"""Top-down scene renderer.

Produces a small PNG image showing Spot's pose, the batteries, and recent
trajectory. Encoded as data-URL for OpenAI multimodal `image_url` content
parts. Cheap (matplotlib + io.BytesIO) so we can call it every skill pick.

Why top-down vs WASM/rerun? The WASM/rerun viewers need a running Atropos
training-side viewer pod (or a JS runtime), while matplotlib is pure-python
and works in any process. The image carries the same spatial signal —
position, heading, battery layout — without the rendering pipeline
overhead. Swap to the WASM render later once we have a clean reuse path.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Lazy-import matplotlib so the env package imports fast even when only
# the env metadata is needed (e.g. for Atropos `serve`'s schema dump).
try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend; mandatory in headless training pods
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrow

    _MATPLOTLIB_OK = True
except Exception as e:  # pragma: no cover
    _MATPLOTLIB_OK = False
    _IMPORT_ERROR = e


@dataclass
class SceneSnapshot:
    """Minimal spatial state needed to render the foraging scene top-down."""

    spot_xy: Tuple[float, float]
    spot_yaw: float                     # radians, world frame, +Z = up
    batteries: List[Tuple[float, float]]
    collected_count: int
    trail: List[Tuple[float, float]]    # historical xy, oldest first
    energy: Optional[float] = None      # 0-1, optional


def render_top_down(
    snap: SceneSnapshot,
    *,
    side_meters: float = 8.0,
    pixels: int = 256,
    title: Optional[str] = None,
) -> bytes:
    """Render the snapshot as a PNG byte string.

    `side_meters` is the world half-extent the view spans (so the figure
    shows ±side_meters in X and Z). `pixels` is the output resolution.
    """
    if not _MATPLOTLIB_OK:
        raise RuntimeError(
            f"matplotlib not available for renderer ({_IMPORT_ERROR}). "
            "Install matplotlib in the env-server container."
        )

    fig, ax = plt.subplots(figsize=(pixels / 100.0, pixels / 100.0), dpi=100)
    ax.set_xlim(-side_meters, side_meters)
    ax.set_ylim(-side_meters, side_meters)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#1a1a1a")
    ax.tick_params(colors="#888888", labelsize=6)
    ax.grid(True, color="#333333", linewidth=0.3)

    # Trail (faded blue).
    if len(snap.trail) >= 2:
        xs = [p[0] for p in snap.trail]
        ys = [p[1] for p in snap.trail]
        ax.plot(xs, ys, color="#4a90e2", linewidth=0.8, alpha=0.5)

    # Batteries (yellow circles).
    for bx, bz in snap.batteries:
        ax.add_patch(Circle((bx, bz), 0.15, color="#ffcc33", zorder=3))

    # Spot (triangle/arrow showing heading).
    sx, sz = snap.spot_xy
    arrow_len = 0.4
    dx = arrow_len * math.cos(snap.spot_yaw)
    dz = arrow_len * math.sin(snap.spot_yaw)
    ax.add_patch(
        FancyArrow(
            sx, sz, dx, dz,
            width=0.08, head_width=0.22, head_length=0.18,
            color="#7dd3fc", zorder=4,
        )
    )
    ax.add_patch(Circle((sx, sz), 0.18, color="#0ea5e9", zorder=4))

    # Compass / scale annotations.
    ax.text(0, side_meters - 0.4, "+X (east)", color="#666", fontsize=5,
            ha="center", va="top")
    ax.text(side_meters - 0.4, 0, "+Z (north)", color="#666", fontsize=5,
            ha="right", va="center", rotation=-90)

    # Status overlay.
    status_lines = [
        f"collected: {snap.collected_count}",
        f"remaining: {len(snap.batteries)}",
    ]
    if snap.energy is not None:
        status_lines.append(f"energy: {snap.energy:.2f}")
    ax.text(
        -side_meters + 0.2, side_meters - 0.4,
        "\n".join(status_lines),
        color="#cccccc", fontsize=6, va="top",
        family="monospace",
    )

    if title:
        ax.set_title(title, color="#cccccc", fontsize=7)

    buf = io.BytesIO()
    fig.tight_layout(pad=0.2)
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def encode_data_url(png_bytes: bytes) -> str:
    """OpenAI multimodal-style data URL for chat `image_url` payloads."""
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


def make_image_content_part(snap: SceneSnapshot, **kwargs) -> dict:
    """OpenAI-format `{"type": "image_url", "image_url": {"url": ...}}` part."""
    return {
        "type": "image_url",
        "image_url": {"url": encode_data_url(render_top_down(snap, **kwargs))},
    }


def snapshot_from_gym(gym_env, trail: Sequence[Tuple[float, float]]) -> SceneSnapshot:
    """Pull a SceneSnapshot from a SpotEnvRapier instance (rapier-gym)."""
    sim = gym_env.sim
    base_pos = np.array(sim.get_base_position(), dtype=np.float32)
    spot_xy = (float(base_pos[0]), float(base_pos[2]))

    # Yaw from the body orientation. SpotEnvRapier's _compute_observation
    # exposes proj_grav at obs[6:9]; for yaw we'd want the world-frame
    # heading, computed from base orientation (we can read it from the
    # raw obs[3:6] which is proj_grav rotated, but the cleanest source is
    # base velocity direction OR a dedicated quaternion accessor).
    # Falling back to estimating heading from recent displacement keeps
    # us decoupled from spot-physics' orientation API surface.
    yaw = 0.0
    if hasattr(sim, "get_base_quaternion"):
        q = sim.get_base_quaternion()
        if len(q) == 4:
            qi, qj, qk, qw = q[0], q[1], q[2], q[3]
            # Rapier uses Y-up. Yaw is rotation around Y.
            # Forward vector projection on X-Z plane:
            yaw = math.atan2(2.0 * (qw * qj + qi * qk), 1.0 - 2.0 * (qj * qj + qk * qk))
    else:
        # Fallback if quaternion API isn't available
        if len(trail) >= 2:
            dx = trail[-1][0] - trail[-2][0]
            dz = trail[-1][1] - trail[-2][1]
            if abs(dx) + abs(dz) > 1e-3:
                yaw = math.atan2(dz, dx)

    batteries_raw = sim.get_battery_positions() or []
    batteries = [(float(b[0]), float(b[2])) for b in batteries_raw]

    collected = int(getattr(gym_env, "collected_targets", 0))

    energy: Optional[float] = None
    try:
        energy = float(sim.get_energy())
    except Exception:
        pass

    return SceneSnapshot(
        spot_xy=spot_xy,
        spot_yaw=yaw,
        batteries=batteries,
        collected_count=collected,
        trail=list(trail),
        energy=energy,
    )
