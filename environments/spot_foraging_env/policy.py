"""Low-level walk PPO policy loader.

Loads a trained walk policy in ONNX format (the artifact produced by
`export_pipeline.export_onnx` in the rapier-gym tree) and returns a
callable suitable for `SkillExecutor`.

Falls back to a "stand-still" zero-action policy when no checkpoint is
available — lets the rest of the env be tested before walk training has
converged. Falling-still-on-the-spot is a degenerate but stable behavior
that keeps episodes running.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np


logger = logging.getLogger(__name__)


PolicyFn = Callable[[np.ndarray], np.ndarray]


def make_zero_policy() -> PolicyFn:
    """Always-zero action. Spot stands still (modulo any default joint biases)."""

    def policy(_obs: np.ndarray) -> np.ndarray:
        return np.zeros(12, dtype=np.float32)

    return policy


def make_random_policy(scale: float = 0.05, seed: int = 0) -> PolicyFn:
    """Tiny gaussian noise. For sanity-testing the env loop produces motion."""
    rng = np.random.default_rng(seed)

    def policy(_obs: np.ndarray) -> np.ndarray:
        return (rng.standard_normal(12) * scale).astype(np.float32)

    return policy


def make_cpg_policy(freq_hz: float = 2.0, amplitude: float = 0.3) -> PolicyFn:
    """Central Pattern Generator (CPG) heuristic walk policy.

    Produces a trot gait via sinusoidal joint targets, sufficient for Spot to
    physically move in the rapier-gym sim when no trained ONNX policy is
    available. Not a substitute for a trained policy — gaits will be unstable
    at high amplitudes — but generates non-zero displacement so the Atropos
    foraging env produces meaningful reward gradients.

    The obs vector contains the command in indices 45:48 (v_x, v_lat, w_z)
    based on the rapier-gym observation layout:
      [0:3]   ang_vel, [3:6] lin_vel, [6:9] proj_grav
      [9:21]  joint_pos, [21:33] joint_vel, [33:45] prev_action
      [45:48] cmd (v_x, v_lat, w_z)
      ... rest: foot_contacts, torques, body_contact, obstacle_rays, behavior, forage, gait_phase

    Joint ordering (12 dims): FL_hip, FL_upper, FL_lower, FR_hip, FR_upper, FR_lower,
                               RL_hip, RL_upper, RL_lower, RR_hip, RR_upper, RR_lower
    """
    _phase = [0.0]  # mutable closure for phase accumulation
    _dt = 1.0 / 50.0  # 50 Hz policy rate

    # Trot: FL+RR swing together, FR+RL swing together (180° phase offset)
    _leg_phase_offsets = np.array([0.0, 0.0, 0.0,   # FL hip/upper/lower
                                    np.pi, np.pi, np.pi,  # FR
                                    np.pi, np.pi, np.pi,  # RL
                                    0.0, 0.0, 0.0],       # RR
                                   dtype=np.float32)

    # Hip: small lateral sway; upper-leg: main swing; lower-leg: follow
    _joint_amp_weights = np.array([0.05, 1.0, -0.5,  # FL
                                    0.05, 1.0, -0.5,  # FR
                                    0.05, 1.0, -0.5,  # RL
                                    0.05, 1.0, -0.5], # RR
                                   dtype=np.float32)

    def policy(obs: np.ndarray) -> np.ndarray:
        # Read command from obs if available.
        cmd_vx = float(obs[45]) if len(obs) > 47 else 1.0
        cmd_lat = float(obs[46]) if len(obs) > 47 else 0.0
        cmd_yaw = float(obs[47]) if len(obs) > 47 else 0.0

        speed = abs(cmd_vx) + abs(cmd_lat) * 0.5 + abs(cmd_yaw) * 0.3
        effective_amp = amplitude * max(0.1, min(1.0, speed))

        phase = _phase[0] + 2 * np.pi * freq_hz * _dt
        _phase[0] = phase % (2 * np.pi)

        action = effective_amp * _joint_amp_weights * np.sin(phase + _leg_phase_offsets)
        return np.clip(action, -0.5, 0.5).astype(np.float32)

    return policy


def make_onnx_policy(onnx_path: str | os.PathLike) -> PolicyFn:
    """Load an ONNX policy via onnxruntime. CPU is fine — the policy is tiny."""
    import onnxruntime as ort  # lazy: not all installs have it

    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    expected_obs_dim = sess.get_inputs()[0].shape[-1]
    if isinstance(expected_obs_dim, int):
        logger.info(
            "Loaded walk policy from %s (obs_dim=%d)", onnx_path, expected_obs_dim
        )

    def policy(obs: np.ndarray) -> np.ndarray:
        if obs.ndim == 1:
            obs = obs[None, :]  # add batch dim
        action = sess.run([out_name], {in_name: obs.astype(np.float32)})[0]
        if action.ndim == 2:
            action = action[0]
        return action.astype(np.float32)

    return policy


def load_walk_policy(
    onnx_path: Optional[str | os.PathLike] = None,
) -> PolicyFn:
    """Load the configured walk policy, falling back to zero if missing.

    Resolution order:
      1. Explicit `onnx_path` argument.
      2. `SPOT_WALK_POLICY_ONNX` env var.
      3. Conventional location: $HOME/.cache/spot_rapier/walk_policy.onnx.
      4. Zero-action fallback (logs a warning).

    The fallback is deliberate — the foraging env is meant to be runnable
    end-to-end for testing the LLM-loop / Atropos plumbing even when no
    walk policy has been trained yet. With zero actions Spot just stands
    there and the LLM's skill picks have no physical effect, but episodes
    still run, reward is computed, and the trainer can collect data.
    """
    candidates = []
    if onnx_path:
        candidates.append(Path(onnx_path).expanduser())
    if os.environ.get("SPOT_WALK_POLICY_ONNX"):
        candidates.append(Path(os.environ["SPOT_WALK_POLICY_ONNX"]).expanduser())
    candidates.append(
        Path.home() / ".cache" / "spot_rapier" / "walk_policy.onnx"
    )

    for c in candidates:
        if c.exists():
            try:
                return make_onnx_policy(c)
            except Exception as e:
                logger.warning("Failed to load ONNX policy %s: %s", c, e)
                continue

    logger.warning(
        "No walk policy ONNX found (tried %s). Falling back to CPG heuristic "
        "trot policy. Train walk via train_rapier.py and export to ONNX, or "
        "set SPOT_WALK_POLICY_ONNX.",
        [str(c) for c in candidates],
    )
    return make_cpg_policy(amplitude=0.04)  # very small — avoids falls while still non-zero
