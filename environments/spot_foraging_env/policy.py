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
        "No walk policy ONNX found (tried %s). Falling back to zero-action "
        "policy. Train walk via train_rapier.py and export to ONNX, or set "
        "SPOT_WALK_POLICY_ONNX.",
        [str(c) for c in candidates],
    )
    return make_zero_policy()
