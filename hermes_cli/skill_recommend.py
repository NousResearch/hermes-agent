"""Read-only advisory skill recommendations for the Hermes CLI.

This module is intentionally a thin CLI/manual wrapper around the Phase 2H
local pgvector recommendation script.  It does not load skills, edit config,
mutate prompts, install hooks, or write telemetry.  The only action is a
foreground subprocess call that returns advisory text for a human/operator to
use manually.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Sequence

DEFAULT_WRAPPER = Path.home() / ".hermes" / "scripts" / "pgvector_recommended_skills.py"
DEFAULT_TOP_K = 3
DEFAULT_MIN_SCORE = 0.04

Runner = Callable[..., subprocess.CompletedProcess[str]]


def _format_min_score(value: float | None) -> str:
    if value is None:
        return "None"
    return f"{value:g}"


def recommend_skills(
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    min_score: float | None = DEFAULT_MIN_SCORE,
    wrapper_path: str | Path | None = None,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Return advisory skill recommendations for ``query``.

    The wrapped script is invoked with ``--json`` and deliberately without
    ``--log-event`` or any flag that would load skills or mutate runtime state.
    """
    if top_k < 1 or top_k > 20:
        raise ValueError("top_k must be between 1 and 20")

    wrapper = Path(wrapper_path) if wrapper_path is not None else DEFAULT_WRAPPER
    if not wrapper.exists():
        raise FileNotFoundError(f"recommendation wrapper not found: {wrapper}")

    cmd: list[str] = [str(wrapper), query, "--top-k", str(top_k)]
    if min_score is None:
        cmd.append("--no-min-score")
    else:
        cmd += ["--min-score", _format_min_score(min_score)]
    cmd.append("--json")

    proc = runner(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout or f"wrapper exited with {proc.returncode}").strip()
        raise RuntimeError(message)

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"recommendation wrapper returned invalid JSON: {exc}") from exc

    payload.setdefault("scope", {})
    payload["scope"].setdefault("manual_only", True)
    payload["scope"].setdefault("loads_skills", False)
    payload["scope"].setdefault("runtime_coupling", False)
    payload["scope"].setdefault("config_changes", False)
    payload["scope"].setdefault("db_writes", False)
    return payload


def render_recommendations(payload: dict[str, Any], *, show_scores: bool = False) -> str:
    """Render recommendation JSON as compact human-readable advisory text."""
    query = payload.get("query") or ""
    diagnostics = payload.get("diagnostics") or {}
    recommendations = payload.get("recommendations") or []

    lines: list[str] = ["Recommended skills:"]
    if not recommendations:
        lines.append("No confident skill recommendation found.")
    else:
        for idx, rec in enumerate(recommendations, start=1):
            rank = rec.get("rank") or idx
            name = rec.get("skill_name") or rec.get("name") or "<unknown>"
            rel_path = rec.get("relative_path") or ""
            line = f"{rank}. {name}"
            if rel_path:
                line += f" — {rel_path}"
            score = rec.get("score")
            if show_scores and score is not None:
                try:
                    line += f" — score {float(score):.6f}"
                except (TypeError, ValueError):
                    line += f" — score {score}"
            lines.append(line)

    if query:
        lines.extend(["", f"Query: {query}"])
    min_score = diagnostics.get("min_score", payload.get("min_score"))
    if min_score is not None:
        lines.append(f"Min score: {min_score}")

    lines.extend(
        [
            "",
            "Advisory only: no runtime hooks, autoloading, prompt mutation, or config changes.",
            "No skills were loaded. Load any recommendation manually with /skill <name> or hermes -s <name>.",
        ]
    )
    return "\n".join(lines)


def recommend_and_render(
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    min_score: float | None = DEFAULT_MIN_SCORE,
    wrapper_path: str | Path | None = None,
    show_scores: bool = False,
) -> str:
    payload = recommend_skills(
        query,
        top_k=top_k,
        min_score=min_score,
        wrapper_path=wrapper_path,
    )
    return render_recommendations(payload, show_scores=show_scores)
