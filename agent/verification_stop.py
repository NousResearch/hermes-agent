"""Turn-end verification guard for coding edits.

This module is intentionally policy-only. It never runs checks itself; it turns
the passive verification ledger into a bounded follow-up when the model tries to
finish immediately after editing code without fresh evidence.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


_MAX_CHANGED_PATHS_IN_NUDGE = 8

# Non-code file extensions whose edits carry no verifiable runtime behavior:
# documentation, prose, and data/markup that no test/build exercises. When a
# turn touches ONLY these, verify-on-stop has nothing to check, so the nudge is
# suppressed (this is fix "C" for the doc/markdown/skill false-positive — a
# SKILL.md or README edit must never demand a /tmp verification script). A turn
# that edits any non-listed path (a real source/code/config file) still nudges.
_NON_CODE_VERIFY_EXTENSIONS = frozenset(
    {
        ".md",
        ".markdown",
        ".mdx",
        ".rst",
        ".txt",
        ".text",
        ".adoc",
        ".asciidoc",
        ".org",
        ".log",
        ".csv",
        ".tsv",
    }
)

# Filenames (case-insensitive, extension-less or otherwise) that are pure prose
# even without a recognized doc extension.
_NON_CODE_VERIFY_FILENAMES = frozenset(
    {
        "license",
        "licence",
        "notice",
        "authors",
        "contributors",
        "changelog",
        "codeowners",
    }
)


def _is_non_code_path(raw: str) -> bool:
    """Return True when a changed path is documentation/prose with nothing to verify."""
    try:
        p = Path(str(raw))
    except Exception:
        return False
    suffix = p.suffix.lower()
    if suffix in _NON_CODE_VERIFY_EXTENSIONS:
        return True
    if not suffix and p.name.lower() in _NON_CODE_VERIFY_FILENAMES:
        return True
    return False


def _filter_verifiable_paths(paths: Iterable[str]) -> list[str]:
    """Drop documentation/prose paths; keep paths that could have verifiable behavior."""
    return [p for p in paths if p and not _is_non_code_path(p)]


def _session_is_messaging_surface() -> bool:
    """Whether this turn is delivered over a human messaging channel.

    Verify-on-stop defaults ON for the interactive coding surfaces and
    programmatic callers, and OFF on a conversational platform (Telegram,
    Discord, Slack, ...) where the verification narrative reaches a human as
    chat noise. The surface classification itself is shared with the other
    consumers of this distinction — see
    ``gateway.session_context.session_is_messaging_surface``.
    """
    try:
        from gateway.session_context import session_is_messaging_surface

        return session_is_messaging_surface()
    except Exception:
        # The gateway package is unreachable, so there is no messaging channel
        # to be on. Reporting a local surface keeps verify-on-stop enabled.
        return False


def verify_on_stop_enabled(config: dict[str, Any] | None = None) -> bool:
    """Return whether edit -> verify-before-finish behavior is enabled.

    Precedence: an explicit ``HERMES_VERIFY_ON_STOP`` env var wins, then an
    explicit ``agent.verify_on_stop`` config value. The default is ``False``
    (opt-in — see ``DEFAULT_CONFIG``): the v31/v32 migrations already turn
    the behavior off for existing installs, so fresh installs match. An
    explicit bool forces the behavior in either direction, and the ``"auto"``
    sentinel opts into the legacy surface-aware behavior: ON for interactive
    coding surfaces (CLI, TUI, desktop) and programmatic callers, OFF for
    conversational messaging surfaces (Telegram, Discord, etc.) where the
    verification narrative would reach a human as chat noise. A missing or
    unrecognized value falls back to OFF.
    """
    env = os.environ.get("HERMES_VERIFY_ON_STOP")
    if env is not None:
        return env.strip().lower() not in {"0", "false", "no", "off"}
    if config is None:
        try:
            from hermes_cli.config import load_config_readonly

            config = load_config_readonly()
        except Exception:
            config = {}
    agent_cfg = (config or {}).get("agent") if isinstance(config, dict) else None
    cfg_val = agent_cfg.get("verify_on_stop") if isinstance(agent_cfg, dict) else None
    if isinstance(cfg_val, bool):
        return cfg_val
    if isinstance(cfg_val, str):
        token = cfg_val.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
        if token == "auto":
            return not _session_is_messaging_surface()
    # Missing or unrecognized value -> OFF, matching the DEFAULT_CONFIG
    # opt-in default. (Only an explicit "auto" opts into the legacy
    # surface-aware behavior.)
    return False


def _candidate_cwds(paths: Iterable[str]) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()
    for raw in paths:
        if not raw:
            continue
        try:
            path = Path(raw).expanduser()
            candidate = path if path.is_dir() else path.parent
            resolved = str(candidate.resolve())
        except Exception:
            continue
        if resolved not in seen:
            seen.add(resolved)
            candidates.append(Path(resolved))
    return candidates


def _verification_snapshot(
    *,
    session_id: str | None,
    changed_paths: list[str],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Return ``(status, facts)`` for the first edited workspace needing proof."""
    try:
        from agent.coding_context import project_facts_for
        from agent.verification_evidence import verification_status
    except Exception:
        return None

    first_snapshot: tuple[dict[str, Any], dict[str, Any]] | None = None
    for cwd in _candidate_cwds(changed_paths):
        facts = project_facts_for(cwd)
        if not facts:
            continue
        status = verification_status(session_id=session_id, cwd=cwd)
        snapshot = (status, facts)
        if first_snapshot is None:
            first_snapshot = snapshot
        if str(status.get("status") or "unverified") != "passed":
            return snapshot
    return first_snapshot


def _format_changed_paths(paths: list[str]) -> str:
    shown = paths[:_MAX_CHANGED_PATHS_IN_NUDGE]
    lines = [f"- `{path}`" for path in shown]
    remaining = len(paths) - len(shown)
    if remaining > 0:
        lines.append(f"- ... and {remaining} more")
    return "\n".join(lines)


def _workspace_has_runnable_recipe(root: Any) -> bool:
    """Whether the workspace has a runtime verify recipe ``hermes verify`` can run.

    True when a saved ``.hermes/environment.json`` manifest exists, or when
    cheap static detection (:func:`agent.verify.recipes.detect_recipe`) finds a
    recipe with a start command. Deliberately fail-silent and cheap — this only
    decorates the nudge text; it must never break or slow the nudge path.
    """
    if not root:
        return False
    try:
        root_path = Path(str(root))
        from agent.verify.environment import manifest_path

        if manifest_path(root_path).is_file():
            return True
        from agent.verify.recipes import detect_recipe

        recipe = detect_recipe(root_path)
        return bool(recipe is not None and recipe.start)
    except Exception:
        return False


def _status_detail(status: dict[str, Any]) -> str:
    state = str(status.get("status") or "unverified")
    evidence = status.get("evidence") if isinstance(status.get("evidence"), dict) else None
    if not evidence:
        return state

    command = evidence.get("canonical_command") or evidence.get("command")
    summary = str(evidence.get("output_summary") or "").strip()
    parts = [state]
    if command:
        parts.append(f"last command `{command}`")
    if summary:
        max_summary = 1200
        if len(summary) > max_summary:
            summary = summary[:max_summary].rstrip() + "\n... [truncated]"
        parts.append(f"last output:\n{summary}")
    return "\n".join(parts)


def build_verify_on_stop_nudge(
    *,
    session_id: str | None,
    changed_paths: Iterable[str],
    attempts: int = 0,
    max_attempts: int = 2,
) -> str | None:
    """Return a synthetic follow-up when edited code lacks fresh verification."""
    # Drop documentation/prose paths (markdown, skills, README, LICENSE, ...) —
    # they carry no verifiable behavior, so a turn that touched only those has
    # nothing to verify and must not nudge.
    paths = sorted({str(p) for p in _filter_verifiable_paths(changed_paths)})
    if not paths or attempts >= max_attempts:
        return None

    snapshot = _verification_snapshot(session_id=session_id, changed_paths=paths)
    if snapshot is None:
        return None
    status, facts = snapshot

    verify_commands = [
        str(cmd).strip()
        for cmd in (facts.get("verifyCommands") or [])
        if str(cmd).strip()
    ]

    state = str(status.get("status") or "unverified")
    if state == "passed":
        return None

    # Optional shipped coding guidance, only paid when this evidence gate fires.
    try:
        from agent.verify_hooks import coding_verify_guidance

        guidance = coding_verify_guidance()
    except Exception:
        guidance = None
    addendum = f"\n\n{guidance}" if guidance else ""

    if verify_commands:
        command_instruction = (
            "Run the relevant verification command now ("
            + ", ".join(f"`{cmd}`" for cmd in verify_commands[:3])
            + (", ..." if len(verify_commands) > 3 else "")
            + "), read any failure, repair the code, and summarize what passed."
        )
        if _workspace_has_runnable_recipe(facts.get("root")):
            command_instruction += (
                " For a full check including a runtime boot (build + test + "
                "start + readiness), prefer `hermes verify --json` — a passing "
                "run records verification evidence for this workspace."
            )
    else:
        temp_dir = os.path.realpath(tempfile.gettempdir())
        if _workspace_has_runnable_recipe(facts.get("root")):
            command_instruction = (
                "No canonical test/lint/build command was detected, but the "
                "project has a runnable verification recipe. Run `hermes verify "
                "--json` (detect -> build -> test -> boot -> readiness poll); a "
                "passing run records verification evidence for this workspace. "
                "Read any failure, repair the code, and summarize what passed."
            )
        else:
            command_instruction = (
                "No canonical test/lint/build command was detected. Create a focused "
                f"temporary verification script under `{temp_dir}` using an OS-safe "
                "`tempfile` path with a `hermes-verify-` filename prefix, run it "
                "against the changed behavior, clean it up when possible, and "
                "summarize it explicitly as ad-hoc verification rather than suite "
                "green."
            )

    return (
        "[System: You edited code in this turn, but the workspace does not have "
        "fresh passing verification evidence yet.\n\n"
        f"Verification status: {_status_detail(status)}\n\n"
        f"Changed paths:\n{_format_changed_paths(paths)}\n\n"
        f"{command_instruction} If verification is not possible, explain the "
        "concrete blocker instead of claiming the work is fully verified."
        f"{addendum}]"
    )


__all__ = ["build_verify_on_stop_nudge", "verify_on_stop_enabled"]
