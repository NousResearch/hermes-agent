"""Turn-end verification guard for coding edits. Policy-only: it never runs
checks itself, it turns the passive verification ledger into a bounded follow-up
when the model tries to finish right after editing code without fresh evidence."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


_MAX_CHANGED_PATHS_IN_NUDGE = 8

# Prose/data extensions and extension-less prose filenames (case-insensitive) with
# no verifiable runtime behavior: a turn touching ONLY these suppresses the nudge
# (a SKILL.md/README edit must never demand a /tmp verification script).
_NON_CODE_VERIFY_EXTENSIONS = frozenset(
    {".md", ".markdown", ".mdx", ".rst", ".txt", ".text", ".adoc", ".asciidoc", ".org", ".log", ".csv", ".tsv"}
)
_NON_CODE_VERIFY_FILENAMES = frozenset(
    {"license", "licence", "notice", "authors", "contributors", "changelog", "codeowners"}
)

_FALSY_TOKENS = {"0", "false", "no", "off"}
_TRUTHY_TOKENS = {"1", "true", "yes", "on"}


def _is_non_code_path(raw: str) -> bool:
    """True when a changed path is documentation/prose with nothing to verify."""
    try:
        p = Path(str(raw))
    except Exception:
        return False
    suffix = p.suffix.lower()
    return suffix in _NON_CODE_VERIFY_EXTENSIONS or (not suffix and p.name.lower() in _NON_CODE_VERIFY_FILENAMES)


def _session_is_messaging_surface() -> bool:
    """Whether this turn is delivered over a human messaging channel. An
    unreachable gateway package means no messaging channel (verify-on-stop stays on)."""
    try:
        from gateway.session_context import session_is_messaging_surface

        return session_is_messaging_surface()
    except Exception:
        return False


def verify_on_stop_enabled(config: dict[str, Any] | None = None) -> bool:
    """Return whether edit -> verify-before-finish behavior is enabled.

    Precedence: ``HERMES_VERIFY_ON_STOP`` env var, then ``agent.verify_on_stop``
    config; default OFF (opt-in). A bool forces the behavior; ``"auto"`` is the
    legacy surface-aware mode: ON for interactive coding surfaces and
    programmatic callers, OFF for messaging surfaces where the verification
    narrative is chat noise. Missing/unrecognized values fall back to OFF.
    """
    env = os.environ.get("HERMES_VERIFY_ON_STOP")
    if env is not None:
        return env.strip().lower() not in _FALSY_TOKENS
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
    token = cfg_val.strip().lower() if isinstance(cfg_val, str) else ""
    if token == "auto":
        return not _session_is_messaging_surface()
    return token in _TRUTHY_TOKENS


def _candidate_cwds(paths: Iterable[str]) -> list[Path]:
    """Distinct resolved directories (a file's parent) for the edited paths, in order."""
    seen: dict[str, None] = {}
    for raw in filter(None, paths):
        try:
            path = Path(raw).expanduser()
            seen.setdefault(str((path if path.is_dir() else path.parent).resolve()))
        except Exception:
            continue
    return [Path(p) for p in seen]


def _verification_snapshot(
    *, session_id: str | None, changed_paths: list[str]
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """``(status, facts)`` for the first edited workspace needing proof, else the
    first recognized workspace when every one is ``passed``."""
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
        first_snapshot = first_snapshot or (status, facts)
        if str(status.get("status") or "unverified") != "passed":
            return status, facts
    return first_snapshot


def _format_changed_paths(paths: list[str]) -> str:
    lines = [f"- `{path}`" for path in paths[:_MAX_CHANGED_PATHS_IN_NUDGE]]
    if len(paths) > _MAX_CHANGED_PATHS_IN_NUDGE:
        lines.append(f"- ... and {len(paths) - _MAX_CHANGED_PATHS_IN_NUDGE} more")
    return "\n".join(lines)


def _workspace_has_runnable_recipe(root: Any) -> bool:
    """Whether ``hermes verify`` has a runtime recipe here: a saved
    ``.hermes/environment.json`` or a statically detected recipe with a start
    command. Fail-silent and cheap — it only decorates the nudge text."""
    if not root:
        return False
    try:
        from agent.verify.environment import manifest_path
        from agent.verify.recipes import detect_recipe

        root_path = Path(str(root))
        if manifest_path(root_path).is_file():
            return True
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
        if len(summary) > 1200:
            summary = summary[:1200].rstrip() + "\n... [truncated]"
        parts.append(f"last output:\n{summary}")
    return "\n".join(parts)


def build_verify_on_stop_nudge(
    *, session_id: str | None, changed_paths: Iterable[str], attempts: int=0, max_attempts: int=2,
) -> str | None:
    """Return a synthetic follow-up when edited code lacks fresh verification."""
    # Prose-only turns (markdown, skills, README, LICENSE, ...) have nothing to verify.
    paths = sorted({str(p) for p in changed_paths if p and not _is_non_code_path(p)})
    if not paths or attempts >= max_attempts:
        return None

    snapshot = _verification_snapshot(session_id=session_id, changed_paths=paths)
    if snapshot is None:
        return None
    status, facts = snapshot
    if str(status.get("status") or "unverified") == "passed":
        return None
    verify_commands = [str(cmd).strip() for cmd in (facts.get("verifyCommands") or []) if str(cmd).strip()]
    has_recipe = _workspace_has_runnable_recipe(facts.get("root"))

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
        if has_recipe:
            command_instruction += (
                " For a full check including a runtime boot (build + test + "
                "start + readiness), prefer `hermes verify --json` — a passing "
                "run records verification evidence for this workspace."
            )
    elif has_recipe:
        command_instruction = (
            "No canonical test/lint/build command was detected, but the "
            "project has a runnable verification recipe. Run `hermes verify "
            "--json` (detect -> build -> test -> boot -> readiness poll); a "
            "passing run records verification evidence for this workspace. "
            "Read any failure, repair the code, and summarize what passed."
        )
    else:
        temp_dir = os.path.realpath(tempfile.gettempdir())
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
