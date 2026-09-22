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


# Markers that make a directory a real project workspace. A deleted file inside
# one is still a real change (its siblings and history exist); a deleted file
# with none of these above it is loose scratch.
_WORKSPACE_MARKERS = (".git", "package.json", "pyproject.toml", "Cargo.toml", "go.mod")

# Upper bound on the parent walk in _in_workspace(). Deep enough for any real
# checkout, shallow enough that a pathological path can't stat its way up the
# whole filesystem.
_MAX_WORKSPACE_WALK_DEPTH = 24


def _in_workspace(resolved: Path) -> bool:
    """True when any parent of *resolved* carries a workspace marker.

    The user's home directory is NOT a workspace even when a stray marker sits
    in it: this Mac has a loose ``~/package.json``, which made every path under
    ``$HOME`` look like a project and defeated the deleted-scratch exemption.
    Home is the boundary, so the walk stops *before* testing it.

    Bounded: stops at home, at the filesystem root, or after
    ``_MAX_WORKSPACE_WALK_DEPTH`` parents, so a pathological path can't turn
    this into an unbounded stat storm.
    """
    try:
        home = Path.home().resolve()
    except Exception:
        home = None
    for depth, parent in enumerate(resolved.parents):
        if depth >= _MAX_WORKSPACE_WALK_DEPTH:
            break
        if home is not None and (parent == home or parent in home.parents):
            break
        for marker in _WORKSPACE_MARKERS:
            try:
                if (parent / marker).exists():
                    return True
            except Exception:
                break
    return False


def _is_ephemeral_verify_artifact(raw: str) -> bool:
    """Return True for paths the verification loop itself produces or consumes.

    Two classes, both under the system temp dir only (a repo file that happens
    to share a name still verifies):

    1. Ad-hoc verification scripts (``hermes-verify-*`` / ``hermes-ad-hoc-*``)
       — this gate ASKS for those scripts, so counting them as changed paths
       makes the gate self-feeding: each round's evidence becomes the next
       round's finding, and the loop has no terminal state.
    2. Temp-dir paths that no longer exist AND are not inside a workspace
       (no ``.git`` / project manifest between them and the temp root).
       Deleted loose scratch (probe scripts, one-shot fixtures) has no behavior
       left to verify; demanding evidence for it can only be satisfied by
       creating another temp script (class 1). A deleted file inside a real
       project that merely lives under temp is still a real change.
    """
    try:
        p = Path(str(raw)).expanduser()
        if not p.is_absolute():
            return False
        resolved = p.resolve()

        def _under_temp(candidate: Path) -> bool:
            try:
                temp_root = Path(tempfile.gettempdir()).resolve()
                if candidate == temp_root or temp_root in candidate.parents:
                    return True
            except Exception:
                pass
            # macOS: TMPDIR lives under /var/folders (= /private/var/folders),
            # but probes are also written to bare /tmp (= /private/tmp). Treat
            # both as temp space.
            posix = candidate.as_posix()
            return posix.startswith(
                ("/tmp/", "/private/tmp/", "/private/var/folders/", "/var/folders/")
            )

        def _temp_root_set() -> set:
            try:
                roots = {Path(tempfile.gettempdir()).resolve()}
            except Exception:
                roots = set()
            roots |= {
                Path(x)
                for x in ("/tmp", "/private/tmp", "/var/folders", "/private/var/folders")
            }
            return roots

        def _in_temp_workspace(candidate: Path) -> bool:
            """Workspace membership for a path that lives under the temp root.

            Walks upward only while still inside temp space and stops AT the
            temp root, so a stray marker sitting directly in /tmp cannot make
            every scratch path look like project code.
            """
            roots = _temp_root_set()
            for parent in candidate.parents:
                if not _under_temp(parent):
                    return False
                if parent in roots:
                    return False
                for marker in _WORKSPACE_MARKERS:
                    try:
                        if (parent / marker).exists():
                            return True
                    except Exception:
                        return False
            return False

        if not _under_temp(resolved):
            # Outside temp the only exemption is a path that no longer exists
            # AND sits in no workspace. A deleted file has no behavior left to
            # verify, so demanding evidence for it is unsatisfiable: the ledger
            # remembers the path forever and re-nudges every turn (observed
            # 2026-08-22 with a throwaway ~/.hermes/scripts restart script
            # created and deleted in one turn, which then nudged three turns
            # running). Restricting this to temp was arbitrary — scratch lives
            # in plenty of non-temp places. Anything that still EXISTS, and any
            # deleted file inside a real project, keeps nudging exactly as
            # before, so no live code loses its gate.
            if resolved.exists():
                return False
            return not _in_workspace(resolved)
        # A tracked file inside a real workspace is never disposable evidence,
        # even when that workspace itself lives under /tmp (disposable update
        # rehearsals use exactly this layout). The temp ROOT is the boundary,
        # exactly as in the deleted-path branch below: a stray /tmp/package.json
        # must not turn every loose probe into "project code".
        if _in_temp_workspace(resolved):
            # Workspace membership decides BEFORE the name check, and without
            # requiring the file to still exist. Otherwise a repo file named
            # hermes-verify-* is exempt when its checkout happens to live under
            # temp but not when it lives anywhere else — the same file, two
            # answers. Disposable update rehearsals clone exactly into temp, so
            # the gate silently weakened in precisely the tree used to prove it.
            return False
        if p.name.startswith(("hermes-verify-", "hermes-ad-hoc-")):
            return True
        if resolved.exists():
            return False
        # Deleted temp path: scratch only when no workspace marker sits on any
        # parent that is still inside temp space (bounded: temp trees are
        # shallow, and the walk stops at the first parent outside temp).
        try:
            _temp_roots = {Path(tempfile.gettempdir()).resolve()}
        except Exception:
            _temp_roots = set()
        _temp_roots |= {
            Path(x) for x in ("/tmp", "/private/tmp", "/var/folders", "/private/var/folders")
        }
        for parent in resolved.parents:
            if not _under_temp(parent):
                break
            # The temp ROOT itself is a boundary, exactly like $HOME in
            # _in_workspace(): a stray /tmp/package.json (observed 2026-08-28)
            # otherwise makes every scratch path look like it lives in a
            # project and permanently defeats the deleted-scratch exemption.
            if parent in _temp_roots:
                break
            for marker in _WORKSPACE_MARKERS:
                try:
                    if (parent / marker).exists():
                        return False
                except Exception:
                    break
        return True
    except Exception:
        return False


def _filter_verifiable_paths(paths: Iterable[str]) -> list[str]:
    """Drop documentation/prose paths and ephemeral verify-loop scratch; keep
    paths that could have verifiable behavior."""
    return [
        p
        for p in paths
        if p and not _is_non_code_path(p) and not _is_ephemeral_verify_artifact(p)
    ]


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
    # Prose-only turns and ephemeral verify-loop scratch have nothing to verify.
    paths = sorted(set(_filter_verifiable_paths(str(p) for p in changed_paths if p)))
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
