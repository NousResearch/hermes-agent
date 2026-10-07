"""File-mutation verification footers and turn-completion explanations for ``AIAgent``.

The footer tells the model (and user) when a claimed file mutation did not land; the explainer
summarises why a turn ended without a final answer. Every method resolves through ``AIAgent``'s MRO.
"""
import os
import re
from contextlib import suppress
from typing import Any, Dict, Optional

from agent.i18n import t
from agent.interrupt_control import STOP_KIND_CLIENT_DISCONNECT
from agent.tool_dispatch_helpers import (
    _extract_error_preview, _extract_file_mutation_targets, _extract_landed_file_mutation_paths
)
from agent.tool_result_classification import (
    FILE_MUTATING_TOOL_NAMES as _FILE_MUTATING_TOOLS, file_mutation_result_landed
)

# One text for "the model produced nothing after retries" on every surface (CLI explainer,
# gateway ``(empty)`` rewrite, desktop). English source kept as a constant for importers; surfaces
# rendering to a human use ``empty_response_explanation`` (``explainer.empty_response``).
EMPTY_RESPONSE_EXPLANATION = (
    "{model} didn't produce a reply this time, even after retries. "
    "Send `continue` to try again, or switch models with /model."
)


def empty_response_explanation(model: str = "") -> str:
    """Localized ``EMPTY_RESPONSE_EXPLANATION`` with the model name (or "The model") filled in."""
    return t("explainer.empty_response", model=model or t("explainer.shared.the_model"))


# Exact ``turn_exit_reason`` → catalog key of the explanation body (prefixed with the no-reply marker).
_EXIT_REASON_EXPLANATIONS: Dict[str, str] = {
    "empty_response_exhausted": "explainer.empty_response",
    "all_retries_exhausted_no_response": "explainer.exit.all_retries_exhausted_no_response",
    "partial_stream_recovery": "explainer.exit.partial_stream_recovery",
    "fallback_prior_turn_content": "explainer.exit.fallback_prior_turn_content",
    "redirect_restart_limit_exceeded": "explainer.exit.redirect_restart_limit_exceeded",
    "rebuilt_restart_limit_exceeded": "explainer.exit.rebuilt_restart_limit_exceeded",
    "budget_exhausted": "explainer.exit.budget_exhausted",
    "ollama_runtime_context_too_small": "explainer.exit.ollama_runtime_context_too_small",
    "pending_tool_result": "explainer.exit.pending_tool_result",
    f"interrupted_during_api_call({STOP_KIND_CLIENT_DISCONNECT})": "explainer.exit.client_disconnect",
    f"interrupted_by_system({STOP_KIND_CLIENT_DISCONNECT})": "explainer.exit.client_disconnect",
}

# Parameterised reasons (``max_iterations_reached(3/3)`` …) matched by prefix.
# ``interrupted_during_api_call(<issuer>)`` names a system watchdog (#112647).
_EXIT_REASON_PREFIX_EXPLANATIONS = tuple(
    (prefix, f"explainer.exit.{prefix}") for prefix in ("interrupted_during_api_call", "max_iterations_reached", "error_near_max_iterations", "repeated_outer_errors")
)

# ``session_persistence_failed`` refined by the classified cause (lock contention ≠ disk full).
_PERSISTENCE_CAUSES = frozenset({"compression", "compression_closed", "turn_lease", "session_row_missing", "locked", "replaced", "deleted_wal", "corrupt", "fts_index", "disk"})


def _persistence_explanation_key(cause: Optional[str]) -> str:
    return f"explainer.persistence.{cause}" if cause in _PERSISTENCE_CAUSES else "explainer.persistence.default"


def _file_mutation_identity(path: str, task_id: Optional[str]) -> str:
    """One key per on-disk target: the file tools' task-resolved absolute path, case-folded
    on case-insensitive hosts. A failure recorded as ``notes.md`` and the write that later
    lands as ``/repo/notes.md`` (or ``Notes.md`` on Windows) must meet on the same key."""
    try:
        from tools.file_tools_paths import _resolve_path_for_task

        resolved = str(_resolve_path_for_task(path, task_id or "default"))
    except Exception:
        resolved = os.path.abspath(os.path.expanduser(path))
    return os.path.normcase(os.path.normpath(resolved))


def _file_stat_signature(identity: str) -> Optional[tuple]:
    """``(mtime_ns, size)`` of the target, ``None`` when it does not exist (or cannot be read)."""
    try:
        st = os.stat(identity)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _display_flag_enabled(agent, *, env_var: str, config_key: str, cache_attr: str) -> bool:
    """``display.<config_key>`` (default True), cached per agent on ``cache_attr``.

    ``env_var`` overrides on every call and is never cached. Reads the persisted config.yaml
    so gateway and CLI share the setting; ``load_config`` is imported lazily (startup cycle,
    and tests patch it at ``hermes_cli.config``). Any failure → True (safe default: on)."""
    try:
        env = os.environ.get(env_var)
        if env is not None:
            return env.strip().lower() not in {"0", "false", "no", "off"}
        cached = getattr(agent, cache_attr, None)
        if cached is not None:
            return cached
        try:
            from hermes_cli.config import load_config as _load_config
            _cfg = _load_config() or {}
        except Exception:
            _cfg = {}
        _display = _cfg.get("display") if isinstance(_cfg, dict) else None
        if isinstance(_display, dict) and config_key in _display:
            enabled = bool(_display.get(config_key))
        else:
            enabled = True
        setattr(agent, cache_attr, enabled)
        return enabled
    except Exception:
        return True


class TurnExplainersMixin:
    """File-mutation failure footer + turn-completion explainer (see module docstring)."""

    def _record_file_mutation_result(
        self, tool_name: str, args: Dict[str, Any], result: Any, is_error: bool,
        *, task_id: Optional[str] = None,
    ) -> None:
        """Record a ``write_file`` / ``patch`` outcome for the turn-end verifier.

        Failures store ``{path: {error_preview, tool, identity, stat}}`` keyed by the model's
        spelling; ``identity`` is the resolved on-disk target and ``stat`` its signature at
        failure time. A later success on the same identity (any spelling) removes the entry.
        No-op when the per-turn state dict is not initialised (tool dispatched outside ``run_conversation``).
        """
        if tool_name not in _FILE_MUTATING_TOOLS:
            return
        state = getattr(self, "_turn_failed_file_mutations", None)
        if state is None:
            return
        targets = _extract_file_mutation_targets(tool_name, args)
        if not targets:
            return
        landed = file_mutation_result_landed(tool_name, result)
        if landed:
            landed_paths = _extract_landed_file_mutation_paths(tool_name, args, result)
            changed = getattr(self, "_turn_file_mutation_paths", None)
            if changed is not None:
                changed.update(landed_paths)
            # Feed the checkpoint agent-write ledger so /rollback's safe mode can tell
            # Hermes-authored content from later user hand-edits.
            mgr = getattr(self, "_checkpoint_mgr", None)
            if mgr is not None and getattr(mgr, "enabled", False):
                from tools.file_tools_paths import container_backend_for_task
                if container_backend_for_task(task_id or "default") is None:  # container paths carry no host ledger entry
                    for _p in landed_paths:
                        with suppress(Exception):
                            mgr.record_agent_write(_p)
        if is_error and not landed:
            # Keep the FIRST error per path unless a later success replaces it.
            preview = _extract_error_preview(result)
            for path in targets:
                identity = _file_mutation_identity(path, task_id)
                state.setdefault(path, {
                    "tool": tool_name, "error_preview": preview,
                    "identity": identity, "stat": _file_stat_signature(identity),
                })
        else:
            cleared = {
                _file_mutation_identity(p, task_id)
                for p in (landed_paths if landed else targets)
            }
            for path, info in list(state.items()):
                if info.get("identity", _file_mutation_identity(path, task_id)) in cleared:
                    state.pop(path, None)

    @staticmethod
    def _file_mutations_still_failed(failed: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Drop entries whose target changed on disk since the failed call.

        The recorder only sees write_file/patch receipts; a terminal redirect or an
        execute_code write leaves none. Re-checking the stat signature at turn end keeps
        the footer from listing a file that was in fact modified later this turn. Entries
        without a snapshot (hand-built dicts) are kept as-is.
        """
        return {
            path: info for path, info in failed.items()
            if "stat" not in info or _file_stat_signature(info["identity"]) == info["stat"]
        }

    def _file_mutation_verifier_enabled(self) -> bool:
        """``display.file_mutation_verifier`` / ``HERMES_FILE_MUTATION_VERIFIER`` (a patchable seam)."""
        return _display_flag_enabled(
            self, env_var="HERMES_FILE_MUTATION_VERIFIER", config_key="file_mutation_verifier",
            cache_attr="_file_mutation_verifier_enabled_cache",
        )

    def _turn_completion_explainer_enabled(self) -> bool:
        """``display.turn_completion_explainer`` / ``HERMES_TURN_COMPLETION_EXPLAINER``."""
        return _display_flag_enabled(
            self, env_var="HERMES_TURN_COMPLETION_EXPLAINER", config_key="turn_completion_explainer",
            cache_attr="_turn_completion_explainer_enabled_cache",
        )

    # Bare absolute / home / Windows-drive paths in a footer line. Mirrors the gateway's
    # extract_local_files detector so anything it WOULD auto-attach is backticked first (#35584).
    _FOOTER_PATH_RE = re.compile(
        r"(?<![/:\w.`])(?:~/|/|[A-Za-z]:[/\\])(?:[\w.\-]+[/\\])*[\w.\-]+\.[\w]+",
    )

    @classmethod
    def _neutralize_footer_paths(cls, text: str) -> str:
        """Backtick bare file paths so the gateway's ``extract_local_files`` never auto-attaches them.

        The extractor skips inline-code spans; already-backticked paths are left alone (no double-wrap).
        """
        if not text:
            return text
        return cls._FOOTER_PATH_RE.sub(lambda m: f"`{m.group(0)}`", text)

    @classmethod
    def _format_file_mutation_failure_footer(cls, failed: Dict[str, Dict[str, Any]]) -> str:
        """Render the per-turn failed-mutation dict as a user-facing footer.

        Up to 10 paths with their first error preview, then an overflow count; "" when nothing failed.
        Every path is backtick-wrapped via ``_neutralize_footer_paths`` so protected files cannot be
        auto-delivered.
        """
        if not failed:
            return ""
        lines = [t("explainer.file_mutation.header", count=len(failed))]
        shown = list(failed.items())[:10]
        for path, info in shown:
            preview = (info.get("error_preview") or "").strip()
            tool = info.get("tool") or "patch"
            lines.append(t("explainer.file_mutation.entry", path=path, tool=tool,
                           preview=preview or t("explainer.file_mutation.failed")))
        remaining = len(failed) - len(shown)
        if remaining > 0:
            lines.append(t("explainer.shared.and_more", count=remaining))
        # Neutralize paths the preview echoed; the lookbehind prevents double-wrapping the bullet path.
        return cls._neutralize_footer_paths("\n".join(lines))

    @staticmethod
    def _format_turn_completion_explanation(
        turn_exit_reason: str, persistence_cause: Optional[str] = None, db_path=None, model: str = "",
    ) -> str:
        """User-facing explanation for an abnormal turn ending, or "" for normal / unknown reasons.

        ``text_response(...)`` is the healthy terminal; unknown/diagnostic-only reasons (e.g.
        ``guardrail_halt``, which surfaces its own message) are not second-guessed.
        """
        if not turn_exit_reason:
            return ""
        reason = str(turn_exit_reason)
        if reason.startswith("text_response"):
            return ""
        key = _EXIT_REASON_EXPLANATIONS.get(reason)
        if key is None:
            for prefix, prefix_key in _EXIT_REASON_PREFIX_EXPLANATIONS:
                if reason.startswith(prefix):
                    key = prefix_key
                    break
        if key is not None:
            body = t(key, model=model or t("explainer.shared.the_model"))
        elif reason == "session_persistence_failed":
            from hermes_constants import display_hermes_home, profile_cli_selector
            from hermes_state_errors import STORAGE_RECOVERY_DOCS_URL

            # Copy-pasteable, so pin every `hermes` command to the profile whose store failed:
            # a multi-profile backend (Desktop serve) hosts sessions whose state.db is NOT the
            # process default, and a bare `hermes` follows active_profile (#105887).
            fill: Dict[str, str] = {
                "home": display_hermes_home(), "profile_arg": profile_cli_selector(),
                "recovery_docs": STORAGE_RECOVERY_DOCS_URL, "db_path": "", "backups_dir": "",
            }
            if persistence_cause in ("corrupt", "fts_index"):
                from hermes_constants import get_default_hermes_root
                from hermes_state import _default_db_path

                fill["db_path"] = str(db_path or _default_db_path())
                fill["backups_dir"] = str(get_default_hermes_root() / "backups")
            body = t(_persistence_explanation_key(persistence_cause), **fill)
        else:
            body = None
        return t("explainer.no_reply_prefix") + body if body else ""
