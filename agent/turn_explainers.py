"""File-mutation verification footers and turn-completion explanations for ``AIAgent``.

The footer tells the model (and user) when a claimed file mutation did not land; the explainer
summarises why a turn ended without a final answer. Every method resolves through ``AIAgent``'s MRO.
"""
import os
import re
from contextlib import suppress
from typing import Any, Dict, Optional

from agent.tool_dispatch_helpers import (
    _extract_error_preview, _extract_file_mutation_targets, _extract_landed_file_mutation_paths
)
from agent.tool_result_classification import (
    FILE_MUTATING_TOOL_NAMES as _FILE_MUTATING_TOOLS, file_mutation_result_landed
)

_NO_REPLY = "⚠️ No reply: "

# One text for "the model produced nothing after retries" on every surface (CLI explainer,
# gateway ``(empty)`` rewrite, desktop); the model name is filled in by the explainer.
EMPTY_RESPONSE_EXPLANATION = (
    "{model} didn't produce a reply this time, even after retries. "
    "Send `continue` to try again, or switch models with /model."
)

# Exact ``turn_exit_reason`` → explanation body (prefixed with ``_NO_REPLY``).
_EXIT_REASON_EXPLANATIONS: Dict[str, str] = {
    "empty_response_exhausted": EMPTY_RESPONSE_EXPLANATION,
    "all_retries_exhausted_no_response": (
        "the model provider didn't answer after all retries. "
        "Send /retry, or switch models with /model."
    ),
    "partial_stream_recovery": (
        "streaming stopped early and only a partial response was "
        "recovered. Send `continue` to resume from where it stopped."
    ),
    "fallback_prior_turn_content": (
        "no new content was produced this turn; showing recovered "
        "prior context. Send `continue` to retry."
    ),
    "redirect_restart_limit_exceeded": (
        "the request was cancelled by a new correction on every attempt, "
        "so the turn stopped instead of retrying forever. Your last "
        "correction is queued as the next message."
    ),
    "rebuilt_restart_limit_exceeded": (
        "every provider in the fallback chain kept failing over, so the "
        "turn stopped instead of retrying forever. Send `continue` or "
        "switch provider."
    ),
    "budget_exhausted": (
        "the per-turn iteration/cost budget was exhausted before a "
        "final answer. Send `continue` to keep going."
    ),
    "ollama_runtime_context_too_small": (
        "the local model's context window was too small to finish. "
        "Increase the context size or use a larger model."
    ),
    "pending_tool_result": (
        "the turn stopped while a tool result was still pending and "
        "the model produced no follow-up text. Send `continue` to "
        "let it summarize."
    ),
}

# Parameterised reasons (``max_iterations_reached(3/3)`` …) matched by prefix.
_EXIT_REASON_PREFIX_EXPLANATIONS = (
    # ``interrupted_during_api_call(<issuer>)`` names a system watchdog (#112647).
    ("interrupted_during_api_call", (
        "the request was interrupted mid-call before a reply was "
        "received. Send `continue` to retry."
    )),
    ("max_iterations_reached", (
        "the maximum tool-iteration limit was reached before a "
        "final answer. Send `continue` to keep going, or raise "
        "`max_iterations`."
    )),
    ("error_near_max_iterations", (
        "an error occurred near the iteration limit before a final "
        "answer. Check the tool output above, then send `continue`."
    )),
    ("repeated_outer_errors", (
        "the turn kept failing with repeated errors and was stopped "
        "early instead of retrying forever. Check the errors above, "
        "then send `continue` to retry."
    )),
)

# ``session_persistence_failed`` refined by the classified cause (lock contention ≠ disk full).
_PERSISTENCE_CAUSE_EXPLANATIONS: Dict[str, str] = {
    "compression": (
        "the turn was stopped because another process was "
        "compressing this session. Your message should already be "
        "saved — please send it again after compression completes."
    ),
    "compression_closed": (
        "the turn was stopped because this session was rotated "
        "by context compression and its live continuation could "
        "not be adopted. The storage itself is healthy — refresh "
        "the client (or start a new turn) so it picks up the new "
        "session id, then send your message again."
    ),
    "turn_lease": (
        "the turn was stopped because another Hermes process "
        "took over this session. Your reply was not saved — wait "
        "for the other process to finish, then send your message "
        "again."
    ),
    "locked": (
        "the turn was stopped because session storage was busy "
        "(another Hermes process was writing to the state "
        "database). Your message should already be saved — "
        "please send it again in a moment."
    ),
    # The forensic runbook for both (WAL generations, manifest.json, sidecars) lives in the
    # logger.error at hermes_state.py::_raise_if_db_replaced — never in the chat reply.
    "replaced": (
        "the session database file was replaced while Hermes was running, so this "
        "message was not saved (a copy is kept in {home}/sessions/). Stop Hermes "
        "(`hermes {profile_arg}gateway stop`), run `hermes {profile_arg}doctor` — not "
        "`hermes {profile_arg}doctor --fix`, which would repair the wrong file in place — "
        "then start it again and send your message once more. Advanced recovery steps are "
        "in the log."
    ),
    "deleted_wal": (
        "another Hermes process still holds an old copy of the session database's write-ahead "
        "log, so Hermes stopped writing to keep the file safe and this message was not saved (a "
        "copy is kept in {home}/sessions/). Nothing is lost. Quit every Hermes process on this "
        "profile (Desktop app, `hermes {profile_arg}gateway stop`, dashboard, cron), run "
        "`hermes {profile_arg}doctor` — it names any process still holding the log — then start "
        "Hermes again and send your message once more. Do not run `doctor --fix` or delete "
        "any state.db files while they run. Guide: {recovery_docs}"
    ),
    "corrupt": (
        "the turn was stopped because the state database "
        "reported structural corruption (the transcript would "
        "have been lost on restart). Freeing disk space will "
        "not help. Recovery options:\n"
        "1. Run `hermes {profile_arg}doctor --fix`\n"
        "2. Stop the gateway, then recover with:\n"
        "   hermes {profile_arg}sessions recover --source {db_path} --inspect-only\n"
        "   (if it reports recoverable) hermes {profile_arg}sessions recover "
        "--source {db_path} --output recovered-state.db\n"
        "   — recovery snapshots the damaged file first; do NOT "
        "run `sqlite3 ... \".recover\"` against the live "
        "state.db, a vulnerable sqlite3 CLI can corrupt it "
        "further\n"
        "3. Restore from a backup in {backups_dir}/\n"
        "Then send your message again."
    ),
    # SQLite scoped the corruption to the FTS index and the derived indexes could not be
    # detached, so this write did not land; the message store itself is intact (#97794).
    "fts_index": (
        "the turn was stopped because the session search index (FTS5) "
        "is corrupt and could not be detached, so this message was not "
        "saved. The message store itself is not damaged: do not run "
        "recovery tools or restore a backup. Run `hermes {profile_arg}doctor --fix` "
        "(or restart Hermes, which repairs the index on open), then "
        "send your message again."
    ),
    "amplification": (
        "the turn stopped because this session reached its physical transcript-row safety "
        "limit. The blocked write was not saved. Start a new session. For controlled "
        "recovery, make a verified backup before cleaning archived generations."
    ),
    "disk": (
        "Hermes couldn't save this conversation to disk, so it stopped rather than lose "
        "your messages. The disk is probably full: free some space (or fix the permissions "
        "on {home}/state.db), then send your message again."
    ),
}
_PERSISTENCE_DEFAULT_EXPLANATION = (
    "Hermes couldn't save this conversation, so it stopped rather than lose your messages. "
    "Possible causes: the drive is out of room, or another Hermes process is holding the "
    "database. Close other Hermes windows, run `hermes {profile_arg}doctor` to check "
    "storage, then send your message again."
)


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
        lines = [
            "⚠️ File-mutation verifier: "
            f"{len(failed)} file edit(s) FAILED this turn despite any "
            "wording above that may suggest otherwise. Run `git status` or "
            "`read_file` to confirm what actually landed."
        ]
        shown = list(failed.items())[:10]
        for path, info in shown:
            preview = (info.get("error_preview") or "").strip()
            tool = info.get("tool") or "patch"
            lines.append(f"  • `{path}` — [{tool}] {preview or 'failed'}")
        remaining = len(failed) - len(shown)
        if remaining > 0:
            lines.append(f"  • … and {remaining} more")
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
        body = _EXIT_REASON_EXPLANATIONS.get(reason)
        if body is None:
            for prefix, text in _EXIT_REASON_PREFIX_EXPLANATIONS:
                if reason.startswith(prefix):
                    body = text
                    break
        if body is not None and "{model}" in body:
            body = body.format(model=model or "The model")
        if body is None and reason == "session_persistence_failed":
            from hermes_constants import display_hermes_home, profile_cli_selector
            from hermes_state_errors import STORAGE_RECOVERY_DOCS_URL

            # Copy-pasteable, so pin every `hermes` command to the profile whose store failed:
            # a multi-profile backend (Desktop serve) hosts sessions whose state.db is NOT the
            # process default, and a bare `hermes` follows active_profile (#105887).
            body = (
                _PERSISTENCE_CAUSE_EXPLANATIONS.get(
                    persistence_cause or "unknown", _PERSISTENCE_DEFAULT_EXPLANATION
                )
                .replace("{home}", display_hermes_home())
                .replace("{profile_arg}", profile_cli_selector())
                .replace("{recovery_docs}", STORAGE_RECOVERY_DOCS_URL)
            )
            if persistence_cause in ("corrupt", "fts_index"):
                from hermes_constants import get_default_hermes_root
                from hermes_state import _default_db_path

                body = body.replace("{db_path}", str(db_path or _default_db_path()))
                body = body.replace(
                    "{backups_dir}", str(get_default_hermes_root() / "backups")
                )
        return _NO_REPLY + body if body else ""
