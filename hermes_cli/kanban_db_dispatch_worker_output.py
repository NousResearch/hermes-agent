"""Worker exit-status decoding and log-output normalization for kanban dispatch.

Split out of ``kanban_db_dispatch.py`` (FILE_LINES ratchet). ``kanban_db`` is
imported lazily inside ``_exit_code_kind``: this module loads while
``kanban_db_dispatch`` is still initializing (``kanban_db`` imports from it
late-bound), so a module-level import would close an import cycle.
"""
from __future__ import annotations


def _wait_status_from_returncode(returncode: int) -> int:
    """Encode a ``Popen.returncode`` in the wait-status layout the registry stores."""
    return (int(returncode) & 0xFF) << 8


def _exit_code_kind(code: int) -> tuple[str, int]:
    """``(kind, code)`` for a worker's exit code, however it was observed."""
    from hermes_cli import kanban_db as _kb
    if code == 0:
        return ("clean_exit", 0)
    if code == _kb.KANBAN_RATE_LIMIT_EXIT_CODE:
        return ("rate_limited", code)
    if code == _kb.KANBAN_TERMINAL_PROVIDER_EXIT_CODE:
        return ("terminal_provider", code)
    return ("nonzero_exit", code)


def _exit_summary_marker() -> str:
    """The CLI exit-summary header (``cli_session_mixin.show_exit_summary``), in the active language."""
    from agent.i18n import t
    return t("cli.session.exit_resume_hint")


def _log_noise_prefixes() -> tuple[str, ...]:
    from agent.i18n import t
    return ("session_id:", "Query:", t("cli.chat.initializing_agent"))
