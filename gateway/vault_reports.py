"""Compatibility shim — vault reports moved to plugins/kanban/vault-reports/.

Repository policy (AGENTS.md:L114-L119) prohibits third-party connectors in the
core tree. The implementation lives in the plugin; this shim re-exports its
public API so existing call sites keep working while the plugin boundary is
respected. Import from the plugin directly in new code.
"""

from __future__ import annotations

try:
    from plugins.kanban.vault_reports import (  # noqa: F401
        ELIGIBLE_EVENTS,
        append_vault_context,
        load_live_config,
        write_terminal_report,
    )
except ImportError:
    # Plugin not installed — provide no-op stubs so call sites stay fail-open.
    from typing import Any, Iterable

    ELIGIBLE_EVENTS: frozenset = frozenset()

    def write_terminal_report(config: object, **kwargs: Any) -> bool:  # type: ignore[misc]
        return False

    def append_vault_context(config: object, response: str, **kwargs: Any) -> str:  # type: ignore[misc]
        return response

    def load_live_config() -> dict:  # type: ignore[misc]
        return {}
