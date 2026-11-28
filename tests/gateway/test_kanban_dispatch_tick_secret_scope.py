"""Secret scope for the dispatch tick's plugin hook under multiplex.

Regression for #151126: ``tick_once_for_board`` runs the tick through
``_to_thread_process_service`` in a fresh Context, so ``get_secret`` fails
closed inside an ``on_kanban_dispatch_tick`` subscriber unless the tick
installs the launch profile's scope — the same contract
``auto_decompose_tick`` already implements, and the one ``gateway/AGENTS.md``
states for every off-turn path.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from agent import secret_scope as ss
from gateway import kanban_watchers_dispatcher as kwd
from gateway.kanban_watchers_common import _to_thread_process_service
from hermes_cli import kanban_db as kb
from hermes_cli.plugins import get_plugin_manager


def _settings() -> kwd._DispatcherSettings:
    return kwd._DispatcherSettings(60.0, None, None, 2, 0, True, None, None)


def test_dispatch_tick_reads_launch_profile_secrets_under_multiplex(monkeypatch, tmp_path):
    """A tick's plugin hook resolves keys from the dispatching profile's .env."""
    (tmp_path / ".env").write_text("SOME_KEY=launch-profile-value\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(kwd, "_board_slugs", lambda _kb: ["default"])
    kb.init_db()

    manager = get_plugin_manager()
    saved = {name: list(callbacks) for name, callbacks in manager._hooks.items()}
    seen: dict[str, str | None] = {}

    def _read_secret_in_hook(**_kwargs):
        # A plugin's hook body: the credential exists only in the profile .env.
        seen["value"] = ss.get_secret("SOME_KEY")

    manager._hooks.setdefault("on_kanban_dispatch_tick", []).append(_read_secret_in_hook)
    dispatcher = kwd._KanbanDispatcher(kb, _settings())
    ss.set_multiplex_active(True)
    try:
        # Same hop the gateway uses: fresh Context, no inherited per-turn scope.
        asyncio.run(_to_thread_process_service(dispatcher.tick_once_for_board, "default"))
    finally:
        ss.set_multiplex_active(False)
        manager._hooks = saved

    assert seen.get("value") == "launch-profile-value", (
        "the dispatch tick hook ran without the profile scope: "
        f"read back {seen.get('value')!r}"
    )
    assert ss.current_secret_scope() is None
