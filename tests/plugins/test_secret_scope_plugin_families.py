"""Regression tests: plugin-family credential reads honor the profile secret scope.

Class-closure follow-up to the profile secret-scope cluster (#76462). Memory,
image_gen, and browser plugins, plus a handful of tier-3 tool helpers, read
credentials straight from ``os.environ``. Under a multiplexed gateway the
process environment may hold ANOTHER profile's key (or none), so every
credential read must route through ``agent.secret_scope.get_secret`` and honor
its verdict — a scoped miss under multiplexing returns the default and must
NOT borrow from ``os.environ``.

One representative test pair (scoped-wins / scoped-miss-no-borrow) per plugin
family, plus the two behavioral sites:

* supermemory ``post_setup`` must not write a profile's key into the
  process-global environ when multiplexing is active (sibling-profile
  pollution).
* google_meet ``process_manager.start`` must resolve OPENAI_API_KEY through
  the scope AT SPAWN TIME and pass it explicitly in the child environment —
  the detached child inherits the process env, not the contextvar scope.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import pytest

from agent.secret_scope import (
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)


@pytest.fixture
def multiplex_scope():
    """Install a secret scope with multiplexing ON; restore state after."""

    def _install(scope: Dict[str, str]):
        set_multiplex_active(True)
        token = set_secret_scope(scope)
        return token

    tokens = []

    def install(scope: Dict[str, str]):
        tokens.append(_install(scope))

    yield install

    for token in tokens:
        reset_secret_scope(token)
    set_multiplex_active(False)


# ---------------------------------------------------------------------------
# Family A — memory plugins
# ---------------------------------------------------------------------------

class TestMemoryFamily:
    def test_retaindb_scoped_key_wins(self, multiplex_scope, monkeypatch):
        monkeypatch.setenv("RETAINDB_API_KEY", "env-other-profile")
        multiplex_scope({"RETAINDB_API_KEY": "scoped-key"})

        from plugins.memory.retaindb import RetainDBMemoryProvider

        assert RetainDBMemoryProvider().is_available() is True

    def test_retaindb_scoped_miss_does_not_borrow_environ(
        self, multiplex_scope, monkeypatch
    ):
        # Env holds another profile's key; the active profile's scope has none.
        monkeypatch.setenv("RETAINDB_API_KEY", "env-other-profile")
        multiplex_scope({})

        from plugins.memory.retaindb import RetainDBMemoryProvider

        assert RetainDBMemoryProvider().is_available() is False

    def test_supermemory_scoped_miss_does_not_borrow_environ(
        self, multiplex_scope, monkeypatch
    ):
        monkeypatch.setenv("SUPERMEMORY_API_KEY", "env-other-profile")
        multiplex_scope({})

        from plugins.memory.supermemory import SupermemoryMemoryProvider

        assert SupermemoryMemoryProvider().is_available() is False

    def test_supermemory_post_setup_no_environ_write_under_multiplex(
        self, multiplex_scope, monkeypatch, tmp_path
    ):
        """post_setup must not pollute process env with a profile's key."""
        multiplex_scope({})
        monkeypatch.delenv("SUPERMEMORY_API_KEY", raising=False)

        import hermes_cli.config as cli_config
        import hermes_cli.memory_setup as memory_setup
        import plugins.memory.supermemory as sm

        monkeypatch.setattr(memory_setup, "_prompt", lambda *a, **k: "sm-fresh-key")
        monkeypatch.setattr(memory_setup, "_write_env_vars", lambda *a, **k: None)
        monkeypatch.setattr(cli_config, "save_config", lambda *a, **k: None)
        monkeypatch.setattr(
            sm,
            "_probe_supermemory_connection",
            lambda *a, **k: {"ok": True, "detail": "stub"},
        )
        monkeypatch.setattr(sm, "_format_connection_summary", lambda s: "stub")

        sm.SupermemoryMemoryProvider().post_setup(str(tmp_path), {})

        assert "SUPERMEMORY_API_KEY" not in os.environ

    def test_supermemory_post_setup_environ_write_kept_single_profile(
        self, monkeypatch, tmp_path
    ):
        """Single-profile (multiplex off): the convenience write still happens."""
        set_multiplex_active(False)
        monkeypatch.delenv("SUPERMEMORY_API_KEY", raising=False)

        import hermes_cli.config as cli_config
        import hermes_cli.memory_setup as memory_setup
        import plugins.memory.supermemory as sm

        monkeypatch.setattr(memory_setup, "_prompt", lambda *a, **k: "sm-fresh-key")
        monkeypatch.setattr(memory_setup, "_write_env_vars", lambda *a, **k: None)
        monkeypatch.setattr(cli_config, "save_config", lambda *a, **k: None)
        monkeypatch.setattr(
            sm,
            "_probe_supermemory_connection",
            lambda *a, **k: {"ok": True, "detail": "stub"},
        )
        monkeypatch.setattr(sm, "_format_connection_summary", lambda s: "stub")

        try:
            sm.SupermemoryMemoryProvider().post_setup(str(tmp_path), {})
            assert os.environ.get("SUPERMEMORY_API_KEY") == "sm-fresh-key"
        finally:
            os.environ.pop("SUPERMEMORY_API_KEY", None)


# ---------------------------------------------------------------------------
# Family B — image_gen plugins
# ---------------------------------------------------------------------------

class TestImageGenFamily:
    def test_deepinfra_scoped_key_wins(self, multiplex_scope, monkeypatch):
        monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
        multiplex_scope({"DEEPINFRA_API_KEY": "scoped-key"})

        from plugins.image_gen.deepinfra import DeepInfraImageGenProvider

        assert DeepInfraImageGenProvider().is_available() is True

    def test_deepinfra_scoped_miss_does_not_borrow_environ(
        self, multiplex_scope, monkeypatch
    ):
        monkeypatch.setenv("DEEPINFRA_API_KEY", "env-other-profile")
        multiplex_scope({})

        from plugins.image_gen.deepinfra import DeepInfraImageGenProvider

        assert DeepInfraImageGenProvider().is_available() is False


# ---------------------------------------------------------------------------
# Family C — browser/web plugins
# ---------------------------------------------------------------------------

class TestBrowserFamily:
    def test_firecrawl_scoped_key_wins(self, multiplex_scope, monkeypatch):
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        multiplex_scope({"FIRECRAWL_API_KEY": "scoped-key"})

        from plugins.browser.firecrawl.provider import FirecrawlBrowserProvider

        provider = FirecrawlBrowserProvider()
        assert provider.is_available() is True
        assert provider._headers()["Authorization"] == "Bearer scoped-key"

    def test_firecrawl_scoped_miss_does_not_borrow_environ(
        self, multiplex_scope, monkeypatch
    ):
        monkeypatch.setenv("FIRECRAWL_API_KEY", "env-other-profile")
        multiplex_scope({})

        from plugins.browser.firecrawl.provider import FirecrawlBrowserProvider

        assert FirecrawlBrowserProvider().is_available() is False


# ---------------------------------------------------------------------------
# Family E — google_meet spawn-wrap
# ---------------------------------------------------------------------------

class TestGoogleMeetSpawn:
    def test_child_env_carries_scoped_openai_key(
        self, multiplex_scope, monkeypatch, tmp_path
    ):
        """start() resolves the key from the scope and injects it explicitly."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("HERMES_MEET_REALTIME_KEY", raising=False)
        multiplex_scope({"OPENAI_API_KEY": "scoped-openai-key"})

        import plugins.google_meet.process_manager as pm

        monkeypatch.setattr(pm, "_root", lambda: tmp_path)

        captured: Dict[str, Any] = {}

        class _FakeProc:
            pid = 4242

        def fake_popen(cmd, **kwargs):
            captured["env"] = kwargs.get("env")
            return _FakeProc()

        monkeypatch.setattr(pm.subprocess, "Popen", fake_popen)

        result = pm.start(
            "https://meet.google.com/abc-defg-hij",
            out_dir=tmp_path / "meeting",
            mode="realtime",
        )

        assert result["ok"] is True
        child_env = captured["env"]
        # The scoped key crosses the process boundary explicitly, not via
        # inherited os.environ (which had no key at all).
        assert child_env["HERMES_MEET_REALTIME_KEY"] == "scoped-openai-key"

    def test_explicit_key_argument_still_wins(
        self, multiplex_scope, monkeypatch, tmp_path
    ):
        multiplex_scope({"OPENAI_API_KEY": "scoped-openai-key"})

        import plugins.google_meet.process_manager as pm

        monkeypatch.setattr(pm, "_root", lambda: tmp_path)

        captured: Dict[str, Any] = {}

        class _FakeProc:
            pid = 4243

        monkeypatch.setattr(
            pm.subprocess,
            "Popen",
            lambda cmd, **kw: (captured.update(env=kw.get("env")), _FakeProc())[1],
        )

        pm.start(
            "https://meet.google.com/abc-defg-hij",
            out_dir=tmp_path / "meeting2",
            realtime_api_key="explicit-key",
        )

        assert captured["env"]["HERMES_MEET_REALTIME_KEY"] == "explicit-key"
