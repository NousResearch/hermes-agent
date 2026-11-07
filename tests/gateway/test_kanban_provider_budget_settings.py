"""Gateway dispatcher settings parsing for the per-provider concurrency
budget (#123654, plan T27): ``_resolve_dispatcher_settings`` carries the raw
``kanban.provider_concurrency`` mapping into ``_DispatcherSettings``, and
``tick_once_for_board`` forwards it to ``dispatch_once`` (spy)."""

from __future__ import annotations


def _settings(overrides: dict | None = None):
    from gateway import kanban_watchers_dispatcher as kwd
    from hermes_cli import kanban_db_dispatch as kbd

    cfg: dict = {"dispatch_interval_seconds": 60, "failure_limit": 3}
    cfg.update(overrides or {})
    return kwd._resolve_dispatcher_settings(cfg, kbd)


class TestResolveSettings:
    def test_absent_key_is_none(self):
        settings = _settings()
        assert settings.provider_concurrency is None

    def test_none_value_is_none(self):
        settings = _settings({"provider_concurrency": None})
        assert settings.provider_concurrency is None

    def test_raw_mapping_carried_verbatim(self):
        # RAW mapping: no parsing at boot — the tick parses (plugin providers
        # may be registered after settings resolution).
        raw = {"anthropic": 2, "custom:https://llm.example.internal/v1": "7"}
        settings = _settings({"provider_concurrency": raw})
        assert settings.provider_concurrency == raw

    def test_non_dict_values_rejected(self):
        assert _settings({"provider_concurrency": "anthropic"}).provider_concurrency is None
        assert _settings({"provider_concurrency": ["anthropic"]}).provider_concurrency is None
        assert _settings({"provider_concurrency": 4}).provider_concurrency is None

    def test_empty_dict_is_none(self):
        assert _settings({"provider_concurrency": {}}).provider_concurrency is None

    def test_boot_log_line(self, caplog):
        import logging

        with caplog.at_level(logging.INFO):
            _settings({"provider_concurrency": {"anthropic": 2}})
        assert any("provider_concurrency=anthropic:2" in r.getMessage() for r in caplog.records)

    def test_boot_log_line_sanitizes_url_keys(self, caplog):
        """R1 scope F2/Q-F2c: the boot INFO line prints keys in their
        log-safe form — userinfo/query in a custom:<url> key never reach the
        log."""
        import logging

        with caplog.at_level(logging.INFO):
            _settings({"provider_concurrency": {
                "custom:https://bob:hunter2@llm.example.internal/v1?token=abc": 2}})
        msgs = [r.getMessage() for r in caplog.records
                if "provider_concurrency=" in r.getMessage()]
        assert msgs
        assert "custom:https://llm.example.internal/v1:2" in msgs[0]
        for r in caplog.records:
            assert "bob:hunter2" not in r.getMessage()
            assert "token=abc" not in r.getMessage()


class TestTickForwards:
    def test_tick_once_for_board_forwards_mapping(self, tmp_path, monkeypatch):
        """tick_once_for_board passes settings.provider_concurrency into
        dispatch_once unchanged (spy on the real call path)."""
        from dataclasses import asdict
        from gateway import kanban_watchers_dispatcher as kwd
        from hermes_cli import kanban_db_dispatch as kbd

        raw = {"anthropic": 2}
        settings = _settings({"provider_concurrency": raw})
        dispatcher = kwd._KanbanDispatcher(kbd, settings)

        seen: dict = {}
        monkeypatch.setattr(
            kwd, "_quarantine_lifted", lambda *a, **kw: True, raising=False)
        # Fingerprint + quarantine: neutralize the board-DB path entirely.
        monkeypatch.setattr(
            dispatcher, "board_db_fingerprint", lambda slug: ("x", 1, 1))
        monkeypatch.setattr(
            dispatcher, "_quarantine_lifted", lambda slug, fp: True)

        class _FakeConn:
            def close(self):
                pass

        monkeypatch.setattr(
            kwd, "_kbc", lambda: type("KBC", (), {"connect": staticmethod(lambda **kw: _FakeConn())}))
        monkeypatch.setattr(
            kwd, "_kbd", lambda: type("KBD", (), {
                "dispatch_once": staticmethod(lambda conn, **kwargs: seen.update(kwargs) or "RESULT")}))

        assert dispatcher.tick_once_for_board("default") == "RESULT"
        assert seen["provider_concurrency"] == raw
        # The settings dict minus interval forwards every field — the new key
        # rides the existing asdict pass-through, not a bespoke call site.
        expected_keys = set(asdict(settings)) - {"interval"}
        assert expected_keys <= set(seen)
