"""Tests for plugins/memory/honcho/cli.py."""

from types import SimpleNamespace


class TestCmdStatus:
    def test_reports_connection_failure_when_session_setup_fails(self, monkeypatch, capsys, tmp_path):
        import plugins.memory.honcho.cli as honcho_cli

        cfg_path = tmp_path / "honcho.json"
        cfg_path.write_text("{}")

        class FakeConfig:
            enabled = True
            api_key = "root-key"
            workspace_id = "hermes"
            host = "hermes"
            base_url = None
            ai_peer = "hermes"
            peer_name = "eri"
            recall_mode = "hybrid"
            user_observe_me = True
            user_observe_others = False
            ai_observe_me = False
            ai_observe_others = True
            write_frequency = "async"
            session_strategy = "per-session"
            context_tokens = 800

            def resolve_session_name(self):
                return "hermes"

        monkeypatch.setattr(honcho_cli, "_read_config", lambda: {"apiKey": "***"})
        monkeypatch.setattr(honcho_cli, "_config_path", lambda: cfg_path)
        monkeypatch.setattr(honcho_cli, "_local_config_path", lambda: cfg_path)
        monkeypatch.setattr(honcho_cli, "_active_profile_name", lambda: "default")
        monkeypatch.setattr(
            "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
            lambda host=None: FakeConfig(),
        )
        monkeypatch.setattr(
            "plugins.memory.honcho.client.get_honcho_client",
            lambda cfg: object(),
        )

        def _boom(hcfg, client):
            raise RuntimeError("Invalid API key")

        monkeypatch.setattr(honcho_cli, "_show_peer_cards", _boom)
        monkeypatch.setitem(__import__("sys").modules, "honcho", SimpleNamespace())

        honcho_cli.cmd_status(SimpleNamespace(all=False))

        out = capsys.readouterr().out
        assert "FAILED (Invalid API key)" in out
        assert "Connection... OK" not in out