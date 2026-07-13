"""Contract tests for the bundled local sxng-search web provider plugin."""
from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest


def _set_config(monkeypatch: pytest.MonkeyPatch, *, command: str = "sxng-search", timeout=45) -> None:
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"web": {"sxng": {"command": command, "timeout": timeout}}},
    )


class TestSxngPluginContract:
    def test_provider_implements_current_abc(self) -> None:
        from agent.web_search_provider import WebSearchProvider
        from plugins.web.sxng.provider import SxngWebSearchProvider

        provider = SxngWebSearchProvider()
        assert isinstance(provider, WebSearchProvider)
        assert provider.name == "sxng"
        assert provider.display_name == "Local sxng-search wrapper"
        assert provider.supports_search() is True
        assert provider.supports_extract() is False

    def test_register_uses_web_provider_hook(self) -> None:
        from plugins.web.sxng import register

        ctx = MagicMock()
        register(ctx)

        ctx.register_web_search_provider.assert_called_once()
        assert ctx.register_web_search_provider.call_args.args[0].name == "sxng"

    def test_setup_schema_has_no_nonsecret_env_fields(self) -> None:
        from plugins.web.sxng.provider import SxngWebSearchProvider

        schema = SxngWebSearchProvider().get_setup_schema()

        assert schema["name"] == "Local sxng-search wrapper"
        assert schema["env_vars"] == []
        assert "search only" in schema["tag"].lower()

    def test_picker_row_is_generated_from_plugin_schema(self) -> None:
        from hermes_cli.tools_config import _plugin_web_search_providers

        rows = _plugin_web_search_providers()
        sxng = next(row for row in rows if row.get("web_backend") == "sxng")

        assert sxng["name"] == "Local sxng-search wrapper"
        assert sxng["web_search_plugin_name"] == "sxng"
        assert sxng["env_vars"] == []


class TestSxngAvailabilityAndConfig:
    def test_available_when_configured_command_is_on_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.web.sxng.provider import SxngWebSearchProvider

        _set_config(monkeypatch, command="custom-sxng")
        monkeypatch.setattr("shutil.which", lambda command: "/usr/bin/custom-sxng" if command == "custom-sxng" else None)

        assert SxngWebSearchProvider().is_available() is True

    def test_unavailable_when_command_cannot_be_resolved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.web.sxng.provider import SxngWebSearchProvider

        _set_config(monkeypatch)
        monkeypatch.setattr("shutil.which", lambda command: None)

        assert SxngWebSearchProvider().is_available() is False

    @pytest.mark.parametrize(
        "configured,expected",
        [(None, 45), ("bad", 45), (0, 1), (-2, 1), (12, 12), (999, 300)],
    )
    def test_timeout_is_config_only_and_bounded(
        self,
        monkeypatch: pytest.MonkeyPatch,
        configured,
        expected: int,
    ) -> None:
        from plugins.web.sxng.provider import _timeout_seconds

        _set_config(monkeypatch, timeout=configured)

        assert _timeout_seconds() == expected


class TestSxngSearch:
    def test_happy_path_normalizes_wrapper_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.web.sxng.provider import SxngWebSearchProvider

        _set_config(monkeypatch, command="custom-sxng", timeout=17)
        monkeypatch.setattr("shutil.which", lambda command: "/usr/bin/custom-sxng")
        completed = MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "results": [
                        {
                            "title": "A",
                            "url": "https://a.example",
                            "content": "desc A",
                        },
                        {
                            "title": "B",
                            "url": "https://b.example",
                            "description": "desc B",
                        },
                    ]
                }
            ),
            stderr="",
        )

        with patch("subprocess.run", return_value=completed) as run:
            result = SxngWebSearchProvider().search("test query", limit=5)

        assert result == {
            "success": True,
            "data": {
                "web": [
                    {
                        "title": "A",
                        "url": "https://a.example",
                        "description": "desc A",
                        "position": 1,
                    },
                    {
                        "title": "B",
                        "url": "https://b.example",
                        "description": "desc B",
                        "position": 2,
                    },
                ]
            },
        }
        assert run.call_args.args[0] == [
            "/usr/bin/custom-sxng",
            "test query",
            "--limit",
            "5",
            "--json",
        ]
        assert run.call_args.kwargs["timeout"] == 17
        assert run.call_args.kwargs["shell"] is False

    def test_accepts_data_web_shape_and_caps_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.web.sxng.provider import SxngWebSearchProvider

        _set_config(monkeypatch)
        monkeypatch.setattr("shutil.which", lambda command: "/usr/bin/sxng-search")
        completed = MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "data": {
                        "web": [
                            {"title": "A", "url": "https://a.example", "snippet": "A"},
                            {"title": "B", "url": "https://b.example", "snippet": "B"},
                        ]
                    }
                }
            ),
            stderr="",
        )

        with patch("subprocess.run", return_value=completed):
            result = SxngWebSearchProvider().search("q", limit=1)

        assert result["success"] is True
        assert len(result["data"]["web"]) == 1
        assert result["data"]["web"][0]["position"] == 1

    def test_missing_command_returns_typed_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.web.sxng.provider import SxngWebSearchProvider

        _set_config(monkeypatch)
        monkeypatch.setattr("shutil.which", lambda command: None)

        result = SxngWebSearchProvider().search("q")

        assert result["success"] is False
        assert "web.sxng.command" in result["error"]

    def test_nonzero_exit_returns_bounded_stderr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.web.sxng.provider import SxngWebSearchProvider

        _set_config(monkeypatch)
        monkeypatch.setattr("shutil.which", lambda command: "/usr/bin/sxng-search")
        completed = MagicMock(returncode=2, stdout="", stderr="backend unavailable")

        with patch("subprocess.run", return_value=completed):
            result = SxngWebSearchProvider().search("q")

        assert result == {"success": False, "error": "backend unavailable"}

    def test_nonzero_exit_redacts_secret_like_stderr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.web.sxng.provider import SxngWebSearchProvider

        _set_config(monkeypatch)
        monkeypatch.setattr("shutil.which", lambda command: "/usr/bin/sxng-search")
        fake_secret = "sk-" + "a" * 24
        completed = MagicMock(
            returncode=2,
            stdout="",
            stderr=f"backend unavailable: {fake_secret}",
        )

        with patch("subprocess.run", return_value=completed):
            result = SxngWebSearchProvider().search("q")

        assert result["success"] is False
        assert fake_secret not in result["error"]

    def test_os_error_does_not_echo_private_command_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from plugins.web.sxng.provider import SxngWebSearchProvider

        private_path = "/private/example/sxng-search"
        _set_config(monkeypatch, command=private_path)
        monkeypatch.setattr("shutil.which", lambda command: private_path)

        with patch("subprocess.run", side_effect=OSError(private_path)):
            result = SxngWebSearchProvider().search("q")

        assert result == {"success": False, "error": "sxng-search failed to start"}
        assert private_path not in result["error"]

    def test_bad_json_returns_typed_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.web.sxng.provider import SxngWebSearchProvider

        _set_config(monkeypatch)
        monkeypatch.setattr("shutil.which", lambda command: "/usr/bin/sxng-search")
        completed = MagicMock(returncode=0, stdout="not-json", stderr="")

        with patch("subprocess.run", return_value=completed):
            result = SxngWebSearchProvider().search("q")

        assert result["success"] is False
        assert "json" in result["error"].lower()

    def test_timeout_returns_typed_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from plugins.web.sxng.provider import SxngWebSearchProvider

        _set_config(monkeypatch, timeout=9)
        monkeypatch.setattr("shutil.which", lambda command: "/usr/bin/sxng-search")

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("sxng-search", 9)):
            result = SxngWebSearchProvider().search("q")

        assert result == {"success": False, "error": "sxng-search timed out after 9 seconds"}


class TestSxngRegistryIntegration:
    def test_real_registry_dispatches_web_search(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        from agent.web_search_registry import _reset_for_tests, register_provider
        from plugins.web.sxng.provider import SxngWebSearchProvider
        from tools import web_tools

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        _set_config(monkeypatch)
        monkeypatch.setattr("shutil.which", lambda command: "/usr/bin/sxng-search")
        monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
        monkeypatch.setattr(web_tools, "_get_search_backend", lambda: "sxng")
        completed = MagicMock(
            returncode=0,
            stdout=json.dumps(
                {"results": [{"title": "Result", "url": "https://example.com"}]}
            ),
            stderr="",
        )

        _reset_for_tests()
        register_provider(SxngWebSearchProvider())
        try:
            with patch("subprocess.run", return_value=completed):
                result = json.loads(web_tools.web_search_tool("integration", limit=1))
        finally:
            _reset_for_tests()

        assert result["success"] is True
        assert result["data"]["web"] == [
            {
                "title": "Result",
                "url": "https://example.com",
                "description": "",
                "position": 1,
            }
        ]
