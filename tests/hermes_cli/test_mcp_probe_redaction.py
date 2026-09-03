"""MCP test/dashboard must not fingerprint Authorization credentials (#97460)."""

import argparse
import itertools
import json

import pytest
import yaml


SYNTHETIC = "SYNTHETIC_MCP_BEARER_NOT_A_SECRET_123456"
SYNTHETIC_PREFIX = "SYNTHETIC_MCP_BEARER"
SYNTHETIC_SUFFIX = "SECRET_123456"
HEADER = f"Authorization: Bearer {SYNTHETIC}"
OPAQUE_API_KEY = "opaquecredential1234567890ABCDEF"
OPAQUE_PREFIX = OPAQUE_API_KEY[:6]
OPAQUE_SUFFIX = OPAQUE_API_KEY[-4:]


def _assert_fully_redacted(text: str) -> None:
    assert SYNTHETIC not in text
    assert SYNTHETIC_PREFIX not in text
    assert SYNTHETIC_SUFFIX not in text
    assert OPAQUE_API_KEY not in text
    assert OPAQUE_PREFIX not in text
    assert OPAQUE_SUFFIX not in text


def _make_args(**kwargs):
    defaults = {
        "name": "test-server",
        "url": None,
        "mcp_command": None,
        "args": None,
        "auth": None,
        "preset": None,
        "env": None,
        "mcp_action": None,
        "connect_timeout": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: tmp_path)
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: config_path)
    monkeypatch.setattr("hermes_cli.config.get_env_path", lambda: env_path)
    return tmp_path


def _seed_config(tmp_path, mcp_servers):
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump({"mcp_servers": mcp_servers, "_config_version": 9}, f)


class TestRedactMcpProbeText:
    def test_authorization_header_is_fully_replaced(self):
        from hermes_cli.mcp_config import redact_mcp_probe_text

        out = redact_mcp_probe_text(f"401 {HEADER}")
        _assert_fully_redacted(out)
        assert "Authorization:" in out
        assert "Bearer ***" in out

    def test_bare_bearer_scheme_is_fully_replaced(self):
        from hermes_cli.mcp_config import redact_mcp_probe_text

        out = redact_mcp_probe_text(f"probe Bearer {SYNTHETIC} failed")
        _assert_fully_redacted(out)
        assert "Bearer ***" in out

    def test_opaque_api_key_header_is_fully_replaced(self):
        from hermes_cli.mcp_config import redact_mcp_probe_text

        out = redact_mcp_probe_text(f"connect failed: X-Api-Key: {OPAQUE_API_KEY}")
        _assert_fully_redacted(out)
        assert "X-Api-Key: ***" in out

    def test_digest_authorization_params_are_fully_replaced(self):
        from hermes_cli.mcp_config import redact_mcp_probe_text

        out = redact_mcp_probe_text(
            f'Authorization: Digest username="u", response="{OPAQUE_API_KEY}"'
        )
        _assert_fully_redacted(out)
        assert "Digest ***" in out

    def test_digest_response_first_does_not_orphan_quoted_value(self):
        from hermes_cli.mcp_config import redact_mcp_probe_text

        out = redact_mcp_probe_text(
            f'Authorization: Digest response="{OPAQUE_API_KEY}", username="u"'
        )
        _assert_fully_redacted(out)
        assert "Digest ***" in out
        assert "response=" not in out

    def test_python_mapping_api_key_is_fully_replaced(self):
        from hermes_cli.mcp_config import redact_mcp_probe_text

        payload = {"X-Api-Key": OPAQUE_API_KEY}
        out = redact_mcp_probe_text(f"headers={payload!r}")
        _assert_fully_redacted(out)
        assert "***" in out

    def test_json_mapping_api_key_is_fully_replaced(self):
        from hermes_cli.mcp_config import redact_mcp_probe_text

        out = redact_mcp_probe_text(json.dumps({"X-Api-Key": OPAQUE_API_KEY}))
        _assert_fully_redacted(out)
        assert "***" in out

    @pytest.mark.parametrize(
        "params",
        list(
            itertools.permutations(
                [
                    ("username", "u"),
                    ("response", OPAQUE_API_KEY),
                    ("opaque", OPAQUE_API_KEY),
                ]
            )
        ),
    )
    @pytest.mark.parametrize("quoted_values", [True, False])
    def test_digest_parameter_order_and_quoting(self, params, quoted_values):
        from hermes_cli.mcp_config import redact_mcp_probe_text

        parts = []
        for key, value in params:
            parts.append(f'{key}="{value}"' if quoted_values else f"{key}={value}")
        wire = "Authorization: Digest " + ", ".join(parts)
        out = redact_mcp_probe_text(wire)
        _assert_fully_redacted(out)
        assert "Digest ***" in out

    @pytest.mark.parametrize("key_quote,value_quote", [
        ("'", "'"),
        ('"', '"'),
        ("'", '"'),
        ('"', "'"),
    ])
    def test_mapping_quotes_on_api_key(self, key_quote, value_quote):
        from hermes_cli.mcp_config import redact_mcp_probe_text

        text = (
            "headers={"
            f"{key_quote}X-Api-Key{key_quote}: "
            f"{value_quote}{OPAQUE_API_KEY}{value_quote}"
            "}"
        )
        out = redact_mcp_probe_text(text)
        _assert_fully_redacted(out)
        assert "***" in out

    def test_mapping_digest_authorization_is_fully_replaced(self):
        from hermes_cli.mcp_config import redact_mcp_probe_text

        value = f'Digest response="{OPAQUE_API_KEY}", username="u"'
        out = redact_mcp_probe_text(f"headers={{'Authorization': {value!r}}}")
        _assert_fully_redacted(out)
        assert "Digest ***" in out

    def test_shared_secret_header_vocabulary_is_fully_replaced(self):
        from agent.redact import _SECRET_HEADER_NAMES
        from hermes_cli.mcp_config import redact_mcp_probe_text

        inner = _SECRET_HEADER_NAMES
        if inner.startswith("(?:") and inner.endswith(")"):
            inner = inner[3:-1]
        for name in inner.split("|"):
            out = redact_mcp_probe_text({name: OPAQUE_API_KEY}.__repr__())
            _assert_fully_redacted(out)

    def test_header_display_masks_opaque_api_key(self):
        from hermes_cli.mcp_config import redact_mcp_header_display

        assert redact_mcp_header_display("X-Api-Key", OPAQUE_API_KEY) == "***"
        assert redact_mcp_header_display(
            "Authorization", f"Bearer ${{MCP_TEST_TOKEN}}; backup={SYNTHETIC}"
        ) == "***"

    def test_header_display_keeps_pure_env_template(self):
        from hermes_cli.mcp_config import redact_mcp_header_display

        assert redact_mcp_header_display(
            "Authorization", "Bearer ${MCP_TEST_TOKEN}"
        ) == "Bearer ${MCP_TEST_TOKEN}"


class TestProbeHelperRedactsBeforeRaise:
    def test_probe_exception_leaving_helper_is_already_safe(self, monkeypatch):
        import tools.mcp_tool as mcp_tool
        from hermes_cli.mcp_config import _probe_single_server

        monkeypatch.setattr(mcp_tool, "_ensure_mcp_loop", lambda: None)
        monkeypatch.setattr(mcp_tool, "_stop_mcp_loop_if_idle", lambda: None)

        def boom(coro, timeout):
            coro.close()
            raise RuntimeError(f"connect failed: {HEADER}")

        monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", boom)

        with pytest.raises(Exception) as caught:
            _probe_single_server("ink", {"url": "https://mcp.example/mcp"})
        _assert_fully_redacted(str(caught.value))
        assert "Bearer ***" in str(caught.value)

    def test_probe_exception_redacts_opaque_api_key(self, monkeypatch):
        import tools.mcp_tool as mcp_tool
        from hermes_cli.mcp_config import _probe_single_server

        monkeypatch.setattr(mcp_tool, "_ensure_mcp_loop", lambda: None)
        monkeypatch.setattr(mcp_tool, "_stop_mcp_loop_if_idle", lambda: None)

        def boom(coro, timeout):
            coro.close()
            raise RuntimeError(f"connect failed: X-Api-Key: {OPAQUE_API_KEY}")

        monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", boom)

        with pytest.raises(Exception) as caught:
            _probe_single_server("ink", {"url": "https://mcp.example/mcp"})
        _assert_fully_redacted(str(caught.value))
        assert "X-Api-Key: ***" in str(caught.value)

    def test_probe_exception_redacts_digest_response_first(self, monkeypatch):
        import tools.mcp_tool as mcp_tool
        from hermes_cli.mcp_config import _probe_single_server

        monkeypatch.setattr(mcp_tool, "_ensure_mcp_loop", lambda: None)
        monkeypatch.setattr(mcp_tool, "_stop_mcp_loop_if_idle", lambda: None)

        def boom(coro, timeout):
            coro.close()
            raise RuntimeError(
                f'connect failed: Authorization: Digest '
                f'response="{OPAQUE_API_KEY}", username="u"'
            )

        monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", boom)

        with pytest.raises(Exception) as caught:
            _probe_single_server("ink", {"url": "https://mcp.example/mcp"})
        _assert_fully_redacted(str(caught.value))
        assert "Digest ***" in str(caught.value)

    def test_probe_exception_redacts_python_mapping_api_key(self, monkeypatch):
        import tools.mcp_tool as mcp_tool
        from hermes_cli.mcp_config import _probe_single_server

        monkeypatch.setattr(mcp_tool, "_ensure_mcp_loop", lambda: None)
        monkeypatch.setattr(mcp_tool, "_stop_mcp_loop_if_idle", lambda: None)

        def boom(coro, timeout):
            coro.close()
            raise RuntimeError(f"headers={{'X-Api-Key': '{OPAQUE_API_KEY}'}}")

        monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", boom)

        with pytest.raises(Exception) as caught:
            _probe_single_server("ink", {"url": "https://mcp.example/mcp"})
        _assert_fully_redacted(str(caught.value))
        assert "***" in str(caught.value)


class TestCmdMcpTestRedaction:
    def test_success_display_keeps_env_template(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setenv("MCP_TEST_TOKEN", SYNTHETIC)
        _seed_config(tmp_path, {
            "ink": {
                "url": "https://mcp.example/mcp",
                "headers": {"Authorization": "Bearer ${MCP_TEST_TOKEN}"},
            },
        })
        monkeypatch.setattr(
            "hermes_cli.mcp_config._probe_single_server",
            lambda *a, **k: [("ping", "Ping")],
        )
        from hermes_cli.mcp_config import cmd_mcp_test

        cmd_mcp_test(argparse.Namespace(name="ink"))
        out = capsys.readouterr().out
        _assert_fully_redacted(out)
        assert "Connected" in out
        assert "${MCP_TEST_TOKEN}" in (tmp_path / "config.yaml").read_text()

    def test_success_display_masks_literal_header(self, tmp_path, capsys, monkeypatch):
        _seed_config(tmp_path, {
            "ink": {
                "url": "https://mcp.example/mcp",
                "headers": {"Authorization": f"Bearer {SYNTHETIC}"},
            },
        })
        monkeypatch.setattr(
            "hermes_cli.mcp_config._probe_single_server",
            lambda *a, **k: [("ping", "Ping")],
        )
        from hermes_cli.mcp_config import cmd_mcp_test

        cmd_mcp_test(argparse.Namespace(name="ink"))
        out = capsys.readouterr().out
        _assert_fully_redacted(out)
        assert "Authorization:" in out

    def test_success_display_masks_opaque_api_key_header(self, tmp_path, capsys, monkeypatch):
        _seed_config(tmp_path, {
            "ink": {
                "url": "https://mcp.example/mcp",
                "headers": {"X-Api-Key": OPAQUE_API_KEY},
            },
        })
        monkeypatch.setattr(
            "hermes_cli.mcp_config._probe_single_server",
            lambda *a, **k: [("ping", "Ping")],
        )
        from hermes_cli.mcp_config import cmd_mcp_test

        cmd_mcp_test(argparse.Namespace(name="ink"))
        out = capsys.readouterr().out
        _assert_fully_redacted(out)
        assert "X-Api-Key:" in out
        assert "***" in out

    def test_success_display_masks_mixed_env_and_literal(self, tmp_path, capsys, monkeypatch):
        _seed_config(tmp_path, {
            "ink": {
                "url": "https://mcp.example/mcp",
                "headers": {
                    "Authorization": f"Bearer ${{MCP_TEST_TOKEN}}; backup={SYNTHETIC}",
                },
            },
        })
        monkeypatch.setattr(
            "hermes_cli.mcp_config._probe_single_server",
            lambda *a, **k: [("ping", "Ping")],
        )
        from hermes_cli.mcp_config import cmd_mcp_test

        cmd_mcp_test(argparse.Namespace(name="ink"))
        out = capsys.readouterr().out
        _assert_fully_redacted(out)
        assert "***" in out

    def test_probe_exception_is_redacted(self, tmp_path, capsys, monkeypatch):
        _seed_config(tmp_path, {
            "ink": {"url": "https://mcp.example/mcp"},
        })

        def boom(*a, **k):
            raise RuntimeError(f"connect failed: {HEADER}")

        monkeypatch.setattr("hermes_cli.mcp_config._probe_single_server", boom)
        from hermes_cli.mcp_config import cmd_mcp_test

        cmd_mcp_test(argparse.Namespace(name="ink"))
        out = capsys.readouterr().out
        _assert_fully_redacted(out)
        assert "Connection failed" in out
        assert "Bearer ***" in out


class TestDashboardMcpTestRedaction:
    def test_probe_error_json_is_redacted(self, tmp_path, monkeypatch):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
        import hermes_cli.mcp_config as mcp_config

        _seed_config(tmp_path, {
            "ink": {"url": "https://mcp.example/mcp"},
        })

        def boom(name, config, connect_timeout=30, details=None):
            raise RuntimeError(f"connect failed: {HEADER}")

        monkeypatch.setattr(mcp_config, "_probe_single_server", boom)
        monkeypatch.setattr(mcp_config, "_get_mcp_servers", lambda: {
            "ink": {"url": "https://mcp.example/mcp"},
        })

        client = TestClient(app)
        client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
        resp = client.post("/api/mcp/servers/ink/test")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is False
        _assert_fully_redacted(body["error"])
        assert "Bearer ***" in body["error"]
        assert SYNTHETIC not in resp.text

    def test_probe_error_json_redacts_digest_and_mapping(self, tmp_path, monkeypatch):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
        import hermes_cli.mcp_config as mcp_config

        _seed_config(tmp_path, {
            "ink": {"url": "https://mcp.example/mcp"},
        })

        def boom(name, config, connect_timeout=30, details=None):
            raise RuntimeError(
                f"headers={{'X-Api-Key': '{OPAQUE_API_KEY}'}}; "
                f'Authorization: Digest response="{OPAQUE_API_KEY}", username="u"'
            )

        monkeypatch.setattr(mcp_config, "_probe_single_server", boom)
        monkeypatch.setattr(mcp_config, "_get_mcp_servers", lambda: {
            "ink": {"url": "https://mcp.example/mcp"},
        })

        client = TestClient(app)
        client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
        resp = client.post("/api/mcp/servers/ink/test")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is False
        _assert_fully_redacted(body["error"])
        assert "***" in body["error"]
        assert SYNTHETIC not in resp.text
        assert OPAQUE_API_KEY not in resp.text


class TestSiblingProbeConsumersRedact:
    def test_mcp_add_redacts_probe_exception(self, tmp_path, capsys, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError(f"connect failed: {HEADER}")

        monkeypatch.setattr("hermes_cli.mcp_config._probe_single_server", boom)
        monkeypatch.setattr("hermes_cli.mcp_config._confirm", lambda *a, **k: False)
        from hermes_cli.mcp_config import cmd_mcp_add

        cmd_mcp_add(_make_args(name="ink", url="https://mcp.example/mcp"))
        out = capsys.readouterr().out
        _assert_fully_redacted(out)
        assert "Failed to connect" in out
        assert "Bearer ***" in out

    def test_mcp_login_redacts_probe_exception(self, tmp_path, capsys, monkeypatch):
        _seed_config(tmp_path, {
            "ink": {"url": "https://mcp.example/mcp", "auth": "oauth"},
        })

        def boom(*a, **k):
            raise RuntimeError(f"connect failed: {HEADER}")

        monkeypatch.setattr("hermes_cli.mcp_config._probe_single_server", boom)
        monkeypatch.setattr(
            "tools.mcp_oauth.humanize_oauth_registration_error",
            lambda *a, **k: None,
        )
        from hermes_cli.mcp_config import cmd_mcp_login

        cmd_mcp_login(_make_args(name="ink"))
        out = capsys.readouterr().out
        _assert_fully_redacted(out)
        assert "Authentication failed" in out
        assert "Bearer ***" in out
