import base64
import json
import os
from pathlib import Path

import yaml
from gateway.modal_runtime import (
    ModalGatewayService,
    _decode_base64_env,
    bootstrap_modal_home,
    named_modal_secret_names,
    normalize_github_token_env,
    render_dashboard_html,
    sanitize_config_for_modal,
    sanitize_env_text_for_modal,
)


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("utf-8")


def test_sanitize_config_for_modal_forces_local_terminal_and_drops_local_urls():
    original = {
        "model": {
            "provider": "openai-codex",
            "default": "gpt-5.4",
            "base_url": "http://localhost:20128/v1",
        },
        "auxiliary": {
            "vision": {"base_url": "http://127.0.0.1:4000/v1"},
            "web_extract": {"base_url": "https://api.example.com/v1"},
        },
        "terminal": {
            "backend": "modal",
            "cwd": ".",
        },
    }

    sanitized = sanitize_config_for_modal(original, project_root="/opt/hermes/hermes-agent")

    assert original["terminal"]["backend"] == "modal"
    assert sanitized["terminal"]["backend"] == "local"
    assert sanitized["terminal"]["cwd"] == "/opt/hermes/hermes-agent"
    assert "base_url" not in sanitized["model"]
    assert "base_url" not in sanitized["auxiliary"]["vision"]
    assert sanitized["auxiliary"]["web_extract"]["base_url"] == "https://api.example.com/v1"


def test_sanitize_env_text_for_modal_drops_localhost_and_modal_credentials():
    raw = """# gateway config
OPENAI_BASE_URL=http://localhost:1234/v1
QUOTED_OPENAI_BASE_URL="http://localhost:2345/v1"
MODAL_TOKEN_ID=abc
MODAL_TOKEN_SECRET=def
HERMES_HOME=/tmp/hermes
MESSAGING_CWD=/tmp/project
TERMINAL_BACKEND=modal
DISCORD_BOT_TOKEN=xyz
OPENAI_API_KEY=sk-live
REMOTE_BASE_URL=https://api.example.com/v1
"""

    sanitized = sanitize_env_text_for_modal(raw)
    sanitized_lines = sanitized.splitlines()

    assert "# gateway config" in sanitized
    assert "OPENAI_BASE_URL=http://localhost:1234/v1" not in sanitized_lines
    assert "QUOTED_OPENAI_BASE_URL" in sanitized
    assert "MODAL_TOKEN_ID" not in sanitized
    assert "MODAL_TOKEN_SECRET" not in sanitized
    assert "HERMES_HOME" not in sanitized
    assert "MESSAGING_CWD" not in sanitized
    assert "TERMINAL_BACKEND" not in sanitized
    assert "DISCORD_BOT_TOKEN=xyz" in sanitized
    assert "OPENAI_API_KEY=sk-live" in sanitized
    assert "REMOTE_BASE_URL=https://api.example.com/v1" in sanitized


def test_render_dashboard_html_includes_runtime_state_and_logs():
    html = render_dashboard_html({
        "runtime_status": {
            "gateway_state": "running",
            "platforms": {"discord": {"state": "running"}},
        },
        "thread_alive": True,
        "gateway_log_tail": "discord connected",
        "error_log_tail": "",
        "last_error": None,
        "hermes_home": "/hermes-state/home",
        "project_root": "/opt/hermes/hermes-agent",
    })

    assert "Hermes Gateway Dashboard" in html
    assert "running" in html
    assert "discord connected" in html


def test_normalize_github_token_env_aliases_nonstandard_secret_keys(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setenv("github-token", "secret-value")

    normalize_github_token_env()

    assert os.environ["GITHUB_TOKEN"] == "secret-value"
    assert os.environ["GH_TOKEN"] == "secret-value"


def test_normalize_github_token_env_prefers_existing_github_token(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "canonical-token")
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setenv("github-token", "alternate-token")

    normalize_github_token_env()

    assert os.environ["GITHUB_TOKEN"] == "canonical-token"
    assert os.environ["GH_TOKEN"] == "canonical-token"


def test_normalize_github_token_env_prefers_existing_gh_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GH_TOKEN", "canonical-token")
    monkeypatch.setenv("github-token", "alternate-token")

    normalize_github_token_env()

    assert os.environ["GITHUB_TOKEN"] == "canonical-token"
    assert os.environ["GH_TOKEN"] == "canonical-token"


def test_named_secret_names_include_github_and_prime_by_default(monkeypatch):
    monkeypatch.delenv("HERMES_MODAL_GITHUB_TOKEN_SECRET", raising=False)
    monkeypatch.delenv("HERMES_MODAL_PRIME_API_KEY_SECRET", raising=False)

    assert named_modal_secret_names() == ["github-token", "PRIME_API_KEY"]


def test_named_secret_names_dedupe_and_skip_empty_values(monkeypatch):
    monkeypatch.setenv("HERMES_MODAL_GITHUB_TOKEN_SECRET", "shared-secret")
    monkeypatch.setenv("HERMES_MODAL_PRIME_API_KEY_SECRET", "shared-secret")

    assert named_modal_secret_names() == ["shared-secret"]


def test_decode_base64_env_returns_none_for_invalid_payload(monkeypatch):
    monkeypatch.setenv("BROKEN_MODAL_ENV", "not-base64")

    assert _decode_base64_env("BROKEN_MODAL_ENV") is None


def test_decode_base64_env_returns_none_for_invalid_utf8(monkeypatch):
    monkeypatch.setenv("BROKEN_MODAL_ENV", base64.b64encode(b"\xff").decode("utf-8"))

    assert _decode_base64_env("BROKEN_MODAL_ENV") is None


def test_sanitize_env_text_for_modal_drops_quoted_local_openai_base_url():
    sanitized = sanitize_env_text_for_modal(
        'OPENAI_BASE_URL="http://localhost:1234/v1"\nOPENAI_API_KEY=sk-live\n'
    )

    assert "OPENAI_BASE_URL" not in sanitized
    assert "OPENAI_API_KEY=sk-live" in sanitized


def test_bootstrap_modal_home_writes_sanitized_files(tmp_path, monkeypatch):
    hermes_home = Path(tmp_path)
    config_yaml = yaml.safe_dump(
        {
            "model": {
                "provider": "openai-codex",
                "base_url": "http://localhost:8000/v1",
            },
            "auxiliary": {
                "vision": {"base_url": "http://127.0.0.1:4000/v1"},
                "web_extract": {"base_url": "https://api.example.com/v1"},
            },
            "terminal": {
                "backend": "modal",
                "cwd": ".",
            },
        }
    )
    env_text = "\n".join(
        [
            "OPENAI_BASE_URL=http://localhost:1234/v1",
            "HERMES_HOME=/tmp/hermes",
            "TERMINAL_BACKEND=modal",
            "OPENAI_API_KEY=sk-live",
            "",
        ]
    )
    auth_text = '{"access_token":"abc"}'
    commit_calls: list[str] = []

    monkeypatch.setenv("HERMES_MODAL_CONFIG_B64", _b64(config_yaml))
    monkeypatch.setenv("HERMES_MODAL_ENV_B64", _b64(env_text))
    monkeypatch.setenv("HERMES_MODAL_AUTH_B64", _b64(auth_text))

    bootstrap_modal_home(
        hermes_home,
        project_root="/opt/hermes/hermes-agent",
        commit_fn=lambda: commit_calls.append("commit"),
    )

    written_config = yaml.safe_load((hermes_home / "config.yaml").read_text())
    assert written_config["terminal"]["backend"] == "local"
    assert written_config["terminal"]["cwd"] == "/opt/hermes/hermes-agent"
    assert "base_url" not in written_config["model"]
    assert "base_url" not in written_config["auxiliary"]["vision"]
    assert written_config["auxiliary"]["web_extract"]["base_url"] == "https://api.example.com/v1"

    written_env = (hermes_home / ".env").read_text()
    assert "OPENAI_BASE_URL" not in written_env
    assert "HERMES_HOME" not in written_env
    assert "TERMINAL_BACKEND" not in written_env
    assert "OPENAI_API_KEY=sk-live" in written_env

    assert (hermes_home / "auth.json").read_text() == auth_text

    bootstrap_metadata = json.loads((hermes_home / "modal-bootstrap.json").read_text())
    assert bootstrap_metadata["project_root"] == "/opt/hermes/hermes-agent"
    assert bootstrap_metadata["has_auth"] is True
    assert "bootstrapped_at" in bootstrap_metadata
    assert commit_calls == ["commit"]


def test_bootstrap_modal_home_ignores_invalid_yaml_config(tmp_path, monkeypatch):
    hermes_home = Path(tmp_path)
    commit_calls: list[str] = []

    monkeypatch.setenv("HERMES_MODAL_CONFIG_B64", _b64("model: ["))
    monkeypatch.delenv("HERMES_MODAL_ENV_B64", raising=False)
    monkeypatch.delenv("HERMES_MODAL_AUTH_B64", raising=False)

    bootstrap_modal_home(
        hermes_home,
        project_root="/opt/hermes/hermes-agent",
        commit_fn=lambda: commit_calls.append("commit"),
    )

    written_config = yaml.safe_load((hermes_home / "config.yaml").read_text())
    assert written_config["terminal"]["backend"] == "local"
    assert written_config["terminal"]["cwd"] == "/opt/hermes/hermes-agent"

    bootstrap_metadata = json.loads((hermes_home / "modal-bootstrap.json").read_text())
    assert bootstrap_metadata["project_root"] == "/opt/hermes/hermes-agent"
    assert bootstrap_metadata["has_auth"] is False
    assert "config_error" in bootstrap_metadata
    assert commit_calls == ["commit"]


def test_modal_gateway_service_snapshot_reads_state_and_logs(tmp_path):
    hermes_home = Path(tmp_path)
    logs_dir = hermes_home / "logs"
    logs_dir.mkdir(parents=True)

    (hermes_home / "gateway_state.json").write_text(
        json.dumps(
            {
                "gateway_state": "running",
                "platforms": {"discord": {"state": "running"}},
            }
        )
    )
    (hermes_home / "gateway.pid").write_text(
        json.dumps(
            {
                "pid": 1234,
                "kind": "hermes-gateway",
                "start_time": 99,
            }
        )
    )
    (logs_dir / "gateway.log").write_text("first line\nsecond line\nthird line\n")
    (logs_dir / "errors.log").write_text("warning line\nerror line\n")

    service = ModalGatewayService(
        hermes_home=hermes_home,
        project_root="/opt/hermes/hermes-agent",
    )
    snapshot = service.snapshot()

    assert snapshot["project_root"] == "/opt/hermes/hermes-agent"
    assert snapshot["hermes_home"] == str(hermes_home)
    assert snapshot["runtime_status"]["gateway_state"] == "running"
    assert snapshot["runtime_status"]["platforms"]["discord"]["state"] == "running"
    assert snapshot["pid_record"]["pid"] == 1234
    assert "third line" in snapshot["gateway_log_tail"]
    assert "error line" in snapshot["error_log_tail"]
    assert snapshot["last_error"] is None


def test_modal_gateway_service_records_bad_project_root(tmp_path):
    hermes_home = Path(tmp_path)
    commit_calls: list[str] = []
    missing_root = hermes_home / "missing-project-root"

    service = ModalGatewayService(
        hermes_home=hermes_home,
        project_root=str(missing_root),
        commit_fn=lambda: commit_calls.append("commit"),
    )

    service._run_gateway()

    assert service._last_error is not None
    assert "FileNotFoundError" in service._last_error
    assert str(missing_root) in service._last_error
    assert commit_calls == ["commit"]
