from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[3] / "docker" / "daimon-sandbox" / "gh_broker.py"
SPEC = importlib.util.spec_from_file_location("daimon_gh_broker_test_module", MODULE_PATH)
gh_broker = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(gh_broker)


def test_validate_allows_issue_list_for_expected_repo():
    argv = ["issue", "list", "-R", "NousResearch/hermes-agent", "--limit", "1"]
    assert gh_broker.validate_argv(argv) == argv


def test_validate_defaults_missing_repo_to_allowed_repo():
    argv = gh_broker.validate_argv(["issue", "list", "--search", "bug"])
    assert argv == [
        "issue",
        "list",
        "--search",
        "bug",
        "-R",
        "NousResearch/hermes-agent",
    ]


def test_auth_status_returns_brokered_auth_message():
    response = json.loads(
        gh_broker.handle_request(json.dumps({"argv": ["auth", "status"]}).encode(), "token").decode()
    )

    assert response["ok"] is True
    assert response["exit_code"] == 0
    assert "Authenticated via Daimon GitHub broker" in response["stdout"]
    assert "token" not in response["stdout"]


@pytest.mark.parametrize(
    "argv",
    [
        ["auth", "token"],
        ["api", "repos/NousResearch/hermes-agent"],
        ["extension", "install", "owner/ext"],
        ["secret", "list"],
        ["issue", "delete", "1", "-R", "NousResearch/hermes-agent"],
        ["issue", "list", "-R", "Other/repo"],
        ["issue", "list", "-R", "NousResearch/hermes-agent", "--hostname", "github.com"],
        ["issue", "list", "-R", "NousResearch/hermes-agent", "--with-token"],
    ],
)
def test_validate_denies_unsupported_or_extracting_shapes(argv):
    with pytest.raises(gh_broker.BrokerError):
        gh_broker.validate_argv(argv)


def test_handle_request_denial_does_not_return_token(monkeypatch):
    token = "github_pat_secret_for_test"
    payload = json.dumps({"argv": ["auth", "token"]}).encode()

    response = json.loads(gh_broker.handle_request(payload, token).decode())

    assert response["ok"] is False
    assert token not in response["stderr"]
    assert "Denied" in response["stderr"]


def test_handle_request_success_preserves_subprocess_result(monkeypatch):
    def fake_run_gh(argv, token, cwd, timeout_sec):
        assert token == "token"
        assert argv == ["issue", "list", "-R", "NousResearch/hermes-agent"]
        return {"ok": True, "exit_code": 0, "stdout": "[]\n", "stderr": ""}

    monkeypatch.setattr(gh_broker, "run_gh", fake_run_gh)
    payload = json.dumps(
        {"argv": ["issue", "list", "-R", "NousResearch/hermes-agent"], "timeout_sec": 3}
    ).encode()

    response = json.loads(gh_broker.handle_request(payload, "token").decode())

    assert response == {"ok": True, "exit_code": 0, "stdout": "[]\n", "stderr": ""}


def test_run_gh_uses_isolated_config_dir(monkeypatch, tmp_path):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        return gh_broker.subprocess.CompletedProcess(cmd, 0, stdout=b"ok\n", stderr=b"")

    monkeypatch.setattr(gh_broker, "GH_CONFIG_DIR", str(tmp_path / "gh-config"))
    monkeypatch.setattr(gh_broker.subprocess, "run", fake_run)

    result = gh_broker.run_gh(
        ["issue", "list", "-R", "NousResearch/hermes-agent"],
        token="token",
        cwd=None,
        timeout_sec=3,
    )

    assert result["ok"] is True
    assert captured["env"]["GH_TOKEN"] == "token"
    assert captured["env"]["GH_CONFIG_DIR"] == str(tmp_path / "gh-config")
    assert captured["env"]["HOME"] == str(tmp_path)
    assert (tmp_path / "gh-config").is_dir()
