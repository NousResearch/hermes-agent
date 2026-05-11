from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SANDBOX_DIR = REPO_ROOT / "docker" / "daimon-sandbox"


def _compose():
    return yaml.safe_load((SANDBOX_DIR / "docker-compose.yml").read_text(encoding="utf-8"))


def test_compose_mounts_token_only_into_broker():
    services = _compose()["services"]
    sandbox_volumes = services["daimon-sandbox"]["volumes"]
    broker_volumes = services["daimon-github-broker"]["volumes"]

    assert not any("/run/secrets/gh_token" in volume for volume in sandbox_volumes)
    assert any("/run/secrets/gh_token" in volume for volume in broker_volumes)
    assert any("GH_TOKEN_PATH:?" in volume for volume in broker_volumes)


def test_compose_uses_shared_network_without_socket_or_token_mounts():
    services = _compose()["services"]

    assert services["daimon-sandbox"]["networks"] == ["daimon-net"]
    assert services["daimon-github-broker"]["networks"] == ["daimon-net"]
    assert "gh-broker-socket" not in _compose().get("volumes", {})
    assert not any("/run/daimon-gh" in volume for volume in services["daimon-sandbox"].get("volumes", []))


def test_agent_dockerfile_has_no_credential_helper_or_server():
    dockerfile = (SANDBOX_DIR / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM base AS agent" in dockerfile
    assert "USER agent" in dockerfile
    assert "credential-server" not in dockerfile
    assert "git-credential-daimon" not in dockerfile
    assert "credential.helper daimon" not in dockerfile


def test_removed_token_extracting_files_are_absent():
    assert not (SANDBOX_DIR / "credential-server.c").exists()
    assert not (SANDBOX_DIR / "git-credential-daimon").exists()
    assert not (SANDBOX_DIR / "gh-wrapper.sh").exists()
