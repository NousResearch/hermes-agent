import pytest

from gateway.restart import (
    GATEWAY_RESTART_APPROVAL_REQUIRED_ENV,
    GATEWAY_RESTART_APPROVED_ENV,
    GATEWAY_RESTART_APPROVAL_REQUIRED_MARKER,
    gateway_restart_approval_required,
    require_gateway_restart_approval,
)


def test_restart_approval_not_required_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv(GATEWAY_RESTART_APPROVAL_REQUIRED_ENV, raising=False)
    monkeypatch.delenv(GATEWAY_RESTART_APPROVED_ENV, raising=False)

    assert gateway_restart_approval_required(tmp_path) is False
    require_gateway_restart_approval(hermes_home=tmp_path)


def test_restart_approval_required_by_env_blocks_without_override(tmp_path, monkeypatch):
    monkeypatch.setenv(GATEWAY_RESTART_APPROVAL_REQUIRED_ENV, "1")
    monkeypatch.delenv(GATEWAY_RESTART_APPROVED_ENV, raising=False)

    with pytest.raises(PermissionError, match="explicit approval is required"):
        require_gateway_restart_approval(source="test restart", hermes_home=tmp_path)


def test_restart_approval_required_by_marker_blocks_without_override(tmp_path, monkeypatch):
    monkeypatch.delenv(GATEWAY_RESTART_APPROVAL_REQUIRED_ENV, raising=False)
    monkeypatch.delenv(GATEWAY_RESTART_APPROVED_ENV, raising=False)
    (tmp_path / GATEWAY_RESTART_APPROVAL_REQUIRED_MARKER).write_text("1")

    with pytest.raises(PermissionError, match="Refusing test restart"):
        require_gateway_restart_approval(source="test restart", hermes_home=tmp_path)


def test_restart_approval_accepts_explicit_argument(tmp_path, monkeypatch):
    monkeypatch.setenv(GATEWAY_RESTART_APPROVAL_REQUIRED_ENV, "1")
    monkeypatch.delenv(GATEWAY_RESTART_APPROVED_ENV, raising=False)

    require_gateway_restart_approval(approved=True, hermes_home=tmp_path)


def test_restart_approval_accepts_single_command_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv(GATEWAY_RESTART_APPROVAL_REQUIRED_ENV, "1")
    monkeypatch.setenv(GATEWAY_RESTART_APPROVED_ENV, "1")

    require_gateway_restart_approval(hermes_home=tmp_path)
