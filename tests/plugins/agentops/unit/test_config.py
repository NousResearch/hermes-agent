from __future__ import annotations

from pathlib import Path

from plugins.agentops.control.config import load_agentops_config
from plugins.agentops.control.models import AuthorityMode


def test_missing_config_is_explicitly_observe_only(tmp_path):
    config = load_agentops_config(tmp_path / "missing.yaml")

    assert config.default_authority is AuthorityMode.OBSERVE_ONLY
    assert config.global_write_enabled is False
    assert "config_missing" in config.safe_start_reasons


def test_invalid_config_stays_observe_only(tmp_path):
    path = tmp_path / "agentops.yaml"
    path.write_text("safety: [", encoding="utf-8")

    config = load_agentops_config(path)

    assert config.default_authority is AuthorityMode.OBSERVE_ONLY
    assert config.global_write_enabled is False
    assert config.safe_start_reasons == ("config_invalid",)


def test_write_request_is_recorded_but_never_enabled(tmp_path):
    path = tmp_path / "agentops.yaml"
    path.write_text("safety:\n  global_write_enabled: true\n", encoding="utf-8")

    config = load_agentops_config(path)

    assert config.default_authority is AuthorityMode.OBSERVE_ONLY
    assert config.global_write_enabled is False
    assert "write_requested_but_disabled" in config.safe_start_reasons


def test_socket_outside_agentops_state_directory_is_rejected(tmp_path):
    path = tmp_path / "agentops.yaml"
    path.write_text(
        "control_plane:\n  socket_path: /tmp/not-agentops.sock\nstorage:\n  sqlite_path: state/state.db\n",
        encoding="utf-8",
    )

    config = load_agentops_config(path)

    assert config.socket_path == Path("/private/tmp/not-agentops.sock")
    assert "socket_outside_state_dir" in config.safe_start_reasons
    assert config.state_dir_safe is False
