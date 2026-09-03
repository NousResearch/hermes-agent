import json

import pytest

from plugins.builder_adapter.errors import AdapterError
from plugins.builder_adapter.profile_install import install_isolated_profile


POLICY = {
    "profile": "deepseek-builder",
    "provider": "deepseek",
    "model": "deepseek-v4-pro",
    "fallback_chain": [],
}


def test_installs_effective_profile_only_in_isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "live"))
    target = install_isolated_profile(
        tmp_path / "isolated",
        POLICY,
        isolation_marker="HERMES_BUILDER_TEST_HOME",
    )
    config = json.loads(target.read_text())
    assert config["model"] == {
        "provider": "deepseek",
        "default": "deepseek-v4-pro",
    }
    assert config["fallback_providers"] == []
    assert config["builder_dispatch"]["confinement"] == {
        "kind": "application_tool_mediated",
        "os_sandbox": False,
        "terminal_tools": False,
        "process_tools": False,
    }


def test_refuses_live_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with pytest.raises(AdapterError, match="live Hermes"):
        install_isolated_profile(
            tmp_path, POLICY, isolation_marker="HERMES_BUILDER_TEST_HOME"
        )
