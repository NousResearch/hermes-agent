"""``hermes doctor`` names the Relay plugin files the runtime actually loads.

Relay discovers user and system ``plugins.toml`` files outside the Hermes home, so doctor is the one
Hermes surface that shows a user which files apply. It must agree with what initialization loads.
"""

import pytest

from agent import relay_runtime
from hermes_cli.doctor_config import _check_relay_plugins

ENABLED_OBSERVABILITY = (
    "version = 1\n\n[[components]]\nkind = \"observability\"\nenabled = true\n\n[components.config]\nversion = 4\n"
)


@pytest.mark.parametrize("explicit", [False, True], ids=["discovered", "explicit"])
def test_doctor_lists_the_files_the_runtime_loads(tmp_path, monkeypatch, capsys, explicit):
    relay = pytest.importorskip("nemo_relay")
    if getattr(relay, "_native", None) is None:
        pytest.skip("NeMo Relay native binding is unavailable on this platform")
    user_config = tmp_path / "xdg" / "nemo-relay" / "plugins.toml"
    user_config.parent.mkdir(parents=True)
    user_config.write_text(ENABLED_OBSERVABILITY, encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("ProgramData", str(tmp_path / "programdata"))
    if explicit:
        selected = tmp_path / "selected.toml"
        selected.write_text(ENABLED_OBSERVABILITY, encoding="utf-8")
        monkeypatch.setenv(relay_runtime.RELAY_PLUGINS_CONFIG_ENV, str(selected))
    else:
        monkeypatch.delenv(relay_runtime.RELAY_PLUGINS_CONFIG_ENV, raising=False)

    _check_relay_plugins(False)
    doctor_output = capsys.readouterr().out

    relay_runtime._reset_for_tests()
    host = relay_runtime.RelayRuntime(relay=relay, profile_key="profile")
    try:
        assert host._plugin_configuration_state is relay_runtime._RelayPluginConfigurationState.ACTIVE
        loaded = relay_runtime._activation_config_paths(relay_runtime._PLUGIN_CONFIGURATION._activation)
    finally:
        host.shutdown()
        relay_runtime._reset_for_tests()

    assert (str(user_config) in loaded) is not explicit
    assert "Relay plugins enabled" in doctor_output
    assert all(path in doctor_output for path in loaded)
    if explicit:
        assert str(user_config) not in doctor_output
