"""`plugins disable` must not need an environment admission (#126711).

Disabling a plugin only shrinks the enabled set, so there is nothing new to
resolve. When the install workspace has no ``pm/uv.lock`` (failed-takeover
residue), the PM sync raises ``FileNotFoundError`` and admission wraps it in
``AdmissionRefused`` — which used to make every disable abort before touching
config. The disable path now persists the deny-list directly; enable still
goes through admission.
"""

from __future__ import annotations

import pytest

from hermes_cli import plugins_cmd
from hermes_cli.config import load_config
from hermes_cli.plugins_admission import AdmissionRefused
from hermes_cli.plugins_discovery import collect_directory_manifests, gate_manifest


@pytest.fixture
def home(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    (hermes_home / "plugins").mkdir(parents=True)
    (hermes_home / "config.yaml").write_text(
        "plugins:\n  enabled: []\n  disabled: []\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


@pytest.fixture
def broken_admission(monkeypatch):
    """Simulate the #126711 workspace: sync_venv raises FileNotFoundError for
    the missing pm/uv.lock, which admission wraps in AdmissionRefused."""
    calls = []

    def admit(enabled, disabled, **kwargs):
        calls.append((sorted(enabled), sorted(disabled)))
        raise AdmissionRefused(
            "[Errno 2] No such file or directory: '.../workspace/pm/uv.lock'")

    monkeypatch.setattr(
        "hermes_cli.plugins_admission.admit_plugin_set_change", admit)
    return calls


def _lists():
    plugins = load_config().get("plugins") or {}
    return set(plugins.get("enabled") or []), set(plugins.get("disabled") or [])


def test_disable_persists_without_environment_admission(home, broken_admission):
    """The reported bug: disable aborts with AdmissionRefused, config untouched."""
    key = plugins_cmd._resolve_plugin_key("photon-platform")
    assert key is not None

    plugins_cmd.cmd_disable("photon-platform")

    assert broken_admission == [], "disable must not attempt env resolution"
    enabled, disabled = _lists()
    assert key in disabled and key not in enabled
    photon = next(m for m in collect_directory_manifests() if m.key == key)
    assert gate_manifest(photon, disabled, enabled).error == "disabled via config"


def test_enable_still_requires_admission(home, broken_admission):
    """Enable resolves something new, so a broken environment must still refuse."""
    with pytest.raises(AdmissionRefused):
        plugins_cmd.cmd_enable("photon-platform")

    assert broken_admission != [], "enable must attempt env resolution"
    enabled, _disabled = _lists()
    assert "photon-platform" not in enabled
