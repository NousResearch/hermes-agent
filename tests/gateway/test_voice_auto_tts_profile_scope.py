"""Secondary profiles must sync their own voice.auto_tts, not the launch profile's.

Regression for #127036: ``_sync_voice_mode_state_to_adapter`` read the default
with a bare ``load_config()``, which follows the ambient (launch) HERMES_HOME —
every multiplexed secondary adapter inherited the default profile's value.
"""

import os
from types import SimpleNamespace

from gateway.config import Platform
from gateway.run import GatewayRunner


def _write_config(home, auto_tts):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        f"voice:\n  auto_tts: {str(auto_tts).lower()}\n", encoding="utf-8"
    )


def _make_homes(tmp_path, monkeypatch, default_tts, secondary_tts):
    launch = tmp_path / "launch-home"
    _write_config(launch, default_tts)
    secondary = launch / "profiles" / "secondary"
    _write_config(secondary, secondary_tts)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    # hermes_constants honors HERMES_HOME (and the data-dir suffix override);
    # keep both pointed at the sandbox so no real home is touched.
    if "HERMES_DATA_DIR_SUFFIX" in os.environ:
        monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX")
    return launch


def _make_runner():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._voice_mode = {}
    return runner


def _make_adapter(owner):
    return SimpleNamespace(
        platform=Platform.TELEGRAM,
        _auto_tts_default=False,
        _auto_tts_disabled_chats=set(),
        _auto_tts_enabled_chats=set(),
        _owner_profile=owner,
    )


def test_secondary_adapter_reads_own_auto_tts_true(tmp_path, monkeypatch):
    """Launch=false, secondary=true -> the secondary adapter speaks (#127036)."""
    _make_homes(tmp_path, monkeypatch, default_tts=False, secondary_tts=True)
    runner, adapter = _make_runner(), _make_adapter("secondary")

    runner._sync_voice_mode_state_to_adapter(adapter)

    assert adapter._auto_tts_default is True


def test_secondary_adapter_reads_own_auto_tts_false(tmp_path, monkeypatch):
    """Launch=true, secondary=false -> the secondary adapter stays text (no leak)."""
    _make_homes(tmp_path, monkeypatch, default_tts=True, secondary_tts=False)
    runner, adapter = _make_runner(), _make_adapter("secondary")

    runner._sync_voice_mode_state_to_adapter(adapter)

    assert adapter._auto_tts_default is False


def test_default_adapter_keeps_launch_home_value(tmp_path, monkeypatch):
    """Owner None (default profile) still reads the ambient launch home."""
    _make_homes(tmp_path, monkeypatch, default_tts=True, secondary_tts=False)
    runner, adapter = _make_runner(), _make_adapter(None)

    runner._sync_voice_mode_state_to_adapter(adapter)

    assert adapter._auto_tts_default is True


def test_unknown_owner_profile_fails_closed(tmp_path, monkeypatch):
    """An unresolvable owner never crashes the connect path; default off."""
    _make_homes(tmp_path, monkeypatch, default_tts=True, secondary_tts=True)
    runner, adapter = _make_runner(), _make_adapter("no-such-profile")

    runner._sync_voice_mode_state_to_adapter(adapter)

    assert adapter._auto_tts_default is False
