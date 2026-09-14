"""Signal-owned cases relocated from cross-cutting gateway/tools tests."""

import inspect

import agent.secret_scope as ss
from gateway.config import PlatformConfig
from plugins.platforms.signal import adapter as sig
from plugins.platforms.signal.adapter import SignalAdapter, _ext_to_mime


def _served(monkeypatch, default_env: dict, secondary_env: dict, build):
    """``(standalone_value, served_value)`` — same contract as gateway profile-parity helper."""
    for k, v in secondary_env.items():
        monkeypatch.setenv(k, v)
    standalone = build()
    for k in secondary_env:
        monkeypatch.delenv(k, raising=False)
    for k, v in default_env.items():
        monkeypatch.setenv(k, v)
    ss.set_multiplex_active(True)
    token = ss.set_secret_scope(dict(secondary_env))
    try:
        served = build()
    finally:
        ss.reset_secret_scope(token)
    return standalone, served


class TestSignalMimeParity:
    """Historical _EXT_TO_MIME table from signal.py, verbatim."""

    HISTORICAL = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".gif": "image/gif", ".webp": "image/webp",
        ".ogg": "audio/ogg", ".mp3": "audio/mpeg", ".wav": "audio/wav",
        ".m4a": "audio/mp4", ".aac": "audio/aac",
        ".mp4": "video/mp4", ".pdf": "application/pdf",
        ".zip": "application/zip",
    }

    def test_table(self):
        for ext, expected in sorted(self.HISTORICAL.items()):
            assert _ext_to_mime(ext) == expected
            assert _ext_to_mime(ext.upper()) == expected


def test_signal_reactions_profile_parity(monkeypatch):
    """SIGNAL_REACTIONS under a served secondary scope matches standalone resolution."""
    var, default_value, secondary_value = "SIGNAL_REACTIONS", "true", "false"

    def resolve():
        adapter = object.__new__(SignalAdapter)
        adapter.dm_allow_from = {"*"}
        return adapter._reactions_enabled()

    for name in (var, "SIGNAL_ACCOUNT"):
        monkeypatch.delenv(name, raising=False)
    creds = {"SIGNAL_ACCOUNT": "+1"}
    standalone, served = _served(
        monkeypatch, {var: default_value, **creds}, {var: secondary_value, **creds}, resolve)
    assert served == standalone, f"{var}: served={served!r} standalone={standalone!r}"


def test_signal_startup_gate_reads_the_profile_env(monkeypatch):
    """A secondary whose SIGNAL_HTTP_URL/SIGNAL_ACCOUNT live only in its own .env must pass the startup
    gate served exactly as it does standalone; a secondary WITHOUT them must not borrow the default's."""
    for k in ("SIGNAL_HTTP_URL", "SIGNAL_ACCOUNT"):
        monkeypatch.delenv(k, raising=False)
    ss.set_multiplex_active(True)
    token = ss.set_secret_scope({"SIGNAL_HTTP_URL": "http://secondary.invalid:8080", "SIGNAL_ACCOUNT": "+15550000001"})
    try:
        assert sig.validate_signal_config(PlatformConfig(enabled=True)) is True
    finally:
        ss.reset_secret_scope(token)
    monkeypatch.setenv("SIGNAL_HTTP_URL", "http://default.invalid:8080")
    monkeypatch.setenv("SIGNAL_ACCOUNT", "+15550000000")
    token = ss.set_secret_scope({})
    try:
        assert sig.validate_signal_config(PlatformConfig(enabled=True)) is False
    finally:
        ss.reset_secret_scope(token)


def test_no_adapter_redeclares_shared_utilities():
    module = inspect.getmodule(SignalAdapter)
    for name in ("_cancel_task", "_is_duplicate", "_bounded_put"):
        assert not hasattr(SignalAdapter, name)
        assert not hasattr(module, name)


def test_send_image_accepts_metadata():
    params = inspect.signature(SignalAdapter.send_image).parameters
    assert "metadata" in params or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    ), "SignalAdapter.send_image must accept metadata"


class TestSignalDelegatesToCentralSniffer:
    """signal._guess_extension audio branches delegate to the shared module."""

    def test_signal_uses_shared_sniffer(self, monkeypatch):
        M4A = b"\x00\x00\x00\x18ftypm4a " + b"\x00" * 8
        calls = []
        real = sig.sniff_container

        def _spy(data):
            calls.append(data[:4])
            return real(data)

        monkeypatch.setattr(sig, "sniff_container", _spy)
        assert sig._guess_extension(M4A) == ".m4a"
        assert calls, "signal._guess_extension did not delegate to the central sniffer"
