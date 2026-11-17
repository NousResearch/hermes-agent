"""``gateway.refuse_secret_files``: a file holding one of the profile's secret values is not attached.

Accident prevention (a debug log or config dump with a live key attached to a chat), opt-in.
"""

import pytest

from agent.redact_env_values import clear_profile_env_secret_cache
from gateway.platforms import base_secret_files
from gateway.platforms.base import BasePlatformAdapter, validate_media_delivery_path
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

OPAQUE = "q7Wz2pLx9Rv4Kd8Nf3Ty"


@pytest.fixture
def profile(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.media_policy.media_delivery_strict", lambda: False)
    monkeypatch.setattr(base_secret_files, "_VERDICTS", {})
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text(f"ACME_SERVICE_TOKEN={OPAQUE}\n", encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    leaky = work / "debug.log"
    # The value straddles the scanner's chunk boundary: a per-chunk check alone would miss it.
    leaky.write_bytes(b"x" * (base_secret_files._CHUNK - 5) + f" auth {OPAQUE} failed\n".encode())
    clean = work / "report.txt"
    clean.write_text("all good\n", encoding="utf-8")
    token = set_hermes_home_override(str(home))
    clear_profile_env_secret_cache()
    yield home, leaky, clean
    reset_hermes_home_override(token)
    clear_profile_env_secret_cache()


def _set_refuse(home, value):
    (home / "config.yaml").write_text(f"gateway:\n  refuse_secret_files: {value}\n", encoding="utf-8")


@pytest.mark.parametrize("enabled", [False, True])
def test_file_holding_a_profile_secret_is_refused_only_when_opted_in(profile, enabled):
    home, leaky, clean = profile
    _set_refuse(home, "true" if enabled else "false")
    reply = f"Here you go\nMEDIA:{leaky}\nMEDIA:{clean}\n"
    media, _ = BasePlatformAdapter.extract_media(reply)
    kept = BasePlatformAdapter.filter_media_delivery_paths(media)
    kept_names = [p.rsplit("/", 1)[-1] for p, _ in kept]
    assert "report.txt" in kept_names
    assert ("debug.log" in kept_names) is not enabled
    assert (validate_media_delivery_path(str(leaky)) is None) is enabled
