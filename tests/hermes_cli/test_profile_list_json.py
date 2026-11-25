"""``hermes profile list`` keeps full model ids machine-readable and marks human truncation."""

import argparse
import json

import pytest

from hermes_cli import profile_cmd, profiles
from hermes_cli.profiles import ProfileInfo

MODEL = "deepseek/deepseek-v4.1-flash-1017"
EXPECTED_KEYS = {
    "name",
    "display_name",
    "model",
    "provider",
    "gateway_running",
    "alias",
    "distribution",
    "is_default",
    "active",
}


@pytest.fixture
def fake_profile(tmp_path, monkeypatch):
    profile = ProfileInfo(
        name="default",
        path=tmp_path,
        is_default=True,
        gateway_running=False,
        model=MODEL,
        provider="deepseek",
    )
    monkeypatch.setattr(profiles, "list_profiles", lambda: [profile])
    monkeypatch.setattr(profiles, "get_active_profile_name", lambda: "default")
    monkeypatch.setattr(profile_cmd, "_unfinished_deletes", lambda: [])
    monkeypatch.setattr(profile_cmd, "_shared_credential_warnings", lambda _profiles: [])
    return profile


def test_json_untruncated(fake_profile, capsys):
    profile_cmd._profile_list(argparse.Namespace(json=True))
    captured = capsys.readouterr()
    rows = json.loads(captured.out)
    assert len(rows) == 1
    assert rows[0]["model"] == MODEL
    assert set(rows[0]) == EXPECTED_KEYS


def test_human_table_ellipsis(fake_profile, capsys):
    profile_cmd._profile_list(argparse.Namespace(json=False))
    out = capsys.readouterr().out
    assert MODEL not in out
    assert "…" in out
