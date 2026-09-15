"""Profile-scoped Groupchat dashboard API tests."""
import json

import pytest

fastapi = pytest.importorskip("fastapi")

from plugins.groupchat.dashboard.plugin_api import (  # noqa: E402
    SettingsUpdate,
    get_settings,
    put_settings,
)


@pytest.fixture(autouse=True)
def profile_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("plugins:\n  enabled: [groupchat]\n")


def test_api_exposes_selected_profile_paths(tmp_path):
    data = get_settings(profile="current")
    assert data["persistent_context_directory"] == str(tmp_path / "groupchat")
    assert data["log_directory"] == str(tmp_path / "logs")
    assert data["decision_logs"]["matrix"]["relevance"] == str(
        tmp_path / "logs/matrix-relevance-decisions.jsonl"
    )


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("relevance", "system_patterns"),
        ("relevance", "multiline_patterns"),
        ("pingpong_guard", "silence_patterns"),
    ],
)
def test_api_rejects_invalid_pattern_without_saving(section, key, tmp_path):
    before = (tmp_path / "config.yaml").read_bytes()
    with pytest.raises(fastapi.HTTPException) as failure:
        put_settings(
            SettingsUpdate(settings={section: {key: ["["]}}),
            profile="current",
        )
    assert failure.value.status_code == 422
    assert "line 1" in failure.value.detail
    assert (tmp_path / "config.yaml").read_bytes() == before


def test_api_roundtrips_patterns_and_exposes_defaults(tmp_path):
    initial = get_settings(profile="current")
    settings = initial["settings"]
    settings["pingpong_guard"]["silence_patterns"] = [r"^test{1,3}$"]
    settings["relevance"]["system_patterns"] = []
    put_settings(SettingsUpdate(settings=settings), profile="current")

    loaded = get_settings(profile="current")
    assert loaded["settings"]["pingpong_guard"]["silence_patterns"] == [
        r"^test{1,3}$"
    ]
    assert loaded["settings"]["relevance"]["system_patterns"] == []
    assert loaded["defaults"]["relevance"]["system_patterns"]


def test_dashboard_save_preserves_unrelated_config(tmp_path):
    from hermes_cli.config import read_raw_config

    original = {
        "plugins": {"enabled": ["another-plugin"]},
        "terminal": {"cwd": "/keep/me"},
    }
    (tmp_path / "config.yaml").write_text(json.dumps(original))
    current = get_settings()["settings"]
    current["enabled"] = True

    result = put_settings(SettingsUpdate(settings=current))
    saved = read_raw_config()

    assert result["restart_required"]
    assert saved["terminal"]["cwd"] == "/keep/me"
    assert saved["plugins"]["enabled"] == ["another-plugin", "groupchat"]
