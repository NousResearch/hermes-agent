"""A config.yaml without ``_config_version`` is current-schema content that was never stamped:
the installers seed it from cli-config.yaml.example and targeted writers (``rabbit config set``,
/personality) never stamp. The one-time migration ladder must not treat it as a v0 install —
its value- and absence-based steps would overwrite what the user chose."""

import shutil
from pathlib import Path

import pytest
import rabbit_yaml as yaml

TEMPLATE = Path(__file__).resolve().parents[2] / "cli-config.yaml.example"

USER_CHOICES = {
    "delegation.max_concurrent_children": "3",
    "delegation.max_iterations": "50",
    "display.background_process_notifications": "all",
    "agent.verify_on_stop": "true",
    "curator.stale_after_days": "30",
    "curator.archive_after_days": "90",
    "model_catalog.ttl_hours": "24",
}
WATCHED = [*USER_CHOICES, "display.personality", "plugins.enabled"]


@pytest.fixture
def rabbit_home(tmp_path, monkeypatch):
    home = tmp_path / ".rabbit"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)  # sibling-profile scan stays in tmp
    monkeypatch.setenv("RABBIT_HOME", str(home))
    return home


def _raw(home: Path) -> dict:
    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}


def _at(raw: dict, dotted: str):
    for part in dotted.split("."):
        raw = raw.get(part) if isinstance(raw, dict) else None
    return raw


def test_update_keeps_user_values_of_an_unversioned_config_and_migrates_legacy_keys(rabbit_home):
    from rabbit_cli.config import DEFAULT_CONFIG, set_config_value
    from rabbit_cli.personality import persist_personality
    from rabbit_cli.update_cmd import _check_and_apply_config_migration

    # Seeded before the template carried a stamp; early-2026 templates shipped this retired key.
    (rabbit_home / "config.yaml").write_text(
        "compression:\n  summary_model: google/gemini-3-flash-preview\n", encoding="utf-8")
    # Installed with `rabbit plugins install` and never enabled.
    plugin = rabbit_home / "plugins" / "notes-helper"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: notes-helper\nversion: 0.1.0\n", encoding="utf-8")
    # User-authored SOUL.md section that happens to carry the v41 heading.
    soul = "# Me\n\n## Messaging other agents\nmy own notes\n\n## Prefs\nkeep\n"
    (rabbit_home / "SOUL.md").write_text(soul, encoding="utf-8")
    assert persist_personality("kawaii")
    for key, value in USER_CHOICES.items():
        set_config_value(key, value)
    before = _raw(rabbit_home)

    _check_and_apply_config_migration()

    after = _raw(rabbit_home)
    assert {k: _at(after, k) for k in WATCHED} == {k: _at(before, k) for k in WATCHED}
    assert "summary_model" not in after["compression"]
    assert _at(after, "auxiliary.compression.model") == _at(before, "compression.summary_model")
    assert after["_config_version"] == DEFAULT_CONFIG["_config_version"]
    assert (rabbit_home / "SOUL.md").read_text(encoding="utf-8") == soul


def test_config_seeded_from_the_template_reads_as_current(rabbit_home):
    """install.sh, install.ps1, docker/stage2-hook.sh and `rabbit doctor --fix` copy the template."""
    from rabbit_cli.config import check_config_version

    shutil.copy(TEMPLATE, rabbit_home / "config.yaml")

    current, latest = check_config_version(raise_on_parse_error=True)
    assert current == latest
