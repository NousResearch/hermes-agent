"""``auxiliary.skills_hub`` is gone: no picker offers it and a leftover block is inert.

Skills Hub search/install never called an LLM, so the slot was a silent no-op (#40703,
#142403). Every surface that lists built-in aux slots (DEFAULT_CONFIG, the ``hermes model``
picker, the dashboard REST allowlist, the web and desktop rows) must agree it is not a slot,
and an existing user config that still carries the block must keep loading.
"""

from __future__ import annotations

from pathlib import Path

import hermes_yaml as yaml

REPO = Path(__file__).resolve().parents[2]
_FRONTEND_ROWS = ("web/src/lib/aux-tasks.ts", "apps/desktop/src/app/settings/aux-task-rows.tsx")


def test_no_surface_offers_skills_hub_slot():
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    from hermes_cli.main_provider_setup import _AUX_TASKS
    from hermes_cli.web_server_config import _AUX_TASK_SLOTS

    assert "skills_hub" not in DEFAULT_CONFIG["auxiliary"]
    assert "skills_hub" not in {key for key, _name, _desc in _AUX_TASKS}
    assert "skills_hub" not in _AUX_TASK_SLOTS
    for rel in _FRONTEND_ROWS:
        assert "skills_hub" not in (REPO / rel).read_text(encoding="utf-8"), rel


def test_leftover_skills_hub_block_still_loads(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump(
        {"auxiliary": {"skills_hub": {"provider": "openrouter", "model": "some/model"}}}))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli.config import load_config
    from hermes_cli.web_server_config import _aux_task_slots

    cfg = load_config()
    assert cfg["auxiliary"]["vision"]["provider"] == "auto"
    assert "skills_hub" not in _aux_task_slots()
