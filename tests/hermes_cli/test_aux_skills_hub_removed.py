"""``auxiliary.skills_hub`` is gone: no picker offers it and a leftover block is inert.

Skills Hub search/install never called an LLM, so the slot was a silent no-op (#40703,
#142403). The web and desktop rows are pinned by their own ``auxTaskRows`` tests.
"""

from __future__ import annotations

import hermes_agent.hermes_yaml as yaml
import pytest
from fastapi import HTTPException


def test_no_backend_surface_offers_skills_hub_slot():
    from hermes_agent.hermes_cli.config_defaults import DEFAULT_CONFIG
    from hermes_agent.hermes_cli.main_provider_setup import _AUX_TASKS
    from hermes_agent.hermes_cli.web_server_config import _AUX_TASK_SLOTS

    assert "skills_hub" not in DEFAULT_CONFIG["auxiliary"]
    assert "skills_hub" not in {key for key, _name, _desc in _AUX_TASKS}
    assert "skills_hub" not in _AUX_TASK_SLOTS


def test_leftover_skills_hub_block_is_kept_but_never_acted_on(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    leftover = {"provider": "openrouter", "model": "some/model"}
    (home / "config.yaml").write_text(yaml.safe_dump({"auxiliary": {"skills_hub": leftover}}))
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_agent.hermes_cli.config import load_config
    from hermes_agent.hermes_cli.web_server_config import _apply_aux_assignment_sync, _stale_aux_pins

    cfg = load_config()
    assert cfg["auxiliary"]["skills_hub"]["model"] == "some/model"
    # Switching main away from openrouter must not nudge the user to reset a dead slot.
    assert _stale_aux_pins(cfg, "anthropic") == []
    with pytest.raises(HTTPException) as exc:
        _apply_aux_assignment_sync(cfg, "openrouter", "m", "skills_hub", "", "")
    assert exc.value.status_code == 400
    # "All auxiliary tasks" leaves the dead block exactly as the user wrote it.
    _apply_aux_assignment_sync(cfg, "openrouter", "m", "", "", "")
    assert cfg["auxiliary"]["skills_hub"] == leftover
