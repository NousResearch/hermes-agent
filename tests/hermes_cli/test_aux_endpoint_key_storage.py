"""An auxiliary custom-endpoint key goes to ``.env``; config.yaml only references it.

Writers: ``hermes model`` -> Configure auxiliary models -> Custom endpoint, and the dashboard's
auxiliary assignment (one task, and all tasks at once).
"""

from __future__ import annotations

import pytest

from hermes_cli.config import get_config_path, load_config

SECRET = "sk-aux-MARKER-SECRET"
URL = "http://127.0.0.1:11434/v1"


def _cli_flow(monkeypatch) -> str:
    import hermes_cli.main_provider_setup as mps
    answers = iter([URL, "qwen3:8b", SECRET])
    monkeypatch.setattr(mps, "_ask", lambda *a, **k: next(answers))
    monkeypatch.setattr(mps, "_prompt_aux_reasoning_effort", lambda *a, **k: None)
    mps._aux_flow_custom_endpoint("compression", {})
    return "compression"


def _dashboard(task: str):
    def run(monkeypatch) -> str:
        from hermes_cli.web_server_config import _apply_aux_assignment_sync
        _apply_aux_assignment_sync(load_config(), "custom", "qwen3:8b", task, URL, SECRET)
        return task or "compression"
    return run


@pytest.mark.parametrize("writer", [_cli_flow, _dashboard("vision"), _dashboard("")],
                         ids=["hermes model aux custom endpoint", "dashboard one task", "dashboard all tasks"])
def test_aux_endpoint_key_is_kept_out_of_config_yaml(monkeypatch, writer):
    get_config_path().write_text("model:\n  provider: openrouter\n  default: x/y\n")

    task = writer(monkeypatch)

    assert SECRET not in get_config_path().read_text()
    assert load_config()["auxiliary"][task]["api_key"] == SECRET


def test_blocked_env_write_stores_neither_the_key_nor_a_dangling_reference(monkeypatch):
    import hermes_cli.config as config
    get_config_path().write_text("model:\n  provider: openrouter\n  default: x/y\n")
    monkeypatch.setattr(config, "_env_write_blocked", lambda *a, **k: True)

    with pytest.raises(RuntimeError):
        _cli_flow(monkeypatch)

    assert SECRET not in get_config_path().read_text()
    assert "HERMES_CUSTOM_AUX" not in get_config_path().read_text()
