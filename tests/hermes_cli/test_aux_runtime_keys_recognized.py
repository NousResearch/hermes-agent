"""Regression: runtime-read keys of an auxiliary task block must be a registered
config key, not flagged as unrecognized.

`hermes config set` walks `DEFAULT_CONFIG` to decide whether a dotted key is
recognized, but the aux runtime reads more off an `auxiliary.<task>` block than the
shared `_aux()` template declared — `key_env`/`api_key_env` and `api_mode` in
`agent/auxiliary_client.py::_resolve_task_provider_model`, `fallback_chain` in
`_try_configured_fallback_chain`, `max_concurrency` in `_get_task_max_concurrency`,
plus the per-task `auxiliary.vision.temperature` / `auxiliary.compression.context_length`
that `tools/vision_tools.py` and `agent/agent_init.py` read. Setting any of them printed

    ⚠ 'auxiliary.compression.key_env' is not a recognized config key — it was saved
    anyway, but Hermes may not read it.

which tells the user the opposite of the truth: the value is saved AND read, and for
`key_env` it silently becomes the whole credential story for a pinned aux model.

These tests walk the real user path (`set_config_value` -> notice + persisted value) and
assert the reader/registry agreement generically, so a new block added outside `_aux()`
fails here instead of shipping a new instance of the notice.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import yaml

from hermes_cli.config import DEFAULT_CONFIG, set_config_value

# Keys every aux task block must declare because the runtime reads them off the block.
_SHARED_RUNTIME_KEYS = ("key_env", "api_key_env", "api_mode", "fallback_chain", "max_concurrency")

# auxiliary.review is the exception: a full subagent on the async delegation rail whose
# credentials resolve like `delegation.provider` pins — agent/review_engine reads exactly
# provider/model/base_url/api_key/api_mode off the block, so it has no key_env to declare.
_DELEGATION_RAIL_TASKS = {"review"}

# (task, key, value as typed on the CLI, value that must land in config.yaml)
_RUNTIME_READ_KEYS = (
    ("compression", "key_env", "CUSTOM_API_KEY", "CUSTOM_API_KEY"),
    ("approval", "api_key_env", "CUSTOM_API_KEY", "CUSTOM_API_KEY"),
    ("compression", "max_concurrency", "2", 2),
    ("compression", "context_length", "32768", 32768),
    ("skills_hub", "api_mode", "anthropic_messages", "anthropic_messages"),
    ("vision", "temperature", "0.3", 0.3),
    (
        "curator",
        "fallback_chain",
        '[{"provider": "openrouter", "model": "google/gemini-3.6-flash"}]',
        [{"provider": "openrouter", "model": "google/gemini-3.6-flash"}],
    ),
)


@pytest.fixture(autouse=True)
def _isolated_hermes_home(tmp_path):
    """Point HERMES_HOME at a temp dir so tests never touch real config."""
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        yield tmp_path


def _saved(_isolated_hermes_home):
    config_path = _isolated_hermes_home / "config.yaml"
    return yaml.safe_load(config_path.read_text()) if config_path.exists() else {}


@pytest.mark.parametrize("task,key,typed,expected", _RUNTIME_READ_KEYS)
def test_aux_runtime_read_key_is_recognized(_isolated_hermes_home, capsys, task, key, typed, expected):
    """No phantom-key notice, and the value lands on the key the runtime reads.

    The persisted value also pins the string-default trap: a key registered with a string
    default stores the user's literal verbatim, so a list-typed key needs a list default
    (the #114471 failure mode) — `fallback_chain` is checked here for exactly that.
    """
    set_config_value(f"auxiliary.{task}.{key}", typed)

    captured = capsys.readouterr()
    assert "not a recognized config key" not in captured.out
    assert "not a recognized config key" not in captured.err
    assert _saved(_isolated_hermes_home)["auxiliary"][task][key] == expected


def test_every_aux_task_block_declares_the_shared_runtime_keys():
    """Generic guard: any task block missing a key its reader consumes fails here."""
    for task, block in DEFAULT_CONFIG["auxiliary"].items():
        if task in _DELEGATION_RAIL_TASKS:
            continue
        if not isinstance(block, dict):
            # Section-level scalars (transient_retries, free_only, openrouter_model, …), not
            # a task block — no per-task keys apply.
            continue
        missing = [key for key in _SHARED_RUNTIME_KEYS if key not in block]
        assert not missing, f"auxiliary.{task} is missing runtime-read key(s): {missing}"
