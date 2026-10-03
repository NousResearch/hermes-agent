"""Removed output-cap config keys must be reported, not silently accepted (#60388).

The provider-defaults change removed the dedicated user-facing output-cap controls; the scope
documented in ``evals/output_caps_scope.md`` and ``website/docs/integrations/providers.md`` says
those settings are no longer read. The ``hermes config`` surface still disagreed:

* ``model.max_tokens`` — still a valid key path (``model`` is a scalar ``str``, so the open
  walk accepts any child) with **no notice at all**: written, never read.
* ``moa.presets.<name>.reference_max_tokens`` / ``max_tokens`` — rejected with a generic
  "not a recognized config key" plus a did-you-mean pointing at ``reference_models``, which is
  an unrelated setting the runtime *does* honor.

Either way the user cannot tell "this key is gone" from "this key is a typo".
"""

from __future__ import annotations

import os
from argparse import Namespace
from unittest.mock import patch

import pytest
import yaml

from hermes_cli.config import (
    _validate_config_key,
    config_command,
    set_config_value,
    unset_config_value,
)


@pytest.fixture(autouse=True)
def _isolated_hermes_home(tmp_path):
    """Point HERMES_HOME at a temp dir so tests never touch real config."""
    (tmp_path / ".env").touch()
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        yield tmp_path


def _config_path(tmp_path):
    return tmp_path / "config.yaml"


def _write_config(tmp_path, payload: dict) -> None:
    _config_path(tmp_path).write_text(yaml.safe_dump(payload, sort_keys=True))


def _get(key: str) -> None:
    config_command(Namespace(config_command="get", key=key, json=False))


# Every key Hermes no longer reads: output-cap controls, per evals/output_caps_scope.md.
REMOVED_OUTPUT_CAP_KEYS = (
    "model.max_tokens",
    "moa.max_tokens",
    "moa.reference_max_tokens",
    "moa.presets.default.max_tokens",
    "moa.presets.default.reference_max_tokens",
    "moa.presets.fast.reference_max_tokens",
    "auxiliary.compression.max_output_tokens",
    "providers.openrouter.max_output_tokens",
    "model_overrides.openrouter.gpt-4o.max_output_tokens",
)

REMOVED_NOTICE = "removed output-cap setting"


class TestRemovedKeysAreRecognizedAsRemoved:
    @pytest.mark.parametrize("key", REMOVED_OUTPUT_CAP_KEYS)
    def test_not_reported_as_an_unknown_key(self, key):
        """A removed key is a fact, not a typo — no did-you-mean for a key that never existed."""
        assert _validate_config_key(key) == (False, None)

    @pytest.mark.parametrize("key", REMOVED_OUTPUT_CAP_KEYS)
    def test_config_set_refuses_it(self, key, _isolated_hermes_home, capsys):
        with pytest.raises(SystemExit) as excinfo:
            set_config_value(key, "4096")

        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert REMOVED_NOTICE in err
        assert f"hermes config unset {key}" in err
        # Nothing may be written on a refusal.
        assert not _config_path(_isolated_hermes_home).exists() or (
            key.split(".")[-1] not in _config_path(_isolated_hermes_home).read_text()
        )

    @pytest.mark.parametrize("key", REMOVED_OUTPUT_CAP_KEYS)
    def test_config_set_force_writes_it(self, key, _isolated_hermes_home, capsys):
        """--force is the documented escape hatch for people wiring these to external tools."""
        set_config_value(key, "4096", force=True)
        captured = capsys.readouterr()

        assert REMOVED_NOTICE not in captured.out + captured.err
        saved = yaml.safe_load(_config_path(_isolated_hermes_home).read_text())
        node = saved
        for segment in key.split("."):
            node = node[segment]
        assert node == 4096


class TestRemovedKeyGetFlagsOnStderr:
    def test_config_get_warns_but_still_prints_the_value(
        self, _isolated_hermes_home, capsys
    ):
        _write_config(_isolated_hermes_home, {"model": {"default": "x", "max_tokens": 4096}})

        _get("model.max_tokens")
        captured = capsys.readouterr()

        # stdout stays parseable; the notice goes to stderr, like the phantom-key notice.
        assert captured.out.strip() == "4096"
        assert REMOVED_NOTICE in captured.err
        assert "hermes config unset model.max_tokens" in captured.err

    def test_config_get_json_output_still_parses(self, _isolated_hermes_home, capsys):
        _write_config(_isolated_hermes_home, {"model": {"default": "x", "max_tokens": 4096}})

        config_command(Namespace(config_command="get", key="model.max_tokens", json=True))
        captured = capsys.readouterr()

        assert captured.out.strip() == "4096"
        assert REMOVED_NOTICE in captured.err

    def test_unset_still_removes_it(self, _isolated_hermes_home):
        """The remedy the notice advertises has to actually work."""
        set_config_value("model.max_tokens", "4096", force=True)
        unset_config_value("model.max_tokens")

        saved = yaml.safe_load(_config_path(_isolated_hermes_home).read_text())
        assert "max_tokens" not in saved.get("model", {})


class TestStillHonoredKeysAreNotFlagged:
    @pytest.mark.parametrize("key", [
        "moa.presets.default.reference_models",
        "moa.presets.default.reference_temperature",
        "moa.presets.default.aggregator",
        "moa.presets.default.enabled",
        "moa.default_preset",
        "moa.presets.fast.reference_timeout",
    ])
    def test_live_moa_keys_stay_configurable(self, key):
        assert _validate_config_key(key) == (True, None)
