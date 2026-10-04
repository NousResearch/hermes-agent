"""Tests for explicit toolset handling in Hermes one-shot mode."""

from unittest.mock import patch

from hermes_cli.oneshot import (
    _normalize_toolsets,
    _validate_explicit_toolsets,
    run_oneshot,
)


def test_none_sentinel_resolves_to_explicit_empty_toolset_list():
    toolsets, error = _validate_explicit_toolsets("none")

    assert toolsets == []
    assert error is None


def test_none_sentinel_cannot_be_combined_with_other_toolsets():
    toolsets, error = _validate_explicit_toolsets("none,file")

    assert toolsets is None
    assert error == "hermes -z: --toolsets none cannot be combined with other toolsets.\n"


def test_omitted_toolsets_still_selects_configured_defaults():
    assert _normalize_toolsets(None) is None


def test_none_sentinel_reaches_agent_as_explicit_empty_list(capsys):
    with patch("hermes_cli.oneshot._run_agent", return_value=("ok", {})) as run_agent:
        assert run_oneshot("prompt", toolsets="none") == 0

    assert capsys.readouterr().out == "ok\n"
    assert run_agent.call_args.kwargs["toolsets"] == []
    assert run_agent.call_args.kwargs["use_config_toolsets"] is False
