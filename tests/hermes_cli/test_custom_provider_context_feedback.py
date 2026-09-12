"""#2513: a blank context-length prompt in the custom-endpoint wizard tells the user whether the
runtime resolver detected a value or will run on the silent default."""
from unittest.mock import patch

import pytest

from hermes_cli import model_setup_flows_custom as flows


@pytest.mark.parametrize("resolved, expect", [
    (200_000, "auto-detected"),
    (None, "using the default"),
])
def test_blank_context_length_reports_detection_outcome(capsys, resolved, expect):
    from agent.model_metadata import DEFAULT_FALLBACK_CONTEXT
    with patch("agent.model_metadata.get_model_context_length",
               return_value=resolved if resolved is not None else DEFAULT_FALLBACK_CONTEXT) as probe:
        flows._report_context_length_detection("some-model", "http://localhost:8000/v1", "k")
    probe.assert_called_once_with("some-model", base_url="http://localhost:8000/v1", api_key="k")
    assert expect in capsys.readouterr().out


def test_probe_failure_never_blocks_the_save(capsys):
    with patch("agent.model_metadata.get_model_context_length", side_effect=RuntimeError("boom")):
        flows._report_context_length_detection("some-model", "http://x/v1", "")
    assert capsys.readouterr().out == ""
