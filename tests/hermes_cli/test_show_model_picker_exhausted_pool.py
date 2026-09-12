"""Regression test for #103829: the interactive CLI's main ``/model`` picker
(``/model`` with no args) must request exhausted-pool visibility.

``_show_model_picker`` feeds ``cli._open_model_picker`` from
``build_models_payload`` but never forwarded ``for_picker`` — a provider whose
credential pool was entirely rate-limited (entries present, none usable until
``last_error_reset_at``) was treated as "not authenticated" and its row
vanished from the main picker until the cooldown elapsed or the user ran
``hermes auth reset``. The gateway interactive picker (#66584) and the aux
pickers (#66624) already forward the flag; this is the same contract for the
CLI's main picker surface.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hermes_cli.cli_model_switch_mixin import _show_model_picker


def _run_picker(payload_providers):
    captured: dict = {}

    def _fake_build(_ctx, **kwargs):
        captured.update(kwargs)
        return {"providers": payload_providers}

    cli = SimpleNamespace(model="some-model", provider="", _open_model_picker=MagicMock())
    ctx = SimpleNamespace(user_providers={}, custom_providers=[])
    with patch("hermes_cli.inventory.build_models_payload", side_effect=_fake_build):
        _show_model_picker(cli, ctx, force_refresh=False)
    return captured, cli


def test_show_model_picker_requests_for_picker():
    """The main CLI picker must pass for_picker=True so exhausted-pool providers stay visible."""
    providers = [{"slug": "openai-codex", "models": ["gpt-5.6-sol"]}]
    captured, cli = _run_picker(providers)

    assert captured.get("for_picker") is True, (
        "_show_model_picker must forward for_picker=True so providers whose credential "
        "pool is entirely in cooldown remain selectable from the main picker (#103829)"
    )
    cli._open_model_picker.assert_called_once()
    assert cli._open_model_picker.call_args[0][0] is providers


def test_show_model_picker_usage_fallback_still_works():
    """Empty provider list keeps printing usage instead of opening the picker."""
    captured, cli = _run_picker([])

    assert captured.get("for_picker") is True
    cli._open_model_picker.assert_not_called()
