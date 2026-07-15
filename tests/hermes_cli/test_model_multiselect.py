from unittest.mock import call, patch


def test_model_picker_keeps_single_select_contract():
    from hermes_cli.auth import _prompt_model_selection

    with (
        patch("hermes_cli.curses_ui.curses_radiolist", return_value=1),
        patch("hermes_cli.curses_ui.curses_checklist") as checklist,
    ):
        selected = _prompt_model_selection(["model-a", "model-b"])

    assert selected == "model-b"
    checklist.assert_not_called()


def test_model_picker_multiselect_uses_menu_order_and_confirms_every_model():
    from hermes_cli.auth import _prompt_model_selection

    with (
        patch(
            "hermes_cli.curses_ui.curses_checklist",
            return_value={0, 2},
        ) as checklist,
        patch(
            "hermes_cli.auth._confirm_expensive_model_selection",
            return_value=True,
        ) as confirm,
    ):
        selected = _prompt_model_selection(
            ["model-a", "model-b", "model-c"],
            current_model="model-b",
            confirm_provider="openrouter",
            confirm_base_url="https://openrouter.ai/api/v1",
            confirm_api_key="test-key",
            multi_select=True,
        )

    # The current model is first in the displayed menu, followed by catalog
    # order. A set returned by curses therefore still produces stable order.
    assert selected == ["model-b", "model-c"]
    assert checklist.call_args.kwargs["searchable"] is True
    confirm.assert_has_calls(
        [
            call(
                "model-b",
                provider="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_key="test-key",
            ),
            call(
                "model-c",
                provider="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_key="test-key",
            ),
        ]
    )


def test_model_picker_aborts_group_when_expensive_confirmation_is_declined():
    from hermes_cli.auth import _prompt_model_selection

    with (
        patch("hermes_cli.curses_ui.curses_checklist", return_value={0, 1}),
        patch(
            "hermes_cli.auth._confirm_expensive_model_selection",
            side_effect=[True, False],
        ) as confirm,
    ):
        selected = _prompt_model_selection(
            ["model-a", "model-b"],
            confirm_provider="openrouter",
            multi_select=True,
        )

    assert selected is None
    assert confirm.call_count == 2
