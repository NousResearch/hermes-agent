"""Projection characterization for Telegram command catalogs."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from hermes_cli.telegram_command_normalization import (
    TelegramCommandAttemptStatus,
    normalize_telegram_command_attempt,
)
from hermes_cli.telegram_command_projection import (
    TELEGRAM_BOT_API_MAX_COMMANDS,
    TelegramMenuOmissionReason,
    build_telegram_command_projection,
)


def _command(name: str, description: str | None = None, **overrides):
    values = {
        "name": name,
        "description": description or f"Run {name}",
        "aliases": (),
        "command_id": None,
        "visibility": None,
        "hidden": False,
        "debug": False,
        "available": True,
        "unsupported_surfaces": (),
        "supported_surfaces": (),
        "cli_only": False,
        "gateway_only": False,
        "presentation_overrides": {},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_projection_uses_stable_id_and_current_v1_name_fallback():
    projection = build_telegram_command_projection(
        [
            _command("new", command_id="session.new"),
            _command("status", command_id="   "),
        ],
        catalog_revision="catalog-7",
    )

    assert projection.catalog_revision == "catalog-7"
    assert [binding.command_id for binding in projection.bindings] == [
        "session.new",
        "status",
    ]
    assert projection.native_payload == (
        ("new", "Run new"),
        ("status", "Run status"),
    )


def test_catalog_object_supplies_revision_and_order():
    catalog = SimpleNamespace(
        revision="rev-object",
        commands=(_command("beta"), _command("alpha")),
    )

    projection = build_telegram_command_projection(catalog)

    assert projection.catalog_revision == "rev-object"
    assert [command.command for command in projection.native_commands] == [
        "beta",
        "alpha",
    ]


def test_pr1_catalog_json_shape_uses_nested_legacy_availability():
    catalog = {
        "revision": "pr1-revision",
        "commands": [
            {
                "command_id": "command.clear",
                "name": "clear",
                "aliases": [],
                "description_fallback": "Clear terminal",
                "legacy": {"cli_only": True, "gateway_only": False},
            },
            {
                "command_id": "command.status",
                "name": "status",
                "aliases": [],
                "description_fallback": "Show status",
                "legacy": {"cli_only": False, "gateway_only": False},
            },
        ],
    }

    projection = build_telegram_command_projection(catalog)

    assert projection.catalog_revision == "pr1-revision"
    assert [binding.command_id for binding in projection.bindings] == [
        "command.status"
    ]
    assert projection.native_payload == (("status", "Show status"),)


def test_projection_is_immutable_and_fingerprinted_deterministically():
    first = build_telegram_command_projection([_command("new"), _command("status")])
    second = build_telegram_command_projection([_command("new"), _command("status")])
    clipped = build_telegram_command_projection(
        [_command("new"), _command("status")], max_commands=1
    )

    assert isinstance(first.bindings, tuple)
    assert first.catalog_revision == second.catalog_revision
    assert first.projection_fingerprint == second.projection_fingerprint
    assert clipped.catalog_revision == first.catalog_revision
    assert clipped.projection_fingerprint != first.projection_fingerprint
    with pytest.raises(FrozenInstanceError):
        first.catalog_revision = "mutated"  # type: ignore[misc]


def test_hidden_debug_and_unsupported_commands_do_not_leak_to_native_menu():
    projection = build_telegram_command_projection(
        [
            _command("visible"),
            _command("hidden", hidden=True),
            _command("debugger", visibility=("debug",)),
            _command("desktop-only", supported_surfaces=("desktop",)),
        ]
    )

    assert [command.command for command in projection.native_commands] == ["visible"]
    assert [binding.canonical_name for binding in projection.bindings] == [
        "visible",
        "hidden",
        "debugger",
    ]
    omitted = {(item.canonical_name, item.reason) for item in projection.omissions}
    assert omitted == {
        ("hidden", TelegramMenuOmissionReason.HIDDEN),
        ("debugger", TelegramMenuOmissionReason.HIDDEN),
    }


def test_visibility_mapping_can_explicitly_include_or_exclude_native_menu():
    projection = build_telegram_command_projection(
        [
            _command("help-only", visibility={"native_menu": False}),
            _command("native", visibility={"native_menu": True}),
            _command("completion", visibility=("completion",)),
            _command("all", visibility=("help", "completion", "native-menu")),
        ]
    )

    assert [command.command for command in projection.native_commands] == [
        "native",
        "all",
    ]


def test_telegram_presentation_override_changes_only_projection_text():
    projection = build_telegram_command_projection(
        [
            _command(
                "status",
                description="Canonical status",
                presentation_overrides={
                    "telegram": {"description": "Telegram  status\nsummary"}
                },
            )
        ]
    )

    assert projection.native_payload == (("status", "Telegram status summary"),)


def test_native_limit_omission_retains_typed_fallback():
    projection = build_telegram_command_projection(
        [_command("alpha"), _command("beta"), _command("gamma")],
        max_commands=1,
    )

    assert projection.native_payload == (("alpha", "Run alpha"),)
    assert {
        omission.canonical_name
        for omission in projection.omissions
        if omission.reason is TelegramMenuOmissionReason.NATIVE_LIMIT
    } == {"beta", "gamma"}

    typed = normalize_telegram_command_attempt("/gamma payload", projection)
    assert typed.status is TelegramCommandAttemptStatus.KNOWN_COMMAND
    assert typed.canonical_name == "gamma"
    assert typed.raw_arguments == "payload"


def test_zero_native_slots_still_preserves_every_typed_command():
    projection = build_telegram_command_projection(
        [_command("alpha"), _command("beta")], max_commands=0
    )

    assert projection.native_payload == ()
    assert normalize_telegram_command_attempt(
        "/beta", projection
    ).status is TelegramCommandAttemptStatus.KNOWN_COMMAND


def test_hyphenated_names_and_aliases_share_one_binding():
    projection = build_telegram_command_projection(
        [_command("codex-runtime", aliases=("cr",), command_id="runtime.codex")]
    )

    assert projection.native_payload == (("codex_runtime", "Run codex-runtime"),)
    for text in ("/codex-runtime", "/codex_runtime", "/cr"):
        attempt = normalize_telegram_command_attempt(text, projection)
        assert attempt.status is TelegramCommandAttemptStatus.KNOWN_COMMAND
        assert attempt.command_id == "runtime.codex"
        assert attempt.canonical_input == "/codex-runtime"



@pytest.mark.parametrize("max_commands", [-1, TELEGRAM_BOT_API_MAX_COMMANDS + 1])
def test_native_limit_bounds_fail_closed(max_commands):
    with pytest.raises(ValueError, match="max_commands must be between"):
        build_telegram_command_projection(
            [_command("status")], max_commands=max_commands
        )


def test_non_integer_native_limit_fails_closed():
    with pytest.raises(TypeError, match="max_commands must be an integer"):
        build_telegram_command_projection(
            [_command("status")], max_commands=True  # type: ignore[arg-type]
        )
