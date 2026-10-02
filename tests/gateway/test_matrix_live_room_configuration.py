"""Configuration admission for the separate-client room-instructions test."""

import inspect
from pathlib import Path

import pytest

from gateway.config import Platform, load_gateway_config
from gateway.run import GatewayRunner
from gateway.pairing import PairingStore
from hermes_cli.env_loader import load_hermes_dotenv
from tests.fakes.fake_llm_provider import write_hermes_home
from tests.integration.matrix_live import conftest as live_fixtures
from tests.integration.matrix_live import test_room_instructions as room_instructions
from tests.integration.matrix_live import test_reaction_menu as reaction_menu


def test_room_configuration_authorizes_only_its_observer_before_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    room = live_fixtures.LiveRoom(
        homeserver="https://matrix.test",
        room_id="!instructions:matrix.test",
        bot=live_fixtures.MatrixAccount("@hermes:matrix.test", "BOT", "bot-token"),
        observer=live_fixtures.MatrixAccount("@alice:matrix.test", "OBSERVER", "observer-token"),
    )
    configuration = inspect.unwrap(room_instructions.gateway_config)(
        inspect.unwrap(live_fixtures.gateway_config)(), room,
    )
    configuration = live_fixtures._gateway_yaml_config(
        configuration, live_fixtures.MatrixFeedbackSettings(),
        live_fixtures.GatewaySettings(), room.room_id, "interrupt",
        inspect.unwrap(room_instructions.gateway_extra_config)(room),
    )
    write_hermes_home(tmp_path, "http://127.0.0.1:1/v1", extra_config=configuration)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("MATRIX_ALLOWED_USERS", raising=False)
    config = load_gateway_config()
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.adapters = {}
    runner._profile_adapters = {}
    runner.pairing_store = PairingStore()
    runner.pairing_stores = {}
    authorize = runner._make_adapter_auth_check(Platform.MATRIX)
    extra = config.platforms[Platform.MATRIX].extra

    assert (
        authorize(room.observer.user_id, "group", room.room_id),
        authorize("@other:matrix.test", "group", room.room_id),
        extra["channel_prompts"],
        extra["channel_skill_bindings"],
    ) == (
        True,
        False,
        {room.room_id: "Follow the configured research method."},
        [{"id": room.room_id, "skills": ["matrix-room-method"]}],
    )


def test_menu_configuration_uses_its_script_and_yaml_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = inspect.unwrap(reaction_menu.gateway_script)()
    configuration = live_fixtures._gateway_yaml_config(
        inspect.unwrap(reaction_menu.gateway_config)(),
        live_fixtures.MatrixFeedbackSettings(), live_fixtures.GatewaySettings(),
        "!menu:matrix.test", "interrupt", "",
    )
    write_hermes_home(tmp_path, "http://127.0.0.1:1/v1", extra_config=configuration)
    (tmp_path / ".env").write_text(
        "MATRIX_ALLOWED_USERS=@alice:matrix.test\nMATRIX_AUTO_THREAD=false\n",
        encoding="utf-8",
    )
    setup = getattr(reaction_menu, "gateway_home_setup", live_fixtures.gateway_home_setup)
    inspect.unwrap(setup)()(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("MATRIX_ALLOWED_USERS", "")
    load_hermes_dotenv(hermes_home=tmp_path, load_external_secrets=False)
    config = load_gateway_config()
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.adapters = {}
    runner._profile_adapters = {}
    runner.pairing_store = PairingStore()
    runner.pairing_stores = {}
    authorize = runner._make_adapter_auth_check(Platform.MATRIX)

    assert (
        inspect.unwrap(reaction_menu.model_responder)(script),
        authorize("@alice:matrix.test", "group", "!menu:matrix.test"),
        authorize("@bob:matrix.test", "group", "!menu:matrix.test"),
        authorize("@other:matrix.test", "group", "!menu:matrix.test"),
        (tmp_path / ".env").read_text(encoding="utf-8"),
    ) == (
        script, True, True, False, "MATRIX_AUTO_THREAD=false\n",
    )
