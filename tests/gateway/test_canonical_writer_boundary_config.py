import hashlib

import pytest

from gateway import canonical_writer_boundary as boundary


@pytest.fixture(autouse=True)
def _fresh_process_policy():
    boundary._reset_frozen_writer_boundary_config_for_tests()
    yield
    boundary._reset_frozen_writer_boundary_config_for_tests()


def _config(**writer):
    return {
        "canonical_brain": {
            "tools_enabled": True,
            "writer_boundary": {"enabled": True, **writer},
        }
    }


def test_production_socket_and_service_units_are_pinned():
    configured = boundary.load_writer_boundary_config(_config())

    assert configured.socket_path == boundary.DEFAULT_SOCKET_PATH
    assert configured.gateway_unit == boundary.DEFAULT_GATEWAY_UNIT
    assert configured.writer_unit == boundary.DEFAULT_WRITER_UNIT

    with pytest.raises(ValueError, match="socket_path is production-pinned"):
        boundary.load_writer_boundary_config(
            _config(socket_path="/tmp/model-controlled.sock")
        )
    with pytest.raises(ValueError, match="gateway_unit is production-pinned"):
        boundary.load_writer_boundary_config(
            _config(gateway_unit="attacker.service")
        )
    with pytest.raises(ValueError, match="writer_unit is production-pinned"):
        boundary.load_writer_boundary_config(
            _config(writer_unit="attacker.service")
        )


def test_tool_availability_is_static_policy_not_socket_health(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: _config(),
    )

    assert boundary.writer_boundary_configured() is True


def test_writer_policy_is_frozen_until_process_restart():
    assert boundary.writer_boundary_configured(_config()) is True

    edited = _config(enabled=False)
    edited["canonical_brain"]["writer_boundary"]["socket_path"] = (
        "/tmp/redirected.sock"
    )

    assert boundary.writer_boundary_configured(edited) is True
    assert boundary.frozen_writer_boundary_config() == (
        boundary.load_writer_boundary_config(_config())
    )


def test_model_tool_policy_is_frozen_with_writer_boundary():
    assert boundary.writer_boundary_configured(_config()) is True
    assert boundary.canonical_model_tools_configured() is True

    edited = _config()
    edited["canonical_brain"]["tools_enabled"] = False

    assert boundary.canonical_model_tools_configured() is True

    boundary._reset_frozen_writer_boundary_config_for_tests()
    assert boundary.writer_boundary_configured(edited) is True
    assert boundary.canonical_model_tools_configured() is False


def test_invalid_enabled_boundary_remains_a_frozen_fail_closed_requirement():
    invalid = _config(socket_path="/tmp/model-controlled.sock")

    assert boundary.writer_boundary_configured(invalid) is False
    assert boundary.writer_boundary_policy_required() is True

    edited = _config(enabled=False)
    assert boundary.writer_boundary_policy_required(edited) is True


def test_enabled_boundary_makes_gateway_nondumpable_before_use(monkeypatch):
    calls = []
    state = {"dumpable": 1}
    boundary.frozen_writer_boundary_config(_config())
    monkeypatch.setattr(boundary.sys, "platform", "linux")
    monkeypatch.setattr(
        boundary,
        "_disable_process_core_dumps",
        lambda: calls.append("core-disabled"),
    )

    def prctl(option, argument):
        calls.append((option, argument))
        if option == boundary._PR_SET_DUMPABLE:
            state["dumpable"] = argument
            return 0
        return state["dumpable"]

    monkeypatch.setattr(boundary, "_linux_prctl", prctl)

    assert boundary.harden_gateway_process_for_writer_boundary() is True
    assert boundary.harden_gateway_process_for_writer_boundary() is True
    assert calls == [
        "core-disabled",
        (boundary._PR_SET_DUMPABLE, 0),
        (boundary._PR_GET_DUMPABLE, 0),
    ]


def test_enabled_boundary_rejects_non_linux_gateway_process(monkeypatch):
    boundary.frozen_writer_boundary_config(_config())
    monkeypatch.setattr(boundary.sys, "platform", "darwin")

    with pytest.raises(RuntimeError, match="requires_linux_process_hardening"):
        boundary.harden_gateway_process_for_writer_boundary()


def test_trusted_runtime_hashes_session_key_and_never_returns_raw(monkeypatch):
    values = {
        "HERMES_SESSION_PLATFORM": "discord",
        "HERMES_SESSION_USER_ID": "owner-1",
        "HERMES_SESSION_KEY": "raw-session-key",
        "HERMES_CAPABILITY_EPOCH_SHA256": "e" * 64,
    }
    monkeypatch.setattr(
        "gateway.session_context.get_session_env",
        lambda name, default="": values.get(name, default),
    )

    runtime = boundary.trusted_runtime_envelope()

    assert runtime["platform"] == "discord"
    assert runtime["user_id"] == "owner-1"
    assert runtime["session_key_sha256"] == hashlib.sha256(
        b"raw-session-key"
    ).hexdigest()
    assert runtime["capability_epoch_sha256"] == "e" * 64
    assert "raw-session-key" not in repr(runtime)


def test_database_adapter_is_unavailable_outside_authenticated_service_scope():
    database = object()
    with pytest.raises(PermissionError, match="requires_writer_service"):
        boundary.require_writer_database()

    with boundary.authenticated_writer_service_scope(
        database=database,
        runtime={"platform": "discord", "thread_id": "thread-1"},
        peer_pid=123,
        peer_uid=456,
    ) as context:
        assert boundary.require_writer_database() is database
        assert context.peer_pid == 123
        assert boundary.in_writer_service() is True

    assert boundary.in_writer_service() is False
    with pytest.raises(PermissionError, match="requires_writer_service"):
        boundary.require_writer_database()
