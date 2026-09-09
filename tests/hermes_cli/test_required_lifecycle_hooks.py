"""Behavior contract for fail-closed required plugin lifecycle hooks."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hermes_cli.plugins import (
    PluginContext,
    PluginManager,
    RequiredLifecycleError,
    get_pre_tool_call_block_message,
    required_hook_result,
)
from hermes_cli.required_lifecycle import (
    parse_required_lifecycle_policy,
    quarantine_required_provider_fields,
    restore_required_provider_fields,
)


REQUIRED = {
    "pre_llm_call": ["behavioral.pre_llm.v1"],
    "pre_tool_call": ["behavioral.pre_tool.v1"],
    "post_tool_call": ["behavioral.post_tool.v1"],
    "transform_llm_output": ["behavioral.output.v1"],
}


def _write_config(home: Path, requirements=REQUIRED) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "plugins": {
                    "enabled": ["atlas"],
                    "required_lifecycle_hooks": {"atlas": requirements},
                }
            }
        ),
        encoding="utf-8",
    )


def _write_plugin(home: Path, register_body: str) -> None:
    root = home / "plugins" / "atlas"
    root.mkdir(parents=True)
    (root / "plugin.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "atlas",
                "version": "1.0.0",
                "description": "test plugin",
            }
        ),
        encoding="utf-8",
    )
    (root / "__init__.py").write_text(
        "from hermes_cli.plugins import required_hook_result\n"
        "def register(ctx):\n"
        + "\n".join(f"    {line}" for line in register_body.splitlines())
        + "\n",
        encoding="utf-8",
    )


def _valid_register_body() -> str:
    return "\n".join(
        (
            'ctx.register_hook("pre_llm_call", lambda **kw: '
            'required_hook_result("behavioral.pre_llm.v1", None), '
            'registration_id="behavioral.pre_llm.v1")',
            'ctx.register_hook("pre_tool_call", lambda **kw: '
            'required_hook_result("behavioral.pre_tool.v1", {}), '
            'registration_id="behavioral.pre_tool.v1")',
            'ctx.register_hook("post_tool_call", lambda **kw: '
            'required_hook_result("behavioral.post_tool.v1", None), '
            'registration_id="behavioral.post_tool.v1")',
            'ctx.register_hook("transform_llm_output", lambda **kw: '
            'required_hook_result("behavioral.output.v1", None), '
            'registration_id="behavioral.output.v1")',
        )
    )


def _manager(tmp_path: Path, monkeypatch, register_body: str) -> PluginManager:
    home = tmp_path / "hermes"
    _write_config(home)
    _write_plugin(home, register_body)
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    manager.discover_and_load()
    return manager


def _payload() -> dict[str, str]:
    return {"session_id": "session-1", "turn_id": "turn-1"}


def test_exact_required_registration_delivers_once_and_unwraps_result(
    tmp_path, monkeypatch
):
    manager = _manager(tmp_path, monkeypatch, _valid_register_body())

    assert manager.invoke_hook("pre_tool_call", **_payload()) == [{}]


@pytest.mark.parametrize(
    ("hook_name", "optional_result", "required_result"),
    [
        pytest.param(
            "pre_tool_call",
            '{"action": "block", "message": "legacy security block"}',
            [{}, {"action": "block", "message": "legacy security block"}],
            id="pre-tool-security-block",
        ),
        pytest.param(
            "pre_tool_call",
            '{"action": "modify", "args": {"command": "unsafe"}}',
            [{}],
            id="pre-tool-mutation",
        ),
        pytest.param(
            "transform_llm_output",
            '"unsafe replacement"',
            [],
            id="terminal-output-replacement",
        ),
    ],
)
def test_required_authority_ignores_optional_semantic_replacements(
    tmp_path, monkeypatch, hook_name, optional_result, required_result
):
    body = _valid_register_body() + (
        f'\nctx.register_hook("{hook_name}", lambda **kw: {optional_result})'
    )
    manager = _manager(tmp_path, monkeypatch, body)

    payload = _payload()
    if hook_name == "transform_llm_output":
        payload["response_text"] = "original"

    assert manager.invoke_hook(hook_name, **payload) == required_result


@pytest.mark.parametrize(
    "register_body",
    [
        pytest.param(
            _valid_register_body().replace(
                'registration_id="behavioral.pre_tool.v1")',
                'registration_id="behavioral.pre_tool.v2")',
            ),
            id="wrong-registration-id",
        ),
        pytest.param(
            _valid_register_body().replace(
                'ctx.register_hook("pre_tool_call", lambda **kw: '
                'required_hook_result("behavioral.pre_tool.v1", {}), '
                'registration_id="behavioral.pre_tool.v1")\n',
                "",
            ),
            id="missing-registration",
        ),
        pytest.param(
            _valid_register_body()
            + '\nctx.register_hook("pre_tool_call", lambda **kw: '
            'required_hook_result("behavioral.pre_tool.v1", {}), '
            'registration_id="behavioral.pre_tool.v1")',
            id="duplicate-registration",
        ),
    ],
)
def test_required_bundle_rejects_missing_wrong_or_duplicate_registration(
    tmp_path, monkeypatch, register_body
):
    manager = _manager(tmp_path, monkeypatch, register_body)

    with pytest.raises(RequiredLifecycleError) as caught:
        manager.invoke_hook("pre_tool_call", **_payload())

    assert caught.value.reason_code == "required_lifecycle_registration_invalid"


@pytest.mark.parametrize(
    "replacement",
    [
        pytest.param('(_ for _ in ()).throw(RuntimeError("secret-value"))', id="raises"),
        pytest.param("None", id="missing-ack"),
        pytest.param(
            'required_hook_result("behavioral.pre_tool.v2", {})',
            id="wrong-ack",
        ),
        pytest.param(
            'required_hook_result("behavioral.pre_tool.v1", "allow")',
            id="malformed-result",
        ),
    ],
)
def test_required_callback_failure_is_redacted_and_latched(
    tmp_path, monkeypatch, replacement
):
    body = _valid_register_body().replace(
        'required_hook_result("behavioral.pre_tool.v1", {})', replacement
    )
    manager = _manager(tmp_path, monkeypatch, body)

    with pytest.raises(RequiredLifecycleError) as first:
        manager.invoke_hook("pre_tool_call", **_payload())
    with pytest.raises(RequiredLifecycleError) as latched:
        manager.invoke_hook(
            "transform_llm_output", response_text="raw secret", **_payload()
        )

    assert first.value.reason_code == "required_lifecycle_delivery_failed"
    assert latched.value.reason_code == first.value.reason_code
    assert "secret-value" not in str(first.value)


def test_required_pre_tool_rejects_non_dict_mapping_directive(
    tmp_path, monkeypatch
):
    body = _valid_register_body().replace(
        'required_hook_result("behavioral.pre_tool.v1", {})',
        'required_hook_result("behavioral.pre_tool.v1", '
        '__import__("types").MappingProxyType('
        '{"action": "block", "message": "must block"}))',
    )
    manager = _manager(tmp_path, monkeypatch, body)

    with pytest.raises(RequiredLifecycleError) as caught:
        manager.invoke_hook("pre_tool_call", **_payload())

    assert caught.value.reason_code == "required_lifecycle_delivery_failed"


def test_registration_generation_drift_during_required_callback_fails_closed(
    tmp_path, monkeypatch
):
    body = _valid_register_body().replace(
        'required_hook_result("behavioral.pre_tool.v1", {})',
        '(setattr(ctx._manager, "_registration_generation", '
        'ctx._manager._registration_generation + 1) or '
        'required_hook_result("behavioral.pre_tool.v1", {}))',
    )
    manager = _manager(tmp_path, monkeypatch, body)

    with pytest.raises(RequiredLifecycleError) as caught:
        manager.invoke_hook("pre_tool_call", **_payload())

    assert caught.value.reason_code == "required_lifecycle_registration_changed"


def test_force_reload_cannot_erase_failed_active_turn_latch(tmp_path, monkeypatch):
    body = _valid_register_body().replace(
        'required_hook_result("behavioral.post_tool.v1", None)',
        '(_ for _ in ()).throw(RuntimeError("post tool unavailable"))',
    )
    manager = _manager(tmp_path, monkeypatch, body)

    with pytest.raises(RequiredLifecycleError):
        manager.invoke_hook("post_tool_call", **_payload())
    with pytest.raises(RequiredLifecycleError) as reload_error:
        manager.discover_and_load(force=True)

    with pytest.raises(RequiredLifecycleError) as caught:
        manager.assert_required_lifecycle_turn_healthy(**_payload())

    assert reload_error.value.reason_code == "required_lifecycle_reload_deferred"
    assert caught.value.reason_code == "required_lifecycle_delivery_failed"


def test_force_reload_invalidates_active_required_turn(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch, _valid_register_body())
    manager.invoke_hook("pre_llm_call", **_payload())

    with pytest.raises(RequiredLifecycleError) as caught:
        manager.discover_and_load(force=True)

    manager.assert_required_lifecycle_turn_healthy(**_payload())

    assert caught.value.reason_code == "required_lifecycle_reload_deferred"


def test_active_turn_rejects_registration_before_live_registry_mutation(
    tmp_path, monkeypatch
):
    manager = _manager(tmp_path, monkeypatch, _valid_register_body())
    manager.invoke_hook("pre_llm_call", **_payload())
    manifest = manager._plugins["atlas"].manifest
    context = PluginContext(manifest, manager)
    before = tuple(manager._hooks.get("other", ()))

    with pytest.raises(RequiredLifecycleError) as caught:
        context.register_hook("other", lambda **_kw: None)

    assert caught.value.reason_code == "required_lifecycle_reload_deferred"
    assert tuple(manager._hooks.get("other", ())) == before
    manager.assert_required_lifecycle_turn_healthy(**_payload())


def test_policy_rejects_multiple_pre_tool_or_output_authorities():
    with pytest.raises(RequiredLifecycleError) as caught:
        parse_required_lifecycle_policy(
            {
                "plugins": {
                    "required_lifecycle_hooks": {
                        "atlas": REQUIRED,
                        "second": {
                            hook: [identifier.replace("behavioral", "second")]
                            for hook, (identifier,) in REQUIRED.items()
                        },
                    }
                }
            }
        )

    assert caught.value.reason_code == "required_lifecycle_policy_invalid"


def test_provider_continuity_is_private_but_restored_on_request_clone():
    agent = SimpleNamespace()
    message = {
        "role": "assistant",
        "timestamp": 1.0,
        "finish_reason": "tool_calls",
        "reasoning_content": "private reasoning",
        "tool_calls": [
            {
                "id": "call-1",
                "call_id": "call-1",
                "extra_content": {"thought_signature": "private signature"},
            }
        ],
    }

    quarantine_required_provider_fields(agent, message)
    request_message = copy.deepcopy(message)
    restore_required_provider_fields(agent, message, request_message)

    assert "reasoning_content" not in message
    assert "extra_content" not in message["tool_calls"][0]
    assert request_message["reasoning_content"] == "private reasoning"
    assert request_message["tool_calls"][0]["extra_content"] == {
        "thought_signature": "private signature"
    }


def test_optional_hooks_keep_legacy_fail_open_behavior(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["atlas"]}}), encoding="utf-8"
    )
    _write_plugin(
        home,
        'ctx.register_hook("pre_tool_call", lambda **kw: '
        '(_ for _ in ()).throw(RuntimeError("optional")))',
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    manager.discover_and_load()

    assert manager.invoke_hook("pre_tool_call", **_payload()) == []


def test_pre_tool_required_delivery_failure_becomes_core_owned_block(monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise RequiredLifecycleError("required_lifecycle_delivery_failed")

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", unavailable)

    assert get_pre_tool_call_block_message(
        "terminal",
        {"command": "true"},
        session_id="session-1",
        turn_id="turn-1",
    ) == (
        "Hermes blocked this turn because a required lifecycle guard was "
        "unavailable."
    )


@pytest.mark.parametrize(
    "requirements",
    [
        pytest.param(
            {"pre_tool_call": ["behavioral.pre_tool.v1"]},
            id="incomplete-bundle",
        ),
        pytest.param(
            {
                **REQUIRED,
                "pre_tool_call": [
                    "behavioral.pre_tool.v1",
                    "behavioral.pre_tool.v2",
                ],
            },
            id="ambiguous-authority",
        ),
    ],
)
def test_malformed_required_policy_cannot_degrade_to_optional(
    tmp_path, monkeypatch, requirements
):
    home = tmp_path / "hermes"
    _write_config(home, requirements)
    _write_plugin(home, _valid_register_body())
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    manager.discover_and_load()

    assert manager.has_hook("pre_llm_call") is True
    with pytest.raises(RequiredLifecycleError) as caught:
        manager.invoke_hook("pre_llm_call", **_payload())

    assert caught.value.reason_code == "required_lifecycle_policy_invalid"
