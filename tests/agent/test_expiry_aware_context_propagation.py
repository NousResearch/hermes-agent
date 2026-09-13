"""Session-local expiry-aware credential-binding propagation tests."""

from concurrent.futures import ThreadPoolExecutor
import threading
import pytest


def _binding(suffix: str) -> dict[str, str]:
    return {
        "provider": "openai-codex",
        "entry_id": f"entry-{suffix}",
        "account_id": f"account-{suffix}",
    }


@pytest.mark.parametrize("invalid", [{}, "invalid", {**_binding("bad"), "entry_id": " padded"}])
def test_invalid_propagated_binding_fails_closed_without_context_leak(invalid):
    from agent import auxiliary_client as aux
    from agent.delegation_context import delegated_child_context, is_delegated_child_context

    with pytest.raises(ValueError):
        with delegated_child_context(credential_binding=invalid):
            pytest.fail("invalid binding admitted")
    assert not is_delegated_child_context()
    with pytest.raises(ValueError):
        aux._normalize_main_runtime({"credential_binding": invalid})


def test_delegated_child_context_scopes_identity_only_credential_binding():
    from agent.delegation_context import (
        delegated_child_context,
        get_delegated_child_credential_binding,
    )

    binding = _binding("parent")
    assert get_delegated_child_credential_binding() is None

    with delegated_child_context(
        credential_binding={**binding, "non_identity_field": "ignored"}
    ):
        assert get_delegated_child_credential_binding() == binding

    assert get_delegated_child_credential_binding() is None


def test_delegated_child_context_keeps_concurrent_bindings_isolated():
    from agent.delegation_context import (
        delegated_child_context,
        get_delegated_child_credential_binding,
    )

    first = _binding("first")
    second = _binding("second")
    barrier = threading.Barrier(2)

    def observe(binding: dict[str, str]) -> dict[str, str] | None:
        with delegated_child_context(credential_binding=binding):
            barrier.wait()
            return get_delegated_child_credential_binding()

    with ThreadPoolExecutor(max_workers=2) as executor:
        observed = list(executor.map(observe, (first, second)))

    assert observed == [first, second]


def test_auxiliary_public_call_inherits_scoped_credential_binding(monkeypatch):
    import agent.auxiliary_client as auxiliary_client

    binding = _binding("auxiliary")
    captured: dict[str, object] = {}

    def fake_call_llm_impl(*args, **kwargs):
        captured["runtime"] = auxiliary_client._normalize_main_runtime(
            kwargs.get("main_runtime")
        )
        return "ok"

    monkeypatch.setattr(auxiliary_client, "_call_llm_impl", fake_call_llm_impl)

    with auxiliary_client.scoped_runtime_main(
        {
            "provider": "openai-codex",
            "model": "gpt-5",
            "credential_binding": {**binding, "non_identity_field": "ignored"},
        }
    ):
        result = auxiliary_client.call_llm(
            messages=[{"role": "user", "content": "test"}],
        )

    assert result == "ok"
    assert captured["runtime"] == {
        "provider": "openai-codex",
        "model": "gpt-5",
        "credential_binding": binding,
    }


def test_tool_runtime_publication_reaches_auxiliary_call(monkeypatch):
    import agent.auxiliary_client as auxiliary_client
    from agent import turn_context

    binding = _binding("tool")
    captured = {}

    class Agent:
        provider = "openai-codex"
        model = "gpt-5"
        requested_provider = "openai-codex"
        base_url = ""
        api_key = ""
        api_mode = ""
        auth_mode = ""
        session_id = "tool-session"
        _session_init_model_config = {
            "credential_binding": {**binding, "non_identity_field": "ignored"}
        }

    def fake_call_llm_impl(*args, **kwargs):
        captured["runtime"] = auxiliary_client._normalize_main_runtime(
            kwargs.get("main_runtime")
        )
        return "ok"

    monkeypatch.setattr(auxiliary_client, "_call_llm_impl", fake_call_llm_impl)
    auxiliary_client.clear_runtime_main()
    try:
        turn_context._publish_runtime_main(Agent())
        assert auxiliary_client.call_llm(
            messages=[{"role": "user", "content": "test"}]
        ) == "ok"
    finally:
        auxiliary_client.clear_runtime_main()

    assert captured["runtime"] == {
        "provider": "openai-codex",
        "requested_provider": "openai-codex",
        "model": "gpt-5",
        "credential_binding": binding,
    }


def test_delegated_child_uses_inherited_binding_without_fresh_selection(tmp_path):
    from agent.agent_init import prepare_expiry_aware_session_credential
    from agent.delegation_context import delegated_child_context
    from hermes_state import SessionDB

    binding = _binding("child")

    class Credential:
        id = binding["entry_id"]
        account_id = binding["account_id"]

    class Pool:
        strategy = "expiry_aware"

        def __init__(self):
            self.select_calls = 0
            self.exact_calls = []
            self.usage_lookup = None

        def set_expiry_aware_usage_lookup(self, lookup):
            self.usage_lookup = lookup

        def select(self):
            self.select_calls += 1
            raise AssertionError("delegated child must not select a new credential")

        def select_exact(self, entry_id):
            self.exact_calls.append(entry_id)
            return Credential()

    session_id = "child-session"
    session_db = SessionDB(tmp_path / "sessions.db")
    session_db.create_session(session_id, "test")
    pool = Pool()

    with delegated_child_context(session_id, credential_binding=binding):
        credential = prepare_expiry_aware_session_credential(
            pool, session_db, session_id, "openai-codex"
        )

    assert credential.id == binding["entry_id"]
    assert pool.select_calls == 0
    assert pool.exact_calls == [binding["entry_id"]]
    assert pool.usage_lookup is None
    import json

    assert json.loads(session_db.get_session(session_id)["model_config"])["credential_binding"] == binding
    session_db.close()


@pytest.mark.parametrize("async_mode", [False, True])
def test_auxiliary_route_uses_exact_pin_without_cached_selection(monkeypatch, async_mode):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from agent import auxiliary_client as aux

    binding = _binding("route")
    entry = SimpleNamespace(id=binding["entry_id"], account_id=binding["account_id"],
                            runtime_api_key="test-only", runtime_base_url="https://example.invalid")
    pool = Mock()
    pool.select_exact.return_value = entry
    monkeypatch.setattr(aux, "load_pool", lambda provider: pool)
    cached = Mock(side_effect=AssertionError("pinned calls must not use shared selection/cache"))
    monkeypatch.setattr(aux, "_get_cached_client", cached)
    client = Mock()
    create = Mock(return_value=client)
    monkeypatch.setattr(aux, "_create_openai_client", create)
    monkeypatch.setattr(aux, "_codex_cloudflare_headers", lambda *args, **kwargs: {})
    monkeypatch.setattr(aux, "_to_async_client", lambda client, model, **kw: (client, model))
    route = aux._resolve_call_client(
        None, provider="openai-codex", model="gpt-5", base_url=None, api_key=None,
        resolved_provider="openai-codex", resolved_model="gpt-5", resolved_base_url=None,
        resolved_api_key=None, resolved_api_mode=None,
        main_runtime={"credential_binding": binding}, async_mode=async_mode,
    )
    assert route.final_model == "gpt-5"
    pool.select_exact.assert_called_once_with(binding["entry_id"])
    pool.select.assert_not_called()
    assert create.call_args.kwargs["api_key"] == "test-only"


def test_pinned_auxiliary_recovery_never_enters_fallback_ladder(monkeypatch):
    from agent import auxiliary_client as aux
    from agent.agent_init import SessionCredentialBindingError
    from unittest.mock import Mock

    ladder = Mock(side_effect=AssertionError("no fallback permitted"))
    monkeypatch.setattr(aux, "_aux_recovery_ladder", ladder)
    with pytest.raises(SessionCredentialBindingError):
        aux._start_recovery_ladder(
            RuntimeError("quota exhausted"), Mock(),
            {"main_runtime": {"credential_binding": _binding("recovery")}, "max_tokens": 64},
            task=None, async_mode=False, route_info=None,
        )
    ladder.assert_not_called()
