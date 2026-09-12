"""Offline integration with real agent constructors, pool and SQLite lineage."""
import json


import pytest


def test_real_agent_pin_survives_child_compaction_and_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("credential_pool_strategies:\n  openai-codex: expiry_aware\n")
    from agent import auxiliary_client as aux
    from agent import account_usage
    from agent.credential_pool import CredentialPool, PooledCredential
    from agent.delegation_context import delegated_child_context
    from agent.agent_init import SessionCredentialBindingError
    from agent.turn_context import _publish_runtime_main
    from hermes_state import SessionDB
    from run_agent import AIAgent

    # Telemetry and transport are the only external boundaries replaced.
    monkeypatch.setattr(account_usage, "fetch_codex_expiry_aware_usage", lambda *a, **k: None)
    entries = [PooledCredential(provider="openai-codex", id=name, label=name,
                               auth_type="api_key", source="manual", priority=i,
                               access_token="test-key-" + name,
                               base_url="https://chatgpt.com/backend-api/codex",
                               extra={"account_id": "account-" + name})
               for i, name in enumerate(["first", "second"])]
    pool = CredentialPool("openai-codex", entries)
    monkeypatch.setattr(aux, "load_pool", lambda provider: pool)
    with SessionDB(tmp_path / "state.db") as db:

        def create(sid):
            return AIAgent(provider="openai-codex", api_mode="codex_responses", model="gpt-5",
                           credential_pool=pool, session_db=db, session_id=sid,
                           quiet_mode=True, skip_memory=True, skip_context_files=True,
                           enabled_toolsets=[], checkpoints_enabled=False,
                           fallback_model={"provider": "anthropic", "model": "other"})

        with aux.scoped_runtime_main({}):
            parent = create("parent")
            binding = json.loads(db.get_session("parent")["model_config"])["credential_binding"]
            assert binding["entry_id"] == "first"
            assert parent._session_init_model_config["credential_binding"] == binding
            assert parent._fallback_chain == []
            _publish_runtime_main(parent)
            seen = []
            real_create = aux._create_openai_client
            monkeypatch.setattr(aux, "_create_openai_client", lambda **kw: seen.append(kw) or real_create(**kw))
            route = aux._resolve_call_client(
                None, provider=None, model=None, base_url=None, api_key=None,
                resolved_provider="auto", resolved_model="gpt-5", resolved_base_url=None,
                resolved_api_key=None, resolved_api_mode=None,
                main_runtime=aux._normalize_main_runtime(None), async_mode=False)
            assert route.resolved_provider == "openai-codex"
            assert seen[-1]["api_key"] == "test-key-first"

            db.create_session("child", "cli")
            pool._entries.reverse()  # A fresh fill-first admission would choose the other entry.
            with delegated_child_context(credential_binding=binding):
                child = create("child")
            assert child._session_init_model_config["credential_binding"] == binding

            assert db.try_acquire_compression_lock("parent", "integration")
            db.publish_compression_child(
                parent_session_id="parent", child_session_id="compacted", source="cli",
                messages=[{"role": "user", "content": "[CONTEXT COMPACTION] summary"}],
                model="gpt-5", model_config={}, system_prompt="", compression_lock_holder="integration")
            resumed = create("compacted")
            assert resumed._session_init_model_config["credential_binding"] == binding
            pool._entries[:] = [entry for entry in pool.entries() if entry.id != "first"]
            with pytest.raises(SessionCredentialBindingError):
                create("compacted")
