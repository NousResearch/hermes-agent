"""Routing-policy outbound and recovery invariants."""

from types import SimpleNamespace

import pytest


class _RecordingCreate:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[])


def test_chat_dispatch_denial_after_request_mutation_sends_nothing(monkeypatch):
    """The last chat-completions seam rejects the mutated model before create()."""
    from agent import chat_completion_helpers as helpers
    from hermes_cli.routing_policy import RoutingPolicyError

    create = _RecordingCreate()
    client = SimpleNamespace(chat=SimpleNamespace(completions=create))
    agent = SimpleNamespace(
        api_mode="chat_completions",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        client=client,
    )
    monkeypatch.setattr(helpers, "current_routing_policy", lambda: {
        "enabled": True, "deny": {"models": ["glm-*"]},
    })

    with pytest.raises(RoutingPolicyError):
        helpers._dispatch_nonstreaming_api_request(
            agent, {"model": "openrouter:z-ai/glm-5.2"}, make_client=lambda *_a, **_kw: client,
        )

    assert create.calls == []


def test_codex_responses_denial_sends_nothing(monkeypatch):
    """Responses transport is guarded separately from chat-completions."""
    from agent.codex_runtime import run_codex_stream
    from hermes_cli.routing_policy import RoutingPolicyError

    create = _RecordingCreate()
    client = SimpleNamespace(responses=create, base_url="https://openrouter.ai/api/v1")
    agent = SimpleNamespace(
        provider="openrouter", base_url="https://openrouter.ai/api/v1", model="z-ai/glm-5.2",
    )
    monkeypatch.setattr("hermes_cli.routing_policy.current_routing_policy", lambda: {
        "enabled": True, "deny": {"models": ["z-ai/*"]},
    })

    with pytest.raises(RoutingPolicyError):
        run_codex_stream(agent, {"model": "z-ai/glm-5.2"}, client=client)

    assert create.calls == []


def test_policy_error_is_terminal_to_auxiliary_recovery_ladder():
    """A denial is not a recoverable provider failure and cannot advance a fallback."""
    from agent.auxiliary_client import _drive_ladder, _rung
    from hermes_cli.routing_policy import RoutingPolicyError

    attempted = []

    def ladder():
        _response, _error = yield from _rung(
            ("primary",), lambda _exc: True,
        )
        attempted.append("fallback")
        yield ("fallback",)

    def perform(step):
        if step == ("primary",):
            raise RoutingPolicyError("denied")
        attempted.append("send")

    with pytest.raises(RoutingPolicyError):
        _drive_ladder(ladder(), perform)

    assert attempted == []


def test_denied_cron_route_is_not_saved(monkeypatch):
    """A rejected cron route fails before the jobs-file write seam."""
    import cron.jobs as jobs
    from hermes_cli.routing_policy import RoutingPolicyError

    writes = []
    monkeypatch.setattr(jobs, "current_routing_policy", lambda: {
        "enabled": True, "deny": {"models": ["z-ai/*"]},
    })
    monkeypatch.setattr(jobs, "save_jobs", lambda records: writes.append(records))

    with pytest.raises(RoutingPolicyError):
        jobs.create_job(prompt="p", schedule="every 1h", provider="openrouter", model="z-ai/glm-5.2")

    assert writes == []



def test_denied_cron_route_update_is_not_saved(monkeypatch):
    """Editing a legacy cron route rejects before the replacement record is saved."""
    import cron.jobs as jobs
    from hermes_cli.routing_policy import RoutingPolicyError

    writes = []
    monkeypatch.setattr(jobs, "current_routing_policy", lambda: {
        "enabled": True, "deny": {"models": ["z-ai/*"]},
    })
    monkeypatch.setattr(jobs, "save_jobs", lambda records: writes.append(records))

    def invoke_apply(_job_id, apply):
        return apply([{"id": "j1"}], 0, {
            "id": "j1", "name": "legacy", "prompt": "p", "schedule": {"kind": "interval"},
            "enabled": True, "state": "scheduled", "next_run_at": "later", "repeat": {"completed": 0},
        })

    monkeypatch.setattr(jobs, "_with_job", invoke_apply)
    with pytest.raises(RoutingPolicyError):
        jobs.update_job("j1", {"model": "z-ai/glm-5.2", "provider": "openrouter"})

    assert writes == []


def test_denied_persisted_session_route_is_not_written(monkeypatch):
    """Session model mutation checks policy before it touches SQLite."""
    from hermes_state import SessionDB
    from hermes_cli.routing_policy import RoutingPolicyError

    db = object.__new__(SessionDB)
    writes = []
    monkeypatch.setattr("hermes_cli.routing_policy.current_routing_policy", lambda: {
        "enabled": True, "deny": {"models": ["z-ai/*"]},
    })
    monkeypatch.setattr(db, "flush_token_counts", lambda: None)
    monkeypatch.setattr(db, "_write_model_config_patch", lambda *_a, **_kw: writes.append(True))

    with pytest.raises(RoutingPolicyError):
        db.update_session_model("s1", "z-ai/glm-5.2", provider="openrouter")

    assert writes == []


def test_multiplex_owner_policy_denies_session_writes_and_dispatch_after_a_to_b_to_a(tmp_path, monkeypatch):
    """A secondary's owner policy wins after the ambient launch profile resumes."""
    from agent import chat_completion_helpers as helpers
    from hermes_state import SessionDB
    from hermes_cli.routing_policy import RoutingPolicyError

    home = tmp_path / "hermes"
    secondary = home / "profiles" / "restricted"
    secondary.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text("routing_policy:\n  enabled: true\n", encoding="utf-8")
    (secondary / "config.yaml").write_text(
        "routing_policy:\n  enabled: true\n  deny:\n    models: ['z-ai/*']\n", encoding="utf-8",
    )

    default_db = SessionDB(db_path=home / "state.db")
    secondary_db = SessionDB(db_path=secondary / "state.db")
    create = _RecordingCreate()
    try:
        # A permits, B denies, then A is ambient again: B's persisted owner must still govern.
        default_db.create_session("a-before", "cli", model="allowed", model_config={"provider": "openrouter"})
        with pytest.raises(RoutingPolicyError):
            secondary_db.create_session(
                "b-create", "cli", model="z-ai/glm-5.2", model_config={"provider": "openrouter"},
            )
        assert secondary_db.get_session("b-create") is None

        secondary_db.create_session("b-update", "cli", model="allowed", model_config={"provider": "openrouter"})
        with pytest.raises(RoutingPolicyError):
            secondary_db.update_session_model("b-update", "z-ai/glm-5.2", provider="openrouter")
        assert secondary_db.get_session("b-update")["model"] == "allowed"

        secondary_db.create_session("b-patch", "cli", model_config={"provider": "openrouter"})
        with pytest.raises(RoutingPolicyError):
            secondary_db.patch_session_model_config("b-patch", {"model": "z-ai/glm-5.2"})
        assert secondary_db.get_session_model_config_value("b-patch", "model") is None

        agent = SimpleNamespace(
            api_mode="chat_completions", provider="openrouter", model="z-ai/glm-5.2",
            base_url="https://openrouter.ai/api/v1", _session_db=secondary_db,
        )
        with pytest.raises(RoutingPolicyError):
            helpers._dispatch_nonstreaming_api_request(
                agent, {"model": "z-ai/glm-5.2"},
                make_client=lambda *_args, **_kwargs: SimpleNamespace(chat=SimpleNamespace(completions=create)),
            )
        assert create.calls == []

        default_db.create_session("a-after", "cli", model="allowed", model_config={"provider": "openrouter"})
    finally:
        secondary_db.close()
        default_db.close()


def test_symlinked_named_profile_session_db_uses_owner_policy_after_a_to_b_to_a(tmp_path, monkeypatch):
    """A logical named-profile symlink keeps B's policy for writes and dispatch."""
    from agent import chat_completion_helpers as helpers
    from hermes_state import SessionDB
    from hermes_cli.routing_policy import RoutingPolicyError

    root = tmp_path / "hermes"
    restricted = root / "profiles" / "restricted"
    outside_restricted = tmp_path / "outside" / "restricted"
    root.mkdir()
    outside_restricted.mkdir(parents=True)
    restricted.parent.mkdir()
    restricted.symlink_to(outside_restricted, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    (root / "config.yaml").write_text("routing_policy:\n  enabled: true\n", encoding="utf-8")
    (outside_restricted / "config.yaml").write_text(
        "routing_policy:\n  enabled: true\n  deny:\n    models: ['z-ai/*']\n", encoding="utf-8",
    )

    default_db = SessionDB(db_path=root / "state.db")
    restricted_db = SessionDB(db_path=restricted / "state.db")
    create = _RecordingCreate()
    try:
        default_db.create_session("a-before", "cli", model="allowed", model_config={"provider": "openrouter"})
        with pytest.raises(RoutingPolicyError):
            restricted_db.create_session(
                "b-denied", "cli", model="z-ai/glm-5.2", model_config={"provider": "openrouter"},
            )
        assert restricted_db.get_session("b-denied") is None

        agent = SimpleNamespace(
            api_mode="chat_completions", provider="openrouter", model="z-ai/glm-5.2",
            base_url="https://openrouter.ai/api/v1", _session_db=restricted_db,
        )
        with pytest.raises(RoutingPolicyError):
            helpers._dispatch_nonstreaming_api_request(
                agent, {"model": "z-ai/glm-5.2"},
                make_client=lambda *_args, **_kwargs: SimpleNamespace(chat=SimpleNamespace(completions=create)),
            )
        assert create.calls == []

        default_db.create_session("a-after", "cli", model="allowed", model_config={"provider": "openrouter"})
    finally:
        restricted_db.close()
        default_db.close()


def test_auxiliary_fallback_send_uses_session_owner_after_a_to_b_to_a(tmp_path, monkeypatch):
    """A fallback final-send remains governed by B after A's ambient scope resumes."""
    from agent import auxiliary_client as auxiliary
    from hermes_state import SessionDB
    from hermes_cli.routing_policy import RoutingPolicyError, profile_home_for_session_db

    root = tmp_path / "hermes"
    restricted = root / "profiles" / "restricted"
    for config, policy in (
        (root / "config.yaml", "routing_policy:\n  enabled: true\n"),
        (restricted / "config.yaml", "routing_policy:\n  enabled: true\n  deny:\n    models: ['z-ai/*']\n"),
    ):
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(policy, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))  # A → B → A: A is ambient at final send.
    db = SessionDB(db_path=restricted / "state.db")
    class RecordingCreate:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(choices=[])

    create = RecordingCreate()
    client = SimpleNamespace(
        base_url="https://allowed.example/v1", chat=SimpleNamespace(completions=create),
    )
    try:
        with pytest.raises(RoutingPolicyError):
            auxiliary._call_fallback_candidate_sync(
                client, "z-ai/glm-5.2", "fallback", task="title_generation", messages=[],
                temperature=None, max_tokens=None, tools=None, effective_timeout=1,
                effective_extra_body={}, reasoning_config=None,
                profile_home=profile_home_for_session_db(db),
            )
    finally:
        db.close()

    assert create.calls == []


def test_compression_publish_rejects_denied_config_route_without_closing_parent(tmp_path, monkeypatch):
    """A rejected compression child must not publish any half of its handoff."""
    from hermes_state import SessionDB
    from hermes_cli.routing_policy import RoutingPolicyError

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "routing_policy:\n  enabled: true\n  deny:\n    models: ['denied-*']\n", encoding="utf-8",
    )
    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session("parent", "cli", model="allowed-model", model_config={"provider": "openrouter"})
        with pytest.raises(RoutingPolicyError, match="selected model"):
            db.publish_compression_child(
                parent_session_id="parent", child_session_id="child", source="cli",
                messages=[{"role": "user", "content": "handoff"}],
                model=None,
                model_config={"provider": "openrouter", "model": "denied-model"},
                require_compression_lease=False,
            )
        assert db.get_session("child") is None
        assert db.get_session("parent")["ended_at"] is None
        assert db.get_session("parent")["end_reason"] is None
    finally:
        db.close()


def test_import_rejects_denied_persisted_route_without_importing_batch(tmp_path, monkeypatch):
    """Portability admission is preflighted, so one denied route writes no batch rows."""
    from hermes_state import SessionDB

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "routing_policy:\n  enabled: true\n  deny:\n    models: ['denied-*']\n", encoding="utf-8",
    )
    db = SessionDB(db_path=home / "state.db")
    try:
        result = db.import_sessions([
            {"id": "allowed", "model": "allowed-model", "model_config": {"provider": "openrouter"},
             "messages": [{"role": "user", "content": "allowed"}]},
            {"id": "denied", "model": "denied-model",
             "model_config": {"provider": "openrouter"},
             "messages": [{"role": "user", "content": "denied"}]},
        ])
        assert result == {
            "ok": False, "imported": 0, "skipped": 0, "detached": 0,
            "errors": [{"index": 1, "session_id": "denied", "error": "routing policy denies the selected model"}],
        }
        assert db.get_session("allowed") is None
        assert db.get_session("denied") is None
    finally:
        db.close()


def test_create_rejects_denied_top_level_model_when_nested_model_is_allowed(tmp_path, monkeypatch):
    """Resume uses sessions.model first, so admission must do the same."""
    from hermes_state import SessionDB
    from hermes_cli.routing_policy import RoutingPolicyError

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "routing_policy:\n  enabled: true\n  deny:\n    models: ['denied-*']\n", encoding="utf-8",
    )
    db = SessionDB(db_path=home / "state.db")
    try:
        with pytest.raises(RoutingPolicyError, match="selected model"):
            db.create_session(
                "conflict", "cli", model="denied-model",
                model_config={"provider": "openrouter", "model": "allowed-model"},
            )
        assert db.get_session("conflict") is None
    finally:
        db.close()


def test_import_rejects_denied_top_level_model_when_nested_model_is_allowed(tmp_path, monkeypatch):
    """One conflicting imported route rejects the whole batch before writes."""
    from hermes_state import SessionDB

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "routing_policy:\n  enabled: true\n  deny:\n    models: ['denied-*']\n", encoding="utf-8",
    )
    db = SessionDB(db_path=home / "state.db")
    try:
        result = db.import_sessions([
            {"id": "allowed", "model": "allowed-model", "model_config": {"provider": "openrouter"},
             "messages": [{"role": "user", "content": "allowed"}]},
            {"id": "conflict", "model": "denied-model",
             "model_config": {"provider": "openrouter", "model": "allowed-model"},
             "messages": [{"role": "user", "content": "conflict"}]},
        ])
        assert result == {
            "ok": False, "imported": 0, "skipped": 0, "detached": 0,
            "errors": [{"index": 1, "session_id": "conflict", "error": "routing policy denies the selected model"}],
        }
        assert db.get_session("allowed") is None
        assert db.get_session("conflict") is None
    finally:
        db.close()


def test_nested_profile_session_db_uses_named_owner_policy_for_persistence_and_dispatch(tmp_path, monkeypatch):
    """A nested store remains owned by its resolved named profile, never its scratch directory."""
    from agent import chat_completion_helpers as helpers
    from hermes_state import SessionDB
    from hermes_cli.routing_policy import RoutingPolicyError

    home = tmp_path / "hermes"
    restricted = home / "profiles" / "restricted"
    scratch = restricted / "scratch"
    scratch.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text("routing_policy:\n  enabled: true\n", encoding="utf-8")
    (restricted / "config.yaml").write_text(
        "routing_policy:\n  enabled: true\n  deny:\n    models: ['z-ai/*']\n", encoding="utf-8",
    )
    (scratch / "config.yaml").write_text("routing_policy:\n  enabled: true\n", encoding="utf-8")

    db = SessionDB(db_path=scratch / "state.db")
    create = _RecordingCreate()
    try:
        with pytest.raises(RoutingPolicyError):
            db.create_session(
                "nested", "cli", model="z-ai/glm-5.2", model_config={"provider": "openrouter"},
            )
        assert db.get_session("nested") is None

        agent = SimpleNamespace(
            api_mode="chat_completions", provider="openrouter", model="z-ai/glm-5.2",
            base_url="https://openrouter.ai/api/v1", _session_db=db,
        )
        with pytest.raises(RoutingPolicyError):
            helpers._dispatch_nonstreaming_api_request(
                agent, {"model": "z-ai/glm-5.2"},
                make_client=lambda *_args, **_kwargs: SimpleNamespace(chat=SimpleNamespace(completions=create)),
            )
        assert create.calls == []
    finally:
        db.close()


def test_ad_hoc_profile_shaped_session_db_uses_active_policy(tmp_path, monkeypatch):
    """A copied store cannot select a permissive policy merely by its path shape."""
    from hermes_cli.routing_policy import (
        RoutingPolicyError,
        check_route,
        current_routing_policy_for_session_db,
    )
    from hermes_state import SessionDB

    home = tmp_path / "hermes"
    ad_hoc = tmp_path / "ad-hoc" / "profiles" / "relaxed"
    home.mkdir()
    ad_hoc.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "routing_policy:\n  enabled: true\n  deny:\n    models: ['z-ai/*']\n", encoding="utf-8",
    )
    (ad_hoc / "config.yaml").write_text("routing_policy:\n  enabled: true\n", encoding="utf-8")

    db = SessionDB(db_path=ad_hoc / "state.db")
    try:
        with pytest.raises(RoutingPolicyError):
            check_route(
                current_routing_policy_for_session_db(db),
                provider="openrouter", model="z-ai/glm-5.2", base_url="",
            )
    finally:
        db.close()


def test_session_meta_and_runtime_lock_reject_before_write_when_explicit_route_is_required(tmp_path, monkeypatch):
    """All public session persistence APIs admit a complete route before durable writes."""
    import json
    from hermes_state import SessionDB
    from hermes_cli.routing_policy import RoutingPolicyError

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "routing_policy:\n  enabled: true\n  require_explicit: true\n", encoding="utf-8",
    )
    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session("s1", "cli")
        with pytest.raises(RoutingPolicyError, match="explicit provider"):
            db.update_session_meta("s1", json.dumps({"model": "permitted-model"}))
        with pytest.raises(RoutingPolicyError, match="explicit provider"):
            db.update_session_runtime_lock("s1", model="permitted-model", confirmed=True)
        row = db.get_session("s1")
        assert row["model_config"] is None
        assert row["model"] is None
    finally:
        db.close()


def test_session_meta_rejects_denied_route_before_write(tmp_path, monkeypatch):
    """Serialized metadata cannot install a denied route for a later resume."""
    import json
    from hermes_state import SessionDB
    from hermes_cli.routing_policy import RoutingPolicyError

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "routing_policy:\n  enabled: true\n  deny:\n    models: ['z-ai/*']\n", encoding="utf-8",
    )
    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session("s1", "cli")
        with pytest.raises(RoutingPolicyError):
            db.update_session_meta("s1", json.dumps({
                "browser_model_lock": {"provider": "openrouter", "model": "z-ai/glm-5.2", "confirmed": True},
            }))
        assert db.get_session("s1")["model_config"] is None
    finally:
        db.close()


def test_denied_host_config_is_canonicalized_before_matching():
    """Operator host spelling and case cannot weaken the deny rule."""
    from hermes_cli.routing_policy import RoutingPolicyError, check_route

    with pytest.raises(RoutingPolicyError, match="base-url host"):
        check_route(
            {"enabled": True, "deny": {"base_url_hosts": ["HTTPS://API.Z.AI/v1/"]}},
            provider="openrouter", model="allowed", base_url="https://api.z.ai/v1",
        )


def test_auxiliary_primary_denial_blocks_sync_async_and_stream_sends(monkeypatch):
    """Resolved auxiliary wire routes are checked before every primary create call."""
    import asyncio
    from agent import auxiliary_client as auxiliary
    from hermes_cli.routing_policy import RoutingPolicyError

    class SyncCreate:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(choices=[])

    class AsyncCreate:
        def __init__(self):
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(choices=[])

    sync_create, async_create = SyncCreate(), AsyncCreate()
    sync_client = SimpleNamespace(
        base_url="https://denied.example/v1",
        chat=SimpleNamespace(completions=sync_create),
    )
    async_client = SimpleNamespace(
        base_url="https://denied.example/v1",
        chat=SimpleNamespace(completions=async_create),
    )
    monkeypatch.setattr(auxiliary, "current_routing_policy", lambda: {
        "enabled": True, "deny": {"base_url_hosts": ["denied.example"]},
    }, raising=False)

    for client, async_mode, stream in ((sync_client, False, False), (sync_client, False, True), (async_client, True, False)):
        req = auxiliary._PreparedAuxRequest(
            client, "allowed-model", {"model": "allowed-model", "messages": []},
            "openrouter", "openrouter", "allowed-model", "https://requested.example/v1",
            None, None, 30.0, {}, "https://requested.example/v1",
        )
        monkeypatch.setattr(auxiliary, "_plan_aux_call", lambda *a, _req=req, **kw: (_req, {}, {}))
        with pytest.raises(RoutingPolicyError):
            if async_mode:
                asyncio.run(auxiliary._async_call_llm_impl(messages=[]))
            else:
                auxiliary._call_llm_impl(messages=[], stream=stream)

    assert sync_create.calls == []
    assert async_create.calls == []


def test_auxiliary_runtime_symlinked_owner_blocks_send_after_a_to_b_to_a(tmp_path, monkeypatch):
    """Auxiliary primary sends retain logical B's policy when A is ambient again."""
    from agent import auxiliary_client as auxiliary
    from hermes_cli.routing_policy import RoutingPolicyError

    root = tmp_path / "hermes"
    logical_profile = root / "profiles" / "restricted"
    outside_profile = tmp_path / "outside" / "restricted"
    root.mkdir()
    logical_profile.parent.mkdir()
    outside_profile.mkdir(parents=True)
    logical_profile.symlink_to(outside_profile, target_is_directory=True)
    (root / "config.yaml").write_text("routing_policy:\n  enabled: true\n", encoding="utf-8")
    (outside_profile / "config.yaml").write_text(
        "routing_policy:\n  enabled: true\n  deny:\n    models: ['z-ai/*']\n", encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(root))  # A → B → A: A is ambient at send time.

    create = _RecordingCreate()
    client = SimpleNamespace(
        base_url="https://allowed.example/v1", chat=SimpleNamespace(completions=create),
    )
    owner = auxiliary._routing_profile_home({"profile_home": str(logical_profile)})

    assert owner == str(logical_profile)
    with pytest.raises(RoutingPolicyError, match="selected model"):
        auxiliary._guard_auxiliary_wire_route(
            client, {"model": "z-ai/glm-5.2"}, "openrouter", profile_home=owner,
        )
    assert create.calls == []


def test_iteration_summary_denial_blocks_direct_create(monkeypatch):
    """The iteration-limit direct summary call checks its request client's wire route."""
    from agent import chat_completion_helpers as helpers
    from hermes_cli.routing_policy import RoutingPolicyError

    create = _RecordingCreate()
    client = SimpleNamespace(base_url="https://denied.example/v1", chat=SimpleNamespace(completions=create))
    agent = SimpleNamespace(
        provider="openrouter", model="allowed", base_url="https://requested.example/v1",
        _build_api_kwargs=lambda _messages: {"model": "allowed", "messages": []},
        _ensure_primary_openai_client=lambda **_kwargs: client,
    )
    monkeypatch.setattr(helpers, "sanitize_outbound_kwargs", lambda *_args: None)
    monkeypatch.setattr(helpers, "current_routing_policy", lambda: {
        "enabled": True, "deny": {"base_url_hosts": ["denied.example"]},
    })

    attempt = helpers._chat_summary_attempt(agent, [], "summary-request")
    with pytest.raises(RoutingPolicyError):
        attempt(0)

    assert create.calls == []


def test_denied_auxiliary_main_fallback_is_terminal_and_does_not_resolve_next(monkeypatch):
    """A denied fallback must not advance the auxiliary main-fallback chain."""
    import agent.auxiliary_client as auxiliary
    from hermes_cli.routing_policy import RoutingPolicyError

    entries = [
        {"provider": "denied", "model": "first"},
        {"provider": "recording", "model": "second"},
    ]
    resolved = []
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"fallback_model": entries})
    monkeypatch.setattr(auxiliary, "_failed_backend_skip", lambda *_a, **_kw: lambda *_a: False)
    monkeypatch.setattr(auxiliary, "_is_provider_unhealthy", lambda *_a, **_kw: False)

    def resolve(entry):
        resolved.append(entry["provider"])
        if entry["provider"] == "denied":
            raise RoutingPolicyError("denied")
        return object(), entry["model"]

    monkeypatch.setattr(auxiliary, "_resolve_fallback_entry", resolve)

    with pytest.raises(RoutingPolicyError, match="denied"):
        auxiliary._try_main_fallback_chain("task")

    assert resolved == ["denied"]
