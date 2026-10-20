"""Behavior contracts for Codex App Server host auth and conservative quota recovery."""

from __future__ import annotations

import base64
import contextvars
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent.credential_pool import CredentialPool, PooledCredential, STATUS_DEAD, STATUS_EXHAUSTED
from agent.transports.codex_app_server import CodexAppServerError, CodexAppServerTransportError
from agent.transports.codex_app_server_quota import (
    is_replay_safe_pretool_quota_failure,
    recover_codex_app_server_quota,
    typed_native_quota_failure,
)
from agent.transports.codex_app_server_session import CodexAppServerSession, TurnResult
from agent.transports.codex_app_server_host_auth import CodexAppServerHostAuth
from agent.codex_runtime import run_codex_app_server_turn


class FakeClient:
    def __init__(self, **options: Any) -> None:
        self.options = options
        self.initialize_params: dict[str, Any] = {}
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.responses: list[tuple[Any, dict[str, Any]]] = []
        self.errors: list[tuple[Any, int, str]] = []
        self.notifications: list[dict[str, Any]] = []
        self.server_requests: list[dict[str, Any]] = []
        self.logged_account: str | None = None
        self.login_response: Any = {"type": "chatgptAuthTokens"}
        self.account_response: Any = None
        self.fail_turn_start = False
        self.missing_turn_id = False
        self.fail_login_message: str | None = None
        self.complete_after_reply = False
        self.closed = False

    def initialize(self, **kwargs: Any) -> dict[str, Any]:
        self.initialize_params = kwargs
        return {}

    def request(self, method: str, params: dict | None = None, timeout: float = 30.0) -> dict:
        params = params or {}
        self.requests.append((method, dict(params)))
        if method == "account/login/start":
            if self.fail_login_message:
                raise CodexAppServerError(code=-32001, message=self.fail_login_message)
            self.logged_account = params.get("chatgptAccountId")
            return self.login_response
        if method == "account/read":
            if self.account_response is not None:
                return self.account_response
            return {"requiresOpenaiAuth": True,
                    "account": {"type": "chatgpt", "email": None, "planType": "plus"}}
        if method == "thread/start":
            return {"thread": {"id": "thread-stable"}}
        if method == "thread/resume":
            return {"thread": {"id": params["threadId"]}}
        if method == "turn/start":
            if self.fail_turn_start:
                raise CodexAppServerTransportError(code=-32000, message="transport lost")
            if self.missing_turn_id:
                return {"turn": {}}
            return {"turn": {"id": f"turn-{sum(m == 'turn/start' for m, _ in self.requests)}"}}
        return {}

    def respond(self, request_id: Any, result: dict[str, Any]) -> None:
        self.responses.append((request_id, result))
        self._complete_after_auth_reply()

    def respond_error(self, request_id: Any, code: int, message: str, data: Any = None) -> None:
        self.errors.append((request_id, code, message))
        self._complete_after_auth_reply()

    def _complete_after_auth_reply(self) -> None:
        if self.complete_after_reply:
            self.complete_after_reply = False
            self.queue_note("turn/completed", threadId="thread-stable",
                            turn={"id": "turn-1", "status": "completed"})

    def take_notification(self, timeout: float = 0.0) -> dict | None:
        return self.notifications.pop(0) if self.notifications else None

    def take_server_request(self, timeout: float = 0.0) -> dict | None:
        return self.server_requests.pop(0) if self.server_requests else None

    def stderr_tail(self, n: int = 20) -> list[str]:
        return []

    def is_alive(self) -> bool:
        return not self.closed

    def close(self) -> None:
        self.closed = True

    def queue_note(self, method: str, **params: Any) -> None:
        self.notifications.append({"method": method, "params": params})

    def queue_request(self, method: str, request_id: str = "request-1", **params: Any) -> None:
        self.server_requests.append({"id": request_id, "method": method, "params": params})



def _jwt(account: str, subject: str) -> str:
    def segment(payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    claims = {"sub": subject, "exp": int(time.time()) + 3600,
              "https://api.openai.com/auth": {"chatgpt_account_id": account}}
    return f"{segment({'alg': 'none'})}.{segment(claims)}.synthetic"


def _entry(credential_id: str, account: str, subject: str, priority: int = 0) -> PooledCredential:
    return PooledCredential(
        provider="openai-codex", id=credential_id, label=f"synthetic-{priority}",
        auth_type="oauth", priority=priority, source="manual:synthetic",
        access_token=_jwt(account, subject), refresh_token=f"refresh-{credential_id}",
    )


def _synthetic_home(tmp_path: Path, monkeypatch: Any) -> Path:
    os_home = tmp_path / "os-home"
    home = tmp_path / "hermes-home"
    home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(os_home))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: os_home)
    return home


def _persist_pool(entries: list[PooledCredential], *, removed_ids: list[str] | None = None) -> None:
    from hermes_cli.auth import write_credential_pool

    write_credential_pool("openai-codex", [entry.to_dict() for entry in entries], removed_ids=removed_ids)


def _session(pool: CredentialPool, entry: PooledCredential, client: FakeClient | None = None):
    client = client or FakeClient()
    binding = CodexAppServerHostAuth.from_selected_credential(pool=pool, entry=entry)
    session = CodexAppServerSession(
        cwd="/synthetic/workspace", codex_home="/synthetic/codex-home", host_auth=binding,
        client_factory=cast(Any, lambda **kwargs: client),
    )
    return session, client, binding


def _failed_quota_note(client: FakeClient, *, code: str = "usageLimitExceeded", turn_id: str = "turn-1") -> None:
    client.queue_note("turn/completed", threadId="thread-stable",
                      turn={"id": turn_id, "status": "failed",
                            "error": {"message": "quota", "codexErrorInfo": code}})


def test_runtime_session_factory_binds_selected_oauth_row(monkeypatch):
    import agent.codex_runtime as runtime
    from agent.transports import codex_app_server_session as native
    from hermes_cli import config

    selected = _entry("factory-row", "acct-selected", "subject-selected")
    pool = CredentialPool("openai-codex", [selected])
    owner = SimpleNamespace(provider="openai-codex", model="synthetic-model", api_key=selected.access_token,
                            _credential_pool=pool, _credential_pool_entry_id=selected.id,
                            session_cwd="/synthetic/workspace")
    captured = {}
    monkeypatch.setattr(runtime, "_codex_developer_instructions", lambda agent: "")
    monkeypatch.setattr(runtime, "_codex_model_provider", lambda agent: None)
    monkeypatch.setattr(runtime, "_stored_codex_thread_id", lambda agent: None)
    monkeypatch.setattr(config, "load_config", dict)
    monkeypatch.setattr(native, "CodexAppServerSession", lambda **kwargs: captured.update(kwargs) or SimpleNamespace())

    runtime._ensure_codex_session(owner, [])

    assert captured.get("host_auth") is not None, "selected OAuth row must reach the native session factory"
    assert captured["host_auth"].credential_id == selected.id


def test_selected_pool_row_authenticates_instead_of_singleton_and_uses_ephemeral_storage(tmp_path, monkeypatch):
    selected = _entry("selected-row", "acct-selected", "subject-selected")
    pool = CredentialPool("openai-codex", [selected])
    singleton = _jwt("acct-singleton", "subject-singleton")
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "auth.json").write_text(json.dumps({"providers": {"openai-codex": {
        "tokens": {"access_token": singleton}}}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    agent = SimpleNamespace(provider="openai-codex", api_key=selected.access_token,
                            _credential_pool=pool, _credential_pool_entry_id=selected.id)
    binding = CodexAppServerHostAuth.from_agent(agent)
    client = FakeClient()
    session = CodexAppServerSession(
        cwd="/synthetic/workspace", codex_home=str(tmp_path / "codex-home"), host_auth=binding,
        client_factory=lambda **kwargs: (captured.update(kwargs) or client),
    )
    captured: dict[str, Any] = {}

    session.ensure_started()

    login = next(params for method, params in client.requests if method == "account/login/start")
    assert login["type"] == "chatgptAuthTokens"
    assert login["chatgptAccountId"] == "acct-selected"
    assert login["accessToken"] == selected.access_token
    assert client.initialize_params["capabilities"]["experimentalApi"] is True
    assert 'cli_auth_credentials_store="ephemeral"' in captured["extra_args"]
    assert captured["env"] == {"OPENAI_API_KEY": None, "OPENAI_BASE_URL": None}
    assert selected.access_token not in repr(captured)
    assert not (tmp_path / "codex-home" / "auth.json").exists()
    assert selected.access_token not in repr(binding)
    assert singleton not in repr(login)


def test_client_env_override_can_remove_inherited_openai_fallback(monkeypatch):
    import io
    import subprocess

    from agent.transports import codex_app_server as transport

    captured: dict[str, Any] = {}

    class FakePopen:
        def __init__(self, _cmd: list[str], *args: Any, **kwargs: Any) -> None:
            captured.update(kwargs["env"])
            self.stdin, self.stdout, self.stderr = io.BytesIO(), io.BytesIO(), io.BytesIO()
            self.pid, self.returncode = 1, 0

        def poll(self) -> int:
            return 0

    monkeypatch.setattr(transport, "hermes_subprocess_env", lambda **_: {
        "OPENAI_API_KEY": "synthetic-api-key", "OPENAI_BASE_URL": "https://synthetic.invalid",
        "PATH": "/synthetic/bin",
    })
    monkeypatch.setattr(subprocess, "Popen", FakePopen)

    client = transport.CodexAppServerClient(
        codex_bin="codex", env={"OPENAI_API_KEY": None, "OPENAI_BASE_URL": None},
    )
    client._closed = True

    assert "OPENAI_API_KEY" not in captured and "OPENAI_BASE_URL" not in captured
    assert captured["PATH"] == "/synthetic/bin"



def test_startup_error_redacts_tokens_and_does_not_write_codex_auth(tmp_path):
    selected = _entry("redact-row", "acct-redact", "subject-redact")
    pool = CredentialPool("openai-codex", [selected])
    session, client, binding = _session(pool, selected)
    client.fail_login_message = f"upstream included {selected.access_token} and refresh-redact"
    session._codex_home = str(tmp_path / "codex-home")
    codex_home = Path(session._codex_home)

    result = session.run_turn("hello", turn_timeout=1)

    assert result.should_retire
    assert "upstream included" not in (result.error or "")
    assert selected.access_token not in repr(result) + repr(binding)
    assert not (codex_home / "auth.json").exists()


@pytest.mark.parametrize("hint", ["matching", "null", "missing"])
def test_refresh_uses_serialized_same_pool_row_and_replies_only_after_identity_verification(tmp_path, monkeypatch, hint):
    from hermes_cli import auth

    _synthetic_home(tmp_path, monkeypatch)
    decoy = _entry("refresh-decoy", "acct-decoy", "subject-decoy")
    original = _entry("refresh-row", "acct-refresh", "subject-refresh", priority=1)
    pool = CredentialPool("openai-codex", [decoy, original])
    _persist_pool([decoy, original])
    rotated = _jwt("acct-refresh", "subject-refresh")
    calls: list[tuple[str, str]] = []
    refresh_ids: list[str | None] = []
    profile_scope = contextvars.ContextVar("test_codex_profile", default="ambient")
    profile_token = profile_scope.set("selected-profile")
    owner = SimpleNamespace(provider="openai-codex", api_key=original.runtime_api_key,
                            base_url="https://synthetic.invalid", _credential_pool=pool,
                            _credential_pool_entry_id=original.id)
    binding = CodexAppServerHostAuth.from_agent(owner)
    client = FakeClient()
    session = CodexAppServerSession(
        cwd="/synthetic/workspace", codex_home="/synthetic/codex-home", host_auth=binding,
        client_factory=cast(Any, lambda **kwargs: client),
    )
    profile_scope.reset(profile_token)
    pool_refresh = pool.try_refresh_matching

    def refresh_selected(
        *, credential_id: str | None = None, api_key_hint: str | None = None,
        expected_principal: tuple[str, str] | None = None,
    ):
        assert profile_scope.get() == "selected-profile"
        refresh_ids.append(credential_id)
        return pool_refresh(
            api_key_hint=api_key_hint, credential_id=credential_id, expected_principal=expected_principal,
        )

    monkeypatch.setattr(pool, "try_refresh_matching", refresh_selected)

    def refresh(access_token: str, refresh_token: str) -> dict[str, str]:
        calls.append((access_token, refresh_token))
        return {"access_token": rotated, "refresh_token": "refresh-next", "last_refresh": "synthetic"}

    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", refresh)
    client.complete_after_reply = True
    refresh_params: dict[str, Any] = {"reason": "unauthorized"}
    if hint == "matching":
        refresh_params["previousAccountId"] = "acct-refresh"
    elif hint == "null":
        refresh_params["previousAccountId"] = None
    client.queue_request("account/chatgptAuthTokens/refresh", **refresh_params)

    result = session.run_turn("continue", turn_timeout=1)

    assert result.error is None
    assert len(calls) == 1 and calls[0][1] == "refresh-refresh-row"
    assert refresh_ids == [original.id]
    assert next(entry for entry in pool.entries() if entry.id == original.id).access_token == rotated
    assert next(entry for entry in pool.entries() if entry.id == decoy.id).access_token == decoy.access_token
    assert len(client.responses) == 1
    assert client.responses[0][1]["accessToken"] == rotated
    assert client.responses[0][1]["chatgptAccountId"] == "acct-refresh"
    assert owner.api_key == rotated and owner._credential_pool_entry_id == original.id
    assert client.errors == []


@pytest.mark.parametrize("mismatch", ["previous_account", "account", "subject", "row"])
def test_refresh_denies_previous_account_principal_or_row_mismatch(tmp_path, monkeypatch, mismatch):
    from hermes_cli import auth

    _synthetic_home(tmp_path, monkeypatch)
    original = _entry(f"mismatch-{mismatch}", "acct-original", "subject-original")
    pool = CredentialPool("openai-codex", [original])
    _persist_pool([original])
    new_account = "acct-other" if mismatch == "account" else "acct-original"
    new_subject = "subject-other" if mismatch == "subject" else "subject-original"
    refresh_calls: list[bool] = []

    def refresh(*_):
        refresh_calls.append(True)
        return {"access_token": _jwt(new_account, new_subject), "refresh_token": "refresh-rotated"}

    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", refresh)
    session, client, _binding = _session(pool, original)
    if mismatch == "row":
        monkeypatch.setattr(pool, "try_refresh_matching", lambda **_: _entry(
            "different-row", "acct-original", "subject-original",
        ))
    client.complete_after_reply = True
    previous = "acct-other" if mismatch == "previous_account" else "acct-original"
    client.queue_request("account/chatgptAuthTokens/refresh", previousAccountId=previous, reason="unauthorized")

    session.run_turn("continue", turn_timeout=1)

    assert client.responses == []
    assert len(client.errors) == 1
    assert "acct-original" not in client.errors[0][2]
    assert "refresh-rotated" not in repr(client.errors)
    assert bool(refresh_calls) is (mismatch in {"account", "subject"})


def test_failed_terminal_pretool_quota_rotates_once_and_retries_same_thread(monkeypatch):
    from hermes_cli import auth

    first, second = _entry("quota-first", "acct-first", "subject-first"), _entry("quota-next", "acct-next", "subject-next", 1)
    pool = CredentialPool("openai-codex", [first, second])
    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", lambda *_: pytest.fail("quota rotation must not refresh"))
    session, client, _binding = _session(pool, first)
    client.queue_note("turn/started", threadId="thread-stable", turn={"id": "turn-1"})
    client.queue_note("thread/status/changed", threadId="thread-stable", status={"type": "active", "activeFlags": []})
    client.queue_note("account/rateLimits/updated", rateLimits={})
    client.queue_note("thread/tokenUsage/updated", threadId="thread-stable", turnId="turn-1",
                      tokenUsage={"last": {}, "total": {}})
    client.queue_note("item/started", threadId="thread-stable",
                      item={"id": "input-echo", "type": "userMessage", "content": [{"type": "text", "text": "same input"}]})
    client.queue_note("item/completed", threadId="thread-stable",
                      item={"id": "input-echo", "type": "userMessage", "content": [{"type": "text", "text": "same input"}]})
    client.queue_note("error", threadId="thread-stable", turnId="turn-1", willRetry=False,
                      error={"message": "quota", "codexErrorInfo": "usageLimitExceeded"})
    _failed_quota_note(client)
    client.queue_note("turn/completed", threadId="thread-stable",
                      turn={"id": "turn-2", "status": "completed"})
    agent = SimpleNamespace(provider="openai-codex", model="gpt-5.6-codex", api_key=first.access_token,
                            base_url="https://chatgpt.com/backend-api/codex", _credential_pool=pool,
                            _credential_pool_entry_id=first.id, _codex_session=session,
                            _codex_session_prompt=None, _codex_session_model_provider=None,
                            request_overrides={}, session_cwd="/synthetic/workspace", _session_db=None,
                            session_id=None, session_api_calls=0, context_compressor=None,
                            _iters_since_skill=0, _skill_nudge_interval=0, valid_tool_names=set(),
                            _interrupt_requested=False, _interrupt_message=None)

    final = run_codex_app_server_turn(
        agent, user_message="same input", original_user_message="same input", messages=[], effective_task_id="synthetic",
    )

    starts = [params for method, params in client.requests if method == "thread/start"]
    turns = [params for method, params in client.requests if method == "turn/start"]
    assert final["error"] is None and final["completed"] is True
    assert final["api_calls"] == 2 and agent.session_api_calls == 2
    assert final["codex_thread_id"] == "thread-stable"
    assert len(starts) == 1
    assert starts[0]["cwd"] == "/synthetic/workspace"
    assert [params["threadId"] for params in turns] == ["thread-stable", "thread-stable"]
    assert [params["input"][0]["text"] for params in turns] == ["same input", "same input"]
    assert agent._credential_pool_entry_id == second.id and agent.api_key == second.access_token
    assert pool.entries()[0].last_status == STATUS_EXHAUSTED
    assert client.logged_account == "acct-next"


def _queue_quota_barrier(barrier: str, client: FakeClient, monkeypatch: Any) -> None:
    def queue_refresh_request() -> None:
        from hermes_cli import auth

        monkeypatch.setattr(auth, "refresh_codex_oauth_pure", lambda *_: {
            "access_token": _jwt("acct-first", "subject-first"), "refresh_token": "refresh-next",
        })
        client.queue_request("account/chatgptAuthTokens/refresh", previousAccountId="acct-first", reason="unauthorized")

    def queue_error(error: dict[str, Any], will_retry: bool) -> None:
        client.queue_note("error", threadId="thread-stable", turnId="turn-1", willRetry=will_retry, error=error)

    item_type_by_barrier = {
        "imageView": "imageView", "command": "commandExecution", "fileChange": "fileChange",
        "unknown_item": "futureAction",
    }
    handlers = {
        "approval": lambda: client.queue_request("item/commandExecution/requestApproval", command="run"),
        "refresh_request": queue_refresh_request,
        "unknown_event": lambda: client.queue_note("future/turnWork", threadId="thread-stable", turnId="turn-1"),
        "unclassified_error": lambda: queue_error({"message": "server overloaded", "codexErrorInfo": "serverOverloaded"}, False),
        "retryable_error": lambda: queue_error({"message": "quota", "codexErrorInfo": "usageLimitExceeded"}, True),
        "output": lambda: client.queue_note("item/agentMessage/delta", threadId="thread-stable", delta="partial output"),
        "reasoning": lambda: client.queue_note("item/reasoning/delta", threadId="thread-stable", delta="partial reasoning"),
        "plan": lambda: client.queue_note("turn/plan/updated", threadId="thread-stable", turnId="turn-1", plan=[]),
        "interruption": lambda: None,
        **{
            name: lambda item_type=item_type: client.queue_note(
                "item/started", threadId="thread-stable", item={"type": item_type, "id": "side-effect"},
            )
            for name, item_type in item_type_by_barrier.items()
        },
    }
    handlers[barrier]()


@pytest.mark.parametrize("barrier", [
    "imageView", "command", "fileChange", "reasoning", "plan", "approval", "refresh_request", "unknown_item",
    "unknown_event", "output", "interruption", "transport", "unclassified_error", "retryable_error",
])
def test_quota_recovery_never_replays_after_any_side_effect_or_ambiguous_failure(barrier, monkeypatch):
    first, second = _entry(f"barrier-{barrier}", "acct-first", "subject-first"), _entry(
        f"barrier-next-{barrier}", "acct-next", "subject-next", 1)
    pool = CredentialPool("openai-codex", [first, second])
    session, client, _binding = _session(pool, first)
    if barrier == "transport":
        client.fail_turn_start = True
    else:
        _queue_quota_barrier(barrier, client, monkeypatch)
        _failed_quota_note(client)
        if barrier == "interruption":
            client.notifications[0]["params"]["turn"]["status"] = "interrupted"
    agent = SimpleNamespace(provider="openai-codex", model="gpt-5.6-codex", api_key=first.access_token,
                            _credential_pool=pool, _credential_pool_entry_id=first.id, _codex_session=session)
    turn = session.run_turn("do not duplicate", turn_timeout=1)
    retried: list[bool] = []

    result = recover_codex_app_server_quota(agent, turn, retry_turn=lambda: retried.append(True) or TurnResult())

    assert retried == []
    assert result is turn
    assert all(entry.last_status != STATUS_EXHAUSTED for entry in pool.entries())
    assert not is_replay_safe_pretool_quota_failure(turn)


@pytest.mark.parametrize("ending", ["completed", "interrupted", "partial_output", "missing_completion"])
def test_quota_error_notification_is_not_a_terminal_failed_turn(ending):
    first = _entry(f"nonterminal-{ending}", "acct-first", "subject-first")
    second = _entry(f"nonterminal-next-{ending}", "acct-next", "subject-next", 1)
    pool = CredentialPool("openai-codex", [first, second])
    session, client, _binding = _session(pool, first)
    client.queue_note("error", threadId="thread-stable", turnId="turn-1", willRetry=False,
                      error={"message": "quota", "codexErrorInfo": "usageLimitExceeded"})
    if ending == "partial_output":
        client.queue_note("item/agentMessage/delta", threadId="thread-stable", turnId="turn-1",
                          delta="already delivered output")
        _failed_quota_note(client)
    elif ending != "missing_completion":
        client.queue_note("turn/completed", threadId="thread-stable",
                          turn={"id": "turn-1", "status": ending})
    agent = SimpleNamespace(provider="openai-codex", model="synthetic-model", api_key=first.access_token,
                            base_url="https://synthetic.invalid", _credential_pool=pool,
                            _credential_pool_entry_id=first.id, _codex_session=session)
    turn = session.run_turn("never duplicate unfinished work", turn_timeout=1)
    retried = []

    result = recover_codex_app_server_quota(agent, turn, retry_turn=lambda: retried.append(True) or TurnResult())

    assert retried == []
    assert result is turn
    assert not is_replay_safe_pretool_quota_failure(turn)
    assert all(entry.last_status != STATUS_EXHAUSTED for entry in pool.entries())


def test_only_native_typed_quota_errors_and_preconnection_http429_are_replayable():
    assert typed_native_quota_failure({"message": "typed", "codexErrorInfo": "usageLimitExceeded"}) == "usageLimitExceeded"
    assert typed_native_quota_failure({"message": "typed", "codexErrorInfo": "rateLimitExceeded"}) == "rateLimitExceeded"
    assert typed_native_quota_failure({"codexErrorInfo": {"httpConnectionFailed": {"httpStatusCode": 429}}}) == "http429"
    assert typed_native_quota_failure({"codexErrorInfo": {"responseStreamConnectionFailed": {"httpStatusCode": 429}}}) == "http429"
    for rejected in (
        {"code": "usageLimitExceeded"},
        {"code": "rateLimitExceeded"},
        {"statusCode": 429, "code": "upstreamError"},
        {"message": "usageLimitExceeded rateLimitExceeded"},
        {"codexErrorInfo": {"httpConnectionFailed": {"httpStatusCode": "429"}}},
        {"codexErrorInfo": {"httpConnectionFailed": {"httpStatusCode": True}}},
        {"codexErrorInfo": {"responseStreamDisconnected": {"httpStatusCode": 429}}},
        {"codexErrorInfo": {"responseTooManyFailedAttempts": {"httpStatusCode": 429}}},
        {"codexErrorInfo": {"httpConnectionFailed": {"httpStatusCode": 429}, "other": {}}},
        {"codexErrorInfo": {"httpConnectionFailed": {"httpStatusCode": 429, "futureField": "ambiguous"}}},
        {"codexErrorInfo": "futureQuotaVariant"},
    ):
        assert typed_native_quota_failure(rejected) is None


def test_host_auth_factory_redacts_pool_getter_errors():
    class PoisonPool:
        def entries(self):
            raise RuntimeError("synthetic-secret-poison")

    agent = SimpleNamespace(provider="openai-codex", api_key="synthetic-api-key",
                            _credential_pool=PoisonPool(), _credential_pool_entry_id="selected-row")

    with pytest.raises(CodexAppServerError, match="host authentication failed") as caught:
        CodexAppServerHostAuth.from_agent(agent)

    assert "synthetic-secret-poison" not in str(caught.value)


def test_host_auth_refuses_missing_selected_pool_id_instead_of_using_singleton():
    selected = _entry("selected-required", "acct-selected", "subject-selected")
    pool = CredentialPool("openai-codex", [selected])
    singleton = _jwt("acct-singleton", "subject-singleton")
    agent = SimpleNamespace(provider="openai-codex", api_key=singleton, _credential_pool=pool,
                            _credential_pool_entry_id=None)

    with pytest.raises(CodexAppServerError, match="host authentication failed"):
        CodexAppServerHostAuth.from_agent(agent)


def test_host_auth_selects_builtin_openai_process_and_thread_provider():
    selected = _entry("provider-row", "acct-provider", "subject-provider")
    pool = CredentialPool("openai-codex", [selected])
    binding = CodexAppServerHostAuth.from_selected_credential(pool=pool, entry=selected)
    client = FakeClient()
    captured: dict[str, Any] = {}
    session = CodexAppServerSession(
        cwd="/synthetic/workspace", host_auth=binding,
        client_factory=cast(Any, lambda **kwargs: (captured.update(kwargs) or client)),
    )

    session.ensure_started()

    thread_start = next(params for method, params in client.requests if method == "thread/start")
    assert session._model_provider == "openai"
    assert thread_start["modelProvider"] == "openai"
    assert 'model_provider="openai"' in captured["extra_args"]


def test_host_auth_rejects_conflicting_native_provider():
    selected = _entry("provider-conflict", "acct-provider", "subject-provider")
    pool = CredentialPool("openai-codex", [selected])
    binding = CodexAppServerHostAuth.from_selected_credential(pool=pool, entry=selected)

    with pytest.raises(CodexAppServerError, match="host authentication failed"):
        CodexAppServerSession(
            cwd="/synthetic/workspace", host_auth=binding, model_provider="external-gateway",
        )


@pytest.mark.parametrize(("login_response", "account_response"), [
    ({}, {"requiresOpenaiAuth": True, "account": {"type": "chatgpt", "email": None, "planType": "plus"}}),
    ({"type": "apiKey"}, {"requiresOpenaiAuth": True,
                         "account": {"type": "chatgpt", "email": None, "planType": "plus"}}),
    ({"type": "chatgptAuthTokens"}, {"account": {"type": "chatgpt"}}),
    ({"type": "chatgptAuthTokens"}, {"requiresOpenaiAuth": False, "account": {"type": "chatgpt"}}),
    ({"type": "chatgptAuthTokens"}, {"requiresOpenaiAuth": False,
                                     "account": {"type": "chatgpt", "requiresOpenaiAuth": True}}),
    ({"type": "chatgptAuthTokens"}, {"account": {"type": "chatgpt", "requiresOpenaiAuth": True}}),
])
def test_host_auth_requires_native_login_and_account_success_shape(login_response, account_response):
    selected = _entry("auth-response", "acct-response", "subject-response")
    pool = CredentialPool("openai-codex", [selected])
    binding = CodexAppServerHostAuth.from_selected_credential(pool=pool, entry=selected)
    client = FakeClient()
    client.login_response = login_response
    client.account_response = account_response

    with pytest.raises(CodexAppServerError, match="host authentication failed"):
        CodexAppServerSession(cwd="/synthetic/workspace", host_auth=binding,
                              client_factory=cast(Any, lambda **_: client)).ensure_started()


def test_pool_host_refresh_does_not_fall_back_from_removed_row_to_current_row(tmp_path, monkeypatch):
    from hermes_cli import auth

    _synthetic_home(tmp_path, monkeypatch)
    removed = _entry("removed-A", "acct-A", "subject-A")
    current = _entry("current-B", "acct-B", "subject-B", priority=1)
    pool = CredentialPool("openai-codex", [removed, current])
    _persist_pool([removed, current])
    session, client, _binding = _session(pool, removed)
    # The in-memory selection is stale, but the pool-store row is authoritative for a host-auth refresh.
    _persist_pool([current], removed_ids=[removed.id])
    refresh_calls: list[str] = []
    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", lambda access, refresh: (
        refresh_calls.append(refresh) or {
            "access_token": _jwt("acct-B", "subject-B"), "refresh_token": "refresh-B-next",
        }
    ))
    client.complete_after_reply = True
    client.queue_request("account/chatgptAuthTokens/refresh", previousAccountId="acct-A", reason="unauthorized")

    session.run_turn("continue", turn_timeout=1)

    from hermes_cli.auth import read_credential_pool
    stored_b = next(row for row in read_credential_pool().get("openai-codex", []) if row["id"] == current.id)
    assert refresh_calls == []
    assert stored_b["refresh_token"] == current.refresh_token
    assert stored_b.get("last_status") in (None, "ok")
    assert client.responses == []
    assert len(client.errors) == 1


def test_pool_refresh_principal_mismatch_quarantines_without_persisting_foreign_tokens_or_singleton(
    tmp_path, monkeypatch,
):
    from hermes_cli import auth

    home = _synthetic_home(tmp_path, monkeypatch)
    original = _entry("refresh-principal", "acct-original", "subject-original")
    pool = CredentialPool("openai-codex", [original])
    _persist_pool([original])
    singleton = {"access_token": "synthetic-singleton-access", "refresh_token": "synthetic-singleton-refresh"}
    auth_file = home / "auth.json"
    auth_data = json.loads(auth_file.read_text(encoding="utf-8"))
    auth_data["providers"] = {"openai-codex": {"tokens": dict(singleton)}}
    auth_file.write_text(json.dumps(auth_data), encoding="utf-8")
    foreign = _jwt("acct-foreign", "subject-foreign")
    refresh_calls: list[str] = []

    def refresh(access: str, refresh_token: str) -> dict[str, str]:
        refresh_calls.append(refresh_token)
        return {"access_token": foreign, "refresh_token": "foreign-next"}

    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", refresh)
    session, client, _binding = _session(pool, original)
    client.complete_after_reply = True
    client.queue_request("account/chatgptAuthTokens/refresh", previousAccountId="acct-original", reason="unauthorized")

    session.run_turn("continue", turn_timeout=1)

    from hermes_cli.auth import read_credential_pool
    rows = cast(list[dict[str, Any]], read_credential_pool("openai-codex"))
    stored = next((row for row in rows if row["id"] == original.id), None)
    assert stored is not None, (
        f"pool row missing; memory statuses={[(row.id, row.last_status) for row in pool.entries()]}, "
        f"persisted ids={[row.get('id') for row in rows]}"
    )
    stored_auth = json.loads(auth_file.read_text(encoding="utf-8"))
    assert refresh_calls == [original.refresh_token]
    assert stored["last_status"] == STATUS_DEAD
    assert stored["access_token"] == original.access_token and stored["refresh_token"] == original.refresh_token
    assert foreign not in json.dumps(stored) and "foreign-next" not in json.dumps(stored)
    assert stored_auth["providers"]["openai-codex"]["tokens"] == singleton
    assert client.responses == [] and len(client.errors) == 1


@pytest.mark.parametrize("change", ["selected-B", "selected-none", "pool-object", "runtime-bearer"])
def test_host_auth_revalidates_runtime_binding_before_second_turn(change):
    first = _entry(f"binding-A-{change}", "acct-A", "subject-A")
    second = _entry(f"binding-B-{change}", "acct-B", "subject-B", priority=1)
    pool = CredentialPool("openai-codex", [first, second])
    owner = SimpleNamespace(provider="openai-codex", api_key=first.runtime_api_key,
                            _credential_pool=pool, _credential_pool_entry_id=first.id)
    binding = CodexAppServerHostAuth.from_agent(owner)
    client = FakeClient()
    session = CodexAppServerSession(
        cwd="/synthetic/workspace", host_auth=binding,
        client_factory=cast(Any, lambda **_: client),
    )
    client.queue_note("turn/completed", threadId="thread-stable", turn={"id": "turn-1", "status": "completed"})
    first_result = session.run_turn("first", turn_timeout=1)
    assert first_result.error is None

    if change == "selected-B":
        owner._credential_pool_entry_id = second.id
    elif change == "selected-none":
        owner._credential_pool_entry_id = None
    elif change == "pool-object":
        owner._credential_pool = CredentialPool("openai-codex", [first, second])
    else:
        owner.api_key = "synthetic-different-runtime-bearer"
    before = len([method for method, _ in client.requests if method == "turn/start"])

    rejected = session.run_turn("second", turn_timeout=1)

    after = len([method for method, _ in client.requests if method == "turn/start"])
    assert rejected.should_retire and rejected.error
    assert after == before == 1


def test_standalone_host_auth_revalidates_exact_bound_row_and_principal():
    first = _entry("standalone-A", "acct-A", "subject-A")
    replacement = _entry("standalone-A", "acct-B", "subject-B")
    pool = CredentialPool("openai-codex", [first])
    binding = CodexAppServerHostAuth.from_selected_credential(pool=pool, entry=first)
    client = FakeClient()
    pool._entries = [replacement]
    session = CodexAppServerSession(
        cwd="/synthetic/workspace", host_auth=binding,
        client_factory=cast(Any, lambda **_: client),
    )

    result = session.run_turn("never sent", turn_timeout=1)

    assert result.should_retire and result.error
    assert not any(method in {"thread/start", "thread/resume", "turn/start"} for method, _ in client.requests)


@pytest.mark.parametrize("scope", [
    "missing_thread", "null_thread", "foreign_thread", "missing_turn", "null_turn", "foreign_turn",
    "contradictory_thread", "missing_start_turn_id",
])
def test_native_quota_retry_requires_explicit_consistent_turn_scope(scope):
    first = _entry(f"scope-A-{scope}", "acct-A", "subject-A")
    second = _entry(f"scope-B-{scope}", "acct-B", "subject-B", priority=1)
    pool = CredentialPool("openai-codex", [first, second])
    session, client, _binding = _session(pool, first)
    params: dict[str, Any] = {
        "threadId": "thread-stable",
        "turn": {"id": "turn-1", "status": "failed",
                 "error": {"message": "quota", "codexErrorInfo": "usageLimitExceeded"}},
    }
    mutations = {
        "missing_thread": lambda: params.pop("threadId"),
        "null_thread": lambda: params.__setitem__("threadId", None),
        "foreign_thread": lambda: params.__setitem__("threadId", "thread-foreign"),
        "missing_turn": lambda: params["turn"].pop("id"),
        "null_turn": lambda: params["turn"].__setitem__("id", None),
        "foreign_turn": lambda: params["turn"].__setitem__("id", "turn-foreign"),
        "contradictory_thread": lambda: params["turn"].__setitem__("threadId", "thread-foreign"),
        "missing_start_turn_id": lambda: client.__dict__.__setitem__("missing_turn_id", True),
    }
    mutations[scope]()
    client.queue_note("turn/completed", **params)
    agent = SimpleNamespace(provider="openai-codex", model="synthetic-model", api_key=first.access_token,
                            base_url="https://synthetic.invalid",
                            _credential_pool=pool, _credential_pool_entry_id=first.id, _codex_session=session)

    turn = session.run_turn("same input", turn_timeout=0.05)
    retried: list[bool] = []
    result = recover_codex_app_server_quota(agent, turn, retry_turn=lambda: retried.append(True) or TurnResult())

    assert retried == [] and result is turn
    assert not is_replay_safe_pretool_quota_failure(turn)
    assert all(entry.last_status != STATUS_EXHAUSTED for entry in pool.entries())


@pytest.mark.parametrize("barrier", ["changed_content", "extra_same_content_id", "unpaired_id"])
def test_user_message_echo_only_allows_one_exact_started_completed_pair(barrier):
    first = _entry(f"echo-A-{barrier}", "acct-A", "subject-A")
    second = _entry(f"echo-B-{barrier}", "acct-B", "subject-B", priority=1)
    pool = CredentialPool("openai-codex", [first, second])
    session, client, _binding = _session(pool, first)
    client.queue_note("turn/started", threadId="thread-stable", turn={"id": "turn-1"})
    submitted = [{"type": "text", "text": "submitted"}]
    echo = {"type": "userMessage", "id": "echo-1", "content": submitted}
    client.queue_note("item/started", threadId="thread-stable", item=echo)
    if barrier == "changed_content":
        changed = {"type": "userMessage", "id": "echo-1", "content": [{"type": "text", "text": "steered"}]}
        client.queue_note("item/completed", threadId="thread-stable", item=changed)
    else:
        client.queue_note("item/completed", threadId="thread-stable", item=echo)
        if barrier == "extra_same_content_id":
            extra = {"type": "userMessage", "id": "echo-2", "content": submitted}
            client.queue_note("item/completed", threadId="thread-stable", item=extra)
        else:
            unmatched = {"type": "userMessage", "id": "other-id", "content": submitted}
            client.queue_note("item/started", threadId="thread-stable", item=unmatched)
    _failed_quota_note(client)
    agent = SimpleNamespace(provider="openai-codex", model="synthetic-model", api_key=first.access_token,
                            base_url="https://synthetic.invalid",
                            _credential_pool=pool, _credential_pool_entry_id=first.id, _codex_session=session)

    turn = session.run_turn("submitted", turn_timeout=1)
    retried: list[bool] = []
    recover_codex_app_server_quota(agent, turn, retry_turn=lambda: retried.append(True) or TurnResult())

    assert retried == []
    assert turn.saw_item and not is_replay_safe_pretool_quota_failure(turn)


def test_native_text_and_image_user_echo_pair_is_safe_only_for_exact_submitted_input():
    first = _entry("echo-image-A", "acct-A", "subject-A")
    second = _entry("echo-image-B", "acct-B", "subject-B", priority=1)
    pool = CredentialPool("openai-codex", [first, second])
    session, client, _binding = _session(pool, first)
    native_content = [
        {"type": "text", "text": "inspect"},
        {"type": "image", "url": "https://synthetic.invalid/image.png"},
    ]
    client.queue_note("item/started", threadId="thread-stable",
                      item={"type": "userMessage", "id": "echo-image", "content": native_content})
    client.queue_note("item/completed", threadId="thread-stable",
                      item={"type": "userMessage", "id": "echo-image", "content": native_content})
    _failed_quota_note(client)
    agent = SimpleNamespace(provider="openai-codex", model="synthetic-model", api_key=first.access_token,
                            base_url="https://synthetic.invalid",
                            _credential_pool=pool, _credential_pool_entry_id=first.id, _codex_session=session)

    turn = session.run_turn(
        [{"type": "text", "text": "inspect"}, {"type": "image", "url": "https://synthetic.invalid/image.png"}],
        turn_timeout=1,
    )
    retried: list[bool] = []
    recover_codex_app_server_quota(agent, turn, retry_turn=lambda: retried.append(True) or TurnResult())

    assert retried == [True]


@pytest.mark.parametrize("failure", ["post-output", "post-tool", "unclassified", "retry-exhausted"])
def test_outer_conversation_loop_does_not_fallback_for_selected_pool_host_auth_result(tmp_path, monkeypatch, failure):
    _synthetic_home(tmp_path, monkeypatch)
    import sys
    from types import ModuleType

    # conversation_loop imports this process bootstrap hook; keep import-time update recovery away from the live checkout.
    bootstrap_stub = ModuleType("hermes_bootstrap")
    bootstrap_stub.__dict__["_happy_eyeballs_create_connection"] = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "hermes_bootstrap", bootstrap_stub)
    from agent.conversation_loop import _codex_app_server_turn

    first = _entry("fallback-A", "acct-A", "subject-A")
    second = _entry("fallback-B", "acct-B", "subject-B", 1)
    pool = CredentialPool("openai-codex", [first, second] if failure == "retry-exhausted" else [first])
    owner = SimpleNamespace(provider="openai-codex", model="synthetic-model", api_key=first.access_token,
                            base_url="https://chatgpt.com/backend-api/codex", _credential_pool=pool,
                            _credential_pool_entry_id=first.id)
    binding = CodexAppServerHostAuth.from_agent(owner)
    client = FakeClient()
    session = CodexAppServerSession(cwd="/synthetic/workspace", host_auth=binding,
                                    client_factory=cast(Any, lambda **_: client))
    if failure == "post-output":
        client.queue_note("item/agentMessage/delta", threadId="thread-stable", turnId="turn-1",
                          delta="already produced output")
    elif failure == "post-tool":
        client.queue_note("item/started", threadId="thread-stable", turnId="turn-1",
                          item={"id": "tool-guard", "type": "commandExecution"})
    if failure == "unclassified":
        client.queue_note("turn/completed", threadId="thread-stable",
                          turn={"id": "turn-1", "status": "failed", "error": {"message": "HTTP 429 quota"}})
    else:
        _failed_quota_note(client)
        if failure == "retry-exhausted":
            _failed_quota_note(client, turn_id="turn-2")
    activated: list[dict[str, Any]] = []
    buffered: list[str] = []
    owner._codex_session = session
    owner._codex_session_prompt = None
    owner._codex_session_model_provider = None
    owner.request_overrides = {}
    owner.session_cwd = "/synthetic/workspace"
    owner._session_db = None
    owner.session_id = None
    owner.session_api_calls = 0
    owner.context_compressor = None
    owner._iters_since_skill = 0
    owner._skill_nudge_interval = 0
    owner.valid_tool_names = set()
    owner._interrupt_requested = False
    owner._interrupt_message = None
    owner._has_pending_fallback = lambda: True
    owner._try_activate_fallback = lambda **kwargs: activated.append(kwargs) or True
    owner._buffer_diagnostic_status = lambda text: buffered.append(text)
    owner._run_codex_app_server_turn = lambda **kwargs: run_codex_app_server_turn(owner, **kwargs)
    state = SimpleNamespace(
        user_message="same input", original_user_message="same input",
        messages=[{"role": "user", "content": "same input"}], effective_task_id="synthetic",
        _should_review_memory=False, api_call_count=0, active_system_prompt="stable prompt",
    )

    result = _codex_app_server_turn(owner, state)

    assert result is not None and result.get("error")
    assert result.get("automatic_replay_prohibited") is True
    expected_calls = 2 if failure == "retry-exhausted" else 1
    assert result["api_calls"] == expected_calls and owner.session_api_calls == expected_calls
    assert len([method for method, _params in client.requests if method == "turn/start"]) == expected_calls
    assert activated == [] and buffered == []
