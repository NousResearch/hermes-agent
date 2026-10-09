"""v5 signals: hermes.tool_unavailable.count, hermes.provider_setup.count, hermes.feature_adoption.count."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from hermes_cli.observability import relay_shared_metrics
from hermes_cli.observability import shared_metrics_contract as contract
from hermes_cli.observability import shared_metrics_setup as setup_metrics
from hermes_cli.observability import shared_metrics_signals as signals
from hermes_cli.observability.shared_metrics import SharedMetricsStore


@pytest.fixture
def marks(tmp_path, monkeypatch):
    captured: list[tuple[str, dict]] = []
    policy = {"on": True}
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(relay_shared_metrics, "enabled", lambda: policy["on"])
    monkeypatch.setattr(relay_shared_metrics, "record_process_mark", lambda mark, data: captured.append((mark, data)))

    def saved(rows):
        captured.extend(rows)
        return len(rows)

    monkeypatch.setattr(relay_shared_metrics, "record_process_marks_saved", saved, raising=False)
    yield SimpleNamespace(rows=captured, policy=policy, home=tmp_path / "home")


def _setup_rows(rows):
    assert all(contract.counter_dimensions_are_valid(contract.PROVIDER_SETUP_METRIC, d) for _, d in rows), rows
    return [(d["surface"], d["provider"], d["event"], d["failure_class"]) for m, d in rows
            if m == contract.PROVIDER_SETUP_MARK]


def _dead_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


# ---- tool unavailable ----

def test_only_a_disabled_shipped_builtin_is_reported(monkeypatch):
    monkeypatch.setattr("tools.skill_provenance.is_background_review", lambda: False)
    route = {"model": "m", "provider": "openrouter"}
    agent = SimpleNamespace(_delegate_depth=0)
    fields = signals.tool_unavailable_fields(agent, "browser_navigate", "unknown_tool", route)
    assert fields == {"model": "m", "provider": "openrouter", "tool_name": "browser_navigate"}
    assert contract.counter_dimensions_are_valid(contract.TOOL_UNAVAILABLE_METRIC, fields)
    # Not shipped (plugin/MCP/hallucinated): stays v4 unknown_tool only.
    assert signals.tool_unavailable_fields(agent, "mcp_github_search", "unknown_tool", route) is None
    assert signals.tool_unavailable_fields(agent, "browser_navigate", "invalid_json", route) is None
    assert signals.tool_unavailable_fields(SimpleNamespace(_delegate_depth=1), "memory", "unknown_tool", route) is None
    monkeypatch.setattr("tools.skill_provenance.is_background_review", lambda: True)
    assert signals.tool_unavailable_fields(agent, "memory", "unknown_tool", route) is None


def test_deferred_builtins_and_cron_narrowing_are_not_unavailable(monkeypatch):
    monkeypatch.setattr("tools.skill_provenance.is_background_review", lambda: False)
    route = {"model": "m", "provider": "openrouter"}
    agent = SimpleNamespace(_delegate_depth=0, enabled_toolsets=None, disabled_toolsets=None)
    # Enabled but behind tool_search (reachable via tool_call): not "disabled in this session".
    assert signals.tool_unavailable_fields(agent, "session_search", "unknown_tool", route) is None
    # The same tool in a session that turned its toolset off is.
    narrowed = SimpleNamespace(_delegate_depth=0, enabled_toolsets=None, disabled_toolsets=["session_search"])
    assert signals.tool_unavailable_fields(narrowed, "session_search", "unknown_tool", route) is not None
    # Cron strips clarify on purpose.
    cron = SimpleNamespace(_delegate_depth=0, platform="cron")
    assert signals.tool_unavailable_fields(cron, "clarify", "unknown_tool", route) is None


def test_turn_validation_reports_disabled_builtin_once_and_unknown_names_never(marks, monkeypatch):
    from hermes_cli.observability.shared_metrics_model import record_tool_call_quality

    monkeypatch.setattr("tools.skill_provenance.is_background_review", lambda: False)
    agent = SimpleNamespace(provider="openrouter", model="anthropic/claude-sonnet-4", valid_tool_names={"memory"},
                            tools=[], _delegate_depth=0)
    calls = [SimpleNamespace(function=SimpleNamespace(name=n, arguments="{}"))
             for n in ("browser_navigate", "memory", "my_plugin_tool")]
    record_tool_call_quality(agent, calls, set())
    unavailable = [d for m, d in marks.rows if m == contract.TOOL_UNAVAILABLE_MARK]
    assert unavailable == [{"model": unavailable[0]["model"], "provider": "openrouter", "tool_name": "browser_navigate"}]
    assert contract.counter_dimensions_are_valid(contract.TOOL_UNAVAILABLE_METRIC, unavailable[0])


# ---- feature adoption ----

def test_counter_rows_map_to_features():
    assert signals.features_for(contract.MEMORY_OP_METRIC, {"origin": "foreground", "outcome": "success"}) == ("memory",)
    assert signals.features_for(contract.MEMORY_OP_METRIC, {"origin": "background_review", "outcome": "success"}) == ()
    assert signals.features_for(contract.TOOL_USAGE_METRIC, {"tool_name": "mcp", "outcome": "success"}) == ("mcp",)
    assert signals.features_for(contract.TOOL_USAGE_METRIC, {"tool_name": "browser_click", "outcome": "failed"}) == ()
    assert signals.features_for(contract.TOOL_USAGE_METRIC, {"tool_name": "kanban_create", "outcome": "success"}) == ("kanban",)
    assert set(signals.features_for(contract.TASK_STARTED_METRIC, {"platform": "telegram", "execution_surface": "desktop"})) \
        == {"gateway_platform", "desktop"}
    assert signals.features_for(contract.FEATURE_USED_MARK, {"feature": "bot_mode"}) == ("bot_mode",)


def test_hermes_internal_work_never_latches_adoption():
    from hermes_cli.observability.shared_metrics_fields import milestones_for

    curator = {"archived_bucket": "0", "created_bucket": "0", "merged_bucket": "0", "patched_bucket": "0",
               "outcome": "success"}
    assert signals.features_for(contract.CURATOR_RUN_METRIC, {**curator, "trigger": "scheduled"}) == ()
    assert signals.features_for(contract.CURATOR_RUN_METRIC, {**curator, "trigger": "manual"}) == ("curator",)
    by_review = {"action": "created", "provenance": "agent_created"}  # the background-review fork
    assert signals.features_for(contract.SKILL_LIFECYCLE_METRIC, by_review) == ()
    assert milestones_for(contract.SKILL_LIFECYCLE_METRIC, by_review) == ()
    by_user = {"action": "created", "provenance": "local"}
    assert signals.features_for(contract.SKILL_LIFECYCLE_METRIC, by_user) == ("skills_created",)
    assert milestones_for(contract.SKILL_LIFECYCLE_METRIC, by_user) == ("first_skill_created",)


def test_unreadable_first_session_reads_unknown(monkeypatch, tmp_path):
    def text_stamp(home):
        return float("yesterday")

    monkeypatch.setattr("hermes_cli.observability.shared_metrics_snapshot._first_session_started_at", text_stamp)
    assert signals.days_since_install_bucket(tmp_path) == "unknown"


@pytest.mark.parametrize(("age_s", "bucket"), [
    (60, "same_day"), (2 * 86_400, "1d_to_7d"), (10 * 86_400, "7d_to_30d"), (40 * 86_400, "30d_to_90d"),
    (200 * 86_400, "gte_90d"),
])
def test_days_since_install_bucket(monkeypatch, tmp_path, age_s, bucket):
    monkeypatch.setattr("hermes_cli.observability.shared_metrics_snapshot._first_session_started_at",
                        lambda home: time.time() - age_s)
    assert signals.days_since_install_bucket(tmp_path) == bucket


def test_feature_adoption_latches_once_per_install(tmp_path):
    resource = {"hermes_version": "1.0.0", "os_family": "linux", "architecture": "x86_64", "install_method": "git"}
    store = SharedMetricsStore(tmp_path / "m.sqlite3", tmp_path / "outbox")
    assert store.record_feature_adoption("memory", "same_day", resource) is True
    again = SharedMetricsStore(tmp_path / "m.sqlite3", tmp_path / "outbox")
    assert again.record_feature_adoption("memory", "1d_to_7d", resource) is False
    assert again.recorded_features() == frozenset({"memory"})
    with pytest.raises(ValueError):
        store.record_feature_adoption("not_a_feature", "same_day", resource)


def test_feature_used_mark_projection_and_disabled_gate(marks):
    event = SimpleNamespace(kind="mark", name=contract.FEATURE_USED_MARK, data={"feature": "projects"})
    signals.record_feature_used("projects")
    signals.record_feature_used("nope")
    assert marks.rows == [(contract.FEATURE_USED_MARK, {"feature": "projects"})]
    marks.policy["on"] = False
    signals.record_feature_used("projects")
    assert len(marks.rows) == 1
    projected = signals.feature_used_counter(event)
    assert projected in (None, (contract.FEATURE_USED_MARK, {"feature": "projects"}))


# ---- provider setup ----

def test_started_then_completed_once_and_marker_cleared(marks):
    flow = setup_metrics.begin_provider_setup("cli_model", "openrouter")
    assert flow is not None and flow.marker.exists()
    setup_metrics.finish_provider_setup(flow, "completed")
    setup_metrics.finish_provider_setup(flow, "failed", "auth")  # already ended
    assert _setup_rows(marks.rows) == [
        ("cli_model", "openrouter", "started", "none"), ("cli_model", "openrouter", "completed", "none")]
    assert not list(setup_metrics.markers_dir(marks.home).iterdir())


def test_custom_endpoint_reads_custom_and_no_raw_values(marks):
    setup_metrics.finish_provider_setup(setup_metrics.begin_provider_setup("desktop", "custom:my-box"), "completed")
    assert {row[1] for row in _setup_rows(marks.rows)} == {"custom"}


def test_disabled_collection_writes_nothing(marks):
    marks.policy["on"] = False
    assert setup_metrics.begin_provider_setup("cli_setup", "openrouter") is None
    assert marks.rows == [] and not setup_metrics.markers_dir(marks.home).exists()


def test_dead_or_stale_marker_is_abandoned_at_the_next_start_and_live_waits(marks):
    from gateway.status import get_process_start_time

    directory = setup_metrics.markers_dir(marks.home)
    directory.mkdir(parents=True)
    now = time.time()
    (directory / "cli_setup-1-1.json").write_text(json.dumps(
        {"pid": _dead_pid(), "start_time": None, "started_at": now, "surface": "cli_setup", "provider": "anthropic"}))
    parent = os.getppid()
    live = directory / f"tui-{parent}-2.json"
    live.write_text(json.dumps({"pid": parent, "start_time": get_process_start_time(parent), "started_at": now,
                                "surface": "tui", "provider": "nous"}))
    (directory / f"dashboard-{parent}-3.json").write_text(json.dumps(
        {"pid": parent, "start_time": get_process_start_time(parent), "started_at": now - 2 * setup_metrics.STALE_AFTER_S,
         "surface": "dashboard", "provider": "xai"}))

    flow = setup_metrics.begin_provider_setup("cli_setup", "openrouter")
    setup_metrics.report_abandoned_setups(marks.home)
    rows = _setup_rows(marks.rows)
    assert sorted(r for r in rows if r[2] == "abandoned") == [
        ("cli_setup", "anthropic", "abandoned", "none"), ("dashboard", "xai", "abandoned", "none")]
    assert live.exists() and flow.marker.exists()


def test_cli_flow_classifies_landed_backed_out_failed_and_raised(marks, monkeypatch):
    route = {"v": ("openrouter", "a", None)}
    monkeypatch.setattr(setup_metrics, "_model_route", lambda: route["v"])

    with setup_metrics.cli_provider_setup("openrouter"):  # no entry-point surface: untracked
        pass
    assert marks.rows == []

    with setup_metrics.provider_setup_surface("cli_setup"):
        with setup_metrics.cli_provider_setup("anthropic"):
            setup_metrics.note_provider_setup_saved()
        with setup_metrics.cli_provider_setup("anthropic"):
            pass  # returned without saving: backed out
        with setup_metrics.cli_provider_setup("nous"):
            setup_metrics.note_provider_setup_failure("no_models")
        with setup_metrics.cli_provider_setup("gemini"):
            route["v"] = ("gemini", "g", None)  # route changed without the save helper
        with pytest.raises(KeyboardInterrupt):
            with setup_metrics.cli_provider_setup("xai"):
                raise KeyboardInterrupt
        with pytest.raises(ConnectionError):
            with setup_metrics.cli_provider_setup("xai"):
                raise ConnectionError("down")
        with setup_metrics.cli_provider_setup("remove-custom"):
            pass
    ends = [r for r in _setup_rows(marks.rows) if r[2] != "started"]
    assert ends == [
        ("cli_setup", "anthropic", "completed", "none"), ("cli_setup", "anthropic", "abandoned", "none"),
        ("cli_setup", "nous", "failed", "no_models"), ("cli_setup", "gemini", "completed", "none"),
        ("cli_setup", "xai", "abandoned", "none"), ("cli_setup", "xai", "failed", "network"),
    ]
    setup_metrics.note_provider_setup_saved()  # outside a flow: inert


def test_cli_setup_navigation_esc_cancels_and_back_resumes_one_flow(marks, monkeypatch):
    from hermes_cli.setup import _SetupCancelled, _SetupGoBack

    monkeypatch.setattr(setup_metrics, "_model_route", lambda: ("openrouter", "a", None))

    def attempt(provider, exc=None):
        with setup_metrics.cli_provider_setup(provider):
            if exc is not None:
                raise exc
            setup_metrics.note_provider_setup_saved()

    with setup_metrics.provider_setup_surface("cli_model"):
        with pytest.raises(_SetupCancelled):
            attempt("xai", _SetupCancelled())  # Esc
        with pytest.raises(_SetupGoBack):
            attempt("anthropic", _SetupGoBack(1))  # Back to the provider menu, then the same provider
        attempt("anthropic")
        with pytest.raises(_SetupGoBack):
            attempt("nous", _SetupGoBack(1))  # Back, then another provider
        attempt("gemini")
    with pytest.raises(_SetupGoBack):  # a wizard section replay keeps the flow open across the surface
        with setup_metrics.provider_setup_surface("cli_setup"):
            attempt("xai", _SetupGoBack(0))
    with setup_metrics.provider_setup_surface("cli_setup"):
        pass  # ...and leaving the entry point without resuming it ends it
    assert _setup_rows(marks.rows) == [
        ("cli_model", "xai", "started", "none"), ("cli_model", "xai", "abandoned", "none"),
        ("cli_model", "anthropic", "started", "none"), ("cli_model", "anthropic", "completed", "none"),
        ("cli_model", "nous", "started", "none"), ("cli_model", "nous", "abandoned", "none"),
        ("cli_model", "gemini", "started", "none"), ("cli_model", "gemini", "completed", "none"),
        ("cli_setup", "xai", "started", "none"), ("cli_setup", "xai", "abandoned", "none"),
    ]


@pytest.mark.parametrize(("sess", "ending"), [
    ({"status": "approved"}, ("completed", "none")),
    ({"status": "expired"}, ("abandoned", "none")),
    ({"status": "error", "reason": "timeout"}, ("abandoned", "none")),
    ({"status": "denied"}, ("failed", "auth")),
    ({"status": "denied", "reason": "user_declined"}, ("abandoned", "none")),
    ({"status": "pending", "cancelled": True}, ("abandoned", "none")),
    ({"status": "error", "reason": "anon_unreachable"}, ("failed", "network")),
])
def test_oauth_session_endings(marks, monkeypatch, sess, ending):
    monkeypatch.setattr(setup_metrics, "web_setup_surface", lambda: "desktop")
    flow = setup_metrics.begin_oauth_setup("nous", None)
    setup_metrics.attach_oauth_setup(sess, flow)
    setup_metrics.settle_oauth_setup(sess)
    setup_metrics.settle_oauth_setup(sess)
    assert [r[2:] for r in _setup_rows(marks.rows)] == [("started", "none"), ending]


def test_cancel_or_lapsed_code_is_never_a_failure_and_poller_errors_keep_their_class(marks, monkeypatch):
    """A user cancel / a sign-in code left to run out is ``abandoned`` on every surface; a dashboard
    poller's exception keeps its closed class instead of the bare session ``error`` -> ``other``."""
    import httpx

    from hermes_cli.auth_constants import _codex_err, _xai_err
    from hermes_cli.auth_device_flow import _poll_for_token
    from hermes_cli.auth_error_copy import device_flow_error
    from hermes_cli.web_server_oauth import _oauth_poller, _oauth_sessions

    monkeypatch.setattr(setup_metrics, "web_setup_surface", lambda: "desktop")
    with setup_metrics.provider_setup_surface("cli_model"):  # CLI: Ctrl-C mid sign-in
        with pytest.raises(KeyboardInterrupt), setup_metrics.cli_provider_setup("nous"):
            raise KeyboardInterrupt
    flow = setup_metrics.begin_oauth_setup("openai-codex", None)  # web start route: cancelled pre-response
    setup_metrics.finish_provider_setup(flow, "failed", setup_metrics.setup_failure_class(KeyboardInterrupt()))

    class Pending:  # the real Nous poll loop running out of time on authorization_pending
        status_code = 400

        def json(self):
            return {"error": "authorization_pending"}

    with pytest.raises(TimeoutError) as lapsed:
        _poll_for_token(SimpleNamespace(post=lambda *a, **k: Pending()), "https://p", "c", "d", 0, 1)
    dropped = _xai_err("xAI OIDC discovery failed", "xai_discovery_failed")
    dropped.__cause__ = httpx.ConnectError("down")
    for provider, exc in [
        ("nous", lapsed.value), ("xai-oauth", _xai_err("Timed out", "device_code_timeout")),
        ("openai-codex", _codex_err("Login timed out after 15 minutes.", "device_code_timeout")),
        ("nous", device_flow_error("access_denied", "declined")), ("xai-oauth", dropped),
        ("nous", device_flow_error("invalid_client", "nope")),
    ]:
        sess = {"status": "pending", "profile": None}
        _oauth_sessions["probe"] = sess
        setup_metrics.attach_oauth_setup(sess, setup_metrics.begin_oauth_setup(provider, None))

        @_oauth_poller(provider)
        def poller(_sid, _sess, exc=exc):
            raise exc

        poller("probe")
        setup_metrics.settle_oauth_setup(sess)
    _oauth_sessions.pop("probe", None)
    assert [r for r in _setup_rows(marks.rows) if r[2] != "started"] == [
        ("cli_model", "nous", "abandoned", "none"), ("desktop", "openai-codex", "abandoned", "none"),
        ("desktop", "nous", "abandoned", "none"), ("desktop", "xai-oauth", "abandoned", "none"),
        ("desktop", "openai-codex", "abandoned", "none"), ("desktop", "nous", "abandoned", "none"),
        ("desktop", "xai-oauth", "failed", "network"), ("desktop", "nous", "failed", "auth"),
    ]


def _wire(status, body):
    """A real httpx client whose every POST gets one canned response (JSON dict, else raw text)."""
    import httpx

    def reply(request):
        if isinstance(body, Exception):
            raise body
        return httpx.Response(status, request=request, **({"json": body} if isinstance(body, dict) else {"text": body}))

    return httpx.Client(transport=httpx.MockTransport(reply))


def _poll(provider, status, body):
    """The real device-code poll loop on a fake clock (each sleep advances it)."""
    def run(monkeypatch):
        from hermes_cli import auth_device_flow
        from hermes_cli.auth_xai import _xai_oauth_poll_device_token

        clock = [0.0]
        fake_time = SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda s: clock.__setitem__(0, clock[0] + max(s, 1)))
        monkeypatch.setattr(auth_device_flow, "time", fake_time)
        if provider == "nous":
            return auth_device_flow._poll_for_token(_wire(status, body), "https://p", "c", "d", 30, 1)
        return _xai_oauth_poll_device_token(
            _wire(status, body), token_endpoint="https://x/token", device_code="d", expires_in=30, poll_interval=1)
    return run


def _guest_sign_in(token_status, token_body):
    """The Desktop free-tier sign-in (guest promotion): the real ``run_sign_in`` generator recorded
    onto a dashboard session; the promotion completes, the token poll gets ``token_*`` until expiry."""
    def run(monkeypatch):
        import httpx

        from hermes_cli import anon_auth, auth_device_flow
        from hermes_cli.web_server_oauth import _record_sign_in_state

        clock = [0.0]
        fake_time = SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda s: clock.__setitem__(0, clock[0] + max(s, 1)))
        monkeypatch.setattr(auth_device_flow, "time", fake_time)
        monkeypatch.setattr(anon_auth, "current_nous_state", lambda: {
            "auth_method": anon_auth.ANON_AUTH_METHOD, "anon_token": "t", "portal_base_url": "https://p"})
        monkeypatch.setattr(anon_auth, "guest_enabled", lambda: True)
        monkeypatch.setattr(anon_auth, "_anon_headers", lambda: {})
        replies = {
            "/api/oauth/device/code": (200, {"device_code": "d", "user_code": "U", "verification_uri": "https://p/v",
                                             "verification_uri_complete": "https://p/v?c=U", "expires_in": 30, "interval": 1}),
            "/api/anonymous/promotion-intent": (200, {"claim_code": "C", "claim_url": "/claim", "interval": 1}),
            "/api/anonymous/promotion-status": (200, {"status": "completed"}),
            "/api/oauth/token": (token_status, token_body)}

        def reply(request):
            status, body = replies[request.url.path]
            return httpx.Response(status, **({"json": body} if isinstance(body, dict) else {"text": body}))
        sess = {"status": "pending"}
        for state in anon_auth.run_sign_in(client_factory=lambda *_: httpx.Client(transport=httpx.MockTransport(reply))):
            _record_sign_in_state(sess, state)
        return sess
    return run


def _codex_start(status, body):
    """The dashboard start route: the worker thread's failure crosses into an HTTPException."""
    def run(monkeypatch):
        import asyncio

        import hermes_cli.web_routers.oauth as routes

        client = _wire(status, body)
        monkeypatch.setattr(routes, "_codex_post", lambda _httpx, url, **kw: client.post(url, **kw))
        asyncio.run(routes._start_codex_device_code(None))
    return run


@pytest.mark.parametrize("producer, ending", [
    (_poll("nous", 400, {"error": "authorization_pending"}), ("abandoned", "none")),  # code left unapproved
    (_poll("nous", 503, "<html>503 unavailable</html>"), ("failed", "network")),  # outage until it ran out
    (_poll("xai", 400, {"error": "access_denied"}), ("abandoned", "none")),  # consent declined
    (_poll("xai", 400, {"error": "expired_token"}), ("abandoned", "none")),  # the server let the code lapse
    (_poll("xai", 400, {"error": "invalid_client"}), ("failed", "auth")),
    (_codex_start(401, {"error": "invalid_client"}), ("failed", "auth")),
    (_codex_start(0, ConnectionRefusedError("down")), ("failed", "network")),
    (_codex_start(503, "<html>503</html>"), ("failed", "network")),
    (_guest_sign_in(400, {"error": "authorization_pending"}), ("abandoned", "none")),
    (_guest_sign_in(503, "<html>503 unavailable</html>"), ("failed", "network")),
], ids=["nous-pending", "nous-503", "xai-denied", "xai-expired", "xai-refused", "codex-401", "codex-down",
        "codex-503", "guest-pending", "guest-503"])
def test_wire_failures_keep_the_producers_class_to_the_recorded_end(marks, monkeypatch, producer, ending):
    """Real poll loops / start route / free-tier sign-in against wire responses: a lapse or a decline
    is a walk-away, an outage or a refusal stays a failure, across the worker -> HTTPException and
    sign-in state -> session boundaries too."""
    import asyncio

    import hermes_cli.web_routers.oauth as routes

    monkeypatch.setattr(setup_metrics, "web_setup_surface", lambda: "desktop")
    flow = setup_metrics.begin_oauth_setup("nous", None)
    try:
        sess = producer(monkeypatch)
    except Exception as exc:
        asyncio.run(routes._end_oauth_setup_metric(flow, exc))
    else:  # a sign-in generator ends on a state, recorded onto its session
        setup_metrics.attach_oauth_setup(sess, flow)
        setup_metrics.settle_oauth_setup(sess)
    assert [r[2:] for r in _setup_rows(marks.rows)] == [("started", "none"), ending]

def test_pending_oauth_session_does_not_settle(marks, monkeypatch):
    monkeypatch.setattr(setup_metrics, "web_setup_surface", lambda: "dashboard")
    sess = {"status": "pending"}
    setup_metrics.attach_oauth_setup(sess, setup_metrics.begin_oauth_setup("xai", None))
    setup_metrics.settle_oauth_setup(sess)
    assert [r[2] for r in _setup_rows(marks.rows)] == ["started"]


def test_api_key_env_maps_to_provider_only():
    assert setup_metrics.provider_for_api_key_env("OPENROUTER_API_KEY") == "openrouter"
    assert setup_metrics.provider_for_api_key_env("GITHUB_TOKEN_FOR_TOOLS_XYZ") is None
    # Ecosystem tokens and keys a tool panel also asks for (Gemini TTS) are not a provider connection.
    for shared in ("GITHUB_TOKEN", "GH_TOKEN", "HF_TOKEN", "GEMINI_API_KEY"):
        assert setup_metrics.provider_for_api_key_env(shared) is None


def test_web_forms_count_only_a_new_provider_key_or_endpoint(marks, monkeypatch):
    from hermes_cli.web_models import CustomEndpointUpdate
    from hermes_cli.web_routers import config_env

    done: list[str] = []
    monkeypatch.setattr(setup_metrics, "record_provider_setup_done", lambda _s, provider, **_k: done.append(provider))
    monkeypatch.setattr(setup_metrics, "web_setup_surface", lambda: "dashboard")
    marks.home.mkdir(parents=True, exist_ok=True)
    for key, value in (("OPENROUTER_API_KEY", "sk-or-a"), ("OPENROUTER_API_KEY", "sk-or-a"),  # re-save
                       ("OPENROUTER_API_KEY", ""), ("OPENROUTER_API_KEY", "sk-or-b"),  # clear, then a new key
                       ("GITHUB_TOKEN", "ghp_" + "b" * 36)):
        config_env._save_env_credential(key, value)
    config_env._save_env_credential("GEMINI_API_KEY", "gm-a")  # Keys page: could be the TTS tool's key
    config_env._save_env_credential("GEMINI_API_KEY", "gm-b", provider_setup=True)  # a provider-connection form
    body = {"name": "Acme LLM", "base_url": "http://10.0.0.5:8080/v1", "model": "acme-70b"}
    endpoint = config_env.upsert_custom_endpoint(CustomEndpointUpdate(**body))["id"]
    config_env.upsert_custom_endpoint(CustomEndpointUpdate(id=endpoint, **{**body, "model": "acme-70b-v2"}))
    assert done == ["openrouter", "openrouter", "gemini", "custom"]


# ---- feature disabled ----

from hermes_cli.observability import shared_metrics_disabled as disabled_metrics  # noqa: E402


def _settle_disabled() -> None:
    """feature_disabled diffs and records off the writer's thread (outside any caller lock)."""
    import threading

    for thread in threading.enumerate():
        if thread.name == "hermes-feature-disabled":
            thread.join(10)


def test_settings_skills_plugins_transitions_only_count_moves_from_and_back_to_default():
    old = {"skills": {"disabled": ["my-private-skill"]}, "plugins": {"disabled": []}}
    new = {
        "memory": {"memory_enabled": False}, "compression": {"enabled": "false"}, "curator": {"enabled": True},
        "display": {"show_reasoning": False, "skin": "mono"},  # a value change on a non-bool: never a row
        "skills": {"disabled": ["my-private-skill", "arxiv"], "platform_disabled": {"telegram": ["totally-custom"]}},
        "plugins": {"disabled": ["disk-cleanup", "telegram", "some-private-plugin"]},
    }
    got = set(disabled_metrics.config_transitions(old, new))
    assert got == {
        ("memory", "memory.memory_enabled", "disabled"), ("compression", "compression.enabled", "disabled"),
        ("setting", "display.show_reasoning", "disabled"),
        ("skill", "arxiv", "disabled"), ("skill", "custom", "disabled"),
        ("plugin", "disk-cleanup", "disabled"), ("platform", "telegram", "disabled"),
        ("plugin", "custom", "disabled"),
    }
    back = set(disabled_metrics.config_transitions(new, old))
    assert ("memory", "memory.memory_enabled", "re_enabled") in back and ("skill", "arxiv", "re_enabled") in back
    assert all(contract.counter_dimensions_are_valid(contract.FEATURE_DISABLED_METRIC, {
        "event": e, "kind": k, "name": n, "surface": "cli_config"}) for k, n, e in got | back)


def test_toolset_transitions_use_the_real_resolver():
    from hermes_cli.tools_config import _get_platform_tools

    default = sorted(_get_platform_tools({}, "cli", include_default_mcp_servers=False))
    assert "memory" in default
    new = {"platform_toolsets": {"cli": [t for t in default if t != "memory"]}}
    assert ("toolset", "memory", "disabled") in disabled_metrics.config_transitions({}, new)
    assert ("toolset", "memory", "re_enabled") in disabled_metrics.config_transitions(new, {})
    assert disabled_metrics.config_transitions({}, {"platform_toolsets": {"cli": default}}) == []


def test_record_config_saved_needs_a_surface_and_collection(marks, monkeypatch):
    monkeypatch.setattr(disabled_metrics, "_process_surface", None)
    disabled_metrics.record_config_saved({}, {"memory": {"memory_enabled": False}})
    assert marks.rows == []  # setup/migration writes: no entry-point surface
    disabled_metrics.set_process_surface("config")
    marks.policy["on"] = False
    disabled_metrics.record_config_saved({}, {"memory": {"memory_enabled": False}})
    assert marks.rows == []
    marks.policy["on"] = True
    disabled_metrics.record_config_saved({}, {"memory": {"memory_enabled": False}})
    _settle_disabled()
    assert marks.rows == [(contract.FEATURE_DISABLED_MARK, {
        "event": "disabled", "kind": "memory", "name": "memory.memory_enabled", "surface": "cli_config"})]
    disabled_metrics.set_process_surface(None)
    assert disabled_metrics.current_surface() == "cli_slash"


def test_store_counts_feature_disabled_once_per_day(tmp_path):
    resource = {"hermes_version": "1.0.0", "os_family": "linux", "architecture": "x86_64", "install_method": "git"}
    store = SharedMetricsStore(tmp_path / "m.sqlite3", tmp_path / "outbox")
    dims = {"event": "disabled", "kind": "toolset", "name": "memory", "surface": "cli_tools"}
    assert store.record_counter_once_per_day(contract.FEATURE_DISABLED_METRIC, dims, resource) is True
    assert store.record_counter_once_per_day(
        contract.FEATURE_DISABLED_METRIC, {**dims, "surface": "desktop"}, resource) is False
    assert store.record_counter_once_per_day(
        contract.FEATURE_DISABLED_METRIC, {**dims, "event": "re_enabled"}, resource) is True


def test_real_config_writes_report_the_move_away_from_default(marks, monkeypatch):
    from hermes_cli.config import load_config, save_config, set_config_value

    marks.home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(disabled_metrics, "_process_surface", "cli_config")
    config = load_config()
    config.setdefault("memory", {})["memory_enabled"] = False
    save_config(config)
    _settle_disabled()
    set_config_value("curator.enabled", "false")
    _settle_disabled()
    rows = [(d["kind"], d["name"], d["event"]) for m, d in marks.rows if m == contract.FEATURE_DISABLED_MARK]
    assert rows == [("memory", "memory.memory_enabled", "disabled"), ("curator", "curator.enabled", "disabled")]


def test_migrations_and_env_templates_are_not_user_disables(marks, monkeypatch):
    from hermes_cli.config import _persist_migration, load_config, save_config

    marks.home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(disabled_metrics, "_process_surface", "cli_config")
    config = load_config()
    config.setdefault("memory", {})["memory_enabled"] = False
    _persist_migration(config)  # e.g. `hermes config migrate` / profile create in the dashboard
    # A ``${VAR}`` template on the raw side vs its expanded value: nothing moved.
    monkeypatch.setenv("MEM_ON", "false")
    (marks.home / "config.yaml").write_text("compression:\n  enabled: ${MEM_ON}\n")
    config = load_config()
    config.setdefault("display", {})["compact"] = True
    save_config(config)
    _settle_disabled()
    assert [d for m, d in marks.rows if m == contract.FEATURE_DISABLED_MARK] == []


def test_diff_and_record_run_off_the_callers_lock_in_the_owning_profile(marks, monkeypatch, tmp_path):
    import threading

    from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override

    monkeypatch.setattr(disabled_metrics, "_process_surface", "cli_config")
    gate, seen = threading.Event(), []
    real = disabled_metrics.config_transitions

    def slow(old, new):
        gate.wait(5)
        seen.append(str(get_hermes_home()))
        return real(old, new)

    monkeypatch.setattr(disabled_metrics, "config_transitions", slow)
    token = set_hermes_home_override(str(tmp_path / "b"))
    try:
        with threading.Lock():  # the caller's write lock (the dashboard's _CONFIG_MUTATION_LOCK)
            disabled_metrics.record_config_saved({}, {"memory": {"memory_enabled": False}})
            assert marks.rows == []  # returned without diffing
    finally:
        reset_hermes_home_override(token)
    gate.set()
    _settle_disabled()
    assert seen == [str(tmp_path / "b")] and [m for m, _ in marks.rows] == [contract.FEATURE_DISABLED_MARK]


def test_off_thread_setup_records_in_the_owning_profile(marks, tmp_path, monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(relay_shared_metrics, "record_process_marks_saved", lambda rows: (
        seen.append(str(__import__("hermes_constants").get_hermes_home())), len(rows))[1])
    home_a, home_b = tmp_path / "a", tmp_path / "b"
    for home in (home_a, home_b, home_a):
        flow = setup_metrics.begin_provider_setup("dashboard", "xai", hermes_home=home)
        assert flow.marker.parent == setup_metrics.markers_dir(home)
        setup_metrics.finish_provider_setup(flow, "completed")
    assert seen == [str(h) for h in (home_a, home_a, home_b, home_b, home_a, home_a)]
