"""PLG-11 (#140058): a memory-provider tool call belongs to the calling turn's profile.

On a multiplexed gateway every served profile's turns run in one process, so anything a provider
module keeps at module or class level is shared by all of them. These tests drive the real
sequential and concurrent dispatch helpers with real ``MemoryManager`` instances.
"""
import json
import threading
from types import SimpleNamespace

import pytest

from agent.inline_tool_executors import resolve_invoke_tool_executor
from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider, ctx_bound
from agent.secret_scope import is_multiplex_active
from agent.tool_executor import _resolve_sequential_dispatch, _ToolCallRef
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

_TOOL = "pw_remember"


class _ProcessWideProvider(MemoryProvider):
    """The reported provider's shape: one store binding per PROCESS, set by whichever profile
    initialized last. A call that names its caller's home rebinds; one that doesn't writes wherever
    the binding points."""

    binding = None
    stores: dict = {}
    calls: list = []

    @property
    def name(self):
        return "processwide"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        type(self).binding = kwargs["hermes_home"]

    def get_tool_schemas(self):
        return [{"name": _TOOL, "description": "remember", "parameters": {"type": "object", "properties": {}}}]

    def handle_tool_call(self, tool_name, args, **kwargs):
        type(self).calls.append(kwargs)
        home = kwargs.get("hermes_home") or type(self).binding
        type(self).stores.setdefault(home, []).append(args["content"])
        return json.dumps({"ok": True})


class _BareSignatureProvider(_ProcessWideProvider):
    """A provider written against the two-argument ``handle_tool_call`` shape."""

    def handle_tool_call(self, tool_name, args):
        type(self).stores.setdefault("bare", []).append(args["content"])
        return json.dumps({"ok": True})


@pytest.fixture(autouse=True)
def _fresh_provider_state():
    _ProcessWideProvider.binding = None
    _ProcessWideProvider.stores = {}
    _ProcessWideProvider.calls = []
    yield


class _Scope:
    def __init__(self, home):
        self.home = home

    def __enter__(self):
        self.token = set_hermes_home_override(str(self.home))

    def __exit__(self, *exc):
        reset_hermes_home_override(self.token)


def _agent(home, gsk, provider_cls=_ProcessWideProvider, *, initialize=True):
    manager = MemoryManager()
    with _Scope(home):
        manager.add_provider(provider_cls())
        if initialize:
            manager.initialize_all(session_id=f"s-{home.name}", platform="api_server", hermes_home=str(home))
    return SimpleNamespace(
        _memory_manager=manager, _context_engine_tool_names=set(), _gateway_session_key=gsk,
        _should_emit_quiet_tool_messages=lambda: False, _should_start_quiet_spinner=lambda: False,
    )


def _sequential(agent, content):
    ref = _ToolCallRef(name=_TOOL, args={"content": content}, task_id="t", call_id="c", trace=[])
    return _resolve_sequential_dispatch(agent, ref, []).execute({"content": content})


def _concurrent(agent, content):
    executor = resolve_invoke_tool_executor(agent, _TOOL)
    assert executor is not None
    return executor(agent, {"content": content}, SimpleNamespace())


@pytest.fixture
def homes(tmp_path):
    root = tmp_path / ".hermes"
    alpha, beta = root / "profiles" / "alpha", root / "profiles" / "beta"
    for home in (alpha, beta):
        home.mkdir(parents=True)
    return alpha, beta


def test_interleaved_turns_of_two_profiles_write_to_their_own_stores(homes):
    alpha_home, beta_home = homes
    alpha = _agent(alpha_home, "agent:alpha:api_server:dm:1")
    beta = _agent(beta_home, "agent:beta:api_server:dm:2")  # initialized last: owns the process-wide binding
    barrier, errors = threading.Barrier(2), []

    def turn(agent, home, label):
        try:
            with _Scope(home):
                barrier.wait(timeout=5)
                for i in range(20):
                    result = (_sequential if i % 2 else _concurrent)(agent, f"{label}-{i}")
                    assert json.loads(result) == {"ok": True}, result
        except Exception as exc:  # surfaced below; a thread can't fail the test itself
            errors.append(exc)

    threads = [threading.Thread(target=turn, args=(alpha, alpha_home, "alpha")),
               threading.Thread(target=turn, args=(beta, beta_home, "beta"))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not errors, errors
    stores = _ProcessWideProvider.stores
    assert sorted(stores.get(str(alpha_home), [])) == sorted(f"alpha-{i}" for i in range(20)), stores
    assert sorted(stores.get(str(beta_home), [])) == sorted(f"beta-{i}" for i in range(20)), stores
    # The caller's gateway session key still travels with each call.
    assert {call["caller_gsk"] for call in _ProcessWideProvider.calls} == {
        "agent:alpha:api_server:dm:1", "agent:beta:api_server:dm:2"}


_REFUSAL_CODE = "memory_caller_profile_unresolved"


def _assert_generic_refusal(result, caplog, *homes):
    """The model/chat sees a generic refusal pointing at a log every host writes (errors.log; gateway.log
    too on a gateway); the log keeps the check's code and the homes."""
    error = result.get("error", "")
    assert "refused" in error and "hermes logs errors" in error, result
    assert _REFUSAL_CODE not in error and not any(str(home) in error for home in homes), result
    logged = "\n".join(r.getMessage() for r in caplog.records if r.name == "agent.memory_caller_scope")
    assert _REFUSAL_CODE in logged, logged
    assert all(str(home) in logged for home in homes), logged


@pytest.fixture
def multiplexed():
    from agent.secret_scope import reset_multiplex_context, set_multiplex_context
    token = set_multiplex_context(True)
    yield
    reset_multiplex_context(token)


@pytest.mark.parametrize("dispatch", [_sequential, _concurrent])
def test_call_from_another_profiles_scope_is_refused(homes, dispatch, caplog):
    alpha_home, beta_home = homes
    alpha = _agent(alpha_home, "agent:alpha:api_server:dm:1")
    with _Scope(beta_home):  # the turn's scope names a profile this manager never served
        result = json.loads(dispatch(alpha, "stray"))
    _assert_generic_refusal(result, caplog, alpha_home, beta_home)
    assert _ProcessWideProvider.stores == {}


def test_multiplexed_call_with_no_known_profile_is_refused(homes, multiplexed, caplog):
    alpha_home, _ = homes
    agent = _agent(alpha_home, "", initialize=False)
    with _Scope(alpha_home):
        result = json.loads(_sequential(agent, "orphan"))
    _assert_generic_refusal(result, caplog, alpha_home)
    assert _ProcessWideProvider.stores == {}


@pytest.mark.parametrize("dispatch", [_sequential, _concurrent])
def test_multiplexed_call_without_a_caller_scope_is_refused_even_for_the_process_profile(
        homes, dispatch, multiplexed, monkeypatch, caplog):
    # The manager serves the process's own (launch) profile, but on a multiplexed process a call that
    # names no profile is not proof it came from that profile's turn.
    alpha_home, _ = homes
    monkeypatch.setenv("HERMES_HOME", str(alpha_home))
    alpha = _agent(alpha_home, "agent:alpha:api_server:dm:1")
    token = set_hermes_home_override(None)
    try:
        result = json.loads(dispatch(alpha, "unscoped"))
    finally:
        reset_hermes_home_override(token)
    _assert_generic_refusal(result, caplog, alpha_home)
    assert _ProcessWideProvider.stores == {}


def test_single_profile_process_keeps_working_without_initialize(homes):
    alpha_home, _ = homes
    agent = _agent(alpha_home, "", initialize=False)
    with _Scope(alpha_home):
        assert json.loads(_sequential(agent, "plain")) == {"ok": True}


def test_two_argument_provider_signature_still_receives_its_calls(homes):
    alpha_home, _ = homes
    agent = _agent(alpha_home, "agent:alpha:api_server:dm:1", _BareSignatureProvider)
    with _Scope(alpha_home):
        assert json.loads(_sequential(agent, "one")) == {"ok": True}
        assert json.loads(_concurrent(agent, "two")) == {"ok": True}
    assert _ProcessWideProvider.stores == {"bare": ["one", "two"]}


# -- Provider-side contract and lifecycle hooks --------------------------------------------------------

_HOOK_TOOL = "ct_remember"


class _InstanceScopedProvider(MemoryProvider):
    """A provider that meets the multiplexing contract: its store binding lives on the instance, set
    from ``initialize``'s ``hermes_home``. Records which binding, and which ambient home, every call
    and hook ran under."""

    events: list = []

    def __init__(self):
        self._home = None

    @property
    def name(self):
        return "instancescoped"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self._home = kwargs["hermes_home"]

    def _record(self, what):
        from hermes_constants import get_hermes_home
        type(self).events.append((what, self._home, str(get_hermes_home())))

    def get_tool_schemas(self):
        return [{"name": _HOOK_TOOL, "description": "remember", "parameters": {"type": "object", "properties": {}}}]

    def handle_tool_call(self, tool_name, args, **kwargs):
        assert kwargs["hermes_home"] == self._home, (kwargs, self._home)
        self._record("tool")
        return json.dumps({"ok": True})

    def prefetch(self, query, *, session_id=""):
        self._record("prefetch")
        return ""

    def queue_prefetch(self, query, *, session_id=""):
        self._record("queue_prefetch")

    def sync_turn(self, user_content, assistant_content, *, session_id=""):
        self._record("sync_turn")

    def on_turn_start(self, turn_number, message, **kwargs):
        self._record("on_turn_start")

    def on_session_end(self, messages):
        self._record("on_session_end")

    def on_session_switch(self, new_session_id, **kwargs):
        self._record("on_session_switch")

    def on_pre_compress(self, messages):
        self._record("on_pre_compress")
        return ""

    def on_memory_write(self, action, target, content, metadata=None):
        self._record("on_memory_write")

    def on_delegation(self, task, result, *, child_session_id="", **kwargs):
        self._record("on_delegation")


@pytest.fixture(autouse=True)
def _fresh_events():
    _InstanceScopedProvider.events = []
    yield


def _manager(home, provider=None):
    manager = MemoryManager()
    with _Scope(home):
        manager.add_provider(provider or _InstanceScopedProvider())
        manager.initialize_all(session_id=f"s-{home.name}", platform="api_server", hermes_home=str(home))
    return manager


def _drain(manager):
    assert manager.flush_pending(timeout=5)


# hook name -> drives it through the manager the way the agent does (background ones are drained)
_HOOKS = {
    "sync_turn": lambda m: (m.sync_all("what did we decide", "the plan", session_id="s"), _drain(m)),
    "queue_prefetch": lambda m: (m.queue_prefetch_all("what did we decide", session_id="s"), _drain(m)),
    "prefetch": lambda m: m.prefetch_all("what did we decide", session_id="s"),
    "on_turn_start": lambda m: m.on_turn_start(1, "what did we decide"),
    "on_session_end": lambda m: m.on_session_end([]),
    "on_session_switch": lambda m: m.on_session_switch("s2"),
    "session_boundary": lambda m: (m.commit_session_boundary_async([], new_session_id="s3"), _drain(m)),
    "on_pre_compress": lambda m: m.on_pre_compress([{"role": "user", "content": "x"}]),
    "on_memory_write": lambda m: m.on_memory_write("add", "memory", "a fact"),
    "on_delegation": lambda m: m.on_delegation("task", "result"),
}


@pytest.mark.parametrize("hook", sorted(_HOOKS))
def test_lifecycle_hook_from_another_profiles_scope_is_refused(homes, hook, caplog):
    alpha_home, beta_home = homes
    alpha = _manager(alpha_home)
    with _Scope(beta_home):
        _HOOKS[hook](alpha)
    assert _InstanceScopedProvider.events == []
    logged = "\n".join(r.getMessage() for r in caplog.records if r.name == "agent.memory_caller_scope")
    assert _REFUSAL_CODE in logged and str(beta_home) in logged, logged


@pytest.mark.parametrize("hook", sorted(_HOOKS))
def test_multiplexed_lifecycle_hook_without_a_caller_scope_is_refused(homes, hook, multiplexed, monkeypatch):
    alpha_home, _ = homes
    monkeypatch.setenv("HERMES_HOME", str(alpha_home))
    alpha = _manager(alpha_home)
    _HOOKS[hook](alpha)
    assert _InstanceScopedProvider.events == []


def test_tool_calls_reuse_what_initialize_bound(homes, monkeypatch):
    # Per call, only the caller's home is resolved: the bound home's key and the provider's
    # handle_tool_call signature are worked out once per manager, not on every call.
    from agent import memory_caller_scope, memory_manager
    alpha_home, _ = homes
    alpha = _manager(alpha_home)
    keys, signatures = [], []
    real_key, real_signature = memory_caller_scope._home_key, memory_manager._signature_params
    monkeypatch.setattr(memory_caller_scope, "_home_key", lambda home: keys.append(home) or real_key(home))
    monkeypatch.setattr(memory_manager, "_signature_params", lambda fn: signatures.append(fn) or real_signature(fn))
    with _Scope(alpha_home):
        for _ in range(3):
            assert json.loads(alpha.handle_tool_call(_HOOK_TOOL, {})) == {"ok": True}
    assert keys == [str(alpha_home)] * 3, keys
    assert len(signatures) <= 1, signatures


def test_refused_checkpoint_compression_fails_closed(homes):
    alpha_home, beta_home = homes
    alpha = _manager(alpha_home)
    with _Scope(beta_home), pytest.raises(RuntimeError, match="hermes logs errors"):
        alpha.on_pre_compress([{"role": "user", "content": "x"}], require_checkpoint=True)


def test_refusals_are_written_to_gateway_log():
    # gateway.log keeps only the gateway component's loggers; the refusals must be among them.
    import logging

    from agent import memory_caller_scope
    from hermes_logging import COMPONENT_PREFIXES, _ComponentFilter
    record = logging.LogRecord(memory_caller_scope.logger.name, logging.WARNING, __file__, 1, "refused", (), None)
    assert _ComponentFilter(COMPONENT_PREFIXES["gateway"]).filter(record)


def test_background_hook_runs_under_the_owning_profile_explicitly(homes, multiplexed):
    # The turn proves its profile through the secret scope alone (no HERMES_HOME override): the
    # background worker must still run the provider under the owner's home, not the process's.
    from agent.secret_scope import reset_secret_scope, set_secret_scope
    alpha_home, _ = homes
    alpha = _manager(alpha_home)
    token = set_secret_scope({}, profile_home=str(alpha_home))
    try:
        alpha.sync_all("what did we decide", "the plan", session_id="s")
        _drain(alpha)
    finally:
        reset_secret_scope(token)
    assert _InstanceScopedProvider.events == [("sync_turn", str(alpha_home), str(alpha_home))]


def test_two_profiles_interleaving_tool_calls_and_hooks_stay_in_their_own_stores(homes, multiplexed):
    alpha_home, beta_home = homes
    managers = {alpha_home: _manager(alpha_home), beta_home: _manager(beta_home)}
    barrier, errors = threading.Barrier(2), []

    def turns(home):
        manager = managers[home]
        agent = SimpleNamespace(
            _memory_manager=manager, _context_engine_tool_names=set(), _gateway_session_key=f"agent:{home.name}:dm",
            _should_emit_quiet_tool_messages=lambda: False, _should_start_quiet_spinner=lambda: False)
        try:
            assert is_multiplex_active(), "the worker lost the multiplexed mode"
            with _Scope(home):
                barrier.wait(timeout=5)
                for i in range(10):
                    ref = _ToolCallRef(name=_HOOK_TOOL, args={}, task_id="t", call_id="c", trace=[])
                    if i % 2:
                        assert json.loads(_resolve_sequential_dispatch(agent, ref, []).execute({})) == {"ok": True}
                    else:
                        assert json.loads(resolve_invoke_tool_executor(agent, _HOOK_TOOL)(agent, {}, SimpleNamespace())) == {"ok": True}
                    for hook in sorted(_HOOKS):
                        _HOOKS[hook](manager)
        except Exception as exc:  # surfaced below
            errors.append(exc)

    # Each worker runs under its own copy of this context, so it inherits the multiplexed mode.
    threads = [threading.Thread(target=ctx_bound(turns), args=(home,)) for home in managers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not errors, errors
    events = _InstanceScopedProvider.events
    for home in managers:
        mine = [e for e in events if e[1] == str(home)]
        assert {e[0] for e in mine} >= {"tool", *{h for h in _HOOKS if h != "session_boundary"}}, mine
        assert all(ambient == str(home) for _, _, ambient in mine), mine
    assert len(events) == sum(1 for e in events if e[1] in {str(alpha_home), str(beta_home)})


def test_one_provider_instance_is_never_served_to_a_second_profile(homes, multiplexed, caplog):
    # A plugin that hands every agent the SAME instance keeps one binding for the whole process.
    # It stays with the profile that initialized it; another profile's manager refuses it.
    alpha_home, beta_home = homes
    shared = _InstanceScopedProvider()
    alpha = _manager(alpha_home, shared)
    beta = _manager(beta_home, shared)
    assert shared._home == str(alpha_home)
    assert not beta.has_tool(_HOOK_TOOL) and beta.providers == []
    with _Scope(alpha_home):
        assert json.loads(alpha.handle_tool_call(_HOOK_TOOL, {})) == {"ok": True}
    assert [e[1] for e in _InstanceScopedProvider.events] == [str(alpha_home)]
    logged = "\n".join(r.getMessage() for r in caplog.records if r.name == "agent.memory_caller_scope")
    assert "memory_provider_instance_shared" in logged and str(beta_home) in logged, logged


def test_agent_whose_provider_instance_is_refused_is_told_and_runs_without_it(homes, multiplexed, monkeypatch, caplog):
    # The real agent-build path: the refused provider is dropped, the user sees a startup warning, and
    # nothing claims the provider "activated".
    import logging

    from agent import agent_init
    from tests.agent.test_memory_migration_notifications import StartupAgent
    alpha_home, beta_home = homes
    shared = _InstanceScopedProvider()
    _manager(alpha_home, shared)
    monkeypatch.setattr("plugins.memory.load_memory_provider", lambda name: shared)
    monkeypatch.setattr(agent_init, "_memory_provider_init_kwargs", lambda agent, platform: {
        "session_id": "s-beta", "platform": platform, "hermes_home": str(beta_home)})
    agent = StartupAgent("api_server")
    with _Scope(beta_home), caplog.at_level(logging.DEBUG):
        agent_init._init_memory(agent, {"memory": {"provider": "instancescoped"}}, False, "api_server")
    assert agent._memory_manager is None
    assert not any("activated" in r.getMessage() for r in caplog.records)
    notices = []
    agent.notice_callback = notices.append
    agent._replay_startup_warnings()
    assert len(notices) == 1 and notices[0].level == "warn" and "instancescoped" in notices[0].text, notices
    assert str(alpha_home) not in notices[0].text and str(beta_home) not in notices[0].text


class _UnhashableProvider(_InstanceScopedProvider):
    __hash__ = None


class _ValueEqualProvider(_InstanceScopedProvider):
    """Every instance compares (and hashes) equal to every other: equality is not identity."""

    def __eq__(self, other):
        return isinstance(other, _ValueEqualProvider)

    def __hash__(self):
        return 0


def test_unhashable_shared_provider_is_refused_for_a_second_profile(homes, multiplexed):
    alpha_home, beta_home = homes
    shared = _UnhashableProvider()
    alpha = _manager(alpha_home, shared)
    beta = _manager(beta_home, shared)
    assert shared._home == str(alpha_home)
    assert alpha.providers and beta.providers == []


def test_value_equal_distinct_instances_are_each_served(homes, multiplexed):
    alpha_home, beta_home = homes
    first, second = _ValueEqualProvider(), _ValueEqualProvider()
    alpha, beta = _manager(alpha_home, first), _manager(beta_home, second)
    assert (first._home, second._home) == (str(alpha_home), str(beta_home))
    assert alpha.providers == [first] and beta.providers == [second]


def test_untrackable_provider_instance_fails_closed_only_when_multiplexed(homes):
    from agent.memory_caller_scope import claim_instance
    from agent.secret_scope import reset_multiplex_context, set_multiplex_context

    class _NoWeakref:
        __slots__ = ("name",)

    alpha_home, _ = homes
    assert claim_instance(_NoWeakref(), alpha_home)
    token = set_multiplex_context(True)
    try:
        assert not claim_instance(_NoWeakref(), alpha_home)
    finally:
        reset_multiplex_context(token)
