"""The classic CLI's model choices reach the gateway session.

``/model`` swaps the live agent, and ``/moa`` rebuilds it on the MoA route. Both have to
show up on the next ``prompt.submit``: a session ``session.mutate model``, or that
admission's ``model_override``.
"""
import queue
from types import SimpleNamespace

from hermes_cli.gateway_client import GatewayClientError
from hermes_cli.model_switch import ModelSwitchResult


class RecordingConnection:
    epoch = 1

    def __init__(self, info):
        from hermes_cli.gateway_connection import SessionState
        self.events = queue.Queue()
        self.calls = []
        self.closed = False
        # The session store is truthy until the connection is final.
        self.final = False
        self._state = SessionState("s")
        self._state.info = dict(info)

    def start(self, timeout=90):
        pass

    def state(self, session_id):
        return self._state

    def reply(self, *_a, **_k):
        return True

    def call(self, method, *, timeout=None, **params):
        self.calls.append((method, params))
        if method == "session.resume":
            return {"revision": 7, "execution_generation": 3, "info": dict(self._state.info), "messages": [],
                    "stored_session_id": "s", "session_id": "s", "prompts": [], "pending": []}
        if method == "session.mutate":
            if params.get("operation") == "model":
                self._state.info.update(model=params["payload"]["model"],
                                        provider=params["payload"].get("provider") or self._state.info.get("provider"))
            return {}
        if method == "prompt.submit":
            self.events.put({"method": "event", "params": {
                "type": "message.complete", "session_id": "s", "admission_id": "a1", "execution_generation": 3,
                "payload": {"text": "ok", "status": "complete", "admission_id": "a1"}}})
            return {"admission_id": "a1", "status": "queued"}
        if method == "prompt.receipt":
            return {"status": "terminal", "outcome": "completed", "result": {"final_response": "ok"}}
        if method == "config.set":
            return {}
        raise GatewayClientError("unsupported_method")


def _agent(conn, **kw):
    from hermes_cli.gateway_agent import GatewayAgent
    return GatewayAgent(connection=conn, session_id="s", **kw)


def _model_routes(conn):
    """Every model the CLI asked the gateway to run: session switches and one-turn overrides."""
    mutates = [p["payload"] for m, p in conn.calls if m == "session.mutate" and p.get("operation") == "model"]
    overrides = [p["model_override"] for m, p in conn.calls if m == "prompt.submit" and "model_override" in p]
    return mutates, overrides


def test_classic_cli_model_switch_reaches_the_gateway_session(monkeypatch):
    import cli as cli_mod
    from tests.hermes_cli.test_cli_model_once import _StubCLI
    conn = RecordingConnection({"model": "old/model", "provider": "openrouter"})
    stub = _StubCLI()
    stub.agent = _agent(conn, model="old/model", provider="openrouter")
    stub._snapshot_model_runtime = cli_mod.HermesCLI._snapshot_model_runtime.__get__(stub)
    stub._session_db = None
    stub.session_id = "s"
    printed = []
    monkeypatch.setattr(cli_mod, "_cprint", lambda s, *a, **k: printed.append(str(s)))
    monkeypatch.setattr("hermes_cli.inventory.load_picker_context", lambda: SimpleNamespace(
        user_providers=None, custom_providers=None,
        with_overrides=lambda **_: SimpleNamespace(user_providers=None, custom_providers=None)))
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **_: ModelSwitchResult(
        success=True, new_model="claude-sonnet-4.6", target_provider="anthropic", api_key="sk-ant",
        base_url="https://api.anthropic.com", api_mode="anthropic_messages", provider_label="Anthropic"))
    monkeypatch.setattr("hermes_cli.model_switch.resolve_display_context_length", lambda *a, **k: None)

    cli_mod.HermesCLI._handle_model_switch(stub, "/model claude-sonnet-4.6 --provider anthropic")

    failed = [line for line in printed if "fail" in line.lower()]
    assert not failed, ("/model printed a failure", failed)
    assert stub.model == "claude-sonnet-4.6", stub.model
    stub.agent.run_conversation("next turn")
    mutates, overrides = _model_routes(conn)
    assert mutates or overrides, ("the next turn runs the old model: nothing reached the gateway", conn.calls)


def test_classic_cli_moa_turn_runs_on_the_moa_route():
    """``/moa <prompt>`` retires the agent; the next turn builds one on the MoA preset."""
    conn = RecordingConnection({"model": "session/model", "provider": "openrouter"})
    agent = _agent(conn, model="moa-default", provider="moa", base_url="moa://local", api_mode="chat_completions")
    result = agent.run_conversation("one prompt through MoA")
    assert not result.get("failed"), result
    mutates, overrides = _model_routes(conn)
    routed = [r for r in mutates + overrides if r.get("model") == "moa-default"]
    assert routed, ("the /moa turn ran the session model, not the MoA preset", conn.calls)
