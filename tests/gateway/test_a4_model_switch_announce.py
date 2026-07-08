"""A4 axis A (gateway) — the deliberate /model switch announce.

Proves the shared ``_announce_model_switch`` helper is config-gated
(model.announce_switch, default ON) and routes through the agent's
``_emit_switch_announce`` -> ``_emit_status`` seam, and that BOTH /model
handlers (the picker path ``_on_model_selected`` and the text-arg path
``_finish_switch``) actually call it — the "fix the whole bug class, not one
site" requirement — asserted at the source level (AST) so a future edit that
drops the announce from one handler fails the gate.
"""

import ast
import os
import types


def _mixin_instance():
    """Bare object carrying just the mixin's _announce_model_switch (bound)."""
    from gateway.slash_commands import GatewaySlashCommandsMixin

    obj = GatewaySlashCommandsMixin.__new__(GatewaySlashCommandsMixin)
    return obj


def _capture_agent():
    a = types.SimpleNamespace()
    a._announced = []
    a._emit_status = lambda m: a._announced.append(m)
    a._last_switch_announced = None
    return a


def test_announce_helper_emits_when_gate_default_on(monkeypatch):
    import gateway.run as run

    # No announce_switch key -> default ON.
    monkeypatch.setattr(run, "_load_gateway_config", lambda: {"model": {}}, raising=True)
    obj = _mixin_instance()
    agent = _capture_agent()
    obj._announce_model_switch(
        agent,
        old_model="claude-opus-4-8", new_model="gpt-5.5",
        old_provider="claude-app", new_provider="openai-codex",
    )
    msgs = [m for m in agent._announced if m.startswith("🔀")]
    assert len(msgs) == 1, agent._announced
    assert "claude-app/claude-opus-4-8" in msgs[0]
    assert "openai-codex/gpt-5.5" in msgs[0]


def test_announce_helper_silent_when_gate_off(monkeypatch):
    import gateway.run as run

    monkeypatch.setattr(
        run, "_load_gateway_config",
        lambda: {"model": {"announce_switch": False}}, raising=True,
    )
    obj = _mixin_instance()
    agent = _capture_agent()
    obj._announce_model_switch(
        agent,
        old_model="a", new_model="b", old_provider="p1", new_provider="p2",
    )
    assert agent._announced == [], f"gate off must be silent, got {agent._announced!r}"


def test_announce_helper_never_raises_on_bad_agent(monkeypatch):
    import gateway.run as run

    monkeypatch.setattr(run, "_load_gateway_config", lambda: {"model": {}}, raising=True)
    obj = _mixin_instance()
    # agent with an _emit_status that raises must not propagate (switch stands).
    bad = types.SimpleNamespace()
    bad._last_switch_announced = None

    def _boom(_m):
        raise RuntimeError("emit broke")

    bad._emit_status = _boom
    obj._announce_model_switch(
        bad, old_model="a", new_model="b", old_provider="p", new_provider="q",
    )  # must not raise


def test_both_model_handlers_call_announce():
    """Source contract: BOTH the picker (_on_model_selected) and the text-arg
    (_finish_switch) handlers must invoke self._announce_model_switch — a fix
    wired into only one site leaves the other silent."""
    import gateway.slash_commands as sc

    src = open(sc.__file__, encoding="utf-8").read()
    tree = ast.parse(src)

    def _calls_announce(node):
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                fn = n.func
                if isinstance(fn, ast.Attribute) and fn.attr == "_announce_model_switch":
                    return True
        return False

    found = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in ("_on_model_selected", "_finish_switch"):
                found[node.name] = _calls_announce(node)

    assert found.get("_on_model_selected") is True, (
        "picker handler _on_model_selected must call _announce_model_switch"
    )
    assert found.get("_finish_switch") is True, (
        "text-arg handler _finish_switch must call _announce_model_switch"
    )
