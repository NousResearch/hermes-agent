"""``pre_update`` directive hook: a plugin can veto ``hermes update`` before state mutation.

The hook fires once per update in :func:`hermes_cli.update_cmd._run_pre_update_hook`,
BEFORE any receipt/backup/git mutation. A plugin returning ``{"action": "block", "message"}``
aborts the update cleanly (exit 1); anything else proceeds. These tests drive the helper
directly with the lifecycle dispatch monkeypatched so no plugin machinery loads.
"""

from types import SimpleNamespace

import pytest

from hermes_cli import update_cmd


def test_pre_update_is_a_valid_hook():
    """The hook name is registered so a plugin may subscribe to it."""
    from hermes_cli.plugins import VALID_HOOKS

    assert "pre_update" in VALID_HOOKS


def test_no_subscriber_short_circuits(monkeypatch):
    """No ``pre_update`` subscriber -> no-op; the hook is never invoked."""
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: False)
    invoked: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda name, **kw: invoked.append(name) or [],
    )
    update_cmd._run_pre_update_hook(SimpleNamespace(), gateway_mode=False)
    assert invoked == []


def test_block_directive_vetoes(monkeypatch):
    """A plugin returning ``{"action": "block", "message"}`` aborts the update (exit 1)."""
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda name, **kw: [{"action": "block", "message": "secrets found in state.db"}],
    )
    stops: list[tuple] = []
    monkeypatch.setattr(
        update_cmd, "_record_stop", lambda reason, **kw: stops.append((reason, kw))
    )
    with pytest.raises(SystemExit) as exc:
        update_cmd._run_pre_update_hook(SimpleNamespace(), gateway_mode=False)
    assert exc.value.code == 1
    assert stops == [("pre_update_blocked", {"without_receipt": "failed"})]


def test_block_without_message_still_vetoes(monkeypatch):
    """A block directive without a message still aborts (message is optional UX)."""
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook", lambda name, **kw: [{"action": "block"}]
    )
    with pytest.raises(SystemExit) as exc:
        update_cmd._run_pre_update_hook(SimpleNamespace(), gateway_mode=False)
    assert exc.value.code == 1


def test_non_block_return_proceeds(monkeypatch):
    """None / non-block / non-dict returns are ignored; the update proceeds."""
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda name, **kw: [None, {"action": "allow"}, "garbage", {"message": "no action"}],
    )
    # No SystemExit means the veto gate passed.
    update_cmd._run_pre_update_hook(SimpleNamespace(), gateway_mode=False)


def test_gateway_mode_and_args_forwarded(monkeypatch):
    """The hook receives ``gateway_mode`` and ``args`` as kwargs."""
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
    captured: dict = {}
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda name, **kw: captured.update(kw) or [],
    )
    args = SimpleNamespace(gateway=True, branch=None)
    update_cmd._run_pre_update_hook(args, gateway_mode=True)
    assert captured == {"gateway_mode": True, "args": args}
