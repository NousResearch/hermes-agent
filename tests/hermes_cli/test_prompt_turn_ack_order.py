"""/plan, /learn, /init acknowledge on the same paint channel as the turn they queue.

Main prints the acknowledgement, then the queued turn paints its ``────`` separator. A bare
``print`` under patch_stdout is flushed by the proxy's writer thread after its batching sleep,
while the separator is a ``_cprint`` on the app loop, so a fast turn start painted the separator
first. Both on ``_cprint`` keeps main's order whatever the turn's start latency."""
import queue
import types

import pytest


@pytest.mark.parametrize(("handler", "arg", "needle"), [
    ("_handle_plan_command", "/plan a tiny plan", "a tiny plan"),
    ("_handle_learn_command", "/learn the deploy flow", None),
    ("_handle_init_command", "/init", None),
])
def test_prompt_turn_ack_paints_through_cprint_before_the_turn_is_queued(handler, arg, needle, monkeypatch, capsys):
    import cli
    from hermes_cli.cli_commands_mixin import CLICommandsMixin
    events = []
    monkeypatch.setattr(cli, "_pt_app_is_running", lambda: True)
    monkeypatch.setattr(cli, "_cprint", lambda text: events.append(("paint", text)))

    class Q(queue.Queue):
        def put(self, item, *a, **k):
            events.append(("queued", None))
            super().put(item, *a, **k)
    host = types.SimpleNamespace(_pending_input=Q())
    host._queue_prompt_turn = lambda *a: CLICommandsMixin._queue_prompt_turn(host, *a)
    getattr(CLICommandsMixin, handler)(host, arg)
    assert capsys.readouterr().out == ""  # nothing through the patch_stdout proxy
    assert [kind for kind, _ in events] == ["paint", "queued"]
    assert events[0][1].startswith("\n")
    if needle:
        assert needle in events[0][1]
