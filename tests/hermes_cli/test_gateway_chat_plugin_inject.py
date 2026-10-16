"""`ctx.inject_message(text)` with no session_key/origin (plugins.md "Active CLI conversation") reaches
the attached classic `hermes chat` session as a prompt.submit, from a plugin's own thread (B2-11)."""
import asyncio
import threading

import pytest


@pytest.mark.asyncio
async def test_no_key_inject_submits_into_the_attached_classic_session(capsys):
    from hermes_cli.gateway_chat_view import GatewayChatView
    from hermes_cli.plugins import PluginContext, get_plugin_manager
    from hermes_cli.plugins_manifest import PluginManifest
    submitted = asyncio.Event()

    class Peer:
        events = asyncio.Queue()
        calls = []

        async def rpc(self, method, **params):
            self.calls.append((method, params))
            if method == "prompt.submit":
                submitted.set()
            return {"admission_id": "adm"}

    view = GatewayChatView(Peer(), {"stored_session_id": "sid"})
    ctx = PluginContext(PluginManifest(name="watcher"), get_plugin_manager())
    accepted = []

    async def read_line(prompt, symbol):
        if accepted:
            return "/quit"
        thread = threading.Thread(target=lambda: accepted.append(ctx.inject_message("New data arrived")))
        thread.start()
        thread.join()
        await asyncio.wait_for(submitted.wait(), 5)
        return ""

    view._read_line = read_line
    assert await view.run() == 0
    assert accepted == [True]
    assert [(m, p["session_id"], p["text"]) for m, p in Peer.calls] == [("prompt.submit", "sid", "New data arrived")]
    assert get_plugin_manager()._cli_ref is None  # detached: the slot is released
    assert "● New data arrived" in capsys.readouterr().out
