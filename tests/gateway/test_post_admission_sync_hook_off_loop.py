"""A slow synchronous ``post_gateway_admission`` plugin callback runs off the gateway event loop:
other coroutines on that loop keep running while the callback works."""
import asyncio
from types import SimpleNamespace

import hermes_yaml as yaml


def test_sync_post_admission_callback_leaves_loop_free(tmp_path, monkeypatch):
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run_inbound_consumer import run_post_admission_hook
    from gateway.session import SessionSource
    from hermes_cli import plugins as plugins_mod

    home = tmp_path / "home"
    plugin = home / "plugins" / "admission-probe"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(yaml.safe_dump({
        "name": "admission-probe", "version": "1.0", "hooks": ["post_gateway_admission"]}), encoding="utf-8")
    # The callback holds its thread until a coroutine on the gateway loop opens GATE; on the loop
    # itself it would wait out the full five seconds and answer "stalled".
    (plugin / "__init__.py").write_text(
        "import threading\n"
        "GATE = threading.Event()\n"
        "def _on_admission(**kwargs):\n"
        "    return {'action': 'handled', 'reply': 'released' if GATE.wait(5) else 'stalled'}\n"
        "def register(ctx):\n"
        "    ctx.register_hook('post_gateway_admission', _on_admission)\n", encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": ["admission-probe"]}}), encoding="utf-8")
    empty_bundled = tmp_path / "bundled"
    empty_bundled.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: empty_bundled)
    plugins_mod.discover_plugins()
    gate = plugins_mod.get_plugin_manager()._plugins["admission-probe"].module.GATE

    source = SessionSource(platform=Platform.DISCORD, chat_id="c1", user_id="u1", chat_type="channel")
    event = MessageEvent(text="hello", source=source, message_id="m1")
    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=False))

    async def gateway_loop():
        admission = asyncio.create_task(run_post_admission_hook(runner, event, source, "session-1"))

        async def other_chat():
            gate.set()

        await asyncio.gather(admission, asyncio.create_task(other_chat()))
        return admission.result()

    assert asyncio.run(gateway_loop()) == (True, "released")
