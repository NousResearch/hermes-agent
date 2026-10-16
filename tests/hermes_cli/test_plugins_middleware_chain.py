"""Request middleware chaining (#128638): rewrites compose in registration order."""

import pytest

from hermes_cli import plugins
from hermes_cli.middleware import apply_llm_request_middleware, apply_tool_request_middleware
from hermes_cli.plugins import PluginManager

_REQUEST_KINDS = [
    ("llm_request", apply_llm_request_middleware),
    ("tool_request", lambda payload: apply_tool_request_middleware("fixture", payload)),
]

# Per request kind: ``first`` rewrites, ``meddler`` mutates its views in place and returns None,
# ``second`` records what it saw and rewrites.
_PLUGIN = """
SEEN = {}

def register(ctx):
    for kind, key in [('llm_request', 'request'), ('tool_request', 'args')]:
        def first(_key=key, **kw):
            return {_key: {**kw[_key], 'steps': ['first']}, 'source': 'first'}
        def meddler(_key=key, **kw):
            kw[_key].setdefault('steps', []).append('meddler')
            kw['original_' + _key]['input'].append('meddler')
        def second(_key=key, _kind=kind, **kw):
            SEEN[_kind] = (kw[_key], kw['original_' + _key])
            return {_key: {**kw[_key], 'second': True}, 'source': 'second'}
        for callback in (first, meddler, second):
            ctx.register_middleware(kind, callback)
"""


@pytest.fixture
def seen(tmp_path, monkeypatch):
    home = tmp_path / "home"
    plugin = home / "plugins" / "request_chain"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: request_chain\nversion: 0.1.0\n")
    (plugin / "__init__.py").write_text(_PLUGIN)
    (home / "config.yaml").write_text("plugins:\n  enabled: [request_chain]\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    manager.discover_and_load()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    return manager._plugins["request_chain"].module.SEEN


@pytest.mark.parametrize("kind,apply", _REQUEST_KINDS, ids=[k for k, _ in _REQUEST_KINDS])
def test_request_middleware_rewrites_compose(seen, kind, apply):
    result = apply({"input": ["original"]})

    # The second callback sees the first one's rewrite, not the original request.
    assert seen[kind][0] == {"input": ["original"], "steps": ["first"]}
    assert result.payload == {"input": ["original"], "steps": ["first"], "second": True}
    assert [entry["source"] for entry in result.trace] == ["first", "second"]


@pytest.mark.parametrize("kind,apply", _REQUEST_KINDS, ids=[k for k, _ in _REQUEST_KINDS])
def test_request_middleware_in_place_mutation_does_not_leak(seen, kind, apply):
    original = {"input": ["original"]}

    result = apply(original)

    # Each callback gets its own copy of the payload and of the pre-middleware snapshot.
    assert "meddler" not in result.payload["steps"]
    assert seen[kind][1] == result.original_payload == original == {"input": ["original"]}


class TestAsyncMiddlewareCallbacks:
    """``async def`` request middleware runs and its rewrite lands (#140140), like hook callbacks."""

    def test_async_middleware_result_is_awaited_alongside_sync(self):
        mgr = PluginManager()

        def sync_rewrite(**kwargs):
            return {"request": {**kwargs["request"], "sync": True}, "source": "sync"}

        async def async_rewrite(**kwargs):
            return {"request": {**kwargs["request"], "async": True}, "source": "async"}

        mgr._middleware.setdefault("llm_request", []).extend([sync_rewrite, async_rewrite])
        results = mgr.invoke_middleware("llm_request", _payload_key="request",
                                        request={"messages": []}, original_request={"messages": []})
        assert [r["source"] for r in results] == ["sync", "async"]

    def test_async_middleware_resolves_under_a_running_loop(self, monkeypatch):
        """Gateway handlers dispatch middleware from inside asyncio; a bare coroutine collected
        there never runs, and the rewrite it returns must still compose into the chain."""
        import asyncio

        mgr = PluginManager()

        async def async_rewrite(**kwargs):
            return {"request": {**kwargs["request"], "async": True}, "source": "async"}

        mgr._middleware.setdefault("llm_request", []).append(async_rewrite)
        monkeypatch.setattr(plugins, "get_plugin_manager", lambda: mgr)

        async def driver():
            return apply_llm_request_middleware({"messages": []})

        result = asyncio.run(driver())
        assert result.payload == {"messages": [], "async": True}
        assert [entry["source"] for entry in result.trace] == ["async"]
