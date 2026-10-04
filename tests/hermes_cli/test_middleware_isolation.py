"""Request rewrites and original snapshots stay isolated across discovered middleware."""

from collections import OrderedDict, defaultdict
from typing import Any, NamedTuple, cast

import pytest

from hermes_cli import plugins
from hermes_cli.config import atomic_config_write
from hermes_cli.middleware import (
    apply_llm_request_middleware,
    apply_tool_request_middleware,
    run_llm_execution_middleware,
    run_tool_execution_middleware,
)


class RequestDict(dict):
    pass


class RequestList(list):
    pass


class RequestTuple(NamedTuple):
    data: Any
    file: Any


class OpaqueMember:
    def __init__(self, related):
        self.related = related

    def __deepcopy__(self, memo):
        memo[id(self.related)] = {"partial": True}
        raise TypeError("opaque member cannot be copied")


def _load_plugin(tmp_path, monkeypatch, source):
    home = tmp_path / "home"
    plugin = home / "plugins" / "copy_probe"
    plugin.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugin.joinpath("plugin.yaml").write_text(
        "name: copy_probe\nversion: 0.1.0\ndescription: Middleware isolation probe\n",
        encoding="utf-8",
    )
    plugin.joinpath("__init__.py").write_text(source, encoding="utf-8")
    atomic_config_write(home / "config.yaml", {"plugins": {"enabled": ["copy_probe"]}})
    manager = plugins.PluginManager()
    manager.discover_and_load()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    assert any(manager.has_middleware(kind) for kind in (
        "llm_request", "tool_request", "llm_execution", "tool_execution",
    ))


def _apply(kind, payload):
    if kind == "llm":
        return apply_llm_request_middleware(payload)
    return apply_tool_request_middleware("probe", payload)


@pytest.mark.parametrize("kind", ["llm", "tool"])
@pytest.mark.parametrize("stage", ["request", "execution", "execution_only"])
@pytest.mark.parametrize("outcome", ["observe", "raise", "rewrite"])
@pytest.mark.parametrize("opaque", [False, True])
def test_original_snapshot_survives_each_middleware_callback(
    tmp_path, monkeypatch, kind, stage, outcome, opaque,
):
    key = "request" if kind == "llm" else "args"
    request_listener = stage != "execution_only"
    if not request_listener:
        stage = "execution"
    _load_plugin(tmp_path, monkeypatch, f"""
KIND = {kind!r}
KEY = {key!r}
STAGE = {stage!r}
OUTCOME = {outcome!r}
REQUEST_LISTENER = {request_listener!r}

def mutate(**kw):
    original = kw['original_' + KEY]
    original['input'].append('snapshot-write')
    original['injected'] = True
    if not REQUEST_LISTENER:
        kw[KEY]['input'].append('execution-write')
    if OUTCOME == 'raise':
        raise RuntimeError('snapshot probe failure')
    if OUTCOME == 'rewrite':
        payload = {{**kw[KEY], 'rewritten': True}}
        if STAGE == 'request':
            return {{KEY: payload, 'source': 'mutate'}}
        return kw['next_call'](payload)
    if STAGE == 'execution':
        return kw['next_call']()

def verify(**kw):
    original = kw['original_' + KEY]
    stage = 'execution' if 'next_call' in kw else 'request'
    payload = {{**kw[KEY], stage + '_snapshot': list(original['input'])}}
    payload[stage + '_injected'] = 'injected' in original
    if 'next_call' in kw:
        return kw['next_call'](payload)
    return {{KEY: payload, 'source': 'verify'}}

def register(ctx):
    ctx.register_middleware(KIND + '_' + STAGE, mutate)
    if REQUEST_LISTENER:
        ctx.register_middleware(KIND + '_request', verify)
    ctx.register_middleware(KIND + '_execution', verify)
""")
    with (tmp_path / "upload.txt").open("w+", encoding="utf-8") as handle:
        original = {"input": ["original"], **({"file": handle} if opaque else {})}
        result = _apply(kind, original) if request_listener else None
        payload = result.payload if result else original
        context = {"original_" + key: result.original_payload} if result else {}
        calls = []

        def terminal(payload):
            calls.append(payload)
            return payload

        if kind == "llm":
            executed = run_llm_execution_middleware(
                payload, terminal, **context,
            )
        else:
            executed = run_tool_execution_middleware(
                "probe", payload, terminal, **context,
            )

        expected_input = ["original"] if request_listener else ["original", "execution-write"]
        assert original["input"] == expected_input
        assert "injected" not in original
        if result:
            assert result.original_payload["input"] == ["original"]
            assert "injected" not in result.original_payload
        for phase in ("request", "execution") if request_listener else ("execution",):
            assert executed[phase + "_snapshot"] == ["original"]
            assert executed[phase + "_injected"] is False
        assert executed["input"] == expected_input
        assert bool(executed.get("rewritten")) == (outcome == "rewrite")
        assert calls == [executed]
        expected_sources = ["verify"]
        if stage == "request" and outcome == "rewrite":
            expected_sources.insert(0, "mutate")
        if result:
            assert [entry["source"] for entry in result.trace] == expected_sources
        if opaque:
            assert executed["file"] is original["file"] is handle
            if result:
                assert result.original_payload["file"] is handle
            assert not handle.closed


def _leaf(node):
    while isinstance(node, dict) and "child" in node:
        node = node["child"]
    while isinstance(node, (list, tuple)):
        node = node[0]
    return node


@pytest.mark.parametrize("kind", ["llm", "tool"])
@pytest.mark.parametrize("container", [
    dict, OrderedDict, defaultdict, RequestDict, list, RequestList, tuple, RequestTuple,
])
@pytest.mark.parametrize("depth,wrapper", [
    (0, dict), (1500, dict), (6000, dict), (6000, list), (6000, tuple),
])
def test_opaque_request_graph_preserves_isolation_aliases_and_cycles(
    tmp_path, monkeypatch, kind, container, depth, wrapper,
):
    key = "request" if kind == "llm" else "args"
    _load_plugin(tmp_path, monkeypatch, f"""
KIND = {kind!r}
KEY = {key!r}

def leaf(node):
    while isinstance(node, dict) and 'child' in node:
        node = node['child']
    while isinstance(node, (list, tuple)):
        node = node[0]
    return node

def first(**kw):
    return {{KEY: {{**kw[KEY], 'accepted': True}}, 'source': 'first'}}

def broken(**kw):
    leaf(kw[KEY]['node'])['steps'].append('broken')
    raise RuntimeError('graph probe failure')

def observer(**kw):
    leaf(kw[KEY]['node'])['steps'].append('observer')

def second(**kw):
    leaf(kw[KEY]['node'])['steps'].append('second')
    return {{KEY: kw[KEY], 'source': 'second'}}

def register(ctx):
    for callback in (first, broken, observer, second):
        ctx.register_middleware(KIND + '_request', callback)
""")
    with (tmp_path / "upload.txt").open("w+", encoding="utf-8") as handle:
        leaf = {"steps": ["original"], "file": handle}
        if issubclass(container, dict):
            node = container(list, leaf) if container is defaultdict else container(leaf)
        elif container is RequestTuple:
            node = container(leaf, handle)
        else:
            node = container([leaf, handle])
        if container in (RequestDict, RequestList):
            node.metadata = ["original"]
        for _ in range(depth):
            node = {"child": node} if wrapper is dict else wrapper([node])
        cycle = []
        pair = (cycle, handle)
        cycle.append(pair)
        opaque = OpaqueMember(node)
        original: dict[str, Any] = {
            "opaque": opaque, "node": node, "alias": node, "cycle": pair,
            "cycle_alias": pair, "tags": {"original"},
            "frozen": frozenset({RequestTuple("original", handle)}),
        }
        try:
            result = _apply(kind, original)
        except RecursionError:
            # Keep pytest from comparing deeply nested frame locals while locating
            # recursion: that can hang its traceback formatter on this graph.
            raise AssertionError("Deep requests must still run middleware") from None

        assert result.payload["accepted"] is True
        assert _leaf(result.payload["node"])["steps"] == ["original", "second"]
        assert _leaf(result.original_payload["node"])["steps"] == ["original"]
        assert _leaf(original["node"])["steps"] == ["original"]
        assert result.payload["node"] is result.payload["alias"]
        assert result.original_payload["node"] is result.original_payload["alias"]
        assert result.payload["cycle"][0][0] is result.payload["cycle"]
        assert result.original_payload["cycle"][0][0] is result.original_payload["cycle"]
        assert result.payload["cycle_alias"] is result.payload["cycle"]
        assert result.original_payload["cycle_alias"] is result.original_payload["cycle"]
        assert result.payload["opaque"] is result.original_payload["opaque"] is opaque
        assert result.payload["frozen"] == result.original_payload["frozen"] == original["frozen"]
        assert _leaf(result.payload["node"])["file"] is handle
        assert not handle.closed
        assert [entry["source"] for entry in result.trace] == ["first", "second"]
        copied_node = result.payload["node"]
        snapshot_node = result.original_payload["node"]
        source_node = cast(Any, original["node"])
        for _ in range(depth):
            index = "child" if wrapper is dict else 0
            copied_node = copied_node[index]
            snapshot_node = snapshot_node[index]
            source_node = source_node[index]
        assert type(copied_node) is type(snapshot_node) is type(source_node) is container
        if container is defaultdict:
            assert copied_node.default_factory is snapshot_node.default_factory is list
        if container in (RequestDict, RequestList):
            copied_node.metadata.append("accepted-only")
            assert source_node.metadata == snapshot_node.metadata == ["original"]
        result.payload["tags"].add("accepted-only")
        assert original["tags"] == result.original_payload["tags"] == {"original"}
