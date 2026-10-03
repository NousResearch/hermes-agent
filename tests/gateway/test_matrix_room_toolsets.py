"""Per-room Matrix toolset overrides (adapter.toolsets_for_source) — local addition.

A matrix room listed in ``platforms.matrix.extra.room_toolsets`` replaces the
platform-level ``platform_toolsets.matrix`` resolution for runs from that room
only (voice room = slim toolset; every other room = full platform set). The
gateway validates the override through the same ``_get_platform_tools`` path as
platform config — mirrors tests/gateway/test_webhook_route_toolsets.py.
"""
import types

try:  # mirrors test_matrix_voice.py: adapter imports mautrix at module load
    import mautrix as _mautrix_probe
    import pytest
    if not isinstance(_mautrix_probe, types.ModuleType) or not hasattr(_mautrix_probe, "__file__"):
        pytest.skip("mautrix in sys.modules is a mock, not the real package", allow_module_level=True)
except ImportError:
    import pytest
    pytest.skip("mautrix not installed", allow_module_level=True)

from types import SimpleNamespace

from gateway.run import GatewayRunner
from hermes_cli.tools_config import _get_platform_tools
from plugins.platforms.matrix.adapter import MatrixAdapter

VOICE_ROOM = "!voice:example.org"
OTHER_ROOM = "!text:example.org"
SLIM = ["memory", "session_search", "terminal", "web", "delegation", "kanban"]
FULL = ["file", "memory", "session_search", "skills", "terminal", "vision", "web", "delegation", "kanban"]


def _make_adapter(extra):
    ma = object.__new__(MatrixAdapter)
    ma.config = SimpleNamespace(extra=extra)
    return ma


class _Src:
    def __init__(self, chat_id, parent_chat_id=None):
        self.chat_id = chat_id
        self.parent_chat_id = parent_chat_id


def _make_runner(adapter):
    gr = object.__new__(GatewayRunner)
    # Resolver renamed upstream (_adapter_for_source -> _delivery_adapter_for);
    # mirror tests/gateway/test_webhook_route_toolsets.py::_make_runner.
    gr._delivery_adapter_for = lambda source: adapter
    return gr


class TestMatrixAdapterRoomToolsets:
    def test_listed_room_returns_override(self):
        ma = _make_adapter({"room_toolsets": {VOICE_ROOM: SLIM}})
        assert ma.toolsets_for_source(_Src(VOICE_ROOM)) == SLIM

    def test_thread_parent_inherits_override(self):
        ma = _make_adapter({"room_toolsets": {VOICE_ROOM: SLIM}})
        assert ma.toolsets_for_source(_Src("$thread", parent_chat_id=VOICE_ROOM)) == SLIM

    def test_unlisted_room_returns_none(self):
        ma = _make_adapter({"room_toolsets": {VOICE_ROOM: SLIM}})
        assert ma.toolsets_for_source(_Src(OTHER_ROOM)) is None

    def test_malformed_entries_return_none(self):
        ma = _make_adapter({"room_toolsets": {VOICE_ROOM: [], "!b:example.org": "terminal", "!c:example.org": ["  ", ""]}})
        assert ma.toolsets_for_source(_Src(VOICE_ROOM)) is None
        assert ma.toolsets_for_source(_Src("!b:example.org")) is None
        assert ma.toolsets_for_source(_Src("!c:example.org")) is None

    def test_no_extra_or_not_dict_returns_none(self):
        ma = _make_adapter({})
        assert ma.toolsets_for_source(_Src(VOICE_ROOM)) is None
        ma = _make_adapter({"room_toolsets": "not-a-dict"})
        assert ma.toolsets_for_source(_Src(VOICE_ROOM)) is None


class TestGatewayResolveWithMatrixRoomOverride:
    def test_voice_room_override_replaces_platform_list(self):
        ma = _make_adapter({"room_toolsets": {VOICE_ROOM: SLIM}})
        gr = _make_runner(ma)
        res = GatewayRunner._resolve_enabled_toolsets_for_source(
            gr, {"platform_toolsets": {"matrix": FULL}}, _Src(VOICE_ROOM), "matrix"
        )
        expected = sorted(_get_platform_tools({"platform_toolsets": {"matrix": list(SLIM)}}, "matrix"))
        assert res == expected
        assert "file" not in res and "skills" not in res and "vision" not in res

    def test_other_room_keeps_full_platform_list(self):
        ma = _make_adapter({"room_toolsets": {VOICE_ROOM: SLIM}})
        gr = _make_runner(ma)
        cfg = {"platform_toolsets": {"matrix": FULL}}
        res = GatewayRunner._resolve_enabled_toolsets_for_source(gr, cfg, _Src(OTHER_ROOM), "matrix")
        assert res == sorted(_get_platform_tools(cfg, "matrix"))
        assert "file" in res and "skills" in res and "vision" in res

    def test_original_config_not_mutated(self):
        cfg = {"platform_toolsets": {"matrix": list(FULL)}}
        ma = _make_adapter({"room_toolsets": {VOICE_ROOM: SLIM}})
        gr = _make_runner(ma)
        GatewayRunner._resolve_enabled_toolsets_for_source(gr, cfg, _Src(VOICE_ROOM), "matrix")
        assert cfg["platform_toolsets"]["matrix"] == FULL

    def test_adapter_exception_falls_back_to_platform(self):
        ma = _make_adapter({})
        ma.toolsets_for_source = lambda source: (_ for _ in ()).throw(RuntimeError("boom"))
        gr = _make_runner(ma)
        cfg = {"platform_toolsets": {"matrix": FULL}}
        res = GatewayRunner._resolve_enabled_toolsets_for_source(gr, cfg, _Src(VOICE_ROOM), "matrix")
        assert res == sorted(_get_platform_tools(cfg, "matrix"))
