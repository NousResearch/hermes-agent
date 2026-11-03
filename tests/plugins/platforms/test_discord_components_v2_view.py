"""Components V2 view construction against the real discord.py library.

Kept out of tests/gateway/ deliberately: that package's conftest installs a MagicMock
`discord`, so discord.ui.Container/LayoutView are not the real classes there and these
assertions would be vacuous. Planning logic (no discord.py needed) is covered by
tests/gateway/test_discord_components_v2.py.

The real library is loaded inside each test rather than at module scope, and sys.modules is
restored afterwards. Importing it at import time would leave a real `discord` cached during
collection, which makes tests/gateway/conftest.py's `_ensure_discord_mock()` short-circuit
for gateway files collected later; those files bind `discord` at import and then fail on
real constructors (`discord.Thread()` wants guild/state/data). That cost 7 unrelated
failures in slash-auth and image-send, each of which passed in isolation.
"""

import importlib
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))


@contextmanager
def real_discord():
    """Swap in the genuine discord.py for the duration of the block, then restore exactly."""
    saved = {name: mod for name, mod in sys.modules.items()
             if name == "discord" or name.startswith("discord.")}
    for name in list(saved):
        del sys.modules[name]
    try:
        try:
            module = importlib.import_module("discord")
        except ImportError:
            pytest.skip("discord.py is not installed")
        if not hasattr(module, "__file__"):
            pytest.skip("a mock discord module is shadowing the real library")
        if not hasattr(module.ui, "LayoutView"):
            pytest.skip("discord.py is too old for Components V2")
        yield module
    finally:
        for name in [n for n in sys.modules if n == "discord" or n.startswith("discord.")]:
            del sys.modules[name]
        sys.modules.update(saved)


def _load_components_v2():
    """Load the renderer by path; its discord import is lazy and happens at call time."""
    path = _ROOT / "plugins" / "platforms" / "discord" / "components_v2.py"
    spec = importlib.util.spec_from_file_location("_hermes_components_v2_view_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestBuildLayoutView:
    def test_returns_none_when_plan_declines(self):
        with real_discord():
            assert _load_components_v2().build_layout_view("x" * 5000) is None

    def test_builds_a_view_for_ordinary_content(self):
        with real_discord():
            assert _load_components_v2().build_layout_view("hello\n\nworld") is not None

    def test_view_declares_components_v2(self):
        """discord.py derives the IS_COMPONENTS_V2 message flag from has_components_v2()."""
        with real_discord():
            view = _load_components_v2().build_layout_view("hello")
            assert view is not None
            assert view.has_components_v2() is True

    def test_view_stays_within_the_40_component_cap(self):
        with real_discord():
            c2 = _load_components_v2()
            view = c2.build_layout_view("\n\n".join(f"para {i}" for i in range(15)))
            assert view is not None
            assert view._total_children <= c2.MAX_TOTAL_COMPONENTS

    def test_code_content_survives_into_the_rendered_components(self):
        with real_discord():
            view = _load_components_v2().build_layout_view("intro\n\n```py\nprint('hi')\n```")
            assert view is not None
            assert "print('hi')" in str(view.to_components())

    def test_fence_and_prose_become_separate_text_displays(self):
        with real_discord():
            view = _load_components_v2().build_layout_view("intro\n\n```py\nx=1\n```")
            assert view is not None
            payload = str(view.to_components())
            assert "intro" in payload and "x=1" in payload

    def test_empty_content_returns_none(self):
        with real_discord():
            assert _load_components_v2().build_layout_view("") is None


def test_sys_modules_is_left_undisturbed():
    """Guard the invariant that makes this file safe to run alongside gateway tests."""
    before = dict(sys.modules)
    with real_discord() as module:
        assert hasattr(module, "__file__")
    after = {n: m for n, m in sys.modules.items()
             if n == "discord" or n.startswith("discord.")}
    expected = {n: m for n, m in before.items()
                if n == "discord" or n.startswith("discord.")}
    assert after == expected
