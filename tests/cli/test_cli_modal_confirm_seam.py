"""Seam tests for the CLIModalConfirmMixin extraction (cli.py god-file slice R3).

Verifies the mixin seam (identity / MRO / no-back-import) plus behavioral
drives of the moved modal/confirm machinery through SimpleNamespace stand-ins
(the ``_bound`` pattern used across tests/cli) — no full HermesCLI construction
required.
"""

from __future__ import annotations

import queue as _stdlib_queue
import subprocess
import sys
import threading
import time as _time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import cli_modal_confirm_mixin as mcm

_REPO_ROOT = Path(__file__).resolve().parents[2]

_MEMBERS = [
    "_run_curses_picker",
    "_prompt_text_input",
    "_prompt_text_input_modal",
    "_submit_slash_confirm_response",
    "_normalize_slash_confirm_choice",
    "_get_slash_confirm_display_fragments",
]


def _bound(fn, instance):
    """Bind an unbound method to a stand-in instance."""
    return fn.__get__(instance, type(instance))


def _make_modal_self(**overrides):
    """Minimal 'self' shaped like HermesCLI for the modal/confirm methods."""
    calls = {"capture": 0, "restore": 0, "invalidate": 0}

    def _invalidate():
        calls["invalidate"] += 1

    self_ = SimpleNamespace(
        _app=SimpleNamespace(loop=object()),
        _slash_confirm_state=None,
        _slash_confirm_deadline=0,
        _status_bar_visible=True,
        _capture_modal_input_snapshot=lambda: calls.__setitem__("capture", calls["capture"] + 1),
        _restore_modal_input_snapshot=lambda: calls.__setitem__("restore", calls["restore"] + 1),
        _invalidate=_invalidate,
        _prompt_text_input=lambda _prompt: "2",
    )
    for key, value in overrides.items():
        setattr(self_, key, value)
    self_._calls = calls
    return self_


class _EmptyOnDrainQueue:
    """queue.Queue stand-in: put()/get() with stdlib Empty semantics."""

    def __init__(self):
        self._items = []

    def put(self, value):
        self._items.append(value)

    def get(self, timeout=None):
        if not self._items:
            raise _stdlib_queue.Empty
        return self._items.pop(0)


# ---------------------------------------------------------------------------
# Seam: identity + MRO
# ---------------------------------------------------------------------------


def test_seam_identity_all_six_members():
    """HermesCLI.<member> IS CLIModalConfirmMixin.<member> for all six."""
    import cli

    for name in _MEMBERS:
        assert getattr(cli.HermesCLI, name) is getattr(mcm.CLIModalConfirmMixin, name), name


def test_seam_mro_order():
    """Mixin is last in the base chain; no member shadowed by earlier mixins."""
    import cli

    mro = cli.HermesCLI.__mro__
    assert mcm.CLIModalConfirmMixin in mro
    # the three pre-existing mixins resolve before the new one
    assert mro.index(mcm.CLIModalConfirmMixin) > mro.index(cli.CLIAgentSetupMixin)
    assert mro.index(mcm.CLIModalConfirmMixin) > mro.index(cli.CLICommandsMixin)
    assert mro.index(mcm.CLIModalConfirmMixin) > mro.index(cli.CLIBillingMixin)


def test_seam_patch_binding_through_seam(monkeypatch):
    """Patching CLIModalConfirmMixin changes what HermesCLI resolves."""
    import cli

    fake = lambda self: "patched"  # noqa: E731
    monkeypatch.setattr(mcm.CLIModalConfirmMixin, "_prompt_text_input_modal", fake)
    assert cli.HermesCLI._prompt_text_input_modal is fake
    assert _bound(cli.HermesCLI._prompt_text_input_modal, SimpleNamespace())() == "patched"


def test_seam_services_stay_on_hermescli():
    """Class services the modal relies on remain reachable on HermesCLI."""
    import cli

    for name in ("_capture_modal_input_snapshot", "_restore_modal_input_snapshot", "_invalidate"):
        assert hasattr(cli.HermesCLI, name), name


# ---------------------------------------------------------------------------
# No-back-import: the mixin must never import `cli`
# ---------------------------------------------------------------------------


def test_no_back_import_cli_blocked():
    """Importing the mixin with cli blocked in sys.modules must succeed."""
    code = (
        "import sys; sys.modules['cli'] = None; "
        "import hermes_cli.cli_modal_confirm_mixin as m; "
        "print(m.CLIModalConfirmMixin.__name__)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=_REPO_ROOT, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "CLIModalConfirmMixin"


def test_import_order_mixin_first_then_cli():
    """Mixin-first import order must not NameError when cli loads after."""
    code = "import hermes_cli.cli_modal_confirm_mixin; import cli; print(cli.HermesCLI.__name__)"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=_REPO_ROOT, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "HermesCLI"


# ---------------------------------------------------------------------------
# Behavioral drives of the moved machinery
# ---------------------------------------------------------------------------


def test_modal_flow_setup_timeout_teardown(monkeypatch):
    """Modal with immediate deadline: setup -> Empty -> timeout -> teardown -> None."""
    import cli

    monkeypatch.setattr(
        mcm, "queue",
        SimpleNamespace(Queue=_EmptyOnDrainQueue, Empty=_stdlib_queue.Empty),
    )
    self_ = _make_modal_self()
    fn = _bound(cli.HermesCLI._prompt_text_input_modal, self_)

    result = fn(
        title="Confirm reset",
        detail="This clears the session.",
        choices=[("once", "Once", "run once"), ("always", "Always", "remember"), ("cancel", "Cancel", "stop")],
        timeout=0,
    )

    assert result is None
    assert self_._slash_confirm_state is None
    assert self_._slash_confirm_deadline == 0
    assert self_._calls["capture"] == 1
    assert self_._calls["restore"] == 1
    assert self_._calls["invalidate"] >= 2


def test_modal_flow_submitted_response():
    """A composer-submitted response flows out of the modal; state fully reset."""
    import cli

    self_ = _make_modal_self()  # _app truthy, main thread -> direct loop dispatch
    fn = _bound(cli.HermesCLI._prompt_text_input_modal, self_)
    result_holder = {}

    def _submitter():
        deadline = _time.monotonic() + 5
        while self_._slash_confirm_state is None and _time.monotonic() < deadline:
            _time.sleep(0.01)
        assert self_._slash_confirm_state is not None, "modal never came up"
        _bound(cli.HermesCLI._submit_slash_confirm_response, self_)("once")

    t = threading.Thread(target=_submitter)
    t.start()
    try:
        result_holder["result"] = fn(
            title="Confirm",
            detail="d",
            choices=[("once", "Once", "run once"), ("always", "Always", "remember"), ("cancel", "Cancel", "stop")],
            timeout=60,
        )
    finally:
        t.join(timeout=10)

    assert not t.is_alive()
    assert result_holder["result"] == "once"
    assert self_._slash_confirm_state is None
    assert self_._slash_confirm_deadline == 0
    assert self_._calls["capture"] == 1
    assert self_._calls["restore"] == 1


def test_modal_no_app_falls_back_to_text_input():
    """No prompt_toolkit app -> plain _prompt_text_input fallback."""
    import cli

    self_ = _make_modal_self(_app=None)
    fn = _bound(cli.HermesCLI._prompt_text_input_modal, self_)
    result = fn(title="t", detail="d", choices=[("once", "Once", ""), ("cancel", "Cancel", "")])
    assert result == "2"  # stand-in _prompt_text_input returns "2"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific deadlock guard (#33961)")
def test_modal_win32_off_main_thread_cancels():
    """Windows + app running + non-main thread -> clean None, no hang."""
    import cli

    self_ = _make_modal_self()
    self_._app = SimpleNamespace(loop=None)
    result_holder = {}

    def _run():
        fn = _bound(cli.HermesCLI._prompt_text_input_modal, self_)
        result_holder["result"] = fn(
            title="t", detail="d", choices=[("once", "Once", ""), ("cancel", "Cancel", "")]
        )

    t = threading.Thread(target=_run)
    t.start()
    t.join(timeout=10)
    assert not t.is_alive()
    assert result_holder["result"] is None
    assert self_._calls["invalidate"] >= 1


def test_normalize_slash_confirm_choice():
    """Alias table, case/whitespace folding, allowed-value passthrough."""
    import cli

    self_ = _make_modal_self()
    fn = _bound(cli.HermesCLI._normalize_slash_confirm_choice, self_)
    choices = [("once", "Once", ""), ("always", "Always", ""), ("cancel", "Cancel", "")]

    assert fn(None, choices) is None
    assert fn("", choices) is None
    assert fn("   ", choices) is None
    assert fn("1", choices) == "once"
    assert fn("YES", choices) == "once"
    assert fn(" y ", choices) == "once"
    assert fn("ok", choices) == "once"
    assert fn("2", choices) == "always"
    assert fn("remember", choices) == "always"
    assert fn("3", choices) == "cancel"
    assert fn("nevermind", choices) == "cancel"
    assert fn("n", choices) == "cancel"
    assert fn("banana", choices) is None
    assert fn("4", choices) is None
    # non-alias raw value that IS an allowed choice value passes through
    custom = [("proceed", "Proceed", ""), ("cancel", "Cancel", "")]
    assert fn("proceed", custom) == "proceed"
    assert fn("PROCEED", custom) == "proceed"
    # alias for a value not offered by these choices is rejected
    assert fn("yes", custom) is None


def test_slash_confirm_display_fragments():
    """Panel rendering: border, title, choices, selected marker, hint; [] when idle."""
    import cli

    self_ = _make_modal_self()
    fn = _bound(cli.HermesCLI._get_slash_confirm_display_fragments, self_)

    assert fn() == []

    self_._slash_confirm_state = {
        "title": "Confirm reset",
        "detail": "This clears the session.\nSecond line.",
        "choices": [
            ("once", "Once", "run once"),
            ("always", "Always", "remember"),
            ("cancel", "Cancel", "stop"),
        ],
        "selected": 1,
        "response_queue": object(),
    }
    lines = fn()
    assert lines, "expected rendered panel lines"
    styles = [style for style, _text in lines]
    assert "class:approval-border" in styles
    assert lines[0][1].startswith("╭")
    assert lines[-1][1].startswith("╰")
    joined = "".join(text for _style, text in lines)
    assert "Confirm reset" in joined
    assert "[1] Once" in joined and "[2] Always" in joined and "[3] Cancel" in joined
    assert "❯ [2]" in joined  # selected marker on index 1 (0-based)
    assert "Type 1/2/3 or use ↑/↓ then Enter. ESC/Ctrl+C cancels." in joined
    assert "Second line." in joined


@pytest.mark.parametrize("member", _MEMBERS)
def test_members_defined_on_mixin(member):
    """Every member is present on the mixin class itself."""
    assert callable(getattr(mcm.CLIModalConfirmMixin, member)), member
