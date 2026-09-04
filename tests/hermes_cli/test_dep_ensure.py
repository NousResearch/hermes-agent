"""ensure_dependency routes through pm: availability checks stay local,
installs go to pm.ensure, and pm's lazy-install policy owns refusal."""

from unittest.mock import patch

import pytest
from tools import browser_tool_install as bt_install


def test_unknown_dep_refused():
    from hermes_cli.dep_ensure import ensure_dependency

    assert ensure_dependency("not-a-dep") is False


def test_available_dep_short_circuits(monkeypatch):
    from hermes_cli import dep_ensure

    monkeypatch.setitem(
        dep_ensure._DEPS, "node", (lambda: True, ("node",))
    )
    called = []
    with patch("pm.ensure", side_effect=lambda *a, **k: called.append(a)):
        assert dep_ensure.ensure_dependency("node") is True
    assert called == []






def test_has_npx_agent_browser_true_when_npx_resolves():
    """agent-browser resolves lazily via npx on the default install (#43564)
    — _has_npx_agent_browser mirrors the runtime cascade so the "browser" dep
    check doesn't wrongly report it missing."""
    from hermes_cli.dep_ensure import _has_npx_agent_browser

    with patch.object(bt_install, "_find_agent_browser", return_value="npx agent-browser"), \
         patch.object(bt_install, "_requires_real_termux_browser_install", return_value=False):
        assert _has_npx_agent_browser() is True


def test_has_npx_agent_browser_false_on_termux_local_bare_npx():
    from hermes_cli.dep_ensure import _has_npx_agent_browser

    with patch.object(bt_install, "_find_agent_browser", return_value="npx agent-browser"), \
         patch.object(bt_install, "_requires_real_termux_browser_install", return_value=True):
        assert _has_npx_agent_browser() is False


def test_has_npx_agent_browser_false_when_nothing_resolves():
    from hermes_cli.dep_ensure import _has_npx_agent_browser

    def _raise(**_kw):
        raise FileNotFoundError("agent-browser CLI not found")

    with patch.object(bt_install, "_find_agent_browser", _raise):
        assert _has_npx_agent_browser() is False


def test_find_agent_browser_lazy_install_cycle_terminates(monkeypatch):
    """tools.browser_tool_install._find_agent_browser's "nothing found" branch calls
    ensure_dependency("browser"), whose "browser" check now includes
    _has_npx_agent_browser() -> _find_agent_browser(validate=False) again.
    That nested call must NOT be able to trigger another ensure_dependency
    call (only validate=True does that) — verifying the cycle is bounded to
    one extra rescan, not unbounded recursion, using the real functions on
    both sides rather than mocking the cycle away."""
    import shutil
    import tools.browser_tool as bt
    from hermes_cli import dep_ensure

    monkeypatch.setattr(bt, "_cached_agent_browser", None)
    monkeypatch.setattr(bt, "_agent_browser_resolved", False)
    monkeypatch.setattr(shutil, "which", lambda *a, **k: None)
    monkeypatch.setattr("tools.browser_tool_install._resolve_npx_bin", lambda: None)
    monkeypatch.setattr(dep_ensure, "_has_system_browser", lambda: False)
    monkeypatch.setattr(dep_ensure, "_has_hermes_agent_browser", lambda: False)
    monkeypatch.setattr(dep_ensure, "_find_install_script", lambda *a, **k: (None, None))

    real_find_agent_browser = bt_install._find_agent_browser
    validate_calls = []

    def counting_find_agent_browser(*, validate=True):
        validate_calls.append(validate)
        return real_find_agent_browser(validate=validate)

    monkeypatch.setattr(bt_install, "_find_agent_browser", counting_find_agent_browser)

    with pytest.raises(FileNotFoundError):
        bt_install._find_agent_browser(validate=True)

    # One outer validate=True call, plus exactly one bounded nested
    # validate=False rescan from _has_npx_agent_browser inside
    # ensure_dependency's "browser" check — not unbounded recursion, and not
    # a second ensure_dependency("browser") call (which would show up as a
    # second `True` in this list).
    assert validate_calls == [True, False]


def test_missing_dep_installs_through_pm(monkeypatch):
    from hermes_cli import dep_ensure

    state = {"installed": False}
    monkeypatch.setitem(
        dep_ensure._DEPS, "node", (lambda: state["installed"], ("node",))
    )

    def fake_ensure(name, **kw):
        assert name == "node"
        state["installed"] = True

    with patch("pm.ensure", side_effect=fake_ensure):
        assert dep_ensure.ensure_dependency("node") is True


def test_pm_refusal_reports_and_returns_false(monkeypatch, capsys):
    from hermes_cli import dep_ensure

    monkeypatch.setitem(
        dep_ensure._DEPS, "node", (lambda: False, ("node",))
    )
    import pm as pm_mod

    def refuse(name, **kw):
        raise pm_mod.InstallError(name, "lazy installs are disabled", "run `hermes pm install`")

    with patch("pm.ensure", side_effect=refuse):
        assert dep_ensure.ensure_dependency("node", interactive=True) is False
    out = capsys.readouterr().out
    assert "hermes pm install" in out


def test_browser_check_consults_pm(monkeypatch):
    from hermes_cli import dep_ensure

    monkeypatch.setattr("shutil.which", lambda *a, **k: None)
    with patch("pm.is_installed", return_value=True):
        assert dep_ensure._browser_available() is True
    with patch("pm.is_installed", return_value=False):
        assert dep_ensure._browser_available() is False
