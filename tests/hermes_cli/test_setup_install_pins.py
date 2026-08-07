"""Setup-time optional installs must resolve pinned/bounded versions.

`hermes setup` and its post-setup hooks install optional packages on a user's
machine. An unpinned spec lets whatever the index serves at that moment become
executing code, so every setup install either

  * inherits exact pins from a ``tools/lazy_deps.LAZY_DEPS`` group — the single
    source of truth, mirroring the pyproject extras — or
  * carries an explicit major bound, for setup-only packages that have no
    LAZY_DEPS group and are installed with ``-U`` by design.

These tests drive the real install paths with ``_pip_install`` stubbed and
assert on the specs those paths actually pass to pip.
"""

import subprocess
import sys

import pytest

import hermes_cli.setup as setup_mod
import hermes_cli.tools_config as tools_config
from hermes_cli.tools_config import (
    _SETUP_INSTALL_BOUNDS,
    _SETUP_INSTALL_FALLBACKS,
    _bounded_spec,
    _pinned_specs,
)
from tools.lazy_deps import LAZY_DEPS


@pytest.fixture
def pip_calls(monkeypatch):
    """Record every spec list handed to ``_pip_install``.

    Patches the module object that is live in ``sys.modules`` rather than the
    one bound at import time: the install sites import ``_pip_install`` inside
    the function body, so they resolve through ``sys.modules`` at call time,
    and another test in the session may have replaced that entry.

    Also blocks ``subprocess.run`` inside tools_config, so a patch that fails
    to take fails the test loudly instead of shelling out to a real pip.
    """
    calls = []

    def _fake(args, **kwargs):
        calls.append(list(args))
        return subprocess.CompletedProcess(list(args), returncode=0, stdout="", stderr="")

    live = sys.modules["hermes_cli.tools_config"]
    monkeypatch.setattr(live, "_pip_install", _fake)
    if live is not tools_config:  # pragma: no cover - only under module reload
        monkeypatch.setattr(tools_config, "_pip_install", _fake)

    def _no_real_installs(*args, **kwargs):
        raise AssertionError(
            f"install path escaped the _pip_install stub and tried to run: {args!r}"
        )

    monkeypatch.setattr(live.subprocess, "run", _no_real_installs)
    return calls


def _specs_only(args):
    """Drop pip flags, leaving the package specs."""
    return [a for a in args if not a.startswith("-")]


# ── The pin tables ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("feature", sorted(_SETUP_INSTALL_FALLBACKS))
def test_fallback_mirrors_lazy_deps(feature):
    """The stripped-install fallback must not drift from the real table."""
    assert _SETUP_INSTALL_FALLBACKS[feature] == LAZY_DEPS[feature]


@pytest.mark.parametrize("feature", sorted(_SETUP_INSTALL_FALLBACKS))
def test_lazy_deps_groups_are_exactly_pinned(feature):
    for spec in LAZY_DEPS[feature]:
        assert "==" in spec, f"{feature} spec {spec!r} is not pinned"


@pytest.mark.parametrize("label", sorted(_SETUP_INSTALL_BOUNDS))
def test_setup_only_specs_are_bounded(label):
    """No LAZY_DEPS group to inherit from → must at least bound the major."""
    spec = _SETUP_INSTALL_BOUNDS[label]
    assert "<" in spec, f"{label} spec {spec!r} has no upper bound"


def test_pinned_specs_reads_lazy_deps():
    assert _pinned_specs("terminal.modal") == list(LAZY_DEPS["terminal.modal"])


def test_pinned_specs_falls_back_when_lazy_deps_unimportable(monkeypatch):
    """Stripped installs have no tools.lazy_deps — the mirror must hold."""
    import builtins

    real_import = builtins.__import__

    def _blow_up(name, *args, **kwargs):
        if name == "tools.lazy_deps":
            raise ImportError("simulated stripped install")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blow_up)
    assert _pinned_specs("terminal.modal") == list(LAZY_DEPS["terminal.modal"])


def test_unknown_feature_raises():
    """A typo must fail loudly, not silently install nothing."""
    with pytest.raises(KeyError):
        _pinned_specs("no.such.feature")
    with pytest.raises(KeyError):
        _bounded_spec("no-such-package")


# ── The install paths actually pass those specs ───────────────────────────────

def test_neutts_install_passes_bounded_spec(monkeypatch, pip_calls):
    monkeypatch.setattr(setup_mod, "_check_espeak_ng", lambda: True)

    assert setup_mod._install_neutts_deps() is True

    assert len(pip_calls) == 1
    specs = _specs_only(pip_calls[0])
    assert specs == [_bounded_spec("neutts")]
    assert "neutts[all]" not in specs, "bare unbounded spec leaked through"


def test_kittentts_install_bounds_its_audio_dep(monkeypatch, pip_calls):
    assert setup_mod._install_kittentts_deps() is True

    specs = _specs_only(pip_calls[0])
    # The wheel URL is itself version-locked; soundfile is the loose one.
    assert any(spec.endswith(".whl") and "0.8.1" in spec for spec in specs)
    assert _bounded_spec("soundfile") in specs
    assert "soundfile" not in specs


@pytest.mark.parametrize(
    "hook, expected",
    [
        ("piper", "piper"),
        ("ddgs", "ddgs"),
        ("langfuse", "langfuse"),
    ],
)
def test_post_setup_hook_installs_bounded_spec(monkeypatch, pip_calls, hook, expected):
    """The post-setup hooks install bounded specs, not floating names."""
    import builtins

    real_import = builtins.__import__

    # Force the ImportError branch — the hook only installs what's missing.
    def _missing(name, *args, **kwargs):
        if name in {"piper", "ddgs", "langfuse"}:
            raise ImportError(f"simulated missing {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _missing)

    try:
        tools_config._run_post_setup(hook)
    except Exception:
        # Hooks do more than install (langfuse also toggles a plugin); the
        # install call is what this test is about.
        pass

    assert pip_calls, f"{hook} hook made no install call"
    specs = _specs_only(pip_calls[0])
    assert _bounded_spec(expected) in specs, specs


def test_faster_whisper_hook_uses_lazy_deps_pins(monkeypatch, pip_calls):
    """faster-whisper HAS a LAZY_DEPS group — the hook must use it verbatim."""
    import builtins

    real_import = builtins.__import__

    def _missing(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("simulated missing faster_whisper")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _missing)

    tools_config._run_post_setup("faster_whisper")

    assert pip_calls, "faster_whisper hook made no install call"
    assert _specs_only(pip_calls[0]) == list(LAZY_DEPS["stt.faster_whisper"])


# ── The fallback ladder is retained, not replaced ─────────────────────────────

def test_pip_install_bootstraps_pip_via_ensurepip(monkeypatch):
    """With uv absent and pip missing, _pip_install must run ensurepip.

    The uv → pip → ensurepip ladder is why setup works on pip-less venvs
    (`uv venv`, Ubuntu's `python -m venv`). Any replacement installer that
    skipped it would regress those installs.
    """
    monkeypatch.setattr(tools_config.shutil, "which", lambda name: None)

    seen = []

    def _fake_run(cmd, **kwargs):
        seen.append(list(cmd))
        if "--version" in cmd:  # the pip probe
            return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="no pip")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(tools_config.subprocess, "run", _fake_run)

    result = tools_config._pip_install(["modal==1.3.4"])

    assert result.returncode == 0
    assert any("ensurepip" in c for cmd in seen for c in cmd), seen
    # And the install itself still ran, with the spec intact.
    assert any("modal==1.3.4" in cmd for cmd in seen), seen


def test_pip_install_prefers_uv_when_available(monkeypatch):
    monkeypatch.setattr(
        tools_config.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None
    )

    seen = []

    def _fake_run(cmd, **kwargs):
        seen.append(list(cmd))
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(tools_config.subprocess, "run", _fake_run)

    tools_config._pip_install(["modal==1.3.4"])

    assert seen[0][:3] == ["/usr/bin/uv", "pip", "install"], seen[0]
    assert "modal==1.3.4" in seen[0]
