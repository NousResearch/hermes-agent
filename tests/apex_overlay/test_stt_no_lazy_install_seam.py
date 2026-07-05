"""Seam-test + behavior test for apex_overlay.stt_no_lazy_install (hc-401 SEAM B).

Pins ``tools.transcription_tools._try_lazy_install_stt`` so an upstream rename
turns a silently-disarmed guard (a managed container OOM-ing on faster-whisper
lazy-install) into a loud CI failure.

Behavior proven:
* with ``HERMES_MANAGED_RUNTIME`` set truthy, the wrapped function returns False
  WITHOUT calling upstream (no install attempt);
* without it, the wrapper delegates to the original implementation unchanged
  (desktop keeps upstream lazy-install behavior).

Run via ``scripts/run_tests_parallel.py`` (per-file fresh interpreter).
"""

from __future__ import annotations

import inspect

from apex_overlay import stt_no_lazy_install


# ---------------------------------------------------------------------------
# Seam assertion — pin the patched symbol
# ---------------------------------------------------------------------------

def test_seam_target_try_lazy_install_stt_exists():
    import tools.transcription_tools as tt

    fn = getattr(tt, stt_no_lazy_install._TARGET_FN, None)
    assert fn is not None, (
        "tools.transcription_tools._try_lazy_install_stt is gone — the managed-"
        "runtime STT guard can no longer neuter the lazy-install. Update "
        "apex_overlay.stt_no_lazy_install._TARGET_FN and the wrapper."
    )
    # Upstream is a zero-arg function; the wrapper forwards (*args, **kwargs) so a
    # future added kwarg wouldn't break it, but the current contract is zero-arg.
    assert callable(fn)
    assert len(inspect.signature(fn).parameters) == 0


def test_apply_wraps_and_is_idempotent():
    import tools.transcription_tools as tt

    stt_no_lazy_install._APPLIED = False
    assert stt_no_lazy_install.apply() is True
    assert getattr(tt._try_lazy_install_stt, stt_no_lazy_install._MARK, False)

    # Idempotent: second apply is a no-op (no double-wrap).
    ref = tt._try_lazy_install_stt
    assert stt_no_lazy_install.apply() is True
    assert tt._try_lazy_install_stt is ref


# ---------------------------------------------------------------------------
# Behavior — managed → False without calling upstream; else delegate
# ---------------------------------------------------------------------------

def test_managed_runtime_returns_false_without_calling_upstream(monkeypatch):
    """HERMES_MANAGED_RUNTIME=1 → wrapper returns False and never calls upstream."""
    import tools.transcription_tools as tt

    calls = {"n": 0}

    def _orig():
        calls["n"] += 1
        return True  # pretend the install would succeed

    wrapped = stt_no_lazy_install._wrap_try_lazy_install_stt(_orig)

    monkeypatch.setenv("HERMES_MANAGED_RUNTIME", "1")
    assert wrapped() is False
    assert calls["n"] == 0, "managed runtime must NOT reach upstream lazy-install"

    # other truthy spellings
    for val in ("true", "YES", "on"):
        calls["n"] = 0
        monkeypatch.setenv("HERMES_MANAGED_RUNTIME", val)
        assert wrapped() is False
        assert calls["n"] == 0


def test_non_managed_delegates_to_upstream(monkeypatch):
    """Without the managed flag, the wrapper delegates to upstream unchanged."""
    calls = {"n": 0}

    def _orig():
        calls["n"] += 1
        return "delegated-sentinel"

    wrapped = stt_no_lazy_install._wrap_try_lazy_install_stt(_orig)

    monkeypatch.delenv("HERMES_MANAGED_RUNTIME", raising=False)
    assert wrapped() == "delegated-sentinel"
    assert calls["n"] == 1

    # explicit falsey value also delegates
    monkeypatch.setenv("HERMES_MANAGED_RUNTIME", "0")
    calls["n"] = 0
    assert wrapped() == "delegated-sentinel"
    assert calls["n"] == 1


def test_is_managed_runtime_helper(monkeypatch):
    monkeypatch.setenv("HERMES_MANAGED_RUNTIME", "1")
    assert stt_no_lazy_install._is_managed_runtime() is True
    monkeypatch.setenv("HERMES_MANAGED_RUNTIME", "off")
    assert stt_no_lazy_install._is_managed_runtime() is False
    monkeypatch.delenv("HERMES_MANAGED_RUNTIME", raising=False)
    assert stt_no_lazy_install._is_managed_runtime() is False


def test_end_to_end_installed_on_module(monkeypatch):
    """After apply(), the module-level function honors the managed gate."""
    import tools.transcription_tools as tt

    stt_no_lazy_install._APPLIED = False
    stt_no_lazy_install.apply()

    monkeypatch.setenv("HERMES_MANAGED_RUNTIME", "1")
    # Managed → False. We assert it does not import/attempt lazy_deps by patching
    # ensure to blow up if reached.
    import tools.lazy_deps as lazy_deps

    def _boom(*a, **k):
        raise AssertionError("lazy_deps.ensure must not be called on managed runtime")

    monkeypatch.setattr(lazy_deps, "ensure", _boom)
    assert tt._try_lazy_install_stt() is False


def test_plugin_register_applies_seam():
    from tests.apex_overlay.conftest import run_plugin_register_with_stubbed_seams

    called = run_plugin_register_with_stubbed_seams("_stt_plugin_under_test")
    assert "stt_no_lazy_install" in called, (
        "plugin.register() must call stt_no_lazy_install.apply()"
    )
