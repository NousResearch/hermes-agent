"""The install hint for an unavailable local_embedded side runtime.

The embedded stack lives in the isolated side env (embedded_runtime); the fix
is always 'hermes memory setup' (Local Embedded) — the main Hermes environment
is never a pip target for it.
"""

import plugins.memory.hindsight as hs
from plugins.memory.hindsight import HindsightMemoryProvider, _local_runtime_hint


def test_hint_for_missing_runtime_names_the_isolated_install():
    hint = _local_runtime_hint("No module named 'hindsight'")
    assert "hermes memory setup" in hint
    assert "hindsight-embed==0.9.2" in hint
    assert "hindsight-api-slim[all]==0.9.2" in hint


def test_hint_includes_probe_reason():
    hint = _local_runtime_hint("Illegal instruction (NumPy SIMD)")
    assert "NumPy SIMD" in hint  # truthful: reinstall may not fix a CPU limit
    assert "hermes memory setup" in hint


def test_hint_without_reason_still_actionable():
    assert "hermes memory setup" in _local_runtime_hint(None)


def test_unavailable_reason_surfaces_hint_for_local_embedded(monkeypatch):
    monkeypatch.setattr(hs, "_load_config", lambda: {"mode": "local_embedded"})
    monkeypatch.setattr(hs, "_check_local_runtime", lambda: (False, "No module named 'hindsight'"))
    reason = HindsightMemoryProvider().unavailable_reason()
    assert "hermes memory setup" in reason
    assert reason == reason.strip()  # no leading/trailing whitespace


def test_unavailable_reason_empty_for_cloud(monkeypatch):
    monkeypatch.setattr(hs, "_load_config", lambda: {"mode": "cloud"})
    # Should not even probe the runtime for a cloud provider.
    monkeypatch.setattr(hs, "_check_local_runtime", lambda: (_ for _ in ()).throw(AssertionError("probed")))
    assert HindsightMemoryProvider().unavailable_reason() == ""


def test_unavailable_reason_empty_when_runtime_present(monkeypatch):
    monkeypatch.setattr(hs, "_load_config", lambda: {"mode": "local_embedded"})
    monkeypatch.setattr(hs, "_check_local_runtime", lambda: (True, None))
    assert HindsightMemoryProvider().unavailable_reason() == ""
