# content_engine/tests/test_harness_collector.py
import activity_collector as ac


def test_harness_changes_shape(monkeypatch):
    fake_log = ("abc123\x1ftune model routing: cheap-first fallback\x1f2026-06-14\n"
                "def456\x1fadd governance fail-closed guard\x1f2026-06-13\n")
    monkeypatch.setattr(ac, "_git_log", lambda repo, n=15: fake_log)
    monkeypatch.setattr(ac, "_is_used", lambda s, i: False)
    sigs = ac.collect_harness_changes({"used": []})
    assert sigs and sigs[0]["signal_type"] == "harness_change"
    assert "routing" in sigs[0]["summary"].lower()
    assert sigs[0]["signal_id"].startswith("harness:")
