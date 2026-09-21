# content_engine/tests/test_context_enrich.py
import context_enrich as ce


def test_enrich_harness_uses_git_show(monkeypatch):
    monkeypatch.setattr(ce, "_git_show", lambda repo, sha: "tune routing\n+ cheap-first\n- old default")
    blob = ce.enrich({"signal_type":"harness_change","repo":"KenseiAgent","sha":"abc","summary":"tune routing"})
    assert "cheap-first" in blob
    assert len(blob) <= ce.MAX_BLOB

def test_enrich_unknown_type_returns_summary():
    blob = ce.enrich({"signal_type":"mystery","summary":"hello world"})
    assert "hello world" in blob
