"""Regression coverage for the source-label audit.

Run: cd content_engine && PYTHONPATH=. ../.venv/bin/python3 -m pytest \
    tests/test_audit_source_labels.py -q
"""

import importlib.util
from pathlib import Path
_spec = importlib.util.spec_from_file_location('ce_source_audit', Path(__file__).resolve().parents[1] / 'tools/audit_source_labels.py')
assert _spec is not None and _spec.loader is not None
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)
audit_all, audit_post = _audit.audit_all, _audit.audit_post


def test_research_paper_with_reference_is_ok(tmp_path):
    post = tmp_path / "paper.mdx"
    post.write_text(
        "---\nsource: research-paper\ntitle: Paper\n---\n\n"
        "See https://arxiv.org/abs/2303.08774.\n"
    )

    assert audit_post(post)["status"] == "ok"


def test_research_paper_without_reference_is_mismatch(tmp_path):
    post = tmp_path / "paper.mdx"
    post.write_text("---\nsource: research-paper\n---\n\nGeneral AI commentary.\n")

    assert audit_post(post)["status"] == "mismatch"


def test_manual_post_with_paper_reference_is_upgrade_candidate(tmp_path):
    post = tmp_path / "manual.mdx"
    post.write_text("---\nsource: manual\n---\n\nSee arxiv.org/abs/2303.08774.\n")

    assert audit_post(post)["status"] == "upgrade"


def test_audit_all_reports_summary_and_posts(tmp_path):
    for index in range(3):
        (tmp_path / f"post-{index}.mdx").write_text("---\nsource: manual\n---\n\nBody.\n")

    report = audit_all(tmp_path)

    assert report["total"] == 3
    assert report["ok"] == 3
    assert len(report["posts"]) == 3
