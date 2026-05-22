import json

from hermes_wiki.config import WikiConfig
from hermes_wiki.frontmatter import write_page
from hermes_wiki.maintenance import generate_maintenance_report, maintenance_report_to_dict, render_maintenance_report, main as maintenance_main
from hermes_wiki.proposals import MemoryProposal, queue_proposal


def _write_page(config, rel_path, fm, body):
    path = config.wiki_path / rel_path
    write_page(path, fm, body)
    return path


def test_generate_maintenance_report_detects_broken_links_and_orphans(tmp_path):
    config = WikiConfig(wiki_path=tmp_path / "wiki", wiki_name="test")
    _write_page(
        config,
        "concepts/a.md",
        {"title": "A", "type": "concept", "created": "2026-01-01", "updated": "2026-01-01"},
        "Links to [[Missing Page]].",
    )
    _write_page(
        config,
        "concepts/orphan.md",
        {"title": "Orphan", "type": "concept", "created": "2026-01-01", "updated": "2026-01-01"},
        "No inbound links.",
    )

    report = generate_maintenance_report(config)

    categories = {issue.category for issue in report.issues}
    assert "broken_link" in categories
    assert "orphan_page" in categories
    assert report.total_pages == 2
    assert report.broken_links == 1
    assert report.orphan_pages >= 1


def test_generate_maintenance_report_detects_pending_proposals(tmp_path):
    config = WikiConfig(wiki_path=tmp_path / "wiki", wiki_name="test")
    queue_proposal(
        config,
        MemoryProposal(
            title="Pending proposal",
            rationale="Needs review.",
            proposed_changes=["Promote after review."],
            source_refs=["test-source"],
        ),
        write=True,
    )

    report = generate_maintenance_report(config)

    assert report.pending_proposals == 1
    assert any(issue.category == "pending_proposal" for issue in report.issues)


def test_generate_maintenance_report_detects_source_coverage_gap(tmp_path):
    config = WikiConfig(wiki_path=tmp_path / "wiki", wiki_name="test")
    _write_page(
        config,
        "concepts/no-source.md",
        {"title": "No Source", "type": "concept", "created": "2026-01-01", "updated": "2026-01-01"},
        "This page has no provenance marker.",
    )

    report = generate_maintenance_report(config)

    assert report.pages_without_sources == 1
    assert any(issue.category == "missing_source_coverage" for issue in report.issues)


def test_maintenance_report_to_dict_is_json_serializable(tmp_path):
    config = WikiConfig(wiki_path=tmp_path / "wiki", wiki_name="test")
    _write_page(
        config,
        "concepts/a.md",
        {"title": "A", "type": "concept", "created": "2026-01-01", "updated": "2026-01-01"},
        "Body.",
    )

    payload = maintenance_report_to_dict(generate_maintenance_report(config))

    assert json.loads(json.dumps(payload))["total_pages"] == 1
    assert "issues" in payload


def test_render_maintenance_report_contains_summary(tmp_path):
    config = WikiConfig(wiki_path=tmp_path / "wiki", wiki_name="test")
    _write_page(
        config,
        "concepts/a.md",
        {"title": "A", "type": "concept", "created": "2026-01-01", "updated": "2026-01-01"},
        "Body.",
    )

    text = render_maintenance_report(generate_maintenance_report(config))

    assert "# LLM Wiki Maintenance Report" in text
    assert "Pages:" in text
    assert "Issues:" in text


def test_maintenance_cli_is_read_only_by_default(tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    wiki_path = tmp_path / "wiki"
    config_path.write_text(f"wiki:\n  path: {wiki_path}\n  name: test\n", encoding="utf-8")

    code = maintenance_main(["--config", str(config_path)])

    assert code == 0
    assert "# LLM Wiki Maintenance Report" in capsys.readouterr().out
    assert not wiki_path.exists()


def test_maintenance_cli_write_report_requires_explicit_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "ambient-home"))

    try:
        maintenance_main(["--write-report", "reports/maintenance.md"])
    except SystemExit:
        pass

    assert not (tmp_path / "ambient-home").exists()


def test_maintenance_cli_write_report_allows_reports_namespace_only(tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    wiki_path = tmp_path / "wiki"
    config_path.write_text(f"wiki:\n  path: {wiki_path}\n  name: test\n", encoding="utf-8")

    code = maintenance_main(["--config", str(config_path), "--write-report", "reports/maintenance.md"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload == {"path": str(wiki_path / "reports" / "maintenance.md"), "written": True}
    assert (wiki_path / "reports" / "maintenance.md").exists()


def test_maintenance_cli_write_report_rejects_canonical_paths(tmp_path):
    config_path = tmp_path / "config.yaml"
    wiki_path = tmp_path / "wiki"
    config_path.write_text(f"wiki:\n  path: {wiki_path}\n  name: test\n", encoding="utf-8")
    canonical = _write_page(
        WikiConfig(wiki_path=wiki_path, wiki_name="test"),
        "concepts/existing.md",
        {"title": "Existing", "type": "concept", "created": "2026-01-01", "updated": "2026-01-01"},
        "Original canonical content.",
    )
    before = canonical.read_text(encoding="utf-8")

    try:
        maintenance_main(["--config", str(config_path), "--write-report", "concepts/existing.md"])
    except SystemExit:
        pass

    assert canonical.read_text(encoding="utf-8") == before


def test_maintenance_cli_write_report_rejects_traversal(tmp_path):
    config_path = tmp_path / "config.yaml"
    wiki_path = tmp_path / "wiki"
    config_path.write_text(f"wiki:\n  path: {wiki_path}\n  name: test\n", encoding="utf-8")

    try:
        maintenance_main(["--config", str(config_path), "--write-report", "../outside.md"])
    except SystemExit:
        pass

    assert not (tmp_path / "outside.md").exists()
