from __future__ import annotations

import json

import pytest
import yaml

from hermes_wiki.caretaker import (
    caretaker_report_to_dict,
    main as caretaker_main,
    render_caretaker_report,
    run_caretaker,
)
from hermes_wiki.config import WikiConfig
from hermes_wiki.frontmatter import write_page
from hermes_wiki.proposals import MemoryProposal, queue_proposal


class FakeSearchResult:
    def __init__(self, page_path: str):
        self.page_path = page_path


class FakeSearcher:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query
        self.calls = []

    def search(self, query, limit=5):
        self.calls.append((query, limit))
        return self.results_by_query.get(query, [])


def _write_page(config: WikiConfig, rel_path: str, fm: dict, body: str):
    path = config.wiki_path / rel_path
    write_page(path, fm, body)
    return path


def test_caretaker_report_is_read_only_and_classifies_pending_proposal(tmp_path):
    config = WikiConfig(wiki_path=tmp_path / "wiki", wiki_name="test")
    queue_proposal(
        config,
        MemoryProposal(
            title="Pending memory",
            rationale="Hermes should self-review this later.",
            proposed_changes=["Store durable agent-owned memory."],
            source_refs=["test-source"],
        ),
        write=True,
    )
    proposal_path = config.wiki_path / "proposals" / "pending-memory.md"
    before = proposal_path.read_text(encoding="utf-8")

    report = run_caretaker(config)

    assert proposal_path.read_text(encoding="utf-8") == before
    assert report.maintenance.pending_proposals == 1
    assert report.has_blockers is False
    assert any(action.kind == "review_pending_proposal" and action.autonomous for action in report.actions)


def test_caretaker_runs_retrieval_eval_when_cases_are_supplied(tmp_path):
    config = WikiConfig(wiki_path=tmp_path / "wiki", wiki_name="test")
    cases_path = tmp_path / "retrieval.yaml"
    cases_path.write_text(
        yaml.safe_dump([
            {
                "query": "How should Hermes use memory?",
                "expected_pages": ["concepts/hermes-memory.md"],
                "top_k": 2,
            }
        ]),
        encoding="utf-8",
    )
    searcher = FakeSearcher({"How should Hermes use memory?": [FakeSearchResult("concepts/hermes-memory.md")]})

    report = run_caretaker(config, eval_cases_path=cases_path, searcher=searcher)

    assert report.retrieval_eval is not None
    assert report.retrieval_eval.passed is True
    assert searcher.calls == [("How should Hermes use memory?", 2)]
    assert report.has_blockers is False


def test_caretaker_marks_eval_failure_as_blocker(tmp_path):
    config = WikiConfig(wiki_path=tmp_path / "wiki", wiki_name="test")
    cases_path = tmp_path / "retrieval.yaml"
    cases_path.write_text(
        yaml.safe_dump([
            {
                "query": "What is the user's preferred name?",
                "expected_pages": ["entities/example-user.md"],
            }
        ]),
        encoding="utf-8",
    )
    searcher = FakeSearcher({"What is the user's preferred name?": [FakeSearchResult("entities/hermes.md")]})

    report = run_caretaker(config, eval_cases_path=cases_path, searcher=searcher)

    assert report.has_blockers is True
    assert any(action.kind == "fix_retrieval_regression" and action.severity == "error" for action in report.actions)


def test_caretaker_report_to_dict_is_json_serializable(tmp_path):
    config = WikiConfig(wiki_path=tmp_path / "wiki", wiki_name="test")

    payload = caretaker_report_to_dict(run_caretaker(config))

    assert json.loads(json.dumps(payload))["maintenance"]["total_pages"] == 0
    assert "actions" in payload


def test_render_caretaker_report_is_agent_native(tmp_path):
    config = WikiConfig(wiki_path=tmp_path / "wiki", wiki_name="test")

    text = render_caretaker_report(run_caretaker(config))

    assert "# LLM Wiki Caretaker Report" in text
    assert "## Hermes actions" in text
    assert "Human UI" not in text


def test_caretaker_cli_is_quiet_when_healthy(tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    wiki_path = tmp_path / "wiki"
    config_path.write_text(f"wiki:\n  path: {wiki_path}\n  name: test\n", encoding="utf-8")

    code = caretaker_main(["--config", str(config_path), "--quiet"])

    assert code == 0
    assert capsys.readouterr().out == ""
    assert not wiki_path.exists()


def test_caretaker_cli_writes_report_only_to_reports_namespace(tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    wiki_path = tmp_path / "wiki"
    config_path.write_text(f"wiki:\n  path: {wiki_path}\n  name: test\n", encoding="utf-8")

    code = caretaker_main(["--config", str(config_path), "--write-report", "reports/caretaker.md"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload == {"path": str(wiki_path / "reports" / "caretaker.md"), "written": True}
    assert (wiki_path / "reports" / "caretaker.md").exists()


def test_caretaker_cli_rejects_write_report_outside_reports(tmp_path):
    config_path = tmp_path / "config.yaml"
    wiki_path = tmp_path / "wiki"
    config_path.write_text(f"wiki:\n  path: {wiki_path}\n  name: test\n", encoding="utf-8")
    canonical = _write_page(
        WikiConfig(wiki_path=wiki_path, wiki_name="test"),
        "concepts/existing.md",
        {"title": "Existing", "type": "concept", "sources": ["raw/articles/source.md"]},
        "Original.",
    )
    before = canonical.read_text(encoding="utf-8")

    with pytest.raises(SystemExit):
        caretaker_main(["--config", str(config_path), "--write-report", "concepts/existing.md"])

    assert canonical.read_text(encoding="utf-8") == before
