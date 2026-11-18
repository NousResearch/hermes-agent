"""UX proposals cannot confer release authority; every publication rechecks it."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts.ux_blockers.github import GitHub
from scripts.ux_blockers.policy import (
    BLOCKER, BOT, CONFIRMED, FRICTION, RECEIPT, REVIEW,
    RECOVERIES, SURFACES, classify, decision, parse_command,
)
from scripts.ux_blockers.service import gate_report, sync_issue

SHA = "a" * 40
OTHER = "b" * 40
URL = "https://github.com/example/project/actions/runs/42"


def report(surface="Desktop", recovery=RECOVERIES[1], progress="Blocked or falsely reports success"):
    return (f"### UX surface\n\n{surface}\n\n### UX journey\n\nUpdate / recovery\n\n"
            f"### UX progress\n\n{progress}\n\n### UX recovery\n\n{recovery}\n\n"
            "### User Journey and Recovery (optional)\n\nUpdate stopped. Ordinary retry failed.")


def comment(number, body, *, login="maintainer", kind="User", edited=False):
    return {"id": number, "body": body, "html_url": f"https://github.com/example/project/issues/1#issuecomment-{number}",
            "user": {"login": login, "type": kind}, "created_at": "2026-11-18T00:00:00Z",
            "updated_at": "2026-11-19T00:00:00Z" if edited else "2026-11-18T00:00:00Z"}


def confirm(number=1, sha=SHA, **kwargs):
    return comment(number, f"/ux confirm {sha} {URL} Supported update strands the user", **kwargs)


def resolve(number=2, sha=SHA, disposition="recovered", **kwargs):
    return comment(number, f"/ux resolve {disposition} {sha} {URL} Relaunch resumes the existing session", **kwargs)


def trusted(login):
    return login == "maintainer"


class Ledger:
    prefix = "repos/example/project"
    repository = "example/project"
    trusted = staticmethod(trusted)

    def __init__(self, body=None):
        self.rows = {1: {"number": 1, "state": "open", "html_url": "https://github.com/example/project/issues/1",
                         "body": body or report(), "labels": []}}
        self.messages = {1: []}
        self.fixes = {1: []}
        self.mutations = []
        self.contained = False

    def issue(self, number):
        return self.rows[number]

    def comments(self, number):
        return list(self.messages[number])

    def links(self, number):
        return self.fixes[number]

    def indexed(self):
        return [row for row in self.rows.values() if CONFIRMED in {x["name"] for x in row["labels"]}]

    def contains(self, candidate, commit):
        assert candidate == SHA and commit == OTHER
        return self.contained

    def write(self, endpoint, method, **fields):
        self.mutations.append((endpoint, method, fields))
        if endpoint.endswith("/labels"):
            self.rows[1]["labels"].extend({"name": label} for label in fields["labels"])
        elif "/labels/" in endpoint:
            from urllib.parse import unquote
            name = unquote(endpoint.rsplit("/", 1)[1])
            self.rows[1]["labels"] = [row for row in self.rows[1]["labels"] if row["name"] != name]
        elif endpoint.endswith("/comments"):
            self.messages[1].append(comment(100 + len(self.messages[1]), fields["body"], login=BOT, kind="Bot"))
        else:
            number = int(endpoint.rsplit("/", 1)[1])
            next(row for row in self.messages[1] if row["id"] == number)["body"] = fields["body"]


@pytest.mark.parametrize("surface", SURFACES)
def test_supported_surfaces_use_recovery_boundary(surface):
    assert classify(report(surface)).priority == "P0"
    assert classify(report(surface, RECOVERIES[0])).priority == "P1"
    assert classify(report(surface, "Unknown")).priority == "unknown"
    assert classify(report(surface, RECOVERIES[1], "Works as expected")).priority == "unknown"


def test_missing_evidence_and_unsupported_journeys_remain_unknown():
    assert classify("### UX surface\nDesktop").priority == "unknown"
    assert classify(report().replace("Update / recovery", "Optional theme")).priority == "unknown"
    assert classify(report().replace("Update stopped. Ordinary retry failed.", "_No response_")).priority == "unknown"


@pytest.mark.parametrize("body", [
    f"> /ux confirm {SHA} {URL} reason", f"```\n/ux confirm {SHA} {URL} reason\n```",
    f"/ux confirm {SHA} http://example.test reason", f"/ux resolve {SHA} {URL} reason",
    f"/ux confirm fixed {SHA} {URL} reason", f"/ux confirm {SHA} https://user:pass@example.test reason",
    f"/ux confirm main {URL} reason", f"/ux confirm {SHA} {URL} \t",
])
def test_command_parser_rejects_quoted_incomplete_or_mutable_authority(body):
    assert parse_command(body) is None


def test_only_current_human_write_permission_can_confirm_and_clear():
    assert decision([confirm(login="reporter"), resolve()], SHA, trusted)["status"] == "blocked"
    assert decision([confirm(kind="Bot"), resolve()], SHA, trusted)["status"] == "blocked"
    assert decision([confirm(), resolve(login="reporter")], SHA, trusted)["status"] == "blocked"
    assert decision([confirm(), resolve()], SHA, trusted)["status"] == "cleared"


def test_dispositions_are_candidate_scoped_and_superseded_by_confirmation():
    assert decision([confirm(), resolve()], OTHER, trusted)["status"] == "blocked"
    assert decision([confirm(), resolve(), confirm(3)], SHA, trusted)["status"] == "blocked"
    assert decision([confirm(), resolve(edited=True)], SHA, trusted)["status"] == "blocked"
    assert decision([confirm(), resolve(), confirm(3, edited=True)], SHA, trusted)["status"] == "blocked"


def test_acknowledged_confirmation_survives_source_deletion_or_edit():
    source = confirm(3)
    receipt = comment(10, RECEIPT + "\n" + json.dumps({"id": 3, "sha256": hashlib.sha256(source["body"].encode()).hexdigest()}),
                      login=BOT, kind="Bot")
    assert decision([confirm(), resolve(), receipt], SHA, trusted)["status"] == "blocked"
    assert decision([confirm(), resolve(), {**source, "body": "withdrawn"}, receipt], SHA, trusted)["status"] == "blocked"
    assert decision([confirm(), resolve(), receipt, resolve(11)], SHA, trusted)["status"] == "cleared"


def test_reporter_markers_do_not_create_authority():
    spoof = comment(10, RECEIPT + '\n{"id":100,"sha256":"bad"}', login="reporter")
    assert decision([confirm(), resolve(), spoof], SHA, trusted)["status"] == "cleared"


def test_proposal_link_refresh_and_confirmation_are_idempotent():
    api = Ledger()
    api.rows[1]["labels"] = [{"name": "P2"}, {"name": "bug"}]
    api.fixes[1] = [{"number": 2, "url": "https://github.com/example/project/pull/2", "state": "OPEN"}]
    result = sync_issue(api, 1)
    assert result["priority"] == "P0" and not result["confirmed"]
    assert {x["name"] for x in api.rows[1]["labels"]} == {"P2", "bug", BLOCKER, REVIEW}
    assert "pull/2 (open)" in api.messages[1][0]["body"]
    assert gate_report(api, SHA)["status"] == "passed"
    count = len(api.mutations)
    sync_issue(api, 1)
    assert len(api.mutations) == count
    api.messages[1].append(confirm(200))
    sync_issue(api, 1)
    assert CONFIRMED in {row["name"] for row in api.rows[1]["labels"]}
    assert any(row["body"].startswith(RECEIPT) for row in api.messages[1])
    api.rows[1].update(state="closed", body="edited away")
    assert gate_report(api, SHA)["blocking_issues"] == [1]
    api.messages[1].append(resolve(300))
    assert gate_report(api, SHA)["status"] == "passed"
    assert gate_report(api, OTHER)["status"] == "blocked"


def test_p1_and_unstructured_reports_do_not_block_or_spam():
    api = Ledger(report(recovery=RECOVERIES[0]))
    assert sync_issue(api, 1)["priority"] == "P1"
    assert FRICTION in {row["name"] for row in api.rows[1]["labels"]}
    api = Ledger("A normal non-core bug")
    assert "skipped" in sync_issue(api, 1)
    assert api.mutations == []


def test_merged_fix_requires_candidate_containment_and_maintainer_recovery_evidence():
    api = Ledger()
    api.rows[1]["labels"] = [{"name": CONFIRMED}]
    api.messages[1] = [confirm()]
    api.fixes[1] = [{"number": 2, "state": "MERGED", "mergeCommit": {"oid": OTHER}}]
    api.contained = True
    assert gate_report(api, SHA)["status"] == "blocked"
    api.messages[1].append(resolve(disposition="fixed"))
    api.contained = False
    assert gate_report(api, SHA)["status"] == "blocked"
    api.contained = True
    assert gate_report(api, SHA)["issues"][0]["contained_fixes"] == [2]
    assert gate_report(api, SHA)["status"] == "passed"


def test_duplicates_require_confirmed_canonical_disposition_and_reject_cycles():
    api = Ledger()
    api.rows[1]["labels"] = [{"name": CONFIRMED}]
    api.rows[2] = {**api.rows[1], "number": 2}
    api.messages = {1: [confirm(), comment(2, "/ux duplicate #2")], 2: [confirm()]}
    api.fixes[2] = []
    assert gate_report(api, SHA)["status"] == "blocked"
    api.messages[2].append(resolve())
    assert gate_report(api, SHA)["status"] == "passed"
    api.messages[2].append(comment(3, "/ux duplicate #1"))
    assert gate_report(api, SHA)["status"] == "blocked"
    del api.rows[2]
    assert gate_report(api, SHA)["status"] == "blocked"


def test_incomplete_api_inventory_and_permission_errors_fail_closed():
    api = GitHub("example/project", run=lambda _: '[{"not":"a page"}]')
    with pytest.raises(ValueError, match="inventory"):
        api.indexed()
    def forbidden(argv):
        raise subprocess.CalledProcessError(1, argv, stderr="HTTP 403")
    with pytest.raises(subprocess.CalledProcessError):
        GitHub("example/project", run=forbidden).trusted("maintainer")
    with pytest.raises(ValueError, match="candidate SHA"):
        gate_report(Ledger(), "main")


def test_permission_is_checked_live_and_cached_per_run():
    calls = []
    def run(argv):
        calls.append(argv)
        return '{"permission":"write"}'
    api = GitHub("example/project", run=run)
    assert api.trusted("maintainer") and api.trusted("maintainer")
    assert len(calls) == 1
    assert calls[0][-1] == "repos/example/project/collaborators/maintainer/permission"


def test_fix_relationships_paginate_completely_and_reject_cursor_stalls():
    responses = [
        {"nodes": [{"number": 2}], "pageInfo": {"hasNextPage": True, "endCursor": "next"}},
        {"nodes": [{"number": 3}], "pageInfo": {"hasNextPage": False, "endCursor": "last"}},
    ]
    def run(_):
        return json.dumps({"data": {"repository": {"issue": {"closedByPullRequestsReferences": responses.pop(0)}}}})
    assert GitHub("example/project", run=run).links(1) == [{"number": 2}, {"number": 3}]
    responses[:] = [{"nodes": [], "pageInfo": {"hasNextPage": True, "endCursor": None}}]
    with pytest.raises(ValueError, match="pagination"):
        GitHub("example/project", run=run).links(1)


@pytest.mark.parametrize("path", ["publish", "advance", "repair", "store"])
def test_live_ux_failure_precedes_every_publication_mutation(monkeypatch, path):
    from scripts.releases import sequencer, stable, channel_releases, docker, store
    from scripts.ux_blockers import service
    record = {"version": "0.23.3", "commit": SHA, "state": "green" if path == "publish" else "published",
              "autopublish": True, "needs_retarget": path == "repair", "release_id": 42,
              "tag": "v0.23.3", "claim_tag": "rc.1-v0.23.3", "lanes": ["windows-store"],
              "candidate_manifest_sha256": None, "docker_manifest_digest": None}
    monkeypatch.setattr(sequencer, "discover", lambda *_: [record])
    calls = []
    def held(repository, candidate, **kwargs):
        calls.append((repository, candidate))
        raise ValueError("confirmed UX hold")
    def mutation(*args, **kwargs):
        pytest.fail("Publication occurred before the live UX gate")
    monkeypatch.setattr(service, "require_candidate", held)
    monkeypatch.setattr(stable, "publish_attempt", mutation)
    monkeypatch.setattr(stable, "edit_draft_release", mutation)
    monkeypatch.setattr(stable, "publish_release_draft", mutation)
    monkeypatch.setattr(channel_releases, "advance_stable", mutation)
    monkeypatch.setattr(docker, "promote_stable", mutation)
    monkeypatch.setattr(store, "release_from_env", mutation)
    # Store recovery still runs when the release is already at the head.
    with pytest.raises(ValueError, match="confirmed UX hold"):
        sequencer.reconcile({"GITHUB_REPOSITORY": "example/project"},
                            read_head=lambda: "0.23.3" if path == "store" else "0.23.2",
                            read_archive=mutation)
    assert calls == [("example/project", SHA)]


def test_hermes_audit_accounting_and_recovery_fields_are_not_release_authority():
    root = Path(__file__).resolve().parents[2]
    sample = json.loads((root / ".github/ux-blockers/hermes-first-wave-2026-11-18.json").read_text())
    assert len(sample["clusters"]) == 19
    qualifying = [row for row in sample["clusters"] if row["historical_ux_p0_qualifying"]]
    assert sum(row["historical_disposition"] in ("open_report_evidenced", "pre_window_carry_in") for row in qualifying) == 10
    assert all(row["current_release_blocker_assessment"] == "not_assessed" for row in sample["clusters"])
    assert sample["counts"]["historical_qualifying_merged_prs"] == 8


def test_bug_form_options_match_classifier_and_gate_is_mandatory():
    root = Path(__file__).resolve().parents[2]
    form = (root / ".github/ISSUE_TEMPLATE/bug_report.yml").read_text()
    for option in (*SURFACES, *RECOVERIES):
        assert f"- {option}\n" in form
    from scripts.releases.stable import gate_expectations
    assert gate_expectations(["ux-blockers"], [])["ux-blockers"] == "success"
    workflow = (root / ".github/workflows/ux-blockers.yml").read_text()
    assert 'ref: ${{ github.event.repository.default_branch }}' in workflow
    assert "github.event.pull_request.head" not in workflow
