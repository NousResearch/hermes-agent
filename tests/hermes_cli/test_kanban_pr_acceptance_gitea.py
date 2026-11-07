"""Gitea exact-PR completion contracts through the real kanban lifecycle.

The trusted gate is a local executable that records its argv/env and prints a
scripted receipt; receipts and the config here are synthetic. What a real gate
prints is pinned by ``test_gitea_acceptance.py`` against a captured forge.
"""
import json
import os
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_pr_acceptance_gitea as transport
from hermes_cli.kanban_db_connect import connect
from hermes_cli.kanban_pr_acceptance import collect_acceptance, validate_contract

ORIGIN = "https://forge.example.test"
URL = ORIGIN + "/acme/widget/pulls/8"
SHA = "a" * 40
pytestmark = pytest.mark.platforms("linux", "macos")


def _receipt(**over):
    receipt = {"ok": True, "classification": "success", "head_sha": SHA, "pr_url": URL, "provider": "gitea",
               "origin": ORIGIN, "repo": "acme/widget", "policy_id": "widget-v1", "qualification_only": False,
               "checks": [{"name": "CI / verify (pull_request)", "id": 11, "run_id": 8, "head_sha": SHA,
                           "classification": "success", "log": "untrusted text"}],
               "merge": {"merged": True, "merge_commit_sha": "b" * 40, "merged_by": "merge-bot", "merged_by_id": 4},
               "review": {"id": 31, "reviewer": "bob", "reviewer_id": 7, "commit_id": SHA},
               "detail": "gate-provided prose is never persisted"}
    receipt.update(over)
    return receipt


@pytest.fixture
def forge(tmp_path, monkeypatch):
    launch = tmp_path / "home"
    profile = launch / "profiles" / "b"
    profile.mkdir(parents=True)
    os.chmod(profile, 0o755)
    (profile / ".env").write_text("GITEA_TOKEN=b-token\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("GITEA_TOKEN", "launch-token")
    monkeypatch.setenv("GH_TOKEN", "launch-gh-token")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/launch/agent.sock")
    gate_dir = tmp_path / "gate"
    gate_dir.mkdir()
    os.chmod(gate_dir, 0o755)
    state = gate_dir / "state.json"
    gate = gate_dir / "gate"
    gate.write_text(
        f"#!{sys.executable}\nimport json, os, sys, time\n"
        f"state = json.load(open({str(state)!r}))\n"
        f"json.dump({{'argv': sys.argv, 'env': dict(os.environ), 'cwd': os.getcwd()}}, open({str(gate_dir / 'seen.json')!r}, 'w'))\n"
        "time.sleep(state.get('sleep', 0))\n"
        "sys.stderr.write('secret-stderr token=xyz\\n')\n"
        "sys.stdout.write(state['stdout'])\n"
        "sys.exit(state.get('exit', 0))\n", encoding="utf-8")
    os.chmod(gate, 0o755)
    entry = {"argv": [str(gate), "--read-only"], "policy_id": "widget-v1", "qualification_only": False,
             "repos": ["acme/widget"]}

    class Forge:
        home, seen = profile, gate_dir / "seen.json"

        def __init__(self):
            self.entry = entry
            self.configure()
            self.respond(_receipt())

        def configure(self, origins=None, mode=0o600, raw=None):
            config = profile / transport.CONFIG_NAME
            config.write_text(raw if raw is not None else json.dumps({"origins": origins or {ORIGIN: self.entry}}),
                              encoding="utf-8")
            os.chmod(config, mode)

        def respond(self, receipt=None, stdout=None, **extra):
            state.write_text(json.dumps({"stdout": stdout if stdout is not None else json.dumps(receipt), **extra}),
                             encoding="utf-8")

        def collect(self, assignee="b", published=None):
            if self.seen.exists():
                self.seen.unlink()
            return collect_acceptance(URL, published, assignee=assignee)

    kb.init_db()
    return Forge()


def test_production_receipt_completes_card_with_sanitized_bound_evidence(forge):
    with connect() as conn:
        tid = kb.create_task(conn, title="ship", completion_contract=URL, assignee="b")
        assert kb.complete_task(conn, tid, result="done", metadata={"published_pr": URL})
        assert kb.get_task(conn, tid).status == "done"
        receipt = json.loads(conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'",
                                          (tid,)).fetchone()[0])
    assert receipt["ok"] and receipt["origin"] == ORIGIN and receipt["policy_id"] == "widget-v1"
    assert receipt["merge"]["merge_commit_sha"] == "b" * 40 and receipt["review"]["commit_id"] == SHA
    assert "log" not in receipt["checks"][0] and "gate-provided" not in json.dumps(receipt)
    seen = json.loads(forge.seen.read_text())
    assert seen["argv"][1:] == ["--read-only", URL]
    assert Path(seen["cwd"]).resolve() == forge.home.resolve()
    assert seen["env"]["GITEA_TOKEN"] == "b-token"
    assert "GH_TOKEN" not in seen["env"] and "SSH_AUTH_SOCK" not in seen["env"]


def test_failed_gate_leaves_card_open_with_recovery(forge):
    forge.respond(_receipt(ok=False, classification="review", checks=[]))
    with connect() as conn:
        tid = kb.create_task(conn, title="ship", completion_contract=URL, assignee="b")
        assert not kb.complete_task(conn, tid, result="done")
        task = kb.get_task(conn, tid)
    assert task.status != "done"
    assert "PR acceptance review" in task.last_failure_error and "retry" in task.last_failure_error


def test_unknown_gate_classification_is_infra(forge):
    forge.respond(_receipt(ok=False, classification="looks-fine-to-me"))
    assert forge.collect()["classification"] == "infra"


def test_qualification_evidence_cannot_complete_production_origin(forge):
    forge.respond(_receipt(qualification_only=True, merge=None, review=None))
    receipt = forge.collect()
    assert receipt["ok"] is False and receipt["classification"] == "qualification"


def test_explicit_qualification_scope_accepts_only_matching_receipt(forge):
    forge.entry = dict(forge.entry, qualification_only=True, policy_id="pilot-v1")
    forge.configure()
    pilot = _receipt(qualification_only=True, policy_id="pilot-v1", merge=None, review=None)
    forge.respond(pilot)
    receipt = forge.collect()
    assert receipt["ok"] and receipt["qualification_only"] is True and "merge" not in receipt
    forge.respond(dict(pilot, qualification_only=False))
    assert forge.collect()["classification"] == "binding"


@pytest.mark.parametrize("field,value", [
    ("pr_url", ORIGIN + "/acme/widget/pulls/9"), ("provider", "github"), ("origin", "https://other.example.test"),
    ("repo", "acme/other"), ("policy_id", "other-v1"), ("qualification_only", None), ("qualification_only", 0),
])
def test_receipt_binding_mismatch_fails_closed(forge, field, value):
    forge.respond(_receipt(**{field: value}))
    assert forge.collect()["classification"] == "binding"


@pytest.mark.parametrize("over", [
    {"head_sha": "A" * 40}, {"head_sha": None}, {"checks": []},
    {"checks": [{"name": "ci", "head_sha": "c" * 40, "classification": "success"}]},
    {"checks": [{"name": "ci", "head_sha": SHA, "classification": "pending"}]},
    {"classification": "pending"}, {"ok": "true"}, {"merge": None}, {"review": None},
    {"review": {"id": 31, "reviewer": "bob", "commit_id": "c" * 40}},
    {"merge": {"merged": True, "merge_commit_sha": "short", "merged_by": "merge-bot"}},
    {"merge": {"merged": False, "merge_commit_sha": "b" * 40, "merged_by": "merge-bot"}},
])
def test_malformed_positive_receipt_is_infra(forge, over):
    forge.respond(_receipt(**over))
    receipt = forge.collect()
    assert receipt["ok"] is False and receipt["classification"] == "infra"


@pytest.mark.parametrize("respond,detail", [
    (dict(stdout="not json"), "malformed JSON"),
    (dict(stdout="[]"), "no receipt object"),
    (dict(stdout="{}", exit=3), "exited with status 3"),
    (dict(stdout="x" * (transport.MAX_OUTPUT_BYTES + 10)), "exceeded its bound"),
])
def test_transport_failures_are_sanitized_infra(forge, respond, detail):
    forge.respond(**respond)
    receipt = forge.collect()
    assert receipt["classification"] == "infra" and detail in receipt["detail"]
    assert "secret-stderr" not in json.dumps(receipt) and "xyz" not in json.dumps(receipt)


def test_transport_timeout_is_bounded(forge, monkeypatch):
    monkeypatch.setattr(transport, "TIMEOUT_SECONDS", 2)
    forge.respond(_receipt(), sleep=30)
    receipt = forge.collect()
    assert receipt["classification"] == "infra" and "timed out" in receipt["detail"]


@pytest.mark.parametrize("change", [
    "unassigned", "missing_config", "unsafe_mode", "symlink", "unknown_key", "allow_qualification",
    "no_qualification_scope", "string_qualification_scope", "relative_argv", "origin_unconfigured",
    "repo_not_allowlisted", "writable_gate_dir", "bad_origin_key", "global_switch",
])
def test_configuration_fails_closed_without_running_gate(forge, change, tmp_path):
    config = forge.home / transport.CONFIG_NAME
    entry = dict(forge.entry)

    def symlink():
        real = tmp_path / "elsewhere.json"
        real.write_text(config.read_text(encoding="utf-8"), encoding="utf-8")
        config.unlink()
        config.symlink_to(real)
    edits = {
        "unassigned": lambda: None,
        "missing_config": config.unlink,
        "unsafe_mode": lambda: forge.configure(mode=0o666),
        "symlink": symlink,
        "unknown_key": lambda: forge.configure(origins={ORIGIN: dict(entry, timeout=5)}),
        "allow_qualification": lambda: forge.configure(origins={ORIGIN: dict(entry, allow_qualification=True)}),
        "no_qualification_scope": lambda: forge.configure(
            origins={ORIGIN: {k: v for k, v in entry.items() if k != "qualification_only"}}),
        "string_qualification_scope": lambda: forge.configure(origins={ORIGIN: dict(entry, qualification_only="false")}),
        "relative_argv": lambda: forge.configure(origins={ORIGIN: dict(entry, argv=["gate"])}),
        "origin_unconfigured": lambda: forge.configure(origins={"https://other.example.test": entry}),
        "repo_not_allowlisted": lambda: forge.configure(origins={ORIGIN: dict(entry, repos=["acme/other"])}),
        "writable_gate_dir": lambda: os.chmod(Path(entry["argv"][0]).parent, 0o777),
        "bad_origin_key": lambda: forge.configure(origins={ORIGIN + "/": entry}),
        "global_switch": lambda: forge.configure(
            raw=json.dumps({"origins": {ORIGIN: entry}, "allow_qualification": True})),
    }
    edits[change]()
    receipt = forge.collect(assignee=None if change == "unassigned" else "b")
    assert receipt["ok"] is False and receipt["classification"] == "auth", receipt
    assert not forge.seen.exists()


def test_unresolvable_assignee_is_auth(forge):
    receipt = forge.collect(assignee="ghost")
    assert receipt["classification"] == "auth" and "cannot be resolved" in receipt["detail"]
    assert not forge.seen.exists()


def test_published_pr_must_equal_declared_contract(forge):
    receipt = forge.collect(published=ORIGIN + "/acme/widget/pulls/9")
    assert receipt["ok"] is False and receipt["classification"] == "missing"
    assert not forge.seen.exists()


def test_contract_validation_accepts_exact_gitea_urls_only():
    assert validate_contract(URL) == URL
    assert validate_contract("https://github.com/acme/widget/pull/8")
    for bad in ("http://forge.example.test/acme/widget/pulls/8", URL + "?x=1", URL + "#c",
                "https://github.com/acme/widget/pulls/8", "https://forge.example.test/acme/widget/pull/8"):
        with pytest.raises(ValueError):
            validate_contract(bad)
