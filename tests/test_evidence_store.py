import os
import json
import pytest
from pathlib import Path
import importlib.util

# Load the hyphenated script name dynamically
repo_root = Path(__file__).parent.parent
script_path = repo_root / "optional-skills" / "security" / "oss-forensics" / "scripts" / "evidence-store.py"

spec = importlib.util.spec_from_file_location("evidence_store", str(script_path))
evidence_store = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence_store)
EvidenceStore = evidence_store.EvidenceStore


def test_evidence_store_init(tmp_path):
    store_file = tmp_path / "test_evidence.json"
    store = EvidenceStore(str(store_file))
    assert store.filepath == str(store_file)
    assert len(store.data["evidence"]) == 0
    assert "metadata" in store.data
    assert store.data["metadata"]["version"] == "2.0"
    assert "chain_of_custody" in store.data


def test_evidence_store_add(tmp_path):
    store_file = tmp_path / "test_evidence.json"
    store = EvidenceStore(str(store_file))

    eid = store.add(
        source="test_source",
        content="test_content",
        evidence_type="git",
        actor="test_actor",
        notes="test_notes",
    )

    assert eid == "EV-0001"
    assert len(store.data["evidence"]) == 1
    assert store.data["evidence"][0]["content"] == "test_content"
    assert store.data["evidence"][0]["id"] == "EV-0001"
    assert store.data["evidence"][0]["actor"] == "test_actor"
    assert store.data["evidence"][0]["notes"] == "test_notes"
    # Verify SHA-256 was computed
    assert store.data["evidence"][0]["content_sha256"] is not None
    assert len(store.data["evidence"][0]["content_sha256"]) == 64


def test_evidence_store_add_persists(tmp_path):
    store_file = tmp_path / "test_evidence.json"
    store = EvidenceStore(str(store_file))
    store.add(source="s1", content="c1", evidence_type="git")

    # Reload from disk
    store2 = EvidenceStore(str(store_file))
    assert len(store2.data["evidence"]) == 1
    assert store2.data["evidence"][0]["id"] == "EV-0001"


def test_evidence_store_sequential_ids(tmp_path):
    store_file = tmp_path / "test_evidence.json"
    store = EvidenceStore(str(store_file))

    eid1 = store.add(source="s1", content="c1", evidence_type="git")
    eid2 = store.add(source="s2", content="c2", evidence_type="gh_api")
    eid3 = store.add(source="s3", content="c3", evidence_type="ioc")

    assert eid1 == "EV-0001"
    assert eid2 == "EV-0002"
    assert eid3 == "EV-0003"


def test_evidence_store_list(tmp_path):
    store_file = tmp_path / "test_evidence.json"
    store = EvidenceStore(str(store_file))

    store.add(source="s1", content="c1", evidence_type="git", actor="a1")
    store.add(source="s2", content="c2", evidence_type="gh_api", actor="a2")

    all_evidence = store.list_evidence()
    assert len(all_evidence) == 2

    git_evidence = store.list_evidence(filter_type="git")
    assert len(git_evidence) == 1
    assert git_evidence[0]["actor"] == "a1"

    actor_evidence = store.list_evidence(filter_actor="a2")
    assert len(actor_evidence) == 1
    assert actor_evidence[0]["type"] == "gh_api"


def test_evidence_store_verify_integrity(tmp_path):
    store_file = tmp_path / "test_evidence.json"
    store = EvidenceStore(str(store_file))

    store.add(source="s1", content="c1", evidence_type="git")
    assert len(store.verify_integrity()) == 0

    # Manually corrupt the content to trigger a hash mismatch
    store.data["evidence"][0]["content"] = "corrupted_content"
    issues = store.verify_integrity()
    assert len(issues) == 1
    assert issues[0]["id"] == "EV-0001"


def test_evidence_store_query(tmp_path):
    store_file = tmp_path / "test_evidence.json"
    store = EvidenceStore(str(store_file))

    store.add(source="github_api", content="malicious activity detected", evidence_type="gh_api")
    store.add(source="manual", content="clean observation", evidence_type="manual")

    results = store.query("malicious")
    assert len(results) == 1
    assert results[0]["source"] == "github_api"

    # Query should be case-insensitive
    results = store.query("MALICIOUS")
    assert len(results) == 1


def test_evidence_store_query_searches_multiple_fields(tmp_path):
    store_file = tmp_path / "test_evidence.json"
    store = EvidenceStore(str(store_file))

    store.add(source="git_fsck", content="dangling commit abc123", evidence_type="git", actor="attacker")
    store.add(source="manual", content="clean", evidence_type="manual")

    # Search by source
    assert len(store.query("fsck")) == 1
    # Search by actor
    assert len(store.query("attacker")) == 1
    # Search returns nothing for non-matching
    assert len(store.query("nonexistent")) == 0


def test_evidence_store_chain_of_custody(tmp_path):
    store_file = tmp_path / "test_evidence.json"
    store = EvidenceStore(str(store_file))

    store.add(source="s1", content="c1", evidence_type="git")
    store.add(source="s2", content="c2", evidence_type="gh_api")

    chain = store.data["chain_of_custody"]
    assert len(chain) == 2
    assert chain[0]["evidence_id"] == "EV-0001"
    assert chain[0]["action"] == "add"
    assert chain[1]["evidence_id"] == "EV-0002"


def test_evidence_store_export_markdown(tmp_path):
    store_file = tmp_path / "test_evidence.json"
    store = EvidenceStore(str(store_file))

    store.add(source="git_log", content="suspicious commit", evidence_type="git", actor="actor1")

    md = store.export_markdown()
    assert "# Evidence Registry" in md
    assert "EV-0001" in md
    assert "Chain of Custody" in md
    assert "actor1" in md


def test_evidence_store_summary(tmp_path):
    store_file = tmp_path / "test_evidence.json"
    store = EvidenceStore(str(store_file))

    store.add(source="s1", content="c1", evidence_type="git", actor="a1")
    store.add(source="s2", content="c2", evidence_type="git", actor="a2")
    store.add(source="s3", content="c3", evidence_type="gh_api", actor="a1")

    s = store.summary()
    assert s["total"] == 3
    assert s["by_type"]["git"] == 2
    assert s["by_type"]["gh_api"] == 1
    assert "a1" in s["unique_actors"]
    assert "a2" in s["unique_actors"]


def test_evidence_store_corrupted_file(tmp_path):
    store_file = tmp_path / "test_evidence.json"
    store_file.write_text("NOT VALID JSON {{{")

    with pytest.raises(SystemExit):
        EvidenceStore(str(store_file))
