import os
import json
import pytest
import sys
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

def test_evidence_store_add(tmp_path):
    store_file = tmp_path / "test_evidence.json"
    store = EvidenceStore(str(store_file))
    
    eid = store.add(
        source="test_source",
        content="test_content",
        evidence_type="git",
        actor="test_actor",
        notes="test_notes"
    )
    
    assert eid == "EV-0001"
    assert len(store.data["evidence"]) == 1
    assert store.data["evidence"][0]["content"] == "test_content"
    assert store.data["evidence"][0]["id"] == "EV-0001"

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
    
    store.add(source="github_api", content="malicious activity", evidence_type="gh_api")
    store.add(source="manual", content="clean", evidence_type="manual")
    
    results = store.query("malicious")
    assert len(results) == 1
    assert results[0]["source"] == "github_api"
