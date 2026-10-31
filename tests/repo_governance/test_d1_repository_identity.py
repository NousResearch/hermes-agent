"""D1.3 request-only repository identity contract tests."""
from __future__ import annotations

import copy
import importlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

CANDIDATE = Path("/Users/ykliu/.hermes/profiles/dev/artifacts/repo-governance/2026-08-11-repo-governance-d1-3-i3-candidate")


def _load_json(name: str):
    return json.loads((CANDIDATE / name).read_text(encoding="utf-8"))


def _module():
    try:
        return importlib.import_module("repo_governance.repository_identity")
    except ModuleNotFoundError:
        pytest.fail("request-only repository identity behavior is absent", pytrace=False)


def test_public_callable_is_request_only_and_rejects_before_observer():
    module = _module()
    assert list(inspect.signature(module.resolve_repository_identity).parameters) == ["request"]
    calls = []
    resolver = module._build_test_resolver(lambda state: (None, None, state["derivedBinding"]), lambda request: calls.append(request))
    forbidden = _load_json("preserved-b03/reachability-profile.json")["forbiddenPublicInputs"]
    for field in forbidden + ["fd", "pid", "state", "environment", "authority"]:
        result = resolver({field: object()})
        assert result == {"kind": "error", "error": "REQUEST_INVALID", "failedStageDecimal": "1", "diagnosticCode": "NONE"}
    assert calls == []


def _materialize(vector, baselines):
    value = copy.deepcopy(vector.get("object", baselines.get(vector.get("baseline"))))
    if "object" in vector:
        return value
    if vector["op"] == "remove":
        value.pop(vector["path"])
    else:
        value[vector["path"]] = copy.deepcopy(vector["value"])
    return value


def test_public_result_object_and_raw_framing_vectors_are_strict():
    module = _module()
    objects = _load_json("public-result-vectors.v2.json")
    assert len(objects["vectors"]) == 64
    for vector in objects["vectors"]:
        candidate = _materialize(vector, objects["baselines"])
        try:
            module._validate_public_result(candidate)
            valid = True
        except (TypeError, ValueError):
            valid = False
        assert valid is vector["expectedValid"], vector["id"]
    frames = _load_json("public-result-framing-vectors.v3.json")
    assert len(frames["vectors"]) == 14
    for vector in frames["vectors"]:
        try:
            module.parse_public_result_frame(bytes.fromhex(vector["bytesHex"]))
            valid = True
        except (TypeError, ValueError):
            valid = False
        assert valid is vector["expectedValid"], vector["id"]


def test_r6_four_positives_118_mutations_16_requirements_and_precedence():
    module = _module()
    vectors = _load_json("bindings/r6-vectors.json")
    profile = _load_json("bindings/r6-profile.json")
    assert len(vectors["positiveStates"]) == 4
    assert len(vectors["mutationVectors"]) == 118
    assert len(profile["namedRequirements"]) == 16
    assert [row["stageDecimal"] for row in profile["stagePredicateTable"]] == [str(i) for i in range(15)]
    for positive in vectors["positiveStates"]:
        assert module._evaluate_r6_state(positive["state"])[0] is None, positive["id"]
    positives = {row["id"]: row["state"] for row in vectors["positiveStates"]}
    for vector in vectors["mutationVectors"]:
        state = copy.deepcopy(positives[vector["baselineId"]])
        if vector["preexistingMutation"]:
            state = module._apply_r6_mutation(state, vector["preexistingMutation"])
        state = module._apply_r6_mutation(state, vector["mutation"])
        actual = module._evaluate_r6_state(state)
        assert actual[:2] == (vector["expectedError"], int(vector["expectedFailedStageDecimal"])), vector["id"]


def test_coherent_reseal_changes_primitive_and_hash_facts_not_request_or_installation():
    module = _module()
    vectors = _load_json("bindings/r6-vectors.json")
    before = vectors["positiveStates"][0]["state"]
    after = module._coherently_reseal_for_test(before, dev_delta=1, inode_delta=7)
    installation_fields = (
        "gitDependencyFirst", "gitDependencySecond", "launchCapability",
        "spawnTrace", "lifecycle", "discoveryFirst", "discoverySecond",
        "dotGit", "layoutKind",
    )
    assert after["request"]["requestSchemaVersion"] == before["request"]["requestSchemaVersion"]
    assert after["request"]["effectiveWorkdirPathB64"] == before["request"]["effectiveWorkdirPathB64"]
    assert all(after[field] == before[field] for field in installation_fields)
    assert after["anchorsFirst"] != before["anchorsFirst"]
    assert after["markerFirst"] != before["markerFirst"]
    assert after["request"]["expectedBinding"]["repositoryIncarnationMarkerFileSha256"] != before["request"]["expectedBinding"]["repositoryIncarnationMarkerFileSha256"]
    assert after["request"]["expectedBinding"]["repositoryKeySha256"] != before["request"]["expectedBinding"]["repositoryKeySha256"]
    assert after["request"]["expectedBinding"]["specificWorktreeKeySha256"] != before["request"]["expectedBinding"]["specificWorktreeKeySha256"]
    fresh = copy.deepcopy(after)
    assert module._evaluate_r6_state(fresh)[0] is None
    tampered = copy.deepcopy(fresh)
    tampered["request"]["expectedBinding"]["specificWorktreeKeySha256"] = "0" * 64
    assert module._evaluate_r6_state(tampered)[0] is not None
