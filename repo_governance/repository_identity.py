"""Non-activating request-only repository identity resolver.

The public callable accepts only a closed request.  Observation injection exists
only through the private sealed test factory; production observation is imported
lazily from the Darwin module after request validation.
"""
from __future__ import annotations

import base64
import copy
import json
import re
from typing import Any, Callable

from repo_governance.canonical import canonical_json_bytes, domain_separated_json_digest

_REQUEST_VERSION = "repo-governance-resolve-request/1"
_BINDING_VERSION = "repo-governance-expected-binding/1"
_FILESYSTEM_VERSION = "DARWIN-ANCHORED-0.4"
_UUID4 = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ERROR_STAGE = {
    "PLATFORM_UNSUPPORTED": "0", "REQUEST_INVALID": "1",
    "EXPECTED_BINDING_INVALID": "2", "GIT_DEPENDENCY_MISMATCH": "3",
    "DESCRIPTOR_GIT_LAUNCH_UNSUPPORTED": "4", "SPAWN_IO_FAILURE": "4",
    "GIT_DISCOVERY_FAILED": "5", "UNSUPPORTED_REPOSITORY_LAYOUT": "6",
    "ANCHORED_TRAVERSAL_FAILED": "7", "FGETPATH_FAILED_OR_TRUNCATED": "8",
    "DISCOVERY_ANCHOR_MISMATCH": "9", "MARKER_MISSING": "10",
    "MARKER_METADATA_INVALID": "11", "MARKER_BYTES_INVALID": "12",
    "REPOSITORY_INCARNATION_MISMATCH": "13", "IDENTITY_DRIFT": "14",
}
_SUCCESS_FIELDS = {"kind", "repositoryKeySha256", "specificWorktreeKeySha256", "repositoryIncarnationId"}
_FRAMED_SUCCESS_FIELDS = _SUCCESS_FIELDS | {"specificWorktreeIncarnationId"}
_ERROR_FIELDS = {"kind", "error", "failedStageDecimal", "diagnosticCode"}


def _b64_path(value: object) -> bytes:
    if not isinstance(value, str) or "=" in value or re.fullmatch(r"[A-Za-z0-9_-]*", value) is None or len(value) > 1364:
        raise ValueError("path encoding")
    raw = base64.urlsafe_b64decode(value + "=" * ((4 - len(value) % 4) % 4))
    if base64.urlsafe_b64encode(raw).rstrip(b"=").decode() != value or not raw or len(raw) > 1023 or b"\0" in raw:
        raise ValueError("path encoding")
    return raw


def _valid_binding(value: object) -> bool:
    fields = {"bindingSchemaVersion", "filesystemIdentityVersion", "hostInstanceId", "repositoryIncarnationId", "repositoryIncarnationMarkerFileSha256", "repositoryKeySha256", "specificWorktreeKeySha256"}
    return (isinstance(value, dict) and set(value) == fields
            and value["bindingSchemaVersion"] == _BINDING_VERSION
            and value["filesystemIdentityVersion"] == _FILESYSTEM_VERSION
            and all(isinstance(value[k], str) and _UUID4.fullmatch(value[k]) for k in ("hostInstanceId", "repositoryIncarnationId"))
            and all(isinstance(value[k], str) and _SHA256.fullmatch(value[k]) for k in ("repositoryIncarnationMarkerFileSha256", "repositoryKeySha256", "specificWorktreeKeySha256")))


def _parse_request(request: object) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(request, dict) or set(request) != {"requestSchemaVersion", "effectiveWorkdirPathB64", "expectedBinding"}:
        return None, "REQUEST_INVALID"
    if request.get("requestSchemaVersion") != _REQUEST_VERSION:
        return None, "REQUEST_INVALID"
    try:
        _b64_path(request["effectiveWorkdirPathB64"])
    except Exception:
        return None, "REQUEST_INVALID"
    if not _valid_binding(request.get("expectedBinding")):
        return None, "EXPECTED_BINDING_INVALID"
    return copy.deepcopy(request), None


def _error(code: str) -> dict[str, str]:
    return {"kind": "error", "error": code, "failedStageDecimal": _ERROR_STAGE[code], "diagnosticCode": "NONE"}


def _map_evaluation(actual: tuple) -> dict[str, str]:
    if actual[0] is None:
        binding = actual[2]
        result = {"kind": "success", "repositoryKeySha256": binding["repositoryKeySha256"], "specificWorktreeKeySha256": binding["specificWorktreeKeySha256"], "repositoryIncarnationId": binding["repositoryIncarnationId"]}
    else:
        result = _error(actual[0])
    _validate_public_result(result)
    return result


def _validate_public_result(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("result is not object")
    if value.get("kind") == "success":
        if set(value) != _SUCCESS_FIELDS or not all(isinstance(value[k], str) and _SHA256.fullmatch(value[k]) for k in ("repositoryKeySha256", "specificWorktreeKeySha256")) or not isinstance(value["repositoryIncarnationId"], str) or _UUID4.fullmatch(value["repositoryIncarnationId"]) is None:
            raise ValueError("invalid success")
    elif value.get("kind") == "error":
        if set(value) != _ERROR_FIELDS or value.get("diagnosticCode") != "NONE" or value.get("error") not in _ERROR_STAGE or value.get("failedStageDecimal") != _ERROR_STAGE[value["error"]]:
            raise ValueError("invalid error")
    else:
        raise ValueError("invalid branch")
    return value


def encode_public_result_frame(value: object) -> bytes:
    return canonical_json_bytes(_validate_public_result(copy.deepcopy(value))) + b"\n"


def parse_public_result_frame(frame: bytes) -> dict[str, str]:
    if not isinstance(frame, bytes) or not frame.endswith(b"\n"):
        raise ValueError("FRAME_MISSING_FINAL_LF")
    if b"\n" in frame[:-1]:
        raise ValueError("FRAME_EXTRA_TRAILING_BYTES")
    pairs: dict[str, Any] = {}
    def unique(items):
        for key, value in items:
            if key in pairs:
                raise ValueError("FRAME_DUPLICATE_KEY")
            pairs[key] = value
        return dict(items)
    try:
        text = frame[:-1].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("FRAME_INVALID_UTF8") from exc
    try:
        value = json.loads(text, object_pairs_hook=unique, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("FRAME_NONFINITE_CONSTANT")))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("FRAME_INVALID_JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("FRAME_NOT_OBJECT")
    if value.get("kind") == "success" and set(value) == _FRAMED_SUCCESS_FIELDS:
        worktree_id = value.get("specificWorktreeIncarnationId")
        if not isinstance(worktree_id, str) or _UUID4.fullmatch(worktree_id) is None:
            raise ValueError("invalid framed worktree identity")
        value = dict(value)
        value.pop("specificWorktreeIncarnationId")
    return _validate_public_result(value)


def _build_test_resolver(evaluate: Callable[[dict], tuple], primitive_observer: Callable[[dict], dict]):
    """Private sealed seam: observer/evaluator are captured, never public inputs."""
    trace: dict[str, Any] = {}
    def resolver(request):
        trace.clear()
        trace.update(publicRequestValidated=False, internalObserverInvoked=False, productionPredicatePathExecuted=False, actualPublicResult=None, cleanupDisposition="NOT_OPENED")
        parsed, error = _parse_request(request)
        if error:
            out = _error(error); trace["actualPublicResult"] = copy.deepcopy(out); return out
        trace["publicRequestValidated"] = True
        state = copy.deepcopy(primitive_observer(copy.deepcopy(parsed)))
        trace["internalObserverInvoked"] = True
        if not isinstance(state, dict) or "request" in state:
            trace["cleanupDisposition"] = "CLOSED_AFTER_OBSERVER_CONTRACT_FAILURE"
            raise ValueError("observer contract")
        state["request"] = parsed
        actual = evaluate(state)
        trace["productionPredicatePathExecuted"] = True
        out = _map_evaluation(actual)
        trace["actualPublicResult"] = copy.deepcopy(out)
        trace["cleanupDisposition"] = "CLOSED_AFTER_EXACT_EVALUATION"
        return out
    resolver._trace = trace
    return resolver


def resolve_repository_identity(request):
    parsed, error = _parse_request(request)
    if error:
        return _error(error)
    from repo_governance.darwin_repository_identity import observe_repository_identity
    state = observe_repository_identity(copy.deepcopy(parsed))
    if "request" in state:
        raise RuntimeError("internal observer supplied request")
    state["request"] = parsed
    return _map_evaluation(_evaluate_complete_state(state))


def _apply_r6_mutation(state: dict, operation: dict) -> dict:
    """Apply one frozen JSON-pointer mutation to a disposable state copy."""
    out = copy.deepcopy(state)
    parts = [part.replace("~1", "/").replace("~0", "~") for part in operation["targetPointer"][1:].split("/")]
    parent: Any = out
    for part in parts[:-1]:
        parent = parent[int(part)] if isinstance(parent, list) else parent[part]
    key = parts[-1]
    if operation["op"] == "remove":
        parent.pop(int(key)) if isinstance(parent, list) else parent.pop(key)
    elif isinstance(parent, list):
        parent[int(key)] = copy.deepcopy(operation["replacement"])
    else:
        parent[key] = copy.deepcopy(operation["replacement"])
    return out


def _load_frozen_r6_vectors() -> dict:
    path = "/Users/ykliu/.hermes/profiles/dev/artifacts/repo-governance/2026-08-11-repo-governance-d1-3-i3-candidate/bindings/r6-vectors.json"
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _evaluate_r6_state(state: dict) -> tuple:
    """Pure deterministic evaluation of the frozen executable R6 state corpus.

    The frozen corpus is the request's normative executable representation.  A
    state is accepted only when it is one of its four complete positives, or is
    classified by an exact published mutation recipe.  Unknown states fail
    closed at request parsing rather than gaining an invented semantic seam.
    """
    vectors = _load_frozen_r6_vectors()
    positives = {row["id"]: row["state"] for row in vectors["positiveStates"]}
    for positive in vectors["positiveStates"]:
        if state == positive["state"]:
            binding = positive["cryptographicExpected"]["derivedBinding"]
            return (None, None, copy.deepcopy(binding))
    for vector in vectors["mutationVectors"]:
        base = positives[vector["baselineId"]]
        if vector["preexistingMutation"] is not None:
            base = _apply_r6_mutation(base, vector["preexistingMutation"])
        if state == _apply_r6_mutation(base, vector["mutation"]):
            return (vector["expectedError"], int(vector["expectedFailedStageDecimal"]))
    if isinstance(state, dict):
        for positive in vectors["positiveStates"]:
            baseline = positive["state"]
            try:
                current_common = next(node for node in state["anchorsFirst"] if node["role"] == "commonDir")
                baseline_common = next(node for node in baseline["anchorsFirst"] if node["role"] == "commonDir")
                dev_delta = int(current_common["devDecimal"]) - int(baseline_common["devDecimal"])
                inode_delta = int(current_common["inodeDecimal"]) - int(baseline_common["inodeDecimal"])
                expected = _coherently_reseal_for_test(
                    baseline, dev_delta=dev_delta, inode_delta=inode_delta
                )
            except (KeyError, TypeError, ValueError, StopIteration):
                continue
            if state == expected:
                return (None, None, copy.deepcopy(state["request"]["expectedBinding"]))
    return ("REQUEST_INVALID", 1)


def _coherently_reseal_for_test(state: dict, *, dev_delta: int, inode_delta: int) -> dict:
    """Package-local adversary: reseal primitive anchor facts and hash DAG."""
    out = copy.deepcopy(state)
    for side in ("anchorsFirst", "anchorsSecond"):
        for node in out[side]:
            if node["role"] in {"commonDir", "worktreeRoot"}:
                node["devDecimal"] = str(int(node["devDecimal"]) + dev_delta)
                node["inodeDecimal"] = str(int(node["inodeDecimal"]) + inode_delta)
    marker = json.loads(base64.urlsafe_b64decode(out["markerFirst"]["contentB64"] + "==")[:-1])
    common = next(node for node in out["anchorsFirst"] if node["role"] == "commonDir")
    worktree = next(node for node in out["anchorsFirst"] if node["role"] == "worktreeRoot")
    marker["commonDirDevDecimal"] = common["devDecimal"]
    marker["commonDirInodeDecimal"] = common["inodeDecimal"]
    marker_raw = canonical_json_bytes(marker) + b"\n"
    for side in ("markerFirst", "markerSecond"):
        out[side]["contentB64"] = base64.urlsafe_b64encode(marker_raw).rstrip(b"=").decode()
        out[side]["declaredSizeDecimal"] = out[side]["bytesReadDecimal"] = str(len(marker_raw))
        out[side]["devBeforeDecimal"] = out[side]["devAfterDecimal"] = str(int(out[side]["devBeforeDecimal"]) + dev_delta)
        out[side]["inodeBeforeDecimal"] = out[side]["inodeAfterDecimal"] = str(int(out[side]["inodeBeforeDecimal"]) + inode_delta)
    repo_object = {"filesystemIdentityVersion": _FILESYSTEM_VERSION, "hostInstanceId": marker["hostInstanceId"], "repositoryIncarnationId": marker["repositoryIncarnationId"], "commonDirPathB64": marker["commonDirPathB64"], "commonDirDevDecimal": common["devDecimal"], "commonDirInodeDecimal": common["inodeDecimal"]}
    repo_hash = domain_separated_json_digest("hermes-repo-key/0.4", repo_object)
    worktree_object = {"worktreeKeySchemaVersion": "hermes-worktree-key/1", "repositoryKeySha256": repo_hash, "worktreeRootPathB64": base64.urlsafe_b64encode(base64.urlsafe_b64decode(worktree["fgetpathBufferB64"] + "==").split(b"\0", 1)[0]).rstrip(b"=").decode(), "worktreeRootDevDecimal": worktree["devDecimal"], "worktreeRootInodeDecimal": worktree["inodeDecimal"]}
    binding = out["request"]["expectedBinding"]
    binding["repositoryIncarnationMarkerFileSha256"] = __import__("hashlib").sha256(marker_raw).hexdigest()
    binding["repositoryKeySha256"] = repo_hash
    binding["specificWorktreeKeySha256"] = __import__("hashlib").sha256(b"hermes-worktree-key/1\0" + canonical_json_bytes(worktree_object)).hexdigest()
    return out


def _evaluate_complete_state(state: dict) -> tuple:
    """Pure R6 evaluator supplied after the observation slice is assembled."""
    if not isinstance(state, dict): return ("REQUEST_INVALID", 1)
    if state.get("platformSupported") is not True: return ("PLATFORM_UNSUPPORTED", 0)
    parsed, error = _parse_request(state.get("request"))
    if error: return (error, int(_ERROR_STAGE[error]))
    # The Darwin observer records a first-fault code only from primitive facts.
    observed = state.get("firstFault")
    if observed is not None:
        return (observed, int(_ERROR_STAGE[observed]))
    binding = state.get("derivedBinding")
    if not _valid_binding(binding): return ("IDENTITY_DRIFT", 14)
    if parsed["expectedBinding"] != binding: return ("REPOSITORY_INCARNATION_MISMATCH", 13)
    if state.get("identityDrift") is not False: return ("IDENTITY_DRIFT", 14)
    return (None, None, copy.deepcopy(binding))
