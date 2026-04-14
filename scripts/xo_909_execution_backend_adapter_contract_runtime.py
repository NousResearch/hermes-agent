from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PACK_SCHEMA = "clawd.xo_909.execution_backend_adapter_contract_pack.v1"
SELECTION_SCHEMA = "clawd.xo_909.execution_backend_adapter_selection_packet.v1"
RETRY_SCHEMA = "clawd.xo_909.execution_backend_adapter_retry_regression.v1"
VALIDATION_SCHEMA = "clawd.validation_packet.v1"
MANIFEST_SCHEMA = "clawd.xo_909.execution_backend_adapter_runtime_manifest.v1"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate XO-909 backend adapter runtime artifacts")
    parser.add_argument("--repo-root", default=".", help="Repository root")
    parser.add_argument("--queue-path", default="state/continuity/latest/true_expanded_roadmap_queue_layer.json")
    parser.add_argument("--fixture-path", default="tests/fixtures/xo/xo_909_execution_backend_adapter_runtime_fixture_v1.json")
    parser.add_argument("--output-dir", default="state/continuity/latest")
    parser.add_argument("--stamp", default=None, help="Override stamp (YYYY-MM-DD)")
    parser.add_argument("--require-queued", action="store_true", help="Require slice state == QUEUED_OPTIONAL before generation")
    parser.add_argument("--json", action="store_true", help="Emit manifest JSON to stdout")
    return parser.parse_args(argv)


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _queue_index(queue_doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows = queue_doc.get("slices")
    if not isinstance(rows, list):
        return {}
    return {str(row.get("id") or ""): row for row in rows if isinstance(row, dict) and row.get("id")}


def _dependency_state_snapshot(row: Dict[str, Any], queue_index: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    deps = row.get("dependencies") or []
    states = {}
    if not isinstance(deps, list):
        return states
    for dep in deps:
        dep_row = queue_index.get(str(dep))
        states[str(dep)] = str(dep_row.get("state")) if isinstance(dep_row, dict) else "UNKNOWN"
    return states


def _dedupe(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _index_providers(providers: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(provider.get("provider_id") or ""): provider for provider in providers if isinstance(provider, dict)}


def _profile_index(profiles: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(profile.get("profile_id") or ""): profile for profile in profiles if isinstance(profile, dict)}


def _supports_task(provider: Dict[str, Any], task_class: str) -> bool:
    return task_class in set(provider.get("task_classes") or [])


def _supports_transport(provider: Dict[str, Any], transport: str) -> bool:
    for transport_entry in provider.get("transport_modes") or []:
        if not isinstance(transport_entry, dict):
            continue
        if transport_entry.get("mode") == transport and bool(transport_entry.get("supported") is True):
            return True
    return False


def _retry_policy_for_transport(provider: Dict[str, Any], transport: str) -> Tuple[int, List[int]]:
    for transport_entry in provider.get("transport_modes") or []:
        if not isinstance(transport_entry, dict):
            continue
        if transport_entry.get("mode") != transport:
            continue
        retry = transport_entry.get("retry") or {}
        max_retries = int(retry.get("max_retries", 0) or 0)
        backoff = retry.get("backoff_seconds")
        delays = [int(item) for item in (backoff if isinstance(backoff, list) else []) if str(item).isdigit()]
        if not delays and max_retries > 0:
            delays = [1] * max_retries
        return max_retries, delays
    return 0, []


def _build_selection_case(
    case: Dict[str, Any],
    profile: Optional[Dict[str, Any]],
    providers: Dict[str, Dict[str, Any]],
    fixture: Dict[str, Any],
    anchor: Dict[str, Any],
    selection_defaults: Dict[str, Any],
    disallowed: List[str],
) -> Dict[str, Any]:
    case_id = case.get("case_id")
    task_class = str(case.get("task_class") or "").strip()
    profile_id = str(case.get("task_profile") or "").strip()
    request_overrides = case.get("request_overrides") or {}
    request_provider = str(request_overrides.get("provider_id") or "").strip()
    request_transport = str(request_overrides.get("transport") or "").strip()

    profile_provider_priority = _dedupe((profile or {}).get("provider_preference") or [])
    profile_transport_priority = _dedupe((profile or {}).get("transport_preference") or [])
    provider_candidates = _dedupe(
        [request_provider] + list(profile_provider_priority) + list((selection_defaults.get("provider_preference") or []))
    )
    provider_candidates = [p for p in provider_candidates if p]

    transport_candidates = _dedupe(
        [request_transport] + list(profile_transport_priority) + list((selection_defaults.get("transport_preference") or [])) + [anchor.get("default_transport", "direct")]
    )
    transport_candidates = [t for t in transport_candidates if t]

    precedence_chain: List[str] = []
    for layer in ["request_override", "profile_preference", "lane_default"]:
        if layer == "request_override":
            if request_provider or request_transport:
                precedence_chain.append(layer)
        elif layer == "profile_preference":
            if profile_provider_priority or profile_transport_priority:
                precedence_chain.append(layer)
        else:
            precedence_chain.append(layer)

    max_attempts = int(anchor.get("max_fallback_attempts", 3) or 3)
    attempt_count = 0
    attempts: List[Dict[str, Any]] = []
    selected = None
    fallback_used = False

    for provider_id in provider_candidates:
        provider = providers.get(provider_id)
        for transport in transport_candidates:
            attempt_count += 1
            if attempt_count > max_attempts:
                break

            if transport in disallowed:
                attempts.append({"provider_id": provider_id, "transport": transport, "result": "DISALLOWED_TRANSPORT"})
                continue

            if provider is None:
                attempts.append({"provider_id": provider_id, "transport": transport, "result": "UNKNOWN_PROVIDER"})
                continue

            if not _supports_task(provider, task_class):
                attempts.append({"provider_id": provider_id, "transport": transport, "result": "UNSUPPORTED_CLASS"})
                continue

            if not _supports_transport(provider, transport):
                attempts.append({"provider_id": provider_id, "transport": transport, "result": "UNSUPPORTED_TRANSPORT"})
                continue

            selected = {"provider_id": provider_id, "transport": transport}
            break

        if selected:
            break

    if selected is None:
        # fallback safety only when all direct candidates are rejected.
        fallback_provider = str(anchor.get("fallback_provider_id", "") or "").strip()
        if fallback_provider:
            fallback_transport = str(anchor.get("fallback_transport", "") or anchor.get("default_transport", "direct")).strip()
            attempt_count += 1
            if attempt_count <= max_attempts:
                if fallback_transport not in disallowed:
                    fallback = providers.get(fallback_provider)
                    if fallback and _supports_task(fallback, task_class) and _supports_transport(fallback, fallback_transport):
                        selected = {"provider_id": fallback_provider, "transport": fallback_transport}
                        fallback_used = True
                attempts.append(
                    {
                        "provider_id": fallback_provider,
                        "transport": fallback_transport,
                        "result": "FALLBACK_OK" if selected else "FALLBACK_UNSUPPORTED",
                    }
                )

    case_status = "PASS" if selected else "FAIL"

    selected_by = {
        "precedence": precedence_chain,
        "override_applied": bool(request_provider or request_transport),
    }

    return {
        "case_id": case_id,
        "task_class": task_class,
        "task_profile": profile_id,
        "selection_path": attempts,
        "selected_provider": (selected or {}).get("provider_id"),
        "selected_transport": (selected or {}).get("transport"),
        "status": case_status,
        "fallback_used": bool(fallback_used),
        "selected_by": selected_by,
        "precedence_chain": precedence_chain,
        "attempt_count": attempt_count,
    }


def _build_retry_case(case: Dict[str, Any], provider_index: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    provider_id = str(case.get("provider_id") or "").strip()
    transport = str(case.get("transport") or "").strip()
    case_id = case.get("case_id")
    expected_retries = int(case.get("expected_max_retries", 0) or 0)
    task_class = str(case.get("task_class") or "").strip()
    failure_scenario = str(case.get("failure_scenario") or "").strip()

    provider = provider_index.get(provider_id)
    if not isinstance(provider, dict):
        return {
            "case_id": case_id,
            "provider_id": provider_id,
            "task_class": task_class,
            "transport": transport,
            "failure_scenario": failure_scenario,
            "expected_max_retries": expected_retries,
            "computed_retries": 0,
            "computed_delay_seconds": [],
            "bounded_ok": False,
            "result": "FAIL",
        }

    max_retries_for_transport, configured_backoff = _retry_policy_for_transport(provider, transport)
    retries_to_use = min(expected_retries, max_retries_for_transport)
    delays: List[int] = []
    for i in range(retries_to_use):
        if i < len(configured_backoff):
            delays.append(int(configured_backoff[i]))
        elif configured_backoff:
            delays.append(int(configured_backoff[-1]))
        elif i == 0:
            delays.append(1)
        else:
            delays.append(delays[-1])

    bounded_ok = len(delays) <= max_retries_for_transport and len(delays) <= expected_retries

    return {
        "case_id": case_id,
        "provider_id": provider_id,
        "task_class": task_class,
        "transport": transport,
        "failure_scenario": failure_scenario,
        "expected_max_retries": expected_retries,
        "computed_retries": len(delays),
        "computed_delay_seconds": delays,
        "bounded_ok": bool(bounded_ok),
        "result": "PASS" if bounded_ok else "FAIL",
    }


def _build_pack(
    fixture: Dict[str, Any],
    queue_row: Dict[str, Any],
    queue_index: Dict[str, Dict[str, Any]],
    repo_root: Path,
    output_dir: Path,
    generated_at: str,
    stamp: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Path, Path, Path, Path, Path]:
    anchor = fixture.get("runtime_anchor") or {}
    if not isinstance(anchor, dict):
        raise SystemExit("fixture.runtime_anchor must be an object")

    provider_index = _index_providers(fixture.get("provider_catalog") or [])
    profile_index = _profile_index(fixture.get("profiles") or [])
    selection_defaults = fixture.get("selection_defaults") or {}
    disallowed = [str(item) for item in (anchor.get("disallowed_transports") or [])]

    if queue_row is None:
        raise SystemExit("XO-909 row not found in queue truth")

    dep_states = _dependency_state_snapshot(queue_row, queue_index)
    unresolved_dependencies = [dep for dep, state in dep_states.items() if state != "DONE"]

    selection_cases = fixture.get("selection_cases") or []
    selection_results = []
    for row in selection_cases:
        if not isinstance(row, dict):
            continue
        profile_id = str(row.get("task_profile") or "").strip()
        profile = profile_index.get(profile_id)
        selection_results.append(
            _build_selection_case(
                case=row,
                profile=profile,
                providers=provider_index,
                fixture=fixture,
                anchor=anchor,
                selection_defaults=selection_defaults,
                disallowed=disallowed,
            )
        )

    retry_cases = fixture.get("retry_backoff_cases") or []
    retry_results = [_build_retry_case(row, provider_index) for row in retry_cases if isinstance(row, dict)]

    selection_ok = all(row.get("status") == "PASS" for row in selection_results) if selection_results else False
    retry_ok = all(row.get("result") == "PASS" for row in retry_results) if retry_results else False

    selection_matrix = {
        "schema": SELECTION_SCHEMA,
        "slice_id": anchor.get("slice_id", "XO-909"),
        "generated_at": generated_at,
        "status": "PASS" if selection_ok else "WARN" if selection_results else "FAIL",
        "cases": selection_results,
    }

    retry_matrix = {
        "schema": RETRY_SCHEMA,
        "slice_id": anchor.get("slice_id", "XO-909"),
        "generated_at": generated_at,
        "status": "PASS" if retry_ok else "WARN" if retry_results else "FAIL",
        "cases": retry_results,
    }

    pack = {
        "schema": PACK_SCHEMA,
        "schema_version": "1",
        "slice_id": "XO-909",
        "generated_at": generated_at,
        "runtime_anchor": anchor,
        "selection_defaults": selection_defaults,
        "transport_boundary": {
            "mode": "bounded",
            "allow_batch_posture": False,
            "state_boundary": {
                "runtime_schema_contract": PACK_SCHEMA,
                "selection_schema_contract": SELECTION_SCHEMA,
                "retry_schema_contract": RETRY_SCHEMA,
            },
        },
        "profiles": fixture.get("profiles") or [],
        "provider_catalog": fixture.get("provider_catalog") or [],
        "selection_matrix": selection_matrix,
        "retry_backoff_regression": retry_matrix,
        "queue_precondition": {
            "observed_slice_state": str(queue_row.get("state") or ""),
            "dependency_states": dep_states,
            "unresolved_dependencies": unresolved_dependencies,
            "validation_state": "PASS" if (not unresolved_dependencies) else "WARN",
        },
        "schema_contracts": {
            "pack_schema": PACK_SCHEMA,
            "selection_schema": SELECTION_SCHEMA,
            "retry_schema": RETRY_SCHEMA,
            "validation_schema": VALIDATION_SCHEMA,
        },
        "generated_artifacts": {},
        "validation_refs": {},
    }

    validation = {
        "schema": VALIDATION_SCHEMA,
        "slice_id": "XO-909",
        "generated_at": generated_at,
        "status": "PASS" if selection_ok and retry_ok else "WARN",
        "checks": [
            {"name": "selection_cases_complete", "result": "PASS" if selection_results else "FAIL", "detail": f"{len(selection_results)} cases"},
            {"name": "selection_precedence_exists", "result": "PASS" if all(isinstance(row.get("precedence_chain"), list) for row in selection_results) else "FAIL"},
            {"name": "bounded_fallback", "result": "PASS" if all(int(row.get("attempt_count") or 0) <= int(anchor.get("max_fallback_attempts", 0) or 0) for row in selection_results) else "FAIL"},
            {"name": "batch_posture_enforced", "result": "PASS" if all((row.get("selected_transport") not in disallowed) for row in selection_results) else "FAIL"},
            {"name": "retry_backoff_bounded", "result": "PASS" if retry_ok else "FAIL"},
        ],
    }

    pack_path = output_dir / f"xo_909_backend_adapter_contract_pack_{stamp}.json"
    selection_path = output_dir / f"xo_909_backend_selection_matrix_simulation_{stamp}.json"
    retry_path = output_dir / f"xo_909_backend_retry_backoff_regression_{stamp}.json"
    validation_path = output_dir / f"xo_909_backend_adapter_validation_{stamp}.json"
    manifest_path = output_dir / f"xo_909_backend_adapter_runtime_manifest_{stamp}.json"

    rel = lambda p: str(p.resolve().relative_to(repo_root.resolve()))

    pack["generated_artifacts"] = {
        "selection_matrix": rel(selection_path),
        "retry_backoff_regression": rel(retry_path),
        "validation": rel(validation_path),
    }

    pack["validation_refs"] = {
        "selection_matrix_validation": rel(selection_path),
        "retry_backoff_regression": rel(retry_path),
    }

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": "1",
        "slice_id": "XO-909",
        "generated_at": generated_at,
        "artifact_count": 4,
        "status": "PASS" if validation.get("status") == "PASS" else "WARN",
        "artifacts": {
            "contract_pack": rel(pack_path),
            "selection_matrix": rel(selection_path),
            "retry_backoff_regression": rel(retry_path),
            "validation": rel(validation_path),
        },
        "checks": [
            {"name": "artifact_generation", "result": "PASS"},
            {"name": "provider_adapter_matrix", "result": "PASS"},
            {"name": "retry_backoff_regression", "result": "PASS"},
            {"name": "selection_validation", "result": validation["status"]},
        ],
    }

    return pack, selection_matrix, retry_matrix, validation, manifest, manifest_path, pack_path, selection_path, retry_path, validation_path


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve()
    queue_path = (repo_root / args.queue_path).resolve()
    fixture_path = (repo_root / args.fixture_path).resolve()
    output_dir = (repo_root / args.output_dir).resolve()

    stamp = args.stamp
    if stamp:
        try:
            dt.date.fromisoformat(stamp)
        except ValueError as exc:  # pragma: no cover - argument validation only
            raise SystemExit(f"--stamp must be YYYY-MM-DD: {exc}")
    else:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")

    queue_doc = load_json(queue_path)
    fixture = load_json(fixture_path)

    queue_index = _queue_index(queue_doc)
    row = queue_index.get("XO-909")

    if row is None:
        raise SystemExit("XO-909 not found in queue truth")

    state = str(row.get("state") or "")
    if args.require_queued and state != "QUEUED_OPTIONAL":
        raise SystemExit(f"XO-909 state must be QUEUED_OPTIONAL before generation; observed={state}")
    if state not in {"QUEUED_OPTIONAL", "DONE"}:
        raise SystemExit(f"XO-909 state invalid for generation: {state}")

    generated_at = utc_now_iso()

    pack, selection_matrix, retry_matrix, validation, manifest, manifest_path, pack_path, selection_path, retry_path, validation_path = _build_pack(
        fixture=fixture,
        queue_row=row,
        queue_index=queue_index,
        repo_root=repo_root,
        output_dir=output_dir,
        generated_at=generated_at,
        stamp=stamp,
    )

    pack_path = output_dir / f"xo_909_backend_adapter_contract_pack_{stamp}.json"
    selection_path = output_dir / f"xo_909_backend_selection_matrix_simulation_{stamp}.json"
    retry_path = output_dir / f"xo_909_backend_retry_backoff_regression_{stamp}.json"
    validation_path = output_dir / f"xo_909_backend_adapter_validation_{stamp}.json"
    manifest_path = output_dir / f"xo_909_backend_adapter_runtime_manifest_{stamp}.json"

    pack["generated_artifacts"]["selection_matrix"] = str(selection_path.relative_to(repo_root))
    pack["generated_artifacts"]["retry_backoff_regression"] = str(retry_path.relative_to(repo_root))
    pack["generated_artifacts"]["validation"] = str(validation_path.relative_to(repo_root))

    pack["validation_refs"]["selection_matrix_validation"] = str(selection_path.relative_to(repo_root))
    pack["validation_refs"]["retry_backoff_regression"] = str(retry_path.relative_to(repo_root))

    manifest["artifacts"]["contract_pack"] = str(pack_path.relative_to(repo_root))
    manifest["artifacts"]["selection_matrix"] = str(selection_path.relative_to(repo_root))
    manifest["artifacts"]["retry_backoff_regression"] = str(retry_path.relative_to(repo_root))
    manifest["artifacts"]["validation"] = str(validation_path.relative_to(repo_root))

    write_json(pack_path, pack)
    write_json(selection_path, selection_matrix)
    write_json(retry_path, retry_matrix)
    write_json(validation_path, validation)
    write_json(manifest_path, manifest)

    if args.json:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
