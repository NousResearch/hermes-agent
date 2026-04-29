"""Hermes-owned Feishu Image2 worker entrypoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from gateway.image2_browser_preflight import BROWSER_STATE_ENV, evaluate_browser_preflight
from gateway.image2_candidate_gate import evaluate_candidate_gate
from gateway.image2_delivery_contract import evaluate_delivery_contract
from gateway.image2_feishu_delivery import send_feishu_files_from_print_package, send_feishu_image_from_contract
from gateway.image2_generation import probe_opencli_browser_state, run_opencli_generation
from gateway.image2_print import package_flat_print_outputs
from gateway.image2_review_gate import evaluate_review_gate
from gateway.image2_store import Image2JobStore
from gateway.image2_visual_reviewer import review_candidate_image

LIVE_ENABLE_ENV = "IMAGE2_WORKER_LIVE_ENABLED"
BROWSER_ENV_OPTIONS = ("OPENCLI_CDP_URL", "CHATGPT_BROWSER_CDP_URL", "OPENCLI_CHROME_CDP_GUIDANCE", BROWSER_STATE_ENV)
REVIEWER_ENV_OPTIONS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "IMAGE2_REVIEWER_PROVIDER")

Generator = Callable[..., dict[str, Any]]
Reviewer = Callable[..., dict[str, Any]]
DeliverySender = Callable[..., dict[str, Any]]
PrintPackager = Callable[..., dict[str, Any]]
FileDeliverySender = Callable[..., dict[str, Any]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hermes-owned Image2 worker")
    parser.add_argument("--db", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--task-id", required=True)
    return parser


def _enabled(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _has_any(environ: Mapping[str, str], keys: Sequence[str]) -> bool:
    return any(str(environ.get(key) or "").strip() for key in keys)


def missing_live_preflight(environ: Mapping[str, str]) -> list[str]:
    """Return missing non-secret live-generation gates without exposing values."""
    missing: list[str] = []
    if not _enabled(environ.get(LIVE_ENABLE_ENV)):
        missing.append(f"{LIVE_ENABLE_ENV}=1")
    if not _has_any(environ, BROWSER_ENV_OPTIONS):
        missing.append("OPENCLI_CDP_URL or CHATGPT_BROWSER_CDP_URL")
    if not _has_any(environ, REVIEWER_ENV_OPTIONS):
        missing.append("GEMINI_API_KEY or GOOGLE_API_KEY or IMAGE2_REVIEWER_PROVIDER")
    return missing


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _prompt_artifacts(job: Mapping[str, Any], runtime_root: Path) -> dict[str, Any]:
    task_id = str(job.get("task_id") or "")
    job_dir = Path(str(job.get("job_dir") or runtime_root / task_id)).expanduser()
    return {
        "job_dir": job_dir,
        "prompt_txt": job_dir / "prompt.txt",
        "compiled_prompt_json": job_dir / "compiled_prompt.json",
        "brief_json": job_dir / "brief.json",
        "message_json": job_dir / "message.json",
    }


def _read_artifacts(job: Mapping[str, Any], runtime_root: Path) -> dict[str, Any]:
    artifacts = _prompt_artifacts(job, runtime_root)
    prompt_path: Path = artifacts["prompt_txt"]
    if not prompt_path.is_file():
        raise FileNotFoundError(str(prompt_path))
    prompt_text = prompt_path.read_text(encoding="utf-8")
    if not prompt_text.strip():
        raise ValueError(f"empty prompt artifact: {prompt_path}")
    compiled = _load_json(artifacts["compiled_prompt_json"])
    message = _load_json(artifacts["message_json"])
    return {
        "job_dir": str(artifacts["job_dir"]),
        "prompt_text": prompt_text,
        "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "compiled_prompt": compiled,
        "message": message,
        "prompt_artifacts": {key: str(value) for key, value in artifacts.items() if key != "job_dir"},
    }


def _source_files(job_dir: Path) -> list[Any]:
    path = job_dir / "source_manifest.json"
    if not path.is_file():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, list) else [value]


def _write_worker_result(job_dir: Path, payload: Mapping[str, Any]) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "worker_result.json").write_text(_safe_json(dict(payload)), encoding="utf-8")


def _terminal_failure(
    *,
    store: Image2JobStore,
    task_id: str,
    worker_id: str,
    job_dir: Path,
    reason: str,
    last_error: str,
    extra: Mapping[str, Any] | None = None,
    exit_code: int = 3,
) -> dict[str, Any]:
    row = store.mark_failed_final(task_id=task_id, worker_id=worker_id, last_error=last_error) or {}
    result = {
        "task_id": task_id,
        "worker_id": worker_id,
        "status": "failed_final",
        "reason": reason,
        "last_error": last_error,
        "exit_code": exit_code,
        "db_status": row.get("status"),
    }
    if extra:
        result.update(dict(extra))
    _write_worker_result(job_dir, result)
    return result


def _terminal_success(
    *,
    store: Image2JobStore,
    task_id: str,
    worker_id: str,
    job_dir: Path,
    payload: Mapping[str, Any],
    reason: str = "Feishu native image read-back verified",
) -> dict[str, Any]:
    row = store.mark_readback_verified(task_id=task_id, worker_id=worker_id) or {}
    result = {
        "task_id": task_id,
        "worker_id": worker_id,
        "status": "readback_verified",
        "reason": reason,
        "exit_code": 0,
        "db_status": row.get("status"),
    }
    result.update(dict(payload))
    _write_worker_result(job_dir, result)
    return result


def _ensure_browser_state(job_dir: Path, env: Mapping[str, str]) -> None:
    if any((job_dir / name).is_file() for name in ("browser_state.json", "browser_preflight_input.json")):
        return
    env_path = str(env.get(BROWSER_STATE_ENV) or "").strip()
    if env_path and Path(env_path).expanduser().is_file():
        return
    if _enabled(env.get("IMAGE2_SKIP_OPENCLI_BROWSER_PROBE")):
        return
    probe_opencli_browser_state(job_dir=job_dir, environ=env, timeout=int(env.get("IMAGE2_BROWSER_PROBE_TIMEOUT") or 60))


def _run_candidate_gate(job_dir: Path, claimed: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_candidate_gate(job_dir=job_dir, generated_after=claimed.get("claimed_at") or claimed.get("created_at"))


def run_worker(
    *,
    db_path: Path,
    runtime_root: Path,
    task_id: str,
    worker_id: str,
    environ: Mapping[str, str] | None = None,
    generator: Generator | None = None,
    reviewer: Reviewer | None = None,
    delivery_sender: DeliverySender | None = None,
    print_packager: PrintPackager | None = None,
    file_delivery_sender: FileDeliverySender | None = None,
) -> dict[str, Any]:
    """Claim, generate, gate, review, deliver/read-back, or package approved print finals."""
    env = dict(environ if environ is not None else os.environ)
    runtime = Path(runtime_root)
    store = Image2JobStore(db_path=Path(db_path), runtime_root=runtime)
    claimed = store.claim_task(task_id=str(task_id), worker_id=str(worker_id))
    if not claimed:
        return {"task_id": str(task_id), "worker_id": str(worker_id), "status": "not_claimed", "reason": "task_not_claimable", "exit_code": 1}

    fallback_job_dir = runtime / str(task_id)
    try:
        artifact_info = _read_artifacts(claimed, runtime)
    except Exception as exc:
        return _terminal_failure(
            store=store,
            task_id=str(task_id),
            worker_id=str(worker_id),
            job_dir=Path(str(claimed.get("job_dir") or fallback_job_dir)),
            reason="missing_prompt_artifact",
            last_error=f"missing_prompt_artifact: {exc}",
            exit_code=2,
        )

    job_dir = Path(str(artifact_info["job_dir"]))
    prompt_text = str(artifact_info["prompt_text"])
    prompt_common = {
        "prompt_sha256": artifact_info["prompt_sha256"],
        "prompt_excerpt": prompt_text[:240],
        "prompt_artifacts": artifact_info["prompt_artifacts"],
    }

    message_for_print = dict(artifact_info.get("message") or _load_json(job_dir / "message.json"))
    print_request = message_for_print.get("print_request") if isinstance(message_for_print.get("print_request"), Mapping) else None
    if print_request:
        if not env.get("FEISHU_APP_ID") or not env.get("FEISHU_APP_SECRET"):
            return _terminal_failure(
                store=store,
                task_id=str(task_id),
                worker_id=str(worker_id),
                job_dir=job_dir,
                reason="delivery_preflight_missing",
                last_error="delivery_preflight_missing: FEISHU_APP_ID / FEISHU_APP_SECRET are required",
                extra={"print_request": print_request, **prompt_common},
                exit_code=6,
            )
        approved_path = Path(str(print_request.get("approved_image_path") or ""))
        expected_approved_sha = str(print_request.get("approved_image_sha256") or "")
        if not approved_path.is_file() or not expected_approved_sha:
            return _terminal_failure(
                store=store,
                task_id=str(task_id),
                worker_id=str(worker_id),
                job_dir=job_dir,
                reason="print_approved_source_missing",
                last_error="print_approved_source_missing: approved image path or sha256 missing",
                extra={"print_request": print_request, **prompt_common},
                exit_code=6,
            )
        actual_approved_sha = hashlib.sha256(approved_path.read_bytes()).hexdigest()
        if actual_approved_sha != expected_approved_sha:
            return _terminal_failure(
                store=store,
                task_id=str(task_id),
                worker_id=str(worker_id),
                job_dir=job_dir,
                reason="print_approved_source_sha_mismatch",
                last_error="print_approved_source_sha_mismatch: approved image bytes changed after preview read-back",
                extra={"print_request": print_request, "approved_sha256_actual": actual_approved_sha, **prompt_common},
                exit_code=6,
            )
        store.mark_status(task_id=str(task_id), status="packaging_print_files", worker_id=str(worker_id), last_error=None)
        try:
            packager = print_packager or package_flat_print_outputs
            package_result = packager(
                job_dir=job_dir,
                approved_image_path=approved_path,
                spec=dict(print_request.get("spec") or {}),
                environ=env,
            )
        except Exception as exc:  # noqa: BLE001
            return _terminal_failure(
                store=store,
                task_id=str(task_id),
                worker_id=str(worker_id),
                job_dir=job_dir,
                reason="print_package_failed",
                last_error=f"print_package_failed: {exc.__class__.__name__}: {exc}",
                extra={"print_request": print_request, **prompt_common},
                exit_code=6,
            )
        (job_dir / "print" / "reports").mkdir(parents=True, exist_ok=True)
        (job_dir / "print" / "reports" / "package_report.json").write_text(_safe_json(package_result), encoding="utf-8")
        if str(package_result.get("approved_sha256") or expected_approved_sha) != expected_approved_sha:
            return _terminal_failure(
                store=store,
                task_id=str(task_id),
                worker_id=str(worker_id),
                job_dir=job_dir,
                reason="print_package_source_sha_mismatch",
                last_error="print_package_source_sha_mismatch: package did not use approved preview bytes",
                extra={"print_request": print_request, "print_package": package_result, **prompt_common},
                exit_code=6,
            )
        if package_result.get("status") != "pass":
            return _terminal_failure(
                store=store,
                task_id=str(task_id),
                worker_id=str(worker_id),
                job_dir=job_dir,
                reason="print_package_rejected",
                last_error="print_package_rejected: " + str(package_result.get("reason") or package_result.get("status")),
                extra={"print_request": print_request, "print_package": package_result, **prompt_common},
                exit_code=6,
            )
        files = [
            {"path": str(package_result.get("psd_path") or ""), "file_name": Path(str(package_result.get("psd_path") or "")).name},
            {"path": str(package_result.get("pdf_path") or ""), "file_name": Path(str(package_result.get("pdf_path") or "")).name},
        ]
        try:
            sender = file_delivery_sender or send_feishu_files_from_print_package
            reply_to = str(message_for_print.get("thread_id") or message_for_print.get("root_id") or message_for_print.get("parent_id") or message_for_print.get("upper_message_id") or message_for_print.get("feishu_message_id") or "")
            delivery_report = sender(files=files, chat_id=str(message_for_print.get("chat_id") or ""), reply_to=reply_to, environ=env)
        except Exception as exc:  # noqa: BLE001
            return _terminal_failure(
                store=store,
                task_id=str(task_id),
                worker_id=str(worker_id),
                job_dir=job_dir,
                reason="print_delivery_failed",
                last_error=f"print_delivery_failed: {exc.__class__.__name__}: {exc}",
                extra={"print_request": print_request, "print_package": package_result, **prompt_common},
                exit_code=6,
            )
        (job_dir / "print" / "reports" / "delivery_report.json").write_text(_safe_json(delivery_report), encoding="utf-8")
        if delivery_report.get("verified") is not True:
            return _terminal_failure(
                store=store,
                task_id=str(task_id),
                worker_id=str(worker_id),
                job_dir=job_dir,
                reason="print_delivery_readback_failed",
                last_error="print_delivery_readback_failed: expected verified file read-backs",
                extra={"print_request": print_request, "print_package": package_result, "print_delivery": delivery_report, **prompt_common},
                exit_code=6,
            )
        return _terminal_success(
            store=store,
            task_id=str(task_id),
            worker_id=str(worker_id),
            job_dir=job_dir,
            payload={"print_request": print_request, "print_package": package_result, "print_delivery": delivery_report, **prompt_common},
            reason="Feishu print files read-back verified",
        )

    missing = missing_live_preflight(env)
    if missing:
        return _terminal_failure(
            store=store,
            task_id=str(task_id),
            worker_id=str(worker_id),
            job_dir=job_dir,
            reason="fail_closed_missing_generation_preflight",
            last_error="fail_closed_missing_generation_preflight: " + ", ".join(missing),
            extra={"missing_preflight": missing, **prompt_common},
            exit_code=3,
        )

    _ensure_browser_state(job_dir, env)
    browser_preflight = evaluate_browser_preflight(job_dir=job_dir, environ=env)
    if browser_preflight.get("status") != "pass":
        reasons = ", ".join(str(reason) for reason in browser_preflight.get("reasons", []))
        return _terminal_failure(
            store=store,
            task_id=str(task_id),
            worker_id=str(worker_id),
            job_dir=job_dir,
            reason="browser_preflight_failed",
            last_error="browser_preflight_failed: " + reasons,
            extra={"browser_preflight": browser_preflight, **prompt_common},
            exit_code=4,
        )

    candidate_gate = _run_candidate_gate(job_dir, claimed)
    generation_result: dict[str, Any] | None = None
    if candidate_gate.get("status") == "no_candidates":
        if not env.get("FEISHU_APP_ID") or not env.get("FEISHU_APP_SECRET"):
            return _terminal_failure(
                store=store,
                task_id=str(task_id),
                worker_id=str(worker_id),
                job_dir=job_dir,
                reason="delivery_preflight_missing",
                last_error="delivery_preflight_missing: FEISHU_APP_ID / FEISHU_APP_SECRET are required before generation",
                extra={"browser_preflight": browser_preflight, "candidate_gate": candidate_gate, "delivery_preflight": {"status": "fail", "missing": ["FEISHU_APP_ID", "FEISHU_APP_SECRET"]}, **prompt_common},
                exit_code=6,
            )
        store.mark_status(task_id=str(task_id), status="generating", worker_id=str(worker_id), last_error=None)
        generation_runner = generator or run_opencli_generation
        generation_result = generation_runner(
            job_dir=job_dir,
            prompt_text=prompt_text,
            environ=env,
            source_files=_source_files(job_dir),
        )
        if not (job_dir / "generation_result.json").is_file():
            (job_dir / "generation_result.json").write_text(_safe_json(generation_result), encoding="utf-8")
        candidate_gate = _run_candidate_gate(job_dir, claimed)

    if candidate_gate.get("status") == "no_candidates":
        return _terminal_failure(
            store=store,
            task_id=str(task_id),
            worker_id=str(worker_id),
            job_dir=job_dir,
            reason="candidate_gate_no_candidates",
            last_error="candidate_gate_no_candidates: generation produced no fresh candidate image",
            extra={"browser_preflight": browser_preflight, "generation_result": generation_result or {}, "candidate_gate": candidate_gate, **prompt_common},
            exit_code=5,
        )
    if candidate_gate.get("status") == "rejected":
        rejected_reasons = sorted({reason for decision in candidate_gate.get("decisions", []) for reason in decision.get("reasons", [])})
        return _terminal_failure(
            store=store,
            task_id=str(task_id),
            worker_id=str(worker_id),
            job_dir=job_dir,
            reason="candidate_gate_rejected",
            last_error="candidate_gate_rejected: " + ", ".join(rejected_reasons),
            extra={"browser_preflight": browser_preflight, "generation_result": generation_result or {}, "candidate_gate": candidate_gate, **prompt_common},
            exit_code=5,
        )
    if candidate_gate.get("status") != "pass":
        return _terminal_failure(
            store=store,
            task_id=str(task_id),
            worker_id=str(worker_id),
            job_dir=job_dir,
            reason="candidate_gate_error",
            last_error=f"candidate_gate_error: unexpected status {candidate_gate.get('status')}",
            extra={"browser_preflight": browser_preflight, "candidate_gate": candidate_gate, **prompt_common},
            exit_code=5,
        )

    accepted = candidate_gate.get("accepted")
    review_runner = reviewer or review_candidate_image
    review_result = review_runner(job_dir=job_dir, candidate=accepted, prompt_text=prompt_text, environ=env)
    if not (job_dir / "review_result.json").is_file():
        (job_dir / "review_result.json").write_text(_safe_json(review_result), encoding="utf-8")
    review_gate = evaluate_review_gate(job_dir=job_dir, candidate=accepted, review_result=review_result)
    if review_gate.get("status") != "pass":
        reasons = ", ".join(str(reason) for reason in review_gate.get("reasons", []))
        return _terminal_failure(
            store=store,
            task_id=str(task_id),
            worker_id=str(worker_id),
            job_dir=job_dir,
            reason="review_gate_rejected",
            last_error="review_gate_rejected: " + reasons,
            extra={"browser_preflight": browser_preflight, "generation_result": generation_result or {}, "candidate_gate": candidate_gate, "review_gate": review_gate, **prompt_common},
            exit_code=5,
        )

    message = dict(artifact_info.get("message") or _load_json(job_dir / "message.json"))
    delivery_contract = evaluate_delivery_contract(job_dir=job_dir, message=message, candidate_gate=candidate_gate, review_gate=review_gate)
    if delivery_contract.get("status") != "ready_to_send":
        reasons = ", ".join(str(reason) for reason in delivery_contract.get("reasons", []))
        return _terminal_failure(
            store=store,
            task_id=str(task_id),
            worker_id=str(worker_id),
            job_dir=job_dir,
            reason="delivery_contract_rejected",
            last_error="delivery_contract_rejected: " + reasons,
            extra={"browser_preflight": browser_preflight, "generation_result": generation_result or {}, "candidate_gate": candidate_gate, "review_gate": review_gate, "delivery_contract": delivery_contract, **prompt_common},
            exit_code=5,
        )

    if not env.get("FEISHU_APP_ID") or not env.get("FEISHU_APP_SECRET"):
        return _terminal_failure(
            store=store,
            task_id=str(task_id),
            worker_id=str(worker_id),
            job_dir=job_dir,
            reason="delivery_preflight_missing",
            last_error="delivery_preflight_missing: FEISHU_APP_ID / FEISHU_APP_SECRET are required",
            extra={"browser_preflight": browser_preflight, "generation_result": generation_result or {}, "candidate_gate": candidate_gate, "review_gate": review_gate, "delivery_contract": delivery_contract, **prompt_common},
            exit_code=6,
        )

    store.mark_status(task_id=str(task_id), status="uploading_to_feishu", worker_id=str(worker_id), last_error=None)
    try:
        delivery_runner = delivery_sender or send_feishu_image_from_contract
        delivery_readback = delivery_runner(
            image_path=Path(str(delivery_contract.get("image_path") or "")),
            chat_id=str(delivery_contract.get("chat_id") or ""),
            reply_to=str(delivery_contract.get("reply_to") or ""),
            candidate_sha256=str(delivery_contract.get("image_sha256") or ""),
            environ=env,
        )
    except Exception as exc:  # noqa: BLE001 - convert to terminal safe error
        return _terminal_failure(
            store=store,
            task_id=str(task_id),
            worker_id=str(worker_id),
            job_dir=job_dir,
            reason="delivery_failed",
            last_error=f"delivery_failed: {exc.__class__.__name__}: {exc}",
            extra={"browser_preflight": browser_preflight, "generation_result": generation_result or {}, "candidate_gate": candidate_gate, "review_gate": review_gate, "delivery_contract": delivery_contract, **prompt_common},
            exit_code=6,
        )

    (job_dir / "delivery_readback.json").write_text(_safe_json(delivery_readback), encoding="utf-8")
    if delivery_readback.get("verified") is not True or delivery_readback.get("readback_msg_type") != "image":
        return _terminal_failure(
            store=store,
            task_id=str(task_id),
            worker_id=str(worker_id),
            job_dir=job_dir,
            reason="delivery_readback_failed",
            last_error="delivery_readback_failed: expected verified image read-back",
            extra={"delivery_readback": delivery_readback, "browser_preflight": browser_preflight, "generation_result": generation_result or {}, "candidate_gate": candidate_gate, "review_gate": review_gate, "delivery_contract": delivery_contract, **prompt_common},
            exit_code=6,
        )

    return _terminal_success(
        store=store,
        task_id=str(task_id),
        worker_id=str(worker_id),
        job_dir=job_dir,
        payload={
            "browser_preflight": browser_preflight,
            "generation_result": generation_result or {},
            "candidate_gate": candidate_gate,
            "review_gate": review_gate,
            "delivery_contract": delivery_contract,
            "delivery_readback": delivery_readback,
            **prompt_common,
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_worker(
        db_path=Path(args.db),
        runtime_root=Path(args.runtime_root),
        worker_id=str(args.worker_id),
        task_id=str(args.task_id),
        environ=os.environ,
    )
    print(_safe_json(result), file=sys.stderr)
    return int(result.get("exit_code") or 0)


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
