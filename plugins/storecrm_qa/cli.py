"""Local CLI for the StoreCRM QA control plane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import Evidence, PolicyMetadata, RunnerOutcome, RunnerResult
from .store import StoreCRMQAStore, default_db_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m plugins.storecrm_qa.cli")
    parser.add_argument("--db", help="SQLite DB path. Defaults to the active Hermes profile.")
    sub = parser.add_subparsers(dest="command", required=True)

    enqueue = sub.add_parser("enqueue", help="Create a QA job with local case payloads.")
    enqueue.add_argument("--name", required=True)
    enqueue.add_argument("--tenant", required=True)
    enqueue.add_argument("--store", required=True)
    enqueue.add_argument("--risk", default="low", choices=("low", "medium", "high"))
    enqueue.add_argument("--allowed-op", action="append", default=["read"])
    enqueue.add_argument("--max-attempts", type=int, default=2)
    enqueue.add_argument("--case", action="append", required=True, help="Case name; may be repeated.")

    list_cmd = sub.add_parser("list", help="List jobs or cases.")
    list_cmd.add_argument("--cases", action="store_true")
    list_cmd.add_argument("--job-id", type=int)
    list_cmd.add_argument("--status")

    lease = sub.add_parser("lease", help="Claim the next runnable case.")
    lease.add_argument("--owner", required=True)
    lease.add_argument("--seconds", type=int, default=300)

    heartbeat = sub.add_parser("heartbeat", help="Extend a case lease.")
    heartbeat.add_argument("--case-id", type=int, required=True)
    heartbeat.add_argument("--owner", required=True)
    heartbeat.add_argument("--seconds", type=int, default=300)

    complete = sub.add_parser("complete", help="Persist a terminal runner result.")
    complete.add_argument("--case-id", type=int, required=True)
    complete.add_argument("--owner", required=True)
    complete.add_argument("--outcome", required=True, choices=[item.value for item in RunnerOutcome])
    complete.add_argument("--summary", required=True)
    complete.add_argument("--evidence", action="append", default=[])

    fail_retry = sub.add_parser("fail-retry", help="Record a failed attempt and requeue if attempts remain.")
    fail_retry.add_argument("--case-id", type=int, required=True)
    fail_retry.add_argument("--owner", required=True)
    fail_retry.add_argument("--summary", required=True)

    recover = sub.add_parser("recover-stale", help="Recover expired leases.")
    recover.set_defaults(command="recover-stale")

    report = sub.add_parser("report", help="Write a redacted local report JSON.")
    report.add_argument("--job-id", type=int, required=True)
    report.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = StoreCRMQAStore(args.db) if args.db else StoreCRMQAStore()

    if args.command == "enqueue":
        job = store.enqueue_job(
            name=args.name,
            metadata=PolicyMetadata(
                tenant_id=args.tenant,
                store_id=args.store,
                risk=args.risk,
                allowed_operations=tuple(args.allowed_op),
            ),
            max_attempts=args.max_attempts,
            cases=[{"name": name, "input": {"case": name}} for name in args.case],
        )
        _print({"job_id": job.id, "status": job.status.value, "db": str(store.db_path)})
        return 0

    if args.command == "list":
        if args.cases:
            rows = [
                {
                    "id": case.id,
                    "job_id": case.job_id,
                    "name": case.name,
                    "status": case.status.value,
                    "attempts": case.attempts,
                    "lease_owner": case.lease_owner,
                }
                for case in store.list_cases(job_id=args.job_id, status=args.status)
            ]
        else:
            rows = [
                {
                    "id": job.id,
                    "name": job.name,
                    "status": job.status.value,
                    "tenant_id": job.metadata.tenant_id,
                    "store_id": job.metadata.store_id,
                }
                for job in store.list_jobs(status=args.status)
            ]
        _print(rows)
        return 0

    if args.command == "lease":
        case = store.claim_next_case(owner=args.owner, lease_seconds=args.seconds)
        _print(None if case is None else _case_payload(case))
        return 0

    if args.command == "heartbeat":
        lease = store.heartbeat(case_id=args.case_id, owner=args.owner, lease_seconds=args.seconds)
        _print(
            {
                "case_id": lease.case_id,
                "job_id": lease.job_id,
                "owner": lease.owner,
                "expires_at": lease.expires_at.isoformat(),
                "attempt": lease.attempt,
            }
        )
        return 0

    if args.command == "complete":
        case = store.complete_case(
            case_id=args.case_id,
            owner=args.owner,
            result=RunnerResult(
                outcome=RunnerOutcome(args.outcome),
                summary=args.summary,
                evidence=tuple(Evidence(kind="note", summary=item) for item in args.evidence),
            ),
        )
        _print(_case_payload(case))
        return 0

    if args.command == "fail-retry":
        case = store.fail_for_retry(case_id=args.case_id, owner=args.owner, summary=args.summary)
        _print(_case_payload(case))
        return 0

    if args.command == "recover-stale":
        _print({"recovered": store.recover_stale_leases()})
        return 0

    if args.command == "report":
        output = store.write_report(args.job_id, Path(args.output))
        _print({"output": str(output)})
        return 0

    raise AssertionError(f"unhandled command {args.command}")


def _case_payload(case) -> dict[str, object]:
    return {
        "id": case.id,
        "job_id": case.job_id,
        "name": case.name,
        "status": case.status.value,
        "attempts": case.attempts,
        "lease_owner": case.lease_owner,
        "lease_expires_at": case.lease_expires_at.isoformat() if case.lease_expires_at else None,
    }


def _print(value: object) -> None:
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
