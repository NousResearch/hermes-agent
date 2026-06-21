"""Dry-run parallel lane runner."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .executors import LaneExecutor
from .lanes import LaneRequest, LaneResult
from .redaction import redact_text


def _sanitize_result(result: LaneResult) -> LaneResult:
    return replace(
        result,
        output=redact_text(result.output) if result.output is not None else None,
        error=redact_text(result.error) if result.error is not None else None,
        artifacts=dict(result.artifacts),
    )


def _write_artifacts(result: LaneResult, log_dir: Path) -> LaneResult:
    log_dir.mkdir(parents=True, exist_ok=True)
    safe = _sanitize_result(result)
    log_path = log_dir / f"{safe.lane_id}.log"
    result_path = log_dir / f"{safe.lane_id}.result.md"
    status_path = log_dir / f"{safe.lane_id}.status.json"

    body_parts: list[str] = []
    if safe.output:
        body_parts.append(safe.output)
    if safe.error:
        body_parts.append(f"ERROR: {safe.error}")
    log_path.write_text("\n".join(body_parts).strip() + ("\n" if body_parts else ""), encoding="utf-8")

    result_path.write_text(
        "\n".join(
            [
                f"# Lane {safe.lane_id}",
                "",
                f"agent: {safe.agent}",
                f"status: {safe.status.value}",
                f"duration_s: {safe.duration_s}",
                "",
                "## Output",
                safe.output or "",
                "",
                "## Error",
                safe.error or "",
                "",
            ]
        ),
        encoding="utf-8",
    )
    artifacts = dict(safe.artifacts)
    artifacts.update({"log": str(log_path), "result": str(result_path), "status": str(status_path)})
    safe = replace(safe, log_path=str(log_path), artifacts=artifacts)
    status_path.write_text(json.dumps(safe.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return safe


def _finalize(result: LaneResult, log_dir: Path | None) -> LaneResult:
    safe = _sanitize_result(result)
    if log_dir is None:
        return safe
    return _write_artifacts(safe, log_dir)


def run_lanes(
    requests: Sequence[LaneRequest],
    *,
    executor: LaneExecutor,
    max_workers: int = 4,
    log_dir: Path | str | None = None,
    degraded_agents: set[str] | None = None,
) -> list[LaneResult]:
    """Run dry-run/fake lanes in parallel and return results in request order."""

    degraded_agents = degraded_agents or set()
    root = Path(log_dir) if log_dir is not None else None
    results: dict[str, LaneResult] = {}
    runnable: list[LaneRequest] = []

    for req in requests:
        if req.agent in degraded_agents:
            results[req.lane_id] = _finalize(LaneResult.skipped(req, "agent degraded"), root)
        else:
            runnable.append(req)

    if runnable:
        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
            futures = {pool.submit(executor.execute, req): req for req in runnable}
            for future in as_completed(futures):
                req = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - executor failures become lane failures
                    result = LaneResult.failed(req, str(exc))
                results[req.lane_id] = _finalize(result, root)

    return [results[req.lane_id] for req in requests]
