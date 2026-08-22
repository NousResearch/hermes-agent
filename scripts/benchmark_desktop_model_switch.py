#!/usr/bin/env python3
"""Reproducible latency benchmark for Desktop's live model-switch path.

The benchmark drives the real :func:`hermes_cli.model_switch.switch_model`
pipeline against a loopback ``/v1/models`` endpoint.  The endpoint adds a
seeded delay to model the remote catalogue validation that a Desktop picker
selection used to repeat even though the gateway had just served that exact
provider/model pair.

Examples::

    python scripts/benchmark_desktop_model_switch.py measure \
        --condition before --output before.csv
    python scripts/benchmark_desktop_model_switch.py measure \
        --condition after --output after.csv
    python scripts/benchmark_desktop_model_switch.py analyse \
        --before before.csv --after after.csv --output-dir results

The ``after`` condition is intentionally unavailable until ``switch_model``
exposes the catalogue-validation fast path.  This prevents accidentally
recording two copies of the baseline under different labels.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import random
import secrets
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable, Sequence
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FOLDS = 5
SAMPLES_PER_FOLD = 12
WARMUPS = 3
SEED = 20_260_820


@dataclass(frozen=True)
class Observation:
    condition: str
    fold: int
    sample: int
    scheduled_delay_ms: float
    latency_ms: float
    catalogue_requests: int


class _CatalogueServer(ThreadingHTTPServer):
    next_delay_seconds = 0.0
    request_count = 0


class _CatalogueHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.rstrip("/") != "/v1/models":
            self.send_error(404)
            return

        self.server.request_count += 1  # type: ignore[attr-defined]
        time.sleep(self.server.next_delay_seconds)  # type: ignore[attr-defined]
        body = json.dumps(
            {
                "data": [
                    {"id": "bench-model", "object": "model"},
                    {"id": "old-model", "object": "model"},
                ]
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def _delay_schedule() -> list[list[float]]:
    """Return five independent, reproducible latency folds in milliseconds."""
    rng = random.Random(SEED)
    return [
        [rng.uniform(35.0, 95.0) for _ in range(SAMPLES_PER_FOLD)]
        for _ in range(FOLDS)
    ]


def _switch_once(base_url: str, *, use_catalogue_proof: bool) -> None:
    from hermes_cli.model_switch import switch_model
    from tui_gateway import server as gateway_server

    kwargs = {
        "raw_input": "bench-model",
        "current_provider": "openrouter",
        "current_model": "old-model",
        "current_base_url": base_url,
        "current_api_key": "benchmark-key",
        "explicit_provider": "openrouter",
    }
    supports_fast_path = "catalogue_proof" in inspect.signature(switch_model).parameters
    proof_session_id = ""
    if use_catalogue_proof:
        if not supports_fast_path:
            raise RuntimeError(
                "The after condition requires switch_model(catalogue_proof=...)."
            )
        proof = secrets.token_urlsafe(32)
        proof_session_id = f"benchmark-catalogue-proof-{proof}"
        with gateway_server._sessions_lock:
            gateway_server._sessions[proof_session_id] = {
                "model_options_catalogue": frozenset({("openrouter", "bench-model")}),
                "model_options_catalogue_at": time.monotonic(),
                "model_options_catalogue_proof": proof,
            }
        kwargs["catalogue_proof"] = proof

    # Keep the benchmark focused on the redundant remote /v1/models validation.
    # Credential resolution is deterministic, and models.dev enrichment is
    # cache-only in the served-picker path this benchmark represents.
    try:
        with (
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                return_value={
                    "api_key": "benchmark-key",
                    "api_mode": "chat_completions",
                    "base_url": base_url,
                },
            ),
            patch("hermes_cli.model_switch.get_model_capabilities", return_value=None),
            patch("hermes_cli.model_switch.get_model_info", return_value=None),
        ):
            result = switch_model(**kwargs)
    finally:
        if proof_session_id:
            with gateway_server._sessions_lock:
                gateway_server._sessions.pop(proof_session_id, None)

    if not result.success or result.new_model != "bench-model":
        raise RuntimeError(f"Benchmark switch failed: {result}")


def measure(condition: str, output: Path) -> None:
    use_catalogue_proof = condition == "after"
    server = _CatalogueServer(("127.0.0.1", 0), _CatalogueHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base_url = f"http://127.0.0.1:{server.server_port}/v1"

    try:
        for _ in range(WARMUPS):
            server.next_delay_seconds = 0.005
            _switch_once(base_url, use_catalogue_proof=use_catalogue_proof)

        rows: list[Observation] = []
        for fold, delays in enumerate(_delay_schedule(), start=1):
            for sample, delay_ms in enumerate(delays, start=1):
                server.next_delay_seconds = delay_ms / 1_000.0
                requests_before = server.request_count
                started = time.perf_counter_ns()
                _switch_once(base_url, use_catalogue_proof=use_catalogue_proof)
                latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
                rows.append(
                    Observation(
                        condition=condition,
                        fold=fold,
                        sample=sample,
                        scheduled_delay_ms=delay_ms,
                        latency_ms=latency_ms,
                        catalogue_requests=server.request_count - requests_before,
                    )
                )
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(asdict(rows[0])),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    print(f"Wrote {len(rows)} {condition} observations to {output}")


def _read(path: Path) -> list[Observation]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [
            Observation(
                condition=row["condition"],
                fold=int(row["fold"]),
                sample=int(row["sample"]),
                scheduled_delay_ms=float(row["scheduled_delay_ms"]),
                latency_ms=float(row["latency_ms"]),
                catalogue_requests=int(row["catalogue_requests"]),
            )
            for row in csv.DictReader(fh)
        ]


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _bootstrap_mean_ci(values: Sequence[float], *, repeats: int = 20_000) -> tuple[float, float]:
    rng = random.Random(SEED + len(values))
    means = [statistics.fmean(rng.choices(values, k=len(values))) for _ in range(repeats)]
    return _percentile(means, 0.025), _percentile(means, 0.975)


def _summary(values: Sequence[float]) -> dict[str, float | int | list[float]]:
    ci = _bootstrap_mean_ci(values)
    return {
        "n": len(values),
        "mean_ms": statistics.fmean(values),
        "sd_ms": statistics.stdev(values),
        "median_ms": statistics.median(values),
        "p95_ms": _percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
        "mean_95_ci_ms": [ci[0], ci[1]],
    }


def _paired_sign_test(before: Sequence[float], after: Sequence[float]) -> tuple[int, int, float]:
    differences = [left - right for left, right in zip(before, after)]
    positive = sum(diff > 0 for diff in differences)
    negative = sum(diff < 0 for diff in differences)
    n = positive + negative
    if n == 0:
        return positive, negative, 1.0
    tail = sum(math.comb(n, k) for k in range(min(positive, negative) + 1)) / (2**n)
    return positive, negative, min(1.0, 2.0 * tail)


def _paired_rows(before: Iterable[Observation], after: Iterable[Observation]) -> list[tuple[Observation, Observation]]:
    before_by_key = {(row.fold, row.sample): row for row in before}
    after_by_key = {(row.fold, row.sample): row for row in after}
    if before_by_key.keys() != after_by_key.keys():
        raise ValueError("Before and after samples do not have matching fold/sample keys")
    pairs = [(before_by_key[key], after_by_key[key]) for key in sorted(before_by_key)]
    for left, right in pairs:
        if not math.isclose(left.scheduled_delay_ms, right.scheduled_delay_ms, abs_tol=1e-9):
            raise ValueError(f"Mismatched scheduled delay for fold/sample {left.fold}/{left.sample}")
    return pairs


def _svg_chart(report: dict[str, object], output: Path) -> None:
    before = report["before"]
    after = report["after"]
    assert isinstance(before, dict) and isinstance(after, dict)
    summaries = [("Before", before, "#D97706"), ("After", after, "#2563EB")]
    maximum = max(float(summary["mean_95_ci_ms"][1]) for _, summary, _ in summaries)
    width, height = 920, 500
    left, right, top, bottom = 150, 70, 125, 95
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_positions = [top + plot_height * 0.3, top + plot_height * 0.72]

    def x(value: float) -> float:
        return left + (value / (maximum * 1.08)) * plot_width

    ticks = 5
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FCFCFD"/>',
        '<text x="150" y="45" font-family="Segoe UI, sans-serif" font-size="24" font-weight="650" fill="#172033">Desktop model-switch acknowledgement latency</text>',
        '<text x="150" y="75" font-family="Segoe UI, sans-serif" font-size="14" fill="#5B6475">Mean and bootstrap 95% CI; 5 folds × 12 paired samples; controlled loopback /v1/models delay</text>',
    ]
    for tick in range(ticks + 1):
        value = maximum * 1.08 * tick / ticks
        xpos = x(value)
        elements.append(
            f'<line x1="{xpos:.2f}" y1="{top - 12}" x2="{xpos:.2f}" y2="{top + plot_height + 10}" stroke="#E3E7EE" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{xpos:.2f}" y="{top + plot_height + 35}" text-anchor="middle" font-family="Consolas, monospace" font-size="12" fill="#687184">{value:.0f}</text>'
        )

    for (label, summary, colour), ypos in zip(summaries, y_positions):
        mean = float(summary["mean_ms"])
        low, high = (float(value) for value in summary["mean_95_ci_ms"])
        label_x = x(high) + 18
        label_y = ypos + 5
        label_anchor = "start"
        if label_x > width - 300:
            label_x = x(mean)
            label_y = ypos - 27
            label_anchor = "middle"
        elements.extend(
            [
                f'<text x="{left - 22}" y="{ypos + 5:.2f}" text-anchor="end" font-family="Segoe UI, sans-serif" font-size="16" font-weight="600" fill="#172033">{label}</text>',
                f'<line x1="{x(low):.2f}" y1="{ypos:.2f}" x2="{x(high):.2f}" y2="{ypos:.2f}" stroke="{colour}" stroke-width="4"/>',
                f'<line x1="{x(low):.2f}" y1="{ypos - 11:.2f}" x2="{x(low):.2f}" y2="{ypos + 11:.2f}" stroke="{colour}" stroke-width="2"/>',
                f'<line x1="{x(high):.2f}" y1="{ypos - 11:.2f}" x2="{x(high):.2f}" y2="{ypos + 11:.2f}" stroke="{colour}" stroke-width="2"/>',
                f'<circle cx="{x(mean):.2f}" cy="{ypos:.2f}" r="9" fill="#FCFCFD" stroke="{colour}" stroke-width="4"/>',
                f'<text x="{label_x:.2f}" y="{label_y:.2f}" text-anchor="{label_anchor}" font-family="Consolas, monospace" font-size="13" fill="#172033">{mean:.2f} ms [{low:.2f}, {high:.2f}]</text>',
            ]
        )

    elements.extend(
        [
            f'<text x="{left + plot_width / 2:.2f}" y="{height - 32}" text-anchor="middle" font-family="Segoe UI, sans-serif" font-size="14" fill="#384255">Acknowledgement latency (milliseconds)</text>',
            f'<text x="{left}" y="{height - 8}" font-family="Segoe UI, sans-serif" font-size="11" fill="#7A8394">Error bars: non-parametric bootstrap 95% confidence intervals of the mean.</text>',
            '</svg>',
        ]
    )
    output.write_text("\n".join(elements), encoding="utf-8")


def analyse(before_path: Path, after_path: Path, output_dir: Path) -> None:
    pairs = _paired_rows(_read(before_path), _read(after_path))
    before_values = [left.latency_ms for left, _ in pairs]
    after_values = [right.latency_ms for _, right in pairs]
    deltas = [left - right for left, right in zip(before_values, after_values)]
    positive, negative, p_value = _paired_sign_test(before_values, after_values)
    fold_rows = []
    for fold in range(1, FOLDS + 1):
        selected = [(left, right) for left, right in pairs if left.fold == fold]
        before_fold = [left.latency_ms for left, _ in selected]
        after_fold = [right.latency_ms for _, right in selected]
        fold_rows.append(
            {
                "fold": fold,
                "before_mean_ms": statistics.fmean(before_fold),
                "after_mean_ms": statistics.fmean(after_fold),
                "mean_reduction_ms": statistics.fmean(
                    left - right for left, right in zip(before_fold, after_fold)
                ),
            }
        )

    report: dict[str, object] = {
        "design": {
            "folds": FOLDS,
            "samples_per_fold": SAMPLES_PER_FOLD,
            "paired_samples": len(pairs),
            "seed": SEED,
            "workload": "real switch_model pipeline with a delayed loopback /v1/models endpoint",
            "error_bar": "non-parametric bootstrap 95% confidence interval of the mean",
            "test": "two-sided exact paired sign test",
        },
        "before": _summary(before_values),
        "after": _summary(after_values),
        "paired_delta": _summary(deltas),
        "relative_mean_reduction_percent": 100.0
        * (statistics.fmean(before_values) - statistics.fmean(after_values))
        / statistics.fmean(before_values),
        "speedup_ratio": statistics.fmean(before_values) / statistics.fmean(after_values),
        "sign_test": {
            "positive_differences": positive,
            "negative_differences": negative,
            "two_sided_p_value": p_value,
        },
        "catalogue_requests": {
            "before": sum(left.catalogue_requests for left, _ in pairs),
            "after": sum(right.catalogue_requests for _, right in pairs),
        },
        "folds": fold_rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "desktop-model-switch-benchmark.json"
    svg_path = output_dir / "desktop-model-switch-latency.svg"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _svg_chart(report, svg_path)
    print(json.dumps(report, indent=2))
    print(f"Wrote {json_path} and {svg_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--condition", choices=("before", "after"), required=True)
    measure_parser.add_argument("--output", type=Path, required=True)

    analyse_parser = subparsers.add_parser("analyse")
    analyse_parser.add_argument("--before", type=Path, required=True)
    analyse_parser.add_argument("--after", type=Path, required=True)
    analyse_parser.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "measure":
        measure(args.condition, args.output)
    else:
        analyse(args.before, args.after, args.output_dir)


if __name__ == "__main__":
    main()
