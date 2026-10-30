"""Local-only technical demonstration, not evidence of Hermes agent invocation."""

import argparse
import csv
from contextlib import ExitStack
import io
import json
import platform
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from catalog import Catalog, server_for
from data import FIELDS, SUPPORT, record, reference


class Budget:
    def __init__(self, seconds):
        self.deadline = time.monotonic() + seconds

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Run duration budget reached")


def request(base, path, payload=None, timeout=3):
    body = json.dumps(payload).encode() if payload is not None else None
    req = Request(base + path, data=body, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8")


def read_ids(call, run, budget):
    ids, page, total = [], 1, None
    while total is None or len(ids) < total:
        budget.check()
        data = json.loads(call("/products?" + urlencode({"run": run, "page": page, "page_size": 200})))
        total = data["total"]
        if not data["rows"] and len(ids) < total:
            raise ValueError("Preparation reconciliation encountered missing rows")
        ids.extend(row["id"] for row in data["rows"])
        page += 1
    if len(set(ids)) != len(ids):
        raise ValueError("Preparation reconciliation encountered duplicate identifiers")
    return set(ids)


def prepare(call, run, target, seed, budget, manifest, save, batch_size=250):
    known = read_ids(call, run, budget)
    if known - set(range(target)):
        raise ValueError("Existing run contains records outside the target prefix")
    missing = sorted(set(range(target)) - known)
    for start in range(0, len(missing), batch_size):
        budget.check()
        rows = [record(seed, index) for index in missing[start:start + batch_size]]
        try:
            counts = json.loads(call("/prepare", {"run": run, "rows": rows}))
            known.update(row["id"] for row in rows)
            if counts["products"] != len(known) or counts["orphaned"]:
                raise ValueError("Preparation returned inconsistent persisted counts")
        except (OSError, ValueError, KeyError, TypeError, URLError):
            # Reconcile actual committed effects once; never blindly retry the write.
            manifest["preparation_error"] = "Batch failed; no write retry attempted"
            try:
                known = read_ids(call, run, budget)
                manifest["reconciled_ids"] = sorted(known)
            finally:
                save()
            raise
        manifest["confirmed_products"] = len(known)
        manifest["confirmed_logical_prefix"] = len(known) if known == set(range(len(known))) else None
        save()
    counts = json.loads(call("/counts?" + urlencode({"run": run})))
    if counts["products"] != target or counts["categories"] != 8 or counts["orphaned"]:
        raise ValueError("Stage target or relationship count not confirmed")
    return counts


def check_pages(call, run, expected, args, budget):
    observed, page, total = [], 1, None
    while total is None or len(observed) < total:
        budget.check()
        data = json.loads(call("/products?" + urlencode({"run": run, "page": page, "page_size": 50, **args})))
        total = data["total"]
        if total != len(expected):
            raise ValueError("Incorrect list total")
        if not data["rows"] and len(observed) < total:
            raise ValueError("Missing page")
        observed.extend(data["rows"])
        page += 1
    result = SUPPORT.compare_rows(observed, expected, FIELDS)
    empty = json.loads(call("/products?" + urlencode({"run": run, "page": page, "page_size": 50, **args})))
    if not result["correct"] or empty["rows"]:
        raise ValueError("Pagination has missing/duplicate/unordered rows or a nonempty page beyond its end")
    return {"parameters": args, "pages": page - 1, **result}


def operations(rows):
    cases = [("listing", {}), ("search", {"q": "Café"}), ("category", {"category": "0"}), ("price-sort", {"sort": "price"})]
    for name, args in cases:
        expected = reference(rows, **args)
        yield name, "/products", {**args, "page": 1}, expected[:50], len(expected)
    expected = reference(rows)
    last = max(1, (len(expected) + 49) // 50)
    yield "last-page", "/products", {"page": last}, expected[(last - 1) * 50:], len(expected)
    for name, args in [("export", {}), ("filtered-export", {"category": "0"})]:
        expected = reference(rows, **args)
        yield name, "/export", args, expected, len(expected)


def measure(call, run, volume, operation, phase):
    name, endpoint, args, expected, total = operation
    sample = {"volume": volume, "operation": name, "phase": phase, "unit": "ms", "error": None, "timeout": False, "correct": None}
    start = time.perf_counter()
    try:
        raw = call(endpoint + "?" + urlencode({"run": run, "page_size": 50, **args}))
        sample["duration_ms"] = (time.perf_counter() - start) * 1000
        if endpoint == "/export":
            observed = list(csv.DictReader(io.StringIO(raw)))
            total_correct = True
        else:
            data = json.loads(raw)
            observed, total_correct = data["rows"], data["total"] == total
        sample.update(SUPPORT.compare_rows(observed, expected, FIELDS))
        sample["correct"] = sample["correct"] and total_correct
    except (OSError, ValueError, KeyError, TypeError, URLError) as error:
        sample["duration_ms"] = (time.perf_counter() - start) * 1000
        sample["error"] = f"{type(error).__name__}: {error}"
        sample["timeout"] = isinstance(error, TimeoutError) or isinstance(getattr(error, "reason", None), TimeoutError)
    return sample


def report(result):
    lines = ["# Synthetic catalog volume test", "", f"Status: {result['status']}. Execution: technical HTTP runner, not Hermes agent E2E.",
             "", f"Environment: {result['environment']}", f"Configuration: {result['plan']}",
             "", "Preparation, page-walk verification and measurement are separate. Concurrency: one sequential client; page size: 50; one warmup and ten warmed samples per operation.",
             "The separately labeled first request is after preparation/verification; neither OS nor SQLite caches were flushed. No cold-cache claim. HTTP body download timing excludes oracle comparison and UI rendering.",
             "No performance SLA was supplied. Ratios are observations, not bugs or infrastructure diagnoses. No sleeps manufacture latency; no percentiles are reported from these small samples.",
             "", "## Confirmed volumes", "| Target products | Actual products | Categories | Orphans |", "| --- | --- | --- | --- |"]
    for stage in result["stages"]:
        lines.append(f"| {stage['target']} | {stage['products']} | {stage['categories']} | {stage['orphaned']} |")
    lines += ["", "## Warmed HTTP measurements", "| Operation | Products | n | Median ms | Min ms | Max ms | Errors | Timeouts | Incorrect | Ratio to baseline |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    baselines = {}
    for stage in result["stages"]:
        for operation in stage.get("operations", []):
            samples = [s for s in result["samples"] if s["volume"] == stage["target"] and s["operation"] == operation and s["phase"] == "warm"]
            metrics = SUPPORT.summarize(samples)
            median = metrics["median_ms"]
            if median is not None:
                baselines.setdefault(operation, median)
            ratio = median / baselines[operation] if median is not None and baselines.get(operation) else None
            display = lambda value: f"{value:.3f}" if value is not None else "unrun"
            lines.append(f"| {operation} | {stage['target']} | {metrics['samples']} | {display(median)} | {display(metrics['min_ms'])} | {display(metrics['max_ms'])} | {metrics['errors']} | {metrics['timeouts']} | {metrics['incorrect']} | {display(ratio)} |")
    failures = [s for s in result["samples"] if s.get("correct") is False or s.get("error")]
    lines += ["", "## Correctness and evidence", f"Incorrect/error samples: {len(failures)}. See raw samples.json and result.json for expected/actual counts, missing/unexpected rows, stage page walks and errors.",
              "All page walks use an independent Python filter/sort reference, including final and empty pages, ties and full identifier coverage.",
              f"Controlled fault enabled: {result['injected_fault']}. Any truncated export in that scenario is deliberately injected, not a spontaneous finding."]
    if failures:
        first = failures[0]
        lines += [f"First failure: {first['operation']} at {first['volume']} products, expected {first.get('expected_count')}, observed {first.get('actual_count')}, error={first.get('error')}.",
                  "Reproduce with the documented --inject-export-bug flag; compare CSV content to the independently generated complete filtered set."]
    if result["ui_samples"]:
        lines += ["", "## Technical UI measurements", f"Browser: {result.get('browser')}. DOM rows, displayed totals and request completion are checked against the independent reference. UI timing includes interaction and waiting for correct rendered content, not just navigation.",
                  "| Operation | Products | Warm n | Median UI ms | Errors | Timeouts |", "| --- | --- | --- | --- | --- | --- |"]
        for volume in result["plan"]["volumes"]:
            for operation in ("listing", "search", "category", "price-sort", "last-page"):
                samples = [s for s in result["ui_samples"] if s["volume"] == volume and s["operation"] == operation and s["phase"] == "warm"]
                metrics = SUPPORT.summarize(samples)
                lines.append(f"| {operation} | {volume} | {metrics['samples']} | {metrics['median_ms']} | {metrics['errors']} | {metrics['timeouts']} |")
            if f"ui-{volume}.png" in result.get("browser", {}).get("screenshots", []):
                lines.append(f"![Catalog at {volume} products](ui-{volume}.png)")
    lines += ["", "## Coverage and limitations", "HTTP listing, pagination, search, category filtering, stable name/price sorting, totals and full CSV content are covered only where stage entries/samples exist.",
              "Real Hermes agent skill invocation, CPU/memory attribution and other platforms are unrun in this technical execution. UI timings/screenshots and console/network capture are covered only when ui_samples/browser evidence is present; otherwise unrun. Local results do not establish production capacity or arbitrary-app support.",
              f"Interruption/blocker: {result.get('interruption') or 'none'}.", "", "## Cleanup", str(result.get("cleanup", {"status": "not attempted"})), ""]
    return "\n".join(lines)


def execute(call, plan, output, injected=False, budget=None, browser=None):
    output = Path(output)
    budget = budget or Budget(plan["duration_seconds"])
    manifest = {"run": plan["run"], "seed": plan["seed"], "confirmed_products": 0, "categories_expected": 8}
    result = {"status": "running", "plan": plan, "stages": [], "samples": [], "ui_samples": [], "injected_fault": injected,
              "environment": {"python": platform.python_version(), "platform": platform.platform(), "sqlite": __import__("sqlite3").sqlite_version,
                              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "started_utc": datetime.now(UTC).isoformat()}}

    def save():
        SUPPORT.checkpoint(output / "manifest.json", manifest)
        SUPPORT.checkpoint(output / "samples.json", result["samples"])
        SUPPORT.checkpoint(output / "result.json", result)

    SUPPORT.checkpoint(output / "plan.json", plan)
    save()
    try:
        for target in plan["volumes"]:
            budget.check()
            if target > plan["max_records"]:
                raise ValueError("Record limit reached")
            counts = prepare(call, plan["run"], target, plan["seed"], budget, manifest, save)
            stage = {"target": target, **counts, "page_walks": [], "operations": []}
            result["stages"].append(stage)
            rows = [record(plan["seed"], i) for i in range(target)]
            for args in ({}, {"q": "Café"}, {"category": "0"}, {"sort": "price"}):
                stage["page_walks"].append(check_pages(call, plan["run"], reference(rows, **args), args, budget))
            for operation in operations(rows):
                stage["operations"].append(operation[0])
                for phase in ["first-request", "warmup"] + ["warm"] * plan["samples"]:
                    budget.check()
                    result["samples"].append(measure(call, plan["run"], target, operation, phase))
                    save()
                    failures = sum(bool(s.get("error")) or s.get("correct") is False for s in result["samples"])
                    if failures >= plan["max_errors"]:
                        raise ValueError("Error/correctness budget reached")
            if browser is not None:
                previous_samples = list(result["ui_samples"])

                def save_ui(samples):
                    result["ui_samples"] = previous_samples + samples
                    result["browser"] = {"version": browser.browser.version, "viewport": {"width": 1100, "height": 800}, "errors": list(browser.errors), "screenshots": list(browser.screenshots)}
                    save()
                    failures = sum(bool(s.get("error")) or s.get("correct") is False for s in result["samples"] + result["ui_samples"])
                    if failures >= plan["max_errors"]:
                        raise ValueError("Error/correctness budget reached")

                browser.capture(plan["run"], target, rows, output, budget, save_ui)
        result["status"] = "completed"
    except (Exception, KeyboardInterrupt) as error:  # preserve a partial report even for browser/runtime failures
        result["status"] = "interrupted"
        result["interruption"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            result["cleanup"] = json.loads(call("/cleanup", {"run": plan["run"]}))
        except (OSError, ValueError, KeyError, TypeError, URLError) as error:
            result["cleanup"] = {"verified": False, "error": str(error), "run": plan["run"]}
        try:
            residual = json.loads(call("/counts?" + urlencode({"run": plan["run"]})))
            result["cleanup"]["residuals"] = residual
            result["cleanup"]["verified"] = residual["products"] == 0 and residual["categories"] == 0 and residual["sentinel_intact"]
        except (OSError, ValueError, KeyError, TypeError, URLError) as error:
            result["cleanup"].update(verified=False, verification_error=str(error))
        result["environment"]["ended_utc"] = datetime.now(UTC).isoformat()
        save()
        (output / "report.md").write_text(report(result), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inject-export-bug", action="store_true")
    parser.add_argument("--browser-executable", help="Optional local Chromium; requires Playwright")
    parser.add_argument("--volumes", type=int, nargs="+", default=[100, 1000, 10000])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=120)
    args = parser.parse_args()
    if args.output.exists() or args.volumes != sorted(set(args.volumes)) or not args.volumes or args.volumes[0] <= 0 or args.volumes[-1] > 10000 or args.duration <= 0:
        parser.error("Use a fresh output, increasing positive absolute volumes <=10000, and positive budget")
    args.output.mkdir(parents=True)
    plan = {"run": "volume-" + uuid.uuid4().hex, "seed": args.seed, "volumes": args.volumes, "samples": 10,
            "timeout_seconds": 3, "duration_seconds": args.duration, "max_records": 10000, "max_errors": 50,
            "concurrency": 1, "page_size": 50, "batch_size": 250, "counted_entity": "products",
            "cleanup": "Delete only products/categories with this exact run; verify unrelated sentinel"}
    SUPPORT.checkpoint(args.output / "plan.json", plan)
    with tempfile.TemporaryDirectory(prefix="volume-catalog-") as directory:
        catalog = Catalog(Path(directory) / "catalog.sqlite", args.inject_export_bug)
        server = server_for(catalog)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            plan["target_url"] = base
            plan["application_repository"] = "tests/skills/fixtures/volume_test_app"
            with ExitStack() as stack:
                browser = None
                if args.browser_executable:
                    from browser_evidence import BrowserEvidence
                    browser = stack.enter_context(BrowserEvidence(base, args.browser_executable))
                result = execute(lambda path, payload=None: request(base, path, payload, plan["timeout_seconds"]), plan, args.output, args.inject_export_bug, browser=browser)
            print(json.dumps({"status": result["status"], "cleanup": result["cleanup"], "samples": len(result["samples"])}))
        finally:
            server.shutdown()
            thread.join()
            server.server_close()
    return 0 if result["status"] == "completed" and result["cleanup"]["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
