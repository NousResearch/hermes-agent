#!/usr/bin/env python3
"""
Adapter registry management for the finetune pipeline.

Handles status reporting, promotion, rollback, garbage collection,
and full pipeline orchestration.

Usage:
    python manage.py {status,promote,rollback,run,gc} [options]
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common import (
    ADAPTERS_DIR, CLUSTERS_DIR, CLUSTER_STATE_PATH,
    EXTRACTED_DIR, FINETUNE_DIR, REGISTRY_PATH, SCORED_DIR,
    ensure_dirs, load_config, load_json, save_json, read_jsonl, logger,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class AdapterRegistry:
    """Manage versioned adapters with promotion, rollback, and status."""

    def __init__(self):
        ensure_dirs()
        self.registry = load_json(REGISTRY_PATH, {"adapters": []})

    def _save(self):
        save_json(REGISTRY_PATH, self.registry)

    def _find_adapter(self, cluster_id: str, version: str = None) -> Optional[Dict]:
        """Find an adapter entry in the registry."""
        for entry in self.registry.get("adapters", []):
            if entry["cluster_id"] == cluster_id:
                if version is None or entry["version"] == version:
                    return entry
        return None

    def _find_active(self, cluster_id: str) -> Optional[Dict]:
        """Find the active adapter for a cluster."""
        for entry in self.registry.get("adapters", []):
            if entry["cluster_id"] == cluster_id and entry["status"] == "active":
                return entry
        return None

    def register_adapter(
        self,
        cluster_id: str,
        version: str,
        maturity: str,
        dataset_size: int,
        eval_results: Dict = None,
    ) -> Dict:
        """Register a new adapter version (initially as 'trained', not promoted)."""
        import hashlib

        adapter_dir = ADAPTERS_DIR / cluster_id / version
        config_path = adapter_dir / "config.yml"

        # Compute hashes for reproducibility
        base_model_hash = ""
        config_hash = ""
        if config_path.exists():
            config_hash = hashlib.sha256(
                config_path.read_bytes()
            ).hexdigest()[:16]

        entry = {
            "cluster_id": cluster_id,
            "cluster_label": "",
            "version": version,
            "status": "trained",
            "maturity": maturity,
            "base_model_hash": base_model_hash,
            "dataset_version": datetime.now().strftime("%Y-%m-%d"),
            "dataset_size": dataset_size,
            "training_config_hash": f"sha256:{config_hash}",
            "eval_results": eval_results or {},
            "trained_at": datetime.now().isoformat(),
            "promoted_at": None,
            "rollback_target": None,
        }

        # Update label from cluster state
        cluster_state = load_json(CLUSTER_STATE_PATH, {})
        cluster_info = cluster_state.get("clusters", {}).get(cluster_id, {})
        entry["cluster_label"] = cluster_info.get("label", f"auto:{cluster_id}")

        # Remove any existing entry for this version
        self.registry["adapters"] = [
            a for a in self.registry.get("adapters", [])
            if not (a["cluster_id"] == cluster_id and a["version"] == version)
        ]
        self.registry["adapters"].append(entry)
        self._save()

        logger.info("Registered adapter: %s %s", cluster_id, version)
        return entry

    def promote(self, cluster_id: str, version: str) -> bool:
        """Promote an adapter to active, demoting the previous one."""
        entry = self._find_adapter(cluster_id, version)
        if not entry:
            logger.error("Adapter not found: %s %s", cluster_id, version)
            return False

        # Demote current active
        current = self._find_active(cluster_id)
        if current:
            current["status"] = "previous"
            entry["rollback_target"] = current["version"]

        entry["status"] = "active"
        entry["promoted_at"] = datetime.now().isoformat()

        # Update symlink
        cluster_dir = ADAPTERS_DIR / cluster_id
        active_link = cluster_dir / "active"
        if active_link.is_symlink() or active_link.exists():
            active_link.unlink()
        active_link.symlink_to(version)

        self._save()
        logger.info("Promoted %s %s to active", cluster_id, version)
        return True

    def rollback(self, cluster_id: str) -> bool:
        """Roll back to the previous adapter version."""
        current = self._find_active(cluster_id)
        if not current:
            logger.error("No active adapter for cluster %s", cluster_id)
            return False

        target_version = current.get("rollback_target")
        if not target_version:
            logger.error("No rollback target for %s", cluster_id)
            return False

        target = self._find_adapter(cluster_id, target_version)
        if not target:
            logger.error("Rollback target %s not found", target_version)
            return False

        # Swap
        current["status"] = "rolled_back"
        target["status"] = "active"
        target["promoted_at"] = datetime.now().isoformat()

        # Update symlink
        cluster_dir = ADAPTERS_DIR / cluster_id
        active_link = cluster_dir / "active"
        if active_link.is_symlink() or active_link.exists():
            active_link.unlink()
        active_link.symlink_to(target_version)

        self._save()
        logger.info("Rolled back %s to %s", cluster_id, target_version)
        return True

    def gc(self, keep_versions: int = 2):
        """Garbage collect old adapter versions, keeping N most recent."""
        for cluster_dir in ADAPTERS_DIR.iterdir():
            if not cluster_dir.is_dir() or cluster_dir.name.startswith("."):
                continue

            versions = sorted(
                (d for d in cluster_dir.iterdir()
                 if d.is_dir() and d.name.startswith("v")),
                key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else 0,
            )

            # Keep active, rollback target, and N most recent
            protected = set()
            for entry in self.registry.get("adapters", []):
                if entry["cluster_id"] == cluster_dir.name:
                    protected.add(entry["version"])
                    if entry.get("rollback_target"):
                        protected.add(entry["rollback_target"])

            to_keep = set(v.name for v in versions[-keep_versions:])
            to_keep.update(protected)

            for v_dir in versions:
                if v_dir.name not in to_keep:
                    import shutil
                    shutil.rmtree(v_dir)
                    logger.info("GC: removed %s/%s", cluster_dir.name, v_dir.name)

    def status(self) -> str:
        """Generate a status report."""
        lines = []
        lines.append("=" * 60)
        lines.append("  FINETUNE PIPELINE STATUS")
        lines.append("=" * 60)

        # Data stats
        extracted_count = sum(
            sum(1 for _ in open(p))
            for p in EXTRACTED_DIR.glob("extract_*.jsonl")
            if p.exists()
        ) if EXTRACTED_DIR.exists() else 0

        scored_count = sum(
            sum(1 for _ in open(p))
            for p in SCORED_DIR.glob("scored_*.jsonl")
            if p.exists()
        ) if SCORED_DIR.exists() else 0

        lines.append(f"\n  Data:")
        lines.append(f"    Extracted sessions: {extracted_count}")
        lines.append(f"    Scored sessions:    {scored_count}")

        # Cluster state
        cluster_state = load_json(CLUSTER_STATE_PATH, {})
        if cluster_state:
            lines.append(f"\n  Clustering:")
            lines.append(f"    Algorithm:        {cluster_state.get('algorithm', 'n/a')}")
            lines.append(f"    Active clusters:  {cluster_state.get('clusters_active', 0)}")
            lines.append(f"    Noise sessions:   {cluster_state.get('noise_sessions', 0)}")
            lines.append(f"    Last run:         {cluster_state.get('last_run', 'never')}")

            for cid, info in cluster_state.get("clusters", {}).items():
                lines.append(f"    {cid}: {info.get('label', '?')} "
                           f"({info.get('session_count', 0)} sessions, "
                           f"maturity={info.get('maturity', '?')})")

        # Adapters
        adapters = self.registry.get("adapters", [])
        if adapters:
            lines.append(f"\n  Adapters:")
            for a in adapters:
                status_icon = {
                    "active": "[*]",
                    "trained": "[ ]",
                    "previous": "[-]",
                    "rolled_back": "[x]",
                }.get(a["status"], "[?]")
                lines.append(
                    f"    {status_icon} {a['cluster_id']} {a['version']} "
                    f"({a['status']}, {a.get('dataset_size', '?')} examples)"
                )
                if a.get("eval_results"):
                    er = a["eval_results"]
                    lines.append(
                        f"        eval: ppl={er.get('perplexity', '?')}, "
                        f"fmt={er.get('format_compliance', '?')}, "
                        f"task={er.get('task_completion', '?')}"
                    )
        else:
            lines.append("\n  Adapters: none")

        lines.append("=" * 60)
        return "\n".join(lines)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
BENCH_ENV_SCRIPT = REPO_ROOT / "environments" / "benchmarks" / "finetune_bench" / "finetune_bench_env.py"
BENCH_DEFAULT_CONFIG = REPO_ROOT / "environments" / "benchmarks" / "finetune_bench" / "default.yaml"
BENCH_RESULTS_DIR = FINETUNE_DIR / "bench" / "results"


def run_bench(prompt_bank: str = None) -> Optional[Path]:
    """
    Run the finetune benchmark via subprocess against the env script.

    Returns the path of the result file written under
    ~/.hermes/finetune/bench/results/, or None if the run failed.
    """
    if not BENCH_ENV_SCRIPT.exists():
        logger.error("Bench env script not found: %s", BENCH_ENV_SCRIPT)
        return None
    if not BENCH_DEFAULT_CONFIG.exists():
        logger.error("Bench config not found: %s", BENCH_DEFAULT_CONFIG)
        return None

    BENCH_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pre_existing = set(BENCH_RESULTS_DIR.glob("bench_*.json"))

    cmd = [
        sys.executable, str(BENCH_ENV_SCRIPT), "evaluate",
        "--config", str(BENCH_DEFAULT_CONFIG),
    ]
    if prompt_bank:
        cmd.extend(["--env.prompt_bank_path", prompt_bank])

    print(f"  Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=False, timeout=3600 * 2)
    except subprocess.TimeoutExpired:
        logger.error("Bench timed out after 2h")
        return None
    except Exception as e:
        logger.error("Bench invocation failed: %s", e)
        return None

    # Find the new result file (bench writes timestamped JSON)
    after = set(BENCH_RESULTS_DIR.glob("bench_*.json"))
    new = after - pre_existing
    if not new:
        # Fall back to most recent file
        candidates = sorted(after, key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else None
    return max(new, key=lambda p: p.stat().st_mtime)


def bench_passes(candidate_path: Path, baseline_path: Path = None) -> tuple:
    """
    Compare a candidate bench result against a baseline.

    Returns (passed: bool, summary: str).
    If baseline_path is None, uses the most recent prior result.
    """
    from eval import compare_metrics, verdict, format_report

    candidate_data = load_json(candidate_path)
    candidate_metrics = candidate_data.get("metrics", {})

    if baseline_path is None:
        # Find the most recent baseline that's older than the candidate
        candidates = sorted(
            (p for p in BENCH_RESULTS_DIR.glob("bench_*.json") if p != candidate_path),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        baseline_path = candidates[0] if candidates else None

    if baseline_path is None:
        return True, "No prior baseline — recording candidate as new baseline."

    baseline_data = load_json(baseline_path)
    baseline_metrics = baseline_data.get("metrics", {})

    comparison = compare_metrics(candidate_metrics, baseline_metrics)
    checks = verdict(comparison)
    passed = checks.get("overall", False)
    report = format_report(
        candidate_metrics, baseline_metrics, comparison, checks,
        cluster_id="(pipeline)", version="(latest)",
    )
    return passed, report


def run_pipeline(dry_run: bool = False, with_bench: bool = False):
    """
    Run the full pipeline: extract → score → cluster → train → register → promote.

    When with_bench=True, the bench env runs after promotion. If it regresses
    against the most recent prior bench result, the new adapters are
    automatically rolled back.
    """
    from extract import SessionExtractor
    from score import QualityScorer
    from cluster import DomainClusterer
    from train import TrainingOrchestrator

    total_steps = 6 if with_bench else 5

    print(f"\n[1/{total_steps}] Extracting sessions...")
    extractor = SessionExtractor()
    sessions = extractor.extract()
    if not sessions:
        sessions = extractor.get_all_extracted()
    print(f"  → {len(sessions)} sessions")

    if not sessions:
        print("No sessions available. Use hermes to generate some conversations first.")
        return

    print(f"\n[2/{total_steps}] Scoring quality...")
    scorer = QualityScorer()
    scored = scorer.score_all(sessions)
    good = sum(1 for s in scored if s.get("scoring", {}).get("bucket") == "good")
    print(f"  → {good} good, {len(scored) - good} other")

    print(f"\n[3/{total_steps}] Discovering domains...")
    clusterer = DomainClusterer()
    cluster_state = clusterer.cluster(scored)
    if cluster_state:
        print(f"  → {cluster_state.get('clusters_active', 0)} clusters")
    else:
        print("  → No clusters (data goes to _general)")

    print(f"\n[4/{total_steps}] Training adapters...")
    orchestrator = TrainingOrchestrator()
    trained = orchestrator.train_eligible(dry_run=dry_run)
    print(f"  → Trained {len(trained)} clusters: {trained}")

    if not (trained and not dry_run):
        print(f"\n[5/{total_steps}] Skipping registration (no training or dry run)")
        print("\nPipeline complete.")
        return

    print(f"\n[5/{total_steps}] Registering and promoting adapters...")
    registry = AdapterRegistry()
    promoted: List[Tuple[str, str]] = []
    for cid in trained:
        cluster_dir = ADAPTERS_DIR / cid
        versions = sorted(
            (d.name for d in cluster_dir.iterdir()
             if d.is_dir() and d.name.startswith("v")),
            key=lambda x: int(x[1:]) if x[1:].isdigit() else 0,
        )
        if not versions:
            continue
        version = versions[-1]
        cluster_info = (cluster_state or {}).get("clusters", {}).get(cid, {})
        train_path = CLUSTERS_DIR / cid / "train.jsonl"
        ds_size = sum(1 for _ in open(train_path)) if train_path.exists() else 0
        registry.register_adapter(
            cluster_id=cid,
            version=version,
            maturity=cluster_info.get("maturity", "established"),
            dataset_size=ds_size,
        )
        registry.promote(cid, version)
        promoted.append((cid, version))
        print(f"  → Promoted {cid} {version}")

    if not with_bench:
        print("\nPipeline complete. Run '/finetune bench' to verify quality.")
        return

    print(f"\n[6/{total_steps}] Running benchmark gate...")
    candidate_path = run_bench()
    if candidate_path is None:
        print("  → Benchmark failed to run. Adapters remain promoted; verify manually.")
        return

    passed, report = bench_passes(candidate_path)
    print(report)

    if passed:
        print("\n  → BENCHMARK PASSED. Adapters remain active.")
        print("\nPipeline complete.")
    else:
        print("\n  → BENCHMARK REGRESSED. Rolling back promoted adapters...")
        for cid, version in promoted:
            if registry.rollback(cid):
                print(f"    Rolled back {cid} (was {version})")
            else:
                print(f"    Could not rollback {cid} — manual intervention required")
        print("\nPipeline complete with regression. Investigate the bench report above.")


CRON_SCHEDULE_MAP = {
    "daily": "0 3 * * *",
    "weekly": "0 3 * * 0",
    "biweekly": "0 3 1,15 * *",
    "monthly": "0 3 1 * *",
}


def setup_cron(schedule: str = "weekly"):
    """Set up scheduled retraining via the Hermes cron system."""
    cron_expr = CRON_SCHEDULE_MAP.get(schedule, schedule)

    prompt = (
        "Run the finetune pipeline: extract new sessions, score quality, "
        "update clusters, retrain eligible adapters, evaluate, and promote "
        "if evaluation passes. Report results including adapter versions, "
        "cluster changes, and any evaluation failures."
    )

    try:
        from tools.cronjob_tools import schedule_cronjob
        result = schedule_cronjob(
            prompt=prompt,
            schedule=cron_expr,
            name="finetune-retrain",
        )
        print(f"Cron job created: {result}")
        print(f"Schedule: {schedule} ({cron_expr})")
    except ImportError:
        # Fall back to shell-level cron
        print(f"Hermes cron not available. Add this to your crontab:")
        scripts_dir = Path(__file__).resolve().parent
        print(f"  {cron_expr}  cd {scripts_dir} && python manage.py run")


def main():
    parser = argparse.ArgumentParser(description="Adapter registry management")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show pipeline status")

    p_promote = sub.add_parser("promote", help="Promote adapter to active")
    p_promote.add_argument("--cluster", required=True)
    p_promote.add_argument("--version", required=True)

    p_rollback = sub.add_parser("rollback", help="Roll back to previous version")
    p_rollback.add_argument("--cluster", required=True)

    p_run = sub.add_parser("run", help="Run full pipeline")
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument(
        "--with-bench", action="store_true",
        help="Run the benchmark gate after promotion. "
             "Auto-rollback adapters that regress vs. the most recent prior result.",
    )

    sub.add_parser("bench", help="Run the finetune benchmark against the active model")

    p_gc = sub.add_parser("gc", help="Garbage collect old versions")
    p_gc.add_argument("--keep", type=int, default=2, help="Versions to keep")

    p_cron = sub.add_parser("cron", help="Set up scheduled retraining")
    p_cron.add_argument("schedule", nargs="?", default="weekly",
                        help="Schedule: daily, weekly, biweekly, monthly, or cron expression")

    args = parser.parse_args()

    if args.command == "status":
        registry = AdapterRegistry()
        print(registry.status())

    elif args.command == "promote":
        registry = AdapterRegistry()
        if registry.promote(args.cluster, args.version):
            print(f"Promoted {args.cluster} {args.version}")
        else:
            print("Promotion failed.")
            sys.exit(1)

    elif args.command == "rollback":
        registry = AdapterRegistry()
        if registry.rollback(args.cluster):
            print(f"Rolled back {args.cluster}")
        else:
            print("Rollback failed.")
            sys.exit(1)

    elif args.command == "run":
        run_pipeline(dry_run=args.dry_run, with_bench=args.with_bench)

    elif args.command == "bench":
        result_path = run_bench()
        if result_path:
            passed, report = bench_passes(result_path)
            print(report)
            print(f"\nResult saved to: {result_path}")
            sys.exit(0 if passed else 1)
        else:
            print("Benchmark failed to run.")
            sys.exit(1)

    elif args.command == "gc":
        registry = AdapterRegistry()
        registry.gc(keep_versions=args.keep)
        print("Garbage collection complete.")

    elif args.command == "cron":
        setup_cron(args.schedule)


if __name__ == "__main__":
    main()
