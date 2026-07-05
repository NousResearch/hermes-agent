"""CLI subcommand: `hermes evolution <subcommand>`.

Thin shell around agent/evolution/. Renders status tables, defines tasks,
runs evaluations, and manages the evolution engine.

Usage:
    hermes evolution status              Show engine status and recent runs
    hermes evolution define-task <file>  Define a task from YAML
    hermes evolution list-tasks          List all defined tasks
    hermes evolution run <task-name>     Run a tracked task attempt
    hermes evolution history [task]      Show evolution history
    hermes evolution variants            Show harness variant stats
    hermes evolution enable/disable      Toggle the evolution engine
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home
from agent.evolution.atropos_export import ATROPOS_FORMAT_VERSION


def _fmt_ts(ts: Optional[str]) -> str:
    if not ts:
        return "never"
    try:
        dt = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return str(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - dt
    secs = int(delta.total_seconds())
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


# ── status ──────────────────────────────────────────────────────────────


def _cmd_status(args) -> int:
    """Show Evolution Engine status and recent runs."""
    try:
        from agent.evolution.config import EvolutionConfig
        config = EvolutionConfig.from_config()
    except Exception as e:
        print(f"Error loading config: {e}")
        return 1

    status_line = "ENABLED" if config.enabled else "DISABLED"
    print(f"Evolution Engine: {status_line}")
    print(f"  mode:           {config.mode}")
    print(f"  max iterations: {config.max_iterations}")
    print(f"  regression gate: {'on' if config.regression_enabled else 'off'}")
    print()

    if not config.enabled:
        print("Enable with: hermes config set evolution.enabled true")
        return 0

    try:
        from agent.evolution.evolution_store import get_evolution_store
        store = get_evolution_store()
        runs = store.list_runs(limit=20)
    except Exception as e:
        print(f"Error accessing evolution store: {e}")
        return 1

    if not runs:
        print("No evolution runs recorded yet.")
        print("Define a task: hermes evolution define-task <file.yaml>")
        return 0

    succeeded = sum(1 for r in runs if r["status"] == "succeeded")
    failed = sum(1 for r in runs if r["status"] == "failed")
    exhausted = sum(1 for r in runs if r["status"] == "exhausted")
    pending = sum(1 for r in runs if r["status"] == "pending")

    print(f"Recent runs ({len(runs)} total):")
    print(f"  succeeded: {succeeded}  failed: {failed}  exhausted: {exhausted}  pending: {pending}")
    print()

    # Table header
    print(f"{'Status':8} {'Task':30} {'Score':8} {'Iters':6} {'When':12}")
    print("-" * 70)

    for r in runs[:20]:
        icon = {"succeeded": "✅", "failed": "❌", "exhausted": "⚠️", "pending": "⏳"}.get(r["status"], "❓")
        score = r.get("final_score")
        score_str = f"{score:.2f}" if score is not None else "N/A"
        iters = f"{r.get('iterations', 0)}/{r.get('max_iterations', 5)}"
        when = _fmt_ts(r.get("created_at"))
        task_name = r["task_name"][:29]
        print(f"{icon} {r['status']:7} {task_name:30} {score_str:8} {iters:6} {when:12}")

    return 0


# ── define-task ─────────────────────────────────────────────────────────


def _cmd_define_task(args) -> int:
    """Define a new evolution task from a YAML file."""
    yaml_path = Path(args.file)
    if not yaml_path.exists():
        print(f"Error: file not found: {yaml_path}")
        return 1

    try:
        from agent.evolution.task_definition import TaskDefinition, save_task
        task = TaskDefinition.from_yaml(yaml_path)
        errors = task.validate()
        if errors:
            print("Task validation failed:")
            for e in errors:
                print(f"  - {e}")
            return 1
        path = save_task(task)
        print(f"Task '{task.name}' defined successfully.")
        print(f"  Path: {path}")
        print(f"  Domain: {task.domain}  Complexity: {task.complexity}/14")
        print(f"  Criteria: {len(task.success_criteria)} ({', '.join(c.type.value for c in task.success_criteria)})")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


# ── list-tasks ──────────────────────────────────────────────────────────


def _cmd_list_tasks(args) -> int:
    """List all defined evolution tasks."""
    try:
        from agent.evolution.task_definition import list_tasks
        tasks = list_tasks()
    except Exception as e:
        print(f"Error: {e}")
        return 1

    if not tasks:
        print("No evolution tasks defined.")
        print("Define one: hermes evolution define-task <file.yaml>")
        return 0

    print(f"Evolution Tasks ({len(tasks)} total):\n")
    for t in tasks:
        criteria = ", ".join(c.type.value for c in t.success_criteria)
        print(f"  {t.name}")
        print(f"    domain={t.domain}  complexity={t.complexity}/14  timeout={t.timeout_seconds}s  max_turns={t.max_turns}")
        print(f"    criteria: {criteria}")
        print()
    return 0


# ── history ─────────────────────────────────────────────────────────────


def _cmd_history(args) -> int:
    """Show detailed evolution history for a task or all tasks."""
    try:
        from agent.evolution.evolution_store import get_evolution_store
        store = get_evolution_store()
    except Exception as e:
        print(f"Error: {e}")
        return 1

    runs = store.list_runs(task_name=args.task or None, limit=args.limit or 50)
    if not runs:
        print("No evolution history found.")
        return 0

    for r in runs:
        icon = {"succeeded": "✅", "failed": "❌", "exhausted": "⚠️"}.get(r["status"], "❓")
        print(f"\n{icon} Run: {r['run_id']}")
        print(f"   Task: {r['task_name']} ({r.get('task_domain', 'general')})")
        print(f"   Status: {r['status']}  Score: {r.get('final_score', 'N/A')}")
        print(f"   Iterations: {r.get('iterations', 0)}/{r.get('max_iterations', 5)}")
        print(f"   Created: {_fmt_ts(r.get('created_at'))}")

        if args.verbose:
            iterations = store.get_iterations(r["run_id"])
            for it in iterations:
                action = it.get("improvement_action") or "none"
                target = it.get("improvement_target") or ""
                score = it.get("score")
                score_str = f"{score:.2f}" if score is not None else "N/A"
                print(f"     iter #{it['iteration_num']}: {it['status']} score={score_str} action={action} target={target}")
    return 0


# ── variants ────────────────────────────────────────────────────────────


def _cmd_variants(args) -> int:
    """Show harness variant statistics."""
    try:
        from agent.evolution.harness_variants import VariantManager
        vm = VariantManager.load()
    except Exception as e:
        print(f"Error loading variants: {e}")
        return 1

    variants = vm.active_variants
    print(f"Active Harness Variants ({len(variants)}):\n")

    for v in variants:
        marker = "→ " if v.variant_id == vm._active_variant_id else "  "
        print(f"{marker}{v.name} ({v.variant_id})")
        print(f"    Success rate: {v.success_rate:.1%} ({v.total_tasks_succeeded}/{v.total_tasks_attempted})")
        print(f"    Avg score: {v.avg_score:.3f}")
        print(f"    Tasks tracked: {len(v.task_scores)}")
        print(f"    Created: {_fmt_ts(v.created_at)}")
        if v.parent_variant:
            print(f"    Forked from: {v.parent_variant}")
        print()
    return 0


# ── enable / disable ────────────────────────────────────────────────────


def _cmd_enable(args) -> int:
    """Enable the Evolution Engine."""
    import subprocess
    result = subprocess.run(
        ["hermes", "config", "set", "evolution.enabled", "true"],
        capture_output=True, text=True,
    )
    print(result.stdout.strip() or "Evolution Engine enabled.")
    return result.returncode


def _cmd_disable(args) -> int:
    """Disable the Evolution Engine."""
    import subprocess
    result = subprocess.run(
        ["hermes", "config", "set", "evolution.enabled", "false"],
        capture_output=True, text=True,
    )
    print(result.stdout.strip() or "Evolution Engine disabled.")
    return result.returncode


# ── run (benchmark mode) ────────────────────────────────────────────────


def _cmd_run(args) -> int:
    """Run an evolution benchmark on a task."""
    task_name = args.task
    iterations = args.iterations or 1

    try:
        from agent.evolution.task_definition import load_task, list_tasks
        task = load_task(task_name)
        if task is None:
            available = [t.name for t in list_tasks()]
            print(f"Task '{task_name}' not found.")
            if available:
                print(f"Available: {', '.join(available)}")
            return 1
    except Exception as e:
        print(f"Error loading task: {e}")
        return 1

    print(f"Task: {task.name}")
    print(f"Description: {task.description}")
    print(f"Criteria: {len(task.success_criteria)}")
    print()

    try:
        from agent.evolution.evaluator import TaskEvaluator, EvaluationContext
        evaluator = TaskEvaluator()
        ctx = EvaluationContext(working_dir=args.cwd or "")

        import time as _time
        total_score = 0.0
        passed_count = 0

        for i in range(iterations):
            start = _time.monotonic()
            result = evaluator.evaluate(task, None, ctx)  # type: ignore[arg-type]
            elapsed = _time.monotonic() - start

            icon = "✅" if result.passed else "❌"
            print(f"  [{i+1}/{iterations}] {icon} score={result.score:.2f} time={elapsed:.2f}s")
            total_score += result.score
            if result.passed:
                passed_count += 1

            if args.verbose:
                for check in result.checks:
                    ci = "✅" if check.passed else "❌"
                    print(f"      {ci} {check.type}: {check.detail[:120]}")

        print()
        print(f"Results: {passed_count}/{iterations} passed ({passed_count/iterations*100:.1f}%)")
        print(f"Average score: {total_score/iterations:.3f}")

    except Exception as e:
        print(f"Error running benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0 if passed_count == iterations else 1


# ── benchmark ───────────────────────────────────────────────────────────


def _cmd_benchmark(args) -> int:
    """Run full evolution benchmark suite."""
    try:
        from agent.evolution.task_definition import list_tasks, load_task
        tasks = list_tasks()
    except Exception as e:
        print(f"Error: {e}")
        return 1

    if not tasks:
        print("No tasks defined. Create one first: hermes evolution define-task <file.yaml>")
        return 1

    if args.task:
        tasks = [t for t in tasks if t.name == args.task]
        if not tasks:
            print(f"Task '{args.task}' not found.")
            return 1

    print(f"Evolution Benchmark — {len(tasks)} task(s)\n")
    print(f"{'Task':30} {'Passed':8} {'Score':8} {'Time':8} {'Details'}")
    print("-" * 80)

    try:
        from agent.evolution.evaluator import TaskEvaluator, EvaluationContext
        import time as _time

        evaluator = TaskEvaluator()
        total_passed = 0
        total_score = 0.0
        total_time = 0.0

        for task in tasks:
            start = _time.monotonic()
            ctx = EvaluationContext(working_dir=args.cwd or "")
            result = evaluator.evaluate(task, None, ctx)  # type: ignore[arg-type]
            elapsed = _time.monotonic() - start

            icon = "✅" if result.passed else "❌"
            details = ", ".join(
                f"{'✓' if c.passed else '✗'}{c.type}"
                for c in result.checks[:3]
            )
            print(f"{icon} {task.name:28} {str(result.passed):8} {result.score:.3f}   {elapsed:.2f}s   {details}")

            if result.passed:
                total_passed += 1
            total_score += result.score
            total_time += elapsed

        print("-" * 80)
        avg_score = total_score / len(tasks) if tasks else 0
        print(f"Summary: {total_passed}/{len(tasks)} passed ({total_passed/len(tasks)*100:.1f}%)")
        print(f"Average score: {avg_score:.3f}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time: {total_time/len(tasks):.2f}s/task" if tasks else "")

    except Exception as e:
        print(f"Error running benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0 if total_passed == len(tasks) else 1


# ── export ──────────────────────────────────────────────────────────────


def _cmd_export(args) -> int:
    """Export evolution runs as Atropos-compatible training data."""
    try:
        from agent.evolution.atropos_export import (
            export_run_to_jsonl,
            export_all_runs,
            get_export_stats,
        )
    except Exception as e:
        print(f"Error loading export module: {e}")
        return 1

    # Show stats first
    stats = get_export_stats(days=args.days or 30)
    print(f"Training Data Available (last {stats['period_days']} days):")
    print(f"  Total runs:       {stats['total_runs']}")
    print(f"  Succeeded:        {stats['succeeded']}")
    print(f"  Failed:           {stats['failed']}")
    print(f"  Exhausted:        {stats['exhausted']}")
    print(f"  Total iterations: {stats['total_iterations']}")
    print(f"  Proposals applied:{stats['total_proposals']}")
    print(f"  Est. records:     {stats['estimated_training_records']}")
    if stats['by_domain']:
        print(f"  By domain:")
        for domain, count in sorted(stats['by_domain'].items()):
            print(f"    {domain}: {count}")
    print()

    if not stats['total_runs']:
        print("No evolution runs to export. Run tasks first: hermes evolution run <task>")
        return 0

    # Export specific run or all
    if args.run_id:
        path = export_run_to_jsonl(args.run_id, output_dir=args.output)
        print(f"Exported run → {path}")
    elif args.all:
        output = args.output or Path(
            get_hermes_home() / "evolution" / "exports" / "training_data.jsonl"
        )
        records = export_all_runs(
            days=args.days or 30,
            status_filter=args.status.split(",") if args.status else None,
            domain_filter=args.domain,
            output_path=output,
        )
        print(f"Exported {len(records)} training records → {output}")
        print(f"Format: Atropos-compatible ShareGPT (v{ATROPOS_FORMAT_VERSION})")
    else:
        print("Specify --run-id <id> for one run, or --all for all runs.")

    return 0


# ── suggest-tasks ───────────────────────────────────────────────────────


def _cmd_suggest_tasks(args) -> int:
    """Auto-discover task definitions from observed agent usage patterns."""
    try:
        from agent.evolution.conversation_observer import (
            get_observer,
            PATTERN_LABELS,
        )
        from agent.evolution.task_definition import save_task, get_task_dir
    except Exception as e:
        print(f"Error loading observer module: {e}")
        return 1

    observer = get_observer()
    suggestions = observer.suggest_tasks(min_occurrences=args.min_occurrences)

    if not suggestions:
        print("No task patterns discovered yet.")
        print()
        print("The Conversation Observer watches your agent usage and detects")
        print("recurring patterns (bug fixes, deployments, file work, etc.).")
        print(f"Patterns need at least {args.min_occurrences} occurrences to be suggested.")
        print()
        print("Keep using Hermes normally — task suggestions will appear as")
        print("the observer learns your workflows.")
        return 0

    print(f"Discovered {len(suggestions)} task pattern(s) from your agent usage:\n")

    saved_count = 0
    for i, pattern in enumerate(suggestions, 1):
        label = PATTERN_LABELS.get(pattern.pattern_type, pattern.pattern_type)
        print(f"  [{i}] {label}")
        print(f"      Task:        {pattern.suggested_task_name}")
        print(f"      Description: {pattern.description}")
        print(f"      Occurrences: {pattern.occurrences} sessions")
        print(f"      Confidence:  {pattern.confidence:.0%}")
        print(f"      Tools used:  {', '.join(pattern.tools_used[:5])}")
        if pattern.commands_seen:
            print(f"      Sample cmd:  {pattern.commands_seen[-1][:80]}")
        print(f"      Criteria:    {len(pattern.suggested_criteria)} suggested")

        if args.verbose:
            yaml_str = observer.suggest_task_yaml(pattern)
            print(f"\n{yaml_str}")

        if args.save:
            try:
                from agent.evolution.task_definition import TaskDefinition
                yaml_str = observer.suggest_task_yaml(pattern)
                task_path = get_task_dir() / f"{pattern.suggested_task_name}.yaml"
                task_path.parent.mkdir(parents=True, exist_ok=True)
                task_path.write_text(yaml_str)
                saved_count += 1
                print(f"      ✅ Saved → {task_path}")
            except Exception as e:
                print(f"      ❌ Save failed: {e}")
        print()

    if args.save and saved_count:
        print(f"Saved {saved_count} task(s). Review them with: hermes evolution list-tasks")
        print("Edit criteria as needed — these are suggestions based on observed patterns.")
    elif not args.save:
        print("Run with --save to auto-create these task definitions.")
        print("Or review with --verbose to see the full YAML before saving.")

    return 0


# ── Parser setup ────────────────────────────────────────────────────────


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register `hermes evolution` subcommands."""
    evo_parser = subparsers.add_parser(
        "evolution",
        help="Autonomous agent evaluation and self-improvement engine",
        description="Manage the Evolution Engine — define evaluation tasks, run benchmarks, and track autonomous improvement.",
    )
    evo_subs = evo_parser.add_subparsers(dest="evolution_action", title="subcommands")

    # status
    p_status = evo_subs.add_parser("status", help="Show Evolution Engine status and recent runs")
    p_status.set_defaults(func=_cmd_status)

    # define-task
    p_define = evo_subs.add_parser("define-task", help="Define a task from a YAML file")
    p_define.add_argument("file", help="Path to YAML task definition")
    p_define.set_defaults(func=_cmd_define_task)

    # list-tasks
    p_list = evo_subs.add_parser("list-tasks", help="List all defined evolution tasks")
    p_list.set_defaults(func=_cmd_list_tasks)

    # history
    p_hist = evo_subs.add_parser("history", help="Show evolution run history")
    p_hist.add_argument("task", nargs="?", help="Filter by task name")
    p_hist.add_argument("--limit", "-n", type=int, default=20, help="Max runs to show")
    p_hist.add_argument("--verbose", "-v", action="store_true", help="Show per-iteration details")
    p_hist.set_defaults(func=_cmd_history)

    # variants
    p_var = evo_subs.add_parser("variants", help="Show harness variant statistics")
    p_var.set_defaults(func=_cmd_variants)

    # enable / disable
    p_enable = evo_subs.add_parser("enable", help="Enable the Evolution Engine")
    p_enable.set_defaults(func=_cmd_enable)

    p_disable = evo_subs.add_parser("disable", help="Disable the Evolution Engine")
    p_disable.set_defaults(func=_cmd_disable)

    # run
    p_run = evo_subs.add_parser("run", help="Run an evolution-tracked benchmark on a task")
    p_run.add_argument("task", help="Task name to run")
    p_run.add_argument("--iterations", "-n", type=int, default=1, help="Number of iterations")
    p_run.add_argument("--cwd", "-C", help="Working directory for evaluation")
    p_run.add_argument("--verbose", "-v", action="store_true", help="Show per-criterion results")
    p_run.set_defaults(func=_cmd_run)

    # benchmark
    p_bench = evo_subs.add_parser("benchmark", help="Run full benchmark suite on all or specific tasks")
    p_bench.add_argument("task", nargs="?", help="Optional specific task to benchmark")
    p_bench.add_argument("--cwd", "-C", help="Working directory for evaluation")
    p_bench.set_defaults(func=_cmd_benchmark)

    # export
    p_export = evo_subs.add_parser("export", help="Export evolution runs as Atropos-compatible training data")
    p_export.add_argument("--run-id", help="Export a specific run by ID")
    p_export.add_argument("--all", action="store_true", help="Export all recent runs")
    p_export.add_argument("--days", type=int, default=30, help="Time window in days (default: 30)")
    p_export.add_argument("--status", help="Filter by status (comma-separated: succeeded,failed,exhausted)")
    p_export.add_argument("--domain", help="Filter by task domain")
    p_export.add_argument("--output", "-o", type=Path, help="Output file path (default: ~/.hermes/evolution/exports/)")
    p_export.set_defaults(func=_cmd_export)

    # suggest-tasks
    p_suggest = evo_subs.add_parser("suggest-tasks", help="Auto-discover task definitions from agent usage patterns")
    p_suggest.add_argument("--min-occurrences", "-n", type=int, default=3, help="Minimum pattern occurrences (default: 3)")
    p_suggest.add_argument("--save", action="store_true", help="Auto-save suggested tasks to task bank")
    p_suggest.add_argument("--verbose", "-v", action="store_true", help="Show full task YAML")
    p_suggest.set_defaults(func=_cmd_suggest_tasks)
