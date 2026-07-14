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
    """Show Evolution Engine dashboard — status, improvement, auto-discoveries."""
    from agent.evolution.config import EvolutionConfig
    config = EvolutionConfig.from_config()

    print("═" * 55)
    print(f"  HERMES EVOLUTION ENGINE")
    print(f"  Status: {'🟢 ENABLED' if config.enabled else '⚫ DISABLED'}")
    print(f"  Mode: {config.mode} | Max Iterations: {config.max_iterations} | Safety: {'on' if config.regression_enabled else 'off'}")
    print("═" * 55)

    if not config.enabled:
        print("\n  Enable: hermes evolution enable")
        return 0

    # ── Run History ──
    from agent.evolution.evolution_store import get_evolution_store
    store = get_evolution_store()
    runs = store.list_runs(limit=20)

    if runs:
        succeeded = sum(1 for r in runs if r["status"] == "succeeded")
        failed = sum(1 for r in runs if r["status"] == "failed")
        total = len(runs)
        print(f"\n  📊 RUN HISTORY ({total} total)")
        bar_w = 30
        s_bar = "█" * int(succeeded/total*bar_w) if total else ""
        f_bar = "░" * int(failed/total*bar_w) if total else ""
        print(f"  Passed:  {succeeded:3d} {s_bar}")
        print(f"  Failed:  {failed:3d} {f_bar}")

    # ── Improvement Metrics ──
    try:
        from agent.evolution.improvement_metrics import get_tracker
        tracker = get_tracker()
        report = tracker.generate_report()
        s = report["summary"]
        if s["total_records"] > 0:
            print(f"\n  📈 IMPROVEMENT PROOF")
            sig = "✅ Significant (p<0.05)" if s["statistically_significant"] else "⏳ Need more data"
            print(f"  Records: {s['total_records']} | Mean Δ: {s['mean_improvement']:+.3f} | Effect: {s['effect_size_label']} (d={s['effect_size']:.2f})")
            print(f"  {sig}")
            if report["tasks_improving"]:
                print(f"  Improving: {', '.join(report['tasks_improving'][:5])}")
    except Exception:
        pass

    # ── Auto-Discoveries ──
    try:
        from agent.evolution.conversation_observer import get_observer
        observer = get_observer()
        stats = observer.get_stats()
        clusters = observer.suggest_tasks(min_occurrences=2, min_confidence=0.3)
        if stats["total_sessions_observed"] > 0:
            print(f"\n  🔍 AUTO-DISCOVERY")
            print(f"  Sessions observed: {stats['total_sessions_observed']}")
            print(f"  Clusters found:    {stats['total_clusters']}")
            print(f"  Ready for tasks:   {len(clusters)}")
            for c in clusters[:3]:
                print(f"    • {c.task_name}: {c.occurrence_count} sessions, {c.confidence:.0%} confidence")
    except Exception:
        pass

    # ── Pre-built Tasks ──
    try:
        from agent.evolution.task_definition import list_tasks
        from pathlib import Path
        tasks = list_tasks()
        bundled = [t for t in tasks if t.name in {
            'bug-fix-verify', 'code-review', 'deploy-verify', 'data-pipeline',
            'api-endpoint', 'security-audit', 'document-generation',
            'refactor-module', 'dependency-update', 'config-migration',
        }]
        custom = [t for t in tasks if t not in bundled]
        if bundled:
            print(f"\n  📦 PRE-BUILT TASKS ({len(bundled)} available)")
            print(f"  Run: hermes evolution benchmark")
            print(f"  Define custom: hermes evolution define-task <file.yaml>")
    except Exception:
        pass

    # ── Training Data ──
    try:
        from agent.evolution.atropos_export import get_export_stats
        es = get_export_stats(days=30)
        if es["total_runs"] > 0:
            print(f"\n  🎯 ATROPOS TRAINING DATA")
            print(f"  Records available: {es['estimated_training_records']} (last 30 days)")
            print(f"  Export: hermes evolution export --all")
    except Exception:
        pass

    # ── Pending PR Proposals ──
    try:
        from agent.evolution.pr_proposer import PRProposer
        proposer = PRProposer()
        lineages = proposer.get_all_lineages()
        pending = [l for l in lineages if not l.resolved]
        if pending:
            print(f"\n  📝 PENDING CODE FIXES ({len(pending)})")
            for lin in pending[:3]:
                print(f"    • {lin.failure_type}: {lin.occurrences} occurrences, {len(lin.generations)} generations")
            print(f"    Review: hermes evolution pr-status")
    except Exception:
        pass

    # ── Quick Start ──
    if not runs:
        benchtasks = []
        try:
            from agent.evolution.task_definition import list_tasks
            benchtasks = list_tasks()
        except Exception:
            pass
        print(f"\n  🚀 QUICK START")
        if benchtasks:
            print(f"  {len(benchtasks)} pre-built tasks ready.")
        print(f"  hermes evolution benchmark           # Run all tasks")
        print(f"  hermes evolution run bug-fix-verify  # Run a specific task")
        print(f"  hermes evolution suggest-tasks       # Auto-discover from your usage")

    print()
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

        store = None
        try:
            from agent.evolution.evolution_store import get_evolution_store
            store = get_evolution_store()
        except Exception: pass

        for i in range(iterations):
            start = _time.monotonic()
            result = evaluator.evaluate(task, None, ctx)
            elapsed = _time.monotonic() - start

            icon = "✅" if result.passed else "❌"
            print(f"  [{i+1}/{iterations}] {icon} score={result.score:.2f} time={elapsed:.2f}s")
            total_score += result.score
            if result.passed:
                passed_count += 1
                if store:
                    try:
                        rid = store.create_run(task_name=task_name, task_domain=task.domain or "general",
                                              task_complexity=task.complexity or 1)
                        store.add_iteration(rid, i+1, "evaluating", score=result.score)
                        store.update_run_status(rid, "succeeded", final_score=result.score)
                        if result.passed:
                            store.set_baseline(task_name, result.score)
                    except Exception: pass

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
        store = None
        try:
            from agent.evolution.evolution_store import get_evolution_store
            store = get_evolution_store()
        except Exception: pass

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
                if store:
                    try:
                        rid = store.create_run(task_name=task.name, task_domain=task.domain,
                                              task_complexity=task.complexity)
                        store.update_run_status(rid, "succeeded", final_score=result.score)
                        store.set_baseline(task.name, result.score)
                    except Exception: pass
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
        from agent.evolution.conversation_observer import get_observer
        from agent.evolution.task_definition import save_task, get_task_dir
    except Exception as e:
        print(f"Error loading observer module: {e}")
        return 1

    observer = get_observer()
    suggestions = observer.suggest_tasks(min_occurrences=args.min_occurrences)

    if not suggestions:
        stats = observer.get_stats()
        print("No task clusters ready for definition yet.\n")
        print(f"  Sessions observed: {stats["total_sessions_observed"]}")
        print(f"  Clusters forming:  {stats["total_clusters"]}")
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
    for i, cluster in enumerate(suggestions, 1):
        print(f"  [{i}] {cluster.task_name}")
        print(f"      Description:  {cluster.description}")
        print(f"      Occurrences:  {cluster.occurrence_count} sessions")
        print(f"      Confidence:   {cluster.confidence:.0%} (α={1+cluster.positive_evidence} β={1+cluster.negative_evidence})")
        print(f"      Complexity:   {cluster.estimated_complexity}/14")
        print(f"      Criteria:     {len(cluster.suggested_criteria)} suggested")

        if args.verbose:
            yaml_str = observer.suggest_task_yaml(cluster)
            print(f"\n{yaml_str}")

        if args.save:
            try:
                yaml_str = observer.suggest_task_yaml(cluster)
                task_path = get_task_dir() / f"{cluster.task_name}.yaml"
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


# ── improvement ────────────────────────────────────────────────────────


def _cmd_improvement(args) -> int:
    """Show statistical proof of agent improvement over time."""
    try:
        from agent.evolution.improvement_metrics import get_tracker
    except Exception as e:
        print(f"Error loading metrics module: {e}")
        return 1

    tracker = get_tracker()
    report = tracker.generate_report()
    s = report["summary"]

    print("Agent Improvement Report")
    print("=" * 50)
    print(f"  Records:            {s['total_records']}")
    print(f"  Unique tasks:       {s['unique_tasks']}")
    print(f"  Mean improvement:   {s['mean_improvement']:+.3f}")
    print(f"  Median improvement: {s['median_improvement']:+.3f}")
    print(f"  Improvement rate:   {s['improvement_rate']:.0%}")
    print(f"  Effect size:        {s['effect_size']:.3f} ({s['effect_size_label']})")
    print(f"  Wilcoxon p-value:   {s['wilcoxon_p_value']:.4f}")
    print(f"  Significant (p<.05): {'✅ YES' if s['statistically_significant'] else '❌ not yet'}")
    print()

    if report["tasks_improving"]:
        print(f"  Tasks improving: {len(report['tasks_improving'])}")
        for name in report["tasks_improving"][:10]:
            td = report["task_details"].get(name, {})
            if td:
                print(f"    {name}: {td.get('first_half_mean', 0):.2f} → {td.get('second_half_mean', 0):.2f} ({td.get('improvement', 0):+.2f})")

    if s["total_records"] < 5:
        print(f"\n  Need {5 - s['total_records']} more improvements for statistical testing.")
    elif s["statistically_significant"]:
        print(f"\n  ✅ HAEE produces statistically significant improvement (p={s['wilcoxon_p_value']:.3f})")
        print(f"  Effect size: {s['effect_size_label']} (d={s['effect_size']:.2f})")

    return 0


# ── pr-status ──────────────────────────────────────────────────────────


def _cmd_pr_status(args) -> int:
    """Show pending code fix proposals from the PR proposer."""
    try:
        from agent.evolution.pr_proposer import PRProposer
        proposer = PRProposer()
    except Exception as e:
        print(f"Error: {e}")
        return 1

    lineages = proposer.get_all_lineages()
    if not lineages:
        print("No code fix proposals pending.")
        print()
        print("The PR proposer creates proposals when HAEE detects")
        print("recurring tool-level failures during normal usage.")
        print("Keep using Hermes — proposals appear automatically.")
        return 0

    print(f"Pending Code Fix Proposals ({len(lineages)}):\n")
    for lin in lineages:
        status = "✅ resolved" if lin.resolved else "🔧 pending"
        print(f"  {status} {lin.failure_type}")
        print(f"    Occurrences: {lin.occurrences} | Generations: {len(lin.generations)}")
        if lin.improvement_delta:
            print(f"    Improvement: {lin.improvement_delta:+.3f}")
        if not lin.resolved:
            print(f"    Approve: hermes evolution approve-pr <tool>")
        print()

    return 0


def _cmd_approve_pr(args) -> int:
    """Approve a pending code fix proposal and create the PR branch."""
    tool = args.tool
    try:
        from agent.evolution.pr_proposer import PRProposer
        proposer = PRProposer()

        # Find the lineage for this tool
        for lin in proposer.get_all_lineages():
            if tool.lower() in lin.failure_id.lower() or tool.lower() in lin.failure_type.lower():
                # Find the selected candidate
                for cid in lin.generations:
                    for c in proposer._candidates:
                        if c.id == cid and c.selected:
                            result = proposer.create_pr(c)
                            if "branch" in result:
                                print(f"✅ PR branch created: {result['branch']}")
                                print(f"   Push: git push origin {result['branch']}")
                                print(f"   Create PR: gh pr create --title 'fix({tool}): address {lin.failure_type}'")
                                return 0
                print(f"No selected candidate found for '{tool}'.")
                print(f"Run 'hermes evolution pr-status' to see pending proposals.")
                return 1

        print(f"No pending proposal found for '{tool}'.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


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

    # improvement
    p_improve = evo_subs.add_parser("improvement", help="Show statistical proof of agent improvement over time")
    p_improve.set_defaults(func=_cmd_improvement)

    # pr-status
    p_pr = evo_subs.add_parser("pr-status", help="Show pending code fix proposals awaiting review")
    p_pr.set_defaults(func=_cmd_pr_status)

    # approve-pr
    p_apr = evo_subs.add_parser("approve-pr", help="Approve a pending code fix proposal")
    p_apr.add_argument("tool", help="Tool name to approve a PR for")
    p_apr.set_defaults(func=_cmd_approve_pr)
