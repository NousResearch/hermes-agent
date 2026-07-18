"""``hermes notify`` and ``hermes omi`` command handlers.

- ``hermes notify status``         — show today's attention-budget usage,
                                     per-category thresholds, and deferred items.
- ``hermes notify keep <category>``  — record positive feedback (lower the bar).
- ``hermes notify mute <category>``  — record a dismissal (raise the bar).
- ``hermes omi scan``               — run the Omi commitment scan now.
- ``hermes omi enable|disable``      — flip the opt-in flag and (de)register the
                                     scheduled scan job.

Handlers are intentionally thin; the logic lives in ``agent.notification_budget``
and ``agent.omi_commitments``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_OMI_JOB_NAME = "omi-commitment-scan"


def notify_command(args) -> int:
    """Dispatch ``hermes notify <action>``."""
    action = getattr(args, "notify_action", None)
    if action == "status":
        return _notify_status()
    if action == "keep":
        return _notify_feedback(args.category, "act")
    if action == "mute":
        return _notify_feedback(args.category, "dismiss")
    if action == "reconcile":
        return _notify_reconcile()
    print(
        "Usage: hermes notify {status|keep <category>|mute <category>|reconcile}"
    )
    return 1


def _notify_status() -> int:
    from agent.notification_budget import budget_status

    status = budget_status()
    if "error" in status:
        print(f"notification budget: error reading status: {status['error']}")
        return 1

    enabled = "on" if status.get("enabled", True) else "off"
    print(f"Attention budget ({status['day']}) — governor {enabled}")
    print(
        f"  Delivered today: {status['allowed']} / cap {status['cap']} "
        f"(hard ceiling {status['ceiling']})"
    )

    categories = status.get("categories", {})
    if categories:
        print("  Per-category thresholds:")
        for cat, stats in sorted(categories.items()):
            if not stats:
                continue
            print(
                f"    {cat}: threshold={stats.get('threshold', 0):.2f} "
                f"p_act={stats.get('p_act_ewma', 0):.2f} "
                f"(sent={stats.get('sent_count', 0)}, "
                f"act={stats.get('act_count', 0)}, "
                f"dismiss={stats.get('dismiss_count', 0)})"
            )

    deferred = status.get("deferred", [])
    if deferred:
        print(f"  Deferred today ({len(deferred)}):")
        for row in deferred[:20]:
            print(
                f"    [{row['category']}] score={row['score']:.2f} "
                f"< thr={row['threshold_used']:.2f}"
            )
    else:
        print("  Deferred today: none")
    return 0


def _notify_feedback(category: str, signal: str) -> int:
    from agent.notification_budget import record_feedback

    record_feedback(category, signal)
    verb = "muted (bar raised)" if signal == "dismiss" else "kept (bar lowered)"
    print(f"notification budget: category '{category}' {verb}.")
    return 0


def _notify_reconcile() -> int:
    """Settle elapsed proactive notifications from implicit engagement.

    Normally runs opportunistically on `notify status` (and can be scheduled
    via cron); this exposes it as an explicit one-shot for inspection.
    """
    from agent.notification_budget import reconcile_implicit_feedback

    result = reconcile_implicit_feedback()
    if result.get("skipped") == "disabled":
        print(
            "implicit learning is disabled "
            "(set notifications.implicit_learning.enabled: true to enable)."
        )
        return 0
    if "error" in result:
        print(f"notification budget: reconcile error: {result['error']}")
        return 1
    print(
        f"notification budget: reconciled {result['reconciled']} "
        f"(acted={result['acted']}, settled={result['settled']})."
    )
    return 0


def omi_command(args) -> int:
    """Dispatch ``hermes omi <action>``."""
    action = getattr(args, "omi_action", None)
    if action == "scan":
        return _omi_scan()
    if action == "enable":
        return _omi_set_enabled(True)
    if action == "disable":
        return _omi_set_enabled(False)
    print("Usage: hermes omi {scan|enable|disable}")
    return 1


def _omi_scan() -> int:
    from agent.omi_commitments import run_omi_commitment_scan

    result = run_omi_commitment_scan()
    if result.get("skipped") == "disabled":
        print(
            "Omi commitment scan is disabled. Enable it with "
            "`hermes omi enable` (opt-in / consent)."
        )
        return 1
    if "error" in result:
        print(f"Omi scan error: {result['error']}")
        return 1
    print(
        f"Omi scan complete: scanned={result.get('scanned', 0)} "
        f"extracted={result.get('extracted', 0)} "
        f"created={result.get('created', 0)} "
        f"notified={result.get('notified', 0)}"
    )
    return 0


def _omi_set_enabled(enabled: bool) -> int:
    """Flip omi_commitments.enabled in config.yaml and (de)register the job."""
    from hermes_cli.config import (
        atomic_config_write,
        get_config_path,
        load_config,
        read_raw_config,
    )

    config_path = get_config_path()
    cfg = load_config()
    try:
        # read_raw_config distinguishes absent from unreadable; atomic_config_write
        # re-checks readability so we never clobber a degraded config, and writes
        # via a temp file + rename so a mid-write crash can't corrupt config.yaml.
        on_disk = read_raw_config() or {}
        section = dict(on_disk.get("omi_commitments", {}))
        section["enabled"] = enabled
        on_disk["omi_commitments"] = section
        atomic_config_write(config_path, on_disk, sort_keys=False)
    except Exception as exc:
        print(f"Could not update {config_path}: {exc}")
        return 1

    interval_hours = int(cfg.get("omi_commitments", {}).get("scan_interval_hours", 6))
    if enabled:
        _register_omi_job(interval_hours)
        print(
            f"Omi commitment scan ENABLED (every {interval_hours}h). "
            "Consent: the wearable transcript will be read and scanned."
        )
    else:
        _deregister_omi_job()
        print("Omi commitment scan DISABLED.")
    return 0


def _register_omi_job(interval_hours: int) -> None:
    """Print how to schedule the recurring Omi scan via `hermes cron`.

    The scan is user-scheduled rather than auto-registered so the schedule
    stays visible and editable in `hermes cron list` like any other job.
    """
    print(
        "To schedule automatic scans, first save a one-line script to "
        "~/.hermes/scripts/omi_scan.py:\n"
        "  from agent.omi_commitments import run_omi_commitment_scan as r; "
        "print(r())\n"
        "then register it:\n"
        f"  hermes cron create 'every {interval_hours} hours' "
        "--name omi-commitment-scan --no-agent --script omi_scan.py\n"
        "(or just run `hermes omi scan` manually anytime)."
    )


def _deregister_omi_job() -> None:
    """Placeholder: scheduled job is user-managed via `hermes cron`."""
    return None


def threads_command(args) -> int:
    """Dispatch ``hermes threads <action>``."""
    action = getattr(args, "threads_action", None)
    if action == "scan":
        return _threads_scan()
    if action == "list":
        return _threads_list()
    if action == "enable":
        return _threads_set_enabled(True)
    if action == "disable":
        return _threads_set_enabled(False)
    print("Usage: hermes threads {scan|list|enable|disable}")
    return 1


def _threads_scan() -> int:
    from agent.stalled_threads import run_stalled_thread_scan

    result = run_stalled_thread_scan()
    if result.get("skipped") == "disabled":
        print(
            "Stalled-thread scan is disabled. Enable it with "
            "`hermes threads enable` (opt-in / consent)."
        )
        return 1
    if "error" in result:
        print(f"Stalled-thread scan error: {result['error']}")
        return 1
    print(
        f"Stalled-thread scan complete: scanned={result.get('scanned', 0)} "
        f"candidates={result.get('candidates', 0)} "
        f"nudged={result.get('nudged', 0)} "
        f"delivered={result.get('delivered', 0)}"
    )
    return 0


def _threads_list() -> int:
    from agent.stalled_threads import list_stalled_candidates

    result = list_stalled_candidates()
    if result.get("skipped") == "disabled":
        print(
            "Stalled-thread scan is disabled. Enable it with "
            "`hermes threads enable` (opt-in / consent)."
        )
        return 1
    candidates = result.get("candidates", [])
    if not candidates:
        print("No open-loop candidates right now.")
        return 0
    print(f"Open-loop candidates ({len(candidates)}):")
    for c in candidates:
        print(f"  [{c.get('kind')}] {c.get('text', '')[:100]}")
    print("(dry run — nothing was nudged; run `hermes threads scan` to act)")
    return 0


def _threads_set_enabled(enabled: bool) -> int:
    """Flip stalled_threads.enabled in config.yaml (atomic, clobber-guarded)."""
    from hermes_cli.config import (
        atomic_config_write,
        get_config_path,
        load_config,
        read_raw_config,
    )

    config_path = get_config_path()
    cfg = load_config()
    try:
        on_disk = read_raw_config() or {}
        section = dict(on_disk.get("stalled_threads", {}))
        section["enabled"] = enabled
        on_disk["stalled_threads"] = section
        atomic_config_write(config_path, on_disk, sort_keys=False)
    except Exception as exc:
        print(f"Could not update {config_path}: {exc}")
        return 1

    interval_hours = int(cfg.get("stalled_threads", {}).get("scan_interval_hours", 12))
    if enabled:
        _register_threads_job(interval_hours)
        print(
            f"Stalled-thread scan ENABLED (every {interval_hours}h). "
            "Consent: your kanban cards and conversation threads will be scanned."
        )
    else:
        print("Stalled-thread scan DISABLED.")
    return 0


def _register_threads_job(interval_hours: int) -> None:
    """Print how to schedule the recurring stalled-thread scan via `hermes cron`."""
    print(
        "To schedule automatic scans, first save a one-line script to "
        "~/.hermes/scripts/threads_scan.py:\n"
        "  from agent.stalled_threads import run_stalled_thread_scan as r; "
        "print(r())\n"
        "then register it:\n"
        f"  hermes cron create 'every {interval_hours} hours' "
        "--name stalled-thread-scan --no-agent --script threads_scan.py\n"
        "(or just run `hermes threads scan` manually anytime)."
    )


def brief_command(args) -> int:
    """Dispatch ``hermes brief <action>``."""
    action = getattr(args, "brief_action", None)
    if action == "show":
        return _brief_show()
    if action == "send":
        return _brief_send()
    if action == "enable":
        return _brief_set_enabled(True)
    if action == "disable":
        return _brief_set_enabled(False)
    print("Usage: hermes brief {show|send|enable|disable}")
    return 1


def _brief_show() -> int:
    from agent.morning_brief import render_brief

    result = render_brief()
    if "error" in result:
        print(f"Morning brief error: {result['error']}")
        return 1
    text = result.get("text", "")
    if not text:
        print("(brief is empty)")
        return 0
    print(text)
    print(f"\n[dry run — {result.get('items', 0)} item(s); not sent]")
    return 0


def _brief_send() -> int:
    from agent.morning_brief import run_morning_brief

    result = run_morning_brief(force=False)
    if result.get("skipped") == "disabled":
        print(
            "Morning brief is disabled. Enable it with "
            "`hermes brief enable` (opt-in / consent)."
        )
        return 1
    if result.get("skipped") == "empty":
        print(
            f"Nothing to send: {result.get('items', 0)} item(s) below the "
            "min_items_to_send threshold. Preview with `hermes brief show`."
        )
        return 0
    if "error" in result:
        print(f"Morning brief error: {result['error']}")
        return 1
    print(
        f"Morning brief: items={result.get('items', 0)} "
        f"delivered={result.get('delivered', 0)}"
    )
    return 0


def _brief_set_enabled(enabled: bool) -> int:
    """Flip morning_brief.enabled in config.yaml (atomic, clobber-guarded)."""
    from hermes_cli.config import (
        atomic_config_write,
        get_config_path,
        load_config,
        read_raw_config,
    )

    config_path = get_config_path()
    cfg = load_config()
    try:
        on_disk = read_raw_config() or {}
        section = dict(on_disk.get("morning_brief", {}))
        section["enabled"] = enabled
        on_disk["morning_brief"] = section
        atomic_config_write(config_path, on_disk, sort_keys=False)
    except Exception as exc:
        print(f"Could not update {config_path}: {exc}")
        return 1

    interval_hours = int(cfg.get("morning_brief", {}).get("scan_interval_hours", 24))
    if enabled:
        _register_brief_job(interval_hours)
        print(
            "Morning brief ENABLED. Consent: your open loops (kanban, threads, "
            "recent Omi) will be composed into a daily digest."
        )
    else:
        print("Morning brief DISABLED.")
    return 0


def _register_brief_job(interval_hours: int) -> None:
    """Print how to schedule the daily brief via `hermes cron`."""
    print(
        "To schedule the daily brief, first save a one-line script to "
        "~/.hermes/scripts/brief_scan.py:\n"
        "  from agent.morning_brief import run_morning_brief as r; print(r())\n"
        "then register it (7am daily shown; adjust the cron expression):\n"
        "  hermes cron create '0 7 * * *' "
        "--name morning-brief --no-agent --script brief_scan.py\n"
        "(or just run `hermes brief send` manually anytime)."
    )


def radar_command(args) -> int:
    """Dispatch ``hermes radar <action>``."""
    action = getattr(args, "radar_action", None)
    if action == "scan":
        return _radar_scan()
    if action == "list":
        return _radar_list()
    if action == "enable":
        return _radar_set_enabled(True)
    if action == "disable":
        return _radar_set_enabled(False)
    print("Usage: hermes radar {scan|list|enable|disable}")
    return 1


def _radar_scan() -> int:
    from agent.deadline_radar import run_deadline_radar

    result = run_deadline_radar()
    if result.get("skipped") == "disabled":
        print(
            "Deadline radar is disabled. Enable it with "
            "`hermes radar enable` (opt-in / consent)."
        )
        return 1
    if "error" in result:
        print(f"Deadline radar error: {result['error']}")
        return 1
    print(
        f"Deadline radar complete: scanned={result.get('scanned', 0)} "
        f"candidates={result.get('candidates', 0)} "
        f"nudged={result.get('nudged', 0)} "
        f"delivered={result.get('delivered', 0)}"
    )
    return 0


def _radar_list() -> int:
    from agent.deadline_radar import list_upcoming_deadlines

    result = list_upcoming_deadlines()
    if result.get("skipped") == "disabled":
        print(
            "Deadline radar is disabled. Enable it with "
            "`hermes radar enable` (opt-in / consent)."
        )
        return 1
    if "error" in result:
        print(f"Deadline radar error: {result['error']}")
        return 1
    candidates = result.get("candidates", [])
    if not candidates:
        print("No upcoming deadlines within the lead-time window.")
        return 0
    print(f"Upcoming deadlines ({len(candidates)}):")
    for c in candidates:
        print(f"  {c.get('text', '')[:100]}")
    print("(dry run — nothing was nudged; run `hermes radar scan` to act)")
    return 0


def _radar_set_enabled(enabled: bool) -> int:
    """Flip deadline_radar.enabled in config.yaml (atomic, clobber-guarded)."""
    from hermes_cli.config import (
        atomic_config_write,
        get_config_path,
        load_config,
        read_raw_config,
    )

    config_path = get_config_path()
    cfg = load_config()
    try:
        on_disk = read_raw_config() or {}
        section = dict(on_disk.get("deadline_radar", {}))
        section["enabled"] = enabled
        on_disk["deadline_radar"] = section
        atomic_config_write(config_path, on_disk, sort_keys=False)
    except Exception as exc:
        print(f"Could not update {config_path}: {exc}")
        return 1

    interval_hours = int(cfg.get("deadline_radar", {}).get("scan_interval_hours", 4))
    if enabled:
        _register_radar_job(interval_hours)
        print(
            f"Deadline radar ENABLED (every {interval_hours}h). "
            "Consent: your kanban cards' due dates will be scanned."
        )
    else:
        print("Deadline radar DISABLED.")
    return 0


def _register_radar_job(interval_hours: int) -> None:
    """Print how to schedule the recurring deadline radar via `hermes cron`."""
    print(
        "To schedule automatic scans, first save a one-line script to "
        "~/.hermes/scripts/radar_scan.py:\n"
        "  from agent.deadline_radar import run_deadline_radar as r; print(r())\n"
        "then register it:\n"
        f"  hermes cron create 'every {interval_hours} hours' "
        "--name deadline-radar --no-agent --script radar_scan.py\n"
        "(or just run `hermes radar scan` manually anytime)."
    )
