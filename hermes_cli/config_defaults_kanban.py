"""Default values for Kanban orchestration and dispatch configuration.

Pure data imported by ``config_defaults.DEFAULT_CONFIG``.
"""

KANBAN_DEFAULTS = {
    # Kanban multi-agent coordination. The dispatcher ticks every N seconds, reclaims stale claims,
    # promotes dependency-satisfied todos to ready, and fires `hermes -p <assignee> chat -q ...` per
    # claimable task. Run ONE dispatcher per profile; two on the same kanban.db race for claims.
    # Auto-subscribe the originating gateway/TUI session to completion + block events when
    # kanban_create is called from a session with a persistent delivery channel. Disable for
    # profiles that prefer explicit kanban_notify-subscribe calls per task.
    "auto_subscribe_on_create": True,
    # Poll and deliver Kanban subscriptions from this gateway. Disable on profiles that do
    # not own notification subscriptions to avoid an idle five-second board probe.
    "notify_in_gateway": True,
    # Run the dispatcher inside the gateway process (~300µs per idle tick). False only if you
    # run it as a separate unit or don't want the gateway spawning workers.
    "dispatch_in_gateway": True,
    # Auto-claim tasks in the review column and spawn the assigned profile with the bundled
    # sdlc-review skill. Disable where every review is done manually from the dashboard.
    "review_dispatch": True,
    # Seconds between dispatcher ticks. Lower = snappier pickup; higher = less SQL pressure.
    "dispatch_interval_seconds": 60,
    # Auto-block after this many consecutive non-success attempts (spawn_failed, timed_out,
    # crashed) for the same task/profile. Reassignment resets the streak.
    "failure_limit": 2,
    # Worker stdout/stderr log rotation at spawn time (2 MiB + one backup). Raise to keep more
    # early failure evidence from long-running workers.
    "worker_log_rotate_bytes": 2 * 1024 * 1024,
    "worker_log_backup_count": 1,
    # Profile for the root/orchestration task after Triage decomposition; "" = default profile.
    # Does not control the decomposer LLM path (see auxiliary.kanban_decomposer).
    "orchestrator_profile": "",
    # Assignee when the orchestrator can't match one to an installed profile; "" = default
    # profile. A task never ends up with assignee=None.
    "default_assignee": "",
    # Global cap: positive int = the HOST never has more than N tasks 'running' across all
    # boards and both dispatch lanes. None = ~MemTotal / 512 MiB clamped to [2, 8]; where
    # MemTotal is unreadable (macOS/Windows) None means no cap.
    # Global concurrency cap (#33488): when set to a positive int, the HOST never has more than N tasks
    # in 'running' at once — counted across every active board and across both the ready and review
    # dispatch lanes (workers are OS processes sharing one machine's memory, so the cap bounds the
    # machine, not each board; OOF-30). Unset (None) means "derive from system memory" (OOF-30/OOF-77):
    # the dispatcher caps concurrency at roughly MemTotal / 512 MiB, clamped to [2, 8] — e.g. 2 workers
    # on a 1 GiB VM. On hosts where total memory can't be read (macOS/Windows), unset falls back to no
    # cap. Set an explicit value to override the derived default in either direction.
    "max_in_progress": None,
    # Per-profile cap: positive int = no single profile runs more than N workers even if the
    # global caps allow; blocked tasks defer to the next tick. None = no per-profile cap. Useful
    # when fan-out would saturate one profile's model/API quota/browser pool.
    # Unset (None) means "no per-profile cap" — backward-compatible with existing installs. Useful for
    # fan-out workflows that would otherwise saturate one profile's local model / API quota / browser
    # pool while leaving other profiles idle. See #21582.
    "max_in_progress_per_profile": None,
    # Per-home claim allowlist for boards shared across Hermes homes (#110995): profile names
    # this home's dispatcher may claim (list or comma-separated string). None = any existing
    # profile is claimable. Set = fail-closed (an empty list claims nothing). Every home has a
    # root profile named "default", so on a shared kanban.db every home can otherwise claim
    # default-assigned cards.
    "dispatch_profiles": None,
    # Profiles omitted from built-in Kanban decomposition. This narrows the eligible roster without
    # changing the dispatchability of explicitly assigned cards.
    "auto_decompose_excluded_profiles": [],
    # Auto-run the decomposer on Triage tasks every tick. False = manual via `hermes kanban
    # decompose <id>` or the dashboard's Decompose button.
    "auto_decompose": True,
    # Max triage tasks decomposed per tick, bounding the aux-LLM burst from a bulk load. Excess
    # defers to the next tick.
    "auto_decompose_per_tick": 3,
    # Running tasks with no heartbeat (last_heartbeat_at) for this many seconds are reclaimed to
    # ready on the next tick; a still-running local worker is terminated first. 0 = off.
    "dispatch_stale_timeout_seconds": 14400,
    # Each tick, requeue 'running' cards with broken claim bookkeeping (claim_lock or
    # claim_expires NULL with a dead worker) that TTL/crash/stale recovery can't see. False
    # keeps orphans frozen for manual forensics.
    "reconcile_orphans": True,
    # Notify subscriptions survive `done` (completion is reversible) and are removed on archive.
    # On boards that never archive, the notifier GC purges subscriptions for tasks done with no
    # activity for this many days, bounding sub-table growth on boards that never archive. 0 = off.
    "done_sub_retention_days": 30,
}
