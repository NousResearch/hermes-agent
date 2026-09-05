"""``hermes cron`` subcommand parser."""

from __future__ import annotations

from typing import Callable

from hermes_cli.subcommands._shared import add_accept_hooks_flag


def _flag(parser, *names, help, **kw):
    parser.add_argument(*names, action="store_true", help=help, **kw)


def build_cron_parser(subparsers, *, cmd_cron: Callable) -> None:
    """Attach the ``cron`` subcommand (and its sub-actions) to ``subparsers``."""
    cron_parser = subparsers.add_parser(
        "cron", help="Cron job management", description="Manage scheduled tasks")
    cron_subparsers = cron_parser.add_subparsers(dest="cron_command")

    cron_list = cron_subparsers.add_parser("list", help="List scheduled jobs")
    _flag(cron_list, "--all", help="Include disabled jobs")

    cron_create = cron_subparsers.add_parser(
        "create", aliases=["add"], help="Create a scheduled job")
    cron_create.add_argument("schedule", help="Schedule like '30m', 'every 2h', or '0 9 * * *'")
    cron_create.add_argument(
        "prompt", nargs="?", help="Optional self-contained prompt or task instruction")
    cron_create.add_argument("--name", help="Optional human-friendly job name")
    cron_create.add_argument("--deliver",
        help="Delivery target: origin, local, telegram, discord, signal, "
            "platform:chat_id, or bot-chat[:profile] (inject output into a "
            "local profile's canonical Bot Chat as a message the bot responds to)")
    cron_create.add_argument("--failure-deliver", dest="failure_deliver",
        help="Override target for FAILURE notices only (same grammar as "
            "--deliver). 'local' suppresses failure notices entirely; run "
            "state stays visible in `hermes cron list`. Omit = failures "
            "follow --deliver.")
    cron_create.add_argument("--repeat", type=int, help="Optional repeat count")
    cron_create.add_argument("--skill", dest="skills", action="append",
        help="Attach a skill. Repeat to add multiple skills.")
    cron_create.add_argument("--script",
        help="Path to a script under ~/.hermes/scripts/. Default mode: "
            "script stdout is injected into the agent's prompt each run. "
            "With --no-agent: the script IS the job and its stdout is "
            "delivered verbatim. .sh/.bash files run via bash, everything "
            "else via Python.")
    _flag(cron_create, "--no-agent", dest="no_agent", default=False,
        help="Skip the LLM entirely — run --script on schedule and deliver "
            "its stdout directly. Empty stdout = silent. Classic watchdog "
            "pattern (memory alerts, disk alerts, CI pings).")
    cron_create.add_argument("--monitor-script", dest="monitor_script",
        help="Monitor mode: path to a cheap source script under "
            "~/.hermes/scripts/ that runs each tick BEFORE the agent. "
            "Unchanged output (exact-bytes hash) suppresses the agent run "
            "entirely; changed output injects a MONITOR CHANGE DETECTED "
            "diff into the prompt. Script output must be stable (no "
            "timestamps). Mutually exclusive with --monitor-url; "
            "incompatible with --no-agent.")
    cron_create.add_argument("--monitor-url", dest="monitor_url",
        help="Monitor mode: http(s) URL fetched with a bounded GET each tick "
            "instead of a script. Same hash-suppression semantics as "
            "--monitor-script.")
    cron_create.add_argument("--workdir",
        help="Absolute path for the job to run from. Injects AGENTS.md / CLAUDE.md / .cursorrules from that directory and uses it as the cwd for terminal/file/code_exec tools. Omit to preserve old behaviour (no project context files).",
    )
    cron_create.add_argument("--model",
        help="Pin this job to a specific inference model (user-owned; the "
            "agent's cronjob tool cannot set this). Omit to follow "
            "cron.model / model.default from config.yaml.")
    cron_create.add_argument("--provider", dest="model_provider",
        help="Inference provider paired with --model (e.g. 'openrouter', 'nous').")
    cron_create.add_argument("--reasoning-effort", dest="reasoning_effort",
        help="Pin this job's reasoning (thinking) effort: none, minimal, low, "
            "medium, high, xhigh, max, or ultra. Overrides agent.reasoning_effort "
            "and agent.reasoning_overrides for this job; unsupported levels are "
            "clamped by the provider at request time. Omit to follow config.")
    cron_create.add_argument(
        "--continuity", dest="continuity", action="store_const", const=True, default=None,
        help="Each run wakes up with the job's own previous output injected "
            "into its prompt, so it can dedupe against what was already "
            "reported and continue where the last run left off (scouts, "
            "monitors, incremental digests). First run is unchanged.")
    cron_create.add_argument(
        "--attach-to-session",
        dest="attach_to_session",
        action="store_const",
        const=True,
        default=None,
        help=(
            "Make the job's delivery a continuable session: the report arrives "
            "as a real message the user can reply to, with the run's context in "
            "scope (briefings, routines that ask follow-up questions). Absent = "
            "one-way delivery."
        ),
    )
    cron_create.add_argument(
        "--prompt-file",
        dest="prompt_file",
        help=(
            "Path to a file whose contents ARE the prompt, read fresh on every "
            "run so edits to the file land on the next fire (single source of "
            "truth for a routine file). Stored as an absolute path. An inline "
            "prompt argument, if given, wins over this. Create-only; not "
            "editable via `cron edit`."
        ),
    )
    cron_create.add_argument(
        "--no-cron-hint",
        dest="no_cron_hint",
        action="store_const",
        const=True,
        default=None,
        help=(
            "Suppress the execution-guidance banner normally prepended to a "
            "cron job's prompt (delivery + [SILENT] convention). Use for jobs "
            "that always deliver via an attached session and manage their own "
            "output."
        ),
    )

    cron_edit = cron_subparsers.add_parser("edit", help="Edit an existing scheduled job")
    cron_edit.add_argument("job_id", help="Job ID to edit")
    cron_edit.add_argument("--schedule", help="New schedule")
    cron_edit.add_argument("--prompt", help="New prompt/task instruction")
    cron_edit.add_argument("--name", help="New job name")
    cron_edit.add_argument("--deliver", help="New delivery target")
    cron_edit.add_argument("--failure-deliver", dest="failure_deliver",
        help="Override target for failure notices (same grammar as --deliver; "
            "'local' suppresses; '' clears the override)")
    cron_edit.add_argument("--repeat", type=int, help="New repeat count")
    cron_edit.add_argument("--skill", dest="skills", action="append",
        help="Replace the job's skills with this set. Repeat to attach multiple skills.")
    cron_edit.add_argument("--add-skill", dest="add_skills", action="append",
        help="Append a skill without replacing the existing list. Repeatable.")
    cron_edit.add_argument("--remove-skill", dest="remove_skills", action="append",
        help="Remove a specific attached skill. Repeatable.")
    _flag(cron_edit, "--clear-skills", help="Remove all attached skills from the job")
    cron_edit.add_argument("--script",
        help="Path to a script under ~/.hermes/scripts/. Pass empty string to clear. "
            "With --no-agent the script IS the job; otherwise its stdout is "
            "injected into the agent's prompt each run.")
    cron_edit.add_argument(
        "--no-agent", dest="no_agent", action="store_const", const=True, default=None,
        help="Enable no-agent mode on this job (requires --script or an "
            "existing script on the job).")
    cron_edit.add_argument("--agent", dest="no_agent", action="store_const", const=False,
        help="Disable no-agent mode on this job (reverts to LLM-driven execution).")
    cron_edit.add_argument(
        "--continuity", dest="continuity", action="store_const", const=True, default=None,
        help="Turn on run-to-run continuity: each run sees the job's own "
            "previous output (dedupe, continue where it left off).")
    cron_edit.add_argument("--no-continuity", dest="continuity", action="store_const", const=False,
        help=("Turn off run-to-run continuity (other context_from job refs are preserved)."))
    cron_edit.add_argument("--monitor-script", dest="monitor_script",
        help="Set/replace the monitor source script (see `hermes cron create "
            "--monitor-script`). Pass empty string to clear.")
    cron_edit.add_argument("--monitor-url", dest="monitor_url",
        help=("Set/replace the monitor source URL. Pass empty string to clear."))
    cron_edit.add_argument("--workdir",
        help="Absolute path for the job to run from (injects AGENTS.md etc. and sets terminal cwd). Pass empty string to clear.",
    )
    cron_edit.add_argument("--model",
        help="Pin this job to a specific inference model (user-owned; the "
            "agent's cronjob tool cannot set this). Pass empty string to "
            "clear the pin and follow cron.model / model.default.")
    cron_edit.add_argument("--provider", dest="model_provider",
        help="Inference provider paired with --model. Pass empty string to clear.")
    cron_edit.add_argument("--reasoning-effort", dest="reasoning_effort",
        help="Pin this job's reasoning (thinking) effort: none, minimal, low, "
            "medium, high, xhigh, max, or ultra. Pass empty string to clear "
            "the pin and follow config resolution.")

    # lifecycle actions
    cron_pause = cron_subparsers.add_parser("pause", help="Pause a scheduled job")
    cron_pause.add_argument("job_id", help="Job ID to pause")

    cron_resume = cron_subparsers.add_parser("resume", help="Resume a paused job")
    cron_resume.add_argument("job_id", help="Job ID to resume")
    cron_resume.add_argument("--at", dest="run_at", help="Re-arm at an ISO-8601 time")
    _flag(cron_resume, "--run-now", help="Re-arm to run now")

    cron_run = cron_subparsers.add_parser("run", help="Run a job on the next scheduler tick")
    cron_run.add_argument("job_id", help="Job ID to trigger")
    add_accept_hooks_flag(cron_run)

    cron_remove = cron_subparsers.add_parser(
        "remove", aliases=["rm", "delete"], help="Remove a scheduled job")
    cron_remove.add_argument("job_id", help="Job ID to remove")

    cron_subparsers.add_parser("status", help="Check if cron scheduler is running")

    cron_runs = cron_subparsers.add_parser(
        "runs", aliases=["history"], help="Show durable execution attempts")
    cron_runs.add_argument("job_id", nargs="?", help="Optional job ID filter")
    cron_runs.add_argument("--limit", type=int, default=20, help="Rows to show (1-500)")

    # cron incidents — durable failure incidents (list/ack)
    cron_incidents = cron_subparsers.add_parser(
        "incidents", help="List or acknowledge durable cron failure incidents")
    cron_incidents.add_argument("--state", choices=["detected", "alerted", "closed"],
        help="Filter incidents by lifecycle state")
    cron_incidents.add_argument(
        "incident_action", nargs="?", default="list", choices=["list", "ack"],
        help="Action (default: list)")
    cron_incidents.add_argument("incident_id", nargs="?", help="Incident ID to acknowledge (ack)")

    # notepad: per-job durable KV, injected into the job prompt each run.
    cron_notepad = cron_subparsers.add_parser(
        "notepad", help="Read/write a job's durable notepad (persistent KV across runs)")
    cron_notepad.add_argument("job_id", help="Job ID the notepad belongs to")
    cron_notepad.add_argument(
        "notepad_action", nargs="?", default="list", choices=["get", "set", "delete", "list"],
        help="Action (default: list)")
    cron_notepad.add_argument("key", nargs="?", help="Notepad key (get/set/delete)")
    cron_notepad.add_argument("value", nargs="?", help="Value to store (set)")

    cron_subparsers.add_parser("doctor", help="Check scheduled jobs for common health issues")

    cron_tick = cron_subparsers.add_parser("tick", help="Run due jobs once and exit")
    add_accept_hooks_flag(cron_tick)
    add_accept_hooks_flag(cron_parser)
    cron_parser.set_defaults(func=cmd_cron)
