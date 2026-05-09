"""
Cron job scheduler - executes due jobs.

Provides tick() which checks for due jobs and runs them. The gateway
calls this every 60 seconds from a background thread.

Uses a file-based lock (~/.hermes/cron/.tick.lock) so only one tick
runs at a time if multiple processes overlap.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import traceback

# fcntl is Unix-only; on Windows use msvcrt for file locking
try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        msvcrt = None
from pathlib import Path
from hermes_constants import get_hermes_home
from hermes_cli.config import load_config
from typing import Optional

from hermes_time import now as _hermes_now

logger = logging.getLogger("cron.scheduler")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cron.jobs import (
    RATCHET_WINDOW_RUNS,
    advance_next_run,
    get_due_jobs,
    load_recent_job_responses,
    mark_job_run,
    save_job_output,
    should_check_persistence_ratchet,
)
from hadto_patches.recurring_routines import (
    build_routine_governance_prompt_prefix,
    classify_cron_delivery,
    sanitize_slack_cron_response,
)

# Sentinel: when a cron agent has nothing new to report, it can start its
# response with this marker to suppress delivery.  Output is still saved
# locally for audit.
SILENT_MARKER = "[SILENT]"

# Resolve Hermes home directory (respects HERMES_HOME override)
_hermes_home = get_hermes_home()
# Preserve the import-time default so tests/monkeypatches that override
# ``_hermes_home`` continue to work, while still honoring runtime env updates.
_HERMES_HOME_IMPORTED = _hermes_home

# File-based lock prevents concurrent ticks from gateway + daemon + systemd timer
_LOCK_DIR = _hermes_home / "cron"
_LOCK_FILE = _LOCK_DIR / ".tick.lock"


def _resolve_hermes_home() -> Path:
    """Return the active Hermes home for the current cron invocation.

    ``_hermes_home`` is evaluated at import time. For cron jobs that run after
    environment changes (for example, when HERMES_HOME is injected at runtime
    via a launcher), prefer the current env value. Keep using an explicitly
    patched ``_hermes_home`` when it differs from the import-time value.
    """

    env_home = os.getenv("HERMES_HOME")
    if _hermes_home != _HERMES_HOME_IMPORTED:
        return _hermes_home
    if env_home:
        return Path(env_home)
    return _hermes_home


def _build_role_prompt_prefix(job: dict) -> str:
    """Return optional role-specific execution guidance for a cron job."""
    role = str(job.get("role") or "").strip().lower()
    if role != "study":
        return ""

    return (
        "[SYSTEM: This cron job is classified as role=study. Treat it as an execution loop, not a passive summary. "
        "When a run confirms a durable gap, convert that gap into explicit follow-through before you report: update the "
        "owning backlog/control surface in the target repo, and if the gap is really Hermes's own capability "
        "(planning, verification, delegation, evidence handling, candidate selection, or similar), also create or "
        "update Hermes self-improvement work via self_improvement_pipeline or an equivalent backlog issue when those "
        "tools are available. For self-improvement reporting, carry forward the usable evidence, decision, and artifact "
        "state that prevents rediscovering the same gap in the next run. If you decide no action is warranted yet, say "
        "why in the report instead of silently continuing.]"
    )


def _build_persistence_ratchet_prompt_prefix(job: dict) -> str:
    """Return recurring-loop guidance for preserving useful state across runs."""
    if not should_check_persistence_ratchet(job):
        return ""

    try:
        from hermes_constants import display_hermes_home

        output_hint = f"{display_hermes_home()}/cron/output/{job.get('id', '<job-id>')}/"
    except Exception:
        output_hint = f"cron/output/{job.get('id', '<job-id>')}/"

    return (
        "[SYSTEM: This is a recurring control loop. Run a bounded persistence-ratchet check before reporting: "
        f"inspect at most the last {RATCHET_WINDOW_RUNS} saved outputs in {output_hint} when prior run evidence is needed, "
        "then distinguish durable carry-forward state from lucky repetition. A useful report should preserve compact, "
        "usable evidence, decisions, or artifacts from prior runs and should surface repeated rediscovery or cleanup drift. "
        "When you send a substantive final response, include a compact 'Persistence Ratchet' block with: "
        "Evidence=<preserved log/file/issue/check facts>; Decisions=<still-valid decisions carried forward>; "
        "Artifacts=<files/issues/PRs/checks updated or preserved>; Carry-forward=<one next durable state/action>; "
        "Drift=<none or repeated rediscovery/cleanup drift>. Keep this tied to operator-value, anti-make-work, and "
        "leading-indicator value. If there is genuinely nothing new and no operator action, respond exactly [SILENT].]"
    )


def _build_trust_contract_prompt_prefix(job: dict) -> str:
    """Return compact trust-contract guidance for recurring classified loops."""
    if not should_check_persistence_ratchet(job):
        return ""

    try:
        from hermes_constants import display_hermes_home

        artifact_hint = f"{display_hermes_home()}/cron/output/{job.get('id', '<job-id>')}/"
    except Exception:
        artifact_hint = f"cron/output/{job.get('id', '<job-id>')}/"

    return (
        "[SYSTEM: This loop is operating under a computational trust contract, not a one-shot prompt. "
        "Before reporting, include a compact 'Trust Contract' block with: "
        "Commitment=<the concrete promise this run was meant to keep>; "
        f"Artifact=<{artifact_hint} or another durable shared artifact surface>; "
        "Verification=<the exact check proving whether the commitment held>; "
        "Outcome=<kept|missed|blocked plus concrete evidence>; "
        "Trust Posture=<one_shot_disconnected|repeated_trust_bearing plus discovery|execution|bridge mode>; "
        "Dignity=<how this run preserved operator agency instead of forcing surrender for basic access>; "
        "Capability=<how this run compounded operator capability instead of hiding judgment behind automation>; "
        "Viability=<how this run kept the surrounding system stable and inspectable enough to rely on>; "
        "Fast Loop=<what this run may change immediately>; "
        "Slow Loop=<what requires slower governance or rule revision>; "
        "Escalate When=<the concrete checkpoint where this run must stop changing the current game and instead request governance revision>; "
        "First Proof Point=<one bounded protected seed where the governance or capability shift is supposed to work first>. "
        "Geometry Shaping=<the concrete path-shaping moves this run made instead of command-style policy language>. "
        "Also include a compact 'First Proof Point' block with: "
        "Seed Surface=<one concrete issue/job/repo path/operator workflow, not a broad rollout>; "
        "Protection Assumptions=<what keeps this seed bounded and safe while tested>; "
        "Success Signal=<the observable result that proves the seed worked>; "
        "Imitation Path=<what another site would copy only after the signal holds>; "
        "Why First=<why this seed is the right first nucleation site>. "
        "Also include a compact 'Geometry Shaping' block with: "
        "Default Changed=<which default or default path changed>; "
        "Channel Opened=<which channel, route, or evidence surface opened>; "
        "Friction Changed=<which friction was added or removed>; "
        "Stale Path Pruned=<which stale branch, route, or behavior was pruned>; "
        "Policy-vs-Path=<how this changed the path itself rather than only restating doctrine>. "
        "If the commitment was missed, name the miss plainly so broken cooperation stays visible in saved output.]"
    )


def _build_coverage_completion_prompt_prefix(job: dict) -> str:
    """Return compact coverage-completion guidance for recurring classified loops."""
    if not should_check_persistence_ratchet(job):
        return ""

    return (
        "[SYSTEM: Treat recurring-loop completion as coverage completion, not activity narration alone. "
        "Before reporting, include a compact 'Coverage Completion' block with: "
        "Evidence Coverage=<what evidence or behavior is now covered by this run>; "
        "Contradictions=<none or the exact conflict that still breaks confidence>; "
        "Uncovered Region=<the next behavior, claim, or surface that still lacks enough coverage>; "
        "Closure Basis=<why this run counts as real progress rather than a status restatement>. "
        "Prefer structural claims that survive translation across logs, issues, checks, and operator summaries. "
        "If the run only produced activity with no new coverage, say so plainly.]"
    )


def _build_value_surfaces_prompt_prefix(job: dict) -> str:
    """Return guidance for separating durable value stores from circulation-only outputs."""
    if not should_check_persistence_ratchet(job):
        return ""

    return (
        "[SYSTEM: Distinguish durable value stores from circulation signals in recurring loops. "
        "Before reporting, include a compact 'Value Surfaces' block with: "
        "Durable Store=<the durable artifact, file, issue/comment state, benchmark, or verified note that compounds across runs>; "
        "Circulation=<the cheap coordination outputs such as Slack heartbeats, transient traces, or status pings>; "
        "Closure Rule=<why circulation-only output cannot count as closure>. "
        "Do not treat closure as satisfied unless a durable artifact updated. "
        "If the run emitted only circulation signals and did not update a durable artifact, say that plainly so the loop cannot price chatter as retained value.]"
    )


def _build_attention_budget_prompt_prefix(job: dict) -> str:
    """Return guidance for pricing operator attention against decision value."""
    if not should_check_persistence_ratchet(job):
        return ""

    return (
        "[SYSTEM: Treat operator attention as a scarce budget on recurring reporting surfaces. "
        "Before reporting, include a compact 'Attention Budget' block with: "
        "Attention Cost=<how much operator attention this run consumed or asked for>; "
        "Decision Value=<what decision, judgment update, or durable state change this attention bought>; "
        "Focus Effect=<whether this shaped useful focus or drifted into low-yield alerting/report spam>; "
        "Do not count low-yield output as closure when attention cost is high and decision value is low or unchanged.]"
    )


def _build_aggregate_stewardship_prompt_prefix(job: dict) -> str:
    """Return guidance for surfacing aggregate stewardship across the wider job economy."""
    if not should_check_persistence_ratchet(job):
        return ""

    return (
        "[SYSTEM: Treat recurring and delegated loops as one agent economy, not isolated wins. "
        "Before reporting, include a compact 'Aggregate Stewardship' block with: "
        "Shared Provider Concentration=<which providers/models/base URLs or auth surfaces concentrate risk across jobs>; "
        "Dependency Choke Points=<which shared artifacts, routes, queues, repos, or operator surfaces many jobs depend on>; "
        "Verification Debt=<which jobs or claims still lack enough verification and how much of the portfolio that debt touches>; "
        "Synchronized Failure Risk=<what could fail many loops at once>; "
        "Portfolio State=<whether the portfolio is locally green but globally fragile, or why it is healthy>; "
        "Shared Artifact=<the durable shared artifact that carries this portfolio view across runs, such as hermes cron topology or inspect_job_topology output>. "
        "Do not report only loop-local success. Name the whole portfolio condition in concrete terms.]"
    )


def _build_ownership_audit_prompt_prefix(job: dict) -> str:
    """Return guidance for visible backlog selection and ownership mutation reasons."""
    if not should_check_persistence_ratchet(job):
        return ""
    role = str(job.get("role") or "").strip().lower()
    scope = str(job.get("scope") or "").strip().lower()
    if role != "coordinate" or scope != "global":
        return ""

    return (
        "[SYSTEM: When this workspace coordinator selects backlog work or attempts Linear ownership writeback, "
        "emit compact ownership audit evidence. Distinguish three surfaces: selected backlog work, live execution, "
        "and ownership mutation. Before reporting a substantive run, include an 'Ownership Decisions' block with "
        "records shaped as: Selection=<selected|denied|skipped reason=... dedupe_key=workspace-orchestrator:HAD-...>; "
        "Execution=<started|skipped reason=live_execution|planning_only|repo_unresolved>; "
        "Comment=<commented|denied|skipped reason=commented|writeback_skipped>; "
        "Delegate=<delegated|denied|skipped reason=delegate_allowed|delegate_denied|owner_label_missing|owner_label_mismatch|owner_label_conflict|human_owned|already_undelegated|writeback_skipped>; "
        "Assign=<assigned|denied|skipped reason=assign_allowed|assign_denied|owner_label_missing|owner_label_mismatch|owner_label_conflict|human_owned|writeback_skipped>. "
        "Keep reason strings terse and stable: owner_label_missing, owner_label_mismatch, owner_label_conflict, human_owned, already_undelegated, "
        "repo_unresolved, planning_only, selected, writeback_skipped, delegate_denied, delegate_allowed, assign_denied, assign_allowed. "
        "Do not create extra Linear comments only for this audit; attach it to the canonical workspace-orchestrator issue comment when a writeback is already being made.]"
    )


def _resolve_origin(job: dict) -> Optional[dict]:
    """Extract origin info from a job, preserving any extra routing metadata."""
    origin = job.get("origin")
    if not origin:
        return None
    platform = origin.get("platform")
    chat_id = origin.get("chat_id")
    if platform and chat_id:
        return origin
    return None


def _resolve_delivery_target(job: dict) -> Optional[dict]:
    """Resolve the concrete auto-delivery target for a cron job, if any."""
    deliver = job.get("deliver", "local")
    origin = _resolve_origin(job)

    if deliver == "local":
        return None

    if deliver == "origin":
        if not origin:
            return None
        return {
            "platform": origin["platform"],
            "chat_id": str(origin["chat_id"]),
            "thread_id": origin.get("thread_id"),
        }

    if ":" in deliver:
        platform_name, rest = deliver.split(":", 1)
        # Check for thread_id suffix (e.g. "telegram:-1003724596514:17")
        if ":" in rest:
            chat_id, thread_id = rest.split(":", 1)
        else:
            chat_id, thread_id = rest, None

        # Resolve human-friendly labels like "Alice (dm)" to real IDs.
        # send_message(action="list") shows labels with display suffixes
        # that aren't valid platform IDs (e.g. WhatsApp JIDs).
        try:
            from gateway.channel_directory import resolve_channel_name
            target = chat_id
            # Strip display suffix like " (dm)" or " (group)"
            if target.endswith(")") and " (" in target:
                target = target.rsplit(" (", 1)[0].strip()
            resolved = resolve_channel_name(platform_name.lower(), target)
            if resolved:
                chat_id = resolved
        except Exception:
            pass

        return {
            "platform": platform_name,
            "chat_id": chat_id,
            "thread_id": thread_id,
        }

    platform_name = deliver
    if origin and origin.get("platform") == platform_name:
        return {
            "platform": platform_name,
            "chat_id": str(origin["chat_id"]),
            "thread_id": origin.get("thread_id"),
        }

    chat_id = os.getenv(f"{platform_name.upper()}_HOME_CHANNEL", "")
    if not chat_id:
        return None

    return {
        "platform": platform_name,
        "chat_id": chat_id,
        "thread_id": None,
    }


def _deliver_result(job: dict, content: str) -> None:
    """
    Deliver job output to the configured target (origin chat, specific platform, etc.).

    Uses the standalone platform send functions from send_message_tool so delivery
    works whether or not the gateway is running.
    """
    target = _resolve_delivery_target(job)
    if not target:
        if job.get("deliver", "local") != "local":
            logger.warning(
                "Job '%s' deliver=%s but no concrete delivery target could be resolved",
                job["id"],
                job.get("deliver", "local"),
            )
        return

    platform_name = target["platform"]
    chat_id = target["chat_id"]
    thread_id = target.get("thread_id")

    from tools.send_message_tool import _send_to_platform
    from gateway.config import load_gateway_config, Platform

    platform_map = {
        "telegram": Platform.TELEGRAM,
        "discord": Platform.DISCORD,
        "slack": Platform.SLACK,
        "whatsapp": Platform.WHATSAPP,
        "signal": Platform.SIGNAL,
        "matrix": Platform.MATRIX,
        "mattermost": Platform.MATTERMOST,
        "homeassistant": Platform.HOMEASSISTANT,
        "dingtalk": Platform.DINGTALK,
        "feishu": Platform.FEISHU,
        "wecom": Platform.WECOM,
        "email": Platform.EMAIL,
        "sms": Platform.SMS,
    }
    platform = platform_map.get(platform_name.lower())
    if not platform:
        logger.warning("Job '%s': unknown platform '%s' for delivery", job["id"], platform_name)
        return

    try:
        config = load_gateway_config()
    except Exception as e:
        logger.error("Job '%s': failed to load gateway config for delivery: %s", job["id"], e)
        return

    pconfig = config.platforms.get(platform)
    if not pconfig or not pconfig.enabled:
        logger.warning("Job '%s': platform '%s' not configured/enabled", job["id"], platform_name)
        return

    # Optionally wrap the content with a header/footer so the user knows this
    # is a cron delivery.  Wrapping is on by default; set cron.wrap_response: false
    # in config.yaml for clean output.
    wrap_response = True
    try:
        user_cfg = load_config()
        wrap_response = user_cfg.get("cron", {}).get("wrap_response", True)
    except Exception:
        pass

    if wrap_response:
        task_name = job.get("name", job["id"])
        delivery_content = (
            f"Cronjob Response: {task_name}\n"
            f"-------------\n\n"
            f"{content}\n\n"
            f"Note: The agent cannot see this message, and therefore cannot respond to it."
        )
    else:
        delivery_content = content

    # Run the async send in a fresh event loop (safe from any thread)
    coro = _send_to_platform(platform, pconfig, chat_id, delivery_content, thread_id=thread_id)
    try:
        result = asyncio.run(coro)
    except RuntimeError:
        # asyncio.run() checks for a running loop before awaiting the coroutine;
        # when it raises, the original coro was never started — close it to
        # prevent "coroutine was never awaited" RuntimeWarning, then retry in a
        # fresh thread that has no running loop.
        coro.close()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, _send_to_platform(platform, pconfig, chat_id, delivery_content, thread_id=thread_id))
            result = future.result(timeout=30)
    except Exception as e:
        logger.error("Job '%s': delivery to %s:%s failed: %s", job["id"], platform_name, chat_id, e)
        return

    if result and result.get("error"):
        logger.error("Job '%s': delivery error: %s", job["id"], result["error"])
    else:
        logger.info("Job '%s': delivered to %s:%s", job["id"], platform_name, chat_id)


_DEFAULT_SCRIPT_TIMEOUT = 120  # seconds
# Backward-compatible module override used by tests and emergency monkeypatches.
_SCRIPT_TIMEOUT = _DEFAULT_SCRIPT_TIMEOUT


def _get_script_timeout() -> int:
    """Resolve cron pre-run script timeout from module/env/config with a safe default."""
    if _SCRIPT_TIMEOUT != _DEFAULT_SCRIPT_TIMEOUT:
        try:
            timeout = int(float(_SCRIPT_TIMEOUT))
            if timeout > 0:
                return timeout
        except Exception:
            logger.warning("Invalid patched _SCRIPT_TIMEOUT=%r; using env/config/default", _SCRIPT_TIMEOUT)

    env_value = os.getenv("HERMES_CRON_SCRIPT_TIMEOUT", "").strip()
    if env_value:
        try:
            timeout = int(float(env_value))
            if timeout > 0:
                return timeout
        except Exception:
            logger.warning("Invalid HERMES_CRON_SCRIPT_TIMEOUT=%r; using config/default", env_value)

    try:
        cfg = load_config() or {}
        cron_cfg = cfg.get("cron", {}) if isinstance(cfg, dict) else {}
        configured = cron_cfg.get("script_timeout_seconds")
        if configured is not None:
            timeout = int(float(configured))
            if timeout > 0:
                return timeout
    except Exception as exc:
        logger.debug("Failed to load cron script timeout from config: %s", exc)

    return _DEFAULT_SCRIPT_TIMEOUT


def _run_job_script(script_path: str) -> tuple[bool, str]:
    """Execute a cron pre-run script and capture sanitized stdout/stderr."""
    scripts_dir = get_hermes_home() / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir_resolved = scripts_dir.resolve()

    raw = Path(script_path).expanduser()
    if raw.is_absolute():
        path = raw.resolve()
    else:
        path = (scripts_dir / raw).resolve()

    try:
        path.relative_to(scripts_dir_resolved)
    except ValueError:
        return False, (
            f"Blocked: script path resolves outside the scripts directory "
            f"({scripts_dir_resolved}): {script_path!r}"
        )

    if not path.exists():
        return False, f"Script not found: {path}"
    if not path.is_file():
        return False, f"Script path is not a file: {path}"

    script_timeout = _get_script_timeout()

    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=script_timeout,
            cwd=str(path.parent),
        )
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        try:
            from agent.redact import redact_sensitive_text

            stdout = redact_sensitive_text(stdout)
            stderr = redact_sensitive_text(stderr)
        except Exception:
            pass

        if result.returncode != 0:
            parts = [f"Script exited with code {result.returncode}"]
            if stderr:
                parts.append(f"stderr:\n{stderr}")
            if stdout:
                parts.append(f"stdout:\n{stdout}")
            return False, "\n".join(parts)

        return True, stdout
    except subprocess.TimeoutExpired:
        return False, f"Script timed out after {script_timeout}s: {path}"
    except Exception as exc:
        return False, f"Script execution failed: {exc}"


def _build_job_prompt(job: dict) -> str:
    """Build the effective prompt for a cron job, optionally loading one or more skills first."""
    prompt = job.get("prompt", "")
    skills = job.get("skills")

    script_path = job.get("script")
    if script_path:
        success, script_output = _run_job_script(script_path)
        if success:
            if script_output:
                prompt = (
                    "## Script Output\n"
                    "The following data was collected by a pre-run script. "
                    "Use it as context for your analysis.\n\n"
                    f"```\n{script_output}\n```\n\n"
                    f"{prompt}"
                )
            else:
                prompt = f"[Script ran successfully but produced no output.]\n\n{prompt}"
        else:
            prompt = (
                "## Script Error\n"
                "The data-collection script failed. Report this to the user.\n\n"
                f"```\n{script_output}\n```\n\n"
                f"{prompt}"
            )

    # Always prepend [SILENT] guidance so the cron agent can suppress
    # delivery when it has nothing new or noteworthy to report.
    silent_hint = (
        "[SYSTEM: If you have a meaningful status report or findings, "
        "send them — that is the whole point of this job. Only respond "
        "with exactly \"[SILENT]\" (nothing else) when there is genuinely "
        "nothing new to report. [SILENT] suppresses delivery to the user. "
        "Never combine [SILENT] with content — either report your "
        "findings normally, or say [SILENT] and nothing more.]\n\n"
    )
    role_prefix = _build_role_prompt_prefix(job)
    ratchet_prefix = _build_persistence_ratchet_prompt_prefix(job)
    trust_prefix = _build_trust_contract_prompt_prefix(job)
    coverage_prefix = _build_coverage_completion_prompt_prefix(job)
    value_surfaces_prefix = _build_value_surfaces_prompt_prefix(job)
    attention_budget_prefix = _build_attention_budget_prompt_prefix(job)
    aggregate_stewardship_prefix = _build_aggregate_stewardship_prompt_prefix(job)
    ownership_audit_prefix = _build_ownership_audit_prompt_prefix(job)
    routine_governance_prefix = build_routine_governance_prompt_prefix(job)
    prompt = (
        silent_hint
        + (routine_governance_prefix + "\n\n" if routine_governance_prefix else "")
        + (role_prefix + "\n\n" if role_prefix else "")
        + (ratchet_prefix + "\n\n" if ratchet_prefix else "")
        + (trust_prefix + "\n\n" if trust_prefix else "")
        + (coverage_prefix + "\n\n" if coverage_prefix else "")
        + (value_surfaces_prefix + "\n\n" if value_surfaces_prefix else "")
        + (attention_budget_prefix + "\n\n" if attention_budget_prefix else "")
        + (aggregate_stewardship_prefix + "\n\n" if aggregate_stewardship_prefix else "")
        + (ownership_audit_prefix + "\n\n" if ownership_audit_prefix else "")
        + prompt
    )
    if skills is None:
        legacy = job.get("skill")
        skills = [legacy] if legacy else []

    skill_names = [str(name).strip() for name in skills if str(name).strip()]
    if not skill_names:
        return prompt

    from tools.skills_tool import skill_view

    parts = []
    skipped: list[str] = []
    for skill_name in skill_names:
        loaded = json.loads(skill_view(skill_name))
        if not loaded.get("success"):
            error = loaded.get("error") or f"Failed to load skill '{skill_name}'"
            logger.warning("Cron job '%s': skill not found, skipping — %s", job.get("name", job.get("id")), error)
            skipped.append(skill_name)
            continue

        content = str(loaded.get("content") or "").strip()
        if parts:
            parts.append("")
        parts.extend(
            [
                f'[SYSTEM: The user has invoked the "{skill_name}" skill, indicating they want you to follow its instructions. The full skill content is loaded below.]',
                "",
                content,
            ]
        )

    if skipped:
        notice = (
            f"[SYSTEM: The following skill(s) were listed for this job but could not be found "
            f"and were skipped: {', '.join(skipped)}. "
            f"Start your response with a brief notice so the user is aware, e.g.: "
            f"'⚠️ Skill(s) not found and skipped: {', '.join(skipped)}']"
        )
        parts.insert(0, notice)

    if prompt:
        parts.extend(["", f"The user has provided the following instruction alongside the skill invocation: {prompt}"])
    return "\n".join(parts)


def run_job(job: dict) -> tuple[bool, str, str, Optional[str]]:
    """
    Execute a single cron job.
    
    Returns:
        Tuple of (success, full_output_doc, final_response, error_message)
    """
    from run_agent import AIAgent
    
    # Initialize SQLite session store so cron job messages are persisted
    # and discoverable via session_search (same pattern as gateway/run.py).
    _session_db = None
    try:
        from hermes_state import SessionDB
        _session_db = SessionDB()
    except Exception as e:
        logger.debug("Job '%s': SQLite session store not available: %s", job.get("id", "?"), e)
    
    job_id = job["id"]
    job_name = job["name"]
    prompt = _build_job_prompt(job)
    origin = _resolve_origin(job)
    _cron_session_id = f"cron_{job_id}_{_hermes_now().strftime('%Y%m%d_%H%M%S')}"

    logger.info("Running job '%s' (ID: %s)", job_name, job_id)
    logger.info("Prompt: %s", prompt[:100])

    # C-09: Mark this as a cron session and as an active cron execution
    # context so cronjob_tools can block recursive job creation without being
    # tripped by unrelated inherited shell env.
    os.environ["HERMES_CRON_SESSION"] = "true"
    os.environ["HERMES_CRON_EXECUTION_CONTEXT"] = "true"

    # Inject origin context so the agent's send_message tool knows the chat
    if origin:
        os.environ["HERMES_SESSION_PLATFORM"] = origin["platform"]
        os.environ["HERMES_SESSION_CHAT_ID"] = str(origin["chat_id"])
        if origin.get("chat_name"):
            os.environ["HERMES_SESSION_CHAT_NAME"] = origin["chat_name"]

    try:
        # Re-read .env and config.yaml fresh every run so provider/key
        # changes take effect without a gateway restart.
        hermes_home = _resolve_hermes_home()
        logger.debug("Job '%s': resolved hermes_home at runtime = %s", job_id, hermes_home)

        from dotenv import load_dotenv

        def _load_dotenv_for(home_path: Path) -> None:
            try:
                load_dotenv(str(home_path / ".env"), override=True, encoding="utf-8")
            except UnicodeDecodeError:
                load_dotenv(str(home_path / ".env"), override=True, encoding="latin-1")

        _load_dotenv_for(hermes_home)
        # .env can change HERMES_HOME as well (e.g. launcher-injected path).
        resolved_home = _resolve_hermes_home()
        if resolved_home != hermes_home:
            logger.debug(
                "Job '%s': hermes_home changed after loading .env (%s -> %s)",
                job_id,
                hermes_home,
                resolved_home,
            )
            hermes_home = resolved_home
            _load_dotenv_for(hermes_home)

        delivery_target = _resolve_delivery_target(job)
        if delivery_target:
            os.environ["HERMES_CRON_AUTO_DELIVER_PLATFORM"] = delivery_target["platform"]
            os.environ["HERMES_CRON_AUTO_DELIVER_CHAT_ID"] = str(delivery_target["chat_id"])
            if delivery_target.get("thread_id") is not None:
                os.environ["HERMES_CRON_AUTO_DELIVER_THREAD_ID"] = str(delivery_target["thread_id"])

        model = job.get("model") or os.getenv("HERMES_MODEL") or ""

        # Load config.yaml for model, reasoning, prefill, toolsets, provider routing
        _cfg = {}
        try:
            import yaml
            _cfg_path = str(hermes_home / "config.yaml")
            if os.path.exists(_cfg_path):
                with open(_cfg_path) as _f:
                    _cfg = yaml.safe_load(_f) or {}
                _model_cfg = _cfg.get("model", {})
                if not job.get("model"):
                    if isinstance(_model_cfg, str):
                        model = _model_cfg
                    elif isinstance(_model_cfg, dict):
                        model = _model_cfg.get("default", model)
        except Exception as e:
            logger.warning("Job '%s': failed to load config.yaml, using defaults: %s", job_id, e)

        # Reasoning config from env or config.yaml
        from hermes_constants import parse_reasoning_effort
        effort = os.getenv("HERMES_REASONING_EFFORT", "")
        if not effort:
            effort = str(_cfg.get("agent", {}).get("reasoning_effort", "")).strip()
        reasoning_config = parse_reasoning_effort(effort)

        # Prefill messages from env or config.yaml
        prefill_messages = None
        prefill_file = os.getenv("HERMES_PREFILL_MESSAGES_FILE", "") or _cfg.get("prefill_messages_file", "")
        if prefill_file:
            import json as _json

            pfpath = Path(prefill_file).expanduser()
            if not pfpath.is_absolute():
                pfpath = hermes_home / pfpath
            if pfpath.exists():
                try:
                    with open(pfpath, "r", encoding="utf-8") as _pf:
                        prefill_messages = _json.load(_pf)
                    if not isinstance(prefill_messages, list):
                        prefill_messages = None
                except Exception as e:
                    logger.warning("Job '%s': failed to parse prefill messages file '%s': %s", job_id, pfpath, e)
                    prefill_messages = None

        # Max iterations
        max_iterations = _cfg.get("agent", {}).get("max_turns") or _cfg.get("max_turns") or 90

        # Provider routing
        pr = _cfg.get("provider_routing", {})
        smart_routing = _cfg.get("smart_model_routing", {}) or {}
        fallback_model = _cfg.get("fallback_providers") or _cfg.get("fallback_model")

        if fallback_model is None:
            fallback_count = 0
        elif isinstance(fallback_model, list):
            fallback_count = len(fallback_model)
        else:
            fallback_count = 1
        logger.debug(
            "Job '%s': fallback_model resolved from config.yaml (type=%s, count=%s)",
            job_id,
            type(fallback_model).__name__,
            fallback_count,
        )

        from hermes_cli.runtime_provider import (
            resolve_runtime_provider,
            format_runtime_provider_error,
        )
        try:
            runtime_kwargs = {
                "requested": job.get("provider") or os.getenv("HERMES_INFERENCE_PROVIDER"),
            }
            if job.get("base_url"):
                runtime_kwargs["explicit_base_url"] = job.get("base_url")
            runtime = resolve_runtime_provider(**runtime_kwargs)
        except Exception as exc:
            message = format_runtime_provider_error(exc)
            raise RuntimeError(message) from exc

        if not model:
            try:
                from hermes_cli.models import _PROVIDER_MODELS
                provider_default_models = _PROVIDER_MODELS.get(runtime.get("provider") or "", [])
                if provider_default_models:
                    model = provider_default_models[0]
            except Exception:
                pass

        from agent.smart_model_routing import resolve_turn_route
        turn_route = resolve_turn_route(
            prompt,
            smart_routing,
            {
                "model": model,
                "api_key": runtime.get("api_key"),
                "base_url": runtime.get("base_url"),
                "provider": runtime.get("provider"),
                "api_mode": runtime.get("api_mode"),
                "command": runtime.get("command"),
                "args": list(runtime.get("args") or []),
            },
        )

        agent = AIAgent(
            model=turn_route["model"],
            api_key=turn_route["runtime"].get("api_key"),
            base_url=turn_route["runtime"].get("base_url"),
            provider=turn_route["runtime"].get("provider"),
            api_mode=turn_route["runtime"].get("api_mode"),
            acp_command=turn_route["runtime"].get("command"),
            acp_args=turn_route["runtime"].get("args"),
            max_iterations=max_iterations,
            reasoning_config=reasoning_config,
            prefill_messages=prefill_messages,
            providers_allowed=pr.get("only"),
            providers_ignored=pr.get("ignore"),
            providers_order=pr.get("order"),
            provider_sort=pr.get("sort"),
            disabled_toolsets=["cronjob", "messaging", "clarify"],
            quiet_mode=True,
            platform="cron",
            session_id=_cron_session_id,
            session_db=_session_db,
            fallback_model=fallback_model,
        )
        
        result = agent.run_conversation(prompt)
        
        final_response = result.get("final_response", "") or ""
        # Use a separate variable for log display; keep final_response clean
        # for delivery logic (empty response = no delivery).
        logged_response = final_response if final_response else "(No response generated)"
        
        output = f"""# Cron Job: {job_name}

**Job ID:** {job_id}
**Run Time:** {_hermes_now().strftime('%Y-%m-%d %H:%M:%S')}
**Schedule:** {job.get('schedule_display', 'N/A')}

## Prompt

{prompt}

## Response

{logged_response}
"""
        
        logger.info("Job '%s' completed successfully", job_name)
        return True, output, final_response, None
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error("Job '%s' failed: %s", job_name, error_msg)
        
        output = f"""# Cron Job: {job_name} (FAILED)

**Job ID:** {job_id}
**Run Time:** {_hermes_now().strftime('%Y-%m-%d %H:%M:%S')}
**Schedule:** {job.get('schedule_display', 'N/A')}

## Prompt

{prompt}

## Error

```
{error_msg}

{traceback.format_exc()}
```
"""
        return False, output, "", error_msg

    finally:
        # Clean up injected env vars so they don't leak to other jobs
        for key in (
            "HERMES_CRON_SESSION",
            "HERMES_CRON_EXECUTION_CONTEXT",
            "HERMES_SESSION_PLATFORM",
            "HERMES_SESSION_CHAT_ID",
            "HERMES_SESSION_CHAT_NAME",
            "HERMES_CRON_AUTO_DELIVER_PLATFORM",
            "HERMES_CRON_AUTO_DELIVER_CHAT_ID",
            "HERMES_CRON_AUTO_DELIVER_THREAD_ID",
        ):
            os.environ.pop(key, None)
        if _session_db:
            try:
                _session_db.end_session(_cron_session_id, "cron_complete")
            except (Exception, KeyboardInterrupt) as e:
                logger.debug("Job '%s': failed to end session: %s", job_id, e)
        try:
            from hermes_cli.ctx_runtime import retire_ctx_binding

            retire_ctx_binding(
                _cron_session_id,
                reason="ctx binding retired: cron job finished",
                preserve_codex_handoff=True,
            )
        except (Exception, KeyboardInterrupt) as e:
            logger.debug("Job '%s': failed to retire ctx binding: %s", job_id, e)
        if _session_db:
            try:
                _session_db.close()
            except (Exception, KeyboardInterrupt) as e:
                logger.debug("Job '%s': failed to close SQLite session store: %s", job_id, e)


def tick(verbose: bool = True) -> int:
    """
    Check and run all due jobs.
    
    Uses a file lock so only one tick runs at a time, even if the gateway's
    in-process ticker and a standalone daemon or manual tick overlap.
    
    Args:
        verbose: Whether to print status messages
    
    Returns:
        Number of jobs executed (0 if another tick is already running)
    """
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)

    # Cross-platform file locking: fcntl on Unix, msvcrt on Windows
    lock_fd = None
    try:
        lock_fd = open(_LOCK_FILE, "w")
        if fcntl:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        elif msvcrt:
            msvcrt.locking(lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
    except (OSError, IOError):
        logger.debug("Tick skipped — another instance holds the lock")
        if lock_fd is not None:
            lock_fd.close()
        return 0

    try:
        due_jobs = get_due_jobs()

        if verbose and not due_jobs:
            logger.info("%s - No jobs due", _hermes_now().strftime('%H:%M:%S'))
            return 0

        if verbose:
            logger.info("%s - %s job(s) due", _hermes_now().strftime('%H:%M:%S'), len(due_jobs))

        executed = 0
        for job in due_jobs:
            try:
                # For recurring jobs (cron/interval), advance next_run_at to the
                # next future occurrence BEFORE execution.  This way, if the
                # process crashes mid-run, the job won't re-fire on restart.
                # One-shot jobs are left alone so they can retry on restart.
                previous_responses = load_recent_job_responses(job["id"], limit=RATCHET_WINDOW_RUNS)
                advance_next_run(job["id"])

                success, output, final_response, error = run_job(job)

                # Decide delivery before saving the new output so no-change
                # gates compare against prior runs only. Output is still saved
                # locally below, even when delivery is suppressed.
                deliver_content = final_response if success else f"⚠️ Cron job '{job.get('name', job['id'])}' failed:\n{error}"
                should_deliver = bool(deliver_content)
                if should_deliver and success:
                    delivery_decision = classify_cron_delivery(
                        job,
                        deliver_content,
                        previous_responses=previous_responses,
                        now=_hermes_now(),
                    )
                    if delivery_decision.get("suppress"):
                        logger.info(
                            "Job '%s': %s — skipping delivery",
                            job["id"],
                            delivery_decision.get("message", "delivery suppressed"),
                        )
                        should_deliver = False

                output_file = save_job_output(job["id"], output)
                if verbose:
                    logger.info("Output saved to: %s", output_file)

                if should_deliver:
                    delivery_target = _resolve_delivery_target(job)
                    if success and delivery_target and str(delivery_target.get("platform", "")).lower() == "slack":
                        slack_decision = sanitize_slack_cron_response(job, deliver_content)
                        if slack_decision.get("suppress"):
                            logger.info(
                                "Job '%s': Slack cron report %s — skipping delivery",
                                job["id"],
                                slack_decision.get("reason", "suppressed"),
                            )
                            should_deliver = False
                        else:
                            deliver_content = str(slack_decision.get("content") or "")

                if should_deliver:
                    try:
                        _deliver_result(job, deliver_content)
                    except Exception as de:
                        logger.error("Delivery failed for job %s: %s", job["id"], de)

                mark_job_run(job["id"], success, error)
                executed += 1

            except Exception as e:
                logger.error("Error processing job %s: %s", job['id'], e)
                mark_job_run(job["id"], False, str(e))

        return executed
    finally:
        if fcntl:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        elif msvcrt:
            try:
                msvcrt.locking(lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
            except (OSError, IOError):
                pass
        lock_fd.close()


if __name__ == "__main__":
    tick(verbose=True)
