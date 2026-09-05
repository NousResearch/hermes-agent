"""Cron job prompt assembly: context_from injection, skill loading, the assembled-prompt
injection scan, and the credential-exfil / config-block guards.

Split out of ``cron.scheduler``. Import names from this module directly (``cron.scheduler`` only
imports the few it calls itself). Origin-resident helpers and sibling split modules are reached
late-bound (``_sched`` / module refs at the bottom) so monkeypatching the defining module works.
"""

from __future__ import annotations

import json
import logging
from hermes_time import now as _hermes_now
from typing import Optional

# Log-record parity with the origin module.
logger = logging.getLogger("cron.scheduler")


def _parse_wake_gate(script_output: str) -> bool:
    """Wake gate: False only if the last non-empty stdout line is JSON ``{"wakeAgent": false}``
    (agent skipped entirely — no LLM run, no delivery); anything else wakes normally.

    Any other output (non-JSON, missing flag, gate absent, or ``wakeAgent: true``) means wake the agent
    normally. See #1232.
    """
    stripped_lines = [line for line in (script_output or "").splitlines() if line.strip()]
    if not stripped_lines:
        return True
    try:
        gate = json.loads(stripped_lines[-1].strip())
    except (json.JSONDecodeError, ValueError):
        return True
    return not isinstance(gate, dict) or gate.get("wakeAgent", True) is not False


def _prepend_context_block(prompt: str, heading: str, intro: str, body: str) -> str:
    """Prefix ``prompt`` with a fenced ``## heading`` data block."""
    return f"## {heading}\n{intro}\n\n```\n{body}\n```\n\n{prompt}"


def _job_skill_names(job: dict) -> list[str]:
    """Normalized skill names from ``skills`` (list or str) or the legacy singular ``skill``."""
    skills = job.get("skills")
    if skills is None:
        legacy = job.get("skill")
        skills = [legacy] if legacy else []
    elif isinstance(skills, str):
        skills = [skills]
    return [str(name).strip() for name in skills if str(name).strip()]


_MAX_CONTEXT_CHARS = 8000

_SELF_CONTEXT_INTRO = (
    "The following is this job's most recent output from its previous run. Use it for "
    "continuity: avoid repeating what was already reported, and continue where the last run "
    "left off."
)
_UPSTREAM_CONTEXT_INTRO = (
    "The following is the most recent output from a preceding cron job. Use it as context for "
    "your analysis."
)


def _inject_context_from(job: dict, prompt: str) -> tuple[str, bool]:
    """Prepend the latest output of each ``context_from`` job; returns ``(prompt, injected)``."""
    context_from = job.get("context_from")
    if not context_from:
        return prompt, False
    from cron.jobs import get_cron_output_dir
    output_dir = get_cron_output_dir()
    if isinstance(context_from, str):
        context_from = [context_from]
    injected = False
    for source_job_id in context_from:
        # "self" = the job's own id: continuity across runs without touching session history.
        if isinstance(source_job_id, str) and source_job_id.strip().lower() == "self":
            source_job_id = str(job.get("id") or "")
        is_self = source_job_id == job.get("id")
        # Traversal guard — valid job IDs are hex strings.
        if not source_job_id or not all(c in "0123456789abcdef" for c in source_job_id):
            logger.warning(
                "context_from: skipping invalid job_id %r for job_id=%r name=%r%s",
                source_job_id, job.get("id"), job.get("name"),
                _delivery._cron_job_origin_log_suffix(job),
            )
            continue
        try:
            output_files = sorted(
                (output_dir / source_job_id).glob("*.md"), key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            if not output_files:
                continue  # silent skip — no output yet
            latest_output = output_files[0].read_text(encoding="utf-8").strip()
            if len(latest_output) > _MAX_CONTEXT_CHARS:
                latest_output = (
                    latest_output[:_MAX_CONTEXT_CHARS] + "\n\n[... output truncated ...]")
            if not latest_output:
                continue  # silent skip — empty output
            if is_self:
                prompt = _prepend_context_block(
                    prompt, "Your previous run's output", _SELF_CONTEXT_INTRO, latest_output)
            else:
                prompt = _prepend_context_block(
                    prompt, f"Output from job '{source_job_id}'", _UPSTREAM_CONTEXT_INTRO,
                    latest_output,
                )
            injected = True
        except (OSError, PermissionError) as e:
            # silent skip — never put error text into the prompt
            logger.warning("context_from: failed to read output for job %r: %s", source_job_id, e)
    return prompt, injected


def _load_cron_skill_parts(job: dict, skill_names: list[str]) -> list[str]:
    """Load each named skill/bundle into prompt parts; unknown ones are skipped with a notice."""
    from tools.skills_tool import skill_view
    from tools.skill_usage import bump_use
    from agent.skill_bundles import build_bundle_invocation_message, resolve_bundle_command_key
    from agent.skill_utils import normalize_skill_lookup_name
    job_label = job.get("name", job.get("id"))
    task_id = str(job.get("id") or "") or None
    parts: list[str] = []
    skipped: list[str] = []

    def _skip(msg: str, *args) -> None:
        logger.warning("Cron job '%s': " + msg, job_label, *args)
        skipped.append(skill_name)

    for skill_name in skill_names:
        # Bundles shadow same-slug skills, mirroring the CLI/gateway slash-command path.
        bundle_key = resolve_bundle_command_key(skill_name.lstrip("/"))
        if bundle_key:
            bundle_payload = build_bundle_invocation_message(
                bundle_key, user_instruction="", task_id=task_id)
            if bundle_payload:
                if parts:
                    parts.append("")
                parts.append(bundle_payload[0])
            else:
                _skip("bundle '%s' could not load any skills, skipping", skill_name)
            continue

        try:
            loaded = json.loads(skill_view(normalize_skill_lookup_name(skill_name)))
        except (json.JSONDecodeError, TypeError):
            _skip("skill '%s' returned invalid JSON, skipping", skill_name)
            continue
        if not loaded.get("success"):
            _skip(
                "skill not found, skipping — %s",
                loaded.get("error") or f"Failed to load skill '{skill_name}'")
            continue

        try:
            bump_use(skill_name, task_id=task_id)
        except Exception:
            logger.debug("Cron job: failed to bump skill usage for '%s'", skill_name, exc_info=True)

        if parts:
            parts.append("")
        parts.extend([
            f'[IMPORTANT: The user has invoked the "{skill_name}" skill, indicating they want you to follow its instructions. The full skill content is loaded below.]',
            "",
            str(loaded.get("content") or "").strip()])

    if skipped:
        parts.insert(0, (
            f"[IMPORTANT: The following skill(s) were listed for this job but could not be found "
            f"and were skipped: {', '.join(skipped)}. "
            f"Start your response with a brief notice so the user is aware, e.g.: "
            f"'⚠️ Skill(s) not found and skipped: {', '.join(skipped)}']"
        ))
    return parts


_CRON_HINT = (
    "[IMPORTANT: You are running as a scheduled cron job. "
    "DELIVERY: Your final response will be automatically delivered "
    "to the user — do NOT use send_message or try to deliver "
    "the output yourself. Just produce your report/output as your "
    "final response and the system handles the rest. "
    "SILENT: If there is genuinely nothing new to report, respond "
    "with exactly \"[SILENT]\" (nothing else) to suppress delivery. "
    "Never combine [SILENT] with content — either report your "
    "findings normally, or say [SILENT] and nothing more.]\n\n"
)


def _build_job_prompt(
    job: dict, prerun_script: Optional[tuple] = None, extra_prompt: Optional[str] = None) -> str:
    """Build the effective prompt for a cron job, optionally loading skills first.
    ``prerun_script``: cached ``(success, stdout)`` from a script the caller already ran (wake-gate
    check) — skips re-execution. ``extra_prompt``: per-run ``## Run Context`` for this fire only,
    never persisted to the job.

    When provided, the script is not re-executed and the cached result is used for prompt injection. When
    omitted, the script (if any) runs inline as before. extra_prompt: Optional per-run context (from
    ``cronjob(action='run')``, 57331 — salvaged from #57342 by @liuhao1024).
    """
    user_prompt = str(job.get("prompt") or "")
    if extra_prompt:
        user_prompt = f"{user_prompt}\n\n## Run Context\n{extra_prompt}"
    prompt = user_prompt
    # Runtime DATA (script stdout, upstream output) legitimately quotes command-shape strings, so it
    # must not be scanned with the strict user-prompt set — see _scan_assembled_cron_prompt.
    has_injected_data = False

    script_path = job.get("script")
    if script_path:
        success, script_output = (
            prerun_script if prerun_script is not None else _script._run_job_script(script_path))
        if success and not script_output:
            return None  # no output → nothing to report, skip the AI call
        heading, intro = (
            ("Script Output", "The following data was collected by a pre-run script. "
                              "Use it as context for your analysis.")
            if success
            else ("Script Error", "The data-collection script failed. Report this to the user.")
        )
        prompt = _prepend_context_block(prompt, heading, intro, script_output)
        has_injected_data = True

    prompt, _ctx_injected = _inject_context_from(job, prompt)
    has_injected_data = has_injected_data or _ctx_injected

    # Durable per-job notepad; empty renders as "" so unused → byte-identical prompt.
    from cron import notepad as cron_notepad
    notepad_section = cron_notepad.render_notepad_section(str(job.get("id") or ""))
    if notepad_section:
        prompt = f"{notepad_section}{prompt}"
        has_injected_data = True

    prompt = _CRON_HINT + prompt
    skill_names = _job_skill_names(job)
    if not skill_names:
        return _scan_assembled_cron_prompt(
            prompt, job, has_skills=False, has_injected_data=has_injected_data,
            user_prompt=user_prompt,
        )

    parts = _load_cron_skill_parts(job, skill_names)
    stable_prefix = None
    if prompt:
        from agent.skill_commands import append_user_instruction
        parts.append("")
        # Skill blocks are stable per job config; the appended instruction is volatile per-run.
        # Declare that boundary for the Anthropic cache planner.
        # The skill blocks (and any skipped-skill notice) above are stable per job config; the appended
        # instruction carries the volatile per-run data (cron hint + prompt + script output + run context).
        # See #81867.
        stable_prefix = append_user_instruction(parts, prompt)
    assembled = _scan_assembled_cron_prompt("\n".join(parts), job, has_skills=True)
    if (
        stable_prefix
        and len(assembled) > len(stable_prefix)
        and assembled.startswith(stable_prefix)
    ):
        # Guarded: the scanner may mutate the bytes; mismatch → whole-message caching.
        from agent.prompt_cache_boundary import register_stable_prefix
        register_stable_prefix(stable_prefix)
    return assembled


def _scan_assembled_cron_prompt(
    assembled: str, job: dict, *, has_skills: bool = False, has_injected_data: bool = False,
    user_prompt: Optional[str] = None,
) -> str:
    """Scan the assembled cron prompt for injection; raise ``CronPromptInjectionBlocked`` on a hit.
    Needed because skill content is loaded from disk at runtime (never scanned at create/update)
    and cron auto-approves tool calls. Tier by what the prompt CONTAINS: user prompt + hint only →
    STRICT ``_scan_cron_prompt``; skills or injected data → LOOSER ``_scan_cron_skill_assembled``
    (command-shape patterns dropped, invisible unicode sanitized not blocked, so a false positive
    cannot permanently kill a job); injected data without skills also scans ``user_prompt`` STRICT.

    Since cron runs non-interactively (auto-approves tool calls), a malicious skill carrying an injection
    payload bypassed every gate. See #3968.
    """
    from tools.cronjob_tools import _scan_cron_prompt
    from tools.cronjob_prompt_scan import _scan_cron_skill_assembled
    if has_skills or has_injected_data:
        # The cleaned (sanitized) prompt is what actually runs.
        assembled, scan_error = _scan_cron_skill_assembled(assembled)
        if not scan_error and not has_skills and user_prompt:
            scan_error = _scan_cron_prompt(user_prompt)
    else:
        scan_error = _scan_cron_prompt(assembled)
    if scan_error:
        logger.warning(
            "Cron job '%s': assembled prompt blocked by injection scanner — %s",
            job.get("name") or job.get("id") or "<unknown>", scan_error)
        raise _sched.CronPromptInjectionBlocked(scan_error)
    return assembled


def _guard_job_credential_exfil(job: dict) -> None:
    """Fail closed (RuntimeError) if the stored provider/base_url pair could exfiltrate a key.
    Runtime backstop: jobs persisted before the create/update guard, or written directly to the
    store, reach provider resolution unchecked. Fallback providers come from operator config and
    are validated by the caller, not here."""
    try:
        from tools.cronjob_tools import _validate_cron_base_url
        err = _validate_cron_base_url(job.get("provider"), job.get("base_url"))
    except Exception as exc:
        # Fail CLOSED on validator/import errors — but only for jobs WITH a base_url override;
        # a job without one cannot exfiltrate via this path, so it still runs.
        err = (
            f"could not validate provider/base_url pair "
            f"({exc.__class__.__name__}: {exc}); refusing to run a job with "
            "an unverified base_url override"
        ) if job.get("base_url") else None
    if err:
        job_id = job.get("id")
        logger.error(
            "Job '%s': refusing to run — unsafe provider/base_url pair could "
            "exfiltrate a stored credential: %s",
            job_id, err)
        raise RuntimeError(f"Cron job '{job_id}' blocked for safety: {err}")


def _block_and_pause_job(
    job_id: str, job_name: str, reason: str) -> tuple[bool, str, str, Optional[str]]:
    """Fail a run closed and pause the job: an unrunnable job left enabled re-fires every tick
    forever; ``paused_at``/``paused_reason`` give an auditable record."""
    from cron.jobs import pause_job
    logger.error("Job '%s': %s", job_id, reason)
    try:
        pause_job(job_id, f"Auto-paused by scheduler: {reason}")
    except Exception:
        logger.exception("Job '%s': failed to auto-pause unrunnable job", job_id)

    now_iso = _hermes_now().strftime("%Y-%m-%d %H:%M:%S")
    doc = (
        f"# Cron Job: {job_name}\n\n"
        f"**Job ID:** {job_id}\n"
        f"**Run Time:** {now_iso}\n"
        f"**Status:** blocked (unrunnable job) — auto-paused\n\n"
        f"{reason}\n"
    )
    alert = f"⚠ Cron job '{job_name}' was auto-paused\n\n{reason}"
    return False, doc, alert, reason


# Late-bound origin namespace (see module docstring). Imported LAST so this module is fully
# populated before ``scheduler`` re-exports from it.
from cron import scheduler as _sched  # noqa: E402
from cron import scheduler_delivery as _delivery  # noqa: E402
from cron import scheduler_script as _script  # noqa: E402
