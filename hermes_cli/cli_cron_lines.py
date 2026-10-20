"""The classic CLI's `/cron` as plain lines: the overview, the list and every store write.

Shared by the classic CLI off the gateway (`CLICommandsMixin._handle_cron_command`) and the shared gateway, which runs
`/cron` for a local session's `slash.exec` (gateway/session_commands.py, gateway/session_cron_commands.py), so the
profile's cron store has one writer: the gateway that owns it.
"""
from __future__ import annotations

import shlex

from hermes_cli.cli_commands_session_tools import _lines, _t


def _cron(**kwargs) -> dict:
    # Looked up on the mixin module at call time so tests that patch `cli_commands_mixin._cron_api` still apply.
    from hermes_cli import cli_commands_mixin
    return cli_commands_mixin._cron_api(**kwargs)


def cron_overview_lines() -> list[str]:
    """Bare ``/cron``: the command card, then the current jobs. Plain text (the shared gateway serves
    the same lines to every local session, gateway/session_cron_commands.py)."""
    out = ["", "+" + "-" * 68 + "+", "|" + _t("cron.overview_title").center(68) + "|",
           "+" + "-" * 68 + "+", "", f"  {_t('cron.overview_commands')}",
           *_lines(_t("cron.overview_usage"), pad="    "), ""]
    result = _cron(action="list")
    jobs = result.get("jobs", []) if result.get("success") else []
    if not jobs:
        return [*out, f"  {_t('cron.no_jobs_hint')}", ""]
    from hermes_cli.cron import _next_run_row
    out += [f"  {_t('cron.current_jobs')}", "  " + "-" * 63]
    for job in jobs:
        out.append(f"    {job['job_id'][:12]:<12} | {job['schedule']:<15} | {job.get('repeat', '?'):<8}")
        if job.get("skills"):
            out.append(f"      {_t('cron.skills', skills=', '.join(job['skills']))}")
        out.append(f"      {job.get('prompt_preview', '')}")
        if job.get("next_run_at"):
            # A stamp parked past the scheduler grace must not read as upcoming (#114309).
            label, value = _next_run_row(job)
            out.append(f"      {_t('cron.next_short') if label == 'Next run' else label}: {value}")
        out.append("")
    return [*out, ""]


def cron_list_lines(*, include_disabled: bool = False) -> list[str]:
    """``/cron list [--all]``."""
    result = _cron(action="list", include_disabled=include_disabled)
    jobs = result.get("jobs", []) if result.get("success") else []
    if not jobs:
        return [_t("cron.no_jobs")]
    from hermes_cli.cron import _next_run_row
    out = ["", _t("cron.list_header"), "-" * 80]
    for job in jobs:
        out += [f"  {_t('cron.id', job_id=job['job_id'])}", f"  {_t('cron.name', name=job['name'])}",
                f"  {_t('cron.state', state=job.get('state', '?'))}",
                f"  {_t('cron.schedule_repeat', schedule=job['schedule'], repeat=job.get('repeat', '?'))}",
                "  {}: {}".format(*_next_run_row(job)) if job.get("next_run_at") else f"  {_t('cron.next_run_na')}"]
        if job.get("skills"):
            out.append(f"  {_t('cron.skills', skills=', '.join(job['skills']))}")
        out.append(f"  {_t('cron.prompt', prompt=job.get('prompt_preview', ''))}")
        if job.get("last_run_at"):
            status = job.get("last_status") or "?"
            # delivery_failed: the run succeeded but delivery didn't — the reason lives
            # in last_delivery_error (last_error is None).
            if status == "delivery_failed" and job.get("last_delivery_error"):
                status = _t("cron.status_delivery_failed", error=job["last_delivery_error"])
            elif status == "error" and job.get("last_error"):
                status = _t("cron.status_error", error=job["last_error"])
            out.append(f"  {_t('cron.last_run', when=job['last_run_at'], status=status)}")
        out.append("")
    return out


# /cron flag tables: flag -> opts key. Order-sensitive in _parse_cron_flags: bool flags never
# consume a value; --repeat is int-validated separately.
_CRON_BOOL_FLAGS = {"--clear-skills": "clear_skills", "--all": "all"}
_CRON_LIST_FLAGS = {"--skill": "skills", "--add-skill": "add_skills", "--remove-skill": "remove_skills"}
_CRON_VALUE_FLAGS = {"--name": "name", "--deliver": "deliver", "--prompt": "prompt", "--schedule": "schedule"}


def _normalize_skills(values) -> list:
    """Strip, drop empties, and dedupe (order-preserving)."""
    normalized = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _parse_cron_flags(tokens):
    """Parse /cron flags into an opts dict (None for a bad --repeat)."""
    opts = {
        "name": None, "deliver": None, "repeat": None, "prompt": None, "schedule": None,
        "skills": [], "add_skills": [], "remove_skills": [],
        "clear_skills": False, "all": False, "positionals": []}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        has_value = i + 1 < len(tokens)
        if token in _CRON_BOOL_FLAGS:
            opts[_CRON_BOOL_FLAGS[token]] = True
            i += 1
        elif token in _CRON_LIST_FLAGS and has_value:
            opts[_CRON_LIST_FLAGS[token]].append(tokens[i + 1])
            i += 2
        elif token == "--repeat" and has_value:
            try:
                opts["repeat"] = int(tokens[i + 1])
            except ValueError:
                return None
            i += 2
        elif token in _CRON_VALUE_FLAGS and has_value:
            opts[_CRON_VALUE_FLAGS[token]] = tokens[i + 1]
            i += 2
        else:
            opts["positionals"].append(token)
            i += 1
    return opts


def _add_lines(subcommand: str, opts: dict) -> list[str]:
    positionals = opts["positionals"]
    if not positionals:
        return [_t("cron.usage_add")]
    schedule = opts["schedule"] or positionals[0]
    prompt = opts["prompt"] or " ".join(positionals[1:])
    skills = _normalize_skills(opts["skills"])
    if not prompt and not skills:
        return [_t("cron.need_prompt_or_skill")]
    result = _cron(
        action="create", schedule=schedule, prompt=prompt or None, name=opts["name"],
        deliver=opts["deliver"], repeat=opts["repeat"], skills=skills or None)
    if not result.get("success"):
        return [_t("cron.create_failed", error=result.get("error"))]
    out = [_t("cron.created", job_id=result["job_id"]), f"  {_t('cron.schedule', schedule=result['schedule'])}"]
    if result.get("skills"):
        out.append(f"  {_t('cron.skills', skills=', '.join(result['skills']))}")
    return [*out, f"  {_t('cron.next_run', value=result['next_run_at'])}"]


def _edit_lines(subcommand: str, opts: dict) -> list[str]:
    from cron import get_job
    positionals = opts["positionals"]
    if not positionals:
        return [_t("cron.usage_edit")]
    job_id = positionals[0]
    existing = get_job(job_id)
    if not existing:
        return [_t("cron.job_not_found", job_id=job_id)]
    # Skill edit precedence: --clear-skills > --skill (replace) > --add/--remove (merge) > untouched.
    final_skills = None
    replacement_skills = _normalize_skills(opts["skills"])
    add_skills = _normalize_skills(opts["add_skills"])
    remove_skills = set(_normalize_skills(opts["remove_skills"]))
    if opts["clear_skills"]:
        final_skills = []
    elif replacement_skills:
        final_skills = replacement_skills
    elif add_skills or remove_skills:
        existing_skills = list(existing.get("skills") or ([existing["skill"]] if existing.get("skill") else []))
        final_skills = [skill for skill in existing_skills if skill not in remove_skills]
        final_skills += [skill for skill in add_skills if skill not in final_skills]
    result = _cron(
        action="update", job_id=job_id, schedule=opts["schedule"], prompt=opts["prompt"],
        name=opts["name"], deliver=opts["deliver"], repeat=opts["repeat"], skills=final_skills)
    if not result.get("success"):
        return [_t("cron.update_failed", error=result.get("error"))]
    job = result["job"]
    return [_t("cron.updated", job_id=job["job_id"]), f"  {_t('cron.schedule', schedule=job['schedule'])}",
            f"  {_t('cron.skills', skills=', '.join(job['skills']))}" if job.get("skills")
            else f"  {_t('cron.skills_none')}"]


def _job_action_lines(subcommand: str, opts: dict) -> list[str]:
    """pause / resume / run / remove (aliases rm, delete) on one job id."""
    positionals = opts["positionals"]
    if not positionals:
        return [_t("cron.usage_action", subcommand=subcommand)]
    job_id = positionals[0]
    action = "remove" if subcommand in {"remove", "rm", "delete"} else subcommand
    result = _cron(action=action, job_id=job_id, reason="paused from /cron" if action == "pause" else None)
    if not result.get("success"):
        return [_t(f"cron.{action}_failed", error=result.get("error"))]
    if action == "remove":
        removed = result.get("removed_job", {})
        return [_t("cron.removed", name=removed.get("name", job_id), job_id=job_id)]
    job = result["job"]
    if action == "run" and job.get("execution_skipped"):
        # A refused run-now (claim lost, paused, gone) must not read as accepted.
        return [_t("cron.did_not_run", name=job["name"], job_id=job_id, reason=job["execution_skipped"])]
    # One full template per action verb (cron.paused / cron.resumed / cron.triggered).
    out = [_t({"pause": "cron.paused", "resume": "cron.resumed", "run": "cron.triggered"}[action],
              name=job["name"], job_id=job_id)]
    if action == "resume":
        out.append(f"  {_t('cron.next_run', value=job.get('next_run_at'))}")
    elif action == "run":
        from hermes_cli.cron import _run_outcome
        out.append(f"  {_run_outcome(job)}")
    return out


# /cron subcommand -> line builder.
_CRON_SUBCOMMANDS = {
    # ``-a``: the list-all alias main's TUI read path accepted (only here, so no other form loses a ``-a`` word).
    "list": lambda _sub, opts: cron_list_lines(include_disabled=opts["all"] or "-a" in opts["positionals"]),
    "add": _add_lines, "create": _add_lines, "edit": _edit_lines,
    **{k: _job_action_lines for k in ("pause", "resume", "run", "remove", "rm", "delete")}}


def cron_command_lines(arg: str) -> list[str]:
    """``/cron [subcommand ...]`` (the text after ``/cron``); raises ValueError on an unclosed quote."""
    tokens = shlex.split(arg)
    if not tokens:
        return cron_overview_lines()
    subcommand, opts = tokens[0].lower(), _parse_cron_flags(tokens[1:])
    if opts is None:
        return [_t("cron.repeat_must_be_integer")]
    handler = _CRON_SUBCOMMANDS.get(subcommand)
    if handler is None:
        return [_t("cron.unknown_command", subcommand=subcommand), f"  {_t('cron.available')}"]
    return handler(subcommand, opts)
