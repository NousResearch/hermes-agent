"""Handle /loop slash command — persistent scheduled agent loops.

Creates cron jobs with loop=True that self-evaluate via judge_goal and
auto-pause after consecutive no-progress detections.

Two modes:
  Timed:   /loop <interval> <prompt>  — fixed schedule (e.g., 30m, 2h)
  Dynamic: /loop <prompt>             — starts at 5m, adapts based on output
"""

import json
import re
from typing import Optional


# Interval token: matches "5m", "2h", "30m", "1d" (the cron engine's floor is
# 1 minute, so seconds are intentionally excluded — see _SUBMINUTE_RE).
_INTERVAL_RE = re.compile(r"^\d+[mhd]$", re.IGNORECASE)
# Sub-minute tokens ("30s") — detected only to reject with a clear error,
# since the scheduler has no sub-minute granularity.
_SUBMINUTE_RE = re.compile(r"^\d+s$", re.IGNORECASE)


def handle_loop_command(
    text: str,
    origin: Optional[dict] = None,
) -> str:
    """Handle /loop slash command.

    Args:
        text: The raw command text after '/loop'.
        origin: Optional origin dict with platform/chat_id/thread_id
            for delivery routing. Passed through to create_job.

    Returns:
        JSON string for tool integration, or human-readable status.
    """
    args = (text or "").strip()

    if not args:
        return _usage()

    lower = args.lower()

    if lower in ("status", "list"):
        return _handle_status()

    if lower == "help":
        return _usage()

    # pause <id>
    m = re.match(r"^pause\s+(\S+)", args, re.IGNORECASE)
    if m:
        return _handle_pause_resume(m.group(1), "pause")
    if lower == "pause":
        return json.dumps({"success": False, "error": "Usage: /loop pause <job_id>\n       (partial IDs work — you can use the first few characters)"})

    # resume <id>
    m = re.match(r"^resume\s+(\S+)", args, re.IGNORECASE)
    if m:
        return _handle_pause_resume(m.group(1), "resume")
    if lower == "resume":
        return json.dumps({"success": False, "error": "Usage: /loop resume <job_id>\n       (partial IDs work — you can use the first few characters)"})

    # stop/remove <id>
    m = re.match(r"^(?:stop|remove)\s+(\S+)", args, re.IGNORECASE)
    if m:
        return _handle_stop(m.group(1))
    if lower in ("stop", "remove"):
        return json.dumps({"success": False, "error": "Usage: /loop stop <job_id>\n       (partial IDs work — you can use the first few characters)"})

    # Parse as create: [interval] <prompt> [--skills ...] [--verify ...] [--name ...]
    return _handle_create(args, origin=origin)


def parse_loop_result(result_json: str) -> tuple[str, bool]:
    """Reduce a handle_loop_command() JSON result to (display_text, is_error).

    Centralizes the result contract so callers (CLI, gateway) don't each
    reimplement the message/error/success key handling. On unparseable input
    the raw string is returned verbatim with is_error=False. An empty
    display_text means "success with nothing to show" — callers decide what,
    if anything, to render.
    """
    try:
        result = json.loads(result_json)
    except (json.JSONDecodeError, TypeError, ValueError):
        return result_json, False

    msg = result.get("message", "")
    if msg:
        return msg, False
    if not result.get("success"):
        return result.get("error", "Unknown loop command error"), True
    return "", False


def _usage() -> str:
    return json.dumps({
        "success": True,
        "message": (
            "Usage: /loop [interval] <prompt> [--skills s1,s2] [--verify 'cmd'] [--name label]\n"
            "       /loop status\n"
            "       /loop pause <id>\n"
            "       /loop resume <id>\n"
            "       /loop stop <id>\n"
            "\n"
            "Modes:\n"
            "  /loop 30m check deploy       → timed: runs every 30 minutes\n"
            "  /loop check deploy            → dynamic: starts at 5m, adapts to output\n"
            "\n"
            "Partial IDs work — you can use the first few characters of a job ID.\n"
            "\n"
            "Examples:\n"
            "  /loop every 30m check the deployment status\n"
            "  /loop 2h monitor disk usage --verify 'df -h /' --name disk-watch\n"
            "  /loop check if tests are passing --verify pytest\n"
            "  /loop status\n"
            "  /loop pause abc123"
        ),
    })


def _classify_schedule(text: str) -> tuple:
    """Split a leading interval token from the prompt and pick the loop mode.

    Returns (schedule, prompt, dynamic, error):
      - timed:   schedule="every Nm", dynamic=False
      - dynamic: schedule="",         dynamic=True  (no leading duration)
      - error:   error set (e.g. a sub-minute interval)

    Loops are always recurring, so a bare duration ("30m") is normalized to its
    recurring form ("every 30m"); a one-shot loop would be a contradiction. The
    scheduler's floor is 1 minute, so sub-minute tokens are rejected outright.
    """
    parts = text.split(None, 1)
    first = parts[0]
    rest = parts[1] if len(parts) > 1 else ""

    # Explicit "every <dur>" form.
    if first.lower() == "every" and rest:
        dur_parts = rest.split(None, 1)
        dur = dur_parts[0]
        prompt = dur_parts[1] if len(dur_parts) > 1 else ""
        if _SUBMINUTE_RE.match(dur):
            return "", "", False, f"Sub-minute intervals aren't supported (minimum 1m): {dur}"
        if _INTERVAL_RE.match(dur):
            return f"every {dur}", prompt, False, None
        # "every" not followed by a duration — treat the whole text as a prompt.
        return "", text, True, None

    # Bare leading duration → normalize to recurring form.
    if _SUBMINUTE_RE.match(first):
        return "", "", False, f"Sub-minute intervals aren't supported (minimum 1m): {first}"
    if _INTERVAL_RE.match(first):
        return f"every {first}", rest, False, None

    # No leading interval → dynamic mode, the whole text is the prompt.
    return "", text, True, None


def _parse_create_args(text: str) -> dict:
    """Parse /loop create arguments.

    Two modes, decided by the leading token:
        Timed:   <interval> <prompt> [--flags]   (e.g. "30m ...", "every 2h ...")
        Dynamic: <prompt> [--flags]              (first word isn't a duration)

    A bare duration is normalized to recurring form and sub-minute intervals are
    rejected — see _classify_schedule.

    Returns dict with keys: schedule, prompt, skills, verify, name, dynamic, error.
    """
    result = {
        "schedule": "", "prompt": "", "skills": None,
        "verify": None, "name": None, "dynamic": False, "error": None,
    }

    # Extract flags in order: --name first, then --skills, then --verify.
    # This ordering prevents --verify's greedy match from eating other flags.

    # --name label
    name_match = re.search(r"--name\s+(\S+)", text)
    if name_match:
        result["name"] = name_match.group(1).strip()
        text = text[:name_match.start()] + text[name_match.end():]

    # --skills s1,s2
    skills_match = re.search(r"--skills?\s+(\S+)", text)
    if skills_match:
        raw_skills = skills_match.group(1)
        result["skills"] = [s.strip() for s in raw_skills.split(",") if s.strip()]
        text = text[:skills_match.start()] + text[skills_match.end():]

    # --verify 'command' or --verify "command" (quoted)
    verify_match = re.search(r"""--verify\s+(['"])(.*?)\1""", text, re.DOTALL)
    if verify_match:
        result["verify"] = verify_match.group(2).strip()
        text = text[:verify_match.start()] + text[verify_match.end():]
    else:
        # --verify command (no quotes, rest of line — safe because
        # --skills and --name are already stripped)
        verify_match = re.search(r"--verify\s+(\S+.*)", text)
        if verify_match:
            result["verify"] = verify_match.group(1).strip()
            text = text[:verify_match.start()]

    text = text.strip()
    if not text:
        result["error"] = "Missing prompt text"
        return result

    schedule, prompt, dynamic, error = _classify_schedule(text)
    if error:
        result["error"] = error
        return result
    if not prompt.strip():
        result["error"] = "Missing prompt text"
        return result

    result["schedule"] = schedule
    result["prompt"] = prompt.strip()
    result["dynamic"] = dynamic
    return result


def _handle_create(text: str, origin: Optional[dict] = None) -> str:
    """Create a new loop cron job."""
    parsed = _parse_create_args(text)
    if parsed.get("error"):
        return json.dumps({"success": False, "error": parsed["error"]})

    dynamic = parsed.get("dynamic", False)
    schedule = parsed["schedule"] if not dynamic else "every 5m"

    try:
        from cron.jobs import create_job
        job = create_job(
            prompt=parsed["prompt"],
            schedule=schedule,
            name=parsed.get("name"),
            deliver="origin",
            origin=origin,
            loop=True,
            loop_dynamic=dynamic,
            loop_verify=parsed.get("verify"),
            skills=parsed.get("skills"),
        )
        job_name = job.get("name", job["id"])
        verify_line = f"   Verify: {parsed['verify']}\n" if parsed.get("verify") else ""
        mode_line = "   Mode: dynamic (starts at 5m, adapts to output changes)\n" if dynamic else ""
        return json.dumps({
            "success": True,
            "action": "created",
            "job_id": job["id"],
            "schedule": job.get("schedule_display", schedule),
            "prompt": parsed["prompt"][:100],
            "dynamic": dynamic,
            "next_run_at": job.get("next_run_at"),
            "skills": parsed.get("skills"),
            "verify": parsed.get("verify"),
            "name": parsed.get("name"),
            "message": (
                f"🔄 Loop created: {job['id']}\n"
                f"   Name: {job_name}\n"
                f"   Schedule: {job.get('schedule_display', schedule)}\n"
                f"   Prompt: {parsed['prompt'][:80]}{'...' if len(parsed['prompt']) > 80 else ''}\n"
                f"{mode_line}"
                f"{verify_line}"
                f"   Next run: {job.get('next_run_at', 'unknown')}\n"
                f"   Auto-pause: after 3 consecutive no-progress detections"
            ),
        })
    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)})


def _handle_status() -> str:
    """List all loop jobs."""
    try:
        from cron.jobs import list_jobs
        all_jobs = list_jobs(include_disabled=True)
        loop_jobs = [j for j in all_jobs if j.get("loop")]

        if not loop_jobs:
            return json.dumps({
                "success": True,
                "jobs": [],
                "message": "No loop jobs configured.",
            })

        lines = ["Loop jobs:"]
        for j in loop_jobs:
            state = j.get("state", "unknown")
            name = j.get("name", j.get("id", "?"))
            schedule = j.get("schedule_display", "?")
            count = j.get("loop_no_progress_count", 0)
            threshold = j.get("loop_no_progress_threshold", 3)
            verify = j.get("loop_verify")
            verify_err = j.get("loop_last_verify_error")
            dynamic = j.get("loop_dynamic", False)
            mode_tag = " [dynamic]" if dynamic else ""
            lines.append(
                f"  {'⏸' if state == 'paused' else '▶'} {j['id']} "
                f"({state}) [{schedule}]{mode_tag} {name[:40]}"
            )
            lines.append(f"    no-progress: {count}/{threshold}")
            if verify:
                status_icon = "❌" if verify_err else "✓"
                lines.append(f"    verify: {status_icon} {verify[:60]}")

        return json.dumps({
            "success": True,
            "jobs": [
                {
                    "id": j["id"],
                    "state": j.get("state", "unknown"),
                    "schedule": j.get("schedule_display", "?"),
                    "name": j.get("name", ""),
                    "dynamic": j.get("loop_dynamic", False),
                    "no_progress_count": j.get("loop_no_progress_count", 0),
                    "threshold": j.get("loop_no_progress_threshold", 3),
                    "verify": j.get("loop_verify"),
                    "verify_error": j.get("loop_last_verify_error"),
                }
                for j in loop_jobs
            ],
            "message": "\n".join(lines),
        })
    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)})


def _handle_pause_resume(job_ref: str, action: str) -> str:
    """Pause or resume a loop job."""
    try:
        from cron.jobs import pause_job, resume_job
        if action == "pause":
            job = pause_job(job_ref, reason="user-paused")
        else:
            job = resume_job(job_ref)

        if not job:
            return json.dumps({"success": False, "error": f"Job not found: {job_ref}"})

        return json.dumps({
            "success": True,
            "action": action + "d",
            "job_id": job["id"],
            "state": job.get("state"),
            "message": f"{'⏸' if action == 'pause' else '▶'} Loop {action}d: {job['id']}",
        })
    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)})


def _handle_stop(job_ref: str) -> str:
    """Stop (remove) a loop job."""
    try:
        from cron.jobs import remove_job
        removed = remove_job(job_ref)
        if removed:
            return json.dumps({
                "success": True,
                "action": "removed",
                "message": f"🗑 Loop stopped and removed: {job_ref}",
            })
        return json.dumps({"success": False, "error": f"Job not found: {job_ref}"})
    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)})
