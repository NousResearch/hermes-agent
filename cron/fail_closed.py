"""Fail-closed delivery gate for monitor cron jobs whose source did not read.

INCIDENT (2026-08-29, job 7993e0b1fa51 "Apon Upwork recovery tracker").  The
monitor script exited 2 ("Failed to fetch Upwork mail across local and remote
sources"), the Mac was unreachable, and the case card could not be loaded --
yet the job delivered a confident case update naming a dispute outcome, a
stated reason and a deadline, none of which had any source in the run at all.
The dollar figures in it were verbatim echoes of the job's OWN prompt baseline.

A fabricated narrative is worse than a missed tick: it reads as fresh
intelligence, and this one recommended a money action off the back of it.

The job prompt already said "Never invent a figure" and "If an amount cannot
be verified, say which one and why".  Prose instruction was not enough.  This
module is the mechanical gate, and it lives at the RUNNER level so it covers
every monitor job rather than one prompt.

THE GATE.  When a monitor source fails, the agent still runs -- local patch
013 (Sam's rule that no raw cron error ever reaches him) hands it the failure
as context so it can verify and, often, repair the source.  Before delivery
the runner RE-RUNS the monitor source and branches on the result:

  * green now  -> the agent actually repaired it.  Its report is grounded in a
    source that demonstrably runs, so it is delivered unchanged.
  * still red  -> the agent did NOT repair it, so anything it says about the
    watched subject has no source behind it.  The response is discarded and
    replaced with one deterministic, CONSTRUCTED line naming the source
    failure.  No figures, no case facts, no recommendation, no raw stderr.

The discriminator is whether the source RUNS, not whether the prose looks
confident.  That is verification, not text matching, and it cannot be talked
past by a more careful-sounding narrative.

Deliberate design notes:

* The replacement line is built by this module from the job definition and a
  timestamp.  It is not model output, so it cannot carry an invented figure.
* Raw stderr is never included.  Patch 013 exists precisely because raw cron
  errors must not reach Sam; this gate must not reintroduce that leak.
* ``[SILENT]`` from the agent is ALSO overridden while the source is still
  failing.  A monitor that has gone blind must say so; silently not watching
  is the failure mode the upstream code warned about.
* Nothing is persisted on recovery.  The stored hash stays untouched so the
  NEXT tick compares the now-working source against the real old baseline and,
  if the content genuinely moved, wakes the agent with a proper MONITOR CHANGE
  block.  Adopting the fresh output here would silently swallow a change the
  agent never actually saw.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Figure-shaped tokens: currency amounts, decimals, and long bare integers.
# Used for FORENSICS only (proving prompt-baseline echo in the audit log) --
# never as the gate itself, because a text heuristic is exactly the kind of
# thing a differently-worded fabrication walks past.
_FIGURE_RE = re.compile(
    r"(?:[$\u20ac\u00a3]\s?\d[\d,]*(?:\.\d+)?)"
    r"|(?:\b\d[\d,]*\.\d{2}\b)"
    r"|(?:\b\d{4,}\b)"
)


@dataclass
class FailClosedOutcome:
    """What the runner should deliver after a failed monitor source."""

    recovered: bool
    replacement: Optional[str] = None
    echoed_figures: tuple = field(default_factory=tuple)


def extract_figures(text: str) -> set:
    """Figure-shaped tokens in ``text``, normalised for comparison."""
    if not text:
        return set()
    out = set()
    for raw in _FIGURE_RE.findall(text):
        out.add(raw.replace(",", "").replace(" ", "").lstrip("$\u20ac\u00a3"))
    return out


def prompt_figure_echo(response: str, prompt: str) -> tuple:
    """Figures the response shares with its own prompt, sorted.

    When a monitor could not read anything, every figure in its response can
    only have come from the prompt's baseline.  Recording the overlap gives
    the audit log hard evidence of the echo without ever letting a text
    heuristic decide delivery.
    """
    shared = extract_figures(response) & extract_figures(prompt)
    return tuple(sorted(shared))


def source_label(job: dict) -> str:
    """Human-safe name for the job's monitor source.

    A ``monitor_url`` is reduced to its hostname: query strings can carry
    tokens and this string is delivered.
    """
    script = str(job.get("monitor_script") or "").strip()
    if script:
        return script.replace("\\", "/").rsplit("/", 1)[-1]
    url = str(job.get("monitor_url") or "").strip()
    if url:
        try:
            from urllib.parse import urlsplit

            host = urlsplit(url).hostname or ""
        except Exception:
            host = ""
        return host or "monitor url"
    return "monitor source"


def source_failure_notice(job_name: str, label: str, failed_at: str) -> str:
    """The ONLY thing a job may say when its monitor source did not read.

    Exactly one line, fully constructed here.  It names what failed and when,
    and states plainly that nothing was read -- so there is no figure, no case
    fact and no recommendation for a reader to act on, because the run has
    nothing to base any of those on.
    """
    return (
        f"Monitor source for '{job_name}' ({label}) failed at {failed_at} and was "
        f"still failing after this run. Nothing was read, so nothing is reported."
    )


def enforce_source_failure_silence(
    job: dict,
    *,
    job_name: str,
    failed_at: str,
    agent_response: str,
    prompt: str = "",
) -> FailClosedOutcome:
    """Decide what may be delivered after this tick's monitor source failed.

    Re-runs the monitor source.  Recovery means the agent repaired it and may
    report freely; a source that still fails means the agent never had one, so
    its response is replaced with :func:`source_failure_notice`.

    Any exception from the re-run is treated as still-failing.  This gate is
    fail-CLOSED by name and by intent: an unprovable source is a silent one.
    """
    from cron.monitor import _run_monitor_source

    try:
        ok, _output = _run_monitor_source(job)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Fail-closed gate: monitor re-run raised for %r (%s); treating as failed",
            job.get("id"),
            exc,
        )
        ok = False

    if ok:
        logger.info(
            "Fail-closed gate: monitor source for %r recovered during the run; "
            "delivering the agent's report",
            job.get("id"),
        )
        return FailClosedOutcome(recovered=True)

    echoed = prompt_figure_echo(agent_response or "", prompt or "")
    if agent_response and agent_response.strip():
        logger.error(
            "Fail-closed gate: monitor source for %r STILL failing after the run; "
            "discarding %d chars of ungrounded agent response (prompt-baseline "
            "figures echoed: %s)",
            job.get("id"),
            len(agent_response),
            ", ".join(echoed) if echoed else "none",
        )
    else:
        logger.error(
            "Fail-closed gate: monitor source for %r STILL failing after the run "
            "and the agent stayed silent; emitting the blind-monitor notice",
            job.get("id"),
        )

    return FailClosedOutcome(
        recovered=False,
        replacement=source_failure_notice(job_name, source_label(job), failed_at),
        echoed_figures=echoed,
    )
