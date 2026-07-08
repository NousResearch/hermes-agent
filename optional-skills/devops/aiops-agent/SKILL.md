---
name: aiops-agent
description: Triage service incidents from logs, alerts, and metrics.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [aiops, incident-response, logs, monitoring, observability, sre, triage]
    category: devops
    requires_toolsets: [terminal]
    related_skills: [watchers, docker-management]
---

# AIOps Agent

An AIOps agent turns raw signal (logs, alerts, metrics) into a diagnosis and a
remediation plan. This skill is the on-demand triage runbook: gather evidence
with `terminal`, summarize and fingerprint errors with the bundled
`log_triage.py`, correlate against recent changes, and propose remediation —
safe steps first, destructive steps only with explicit confirmation.

## When to Use

- User reports a service/system is degraded, erroring, or down and wants root cause
- User pastes or points at a log dump and asks "what's wrong here?"
- User wants a recurring class of incident turned into an automated check
- Don't use for: continuously polling a feed/API when nothing is wrong yet — use the `watchers` skill for that
- Don't use for: container lifecycle management with no incident involved — use the `docker-management` skill for that

## Prerequisites

- `terminal` toolset enabled (required)
- Read access to the relevant log source for the target platform: `journalctl`,
  `docker logs`, `kubectl logs`, Windows Event Log, or the macOS unified log
- Optional: `docker`/`kubectl` CLI configured if triaging containerized workloads
- Python 3 available to run `scripts/log_triage.py` (stdlib only, no extra installs)

## Quick Reference

| Task | Command |
|---|---|
| Tail a systemd service's logs (Linux) | `journalctl -u <service> -n 200 --no-pager` |
| Follow a container's logs | `docker logs --tail 200 -f <container>` |
| Tail a Kubernetes pod's logs | `kubectl logs <pod> -n <namespace> --tail=200` |
| Logs from a pod's previous (crashed) instance | `kubectl logs <pod> -n <namespace> --previous` |
| Recent Windows Event Log errors | `Get-WinEvent -LogName System -MaxEvents 200 \| Where-Object LevelDisplayName -eq 'Error'` |
| macOS unified log, last hour | `log show --predicate 'eventMessage contains "error"' --last 1h` |
| Host resource snapshot (Linux/macOS) | `uptime; free -h; df -h` |
| Host resource snapshot (Windows) | `Get-CimInstance Win32_OperatingSystem \| Select-Object FreePhysicalMemory; Get-PSDrive` |
| Recent deploys/changes in a repo | `git log --oneline --since="2 hours ago"` |
| Summarize + fingerprint a log dump | `python scripts/log_triage.py --file app.log` |

## Procedure

1. **Scope the incident.** Done when you can state the affected service, the
   time window, and the user-visible symptom in one sentence.
2. **Collect evidence.** Pull logs from the relevant surface (systemd/docker/
   kubectl/cloud) plus a change timeline (deploys, config edits, dependency
   bumps) for the same window. Done when you have raw log text and a change
   timeline covering the incident window.
3. **Run `log_triage.py`** on the collected logs to get severity counts, a
   ranked list of error signatures, and spike detection. Done when you have
   the ranked signature list in hand.
4. **Correlate.** Check whether the top signature's onset lines up with an
   entry in the change timeline. Done when you can name a leading hypothesis
   or explicitly state that nothing correlates.
5. **Verify the hypothesis** with a targeted check — reproduce it, read the
   failing code path, or check a dependency's status page. Done when the
   hypothesis is confirmed, refuted, or escalated with evidence either way.
6. **Propose remediation**, splitting steps into **safe** (restart a crashed
   pod, roll back a bad config, clear a full disk's temp files) and
   **destructive** (force-delete data, drop a database, kill unrelated
   processes). Destructive steps always require explicit user confirmation
   before execution. Done when every step in the plan is labeled.
7. **Execute approved steps one at a time**, re-checking the original symptom
   after each. Done when the symptom is cleared and re-verified, or you
   escalate because it isn't.
8. **Document and close the loop.** Record root cause, timeline, and the fix
   in a short incident summary. If this is a recurring failure class, propose
   a `watchers` job or a `cronjob` health check to catch it earlier next
   time. Done when the summary is written and any follow-up monitor is
   proposed.

## Common Pitfalls

| Problem | Cause | Fix |
|---|---|---|
| Chasing every WARN line | Warn-level noise dominates volume | Filter to ERROR/CRITICAL first — `log_triage.py`'s counts show the split |
| False "spike" from a restart burst | App logs a burst of startup errors on boot | Exclude the first N seconds after a known restart from the analysis window |
| Auto-restarting in a crash loop | Remediation restarts blindly on every failure | Cap restart attempts; if it crashes again within minutes, stop and escalate |
| Pasting raw logs with secrets into chat/tickets | Logs often carry tokens, connection strings, or PII | Redact example lines before sharing; don't forward raw log dumps verbatim |
| Treating correlation as root cause | Deploy timing coincidence isn't causation | Confirm with a targeted reproduction or code read before declaring root cause |

## Verification Checklist

- [ ] Affected service, time window, and symptom stated in one sentence
- [ ] Logs and a change timeline collected for the incident window
- [ ] `log_triage.py` run and top error signatures reviewed
- [ ] Hypothesis confirmed or refuted with concrete evidence
- [ ] Remediation steps labeled safe/destructive; destructive steps confirmed with the user
- [ ] Symptom re-checked after remediation and confirmed resolved
- [ ] Root cause and fix documented; recurring-failure follow-up proposed if relevant
