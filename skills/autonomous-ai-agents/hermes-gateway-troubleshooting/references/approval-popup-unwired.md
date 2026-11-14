# Approval popup unwired after desktop restart — evidence & PR brief (2026-08-01)

Companion to SKILL.md §5 "Desktop: Approval popup never appears — instant silent
decline after app restart". Everything needed to re-diagnose the failure or hand
the upstream fix to a coding agent.

## Symptom timeline (observed 2026-08-01)

| Time | What happened |
|---|---|
| 05:53–06:21 | Approvals **worked** — requests took 4–11s each = user clicked the popup |
| 08:26 | Desktop app restarted (for an unrelated Ship-By fix) |
| 11:05–11:07 | All 4 SMS approvals returned **decline in 0.01s** — no human could respond that fast |
| after full desktop restart | Same send approved instantly, popup appeared, message delivered (msg id 256692008) |

## Root cause

1. The desktop registers a per-session approval callback:
   `register_gateway_notify(key, lambda data: _emit_approval_request(sid, data))`
   in `tui_gateway/server.py` (two sites, ~line 1994 and ~line 6276; the second is
   inside the session-build path). The callback emits the `approval.request`
   websocket event that renders the popup.
2. In `tools/approval.py`, the gateway branch (`check_dangerous_command` and
   `check_execute_code_guard`) does:
   ```python
   notify_cb = None
   with _lock:
       notify_cb = _gateway_notify_cbs.get(session_key)
   if notify_cb is not None:
       # ... blocking gateway approval (popup path) ...
   # else: FALLS THROUGH to the CLI input() fallback
   ```
3. The CLI fallback (`_request_approval`-style path, ~line 2497+) runs
   `input(prompt)` on a daemon thread; in a headless desktop context stdin is
   EOF, so `get_input()` catches `(EOFError, OSError)` and returns `""` → deny.
   Hence: instant, silent decline with no popup and no error in gateway logs.
4. Whether the popup path is even attempted depends on `is_gateway` /
   `is_cli`, which are derived from `HERMES_SESSION_PLATFORM` contextvars
   (`tools/approval.py::_is_gateway_approval_context`, ~line 227). After the
   08:26 desktop restart the per-session callback was not registered for the
   active session (registration is wrapped in `except Exception: pass` in
   `tui_gateway/server.py`, so a registration failure is invisible), so every
   approval fell to the input() fallback.

The gate fails **closed** (anything that isn't a real "accept" = decline), so
nothing slips out unapproved — the failure mode is availability (popup never
reaches the user), not security.

## Evidence file paths (Windows)

- Gateway status: `hermes gateway status` (healthy while bug active)
- MCP stderr: `~/AppData/Local/hermes/profiles/poncho/logs/mcp-stderr.log`
  (grep for `(declined/timeout)`)
- Desktop server: `C:\Users\dalia\AppData\Local\hermes\hermes-agent\tui_gateway\server.py`
  (`register_gateway_notify` call sites ~1994, ~6276; `_emit_approval_request` ~1592)
- Approval gate: `C:\Users\dalia\AppData\Local\hermes\hermes-agent\tools\approval.py`
  (gateway branch `_gateway_notify_cbs.get(session_key)` ~2947 and ~3636;
  CLI `input()` fallback ~2513-2525)
- Repo (live install): `C:\Users\dalia\AppData\Local\hermes\hermes-agent`
  (git remote `github.com/NousResearch/hermes-agent`, branch `main`)

## Ready-made Claude Code brief (upstream fix + PR)

Paste into Claude Code with the `hermes-agent` folder as the workspace:

> **Fix silent instant-decline of approvals in desktop sessions after an app restart, and open a PR.**
>
> **Repo:** This workspace (`C:\Users\dalia\AppData\Local\hermes\hermes-agent`, git remote `github.com/NousResearch/hermes-agent`, branch `main`).
>
> **Bug:** After the desktop app restarts, every dangerous-command / MCP approval returns `decline` in ~0.01s with no popup shown. The desktop's per-session approval callback (`register_gateway_notify(key, cb)` in `tui_gateway/server.py`, sites ~line 1994 and ~line 6276) is not registered for the active session, so `tools/approval.py`'s gateway branch falls through to the CLI `input()` fallback, which hits `EOFError` instantly in a headless context and silently denies.
>
> **Evidence:** `tools/approval.py` gateway branch (`_gateway_notify_cbs.get(session_key)` ~line 2947 in `check_dangerous_command`, ~line 3636 in `check_execute_code_guard`): when `notify_cb is None` the code falls through to the `input()` path (~line 2497-2525) instead of failing loudly. The `register_gateway_notify` call in `tui_gateway/server.py` is wrapped in `except Exception: pass`, so a registration failure is swallowed with no log.
>
> **Fix — do both:**
> 1. In `tools/approval.py`: when in a gateway/desktop context (`is_gateway` true) but `notify_cb is None` for the session key, do NOT fall through to the CLI `input()` path. Return a distinct `BLOCKED: no approval listener registered for this session` result (or equivalent) and log the session key loudly, so the agent surfaces a real error instead of a fake user decline.
> 2. In `tui_gateway/server.py`: replace the silent `except Exception: pass` around `register_gateway_notify(...)` with a logged warning (`logger.warning` with exc_info), so a failed registration is visible in logs instead of invisible.
>
> **Verification:** run existing approval tests; then start the desktop app, trigger a gated tool call, and confirm the popup appears (not a 0.01s decline). Also verify a session with NO registered callback returns the new BLOCKED message rather than falling into `input()`.
>
> **Deliverable:** commit on a new branch (e.g. `fix/approval-popup-unwired-after-restart`), push, and open a PR to upstream (fork first if no push access to NousResearch/hermes-agent). Explain root cause + fix in the PR description.

## Operational context

- While the bug is active, ANY gated action is impossible (SMS, dangerous
  commands) — the user sees "not approved" with no popup. Do not interpret it
  as the user declining; it is a wiring failure.
- The gate fails closed, so there is no security hole — only an availability
  problem. The desktop app restart is the reliable immediate fix.
