# BRIEF — hermes-agent: gateway must FAIL LOUD when an essential port-binding platform cannot bind

Repo worktree: /Users/engineer/workspace/hermes-worktrees/t_b241802d
Branch: fix/gateway-eaddrinuse-fail-loud (based on origin/main c80d12b9b9)
Venv: .venv (already synced: `uv sync --locked --python 3.11 --extra all --extra dev`). Use `.venv/bin/python -m pytest`.

## 0. What is ALREADY TRUE on main (I measured it — do not re-litigate, but do not assume more than this)

I ran `.venv/bin/python repro_eaddrinuse.py` (committed at ecf320c256) against real code. Verbatim:

```
[Api_Server] Could not bind 127.0.0.1:58553: [Errno 48] error while attempting to bind on address ('127.0.0.1', 58553): address already in use. Set a different port in config.yaml: platforms.api_server.port
1 configured platform(s) failed to start and are parked (fix the reported error, then `hermes gateway restart`): api_server: Port 58553 already in use. ... The gateway is DEGRADED — it serves the remaining platform(s) with those unserved.
[repro] api_server connect() -> False
[repro] has_fatal_error=True code='api_server_port_in_use' retryable=False
[repro] _start_handle_no_connections -> must_exit=False
[repro] _serving_state() -> 'degraded'
[repro] queued for retry: {}
[repro] readiness.status=degraded gateway_check={'status': 'degraded', 'state': 'degraded', 'connected_platforms': 1, 'platforms': 2}
```

So: the runtime status file and /readiness ALREADY say degraded. The card's claim that the
degraded state is invisible everywhere is partially false and MUST NOT be repeated in the PR body.

What is genuinely broken, and what you are fixing:
1. Nothing ever retries the bind. `_BIND_ATTEMPTS = 5` with `0.2*(attempt+1)` sleeps = ~3s total,
   inside connect(), and then the adapter is classified non-retryable and dropped from the
   reconnect queue forever (`queued for retry: {}` above). When the competing process is stopped
   ten seconds later, the port stays dark for the life of the process.
2. The process stays up and `systemctl is-active` reports `active`. A provisioner that polls the
   PORT (which is what AgentPod does: `curl 127.0.0.1:18789`) sees nothing, forever, with the
   service green. That is the 1200s `gateway_ready` timeout.

## 1. The decision (already made — implement it, do not redesign)

A platform that BINDS A PORT is categorically different from a messaging platform: its absence is
not just degraded service, it is an unreachable contract surface that no external poller can
distinguish from a hung host. So:

**Rule: at startup, a non-retryable connect failure of a platform for which
`gateway.config.platform_binds_port(platform.value, extra)` is True is FATAL — the gateway exits
`GATEWAY_FATAL_CONFIG_EXIT_CODE` (78) even when sibling platforms connected.**

Engage with the existing counter-reasoning in the docstring at run_startup.py:1303-1319 rather
than deleting it. The reasoning there ("exiting 78 would take the gateway PERMANENTLY down over a
blip, and the retryable platforms never get their retry") is correct for a WhatsApp pairing
failure next to a transient Telegram timeout. It does not transfer to a port binder: systemd's
`RestartPreventExitStatus=78` parking the unit makes `systemctl is-active` report `failed`, which
is exactly the loud, pollable signal that is missing. A human losing Telegram for the minutes it
takes to read the journal is strictly cheaper than a tenant wedged in `provisioning` for 20
minutes with everything green. Say this in the code comment; do not just assert it.

**Plus: widen the bind retry window so the losing-race case self-heals before it ever becomes
fatal.** EADDRINUSE from a restart race resolves in seconds. In `api_server.py` connect(), replace
the fixed `_BIND_ATTEMPTS = 5` / `0.2*(attempt+1)` schedule with a bounded WALL-CLOCK budget
(~30s, exponential backoff capped at ~2s per sleep, still EADDRINUSE-only). Keep the final
classification non-retryable — the reconnect-watcher fd-leak reasoning at api_server.py:4245-4258
is still valid and must survive. Do not remove that comment.

### Derivation requirement (hard — a hand list is an automatic rejection)

The fatal set must be DERIVED from `gateway.config.PORT_BINDING_PLATFORM_VALUES` /
`platform_binds_port()`, which is the existing source of truth. Do NOT write a literal tuple like
`{"api_server", "webhook"}` anywhere in run_startup.py. And add a derive-check test that FAILS if
someone reintroduces a literal: assert the fatal classification's answer for every member of
`PORT_BINDING_PLATFORM_VALUES` is True and for a sample of non-port platforms is False, computed
by calling the production helper, not by restating the set.

## 2. Where to change

- `gateway/run_startup.py` `_start_handle_no_connections` (line ~1272): the `connected_count != 0`
  branch currently always `return False`. It must first partition `startup_nonretryable_errors`
  into port-binders and the rest. You will need the platform identity, which that function does
  not currently receive — `_start_aggregate_connect_results` (line ~1172) appends formatted
  strings `f"{platform.value}: {msg}"` into the list. Do NOT parse those strings back apart.
  Thread structured data through instead (e.g. append `(platform, adapter.config.extra, message)`
  tuples to a parallel list, or change the list element type and format at the log sites). Keep
  the log wording that exists.
- Fatal path: reuse `self._startup_fail_fatal_config(reason)` and `return True` so the existing
  `gateway_state="startup_failed"` + exit-78 plumbing is used unchanged. Do not invent a new exit code.
- `gateway/platforms/api_server.py` connect() bind loop (line ~4226-4263): wall-clock retry budget.

## 3. Tests (new file `tests/gateway/test_startup_port_binder_fatal.py`)

1. Real-socket regression: squat a real ephemeral port, build a real `APIServerAdapter` with a
   valid `API_SERVER_KEY` (>=16 hex chars; a short key trips a DIFFERENT guard and silently tests
   nothing — I hit that), assert connect() is False with `fatal_error_code ==
   "api_server_port_in_use"`.
2. Startup gate: with `connected_count=1` (a sibling connected) and the api_server failure in the
   non-retryable list, assert `_start_handle_no_connections` returns True, `runner._exit_code ==
   GATEWAY_FATAL_CONFIG_EXIT_CODE`, and the runtime status on disk reads `gateway_state ==
   "startup_failed"` (use `gateway.status.read_runtime_status`, HERMES_HOME in tmp_path).
   Assert explicitly that it does NOT read "running" and does NOT read "degraded".
3. Non-regression: a NON-port-binding non-retryable failure (e.g. whatsapp pairing) with a
   connected sibling must still return False and stay up with `_serving_state() == "degraded"`.
   This is the behaviour the old comment protects; breaking it is a rejection.
4. Derive-check as described in §1.
5. Bind-retry: assert the retry loop keeps trying past 5 attempts within its budget (monkeypatch
   the clock/sleep; do not sleep 30s in CI) and that a port freed mid-window binds successfully.

## 4. Mutation proof (mandatory, both directions, verbatim output in MUTATION_PROOF.md)

Commit first — `git status --short` must be empty before you mutate anything.
For EACH of: (a) the fatal classification, (b) the derive-check, (c) the widened bind budget:
break it, run the new test file, capture RED; `git checkout -- <file>`, rerun, capture GREEN.
Paste both transcripts with the command lines.

## 5. Also run

`.venv/bin/python -m pytest tests/gateway/test_runner_startup_failures.py tests/gateway/test_api_server_bind_guard.py tests/gateway/test_api_server.py -q`
plus your new file. Report the real counts. Do not pipe through tail.

## 6. Housekeeping

- DELETE `repro_eaddrinuse.py` from the branch in your final commit (it was my scratch repro; the
  evidence lives in the PR body and MUTATION_PROOF.md, not in the tree). Its removal must be in
  the diff.
- Do NOT touch AgentPod, do NOT bump any pin, do NOT open the PR — I open it.
- No new config knobs.

## 7. Escape hatch

If anything in this brief is wrong — in particular if the port-binder fatal rule breaks an
existing test that encodes a deliberate opposite decision, or if `platform_binds_port` turns out
to be the wrong source of truth — write the objection to OBJECTION.md and STOP. Do not improvise a
different design.
