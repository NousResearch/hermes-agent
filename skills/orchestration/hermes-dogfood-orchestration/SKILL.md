---
name: hermes-dogfood-orchestration
description: Self-test checklist for verifying the orchestration stack (delegation, profiles, tool search, smart routing) is wired correctly on this Hermes install. Trigger after an upgrade, a config change, or an "is everything working?" request.
version: 1.0.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [orchestration, dogfood, self-test, diagnostics, verification]
    related_skills: [orchestration-patterns, task-router, model-selection-router]
---

# Hermes Dogfood — Orchestration Self-Test

> A scripted smoke test you run on yourself to confirm the orchestration stack is intact. Run it whenever the ground may have shifted under you.

## Trigger

Apply after:

- An **upgrade** (ran `upgrade-hermes.sh`, bumped a model, swapped a provider).
- A **config change** (edited a profile, changed `delegation.model`, toggled smart routing).
- An explicit **"is everything working?"** / "did the upgrade break anything?" request.

## Checklist

Run these in order. Each line is a single check with a clear pass condition.

1. **Version** — `hermes --version`
   - Pass: prints a version string matching what you expect post-upgrade.

2. **Doctor** — `hermes doctor`
   - Pass: clean run, no errors. Note any warnings to report.

3. **Delegation smoke** — `delegate_task` with a trivial task (e.g. "return the string OK").
   - Pass: a child agent spawns, completes, and returns its result to you. Confirms the dispatcher and `delegation.model` (`deepseek/deepseek-v4-flash`) are live.

4. **Profile smoke** — invoke the `engineer` profile with a 1-line task (e.g. "print the Python version via terminal").
   - Pass: it runs on the engineer toolset (terminal available) and uses the expected model. Confirms profile loading + toolset wiring.

5. **Tool search** — ask a question that forces a `tool_search` (e.g. a capability you don't have loaded by name).
   - Pass: the reranker returns relevant tool schemas. Confirms tool search + reranker are active.

6. **Smart routing (gateway only)** — send a short, simple message through the gateway.
   - Pass: it auto-classifies as short/simple and routes to `deepseek/deepseek-v4-flash`. Note: smart routing is **gateway-only** — it does **not** apply to the CLI, so this check is meaningless from a CLI session. Skip it (and say so) if you're not on the gateway.

## What "all green" looks like

- Version matches the target; `hermes doctor` clean.
- A delegated child round-trips a result.
- The `engineer` profile runs with terminal access on its expected model.
- Tool search returns usable schemas.
- (Gateway) a short message lands on the cheap model.

Report the result as a compact table: check / pass-fail / note.

## If something fails

- **`hermes doctor` errors** → fix those first; most downstream failures cascade from a bad doctor run.
- **Delegation child never returns** → check the dispatcher is running and `delegation.model` resolves to a reachable provider (OpenRouter for `deepseek-v4-flash`).
- **Profile uses the wrong model / lacks tools** → inspect the profile config on disk; a partial upgrade can leave a profile pointing at an old/removed model id.
- **Tool search returns nothing relevant** → the reranker or index may not have come back up after restart; re-run after confirming the index service is healthy.
- **Smart routing didn't downshift** → confirm you're actually on the gateway, not the CLI, and that smart routing is enabled in config.

## Related: the upgrade script

The relevant upgrade entry point on this install is:

```
/opt/hermes/home/.hermes/upgrade-hermes.sh
```

Run this self-test immediately after that script completes to confirm the orchestration stack survived the upgrade before handing the system back to normal use.
