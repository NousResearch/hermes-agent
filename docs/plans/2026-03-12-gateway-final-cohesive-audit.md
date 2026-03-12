# Gateway Final Cohesive Audit (2026-03-12)

> Scope: reconstruct the phase plan + review outcomes, validate implementation end-to-end, and produce rollout-ready operating docs.

## 1) What was audited

### Reviewed implementation areas
- Runtime override UX and routing controls in `gateway/run.py`
  - `/ask` one-turn overrides (`model=...`, `provider=...`, `reasoning=...`)
  - `/modelpin`, `/reasoning`, `/route`, `/runtime`
  - command confirmation framing and receipts
- Command governance in `gateway/run.py`
  - write-command RBAC checks
  - write-command audit log (`~/.hermes/logs/gateway_command_audit.jsonl`)
- Execution-owner command surface in `gateway/run.py`
  - `/now`, `/blocked`, `/next` snapshot reporting
- Discord ingest hardening in `gateway/platforms/discord.py`
  - free-response inheritance in thread children
  - document/archive attachment caching
  - markdown/text attachment content injection
  - slash parity for `/ask` model/reasoning + `/now` `/blocked` `/next`
- Provider/model parsing hardening in `hermes_cli/models.py`
  - `provider/model` support for hyphenated providers
  - Codex shorthand inference under openrouter/auto defaults

### Review context from commit history
- `58fa624` — session lifecycle + telemetry/steer/queue controls
- `60fa624` — retry compression and service/tool-call UX improvements
- `87ad9c5` — runtime telemetry coverage and rollout drills
- `1827968` — phase6 runtime routing controls/hardening
- `bc62697` — free-response behavior in Discord threads

## 2) Test evidence (final pass)

### Targeted feature tests
```bash
source venv/bin/activate
python -m pytest \
  tests/gateway/test_routing_policy.py \
  tests/gateway/test_command_rbac_audit.py \
  tests/gateway/test_exec_owner_commands.py \
  tests/gateway/test_discord_attachments.py \
  tests/hermes_cli/test_model_validation.py -q
```
Result: **58 passed**.

### Gateway suite sanity
```bash
source venv/bin/activate
python -m pytest tests/gateway/ -q
```
Result: **543 passed**.

### Syntax sanity
```bash
source venv/bin/activate
python -m py_compile gateway/run.py gateway/platforms/discord.py hermes_cli/models.py
```
Result: success.

## 3) Cohesive findings

### ✅ Confirmed good
1. Per-query high-reasoning flow is wired through `/ask ... reasoning=high <prompt>` and is surfaced with a visible runtime receipt.
2. Thread/session runtime state controls are coherent (`/modelpin`, `/reasoning`, `/route`, `/runtime`).
3. Command responses now use consistent, explicit confirmation formatting.
4. Write-command RBAC + audit trail are enforced at command dispatch point.
5. Discord attachment ingestion now supports text/docs/archives with local caching and markdown injection.
6. Execution-owner commands (`/now`, `/blocked`, `/next`) produce actionable health snapshots and bridge suggestions.

### ⚠️ Remaining operational risks (non-blocking for merge)
1. `AGENTS.md` says `.venv`, environment currently uses `venv` in this checkout.
2. Transient runtime warnings in unrelated gateway tests (AsyncMock warnings) still exist; they do not fail tests but should be cleaned in future hygiene pass.
3. Untracked artifact file exists: `hermes_cli/models.py.bak.codex-provider-infer-20260311-213529` (remove before release commit unless intentionally kept).

## 4) Release recommendation

**Recommendation: ship-ready for the audited scope**, with one pre-merge cleanup step:
- remove backup artifact file from working tree.

## 5) Post-merge verification checklist (Discord)

1. `/help` lists: `/ask`, `/modelpin`, `/reasoning`, `/route`, `/runtime`, `/now`, `/blocked`, `/next`.
2. Run:
   - `/ask reasoning=high summarize current blockers`
   - confirm runtime receipt shows reasoning `high`.
3. In an allowed write user:
   - run `/sethome` and confirm audit line appended.
4. In a non-allowlisted user:
   - run `/sethome` and confirm explicit authorization denial.
5. Send `.md` attachment with a question and confirm injected text appears in assistant context handling.

## 6) Suggested commit grouping

1. `feat(gateway): finalize runtime override, RBAC audit, and exec-owner command surface`
2. `feat(discord): attachment/document caching + slash command parity`
3. `feat(models): provider/model parse hardening and codex shorthand inference`
4. `test(gateway): add routing/RBAC/exec-owner/attachment coverage`
5. `docs: add final cohesive audit + runtime ops runbook`
