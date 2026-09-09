# Route Registry — Central Slot Governance for the 64 Live Hermes Config Surfaces

Build artifact from the approved proposal
(`/home/kensei/provider-model-mapping-draft.md` v3, route governance design).
**Nothing here has been applied to any live config.** Build/deploy separation is preserved:
this directory produces plans only; applying them to live configs is a separate,
explicitly approved step.

## What lives here

| Path | Purpose |
|------|---------|
| `registry/route-slots.yaml` | Central slot registry: 34 route slots across 9 model slot sets + the Nous $20 emergency-reserve record. No credentials — symbolic provider_account IDs only. Also declares `pool_requirements` (native credential-pool cascade) and `env_bypass` (env keys that bypass the pool). |
| `registry/surfaces.yaml` | All 64 surfaces (63 profiles + root `default`): tier, preserved main model, slot chain, `routes_policy`. |
| `route_registry/generator.py` | Generator engine: validation, chain building, deployment dedup, diffing, removed-capability summary, pool/env checks, atomic plan/apply. |
| `generator_cli.py` | CLI entry point (dry-run default). |
| `local_transition_cli.py` | Isolated CAS plan/apply/rollback for the governed turbohaul-to-turbofit local transition. |
| `tests/test_generator.py` | 49 unit + integration + adversarial tests (run against temp copies only). |
| `out/dryrun-plan-20260901.json` | Generated dry-run plan for approval (deterministic; byte-identical across runs modulo the target-root path). |
| `route_registry/backups/` | apply-mode timestamped backups. Empty until an approved apply runs; `_archive-*` dirs are pre-remediation test artifacts retained for audit. |

## Multi-account design (approved architecture decision)

One fallback entry per **(provider, exact model, base_url)**. Multiple approved
accounts for the same deployment are represented as NATIVE credential-pool
membership (`credential_pool` + `pool_accounts` metadata on the single entry);
the runtime credential pool (`credential_pool_strategies`, `fill_first`)
performs the account 1→2 cascade. Duplicate same-deployment entries are never
emitted — the Hermes runtime dedup-skips chain entries resolving to the same
deployment (`agent/backend_identity.same_deployment` +
`chat_completion_helpers.should_skip_candidate`), so duplicates are dead weight,
never rotation.

**Env-key pool bypass:** Hermes resolves ollama.com endpoints via
`get_secret("OLLAMA_API_KEY")` (env preferred over the pool). Profiles whose
`.env` carries that key pin all ollama-cloud chain entries to one account.
Every dry-run/apply plan reports each hit as a deployment BLOCKER requiring
later approved auth cleanup. The scan reads env KEY NAMES only — values are
never read, copied, or logged.

## Quick start

Dry-run (default — never modifies anything; `--target-root` is REQUIRED):

```bash
# make a temp copy of the config surfaces first (NEVER point --target-root at live ~/.hermes)
rm -rf /tmp/rreg-hermes-copy && mkdir -p /tmp/rreg-hermes-copy/profiles
cp ~/.hermes/config.yaml /tmp/rreg-hermes-copy/config.yaml
for d in ~/.hermes/profiles/*/; do n=$(basename "$d"); \
  [ -f "$d/config.yaml" ] && mkdir -p "/tmp/rreg-hermes-copy/profiles/$n" && \
  cp "$d/config.yaml" "/tmp/rreg-hermes-copy/profiles/$n/config.yaml"; done

cd /home/kensei/repos/KenseiAgent/route-registry
../.venv/bin/python generator_cli.py \
  --target-root /tmp/rreg-hermes-copy \
  --out out/dryrun-plan.json
```

Apply (quadruple-guarded — refuses unless ALL hold):

```bash
# 1. explicit --apply  2. exact confirmation string  3. non-interactive stdin
#    4. target root is NOT the live ~/.hermes (or --allow-live-apply also passed)
python3 generator_cli.py --target-root /tmp/hermes-copy \
  --apply --confirm YES-APPLY-ROUTES --out applied.json
```

Apply refuses with exit code 2 when:
- `--confirm` is missing or not exactly `YES-APPLY-ROUTES`, or
- stdin is a TTY (interactive use blocked by design), or
- the target root does not exist.

Apply refuses with exit code 3 when the target root resolves to the LIVE
`~/.hermes` tree and the separate `--allow-live-apply` approval flag was not
supplied. Live-apply is a distinct, explicit approval step on top of the
triple guard.

Apply refuses with exit code 4 (fail-closed blocker gate) while:
- `env_key_bypass.blockers` is non-empty (profile .env keys bypass the native
  credential pool — approved auth cleanup required), or
- any `pool_requirement` entry has `status != met` (or carries blockers).

This gate is defense-in-depth: enforced both in the CLI (exit 4, before any
write) and inside `apply_plan` (raises `ValidationError` before any
snapshot/backup/write). Dry-run is never gated — it still generates the full
plan and reports the blockers for review.

Every apply is **atomic**: all originals are snapshotted before any write; any
failure mid-apply rolls back EVERY already-written file from the snapshots
(the target tree is left byte-identical to its pre-apply state), and backups
of all originals are retained in `backups/<UTC stamp>/`. Rollback = copy the
backup file back (or call `verify_rollback()` to confirm).

## Tests

```bash
cd /home/kensei/repos/KenseiAgent/route-registry
../.venv/bin/python -m pytest tests/ -q
# → 49 passed
```

Tests cover:

- registry schema completeness, class/status/disabled invariants (all GATED and
  NEEDS-CLASSIFICATION slots born `disabled: true`, fail closed)
- ≤5 same-model routes per surface, unique provider_account per slot set
- shape-agnostic main-model extraction vs all 64 live configs
- same-model invariant: every emitted route serves exactly the surface's current main model
- **no duplicate (provider, model, base_url) deployment entries in any emitted chain**
- **single ollama-cloud entry per surface carrying both approved pool accounts**
  (`pool_accounts: [ollama-cloud/1, ollama-cloud/2]`, `credential_pool` metadata)
- **pool cascade requirement** declared in the registry and checked against the
  target config's `credential_pool_strategies` (missing strategy → blocked)
- **env-key pool bypass blockers**: real .env scan (key names only), every
  OLLAMA_API_KEY-carrying profile reported as a deployment blocker
- **expired / usage-capped / allow-list-disallowed slots cannot emit** (fail-closed
  exclusion + registry-level validation refusal)
- effective chain = approved+enabled PERM slots only → Codex Sol/Luna (tier-correct) →
  local `custom:turbohaul-local/qwen3.8-27b`
- Nous emergency-reserve slot: never referenced, never emitted
- NIM: no slot exists until the 60-request window is verified from the live account
- `keep_current` surfaces emit zero changes
- **removed-capability summary**: every (provider, model) removed from existing
  chains is reported per-surface and in aggregate
- **real credential scan** (replaces the placebo docstring scan): emitted plans
  contain no key material and no env values; .env scanning sees key NAMES only
- CLI: dry-run default writes plans only; `--apply` without `--confirm
  YES-APPLY-ROUTES` refused (exit 2); **apply against the live tree refused
  (exit 3) without `--allow-live-apply`**; apply writes backups and produces
  perm-only, deduped chains
- **atomic apply with rollback**: injected mid-apply failure leaves every target
  byte-identical to pre-apply state; backups retained; rollback verifier
- plan determinism: identical runs produce byte-identical JSON (modulo target path)

## Governance rules enforced by the validator

1. **Exact-model invariant** — a slot's `model_id` must equal its surface's current
   main model; validator hard-fails on any mismatch.
2. **Max 5 same-model routes** per surface (enforced in registry validation).
3. **Gated/NC fail closed** — `disabled: true` slots are never emitted; only
   `class: perm, status: approved, disabled: false, hermes: != null` slots enter chains.
4. **One entry per deployment** — duplicate (provider, model, base_url) entries are
   collapsed into a single entry with native credential-pool membership metadata;
   account cascade is the pool's job (`credential_pool_strategies`).
5. **Env-key bypass blockers** — profiles with a provider env key in `.env`
   (e.g. `OLLAMA_API_KEY`) are reported as deployment blockers; apply approval
   must wait for approved auth cleanup.
6. **Expiry / usage-cap / allow-list fail closed** — expired slots, capped slots
   with unverified windows, and slots whose surface is not in `allowed_profiles`
   can never emit; an enabled slot with a past `expires_at` fails registry
   validation outright.
7. **Live-tree apply is separately guarded** — target root resolving to the live
   `~/.hermes` is refused (exit 3) unless `--allow-live-apply` is also passed.
8. **Atomic apply** — all targets are snapshotted before any write; any mid-apply
   failure rolls back every written file from the snapshots (byte-identical
   restore), and backups are always retained.
9. **Nous $20 emergency credit** — `slot-nous-emergency-credit` is
   `class: emergency_reserve, disabled: true, hermes: null`. It is structurally
   impossible to emit it into any chain; any surface referencing it fails validation.
10. **NIM 60-limit** — recorded as `nim_request_limit: 60`,
   `nim_window_verified: false`; validator rejects any NIM slot until the window is
   verified, and NIM can never be a model's sole route.

## Approval workflow (build → deploy separation)

1. Review `out/dryrun-plan-20260901.json`: 54 surfaces changed, 10 untouched
   keep-current surfaces, 0 gated routes active anywhere, 0 duplicate deployment
   entries. Check `removed_capabilities` (capability loss is intentional but must
   be reviewed), `pool_requirements` (must be `met`), and `env_key_bypass`
   blockers (must be resolved via approved auth cleanup BEFORE apply).
2. Sahil approves specific gated/NC slots by setting `status: approved`,
   `approved_at`, and `disabled: false` on those exact slots (per-route approval only —
   no bulk approvals).
3. Re-run the generator (dry-run) to produce the updated plan.
4. Only after explicit approval: run with `--apply --confirm YES-APPLY-ROUTES`
   against the target tree; backups land in `backups/<stamp>/`. The apply is
   atomic — any failure rolls the whole batch back.
5. Rollback: restore from `backups/<stamp>/`.
