# Hermes Phase 6 — Authentication and Credential Ownership

Status: 6.1–6.8 implementation and closeout complete on the independent baseline. Validation results, baseline failures and the full TUI gate limitation are recorded in `PHASE6_AUTH_CLOSEOUT.md`; the independent PR remains draft.
See `PHASE6_AUTH_STORAGE.md`, `PHASE6_AUTH_POOL.md`, `PHASE6_AUTH_OAUTH.md` `PHASE6_AUTH_CONSUMERS.md` and `PHASE6_AUTH_PRESENTATION.md` for ownership cuts and verification.
Branch: `refactor/phase6-auth-credentials`.
Base: One Gateway `bc1b572e2c`. Phase 6 is independent of Phase 5.

## Problem
Runtime authentication currently straddles `hermes_cli.auth*`, `agent.credential_*`,
and provider plugins. Runtime consumers import CLI-owned state and persistence.
Move once to one canonical auth owner without changing observable behaviour.

## Target architecture
- `auth/` owns credential selection, pool lifecycle, storage, source suppression,
  shared OAuth lifecycle, single-use grant hygiene, and runtime refresh invocation.
- `providers/` owns canonical provider metadata and provider plugin discovery.
  Provider-specific plugins continue to own concrete protocol hooks/endpoints.
- `agent/`, `gateway/`, TUI and auxiliary clients consume auth results for a
  selected provider, endpoint and explicit profile/execution scope.
- The CLI owns prompts, browser login presentation, status output and argument parsing.
- The existing config subsystem owns raw config reads and writes. Provide auth
  with explicit application-level configuration; never import CLI config from auth.

Do not create parallel auth registries, stores, credential pools or internal shims.
The planned credential request carries provider ID, endpoint identity, optional
model cooldown context and profile scope. The result identifies the selected
credential/lease and distinguishes unavailable, exhausted, dead, and failed refresh.
Retain `ProviderProfile.auth_handler(action, args)` as an external CLI-edge hook
and `refresh_credential(entry)` as the provider's runtime refresh hook.

## Independence and Phase 0 boundary
One Gateway at this base does not contain `nous_cli/`. Do not make Phase 6
depend on Phase 0/5 or create a disposable second CLI. For the independent
Phase 6 PR, leave real auth command presentation in `hermes_cli`, with *zero*
shared runtime auth implementation in that namespace. Relocate that presentation
to `nous_cli` only during subsequent CLI integration with Phase 0.
Phase 5 chooses provider/model/route; Phase 6 chooses credentials for that route.

## Implementation slices
1. 6.1 Bootstrap branch/worktree, inventory and baseline tests; add `auth/`
   package and a guard prohibiting auth -> CLI imports.
2. 6.2 Define canonical auth request/result/lease contracts and application-
   supplied configuration/context. Preserve lazy provider discovery behaviour.
3. 6.3 Hard-cut auth.json store, locks, writes, provider state, suppression,
   and coordinated removal into auth. Keep file format and locations stable.
4. 6.4 Move credential pool, cooldowns, rotation, administration and
   credential-specific source/persistence behaviour from agent to auth.
5. 6.5 Move shared OAuth and built-in runtime authentication into auth.
   Preserve provider-specific hooks and CLI-only user interaction.
6. 6.6 Rewire agent, Gateway, TUI, auxiliary and plugin runtime consumers directly.
   Delete obsolete internal runtime implementations and import paths.
7. 6.7 Leave the independent branch's CLI presentation-only; document
   the `nous_cli` presentation cutover for integration without a temporary facade.
8. 6.8 Run package, regression and structural gates; independent PR.

## Non-negotiable invariants
- Credential isolation across provider, endpoint and served profile.
- Exactly one owner for single-use OAuth grants; cloned profiles must not fork.
- Concurrent refresh adopts another process's successful rotation.
- Persistence failures never publish invalid rotated credentials as durable.
- A removed source stays suppressed until deliberately re-added.
- Preserve external Codex/Claude login adoption settings and source provenance.
- Preserve pool strategies, cooldowns, terminal-failure semantics and recovery.
- Preserve redaction, lock discipline, private file permissions and on-disk data.
- Keep dashboard session auth, platform authz and generic secret backends separate.

## Migration rules and verification
Staged hard cut: define final owner first, move implementation once, rewire
consumers directly, then delete the old path. Temporary targeted test failures
are acceptable inside declared migration windows; no internal aliases, dual
writes, forwarding facades or work solely for intermediary greenness.
Named external persisted formats and public plugin hooks remain compatible.

Module gate: targeted store/pool/OAuth and plugin hook tests, direct import audit.
Consumer gate: agent/Gateway/TUI/auxiliary/CLI contract and isolation regressions.
Final gate: whole-repo build, relevant suite baseline and wheel packaging;
no runtime imports of retired `hermes_cli.auth*` or `agent.credential_pool*` internally,
no auth -> CLI imports, no redundant legacy implementation, same persisted
format and supported external hooks. Arcana supplements direct source checks
when its graph is valid; it cannot substitute for regression tests.

Out of scope: Phase 5 model routing, config PR #122245, dashboard user auth,
Gateway authz, unrelated vault redesign, new auth types and global CLI cutover.
