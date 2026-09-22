# Desktop: real multi-gateway support — plan (2026-09-22)

**Status:** implemented on `fix/desktop-multi-gateway`; ships as fork `v0.21.1`.
**Canonical plan and results:** `tf-home-infra` `docs/agent-plans/0014-desktop-multi-gateway-and-agent-front-checks.md`
and its `0014R` results doc (the operator's infra repo, where the investigation started).
This copy is the fork-local record the Plan-Saving Rule in `AGENTS.md` asks for.

## Context

The operator added a second remote gateway (a second Agent Command deployment,
password-authenticated dashboard) in **Settings → Connections**. Sign-in appeared
to succeed, but switching to it reported "Remote Hermes gateway uses OAuth, but you
are not signed in". Separately, a reset of the client then failed with TLS alert 112
against the first gateway — a server-side ingress fault (the proxy host had dropped),
which the client reported as raw OpenSSL text directly above the "Allow self-signed
certificate" toggle that cannot fix it.

## Findings (verified against source)

1. **Cookie jar follows status, cookies do not.** `resolveOauthPartition`
   (`apps/desktop/electron/oauth-partition.ts`, upstream #92183) gives a non-primary
   `remote` + `oauth` registry entry its own jar and everything else — the primary, the
   v1 remote, and any URL the registry does not know — the legacy shared jar. The
   Connections editor signs in *before* Save (`connections-registry.tsx`), so the login
   lands in the legacy jar and every read after Save goes to the entry's empty jar.
   "Make primary" and applying from the connection dialog swap jars the same way.
   Password-only dashboards always take the embedded cookie flow
   (`resolveLoginStrategy` / `oauthGuardMayHardFail`), so every lab gateway hits it.
2. **The self-signed opt-in never reached the v2 registry.** Every normalizer in
   `connection-registry.ts` rebuilt entries from a fixed field list and dropped
   `allowInvalidCertificate`; per-connection jars had no certificate verify proc;
   `hostAllowsInvalidCertificate` read only v1 state.
3. **Alert 112 read as a certificate problem** in the onboarding / Settings probe.

## Work items

- [x] `electron/oauth-partition-moves.ts` — pure planner: diff two registry snapshots,
      URLs still registered after the mutation only (a removed gateway's session must
      never land in the shared jar), legacy-source moves first.
- [x] `electron/oauth-cookie-move.ts` — move `hermes_session*` cookies between
      partitions against the gateway's own origin; host-only cookies re-set without a
      domain (`__Host-` requires it); skip a destination that is already signed in.
- [x] `main.ts` wiring: `migrateOauthCookiesForRegistryChange` in
      `saveRegistryConnection`, `connections:set-primary`, and inside the `apply`
      callback of `connection-config:apply` (after the writes, before the re-home dial,
      reversed if activation throws); `warmOauthCookieStore` waits for an in-flight move.
- [x] `allowInvalidCertificate` threaded through the registry (type, input, normalize,
      merge, dial-fields-changed, normalizeRegistry, migrateV1ToRegistry,
      reconcileAppliedGlobalConnection), `hostAllowsInvalidCertificate` reads the
      registry, `gatewayCertificateVerifyProc` installed on per-connection jars,
      `connections:test` skips its Node WebSocket leg under the opt-in, editor toggle.
- [x] `electron/gateway-probe-errors.ts` — plain-language text for alert 112.
- [x] Tests: `oauth-partition-moves.test.ts`, `oauth-cookie-move.test.ts`,
      `gateway-probe-errors.test.ts`, contract case in `oauth-partition.test.ts`, the
      fork block in `connection-registry.test.ts`, two cases in
      `connections-registry.test.tsx`.
- [x] Proven in real Chromium (Electron 40.10.6 / Chrome 144, Xvfb in the devcontainer):
      the real move code carried a `__Host-` cookie and a plain-HTTP-shaped cookie into a
      per-connection jar with attributes intact, and both survived a relaunch; the real
      alert-112 string was reproduced (Electron's BoringSSL and plain Node's OpenSSL) and
      mapped.
- [x] Docs: patch inventory (two entries), `docs/site/fork/forgeguard-changes.md`,
      `website/docs/user-guide/desktop.md` "macOS: start completely fresh".
- [x] Version `0.21.1`: `pyproject.toml`, `hermes_cli/__init__.py`, `uv.lock`,
      `apps/desktop/package.json`, `package-lock.json`, `docs/site/fork/compatibility.md`.
- [ ] Operator: merge the PR into `main` (release-on-merge cuts `v0.21.1`), install the
      macOS build, run the manual two-gateway pass in the canonical plan.

## Validation

- `npx vitest run --project electron` — 2036 passed; 1 failure in
  `remote-lifecycle.test.ts` (`pidIsOurDashboard … installer wrapper`) that fails
  identically on the untouched tree (it symlinks `command -v python3`, a mise shim here).
- `npx vitest run --project ui` — 691 files, 6842 passed.
- `npm run typecheck` and `eslint` over every changed file — clean.

## Known limitation (unchanged, upstream)

Two gateways on one host where one of them is the primary still share the legacy jar by
host — Chromium cookie jars ignore the port. Pre-existing #92183 behaviour.
