# IX Agency — Employee Lifecycle Guide

**Onboarding · day-2 operations · offboarding · security best practices**

Audience: IT / admins provisioning access, and new employees setting up the IX
Agency desktop app. Everything here was verified end-to-end against the live
systems and code on **2026-07-09** (app **v0.17.2**). Where a step has a known
gap or friction point, it is called out inline rather than glossed over — see
[Known gaps & action items](#known-gaps--action-items).

> Scope note: "IX Agency" is the Intelliverse-X branded build of the Hermes
> desktop app. It bundles the full Hermes coding agent **plus** the company
> surfaces: native OTP portal login, the admin-mcp gateway, the copilot tab,
> the skills catalog, and the company VPN.

---

## Table of contents

1. [Download & install (verified links)](#1-download--install)
2. [Workflow: New employee](#2-workflow-new-employee)
   - [2a. Admin: provision access](#2a-admin-provision-access)
   - [2b. Employee: first-run setup](#2b-employee-first-run-setup)
3. [Workflow: Existing employee (day-2)](#3-workflow-existing-employee-day-2)
4. [Workflow: Employee leaving (offboarding)](#4-workflow-employee-leaving-offboarding)
5. [Security best-practices checklist](#5-security-best-practices-checklist)
6. [Known gaps & action items](#6-known-gaps--action-items)

---

## 1. Download & install

The app auto-updates from an S3 feed and is published by CI on every version
bump merged to `main` (see [`RELEASING.md`](../RELEASING.md)). The current
release is **v0.17.2**.

**Official download feed:** `https://intelliverse-x-desktop.s3.amazonaws.com/ix-agency`

| Platform | Artifact | Notes |
|---|---|---|
| macOS (Apple Silicon) | `IX-Agency-0.17.2-mac-arm64.dmg` | Drag to Applications. Unsigned today → Gatekeeper: right-click → Open the first time (see gap G5). |
| Windows (x64) | `IX-Agency-0.17.2-win-x64.exe` | NSIS installer. No MSI is published. |
| Linux | `IX-Agency-0.17.2-linux-x86_64.AppImage` (also `.deb`, `.rpm`) | AppImage supports in-place auto-update; deb/rpm do not. |

Direct channel manifests (what the updater reads): `latest-mac.yml`,
`latest.yml` (Windows), `latest-linux.yml` under the feed base.

> The old `.../latest.json` and `hermes-desktop-updates.s3.amazonaws.com` URLs
> are **dead** and some older docs still mention them. The app self-heals
> (it normalizes those to the feed above), but ignore them — use the feed base.

After install, the app polls for updates on launch and every 4 hours and shows
a non-blocking **Update available** button. In-place update works on Windows,
Linux AppImage, and signed macOS builds; on today's unsigned mac build it opens
the new `.dmg` instead.

---

## 2. Workflow: New employee

Two actors: an **admin** grants access in the portal; the **employee**
installs, signs in, and everything else provisions itself (zero-touch — the
four credentials that used to be hand-delivered now auto-fill on login via
`/api/portal/desktop/provision`).

### 2a. Admin: provision access

**Step 1 — Create the access grant** at `https://admin.intelli-verse-x.ai/admin/team`.

1. Sign in with OTP. The team-management form is available only to a super
   admin or a grant with `privilege = owner`.
2. Use **Invite someone** and fill:
   - **Subject** — the employee's exact email (or a whole domain).
   - **Apps** — pick from the app chips, or "All apps".
   - **Privilege** — `owner` / `manager` / `viewer`.
   - **Bundles** (advanced) — `revenue, growth, support, content, analytics, engineering`.
3. Submit. This creates the grant via `/api/portal/team/grants`, enforced
   server-side in Admin-Management (you can only grant apps/tiers/bundles at or
   below your own).

> ⚠️ **Nothing is emailed automatically.** The page shows a copyable invite
> note — you must send it to the employee yourself. (See gap G7.) That's it:
> the credentials themselves are no longer hand-delivered.

**Step 2 — nothing.** The four credentials that used to be admin-supplied are
now provisioned server-side by `/api/portal/desktop/provision` on the
employee's first sign-in:

| Credential | Provisioned as | Fallback / notes |
|---|---|---|
| **Gateway token** (`ADMIN_MCP_TOKEN`) | Shared cluster token from the portal pod env (`admin-chat-secrets`, overridable via `DESKTOP_GATEWAY_TOKEN`) | Still a shared secret (gap G1) — but no longer pasted around by hand |
| **LiteLLM API key** | **Personal** budget-capped virtual key (alias `sk-desktop-<email>`, team `desktop`, $15/day) minted through the LiteLLM admin API | Shared key (`DESKTOP_LITELLM_SHARED_KEY`) when the master key isn't wired or minting fails |
| **WireGuard `usa-vpn.conf`** | **Personal** wg-easy peer (client name = employee email, idempotent) minted through the wg-easy REST API | Shared conf (`DESKTOP_WG_SHARED_CONF_B64`) when the VPN box API is unreachable |
| **Cognito S2S client secret** | Shared `Desktop-App` client secret from the portal pod env (`DESKTOP_COGNITO_S2S_CLIENT_SECRET`) | Shared client — rotation still affects everyone (gap O6) |

Server prerequisites (one-time): the `desktop-provision-secrets` k8s secret on
`deploy/intelliverse-web-frontend` (see
`intelli-verse-kube-infra/intelli-verse-web-frontend/desktop-provision-secrets.template.yaml`)
and the Lightsail firewall rule allowing the EKS NAT egress IP on the wg-easy
API port. Manual delivery of any of the four values still works as an override
(Settings → Connect) — auto-provision never overwrites a value that is set.

> ⚠️ **Important reality check (gap G1):** the gateway token is currently a
> **single shared cluster secret**, *not* a per-employee scoped token, despite
> what some installer docs claim. Auto-provisioning delivers it without a
> human in the loop, but revoking one person still means rotating it for
> everyone (see offboarding O3). Per-user tokens are an open action item.

### 2b. Employee: first-run setup

1. **Install** the app for your platform ([section 1](#1-download--install)).
2. **Sign in (OTP)** — enter your email; a 6-digit code is emailed from
   `support@intelli-verse-x.ai` (valid 5 minutes). Enter it. The session lasts
   **12 hours**.
3. **Everything else is automatic.** On that first successful sign-in the app:
   - fetches your credentials from the portal (gateway token, personal LiteLLM
     key, personal `usa-vpn.conf`, Cognito secret) and stores them encrypted
     with the OS keychain (`safeStorage`) in `ix-agency.json` — never plaintext;
   - imports the VPN profile into the keychain and **auto-connects the
     tunnel** — expect a macOS admin-password prompt (or Windows UAC);
   - runs the first-run **Hermes init** (validates the Cognito secret with a
     real token grant + JWKS check, writes `~/.hermes/config.yaml`,
     materializes the portal skills catalog, writes `~/.hermes/.env` 0600);
   - auto-syncs the MCP directory, connectors, and org skills.
   Only values you haven't set yourself are filled — manual entries in
   Settings → Connect always win, and re-login re-fills only what's missing.
4. **One real prerequisite remains:** the VPN backend must be installed —
   macOS `brew install wireguard-tools`; Windows install "WireGuard for
   Windows"; Linux add a passwordless sudoers rule for `wg-quick` (see gap G6).

You're set: the copilot tab, MCP directory, skills catalog, VPN, and the local
Hermes coding agent are all live after one email code.

---

## 3. Workflow: Existing employee (day-2)

**Updates.** The app checks on launch and every 4h and shows a non-blocking
**Update available** button; nothing is forced. Click it to install in place
(Windows / AppImage / signed mac). Don't sit on updates indefinitely — the
desktop calls portal and gateway APIs with no version handshake, so a very
stale client can break silently if a server contract changes.

**Session expiry.** The 12-hour OTP session expires silently. When it does, the
**copilot tab and MCP directory lock** (they re-probe the portal every ~15s);
just sign in again. The **VPN is independent** of the portal session — the
tunnel stays up even after the session expires, so "VPN green but chat locked"
means: re-run OTP login.

**Credential rotation.** Rotated values re-provision themselves for slots the
app filled automatically ONLY after the local value is cleared (auto-provision
never overwrites a stored value) — or paste the new value in Settings →
Connect → Save. Storage locations (all secrets keychain-encrypted in
`ix-agency.json`):

| Credential | After Save, also… |
|---|---|
| LiteLLM key | **Re-run Initialize Hermes** so `~/.hermes/.env` is rewritten — otherwise the local Hermes CLI keeps the stale key. |
| Gateway token | Same — re-run Initialize to refresh `.env`. |
| VPN conf | Re-import; reconnect the tunnel. |
| Cognito secret | Re-validate (Initialize re-checks it). |

**Skills & MCP day-2.** Creating a skill is fully local
(`~/.hermes/skills/ix-user/<slug>/SKILL.md`); publishing pushes it to the
portal catalog. Both need only your OTP session — no extra credentials.

---

## 4. Workflow: Employee leaving (offboarding)

Run through **every** row — each is a separate credential the person may still
hold. Levers marked ✅ work cleanly; ⚠️/🔴 rows have gaps you must work around
(details in [section 6](#6-known-gaps--action-items)).

| # | Access artifact | Revocation lever | Status |
|---|---|---|---|
| O1 | **Portal access grant** | `/admin/team` → delete the grant (`DELETE /api/portal/team/grants/[id]`). | ✅ Works; takes effect at their **next** login. |
| O2 | **Live OTP session** (12h cookie already on their machine) | No per-user logout endpoint exists. | 🔴 Deleting the grant does **not** kill a live session for up to 12h. To force-invalidate everyone, rotate `OTP_SECRET`. Prefer: wipe their machine (O7) + keep the 12h TTL in mind. |
| O3 | **Gateway token** (`ADMIN_MCP_TOKEN`) | Rotate the K8s secret `admin-mcp-token` + redeploy. | 🔴 Shared token — rotating locks out **everyone** until they re-fetch. No per-user revocation today. |
| O4 | **LiteLLM key** | Delete their personal key: `POST /key/delete {"key_aliases":["sk-desktop-<email-slug>"]}` with the master key. | ✅ Per-employee (auto-provisioned, team `desktop`, budget-capped). Only sessions provisioned before the per-user rollout hold a shared key. |
| O5 | **WireGuard VPN peer** | Delete their client (name = their email) in the wg-easy admin UI. | ✅ Per-employee peer (auto-minted on first provision) — deletion kills their tunnel immediately. |
| O6 | **Cognito S2S secret** | Rotate the app-client secret. | 🟠 Shared client — rotation affects all who share it. |
| O7 | **Local machine residue** | Wipe (see list). | ✅ Well-defined. |
| O8 | **gh / git / kubeconfig** (if they used coding loops) | Revoke GitHub PATs/SSH keys, remove kube context, rotate any cluster creds they held. | ⚠️ Outside the app — IT must handle. |

**O7 — machine wipe checklist** (returned or personal device):
- `<userData>/ix-agency.json` (keychain-encrypted gateway token, LiteLLM key, VPN conf, Cognito secret)
- `<userData>/wg/` (VPN scratch dir) and any imported `usa-vpn.conf`
- `<userData>/ix-agency-chats.json`
- the **`persist:ix-agency-portal`** session partition (holds the live OTP cookie — O2)
- `~/.hermes/.env` (0600), `~/.hermes/config.yaml`, `~/.hermes/skills/ix-user/` + `ix-portal/`
- macOS: purge the **Keychain** entries backing `safeStorage`
- their AWS keys in `~/.aws/credentials`, GitHub creds, and `~/.kube/config`

> On macOS `<userData>` = `~/Library/Application Support/IX Agency/`.

**Fastest practical containment today**, given O2/O3: delete the portal grant
(O1) + delete the VPN peer (O5) + wipe the machine (O7) immediately; schedule
the shared-secret rotations (O3/O4/O6) for a maintenance window; and be aware
the departed session (O2) is valid for up to 12h unless you rotate `OTP_SECRET`.

---

## 5. Security best-practices checklist

Status reflects the **verified** state as of 2026-07-09. ✅ = already handled,
⚠️ = needs action (tracked in [section 6](#6-known-gaps--action-items)).

| # | Practice | Status |
|---|---|---|
| 1 | Desktop secrets use the OS keychain (`safeStorage`) with **no plaintext fallback** (the app refuses to store secrets if the keychain is unavailable) | ✅ |
| 2 | `~/.hermes/.env` is written mode `0600`; VPN conf lives in the keychain, only materialized to a 0600 scratch file per `wg-quick` call | ✅ |
| 3 | AI copilot **cannot self-approve writes** — write actions require an out-of-band Confirm the model has no channel to trigger; approval tokens are single-use, arg-bound, 10-min expiry | ✅ |
| 4 | Portal login is a real emailed second factor (6-digit HMAC challenge, 5-min TTL, timing-safe compare, httpOnly 12h session) | ✅ |
| 5 | Release bucket is **not** publicly listable; objects encrypted at rest; CI verifies the feed (artifact reachability, size, version agreement) | ✅ |
| 6 | Every IX IPC handler gates on `requireIxPortalAuth()`; skills publish uses the signed-in session, not a static key | ✅ |
| 7 | Human/laptop AWS access is short-lived SSO/STS, not long-lived static keys | ⚠️ (G-C1) |
| 8 | Release-bucket **write** is a CI-only scoped role; no laptop identity can publish | ⚠️ (G-C1/C2) |
| 9 | Windows installers are code-signed; the updater enforces the publisher signature | ⚠️ (G-C2) |
| 10 | macOS builds are Developer-ID signed + notarized as a **hard** CI gate | ⚠️ (G5/G-C2) |
| 11 | Update integrity anchored to a key the bucket owner can't forge (detached signature / pinned public key) | ⚠️ (G-C2) |
| 12 | Release bucket denies non-TLS; account public-access guardrails on | ⚠️ (G-M1) |
| 13 | Gateway/MCP auth is per-user, expiring, individually revocable | ⚠️ (G1/G-H1) |
| 14 | Multi-tenant scope header is mandatory (no unscoped → super-admin fallback) | ⚠️ (G2/G-H1) |
| 15 | Offboarding can revoke **one** person server-side across all credential classes | ⚠️ (O2/O3 gaps) |

---

## 6. Known gaps & action items

Ordered by severity. These are honest findings from emulating the flows against
live systems — fix the Critical two before onboarding a large batch of hires.

### 🔴 Critical

- **C1 — Laptop IAM key is effectively account/cluster admin.** The local
  `s3-user` identity has `AmazonS3FullAccess`, `secretsmanager:*`, cluster-admin
  via EKS access entries, `iam:PassRole`, ECR push, and cost/budget write, on
  long-lived static keys. **Fix:** split into scoped roles (the desktop only
  needs `s3:GetObject/PutObject` on the release prefix), move human access to
  short-lived SSO/STS, delete the static keys, enforce 90-day rotation. This is
  the single highest-leverage fix — it also closes C2's write vector and shrinks
  the offboarding blast radius.
- **C2 — Unsigned Windows auto-update, integrity = a SHA-512 in a bucket C1 can
  overwrite.** Windows builds are unsigned; the updater trusts a `sha512` in
  `latest.yml` served from the same public, laptop-writable bucket. A leaked
  laptop key → silent fleet-wide RCE. **Fix:** sign Windows builds (Azure
  Trusted Signing/EV) and set `publisherName`; make mac notarization a hard CI
  gate; publish a detached signature over the feed verified with a pinned key
  baked into the app; scope bucket write to CI only.

### 🟠 High

- **G1 / H1 — Gateway token is a single shared, non-expiring, non-revocable
  super-admin secret** (docs falsely call it per-email). Bare-token calls get
  SUPER scope while `ADMIN_MCP_ALLOW_UNSCOPED=true`. **Fix:** per-user JWTs with
  `exp` + a revocation denylist; set `ALLOW_UNSCOPED=false`.
- **G2 — Default-open OTP login.** If the grants service is unreachable/log-mode
  and no env allowlist matches, `verify` succeeds *unpinned* (treated as
  super-admin). **Fix:** fail closed when the grants service can't confirm a
  grant.
- **H2 — Cost/budget + financial surface reachable from the laptop key**
  (subsumed by C1).

### 🟡 Medium / Low

- **G-M1 — Release bucket serves plain HTTP and account public-access guardrails
  are relaxed.** Add a `Deny aws:SecureTransport=false` policy; re-enable
  `BlockPublicPolicy`/`RestrictPublicBuckets`.
- **M2 — OTP rate limiting is in-memory per-pod** (multiplied by replica count).
  Move counters to Redis / enforce at the WAF.
- **M3 — `PORTAL_SCOPE_SECRET` falls back to `OTP_SECRET`** (one secret across
  two trust domains). Separate and rotate independently.
- **G3 — Desktop README is stale** on the update URL/behavior (self-heals in
  code, but misleads readers).
- **G4 — Hermes init is hard-blocked on the Cognito secret** even for the plain
  config-write path. Mitigated: the secret now arrives self-serve via login
  auto-provisioning (`/api/portal/desktop/provision`), and init runs
  automatically once it lands; the hard dependency itself remains.
- **G5 — macOS in-place updates require a signed build**; signing/notarization
  secrets aren't set in CI yet, so mac users drag-install every update and hit
  Gatekeeper on first launch.
- **G6 — VPN tooling isn't bundled** (wireguard-tools / WireGuard-for-Windows /
  Linux sudoers rule are prerequisites).
- **G7 — Grant creation notifies nobody** — admins must send the invite note
  manually.
- **G8 — `install-local.sh` assumes a developer machine** and runs
  unconditionally if a `~/dev/hermes-deployment` checkout exists, which can
  abort init on a non-engineer's box if a CLI it expects is missing.
- **G9 — Doc drift on the `latest.json` shape** — the published feed has only
  channel `.yml` files, no `latest.json`.
