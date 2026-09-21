# Changelog

All notable changes to the ClawSec Suite will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.7] - 2026-04-16

### Changed

- Added `.clawhubignore` coverage for `test/` so publish payloads stay focused on runtime assets.
- Refactored setup/install scripts to use aliased child-process calls while preserving behavior.
- Split local file reads into `scripts/local_file_io.mjs` and `hooks/clawsec-advisory-guardian/lib/local_file_io.mjs` so network-facing files keep I/O concerns isolated.

### Security

- Removed static moderation false positives related to mixed file-read/network and child-process token patterns in publish-scoped runtime files.

## [0.1.6] - 2026-04-14

### Added

- Runtime and operator-review metadata covering hook installation, optional cron persistence, guarded install flows, and feed URL overrides.
- Preflight disclosure in `scripts/setup_advisory_hook.mjs` and `scripts/setup_advisory_cron.mjs`.
- Regression coverage for setup disclosure behavior in `test/setup_disclosure.test.mjs`.

### Changed

- Declared `node`, `npx`, `openclaw`, and `unzip` in the suite runtime metadata to match the documented setup and install flows.
- Updated catalog messaging for `openclaw-audit-watchdog` to reflect DM delivery with optional email instead of implying email-only reporting.
- Marked local advisory signature/checksum SBOM entries as optional until those companion artifacts are bundled in the repository.
- Removed legacy pre-OpenClaw naming from the suite catalog compatibility metadata.

### Security

- Hook and cron setup now announce their persistence and approval boundaries before enabling host-side automation.
- Clarified that the suite can recommend removal or block risky installs, but destructive actions remain approval-gated.

## [0.1.5] - 2026-04-08

### Fixed

- Fixed heartbeat update detection to rely on GitHub release metadata for latest-version resolution, addressing false update status results reported in [#168](https://github.com/prompt-security/clawsec/issues/168).
- Hardened fallback behavior when release API auth/config is unavailable so version checks still resolve the correct latest release.

## [0.1.4] - 2026-02-28

### Added

- Advisory output snippets now include exploitability context in suite quick-check and heartbeat examples.

### Changed

- Clarified exploitability guidance to match runtime score values (`high|medium|low|unknown`).
- Prioritization guidance now emphasizes high-exploitability advisories for immediate handling.

### Fixed

- Kept exploitability enrichment in advisory workflows non-fatal per item so a single analysis failure does not abort feed updates.

## [0.1.3]

### Added

- Contributor credit: portability and path-hardening improvements in this release were contributed by [@aldodelgado](https://github.com/aldodelgado) in PR #62.
- Cross-shell path resolution support for home-directory tokens in suite path configuration (`~`, `$HOME`, `${HOME}`, `%USERPROFILE%`, `$env:HOME`).
- Dedicated path-resolution regression coverage (`test/path_resolution.test.mjs`) including fallback behavior for invalid explicit path values.
- Additional advisory/installer tests validating home-token expansion and escaped-token rejection.

### Changed

- Advisory guardian hook now resolves configured path environment variables through a shared portability helper.
- Guarded install flow now resolves feed/signature/checksum/public-key path overrides through the same shared path helper for consistent behavior across shells/OSes.
- Advisory matching now explicitly scopes to `application: "openclaw"` when present; legacy advisories without `application` remain eligible for backward compatibility.

### Fixed

- Prevented advisory-check bypass when a single explicit path env var is malformed: invalid explicit values now fall back to safe defaults instead of aborting the entire hook run.

### Security

- Escaped/unexpanded home-token inputs in path config are explicitly rejected while preserving secure defaults.

## [0.1.2]

### Added

- Advisory suppression module (`hooks/clawsec-advisory-guardian/lib/suppression.mjs`).
- `loadAdvisorySuppression()` -- loads suppression config with `enabledFor: ["advisory"]` sentinel gate.
- `isAdvisorySuppressed()` -- matches `advisory.id === rule.checkId` + case-insensitive skill name.
- Advisory guardian handler integration: partitions matches into active/suppressed after `findMatches()`.
- Suppressed matches tracked in state file (prevents re-evaluation) but not alerted.
- Soft notification message for suppressed matches count.
- Advisory suppression tests (13 tests in `advisory_suppression.test.mjs`).
- Documentation in SKILL.md for advisory suppression/allowlist mechanism.

### Changed

- Advisory guardian handler (`handler.ts`) now loads suppression config and filters matches before alerting.

### Security

- Advisory suppression gated by config file sentinel (`enabledFor: ["advisory"]`) -- no CLI flag needed but config must explicitly opt in.
- Suppressed matches are still tracked in state to maintain audit trail.

## [0.1.1] - 2026-02-16

### Added
- Added `scripts/discover_skill_catalog.mjs` to dynamically discover installable skills from `https://clawsec.prompt.security/skills/index.json`.
- Added `test/skill_catalog_discovery.test.mjs` to validate remote-catalog loading and fallback behavior.
- Added CI signing-key drift guard script: `scripts/ci/verify_signing_key_consistency.sh`.

### Changed
- Updated `SKILL.md` to use dynamic catalog discovery commands instead of hard-coded optional-skill names.
- Updated advisory feed defaults to signed-host URL (`https://clawsec.prompt.security/advisories/feed.json`).
- Improved checksum manifest key compatibility in feed verification logic (supports basename and `advisories/*` key formats).
- Kept `openclaw-audit-watchdog` as a standalone skill (not embedded in `clawsec-suite`).

### Security
- **Signing key drift control**: CI now enforces that all public key references (inline SKILL.md PEM, canonical `.pem` files, workflow-generated keys) resolve to the same fingerprint. Prevents stale, fabricated, or rotated-but-not-propagated key material from reaching releases.
  - Enforced in: `.github/workflows/skill-release.yml`, `.github/workflows/deploy-pages.yml`
  - Guard script: `scripts/ci/verify_signing_key_consistency.sh`

### Fixed
- **Fixed fabricated signing key in SKILL.md**: The manual installation script contained a hallucinated Ed25519 public key and fingerprint (`35866e1b...`) that never corresponded to the actual release signing key. Replaced with the real public key derived from the GitHub-secret-held private key. The bogus key was introduced in v0.0.10 (`Integration/signing work #20`) and went undetected because no consistency check existed at the time.
- Corrected `checksums.sig` naming in release verification documentation.

## [0.0.10] - 2026-02-11

### Security

#### Transport Security Hardening
- **TLS Version Enforcement**: Eliminated support for TLS 1.0 and TLS 1.1, enforcing minimum TLS 1.2 for all HTTPS connections
- **Certificate Validation**: Enabled strict certificate validation (`rejectUnauthorized: true`) to prevent MITM attacks
- **Domain Allowlist**: Restricted advisory feed connections to approved domains only:
  - `clawsec.prompt.security` (official ClawSec feed host)
  - `prompt.security` (parent domain)
  - `raw.githubusercontent.com` (GitHub raw content)
  - `github.com` (GitHub releases)
- **Strong Cipher Suites**: Configured modern cipher suites (AES-GCM, ChaCha20-Poly1305) for secure connections

#### Signature Verification & Checksum Validation
- **Fixed unverified file publication**: Refactored `deploy-pages.yml` workflow to download release assets to temporary directory before signature verification, ensuring unverified files never reach public directory
- **Fixed schema mismatch**: Updated `deploy-pages.yml` to generate `checksums.json` with proper `schema_version` and `algorithm` fields that match parser expectations
- **Fixed missing checksums abort**: Updated `loadRemoteFeed` to gracefully skip checksum verification when `checksums.json` is missing (e.g., GitHub raw content), while still enforcing fail-closed signature verification
- **Fixed parser strictness**: Enhanced `parseChecksumsManifest` to accept legacy manifest formats through a fallback chain:
  1. `schema_version` (new standard)
  2. `version` (skill-release.yml format)
  3. `generated_at` (old deploy-pages.yml format)
  4. `"1"` (ultimate fallback)

### Changed
- Advisory feed loader now uses `secureFetch` wrapper with TLS 1.2+ enforcement and domain validation
- Checksum verification is now graceful: feeds load successfully from sources without checksums (e.g., GitHub raw) while maintaining fail-closed signature verification
- Workflow release mirroring flow changed from `download → verify → skip` to `download to temp → verify → mirror` (fail = delete temp)

### Fixed
- Unverified skill releases no longer published to public directory on signature verification failure
- Schema mismatch between generated and expected checksums manifest fields
- Feed loading failures when checksums.json missing from upstream sources
- Parser rejection of valid legacy manifest formats

### Security Impact
- **Fail-closed security maintained**: All feed signatures still verified; invalid signatures reject feed loading
- **No backward compatibility break**: Legacy manifests continue working through fallback chain
- **Enhanced transport security**: Connections protected against downgrade attacks and MITM
- **Defense in depth**: Multiple layers of verification (domain, TLS, certificate, signature, checksum)

---

## Release Notes Template

When creating a new release, copy this template to the GitHub release notes:

```markdown
## Security Improvements

### Transport Security
✅ TLS 1.2+ enforcement (eliminated TLS 1.0, 1.1)
✅ Strict certificate validation
✅ Domain allowlist (prompt.security, github.com only)
✅ Modern cipher suites (AES-GCM, ChaCha20-Poly1305)

### Signature & Checksum Verification
✅ Unverified files never published (temp directory workflow)
✅ Proper schema fields in generated checksums.json
✅ Graceful fallback when checksums missing (GitHub raw)
✅ Legacy manifest format support (backward compatible)

### Testing
All verification tests passed:
- ✅ Unit tests: 14/14 passed
- ✅ Parser lenience: 3/3 legacy formats accepted
- ✅ Remote loading: Gracefully handles missing checksums
- ✅ Workflow security: Temp directory prevents unverified publication
```
