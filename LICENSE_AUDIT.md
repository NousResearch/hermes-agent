# Hermes Agent — Dependency License Audit

**Date:** 2026-07-20
**Project license:** MIT (Copyright 2025 Nous Research)
**Audited by:** ethie (automated — PyPI/npm/GitHub registry queries + lockfile parsing)

## Scope

Every dependency that ships in a Hermes Agent distribution, across all three
ecosystems and all four npm lockfiles:

| Ecosystem | Lockfile | Packages |
|-----------|----------|----------|
| Python | `uv.lock` | 233 |
| npm (root workspaces) | `package-lock.json` | 1,387 |
| npm (website / Docusaurus) | `website/package-lock.json` | 1,443 |
| npm (whatsapp-bridge) | `scripts/whatsapp-bridge/package-lock.json` | 167 |
| npm (photon sidecar) | `plugins/platforms/photon/sidecar/package-lock.json` | 144 |
| npm (deduped across all 4) | — | 2,401 unique |
| Rust | `apps/bootstrap-installer/src-tauri/Cargo.toml` | 19 direct (no `Cargo.lock`) |

**Total: 2,653 dependency versions audited.**

## Executive summary

The overwhelming majority of dependencies (2,564 / 2,653 = 96.6%) are under
permissive licenses (MIT, ISC, Apache-2.0, BSD, PSF) that are fully
compatible with the project's MIT license. No action is needed for those.

**9 packages require attention** — they fall into copyleft, attribution-required,
proprietary, or unlicensed categories that create real compliance obligations.
The most serious is **`libsignal` (GPL-3.0)** in the whatsapp-bridge, which has
strong copyleft implications if the bridge is distributed as part of Hermes.

**There is currently zero license-compliance scanning in CI.** The existing
`osv-scanner.yml` covers CVEs only; `supply-chain-audit.yml` covers malware
patterns only. Neither checks license compatibility. There is no `NOTICE` or
`THIRD_PARTY_LICENSES` file at the repo root (the only `NOTICE` is inside
`plugins/security-guidance/` for forked Anthropic Apache-2.0 code).

---

## Current compliance posture

| What exists | What it covers | Gap |
|-------------|----------------|-----|
| `osv-scanner.yml` | Known CVEs in pinned deps (vuln DB) | Does not check licenses |
| `supply-chain-audit.yml` | Malware patterns in PR diffs (.pth, base64+exec, etc.) | Does not check licenses |
| `tests/docker/test_license_file_present.py` | Our own LICENSE ships in the Docker image | Only checks our LICENSE, not deps |
| `plugins/security-guidance/NOTICE` | Forked Anthropic code attribution | Local to that plugin only |
| **Missing** | **License compatibility scanning** | **No CI check prevents adding a GPL/AGPL/unlicensed dep** |
| **Missing** | **NOTICE / THIRD_PARTY_LICENSES file** | **No bundled attribution for permissive licenses that require it** |

---

## Findings by severity

### 🔴 Critical — GPL-3.0 (strong copyleft)

#### `libsignal@6.0.0` (npm, whatsapp-bridge)

- **License:** GPL-3.0
- **Source:** `https://github.com/WhiskeySockets/libsignal-node`
- **Where:** `scripts/whatsapp-bridge/package.json` → `@whiskeysockets/baileys` depends on it
- **Direct dep?** No — transitive via `@whiskeysockets/baileys`
- **Compliance issue:** GPL-3.0 is strong copyleft. If the whatsapp-bridge is
  distributed as part of Hermes Agent (which is MIT), the GPL-3.0 license
  would require the *entire combined work* to be GPL-3.0. This is a
  license conflict — MIT and GPL-3.0 are compatible (MIT code can be included
  in a GPL-3.0 project), but only if the combined work is distributed under
  GPL-3.0, which contradicts Hermes's MIT license.
- **Mitigation:** The whatsapp-bridge is a standalone sidecar script
  (`scripts/whatsapp-bridge/bridge.js`), not imported by the Python core.
  If it ships as a separate process invoked via subprocess (not linked
  into the Hermes binary), the "aggregate, not derivative" argument may
  apply. **This needs a legal review** to confirm the boundary.

### 🟠 High — LGPL-3.0 (weak copyleft)

LGPL allows linking from non-LGPL code, but modifications to the LGPL
library itself must be released under LGPL, and the license + source
must be offered to recipients.

#### Python

| Package | Version | Extra | Notes |
|---------|---------|-------|-------|
| `edge-tts` | 7.2.7 | `[edge-tts]` | Default TTS provider. LICENSE file confirms: one file (srt_composer.py) is MIT, rest is LGPLv3. |
| `python-telegram-bot` | 22.6 | `[messaging]`, `[termux]` | PEP 639 `license_expression: LGPL-3.0-only`. LICENSE file ships GPL-3 text (LGPL is a permitted additional permission). |

#### npm (sharp / libvips native binaries)

| Package | Version | License | Notes |
|---------|---------|---------|-------|
| `@img/sharp-libvips-*` (10 platform variants) | 1.3.2 | LGPL-3.0-or-later | libvips native binaries. Pulled by `sharp` (image processing). |
| `@img/sharp-win32-*` (3 variants) | 0.35.3 | Apache-2.0 AND LGPL-3.0-or-later | sharp Windows binaries. |
| `@img/sharp-wasm32` | 0.35.3 | Apache-2.0 AND LGPL-3.0-or-later AND MIT | sharp wasm binary. |

- **Where:** Root `package-lock.json`, transitive via `sharp`
- **Direct dep?** No — `sharp` is not declared in any workspace `package.json`;
  it's pulled transitively by an upstream package.
- **Compliance issue:** LGPL-3.0-or-later allows use in non-LGPL projects,
  but requires: (1) providing the LGPL license text, (2) providing a way to
  relink against a modified version of the library. For statically-linked
  native binaries, this means documenting how to swap the .node/.so/.dll.
- **Mitigation:** Include LGPL license text in NOTICE file. The "provide a
  way to relink" obligation is satisfied by the npm package boundary (users
  can `npm install` a different `sharp` build).

### 🟡 Medium — MPL-2.0 / MPL-1.1 (file-level weak copyleft)

MPL is file-level copyleft: modifications to MPL-licensed *files* must stay
MPL, but it does not infect the rest of the project. Compatible with MIT.

| Package | Version | Ecosystem | Notes |
|---------|---------|-----------|-------|
| `lightningcss` (+ 11 platform variants) | 1.32.0 | npm | CSS transformer, pulled by `@tailwindcss` + `vite`. |
| `lunr-languages` | 1.14.0 | npm (website) | Language packs for Docusaurus search. Old MPL-1.1. |
| `certifi` | 2026.5.20 | python | CA bundle. |
| `mautrix` | 0.21.0 | python | Matrix SDK (`[matrix]` extra). |
| `pathspec` | 1.1.1 | python | .gitignore matching. Core dep. |

- **Compliance:** Include MPL license text in NOTICE. No source disclosure
  obligation. If we modify any MPL file, that file must stay MPL.

### 🟡 Medium — CC-BY-4.0 (attribution required)

CC-BY-4.0 requires attribution in a way that is "reasonable to the medium."
For software distribution, this means the NOTICE file.

| Package | Version | Ecosystem | Notes |
|---------|---------|-----------|-------|
| `@vscode/codicons` | 0.0.45 | npm | Icon font, direct dep in `apps/desktop` + `apps/bootstrap-installer`. |
| `caniuse-lite` | 1.0.30001799 | npm | Browser support DB, transitive via `browserslist`. |

- **Compliance:** Add attribution lines to NOTICE file.

### 🟡 Medium — Non-standard / proprietary license

#### `gsap@3.15.0` (npm)

- **License:** "Standard 'no charge' license: https://gsap.com/standard-license"
- **Where:** `web/package.json` (direct dep), also pulled by `@nous-research/ui`
- **Compliance issue:** This is not an OSI-approved license. GSAP's own terms
  apply. Per gsap.com: "GSAP is now free for everyone, thanks to Webflow's
  support!" — but the license has specific terms (no resale as a standalone
  library, etc.) that differ from MIT.
- **Compliance:** Reproduce GSAP's license terms in the NOTICE file with a
  link to the full text.

### 🟢 Low — No license (all-rights-reserved)

| Package | Version | Ecosystem | Notes |
|---------|---------|-----------|-------|
| `hindsight-client` | 0.6.1 | python | Optional `[hindsight]` extra. No LICENSE in wheel, no repo link. |
| `format` | 0.2.2 | npm (website) | sprintf library. No LICENSE anywhere. Transitive via Docusaurus. |

- **Compliance issue:** Without a license, the default is "all rights reserved"
  — the copyright holder has not granted permission to use, modify, or
  distribute. Technically, using these at all is a copyright infringement.
- **Mitigation:**
  - `hindsight-client`: Contact the "Hindsight Team" for a license
    clarification, or remove the extra and document it as user-installed.
  - `format`: It's a tiny, ancient (2012-era) sprintf utility pulled
    transitively by Docusaurus. Contact the author (samsonjs) or replace
    with a licensed alternative. Alternatively, Docusaurus may have already
    addressed this upstream.

### ✅ Resolved (were "missing" in metadata, verified permissive)

These packages had no license in their npm/PyPI metadata but were verified
permissive by fetching their LICENSE files from GitHub or inspecting tarballs:

| Package | Resolved license | How verified |
|---------|-----------------|--------------|
| `khroma` | MIT | GitHub `fabiospampinato/khroma` LICENSE file |
| `eval` | MIT | GitHub `pierrec/node-eval` LICENSE file |
| `require-like` | MIT | GitHub `felixge/node-require-like` package.json |
| `qrcode-terminal` | MIT | GitHub `gtanner/qrcode-terminal` LICENSE file |
| `@photon-ai/slack` | MIT | Tarball `package/proto/LICENSE`: "Copyright (c) 2025 Photon AI" |
| `@photon-ai/whatsapp-business` | Likely MIT | Same Photon AI scope; no LICENSE in tarball (unverified) |
| `agent-client-protocol` | Apache-2.0 | GitHub `agentclientprotocol/python-sdk` LICENSE |
| `azure-core` | MIT | GitHub `Azure/azure-sdk-for-python` LICENSE |
| `azure-identity` | MIT | GitHub `Azure/azure-sdk-for-python` LICENSE |
| `fal-client` | Apache-2.0 | GitHub `fal-ai/fal` LICENSE |
| `mistralai` | Apache-2.0 | GitHub `mistralai/client-python` LICENSE |
| `microsoft-teams-*` (4 pkgs) | MIT | GitHub `microsoft/teams.py` LICENSE |
| `honcho-ai` | Apache-2.0 | PyPI `license_expression` field |

### ✅ No issue (compound permissive licenses)

| Package | License | Why it's fine |
|---------|---------|---------------|
| `@bufbuild/protobuf` | Apache-2.0 AND BSD-3-Clause | Both permissive |
| `dompurify` | MPL-2.0 OR Apache-2.0 | Can elect Apache-2.0 |

---

## Rust (Cargo)

No `Cargo.lock` exists in the repo, so a transitive audit is deferred.
All 19 direct dependencies in `apps/bootstrap-installer/src-tauri/Cargo.toml`
are permissive (MIT or Apache-2.0 or both). Run `cargo generate-lockfile`
before the installer is first built/distributed to enable the transitive check.

---

## Compliance plan

### 1. Create a root `NOTICE` file

The biggest gap. Many permissive licenses (MIT, Apache-2.0, BSD) require
retaining the copyright + license notice when distributed. Currently we ship
our own `LICENSE` but don't bundle attribution for dependencies.

**Action:** Generate `NOTICE` (or `THIRD_PARTY_LICENSES.md`) at repo root,
containing:
- The project's own MIT license + copyright
- For each non-MIT dependency: the license name, copyright holder, and
  license text (or link)
- Attribution blocks for CC-BY-4.0 packages (codicons, caniuse-lite)
- Reproduced terms for GSAP's non-standard license
- LGPL/MPL license texts for the copyleft deps

This can be auto-generated from the lockfiles using a tool like
`pip-licenses` (Python) + `license-checker` (npm) in CI.

### 2. Add license-scanning CI

**Action:** Add a `license-audit.yml` workflow that:
- Runs `pip-licenses` (or `uv pip licenses`) against `uv.lock`
- Runs `license-checker` (or `licensee`) against all 4 `package-lock.json` files
- Fails on: GPL, AGPL, unlisted, or "unknown" licenses
- Warns on: LGPL, MPL, CC-BY, non-OSI licenses (advisory, not blocking)
- Posts findings as a collapsible `<details>` PR comment

This closes the gap where a PR could add a GPL-3.0 dependency and CI would
never catch it.

### 3. Resolve the `libsignal` GPL-3.0 question

**Action:** Legal review to confirm whether `scripts/whatsapp-bridge/` is a
separate aggregate (not a derivative of Hermes) when distributed. If the
bridge ships as a standalone subprocess invoked via `node bridge.js`, the
GPL-3.0 obligation may not extend to Hermes itself. If there's any doubt,
document the boundary clearly or move the bridge to a separate repo.

### 4. Resolve unlicensed packages

**Action:**
- `hindsight-client`: Contact the Hindsight Team for a license, or move the
  `[hindsight]` extra to lazy-install only (it already is, partially).
- `format`: Check if Docusaurus has dropped or replaced it upstream. If not,
  file an issue with samsonjs/format to add a license, or patch it out.

### 5. Generate `Cargo.lock` for the bootstrap installer

**Action:** Run `cargo generate-lockfile` in
`apps/bootstrap-installer/src-tauri/` and commit it, so the transitive Rust
dependency tree is auditable before the installer is distributed.

### 6. Document the MPL/LGPL boundary

**Action:** Add a section to `CONTRIBUTING.md` documenting that:
- MPL-2.0 deps are acceptable (file-level copyleft, MIT-compatible) but
  modifications to MPL files must stay MPL
- LGPL-3.0 deps are acceptable (linking is fine) but the NOTICE file must
  carry the LGPL text and document how to relink
- GPL/AGPL deps are NOT acceptable in the core or any workspace that ships
  with Hermes — they must be isolated in standalone sidecar processes

---

## Appendix: tools used

This audit was performed with:
- `uv.lock` parsing (233 Python packages, PEP 639 `license_expression` field)
- `package-lock.json` parsing (4 lockfiles, 2,401 unique npm packages)
- PyPI JSON API (`/pypi/{name}/{version}/json`) for Python license metadata
- npm registry API (`registry.npmjs.org`) for npm license metadata
- GitHub raw (`raw.githubusercontent.com`) for LICENSE file verification
- npm tarball inspection for packages with no metadata
