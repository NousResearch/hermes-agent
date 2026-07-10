# HT AI Agent — Release & Distribution Runbook

This fork is **publish-ready** under the `HT AI Agent` / `ht-ai-agent` identity:
the package name, image name, Nix package, update channel, and bundle IDs are
all rebranded. What remains to actually *ship* needs credentials, accounts, and
build environments that live outside the repo. This file lists exactly what to
provide and how each channel publishes.

Repo: `github.com/uaixo/awesome-hermes-agent` · PyPI project: `ht-ai-agent` ·
Docker image: `uaixo/ht-ai-agent`.

---

## 1. PyPI (`ht-ai-agent`)

**In-repo (done):** `pyproject.toml` `name = "ht-ai-agent"`, all self-referencing
extras, `[project.urls]` → the fork, `uv.lock` regenerated. The publish workflow
(`.github/workflows/upload_to_pypi.yml`) builds the wheel/sdist and publishes to
`https://pypi.org/p/ht-ai-agent` via the `pypi` GitHub Environment.

**You must provide:**
1. Register the `ht-ai-agent` project on PyPI (first upload, or reserve it).
2. Configure a **PyPI Trusted Publisher** for the project pointing at this repo +
   the `upload_to_pypi.yml` workflow + the `pypi` environment (PyPI → project →
   Publishing → Add GitHub publisher). No API token secret is needed with trusted
   publishing.
3. Publish by pushing a CalVer tag (`scripts/release.py` does this), e.g.
   `v2026.7.9`. The workflow triggers on `v20*` tags.

**Verify before first publish:** `uv build` locally produces
`dist/ht_ai_agent-<version>-py3-none-any.whl` and `ht_ai_agent-<version>.tar.gz`.

---

## 2. Docker (`uaixo/ht-ai-agent`)

**In-repo (done):** `.github/workflows/docker.yml` `IMAGE_NAME: uaixo/ht-ai-agent`,
repo guards set to `uaixo/awesome-hermes-agent`, `docker-compose*.yml` images
build/pull `ht-ai-agent`.

**You must provide** two repository secrets so the publish job can push to Docker
Hub (Settings → Secrets and variables → Actions):
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN` (a Docker Hub access token with write scope for the
  `uaixo/ht-ai-agent` repository).

Publishing happens on pushes to `main` and on releases (per the workflow's `if`
guards). If your Docker Hub namespace differs from `uaixo`, update `IMAGE_NAME`
in `docker.yml` and the `image:` lines in the compose files.

---

## 3. Nix

**In-repo (done):** `flake.nix` description, `nix/hermes-agent.nix` `pname =
"ht-ai-agent"`, `ht` added to the wrapped binaries, `mainProgram = "ht"`, meta
homepage → the fork; `nix/checks.nix` version check runs `ht version`.

**You must verify locally** (Nix is **not** built in CI):
```bash
nix build .#ht-ai-agent
nix flake check          # runs nix/checks.nix
nix run .#ht-ai-agent -- version
```
Not yet rebranded (load-bearing internal paths — deferred): the
`share/hermes-agent/` bundle directory and the `hermes-desktop` / `hermes-tui`
sub-package pnames. These are internal build paths, invisible to users; rename
them only in a coordinated internal-identifier pass (Phase 6).

---

## 4. Install scripts + docs hosting

**In-repo (done):** `scripts/install.sh|ps1|cmd` and `setup-hermes.sh` clone
from `github.com/uaixo/awesome-hermes-agent` and install the `ht` command.

**You must provide** (optional, for a one-line install UX):
- Host `install.sh` / `install.ps1` at a domain you control if you want a
  `curl … | bash` one-liner (the README currently instructs clone-then-run,
  which needs no hosting).
- The Docusaurus site (`website/`) deploys via `deploy-site.yml`, which is
  guarded to the upstream repo and self-disabled here. To host docs, point
  `docusaurus.config.ts` `url`/`baseUrl` at your Pages/host and enable a deploy
  workflow.

---

## 5. Desktop & installer apps — remaining work (needs your build + certs)

**In-repo (done):** product/display names, window & dialog copy, i18n, the `◆`
brand mark, and the **bundle identifiers**:
- desktop `appId` + `setAppUserModelId` → `io.github.uaixo.ht-ai-agent`
- bootstrap-installer tauri `identifier` → `io.github.uaixo.ht-ai-agent.setup`

**Deliberately deferred** because they can only be verified against a real
`electron-builder` / `tauri build` (CI builds only the renderer via `vite`,
not the packaged app), and they're entangled with code-signing you must set up
anyway:

1. **Executable / artifact names.** `apps/desktop/package.json` still ships
   `productName` / `executableName` `"Hermes"` and `artifactName
   "Hermes-${version}-…"`; the Tauri installer ships the `Hermes-Setup` binary.
   Renaming these ripples through hardcoded paths in:
   `apps/desktop/electron/main.cjs` (mac self-update swap of `Hermes.app`),
   `desktop-uninstall.cjs`, `test-desktop.mjs`, `scripts/install.sh|ps1`,
   `hermes_cli/main.py` (desktop launch resolver), `gui_uninstall.py`, and the
   bootstrap-installer's `src-tauri/{bootstrap,update}.rs` + `[[bin]]` name.
   **Note:** `productName` also sets Electron `app.name`, which determines the
   `userData` directory (where localStorage/settings live) — renaming it moves
   that directory, so existing installs need a migration or lose local state.
   Do this as one coordinated change and verify with a full packaged build.

2. **Protocol scheme.** Deep links use `hermes://` (`HERMES_PROTOCOL` in
   `main.cjs`, the `protocols` block in `package.json`, and test data in
   `update-relaunch.test.cjs`). Rename to a fork scheme (e.g. `htaiagent://`)
   in lockstep across those three, then verify deep-link launch on each OS.

3. **Code signing.** Shipping installable binaries requires:
   - macOS: an Apple Developer ID certificate + notarization (electron-builder
     `mac.notarize`); the new `io.github.uaixo.ht-ai-agent` bundle id means macOS
     treats it as a fresh app (TCC mic/screen permissions reset — expected).
   - Windows: an Authenticode code-signing certificate (or the app is flagged by
     SmartScreen); the new AppUserModelId means a fresh Start-menu/uninstall
     identity.
   Add the signing secrets to the release workflow and point the desktop
   self-update remote (already `github.com/uaixo/awesome-hermes-agent` in
   `electron/update-remote.cjs`) at your release channel.

4. **Icons.** Replace the placeholder/old brand art (`apps/desktop/assets/
   icon.{icns,ico,png}`, `apps/bootstrap-installer/src-tauri/icons/*`, the pet
   sprites, `website/static/img/logo.png`). The UI currently renders a neutral
   `◆` glyph via `brand-mark.tsx` in place of the removed `nous-girl.jpg`.

---

## Summary: what's blocked on you

| Channel | In-repo status | You provide |
|---|---|---|
| PyPI | ✅ ready | Trusted Publisher for `ht-ai-agent`; push a CalVer tag |
| Docker Hub | ✅ ready | `DOCKERHUB_USERNAME` + `DOCKERHUB_TOKEN` secrets |
| Nix | ✅ ready | Local `nix build` / `nix flake check` verification |
| Install one-liner / docs site | ✅ clone-and-run | (optional) a hosting domain |
| Desktop/installer binaries | ⚠️ identity done | executable-name rename + signing certs + icons + a packaged build |
