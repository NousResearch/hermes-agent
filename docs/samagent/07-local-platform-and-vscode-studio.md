# 07 — SamAgent Local Installed Platform & VS Code Pre-Production Studio

SamAgent is packaged and operated as an **installed local platform** on your machine (desktop application + background daemon + web studio at `http://127.0.0.1:8080`) that synchronizes directly with **VS Code / Cursor / VSCodium** so you never need to use a CLI and can inspect, edit, run, and verify your application locally before deploying to production.

---

## 1. One-Click Local Machine Installation

Run the platform installer once (or click **"Install Local Platform & VS Code Bridge"** inside the platform UI):

- **macOS / Linux:**
  ```bash
  ./scripts/install-samagent-platform.sh
  ```
- **Windows (PowerShell):**
  ```powershell
  .\scripts\install-samagent-platform.ps1
  ```

### What Gets Installed on Your Machine
1. **OS Desktop Application & Background Service (`samagent/platform_installer.py`)**:
   - **macOS**: `~/Applications/SamAgent.app` native application bundle + `~/Library/LaunchAgents/com.samjuniors.samagent.plist` background daemon.
   - **Linux**: `~/.local/share/applications/samagent-platform.desktop` application launcher + `~/.config/systemd/user/samagent-platform.service` user daemon.
   - **Electron Desktop Identity**: Added the `samagent` variant (`HERMES_DESKTOP_VARIANT=samagent` → `SamAgent Platform`) in `apps/desktop/product-identity.cjs`.
2. **Persistent Local Projects Folder (`~/SamAgentProjects/<project-slug>`)**:
   - Every app you build in SamAgent is saved to a real local Git repository on your disk (`~/SamAgentProjects/yoga-studio-local` by default, or any project created in the **Local Workspace Switcher** bar).
3. **Bundled VS Code / Cursor Extension (`integrations/vscode-samagent/`)**:
   - Automatically installed into `~/.vscode/extensions/samjuniors.samagent-vscode-0.1.0` and `~/.cursor/extensions/samjuniors.samagent-vscode-0.1.0`.
   - Provides:
     - **Verify-on-Save (`samagent.verifyOnSave`)**: Saving any file in VS Code automatically triggers SamAgent's L0–L4 + OWASP security verification on your local workspace.
     - **Status Bar Readiness Indicator**: Displays `$(pass-filled) SamAgent: Pre-Prod Ready (L0–L4 PASS)` or `$(error) SamAgent: Pre-Prod Gate Blocked` inside VS Code.
     - **Activity Bar Webview (`samagent.missionControlView`)**: Embeds Mission Control and the Pre-Production Gate directly inside VS Code's sidebar.

---

## 2. Zero-Config `.vscode/` Workspace in Every Generated Project (`samagent/ide_bridge.py`)

Every project built by SamAgent automatically includes:
- `.vscode/tasks.json`:
  - `SamAgent: Run Local Dev Server (Port 3000)` (`python3 app/main.py --serve --port 3000`)
  - `SamAgent: Run Acceptance & OWASP Security Suite (L0–L3)`
  - `SamAgent: Verify Pre-Production Readiness`
- `.vscode/launch.json`:
  - Press **`F5`** in VS Code to launch and step-debug the local development server (`app/main.py --serve --port 3000`) or step through the generated acceptance tests (`.samagent/tests/test_acceptance.py`).
- `.vscode/settings.json` & `project.code-workspace`:
  - Pre-configures `pytest` discovery for `.samagent/tests` and excludes internal `.samagent/worktrees` from search noise.

---

## 3. How to Develop in VS Code & Test Locally Before Production Deployment

1. **Open Your Local Workspace in VS Code**:
   - Click **"Open Workspace in VS Code"** in the top platform bar (or click any file's `vscode:// ↗` link in **Tab 1: VS Code Studio & Pre-Prod Gate**).
2. **Edit Code in VS Code & Inspect Live Git Diffs**:
   - Edit `app/main.py`, `app/static/index.html`, or any file in VS Code.
   - Click **"Sync from VS Code & Verify"** (or simply save in VS Code with the extension active).
   - **Tab 1 (`1. VS Code Studio & Pre-Prod Gate`)** immediately displays your uncommitted `git status` and `git diff` alongside the updated L0–L4 verification results.
3. **Test Multi-Role Flows Locally (`2. Live Local Dev & Multi-Role Sandbox`)**:
   - Switch between **`Role: Visitor (Anon)`**, **`Role: Member Alice (u_member_a)`**, **`Role: Member Bob (u_member_b)`**, and **`Role: Admin (u_admin)`** against your local SQLite database (`app/db/app.sqlite3`):
     - Book a class as Member Alice (`201 Created`), test duplicate booking protection (`409 Conflict`), and verify anonymous visitors are blocked (`401 Unauthorized`).
     - Switch to Member Bob and click **"Inspect as member (u_member_b)"** on Alice's booking to watch IDOR protection block cross-user data access (`403 Forbidden`).
     - Switch to Admin to add a new class (`201 Created`) and watch it appear live in the schedule.
4. **Pre-Production Deployment Gate (`evaluate_pre_production_gate`) & Signed Production Release Bundler (`samagent/prod_bundler.py`)**:
   - Production deployment is blocked unless all 7 pre-production gates are green:
     1. `local_dev_app_ready`
     2. `vscode_workspace_configured`
     3. `l0_l1_syntax_contract`
     4. `l2_ownership_and_tdd_red_green`
     5. `l3_security_owasp_idor_rbac`
     6. `l4_live_browser_dom_smoke`
     7. `no_secret_leaks`
   - Once all 7 checks pass, click **"Promote to Production Release"** (or **"Generate Production Bundle (Dockerfile + Manifest)"**) in **Tab 1: VS Code Studio & Pre-Prod Gate** (`POST /api/plugins/samagent/promote-prod`) to generate:
     - `Dockerfile` (non-root `appuser`, `/healthz` healthcheck, port `8000`)
     - `docker-compose.prod.yml`
     - `.env.example`
     - `.samagent/releases/<release_id>/RELEASE_MANIFEST.json` with SHA-256 checksums of every verified workspace file and the L0–L4 + OWASP security attestation.
   - If a developer edits any file in VS Code and introduces a failing acceptance test, an IDOR/RBAC regression, or a leaked secret (`sk-...`), `promote_to_production()` automatically blocks the release and lists the exact failing gate.

---

## 4. Automatic IDE Disk Watcher, Managed Dev Server & ACP Bridge

1. **Zero-Config External IDE Auto-Watcher (`samagent/ide_watcher.py`)**:
   - Tracks SHA-256 digests of all user-editable files (`app/**`, `.vscode/**`, `.samagent/spec.yaml`).
   - Whenever you save a file in **VS Code, Cursor, Zed, JetBrains, or Neovim**, SamAgent automatically detects the on-disk change, runs L0–L4 + OWASP verification, updates `.samagent/runs/run_ide_autosync_*/deliverable.json`, and records the edit + verification outcome in the bi-temporal Project Ledger (`scope="ide_sync", kind="external_edit"`).
2. **Managed Local Dev Server Controller (`samagent/dev_server_manager.py`)**:
   - Controls the live local development server (`python3 app/main.py --serve --host 0.0.0.0 --port 3000`) directly from the SamAgent Local Platform header (**`DEV SERVER :3000 ONLINE`** pill + **`Restart Dev Server (:3000)`** button).
3. **Agent Client Protocol (ACP) Slash Commands (`samagent/acp_bridge.py` + `acp_adapter/commands.py`)**:
   - Any external IDE connected over ACP (Zed, VS Code ACP, JetBrains ACP) can run `/verify`, `/preprod`, and `/promote` directly inside the IDE agent panel.


