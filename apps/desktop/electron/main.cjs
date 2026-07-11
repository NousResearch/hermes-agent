const {
  app,
  BrowserWindow,
  Menu,
  Notification,
  clipboard,
  dialog,
  ipcMain,
  nativeImage,
  nativeTheme,
  net: electronNet,
  powerMonitor,
  protocol,
  safeStorage,
  session,
  shell,
  systemPreferences
} = require('electron')
const crypto = require('node:crypto')
const fs = require('node:fs')
const http = require('node:http')
const https = require('node:https')
const net = require('node:net')
const path = require('node:path')
const { pathToFileURL } = require('node:url')
const { execFileSync, spawn } = require('node:child_process')
const { detectRemoteDisplay, isWindowsBinaryPathInWsl, isWslEnvironment } = require('./bootstrap-platform.cjs')
const { runBootstrap } = require('./bootstrap-runner.cjs')
const {
  resolveLatestRuntimePin,
  checkForRuntimeUpdate,
  overlayStampWithPin
} = require('./apex-runtime-latest.cjs')
const {
  canUseOnDiskRuntime,
  resolvePreBootstrapDecision,
  resolveBootstrapFailureFallback
} = require('./apex-runtime-select.cjs')
// hc-472 P1 · versioned runtime bundle (opt-in, behind HERMES_BUNDLE_MODE).
// main talks to the layout only through the migrate + diskspace facades (which
// wrap reconcile/GC with the D1 migration + C2 watermark), so it does not
// require apex-bundle-layout directly.
const bundleMigrate = require('./apex-bundle-migrate.cjs')
const bundleDiskspace = require('./apex-bundle-diskspace.cjs')
const { downloadWithResume } = require('./apex-bundle-download.cjs')
const { applyBundleUpdate: applyRuntimeBundleUpdate } = require('./apex-bundle-install.cjs')
const { createShellUpdater } = require('./shell-updater.cjs')
const {
  applyConfigYamlKeys,
  fetchClientConfig,
  normalizeStoredClientConfig,
  shouldApply: shouldApplyClientConfig
} = require('./apex-client-config.cjs')
const {
  buildSessionWindowUrl,
  chatWindowWebPreferences,
  createSessionWindowRegistry,
  SESSION_WINDOW_MIN_HEIGHT,
  SESSION_WINDOW_MIN_WIDTH
} = require('./session-windows.cjs')
const { canImportHermesCli, verifyHermesCli } = require('./backend-probes.cjs')
const { probeGatewayWebSocket } = require('./gateway-ws-probe.cjs')
const { adoptServedDashboardToken } = require('./dashboard-token.cjs')
const { waitForDashboardPort } = require('./backend-ready.cjs')
const { serializeJsonBody, setJsonRequestHeaders } = require('./oauth-net-request.cjs')
const { fetchMarketplaceThemes, searchMarketplaceThemes } = require('./vscode-marketplace.cjs')
const { buildDesktopBackendEnv, normalizeHermesHomeRoot } = require('./backend-env.cjs')
const { readWindowsUserEnvVar } = require('./windows-user-env.cjs')
const { readDirForIpc } = require('./fs-read-dir.cjs')
const { gitRootForIpc } = require('./git-root.cjs')
const { worktreesForIpc } = require('./git-worktrees.cjs')
const { OFFICIAL_REPO_HTTPS_URL, isOfficialSshRemote } = require('./update-remote.cjs')
const { runRebuildWithRetry } = require('./update-rebuild.cjs')
const {
  buildPosixCleanupScript,
  buildWindowsCleanupScript,
  modeRemovesAgent,
  modeRemovesUserData,
  resolveRemovableAppPath,
  shouldRemoveAppBundle,
  uninstallArgsForMode
} = require('./desktop-uninstall.cjs')
const { isPackagedInstallPath: isPackagedInstallPathUnderRoots } = require('./workspace-cwd.cjs')
const {
  authModeFromStatus,
  buildGatewayWsUrl,
  buildGatewayWsUrlWithTicket,
  connectionScopeKey,
  cookiesHaveSession,
  cookiesHaveLiveSession,
  normAuthMode,
  normalizeRemoteBaseUrl,
  pathWithGlobalRemoteProfile,
  profileRemoteOverride,
  resolveAuthMode,
  resolveTestWsUrl,
  tokenPreview
} = require('./connection-config.cjs')
const {
  accessTokenFromLogin,
  accountFromLogin,
  apexWebLoginUrl,
  buildManagedModelConfig,
  defaultModelPath,
  googleStartUrl,
  isManagedEnabled,
  isRelayUnauthorized,
  managedModelConfigYaml,
  ensurePluginsEnabledYaml,
  ensureSkillsDisabledYaml,
  modelDisabledProvidersYaml,
  seedSkillsBlockYaml,
  seedPluginsBlockYaml,
  MANAGED_PROVIDER_NAME,
  MODEL_DISABLED_PROVIDERS,
  parseProvisionResponse,
  resolveApexEndpoints,
  shouldAttemptReprovision,
  syncCustomProviderKeyYaml
} = require('./apex-managed.cjs')
const {
  buildFeishuBackendEnv,
  feishuCredentialsUrl,
  normalizeStoredFeishu,
  parseFeishuCredentialsResponse
} = require('./apex-feishu.cjs')
const { startLoopbackLogin } = require('./apex-loopback.cjs')
const {
  DATA_URL_READ_MAX_BYTES,
  DEFAULT_FETCH_TIMEOUT_MS,
  TEXT_PREVIEW_SOURCE_MAX_BYTES,
  encryptDesktopSecret: encryptDesktopSecretStrict,
  resolveReadableFileForIpc,
  resolveRequestedPathForIpc,
  resolveTimeoutMs
} = require('./hardening.cjs')

let nodePty = null
let nodePtyDir = null

try {
  nodePty = require('node-pty')
  nodePtyDir = path.dirname(require.resolve('node-pty/package.json'))
} catch {
  // Packaged builds set `files:` in package.json, which excludes node_modules
  // from the asar.  Workspace dedup also hoists this native dep to the repo
  // root's node_modules, out of reach of electron-builder's collector.  We
  // ship a minimal copy under resources/native-deps/ via extraResources +
  // scripts/stage-native-deps.cjs; resolve from there when the normal
  // require() fails.  Dev mode never reaches this branch -- the hoisted
  // resolve succeeds via Node's normal module lookup.
  try {
    const path = require('node:path')
    const resourcesPath = process.resourcesPath
    if (resourcesPath) {
      nodePtyDir = path.join(resourcesPath, 'native-deps', 'node-pty')
      nodePty = require(nodePtyDir)
    }
  } catch {
    console.log(`[terminal] failed to load node-pty from path ${nodePtyDir}`)
    nodePty = null
    nodePtyDir = null
  }
}

// Data continuity across the APEX brand rename: Electron derives the DEFAULT
// userData dir from productName, so renaming "ApexNodes" → "APEX" would move
// it to …/APEX and abandon every existing install's state (connection.json,
// updates.json and apex-managed.json — the managed-LLM login). Pin userData to
// the historical ApexNodes directory BEFORE any app.getPath('userData') use
// below. Verified on Electron 40: this single setPath also re-points
// sessionData (cookies/localStorage), so remote-gateway sessions survive too.
// HERMES_DESKTOP_USER_DATA_DIR still wins when set (tests / sandboxed runs).
const { resolveUserDataDir } = require('./user-data-dir.cjs')
const USER_DATA_OVERRIDE = process.env.HERMES_DESKTOP_USER_DATA_DIR
const RESOLVED_USER_DATA_DIR = resolveUserDataDir(app.getPath('appData'), USER_DATA_OVERRIDE)
fs.mkdirSync(RESOLVED_USER_DATA_DIR, { recursive: true })
app.setPath('userData', RESOLVED_USER_DATA_DIR)

const DEV_SERVER = process.env.HERMES_DESKTOP_DEV_SERVER
const IS_PACKAGED = app.isPackaged
const IS_MAC = process.platform === 'darwin'
const IS_WINDOWS = process.platform === 'win32'
const IS_WSL = isWslEnvironment()
const APP_ROOT = app.getAppPath()

// Public-read COS bucket base URL that hosts the ApexNodes runtime source
// tarball + uv binary for mainland-China first-launch installs (published by
// scripts/publish-runtime-tarball.sh). When packaged, bootstrap-runner turns on
// install.sh's CN mirror mode and points its runtime source here:
//   <base>/hermes-agent-<commit>.tar.gz  and  <base>/uv-<triple>.tar.gz
// Override at pack time via HERMES_RUNTIME_COS_BASE. If this is ever cleared,
// install.sh CN mode degrades gracefully to git clone / astral.sh.
const RUNTIME_COS_BASE =
  process.env.HERMES_RUNTIME_COS_BASE ||
  'https://apexnodes-runtime-202606250443-1300912302.cos.ap-guangzhou.myqcloud.com/runtime'

function hiddenWindowsChildOptions(options = {}) {
  if (!IS_WINDOWS || Object.prototype.hasOwnProperty.call(options, 'windowsHide')) {
    return options
  }
  return { ...options, windowsHide: true }
}

// Remote displays (SSH X11 forwarding, VNC, RDP) make Chromium's GPU
// compositor flicker — accelerated layers can't be presented cleanly over the
// wire, so the window flashes during scroll/streaming/animation. Local
// Windows/macOS (and WSLg, which renders locally via vGPU) composite on the
// GPU and never see it. Fall back to software rendering when a remote display
// is detected; it's rock-steady over the wire and the CPU cost is negligible
// next to the connection's latency. Must run before app `ready` — these
// switches only apply pre-launch. Override with HERMES_DESKTOP_DISABLE_GPU
// (1/true → always disable, 0/false → keep GPU on).
const REMOTE_DISPLAY_REASON = detectRemoteDisplay()
if (REMOTE_DISPLAY_REASON) {
  app.disableHardwareAcceleration()
  // Belt-and-suspenders for X11/VNC, where the Viz compositor can still glitch
  // with only --disable-gpu: force compositing onto the CPU too.
  app.commandLine.appendSwitch('disable-gpu-compositing')
  console.log(
    `[hermes] remote display detected (${REMOTE_DISPLAY_REASON}); disabling GPU hardware acceleration to prevent flicker`
  )
}

// Keep the renderer running at full speed while the window is in the background
// or occluded. The chat transcript streams to screen through a
// requestAnimationFrame-gated flush; Chromium pauses rAF (and clamps timers)
// for backgrounded/occluded renderers, so without these the live answer stalls
// whenever the window loses focus (switching to your editor mid-turn, detached
// devtools, another window covering it) and only paints on refocus or refresh.
// `backgroundThrottling: false` on the BrowserWindow covers the blurred case;
// these process-level switches additionally stop Chromium from backgrounding or
// occlusion-throttling the renderer. Must run before app `ready`.
app.commandLine.appendSwitch('disable-renderer-backgrounding')
app.commandLine.appendSwitch('disable-backgrounding-occluded-windows')
app.commandLine.appendSwitch('disable-background-timer-throttling')

const SOURCE_REPO_ROOT = path.resolve(APP_ROOT, '../..')

// Build-time install stamp -- the git ref this .exe was built against.
//
// Written by apps/desktop/scripts/write-build-stamp.cjs during `npm run build`
// and bundled into packaged apps via electron-builder's extraResources entry,
// so the runtime stamp ends up at process.resourcesPath/install-stamp.json
// after install. The bootstrap runner (Phase 1D) reads it to know which
// commit to clone when running install.ps1 stages at first launch.
//
// Returns null when the file is missing (dev runs from a checkout where
// build hasn't been invoked, or schema mismatch). Callers must handle null.
//
// Schema:
//   { schemaVersion: 1, commit, branch, builtAt, dirty, source }
const INSTALL_STAMP_SCHEMA_VERSION = 1
function loadInstallStamp() {
  // Try packaged location first (resources/install-stamp.json), then the
  // dev/local build output (apps/desktop/build/install-stamp.json) so
  // someone running `npm run start` after a local `npm run build` also
  // sees a stamp without needing a packaged build.
  const candidates = [
    process.resourcesPath ? path.join(process.resourcesPath, 'install-stamp.json') : null,
    path.join(APP_ROOT, 'build', 'install-stamp.json')
  ].filter(Boolean)
  for (const p of candidates) {
    try {
      const raw = fs.readFileSync(p, 'utf8')
      const parsed = JSON.parse(raw)
      if (parsed && typeof parsed === 'object' && typeof parsed.commit === 'string' && parsed.commit.length >= 7) {
        if (parsed.schemaVersion !== INSTALL_STAMP_SCHEMA_VERSION) {
          console.warn(
            `[hermes] install-stamp.json schemaVersion ${parsed.schemaVersion} != expected ${INSTALL_STAMP_SCHEMA_VERSION}; ignoring`
          )
          continue
        }
        return Object.freeze({
          schemaVersion: parsed.schemaVersion,
          commit: parsed.commit,
          branch: parsed.branch || null,
          builtAt: parsed.builtAt || null,
          dirty: Boolean(parsed.dirty),
          source: parsed.source || null,
          path: p
        })
      }
    } catch {
      // Either ENOENT or malformed JSON; try the next candidate
    }
  }
  return null
}
const INSTALL_STAMP = loadInstallStamp()
if (INSTALL_STAMP) {
  console.log(
    `[hermes] install stamp: ${INSTALL_STAMP.commit.slice(0, 12)}${INSTALL_STAMP.branch ? ` (${INSTALL_STAMP.branch})` : ''}${INSTALL_STAMP.dirty ? ' [DIRTY]' : ''} from ${INSTALL_STAMP.source || 'unknown'}`
  )
} else if (IS_PACKAGED) {
  // Dev builds without a stamp are normal; packaged builds without one
  // mean the bootstrap won't know what to clone. Surface clearly.
  console.error(
    '[hermes] WARNING: no install-stamp.json found in packaged build. First-launch bootstrap will not have a pinned ref to install.'
  )
}

// HERMES_HOME — the user-facing root for everything Hermes-related. Mirrors
// scripts/install.ps1's $HermesHome and scripts/install.sh's $HERMES_HOME.
//
// Defaults:
//   Windows: %LOCALAPPDATA%\hermes (matches install.ps1)
//   macOS / Linux: ~/.hermes (matches install.sh)
//
// Special case for Windows: if the user has a legacy ~/.hermes directory
// (e.g., from a prior pip install or a manual setup) AND no
// %LOCALAPPDATA%\hermes yet, prefer the legacy path so we don't orphan their
// existing config / sessions / .env. New installs go to %LOCALAPPDATA%.
//
// HERMES_DESKTOP_USER_DATA_DIR (used by test:desktop:fresh) puts the sandbox
// HERMES_HOME beneath the throwaway userData dir so a fresh-install run never
// touches the user's real ~/.hermes / %LOCALAPPDATA%\hermes.
function resolveHermesHome() {
  if (process.env.HERMES_HOME) return normalizeHermesHomeRoot(process.env.HERMES_HOME)
  if (USER_DATA_OVERRIDE) return path.join(path.resolve(USER_DATA_OVERRIDE), 'hermes-home')
  if (IS_WINDOWS) {
    // A GUI app launched from Explorer inherits the environment block captured
    // at login, so a HERMES_HOME set via `setx` AFTER login is invisible in
    // process.env even though the CLI (a fresh shell) sees it. Without this the
    // backend silently falls back to %LOCALAPPDATA%\hermes and reports "No
    // inference provider configured" despite a valid configured home (#45471).
    // Consult the live User-scoped registry value before the default below.
    const fromRegistry = readWindowsUserEnvVar('HERMES_HOME')
    if (fromRegistry) return normalizeHermesHomeRoot(fromRegistry)
  }
  if (IS_WINDOWS && process.env.LOCALAPPDATA) {
    const localappdata = path.join(process.env.LOCALAPPDATA, 'apexnodes')
    const legacy = path.join(app.getPath('home'), '.apexnodes')
    // Migrate transparently to LOCALAPPDATA, but honour an existing legacy
    // ~/.apexnodes setup (no LOCALAPPDATA install yet) so users don't lose state.
    if (!directoryExists(localappdata) && directoryExists(legacy)) return legacy
    return localappdata
  }
  // ApexNodes Desktop runs an independent runtime home (~/.apexnodes) so it
  // never collides with a stock Hermes install (~/.hermes) sharing the machine.
  return path.join(app.getPath('home'), '.apexnodes')
}

const HERMES_HOME = resolveHermesHome()
// Force the resolved root into the environment so every child process —
// install.sh's first-launch clone, the dashboard backend, the hermes CLI —
// inherits the SAME independent home instead of re-deriving its own ~/.hermes
// default. Idempotent: when HERMES_HOME came from an explicit override this is a
// no-op; on a default install it pins clone/config/data under ~/.apexnodes.
process.env.HERMES_HOME = HERMES_HOME
// ACTIVE_HERMES_ROOT — the canonical mutable Hermes install. Same path
// install.ps1 / install.sh use, so a desktop-only user and a CLI-only user end
// up with identical layouts and can share one install.
const ACTIVE_HERMES_ROOT = path.join(HERMES_HOME, 'hermes-agent')
// VENV_ROOT — venv lives inside the repo, exactly like install.ps1 does it.
const VENV_ROOT = path.join(ACTIVE_HERMES_ROOT, 'venv')
// BOOTSTRAP_COMPLETE_MARKER — written by the first-launch bootstrap runner
// (Phase 1D) after install.ps1 has completed all stages and the user has
// finished initial configuration. Presence of this marker means the install
// is in a known-good state and we can skip the bootstrap flow on subsequent
// boots, going straight to `resolveHermesBackend()`. Missing or stale marker
// means we re-run the bootstrap; install.ps1's stages are idempotent so a
// re-run on an already-good install just discovers everything in place.
//
// We deliberately put the marker INSIDE ACTIVE_HERMES_ROOT (not alongside)
// so that deleting the checkout to start fresh also deletes the marker --
// avoids the confusing "marker exists but checkout is gone" state.
const BOOTSTRAP_COMPLETE_MARKER = path.join(ACTIVE_HERMES_ROOT, '.hermes-bootstrap-complete')
const BOOTSTRAP_MARKER_SCHEMA_VERSION = 1

// ── Runtime opt-in update — durable pin override (R4/R5) ────────────────────
// The build-time install-stamp pins the runtime commit the .app was shipped
// against. R5's opt-in update re-points that pin to the admin-set default
// (GET /api/v1/runtime/latest) WITHOUT re-shipping the app. The chosen pin must
// survive a restart (an update the user triggered should still take after they
// quit before bootstrap finished), so we persist it here.
//
// Lives under HERMES_HOME (NOT inside ACTIVE_HERMES_ROOT): the re-bootstrap that
// applies an update can wipe/replace the checkout, and the override — plus the
// snapshot of the marker we are replacing, for rollback — must outlive that.
//
// Schema:
//   {
//     schemaVersion: 1,
//     commit: "<sha>" | null,
//     branch: "<tag/branch>" | null,
//     version: "<label>" | null,
//     requestedAt: "<ISO>",
//     // rollback snapshot of the bootstrap marker this update is replacing:
//     previousMarker: { ...marker } | null
//   }
const RUNTIME_PIN_OVERRIDE_PATH = path.join(HERMES_HOME, '.apexnodes-runtime-override.json')
const RUNTIME_PIN_OVERRIDE_SCHEMA_VERSION = 1

function readRuntimePinOverride() {
  const parsed = readJson(RUNTIME_PIN_OVERRIDE_PATH)
  if (!parsed || typeof parsed !== 'object') return null
  if (parsed.schemaVersion !== RUNTIME_PIN_OVERRIDE_SCHEMA_VERSION) return null
  // Must carry at least one usable pin field, else it's meaningless.
  if (!parsed.commit && !parsed.branch) return null
  return parsed
}

function writeRuntimePinOverride(payload) {
  fs.mkdirSync(path.dirname(RUNTIME_PIN_OVERRIDE_PATH), { recursive: true })
  const merged = {
    schemaVersion: RUNTIME_PIN_OVERRIDE_SCHEMA_VERSION,
    commit: payload.commit || null,
    branch: payload.branch || null,
    version: payload.version || null,
    requestedAt: new Date().toISOString(),
    previousMarker: payload.previousMarker || null
  }
  writeFileAtomic(RUNTIME_PIN_OVERRIDE_PATH, JSON.stringify(merged, null, 2) + '\n', 'utf8')
  return merged
}

function clearRuntimePinOverride() {
  try {
    if (fileExists(RUNTIME_PIN_OVERRIDE_PATH)) {
      fs.rmSync(RUNTIME_PIN_OVERRIDE_PATH, { force: true })
    }
  } catch (error) {
    rememberLog(`[runtime-update] failed to clear pin override: ${error && error.message}`)
  }
}

// The ApexNodes API base the desktop talks to for managed endpoints (login,
// provision-key) and the public runtime /latest discovery. resolveApexEndpoints
// applies the same APEXNODES_API_BASE override the V0.2 managed flow uses, so a
// staging build retargets both with one env var.
function apexApiBase() {
  try {
    return resolveApexEndpoints(process.env).apiBase
  } catch {
    return ''
  }
}

// Resolve the install pin for a bootstrap run (R4 first install + R5 applied
// override). NEVER throws — every failure path degrades to the build-time stamp.
//
//   1. A persisted opt-in override (R5) wins: the user explicitly chose a
//      version; honor it across restarts until it installs (or is rolled back).
//   2. No override -> R4: fetch the admin-set default (GET /api/v1/runtime/latest)
//      and overlay it onto the baked stamp for THIS install only (not persisted —
//      a fresh machine just tracks the current admin default at install time).
//   3. Cloud unreachable / no default / parse error -> the baked stamp verbatim.
async function resolveBootstrapStamp(bakedStamp) {
  // (1) Persisted override takes precedence and short-circuits the network.
  const override = readRuntimePinOverride()
  if (override) {
    const merged = overlayStampWithPin(
      bakedStamp || INSTALL_STAMP,
      { commit: override.commit, branch: override.branch, version: override.version },
      'opt-in-update'
    )
    rememberLog(
      `[runtime-update] using persisted opt-in pin override: version=${override.version || '?'} ` +
        `commit=${override.commit ? String(override.commit).slice(0, 12) : '-'} branch=${override.branch || '-'}`
    )
    return merged
  }

  // (2) R4: live admin-latest overlay. Bounded, best-effort, never fatal.
  const apiBase = apexApiBase()
  let pin = null
  try {
    pin = await resolveLatestRuntimePin({
      apiBase,
      fetchJson: fetchPublicJson,
      timeoutMs: 10_000,
      log: msg => rememberLog(msg)
    })
  } catch (error) {
    // resolveLatestRuntimePin already swallows; this is belt-and-suspenders so a
    // surprise throw can never abort a first install.
    rememberLog(`[runtime-update] latest-pin resolution errored (ignored): ${error && error.message}`)
    pin = null
  }

  if (!pin) {
    rememberLog('[runtime-update] no admin latest available; installing the build-time pin')
    return bakedStamp || INSTALL_STAMP
  }

  // (3) Overlay the admin latest onto the baked stamp for this install.
  const merged = overlayStampWithPin(bakedStamp || INSTALL_STAMP, pin, 'api-latest')
  rememberLog(
    `[runtime-update] first-install pinning to admin latest: version=${pin.version || '?'} ` +
      `commit=${pin.commit ? pin.commit.slice(0, 12) : '-'} branch=${pin.branch || '-'}`
  )
  return merged
}

// Best-effort reachability probe for an update artifact (the COS source tarball)
// BEFORE we retarget the install pin and re-run bootstrap. This is the
// don't-brick guard: install.sh's CN path deletes INSTALL_DIR if the new tarball
// extract fails, so confirming the object actually exists first keeps a working
// install from being torn down for a 404. Resolves true on a 2xx/3xx HEAD,
// false on 4xx/5xx or a network error. Never throws. A missing URL resolves
// true (the non-CN git-clone path doesn't use COS and verifies via git itself).
function isUpdateArtifactReachable(url, { timeoutMs = 8000 } = {}) {
  return new Promise(resolve => {
    const clean = String(url || '').trim()
    if (!clean) {
      resolve(true)
      return
    }
    let parsed
    try {
      parsed = new URL(clean)
    } catch {
      resolve(false)
      return
    }
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      resolve(false)
      return
    }
    const client = parsed.protocol === 'https:' ? https : http
    let settled = false
    const done = value => {
      if (settled) return
      settled = true
      resolve(value)
    }
    const req = client.request(parsed, { method: 'HEAD' }, res => {
      const code = res.statusCode || 0
      res.resume() // drain
      done(code >= 200 && code < 400)
    })
    req.on('error', () => done(false))
    req.setTimeout(timeoutMs, () => {
      try {
        req.destroy()
      } catch {
        void 0
      }
      done(false)
    })
    req.end()
  })
}

// Roll an opt-in update back after a failed re-bootstrap: restore the marker the
// update was replacing (so the next launch boots the OLD, known-good runtime
// that is still on disk) and drop the override so we don't re-attempt the broken
// pin. Called from the bootstrap-failure path. Idempotent / best-effort.
function rollbackRuntimePinOverride(reason) {
  const override = readRuntimePinOverride()
  if (!override) return false
  rememberLog(`[runtime-update] rolling back opt-in update (${reason || 'failed'})`)
  try {
    if (override.previousMarker && typeof override.previousMarker === 'object') {
      fs.mkdirSync(path.dirname(BOOTSTRAP_COMPLETE_MARKER), { recursive: true })
      writeFileAtomic(
        BOOTSTRAP_COMPLETE_MARKER,
        JSON.stringify(override.previousMarker, null, 2) + '\n',
        'utf8'
      )
      rememberLog('[runtime-update] restored previous bootstrap marker (old runtime remains active)')
    }
  } catch (error) {
    rememberLog(`[runtime-update] failed to restore previous marker on rollback: ${error && error.message}`)
  }
  clearRuntimePinOverride()
  return true
}

// ── hc-472 P1 · versioned runtime bundle wiring (opt-in) ────────────────────
// Everything below is gated on HERMES_BUNDLE_MODE and is a strict no-op when the
// switch is off (the default). The three consumer pieces live in dedicated,
// unit-tested modules: apex-bundle-layout (versioned catalog + pointer-truth
// atomic switch + rollback + startup GC), apex-bundle-download (Range-resumable,
// sha-gated downloader) and apex-bundle-install (never-extract-in-place staging
// + the F1→F2→C1 orchestration). Any bundle-path failure returns cleanly so the
// caller falls back to the existing install chain — a bundle attempt can never
// brick a working install (the design's core回退 contract).

function bundleModeEnabled() {
  const v = String(process.env.HERMES_BUNDLE_MODE || '').trim().toLowerCase()
  return v === '1' || v === 'true' || v === 'on' || v === 'yes'
}

// This machine's bundle (os, arch). P1 ships win-x64; mac legs arrive with P2.
// null = unsupported platform (caller falls back to the legacy chain).
function desktopBundleTarget() {
  if (IS_WINDOWS) return process.arch === 'x64' ? { os: 'win', arch: 'x64' } : null
  if (IS_MAC) return { os: 'mac', arch: process.arch === 'arm64' ? 'arm64' : 'x64' }
  return null
}

// Extract with bsdtar exactly as the bundle build/install contract expects:
// System32\tar.exe on Windows (immune to the GNU-tar "C: is a remote host" trap
// — same reason build-runtime-bundle.mjs::tarBin pins it), `tar` elsewhere.
// Async spawn so a multi-minute extract never freezes the electron main thread.
function extractBundleArchive(archivePath, destDir) {
  return new Promise((resolve, reject) => {
    const tarExe = IS_WINDOWS
      ? path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'tar.exe')
      : 'tar'
    const child = spawn(tarExe, ['-xzf', archivePath, '-C', destDir], { stdio: ['ignore', 'ignore', 'pipe'] })
    let stderr = ''
    child.stderr.on('data', d => {
      stderr = (stderr + String(d)).slice(-2000)
    })
    child.on('error', reject)
    child.on('close', code => (code === 0 ? resolve() : reject(new Error(`tar exited ${code}: ${stderr.slice(-400)}`))))
  })
}

// Run the BUNDLED node against the BUNDLED tool copy (fixup / verify). The
// bundle ships scripts/build-runtime-bundle.mjs + its own node, so there is no
// external fixup binary to keep in lockstep (manifest.fixup drives the argv).
function runBundledTool(exe, argv, label) {
  return new Promise((resolve, reject) => {
    const child = spawn(exe, argv, { stdio: ['ignore', 'pipe', 'pipe'] })
    let tail = ''
    const cap = d => {
      tail = (tail + String(d)).slice(-2000)
    }
    child.stdout.on('data', cap)
    child.stderr.on('data', cap)
    child.on('error', reject)
    child.on('close', code => (code === 0 ? resolve() : reject(new Error(`bundle ${label} exited ${code}: ${tail.slice(-400)}`))))
  })
}

function bundleRuntimeDownload({ url, dest, sha256, size }) {
  return downloadWithResume({ url, dest, sha256, size, log: msg => rememberLog(msg) })
}

// Startup self-heal: rebuild the active link from the truth pointer (a switch or
// legacy migration interrupted mid-repoint — D1), GC everything but
// current+previous and any `.tmp` half-installs, tighten past the disk budget if
// versions/ overflowed (C2), and reap the legacy in-place fallback once it is no
// longer a rollback target (D1). Runs before any runtime child is spawned
// (nothing holds a handle on an old venv), and is fully fail-soft.
function reconcileAndGcBundleRuntime() {
  if (!bundleModeEnabled()) return
  try {
    const rec = bundleMigrate.reconcileMigration(HERMES_HOME)
    if (rec.reconciled) rememberLog(`[bundle] healed active link (${rec.action}) -> ${rec.key || '?'}`)
    // Watermark-aware GC: normal keep current+previous, or drop previous when
    // versions/ blew past its disk budget. One pass (the watermark check runs GC).
    const water = bundleDiskspace.enforceVersionsWatermark(HERMES_HOME)
    const gc = water.gc
    if (gc && (gc.removed.length || gc.orphansRemoved.length || gc.skipped.length)) {
      rememberLog(
        `[bundle] GC removed=${JSON.stringify(gc.removed)} staging=${JSON.stringify(gc.orphansRemoved)} ` +
          `skipped=${JSON.stringify(gc.skipped)}`
      )
    }
    if (water.warning) rememberLog(`[bundle] ${water.warning}`)
    // Reap the legacy in-place fallback once the sentinel has left the pointer.
    const asideGc = bundleMigrate.gcLegacyAside(HERMES_HOME)
    if (asideGc.removed) rememberLog(`[bundle] reaped legacy in-place fallback ${asideGc.path}`)
  } catch (error) {
    rememberLog(`[bundle] reconcile/GC errored (ignored): ${error && error.message}`)
  }
}

// R5 opt-in update via the bundle set. Resolves {ok:true,...} on a completed
// pointer+link switch, or {ok:false, code, ...} so apply-update can fall back to
// the legacy marker-drop re-bootstrap. Never throws.
async function applyRuntimeBundleUpdateFlow(pin) {
  const target = desktopBundleTarget()
  if (!target) return { ok: false, code: 'unsupported_platform' }
  if (!pin || !pin.key) return { ok: false, code: 'no_pin_key' }
  return applyRuntimeBundleUpdate({
    hermesHome: HERMES_HOME,
    os: target.os,
    arch: target.arch,
    key: String(pin.key),
    desktopVersion: app.getVersion(),
    cosBase: process.env.HERMES_RUNTIME_COS_BASE || '',
    fetchManifest: url => fetchPublicJson(url, { timeoutMs: 20000 }),
    download: bundleRuntimeDownload,
    extract: extractBundleArchive,
    runTool: runBundledTool,
    log: msg => rememberLog(msg)
  })
}

// Shell UI locale block, appended to every seed. The runtime writes
// display.language: en by default, which beats the China-first zh fallback (it
// only triggers when the key is absent); pre-seeding zh makes a fresh install
// open in Simplified Chinese. show_reasoning is a product decision: the runtime
// defaults it to false (hermes_cli/config.py display.show_reasoning), but a
// fresh APEX install should show the reasoning blocks (推理过程块) out of the
// box — it lives here because it shares the display: mapping.
const SEED_DISPLAY_BLOCK =
  '# Shell UI locale. The runtime writes display.language: en by default, which\n' +
  '# beats the China-first zh fallback (it only triggers when the key is\n' +
  '# absent); pre-seeding zh makes a fresh install open in Simplified Chinese.\n' +
  '# show_reasoning: APEX product default — reasoning blocks visible on a fresh\n' +
  '# install (the runtime defaults to false).\n' +
  'display:\n' +
  '  language: zh\n' +
  '  show_reasoning: true\n'

// APEX product defaults appended to every seed alongside SEED_DISPLAY_BLOCK.
// Both keys exist in the runtime schema (hermes_cli/config.py) and both values
// MATCH today's runtime defaults — seeded explicitly to pin the product
// behavior against upstream default drift:
//   agent.image_input_mode: auto — image attachments go native only to
//     vision-capable models, otherwise text pre-analysis (config.py agent block).
//   timezone: '' — empty means "server-local time" (config.py top-level
//     timezone), which on a desktop IS the OS timezone, i.e. follow-the-OS.
// Top-level keys here (agent:, timezone:) must not collide with the other seed
// blocks (model:/custom_providers:/display:/skills:/plugins: — see
// seedDefaultModelConfig).
const SEED_PRODUCT_DEFAULTS_BLOCK =
  '# APEX product defaults: image attachments auto-routed by model vision\n' +
  "# support; empty timezone = follow the OS (server-local) clock.\n" +
  'agent:\n' +
  '  image_input_mode: auto\n' +
  "timezone: ''\n"

// Curated domestic MoA preset (managed seed only — every slot routes through
// the relay via the global `custom` endpoint, so BYOK installs without a relay
// key can't run it). Orchestration rationale — anchored on the 2026-06-29
// five-model agentic eval + a 2026-07-04 real-world rerun:
//   * The aggregator is the ACTING model (holds the tools, takes every turn —
//     agent/moa_loop.py), so its execution discipline/speed/style dominate the
//     residual quality gap. qwen3.7-max is the domestic best on exactly those
//     (fastest run, fewest self-repair loops, best code modularity); its known
//     weakness (self-checking) is precisely what the reference panel fixes.
//   * GLM-5.2 as brain was measured TWICE at ~30min/run (slowest + priciest
//     output tokens) — eliminated as aggregator, kept as the polish/design
//     ADVISOR where its strength (presentation) arrives as cheap advice.
//   * deepseek-v4-pro referees correctness and is the cost-tier brain
//     alternative (cache pricing) for long agentic loops.
//   * deepseek-v4-flash is deliberately absent: weaker sibling of a ref that
//     is already present — adds cost, not diversity.
// Upstream's own default preset points at GPT-5.5 / OpenRouter / Claude — all
// unreachable from mainland China, which is exactly why the seed replaces it.
// Temperatures follow the upstream preset defaults.
const SEED_MOA_BLOCK =
  '# APEX 多模型协作(MoA)预设:参考模型出多样性,聚合模型执行。全部经由\n' +
  '# APEX 中转,无需额外配置。/moa apex-moa 或模型菜单里启用。\n' +
  'moa:\n' +
  '  default_preset: apex-moa\n' +
  '  presets:\n' +
  '    apex-moa:\n' +
  '      reference_models:\n' +
  '      - model: deepseek-v4-pro\n' +
  '        provider: custom:apex-nodes.com\n' +
  '      - model: kimi-k2.7-code\n' +
  '        provider: custom:apex-nodes.com\n' +
  '      - model: glm-5.2\n' +
  '        provider: custom:apex-nodes.com\n' +
  '      aggregator:\n' +
  '        provider: custom:apex-nodes.com\n' +
  '        model: qwen3.7-max\n' +
  '      reference_temperature: 0.6\n' +
  '      aggregator_temperature: 0.4\n'

// ── ApexNodes default model preset ─────────────────────────────────────────
// We pre-seed config.yaml BEFORE the first-launch installer runs: install.sh
// only creates config.yaml from its template when absent, so this seed wins
// WITHOUT forking the runtime. Idempotent + non-destructive: an existing
// config.yaml (returning user, or one they edited) is left untouched.
//
// Two default paths (see apex-managed.cjs):
//   - MANAGED (V0.2, preferred): a signed-in user's relay key is on disk, so we
//     point the runtime's inference at the ApexNodes relay (provider=custom +
//     base_url=/relay/v1 + the user's key + deepseek-v4-pro). Zero-key chat — the
//     user pays via their cloud account; the relay decouples display vs routed
//     model (hc-184). Uses the same model.base_url/api_key fields the "Local /
//     custom endpoint" BYOK flow writes, so no new runtime plumbing.
//   - BYOK (fallback): no relay key (managed disabled, or not signed in yet) →
//     ship DeepSeek direct, so a fresh install only needs the user's own
//     DEEPSEEK_API_KEY, added in Settings › Providers (the DeepSeek card).
//     We intentionally do NOT set model.base_url here — the `deepseek` provider
//     already pins inference_base_url=https://api.deepseek.com/v1, and a bare
//     api.deepseek.com (missing /v1) would 404.
function seedDefaultModelConfig() {
  try {
    const configPath = path.join(HERMES_HOME, 'config.yaml')
    if (fs.existsSync(configPath)) return
    fs.mkdirSync(HERMES_HOME, { recursive: true })

    const managed = resolveManagedConfig()
    // hc-392 China profile: the same skills.disabled (49) + model.disabled_providers
    // ([copilot]) that cli-config.yaml.example carries must be folded into the
    // desktop seed, because this seed pre-empts install.sh's example-copy (both
    // are absent-gated and this one runs first) — otherwise skill-cut +
    // Copilot-disable would be a no-op on a fresh desktop install. The
    // denylist sits INSIDE the model: block (a 2nd top-level model: key would
    // be invalid YAML); the skills block is its own top-level key. Same story
    // for plugins.enabled: the runtime's standalone plugin loader is opt-in,
    // so a seed without that block would ship apex-overlay + the apexnodes-*
    // tool plugins disabled on every fresh install (see MANAGED_PLUGIN_NAMES).
    const skillsBlock = seedSkillsBlockYaml()
    const pluginsBlock = seedPluginsBlockYaml()
    let seed
    if (defaultModelPath({ enabled: isManagedEnabled(process.env), key: managed.key }) === 'managed') {
      const block = managedModelConfigYaml(
        buildManagedModelConfig(managed.key, process.env, { baseUrl: managed.baseUrl, model: managed.model }),
        { disabledProviders: MODEL_DISABLED_PROVIDERS }
      )
      seed =
        '# Seeded by ApexNodes Desktop (V0.2 — managed).\n' +
        '# Inference is routed through the ApexNodes relay using your signed-in\n' +
        '# cloud account. Switch to your own provider any time in\n' +
        '# Settings › Providers.\n' +
        block +
        SEED_DISPLAY_BLOCK +
        SEED_PRODUCT_DEFAULTS_BLOCK +
        SEED_MOA_BLOCK +
        skillsBlock +
        pluginsBlock
      rememberLog(`[apexnodes] seeded managed relay config at ${configPath}`)
    } else {
      seed =
        '# Seeded by ApexNodes Desktop (BYOK).\n' +
        '# DeepSeek is the default provider. Add your key in Settings › Providers\n' +
        '# (the DeepSeek card), which writes DEEPSEEK_API_KEY.\n' +
        'model:\n' +
        '  default: deepseek-v4-pro\n' +
        '  provider: deepseek\n' +
        modelDisabledProvidersYaml() +
        SEED_DISPLAY_BLOCK +
        SEED_PRODUCT_DEFAULTS_BLOCK +
        skillsBlock +
        pluginsBlock
      rememberLog(`[apexnodes] seeded default DeepSeek (BYOK) config at ${configPath}`)
    }
    fs.writeFileSync(configPath, seed, { encoding: 'utf8' })
  } catch (err) {
    rememberLog(`[apexnodes] could not seed default config: ${err && err.message ? err.message : err}`)
  }
}

// Keep the registered relay custom_providers entry's api_key in lockstep with
// the freshly provisioned relay key (see syncCustomProviderKeyYaml — provision
// rotates the key on every sign-in, and the runtime's dedupe never refreshes a
// registered entry's key, stranding the picker's live model listing on a dead
// credential). Runs at boot and right after provisioning; no-op when signed
// out, config missing, or already in sync.
function syncManagedCustomProviderKey() {
  try {
    const managed = resolveManagedConfig()
    if (!managed.key || !managed.baseUrl) return
    const configPath = path.join(HERMES_HOME, 'config.yaml')
    if (!fs.existsSync(configPath)) return
    const raw = fs.readFileSync(configPath, 'utf8')
    const { changed, next } = syncCustomProviderKeyYaml(raw, managed.baseUrl, managed.key)
    if (!changed) return
    fs.writeFileSync(configPath, next, { encoding: 'utf8' })
    rememberLog('[apexnodes] refreshed relay custom_providers api_key after key rotation')
  } catch (err) {
    rememberLog(`[apexnodes] custom provider key sync skipped: ${err && err.message ? err.message : err}`)
  }
}

const DESKTOP_CONNECTION_CONFIG_PATH = path.join(app.getPath('userData'), 'connection.json')
const DESKTOP_UPDATE_CONFIG_PATH = path.join(app.getPath('userData'), 'updates.json')
// active-profile.json records which Hermes profile the desktop launches its
// local backend as. When set, startHermes() passes `hermes --profile <name>
// dashboard …`, which deterministically pins HERMES_HOME (see
// _apply_profile_override in hermes_cli/main.py) and bypasses the sticky
// ~/.hermes/active_profile file. Unset (null) preserves the legacy behavior:
// no --profile flag, so the backend honors active_profile / default.
const DESKTOP_PROFILE_CONFIG_PATH = path.join(app.getPath('userData'), 'active-profile.json')
// apex-managed.json holds the signed-in user's ApexNodes relay key (encrypted
// with safeStorage, same as the remote-gateway token). It backs the managed-LLM
// default path: seedDefaultModelConfig reads it to seed config.yaml with the
// relay endpoint so a fresh, signed-in install gets zero-key chat. Kept in its
// own file (not connection.json) because the managed-LLM credential and the
// remote-gateway session are unrelated concerns.
const DESKTOP_MANAGED_CONFIG_PATH = path.join(app.getPath('userData'), 'apex-managed.json')
// apex-client-config.json caches the platform-served versioned client config
// ({ version, payload, fetchedAt, appliedVersion } — see apex-client-config.cjs).
// Refreshed fail-soft at boot and after a successful managed sign-in; the
// renderer reads it over IPC and applies payload.config_yaml through the
// runtime's global-config API once the gateway is open. No secrets inside, so
// plain JSON (no safeStorage), unlike apex-managed.json.
const DESKTOP_CLIENT_CONFIG_PATH = path.join(app.getPath('userData'), 'apex-client-config.json')
// apex-feishu.json holds the signed-in user's OWN Feishu app credential mirrored
// from the cloud (hc-444). The app_secret is a real secret → stored ENCRYPTED
// (safeStorage, same treatment as the managed relay key in apex-managed.json);
// app_id / domain / agent_name / status are non-secret and kept in clear. main
// injects the decrypted creds JUST-IN-TIME into the backend spawn env
// (FEISHU_APP_ID / FEISHU_APP_SECRET / FEISHU_DOMAIN) so the runtime's Feishu
// adapter + lark doc/drive tools light up — the secret never touches a plaintext
// .env and is never logged. Own file (not apex-managed.json) because the Feishu
// office-suite credential and the managed-LLM relay key are unrelated concerns.
const DESKTOP_FEISHU_CONFIG_PATH = path.join(app.getPath('userData'), 'apex-feishu.json')
// Mirrors hermes_cli.profiles._PROFILE_ID_RE so we never hand the backend a
// value its profile resolver would reject and exit on.
const PROFILE_NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/
// Branch we track for self-update. The GUI work has merged to main, so this
// tracks main. User can also override at runtime via
// hermesDesktop.updates.setBranch().
const DEFAULT_UPDATE_BRANCH = 'main'
// desktop.log lives under HERMES_HOME/logs/ so it sits next to agent.log,
// errors.log, gateway.log produced by hermes_logging.setup_logging — one log
// directory per user, regardless of which UI surface produced the line.
const DESKTOP_LOG_PATH = path.join(HERMES_HOME, 'logs', 'desktop.log')
const DESKTOP_LOG_FLUSH_MS = 120
const DESKTOP_LOG_BUFFER_MAX_CHARS = 64 * 1024
// Bound desktop.log on disk. It is an append-only forensic log, so a boot loop
// (version-skew crash -> backend exits instantly -> renderer keeps hitting
// Retry) appends the full bootstrap transcript every attempt and grows without
// bound — we have seen it reach ~326 GB and exhaust the disk, which then breaks
// update/install (no room for git/venv/npm temp files).
//
// Mirror the Python logs (hermes_logging.py RotatingFileHandler, maxBytes x
// backupCount): cascade live -> .1 -> .2 -> .3, drop the oldest. Steady-state
// stays bounded at ~(backupCount + 1) x cap however hard the app loops.
//
// Bounding alone never RECLAIMS an already-huge file: a plain rotation just
// renames the monster to .1 and strands it for a cycle a healthy app may never
// reach. A multi-GB boot-loop transcript has no diagnostic value, so anything
// past the discard ceiling is deleted outright — the updated app self-heals a
// disk a stale build filled, on the next launch.
const DESKTOP_LOG_MAX_BYTES = 10 * 1024 * 1024
const DESKTOP_LOG_BACKUP_COUNT = 3
const DESKTOP_LOG_DISCARD_BYTES = DESKTOP_LOG_MAX_BYTES * 4
const desktopLogBackupPath = n => `${DESKTOP_LOG_PATH}.${n}`
const BOOT_FAKE_MODE = process.env.HERMES_DESKTOP_BOOT_FAKE === '1'
const BOOT_FAKE_STEP_MS = (() => {
  const raw = Number.parseInt(String(process.env.HERMES_DESKTOP_BOOT_FAKE_STEP_MS || ''), 10)
  if (!Number.isFinite(raw) || raw <= 0) return 650
  return Math.max(120, raw)
})()
// User-visible product name: drives app.setName (macOS menu-bar app menu —
// 关于/隐藏/退出), the native About panel, and the fallback notification title.
// The userData pin near the top of this file deliberately does NOT follow this
// name — see user-data-dir.cjs.
const APP_NAME = 'APEX'
const TITLEBAR_HEIGHT = 34
const MACOS_TRAFFIC_LIGHTS_HEIGHT = 14
const WINDOW_BUTTON_POSITION = {
  x: 24,
  y: TITLEBAR_HEIGHT / 2 - MACOS_TRAFFIC_LIGHTS_HEIGHT / 2
}
// Width Electron reserves for the Windows/Linux native min/max/close cluster
// when `titleBarOverlay` is enabled. The OS paints these buttons in the
// top-right corner of the renderer; we have to leave that much room on the
// right edge so our system tools (file browser, haptics, settings) don't sit
// underneath them. macOS uses left-side traffic lights instead and reports a
// position via getWindowButtonPosition(), so this width is non-zero only on
// non-macOS platforms.
const NATIVE_OVERLAY_BUTTON_WIDTH = 144
const APP_ICON_PATHS = [
  path.join(APP_ROOT, 'public', 'apple-touch-icon.png'),
  path.join(APP_ROOT, 'dist', 'apple-touch-icon.png'),
  path.join(unpackedPathFor(APP_ROOT), 'dist', 'apple-touch-icon.png')
]

let rendererTitleBarTheme = null
const terminalSessions = new Map()

// Force the NATIVE window appearance (vibrancy material, titlebar, the
// pre-first-paint window background) to follow the APP theme instead of the
// OS appearance. With `vibrancy` set, macOS paints an NSVisualEffectView that
// tracks the window's effective appearance and ignores `backgroundColor` —
// so a dark-themed app on a light-mode Mac flashes a white material on every
// new window until the renderer covers it. The renderer reports its mode via
// 'hermes:native-theme' ('dark' | 'light' | 'system'); we pin
// nativeTheme.themeSource to it and persist the value so cold launches paint
// correctly before the renderer has even loaded.
const NATIVE_THEME_CONFIG_PATH = path.join(app.getPath('userData'), 'native-theme.json')
const THEME_SOURCES = new Set(['dark', 'light', 'system'])

function readPersistedThemeSource() {
  try {
    const parsed = JSON.parse(fs.readFileSync(NATIVE_THEME_CONFIG_PATH, 'utf8'))

    if (parsed && THEME_SOURCES.has(parsed.themeSource)) {
      return parsed.themeSource
    }
  } catch {
    // Missing / malformed → follow the OS like a fresh install.
  }

  return 'system'
}

function writePersistedThemeSource(mode) {
  try {
    fs.mkdirSync(path.dirname(NATIVE_THEME_CONFIG_PATH), { recursive: true })
    fs.writeFileSync(NATIVE_THEME_CONFIG_PATH, JSON.stringify({ themeSource: mode }, null, 2), 'utf8')
  } catch (error) {
    rememberLog(`[theme] write native theme failed: ${error.message}`)
  }
}

nativeTheme.themeSource = readPersistedThemeSource()

// Window translucency (see-through window). One lever, 0–100; 0 = off (the
// default). Mapped to the native window opacity so the desktop shows through
// the whole window. Persisted so a cold launch applies it at window creation,
// before the renderer reports its value. macOS + Windows only; `setOpacity` is
// a no-op on Linux. See store/translucency.
const TRANSLUCENCY_CONFIG_PATH = path.join(app.getPath('userData'), 'translucency.json')

function clampIntensity(value) {
  const n = Math.round(Number(value))

  return Number.isFinite(n) ? Math.min(100, Math.max(0, n)) : 0
}

function readPersistedTranslucency() {
  try {
    return clampIntensity(JSON.parse(fs.readFileSync(TRANSLUCENCY_CONFIG_PATH, 'utf8')).intensity)
  } catch {
    return 0
  }
}

function writePersistedTranslucency(intensity) {
  try {
    fs.mkdirSync(path.dirname(TRANSLUCENCY_CONFIG_PATH), { recursive: true })
    fs.writeFileSync(TRANSLUCENCY_CONFIG_PATH, JSON.stringify({ intensity }, null, 2), 'utf8')
  } catch (error) {
    rememberLog(`[translucency] write failed: ${error.message}`)
  }
}

let translucencyIntensity = readPersistedTranslucency()

// Map the 0–100 lever to a window opacity. Floor at 0.3 so the most see-through
// setting is still usable rather than nearly invisible. 0 → fully opaque.
function windowOpacity() {
  return 1 - (translucencyIntensity / 100) * 0.7
}

// Re-apply translucency to a live window (runtime toggle, no recreation).
// `setOpacity` is a no-op on Linux, which is fine — it just stays opaque there.
function applyWindowTranslucency(win) {
  if (!win || win.isDestroyed() || typeof win.setOpacity !== 'function') {
    return
  }

  try {
    win.setOpacity(windowOpacity())
  } catch (error) {
    rememberLog(`[translucency] apply failed: ${error.message}`)
  }
}

function isHexColor(value) {
  return typeof value === 'string' && /^#[0-9a-f]{6}$/i.test(value)
}

// Background color to paint a window with BEFORE its renderer loads, so a new
// (or reopened) window doesn't flash white/light in dark mode. Prefer the theme
// the renderer last reported; fall back to the OS preference on first launch.
function getWindowBackgroundColor() {
  if (rendererTitleBarTheme && isHexColor(rendererTitleBarTheme.background)) {
    return rendererTitleBarTheme.background
  }

  return nativeTheme.shouldUseDarkColors ? '#111111' : '#f7f7f7'
}

function getTitleBarOverlayOptions() {
  if (IS_MAC) {
    return { height: TITLEBAR_HEIGHT }
  }

  if (rendererTitleBarTheme) {
    return {
      color: rendererTitleBarTheme.background,
      height: TITLEBAR_HEIGHT,
      symbolColor: rendererTitleBarTheme.foreground
    }
  }

  const useDarkColors = nativeTheme.shouldUseDarkColors

  return {
    color: useDarkColors ? '#111111' : '#f7f7f7',
    height: TITLEBAR_HEIGHT,
    symbolColor: useDarkColors ? '#f7f7f7' : '#242424'
  }
}

const MEDIA_MIME_TYPES = {
  '.avi': 'video/x-msvideo',
  '.bmp': 'image/bmp',
  '.flac': 'audio/flac',
  '.gif': 'image/gif',
  '.jpeg': 'image/jpeg',
  '.jpg': 'image/jpeg',
  '.m4a': 'audio/mp4',
  '.mkv': 'video/x-matroska',
  '.mov': 'video/quicktime',
  '.mp3': 'audio/mpeg',
  '.mp4': 'video/mp4',
  '.ogg': 'audio/ogg',
  '.opus': 'audio/ogg; codecs=opus',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.wav': 'audio/wav',
  '.webm': 'video/webm',
  '.webp': 'image/webp'
}

const PREVIEW_HTML_EXTENSIONS = new Set(['.html', '.htm'])
const PREVIEW_WATCH_DEBOUNCE_MS = 120
const LOCAL_PREVIEW_HOSTS = new Set(['0.0.0.0', '127.0.0.1', '::1', '[::1]', 'localhost'])
const TEXT_PREVIEW_MAX_BYTES = 512 * 1024
const PREVIEW_LANGUAGE_BY_EXT = {
  '.c': 'c',
  '.conf': 'ini',
  '.cpp': 'cpp',
  '.css': 'css',
  '.csv': 'csv',
  '.go': 'go',
  '.graphql': 'graphql',
  '.h': 'c',
  '.hpp': 'cpp',
  '.html': 'html',
  '.java': 'java',
  '.js': 'javascript',
  '.json': 'json',
  '.jsx': 'jsx',
  '.kt': 'kotlin',
  '.lua': 'lua',
  '.md': 'markdown',
  '.mjs': 'javascript',
  '.py': 'python',
  '.rb': 'ruby',
  '.rs': 'rust',
  '.sh': 'shell',
  '.sql': 'sql',
  '.svg': 'xml',
  '.toml': 'toml',
  '.ts': 'typescript',
  '.tsx': 'tsx',
  '.txt': 'text',
  '.xml': 'xml',
  '.yaml': 'yaml',
  '.yml': 'yaml',
  '.zsh': 'shell'
}

function looksBinary(buffer) {
  if (!buffer.length) return false

  let suspicious = 0

  for (const byte of buffer) {
    if (byte === 0) return true
    // Allow common whitespace controls: tab, LF, CR.
    if (byte < 32 && byte !== 9 && byte !== 10 && byte !== 13) suspicious += 1
  }

  return suspicious / buffer.length > 0.12
}

function previewFileMetadata(filePath, mimeType) {
  let byteSize = 0
  let binary = false

  try {
    const stat = fs.statSync(filePath)
    byteSize = stat.size

    if (!mimeType.startsWith('image/')) {
      const fd = fs.openSync(filePath, 'r')

      try {
        const sample = Buffer.alloc(Math.min(byteSize, 4096))
        const bytesRead = fs.readSync(fd, sample, 0, sample.length, 0)
        binary = looksBinary(sample.subarray(0, bytesRead))
      } finally {
        fs.closeSync(fd)
      }
    }
  } catch {
    // Metadata is best-effort; the read handlers surface hard errors later.
  }

  return {
    binary,
    byteSize,
    large: byteSize > TEXT_PREVIEW_MAX_BYTES
  }
}

app.setName(APP_NAME)
// Seed the native About panel with the live Hermes version. This is refreshed
// on every open via the explicit "About" menu handler (refreshAboutPanel), so
// an in-place `hermes update` mid-session is reflected without an app restart;
// the seed here just covers the first open and any non-menu invocation path.
app.setAboutPanelOptions({
  applicationName: APP_NAME,
  applicationVersion: resolveHermesVersion(),
  copyright: 'Copyright © 2026 ApexNodes'
})

// Custom scheme for streaming local media (video/audio) into the renderer.
// Reading large media through `readFileDataUrl` failed: it base64-loads the
// whole file into memory and is hard-capped at DATA_URL_READ_MAX_BYTES (16 MB),
// so any non-trivial video silently refused to load. Streaming via a protocol
// handler removes the size cap and gives the <video> element seekable,
// range-aware playback. Must be registered before the app is ready.
const MEDIA_PROTOCOL = 'hermes-media'
// Only audio/video may be streamed. Without this the handler would read any
// non-blocklisted local file (no size cap) for any `fetch(hermes-media://…)`.
const STREAMABLE_MEDIA_EXTS = new Set([
  '.avi',
  '.flac',
  '.m4a',
  '.mkv',
  '.mov',
  '.mp3',
  '.mp4',
  '.ogg',
  '.opus',
  '.wav',
  '.webm'
])

protocol.registerSchemesAsPrivileged([
  {
    scheme: MEDIA_PROTOCOL,
    privileges: {
      secure: true,
      standard: true,
      stream: true,
      supportFetchAPI: true
    }
  }
])

function registerMediaProtocol() {
  protocol.handle(MEDIA_PROTOCOL, async request => {
    let resolvedPath
    try {
      const url = new URL(request.url)
      const filePath = decodeURIComponent(url.pathname.replace(/^\/+/, ''))
      ;({ resolvedPath } = await resolveReadableFileForIpc(filePath, { purpose: 'Media stream' }))
    } catch {
      return new Response('Media not found', { status: 404 })
    }

    if (!STREAMABLE_MEDIA_EXTS.has(path.extname(resolvedPath).toLowerCase())) {
      return new Response('Unsupported media type', { status: 415 })
    }

    // Delegate to Electron's net stack on a file:// URL — it resolves the
    // content-type and honors Range requests so seeking works. Forward the
    // renderer's headers (notably Range) and skip custom-protocol re-entry.
    return electronNet.fetch(pathToFileURL(resolvedPath).toString(), {
      bypassCustomProtocolHandlers: true,
      headers: request.headers
    })
  })
}

let mainWindow = null
let hermesProcess = null
let connectionPromise = null
// Additional per-profile backends, keyed by profile name. The PRIMARY backend
// (the desktop's launch profile) stays managed by hermesProcess +
// connectionPromise + startHermes(); this pool only holds EXTRA profile
// backends spawned lazily when a session belongs to a different profile. A user
// with no named profiles never populates this map, so their experience is
// byte-for-byte the single-backend behavior.
const backendPool = new Map() // profile -> { process, port, token, connectionPromise, lastActiveAt }
// Keep the pool light: cap concurrent profile backends (LRU eviction) and reap
// idle ones. A user idles at exactly the primary backend; pool backends only
// exist while a non-primary profile is actively being chatted through.
const POOL_MAX_BACKENDS = Math.max(1, Number(process.env.HERMES_DESKTOP_POOL_MAX) || 3)
const POOL_IDLE_MS = Math.max(60_000, Number(process.env.HERMES_DESKTOP_POOL_IDLE_MS) || 10 * 60_000)
// A backend touched within this window has a live renderer socket (the keepalive
// pings every 60s for every open profile). LRU eviction must spare these — a
// concurrent multi-profile session keeps several backends "fresh" at once, and
// killing one to honor the soft cap would abort a running agent.
const POOL_KEEPALIVE_FRESH_MS = 90_000
let poolIdleReaper = null
// Auto-reload budget for renderer crashes. A deterministic startup crash would
// otherwise loop forever (reload → crash → reload), pinning CPU and spamming
// logs. Allow a few reloads per rolling window, then stop and leave the dead
// window so the user can read the error / quit.
const RENDERER_RELOAD_WINDOW_MS = 60_000
const RENDERER_RELOAD_MAX = 3
let rendererReloadTimes = []
// Latched bootstrap failure: when the first-launch install fails, we hold
// onto the error so subsequent startHermes() calls (e.g. the renderer's
// ensureGatewayOpen retrying after the WS won't open) return the same error
// instead of re-running install.ps1 in a hot loop. Cleared explicitly by
// the renderer's "Reload and retry" path or by quitting the app.
let bootstrapFailure = null
// Active first-launch install, so the renderer's Cancel button (and app quit)
// can abort the in-flight install.sh/ps1 instead of leaving it running.
let bootstrapAbortController = null
let connectionConfigCache = null
let connectionConfigCacheMtime = null
const hermesLog = []
const previewWatchers = new Map()
let previewShortcutActive = false
let desktopLogBuffer = ''
let desktopLogFlushTimer = null
let desktopLogFlushPromise = Promise.resolve()
let nativeThemeListenerInstalled = false
let bootProgressState = {
  error: null,
  fakeMode: BOOT_FAKE_MODE,
  message: 'Waiting to start Hermes backend',
  phase: 'idle',
  progress: 0,
  running: false,
  timestamp: Date.now()
}

// Pure planner: ordered fs ops to bound a live log of `size`. [] = nothing.
// Each step is ['rm', path] or ['mv', src, dst]; executed best-effort so a
// missing chain link never aborts the rest.
function planDesktopLogRotation(size) {
  if (size < DESKTOP_LOG_MAX_BYTES) return []
  const backups = n => Array.from({ length: n }, (_, i) => desktopLogBackupPath(i + 1))
  // Pathological boot-loop log: reclaim live + every backup outright.
  if (size > DESKTOP_LOG_DISCARD_BYTES) {
    return [DESKTOP_LOG_PATH, ...backups(DESKTOP_LOG_BACKUP_COUNT)].map(p => ['rm', p])
  }
  // Cascade: drop oldest, shift each up, live -> .1.
  const ops = [['rm', desktopLogBackupPath(DESKTOP_LOG_BACKUP_COUNT)]]
  for (let i = DESKTOP_LOG_BACKUP_COUNT - 1; i >= 1; i--) {
    ops.push(['mv', desktopLogBackupPath(i), desktopLogBackupPath(i + 1)])
  }
  ops.push(['mv', DESKTOP_LOG_PATH, desktopLogBackupPath(1)])
  return ops
}

function rotateDesktopLogIfNeededSync() {
  let size
  try {
    size = fs.statSync(DESKTOP_LOG_PATH).size
  } catch {
    return // No live file yet — the append (re)creates it.
  }
  for (const [op, src, dst] of planDesktopLogRotation(size)) {
    try {
      if (op === 'rm') fs.rmSync(src, { force: true })
      else fs.renameSync(src, dst)
    } catch {
      // Best-effort — logging must never block startup/shutdown.
    }
  }
}

async function rotateDesktopLogIfNeededAsync() {
  let size
  try {
    size = (await fs.promises.stat(DESKTOP_LOG_PATH)).size
  } catch {
    return // No live file yet — the append (re)creates it.
  }
  for (const [op, src, dst] of planDesktopLogRotation(size)) {
    try {
      if (op === 'rm') await fs.promises.rm(src, { force: true })
      else await fs.promises.rename(src, dst)
    } catch {
      // Best-effort — logging must never crash the shell.
    }
  }
}

function flushDesktopLogBufferSync() {
  if (!desktopLogBuffer) return
  const chunk = desktopLogBuffer
  desktopLogBuffer = ''

  try {
    fs.mkdirSync(path.dirname(DESKTOP_LOG_PATH), { recursive: true })
    rotateDesktopLogIfNeededSync()
    fs.appendFileSync(DESKTOP_LOG_PATH, chunk)
  } catch {
    // Logging must never block app startup/shutdown.
  }
}

function flushDesktopLogBufferAsync() {
  if (!desktopLogBuffer) return desktopLogFlushPromise
  const chunk = desktopLogBuffer
  desktopLogBuffer = ''

  desktopLogFlushPromise = desktopLogFlushPromise
    .then(async () => {
      await fs.promises.mkdir(path.dirname(DESKTOP_LOG_PATH), { recursive: true })
      await rotateDesktopLogIfNeededAsync()
      await fs.promises.appendFile(DESKTOP_LOG_PATH, chunk)
    })
    .catch(() => {
      // Logging must never crash the desktop shell.
    })

  return desktopLogFlushPromise
}

function scheduleDesktopLogFlush() {
  if (desktopLogFlushTimer) return
  desktopLogFlushTimer = setTimeout(() => {
    desktopLogFlushTimer = null
    void flushDesktopLogBufferAsync()
  }, DESKTOP_LOG_FLUSH_MS)
}

function rememberLog(chunk) {
  const text = String(chunk || '').trim()
  if (!text) return
  const lines = text.split(/\r?\n/).map(line => `[hermes] ${line}`)
  hermesLog.push(...lines)
  if (hermesLog.length > 300) {
    hermesLog.splice(0, hermesLog.length - 300)
  }

  desktopLogBuffer += `${lines.join('\n')}\n`

  if (desktopLogBuffer.length >= DESKTOP_LOG_BUFFER_MAX_CHARS) {
    if (desktopLogFlushTimer) {
      clearTimeout(desktopLogFlushTimer)
      desktopLogFlushTimer = null
    }
    void flushDesktopLogBufferAsync()

    return
  }

  scheduleDesktopLogFlush()
}

function openExternalUrl(rawUrl) {
  const raw = String(rawUrl || '').trim()
  if (!raw) return false

  let parsed
  try {
    parsed = new URL(raw)
  } catch {
    return false
  }

  // `file://` URLs come from the artifacts panel (the renderer can't open
  // them itself because Chromium blocks file:// navigation from the app
  // origin). Hand them to `shell.openPath`, which dispatches to the OS
  // file association. If the OS can't open it (`error` is a non-empty
  // string), fall back to revealing the file in the system file manager.
  if (parsed.protocol === 'file:') {
    let localPath
    try {
      localPath = resolveRequestedPathForIpc(parsed.toString(), { purpose: 'Open external file' })
    } catch {
      return false
    }

    void shell
      .openPath(localPath)
      .then(error => {
        if (!error) {
          return
        }

        rememberLog(`[file] openPath failed: ${error}; revealing in folder instead`)

        try {
          shell.showItemInFolder(localPath)
        } catch (revealError) {
          rememberLog(`[file] showItemInFolder failed: ${revealError.message}`)
        }
      })
      .catch(error => rememberLog(`[file] openPath rejected: ${error.message}`))

    return true
  }

  if (!['http:', 'https:', 'mailto:'].includes(parsed.protocol)) {
    return false
  }

  const url = parsed.toString()

  if (IS_WSL) {
    rememberLog(`[link] opening via WSL→Windows: ${url}`)
    const proc = spawn('cmd.exe', ['/c', 'start', '""', url], {
      detached: true,
      stdio: 'ignore',
      windowsHide: true
    })
    proc.on('error', error => {
      rememberLog(`[link] cmd.exe start failed: ${error.message}; falling back to xdg-open`)
      shell.openExternal(url).catch(fallback => rememberLog(`[link] xdg-open failed: ${fallback.message}`))
    })
    proc.unref()

    return true
  }

  shell.openExternal(url).catch(error => rememberLog(`[link] openExternal failed: ${error.message}`))

  return true
}

function ensureWslWindowsFonts() {
  if (!IS_WSL) return

  const fontsDir = ['/mnt/c/Windows/Fonts', '/mnt/c/windows/fonts'].find(candidate => {
    try {
      return fs.statSync(candidate).isDirectory()
    } catch {
      return false
    }
  })
  if (!fontsDir) return

  try {
    const confDir = path.join(app.getPath('home'), '.config', 'fontconfig', 'conf.d')
    const confPath = path.join(confDir, '99-hermes-wsl-windows-fonts.conf')
    let existing = ''
    try {
      existing = fs.readFileSync(confPath, 'utf8')
    } catch {
      existing = ''
    }
    if (existing.includes(fontsDir)) return

    fs.mkdirSync(confDir, { recursive: true })
    fs.writeFileSync(
      confPath,
      `<?xml version="1.0"?>\n<!DOCTYPE fontconfig SYSTEM "fonts.dtd">\n<fontconfig>\n  <dir>${fontsDir}</dir>\n</fontconfig>\n`
    )
    rememberLog(`[fonts] wired WSL Windows fonts for renderer: ${fontsDir}`)

    const cache = spawn('fc-cache', ['-f', fontsDir], { detached: true, stdio: 'ignore' })
    cache.on('error', () => undefined)
    cache.unref()
  } catch (error) {
    rememberLog(`[fonts] WSL font setup skipped: ${error.message}`)
  }
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

function clampBootProgress(value) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return 0
  return Math.max(0, Math.min(100, Math.round(numeric)))
}

function broadcastBootProgress() {
  if (!mainWindow || mainWindow.isDestroyed()) return
  const { webContents } = mainWindow
  if (!webContents || webContents.isDestroyed()) return
  webContents.send('hermes:boot-progress', bootProgressState)
}

// Bootstrap-event broadcast channel + state. The bootstrap runner emits a
// stream of events (manifest, stage, log, complete, failed) that the renderer
// install overlay subscribes to. We also keep a running snapshot:
//   - manifest: the stage list (rendered as a checklist in the overlay)
//   - stages:   per-stage state ('pending' | 'running' | 'succeeded' |
//               'skipped' | 'failed') keyed by stage name
//   - active:   true while a bootstrap is in flight; false otherwise
//   - error:    last 'failed' event's error message
//   - log:      bounded ring buffer of the last 200 log lines for the
//               "Show details" affordance in the overlay
//
// The snapshot is queryable via the hermes:bootstrap:get IPC handler so a
// reloaded renderer (e.g. devtools reload during dev) recovers state.
// Bootstrap log ring: bounded buffer so a long install (npm + playwright
// downloads can emit thousands of lines) doesn't grow unbounded in memory
// AND so the renderer's getBootstrapState() reply stays a reasonable size.
// We keep enough to cover an entire failed stage's transcript so the
// 'Copy output' button gives the user actually-actionable context, not
// just the last few lines.
const BOOTSTRAP_LOG_RING_MAX = 500
let bootstrapState = {
  active: false,
  manifest: null,
  stages: {},
  error: null,
  log: [],
  startedAt: null,
  completedAt: null,
  unsupportedPlatform: null
}

function broadcastBootstrapEvent(ev) {
  if (ev.type === 'manifest') {
    bootstrapState.manifest = ev
    bootstrapState.active = true
    bootstrapState.startedAt = bootstrapState.startedAt || Date.now()
    bootstrapState.stages = {}
    for (const stage of ev.stages || []) {
      bootstrapState.stages[stage.name] = { state: 'pending', json: null, durationMs: null, error: null }
    }
  } else if (ev.type === 'stage') {
    bootstrapState.stages[ev.name] = {
      state: ev.state,
      durationMs: ev.durationMs ?? null,
      json: ev.json ?? null,
      error: ev.error ?? null
    }
  } else if (ev.type === 'log') {
    bootstrapState.log.push({ ts: Date.now(), stage: ev.stage || null, line: ev.line, stream: ev.stream || 'stdout' })
    if (bootstrapState.log.length > BOOTSTRAP_LOG_RING_MAX) {
      bootstrapState.log.splice(0, bootstrapState.log.length - BOOTSTRAP_LOG_RING_MAX)
    }
  } else if (ev.type === 'complete') {
    bootstrapState.active = false
    bootstrapState.completedAt = Date.now()
    bootstrapState.error = null
    bootstrapState.unsupportedPlatform = null
  } else if (ev.type === 'failed') {
    bootstrapState.active = false
    bootstrapState.error = ev.error || 'unknown error'
  } else if (ev.type === 'unsupported-platform') {
    bootstrapState.active = false
    bootstrapState.unsupportedPlatform = {
      platform: ev.platform,
      activeRoot: ev.activeRoot,
      installCommand: ev.installCommand,
      docsUrl: ev.docsUrl
    }
  }

  if (!mainWindow || mainWindow.isDestroyed()) return
  const { webContents } = mainWindow
  if (!webContents || webContents.isDestroyed()) return
  webContents.send('hermes:bootstrap:event', ev)
}

function getBootstrapState() {
  return bootstrapState
}

function updateBootProgress(update, options = {}) {
  const nextProgressRaw =
    typeof update.progress === 'number' ? clampBootProgress(update.progress) : bootProgressState.progress
  const nextProgress = options.allowDecrease ? nextProgressRaw : Math.max(bootProgressState.progress, nextProgressRaw)

  bootProgressState = {
    ...bootProgressState,
    ...update,
    error: update.error === undefined ? bootProgressState.error : update.error,
    fakeMode: BOOT_FAKE_MODE || Boolean(update.fakeMode),
    progress: nextProgress,
    timestamp: Date.now()
  }

  if (update.message) {
    rememberLog(`[boot] ${update.message}`)
  }

  broadcastBootProgress()
}

async function advanceBootProgress(phase, message, progress) {
  updateBootProgress({
    phase,
    message,
    progress,
    running: true,
    error: null
  })

  if (BOOT_FAKE_MODE) {
    await sleep(BOOT_FAKE_STEP_MS)
  }
}

function fileExists(filePath) {
  try {
    return fs.statSync(filePath).isFile()
  } catch {
    return false
  }
}

function directoryExists(filePath) {
  try {
    return fs.statSync(filePath).isDirectory()
  } catch {
    return false
  }
}

function unpackedPathFor(filePath) {
  return filePath.replace(/app\.asar(?=$|[\\/])/, 'app.asar.unpacked')
}

function findOnPath(command) {
  if (!command) return null

  if (path.isAbsolute(command) || command.includes(path.sep) || (IS_WINDOWS && command.includes('/'))) {
    if (!fileExists(command)) return null
    if (isWindowsBinaryPathInWsl(command, { isWsl: IS_WSL })) return null
    return command
  }

  const pathEntries = String(process.env.PATH || '')
    .split(path.delimiter)
    .filter(Boolean)
  const extensions = IS_WINDOWS
    ? ['', ...(process.env.PATHEXT || '.COM;.EXE;.BAT;.CMD').split(';').filter(Boolean)]
    : ['']

  for (const entry of pathEntries) {
    for (const extension of extensions) {
      const candidate = path.join(entry, `${command}${extension}`)
      if (fileExists(candidate)) return candidate
    }
  }

  return null
}

function isCommandScript(command) {
  return IS_WINDOWS && /\.(cmd|bat)$/i.test(command || '')
}

function normalizeExecutablePathForCompare(commandPath) {
  if (!commandPath) return null

  let resolved = path.resolve(String(commandPath))
  try {
    resolved = fs.realpathSync.native ? fs.realpathSync.native(resolved) : fs.realpathSync(resolved)
  } catch {
    // Fallback to path.resolve() above.
  }

  return IS_WINDOWS ? resolved.toLowerCase() : resolved
}

function looksLikeDesktopAppBinary(commandPath) {
  if (!IS_WINDOWS || !commandPath) return false

  const normalizedCandidate = normalizeExecutablePathForCompare(commandPath)
  const normalizedCurrentExec = normalizeExecutablePathForCompare(process.execPath)
  if (normalizedCandidate && normalizedCurrentExec && normalizedCandidate === normalizedCurrentExec) {
    return true
  }

  let resolved = path.resolve(String(commandPath))
  try {
    resolved = fs.realpathSync.native ? fs.realpathSync.native(resolved) : fs.realpathSync(resolved)
  } catch {
    // Keep resolved path fallback.
  }

  const resourcesDir = path.join(path.dirname(resolved), 'resources')
  return (
    fileExists(path.join(resourcesDir, 'app.asar')) || directoryExists(path.join(resourcesDir, 'app.asar.unpacked'))
  )
}

function isHermesSourceRoot(root) {
  return directoryExists(root) && fileExists(path.join(root, 'hermes_cli', 'main.py'))
}

function findPythonForRoot(root) {
  const override = process.env.HERMES_DESKTOP_PYTHON
  if (override && fileExists(override)) return override

  const relativePaths = IS_WINDOWS
    ? [path.join('.venv', 'Scripts', 'python.exe'), path.join('venv', 'Scripts', 'python.exe')]
    : [path.join('.venv', 'bin', 'python'), path.join('venv', 'bin', 'python')]

  for (const relativePath of relativePaths) {
    const candidate = path.join(root, relativePath)
    if (fileExists(candidate)) return candidate
  }

  return findSystemPython()
}

function findSystemPython() {
  if (!IS_WINDOWS) {
    // POSIX systems: PATH lookup is safe.
    for (const command of ['python3', 'python']) {
      const candidate = findOnPath(command)
      if (candidate) return candidate
    }
    return null
  }

  // Windows: PATH-based detection has TWO landmines we have to dodge.
  //
  //  (1) The Microsoft Store "Python stub" lives at
  //      %LOCALAPPDATA%\Microsoft\WindowsApps\python.exe and is on PATH
  //      by default on modern Windows. It's a redirector that opens the
  //      Store window if no Store Python is installed. Running it for
  //      `-m venv` would either succeed (real Store install — fine) or
  //      pop the Store dialog (bad UX during boot).
  //  (2) `py.exe` (Python launcher) is missing from per-user installs
  //      that didn't check the launcher option, so PATH-only checks
  //      miss real Python 3.13 installs (user-reported case).
  //
  // We also restrict ourselves to Python 3.11–3.13. 3.14 is the latest
  // CPython but several Hermes deps (notably pywinpty's Rust-built
  // windows_x86_64_msvc crate) don't yet publish 3.14 wheels, and
  // `pip install -e .` falls back to source-build, which fails without
  // a Rust toolchain. install.ps1 sidesteps this by pinning to 3.11
  // via uv; until we add the same uv-managed Python pathway here, the
  // simplest fix is to refuse 3.14 detection and let the NSIS prereq
  // page offer to install 3.11 alongside.
  //
  // Strategy: probe in three passes, in order from most-precise to
  // least-precise, and ONLY use PATH lookup as a last resort after
  // confirming the candidate isn't the WindowsApps redirector.
  //
  //  Pass 1: PEP 514 registry — every standards-compliant Python
  //          installer registers itself at SOFTWARE\Python\PythonCore.
  //          The MS Store stub does NOT register here, so a hit means
  //          a real Python install. Versions are explicit so we
  //          inherently filter 3.14 out.
  //  Pass 2: Filesystem probe of standard install locations
  //          (Program Files, LocalAppData\Programs\Python). Same
  //          version filtering by directory name.
  //  Pass 3: PATH lookup of `py.exe` (the launcher itself never
  //          triggers the Store) — but call it with a version flag so
  //          we resolve to a SPECIFIC supported version, not whatever
  //          py.exe's default is (which on a 3.14-only box would be
  //          3.14).

  const SUPPORTED_VERSIONS = ['3.11', '3.12', '3.13']
  const SUPPORTED_VERSIONS_NO_DOT = ['311', '312', '313']

  // Pass 1: registry. Use `reg query` since main process doesn't have
  // a reliable in-process registry API across all electron versions.
  for (const hive of ['HKLM', 'HKCU']) {
    for (const version of SUPPORTED_VERSIONS) {
      try {
        const out = execFileSync(
          'reg',
          ['query', `${hive}\\SOFTWARE\\Python\\PythonCore\\${version}\\InstallPath`, '/ve', '/reg:64'],
          hiddenWindowsChildOptions({ encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] })
        )
        // Output format: "    (Default)    REG_SZ    C:\Path\To\Python\"
        const match = out.match(/REG_SZ\s+(.+?)\s*$/m)
        if (match) {
          const installPath = match[1].trim()
          const pythonExe = path.join(installPath, 'python.exe')
          if (fileExists(pythonExe)) return pythonExe
        }
      } catch {
        // Key not present — try next.
      }
    }
  }

  // Pass 2: filesystem probe of standard locations.
  const programFiles = process.env['ProgramFiles'] || 'C:\\Program Files'
  const localAppData = process.env.LOCALAPPDATA || ''
  for (const versionDir of SUPPORTED_VERSIONS_NO_DOT) {
    const systemWide = path.join(programFiles, `Python${versionDir}`, 'python.exe')
    if (fileExists(systemWide)) return systemWide
    if (localAppData) {
      const perUser = path.join(localAppData, 'Programs', 'Python', `Python${versionDir}`, 'python.exe')
      if (fileExists(perUser)) return perUser
    }
  }

  // Pass 3: py.exe with explicit version flag. The launcher itself is
  // safe to invoke (no Store popup) and `py -3.13 -c "import sys;
  // print(sys.executable)"` resolves to the actual python.exe path of
  // the requested version. We try in version-priority order so the
  // first hit wins.
  const pyExe = findOnPath('py.exe')
  if (pyExe) {
    for (const version of SUPPORTED_VERSIONS) {
      try {
        const out = execFileSync(
          pyExe,
          [`-${version}`, '-c', 'import sys; print(sys.executable)'],
          hiddenWindowsChildOptions({
            encoding: 'utf8',
            stdio: ['ignore', 'pipe', 'ignore']
          })
        )
        const candidate = out.trim()
        if (candidate && fileExists(candidate)) return candidate
      } catch {
        // py couldn't find that version — try next.
      }
    }
  }

  // We deliberately do NOT fall back to plain `python.exe` on PATH.
  // Without a way to verify the version safely (running `python -V`
  // risks the Microsoft Store popup), accepting whatever's there
  // could land us on 3.14 and trigger the Rust-build-from-source
  // failure. Better to return null and let the NSIS prereq page
  // offer to install a known-good 3.11 via winget.
  return null
}

// findGitBash — locate bash.exe on Windows. Hermes' terminal tool requires
// bash (POSIX shell), and on Windows that's almost always Git for Windows'
// bundled Git Bash. We check the same set of locations tools/environments/
// local.py:_find_bash() checks at runtime, so a positive result here means
// the agent will be able to start a terminal too.
//
// On non-Windows hosts bash is part of the OS and this just returns the
// first bash on PATH.
function findGitBash() {
  if (!IS_WINDOWS) {
    return findOnPath('bash')
  }

  // install.ps1 drops PortableGit at %LOCALAPPDATA%\apexnodes\git\... — checked
  // first so users who installed via install.ps1 are detected before we
  // start probing system-wide locations.
  const localAppData = process.env.LOCALAPPDATA || ''
  const candidates = []
  if (localAppData) {
    candidates.push(path.join(localAppData, 'apexnodes', 'git', 'bin', 'bash.exe'))
    candidates.push(path.join(localAppData, 'apexnodes', 'git', 'usr', 'bin', 'bash.exe'))
  }

  // Standard Git for Windows install locations.
  candidates.push(path.join(process.env['ProgramFiles'] || 'C:\\Program Files', 'Git', 'bin', 'bash.exe'))
  candidates.push(path.join(process.env['ProgramFiles(x86)'] || 'C:\\Program Files (x86)', 'Git', 'bin', 'bash.exe'))
  if (localAppData) {
    candidates.push(path.join(localAppData, 'Programs', 'Git', 'bin', 'bash.exe'))
  }

  for (const candidate of candidates) {
    if (fileExists(candidate)) return candidate
  }

  // Last resort — bash on PATH (covers WSL bash, MSYS2, custom installs).
  // On WSL hosts findOnPath itself filters out Windows-binary paths via
  // isWindowsBinaryPathInWsl, so we won't hand back a wsl.exe shim either.
  return findOnPath('bash')
}

function getVenvPython(venvRoot) {
  return path.join(venvRoot, IS_WINDOWS ? path.join('Scripts', 'python.exe') : path.join('bin', 'python'))
}

// resolveGitBinary — locate git.exe on Windows. A fresh installer-driven
// install only has PortableGit under %LOCALAPPDATA%\apexnodes\git (never on
// PATH), so a bare spawn('git') ENOENTs and self-update checks fail with
// "Couldn't check for updates". Mirror findGitBash: PortableGit first, then
// standard Git-for-Windows locations, then PATH. Cached after first probe.
let _gitBinaryCache = null
function resolveGitBinary() {
  if (_gitBinaryCache) return _gitBinaryCache
  if (!IS_WINDOWS) {
    _gitBinaryCache = findOnPath('git') || 'git'
    return _gitBinaryCache
  }

  const localAppData = process.env.LOCALAPPDATA || ''
  const candidates = []
  if (localAppData) {
    candidates.push(path.join(localAppData, 'apexnodes', 'git', 'cmd', 'git.exe'))
    candidates.push(path.join(localAppData, 'apexnodes', 'git', 'bin', 'git.exe'))
  }
  candidates.push(path.join(process.env['ProgramFiles'] || 'C:\\Program Files', 'Git', 'cmd', 'git.exe'))
  candidates.push(path.join(process.env['ProgramFiles(x86)'] || 'C:\\Program Files (x86)', 'Git', 'cmd', 'git.exe'))
  if (localAppData) {
    candidates.push(path.join(localAppData, 'Programs', 'Git', 'cmd', 'git.exe'))
  }

  _gitBinaryCache = candidates.find(fileExists) || findOnPath('git') || 'git'
  return _gitBinaryCache
}

function recentHermesLog() {
  return hermesLog.slice(-20).join('\n')
}

// ─── Self-update (git-pull against the running backend's hermes root) ──────

function readDesktopUpdateConfig() {
  try {
    const parsed = JSON.parse(fs.readFileSync(DESKTOP_UPDATE_CONFIG_PATH, 'utf8'))
    const branch = typeof parsed?.branch === 'string' ? parsed.branch.trim() : ''
    return { branch: branch || DEFAULT_UPDATE_BRANCH }
  } catch {
    return { branch: DEFAULT_UPDATE_BRANCH }
  }
}

// Atomic file write: temp + rename (atomic on all platforms). Prevents
// partial writes on crash/power loss that corrupt JSON config files.
function writeFileAtomic(targetPath, data, encoding) {
  const tmp = targetPath + '.tmp'
  fs.writeFileSync(tmp, data, encoding)
  fs.renameSync(tmp, targetPath)
}

function writeDesktopUpdateConfig(config) {
  fs.mkdirSync(path.dirname(DESKTOP_UPDATE_CONFIG_PATH), { recursive: true })
  writeFileAtomic(DESKTOP_UPDATE_CONFIG_PATH, JSON.stringify(config, null, 2))
}

// Match the backend's source resolution but bias toward a real git checkout.
// Dev → SOURCE_REPO_ROOT. Packaged/CLI install → ACTIVE_HERMES_ROOT.
// HERMES_DESKTOP_HERMES_ROOT always wins so devs can pin a worktree.
function resolveUpdateRoot() {
  const candidates = [
    process.env.HERMES_DESKTOP_HERMES_ROOT && path.resolve(process.env.HERMES_DESKTOP_HERMES_ROOT),
    !IS_PACKAGED && isHermesSourceRoot(SOURCE_REPO_ROOT) ? SOURCE_REPO_ROOT : null,
    isHermesSourceRoot(ACTIVE_HERMES_ROOT) ? ACTIVE_HERMES_ROOT : null
  ].filter(Boolean)

  return candidates.find(c => directoryExists(path.join(c, '.git'))) || candidates[0] || ACTIVE_HERMES_ROOT
}

function runGit(args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(
      resolveGitBinary(),
      IS_WINDOWS ? ['-c', 'windows.appendAtomically=false', ...args] : args,
      hiddenWindowsChildOptions({
        cwd: options.cwd,
        env: { ...process.env, ...(options.env || {}), GIT_TERMINAL_PROMPT: '0' },
        stdio: ['ignore', 'pipe', 'pipe']
      })
    )

    let stdout = ''
    let stderr = ''
    child.stdout.on('data', chunk => {
      const text = chunk.toString()
      stdout += text
      options.onLine?.('stdout', text)
    })
    child.stderr.on('data', chunk => {
      const text = chunk.toString()
      stderr += text
      options.onLine?.('stderr', text)
    })
    child.once('error', reject)
    child.once('exit', code => resolve({ code, stdout, stderr }))
  })
}

const firstLine = text => (text || '').split('\n').find(Boolean) || ''

async function getOriginUrl(updateRoot) {
  const origin = await runGit(['remote', 'get-url', 'origin'], { cwd: updateRoot })
  return origin.code === 0 ? origin.stdout.trim() : ''
}

function emitUpdateProgress(payload) {
  const merged = { stage: 'idle', message: '', percent: null, error: null, ...payload, at: Date.now() }
  rememberLog(`[updates] ${merged.stage}: ${merged.message || merged.error || ''}`)
  for (const window of BrowserWindow.getAllWindows()) {
    window.webContents.send('hermes:updates:progress', merged)
  }
}

// Self-heal the tracked update branch: if origin no longer publishes it (e.g.
// bb/gui was merged into main and deleted), fall back to main and persist so
// every later check/apply follows main — no manual flip, even for already-
// installed clients. Read-only ls-remote probe; only flips on a definitive
// "ref absent" (exit 2), never on a transient network error, so a flaky
// connection can't strand a user on the wrong branch.
async function resolveHealedBranch(updateRoot, branch) {
  if (!branch || branch === 'main') {
    return branch || 'main'
  }

  const originUrl = await getOriginUrl(updateRoot)
  const remote = isOfficialSshRemote(originUrl) ? OFFICIAL_REPO_HTTPS_URL : 'origin'
  const probe = await runGit(['ls-remote', '--exit-code', '--heads', remote, branch], { cwd: updateRoot })
  if (probe.code !== 2) {
    return branch
  }

  rememberLog(`[updates] origin/${branch} is gone (merged?); falling back to main`)
  const config = readDesktopUpdateConfig()
  if (config.branch !== 'main') {
    writeDesktopUpdateConfig({ ...config, branch: 'main' })
  }
  return 'main'
}

async function checkUpdates() {
  const updateRoot = resolveUpdateRoot()
  let { branch } = readDesktopUpdateConfig()
  const gitDir = path.join(updateRoot, '.git')
  if (!directoryExists(gitDir)) {
    return {
      supported: false,
      reason: 'not-a-git-checkout',
      message: `${updateRoot} isn't a git checkout — desktop self-update only runs against a source install.`,
      hermesRoot: updateRoot,
      branch
    }
  }

  branch = await resolveHealedBranch(updateRoot, branch)
  const originUrl = await getOriginUrl(updateRoot)
  if (isOfficialSshRemote(originUrl)) {
    const git = args => runGit(args, { cwd: updateRoot }).then(r => r.stdout.trim())
    const [currentSha, target, dirtyStr, currentBranch] = await Promise.all([
      git(['rev-parse', 'HEAD']),
      runGit(['ls-remote', OFFICIAL_REPO_HTTPS_URL, `refs/heads/${branch}`], { cwd: updateRoot }),
      git(['status', '--porcelain']),
      git(['rev-parse', '--abbrev-ref', 'HEAD'])
    ])
    const targetSha = firstLine(target.stdout).split(/\s+/)[0] || ''
    if (target.code !== 0 || !targetSha) {
      return {
        supported: true,
        branch,
        error: 'fetch-failed',
        message: firstLine(target.stderr) || 'git ls-remote failed.',
        hermesRoot: updateRoot,
        fetchedAt: Date.now()
      }
    }
    return {
      supported: true,
      branch,
      currentBranch,
      behind: currentSha && currentSha === targetSha ? 0 : 1,
      currentSha,
      targetSha,
      commits: [],
      dirty: dirtyStr.length > 0,
      hermesRoot: updateRoot,
      fetchedAt: Date.now()
    }
  }

  const fetched = await runGit(['fetch', '--quiet', 'origin', branch], { cwd: updateRoot })
  if (fetched.code !== 0) {
    return {
      supported: true,
      branch,
      error: 'fetch-failed',
      message: firstLine(fetched.stderr) || 'git fetch failed.',
      hermesRoot: updateRoot,
      fetchedAt: Date.now()
    }
  }

  const git = args => runGit(args, { cwd: updateRoot }).then(r => r.stdout.trim())
  const [currentSha, targetSha, countStr, dirtyStr, currentBranch] = await Promise.all([
    git(['rev-parse', 'HEAD']),
    git(['rev-parse', `origin/${branch}`]),
    git(['rev-list', `HEAD..origin/${branch}`, '--count']),
    git(['status', '--porcelain']),
    git(['rev-parse', '--abbrev-ref', 'HEAD'])
  ])

  const behind = Number.parseInt(countStr, 10) || 0
  const commits = behind > 0 ? await readCommitLog(updateRoot, branch) : []

  return {
    supported: true,
    branch,
    currentBranch,
    behind,
    currentSha,
    targetSha,
    commits,
    dirty: dirtyStr.length > 0,
    hermesRoot: updateRoot,
    fetchedAt: Date.now()
  }
}

async function readCommitLog(cwd, branch) {
  const SEP = '\x1f'
  const REC = '\x1e'
  const { stdout } = await runGit(
    ['log', `HEAD..origin/${branch}`, `--pretty=format:%H${SEP}%s${SEP}%an${SEP}%at${REC}`, '-n', '40'],
    { cwd }
  )

  return stdout
    .split(REC)
    .map(line => line.trim())
    .filter(Boolean)
    .map(line => {
      const [sha, summary, author, at] = line.split(SEP)
      return { sha, summary, author, at: Number.parseInt(at, 10) * 1000 }
    })
}

let updateInFlight = false

// Resolve the staged updater binary. The Tauri installer copies itself to
// HERMES_HOME/hermes-setup.exe on a successful install (see
// apps/bootstrap-installer paths::copy_self_to_hermes_home). That binary owns
// ALL repo mutation — running `hermes update` + rebuilding the desktop — so
// the desktop never touches its own bits while running. Returns null when the
// updater isn't staged (e.g. a dev/source run that never went through the
// installer); callers degrade gracefully.
function resolveUpdaterBinary() {
  const name = IS_WINDOWS ? 'hermes-setup.exe' : 'hermes-setup'
  const candidate = path.join(HERMES_HOME, name)
  return fileExists(candidate) ? candidate : null
}

function repairMacUpdaterHelper(updater) {
  if (!IS_MAC || !updater) return

  try {
    execFileSync('/usr/bin/xattr', ['-cr', updater], { stdio: 'ignore' })
  } catch (err) {
    rememberLog(`[updates] macOS updater helper quarantine repair skipped: ${err.message}`)
  }

  try {
    execFileSync('/usr/bin/codesign', ['--verify', updater], { stdio: 'ignore' })
    return
  } catch {
    // Unsigned or invalid helper. Apply a local ad-hoc signature so Gatekeeper
    // does not block the staged updater before it can run.
  }

  try {
    execFileSync('/usr/bin/codesign', ['--force', '--sign', '-', updater], { stdio: 'ignore' })
    rememberLog('[updates] repaired macOS updater helper signature')
  } catch (err) {
    rememberLog(`[updates] macOS updater helper signature repair skipped: ${err.message}`)
  }
}

// Path to the venv shim whose lock decides whether `hermes update` can write
// fresh entry points. On Windows this is the file the running backend
// `hermes.exe` holds open; on POSIX it's never mandatory-locked.
function venvHermesShimPath(updateRoot) {
  return IS_WINDOWS
    ? path.join(updateRoot, 'venv', 'Scripts', 'hermes.exe')
    : path.join(updateRoot, 'venv', 'bin', 'hermes')
}

// Best-effort lock probe mirroring the Rust updater's is_locked(): a running
// .exe on Windows refuses an O_RDWR open with a sharing violation. On POSIX
// this practically always succeeds (no mandatory locking), so it returns false
// — correct, since the shim-contention brick is Windows-only.
function isShimLocked(shimPath) {
  if (!IS_WINDOWS) return false
  let fd
  try {
    fd = fs.openSync(shimPath, 'r+')
    return false
  } catch (err) {
    // ENOENT ⇒ not there ⇒ nothing locking it. Anything else (EBUSY/EPERM/
    // EACCES) on Windows means a live handle holds it.
    return err && err.code !== 'ENOENT'
  } finally {
    if (fd !== undefined) {
      try {
        fs.closeSync(fd)
      } catch {
        void 0
      }
    }
  }
}

// Force-kill the entire process TREE rooted at each PID. Node's child.kill()
// only signals the direct child, so on Windows a backend `hermes.exe` that
// spawned its own grandchildren (a `hermes` REPL, a pty terminal session, the
// gateway) would survive and keep the venv shim locked. taskkill /T /F reaps
// the whole tree synchronously. Windows-only: this is called solely from the
// Windows shim-unlock path, and the backend is NOT spawned detached (so it's
// not a process-group leader — a POSIX negative-pgid kill would be meaningless
// here anyway). POSIX teardown stays with the existing before-quit SIGTERM.
function forceKillProcessTree(pid) {
  if (!IS_WINDOWS) return
  if (!Number.isInteger(pid) || pid <= 0) return
  try {
    execFileSync('taskkill', ['/PID', String(pid), '/T', '/F'], hiddenWindowsChildOptions({ stdio: 'ignore' }))
  } catch {
    // Already gone, or no permission — best effort; the unlock wait below is
    // the real gate.
  }
}

// Before handing off the update on Windows, the desktop MUST stop every backend
// it spawned and WAIT for the venv shim to actually unlock. The old code did
// `hermesProcess.kill('SIGTERM')` + `app.quit()` fire-and-forget: SIGTERM on
// Windows doesn't reap the backend's grandchildren, and quit didn't wait for
// teardown, so the updater raced a still-locked `hermes.exe`, the quarantine
// rename failed, uv's `pip install` hit "Access is denied", and the git path
// bailed into a full ZIP re-download that ALSO couldn't write the locked shim —
// a half-applied install (ryanc's update.log). Here we tree-kill the primary +
// pool backends and poll the shim until it's writable (or a bounded timeout),
// so by the time we spawn the updater the lock is genuinely gone.
//
// Windows-only: the venv-shim mandatory lock is a Windows phenomenon. On
// macOS/Linux there's no REPLACE-on-running-exe block, the existing before-quit
// SIGTERM + app.quit() teardown already works (the macOS path is flawless), and
// aggressively SIGKILL-ing the backend here would be an untested behavior change
// for no benefit. So we no-op off Windows and leave that path exactly as it was.
async function releaseBackendLockForUpdate(updateRoot) {
  return releaseBackendLock(updateRoot, 'updates')
}

// Shared backend teardown + venv-shim unlock wait. Used by BOTH the self-update
// hand-off and the desktop uninstaller — they have the identical Windows
// problem: the desktop's backend (and the grandchildren IT spawned — a hermes
// REPL, a pty terminal, the gateway) keep `hermes.exe` and other files in the
// venv mandatory-locked, so any in-place replace/delete of the install tree
// races a live handle and half-fails (#37532). We tree-kill every backend PID
// the desktop owns, then poll the shim until it's genuinely writable.
//
// `tag` only flavors the log lines. No-op off Windows (POSIX has no mandatory
// locks — the before-quit SIGTERM + the cleanup script's own PID-wait suffice).
async function releaseBackendLock(updateRoot, tag) {
  if (!IS_WINDOWS) return { unlocked: true }

  // Collect every backend PID the desktop owns: primary window backend + pool.
  const pids = []
  if (hermesProcess && Number.isInteger(hermesProcess.pid)) pids.push(hermesProcess.pid)
  for (const entry of backendPool.values()) {
    if (entry.process && Number.isInteger(entry.process.pid)) pids.push(entry.process.pid)
  }

  // Graceful first (lets Python flush), then tree-kill to catch grandchildren.
  if (hermesProcess && !hermesProcess.killed) {
    try {
      hermesProcess.kill('SIGTERM')
    } catch {
      void 0
    }
  }
  stopAllPoolBackends()
  for (const pid of pids) forceKillProcessTree(pid)

  const shim = venvHermesShimPath(updateRoot)
  const deadlineMs = Date.now() + 15000
  while (Date.now() < deadlineMs) {
    if (!isShimLocked(shim)) {
      rememberLog(`[${tag}] venv shim unlocked; safe to proceed`)
      return { unlocked: true }
    }
    await new Promise(r => setTimeout(r, 300))
  }
  rememberLog(`[${tag}] venv shim still locked after 15s; proceeding anyway (force)`)
  return { unlocked: false }
}

// applyUpdates — hand off to the installer's --update flow, then exit.
//
// The desktop is a pure consumer: it does NOT git pull / pip install / rebuild
// itself (the old open-coded git dance lived here and drifted from
// `hermes update`). Instead we spawn the staged Hermes-Setup binary with
// --update and quit, so it can run `hermes update` (which refuses while we
// hold the venv shim) and rebuild the desktop with our exe already gone.
//
// Detection (checkUpdates / commit changelog / "N behind") stays in the UI;
// only this apply action changed.
async function applyUpdates(opts = {}) {
  if (updateInFlight) {
    throw new Error('An update is already in progress.')
  }
  updateInFlight = true

  try {
    const updater = resolveUpdaterBinary()
    if (!updater && !IS_WINDOWS) {
      // macOS/Linux drag-install: no staged Tauri hermes-setup. Unlike Windows
      // (where a venv-shim file lock forces the quit→hand-off→rebuild dance),
      // there's no mandatory file locking here, so the desktop can drive the
      // whole update itself: `hermes update` (backend) + `hermes desktop
      // --build-only` (OS-aware GUI rebuild), then swap the running .app bundle
      // with the freshly built one and relaunch.
      return await applyUpdatesPosixInApp(opts)
    }
    if (!updater) {
      // No staged updater binary — this is a CLI-installed user (they ran
      // `hermes desktop`, never the Tauri installer that self-copies
      // hermes-setup.exe into HERMES_HOME). They DO have a working `hermes`
      // on PATH / in the venv, so the correct path is the one-liner in their
      // native medium. We show the EXACT command, branch-pinned to the
      // checkout they're on — bare `hermes update` defaults to main and would
      // silently switch a bb/gui (or any non-main) install off-branch. Mirror
      // the GUI button's contract: append --branch <current> for non-main
      // checkouts, keep it bare for main so the card stays clean.
      const updateRoot = resolveUpdateRoot()
      let command = 'hermes update'
      try {
        const head = await runGit(['rev-parse', '--abbrev-ref', 'HEAD'], { cwd: updateRoot })
        const current = (head.stdout || '').trim()
        if (head.code === 0 && current && current !== 'HEAD') {
          const branch = await resolveHealedBranch(updateRoot, current)
          if (branch !== 'main') command = `hermes update --branch ${branch}`
        }
      } catch {
        // Best-effort: fall back to bare `hermes update` if branch detection fails.
      }
      rememberLog(`[updates] no staged updater; surfacing manual \`${command}\` for CLI install at ${updateRoot}`)
      emitUpdateProgress({ stage: 'manual', message: command, percent: null })
      return { ok: true, manual: true, command, hermesRoot: updateRoot }
    }

    emitUpdateProgress({ stage: 'restart', message: 'Handing off to the Hermes updater…', percent: 100 })
    repairMacUpdaterHelper(updater)

    const updateRoot = resolveUpdateRoot()
    const { branch: configuredBranch } = readDesktopUpdateConfig()
    const branch = await resolveHealedBranch(updateRoot, configuredBranch || DEFAULT_UPDATE_BRANCH)
    const updaterArgs = ['--update', '--branch', branch]
    const targetApp = IS_MAC ? runningAppBundle() : null
    if (targetApp) {
      updaterArgs.push('--target-app', targetApp)
    }
    const venvBin = path.join(updateRoot, 'venv', IS_WINDOWS ? 'Scripts' : 'bin')

    // Stop our own backend(s) and wait for the venv shim to unlock BEFORE we
    // spawn the updater. Without this the updater races a still-locked
    // hermes.exe (held by the backend child / its grandchildren) and the update
    // bricks. See releaseBackendLockForUpdate for the full failure analysis.
    await releaseBackendLockForUpdate(updateRoot)

    // Detached so the updater outlives this process — it needs us GONE before
    // `hermes update` will run (the venv shim is locked while we live).
    const child = spawn(updater, updaterArgs, {
      cwd: HERMES_HOME,
      env: {
        ...process.env,
        HERMES_HOME,
        PATH: [path.join(HERMES_HOME, 'node', 'bin'), venvBin, process.env.PATH].filter(Boolean).join(path.delimiter)
      },
      detached: true,
      stdio: 'ignore',
      windowsHide: false
    })
    child.unref()

    rememberLog(`[updates] launched updater: ${updater} ${updaterArgs.join(' ')}; exiting desktop to release venv shim`)

    // Give the OS a beat to register the new process, then quit. The updater
    // rebuilds and relaunches us when it's done.
    setTimeout(() => {
      app.quit()
    }, 600)

    return { ok: true, handedOff: true, updater }
  } finally {
    updateInFlight = false
  }
}

async function handOffWindowsBootstrapRecovery(reason) {
  if (!IS_WINDOWS || !IS_PACKAGED) return false

  const updater = resolveUpdaterBinary()
  if (!updater) return false

  const updateRoot = resolveUpdateRoot()
  const { branch: configuredBranch } = readDesktopUpdateConfig()
  const branch = directoryExists(path.join(updateRoot, '.git'))
    ? await resolveHealedBranch(updateRoot, configuredBranch || DEFAULT_UPDATE_BRANCH)
    : configuredBranch || DEFAULT_UPDATE_BRANCH
  const venvBin = path.join(updateRoot, 'venv', IS_WINDOWS ? 'Scripts' : 'bin')
  const venvHermes = path.join(venvBin, IS_WINDOWS ? 'hermes.exe' : 'hermes')
  const updaterArgs = fileExists(venvHermes) ? ['--update', '--branch', branch] : ['--repair', '--branch', branch]

  await releaseBackendLockForUpdate(updateRoot)

  const child = spawn(updater, updaterArgs, {
    cwd: HERMES_HOME,
    env: {
      ...process.env,
      HERMES_HOME,
      PATH: [path.join(HERMES_HOME, 'node', 'bin'), venvBin, process.env.PATH].filter(Boolean).join(path.delimiter)
    },
    detached: true,
    stdio: 'ignore',
    windowsHide: false
  })
  child.unref()

  rememberLog(`[bootstrap] handed off ${reason} recovery to updater: ${updater} ${updaterArgs.join(' ')}; exiting desktop to release app.asar`)
  setTimeout(() => {
    app.quit()
  }, 600)

  return true
}

// Resolve the hermes CLI to drive an in-app update: prefer the venv shim in
// the install we're updating, fall back to `hermes` on PATH.
function resolveHermesCliBinary(updateRoot) {
  const venvHermes = path.join(updateRoot, 'venv', 'bin', 'hermes')
  if (fileExists(venvHermes)) return venvHermes
  return findOnPath('hermes') || null
}

// Spawn a command and stream each output line to the update progress channel.
function runStreamedUpdate(command, args, { cwd, env, stage } = {}) {
  return new Promise(resolve => {
    let child
    try {
      child = spawn(
        command,
        args,
        hiddenWindowsChildOptions({
          cwd,
          env: { ...process.env, ...(env || {}) },
          stdio: ['ignore', 'pipe', 'pipe']
        })
      )
    } catch (err) {
      resolve({ code: 1, error: err.message })
      return
    }
    const emitLines = chunk => {
      for (const line of chunk.toString().split('\n')) {
        const trimmed = line.trim()
        if (trimmed) emitUpdateProgress({ stage, message: trimmed, percent: null })
      }
    }
    child.stdout.on('data', emitLines)
    child.stderr.on('data', emitLines)
    child.once('error', err => resolve({ code: 1, error: err.message }))
    child.once('exit', code => resolve({ code }))
  })
}

// The running app's .app bundle (packaged macOS): execPath is
// <App>.app/Contents/MacOS/<exe>; climb three levels to the bundle root.
function runningAppBundle() {
  if (!IS_MAC) return null
  let dir = path.dirname(app.getPath('exe')) // .../Contents/MacOS
  for (let i = 0; i < 2; i++) dir = path.dirname(dir) // -> .../X.app
  return dir.endsWith('.app') ? dir : null
}

function shellQuote(value) {
  return `'${String(value).replace(/'/g, `'\\''`)}'`
}

// macOS/Linux in-app update: backend (`hermes update`) + OS-aware GUI rebuild
// (`hermes desktop --build-only`), then atomically swap the running .app bundle
// with the freshly built one and relaunch. Degrades to "backend updated,
// restart to load the new GUI" if the swap can't be performed.
async function applyUpdatesPosixInApp() {
  const updateRoot = resolveUpdateRoot()
  const hermes = resolveHermesCliBinary(updateRoot)
  if (!hermes) {
    emitUpdateProgress({ stage: 'manual', message: 'hermes update', percent: null })
    return { ok: true, manual: true, command: 'hermes update', hermesRoot: updateRoot }
  }

  // Put the Hermes-managed Node and the venv on PATH so `hermes desktop`'s
  // npm build can find them on a machine with no system Node.
  const extraPath = [path.join(HERMES_HOME, 'node', 'bin'), path.join(updateRoot, 'venv', 'bin')]
    .filter(Boolean)
    .join(path.delimiter)
  const env = {
    HERMES_HOME,
    PATH: [extraPath, process.env.PATH].filter(Boolean).join(path.delimiter)
  }

  // `hermes update` reaps stale `hermes dashboard` backends (a code update
  // leaves the running process serving old Python against the freshly-updated
  // JS bundle). But OUR backend is one of those processes, and killing it
  // mid-update produces the boot→kill→crash loop in #37532 — the desktop
  // already restarts its own backend via the rebuild+relaunch below, so the
  // reap must spare it. Hand the live backend's PID to the update process;
  // _kill_stale_dashboard_processes reads HERMES_DESKTOP_CHILD_PID and excludes
  // it while still reaping any genuinely-orphaned dashboards. (#37532)
  // Exclude every desktop-managed backend (primary + all pool profiles) from
  // the update reaper. _kill_stale_dashboard_processes accepts a comma-separated
  // list (a single int still parses for back-compat).
  const desktopChildPids = []
  if (hermesProcess && Number.isInteger(hermesProcess.pid)) {
    desktopChildPids.push(hermesProcess.pid)
  }
  for (const entry of backendPool.values()) {
    if (entry.process && Number.isInteger(entry.process.pid)) {
      desktopChildPids.push(entry.process.pid)
    }
  }
  if (desktopChildPids.length) {
    env.HERMES_DESKTOP_CHILD_PID = desktopChildPids.join(',')
  }

  // Branch-pin so a non-main checkout doesn't get switched to main (and self-heal
  // to main when the pinned branch no longer exists on origin).
  let branchArgs = []
  try {
    const head = await runGit(['rev-parse', '--abbrev-ref', 'HEAD'], { cwd: updateRoot })
    const current = (head.stdout || '').trim()
    if (head.code === 0 && current && current !== 'HEAD') {
      branchArgs = ['--branch', await resolveHealedBranch(updateRoot, current)]
    }
  } catch {
    // best effort
  }

  emitUpdateProgress({ stage: 'update', message: 'Updating Hermes (git + dependencies)…', percent: 10 })
  const updated = await runStreamedUpdate(hermes, ['update', '--yes', ...branchArgs], {
    cwd: updateRoot,
    env,
    stage: 'update'
  })
  if (updated.code !== 0) {
    emitUpdateProgress({ stage: 'error', message: 'hermes update failed.', error: updated.error || 'update-failed' })
    return { ok: false, error: 'hermes update failed' }
  }

  emitUpdateProgress({ stage: 'rebuild', message: 'Rebuilding the desktop app…', percent: 60 })
  // Retry-once: a first rebuild can fail on a still-settling tree or a
  // self-healed (network-blocked) Electron download; a second run builds clean
  // off the healed dist so we reach the swap+relaunch below instead of bailing.
  const rebuilt = await runRebuildWithRetry(attempt => {
    if (attempt > 0) {
      emitUpdateProgress({ stage: 'rebuild', message: 'Retrying the desktop rebuild…', percent: 60 })
    }
    return runStreamedUpdate(hermes, ['desktop', '--build-only'], { cwd: updateRoot, env, stage: 'rebuild' })
  })
  if (rebuilt.code !== 0) {
    emitUpdateProgress({
      stage: 'error',
      message: 'Backend updated, but the desktop rebuild failed. Restart Hermes to retry.',
      error: rebuilt.error || 'rebuild-failed'
    })
    return { ok: false, backendUpdated: true, error: 'desktop rebuild failed' }
  }

  // The rebuilt bundle's name comes from the SOURCE tree's productName at
  // update time: 'APEX.app' after the brand rename, 'ApexNodes.app' for an
  // older checkout. Accept both so an in-flight update never misses the build.
  const rebuiltApp = [
    path.join(updateRoot, 'apps', 'desktop', 'release', 'mac-arm64', 'APEX.app'),
    path.join(updateRoot, 'apps', 'desktop', 'release', 'mac', 'APEX.app'),
    path.join(updateRoot, 'apps', 'desktop', 'release', 'mac-arm64', 'ApexNodes.app'),
    path.join(updateRoot, 'apps', 'desktop', 'release', 'mac', 'ApexNodes.app')
  ].find(directoryExists)
  const targetApp = runningAppBundle()

  // No bundle to swap (dev run, Linux AppImage, or unresolved paths): the
  // backend is updated; the next launch picks up the rebuilt GUI.
  if (!rebuiltApp || !targetApp) {
    emitUpdateProgress({
      stage: 'done',
      message: 'Backend updated. Restart Hermes to load the new version.',
      percent: 100
    })
    return { ok: true, backendUpdated: true, rebuiltApp: rebuiltApp || null }
  }

  emitUpdateProgress({ stage: 'restart', message: 'Installing the updated app and restarting…', percent: 95 })

  // Detached swapper: wait for THIS process to exit (so the bundle is free),
  // ditto the rebuilt app over the running one, clear quarantine, relaunch.
  const swapScript = `#!/bin/bash
set -u
APP_PID=${process.pid}
SRC=${shellQuote(rebuiltApp)}
DST=${shellQuote(targetApp)}
for _ in $(seq 1 240); do
  kill -0 "$APP_PID" 2>/dev/null || break
  sleep 0.5
done
if [ "$SRC" != "$DST" ]; then
  if /usr/bin/ditto "$SRC" "$DST.hermes-update-new"; then
    rm -rf "$DST.hermes-update-old" 2>/dev/null || true
    mv "$DST" "$DST.hermes-update-old" 2>/dev/null || rm -rf "$DST"
    mv "$DST.hermes-update-new" "$DST"
    rm -rf "$DST.hermes-update-old" 2>/dev/null || true
  fi
fi
/usr/bin/xattr -dr com.apple.quarantine "$DST" 2>/dev/null || true
/usr/bin/open "$DST"
`
  const scriptPath = path.join(app.getPath('temp'), `hermes-desktop-update-${Date.now()}.sh`)
  try {
    fs.writeFileSync(scriptPath, swapScript, { mode: 0o755 })
  } catch (err) {
    emitUpdateProgress({
      stage: 'done',
      message: 'Backend + app updated. Restart Hermes to load the new version.',
      percent: 100
    })
    rememberLog(`[updates] could not write swap script: ${err.message}; rebuilt app at ${rebuiltApp}`)
    return { ok: true, backendUpdated: true, rebuiltApp }
  }

  const child = spawn('/bin/bash', [scriptPath], { detached: true, stdio: 'ignore' })
  child.unref()
  rememberLog(`[updates] launched mac swap+relaunch: ${scriptPath} (${rebuiltApp} -> ${targetApp})`)

  setTimeout(() => app.quit(), 600)
  return { ok: true, handedOff: true, rebuiltApp, targetApp }
}

function readJson(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'))
  } catch {
    return null
  }
}

// Bootstrap-complete marker helpers. The marker is written ONCE by the
// first-launch bootstrap runner (Phase 1D) after install.ps1 stages succeed
// AND the user has finished initial configuration. On every subsequent boot
// we check `isBootstrapComplete()` and skip the bootstrap flow entirely if
// the marker is present and current-schema.
//
// Marker schema (version 1):
//   {
//     schemaVersion: 1,
//     pinnedCommit: "<40-char SHA>",       // what install.ps1 was driven against
//     pinnedBranch: "<branch name>" | null,
//     completedAt:  "<ISO 8601>",
//     desktopVersion: "<app.getVersion()>"  // for forensics
//   }
function readBootstrapMarker() {
  return readJson(BOOTSTRAP_COMPLETE_MARKER)
}

function isBootstrapComplete() {
  const marker = readBootstrapMarker()
  if (!marker || typeof marker !== 'object') return false
  if (marker.schemaVersion !== BOOTSTRAP_MARKER_SCHEMA_VERSION) return false
  if (typeof marker.pinnedCommit !== 'string' || marker.pinnedCommit.length < 7) return false
  // We DELIBERATELY do NOT verify that the checkout is currently at the
  // pinned commit -- users update via the in-app update path or `hermes
  // update`, which moves HEAD legitimately. The marker just attests "we
  // ran the bootstrap successfully at least once." We DO additionally require
  // a runnable venv: an interrupted or split-home install can leave the marker
  // + checkout without a venv, and trusting that spawns a dead backend
  // ("gateway offline") instead of re-running bootstrap to repair it.
  return isHermesSourceRoot(ACTIVE_HERMES_ROOT) && fileExists(getVenvPython(VENV_ROOT))
}

// Probe the on-disk canonical install for the runtime-select fail-open logic.
// Reports the two facts canUseOnDiskRuntime() needs: is the runtime SOURCE
// present (hermes_cli/main.py) and is a runnable interpreter present. We scope
// "python present" to the co-located venv on purpose: ensureRuntime()'s adoption
// path (the createActiveBackend venv-wiring branch) REQUIRES getVenvPython(
// VENV_ROOT) and throws without it, so adopting on the strength of a mere system
// Python would just trade a bootstrap brick for a venv-missing brick. The pair
// here is therefore exactly the pair isBootstrapComplete() checks — the only
// difference the fail-open path cares about is the presence/absence of the
// attesting MARKER, not the runnability of the install.
function probeOnDiskRuntime() {
  return {
    sourcePresent: isHermesSourceRoot(ACTIVE_HERMES_ROOT),
    pythonPresent: fileExists(getVenvPython(VENV_ROOT))
  }
}

function writeBootstrapMarker(payload) {
  fs.mkdirSync(path.dirname(BOOTSTRAP_COMPLETE_MARKER), { recursive: true })
  const merged = {
    schemaVersion: BOOTSTRAP_MARKER_SCHEMA_VERSION,
    pinnedCommit: payload.pinnedCommit || null,
    pinnedBranch: payload.pinnedBranch || null,
    // The admin runtime version label this install landed on, when known (R4/R5
    // thread it through the stamp). Lets the opt-in update check compare the
    // installed version against /latest even when the commit key is unchanged
    // (a re-publish under the same key with a bumped label).
    version: payload.version || null,
    completedAt: new Date().toISOString(),
    desktopVersion: app.getVersion()
  }
  writeFileAtomic(BOOTSTRAP_COMPLETE_MARKER, JSON.stringify(merged, null, 2) + '\n', 'utf8')
  return merged
}

function resolveWebDist() {
  const override = process.env.HERMES_DESKTOP_WEB_DIST
  if (override && directoryExists(path.resolve(override))) return path.resolve(override)

  const unpackedDist = path.join(unpackedPathFor(APP_ROOT), 'dist')
  if (directoryExists(unpackedDist)) return unpackedDist

  // Final fallback: APP_ROOT/dist. When packaged with asar:true this lives
  // INSIDE app.asar — not a servable filesystem directory — so the embedded
  // dashboard backend 404s on static routes (see #41327, #39472). The durable
  // fix is unpacking dist/ (PR #41411 adds dist/** to asarUnpack so the tier-2
  // unpackedDist above resolves). If we still land here while packaged, log it
  // so the cause isn't silent.
  const fallback = path.join(APP_ROOT, 'dist')
  if (IS_PACKAGED && /app\.asar(?=$|[\\/])/.test(fallback) && !directoryExists(fallback)) {
    rememberLog(
      `[web-dist] dashboard frontend dir resolved to an asar-internal path that ` +
        `is not a real directory: ${fallback}. Static routes will 404. ` +
        `Ensure dist/** is unpacked (asarUnpack) or set HERMES_DESKTOP_WEB_DIST.`
    )
  }
  return fallback
}

function resolveRendererIndex() {
  const candidates = [path.join(APP_ROOT, 'dist', 'index.html'), path.join(resolveWebDist(), 'index.html')]
  const found = candidates.find(fileExists)
  if (found) return found
  // Nothing on disk. A packaged build with no renderer bundle blank-pages with
  // a bare ERR_FILE_NOT_FOUND and no clue why (see #39484). Surface the cause
  // and the fix before Electron loads the missing file.
  rememberLog(
    `[renderer] index.html not found — the desktop app was packaged without a ` +
      `renderer bundle. Tried: ${candidates.join(', ')}. ` +
      `Rebuild with: hermes desktop --force-build`
  )
  return candidates[0]
}

// True when `dir` lives inside the packaged app bundle / install tree.
// Packaged Electron's process.cwd() (and npm's INIT_CWD when dev tooling
// leaked into a release build) often resolve here — e.g. win-unpacked on
// Windows — which is exactly where PR #37536 item 16 said we must NOT run.
function isPackagedInstallPath(dir) {
  return isPackagedInstallPathUnderRoots(dir, {
    isPackaged: IS_PACKAGED,
    installRoots: [
      APP_ROOT,
      path.dirname(process.execPath),
      resolveRemovableAppPath(process.execPath, process.platform, process.env)
    ]
  })
}

function resolveHermesCwd() {
  // In a packaged build, `process.cwd()` resolves to the install root (e.g.
  // `…/win-unpacked` on Windows or `/Applications/Hermes.app/Contents/...`
  // on macOS). Sessions spawned there leave files inside the app bundle
  // and bewilder users when "where did my files go?" is the install dir.
  // The user-configurable default project directory wins over everything,
  // followed by env hints (only honored when packaged if they point at a
  // real directory), then the home dir.
  const candidates = [
    readDefaultProjectDir(),
    process.env.HERMES_DESKTOP_CWD,
    IS_PACKAGED ? null : process.env.INIT_CWD,
    IS_PACKAGED ? null : process.cwd(),
    !IS_PACKAGED ? SOURCE_REPO_ROOT : null,
    app.getPath('home')
  ]

  for (const candidate of candidates) {
    if (!candidate) continue
    const resolved = path.resolve(String(candidate))

    if (isPackagedInstallPath(resolved)) {
      continue
    }

    if (directoryExists(resolved)) return resolved
  }

  return app.getPath('home')
}

function sanitizeWorkspaceCwd(cwd) {
  const trimmed = typeof cwd === 'string' ? cwd.trim() : ''

  if (!trimmed || isPackagedInstallPath(trimmed)) {
    return { cwd: resolveHermesCwd(), sanitized: Boolean(trimmed) }
  }

  try {
    const resolved = path.resolve(trimmed)

    if (directoryExists(resolved)) {
      return { cwd: resolved, sanitized: false }
    }
  } catch {
    // Fall through to the resolved default.
  }

  return { cwd: resolveHermesCwd(), sanitized: Boolean(trimmed) }
}

// Persisted "Default project directory" — surfaced as a setting in the
// renderer (see app/settings/sessions-settings.tsx). Stored as JSON in
// userData so it survives self-updates without bleeding into the new
// install. `null` means "no preference, fall back to the usual chain".
const DEFAULT_PROJECT_DIR_CONFIG_FILENAME = 'project-dir.json'

function defaultProjectDirConfigPath() {
  return path.join(app.getPath('userData'), DEFAULT_PROJECT_DIR_CONFIG_FILENAME)
}

function readDefaultProjectDir() {
  try {
    const raw = fs.readFileSync(defaultProjectDirConfigPath(), 'utf8')
    const parsed = JSON.parse(raw)

    if (parsed && typeof parsed.dir === 'string' && parsed.dir.trim()) {
      const resolved = path.resolve(parsed.dir)

      if (directoryExists(resolved)) {
        return resolved
      }
    }
  } catch {
    // Missing / unreadable / malformed → fall through to the rest of the
    // candidate chain.
  }

  return null
}

function writeDefaultProjectDir(dir) {
  const target = defaultProjectDirConfigPath()
  const payload = dir ? JSON.stringify({ dir: path.resolve(dir) }, null, 2) : JSON.stringify({}, null, 2)

  try {
    fs.mkdirSync(path.dirname(target), { recursive: true })
    fs.writeFileSync(target, payload, 'utf8')
  } catch (error) {
    rememberLog(`[settings] write default project dir failed: ${error.message}`)
  }
}

function createPythonBackend(root, label, dashboardArgs, options = {}) {
  const python = findPythonForRoot(root)
  if (!python) return null

  return {
    kind: 'python',
    label,
    command: python,
    args: ['-m', 'hermes_cli.main', ...dashboardArgs],
    env: buildDesktopBackendEnv({
      hermesHome: HERMES_HOME,
      pythonPathEntries: [root],
      venvRoot: path.join(root, 'venv')
    }),
    root,
    bootstrap: Boolean(options.bootstrap),
    shell: false
  }
}

// createActiveBackend — build a backend pointing at ACTIVE_HERMES_ROOT, the
// canonical install location shared with the CLI installer. The venv at
// VENV_ROOT may not exist yet on first run; bootstrap=true tells
// ensureRuntime() to create / refresh it before launch.
function createActiveBackend(dashboardArgs) {
  const venvPython = getVenvPython(VENV_ROOT)

  return {
    kind: 'python',
    label: `Hermes at ${ACTIVE_HERMES_ROOT}`,
    command: fileExists(venvPython) ? venvPython : findSystemPython(),
    args: ['-m', 'hermes_cli.main', ...dashboardArgs],
    env: buildDesktopBackendEnv({
      hermesHome: HERMES_HOME,
      pythonPathEntries: [ACTIVE_HERMES_ROOT],
      venvRoot: VENV_ROOT
    }),
    root: ACTIVE_HERMES_ROOT,
    bootstrap: true,
    shell: false
  }
}

function resolveHermesBackend(dashboardArgs) {
  // 1. Explicit override -- HERMES_DESKTOP_HERMES_ROOT points at a developer
  //    checkout. Honour it as-is (no bootstrap; the user is driving).
  const overrideRoot = process.env.HERMES_DESKTOP_HERMES_ROOT && path.resolve(process.env.HERMES_DESKTOP_HERMES_ROOT)
  if (overrideRoot && isHermesSourceRoot(overrideRoot)) {
    const backend = createPythonBackend(overrideRoot, `Hermes source at ${overrideRoot}`, dashboardArgs)
    if (backend) return backend
  }

  // 2. Development source -- when running `npm run dev` from a checkout, the
  //    cloned repo at SOURCE_REPO_ROOT takes precedence over ACTIVE and any
  //    installed `hermes` on PATH so local Python edits are actually exercised.
  //    (In dev with no checkout, SOURCE_REPO_ROOT won't pass isHermesSourceRoot.)
  if (!IS_PACKAGED && isHermesSourceRoot(SOURCE_REPO_ROOT)) {
    const backend = createPythonBackend(SOURCE_REPO_ROOT, `Hermes source at ${SOURCE_REPO_ROOT}`, dashboardArgs)
    if (backend) return backend
  }

  // 3. Bootstrap-complete ACTIVE_HERMES_ROOT -- the canonical install at
  //    %LOCALAPPDATA%\hermes\hermes-agent (Windows) or ~/.hermes/hermes-agent.
  //    The bootstrap marker means install.ps1 stages finished and the user
  //    completed initial configuration; we trust the install and go straight
  //    to spawning hermes. Updates flow through the in-app update path
  //    (applyUpdates -> git pull) or `hermes update` from the CLI.
  if (isBootstrapComplete()) {
    return createActiveBackend(dashboardArgs)
  }

  // 3.5 FAIL-OPEN (2026-07-06 incident): a usable runtime is already extracted at
  //     ACTIVE_HERMES_ROOT (source + venv on disk) but the bootstrap-complete
  //     marker is absent/stale — an interrupted install, a dropped marker, a
  //     legacy install predating the marker, or a COS-tarball extract that was
  //     never registered on PATH. WITHOUT this, resolution falls through to the
  //     bootstrap-needed sentinel (step 6), which fires the network runtime-latest
  //     resolve; when the cloud advertises a version whose COS tarball is not yet
  //     published, install.sh 404s and the WHOLE gateway refuses to start —
  //     stranding the user on an error page despite a perfectly runnable runtime
  //     sitting right there. Adopt the on-disk runtime directly instead (same
  //     venv-wiring adoption path createActiveBackend feeds). We do NOT do this
  //     while an opt-in update is pending: the user chose a new version and
  //     adopting the old one would silently no-op their request (that case must
  //     drive the bootstrap re-run below). The client must self-heal against a
  //     wrong/ahead server answer rather than assume the cloud is always right.
  const preBootstrap = resolvePreBootstrapDecision({
    markerComplete: false, // isBootstrapComplete() already returned false above
    onDiskUsable: canUseOnDiskRuntime(probeOnDiskRuntime()),
    updatePending: readRuntimePinOverride() !== null
  })
  if (preBootstrap === 'use-installed') {
    rememberLog(
      '[runtime-select] bootstrap marker absent/stale but a runnable runtime is on disk at ' +
        `${ACTIVE_HERMES_ROOT}; adopting it instead of re-bootstrapping (fail-open — avoids ` +
        'bricking on an unpublished admin-latest / package fetch failure).'
    )
    return createActiveBackend(dashboardArgs)
  }

  // R5: a pending opt-in update (override file present, marker just dropped by
  // hermes:runtime:apply-update) MUST drive the bootstrap re-run so install.sh
  // re-fetches the new pin. Skip the "use an existing install" steps 4-5 — the
  // prior install's `hermes` is still on PATH (and its venv on disk), and trusting
  // it here would silently spawn the OLD runtime and no-op the update. Only the
  // bootstrap path (step 6) honors resolveBootstrapStamp()'s override. When the
  // override is absent this is a no-op and resolution behaves exactly as before.
  const runtimeUpdatePending = readRuntimePinOverride() !== null
  if (runtimeUpdatePending) {
    rememberLog('[runtime-update] pin override pending; forcing bootstrap re-run (skipping existing-install reuse)')
  }

  // 4. Existing `hermes` on PATH -- installed via install.ps1 / install.sh from
  //    a previous tool-only setup, or pip-installed system-wide. Use it but
  //    do NOT write a bootstrap marker; the user did this themselves and we
  //    don't want to take ownership of an install we didn't perform.
  //    HERMES_DESKTOP_IGNORE_EXISTING=1 forces the bootstrap path for testing.
  if (!runtimeUpdatePending && process.env.HERMES_DESKTOP_IGNORE_EXISTING !== '1') {
    let hermesCommand = null
    const hermesOverride = process.env.HERMES_DESKTOP_HERMES

    if (hermesOverride) {
      const resolvedOverride = findOnPath(hermesOverride)
      if (resolvedOverride) {
        hermesCommand = resolvedOverride
      } else if (!isWindowsBinaryPathInWsl(hermesOverride, { isWsl: IS_WSL })) {
        hermesCommand = hermesOverride
      } else {
        rememberLog(`Ignoring Windows Hermes override under WSL: ${hermesOverride}`)
      }
    } else {
      hermesCommand = findOnPath('hermes')
    }

    if (hermesCommand) {
      if (looksLikeDesktopAppBinary(hermesCommand)) {
        rememberLog(`Ignoring desktop app executable on PATH while resolving Hermes CLI: ${hermesCommand}`)
        hermesCommand = null
      }
    }

    if (hermesCommand) {
      // Smoke-test the candidate before trusting it. A `hermes` shim
      // left behind by a half-uninstalled pip install (or a venv
      // entry-point pointing at a deleted interpreter) still resolves
      // via findOnPath but explodes on spawn -- the user then sees a
      // dead backend instead of the first-launch installer. The cheap
      // `--version` probe (see backend-probes.cjs) catches that case
      // and lets the resolver fall through to step 6 / bootstrap.
      const shellForProbe = isCommandScript(hermesCommand)
      if (verifyHermesCli(hermesCommand, { shell: shellForProbe })) {
        return {
          label: `existing Hermes CLI at ${hermesCommand}`,
          command: hermesCommand,
          args: dashboardArgs,
          bootstrap: false,
          env: {},
          kind: 'command',
          shell: shellForProbe
        }
      }
      rememberLog(
        `Ignoring existing Hermes CLI at ${hermesCommand}: --version probe failed; falling through to bootstrap.`
      )
    }
  }

  // 5. Last-ditch: pip-installed hermes_cli module via system Python.
  //    Same rationale as #4 -- the user installed this; we use it but don't
  //    take ownership. Also skipped while a runtime update is pending (step 4).
  const python = runtimeUpdatePending ? null : findSystemPython()
  if (python) {
    // Same smoke-test rationale as step 4: a system Python in the
    // SUPPORTED_VERSIONS range can be registered (PEP 514) without
    // having hermes_cli installed -- common on dev boxes that have
    // a python.org install from prior unrelated work. Returning that
    // backend hands the spawn step a guaranteed ModuleNotFoundError.
    // Verify the import works before trusting the candidate; on
    // failure, fall through to step 6 so the bootstrap runner pulls
    // a uv-managed 3.11 into %LOCALAPPDATA%\hermes\hermes-agent\venv.
    if (canImportHermesCli(python)) {
      return {
        kind: 'python',
        label: `installed hermes_cli module via ${python}`,
        command: python,
        args: ['-m', 'hermes_cli.main', ...dashboardArgs],
        bootstrap: false,
        env: {},
        shell: false
      }
    }
    rememberLog(`Ignoring system Python ${python}: hermes_cli is not importable; falling through to bootstrap.`)
  }

  // 6. Nothing usable yet -- signal the bootstrap runner that we need to
  //    clone+install. Phase 1D's bootstrap-runner consumes this sentinel
  //    and drives install.ps1 stages with a progress UI. Until 1D lands,
  //    callers see the sentinel and surface it as a user-facing error
  //    explaining what's missing.
  //
  //    We deliberately do NOT throw here -- throwing inside
  //    resolveHermesBackend was the old "no payload" path and forced the
  //    user into a dead end. With the bootstrap protocol, "no install yet"
  //    is a recoverable state the GUI can drive through.
  return {
    kind: 'bootstrap-needed',
    label: 'Hermes Agent not installed yet; bootstrap required',
    command: null,
    args: dashboardArgs,
    bootstrap: true,
    env: {},
    shell: false,
    // Hints for the bootstrap runner / UI layer:
    activeRoot: ACTIVE_HERMES_ROOT,
    installStamp: INSTALL_STAMP, // may be null in dev
    isPackaged: IS_PACKAGED,
    platform: process.platform
  }
}

async function ensureRuntime(backend) {
  // Every boot path (existing install or fresh bootstrap) passes through here
  // before the gateway starts — heal a rotated relay key in the registered
  // custom provider so the model picker's live listing works this launch,
  // then fold any newer platform config into config.yaml (line surgery; the
  // gateway loads the result fresh).
  syncManagedCustomProviderKey()
  applyClientConfigToRuntime('boot')
  guardConfigYamlProductBlocks('boot')
  watchConfigYamlProductBlocks()

  if (!backend.bootstrap) {
    await advanceBootProgress('runtime.external', `Using ${backend.label}`, 32)
    return backend
  }

  // backend.kind === 'bootstrap-needed' means resolveHermesBackend couldn't
  // find anything to spawn. Hand off to the bootstrap runner which drives the
  // platform installer, writes the bootstrap-complete marker on success, then
  // we re-resolve to get the now-installed backend.
  //
  // Phase 1D status: bootstrap runs but events go to desktop.log only
  // (renderer window isn't created until later in startBackend). Phase 1E
  // will rewire startup to spawn the window first and route bootstrap events
  // to a renderer-side install overlay.
  if (backend.kind === 'bootstrap-needed') {
    rememberLog('[bootstrap] no Hermes install found; starting first-launch bootstrap')

    // ApexNodes: seed the DeepSeek default into config.yaml before install.sh
    // runs (it keeps an existing config.yaml), so a fresh install boots with
    // DeepSeek preselected and only needs the user's API key.
    seedDefaultModelConfig()

    if (await handOffWindowsBootstrapRecovery('bootstrap-needed')) {
      const handoffError = new Error('Hermes recovery was handed off to Hermes Setup. The desktop will restart when recovery completes.')
      handoffError.isBootstrapFailure = true
      handoffError.bootstrapHandedOff = true
      bootstrapFailure = handoffError
      throw handoffError
    }

    // hc-452: is this a re-bootstrap for an opt-in runtime UPDATE (marker
    // dropped by hermes:runtime:apply-update, an override pin waiting to be
    // installed), or a genuine first-ever install (no prior runtime, no
    // override)? Same signal the fail-open/rollback logic below already reads
    // via readRuntimePinOverride() !== null (see wasOptInUpdate at the ok===false
    // branch). Read synchronously and early (before the eager synthetic-manifest
    // broadcast just below) so even the very first UI frame — shown before the
    // real manifest fetch resolves — carries the right "updating" vs
    // "first-time setup" signal instead of flashing the wrong copy on a slow
    // network. override.previousMarker (persisted by hermes:runtime:apply-update
    // right before it drops the marker) carries the version being replaced.
    const runtimeUpdateOverride = readRuntimePinOverride()
    const isRuntimeUpdate = runtimeUpdateOverride !== null
    const bootstrapUpdateInfoEarly = isRuntimeUpdate
      ? {
          isUpdate: true,
          // The resolved target version isn't known yet at this point
          // (resolveBootstrapStamp hasn't run) — the real 'manifest' event
          // fills this in once bootstrapStamp resolves, just below.
          toVersion: null,
          fromVersion:
            runtimeUpdateOverride.previousMarker && runtimeUpdateOverride.previousMarker.version
              ? runtimeUpdateOverride.previousMarker.version
              : null
        }
      : { isUpdate: false, toVersion: null, fromVersion: null }

    // Eagerly flip the bootstrap UI state to 'active' so the renderer
    // shows the install overlay BEFORE the runner finishes fetching the
    // manifest (which on slow networks can take tens of seconds and would
    // otherwise leave the user staring at the generic 'Preparing' splash).
    // We emit a synthetic manifest with an empty stages list -- the real
    // manifest event will overwrite it once install.ps1 -Manifest returns.
    try {
      broadcastBootstrapEvent({
        type: 'manifest',
        stages: [],
        protocolVersion: null,
        updateInfo: bootstrapUpdateInfoEarly
      })
    } catch {
      void 0
    }

    bootstrapAbortController = new AbortController()

    // ── R4/R5: resolve the pin the installer should use ──────────────────────
    // Precedence: a persisted opt-in override (R5) > a live admin-latest overlay
    // fetched now (R4 first install) > the build-time stamp (offline fallback).
    // resolveBootstrapStamp NEVER throws — on any failure it returns the baked
    // stamp, so a fresh install proceeds even when the cloud is unreachable.
    const bootstrapStamp = await resolveBootstrapStamp(backend.installStamp)

    // Now that bootstrapStamp is resolved, fill in the target version the
    // real bootstrap run (and its 'manifest' event, emitted from inside
    // runBootstrap once install.ps1/.sh -Manifest returns) will carry.
    const bootstrapUpdateInfo = {
      ...bootstrapUpdateInfoEarly,
      toVersion: bootstrapStamp && bootstrapStamp.version ? bootstrapStamp.version : null
    }

    const bootstrapResult = await runBootstrap({
      installStamp: bootstrapStamp,
      activeRoot: backend.activeRoot,
      sourceRepoRoot: SOURCE_REPO_ROOT,
      resourcesPath: process.resourcesPath,
      hermesHome: HERMES_HOME,
      logRoot: path.join(HERMES_HOME, 'logs'),
      abortSignal: bootstrapAbortController.signal,
      updateInfo: bootstrapUpdateInfo,
      // Region (CN mirrors vs upstream defaults) is auto-detected per machine by
      // install.sh / install.ps1 themselves (IP/timezone heuristic), so a
      // packaged build serves both foreign and mainland-China users correctly —
      // we deliberately do NOT force cnMirrors here (the old "packaged == China"
      // assumption wrongly gave every overseas user the slow/blocked CN
      // mirrors). Escape hatches still win: an explicit HERMES_CN_MIRRORS in the
      // environment, or APEXNODES_REGION=cn|global, override auto-detection.
      //
      // We DO always thread the COS base through (decoupled from the mirror
      // flag) so that when the installer auto-detects CN it can fetch the
      // runtime tarball + uv from our public bucket instead of github.com.
      runtimeCosBase: RUNTIME_COS_BASE,
      onEvent: ev => {
        // Tee every bootstrap event to (a) the desktop log for forensics
        // and (b) the renderer for live progress UI. Either may be absent;
        // tolerate both gracefully so a renderer crash doesn't stall the
        // bootstrap and a log-write failure doesn't suppress the UI signal.
        try {
          rememberLog(`[bootstrap] ${JSON.stringify(ev)}`)
        } catch {
          void 0
        }
        try {
          broadcastBootstrapEvent(ev)
        } catch {
          void 0
        }
      },
      writeMarker: writeBootstrapMarker
    })

    bootstrapAbortController = null

    if (bootstrapResult.cancelled) {
      // A cancelled opt-in update must not leave the install half-retargeted:
      // restore the previous marker so the old runtime stays active.
      rollbackRuntimePinOverride('install cancelled')
      const cancelledError = new Error('Hermes install was cancelled.')
      cancelledError.isBootstrapFailure = true
      cancelledError.bootstrapCancelled = true
      bootstrapFailure = cancelledError
      throw cancelledError
    }

    if (!bootstrapResult.ok) {
      // Capture whether this was an opt-in update BEFORE any rollback clears the
      // override (the fail-open decision below needs to know).
      const wasOptInUpdate = readRuntimePinOverride() !== null

      // FAIL-OPEN safety net (2026-07-06 incident): a bootstrap can fail because
      // the cloud advertised a version whose COS tarball isn't published yet
      // (install.sh 404) or any transient network/checksum error. For a plain
      // first-install/marker-repair run (NOT an opt-in update), if a runnable
      // runtime is STILL on disk after the failed attempt, start the gateway
      // with it instead of latching a fatal failure and stranding the user.
      // Step 3.5 in resolveHermesBackend normally prevents us from ever reaching
      // here with a usable on-disk runtime, but this backstops any failure that
      // slips past it (e.g. a genuine fresh install whose download 404s while a
      // prior good extract survives). Opt-in updates deliberately fall through
      // to the rollback path below (restores the previous marker → old runtime
      // boots next launch); we must not silently no-op the user's chosen version.
      if (!wasOptInUpdate && canUseOnDiskRuntime(probeOnDiskRuntime())) {
        const fallback = resolveBootstrapFailureFallback({
          onDiskUsable: true,
          updatePending: false
        })
        if (fallback === 'fallback-to-disk') {
          rememberLog(
            `[runtime-select] bootstrap failed${
              bootstrapResult.failedStage ? ` at stage '${bootstrapResult.failedStage}'` : ''
            } (${bootstrapResult.error || 'unknown error'}); a runnable runtime remains on disk at ` +
              `${ACTIVE_HERMES_ROOT} — degrading to it and starting the gateway (fail-open) instead of ` +
              'bricking. This typically means the admin latest advertised an unpublished/unreachable ' +
              'package; the existing runtime is used until a valid update is available.'
          )
          // Re-resolve; step 3.5 now adopts the on-disk runtime and wires venv.
          return ensureRuntime(resolveHermesBackend(backend.args))
        }
      }

      // R5 don't-brick guard: a failed re-bootstrap of an opt-in update rolls
      // back to the previous marker so the next launch boots the OLD runtime
      // (still on disk) instead of bricking on the new pin.
      rollbackRuntimePinOverride(bootstrapResult.failedStage || 'bootstrap failed')
      const bootstrapError = new Error(
        `Hermes bootstrap failed${bootstrapResult.failedStage ? ` at stage '${bootstrapResult.failedStage}'` : ''}: ` +
          `${bootstrapResult.error || 'unknown error'}. ` +
          `Check ${path.join(HERMES_HOME, 'logs', 'desktop.log')} for the full transcript.`
      )
      bootstrapError.isBootstrapFailure = true
      bootstrapError.failedStage = bootstrapResult.failedStage || null
      // Latch the failure so subsequent startHermes() calls return this
      // same error without re-running install.ps1.  Cleared by the
      // hermes:bootstrap:reset IPC (renderer's "Reload and retry").
      bootstrapFailure = bootstrapError
      throw bootstrapError
    }

    rememberLog('[bootstrap] bootstrap complete; marker written. Re-resolving backend.')
    // An opt-in update (R5) succeeded — the freshly written marker is now the
    // source of truth for what's installed, so retire the pending override.
    // (No-op for a normal first install, where no override exists.)
    if (readRuntimePinOverride()) {
      rememberLog('[runtime-update] opt-in update installed successfully; clearing pin override')
      clearRuntimePinOverride()
    }
    // Re-resolve now that the install exists. The new resolution lands in
    // step 3 (bootstrap-complete marker) and we recurse to wire venvPython.
    return ensureRuntime(resolveHermesBackend(backend.args))
  }

  // bootstrap=true with a real backend (createActiveBackend path) means we
  // have a checkout and need to ensure the venv-derived Python command is
  // wired into the backend before launch. Same code path the old factory
  // sync flow exited through, minus all the factory/pip/marker machinery
  // (install.ps1 owns those concerns now and the bootstrap-complete marker
  // attests they ran successfully).
  if (!isHermesSourceRoot(ACTIVE_HERMES_ROOT)) {
    throw new Error(
      `Hermes install at ${ACTIVE_HERMES_ROOT} is missing or incomplete. ` +
        'Reinstall via the desktop installer or scripts/install.ps1.'
    )
  }

  // On Windows, preflight Git Bash. Hermes' terminal tool calls bash.exe
  // directly (tools/environments/local.py); without it the agent can't run
  // terminal commands. install.ps1's Stage-Git puts PortableGit at
  // %LOCALAPPDATA%\apexnodes\git\, which findGitBash() picks up, so for any
  // user who completed the bootstrap this is a no-op. For users who got
  // here via an external `hermes` on PATH, this check still helps.
  if (IS_WINDOWS && !findGitBash()) {
    throw new Error(
      'Git for Windows is required for Hermes on Windows (provides Git Bash, ' +
        "which the agent's terminal tool uses). Install it from " +
        'https://git-scm.com/download/win or run `winget install -e --id Git.Git`, ' +
        'then relaunch Hermes.'
    )
  }

  const venvPython = getVenvPython(VENV_ROOT)
  if (!fileExists(venvPython)) {
    // No venv at the expected location AND no bootstrap-needed sentinel
    // means we have a half-installed checkout: .git exists, source files
    // exist, but venv is missing or broken. This shouldn't happen in
    // normal flow because isBootstrapComplete() requires
    // isHermesSourceRoot() and the bootstrap writes the marker only after
    // install.ps1 succeeds. If we hit this, the user (or a deleted venv)
    // broke the invariant; tell them to re-run the install.
    throw new Error(
      `Hermes venv missing at ${VENV_ROOT}. Re-run the desktop installer or ` + '`scripts/install.ps1` to rebuild it.'
    )
  }

  backend.command = venvPython
  backend.label = `Hermes at ${ACTIVE_HERMES_ROOT} (venv: ${VENV_ROOT})`
  updateBootProgress({
    phase: 'runtime.ready',
    message: 'Hermes runtime is ready',
    progress: 82,
    running: true,
    error: null
  })
  return backend
}


function fetchJson(url, token, options = {}) {
  return new Promise((resolve, reject) => {
    const body = options.body === undefined ? undefined : Buffer.from(JSON.stringify(options.body))
    const parsed = new URL(url)
    const client = parsed.protocol === 'https:' ? https : http
    const timeoutMs = resolveTimeoutMs(options.timeoutMs, DEFAULT_FETCH_TIMEOUT_MS)

    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      reject(new Error(`Unsupported Hermes backend URL protocol: ${parsed.protocol}`))
      return
    }

    const req = client.request(
      parsed,
      {
        method: options.method || 'GET',
        headers: {
          'Content-Type': 'application/json',
          'X-Hermes-Session-Token': token,
          ...(body ? { 'Content-Length': String(body.length) } : {})
        }
      },
      res => {
        const chunks = []
        res.on('error', reject)
        res.on('data', chunk => chunks.push(chunk))
        res.on('end', () => {
          const text = Buffer.concat(chunks).toString('utf8')
          if ((res.statusCode || 500) >= 400) {
            reject(new Error(`${res.statusCode}: ${text || res.statusMessage}`))
            return
          }
          if (!text) {
            resolve(null)
            return
          }
          // A 2xx response whose body is HTML means the request fell through
          // to the SPA index.html (e.g. an unregistered /api path). JSON.parse
          // would throw an opaque `Unexpected token '<'` here, so surface a
          // clear diagnostic with the offending URL instead.
          const looksHtml = /^\s*<(?:!doctype|html)/i.test(text)
          const contentType = String(res.headers['content-type'] || '')
          if (looksHtml || contentType.includes('text/html')) {
            reject(
              new Error(
                `Expected JSON from ${url} but got HTML (status ${res.statusCode}). ` +
                  'The endpoint is likely missing on the Hermes backend.'
              )
            )
            return
          }
          try {
            resolve(JSON.parse(text))
          } catch {
            reject(new Error(`Invalid JSON from ${url} (status ${res.statusCode}): ${text.slice(0, 200)}`))
          }
        })
      }
    )

    req.on('error', reject)
    req.setTimeout(timeoutMs, () => {
      req.destroy(new Error(`Timed out connecting to Hermes backend after ${timeoutMs}ms`))
    })
    if (body) req.write(body)
    req.end()
  })
}

function fetchPublicJson(url, options = {}) {
  // Credential-free JSON GET/POST for public gateway endpoints
  // (``/api/status``, ``/api/auth/providers``). Unlike ``fetchJson`` it sends
  // NO ``X-Hermes-Session-Token`` header — used by the auth-mode probe before
  // any credentials exist, and any time we must not leak a token to an
  // endpoint that doesn't need one.
  return new Promise((resolve, reject) => {
    const body = options.body === undefined ? undefined : Buffer.from(JSON.stringify(options.body))
    let parsed
    try {
      parsed = new URL(url)
    } catch (error) {
      reject(new Error(`Invalid URL: ${error.message}`))
      return
    }
    const client = parsed.protocol === 'https:' ? https : http
    const timeoutMs = resolveTimeoutMs(options.timeoutMs, DEFAULT_FETCH_TIMEOUT_MS)

    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      reject(new Error(`Unsupported Hermes backend URL protocol: ${parsed.protocol}`))
      return
    }

    const req = client.request(
      parsed,
      {
        method: options.method || 'GET',
        headers: {
          'Content-Type': 'application/json',
          ...(body ? { 'Content-Length': String(body.length) } : {})
        }
      },
      res => {
        const chunks = []
        res.on('data', chunk => chunks.push(chunk))
        res.on('end', () => {
          const text = Buffer.concat(chunks).toString('utf8')
          if ((res.statusCode || 500) >= 400) {
            reject(new Error(`${res.statusCode}: ${text || res.statusMessage}`))
            return
          }
          if (!text) {
            resolve(null)
            return
          }
          const looksHtml = /^\s*<(?:!doctype|html)/i.test(text)
          const contentType = String(res.headers['content-type'] || '')
          if (looksHtml || contentType.includes('text/html')) {
            reject(
              new Error(
                `Expected JSON from ${url} but got HTML (status ${res.statusCode}). ` +
                  'The endpoint is likely missing on the Hermes backend.'
              )
            )
            return
          }
          try {
            resolve(JSON.parse(text))
          } catch {
            reject(new Error(`Invalid JSON from ${url} (status ${res.statusCode}): ${text.slice(0, 200)}`))
          }
        })
      }
    )

    req.on('error', reject)
    req.setTimeout(timeoutMs, () => {
      req.destroy(new Error(`Timed out connecting to Hermes backend after ${timeoutMs}ms`))
    })
    if (body) req.write(body)
    req.end()
  })
}

function mimeTypeForPath(filePath) {
  const ext = path.extname(filePath || '').toLowerCase()

  return MEDIA_MIME_TYPES[ext] || 'application/octet-stream'
}

function extensionForMimeType(mimeType) {
  const type = String(mimeType || '')
    .split(';')[0]
    .trim()
    .toLowerCase()
  if (type === 'image/png') return '.png'
  if (type === 'image/jpeg') return '.jpg'
  if (type === 'image/gif') return '.gif'
  if (type === 'image/webp') return '.webp'
  if (type === 'image/bmp') return '.bmp'
  if (type === 'image/svg+xml') return '.svg'
  return ''
}

function filenameFromUrl(rawUrl, fallback = 'image') {
  try {
    const parsed = new URL(rawUrl)
    const base = path.basename(decodeURIComponent(parsed.pathname || ''))
    return base && base.includes('.') ? base : fallback
  } catch {
    return fallback
  }
}

// Link title resolution — curl (tier 1) → hidden BrowserWindow (tier 2).
const titleCache = new Map()
const titleInflight = new Map()
const TITLE_CACHE_LIMIT = 500
const TITLE_BYTE_BUDGET = 96 * 1024
const TITLE_TIMEOUT_MS = 5000
const TITLE_MAX_REDIRECTS = 3
// Browser-shaped UA — many bot-walled sites (GetYourGuide, Cloudflare-protected
// pages) refuse anything that doesn't look like a real Chrome.
const TITLE_USER_AGENT =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36'
const TITLE_ERROR_RE =
  /\b(access denied|attention required|captcha|error|forbidden|just a moment|request blocked|too many requests)\b/i
const HTML_ENTITIES = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ', '#39': "'" }

// Tier-2 renderer fallback config. Only invoked when curl came back empty or
// matched TITLE_ERROR_RE — keeps cold/CDN-cached pages on the cheap path.
const RENDER_TITLE_MAX_CONCURRENT = 2
const RENDER_TITLE_TIMEOUT_MS = 8000
const RENDER_TITLE_GRACE_MS = 700
// Resource types we cancel before the network even fires — keeps the hidden
// renderer fast and cuts third-party tracking noise.
const RENDER_TITLE_BLOCKED_RESOURCES = new Set([
  'cspReport',
  'font',
  'imageset',
  'media',
  'object',
  'ping',
  'stylesheet'
])

let linkTitleSession = null
let oauthSession = null
let renderTitleInFlight = 0
const renderTitleQueue = []

function canonicalTitleCacheKey(rawUrl) {
  const value = String(rawUrl || '').trim()
  if (!value) return ''

  try {
    const url = new URL(value)
    const host = url.hostname.replace(/^www\./i, '').toLowerCase()
    const pathname = url.pathname === '/' ? '/' : url.pathname.replace(/\/+$/, '') || '/'

    return `${host}${pathname}${url.search || ''}`
  } catch {
    return value
  }
}

function cacheTitle(key, title) {
  if (titleCache.size >= TITLE_CACHE_LIMIT) titleCache.delete(titleCache.keys().next().value)
  titleCache.set(key, title)
}

function decodeHtmlEntities(value) {
  return value
    .replace(/&(amp|lt|gt|quot|apos|nbsp|#39);/gi, (_, k) => HTML_ENTITIES[k.toLowerCase()] ?? '')
    .replace(/&#x([0-9a-f]+);/gi, (_, hex) => String.fromCodePoint(parseInt(hex, 16) || 32))
    .replace(/&#(\d+);/g, (_, dec) => String.fromCodePoint(parseInt(dec, 10) || 32))
}

function parseHtmlTitle(html) {
  const raw = html.match(/<title[^>]*>([\s\S]*?)<\/title>/i)?.[1]
  return raw ? decodeHtmlEntities(raw).replace(/\s+/g, ' ').trim() : ''
}

function fetchHtmlTitleWithCurl(rawUrl) {
  return new Promise(resolve => {
    const url = String(rawUrl || '').trim()
    if (!url) return resolve('')

    const args = [
      '--silent',
      '--show-error',
      '--location',
      '--max-redirs',
      String(TITLE_MAX_REDIRECTS),
      '--max-time',
      String(Math.max(2, Math.ceil(TITLE_TIMEOUT_MS / 1000))),
      '--connect-timeout',
      '4',
      '--user-agent',
      TITLE_USER_AGENT,
      '--header',
      'Accept: text/html,application/xhtml+xml;q=0.9,*/*;q=0.5',
      '--header',
      'Accept-Language: en-US,en;q=0.7',
      '--header',
      'Accept-Encoding: identity',
      '--raw',
      url
    ]
    const child = spawn('curl', args, hiddenWindowsChildOptions({ stdio: ['ignore', 'pipe', 'ignore'] }))
    const chunks = []
    let bytes = 0

    child.stdout.on('data', chunk => {
      if (bytes >= TITLE_BYTE_BUDGET) return
      const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)
      const remaining = TITLE_BYTE_BUDGET - bytes
      const next = buffer.length > remaining ? buffer.subarray(0, remaining) : buffer
      chunks.push(next)
      bytes += next.length
    })

    child.on('error', () => resolve(''))
    child.on('close', () => {
      if (!chunks.length) return resolve('')
      resolve(parseHtmlTitle(Buffer.concat(chunks).toString('utf8')))
    })
  })
}

function getLinkTitleSession() {
  if (linkTitleSession || !app.isReady()) return linkTitleSession
  linkTitleSession = session.fromPartition('hermes:link-titles', { cache: false })
  linkTitleSession.webRequest.onBeforeRequest((details, callback) => {
    callback({ cancel: RENDER_TITLE_BLOCKED_RESOURCES.has(details.resourceType) })
  })
  return linkTitleSession
}

function dequeueRenderTitle() {
  while (renderTitleInFlight < RENDER_TITLE_MAX_CONCURRENT && renderTitleQueue.length) {
    const item = renderTitleQueue.shift()
    renderTitleInFlight += 1
    runRenderTitleJob(item.url).then(title => {
      renderTitleInFlight -= 1
      item.resolve(title)
      dequeueRenderTitle()
    })
  }
}

function runRenderTitleJob(rawUrl) {
  return new Promise(resolve => {
    if (!app.isReady()) return resolve('')

    const partitionSession = getLinkTitleSession()
    if (!partitionSession) return resolve('')

    let settled = false
    let window = null
    let hardTimer = null
    let graceTimer = null

    const finish = title => {
      if (settled) return
      settled = true
      if (hardTimer) clearTimeout(hardTimer)
      if (graceTimer) clearTimeout(graceTimer)
      const value = (title || '').replace(/\s+/g, ' ').trim()
      try {
        if (window && !window.isDestroyed()) window.destroy()
      } catch {
        // BrowserWindow may already be torn down; ignore.
      }
      resolve(value)
    }

    try {
      window = new BrowserWindow({
        show: false,
        width: 1280,
        height: 800,
        webPreferences: {
          backgroundThrottling: false,
          contextIsolation: true,
          javascript: true,
          nodeIntegration: false,
          sandbox: true,
          session: partitionSession,
          webSecurity: true
        }
      })
    } catch {
      return finish('')
    }

    const readTitle = () => window?.webContents?.getTitle?.() || ''
    const scheduleGrace = () => {
      if (graceTimer) clearTimeout(graceTimer)
      graceTimer = setTimeout(() => finish(readTitle()), RENDER_TITLE_GRACE_MS)
    }

    hardTimer = setTimeout(() => finish(readTitle()), RENDER_TITLE_TIMEOUT_MS)

    window.webContents.setUserAgent(TITLE_USER_AGENT)
    window.webContents.on('page-title-updated', scheduleGrace)
    window.webContents.on('did-finish-load', scheduleGrace)
    window.webContents.on('did-fail-load', (_event, _code, _desc, _validatedURL, isMainFrame) => {
      if (isMainFrame) finish('')
    })

    window
      .loadURL(rawUrl, {
        httpReferrer: 'https://www.google.com/',
        userAgent: TITLE_USER_AGENT
      })
      .catch(() => finish(''))
  })
}

function fetchHtmlTitleWithRenderer(rawUrl) {
  return new Promise(resolve => {
    renderTitleQueue.push({ resolve, url: rawUrl })
    dequeueRenderTitle()
  })
}

// Strips known error/captcha titles (e.g. "GetYourGuide – Error", "Just a
// moment...") so they don't get cached as the resolved title.
const usableTitle = value => (value && !TITLE_ERROR_RE.test(value) ? value : '')

function fetchLinkTitle(rawUrl) {
  const url = String(rawUrl || '').trim()
  const key = canonicalTitleCacheKey(url)
  if (!key) return Promise.resolve('')
  if (titleCache.has(key)) return Promise.resolve(titleCache.get(key))
  if (titleInflight.has(key)) return titleInflight.get(key)

  const pending = fetchHtmlTitleWithCurl(url)
    .catch(() => '')
    .then(value => usableTitle((value || '').slice(0, 240)))
    .then(
      async value => value || usableTitle(((await fetchHtmlTitleWithRenderer(url).catch(() => '')) || '').slice(0, 240))
    )
    .then(clean => {
      cacheTitle(key, clean)
      titleInflight.delete(key)
      return clean
    })

  titleInflight.set(key, pending)
  return pending
}

async function resourceBufferFromUrl(rawUrl) {
  if (!rawUrl) throw new Error('Missing URL')
  if (rawUrl.startsWith('data:')) {
    const match = rawUrl.match(/^data:([^;,]+)?(;base64)?,(.*)$/s)
    if (!match) throw new Error('Invalid data URL')
    const mimeType = match[1] || 'application/octet-stream'
    const encoded = match[3] || ''
    const buffer = match[2] ? Buffer.from(encoded, 'base64') : Buffer.from(decodeURIComponent(encoded), 'utf8')
    return { buffer, mimeType }
  }
  if (/^file:/i.test(rawUrl)) {
    const { resolvedPath } = await resolveReadableFileForIpc(rawUrl, { purpose: 'Image file' })
    const buffer = await fs.promises.readFile(resolvedPath)
    return { buffer, mimeType: mimeTypeForPath(resolvedPath) }
  }

  const parsed = new URL(rawUrl)
  const client = parsed.protocol === 'https:' ? https : http
  return new Promise((resolve, reject) => {
    const req = client.get(parsed, res => {
      if ((res.statusCode || 500) >= 400) {
        reject(new Error(`Failed to fetch ${rawUrl}: ${res.statusCode}`))
        res.resume()
        return
      }
      const chunks = []
      res.on('error', reject)
      res.on('data', chunk => chunks.push(chunk))
      res.on('end', () => {
        resolve({
          buffer: Buffer.concat(chunks),
          mimeType: res.headers['content-type'] || 'application/octet-stream'
        })
      })
    })
    req.on('error', reject)
  })
}

async function copyImageFromUrl(rawUrl) {
  const { buffer } = await resourceBufferFromUrl(rawUrl)
  const image = nativeImage.createFromBuffer(buffer)
  if (image.isEmpty()) throw new Error('Could not read image')
  clipboard.writeImage(image)
}

async function saveImageFromUrl(rawUrl) {
  const { buffer, mimeType } = await resourceBufferFromUrl(rawUrl)
  const fallbackName = filenameFromUrl(rawUrl, `image${extensionForMimeType(mimeType) || '.png'}`)
  const result = await dialog.showSaveDialog(mainWindow, {
    title: 'Save Image',
    defaultPath: fallbackName
  })
  if (result.canceled || !result.filePath) return false
  await fs.promises.writeFile(result.filePath, buffer)
  return true
}

async function writeComposerImage(buffer, ext = '.png') {
  const rawExt = String(ext || '.png')
    .trim()
    .toLowerCase()
  const normalizedExt = rawExt.startsWith('.') ? rawExt : `.${rawExt}`
  const safeExt = /^\.[a-z0-9]{1,5}$/.test(normalizedExt) ? normalizedExt : '.png'
  const dir = path.join(app.getPath('userData'), 'composer-images')
  await fs.promises.mkdir(dir, { recursive: true })
  const stamp = new Date().toISOString().replace(/[:.]/g, '-').replace('T', '_').replace('Z', '')
  const random = crypto.randomBytes(3).toString('hex')
  const filePath = path.join(dir, `composer_${stamp}_${random}${safeExt}`)
  await fs.promises.writeFile(filePath, buffer)
  return filePath
}

function previewLabelForUrl(url) {
  return `${url.host}${url.pathname === '/' ? '' : url.pathname}`
}

function expandUserPath(filePath) {
  const value = String(filePath || '').trim()

  if (value === '~') {
    return app.getPath('home')
  }

  if (value.startsWith(`~${path.sep}`) || value.startsWith('~/')) {
    return path.join(app.getPath('home'), value.slice(2))
  }

  return value
}

async function previewFileTarget(rawTarget, baseDir) {
  const raw = String(rawTarget || '').trim()
  const base = baseDir ? path.resolve(expandUserPath(baseDir)) : resolveHermesCwd()
  let resolved = resolveRequestedPathForIpc(/^file:/i.test(raw) ? raw : expandUserPath(raw), {
    baseDir: base,
    purpose: 'Preview target'
  })

  if (directoryExists(resolved)) {
    resolved = path.join(resolved, 'index.html')
  }

  const ext = path.extname(resolved).toLowerCase()
  if (!fileExists(resolved)) {
    return null
  }

  ;({ resolvedPath: resolved } = await resolveReadableFileForIpc(resolved, { purpose: 'Preview target' }))

  const mimeType = mimeTypeForPath(resolved)
  const metadata = previewFileMetadata(resolved, mimeType)
  const isHtml = PREVIEW_HTML_EXTENSIONS.has(ext)
  const isImage = mimeType.startsWith('image/')
  const previewKind = isHtml ? 'html' : isImage ? 'image' : metadata.binary ? 'binary' : 'text'

  return {
    binary: metadata.binary,
    byteSize: metadata.byteSize,
    kind: 'file',
    large: metadata.large,
    label: path.basename(resolved),
    language: PREVIEW_LANGUAGE_BY_EXT[ext] || 'text',
    mimeType,
    path: resolved,
    previewKind,
    source: raw,
    url: pathToFileURL(resolved).toString()
  }
}

function previewUrlTarget(rawTarget) {
  const raw = String(rawTarget || '').trim()
  const url = new URL(raw)

  if (!['http:', 'https:'].includes(url.protocol)) {
    return null
  }

  if (!LOCAL_PREVIEW_HOSTS.has(url.hostname.toLowerCase())) {
    return null
  }

  if (url.hostname === '0.0.0.0') {
    url.hostname = '127.0.0.1'
  }

  return {
    kind: 'url',
    label: previewLabelForUrl(url),
    source: raw,
    url: url.toString()
  }
}

async function normalizePreviewTarget(rawTarget, baseDir) {
  const raw = String(rawTarget || '').trim()

  if (!raw) {
    return null
  }

  try {
    if (/^https?:\/\//i.test(raw)) {
      return previewUrlTarget(raw)
    }

    return await previewFileTarget(raw, baseDir)
  } catch {
    return null
  }
}

async function filePathFromPreviewUrl(rawUrl) {
  const { resolvedPath } = await resolveReadableFileForIpc(String(rawUrl || ''), { purpose: 'Preview file' })
  return resolvedPath
}

function sendPreviewFileChanged(payload) {
  if (!mainWindow || mainWindow.isDestroyed()) return
  const { webContents } = mainWindow
  if (!webContents || webContents.isDestroyed()) return
  webContents.send('hermes:preview-file-changed', payload)
}

async function watchPreviewFile(rawUrl) {
  const filePath = await filePathFromPreviewUrl(rawUrl)
  const watchDir = path.dirname(filePath)
  const targetName = path.basename(filePath)
  const id = crypto.randomBytes(12).toString('base64url')
  let timer = null
  const watcher = fs.watch(watchDir, (_eventType, filename) => {
    const changedName = filename ? path.basename(String(filename)) : ''

    if (changedName && changedName !== targetName) {
      return
    }

    if (timer) clearTimeout(timer)
    timer = setTimeout(() => {
      timer = null
      if (!fileExists(filePath)) return
      sendPreviewFileChanged({ id, path: filePath, url: pathToFileURL(filePath).toString() })
    }, PREVIEW_WATCH_DEBOUNCE_MS)
  })

  previewWatchers.set(id, {
    close: () => {
      if (timer) clearTimeout(timer)
      watcher.close()
    }
  })

  return { id, path: filePath }
}

function stopPreviewFileWatch(id) {
  const watcher = previewWatchers.get(id)

  if (!watcher) {
    return false
  }

  watcher.close()
  previewWatchers.delete(id)

  return true
}

function closePreviewWatchers() {
  for (const id of previewWatchers.keys()) {
    stopPreviewFileWatch(id)
  }
}

async function waitForHermes(baseUrl, token) {
  const deadline = Date.now() + 45_000
  let lastError = null

  while (Date.now() < deadline) {
    try {
      await fetchJson(`${baseUrl}/api/status`, token)
      return
    } catch (error) {
      lastError = error
      await new Promise(resolve => setTimeout(resolve, 500))
    }
  }

  throw new Error(`Hermes backend did not become ready: ${lastError?.message || 'timeout'}`)
}

function getWindowButtonPosition() {
  if (!IS_MAC) return null
  return mainWindow?.getWindowButtonPosition?.() || WINDOW_BUTTON_POSITION
}

function getNativeOverlayWidth() {
  // macOS reports traffic-light coords via windowButtonPosition; the
  // titlebarOverlay there doesn't reserve right-edge space. Windows/Linux
  // render the native window-controls overlay on the right, so the renderer
  // needs to inset its right cluster by this much to clear them.
  return IS_MAC ? 0 : NATIVE_OVERLAY_BUTTON_WIDTH
}

function getWindowState() {
  return {
    isFullscreen: Boolean(mainWindow?.isFullScreen?.()),
    nativeOverlayWidth: getNativeOverlayWidth(),
    windowButtonPosition: getWindowButtonPosition()
  }
}

function sendBackendExit(payload) {
  if (!mainWindow || mainWindow.isDestroyed()) return
  const { webContents } = mainWindow
  if (!webContents || webContents.isDestroyed()) return
  webContents.send('hermes:backend-exit', payload)
}

function sendClosePreviewRequested() {
  if (!mainWindow || mainWindow.isDestroyed()) return
  const { webContents } = mainWindow
  if (!webContents || webContents.isDestroyed()) return
  webContents.send('hermes:close-preview-requested')
}

// Tell the renderer the machine just woke. Sleep silently drops the
// renderer's WebSocket to the local backend; the renderer reconnects on this
// signal so the chat composer doesn't stay stuck on "Starting Hermes...".
function sendPowerResume() {
  if (!mainWindow || mainWindow.isDestroyed()) return
  const { webContents } = mainWindow
  if (!webContents || webContents.isDestroyed()) return
  webContents.send('hermes:power-resume')
}

let powerResumeRegistered = false

function registerPowerResumeListeners() {
  if (powerResumeRegistered) return
  powerResumeRegistered = true
  try {
    // 'resume' covers sleep/wake; 'unlock-screen' covers lock/unlock without a
    // full suspend. Either can drop an idle socket.
    powerMonitor.on('resume', sendPowerResume)
    powerMonitor.on('unlock-screen', sendPowerResume)
  } catch {
    // powerMonitor is unavailable before app 'ready' on some platforms; the
    // caller registers after 'ready', so this should not normally throw.
  }
}

function getAppIconPath() {
  return APP_ICON_PATHS.find(fileExists)
}

function sendOpenUpdatesRequested() {
  if (!mainWindow || mainWindow.isDestroyed()) return
  const { webContents } = mainWindow
  if (!webContents || webContents.isDestroyed()) return
  webContents.send('hermes:open-updates')
  if (!mainWindow.isVisible()) mainWindow.show()
  mainWindow.focus()
}

function sendWindowStateChanged(nextIsFullscreen) {
  if (!mainWindow || mainWindow.isDestroyed()) return
  const { webContents } = mainWindow
  if (!webContents || webContents.isDestroyed()) return
  const state = getWindowState()

  if (typeof nextIsFullscreen === 'boolean') {
    state.isFullscreen = nextIsFullscreen
  }

  webContents.send('hermes:window-state-changed', state)
}

function buildApplicationMenu() {
  const template = []
  const checkForUpdatesItem = {
    label: 'Check for Updates…',
    click: () => sendOpenUpdatesRequested()
  }
  if (IS_MAC) {
    template.push({
      label: APP_NAME,
      submenu: [
        { label: `About ${APP_NAME}`, click: () => showAboutPanelFresh() },
        checkForUpdatesItem,
        { type: 'separator' },
        { role: 'services' },
        { type: 'separator' },
        { role: 'hide' },
        { role: 'hideOthers' },
        { role: 'unhide' },
        { type: 'separator' },
        { role: 'quit' }
      ]
    })
  }

  template.push({
    label: 'File',
    submenu: [
      IS_MAC
        ? {
            accelerator: 'CommandOrControl+W',
            click: () => {
              if (previewShortcutActive) {
                sendClosePreviewRequested()
              } else {
                mainWindow?.close()
              }
            },
            label: 'Close'
          }
        : { role: 'quit' }
    ]
  })
  template.push({
    label: 'Edit',
    submenu: [
      { role: 'undo' },
      { role: 'redo' },
      { type: 'separator' },
      { role: 'cut' },
      { role: 'copy' },
      { role: 'paste' },
      { role: 'delete' },
      { role: 'selectAll' }
    ]
  })
  template.push({
    label: 'View',
    submenu: [
      { role: 'reload' },
      { role: 'forceReload' },
      { role: 'toggleDevTools' },
      { type: 'separator' },
      {
        label: 'Actual Size',
        accelerator: 'CommandOrControl+0',
        click: () => {
          setAndPersistZoomLevel(mainWindow, 0)
        }
      },
      {
        label: 'Zoom In',
        accelerator: 'CommandOrControl+Plus',
        click: () => {
          if (mainWindow && !mainWindow.isDestroyed()) {
            setAndPersistZoomLevel(mainWindow, mainWindow.webContents.getZoomLevel() + 0.1)
          }
        }
      },
      {
        label: 'Zoom Out',
        accelerator: 'CommandOrControl+-',
        click: () => {
          if (mainWindow && !mainWindow.isDestroyed()) {
            setAndPersistZoomLevel(mainWindow, mainWindow.webContents.getZoomLevel() - 0.1)
          }
        }
      },
      { type: 'separator' },
      { role: 'togglefullscreen' }
    ]
  })
  template.push({
    label: 'Window',
    submenu: IS_MAC
      ? [{ role: 'minimize' }, { role: 'zoom' }, { role: 'front' }]
      : [{ role: 'minimize' }, { role: 'close' }]
  })
  template.push({
    label: 'Help',
    role: 'help',
    submenu: [checkForUpdatesItem]
  })

  return Menu.buildFromTemplate(template)
}

function toggleDevTools(window) {
  // DevTools is enabled in packaged builds so users can diagnose renderer
  // issues without needing a dev build. Trade-off: tiny attack surface
  // increase versus a much better support story when WS connection or
  // CSP issues surface in the field.
  const { webContents } = window
  if (webContents.isDevToolsOpened()) {
    webContents.closeDevTools()
  } else {
    webContents.openDevTools({ mode: 'detach' })
  }
}

function installDevToolsShortcut(window) {
  // F12 / Cmd+Opt+I works in both dev and packaged builds.
  window.webContents.on('before-input-event', (event, input) => {
    const key = input.key.toLowerCase()
    const isInspectShortcut =
      input.key === 'F12' ||
      (IS_MAC && input.meta && input.alt && key === 'i') ||
      (!IS_MAC && input.control && input.shift && key === 'i')
    if (!isInspectShortcut) return
    event.preventDefault()
    toggleDevTools(window)
  })
}

function installPreviewShortcut(window) {
  window.webContents.on('before-input-event', (event, input) => {
    const key = String(input.key || '').toLowerCase()
    const isPreviewCloseShortcut = key === 'w' && (IS_MAC ? input.meta : input.control) && !input.alt && !input.shift

    if (!isPreviewCloseShortcut || !previewShortcutActive) return

    event.preventDefault()
    sendClosePreviewRequested()
  })
}

// Zoom level is persisted in the renderer's own localStorage (per-origin,
// survives reloads/restarts) rather than a main-process JSON file. The main
// process owns setZoomLevel, so we mirror each change into localStorage and
// read it back on did-finish-load to re-apply after reloads or crash recovery.
const ZOOM_STORAGE_KEY = 'hermes:desktop:zoomLevel'

function clampZoomLevel(value) {
  if (!Number.isFinite(value)) return 0
  return Math.min(Math.max(value, -9), 9)
}

function setAndPersistZoomLevel(window, zoomLevel) {
  if (!window || window.isDestroyed()) return
  const next = clampZoomLevel(zoomLevel)
  window.webContents.setZoomLevel(next)
  window.webContents
    .executeJavaScript(
      `try { localStorage.setItem(${JSON.stringify(ZOOM_STORAGE_KEY)}, ${JSON.stringify(String(next))}) } catch {}`
    )
    .catch(error => rememberLog(`[zoom] persist failed: ${error?.message || error}`))
}

function restorePersistedZoomLevel(window) {
  if (!window || window.isDestroyed()) return
  window.webContents
    .executeJavaScript(
      `(() => { try { return localStorage.getItem(${JSON.stringify(ZOOM_STORAGE_KEY)}) } catch { return null } })()`
    )
    .then(stored => {
      if (stored == null || !window || window.isDestroyed()) return
      const level = clampZoomLevel(Number(stored))
      window.webContents.setZoomLevel(level)
    })
    .catch(error => rememberLog(`[zoom] restore failed: ${error?.message || error}`))
}

function installZoomShortcuts(window) {
  // Override Ctrl/Cmd + +/-/0 with half the default zoom step (0.1 vs 0.2).
  // The menu items handle this on macOS (where the menu is always present),
  // but on Linux/Windows the menu is null and Chromium's default handler
  // would use the full 0.2 step, so we intercept here for consistency.
  const ZOOM_STEP = 0.1
  window.webContents.on('before-input-event', (event, input) => {
    const mod = IS_MAC ? input.meta : input.control
    if (!mod || input.alt || input.shift) return

    const key = input.key
    if (key === '0') {
      event.preventDefault()
      setAndPersistZoomLevel(window, 0)
    } else if (key === '=' || key === '+') {
      event.preventDefault()
      setAndPersistZoomLevel(window, window.webContents.getZoomLevel() + ZOOM_STEP)
    } else if (key === '-') {
      event.preventDefault()
      setAndPersistZoomLevel(window, window.webContents.getZoomLevel() - ZOOM_STEP)
    }
  })
}

function installContextMenu(window) {
  window.webContents.on('context-menu', (_event, params) => {
    const template = []
    const hasSelection = Boolean(params.selectionText?.trim())
    const hasImage = params.mediaType === 'image' && Boolean(params.srcURL)
    const hasLink = Boolean(params.linkURL)
    const isEditable = Boolean(params.isEditable)

    if (hasImage) {
      template.push(
        {
          label: 'Open Image',
          click: () => {
            if (params.srcURL && !params.srcURL.startsWith('data:')) {
              openExternalUrl(params.srcURL)
            }
          },
          enabled: !params.srcURL.startsWith('data:')
        },
        {
          label: 'Copy Image',
          click: () => {
            void copyImageFromUrl(params.srcURL).catch(error => rememberLog(`Copy image failed: ${error.message}`))
          }
        },
        {
          label: 'Copy Image Address',
          click: () => clipboard.writeText(params.srcURL)
        },
        {
          label: 'Save Image As...',
          click: () => {
            void saveImageFromUrl(params.srcURL).catch(error => rememberLog(`Save image failed: ${error.message}`))
          }
        }
      )
    }

    if (hasLink) {
      if (template.length) template.push({ type: 'separator' })
      template.push(
        {
          label: 'Open Link',
          click: () => openExternalUrl(params.linkURL)
        },
        {
          label: 'Copy Link',
          click: () => clipboard.writeText(params.linkURL)
        }
      )
    }

    // Spell-check suggestions for the misspelled word under the caret.
    // Chromium surfaces them on `params.dictionarySuggestions`; we offer the
    // top 5 plus a "Add to dictionary" affordance.
    const suggestions = Array.isArray(params.dictionarySuggestions) ? params.dictionarySuggestions : []

    if (isEditable && params.misspelledWord && suggestions.length > 0) {
      if (template.length) template.push({ type: 'separator' })

      for (const suggestion of suggestions.slice(0, 5)) {
        template.push({
          label: suggestion,
          click: () => window.webContents.replaceMisspelling(suggestion)
        })
      }

      template.push({ type: 'separator' })
      template.push({
        label: 'Add to dictionary',
        click: () => window.webContents.session.addWordToSpellCheckerDictionary(params.misspelledWord)
      })
    }

    if (hasSelection || isEditable) {
      if (template.length) template.push({ type: 'separator' })
      if (isEditable) {
        template.push(
          { role: 'cut', enabled: params.editFlags.canCut },
          { role: 'copy', enabled: params.editFlags.canCopy },
          { role: 'paste', enabled: params.editFlags.canPaste },
          { type: 'separator' },
          { role: 'selectAll', enabled: params.editFlags.canSelectAll }
        )
      } else {
        template.push({ role: 'copy', enabled: params.editFlags.canCopy })
      }
    }

    if (!template.length) {
      template.push({ role: 'selectAll' })
    }

    Menu.buildFromTemplate(template).popup({ window })
  })
}

// Microphone capture for the voice composer. The renderer drives mic access
// through getUserMedia, which Chromium gates behind these two session hooks.
//
// The naive `details.mediaTypes.includes('audio')` check works on macOS but
// breaks on Windows: Chromium frequently fires the mic permission request with
// an empty/undefined `mediaTypes`, so the strict check denies it and
// getUserMedia throws NotAllowedError ("Microphone permission was denied").
// We therefore treat an audio-capture request as allowed whenever it's the
// 'media'/'audioCapture' permission AND mediaTypes either includes 'audio' OR
// is empty/absent (the Windows case). Video is still denied.
function isAudioCapturePermission(permission, details) {
  if (permission === 'audioCapture') {
    return true
  }
  if (permission !== 'media') {
    return false
  }
  const mediaTypes = details?.mediaTypes
  if (!Array.isArray(mediaTypes) || mediaTypes.length === 0) {
    // Windows: mediaTypes is often empty for a mic request. Don't deny on
    // missing metadata. (A video request would carry mediaTypes:['video'].)
    return true
  }
  return mediaTypes.includes('audio') && !mediaTypes.includes('video')
}

function installMediaPermissions() {
  // Async request handler: the prompt-style path (most platforms).
  session.defaultSession.setPermissionRequestHandler((_webContents, permission, callback, details) => {
    callback(isAudioCapturePermission(permission, details))
  })

  // Synchronous check handler: Chromium consults this for getUserMedia on
  // Windows in addition to (or instead of) the request handler. Without it,
  // the check defaults to false and the mic is denied before the request
  // handler ever runs.
  session.defaultSession.setPermissionCheckHandler((_webContents, permission, _origin, details) => {
    if (permission === 'media' || permission === 'audioCapture') {
      // details.mediaType is a single string here (not the mediaTypes array).
      const mediaType = details?.mediaType
      if (mediaType === 'video') {
        return false
      }

      return true
    }

    return false
  })
}

// ---------------------------------------------------------------------------
// OAuth remote-gateway auth.
//
// Hosted Hermes gateways gate the dashboard behind an OAuth provider (e.g.
// Nous Research) instead of a static session token. The auth model is
// fundamentally different from the token path:
//
//   * REST is authed by HttpOnly session cookies (``hermes_session_at``),
//     established by a browser redirect round-trip (/login → IDP →
//     /auth/callback sets cookies). We cannot read the HttpOnly cookie value
//     in JS — instead we let an Electron BrowserWindow complete the round
//     trip into a PERSISTENT session partition, and thereafter route our REST
//     through Electron's ``net`` bound to that same partition so the cookie
//     jar attaches the cookie automatically.
//   * WebSocket upgrades require a single-use ``?ticket=`` minted at
//     ``POST /api/auth/ws-ticket`` (cookie-authed). The legacy ``?token=``
//     path is unconditionally rejected by gated gateways.
//   * Nous Portal now issues a 24h ROTATING, reuse-detected refresh token
//     alongside the ~15-min access token (Portal NAS #293 / hermes #37247).
//     Both are set as HttpOnly cookies (``hermes_session_at`` ~15 min,
//     ``hermes_session_rt`` 24h). When the AT cookie lapses but the RT cookie
//     is still alive, the gateway middleware transparently rotates a fresh AT
//     on the next authenticated request — so connectivity must NOT be gated on
//     the AT cookie alone. We probe liveness by actually minting a ws-ticket
//     (which triggers that server-side refresh) and treat a real 401 as
//     "needs re-login"; the AT-or-RT cookie presence check is only a cheap
//     "is the user signed in at all?" gate / display signal.
// ---------------------------------------------------------------------------

const OAUTH_SESSION_PARTITION = 'persist:hermes-remote-oauth'

function getOauthSession() {
  if (oauthSession || !app.isReady()) return oauthSession
  oauthSession = session.fromPartition(OAUTH_SESSION_PARTITION)
  return oauthSession
}

// Bare + prefixed variants of the session cookies live in
// connection-config.cjs (cookiesHaveSession / cookiesHaveLiveSession). See
// that module for details.

async function hasOauthSessionCookie(baseUrl) {
  const sess = getOauthSession()
  if (!sess) return false
  const parsed = new URL(baseUrl)
  try {
    // Query by URL so the cookie jar applies Domain/Path/Secure scoping for us.
    const cookies = await sess.cookies.get({ url: baseUrl })
    return cookiesHaveSession(cookies)
  } catch {
    // Fall back to a host match if the URL query path errors.
    try {
      const cookies = await sess.cookies.get({ domain: parsed.hostname })
      return cookiesHaveSession(cookies)
    } catch {
      return false
    }
  }
}

// Like hasOauthSessionCookie, but returns true when EITHER a live access-token
// cookie OR a (longer-lived) refresh-token cookie is present. This is the right
// "is the user signed in at all?" check: an expired AT with a live RT is still
// a connectable session because the gateway rotates a fresh AT server-side on
// the next authenticated request. Gating on the AT alone forces a needless full
// re-login every ~15 min. Used for the Settings "connected" indicator and as a
// cheap early-out before attempting a network round-trip in resolveRemoteBackend.
async function hasLiveOauthSession(baseUrl) {
  const sess = getOauthSession()
  if (!sess) return false
  const parsed = new URL(baseUrl)
  try {
    const cookies = await sess.cookies.get({ url: baseUrl })
    return cookiesHaveLiveSession(cookies)
  } catch {
    try {
      const cookies = await sess.cookies.get({ domain: parsed.hostname })
      return cookiesHaveLiveSession(cookies)
    } catch {
      return false
    }
  }
}

async function clearOauthSession(baseUrl) {
  const sess = getOauthSession()
  if (!sess) return
  try {
    const cookies = await sess.cookies.get(baseUrl ? { url: baseUrl } : {})
    await Promise.all(
      cookies.map(c => {
        const scheme = c.secure ? 'https' : 'http'
        const cookieUrl = `${scheme}://${c.domain.replace(/^\./, '')}${c.path || '/'}`
        return sess.cookies.remove(cookieUrl, c.name).catch(() => undefined)
      })
    )
  } catch {
    // Best effort — a stale cookie self-expires anyway.
  }
}

// Open the gateway's /login page in a visible window using the OAuth session
// partition, and resolve once the access-token cookie appears (login done) or
// reject if the user closes the window first. The window navigates through the
// IDP and back to /auth/callback, which sets the session cookies on the
// partition; we poll the cookie jar rather than try to read the HttpOnly value.
function openOauthLoginWindow(baseUrl) {
  return new Promise((resolve, reject) => {
    if (!app.isReady()) {
      reject(new Error('Desktop is not ready to start an OAuth login.'))
      return
    }
    const sess = getOauthSession()
    if (!sess) {
      reject(new Error('OAuth session partition is unavailable.'))
      return
    }

    let settled = false
    let win = null
    let pollTimer = null

    const finish = err => {
      if (settled) return
      settled = true
      if (pollTimer) clearInterval(pollTimer)
      try {
        if (win && !win.isDestroyed()) win.destroy()
      } catch {
        // window already torn down
      }
      if (err) reject(err)
      else resolve({ baseUrl, ok: true })
    }

    const checkCookie = async () => {
      if (settled) return
      if (await hasOauthSessionCookie(baseUrl)) finish(null)
    }

    try {
      win = new BrowserWindow({
        width: 520,
        height: 720,
        title: '登录 APEX',
        autoHideMenuBar: true,
        webPreferences: {
          contextIsolation: true,
          nodeIntegration: false,
          sandbox: true,
          session: sess,
          webSecurity: true
        }
      })
    } catch (error) {
      finish(error instanceof Error ? error : new Error(String(error)))
      return
    }

    // Re-check the cookie jar on every successful navigation (the callback
    // redirect is the moment cookies get set) plus a low-frequency poll as a
    // belt-and-braces fallback for IDPs that finish via in-page JS.
    win.webContents.on('did-navigate', () => void checkCookie())
    win.webContents.on('did-redirect-navigation', () => void checkCookie())
    win.webContents.on('did-frame-navigate', () => void checkCookie())
    pollTimer = setInterval(() => void checkCookie(), 750)

    win.on('closed', () => {
      if (!settled) finish(new Error('Login window closed before authentication completed.'))
    })

    // ``next`` is intentionally omitted: the gateway lands on ``/`` after
    // login, which is a valid authenticated page that sets the cookies. We
    // only care that the cookie jar is populated.
    const loginUrl = `${normalizeRemoteBaseUrl(baseUrl)}/login`
    win.loadURL(loginUrl).catch(error => {
      finish(error instanceof Error ? error : new Error(String(error)))
    })
  })
}

// JSON request routed through the OAuth session partition so the HttpOnly
// session cookie is attached automatically by Electron's net stack. Used for
// authed REST against a gated gateway, including minting WS tickets.
function fetchJsonViaOauthSession(url, options = {}) {
  return new Promise((resolve, reject) => {
    const sess = getOauthSession()
    if (!sess) {
      reject(new Error('OAuth session partition is unavailable.'))
      return
    }
    let parsed
    try {
      parsed = new URL(url)
    } catch (error) {
      reject(new Error(`Invalid URL: ${error.message}`))
      return
    }
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      reject(new Error(`Unsupported Hermes backend URL protocol: ${parsed.protocol}`))
      return
    }
    const body = serializeJsonBody(options.body)
    const timeoutMs = resolveTimeoutMs(options.timeoutMs, DEFAULT_FETCH_TIMEOUT_MS)

    const request = electronNet.request({
      method: options.method || 'GET',
      url,
      session: sess,
      useSessionCookies: true,
      redirect: 'follow'
    })
    setJsonRequestHeaders(request)

    let timedOut = false
    const timer = setTimeout(() => {
      timedOut = true
      try {
        request.abort()
      } catch {
        // already finished
      }
      reject(new Error(`Timed out connecting to Hermes backend after ${timeoutMs}ms`))
    }, timeoutMs)

    request.on('response', res => {
      const chunks = []
      res.on('data', chunk => chunks.push(Buffer.from(chunk)))
      res.on('end', () => {
        if (timedOut) return
        clearTimeout(timer)
        const text = Buffer.concat(chunks).toString('utf8')
        const statusCode = res.statusCode || 500
        if (statusCode >= 400) {
          const err = new Error(`${statusCode}: ${text || ''}`)
          err.statusCode = statusCode
          reject(err)
          return
        }
        if (!text) {
          resolve(null)
          return
        }
        const looksHtml = /^\s*<(?:!doctype|html)/i.test(text)
        const contentType = String(res.headers['content-type'] || res.headers['Content-Type'] || '')
        if (looksHtml || contentType.includes('text/html')) {
          reject(new Error(`Expected JSON from ${url} but got HTML (status ${statusCode}).`))
          return
        }
        try {
          resolve(JSON.parse(text))
        } catch {
          reject(new Error(`Invalid JSON from ${url} (status ${statusCode}): ${text.slice(0, 200)}`))
        }
      })
    })
    request.on('error', error => {
      if (timedOut) return
      clearTimeout(timer)
      reject(error)
    })
    if (body) request.write(body)
    request.end()
  })
}

// Mint a single-use WS ticket for a gated gateway. Returns the ticket string.
// Throws (with statusCode 401) if the session cookie is missing/expired —
// callers treat that as "needs re-login".
async function mintGatewayWsTicket(baseUrl) {
  const body = await fetchJsonViaOauthSession(`${baseUrl}/api/auth/ws-ticket`, {
    method: 'POST',
    timeoutMs: 8_000
  })
  const ticket = body?.ticket
  if (!ticket || typeof ticket !== 'string') {
    throw new Error('Gateway did not return a WS ticket.')
  }
  return ticket
}

// Build a fresh WS URL for the *current* connection. Critical for reconnects:
// OAuth WS tickets are single-use with a ~30s TTL, so the ticket baked into
// the cached connection's wsUrl is stale on the second connect. The renderer
// calls this immediately before every gateway.connect() so each WS upgrade
// carries a freshly-minted ticket. For local/token connections this just
// reuses the static token (no minting needed).
async function freshGatewayWsUrl(profile) {
  // Mint for the requested profile's backend, NOT always the primary. The
  // renderer re-mints right before every gateway.connect(); when swapping to a
  // pooled profile we must return THAT backend's ws URL, otherwise the connect
  // silently lands back on the primary (default) backend and writes sessions to
  // the wrong profile's DB. A null/empty profile resolves to the primary, so
  // legacy callers and single-profile users are unchanged.
  const connection = await ensureBackend(profile)
  if (connection.authMode === 'oauth') {
    const ticket = await mintGatewayWsTicket(connection.baseUrl)
    return buildGatewayWsUrlWithTicket(connection.baseUrl, ticket)
  }
  // Local/token: the cached wsUrl already carries the (long-lived) token.
  return connection.wsUrl
}

function encryptDesktopSecret(value) {
  return encryptDesktopSecretStrict(value, safeStorage)
}

function decryptDesktopSecret(secret) {
  if (!secret || typeof secret !== 'object') {
    return ''
  }

  const value = String(secret.value || '')

  if (!value) {
    return ''
  }

  if (secret.encoding === 'safeStorage') {
    try {
      return safeStorage.decryptString(Buffer.from(value, 'base64'))
    } catch {
      return ''
    }
  }

  return value
}

// Validate + normalize the per-profile remote overrides map read from disk.
// Drops malformed names/entries and keeps only the recognized fields so a
// hand-edited or stale connection.json can't inject junk into resolution.
function sanitizeConnectionProfiles(raw) {
  if (!raw || typeof raw !== 'object') {
    return {}
  }

  const out = {}
  for (const [name, entry] of Object.entries(raw)) {
    if (!entry || typeof entry !== 'object') {
      continue
    }
    if (name !== 'default' && !PROFILE_NAME_RE.test(name)) {
      continue
    }

    const cleaned = { mode: entry.mode === 'remote' ? 'remote' : 'local' }
    const url = String(entry.url || '').trim()
    if (url) {
      cleaned.url = url
    }
    cleaned.authMode = normAuthMode(entry.authMode)
    if (entry.token && typeof entry.token === 'object') {
      cleaned.token = entry.token
    }
    out[name] = cleaned
  }

  return out
}

function readDesktopConnectionConfig() {
  // Check if file changed on disk since last read (e.g. modified by another
  // process or an external tool).  Our own writes update the cache inline
  // via writeDesktopConnectionConfig, but external changes would be missed.
  let mtime = null
  try {
    mtime = fs.statSync(DESKTOP_CONNECTION_CONFIG_PATH).mtimeMs
  } catch {
    mtime = null
  }

  if (connectionConfigCache && connectionConfigCacheMtime === mtime) {
    return connectionConfigCache
  }

  let config = { mode: 'local', remote: {}, profiles: {} }

  try {
    const raw = fs.readFileSync(DESKTOP_CONNECTION_CONFIG_PATH, 'utf8')
    const parsed = JSON.parse(raw)

    if (parsed && typeof parsed === 'object') {
      const remote = parsed.remote && typeof parsed.remote === 'object' ? parsed.remote : {}
      // authMode lives on the remote sub-object: 'oauth' (cookie + ws-ticket)
      // or 'token' (legacy static session token). Default to 'token' for
      // backward compatibility with configs written before OAuth support.
      remote.authMode = remote.authMode === 'oauth' ? 'oauth' : 'token'
      config = {
        mode: parsed.mode === 'remote' ? 'remote' : 'local',
        remote,
        // Per-profile remote overrides: each profile may point at its own
        // backend (local spawn or its own remote URL). Preserved verbatim so
        // profileRemoteOverride() can resolve them; normalized lazily on save.
        profiles: sanitizeConnectionProfiles(parsed.profiles)
      }
    }
  } catch {
    // Missing or malformed connection settings should fall back to local.
  }

  connectionConfigCache = config
  connectionConfigCacheMtime = mtime

  return config
}

function writeDesktopConnectionConfig(config) {
  fs.mkdirSync(path.dirname(DESKTOP_CONNECTION_CONFIG_PATH), { recursive: true })
  writeFileAtomic(DESKTOP_CONNECTION_CONFIG_PATH, JSON.stringify(config, null, 2))
  connectionConfigCache = config
  connectionConfigCacheMtime = fs.statSync(DESKTOP_CONNECTION_CONFIG_PATH).mtimeMs
}

// ── ApexNodes managed-LLM credential persistence ────────────────────────────
// The provision-key response { api_key, base_url, model } is stored in
// apex-managed.json under userData — api_key encrypted (safeStorage, same as the
// remote-gateway token); base_url + model in clear (server-truth routing, not
// secrets). Read synchronously by seedDefaultModelConfig at boot, so it must be
// cheap and never throw.

function readManagedConfig() {
  try {
    const raw = fs.readFileSync(DESKTOP_MANAGED_CONFIG_PATH, 'utf8')
    const parsed = JSON.parse(raw)
    return parsed && typeof parsed === 'object' ? parsed : {}
  } catch {
    return {}
  }
}

// Normalize the stored (clear, non-secret) account descriptor for the account
// panel. Missing → empty object so callers can read fields safely.
function readManagedAccount(stored) {
  const account = stored && typeof stored.account === 'object' && stored.account ? stored.account : {}
  const str = value => (typeof value === 'string' ? value.trim() : '')
  return { email: str(account.email), name: str(account.name), plan: str(account.plan) }
}

// The stored managed config: { key, baseUrl, model, account, accessToken }. key
// is '' when none is stored / managed is disabled. Centralizes the "do we have a
// managed credential?" question for the boot seed, onboarding gate, and IPC
// status. The server is the source of truth for baseUrl/model (from
// provision-key); env defaults only fill gaps. `account` is display-only
// identity (email/name/plan), never a secret. `accessToken` is the login JWT
// (decrypted), kept ONLY so the boot 401-self-heal can re-provision a rotated
// relay key without a re-login — it is never used for authorization here beyond
// re-calling provision-key (server-validated). '' when none stored / env key.
function resolveManagedConfig() {
  if (!isManagedEnabled(process.env)) {
    return { key: '', baseUrl: '', model: '', account: { email: '', name: '', plan: '' }, accessToken: '' }
  }
  const endpoints = resolveApexEndpoints(process.env)
  const stored = readManagedConfig()
  const account = readManagedAccount(stored)
  // An explicit env key (e.g. a CI/dev/admin-provisioned key for real-machine
  // testing) wins over stored state, using env/default base_url + model. No JWT
  // in this path — an env key is managed out-of-band, so self-heal stays off
  // (shouldAttemptReprovision gates on hasToken).
  const fromEnv = String(process.env.APEXNODES_RELAY_KEY || '').trim()
  if (fromEnv) {
    return { key: fromEnv, baseUrl: endpoints.relayBaseUrl, model: endpoints.model, account, accessToken: '' }
  }
  return {
    key: decryptDesktopSecret(stored.relayKey),
    baseUrl: String(stored.baseUrl || '').trim() || endpoints.relayBaseUrl,
    model: String(stored.model || '').trim() || endpoints.model,
    account,
    accessToken: decryptDesktopSecret(stored.accessToken)
  }
}

// Just the decrypted relay key (or '') — thin wrapper for call sites that only
// need to answer "is the user signed in to managed?".
function resolveManagedRelayCredential() {
  return resolveManagedConfig().key
}

// Persist the provision-key result. Pass null/empty to clear. `provisioned` may
// carry an optional display-only `account` ({ email, name, plan }) captured from
// the login response / JWT claims — stored in clear (it is not a secret) so the
// account panel can render who is signed in. It may also carry the login JWT as
// `accessToken` — persisted ENCRYPTED (same safeStorage as relayKey) so the boot
// 401-self-heal can re-provision a rotated relay key without a re-login. The JWT
// lives 7 days server-side (ACCESS_TOKEN_EXPIRE_DAYS); once it expires,
// provision-key 401s and the self-heal stops (the user re-logs in via the normal
// flow). This rewrites the WHOLE record on every write (each follows a fresh
// provision), so a provision that carries no token simply stores none — we never
// resurrect a stale token, and clearing (no key) wipes the token too.
function writeManagedConfig(provisioned) {
  fs.mkdirSync(path.dirname(DESKTOP_MANAGED_CONFIG_PATH), { recursive: true })
  const key = provisioned && typeof provisioned.apiKey === 'string' ? provisioned.apiKey.trim() : ''
  const account = provisioned && provisioned.account ? readManagedAccount({ account: provisioned.account }) : null
  const accessToken = provisioned && typeof provisioned.accessToken === 'string' ? provisioned.accessToken.trim() : ''
  const next = key
    ? {
        relayKey: encryptDesktopSecret(key),
        baseUrl: String(provisioned.baseUrl || '').trim(),
        model: String(provisioned.model || '').trim(),
        ...(account && (account.email || account.name || account.plan) ? { account } : {}),
        ...(accessToken ? { accessToken: encryptDesktopSecret(accessToken) } : {}),
        savedAt: Date.now()
      }
    : {}
  writeFileAtomic(DESKTOP_MANAGED_CONFIG_PATH, JSON.stringify(next, null, 2))
}

function clearManagedRelayCredential() {
  try {
    fs.rmSync(DESKTOP_MANAGED_CONFIG_PATH, { force: true })
  } catch {
    // Best effort.
  }
}

// ── hc-444: desktop ↔ cloud Feishu credential bridge ────────────────────────
// The cloud Feishu line is complete (each user self-registers their own app; the
// creds live in agent_entries). This mirrors the user's OWN credential down to the
// desktop so the local runtime's Feishu adapter + lark tools light up. The
// app_secret is persisted ENCRYPTED (safeStorage, same as the managed relay key);
// app_id/domain/agent_name/status are non-secret and stored in clear. See
// apex-feishu.cjs for the pure shaping/gating helpers.

// Read + decrypt the stored Feishu credential into the normalized runtime shape
// ({ connected, appId, appSecret, domain, agentName, credentialStatus, syncedAt }).
// Synchronous + never throws (read at spawn time); a decrypt failure blanks the
// secret, which normalizeStoredFeishu degrades to `connected:false`.
function resolveFeishuConfig() {
  let raw
  try {
    raw = JSON.parse(fs.readFileSync(DESKTOP_FEISHU_CONFIG_PATH, 'utf8'))
  } catch {
    return normalizeStoredFeishu(null)
  }
  const appSecret = raw && typeof raw === 'object' ? decryptDesktopSecret(raw.appSecret) : ''
  // Hand normalizeStoredFeishu the record with the secret already decrypted; the
  // stored `appSecret` is ciphertext, so replace it with the plaintext (or '').
  return normalizeStoredFeishu(raw && typeof raw === 'object' ? { ...raw, appSecret } : null)
}

// Persist a fetched Feishu credential. Pass a parsed credential
// ({ appId, appSecret, domain, agentName, credentialStatus }) to store, or
// null/empty to clear. The whole record is rewritten each call (each follows a
// fresh fetch), so clearing wipes the secret too. app_secret is encrypted; the
// rest is clear (non-secret display/routing).
function writeFeishuConfig(credential) {
  fs.mkdirSync(path.dirname(DESKTOP_FEISHU_CONFIG_PATH), { recursive: true })
  const appId = credential && typeof credential.appId === 'string' ? credential.appId.trim() : ''
  const appSecret = credential && typeof credential.appSecret === 'string' ? credential.appSecret.trim() : ''
  const next =
    appId && appSecret
      ? {
          appId,
          appSecret: encryptDesktopSecret(appSecret),
          domain: String(credential.domain || '').trim() || 'feishu',
          agentName: String(credential.agentName || '').trim(),
          credentialStatus: String(credential.credentialStatus || '').trim(),
          syncedAt: Date.now()
        }
      : {}
  writeFileAtomic(DESKTOP_FEISHU_CONFIG_PATH, JSON.stringify(next, null, 2))
}

function clearFeishuConfig() {
  try {
    fs.rmSync(DESKTOP_FEISHU_CONFIG_PATH, { force: true })
  } catch {
    // Best effort.
  }
}

// Build the FEISHU_* spawn-env fragment for the local backend from the stored
// (decrypted) credential — but ADD-ONLY, never clobbering a FEISHU_APP_ID the
// parent env already set (a power-user / staging / CI that wants to test with
// their own app credential out-of-band). Mirrors the HF_ENDPOINT add-only rule in
// backend-env.cjs. Returns {} for a not-connected user, so a spread merge is a
// safe no-op. Called at spawn time (not cached) so a mid-session sync/disconnect
// takes effect on the next backend (re)start.
function desktopFeishuSpawnEnv() {
  // An explicit parent-env credential wins — leave it untouched.
  if (String(process.env.FEISHU_APP_ID || '').trim() && String(process.env.FEISHU_APP_SECRET || '').trim()) {
    return {}
  }
  return buildFeishuBackendEnv(resolveFeishuConfig())
}

// Fetch the signed-in user's Feishu credential from the cloud and persist it
// (encrypted). Authenticates with the STORED login JWT (the same encrypted JWT
// the managed self-heal reuses) — no re-login needed for a user already signed in
// to managed. Returns a status object the IPC layer relays to the renderer:
//   { ok, hasEntry, agentName, domain, credentialStatus, needsSignIn?, message? }
// NEVER throws; a fetch failure resolves ok:false with a message. The secret is
// never logged — only counts/flags are.
async function fetchAndStoreFeishuCredentials() {
  const managed = resolveManagedConfig()
  const token = String(managed.accessToken || '').trim()
  if (!token) {
    // No stored JWT → the user must sign in (managed) first; the renderer opens
    // the sign-in / web flow. Not an error — an expected pre-condition.
    return { ok: false, needsSignIn: true, hasEntry: false, message: 'NOT_SIGNED_IN' }
  }

  const endpoints = resolveApexEndpoints(process.env)
  let body
  try {
    body = await apexAuthGetJson(feishuCredentialsUrl(endpoints.apiBase), { bearer: token })
  } catch (error) {
    // A 401 means the stored JWT expired → treat as "needs sign-in" so the
    // renderer routes the user back through login; other errors are transient.
    if (error && error.statusCode === 401) {
      return { ok: false, needsSignIn: true, hasEntry: false, message: 'SESSION_EXPIRED' }
    }
    rememberLog(`[feishu-bridge] credential fetch failed: ${error && error.message ? error.message : error}`)
    return { ok: false, hasEntry: false, message: 'FETCH_FAILED' }
  }

  const parsed = parseFeishuCredentialsResponse(body)
  if (!parsed) {
    rememberLog('[feishu-bridge] credential response malformed')
    return { ok: false, hasEntry: false, message: 'FETCH_FAILED' }
  }

  if (!parsed.hasEntry) {
    // The user has not bound a Feishu app in the cloud yet — clear any stale
    // local credential and tell the renderer to guide them into the web flow.
    clearFeishuConfig()
    rememberLog('[feishu-bridge] no cloud Feishu entry for this user; guiding to web binding')
    return { ok: true, hasEntry: false, credentialStatus: parsed.credentialStatus }
  }

  writeFeishuConfig(parsed)
  rememberLog(
    `[feishu-bridge] synced Feishu credential (app ${parsed.appId}, domain ${parsed.domain}, status ${parsed.credentialStatus || 'unknown'})`
  )
  return {
    ok: true,
    hasEntry: true,
    agentName: parsed.agentName,
    domain: parsed.domain,
    credentialStatus: parsed.credentialStatus
  }
}

// ── Platform client-config sync (apex-client-config.cjs) ────────────────────
// The cloud serves a versioned client config (PUBLIC GET
// /api/v1/desktop/client-config). We refresh the on-disk cache fail-soft at
// boot and after every successful managed sign-in; the MAIN process then
// applies payload.config_yaml straight into config.yaml via line surgery
// BEFORE the gateway spawns (applyClientConfigToRuntime). The old renderer
// path — full-record /api/config round-trip — is retired: the dashboard GET
// normalizes the config for the web schema and silently drops keys outside it
// (custom_providers / skills / timezone), so PUT-ing that projection back
// wiped the relay registration on a live install. An offline user boots
// exactly as before — every failure path keeps the cached state.

function readClientConfigState() {
  try {
    const raw = fs.readFileSync(DESKTOP_CLIENT_CONFIG_PATH, 'utf8')
    return normalizeStoredClientConfig(JSON.parse(raw))
  } catch {
    return normalizeStoredClientConfig(null)
  }
}

function writeClientConfigState(next) {
  fs.mkdirSync(path.dirname(DESKTOP_CLIENT_CONFIG_PATH), { recursive: true })
  writeFileAtomic(DESKTOP_CLIENT_CONFIG_PATH, JSON.stringify(next, null, 2))
}

// Fetch the platform config and store it when a NEWER version arrived.
// Non-blocking by contract (callers `void` it), bounded (~5s), and never
// throws — any error only logs and leaves the cached state untouched.
async function refreshClientConfigFromPlatform(reason) {
  try {
    const stored = readClientConfigState()
    const fetched = await fetchClientConfig({
      apiBase: apexApiBase(),
      fetchJson: fetchPublicJson,
      knownVersion: stored.version,
      timeoutMs: 5_000,
      log: msg => rememberLog(msg)
    })
    if (!fetched) return // offline / 404 no-active-config / garbage → cache stands
    if (fetched.unchanged) {
      rememberLog(`[client-config] v${fetched.version} unchanged (${reason})`)
      return
    }
    if (!shouldApplyClientConfig(fetched.version, stored.version)) {
      rememberLog(
        `[client-config] fetched v${fetched.version} is not newer than cached v${stored.version}; ignoring (${reason})`
      )
      return
    }
    writeClientConfigState({
      version: fetched.version,
      payload: fetched.payload,
      fetchedAt: Date.now(),
      // Preserve what the renderer already applied — the gap between version
      // and appliedVersion is exactly what triggers the next apply pass.
      appliedVersion: stored.appliedVersion
    })
    rememberLog(`[client-config] stored platform config v${fetched.version} (${reason})`)
  } catch (error) {
    rememberLog(`[client-config] refresh failed (ignored): ${error && error.message ? error.message : error}`)
  }
}

// Product-critical config.yaml blocks watchdog. The dashboard's full-record
// config save (settings pages still use it) has at least once dropped blocks
// it didn't round-trip (custom_providers — killing relay routing with
// "Unknown provider 'custom:apex-nodes.com'" — plus skills/timezone). Exact
// writer unconfirmed (candidates: PUT denormalize, profile scoping, boot-time
// writer race) — this guard makes the whole CLASS non-fatal: whenever the file
// loses a product-critical block, restore it. Idempotent; append-only; never
// touches a block that exists.
function guardConfigYamlProductBlocks(reason) {
  try {
    const configPath = path.join(HERMES_HOME, 'config.yaml')
    if (!fs.existsSync(configPath)) return
    let raw = fs.readFileSync(configPath, 'utf8')
    const fixed = []

    const managed = resolveManagedConfig()
    const endpoints = resolveApexEndpoints(process.env)
    const relayEntryLines =
      `- api_key: ${managed.key}\n` +
      `  base_url: ${managed.baseUrl}\n` +
      `  model: ${endpoints.modelDisplay}\n` +
      `  name: ${MANAGED_PROVIDER_NAME}\n`

    if (managed.key && managed.baseUrl && !/^custom_providers:/m.test(raw)) {
      raw = raw.replace(/\n*$/, '\n') + 'custom_providers:\n' + relayEntryLines
      fixed.push('custom_providers')
    } else if (
      managed.key &&
      managed.baseUrl &&
      // Header present but NO list entry under it (a wiper once left the bare
      // header behind — the earlier existence check sailed right past it).
      /^custom_providers:[ \t]*\n(?![ \t]|- )/m.test(raw)
    ) {
      raw = raw.replace(/^custom_providers:[ \t]*\n/m, 'custom_providers:\n' + relayEntryLines)
      fixed.push('custom_providers(empty-header)')
    }

    // A /moa toggle has been observed persisting itself as the GLOBAL default
    // (model.provider: moa + base_url: moa://local) — every new chat then
    // opens on MoA (slow + expensive), violating selective routing, and the
    // relay URL is clobbered. MoA is per-session only on managed installs:
    // heal the model block back to the relay default.
    if (managed.key && managed.baseUrl && /^model:[\s\S]*?^\s{2}provider:\s*moa\s*$/m.test(raw)) {
      raw = raw
        .replace(/^(\s{2}provider:\s*)moa\s*$/m, `$1custom`)
        .replace(/^(\s{2}base_url:\s*)moa:\/\/local\s*$/m, `$1${managed.baseUrl}`)
        .replace(/^(\s{2}default:\s*)\S.*$/m, `$1${endpoints.modelDisplay}`)
      fixed.push('model(moa-global)')
    }

    // Skills: union the managed disabled names back in — add-only, so a user's
    // enable-toggles (names they removed) survive. This is BOTH the "block was
    // wiped entirely" heal AND the hc-406 UPGRADE path: an install seeded under
    // v0.17 keeps its old 49-name skills.disabled after a bump to v0.18, so the
    // newly-graded-OFF bundled skills (huggingface-hub / maps / plan) would ship
    // ACTIVE without this reconcile. seedSkillsBlockYaml is the append when the
    // block is wholly absent (ensureSkillsDisabledYaml delegates to it).
    const skillsHeal = ensureSkillsDisabledYaml(raw)
    if (skillsHeal.changed) {
      raw = skillsHeal.next
      fixed.push(`skills.disabled(+${skillsHeal.added.length})`)
    }

    if (!/^timezone:/m.test(raw)) {
      raw = raw.replace(/\n*$/, '\n') + "timezone: ''\n"
      fixed.push('timezone')
    }

    // hc-392 China profile: losing model.disabled_providers silently re-enables
    // the Copilot provider probe — GitHub is near-unreachable from the
    // mainland, so its probe saturates the gateway RPC pool (slow model list,
    // spinning @-completions). Re-pin it whenever the model block loses it.
    if (/^model:/m.test(raw) && !/^\s{2}disabled_providers:/m.test(raw)) {
      raw = raw.replace(/^(model:\n)/m, `$1${modelDisabledProvidersYaml()}`)
      fixed.push('model.disabled_providers')
    }

    // Standalone plugins are opt-in: a config.yaml without (or with an
    // emptied) plugins.enabled list silently disables apex-overlay + the
    // apexnodes-* tool plugins on the next backend start. Union the managed
    // names back in — add-only, so user-added plugin entries survive. This
    // boot-time pass is also the UPGRADE path for installs seeded before the
    // plugins block existed.
    const pluginsHeal = ensurePluginsEnabledYaml(raw)
    if (pluginsHeal.changed) {
      raw = pluginsHeal.next
      fixed.push(`plugins.enabled(+${pluginsHeal.added.length})`)
    }

    if (fixed.length) {
      fs.writeFileSync(configPath, raw, { encoding: 'utf8' })
      rememberLog(`[config-guard] restored missing block(s): ${fixed.join(', ')} (${reason})`)
    }
  } catch (err) {
    rememberLog(`[config-guard] skipped: ${err && err.message ? err.message : err}`)
  }
}

// Keep the guard live while the app runs: any writer (dashboard save, the
// runtime itself) that drops a product block gets healed within seconds. The
// watcher is best-effort — boot-time invocation is the reliable baseline.
let configGuardTimer = null
function watchConfigYamlProductBlocks() {
  try {
    const configPath = path.join(HERMES_HOME, 'config.yaml')
    if (!fs.existsSync(configPath)) return
    fs.watch(configPath, { persistent: false }, () => {
      clearTimeout(configGuardTimer)
      configGuardTimer = setTimeout(() => guardConfigYamlProductBlocks('watch'), 2_000)
    })
  } catch (err) {
    rememberLog(`[config-guard] watcher unavailable: ${err && err.message ? err.message : err}`)
  }
}

// Apply the cached platform config to config.yaml — main-process line surgery,
// run BEFORE the gateway spawns so the runtime loads the result fresh. Only
// scalar dotted keys are written (see applyConfigYamlKeys); all-or-nothing:
// appliedVersion advances only after a successful write, so a failure retries
// next boot. Fail-soft — a broken payload can never block booting.
function applyClientConfigToRuntime(reason) {
  try {
    const stored = readClientConfigState()
    if (!stored.version || stored.version <= (stored.appliedVersion || 0)) return
    const entries =
      stored.payload && typeof stored.payload === 'object' && stored.payload.config_yaml &&
      typeof stored.payload.config_yaml === 'object'
        ? stored.payload.config_yaml
        : null
    const configPath = path.join(HERMES_HOME, 'config.yaml')
    if (entries && Object.keys(entries).length > 0) {
      if (!fs.existsSync(configPath)) {
        // Seed hasn't produced a config yet (ultra-fresh install) — retry on
        // the next boot rather than inventing a file the seed would then skip.
        rememberLog(`[client-config] config.yaml absent; deferring v${stored.version} apply (${reason})`)
        return
      }
      const raw = fs.readFileSync(configPath, 'utf8')
      const { changed, next, applied, skipped } = applyConfigYamlKeys(raw, entries)
      if (changed) fs.writeFileSync(configPath, next, { encoding: 'utf8' })
      rememberLog(
        `[client-config] applied v${stored.version} (${reason}): ${applied.join(', ') || 'no-op'}` +
          (skipped.length ? `; skipped: ${skipped.join(', ')}` : '')
      )
    } else {
      rememberLog(`[client-config] v${stored.version} carries no config_yaml keys (${reason})`)
    }
    writeClientConfigState({ ...stored, appliedVersion: stored.version })
  } catch (error) {
    rememberLog(`[client-config] apply failed (will retry next boot): ${error && error.message ? error.message : error}`)
  }
}

// POST JSON to an ApexNodes auth endpoint, optionally with a Bearer JWT. Reuses
// the oauth-net-request helpers (serializeJsonBody / setJsonRequestHeaders) +
// Electron's net stack, the same transport fetchJsonViaOauthSession uses — but
// WITHOUT the OAuth cookie session (managed-LLM auth is JWT Bearer, a separate
// concern from the remote-gateway cookie jar).
function apexAuthPostJson(url, { body, bearer, timeoutMs = 12_000 } = {}) {
  return new Promise((resolve, reject) => {
    let parsed
    try {
      parsed = new URL(url)
    } catch (error) {
      reject(new Error(`Invalid ApexNodes URL: ${error.message}`))
      return
    }
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      reject(new Error(`Unsupported ApexNodes URL protocol: ${parsed.protocol}`))
      return
    }

    const payload = serializeJsonBody(body)
    const request = electronNet.request({ method: 'POST', url, redirect: 'follow' })
    setJsonRequestHeaders(request)
    if (bearer) {
      request.setHeader('Authorization', `Bearer ${bearer}`)
    }

    let timedOut = false
    const timer = setTimeout(() => {
      timedOut = true
      try {
        request.abort()
      } catch {
        // already finished
      }
      reject(new Error(`Timed out connecting to ApexNodes after ${timeoutMs}ms`))
    }, timeoutMs)

    request.on('response', res => {
      const chunks = []
      res.on('data', chunk => chunks.push(Buffer.from(chunk)))
      res.on('end', () => {
        if (timedOut) return
        clearTimeout(timer)
        const text = Buffer.concat(chunks).toString('utf8')
        const statusCode = res.statusCode || 500
        if (statusCode >= 400) {
          const err = new Error(`${statusCode}: ${text || ''}`)
          err.statusCode = statusCode
          reject(err)
          return
        }
        if (!text) {
          resolve(null)
          return
        }
        try {
          resolve(JSON.parse(text))
        } catch {
          reject(new Error(`Invalid JSON from ${url} (status ${statusCode}): ${text.slice(0, 200)}`))
        }
      })
    })
    request.on('error', error => {
      if (timedOut) return
      clearTimeout(timer)
      reject(error)
    })
    if (payload) request.write(payload)
    request.end()
  })
}

// Bearer-authed GET returning parsed JSON — the read counterpart to
// apexAuthPostJson (same electronNet transport + explicit timeout + statusCode on
// the rejection). Used by the hc-444 Feishu bridge to fetch the signed-in user's
// credential (GET /api/v1/desktop/feishu-credentials). A >=400 rejects with an
// Error carrying `.statusCode` so the caller can distinguish 401 (expired JWT →
// re-login) from a transient failure.
function apexAuthGetJson(url, { bearer, timeoutMs = 12_000 } = {}) {
  return new Promise((resolve, reject) => {
    let parsed
    try {
      parsed = new URL(url)
    } catch (error) {
      reject(new Error(`Invalid ApexNodes URL: ${error.message}`))
      return
    }
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      reject(new Error(`Unsupported ApexNodes URL protocol: ${parsed.protocol}`))
      return
    }

    const request = electronNet.request({ method: 'GET', url, redirect: 'follow' })
    request.setHeader('Accept', 'application/json')
    if (bearer) {
      request.setHeader('Authorization', `Bearer ${bearer}`)
    }

    let timedOut = false
    const timer = setTimeout(() => {
      timedOut = true
      try {
        request.abort()
      } catch {
        // already finished
      }
      reject(new Error(`Timed out connecting to ApexNodes after ${timeoutMs}ms`))
    }, timeoutMs)

    request.on('response', res => {
      const chunks = []
      res.on('data', chunk => chunks.push(Buffer.from(chunk)))
      res.on('end', () => {
        if (timedOut) return
        clearTimeout(timer)
        const text = Buffer.concat(chunks).toString('utf8')
        const statusCode = res.statusCode || 500
        if (statusCode >= 400) {
          const err = new Error(`${statusCode}: ${text || ''}`)
          err.statusCode = statusCode
          reject(err)
          return
        }
        if (!text) {
          resolve(null)
          return
        }
        try {
          resolve(JSON.parse(text))
        } catch {
          reject(new Error(`Invalid JSON from ${url} (status ${statusCode}): ${text.slice(0, 200)}`))
        }
      })
    })
    request.on('error', error => {
      if (timedOut) return
      clearTimeout(timer)
      reject(error)
    })
    request.end()
  })
}

// Probe the relay's OpenAI-compatible model listing with a Bearer relay key —
// the SAME `GET {base_url}/v1/models` the runtime's model picker calls to build
// its live "APEX-NODES.COM" model group. We only need the status code: 401/403
// means the stored relay key is dead (rotated out) and the picker list has
// collapsed; that is the self-heal trigger. Returns { ok, statusCode }; on a
// timeout / network error resolves { ok:false, statusCode:0 } (NOT an auth
// failure — we must not re-provision on a transient outage). Mirrors
// apexAuthPostJson's transport (electronNet + explicit timeout), GET + no body.
//
// base_url already ends at the relay `/v1` segment (see DEFAULT_RELAY_BASE_URL),
// so the listing path is `${base_url}/models`.
function apexRelayGetModels(baseUrl, key, { timeoutMs = 10_000 } = {}) {
  return new Promise(resolve => {
    const base = String(baseUrl || '').trim().replace(/\/+$/, '')
    const relayKey = String(key || '').trim()
    if (!base || !relayKey) {
      resolve({ ok: false, statusCode: 0 })
      return
    }
    const url = `${base}/models`
    let parsed
    try {
      parsed = new URL(url)
    } catch {
      resolve({ ok: false, statusCode: 0 })
      return
    }
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      resolve({ ok: false, statusCode: 0 })
      return
    }

    let settled = false
    const done = result => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      resolve(result)
    }
    const request = electronNet.request({ method: 'GET', url, redirect: 'follow' })
    request.setHeader('Authorization', `Bearer ${relayKey}`)
    request.setHeader('Accept', 'application/json')

    const timer = setTimeout(() => {
      try {
        request.abort()
      } catch {
        // already finished
      }
      done({ ok: false, statusCode: 0 })
    }, timeoutMs)

    request.on('response', res => {
      const statusCode = res.statusCode || 0
      // Drain so the socket can be reused/closed cleanly; body is irrelevant.
      res.on('data', () => {})
      res.on('end', () => done({ ok: statusCode >= 200 && statusCode < 400, statusCode }))
    })
    request.on('error', () => done({ ok: false, statusCode: 0 }))
    request.end()
  })
}

// Timestamp (ms) of the last self-heal re-provision ATTEMPT — module-level so the
// anti-storm cooldown (shouldAttemptReprovision) survives across boot probes
// within one app run. Reset only by restarting the app; a genuine rotation heals
// on the first attempt, so the cooldown only matters when re-provision keeps
// failing (expired JWT / provision-key down), which must not loop.
let lastManagedReprovisionAttemptAt = 0

// Boot self-heal: if the stored relay key is dead (relay /v1/models → 401/403),
// re-provision it in place using the stored login JWT, then re-sync the
// custom_providers entry so the model picker's live listing recovers THIS launch
// — fixing the "过几天列表缩水到只剩一个" bug without a manual re-login.
//
// Gated + rate-limited via the pure shouldAttemptReprovision (managed enabled +
// relay key present + login JWT present + cooldown elapsed), so BYOK / signed-out
// / env-key installs are strict no-ops. A probe that is anything other than a
// clean 401/403 (2xx, 5xx, timeout, offline) does NOTHING — we never burn the
// re-provision on a key that is actually fine or a relay that is merely down.
// If provision-key itself 401s (the stored JWT has also expired), we stop and
// log; re-login UX is the existing sign-in flow's job, not a popup storm.
//
// Fire-and-forget from the boot path (app.whenReady, alongside the client-config
// boot sync); never blocks the gateway spawn and only ever logs on failure.
// Returns a structured outcome so the on-demand caller (a renderer-reported
// runtime 401, hermes:managed:selfHeal) can act on it — apply + retry when
// healed, or route to re-sign-in when there is no reusable login token. The
// boot caller ignores the return (fire-and-forget). Shape:
//   { ok, relayUnauthorized, healed, hasToken }
//   - relayUnauthorized=false → relay accepted the key (or managed off / no
//     key): nothing to heal, not a managed-relay auth problem.
//   - relayUnauthorized=true, healed=true → fresh key minted + config re-synced.
//   - relayUnauthorized=true, healed=false, hasToken=false → seed/env key or a
//     cleared token: can't re-provision, the user must sign in again.
//   - relayUnauthorized=true, healed=false, hasToken=true → the stored JWT is
//     itself expired (provision-key rejected it) — sign in again.
async function selfHealManagedKeyOn401() {
  try {
    const managed = resolveManagedConfig()
    if (!isManagedEnabled(process.env)) return { ok: true, relayUnauthorized: false }
    if (!managed.key || !managed.baseUrl) return { ok: true, relayUnauthorized: false }

    // Cheap probe of the exact listing the picker uses. Only a hard auth
    // rejection is actionable — a 2xx/5xx/offline probe heals nothing and (for
    // an on-demand call) tells the renderer this wasn't a relay-auth failure.
    const probe = await apexRelayGetModels(managed.baseUrl, managed.key)
    if (!isRelayUnauthorized(probe.statusCode)) return { ok: true, relayUnauthorized: false }

    const hasToken = Boolean(managed.accessToken)
    if (
      !shouldAttemptReprovision({
        enabled: true,
        hasKey: Boolean(managed.key),
        hasToken,
        lastAttemptAt: lastManagedReprovisionAttemptAt,
        now: Date.now()
      })
    ) {
      if (!hasToken) {
        rememberLog(
          '[apexnodes] relay key rejected (401) but no stored login token to re-provision with; ' +
            'sign in again to refresh (self-heal skipped).'
        )
      }
      return { ok: true, relayUnauthorized: true, healed: false, hasToken }
    }

    lastManagedReprovisionAttemptAt = Date.now()
    rememberLog('[apexnodes] relay key rejected (401); auto re-provisioning with stored login token…')
    // Re-run the SAME provision chain the sign-in routes use: mints a fresh relay
    // key (server rotates), persists it (+ the — possibly unchanged — JWT), and
    // syncs the custom_providers entry. A stored account keeps the panel intact.
    const result = await provisionManagedFromAccessToken(managed.accessToken, managed.account || null)
    if (result && result.hasRelayKey) {
      rememberLog('[apexnodes] relay key self-heal succeeded; model picker list restored.')
      return { ok: true, relayUnauthorized: true, healed: true, hasToken: true }
    }
    // provision-key returned no key: JWT expired (401) or endpoint unavailable.
    // Stop here — the cooldown prevents a retry storm; the user re-logs in via
    // the normal flow when the token is truly dead.
    rememberLog(
      '[apexnodes] relay key self-heal could not re-provision (login token likely expired); ' +
        'sign in again to refresh.'
    )
    return { ok: true, relayUnauthorized: true, healed: false, hasToken: true }
  } catch (error) {
    rememberLog(`[apexnodes] relay key self-heal skipped: ${error && error.message ? error.message : error}`)
    return { ok: false, relayUnauthorized: false, healed: false, hasToken: false }
  }
}

// Shared post-auth path for EVERY managed sign-in route (email/password,
// Google, APEX-web). Given a platform access-token JWT, provision a relay-valid
// key for this user and persist it. Tolerates "provision-key not deployed yet"
// (404/501 or any fetch error): keeps the BYOK fallback rather than failing the
// sign-in, so a missing endpoint is NOT a login failure. base_url + model come
// FROM THE RESPONSE (server-truth).
//
// Returns { ok, hasRelayKey }:
//   - ok=true, hasRelayKey=true  → key + base_url + model stored; managed live.
//   - ok=true, hasRelayKey=false → token valid but provision-key unavailable —
//     caller falls back to BYOK.
async function provisionManagedFromAccessToken(accessToken, account = null) {
  const token = String(accessToken || '').trim()
  if (!token) {
    throw new Error('ApexNodes sign-in did not return an access token.')
  }

  const endpoints = resolveApexEndpoints(process.env)
  // Display-only identity for the account panel. Prefer a caller-supplied
  // account (email from the login body); always fold in the JWT claims as a
  // fallback so a browser-flow sign-in (no login body) still gets an email.
  const resolvedAccount = accountFromLogin(account || {}, token)

  let provisioned = null
  try {
    const body = await apexAuthPostJson(endpoints.provisionKeyUrl, {
      bearer: token,
      body: {}
    })
    provisioned = parseProvisionResponse(body, process.env)
  } catch (error) {
    rememberLog(
      `[apexnodes] provision-key unavailable (${error && error.message ? error.message : error}); ` +
        'managed default disabled, falling back to BYOK.'
    )
  }

  if (provisioned) {
    // The provision endpoint is JWT-authed and returns the signed-in user's own
    // email/name/plan — authoritative. Prefer it, falling back to the login-body
    // / JWT-claim values (a Google/browser sign-in JWT may omit the email).
    const account2 = {
      email: provisioned.email || resolvedAccount.email,
      name: provisioned.name || resolvedAccount.name,
      plan: provisioned.plan || resolvedAccount.plan
    }
    // Persist the login JWT (encrypted) alongside the fresh relay key so the boot
    // 401-self-heal can silently re-provision if this key is later rotated out.
    writeManagedConfig({ ...provisioned, account: account2, accessToken: token })
    // A re-login just ROTATED the relay key — refresh the registered custom
    // provider entry immediately so the model picker's live listing doesn't
    // run on the dead key until the next app restart.
    syncManagedCustomProviderKey()
    // A successful sign-in is a sync point for the platform client config
    // (contract: check at boot AND after every successful sign-in).
    // Fire-and-forget — provisioning must not wait on it.
    void refreshClientConfigFromPlatform('sign-in')
    return { ok: true, hasRelayKey: true }
  }
  // Sign-in itself succeeded (valid token) even though provisioning fell back
  // to BYOK — still a sync point for the platform client config.
  void refreshClientConfigFromPlatform('sign-in')
  return { ok: true, hasRelayKey: false }
}

/**
 * Sign in to ApexNodes with email + password and provision the managed relay
 * key for this install via the P0 contract:
 *   POST {AUTH_BASE}/api/v1/auth/login  → { access_token }
 *   POST {API_BASE}/api/v1/desktop/provision-key  (Bearer JWT, body {})
 *
 * login-or-register (mirrors web public-login-page.tsx): if /auth/login returns
 * 401 the email may simply not be registered yet, so we POST /auth/register with
 * the same credentials. A successful register yields a token we continue with; a
 * register that also fails (e.g. 202 magic-link / already-registered → wrong
 * password) surfaces as a login failure (the Chinese message is applied in the
 * renderer). Any other login error (non-401) is rethrown as-is.
 */
async function apexManagedSignIn({ email, password }) {
  const endpoints = resolveApexEndpoints(process.env)
  const cleanEmail = String(email || '').trim()
  const cleanPassword = String(password || '')

  let accessToken = ''
  // The auth-response body (login or register) is the best source of the user's
  // email/plan for the account panel; keep it to fold into the stored account.
  let authBody = null
  try {
    const loginBody = await apexAuthPostJson(endpoints.loginUrl, {
      body: { email: cleanEmail, password: cleanPassword }
    })
    authBody = loginBody
    accessToken = accessTokenFromLogin(loginBody) || ''
  } catch (error) {
    // Only a 401 means "wrong creds OR unknown email" — try registering. Any
    // other status (network, 5xx, …) is a real error: rethrow.
    if (error && error.statusCode === 401) {
      const registerBody = await apexAuthPostJson(endpoints.registerUrl, {
        body: { email: cleanEmail, password: cleanPassword, name: '', locale: 'zh' }
      }).catch(() => {
        // Register failed too (e.g. 202 magic-link for an already-registered
        // email → the 401 above was a wrong password; or any other reject).
        // Either way the user-facing outcome is "check credentials" — throw a
        // marker the renderer maps to the Chinese login-failed string.
        const wrongCreds = new Error('INVALID_CREDENTIALS')
        wrongCreds.code = 'INVALID_CREDENTIALS'
        throw wrongCreds
      })
      authBody = registerBody
      accessToken = accessTokenFromLogin(registerBody) || ''
      if (!accessToken) {
        // register returned 2xx but no token (e.g. 202 magic-link path) → treat
        // as invalid credentials, same as the web flow's 202 branch.
        const wrongCreds = new Error('INVALID_CREDENTIALS')
        wrongCreds.code = 'INVALID_CREDENTIALS'
        throw wrongCreds
      }
    } else {
      throw error
    }
  }

  // The typed email is always a valid identity fallback even if the body omits it.
  const account = { email: cleanEmail, ...(authBody && typeof authBody === 'object' ? authBody : {}) }
  return provisionManagedFromAccessToken(accessToken, account)
}

// Returns the desktop's chosen profile name, or null when unset. "default" is
// a valid stored value (pins the root HERMES_HOME explicitly); null means "no
// preference" and preserves the legacy launch (no --profile flag).
function readActiveDesktopProfile() {
  try {
    const raw = fs.readFileSync(DESKTOP_PROFILE_CONFIG_PATH, 'utf8')
    const parsed = JSON.parse(raw)
    const name = parsed && typeof parsed.profile === 'string' ? parsed.profile.trim() : ''

    if (name && (name === 'default' || PROFILE_NAME_RE.test(name))) {
      return name
    }
  } catch {
    // Missing or malformed → no preference.
  }

  return null
}

function writeActiveDesktopProfile(name) {
  const value = typeof name === 'string' ? name.trim() : ''

  if (value && value !== 'default' && !PROFILE_NAME_RE.test(value)) {
    throw new Error(`Invalid profile name: ${value}`)
  }

  fs.mkdirSync(path.dirname(DESKTOP_PROFILE_CONFIG_PATH), { recursive: true })
  writeFileAtomic(DESKTOP_PROFILE_CONFIG_PATH, JSON.stringify({ profile: value || null }, null, 2))

  return value || null
}

// Sanitize a connection config into the renderer-facing shape. With no
// `profile` this describes the global/default connection (the existing
// behavior); with a `profile` it describes that profile's per-profile remote
// override (or an empty "local/inherit" view when the profile has none).
async function sanitizeDesktopConnectionConfig(config = readDesktopConnectionConfig(), profile = null) {
  const key = connectionScopeKey(profile)
  const scoped = key ? config.profiles?.[key] || null : null
  const block = key ? scoped || {} : config.remote || {}

  const envOverride = key ? false : Boolean(process.env.HERMES_DESKTOP_REMOTE_URL)

  const remoteToken = decryptDesktopSecret(block.token)
  const authMode = normAuthMode(block.authMode)
  const remoteUrl = envOverride ? String(process.env.HERMES_DESKTOP_REMOTE_URL || '') : String(block.url || '')
  const mode = envOverride || (key ? scoped?.mode : config.mode) === 'remote' ? 'remote' : 'local'

  let remoteOauthConnected = false
  if (authMode === 'oauth' && remoteUrl) {
    try {
      // Display signal: treat a live RT cookie as "connected" even if the AT
      // cookie has lapsed — the gateway refreshes the AT on the next request,
      // so the session is still usable. The authoritative liveness check is
      // the ws-ticket mint in resolveRemoteBackend at actual connect time.
      remoteOauthConnected = await hasLiveOauthSession(remoteUrl)
    } catch {
      remoteOauthConnected = false
    }
  }

  return {
    mode,
    // Echo the scope back so the UI knows which profile (if any) this reflects.
    profile: key,
    remoteAuthMode: authMode,
    remoteOauthConnected,
    remoteUrl,
    remoteTokenPreview: tokenPreview(remoteToken),
    remoteTokenSet: Boolean(remoteToken),
    // The env override only forces the global/primary connection; a per-profile
    // scope is never overridden by HERMES_DESKTOP_REMOTE_URL.
    envOverride
  }
}

// Build + validate a `{ url, authMode, token }` remote block. OAuth gateways
// authenticate via the login-window session cookie (verified at connect time in
// resolveRemoteBackend), so only token-auth remotes require a saved token.
function buildRemoteBlock(remoteUrl, authMode, token) {
  if (authMode !== 'oauth' && !decryptDesktopSecret(token)) {
    throw new Error('Remote gateway session token is required.')
  }
  return { url: normalizeRemoteBaseUrl(remoteUrl), authMode, token }
}

function coerceDesktopConnectionConfig(input = {}, existing = readDesktopConnectionConfig(), options = {}) {
  const persistToken = options.persistToken !== false
  const key = connectionScopeKey(input.profile)
  const mode = input.mode === 'remote' ? 'remote' : 'local'

  // The block being edited: a per-profile entry or the global remote block.
  const existingBlock = key ? existing.profiles?.[key] || {} : existing.remote || {}
  const remoteUrl = String(input.remoteUrl ?? existingBlock.url ?? '').trim()
  // authMode: explicit input wins; otherwise inherit the saved value, default 'token'.
  const authMode = resolveAuthMode(input.remoteAuthMode, existingBlock.authMode)
  const incomingToken = typeof input.remoteToken === 'string' ? input.remoteToken.trim() : ''
  const nextToken = incomingToken
    ? persistToken
      ? encryptDesktopSecret(incomingToken)
      : { encoding: 'plain', value: incomingToken }
    : existingBlock.token

  if (key) {
    // Per-profile scope: a remote entry pins this profile to its own backend; a
    // local entry clears the override so the profile inherits the default.
    const profiles = { ...(existing.profiles || {}) }
    if (mode === 'remote') {
      profiles[key] = { mode: 'remote', ...buildRemoteBlock(remoteUrl, authMode, nextToken) }
    } else {
      delete profiles[key]
    }
    return { mode: existing.mode === 'remote' ? 'remote' : 'local', remote: existing.remote || {}, profiles }
  }

  const nextRemote =
    mode === 'remote'
      ? buildRemoteBlock(remoteUrl, authMode, nextToken)
      : { url: remoteUrl ? normalizeRemoteBaseUrl(remoteUrl) : remoteUrl, authMode, token: nextToken }

  // Preserve per-profile overrides when saving the global connection.
  return { mode, remote: nextRemote, profiles: existing.profiles || {} }
}

// Build a remote backend connection descriptor from an already-resolved remote
// config. Handles both auth models (OAuth ws-ticket vs static session token)
// and is shared by the per-profile, env, and global resolution paths. `token`
// is the DECRYPTED static token (or null in OAuth mode). `source` is a label
// for diagnostics ('profile' | 'env' | 'settings').
async function buildRemoteConnection(rawUrl, authMode, token, source) {
  const baseUrl = normalizeRemoteBaseUrl(rawUrl)

  if (authMode === 'oauth') {
    // OAuth gateway: auth comes from the session cookies in the OAuth
    // partition. Liveness is NOT "is the access-token cookie present?" —
    // Portal issues a 24h rotating refresh token (hermes #37247), and the
    // gateway middleware transparently rotates a fresh ~15-min access token
    // from it on the next authenticated request. So a session with an expired
    // AT cookie but a live RT cookie is still perfectly connectable. We
    // early-out only when neither cookie is present, then mint a ws-ticket as
    // the authoritative liveness check.
    if (!(await hasLiveOauthSession(baseUrl))) {
      const err = new Error(
        'Remote Hermes gateway uses OAuth, but you are not signed in. ' +
          'Open Settings → Gateway and click "Sign in", or switch back to Local.'
      )
      err.needsOauthLogin = true
      throw err
    }

    let ticket
    try {
      ticket = await mintGatewayWsTicket(baseUrl)
    } catch (error) {
      const err = new Error(
        'Your remote gateway session has expired. ' + 'Open Settings → Gateway and click "Sign in" again.'
      )
      err.needsOauthLogin = true
      err.cause = error
      throw err
    }

    return {
      baseUrl,
      mode: 'remote',
      source,
      authMode: 'oauth',
      // No static token in OAuth mode; REST is cookie-authed via the partition.
      token: null,
      wsUrl: buildGatewayWsUrlWithTicket(baseUrl, ticket)
    }
  }

  if (!token) {
    throw new Error(
      'Remote Hermes gateway is selected, but no session token is saved. ' +
        'Open Settings → Gateway and save a token, or switch back to Local.'
    )
  }

  return {
    baseUrl,
    mode: 'remote',
    source,
    authMode: 'token',
    token,
    wsUrl: buildGatewayWsUrl(baseUrl, token)
  }
}

// Resolve the remote backend for a given profile, or null when that profile
// should run a LOCAL backend. Precedence:
//   1. explicit per-profile remote override (connection.json `profiles[name]`)
//   2. env override (HERMES_DESKTOP_REMOTE_URL/_TOKEN) — applies app-wide
//   3. global remote (connection.json `mode: 'remote'`)
// A null/empty profile resolves the env/global remote, so legacy callers and
// the connection test (which pass no profile) are unchanged.
async function resolveRemoteBackend(profile) {
  const config = readDesktopConnectionConfig()

  // 1. Per-profile override — "a profile with its own remote host". Wins even
  //    over the env override so an explicitly-configured profile always
  //    reaches its intended backend.
  const override = profileRemoteOverride(config, profile)
  if (override) {
    const token = override.authMode === 'oauth' ? null : decryptDesktopSecret(override.token)
    return buildRemoteConnection(override.url, override.authMode, token, 'profile')
  }

  // 2. Env override (global, token-auth only).
  const rawEnvUrl = process.env.HERMES_DESKTOP_REMOTE_URL
  const rawEnvToken = process.env.HERMES_DESKTOP_REMOTE_TOKEN
  if (rawEnvUrl) {
    if (!rawEnvToken) {
      throw new Error(
        'HERMES_DESKTOP_REMOTE_URL is set but HERMES_DESKTOP_REMOTE_TOKEN is not. ' +
          'Both must be provided to connect to a remote Hermes backend.'
      )
    }
    return buildRemoteConnection(rawEnvUrl, 'token', rawEnvToken, 'env')
  }

  // 3. Global remote.
  if (config.mode !== 'remote') {
    return null
  }
  const authMode = normAuthMode(config.remote?.authMode)
  const token = authMode === 'oauth' ? null : decryptDesktopSecret(config.remote?.token)
  return buildRemoteConnection(config.remote?.url, authMode, token, 'settings')
}

// A remote profile's sessions live on its remote host's state.db, not on a local
// file the primary can open — so reads for it must route to the remote backend,
// not the local-disk fast path. These three helpers drive that (see
// interceptSessionReadForRemote).
function profileHasRemoteOverride(profile) {
  return Boolean(profileRemoteOverride(readDesktopConnectionConfig(), profile))
}

function configuredRemoteProfileNames() {
  const config = readDesktopConnectionConfig()
  return Object.keys(config.profiles || {}).filter(name => profileRemoteOverride(config, name))
}

// True when the app is in app-global remote mode (Settings → "All profiles" →
// Remote, or the env override): a SINGLE remote backend serves every profile via
// ?profile=. Distinct from per-profile overrides — here there's one host for all.
function globalRemoteActive() {
  if (process.env.HERMES_DESKTOP_REMOTE_URL) {
    return true
  }
  return readDesktopConnectionConfig().mode === 'remote'
}

// GET a profile's resolved backend (remote pool or local primary), parsed JSON.
async function fetchJsonForProfile(profile, path) {
  return requestJsonForProfile(profile, path, 'GET')
}

// Issue an arbitrary method against a profile's resolved backend, parsed JSON.
async function requestJsonForProfile(profile, path, method, body) {
  const conn = await ensureBackend(profile)
  const url = `${conn.baseUrl}${path}`
  const opts = { method, body, timeoutMs: DEFAULT_FETCH_TIMEOUT_MS }
  return conn.authMode === 'oauth' ? fetchJsonViaOauthSession(url, opts) : fetchJson(url, conn.token, opts)
}

async function probeRemoteAuthMode(rawUrl) {
  // Determine how a remote gateway expects callers to authenticate, WITHOUT
  // sending any credentials. ``/api/status`` is public on every Hermes
  // gateway (it backs the portal liveness probe) and reports:
  //   auth_required: true  → OAuth gate is engaged (cookie + ws-ticket auth)
  //   auth_required: false → loopback/--insecure: legacy session-token auth
  // ``/api/auth/providers`` (also public, only meaningful when gated) gives
  // the human-facing provider name(s) for the login button label.
  //
  // The settings UI calls this as the user types a URL so it can render an
  // OAuth login button vs a session-token entry box. Network/parse failures
  // surface as ``reachable: false`` rather than throwing, so a half-typed or
  // unreachable URL degrades to "can't tell yet" instead of a hard error.
  const baseUrl = normalizeRemoteBaseUrl(rawUrl)

  let status
  try {
    status = await fetchPublicJson(`${baseUrl}/api/status`, { timeoutMs: 8_000 })
  } catch (error) {
    return {
      baseUrl,
      reachable: false,
      authMode: 'unknown',
      providers: [],
      version: null,
      error: error instanceof Error ? error.message : String(error)
    }
  }

  const authRequired = authModeFromStatus(status) === 'oauth'
  let providers = []

  if (authRequired) {
    // Best-effort: a gated gateway exposes the registered providers so the
    // button can read "Sign in with Nous Research" instead of a generic
    // label, and so a username/password provider can be distinguished from
    // an OAuth-redirect one (``supports_password``). A failure here doesn't
    // change the auth mode, so swallow it.
    try {
      const body = await fetchPublicJson(`${baseUrl}/api/auth/providers`, { timeoutMs: 8_000 })
      if (Array.isArray(body?.providers)) {
        providers = body.providers
          .filter(p => p && typeof p === 'object')
          .map(p => ({
            name: String(p.name || ''),
            displayName: String(p.display_name || p.name || ''),
            supportsPassword: Boolean(p.supports_password)
          }))
          .filter(p => p.name)
      }
    } catch {
      // Provider listing is optional metadata; the auth mode is already known.
    }
  }

  return {
    baseUrl,
    reachable: true,
    authMode: authRequired ? 'oauth' : 'token',
    providers,
    version: status?.version || null,
    error: null
  }
}

async function testDesktopConnectionConfig(input = {}) {
  const config = coerceDesktopConnectionConfig(input, readDesktopConnectionConfig(), { persistToken: false })
  const key = connectionScopeKey(input.profile)
  // The block under test: a per-profile entry or the global remote. Coerce has
  // already normalized the URL and resolved token inheritance for the scope.
  const block = key ? config.profiles?.[key] || null : config.remote
  const wantRemote =
    block?.mode === 'remote' || (!key && config.mode === 'remote') || (input.mode === 'remote' && block)
  // ``/api/status`` is public on every gateway (no creds needed), so a
  // reachability test works for local, token, and oauth modes alike — we only
  // need a base URL. For a remote config we normalize the URL from the input;
  // for local we fall back to the resolved/started backend.
  let baseUrl
  let token = null
  let authMode = 'token'
  if (wantRemote && block?.url) {
    baseUrl = normalizeRemoteBaseUrl(block.url)
    authMode = normAuthMode(block.authMode)
    if (authMode !== 'oauth') {
      token = decryptDesktopSecret(block.token)
    }
  } else {
    const remote = (await resolveRemoteBackend(key)) || (await startHermes())
    baseUrl = remote.baseUrl
    token = remote.token
    authMode = normAuthMode(remote.authMode)
  }
  const status = await fetchJson(`${baseUrl}/api/status`, token, { timeoutMs: 8_000 })

  // The HTTP status check above proves the backend is reachable, but the chat
  // surface only works once the renderer's live WebSocket to ``/api/ws``
  // connects — a separate transport with separate server-side guards (Host/
  // Origin, ws-ticket/token auth). Validating only the HTTP side produced a
  // false-positive "reachable" while the real boot still failed with "Could not
  // connect to Hermes gateway". Mirror the renderer's connect here so the test
  // reflects the full path the app actually uses.
  const wsUrl = await resolveTestWsUrl(baseUrl, authMode, token, { mintTicket: mintGatewayWsTicket })
  // Skip the WS leg only when the runtime genuinely lacks a WebSocket (so an
  // older Electron/Node never fails the test spuriously); Electron's main
  // process ships a global WebSocket on every supported version.
  if (wsUrl && typeof globalThis.WebSocket === 'function') {
    const probe = await probeGatewayWebSocket(wsUrl, { WebSocketImpl: globalThis.WebSocket })
    if (!probe.ok) {
      throw new Error(
        `Reached the gateway over HTTP, but the live WebSocket (/api/ws) connection failed: ${probe.reason} ` +
          'The HTTP check can pass while the WebSocket is blocked by a proxy, firewall, or gateway auth/origin guard.'
      )
    }
  }

  return {
    ok: true,
    baseUrl,
    version: status?.version || null
  }
}

function resetBootProgressForReconnect() {
  updateBootProgress(
    {
      error: null,
      message: 'Restarting desktop connection',
      phase: 'backend.resolve',
      progress: 4,
      running: true
    },
    { allowDecrease: true }
  )
}

function resetHermesConnection() {
  connectionPromise = null

  if (hermesProcess && !hermesProcess.killed) {
    hermesProcess.kill('SIGTERM')
  }

  hermesProcess = null
  resetBootProgressForReconnect()
}

// Re-home the primary backend: reset connection state, then wait for the live
// dashboard process to actually exit (SIGKILL after 5s) so the next
// startHermes() spawns fresh instead of racing the dying one. Shared by the
// connection-config and profile switch flows.
async function teardownPrimaryBackendAndWait() {
  // Capture the reference before resetHermesConnection() nulls hermesProcess.
  const dying = hermesProcess && !hermesProcess.killed ? hermesProcess : null
  resetHermesConnection()

  await waitForBackendExit(dying)
}

async function waitForBackendExit(child, timeoutMs = 5000) {
  if (!child) {
    return
  }
  if (child.exitCode !== null || child.signalCode !== null) {
    return
  }

  await new Promise(resolve => {
    const timer = setTimeout(() => {
      try {
        if (IS_WINDOWS && Number.isInteger(child.pid)) {
          forceKillProcessTree(child.pid)
        } else {
          child.kill('SIGKILL')
        }
      } catch {
        // Already gone.
      }
      resolve()
    }, timeoutMs)
    child.once('exit', () => {
      clearTimeout(timer)
      resolve()
    })
  })
}

// The profile the primary (window) backend runs as. readActiveDesktopProfile()
// returns the desktop's stored preference, or null when unset (legacy launch
// that defers to active_profile / default).
function primaryProfileKey() {
  return readActiveDesktopProfile() || 'default'
}

// Resolve a backend connection for the given profile. Routes the primary
// profile to startHermes() (the window backend: boot UI, bootstrap, remote
// mode), and any OTHER profile to a lazily-spawned pool backend. An empty /
// unknown profile resolves to the primary, so all legacy callers are unchanged.
async function ensureBackend(profile) {
  const key = profile && String(profile).trim() ? String(profile).trim() : primaryProfileKey()

  if (key === primaryProfileKey()) {
    return startHermes()
  }

  const existing = backendPool.get(key)
  if (existing) {
    existing.lastActiveAt = Date.now()
    return existing.connectionPromise
  }

  evictLruPoolBackends(POOL_MAX_BACKENDS - 1)

  const entry = { process: null, port: null, token: null, connectionPromise: null, lastActiveAt: Date.now() }
  entry.connectionPromise = spawnPoolBackend(key, entry).catch(error => {
    backendPool.delete(key)
    throw error
  })
  backendPool.set(key, entry)
  startPoolIdleReaper()
  return entry.connectionPromise
}

// Mark a pool profile as recently used so the idle reaper spares it. The
// renderer calls this when it opens a profile's chat WS and periodically while
// streaming, since the main process can't see the direct renderer↔backend WS.
function touchPoolBackend(profile) {
  const key = profile && String(profile).trim() ? String(profile).trim() : null
  if (!key) return
  const entry = backendPool.get(key)
  if (entry) entry.lastActiveAt = Date.now()
}

// Evict least-recently-used pool backends until at most `keep` remain — but only
// ever evict backends without a live renderer socket (stale beyond the keepalive
// window). When every backend is actively kept alive we let the pool exceed the
// soft cap rather than kill a running session.
function evictLruPoolBackends(keep) {
  if (backendPool.size <= keep) return
  const now = Date.now()
  const evictable = [...backendPool.entries()]
    .filter(([, entry]) => now - (entry.lastActiveAt || 0) > POOL_KEEPALIVE_FRESH_MS)
    .sort((a, b) => (a[1].lastActiveAt || 0) - (b[1].lastActiveAt || 0))
  let removable = backendPool.size - Math.max(0, keep)
  for (const [profile] of evictable) {
    if (removable <= 0) break
    rememberLog(`Evicting idle profile backend "${profile}" (LRU cap ${POOL_MAX_BACKENDS})`)
    stopPoolBackend(profile)
    removable -= 1
  }
}

function startPoolIdleReaper() {
  if (poolIdleReaper) return
  poolIdleReaper = setInterval(() => {
    const now = Date.now()
    for (const [profile, entry] of [...backendPool.entries()]) {
      if (now - (entry.lastActiveAt || 0) > POOL_IDLE_MS) {
        rememberLog(`Reaping idle profile backend "${profile}" (idle > ${Math.round(POOL_IDLE_MS / 1000)}s)`)
        stopPoolBackend(profile)
      }
    }
    if (backendPool.size === 0 && poolIdleReaper) {
      clearInterval(poolIdleReaper)
      poolIdleReaper = null
    }
  }, 60_000)
  if (typeof poolIdleReaper.unref === 'function') poolIdleReaper.unref()
}

// Spawn an additional dashboard backend pinned to a named profile. Mirrors the
// local-spawn portion of startHermes() but without the boot-progress UI,
// bootstrap, or remote handling (those belong to the primary backend only).
async function spawnPoolBackend(profile, entry) {
  // A profile may point at its OWN remote backend (connection.json
  // `profiles[name]`), or inherit the app-wide remote (env / global settings).
  // In either case there is no local child to spawn — we just verify the
  // remote is reachable and hand back its connection descriptor. The pool
  // entry keeps `entry.process === null`, which stopPoolBackend/evict already
  // tolerate.
  const remote = await resolveRemoteBackend(profile)
  if (remote) {
    await waitForHermes(remote.baseUrl, remote.token)
    return {
      ...remote,
      profile,
      logs: hermesLog.slice(-80),
      ...getWindowState()
    }
  }

  const token = crypto.randomBytes(32).toString('base64url')
  // --profile wins over the inherited HERMES_HOME env (see _apply_profile_override
  // step 3 in hermes_cli/main.py), so the child re-homes to this profile.
  // --port 0: the OS assigns an ephemeral port; the child announces it on stdout.
  const dashboardArgs = ['--profile', profile, 'dashboard', '--no-open', '--host', '127.0.0.1', '--port', '0']
  const backend = await ensureRuntime(resolveHermesBackend(dashboardArgs))
  const hermesCwd = resolveHermesCwd()
  const webDist = resolveWebDist()

  rememberLog(`Starting Hermes backend for profile "${profile}" via ${backend.label}`)

  const child = spawn(
    backend.command,
    backend.args,
    hiddenWindowsChildOptions({
      cwd: hermesCwd,
      env: {
        ...process.env,
        HERMES_HOME,
        ...backend.env,
        // hc-444: inject the signed-in user's mirrored Feishu credential
        // (FEISHU_APP_ID / FEISHU_APP_SECRET / FEISHU_DOMAIN, decrypted just in
        // time) so the runtime's Feishu adapter + lark doc/drive tools light up.
        // {} (no keys) when not connected; add-only vs an explicit parent-env
        // credential.
        ...desktopFeishuSpawnEnv(),
        // Pin the gateway's tool/terminal cwd to the same directory we chose for
        // the child process. Inherited TERMINAL_CWD (or a stale config bridge)
        // can still point at the install dir even when spawn cwd is home.
        TERMINAL_CWD: hermesCwd,
        HERMES_DASHBOARD_SESSION_TOKEN: token,
        // Marks this dashboard backend as desktop-spawned so it runs the cron
        // scheduler tick loop (the gateway isn't running under the app).
        HERMES_DESKTOP: '1',
        HERMES_WEB_DIST: webDist
      },
      shell: backend.shell,
      stdio: ['ignore', 'pipe', 'pipe']
    })
  )
  entry.process = child
  entry.token = token

  child.stdout.on('data', rememberLog)
  child.stderr.on('data', rememberLog)

  let ready = false
  let rejectStart = null
  const startFailed = new Promise((_resolve, reject) => {
    rejectStart = reject
  })
  child.once('error', error => {
    rememberLog(`Hermes backend for profile "${profile}" failed to start: ${error.message}`)
    backendPool.delete(profile)
    rejectStart?.(error)
  })
  child.once('exit', (code, signal) => {
    rememberLog(`Hermes backend for profile "${profile}" exited (${signal || code})`)
    backendPool.delete(profile)
    if (!ready) {
      rejectStart?.(
        new Error(`Hermes backend for profile "${profile}" exited before it became ready (${signal || code}).`)
      )
    }
  })

  // Discover the ephemeral port the child bound to
  const port = await Promise.race([waitForDashboardPort(child), startFailed])
  entry.port = port

  const baseUrl = `http://127.0.0.1:${port}`
  await Promise.race([waitForHermes(baseUrl, token), startFailed])
  ready = true
  const authToken = await adoptServedDashboardToken(baseUrl, token, {
    childAlive: () => child.exitCode === null && !child.killed,
    label: `Hermes backend for profile "${profile}"`,
    rememberLog
  })
  entry.token = authToken

  return {
    baseUrl,
    mode: 'local',
    source: 'local',
    authMode: 'token',
    token: authToken,
    profile,
    wsUrl: `ws://127.0.0.1:${port}/api/ws?token=${encodeURIComponent(authToken)}`,
    logs: hermesLog.slice(-80),
    ...getWindowState()
  }
}

function stopPoolBackend(profile) {
  const entry = backendPool.get(profile)
  if (!entry) return
  backendPool.delete(profile)
  if (entry.process && !entry.process.killed) {
    try {
      entry.process.kill('SIGTERM')
    } catch {
      // Already gone.
    }
  }
}

async function teardownPoolBackendAndWait(profile) {
  const entry = backendPool.get(profile)
  if (!entry) return
  backendPool.delete(profile)

  if (entry.process && !entry.process.killed) {
    try {
      entry.process.kill('SIGTERM')
    } catch {
      // Already gone.
    }
  }

  await waitForBackendExit(entry.process)
}

function stopAllPoolBackends() {
  for (const profile of [...backendPool.keys()]) {
    stopPoolBackend(profile)
  }
}

function profileNameFromDeleteRequest(request) {
  if (!request || String(request.method || 'GET').toUpperCase() !== 'DELETE') {
    return null
  }

  const match = String(request.path || '').match(/^\/api\/profiles\/([^/?#]+)(?:[?#].*)?$/)
  if (!match) {
    return null
  }

  let raw = ''
  try {
    raw = decodeURIComponent(match[1])
  } catch {
    return null
  }

  const name = raw.trim()
  if (!name) {
    return null
  }
  if (name.toLowerCase() === 'default') {
    return 'default'
  }
  return name.toLowerCase()
}

async function prepareProfileDeleteRequest(request) {
  const profile = profileNameFromDeleteRequest(request)
  if (!profile || profile === 'default' || !PROFILE_NAME_RE.test(profile)) {
    return
  }

  if (profile === primaryProfileKey()) {
    writeActiveDesktopProfile('default')
    await teardownPrimaryBackendAndWait()
    return
  }

  await teardownPoolBackendAndWait(profile)
}

async function startHermes() {
  // Latched-failure short-circuit: once bootstrap has failed in this
  // process, every subsequent startHermes() call re-throws the same error
  // without re-running install.ps1. This prevents the renderer's
  // ensureGatewayOpen retries (and any other getConnection callers) from
  // restarting a 5-10 minute install loop while the user is still reading
  // the failure overlay.
  if (bootstrapFailure) {
    throw bootstrapFailure
  }
  if (connectionPromise) return connectionPromise

  connectionPromise = (async () => {
    await advanceBootProgress('backend.resolve', 'Resolving Hermes backend', 8)
    // Resolve for the desktop's primary profile so a per-profile remote
    // override on the active profile is honored (falls back to env / global).
    const remote = await resolveRemoteBackend(primaryProfileKey())
    if (remote) {
      await advanceBootProgress('backend.remote', `Connecting to remote Hermes backend at ${remote.baseUrl}`, 24)
      await waitForHermes(remote.baseUrl, remote.token)
      updateBootProgress({
        phase: 'backend.ready',
        message: 'Remote Hermes backend is ready',
        progress: 94,
        running: true,
        error: null
      })
      return {
        baseUrl: remote.baseUrl,
        mode: 'remote',
        source: remote.source,
        authMode: remote.authMode || 'token',
        token: remote.token,
        wsUrl: remote.wsUrl,
        logs: hermesLog.slice(-80),
        ...getWindowState()
      }
    }

    const token = crypto.randomBytes(32).toString('base64url')
    // --port 0: the OS assigns an ephemeral port; the child announces it on stdout.
    const dashboardArgs = ['dashboard', '--no-open', '--host', '127.0.0.1', '--port', '0']
    // Pin the desktop's chosen profile via the global --profile flag. This is
    // deterministic (it wins over the sticky ~/.hermes/active_profile file) and
    // resolves HERMES_HOME the same way `hermes -p <name>` does on the CLI. An
    // unset preference keeps the legacy launch so existing installs are
    // unaffected.
    const activeProfile = readActiveDesktopProfile()
    if (activeProfile) {
      dashboardArgs.unshift('--profile', activeProfile)
    }
    await advanceBootProgress('backend.runtime', 'Resolving Hermes runtime', 28)
    const backend = await ensureRuntime(resolveHermesBackend(dashboardArgs))
    const hermesCwd = resolveHermesCwd()
    const webDist = resolveWebDist()

    await advanceBootProgress('backend.spawn', `Starting Hermes backend via ${backend.label}`, 84)
    rememberLog(`Starting Hermes backend via ${backend.label}`)

    hermesProcess = spawn(
      backend.command,
      backend.args,
      hiddenWindowsChildOptions({
        cwd: hermesCwd,
        env: {
          ...process.env,
          // Explicitly pin HERMES_HOME for the child so Python's get_hermes_home()
          // resolves to the SAME location our resolveHermesHome() picked. Without
          // this pin, Python falls back to ~/.hermes on every platform — fine on
          // mac/linux (where our default matches), but on Windows our default is
          // %LOCALAPPDATA%\hermes, which differs from C:\Users\<u>\.hermes.
          // Mismatch would split config / sessions / .env / logs across two
          // directories. install.ps1 sets HERMES_HOME via setx; the desktop
          // can't reliably do that, so we set it inline for every spawn.
          HERMES_HOME,
          ...backend.env,
          // hc-444: inject the signed-in user's mirrored Feishu credential
          // (FEISHU_APP_ID / FEISHU_APP_SECRET / FEISHU_DOMAIN, decrypted just in
          // time) so the runtime's Feishu adapter + lark doc/drive tools light up.
          // {} (no keys) when not connected; add-only vs an explicit parent-env
          // credential.
          ...desktopFeishuSpawnEnv(),
          TERMINAL_CWD: hermesCwd,
          HERMES_DASHBOARD_SESSION_TOKEN: token,
          // Marks this dashboard backend as desktop-spawned so it runs the cron
          // scheduler tick loop (the gateway isn't running under the app).
          HERMES_DESKTOP: '1',
          HERMES_WEB_DIST: webDist
        },
        shell: backend.shell,
        stdio: ['ignore', 'pipe', 'pipe']
      })
    )

    hermesProcess.stdout.on('data', rememberLog)
    hermesProcess.stderr.on('data', rememberLog)
    let backendReady = false
    let rejectBackendStart = null
    const backendStartFailed = new Promise((_resolve, reject) => {
      rejectBackendStart = reject
    })
    hermesProcess.once('error', error => {
      rememberLog(`Hermes backend failed to start: ${error.message}`)
      updateBootProgress(
        {
          error: error.message,
          message: `Hermes backend failed to start: ${error.message}`,
          phase: 'backend.error',
          running: false
        },
        { allowDecrease: true }
      )
      hermesProcess = null
      connectionPromise = null
      sendBackendExit({ code: null, signal: null, error: error.message })
      rejectBackendStart?.(error)
    })
    hermesProcess.once('exit', (code, signal) => {
      rememberLog(`Hermes backend exited (${signal || code})`)
      hermesProcess = null
      connectionPromise = null
      sendBackendExit({ code, signal })
      if (!backendReady) {
        const message = `Hermes backend exited before it became ready (${signal || code}).`
        updateBootProgress(
          {
            error: message,
            message,
            phase: 'backend.error',
            running: false
          },
          { allowDecrease: true }
        )
        rejectBackendStart?.(
          new Error(
            `Hermes backend exited before it became ready (${signal || code}). Log: ${DESKTOP_LOG_PATH}\n${recentHermesLog()}`
          )
        )
      }
    })

    await advanceBootProgress('backend.port', 'Waiting for Hermes backend to launch', 86)
    // Discover the ephemeral port the child bound to
    const port = await Promise.race([waitForDashboardPort(hermesProcess), backendStartFailed])

    const baseUrl = `http://127.0.0.1:${port}`
    await advanceBootProgress('backend.wait', 'Waiting for Hermes backend to become ready', 90)
    await Promise.race([waitForHermes(baseUrl, token), backendStartFailed])
    backendReady = true
    const authToken = await adoptServedDashboardToken(baseUrl, token, {
      // The exit/error handlers null hermesProcess when the child dies.
      childAlive: () => hermesProcess !== null && hermesProcess.exitCode === null && !hermesProcess.killed,
      rememberLog
    })
    updateBootProgress({
      phase: 'backend.ready',
      message: 'Hermes backend is ready. Finalizing desktop startup',
      progress: 94,
      running: true,
      error: null
    })

    return {
      baseUrl,
      mode: 'local',
      source: 'local',
      authMode: 'token',
      token: authToken,
      wsUrl: `ws://127.0.0.1:${port}/api/ws?token=${encodeURIComponent(authToken)}`,
      logs: hermesLog.slice(-80),
      ...getWindowState()
    }
  })().catch(error => {
    const message = error instanceof Error ? error.message : String(error)
    updateBootProgress(
      {
        error: message,
        message: `Desktop boot failed: ${message}`,
        phase: 'backend.error',
        running: false
      },
      { allowDecrease: true }
    )
    connectionPromise = null
    throw error
  })

  return connectionPromise
}

// Shared navigation guards + window chrome wiring applied to every window
// (the primary plus any secondary session windows). Factored out of
// createWindow() so secondary windows can't drift from the main window's
// security posture: external links open in the OS browser, in-app navigation
// stays confined to the dev server / packaged file URL, and the preview /
// devtools / zoom / context-menu affordances behave identically everywhere.
function wireCommonWindowHandlers(win) {
  installPreviewShortcut(win)
  installDevToolsShortcut(win)
  installZoomShortcuts(win)
  installContextMenu(win)
  win.webContents.setWindowOpenHandler(details => {
    openExternalUrl(details.url)

    return { action: 'deny' }
  })
  win.webContents.on('will-navigate', (event, url) => {
    if ((DEV_SERVER && url.startsWith(DEV_SERVER)) || (!DEV_SERVER && url.startsWith('file:'))) {
      return
    }

    event.preventDefault()
    openExternalUrl(url)
  })
}

// Secondary "session windows" — one extra OS window per chat so a user can
// work with multiple chats side by side. The registry guarantees one window
// per sessionId (re-opening focuses the existing window) and self-cleans on
// close. The primary mainWindow is never tracked here. Pure logic + the URL
// builder live in session-windows.cjs so they stay unit-testable.
const sessionWindows = createSessionWindowRegistry()

function focusWindow(win) {
  if (!win || win.isDestroyed()) return
  if (win.isMinimized()) win.restore()
  if (!win.isVisible()) win.show()
  win.focus()
}

function spawnSecondaryWindow({ sessionId, watch, newSession } = {}) {
  const icon = getAppIconPath()
  const win = new BrowserWindow({
    width: SESSION_WINDOW_MIN_WIDTH,
    height: SESSION_WINDOW_MIN_HEIGHT,
    minWidth: SESSION_WINDOW_MIN_WIDTH,
    minHeight: SESSION_WINDOW_MIN_HEIGHT,
    title: 'APEX',
    titleBarStyle: 'hidden',
    titleBarOverlay: getTitleBarOverlayOptions(),
    trafficLightPosition: IS_MAC ? WINDOW_BUTTON_POSITION : undefined,
    vibrancy: IS_MAC ? 'sidebar' : undefined,
    opacity: windowOpacity(),
    icon,
    // Don't show until the renderer's first themed paint is ready. macOS
    // `vibrancy` ignores `backgroundColor` and paints a translucent OS
    // material (which follows the OS appearance, not the app theme), so a
    // dark-themed app on a light-mode Mac flashes white until the renderer
    // covers it. ready-to-show fires after the boot-time paint in
    // themes/context.tsx, so the window appears already themed.
    show: false,
    backgroundColor: getWindowBackgroundColor(),
    webPreferences: chatWindowWebPreferences(path.join(__dirname, 'preload.cjs'))
  })

  if (IS_MAC) {
    win.setWindowButtonPosition?.(WINDOW_BUTTON_POSITION)
  }

  win.once('ready-to-show', () => {
    if (!win.isDestroyed()) win.show()
  })

  win.on('will-enter-full-screen', () => sendWindowStateChanged(true))
  win.on('enter-full-screen', () => sendWindowStateChanged(true))
  win.on('will-leave-full-screen', () => sendWindowStateChanged(false))
  win.on('leave-full-screen', () => sendWindowStateChanged(false))

  wireCommonWindowHandlers(win)

  win.loadURL(
    buildSessionWindowUrl(sessionId, {
      devServer: DEV_SERVER,
      rendererIndexPath: DEV_SERVER ? undefined : resolveRendererIndex(),
      watch,
      newSession
    })
  )

  return win
}

// Open (or focus) a standalone window for a single chat session.
function createSessionWindow(sessionId, { watch = false } = {}) {
  return sessionWindows.openOrFocus(sessionId, () => spawnSecondaryWindow({ sessionId, watch }))
}

// Open a fresh compact window on the new-session draft (#/). Not registry-keyed:
// like ⌘N in a browser, every press opens a new window — and a draft window that
// later converts to a real session must not get refocused as if it were blank.
function createNewSessionWindow() {
  return spawnSecondaryWindow({ newSession: true })
}

function createWindow() {
  const icon = getAppIconPath()
  mainWindow = new BrowserWindow({
    width: 1220,
    height: 800,
    minWidth: 400,
    minHeight: 620,
    title: 'APEX',
    // Frameless title bar on every platform so the renderer can paint the
    // "hide sidebar" button (and other left-side titlebar tools) flush with
    // the top edge — matching the macOS layout where the traffic lights sit
    // inside the same band. On Windows/Linux, titleBarOverlay tells Electron
    // to paint native min/max/close in the top-right of the renderer; on
    // macOS it just reserves a content inset alongside the traffic lights.
    titleBarStyle: 'hidden',
    titleBarOverlay: getTitleBarOverlayOptions(),
    trafficLightPosition: IS_MAC ? WINDOW_BUTTON_POSITION : undefined,
    vibrancy: IS_MAC ? 'sidebar' : undefined,
    opacity: windowOpacity(),
    icon,
    // Hidden until the first themed paint so macOS `vibrancy` (which ignores
    // `backgroundColor` and follows the OS appearance) can't flash a light
    // material before the renderer paints the app theme. See createSessionWindow.
    show: false,
    backgroundColor: getWindowBackgroundColor(),
    // Shared with the secondary session windows (chatWindowWebPreferences) so
    // both keep `backgroundThrottling: false` — the chat transcript streams via
    // a requestAnimationFrame-gated flush that Chromium pauses for blurred
    // windows, stalling the live answer until refocus. See session-windows.cjs.
    webPreferences: chatWindowWebPreferences(path.join(__dirname, 'preload.cjs'))
  })

  if (IS_MAC) {
    mainWindow.setWindowButtonPosition?.(WINDOW_BUTTON_POSITION)
    if (icon) {
      app.dock?.setIcon(icon)
    }
  }

  if (!IS_MAC) {
    if (!nativeThemeListenerInstalled) {
      nativeThemeListenerInstalled = true
      nativeTheme.on('updated', () => {
        mainWindow?.setTitleBarOverlay?.(getTitleBarOverlayOptions())
      })
    }
  }

  mainWindow.once('ready-to-show', () => {
    if (mainWindow && !mainWindow.isDestroyed()) mainWindow.show()
  })

  mainWindow.on('will-enter-full-screen', () => sendWindowStateChanged(true))
  mainWindow.on('enter-full-screen', () => sendWindowStateChanged(true))
  mainWindow.on('will-leave-full-screen', () => sendWindowStateChanged(false))
  mainWindow.on('leave-full-screen', () => sendWindowStateChanged(false))

  wireCommonWindowHandlers(mainWindow)

  mainWindow.webContents.on('render-process-gone', (_event, details) => {
    rememberLog(`[renderer] render-process-gone reason=${details?.reason} exitCode=${details?.exitCode}`)

    if (details?.reason === 'crashed' || details?.reason === 'oom') {
      const now = Date.now()
      rendererReloadTimes = rendererReloadTimes.filter(t => now - t < RENDERER_RELOAD_WINDOW_MS)

      if (rendererReloadTimes.length >= RENDERER_RELOAD_MAX) {
        rememberLog(
          `[renderer] suppressing reload: ${rendererReloadTimes.length} crashes within ${RENDERER_RELOAD_WINDOW_MS}ms (likely a crash loop)`
        )

        return
      }

      rendererReloadTimes.push(now)
      setImmediate(() => {
        if (!mainWindow || mainWindow.isDestroyed()) return
        try {
          mainWindow.webContents.reload()
        } catch (err) {
          rememberLog(`[renderer] reload after crash failed: ${err?.message || err}`)
        }
      })
    }
  })

  mainWindow.webContents.on('unresponsive', () => rememberLog('[renderer] webContents became unresponsive'))

  // Electron always passes the event first. The canonical (Electron 36+) shape
  // is (event, messageDetails); the deprecated positional shape is
  // (event, level, message, line, sourceId). Handle both. `level` is numeric
  // (0..3), where 3 === error.
  mainWindow.webContents.on('console-message', (_event, detailsOrLevel, message, line, sourceId) => {
    const details = detailsOrLevel && typeof detailsOrLevel === 'object' ? detailsOrLevel : null
    const level = details ? details.level : detailsOrLevel

    if (level !== 3) return

    const text = details ? details.message : message
    const src = details ? details.sourceUrl : sourceId
    const lineNo = details ? details.lineNumber : line
    rememberLog(`[renderer console] ${text} (${src}:${lineNo})`)
  })

  if (DEV_SERVER) {
    mainWindow.loadURL(DEV_SERVER)
  } else {
    mainWindow.loadURL(pathToFileURL(resolveRendererIndex()).toString())
  }

  mainWindow.webContents.once('did-finish-load', () => {
    restorePersistedZoomLevel(mainWindow)
    broadcastBootProgress()
    sendWindowStateChanged()
    startHermes().catch(error => rememberLog(error.stack || error.message))
  })
}

ipcMain.handle('hermes:connection', async (_event, profile) => ensureBackend(profile))
// Reconnect-after-wake recovery. A REMOTE primary backend has no child process,
// so the 'exit'/'error' handlers that would clear a dead connectionPromise never
// fire — once the remote becomes unreachable across a sleep/wake the renderer
// re-dials the same dead descriptor forever and the composer stays stuck on
// "Starting Hermes…". Before the renderer's backoff loop reconnects, it asks us
// to confirm the cached PRIMARY backend is still reachable; if a remote one is
// not, we drop the cache so the next getConnection() rebuilds it. Local backends
// self-heal via their child 'exit' handler, so we never touch them here.
ipcMain.handle('hermes:connection:revalidate', async () => {
  if (!connectionPromise) {
    return { ok: true, rebuilt: false }
  }

  let conn = null
  try {
    conn = await connectionPromise
  } catch {
    // The cached boot already rejected (its own catch nulls connectionPromise);
    // nothing to revalidate — the next getConnection() builds fresh.
    return { ok: true, rebuilt: false }
  }

  if (!conn || conn.mode !== 'remote' || !conn.baseUrl) {
    return { ok: true, rebuilt: false }
  }

  const base = conn.baseUrl.replace(/\/+$/, '')
  try {
    await fetchPublicJson(`${base}/api/status`, { timeoutMs: 2_500 })
    return { ok: true, rebuilt: false }
  } catch {
    // Unreachable remote: drop the stale cache so the renderer's next reconnect
    // tick rebuilds a fresh, reachable descriptor. resetHermesConnection only
    // nulls connectionPromise for a remote (no child to SIGTERM).
    rememberLog('Cached remote Hermes backend failed liveness probe; dropping stale connection.')
    resetHermesConnection()
    return { ok: true, rebuilt: true }
  }
})
ipcMain.handle('hermes:backend:touch', async (_event, profile) => {
  touchPoolBackend(profile)
  return { ok: true }
})
ipcMain.handle('hermes:gateway:ws-url', async (_event, profile) => freshGatewayWsUrl(profile))
ipcMain.handle('hermes:window:openSession', async (_event, sessionId, opts) => {
  if (typeof sessionId !== 'string' || !sessionId.trim()) {
    return { ok: false, error: 'invalid-session-id' }
  }

  createSessionWindow(sessionId.trim(), { watch: opts?.watch === true })

  return { ok: true }
})
ipcMain.handle('hermes:window:openNewSession', async () => {
  createNewSessionWindow()

  return { ok: true }
})
ipcMain.handle('hermes:bootstrap:reset', async () => {
  // Renderer's "Reload and retry" path. Clear the latched failure and
  // reset connection state so the next startHermes() call restarts the
  // full backend flow (including a fresh runBootstrap pass).
  rememberLog('[bootstrap] reset requested by renderer; clearing latched failure')
  await teardownPrimaryBackendAndWait()
  bootstrapFailure = null
  bootstrapState = {
    active: false,
    manifest: null,
    stages: {},
    error: null,
    log: [],
    startedAt: null,
    completedAt: null,
    unsupportedPlatform: null
  }
  return { ok: true }
})
ipcMain.handle('hermes:bootstrap:repair', async () => {
  // Forceful repair: drop the bootstrap-complete marker so the next
  // startHermes() re-runs the full installer (refreshing a broken/partial
  // venv), and clear any latched failure + live connection. The renderer
  // reloads afterwards to re-drive the boot flow from scratch.
  rememberLog('[bootstrap] repair requested by renderer; clearing marker + latched failure')
  try {
    if (fileExists(BOOTSTRAP_COMPLETE_MARKER)) {
      fs.rmSync(BOOTSTRAP_COMPLETE_MARKER, { force: true })
    }
  } catch (error) {
    rememberLog(`[bootstrap] failed to remove marker during repair: ${error.message}`)
  }
  bootstrapFailure = null
  resetHermesConnection()
  return { ok: true }
})
ipcMain.handle('hermes:bootstrap:cancel', async () => {
  // Renderer's Cancel button during first-launch install. Abort the running
  // install script (SIGTERM via the runner's abortSignal). runBootstrap
  // resolves with { cancelled: true }, which surfaces the recovery overlay.
  if (bootstrapAbortController) {
    try {
      bootstrapAbortController.abort()
    } catch {
      void 0
    }
    return { ok: true, cancelled: true }
  }
  return { ok: false, cancelled: false }
})

// ── R5/R6: desktop opt-in runtime update ────────────────────────────────────
// version (R6): the installed engine version, read purely from the local
// bootstrap marker. No network, no state change — so the About panel can show
// the current engine version on open without an opt-in update check. Mirrors
// how checkForRuntimeUpdate derives `current` (commit||branch is the key).
ipcMain.handle('hermes:runtime:version', async () => {
  try {
    const marker = readBootstrapMarker()
    const commit = (marker && marker.pinnedCommit) || null
    const branch = (marker && marker.pinnedBranch) || null
    const version = (marker && marker.version) || null
    return { ok: true, version, commit, branch, key: commit || branch || null }
  } catch (error) {
    rememberLog(`[runtime-update] version read errored: ${error && error.message}`)
    return { ok: false, version: null, commit: null, branch: null, key: null }
  }
})

// check-update: compare the installed runtime (bootstrap marker) against the
// admin-set default (GET /api/v1/runtime/latest). Read-only; never mutates.
ipcMain.handle('hermes:runtime:check-update', async () => {
  try {
    const result = await checkForRuntimeUpdate({
      apiBase: apexApiBase(),
      fetchJson: fetchPublicJson,
      marker: readBootstrapMarker(),
      log: msg => rememberLog(msg)
    })
    return { ok: true, ...result }
  } catch (error) {
    // checkForRuntimeUpdate already swallows; defensive only.
    rememberLog(`[runtime-update] check-update errored: ${error && error.message}`)
    return { ok: false, updateAvailable: false, error: (error && error.message) || String(error) }
  }
})

// apply-update: opt-in (user-triggered) update to the admin default. Rollback-
// safe — we verify the new artifact is reachable, snapshot the current marker,
// persist a durable pin override, then drop the marker so the next boot re-runs
// our own bootstrap against the new pin. A failed/cancelled re-bootstrap rolls
// back to the snapshot (see rollbackRuntimePinOverride), so a working install is
// never bricked. The renderer reloads to drive the boot flow.
ipcMain.handle('hermes:runtime:apply-update', async () => {
  // 1. Resolve the target pin. No managed latest / offline -> nothing to do.
  let pin = null
  try {
    pin = await resolveLatestRuntimePin({
      apiBase: apexApiBase(),
      fetchJson: fetchPublicJson,
      log: msg => rememberLog(msg)
    })
  } catch (error) {
    return { ok: false, error: (error && error.message) || String(error) }
  }
  if (!pin) {
    return { ok: false, error: 'no_admin_latest_available' }
  }

  // 2. Skip if already on this pin (compare against the installed marker key).
  const marker = readBootstrapMarker()
  const installedKey = (marker && (marker.pinnedCommit || marker.pinnedBranch)) || null
  if (installedKey && String(installedKey) === String(pin.key)) {
    const versionMoved = Boolean(marker && marker.version && pin.version && marker.version !== pin.version)
    if (!versionMoved) {
      return { ok: true, applied: false, alreadyCurrent: true, latest: { version: pin.version, key: pin.key } }
    }
  }

  // 2b. hc-472 opt-in: when HERMES_BUNDLE_MODE is on, apply via the versioned
  //     bundle set (download → sha gate → never-in-place stage/verify → atomic
  //     pointer+link switch). Its own manifest fetch IS the reachability proof,
  //     so this runs BEFORE the legacy tarball reachability check. A min-desktop
  //     incompatibility is a hard stop (拒装+提示); any other failure falls
  //     through to the legacy re-bootstrap path below (the design回退 contract).
  if (bundleModeEnabled()) {
    rememberLog(`[bundle] apply-update via bundle set: key=${pin.key} version=${pin.version || '?'}`)
    const result = await applyRuntimeBundleUpdateFlow(pin)
    if (result.ok) {
      // Stamp the bootstrap-complete marker into the freshly-activated version
      // (design §4 step 4) so the next boot treats the switched runtime as good
      // and does NOT re-run install.ps1. The link already points at versions/<key>.
      try {
        writeBootstrapMarker({
          pinnedCommit: result.runtimeCommit || pin.commit || pin.key,
          pinnedBranch: pin.branch,
          version: pin.version
        })
      } catch (error) {
        rememberLog(`[bundle] switched but failed to stamp marker: ${error && error.message}`)
      }
      bootstrapFailure = null
      resetHermesConnection()
      return {
        ok: true,
        applied: true,
        via: 'bundle',
        reloadRequired: true,
        latest: { version: pin.version, key: pin.key, compatibilityNotes: pin.compatibilityNotes }
      }
    }
    if (result.code === 'min_desktop_version') {
      rememberLog(`[bundle] refusing update: ${result.error}`)
      return {
        ok: false,
        error: 'min_desktop_version',
        required: result.required,
        current: result.current,
        latest: { version: pin.version, key: pin.key }
      }
    }
    if (result.code === 'insufficient_disk') {
      // Hard stop: the legacy chain re-bootstraps IN PLACE and needs even more
      // disk, so falling back would only fail worse (and mutate a working
      // runtime). Surface the readable precheck message (C2, design §8).
      rememberLog(`[bundle] refusing update: ${result.error}`)
      return {
        ok: false,
        error: 'insufficient_disk',
        message: result.error,
        freeBytes: result.freeBytes,
        requiredBytes: result.requiredBytes,
        latest: { version: pin.version, key: pin.key }
      }
    }
    rememberLog(`[bundle] apply failed (${result.code}@${result.stage || '?'}); falling back to legacy install chain`)
    // fall through to the legacy steps 3-4 below
  }

  // 3. Don't-brick pre-flight: confirm the new source tarball actually exists
  //    before we retarget. (No URL -> non-CN git path, which verifies itself.)
  const reachable = await isUpdateArtifactReachable(pin.cosTarballUrl)
  if (!reachable) {
    rememberLog(
      `[runtime-update] aborting apply: update artifact not reachable (${pin.cosTarballUrl || 'n/a'}); ` +
        'keeping current runtime'
    )
    return { ok: false, error: 'update_artifact_unreachable', latest: { version: pin.version, key: pin.key } }
  }

  // 4. Persist the durable override WITH a rollback snapshot of the current
  //    marker, then drop the marker + reset the connection. The renderer
  //    reloads -> startHermes() re-runs bootstrap with resolveBootstrapStamp(),
  //    which reads the persisted override first.
  try {
    writeRuntimePinOverride({
      commit: pin.commit,
      branch: pin.branch,
      version: pin.version,
      previousMarker: marker || null
    })
  } catch (error) {
    return { ok: false, error: `failed_to_persist_override: ${(error && error.message) || error}` }
  }

  rememberLog(
    `[runtime-update] opt-in update armed: version=${pin.version || '?'} key=${pin.key}; ` +
      'dropping marker and re-running bootstrap'
  )
  try {
    if (fileExists(BOOTSTRAP_COMPLETE_MARKER)) {
      fs.rmSync(BOOTSTRAP_COMPLETE_MARKER, { force: true })
    }
  } catch (error) {
    // If we can't drop the marker the update won't trigger; roll back so we
    // don't leave a dangling override that fights the installed runtime.
    rollbackRuntimePinOverride('failed to drop marker')
    return { ok: false, error: `failed_to_clear_marker: ${(error && error.message) || error}` }
  }
  bootstrapFailure = null
  resetHermesConnection()
  return {
    ok: true,
    applied: true,
    reloadRequired: true,
    latest: { version: pin.version, key: pin.key, compatibilityNotes: pin.compatibilityNotes }
  }
})
ipcMain.handle('hermes:boot-progress:get', async () => bootProgressState)
ipcMain.handle('hermes:bootstrap:get', async () => getBootstrapState())
ipcMain.handle('hermes:connection-config:get', async (_event, profile) =>
  sanitizeDesktopConnectionConfig(readDesktopConnectionConfig(), profile)
)
ipcMain.handle('hermes:connection-config:test', async (_event, payload) => testDesktopConnectionConfig(payload))
ipcMain.handle('hermes:connection-config:probe', async (_event, rawUrl) => probeRemoteAuthMode(rawUrl))
ipcMain.handle('hermes:connection-config:oauth-login', async (_event, rawUrl) => {
  // Open the gateway's OAuth login window and wait for the session cookie to
  // land in the OAuth partition. The caller (settings UI) typically saves the
  // remote config with authMode='oauth' first, then calls this. We normalize
  // the URL defensively so a login can be driven from a raw URL too.
  const baseUrl = normalizeRemoteBaseUrl(rawUrl)
  await openOauthLoginWindow(baseUrl)
  return { ok: true, baseUrl, connected: await hasOauthSessionCookie(baseUrl) }
})
ipcMain.handle('hermes:connection-config:oauth-logout', async (_event, rawUrl) => {
  const baseUrl = rawUrl ? normalizeRemoteBaseUrl(rawUrl) : ''
  await clearOauthSession(baseUrl || undefined)
  // Report against the SAME liveness notion the Settings indicator uses
  // (AT-or-RT) so a logout that left any session cookie behind is reflected
  // as still-connected rather than silently signed-out.
  return { ok: true, connected: baseUrl ? await hasLiveOauthSession(baseUrl) : false }
})
ipcMain.handle('hermes:connection-config:save', async (_event, payload) => {
  const config = coerceDesktopConnectionConfig(payload)
  writeDesktopConnectionConfig(config)

  return sanitizeDesktopConnectionConfig(config, payload?.profile)
})
ipcMain.handle('hermes:connection-config:apply', async (_event, payload) => {
  const config = coerceDesktopConnectionConfig(payload)
  writeDesktopConnectionConfig(config)

  const key = connectionScopeKey(payload?.profile)

  if (key && key !== primaryProfileKey()) {
    // Editing a NON-primary profile's connection: don't disturb the window's
    // primary backend. Drop the profile's pooled backend so the next switch
    // re-resolves against the new remote/local target.
    stopPoolBackend(key)
  } else {
    // Global connection, or the primary profile's connection: re-home the
    // window backend by tearing it down and reloading the renderer.
    await teardownPrimaryBackendAndWait()
    mainWindow?.reload()
  }

  return sanitizeDesktopConnectionConfig(config, payload?.profile)
})

// ── ApexNodes managed-LLM IPC ───────────────────────────────────────────────
// status: whether the managed default is enabled for this build and whether the
// user is already signed in (relay key on disk). The renderer uses this to skip
// the BYOK picker on first run and show "managed, zero-key" instead.
ipcMain.handle('hermes:managed:status', async () => {
  const endpoints = resolveApexEndpoints(process.env)
  const managed = resolveManagedConfig()
  const account = managed.account || { email: '', name: '', plan: '' }
  return {
    enabled: isManagedEnabled(process.env),
    signedIn: Boolean(managed.key),
    // True only when a reusable login JWT is on disk — i.e. a real cloud
    // sign-in that CAN self-heal a rotated/expired relay key. A seeded/env key
    // (e.g. a `*.local` release account or a CI-provisioned test key) has a
    // relay key but no token: signedIn=true yet hasToken=false, so the UI can
    // show an honest "not connected to platform" state instead of pretending a
    // silent-failing account is a live managed sign-in.
    hasToken: Boolean(managed.accessToken),
    // When signed in, reflect the server-provided routing (base_url/model);
    // otherwise the env/default for display.
    model: managed.model || endpoints.model,
    modelDisplay: endpoints.modelDisplay,
    provider: endpoints.provider,
    baseUrl: managed.baseUrl || endpoints.relayBaseUrl,
    // Display-only identity for the account panel (empty strings when unknown /
    // signed out). Never a secret — the relay key stays encrypted on disk.
    email: account.email || '',
    name: account.name || '',
    plan: account.plan || ''
  }
})
// Shape a managed sign-in result into the IPC payload the renderer applies. When
// a relay key was provisioned, build the assignment from the STORED provision
// result (server-truth base_url + model), not env defaults. When provision-key
// wasn't available, assignment is null and the renderer falls back to BYOK.
function managedSignInResultPayload(result) {
  if (!result.hasRelayKey) {
    return { ok: true, hasRelayKey: false, assignment: null }
  }
  const managed = resolveManagedConfig()
  const block = buildManagedModelConfig(managed.key, process.env, {
    baseUrl: managed.baseUrl,
    model: managed.model
  })
  return {
    ok: true,
    hasRelayKey: true,
    assignment: {
      scope: 'main',
      provider: block.provider,
      model: block.default,
      base_url: block.base_url,
      api_key: block.api_key
    }
  }
}

// Map an error thrown by a sign-in path to the IPC `message`. The renderer turns
// these into Chinese copy. INVALID_CREDENTIALS is the login-or-register
// "wrong email/password" marker; everything else passes its message through.
function managedSignInErrorMessage(error) {
  if (error && error.code === 'INVALID_CREDENTIALS') {
    return 'INVALID_CREDENTIALS'
  }
  return error && error.message ? error.message : String(error)
}

// signIn: email+password → login-or-register → provision relay key
// (POST /api/v1/desktop/provision-key). Returns the model assignment the renderer
// should apply via /api/model/set (the SAME path the BYOK local-endpoint flow
// uses), so applying managed needs no new runtime plumbing. When provision-key
// isn't deployed yet, hasRelayKey=false and `assignment` is null — the renderer
// then falls back to the BYOK onboarding.
ipcMain.handle('hermes:managed:signIn', async (_event, payload) => {
  const email = String(payload?.email || '').trim()
  const password = String(payload?.password || '')
  if (!email || !password) {
    // EMPTY_FIELDS marker → renderer shows the Chinese "请输入邮箱和密码".
    return { ok: false, message: 'EMPTY_FIELDS' }
  }
  try {
    const result = await apexManagedSignIn({ email, password })
    return managedSignInResultPayload(result)
  } catch (error) {
    return { ok: false, message: managedSignInErrorMessage(error) }
  }
})

// browserSignIn: "用 Google 登录" / "用 APEX 登录". Open a loopback listener +
// random state, launch the system browser at the provider's start URL with our
// loopback redirect_uri, and wait for the browser to redirect back with
// `?token=<JWT>&state=<s>`. Validate state (CSRF), then run the SAME post-auth
// path (provision-key → assignment) as the email/password flow. Loopback is
// 127.0.0.1 only; the backend MUST also validate redirect_uri/desktop_cb.
ipcMain.handle('hermes:managed:browserSignIn', async (_event, payload) => {
  const provider = String(payload?.provider || '').trim()
  if (provider !== 'google' && provider !== 'apex') {
    return { ok: false, message: `Unknown browser sign-in provider: ${provider}` }
  }

  let loopback = null
  try {
    loopback = await startLoopbackLogin()
  } catch (error) {
    return { ok: false, message: error && error.message ? error.message : String(error) }
  }

  try {
    const startUrl =
      provider === 'google'
        ? googleStartUrl(loopback.redirectUri, loopback.state, process.env)
        : apexWebLoginUrl(loopback.redirectUri, loopback.state, process.env)

    if (!openExternalUrl(startUrl)) {
      loopback.close()
      return { ok: false, message: 'Could not open the system browser for sign-in.' }
    }

    // Block until the browser redirects back (or the watchdog/abort fires).
    const { token } = await loopback.result
    const result = await provisionManagedFromAccessToken(token)
    return managedSignInResultPayload(result)
  } catch (error) {
    loopback.close()
    return { ok: false, message: managedSignInErrorMessage(error) }
  }
})
// signOut: forget the relay key. The renderer is responsible for re-pointing the
// model at a BYOK provider if the user wants to keep chatting.
ipcMain.handle('hermes:managed:signOut', async () => {
  clearManagedRelayCredential()
  return { ok: true }
})

// selfHeal: on-demand relay-key recovery, triggered by the renderer when a chat
// turn fails with a relay auth error (HTTP 401/403). Runs the SAME gated probe +
// re-provision as the boot self-heal and reports the outcome so the renderer can
// either apply the fresh key + retry once (healed), or route to re-sign-in when
// there is no reusable login token (a `*.local`/env seed key, or an expired JWT)
// — turning a silent 401 loop into a visible, actionable state. Never throws.
ipcMain.handle('hermes:managed:selfHeal', async () => {
  const outcome = await selfHealManagedKeyOn401()
  // Relay accepted the key (or managed off / no key): not a managed-relay auth
  // problem — let the renderer's generic error path surface it.
  if (!outcome || !outcome.relayUnauthorized) {
    return { ok: true, relayUnauthorized: false, healed: false, needsSignIn: false, assignment: null }
  }
  if (outcome.healed) {
    // Fresh key on disk + config.yaml re-synced; hand back the assignment so the
    // renderer applies it via /api/model/set (same path as sign-in) and retries.
    return {
      ok: true,
      relayUnauthorized: true,
      healed: true,
      needsSignIn: false,
      assignment: managedSignInResultPayload({ hasRelayKey: true }).assignment
    }
  }
  // Could not heal (no token, or the stored JWT is itself expired) → the user
  // must sign in again. Honest, visible state instead of a silent 401 loop.
  return { ok: true, relayUnauthorized: true, healed: false, needsSignIn: true, assignment: null }
})

// ── hc-444: Feishu bridge (renderer surface) ────────────────────────────────
// status: read-only view of the LOCAL stored credential for the settings card —
// no network, no secret. `connected` reflects a stored, injectable credential;
// `signedIn` tells the card whether a managed sign-in exists (the prerequisite
// for sync, since sync authenticates with the stored login JWT).
ipcMain.handle('hermes:feishu:status', async () => {
  const stored = resolveFeishuConfig()
  const managed = resolveManagedConfig()
  return {
    connected: stored.connected,
    signedIn: Boolean(String(managed.accessToken || '').trim()),
    agentName: stored.agentName || '',
    domain: stored.domain || '',
    credentialStatus: stored.credentialStatus || '',
    syncedAt: stored.syncedAt || null
  }
})

// sync: fetch the signed-in user's cloud Feishu credential and persist it
// (encrypted), then re-home the backend so the runtime boots with the new
// FEISHU_* env and the Feishu adapter + lark tools come alive. On hasEntry=false
// the renderer opens the web binding flow (openBind). On needsSignIn the renderer
// routes the user through managed sign-in first. The backend restart only happens
// when a credential was actually stored (hasEntry) — a no-op sync shouldn't churn
// the runtime.
ipcMain.handle('hermes:feishu:sync', async () => {
  const result = await fetchAndStoreFeishuCredentials()
  if (result.ok && result.hasEntry) {
    // Re-home the local backend (same teardown+reload path as a profile switch)
    // so the freshly-injected FEISHU_* env takes effect immediately.
    await teardownPrimaryBackendAndWait()
    mainWindow?.reload()
  }
  return result
})

// disconnect: forget the local Feishu credential and restart the backend so the
// adapter goes dark on the next boot. Does NOT touch the cloud entry (the user's
// app binding stays intact for the cloud webhook line + other devices) — this is
// a desktop-local un-sync only.
ipcMain.handle('hermes:feishu:disconnect', async () => {
  clearFeishuConfig()
  await teardownPrimaryBackendAndWait()
  mainWindow?.reload()
  return { ok: true }
})

// openBind: open the cloud web binding flow in the system browser for a user who
// has no Feishu app bound yet. China-first locale (matches apexWebLoginUrl's /zh
// pin). After the user finishes binding in the browser, they press "Sync" back on
// the card. Returns { ok } — the actual sync stays an explicit user action so we
// never poll a browser tab we don't control.
ipcMain.handle('hermes:feishu:openBind', async () => {
  const endpoints = resolveApexEndpoints(process.env)
  const url = `${endpoints.authBase}/zh/createbot`
  const opened = openExternalUrl(url)
  return { ok: opened, url }
})

// ── Platform client-config sync (renderer surface) ──────────────────────────
// get: the cached state from disk — no network, informational only. The APPLY
// now happens entirely in the main process pre-gateway (applyClientConfigToRuntime);
// the renderer neither applies nor records versions anymore.
ipcMain.handle('hermes:clientConfig:get', async () => {
  const state = readClientConfigState()
  return { version: state.version, payload: state.payload, appliedVersion: state.appliedVersion }
})

ipcMain.handle('hermes:profile:get', async () => ({ profile: readActiveDesktopProfile() }))
ipcMain.handle('hermes:profile:set', async (_event, name) => {
  const next = writeActiveDesktopProfile(name)

  // Switching profiles is a backend re-home: relaunch the dashboard under the
  // new HERMES_HOME. Pool backends keep their own homes, so only the primary
  // is torn down.
  await teardownPrimaryBackendAndWait()
  mainWindow?.reload()

  return { profile: next }
})

ipcMain.on('hermes:previewShortcutActive', (_event, active) => {
  previewShortcutActive = Boolean(active)
})

ipcMain.handle('hermes:requestMicrophoneAccess', async () => {
  if (!IS_MAC || typeof systemPreferences.askForMediaAccess !== 'function') {
    return true
  }

  return systemPreferences.askForMediaAccess('microphone')
})

// Re-route remote-profile session requests to the owning remote backend. Returns
// `undefined` when not interceptable (caller takes the normal local path), else
// the response. Reads tag the profile as ?profile=<name>; mutations carry it in
// request.profile. Either way, a remote profile's session lives only on its
// remote host, so the request must go there (where it serves its own state.db).
//   GET    /api/profiles/sessions        → splice each remote profile's rows in
//   GET    /api/sessions/{id}[/messages] → read from remote
//   DELETE /api/sessions/{id}            → delete on remote
//   PATCH  /api/sessions/{id}            → rename/archive on remote
async function interceptSessionRequestForRemote(request) {
  if (typeof request?.path !== 'string') {
    return undefined
  }
  const method = (request.method || 'GET').toUpperCase()

  let parsed
  try {
    parsed = new URL(request.path, 'http://x')
  } catch {
    return undefined
  }
  const { pathname, searchParams } = parsed

  if (method === 'GET' && pathname === '/api/profiles/sessions') {
    const remoteProfiles = configuredRemoteProfileNames()
    if (remoteProfiles.length === 0) {
      return undefined // no remote profiles → local fast path
    }
    const requested = (searchParams.get('profile') || 'all').trim() || 'all'
    if (requested !== 'all') {
      return profileHasRemoteOverride(requested) ? remoteSessionList(requested, searchParams) : undefined
    }
    return mergeRemoteProfileSessions(searchParams, remoteProfiles)
  }

  // Per-session read/mutation. Owner is in ?profile= (reads) or request.profile
  // (mutations). Two remote shapes:
  //  - per-profile override: route to that profile's own remote, sans profile
  //    param (it serves its own state.db natively).
  //  - global remote mode: ONE backend serves every profile via ?profile=, so
  //    route there and KEEP the profile param so it opens the right state.db.
  if (/^\/api\/sessions\/[^/]+(\/messages)?$/.test(pathname)) {
    const profile = (searchParams.get('profile') || request.profile || '').trim()
    if (!profile) {
      return undefined
    }
    if (profileHasRemoteOverride(profile)) {
      if (method === 'GET') {
        return fetchJsonForProfile(profile, pathname)
      }
      const body = request.body && typeof request.body === 'object' ? { ...request.body } : request.body
      if (body) delete body.profile
      return requestJsonForProfile(profile, pathname, method, body)
    }
    if (globalRemoteActive()) {
      // Single global backend: keep ?profile= so it opens the right state.db.
      const sep = pathname.includes('?') ? '&' : '?'
      const path = `${pathname}${sep}profile=${encodeURIComponent(profile)}`
      if (method === 'GET') {
        return fetchJsonForProfile(null, path)
      }
      const body = request.body && typeof request.body === 'object' ? { ...request.body, profile } : { profile }
      return requestJsonForProfile(null, path, method, body)
    }
    return undefined
  }

  return undefined
}

const rowsOf = data => (Array.isArray(data?.sessions) ? data.sessions : [])

// A remote profile's session list, read from its remote host and tagged with the
// desktop-facing profile name (the remote's /api/sessions doesn't know it).
async function remoteSessionList(profile, searchParams) {
  const qs = new URLSearchParams(searchParams)
  qs.delete('profile') // remote serves its own db; no cross-profile read there
  const data = await fetchJsonForProfile(profile, `/api/sessions?${qs}`)
  for (const s of rowsOf(data)) {
    s.profile = profile
    s.is_default_profile = false
  }
  return { ...data, sessions: rowsOf(data) }
}

// Unified list: primary's local aggregate, with each remote profile's stale local
// rows/totals swapped for the remote's real ones, re-sorted by recency and
// re-windowed to the requested page. A dead remote contributes nothing rather
// than breaking the sidebar.
async function mergeRemoteProfileSessions(searchParams, remoteProfiles) {
  const limit = Math.max(1, Number(searchParams.get('limit')) || 20)
  const offset = Math.max(0, Number(searchParams.get('offset')) || 0)
  const order = searchParams.get('order') === 'created' ? 'started_at' : 'last_active'

  const primary = await ensureBackend(null)
  const base = await fetchJson(`${primary.baseUrl}/api/profiles/sessions?${searchParams}`, primary.token, {
    method: 'GET',
    timeoutMs: DEFAULT_FETCH_TIMEOUT_MS
  }).catch(() => ({ sessions: [], total: 0, profile_totals: {} }))

  // Over-fetch each remote from offset 0 (limit+offset rows) so the merged window
  // is correct for this page — mirrors the primary's per-profile over-fetch.
  const remoteParams = new URLSearchParams(searchParams)
  remoteParams.set('limit', String(limit + offset))
  remoteParams.set('offset', '0')

  const remoteSet = new Set(remoteProfiles)
  const merged = rowsOf(base).filter(s => !remoteSet.has(s?.profile))
  const profileTotals = { ...(base.profile_totals || {}) }
  let total = (Number(base.total) || 0) - remoteProfiles.reduce((n, p) => n + (profileTotals[p] || 0), 0)

  // Swap each remote profile's stale local rows/total for the remote's real ones.
  await Promise.all(
    remoteProfiles.map(async name => {
      const list = await remoteSessionList(name, remoteParams).catch(() => null)
      if (!list) {
        delete profileTotals[name] // dead remote → drop its stale local total too
        return
      }
      const rows = rowsOf(list)
      merged.push(...rows)
      profileTotals[name] = Number(list.total) || rows.length
      total += profileTotals[name]
    })
  )

  const recency = s => s?.[order] ?? s?.started_at ?? 0
  merged.sort((a, b) => recency(b) - recency(a))
  return { ...base, sessions: merged.slice(offset, offset + limit), total, profile_totals: profileTotals }
}

// Both fetchJson and fetchJsonViaOauthSession reject with a message shaped
// `"<statusCode>: <body>"`. Parse the leading status code (fetchJsonViaOauthSession
// also attaches err.statusCode). Returns null when it isn't an HTTP-status error.
function httpStatusFromError(error) {
  if (error && typeof error.statusCode === 'number') {
    return error.statusCode
  }
  const message = error && error.message ? String(error.message) : ''
  const match = /^(\d{3}):/.exec(message)
  return match ? Number(match[1]) : null
}

// Continuous auth gate: when a backend call comes back 401 (login lost / token
// invalid) or 403 account_disabled (account abnormal), tell every renderer so it
// can clear auth and return to the login screen. The 403 case is narrowed to an
// `account_disabled` body so an ordinary permission 403 doesn't sign the user
// out. Best-effort + never throws — a broadcast failure must not break the call's
// own error handling.
function broadcastAuthGate(error) {
  try {
    const statusCode = httpStatusFromError(error)
    if (statusCode !== 401 && statusCode !== 403) {
      return
    }
    const message = error && error.message ? String(error.message) : ''
    const body = message.replace(/^\d{3}:\s*/, '')
    const disabled = /account_disabled/i.test(body)
    // A 403 that is NOT an account-disabled signal is a routine authorization
    // error, not a login-lost event — leave the session intact.
    if (statusCode === 403 && !disabled) {
      return
    }
    const reason = statusCode === 403 ? 'account_disabled' : 'unauthorized'
    for (const win of BrowserWindow.getAllWindows()) {
      if (!win.isDestroyed()) {
        win.webContents.send('hermes:auth-gate', { statusCode, reason })
      }
    }
  } catch {
    // Never let the gate broadcast interfere with the caller's error path.
  }
}

ipcMain.handle('hermes:api', async (_event, request) => {
  // Remote-profile session requests would otherwise hit the local primary off
  // each profile's on-disk state.db — fine for local profiles, but a remote
  // profile's sessions live on its remote host, so the UI's IDs 404 (or mutations
  // no-op) the moment they run there. Route reads + mutations to the remote.
  const rerouted = await interceptSessionRequestForRemote(request)
  if (rerouted !== undefined) {
    return rerouted
  }

  await prepareProfileDeleteRequest(request)

  const profile = request?.profile
  const connection = await ensureBackend(profile)
  const timeoutMs = resolveTimeoutMs(request?.timeoutMs, DEFAULT_FETCH_TIMEOUT_MS)
  const requestPath = pathWithGlobalRemoteProfile(request.path, profile, {
    globalRemote: globalRemoteActive(),
    profileRemoteOverride: profileHasRemoteOverride(profile)
  })
  const url = `${connection.baseUrl}${requestPath}`
  // OAuth gateways authenticate REST via the HttpOnly session cookie held in
  // the OAuth partition — route through Electron's net stack bound to that
  // session so the cookie attaches automatically. Token/local modes keep using
  // the static session-token header.
  try {
    if (connection.authMode === 'oauth') {
      return await fetchJsonViaOauthSession(url, {
        method: request?.method,
        body: request?.body,
        timeoutMs
      })
    }
    return await fetchJson(url, connection.token, {
      method: request?.method,
      body: request?.body,
      timeoutMs
    })
  } catch (error) {
    // Fire the continuous auth gate on 401 / 403 account_disabled, then rethrow
    // so the caller's own error handling is unchanged.
    broadcastAuthGate(error)
    throw error
  }
})

ipcMain.handle('hermes:notify', (_event, payload) => {
  if (!Notification.isSupported()) return false
  // Action buttons render only on signed macOS builds; elsewhere they're dropped
  // and the body click still works.
  const actions = Array.isArray(payload?.actions) ? payload.actions : []
  const notification = new Notification({
    title: payload?.title || APP_NAME,
    body: payload?.body || '',
    silent: Boolean(payload?.silent),
    actions: actions.map(action => ({ type: 'button', text: String(action?.text || '') }))
  })
  notification.on('click', () => {
    if (!mainWindow || mainWindow.isDestroyed()) return
    focusWindow(mainWindow)
    if (payload?.sessionId) {
      mainWindow.webContents.send('hermes:focus-session', payload.sessionId)
    }
  })
  notification.on('action', (_actionEvent, index) => {
    if (!mainWindow || mainWindow.isDestroyed()) return
    const action = actions[index]
    if (action?.id) {
      mainWindow.webContents.send('hermes:notification-action', { sessionId: payload?.sessionId, actionId: action.id })
    }
  })
  notification.show()
  return true
})

ipcMain.handle('hermes:readFileDataUrl', async (_event, filePath) => {
  const { resolvedPath } = await resolveReadableFileForIpc(filePath, {
    maxBytes: DATA_URL_READ_MAX_BYTES,
    purpose: 'File preview'
  })
  const data = await fs.promises.readFile(resolvedPath)
  return `data:${mimeTypeForPath(resolvedPath)};base64,${data.toString('base64')}`
})

ipcMain.handle('hermes:readFileText', async (_event, filePath) => {
  const { resolvedPath, stat } = await resolveReadableFileForIpc(filePath, {
    maxBytes: TEXT_PREVIEW_SOURCE_MAX_BYTES,
    purpose: 'Text preview'
  })
  const ext = path.extname(resolvedPath).toLowerCase()
  const handle = await fs.promises.open(resolvedPath, 'r')
  const bytesToRead = Math.min(stat.size, TEXT_PREVIEW_MAX_BYTES)

  try {
    const buffer = Buffer.alloc(bytesToRead)
    const { bytesRead } = await handle.read(buffer, 0, bytesToRead, 0)

    return {
      binary: looksBinary(buffer.subarray(0, Math.min(bytesRead, 4096))),
      byteSize: stat.size,
      language: PREVIEW_LANGUAGE_BY_EXT[ext] || 'text',
      mimeType: mimeTypeForPath(resolvedPath),
      path: resolvedPath,
      text: buffer.subarray(0, bytesRead).toString('utf8'),
      truncated: stat.size > TEXT_PREVIEW_MAX_BYTES
    }
  } finally {
    await handle.close()
  }
})

ipcMain.handle('hermes:selectPaths', async (_event, options = {}) => {
  const properties = options?.directories ? ['openDirectory'] : ['openFile']
  if (options?.multiple !== false) properties.push('multiSelections')

  let resolvedDefaultPath
  if (options?.defaultPath) {
    try {
      resolvedDefaultPath = path.resolve(String(options.defaultPath))
    } catch {
      resolvedDefaultPath = undefined
    }
  }

  const result = await dialog.showOpenDialog(mainWindow, {
    title: options?.title || 'Add context',
    defaultPath: resolvedDefaultPath,
    properties,
    filters: Array.isArray(options?.filters) ? options.filters : undefined
  })

  if (result.canceled) return []
  return result.filePaths
})

ipcMain.handle('hermes:writeClipboard', (_event, text) => {
  clipboard.writeText(String(text || ''))
  return true
})

ipcMain.handle('hermes:saveImageFromUrl', (_event, url) => saveImageFromUrl(String(url || '')))

ipcMain.handle('hermes:saveImageBuffer', async (_event, payload) => {
  const data = payload?.data
  if (!data) throw new Error('saveImageBuffer: missing data')

  const buffer = Buffer.isBuffer(data) ? data : Buffer.from(data)
  return writeComposerImage(buffer, payload?.ext || '.png')
})

ipcMain.handle('hermes:saveClipboardImage', async () => {
  const image = clipboard.readImage()
  if (!image || image.isEmpty()) {
    return ''
  }

  return writeComposerImage(image.toPNG(), '.png')
})

ipcMain.handle('hermes:normalizePreviewTarget', (_event, target, baseDir) =>
  normalizePreviewTarget(String(target || ''), baseDir ? String(baseDir) : '')
)

ipcMain.handle('hermes:watchPreviewFile', (_event, url) => watchPreviewFile(String(url || '')))

ipcMain.handle('hermes:stopPreviewFileWatch', (_event, id) => stopPreviewFileWatch(String(id || '')))

ipcMain.on('hermes:titlebar-theme', (_event, payload) => {
  if (!payload || !isHexColor(payload.background) || !isHexColor(payload.foreground)) {
    return
  }

  rendererTitleBarTheme = {
    background: payload.background,
    foreground: payload.foreground
  }
  mainWindow?.setTitleBarOverlay?.(getTitleBarOverlayOptions())
})

// Pin the native appearance to the app theme (see NATIVE_THEME_CONFIG_PATH).
ipcMain.on('hermes:native-theme', (_event, mode) => {
  if (!THEME_SOURCES.has(mode)) {
    return
  }

  if (nativeTheme.themeSource !== mode) {
    nativeTheme.themeSource = mode
    writePersistedThemeSource(mode)
  }
})

// See-through window translucency. Persist + re-apply opacity to every open
// window at runtime (no recreation, so caching/sessions are untouched).
ipcMain.on('hermes:translucency', (_event, payload) => {
  const next = clampIntensity(payload && payload.intensity)

  if (next === translucencyIntensity) {
    return
  }

  translucencyIntensity = next
  writePersistedTranslucency(next)

  for (const win of BrowserWindow.getAllWindows()) {
    applyWindowTranslucency(win)
  }
})

ipcMain.handle('hermes:openExternal', (_event, url) => {
  if (!openExternalUrl(url)) {
    throw new Error('Invalid external URL')
  }
})

// User-configurable default project directory. The renderer reads this on
// settings mount and seeds the value into the picker; writing back persists
// it via writeDefaultProjectDir so resolveHermesCwd picks it up on the next
// session spawn (no app restart needed).
ipcMain.handle('hermes:setting:defaultProjectDir:get', async () => ({
  dir: readDefaultProjectDir(),
  defaultLabel: app.getPath('home'),
  resolvedCwd: resolveHermesCwd()
}))

ipcMain.handle('hermes:workspace:sanitize', async (_event, cwd) => sanitizeWorkspaceCwd(cwd))

ipcMain.handle('hermes:setting:defaultProjectDir:set', async (_event, dir) => {
  const next = typeof dir === 'string' && dir.trim() ? dir.trim() : null

  if (next) {
    try {
      fs.mkdirSync(next, { recursive: true })
    } catch (error) {
      throw new Error(`Could not create directory: ${error.message}`)
    }
  }

  writeDefaultProjectDir(next)

  return { dir: next }
})

ipcMain.handle('hermes:setting:defaultProjectDir:pick', async () => {
  const result = await dialog.showOpenDialog({
    title: 'Choose default project directory',
    properties: ['openDirectory', 'createDirectory'],
    defaultPath: readDefaultProjectDir() || app.getPath('home')
  })

  if (result.canceled || result.filePaths.length === 0) {
    return { canceled: true, dir: null }
  }

  return { canceled: false, dir: result.filePaths[0] }
})

ipcMain.handle('hermes:fetchLinkTitle', (_event, url) => fetchLinkTitle(url))

ipcMain.handle('hermes:logs:reveal', async () => {
  try {
    await fs.promises.mkdir(path.dirname(DESKTOP_LOG_PATH), { recursive: true })
    if (!fileExists(DESKTOP_LOG_PATH)) {
      await fs.promises.appendFile(DESKTOP_LOG_PATH, '')
    }
    shell.showItemInFolder(DESKTOP_LOG_PATH)
    return { ok: true, path: DESKTOP_LOG_PATH }
  } catch (error) {
    return { ok: false, path: DESKTOP_LOG_PATH, error: error.message }
  }
})

ipcMain.handle('hermes:logs:recent', async () => ({ path: DESKTOP_LOG_PATH, lines: hermesLog.slice(-200) }))

function isExecutableFile(filePath) {
  if (!filePath || !path.isAbsolute(filePath)) {
    return false
  }

  try {
    fs.accessSync(filePath, fs.constants.X_OK)
    return true
  } catch {
    return false
  }
}

function posixShellSpec(shellPath) {
  const shellName = path.basename(shellPath)
  const interactiveArgs = shellName.includes('zsh') || shellName.includes('bash') ? ['-il'] : ['-i']

  return { args: interactiveArgs, command: shellPath, name: shellName }
}

let spawnHelperChecked = false

// node-pty execs a `spawn-helper` binary on macOS/Linux to launch the shell in a
// fresh session. The prebuilt that ships in node-pty's `prebuilds/` (and the
// staged copy under resources/native-deps) loses its execute bit through npm
// pack / electron-builder file collection, so every nodePty.spawn() dies with
// "posix_spawnp failed". Restore +x once, lazily, before the first spawn.
function ensureSpawnHelperExecutable() {
  if (spawnHelperChecked || IS_WINDOWS || !nodePtyDir) {
    return
  }

  spawnHelperChecked = true

  const arch = process.arch
  const candidates = [
    path.join(nodePtyDir, 'build', 'Release', 'spawn-helper'),
    path.join(nodePtyDir, 'prebuilds', `${process.platform}-${arch}`, 'spawn-helper')
  ]

  for (const helper of candidates) {
    try {
      const mode = fs.statSync(helper).mode

      if ((mode & 0o111) !== 0o111) {
        fs.chmodSync(helper, mode | 0o755)
      }
    } catch {
      // Not present in this layout (e.g. compiled build vs prebuild); skip.
    }
  }
}

// Windows PowerShell 5.1 ships at a fixed System32 path on every Windows box;
// prefer it only after PowerShell 7+ (`pwsh`).
function windowsPowerShellPath() {
  const systemRoot = process.env.SystemRoot || process.env.windir || 'C:\\Windows'
  const builtin = path.join(systemRoot, 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe')

  return isExecutableFile(builtin) ? builtin : findOnPath('powershell.exe')
}

// Map a resolved shell path to its spawn spec, picking interactive flags by
// family: PowerShell drops its logo banner (so the prompt sits flush like the
// POSIX shells), cmd needs nothing, and everything else (zsh/bash/fish/sh…)
// gets POSIX interactive-login flags.
function shellSpecFor(shellPath) {
  const name = path.basename(shellPath).toLowerCase()

  if (name.startsWith('pwsh') || name.startsWith('powershell')) {
    return { args: ['-NoLogo'], command: shellPath, name }
  }

  if (name.startsWith('cmd')) {
    return { args: [], command: shellPath, name }
  }

  return posixShellSpec(shellPath)
}

// Best installed Windows shell: PowerShell 7+ (`pwsh`), then Windows PowerShell
// 5.1, then comspec/cmd.exe as the universal fallback.
function windowsShellSpec() {
  const command =
    findOnPath('pwsh.exe') || findOnPath('pwsh') || windowsPowerShellPath() || process.env.COMSPEC || 'cmd.exe'

  return shellSpecFor(command)
}

// Resolve the interactive shell for the embedded terminal: an explicit user
// override wins, otherwise auto-detect the best one installed for the platform.
function terminalShellCommand() {
  // HERMES_DESKTOP_SHELL is the cross-platform escape hatch (a path or a bare
  // name on PATH); $SHELL is honored on POSIX, where it's the user's canonical
  // choice, but ignored on Windows, where it's usually a stray MSYS/Git path
  // node-pty can't spawn natively.
  const override = (process.env.HERMES_DESKTOP_SHELL || (IS_WINDOWS ? '' : process.env.SHELL) || '').trim()

  if (override) {
    const resolved = isExecutableFile(override) ? override : findOnPath(override)

    if (resolved) {
      return shellSpecFor(resolved)
    }
  }

  if (IS_WINDOWS) {
    return windowsShellSpec()
  }

  const shellPath = ['/bin/zsh', '/bin/bash', '/bin/sh'].find(candidate => isExecutableFile(candidate))

  return posixShellSpec(shellPath || '/bin/sh')
}

function safeTerminalCwd(cwd) {
  const candidate = path.resolve(String(cwd || app.getPath('home')))

  try {
    const stat = fs.statSync(candidate)

    return stat.isDirectory() ? candidate : path.dirname(candidate)
  } catch {
    return app.getPath('home')
  }
}

function terminalShellEnv() {
  const env = { ...process.env }

  // Electron is commonly launched through `npm run dev`; do not leak npm's
  // managed prefix into a user's interactive shell (nvm/proto warn loudly).
  for (const key of Object.keys(env)) {
    if (key === 'npm_config_prefix' || key.startsWith('npm_config_') || key.startsWith('npm_package_')) {
      delete env[key]
    }
  }

  // Strip color/theme-detection vars that ride along when Electron is launched
  // from a non-tty agent shell (Cursor's runner sets NO_COLOR/FORCE_COLOR=0
  // /TERM=dumb; some terminals set COLORFGBG which would flip Hermes' TUI into
  // light-mode). Our PTY is a real xterm-compat terminal — force truecolor.
  delete env.NO_COLOR
  delete env.FORCE_COLOR
  delete env.COLORFGBG

  env.COLORTERM = 'truecolor'
  env.LC_CTYPE = env.LC_CTYPE || 'UTF-8'
  env.TERM = 'xterm-256color'
  env.TERM_PROGRAM = 'Hermes'
  env.TERM_PROGRAM_VERSION = app.getVersion()

  // Let a hermes/--tui launched in this pane know it's embedded in the desktop
  // GUI (build_environment_hints surfaces this). Distinct from HERMES_DESKTOP,
  // which marks the agent *backend* and gates cron/gateway behavior.
  env.HERMES_DESKTOP_TERMINAL = '1'

  return env
}

function terminalChannel(id, suffix) {
  return `hermes:terminal:${id}:${suffix}`
}

function disposeTerminalSession(id) {
  const sessionInfo = terminalSessions.get(id)

  if (!sessionInfo) {
    return false
  }

  terminalSessions.delete(id)

  try {
    sessionInfo.pty.kill()
  } catch {
    // Process may already be gone.
  }

  return true
}

ipcMain.handle('hermes:fs:readDir', async (_event, dirPath) => readDirForIpc(dirPath))

ipcMain.handle('hermes:fs:gitRoot', async (_event, startPath) => gitRootForIpc(startPath))

ipcMain.handle('hermes:fs:worktrees', async (_event, cwds) => worktreesForIpc(cwds))

ipcMain.handle('hermes:terminal:start', async (event, payload = {}) => {
  if (!nodePty) {
    throw new Error('PTY support is unavailable. Reinstall desktop dependencies and restart Hermes.')
  }

  ensureSpawnHelperExecutable()

  const id = crypto.randomUUID()
  const { args, command, name } = terminalShellCommand()
  const cwd = safeTerminalCwd(payload?.cwd)
  const cols = Math.max(2, Number.parseInt(String(payload?.cols || 80), 10) || 80)
  const rows = Math.max(2, Number.parseInt(String(payload?.rows || 24), 10) || 24)
  const ptyProcess = nodePty.spawn(command, args, {
    cols,
    cwd,
    env: terminalShellEnv(),
    name: 'xterm-256color',
    rows
  })

  terminalSessions.set(id, { pty: ptyProcess, webContentsId: event.sender.id })

  const send = (suffix, payload) => {
    if (event.sender.isDestroyed()) {
      return
    }

    event.sender.send(terminalChannel(id, suffix), payload)
  }

  ptyProcess.onData(data => send('data', data))
  ptyProcess.onExit(({ exitCode, signal }) => {
    terminalSessions.delete(id)
    send('exit', { code: exitCode, signal: signal || null })
  })
  event.sender.once('destroyed', () => disposeTerminalSession(id))

  return { cwd, id, shell: name }
})

ipcMain.handle('hermes:terminal:write', (_event, id, data) => {
  const sessionInfo = terminalSessions.get(String(id || ''))

  if (!sessionInfo) {
    return false
  }

  sessionInfo.pty.write(String(data || ''))

  return true
})

ipcMain.handle('hermes:terminal:resize', (_event, id, size = {}) => {
  const sessionInfo = terminalSessions.get(String(id || ''))

  if (!sessionInfo) {
    return false
  }

  const cols = Math.max(2, Number.parseInt(String(size?.cols || 80), 10) || 80)
  const rows = Math.max(2, Number.parseInt(String(size?.rows || 24), 10) || 24)

  sessionInfo.pty.resize(cols, rows)

  return true
})
ipcMain.handle('hermes:terminal:dispose', (_event, id) => disposeTerminalSession(String(id || '')))

ipcMain.handle('hermes:updates:check', async () =>
  checkUpdates().catch(error => ({
    supported: true,
    branch: readDesktopUpdateConfig().branch,
    error: 'check-failed',
    message: error?.message || String(error),
    fetchedAt: Date.now()
  }))
)

ipcMain.handle('hermes:updates:apply', async (_event, payload) =>
  applyUpdates(payload || {}).catch(error => ({
    ok: false,
    error: 'apply-failed',
    message: error?.message || String(error)
  }))
)

ipcMain.handle('hermes:updates:branch:get', async () => readDesktopUpdateConfig())

ipcMain.handle('hermes:updates:branch:set', async (_event, name) => {
  const branch = typeof name === 'string' && name.trim() ? name.trim() : DEFAULT_UPDATE_BRANCH
  writeDesktopUpdateConfig({ branch })
  return { branch }
})

// Resolve the canonical Hermes version (the one `release.py` bumps in
// hermes_cli/__init__.py + pyproject.toml) so the desktop About panel shows the
// real Hermes version instead of the Electron app's own package.json version,
// which historically drifted (stuck at 0.0.2). Falls back to app.getVersion()
// when the source tree can't be read (e.g. a packaged build without the repo).
function resolveHermesVersion() {
  try {
    const root = resolveUpdateRoot()
    const initPath = path.join(root, 'hermes_cli', '__init__.py')
    if (fileExists(initPath)) {
      const raw = fs.readFileSync(initPath, 'utf8')
      const match = raw.match(/__version__\s*=\s*["']([^"']+)["']/)
      if (match) {
        return match[1]
      }
    }
  } catch {
    // Fall through to the Electron app version below.
  }
  return app.getVersion()
}

// Re-resolve the live Hermes version and push it into the native About panel
// just before showing it, so an in-place `hermes update` is reflected without
// an app restart. macOS only — `showAboutPanel()` is a no-op elsewhere, and the
// other platforms don't use this menu item.
function showAboutPanelFresh() {
  app.setAboutPanelOptions({
    applicationName: APP_NAME,
    // Shell (installer) version — the engine (runtime) version is shown by
    // About's engine-update section. resolveHermesVersion() reads the ENGINE
    // source tree and confusingly surfaced e.g. "0.17.0" as the app version.
    applicationVersion: app.getVersion(),
    copyright: 'Copyright © 2026 ApexNodes'
  })
  app.showAboutPanel()
}

ipcMain.handle('hermes:version', async () => ({
  appVersion: app.getVersion(),
  electronVersion: process.versions.electron,
  nodeVersion: process.versions.node,
  platform: process.platform,
  hermesRoot: resolveUpdateRoot()
}))

// ===========================================================================
// Uninstall — remove the Chat GUI (and optionally the agent / user data).
// ===========================================================================
//
// The renderer's About → Danger Zone surfaces three options that mirror the
// CLI exactly: GUI only, Lite (keep user data), Full. We ask the agent to do
// the actual removal via `hermes uninstall …` so the cross-platform PATH /
// registry / service / node-symlink cleanup all lives in one place
// (hermes_cli/uninstall.py + hermes_cli/gui_uninstall.py).
//
// getUninstallSummary() shells out to `--gui-summary` (a fast, no-side-effect
// JSON probe) so the UI can gate options on what's actually installed — and
// detect a missing agent (a future "lite client" that ships without the
// bundled agent), hiding the agent/full options when there's nothing to remove.

function uninstallVenvPython() {
  return getVenvPython(VENV_ROOT)
}

async function getUninstallSummary() {
  const py = uninstallVenvPython()
  const agentRoot = ACTIVE_HERMES_ROOT
  // Fast JS-side fallback used when the agent venv is gone (lite client) or the
  // probe fails — the renderer still needs *something* to render options from.
  const fallback = () => ({
    hermes_home: HERMES_HOME,
    agent_installed: isHermesSourceRoot(agentRoot) && fileExists(py),
    gui_installed: true,
    source_built_artifacts: [],
    packaged_app_paths: [],
    userdata_dir: app.getPath('userData'),
    userdata_exists: true,
    platform: process.platform,
    probe: 'fallback'
  })

  if (!fileExists(py)) {
    return fallback()
  }

  return new Promise(resolve => {
    let stdout = ''
    let settled = false
    const done = value => {
      if (settled) return
      settled = true
      resolve(value)
    }
    try {
      const child = spawn(
        py,
        ['-m', 'hermes_cli.main', 'uninstall', '--gui-summary'],
        hiddenWindowsChildOptions({
          cwd: agentRoot,
          env: { ...process.env, HERMES_HOME, NO_COLOR: '1' },
          stdio: ['ignore', 'pipe', 'ignore']
        })
      )
      child.stdout.on('data', chunk => {
        stdout += chunk.toString()
      })
      child.on('error', () => done(fallback()))
      child.on('exit', code => {
        if (code !== 0) return done(fallback())
        try {
          const line = stdout.trim().split('\n').filter(Boolean).pop() || '{}'
          const parsed = JSON.parse(line)
          // The app bundle the renderer would be removing on *this* machine,
          // resolved from the running exe (the Python probe only knows the
          // standard locations, not where THIS build actually runs from).
          parsed.running_app_path = resolveRemovableAppPath(process.execPath, process.platform, process.env)
          done(parsed)
        } catch {
          done(fallback())
        }
      })
      setTimeout(() => done(fallback()), 8000)
    } catch {
      done(fallback())
    }
  })
}

async function runDesktopUninstall(mode) {
  let uninstallArgs
  try {
    uninstallArgs = uninstallArgsForMode(mode)
  } catch (error) {
    return { ok: false, error: 'invalid-mode', message: error.message }
  }

  const venvPy = uninstallVenvPython()
  if (!fileExists(venvPy)) {
    return {
      ok: false,
      error: 'agent-missing',
      message: `Can't run the uninstaller: no Hermes agent venv at ${VENV_ROOT}.`
    }
  }

  // Interpreter choice (Finding 3): lite/full rmtree the venv that holds the
  // running python.exe. On Windows a running .exe is mandatory-locked, so the
  // rmtree must NOT be driven by the venv's own interpreter — use a system
  // Python with PYTHONPATH=<agentRoot> so `import hermes_cli` resolves from
  // source while the venv is torn down. gui-only doesn't touch the venv, so the
  // venv python is fine there. If no system Python exists (the Windows edge
  // case), fall back to the venv python — gui-only is unaffected; lite/full may
  // leave venv remnants the user can delete, which we log.
  let py = venvPy
  let pythonPath = null
  if (modeRemovesAgent(mode)) {
    const sysPy = findSystemPython()
    if (sysPy) {
      py = sysPy
      pythonPath = ACTIVE_HERMES_ROOT
    } else if (IS_WINDOWS) {
      rememberLog(
        '[uninstall] no system Python found for lite/full on Windows; falling back ' +
          'to the venv python — venv files locked by the running interpreter may ' +
          'remain and need manual deletion.'
      )
    }
  }

  const appPath = resolveRemovableAppPath(process.execPath, process.platform, process.env)
  const removeBundle = shouldRemoveAppBundle(IS_PACKAGED, appPath) ? appPath : null

  // CRITICAL (Windows): tear down every backend the desktop owns and wait for
  // the venv shim to unlock BEFORE the cleanup script runs. lite/full delete
  // the venv, and even gui-only removes the install tree's GUI artifacts — a
  // live backend grandchild (gateway / pty / REPL) holding a mandatory file
  // lock would make the script's rmdir half-fail (#37532 for the update path).
  // Reuses the incident-hardened update teardown; no-op on macOS/Linux.
  try {
    await releaseBackendLock(ACTIVE_HERMES_ROOT, 'uninstall')
  } catch (error) {
    rememberLog(`[uninstall] backend teardown errored (continuing): ${error.message}`)
  }

  const scriptArgs = {
    desktopPid: process.pid,
    pythonExe: py,
    pythonPath,
    agentRoot: ACTIVE_HERMES_ROOT,
    uninstallArgs,
    appPath: removeBundle,
    hermesHome: HERMES_HOME
  }

  let scriptPath
  let runner
  let runnerArgs
  try {
    if (IS_WINDOWS) {
      scriptPath = path.join(app.getPath('temp'), `hermes-uninstall-${Date.now()}.cmd`)
      fs.writeFileSync(scriptPath, buildWindowsCleanupScript(scriptArgs))
      runner = process.env.ComSpec || 'cmd.exe'
      runnerArgs = ['/c', scriptPath]
    } else {
      scriptPath = path.join(app.getPath('temp'), `hermes-uninstall-${Date.now()}.sh`)
      fs.writeFileSync(scriptPath, buildPosixCleanupScript(scriptArgs), { mode: 0o755 })
      runner = '/bin/bash'
      runnerArgs = [scriptPath]
    }
  } catch (error) {
    return { ok: false, error: 'script-write-failed', message: error.message }
  }

  try {
    const child = spawn(runner, runnerArgs, {
      detached: true,
      stdio: 'ignore',
      windowsHide: true
    })
    child.unref()
  } catch (error) {
    return { ok: false, error: 'spawn-failed', message: error.message }
  }

  rememberLog(
    `[uninstall] launched detached cleanup (${mode}): ${scriptPath} ` +
      `(removesAgent=${modeRemovesAgent(mode)} removesUserData=${modeRemovesUserData(mode)} bundle=${removeBundle || 'none'})`
  )

  // Give the renderer a beat to show its "uninstalling…" state, then quit so
  // the venv python shim + app bundle unlock and the cleanup script can run.
  setTimeout(() => app.quit(), 800)
  return { ok: true, mode, willRemoveAppBundle: Boolean(removeBundle), scriptPath }
}

ipcMain.handle('hermes:uninstall:summary', async () => getUninstallSummary())
ipcMain.handle('hermes:uninstall:run', async (_event, payload) => {
  const mode = payload && typeof payload === 'object' ? payload.mode : payload
  return runDesktopUninstall(String(mode || ''))
})

// Download a VS Code Marketplace extension and return the raw color-theme JSON
// it contributes. No theme code is executed — we only read JSON from the .vsix.
ipcMain.handle('hermes:vscode-theme:fetch', async (_event, id) => fetchMarketplaceThemes(String(id || '')))

// Search the Marketplace for color-theme extensions (empty query = top installs).
ipcMain.handle('hermes:vscode-theme:search', async (_event, query) => searchMarketplaceThemes(String(query || ''), 20))

// ---------------------------------------------------------------------------
// hermes:// deep links (e.g. hermes://blueprint/morning-brief?time=08:00).
// A docs/dashboard "Send to App" button opens this URL; we route it into the
// running app's chat composer. Three delivery paths: macOS 'open-url',
// Win/Linux running-app 'second-instance' (argv), Win/Linux cold-start argv.
// ---------------------------------------------------------------------------
const HERMES_PROTOCOL = 'apexnodes'
let _pendingDeepLink = null
let _rendererReadyForDeepLink = false

function _extractDeepLink(argv) {
  if (!Array.isArray(argv)) return null
  return argv.find(a => typeof a === 'string' && a.startsWith(`${HERMES_PROTOCOL}://`)) || null
}

function handleDeepLink(url) {
  if (!url || typeof url !== 'string') return
  let parsed
  try {
    parsed = new URL(url)
  } catch {
    rememberLog(`[deeplink] ignoring malformed url: ${url}`)
    return
  }
  // hermes://blueprint/<key>?slot=val  -> host="blueprint", path="/<key>"
  const kind = parsed.hostname || ''
  const name = decodeURIComponent((parsed.pathname || '').replace(/^\//, ''))
  const params = {}
  parsed.searchParams.forEach((v, k) => {
    params[k] = v
  })
  const payload = { kind, name, params }

  if (!_rendererReadyForDeepLink || !mainWindow || mainWindow.isDestroyed()) {
    _pendingDeepLink = payload
    return
  }
  try {
    if (mainWindow.isMinimized()) mainWindow.restore()
    mainWindow.focus()
    mainWindow.webContents.send('hermes:deep-link', payload)
    rememberLog(`[deeplink] delivered ${kind}/${name}`)
  } catch (err) {
    rememberLog(`[deeplink] delivery failed: ${err.message}`)
  }
}

// Renderer calls this (via IPC) once it has mounted its deep-link listener, so
// a link that arrived during boot/install is flushed exactly once.
ipcMain.handle('hermes:deep-link-ready', () => {
  _rendererReadyForDeepLink = true
  if (_pendingDeepLink) {
    const queued = _pendingDeepLink
    _pendingDeepLink = null
    handleDeepLink(
      `${HERMES_PROTOCOL}://${queued.kind}/${encodeURIComponent(queued.name)}` +
        (Object.keys(queued.params).length ? '?' + new URLSearchParams(queued.params).toString() : '')
    )
  }
  return { ok: true }
})

function registerDeepLinkProtocol() {
  try {
    if (process.defaultApp && process.argv.length >= 2) {
      // Dev: register with the electron exec path + entry script so the OS can
      // relaunch us with the URL.
      app.setAsDefaultProtocolClient(HERMES_PROTOCOL, process.execPath, [path.resolve(process.argv[1])])
    } else {
      app.setAsDefaultProtocolClient(HERMES_PROTOCOL)
    }
  } catch (err) {
    rememberLog(`[deeplink] protocol registration failed: ${err.message}`)
  }
}

// 壳自更新(electron-updater)装配 —— 和引擎(runtime)的 opt-in 更新是两条
// 互不相扰的通道。策略全静默:60s 后首查 + 每 6h 重查,autoDownload 下载,
// downloaded 状态推给侧栏胶囊出「重启以更新」;错误只进 desktop log。dev
// (未打包)不 require electron-updater,整体停用(IPC 面保留,renderer 免探测)。
function initShellUpdater() {
  let autoUpdater = null
  if (app.isPackaged) {
    try {
      autoUpdater = require('electron-updater').autoUpdater
    } catch {
      // Packaged builds set `files:` in package.json AND `beforeBuild` returns
      // false, so electron-builder's node_modules collector never runs and no
      // production dependency (electron-updater included) lands in the asar.
      // Workspace dedup also hoists electron-updater to the repo-root
      // node_modules, out of the app matcher's reach. We ship a minimal copy of
      // electron-updater + its full dependency closure under
      // resources/updater-deps/vendor/node_modules/ via extraResources +
      // scripts/stage-updater-deps.cjs; resolve from there when the normal
      // require() fails. This is the SAME pattern as node-pty above. Dev mode
      // never reaches this branch (hoisted resolve succeeds). Before this fix
      // the require threw "Cannot find module 'electron-updater'" and shell
      // self-update was silently disabled from 0.16.1 onward.
      try {
        const resourcesPath = process.resourcesPath
        if (resourcesPath) {
          const updaterPath = path.join(
            resourcesPath,
            'updater-deps',
            'vendor',
            'node_modules',
            'electron-updater'
          )
          autoUpdater = require(updaterPath).autoUpdater
        }
      } catch (fallbackError) {
        // 依赖缺失(异常打包)降级为停用,绝不拦启动。
        rememberLog(
          `[shell-update] electron-updater unavailable (disabled): ${fallbackError && fallbackError.message}`
        )
      }
    }
  }
  createShellUpdater({
    autoUpdater,
    ipcMain,
    isPackaged: app.isPackaged,
    log: rememberLog,
    broadcast: (channel, payload) => {
      for (const win of BrowserWindow.getAllWindows()) {
        if (!win.isDestroyed()) {
          win.webContents.send(channel, payload)
        }
      }
    }
  })
}

// Single-instance lock: deep links on a running app (Win/Linux) arrive as a
// second-instance argv. Without the lock a second `hermes://` launch spawns a
// whole new app instead of routing into the running one.
const _gotSingleInstanceLock = app.requestSingleInstanceLock()
if (!_gotSingleInstanceLock) {
  app.quit()
} else {
  app.on('second-instance', (_event, argv) => {
    const url = _extractDeepLink(argv)
    if (url) handleDeepLink(url)
    else if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore()
      mainWindow.focus()
    }
  })
}

// macOS delivers deep links via 'open-url' — register early (can fire before
// whenReady; handleDeepLink queues until the renderer is ready).
app.on('open-url', (event, url) => {
  event.preventDefault()
  handleDeepLink(url)
})

app.whenReady().then(() => {
  // hc-472: heal the runtime bundle's active link from the truth pointer and GC
  // stale versions/.tmp BEFORE any runtime child spawns (no handle held on an
  // old venv yet). No-op unless HERMES_BUNDLE_MODE is on; fully fail-soft.
  reconcileAndGcBundleRuntime()
  if (IS_MAC) {
    Menu.setApplicationMenu(buildApplicationMenu())
  } else {
    Menu.setApplicationMenu(null)
  }
  installMediaPermissions()
  registerMediaProtocol()
  registerDeepLinkProtocol()
  ensureWslWindowsFonts()
  configureSpellChecker()
  registerPowerResumeListeners()
  createWindow()

  // Platform client-config sync: non-blocking boot check (contract: every boot
  // + after every successful sign-in). Bounded at ~5s and strictly fail-soft —
  // an offline user boots exactly as before, on the cached state.
  void refreshClientConfigFromPlatform('boot')

  // Managed relay-key self-heal: non-blocking boot probe of the relay's
  // /v1/models. If the stored key was rotated out (401), auto re-provision with
  // the stored login JWT and re-sync config.yaml so the model picker's live
  // listing recovers without a manual re-login (the "list shrinks to one model
  // after a few days" bug). Gated to signed-in managed installs; strict no-op
  // for BYOK / signed-out / offline. Fire-and-forget, same as the boot config
  // sync above — the picker reads config.yaml fresh per open, so a heal that
  // lands after the gateway is up still takes effect on the next picker open.
  void selfHealManagedKeyOn401()

  // 壳自更新:首查本身就延迟 60s(shell-updater.cjs),不和启动高峰抢资源。
  initShellUpdater()

  // Win/Linux cold start: the launching hermes:// URL is in our own argv.
  const _coldStartLink = _extractDeepLink(process.argv)
  if (_coldStartLink) handleDeepLink(_coldStartLink)

  app.on('activate', () => {
    // Recreate the primary window if it's gone. Guard on mainWindow directly
    // (not just total window count) so a dock click still restores the main
    // window when only secondary session windows remain open.
    if (!mainWindow || mainWindow.isDestroyed()) {
      createWindow()
    } else {
      focusWindow(mainWindow)
    }
  })
})

// Seed Chromium's spellchecker with the system locale (falling back to en-US).
// On macOS Electron uses the native spellchecker which ignores this list, but
// on Windows/Linux Chromium downloads Hunspell dictionaries on demand and
// won't enable any without an explicit language.
function configureSpellChecker() {
  try {
    const defaultSession = session.defaultSession

    if (!defaultSession || typeof defaultSession.setSpellCheckerLanguages !== 'function') {
      return
    }

    const available = defaultSession.availableSpellCheckerLanguages || []
    const locale = (app.getLocale && app.getLocale()) || 'en-US'
    const candidates = [locale, locale.split('-')[0], 'en-US', 'en']
    const chosen = candidates.find(lang => available.includes(lang)) || 'en-US'

    defaultSession.setSpellCheckerLanguages([chosen])
  } catch (error) {
    rememberLog(`Spellchecker setup failed: ${error.message}`)
  }
}

app.on('before-quit', () => {
  // Quitting mid-install should stop the installer, not orphan it.
  if (bootstrapAbortController) {
    try {
      bootstrapAbortController.abort()
    } catch {
      void 0
    }
  }

  if (desktopLogFlushTimer) {
    clearTimeout(desktopLogFlushTimer)
    desktopLogFlushTimer = null
  }
  flushDesktopLogBufferSync()
  closePreviewWatchers()

  // Kill open PTYs before environment teardown to avoid the node-pty#904
  // ThreadSafeFunction SIGABRT race.
  for (const id of [...terminalSessions.keys()]) {
    disposeTerminalSession(id)
  }

  if (hermesProcess && !hermesProcess.killed) {
    hermesProcess.kill('SIGTERM')
  }
  stopAllPoolBackends()
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})
