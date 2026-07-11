type Method = string
import type { ExecFileSyncOptionsWithStringEncoding } from 'node:child_process'
import { execFileSync, spawn } from 'node:child_process'
import crypto from 'node:crypto'
import fs from 'node:fs'
import http from 'node:http'
import https from 'node:https'
import os from 'node:os'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

import {
  app,
  BrowserWindow,
  clipboard,
  dialog,
  net as electronNet,
  ipcMain,
  Menu,
  nativeImage,
  nativeTheme,
  Notification,
  powerMonitor,
  protocol,
  safeStorage,
  screen,
  session,
  shell,
  systemPreferences
} from 'electron'
import nodePty from 'node-pty'

import { dashboardFallbackArgs, sourceDeclaresServe } from './backend-command'
import { buildDesktopBackendEnv, normalizeHermesHomeRoot } from './backend-env'
import { canImportHermesCli, verifyHermesCli } from './backend-probes'
import { waitForDashboardPortAnnouncement } from './backend-ready'
import { detectRemoteDisplay, isWindowsBinaryPathInWsl, isWslEnvironment } from './bootstrap-platform'
import { runBootstrap } from './bootstrap-runner'
import {
  authModeFromStatus,
  buildGatewayWsUrl,
  buildGatewayWsUrlWithTicket,
  connectionScopeKey,
  cookiesHaveLiveSession,
  cookiesHavePrivySession,
  cookiesHaveSession,
  modeIsRemoteLike,
  normalizeRemoteBaseUrl,
  normAuthMode,
  pathWithGlobalRemoteProfile,
  profileRemoteOverride,
  resolveAuthMode,
  resolveTestWsUrl,
  tokenPreview
} from './connection-config'
import { adoptServedDashboardToken } from './dashboard-token'
import {
  buildPosixCleanupScript,
  buildWindowsCleanupScript,
  modeRemovesAgent,
  modeRemovesUserData,
  resolveRemovableAppPath,
  shouldRemoveAppBundle,
  uninstallArgsForMode
} from './desktop-uninstall'
import { installEmbedReferer } from './embed-referer'
import { readDirForIpc } from './fs-read-dir'
import { probeGatewayWebSocket } from './gateway-ws-probe'
import { scanGitRepos } from './git-repo-scan'
import {
  fileDiffVsHead,
  repoStatus,
  reviewCommit,
  reviewCommitContext,
  reviewCreatePr,
  reviewDiff,
  reviewList,
  reviewPush,
  reviewRevert,
  reviewRevParse,
  reviewShipInfo,
  reviewStage,
  reviewUnstage
} from './git-review-ops'
import { gitRootForIpc } from './git-root'
import { addWorktree, listBranches, listWorktrees, removeWorktree, switchBranch } from './git-worktree-ops'
import {
  DATA_URL_READ_MAX_BYTES,
  DEFAULT_FETCH_TIMEOUT_MS,
  encryptDesktopSecret as encryptDesktopSecretStrict,
  resolveReadableFileForIpc,
  resolveRequestedPathForIpc,
  resolveTimeoutMs,
  TEXT_PREVIEW_SOURCE_MAX_BYTES
} from './hardening'
import { createLinkTitleWindow, guardLinkTitleSession, readLinkTitleWindowTitle } from './link-title-window'
import { serializeJsonBody, setJsonRequestHeaders } from './oauth-net-request'
import {
  buildSessionWindowUrl,
  chatWindowWebPreferences,
  createSessionWindowRegistry,
  SESSION_WINDOW_MIN_HEIGHT,
  SESSION_WINDOW_MIN_WIDTH
} from './session-windows'
import { nativeOverlayWidth as computeNativeOverlayWidth, macTitleBarOverlayHeight } from './titlebar-overlay-width'
import { resolveBehindCount, shouldCountCommits } from './update-count'
import { readLiveUpdateMarker, writeUpdateMarker } from './update-marker'
import { runRebuildWithRetry } from './update-rebuild'
import {
  buildRelaunchScript,
  collectRelaunchArgs,
  collectRelaunchEnv,
  decideRelaunchOutcome,
  resolveUnpackedRelease,
  sandboxFallbackFromEnv,
  sandboxPreflight
} from './update-relaunch'
import { isOfficialSshRemote, OFFICIAL_REPO_HTTPS_URL } from './update-remote'
import { fetchMarketplaceThemes, searchMarketplaceThemes } from './vscode-marketplace'
import {
  computeWindowOptions,
  debounce,
  sanitizeWindowState,
  MIN_HEIGHT as WINDOW_MIN_HEIGHT,
  MIN_WIDTH as WINDOW_MIN_WIDTH
} from './window-state'
import { readWindowsUserEnvVar } from './windows-user-env'
import { isPackagedInstallPath as isPackagedInstallPathUnderRoots } from './workspace-cwd'
import { readWslWindowsClipboardImage } from './wsl-clipboard-image'

const USER_DATA_OVERRIDE = process.env.HERMES_DESKTOP_USER_DATA_DIR

if (USER_DATA_OVERRIDE) {
  const resolvedUserData = path.resolve(USER_DATA_OVERRIDE)
  fs.mkdirSync(resolvedUserData, { recursive: true })
  app.setPath('userData', resolvedUserData)
}

const DEV_SERVER = process.env.HERMES_DESKTOP_DEV_SERVER
const IS_PACKAGED = app.isPackaged || Boolean(process.env.HERMES_DESKTOP_IS_PACKAGED)
const IS_MAC = process.platform === 'darwin'
const IS_WINDOWS = process.platform === 'win32'
const IS_WSL = isWslEnvironment()
// Truthful macOS kernel major (Tahoe = 25). Product version lies (16 vs 26) per
// build SDK, so gate Tahoe workarounds on Darwin instead.
const DARWIN_MAJOR = IS_MAC ? Number.parseInt(os.release(), 10) || 0 : 0
const APP_ROOT = app.getAppPath()

// Preload must be plain JS — Electron's sandbox can't run .ts, and tsx's
// ESM loader is broken on Electron 40's Node (ERR_INVALID_RETURN_PROPERTY_VALUE).
// Dev (`npm run dev`) and prod both load the esbuild output from dist/.
const PRELOAD_PATH = path.join(APP_ROOT, 'dist', 'electron-preload.js')

function hiddenWindowsChildOptions(options: any = {}): ExecFileSyncOptionsWithStringEncoding {
  if (!IS_WINDOWS || Object.prototype.hasOwnProperty.call(options, 'windowsHide')) {
    return options as any
  }

  return { ...options, windowsHide: true } as any
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

// WSLg: Chromium blocklists the Mesa vGPU → software compositing → typing lag.
// /dev/dxg means a real GPU is available; un-blocklist it. Skipped when a remote
// display already forced software (SSH'd-into-WSL).
if (IS_WSL && !REMOTE_DISPLAY_REASON && fs.existsSync('/dev/dxg')) {
  app.commandLine.appendSwitch('ignore-gpu-blocklist')
  app.commandLine.appendSwitch('enable-gpu-rasterization')
  app.commandLine.appendSwitch('enable-zero-copy')
  console.log('[hermes] WSL GPU passthrough (/dev/dxg) detected; enabling GPU acceleration')
}

ipcMain.handle('hermes:get-remote-display-reason', () => REMOTE_DISPLAY_REASON)

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
// Written by apps/desktop/scripts/write-build-stamp.mjs during `npm run build`
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
    } catch (e) {
      console.warn(`[hermes] install-stamp.json found at ${p} , but parsing failed with ${e}`)
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
  if (process.env.HERMES_HOME) {
    return normalizeHermesHomeRoot(process.env.HERMES_HOME)
  }

  if (USER_DATA_OVERRIDE) {
    return path.join(path.resolve(USER_DATA_OVERRIDE), 'hermes-home')
  }

  if (IS_WINDOWS) {
    // A GUI app launched from Explorer inherits the environment block captured
    // at login, so a HERMES_HOME set via `setx` AFTER login is invisible in
    // process.env even though the CLI (a fresh shell) sees it. Without this the
    // backend silently falls back to %LOCALAPPDATA%\hermes and reports "No
    // inference provider configured" despite a valid configured home (#45471).
    // Consult the live User-scoped registry value before the default below.
    const fromRegistry = readWindowsUserEnvVar('HERMES_HOME')

    if (fromRegistry) {
      return normalizeHermesHomeRoot(fromRegistry)
    }
  }

  if (IS_WINDOWS && process.env.LOCALAPPDATA) {
    const localappdata = path.join(process.env.LOCALAPPDATA, 'hermes')
    const legacy = path.join(app.getPath('home'), '.hermes')

    // Migrate transparently to LOCALAPPDATA, but honour an existing legacy
    // ~/.hermes setup (no LOCALAPPDATA install yet) so users don't lose state.
    if (!directoryExists(localappdata) && directoryExists(legacy)) {
      return legacy
    }

    return localappdata
  }

  return path.join(app.getPath('home'), '.hermes')
}

const HERMES_HOME = resolveHermesHome()

function hermesManagedNodePathEntries() {
  // NOTE: keep this ordering in sync with iter_hermes_node_dirs() in
  // hermes_constants.py — this Node main process cannot import the Python
  // module, so the platform-ordering rule is mirrored here.
  const root = path.join(HERMES_HOME, 'node')
  const bin = path.join(root, 'bin')
  const entries = IS_WINDOWS ? [root, bin] : [bin, root]

  return entries.filter(directoryExists)
}

function pathWithHermesManagedNode(...entries) {
  return [...hermesManagedNodePathEntries(), ...entries, process.env.PATH].filter(Boolean).join(path.delimiter)
}

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

const DESKTOP_CONNECTION_CONFIG_PATH = path.join(app.getPath('userData'), 'connection.json')
const DESKTOP_UPDATE_CONFIG_PATH = path.join(app.getPath('userData'), 'updates.json')
const DESKTOP_WINDOW_STATE_PATH = path.join(app.getPath('userData'), 'window-state.json')
// active-profile.json records which Hermes profile the desktop launches its
// local backend as. When set, startHermes() passes `hermes --profile <name>
// dashboard …`, which deterministically pins HERMES_HOME (see
// _apply_profile_override in hermes_cli/main.py) and bypasses the sticky
// ~/.hermes/active_profile file. Unset (null) preserves the legacy behavior:
// no --profile flag, so the backend honors active_profile / default.
const DESKTOP_PROFILE_CONFIG_PATH = path.join(app.getPath('userData'), 'active-profile.json')
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

  if (!Number.isFinite(raw) || raw <= 0) {
    return 650
  }

  return Math.max(120, raw)
})()

const APP_NAME = process.env.HERMES_DESKTOP_APP_NAME || 'Hermes'
const TITLEBAR_HEIGHT = 34
const MACOS_TRAFFIC_LIGHTS_HEIGHT = 14

const WINDOW_BUTTON_POSITION = {
  x: 24,
  y: TITLEBAR_HEIGHT / 2 - MACOS_TRAFFIC_LIGHTS_HEIGHT / 2
}

// Right-edge window-control reservation lives in titlebar-overlay-width.ts
// (pure + unit-testable); computeNativeOverlayWidth() applies it per platform.
// It's only the pre-layout fallback — the renderer measures the exact overlay
// width live via the Window Controls Overlay API.
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

// Transparent WCO — renderer chrome shows through. rgba(0,0,0,0) can fall back
// to GetFrameColor() on some Electron builds; rgba(1,0,0,0) is the escape hatch.
const TITLEBAR_OVERLAY_COLOR = 'rgba(1, 0, 0, 0)'

function getTitleBarOverlayOptions() {
  if (IS_MAC) {
    // Tahoe (Darwin 25+) misplaces the traffic lights when the overlay has a
    // nonzero height (electron#49183); 0 there keeps them at the configured
    // inset. See macTitleBarOverlayHeight.
    return { height: macTitleBarOverlayHeight({ darwinMajor: DARWIN_MAJOR, titlebarHeight: TITLEBAR_HEIGHT }) }
  }

  // WSLg paints WCO via the RDP host's own min/max/close, so requesting
  // an Electron overlay there just leaves a dead gap. Plain Linux (KDE,
  // GNOME) can use the native overlay — let it through.
  if (!IS_WINDOWS && IS_WSL) {
    return false
  }

  return {
    color: TITLEBAR_OVERLAY_COLOR,
    height: TITLEBAR_HEIGHT,
    symbolColor:
      rendererTitleBarTheme && isHexColor(rendererTitleBarTheme.foreground)
        ? rendererTitleBarTheme.foreground
        : nativeTheme.shouldUseDarkColors
          ? '#f7f7f7'
          : '#242424'
  }
}

// Push refreshed overlay options to a live window after a theme/appearance
// change. No-op only on plain (non-WSL) Linux, where getTitleBarOverlayOptions()
// returns false; the try/catch additionally guards builds where
// setTitleBarOverlay isn't supported.
function applyTitleBarOverlay(win) {
  const options = getTitleBarOverlayOptions()

  if (!options || typeof options !== 'object') {
    return
  }

  try {
    win?.setTitleBarOverlay?.(options)
  } catch {
    // Overlay not supported on this platform/build — leave the frameless
    // titlebar as-is.
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
  if (!buffer.length) {
    return false
  }

  let suspicious = 0

  for (const byte of buffer) {
    if (byte === 0) {
      return true
    }

    // Allow common whitespace controls: tab, LF, CR.
    if (byte < 32 && byte !== 9 && byte !== 10 && byte !== 13) {
      suspicious += 1
    }
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

// Windows toast notifications silently no-op unless an AppUserModelID is set:
// `new Notification().show()` returns without error and nothing appears. The
// AUMID must match the installed Start Menu shortcut's AUMID, which
// electron-builder derives from the build `appId` (com.nousresearch.hermes) —
// keep this string in sync with package.json `build.appId`. macOS/Linux don't
// need this, so gate it on Windows. (Fixes: desktop approval/turn notifications
// never firing on Windows.)
if (IS_WINDOWS) {
  app.setAppUserModelId('com.nousresearch.hermes')
}

// Seed the native About panel with the live Hermes version. This is refreshed
// on every open via the explicit "About" menu handler (refreshAboutPanel), so
// an in-place `hermes update` mid-session is reflected without an app restart;
// the seed here just covers the first open and any non-menu invocation path.
app.setAboutPanelOptions({
  applicationName: APP_NAME,
  applicationVersion: resolveHermesVersion(),
  copyright: 'Copyright © 2026 Nous Research'
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
// True while connection-config:apply soft-rehomes the primary — suppresses the
// backend-exit toast so an intentional kill doesn't look like a crash.
let softRehomeInProgress = false
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
// Latched non-bootstrap backend spawn failure — stops getConnection() from
// respawning hermes serve backend children in a tight loop while boot is broken.
let backendStartFailure = null
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
  if (size < DESKTOP_LOG_MAX_BYTES) {
    return []
  }
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
      if (op === 'rm') {
        fs.rmSync(src, { force: true })
      } else {
        fs.renameSync(src, dst)
      }
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
      if (op === 'rm') {
        await fs.promises.rm(src, { force: true })
      } else {
        await fs.promises.rename(src, dst)
      }
    } catch {
      // Best-effort — logging must never crash the shell.
    }
  }
}

function flushDesktopLogBufferSync() {
  if (!desktopLogBuffer) {
    return
  }
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
  if (!desktopLogBuffer) {
    return desktopLogFlushPromise
  }
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
  if (desktopLogFlushTimer) {
    return
  }
  desktopLogFlushTimer = setTimeout(() => {
    desktopLogFlushTimer = null
    void flushDesktopLogBufferAsync()
  }, DESKTOP_LOG_FLUSH_MS)
}

function rememberLog(chunk) {
  const text = String(chunk || '').trim()

  if (!text) {
    return
  }
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

  if (!raw) {
    return false
  }

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

async function openPreviewInBrowser(rawUrl) {
  const raw = String(rawUrl || '').trim()

  if (!raw) {
    return false
  }

  let parsed

  try {
    parsed = new URL(raw)
  } catch {
    return false
  }

  if (parsed.protocol === 'file:') {
    let localPath

    try {
      localPath = resolveRequestedPathForIpc(parsed.toString(), { purpose: 'Open preview in browser' })
    } catch {
      return false
    }

    await shell.openExternal(pathToFileURL(localPath).toString())

    return true
  }

  return openExternalUrl(raw)
}

function ensureWslWindowsFonts() {
  if (!IS_WSL) {
    return
  }

  const fontsDir = ['/mnt/c/Windows/Fonts', '/mnt/c/windows/fonts'].find(candidate => {
    try {
      return fs.statSync(candidate).isDirectory()
    } catch {
      return false
    }
  })

  if (!fontsDir) {
    return
  }

  try {
    const confDir = path.join(app.getPath('home'), '.config', 'fontconfig', 'conf.d')
    const confPath = path.join(confDir, '99-hermes-wsl-windows-fonts.conf')
    let existing = ''

    try {
      existing = fs.readFileSync(confPath, 'utf8')
    } catch {
      existing = ''
    }

    if (existing.includes(fontsDir)) {
      return
    }

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

  if (!Number.isFinite(numeric)) {
    return 0
  }

  return Math.max(0, Math.min(100, Math.round(numeric)))
}

function broadcastBootProgress() {
  if (!mainWindow || mainWindow.isDestroyed()) {
    return
  }
  const { webContents } = mainWindow

  if (!webContents || webContents.isDestroyed()) {
    return
  }
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

  if (!mainWindow || mainWindow.isDestroyed()) {
    return
  }
  const { webContents } = mainWindow

  if (!webContents || webContents.isDestroyed()) {
    return
  }
  webContents.send('hermes:bootstrap:event', ev)
}

function getBootstrapState() {
  return bootstrapState
}

function updateBootProgress(update, options: { allowDecrease?: boolean } = {}) {
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

// --- in-app update mutual exclusion (#50238) -------------------------------
// The Tauri updater writes HERMES_HOME/.hermes-update-in-progress for the whole
// duration of an `--update` run (see update.rs UpdateMarkerGuard). If the user
// relaunches the desktop mid-update — because the window vanished with no
// progress and looks crashed — a fresh instance must NOT spawn its own local
// backend: that backend re-locks the venv shim, the updater's straggler cleanup
// (`force_kill_other_hermes`, taskkill /IM hermes.exe) kills it, the launch
// fails with the 45s "backend didn't come up" error, and the relaunch/kill
// cycle loops. Instead the fresh instance parks until the update finishes, then
// brings the backend up itself (it is the surviving instance — the updater's
// own relaunch hits our single-instance lock and quits). Marker parsing +
// staleness self-heal live in update-marker.ts (unit-tested).

// How long we'll park the launch waiting for a live update to finish before
// giving up and starting the backend anyway (belt-and-suspenders alongside the
// marker's own age ceiling; covers a stuck-but-alive updater).
const UPDATE_WAIT_TIMEOUT_MS = 20 * 60 * 1000
const UPDATE_WAIT_POLL_MS = 1000
// How long the desktop lingers on the "updating, don't reopen" overlay after
// spawning the detached updater, before it quits to release the venv shim. The
// old 600ms was long enough to register the child process but far too short for
// the user to READ the overlay — the window just vanished, looked like a crash,
// and the user relaunched mid-update (the #50238 restart-loop trigger). A
// couple of seconds lets the message land and bridges the gap until the
// updater's own progress window appears. (#50419)
const UPDATE_HANDOFF_DWELL_MS = 2500

// Block until no live update is in progress (or we hit the wait timeout).
// Emits a boot-progress phase so the renderer shows "Update in progress…"
// rather than a frozen splash. Returns true if it parked at all.
async function waitForUpdateToFinish() {
  let marker = readLiveUpdateMarker(HERMES_HOME)

  if (!marker) {
    return false
  }

  rememberLog(`[updates] update in progress (pid=${marker.pid}); deferring backend start until it finishes`)
  const deadline = Date.now() + UPDATE_WAIT_TIMEOUT_MS

  while (marker && Date.now() < deadline) {
    await advanceBootProgress(
      'backend.update-wait',
      'An update is finishing — Hermes will start automatically when it completes…',
      12
    )
    await new Promise(r => setTimeout(r, UPDATE_WAIT_POLL_MS))
    marker = readLiveUpdateMarker(HERMES_HOME)
  }

  if (marker) {
    rememberLog('[updates] update still in progress after wait timeout; starting backend anyway')
  } else {
    rememberLog('[updates] update finished; proceeding with backend start')
  }

  return true
}

function unpackedPathFor(filePath) {
  return filePath.replace(/app\.asar(?=$|[\\/])/, 'app.asar.unpacked')
}

function findOnPath(command) {
  if (!command) {
    return null
  }

  if (path.isAbsolute(command) || command.includes(path.sep) || (IS_WINDOWS && command.includes('/'))) {
    if (!fileExists(command)) {
      return null
    }

    if (isWindowsBinaryPathInWsl(command, { isWsl: IS_WSL })) {
      return null
    }

    return command
  }

  const pathEntries = String(process.env.PATH || '')
    .split(path.delimiter)
    .filter(Boolean)

  // On Windows, try PATHEXT extensions BEFORE the bare (empty-extension) name.
  // A real command must resolve via its .exe/.cmd (Windows command-resolution
  // semantics consult PATHEXT); an extensionless file — e.g. a Git-Bash
  // shell-script shim named `hermes` — must not shadow `hermes.cmd`/`hermes.exe`.
  // The empty entry is kept LAST so callers that already include the extension
  // (py.exe, pwsh.exe, powershell.exe) still resolve.
  const extensions = IS_WINDOWS
    ? [...(process.env.PATHEXT || '.COM;.EXE;.BAT;.CMD').split(';').filter(Boolean), '']
    : ['']

  for (const entry of pathEntries) {
    for (const extension of extensions) {
      const candidate = path.join(entry, `${command}${extension}`)

      if (fileExists(candidate)) {
        return candidate
      }
    }
  }

  return null
}

function isCommandScript(command) {
  return IS_WINDOWS && /\.(cmd|bat)$/i.test(command || '')
}

function unwrapWindowsVenvHermesCommand(command, backendArgs) {
  if (!IS_WINDOWS || !command || isCommandScript(command)) {
    return null
  }

  const resolved = path.resolve(String(command))

  if (!/^hermes(?:\.exe)?$/i.test(path.basename(resolved))) {
    return null
  }

  const scriptsDir = path.dirname(resolved)

  if (path.basename(scriptsDir).toLowerCase() !== 'scripts') {
    return null
  }

  const venvRoot = path.dirname(scriptsDir)
  const python = getVenvPython(venvRoot)

  if (!fileExists(python)) {
    return null
  }

  const root = path.dirname(venvRoot)

  // Smoke-test the venv interpreter before trusting it. A venv whose update
  // died mid-`pip install` still has python.exe + hermes.exe on disk, but the
  // backend dies on its first import (e.g. ModuleNotFoundError: dotenv) before
  // the gateway ever binds. Returning it here also BYPASSED the caller's
  // `--version` probe, so Retry/"Repair install" re-resolved the same broken
  // venv forever instead of falling through to the bootstrap installer.
  // Mirror isActiveRuntimeUsable(): probe with the checkout on PYTHONPATH so a
  // healthy source-tree venv passes.
  if (
    !canImportHermesCli(python, {
      env: {
        PYTHONPATH: [...(directoryExists(root) ? [root] : []), process.env.PYTHONPATH]
          .filter(Boolean)
          .join(path.delimiter)
      }
    })
  ) {
    rememberLog(
      `Ignoring venv Hermes at ${python}: runtime import probe failed (broken/partial venv); falling through to bootstrap.`
    )

    return null
  }

  return {
    label: `existing Hermes Python at ${python}`,
    command: python,
    args: ['-m', 'hermes_cli.main', ...backendArgs],
    bootstrap: false,
    env: buildDesktopBackendEnv({
      hermesHome: HERMES_HOME,
      pythonPathEntries: [...(directoryExists(root) ? [root] : []), ...getVenvSitePackagesEntries(venvRoot)],
      venvRoot
    }),
    kind: 'python',
    // Surfaced so backendSupportsServe() can read this runtime's source for the
    // `serve` capability check instead of falling back to a heavyweight probe.
    root,
    shell: false
  }
}

// Does the resolved runtime understand the `serve` subcommand? The desktop
// spawns `hermes serve`; runtimes older than serve only have `dashboard`. We
// detect support so getBackendArgsForRuntime() can route old runtimes through
// the legacy `dashboard --no-open` form instead of crashing on an unknown
// subcommand (would brick every user mid-upgrade — #54568 follow-up).
//
// Fast path: read the runtime's own dashboard.py (instant, covers managed
// installs, dev checkouts, and the Windows venv). Fallback: probe the CLI once
// (covers a bare `hermes` resolved from PATH with no known source root). Result
// is cached per resolved runtime so we probe at most once per backend.
const _serveSupportCache = new Map()

function backendSupportsServe(backend) {
  if (!backend || !backend.command) {
    return true
  }
  const key = `${backend.command}::${backend.root || ''}`

  if (_serveSupportCache.has(key)) {
    return _serveSupportCache.get(key)
  }

  let supported = null

  if (backend.root) {
    try {
      const src = fs.readFileSync(path.join(backend.root, 'hermes_cli', 'subcommands', 'dashboard.py'), 'utf8')
      supported = sourceDeclaresServe(src)
    } catch {
      supported = null // source unreadable — fall through to the probe
    }
  }

  if (supported === null) {
    try {
      const prefix = backend.args && backend.args[0] === '-m' ? backend.args.slice(0, 2) : []
      execFileSync(backend.command, [...prefix, 'serve', '--help'], {
        cwd: backend.root || undefined,
        env: { ...process.env, HERMES_HOME, ...(backend.env || {}) },
        timeout: 15000,
        stdio: 'ignore',
        windowsHide: true
      })
      supported = true
    } catch {
      supported = false
    }
  }

  _serveSupportCache.set(key, supported)
  rememberLog(
    `[backend] \`serve\` ${supported ? 'supported' : 'unsupported → routing via legacy `dashboard`'} for ${backend.label || key}`
  )

  return supported
}

// Given a resolved backend whose args target `serve`, return the args the
// runtime actually understands: unchanged when `serve` is supported, or
// rewritten to `dashboard --no-open` for older runtimes.
function getBackendArgsForRuntime(backend) {
  return backendSupportsServe(backend) ? backend.args : dashboardFallbackArgs(backend.args)
}

function normalizeExecutablePathForCompare(commandPath) {
  if (!commandPath) {
    return null
  }

  let resolved = path.resolve(String(commandPath))

  try {
    resolved = fs.realpathSync.native ? fs.realpathSync.native(resolved) : fs.realpathSync(resolved)
  } catch {
    // Fallback to path.resolve() above.
  }

  return IS_WINDOWS ? resolved.toLowerCase() : resolved
}

function looksLikeDesktopAppBinary(commandPath) {
  if (!IS_WINDOWS || !commandPath) {
    return false
  }

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

  if (override && fileExists(override)) {
    return override
  }

  const relativePaths = IS_WINDOWS
    ? [path.join('.venv', 'Scripts', 'python.exe'), path.join('venv', 'Scripts', 'python.exe')]
    : [path.join('.venv', 'bin', 'python'), path.join('venv', 'bin', 'python')]

  for (const relativePath of relativePaths) {
    const candidate = path.join(root, relativePath)

    if (fileExists(candidate)) {
      return candidate
    }
  }

  return findSystemPython()
}

function findSystemPython() {
  if (!IS_WINDOWS) {
    // POSIX systems: PATH lookup is safe.
    for (const command of ['python3', 'python']) {
      const candidate = findOnPath(command)

      if (candidate) {
        return candidate
      }
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

          if (fileExists(pythonExe)) {
            return pythonExe
          }
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

    if (fileExists(systemWide)) {
      return systemWide
    }

    if (localAppData) {
      const perUser = path.join(localAppData, 'Programs', 'Python', `Python${versionDir}`, 'python.exe')

      if (fileExists(perUser)) {
        return perUser
      }
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

        if (candidate && fileExists(candidate)) {
          return candidate
        }
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

  // install.ps1 drops PortableGit at %LOCALAPPDATA%\hermes\git\... — checked
  // first so users who installed via install.ps1 are detected before we
  // start probing system-wide locations.
  const localAppData = process.env.LOCALAPPDATA || ''
  const candidates = []

  if (localAppData) {
    candidates.push(path.join(localAppData, 'hermes', 'git', 'bin', 'bash.exe'))
    candidates.push(path.join(localAppData, 'hermes', 'git', 'usr', 'bin', 'bash.exe'))
  }

  // Standard Git for Windows install locations.
  candidates.push(path.join(process.env['ProgramFiles'] || 'C:\\Program Files', 'Git', 'bin', 'bash.exe'))
  candidates.push(path.join(process.env['ProgramFiles(x86)'] || 'C:\\Program Files (x86)', 'Git', 'bin', 'bash.exe'))

  if (localAppData) {
    candidates.push(path.join(localAppData, 'Programs', 'Git', 'bin', 'bash.exe'))
  }

  for (const candidate of candidates) {
    if (fileExists(candidate)) {
      return candidate
    }
  }

  // Last resort — bash on PATH (covers WSL bash, MSYS2, custom installs).
  // On WSL hosts findOnPath itself filters out Windows-binary paths via
  // isWindowsBinaryPathInWsl, so we won't hand back a wsl.exe shim either.
  return findOnPath('bash')
}

function getVenvPython(venvRoot) {
  return path.join(venvRoot, IS_WINDOWS ? path.join('Scripts', 'python.exe') : path.join('bin', 'python'))
}

// Windows console-window flashes are governed by the *parent's* console, not by
// each child spawn. A GUI-subsystem parent (pythonw.exe) has no console, so every
// console-subsystem child it spawns (git, gh, cmd, ...) must allocate its own —
// which flashes a window. A console-subsystem parent (python.exe) instead owns a
// single console that all of its children inherit, so none of them flash.
//
// Note this change adds no new creationflag: the backend spawn is ALREADY wrapped
// in hiddenWindowsChildOptions() (windowsHide: true), but that setting is INERT
// against pythonw.exe — a GUI-subsystem process has no console for it to act on.
// Switching the backend to the venv's console python.exe is what makes the
// existing wrapper load-bearing: with windowsHide the process comes up owning a
// *windowless* console (verified at runtime — it has an attachable console whose
// window handle is NULL), and its children inherit that one windowless console
// instead of each allocating a visible one.
//
// This makes "no flashing windows" a property of the one backend launch rather
// than a flag that has to be remembered at every descendant spawn site. Restoring
// console python also restores stdout, so the backend announces its port on the
// normal HERMES_DASHBOARD_READY stdout line and no ready-file side channel is
// needed.

function getVenvSitePackagesEntries(venvRoot) {
  const entries = []

  if (!venvRoot) {
    return entries
  }

  if (IS_WINDOWS) {
    const sitePackages = path.join(venvRoot, 'Lib', 'site-packages')

    if (directoryExists(sitePackages)) {
      entries.push(sitePackages)
    }

    return entries
  }

  const version = (() => {
    try {
      const cfg = fs.readFileSync(path.join(venvRoot, 'pyvenv.cfg'), 'utf8')
      const match = cfg.match(/^version_info\s*=\s*(\d+\.\d+)/im)

      return match ? match[1].trim() : null
    } catch {
      return null
    }
  })()

  if (version) {
    const sitePackages = path.join(venvRoot, 'lib', `python${version}`, 'site-packages')

    if (directoryExists(sitePackages)) {
      entries.push(sitePackages)
    }
  }

  return entries
}

function makeDashboardReadyFile() {
  const dir = path.join(app.getPath('userData'), 'backend-ready')
  fs.mkdirSync(dir, { recursive: true })

  return path.join(dir, `dashboard-${process.pid}-${Date.now()}-${crypto.randomBytes(6).toString('hex')}.json`)
}

// resolveGitBinary — locate git.exe on Windows. A fresh installer-driven
// install only has PortableGit under %LOCALAPPDATA%\hermes\git (never on
// PATH), so a bare spawn('git') ENOENTs and self-update checks fail with
// "Couldn't check for updates". Mirror findGitBash: PortableGit first, then
// standard Git-for-Windows locations, then PATH. Cached after first probe.
let _gitBinaryCache = null

function resolveGitBinary() {
  if (_gitBinaryCache) {
    return _gitBinaryCache
  }

  if (!IS_WINDOWS) {
    _gitBinaryCache = findOnPath('git') || 'git'

    return _gitBinaryCache
  }

  const localAppData = process.env.LOCALAPPDATA || ''
  const candidates = []

  if (localAppData) {
    candidates.push(path.join(localAppData, 'hermes', 'git', 'cmd', 'git.exe'))
    candidates.push(path.join(localAppData, 'hermes', 'git', 'bin', 'git.exe'))
  }

  candidates.push(path.join(process.env['ProgramFiles'] || 'C:\\Program Files', 'Git', 'cmd', 'git.exe'))
  candidates.push(path.join(process.env['ProgramFiles(x86)'] || 'C:\\Program Files (x86)', 'Git', 'cmd', 'git.exe'))

  if (localAppData) {
    candidates.push(path.join(localAppData, 'Programs', 'Git', 'cmd', 'git.exe'))
  }

  _gitBinaryCache = candidates.find(fileExists) || findOnPath('git') || 'git'

  return _gitBinaryCache
}

// resolveGhBinary — locate the GitHub CLI. GUI-launched apps get a minimal PATH
// that omits Homebrew (/opt/homebrew/bin, /usr/local/bin) where `gh` usually
// lives, so a bare spawn('gh') ENOENTs even though `gh` works in the user's
// terminal. Check the common install locations first, then PATH. Cached.
let _ghBinaryCache = null

function resolveGhBinary() {
  if (_ghBinaryCache) {
    return _ghBinaryCache
  }

  const candidates = []

  if (IS_WINDOWS) {
    candidates.push(path.join(process.env['ProgramFiles'] || 'C:\\Program Files', 'GitHub CLI', 'gh.exe'))

    if (process.env.LOCALAPPDATA) {
      candidates.push(path.join(process.env.LOCALAPPDATA, 'Microsoft', 'WinGet', 'Links', 'gh.exe'))
    }
  } else {
    const home = app.getPath('home')
    candidates.push('/opt/homebrew/bin/gh', '/usr/local/bin/gh', '/usr/bin/gh', path.join(home, '.local', 'bin', 'gh'))
  }

  _ghBinaryCache = candidates.find(fileExists) || findOnPath('gh') || 'gh'

  return _ghBinaryCache
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
function writeFileAtomic(targetPath, data, encoding?: BufferEncoding) {
  const tmp = targetPath + '.tmp'
  fs.writeFileSync(tmp, data, encoding)
  fs.renameSync(tmp, targetPath)
}

function writeDesktopUpdateConfig(config) {
  fs.mkdirSync(path.dirname(DESKTOP_UPDATE_CONFIG_PATH), { recursive: true })
  writeFileAtomic(DESKTOP_UPDATE_CONFIG_PATH, JSON.stringify(config, null, 2))
}

// ─── Main-window geometry persistence (window-state.json) ──────────────────

function readWindowState() {
  try {
    return sanitizeWindowState(JSON.parse(fs.readFileSync(DESKTOP_WINDOW_STATE_PATH, 'utf8')))
  } catch {
    return null
  }
}

// Persist the window's restored (non-maximized) bounds plus its maximized flag.
// getNormalBounds() keeps the pre-maximize size, so un-maximizing next session
// lands back where the user actually sized the window.
function persistWindowState() {
  if (!mainWindow || mainWindow.isDestroyed() || mainWindow.isMinimized()) {
    return
  }

  try {
    const { x, y, width, height } = mainWindow.getNormalBounds()
    fs.mkdirSync(path.dirname(DESKTOP_WINDOW_STATE_PATH), { recursive: true })
    writeFileAtomic(
      DESKTOP_WINDOW_STATE_PATH,
      JSON.stringify({ x, y, width, height, isMaximized: mainWindow.isMaximized() }, null, 2)
    )
  } catch (err) {
    rememberLog(`[window-state] persist failed: ${err?.message || err}`)
  }
}

// resized/moved fire many times mid-drag on Linux; debounce to one write.
const schedulePersistWindowState = debounce(persistWindowState, 250)

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

function runGit(args, options: any = {}): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn(
      resolveGitBinary(),
      IS_WINDOWS ? ['-c', 'windows.appendAtomically=false', ...args] : args,
      hiddenWindowsChildOptions({
        cwd: options.cwd,
        env: { ...process.env, ...((options.env || {}) as any), GIT_TERMINAL_PROMPT: '0' },
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

  const [currentSha, targetSha, dirtyStr, currentBranch, shallowStr, mergeBaseStr] = await Promise.all([
    git(['rev-parse', 'HEAD']),
    git(['rev-parse', `origin/${branch}`]),
    git(['status', '--porcelain']),
    git(['rev-parse', '--abbrev-ref', 'HEAD']),
    git(['rev-parse', '--is-shallow-repository']),
    // merge-base exits non-zero with empty stdout when HEAD shares no common
    // ancestor with the freshly fetched tip — exactly the shallow-clone case.
    git(['merge-base', 'HEAD', `origin/${branch}`])
  ])

  const isShallow = shallowStr === 'true'
  const hasMergeBase = Boolean(mergeBaseStr)

  // Only enumerate the commit count when it is meaningful. On a shallow checkout
  // with no merge-base, `rev-list --count` walks the entire remote ancestry
  // (thousands of commits, see #51922) and resolveBehindCount discards the
  // result anyway in favour of a SHA compare — so skip the expensive query.
  const countStr = shouldCountCommits({ isShallow, hasMergeBase })
    ? await git(['rev-list', `HEAD..origin/${branch}`, '--count'])
    : ''

  const behind = resolveBehindCount({
    countStr,
    currentSha,
    targetSha,
    isShallow,
    hasMergeBase
  })

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

// Set to true when the desktop is about to quit so a detached swap/install/
// uninstall script can take over. On macOS, app.quit() closes windows but
// window-all-closed deliberately keeps the process alive (standard Electron
// macOS convention). Without this flag the process never exits — the detached
// hand-off script spins its PID-wait for the full timeout, and the user sees a
// blank app with no window (and an uninstall that appears to do nothing). When
// set, window-all-closed calls app.quit() on every platform so the process
// actually dies and the hand-off script can proceed immediately.
let isQuittingForHandoff = false

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
  if (!IS_MAC || !updater) {
    return
  }

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
  if (!IS_WINDOWS) {
    return false
  }
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
  if (!IS_WINDOWS) {
    return
  }

  if (!Number.isInteger(pid) || pid <= 0) {
    return
  }

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
  if (!IS_WINDOWS) {
    return { unlocked: true }
  }

  // Collect every backend PID the desktop owns: primary window backend + pool.
  const pids = []

  if (hermesProcess && Number.isInteger(hermesProcess.pid)) {
    pids.push(hermesProcess.pid)
  }

  for (const entry of backendPool.values()) {
    if (entry.process && Number.isInteger(entry.process.pid)) {
      pids.push(entry.process.pid)
    }
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

  for (const pid of pids) {
    forceKillProcessTree(pid)
  }

  const shim = venvHermesShimPath(updateRoot)
  const deadlineMs = Date.now() + 15000

  while (Date.now() < deadlineMs) {
    if (!isShimLocked(shim)) {
      rememberLog(`[${tag}] venv shim unlocked; safe to proceed`)

      return { unlocked: true }
    }

    // A supervised backend can respawn between kill and check (grandchildren,
    // pool entries registered mid-teardown). Re-collect and re-kill each pass
    // instead of trusting the initial sweep.
    const stragglers = []

    if (hermesProcess && Number.isInteger(hermesProcess.pid)) {
      stragglers.push(hermesProcess.pid)
    }

    for (const entry of backendPool.values()) {
      if (entry.process && Number.isInteger(entry.process.pid)) {
        stragglers.push(entry.process.pid)
      }
    }

    for (const pid of stragglers) {
      forceKillProcessTree(pid)
    }
    await new Promise(r => setTimeout(r, 300))
  }

  // Do NOT proceed past a held lock: handing off to the updater while another
  // process (a second desktop window, a user terminal, an unkillable child)
  // still maps the venv's files guarantees a half-updated venv — the updater's
  // dependency sync dies on access-denied partway through uninstalls, leaving
  // imports broken (the July 2026 brotlicffi/_sodium.pyd incidents). Failing
  // the update loudly and keeping the app running is strictly better than a
  // bricked install that needs manual venv surgery.
  rememberLog(
    `[${tag}] venv shim still locked after 15s; aborting hand-off (something outside this app holds the venv)`
  )

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

          if (branch !== 'main') {
            command = `hermes update --branch ${branch}`
          }
        }
      } catch {
        // Best-effort: fall back to bare `hermes update` if branch detection fails.
      }

      rememberLog(`[updates] no staged updater; surfacing manual \`${command}\` for CLI install at ${updateRoot}`)
      emitUpdateProgress({ stage: 'manual', message: command, percent: null })

      return { ok: true, manual: true, command, hermesRoot: updateRoot }
    }

    emitUpdateProgress({
      stage: 'restart',
      message:
        'Updating Hermes — this window will close and the updater will open. Don’t reopen Hermes yourself; it restarts automatically when the update finishes.',
      percent: 100
    })
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
    const lock = await releaseBackendLockForUpdate(updateRoot)

    if (!lock.unlocked) {
      // Something OUTSIDE this app holds the venv (a second window, a user
      // terminal running hermes, an unkillable child). Handing off anyway
      // guarantees a half-updated venv — abort loudly instead and let the
      // user close the holder and retry. Restart our own backend so the app
      // keeps working after the failed attempt.
      const message =
        'Update aborted: another process is holding the Hermes install open ' +
        '(a second Hermes window or a terminal running hermes?). Close it and retry.'

      emitUpdateProgress({ stage: 'error', message, percent: null })
      startHermes().catch(() => {})

      return { ok: false, error: message }
    }

    // Detached so the updater outlives this process — it needs us GONE before
    // `hermes update` will run (the venv shim is locked while we live).
    const child = spawn(updater, updaterArgs, {
      cwd: HERMES_HOME,
      env: {
        ...process.env,
        HERMES_HOME,
        PATH: pathWithHermesManagedNode(venvBin)
      },
      detached: true,
      stdio: 'ignore',
      windowsHide: false
    })

    child.unref()

    // Write the update-in-progress marker IMMEDIATELY — before the 2.5s
    // quit dwell. The Tauri updater won't write its own marker for several
    // seconds (window init + manifest), and during that gap our renderer
    // can reconnect and spawn a fresh backend that re-locks .pyd files in
    // the venv. By writing the marker ourselves the renderer's
    // waitForUpdateToFinish() gate sees a live update and parks instead.
    // The updater overwrites this with its own PID later; same format.
    if (Number.isInteger(child.pid)) {
      writeUpdateMarker(HERMES_HOME, child.pid)
    }

    rememberLog(`[updates] launched updater: ${updater} ${updaterArgs.join(' ')}; exiting desktop to release venv shim`)

    // Linger on the "updating — don't reopen" overlay long enough for the user
    // to actually read it (and to bridge the gap until the updater's own window
    // appears), THEN quit to release the venv shim. The updater rebuilds and
    // relaunches us when it's done. (#50419 — a 600ms quit looked like a crash
    // and lured users into the #50238 relaunch loop.)
    isQuittingForHandoff = true
    setTimeout(() => {
      app.quit()
    }, UPDATE_HANDOFF_DWELL_MS)

    return { ok: true, handedOff: true, updater }
  } finally {
    updateInFlight = false
  }
}

async function handOffWindowsBootstrapRecovery(reason) {
  if (!IS_WINDOWS || !IS_PACKAGED) {
    return false
  }

  const updater = resolveUpdaterBinary()

  if (!updater) {
    return false
  }

  const updateRoot = resolveUpdateRoot()
  const { branch: configuredBranch } = readDesktopUpdateConfig()

  const branch = directoryExists(path.join(updateRoot, '.git'))
    ? await resolveHealedBranch(updateRoot, configuredBranch || DEFAULT_UPDATE_BRANCH)
    : configuredBranch || DEFAULT_UPDATE_BRANCH

  const venvBin = path.join(updateRoot, 'venv', IS_WINDOWS ? 'Scripts' : 'bin')
  const venvHermes = path.join(venvBin, IS_WINDOWS ? 'hermes.exe' : 'hermes')
  const venvPython = path.join(venvBin, IS_WINDOWS ? 'python.exe' : 'python')

  // Choose the gentle in-place --update when ANY real-install signal is present,
  // not just the `hermes.exe` console-script shim. That shim is generated at the
  // END of venv setup and is absent in exactly the interrupted/quarantined states
  // this recovery exists to heal — gating on it alone forced the destructive
  // --repair (full venv recreate) and drove reinstall loops. The venv interpreter
  // and the bootstrap-complete marker are present earlier and are better signals.
  const haveRealInstall =
    fileExists(venvPython) || fileExists(venvHermes) || fileExists(path.join(updateRoot, '.hermes-bootstrap-complete'))

  const updaterArgs = haveRealInstall ? ['--update', '--branch', branch] : ['--repair', '--branch', branch]

  await releaseBackendLockForUpdate(updateRoot)

  const child = spawn(updater, updaterArgs, {
    cwd: HERMES_HOME,
    env: {
      ...process.env,
      HERMES_HOME,
      PATH: pathWithHermesManagedNode(venvBin)
    },
    detached: true,
    stdio: 'ignore',
    windowsHide: false
  })

  child.unref()

  // Same marker pre-write as applyUpdates — see comment there. The recovery
  // hand-off has the same window where the renderer can respawn a backend
  // before the updater writes its own marker.
  if (Number.isInteger(child.pid)) {
    writeUpdateMarker(HERMES_HOME, child.pid)
  }

  rememberLog(
    `[bootstrap] handed off ${reason} recovery to updater: ${updater} ${updaterArgs.join(' ')}; exiting desktop to release app.asar`
  )
  // Same dwell as the in-app update hand-off (#50419): give the updater's
  // window time to appear before we vanish, so the recovery doesn't look like
  // a crash and provoke a mid-recovery relaunch.
  isQuittingForHandoff = true
  setTimeout(() => {
    app.quit()
  }, UPDATE_HANDOFF_DWELL_MS)

  return true
}

// Resolve the hermes CLI to drive an in-app update: prefer the venv shim in
// the install we're updating, fall back to `hermes` on PATH.
function resolveHermesCliBinary(updateRoot) {
  const venvHermes = path.join(updateRoot, 'venv', 'bin', 'hermes')

  if (fileExists(venvHermes)) {
    return venvHermes
  }

  return findOnPath('hermes') || null
}

// Spawn a command and stream each output line to the update progress channel.
function runStreamedUpdate(command, args, { cwd, env, stage }: any = {}) {
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

        if (trimmed) {
          emitUpdateProgress({ stage, message: trimmed, percent: null })
        }
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
  if (!IS_MAC) {
    return null
  }
  let dir = path.dirname(app.getPath('exe')) // .../Contents/MacOS

  for (let i = 0; i < 2; i++) {
    dir = path.dirname(dir)
  } // -> .../X.app

  return dir.endsWith('.app') ? dir : null
}

function shellQuote(value) {
  return `'${String(value).replace(/'/g, `'\\''`)}'`
}

// macOS/Linux in-app update: backend (`hermes update`) + OS-aware GUI rebuild
// (`hermes desktop --build-only`), then atomically swap the running .app bundle
// with the freshly built one and relaunch. Degrades to "backend updated,
// restart to load the new GUI" if the swap can't be performed.
async function applyUpdatesPosixInApp(opts: any) {
  const updateRoot = resolveUpdateRoot()
  const hermes = resolveHermesCliBinary(updateRoot)

  if (!hermes) {
    emitUpdateProgress({ stage: 'manual', message: 'hermes update', percent: null })

    return { ok: true, manual: true, command: 'hermes update', hermesRoot: updateRoot }
  }

  // Put the Hermes-managed Node and the venv on PATH so `hermes desktop`'s
  // npm build can find them on a machine with no system Node. Windows portable
  // Node lives directly under %LOCALAPPDATA%\hermes\node, not node\bin.
  const env: Record<string, string> = {
    HERMES_HOME,
    PATH: pathWithHermesManagedNode(path.join(updateRoot, 'venv', 'bin'))
  }

  // `hermes update` reaps stale `hermes serve` backends (a code update
  // leaves the running process serving old Python against the freshly-updated
  // JS bundle). But OUR backend is one of those processes, and killing it
  // mid-update produces the boot→kill→crash loop in #37532 — the desktop
  // already restarts its own backend via the rebuild+relaunch below, so the
  // reap must spare it. Hand the live backend's PID to the update process;
  // _kill_stale_dashboard_processes reads HERMES_DESKTOP_CHILD_PID and excludes
  // it while still reaping any genuinely-orphaned backends. (#37532)
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

  const updated = (await runStreamedUpdate(hermes, ['update', '--yes', ...branchArgs], {
    cwd: updateRoot,
    env,
    stage: 'update'
  })) as any

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

  // Linux in-app update terminal state (#45205). `hermes desktop --build-only`
  // rebuilds the unpacked app in place under apps/desktop/release/<plat>-unpacked.
  // We can only HONESTLY relaunch into the new GUI when the *running* binary IS
  // that rebuilt one — i.e. execPath lives under release/<plat>-unpacked. The
  // outcome is decided by three signals (see update-relaunch.ts):
  //
  //   underUnpacked + sandboxOk  → 'relaunch': detached watcher re-execs us in
  //       place (mirrors the macOS handoff). Without it the update succeeds but
  //       the app never restarts and the overlay hangs on "applying" forever.
  //   !underUnpacked             → 'guiSkew': the running shell is an AppImage/
  //       .deb/.rpm/dev/unresolved binary we did NOT replace. Claiming "loads
  //       next launch" is a lie (GUI/backend skew, #37541) — surface an
  //       explicit closeable terminal state telling the user the GUI package
  //       was NOT changed and must be updated/reinstalled.
  //   underUnpacked + !sandboxOk → 'manual': we'd be relaunching the rebuilt
  //       binary, but a fresh rebuild can leave chrome-sandbox without
  //       root:root + setuid (mode 4755) and Electron then refuses to launch
  //       ("quit and never came back"). DO NOT quit into a dead app — keep the
  //       working window and surface the closeable manual-restart state.
  if (!IS_MAC) {
    const unpackedDir = resolveUnpackedRelease(process.execPath, updateRoot, process.platform)
    const underUnpacked = unpackedDir !== null

    const preflight = underUnpacked
      ? sandboxPreflight(unpackedDir, p => fs.statSync(p))
      : { ok: false, reason: 'not-under-unpacked', path: null }

    const sandboxFallback = sandboxFallbackFromEnv(process.env, process.argv.slice(1))
    const sandboxOk = preflight.ok || sandboxFallback

    if (underUnpacked && !preflight.ok) {
      rememberLog(
        `[updates] sandbox preflight: not launchable (${preflight.reason}) at ${preflight.path}; ` +
          `fallback=${sandboxFallback ? 'env/--no-sandbox' : 'none'}`
      )
    }

    const outcome = decideRelaunchOutcome({ underUnpacked, sandboxOk })

    if (outcome === 'relaunch') {
      emitUpdateProgress({ stage: 'restart', message: 'Restarting Hermes…', percent: 100 })
      // Preserve launch context across the re-exec: replay the original args
      // (filtered of Electron internals) and the env/cwd that define which
      // backend/profile/root this instance talks to. Without this the
      // relaunched instance comes up with default context instead of the user's.
      const relaunchArgs = collectRelaunchArgs(process.argv.slice(1))
      const relaunchEnv = collectRelaunchEnv(process.env)

      const relaunchScript = buildRelaunchScript({
        pid: process.pid,
        execPath: process.execPath,
        args: relaunchArgs,
        env: relaunchEnv,
        cwd: process.cwd()
      })

      const scriptPath = path.join(app.getPath('temp'), `hermes-desktop-update-${Date.now()}.sh`)

      try {
        fs.writeFileSync(scriptPath, relaunchScript, { mode: 0o755 })
        const child = spawn('/bin/bash', [scriptPath], { detached: true, stdio: 'ignore' })
        child.unref()
        rememberLog(
          `[updates] launched linux relaunch: ${scriptPath} -> ${process.execPath} ` +
            `(args=${relaunchArgs.length}, env=${Object.keys(relaunchEnv).length})`
        )
        isQuittingForHandoff = true
        setTimeout(() => app.quit(), UPDATE_HANDOFF_DWELL_MS)

        return { ok: true, handedOff: true }
      } catch (err) {
        rememberLog(`[updates] linux relaunch failed: ${err.message}; falling back to manual restart`)

        return {
          ok: true,
          backendUpdated: true,
          guiUpdated: false,
          manualRestart: true,
          message: 'Backend updated. Quit and reopen Hermes to load the new version.'
        }
      }
    }

    if (outcome === 'guiSkew') {
      emitUpdateProgress({
        stage: 'guiSkew',
        message:
          'Backend updated, but the desktop app package was not changed. ' +
          'Update or reinstall the Hermes desktop app to match.',
        percent: 100
      })
      rememberLog(
        `[updates] gui/backend skew: execPath ${process.execPath} not under release/*-unpacked; ` +
          'backend updated, GUI package unchanged (AppImage/.deb/.rpm/dev/unresolved)'
      )

      return { ok: true, backendUpdated: true, guiUpdated: false, guiSkew: true }
    }

    // outcome === 'manual': we're the rebuilt binary, but its sandbox helper is
    // not launchable and no fallback applies. Keep this working window alive.
    rememberLog(
      `[updates] sandbox not launchable (${preflight.reason}); skipping auto-relaunch, ` +
        'returning manual-restart so the user keeps a working window'
    )

    return {
      ok: true,
      backendUpdated: true,
      guiUpdated: false,
      manualRestart: true,
      sandboxBlocked: true,
      message:
        'Backend updated. The rebuilt app can’t relaunch automatically ' +
        '(sandbox helper needs root). Quit and reopen Hermes to finish.'
    }
  }

  const rebuiltApp = [
    path.join(updateRoot, 'apps', 'desktop', 'release', 'mac-arm64', 'Hermes.app'),
    path.join(updateRoot, 'apps', 'desktop', 'release', 'mac', 'Hermes.app')
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

  isQuittingForHandoff = true
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

// Marker-independent: is the canonical install at ACTIVE_HERMES_ROOT actually
// runnable right now? A complete CLI install (`install.sh --include-desktop`)
// or a DMG launch over a prior CLI install satisfies this WITHOUT the desktop
// ever having written the bootstrap marker -- so we must be able to recognise
// "already installed" off the filesystem alone, not just the marker.
function isActiveRuntimeUsable() {
  const venvPython = getVenvPython(VENV_ROOT)

  return (
    isHermesSourceRoot(ACTIVE_HERMES_ROOT) &&
    fileExists(venvPython) &&
    canImportHermesCli(venvPython, {
      env: {
        PYTHONPATH: [ACTIVE_HERMES_ROOT, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter)
      }
    })
  )
}

function isBootstrapComplete() {
  const marker = readBootstrapMarker()

  if (!marker || typeof marker !== 'object') {
    return false
  }

  if (marker.schemaVersion !== BOOTSTRAP_MARKER_SCHEMA_VERSION) {
    return false
  }

  if (typeof marker.pinnedCommit !== 'string' || marker.pinnedCommit.length < 7) {
    return false
  }

  // We DELIBERATELY do NOT verify that the checkout is currently at the
  // pinned commit -- users update via the in-app update path or `hermes
  // update`, which moves HEAD legitimately. The marker just attests "we
  // ran the bootstrap successfully at least once." We DO additionally require
  // a runnable venv: an interrupted or split-home install can leave the marker
  // + checkout without a venv, and trusting that spawns a dead backend
  // ("gateway offline") instead of re-running bootstrap to repair it.
  return isActiveRuntimeUsable()
}

function writeBootstrapMarker(payload) {
  fs.mkdirSync(path.dirname(BOOTSTRAP_COMPLETE_MARKER), { recursive: true })

  const merged = {
    schemaVersion: BOOTSTRAP_MARKER_SCHEMA_VERSION,
    pinnedCommit: payload.pinnedCommit || null,
    pinnedBranch: payload.pinnedBranch || null,
    completedAt: new Date().toISOString(),
    desktopVersion: app.getVersion()
  }

  writeFileAtomic(BOOTSTRAP_COMPLETE_MARKER, JSON.stringify(merged, null, 2) + '\n', 'utf8')

  return merged
}

function resolveWebDist() {
  const override = process.env.HERMES_DESKTOP_WEB_DIST

  if (override && directoryExists(path.resolve(override))) {
    return path.resolve(override)
  }

  const unpackedDist = path.join(unpackedPathFor(APP_ROOT), 'dist')

  if (directoryExists(unpackedDist)) {
    return unpackedDist
  }

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

  if (found) {
    return found
  }
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
    if (!candidate) {
      continue
    }
    const resolved = path.resolve(String(candidate))

    if (isPackagedInstallPath(resolved)) {
      continue
    }

    if (directoryExists(resolved)) {
      return resolved
    }
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

function createPythonBackend(root, label, backendArgs, options: any = {}) {
  const python = findPythonForRoot(root)

  if (!python) {
    return null
  }

  const venvRoot = path.join(root, 'venv')
  const venvPython = getVenvPython(venvRoot)
  const command = IS_WINDOWS && fileExists(venvPython) ? venvPython : python

  return {
    kind: 'python',
    label,
    command,
    args: ['-m', 'hermes_cli.main', ...backendArgs],
    env: buildDesktopBackendEnv({
      hermesHome: HERMES_HOME,
      pythonPathEntries: [root, ...getVenvSitePackagesEntries(venvRoot)],
      venvRoot
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
function createActiveBackend(backendArgs) {
  const venvPython = getVenvPython(VENV_ROOT)
  const command = fileExists(venvPython) ? venvPython : findSystemPython()

  return {
    kind: 'python',
    label: `Hermes at ${ACTIVE_HERMES_ROOT}`,
    command,
    args: ['-m', 'hermes_cli.main', ...backendArgs],
    env: buildDesktopBackendEnv({
      hermesHome: HERMES_HOME,
      pythonPathEntries: [ACTIVE_HERMES_ROOT, ...getVenvSitePackagesEntries(VENV_ROOT)],
      venvRoot: VENV_ROOT
    }),
    root: ACTIVE_HERMES_ROOT,
    bootstrap: true,
    shell: false
  }
}

function resolveHermesBackend(backendArgs) {
  // 1. Explicit override -- HERMES_DESKTOP_HERMES_ROOT points at a developer
  //    checkout. Honour it as-is (no bootstrap; the user is driving).
  const overrideRoot = process.env.HERMES_DESKTOP_HERMES_ROOT && path.resolve(process.env.HERMES_DESKTOP_HERMES_ROOT)

  if (overrideRoot && isHermesSourceRoot(overrideRoot)) {
    const backend = createPythonBackend(overrideRoot, `Hermes source at ${overrideRoot}`, backendArgs)

    if (backend) {
      return backend
    }
  }

  // 2. Development source -- when running `npm run dev` from a checkout, the
  //    cloned repo at SOURCE_REPO_ROOT takes precedence over ACTIVE and any
  //    installed `hermes` on PATH so local Python edits are actually exercised.
  //    (In dev with no checkout, SOURCE_REPO_ROOT won't pass isHermesSourceRoot.)
  if (!IS_PACKAGED && isHermesSourceRoot(SOURCE_REPO_ROOT)) {
    const backend = createPythonBackend(SOURCE_REPO_ROOT, `Hermes source at ${SOURCE_REPO_ROOT}`, backendArgs)

    if (backend) {
      return backend
    }
  }

  // 3. Bootstrap-complete ACTIVE_HERMES_ROOT -- the canonical install at
  //    %LOCALAPPDATA%\hermes\hermes-agent (Windows) or ~/.hermes/hermes-agent.
  //    The bootstrap marker means install.ps1 stages finished and the user
  //    completed initial configuration; we trust the install and go straight
  //    to spawning hermes. Updates flow through the in-app update path
  //    (applyUpdates -> git pull) or `hermes update` from the CLI.
  if (isBootstrapComplete()) {
    return createActiveBackend(backendArgs)
  }

  // 4. Existing `hermes` on PATH -- installed via install.ps1 / install.sh from
  //    a previous tool-only setup, or pip-installed system-wide. Use it but
  //    do NOT write a bootstrap marker; the user did this themselves and we
  //    don't want to take ownership of an install we didn't perform.
  //    HERMES_DESKTOP_IGNORE_EXISTING=1 forces the bootstrap path for testing.
  if (process.env.HERMES_DESKTOP_IGNORE_EXISTING !== '1') {
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
      const unwrapped = unwrapWindowsVenvHermesCommand(hermesCommand, backendArgs)

      if (unwrapped) {
        return unwrapped
      }

      // Smoke-test the candidate before trusting it. A `hermes` shim
      // left behind by a half-uninstalled pip install (or a venv
      // entry-point pointing at a deleted interpreter) still resolves
      // via findOnPath but explodes on spawn -- the user then sees a
      // dead backend instead of the first-launch installer. The cheap
      // `--version` probe (see backend-probes.ts) catches that case
      // and lets the resolver fall through to step 6 / bootstrap.
      const shellForProbe = isCommandScript(hermesCommand)

      if (verifyHermesCli(hermesCommand, { shell: shellForProbe })) {
        return (
          unwrapWindowsVenvHermesCommand(hermesCommand, backendArgs) || {
            label: `existing Hermes CLI at ${hermesCommand}`,
            command: hermesCommand,
            args: backendArgs,
            bootstrap: false,
            env: {},
            kind: 'command',
            shell: shellForProbe
          }
        )
      }

      rememberLog(
        `Ignoring existing Hermes CLI at ${hermesCommand}: --version probe failed; falling through to bootstrap.`
      )
    }
  }

  // 5. Last-ditch: pip-installed hermes_cli module via system Python.
  //    Same rationale as #4 -- the user installed this; we use it but don't
  //    take ownership.
  const python = findSystemPython()

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
        args: ['-m', 'hermes_cli.main', ...backendArgs],
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
    args: backendArgs,
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

    if (await handOffWindowsBootstrapRecovery('bootstrap-needed')) {
      const handoffError: Error & { isBootstrapFailure?: boolean; bootstrapHandedOff?: boolean } = new Error(
        'Hermes recovery was handed off to Hermes Setup. The desktop will restart when recovery completes.'
      )

      handoffError.isBootstrapFailure = true
      handoffError.bootstrapHandedOff = true
      bootstrapFailure = handoffError
      throw handoffError
    }

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
        protocolVersion: null
      })
    } catch {
      void 0
    }

    bootstrapAbortController = new AbortController()

    const bootstrapResult = await runBootstrap({
      installStamp: backend.installStamp,
      activeRoot: backend.activeRoot,
      sourceRepoRoot: SOURCE_REPO_ROOT,
      hermesHome: HERMES_HOME,
      logRoot: path.join(HERMES_HOME, 'logs'),
      abortSignal: bootstrapAbortController.signal,
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
      const cancelledError = new Error('Hermes install was cancelled.') as any
      cancelledError.isBootstrapFailure = true
      cancelledError.bootstrapCancelled = true
      bootstrapFailure = cancelledError
      throw cancelledError
    }

    if (!bootstrapResult.ok) {
      const bootstrapError = new Error(
        `Hermes bootstrap failed${bootstrapResult.failedStage ? ` at stage '${bootstrapResult.failedStage}'` : ''}: ` +
          `${bootstrapResult.error || 'unknown error'}. ` +
          `Check ${path.join(HERMES_HOME, 'logs', 'desktop.log')} for the full transcript.`
      ) as any

      bootstrapError.isBootstrapFailure = true
      bootstrapError.failedStage = bootstrapResult.failedStage || null
      // Latch the failure so subsequent startHermes() calls return this
      // same error without re-running install.ps1.  Cleared by the
      // hermes:bootstrap:reset IPC (renderer's "Reload and retry").
      bootstrapFailure = bootstrapError
      throw bootstrapError
    }

    rememberLog('[bootstrap] bootstrap complete; marker written. Re-resolving backend.')

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
  // %LOCALAPPDATA%\hermes\git\, which findGitBash() picks up, so for any
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

  backend.command = getVenvPython(VENV_ROOT)
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

function fetchJson(url, token, options: any = {}) {
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

    if (body) {
      req.write(body)
    }
    req.end()
  })
}

function fetchPublicJson(url, options: any = {}) {
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

    if (body) {
      req.write(body)
    }
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

  if (type === 'image/png') {
    return '.png'
  }

  if (type === 'image/jpeg') {
    return '.jpg'
  }

  if (type === 'image/gif') {
    return '.gif'
  }

  if (type === 'image/webp') {
    return '.webp'
  }

  if (type === 'image/bmp') {
    return '.bmp'
  }

  if (type === 'image/svg+xml') {
    return '.svg'
  }

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

  if (!value) {
    return ''
  }

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
  if (titleCache.size >= TITLE_CACHE_LIMIT) {
    titleCache.delete(titleCache.keys().next().value)
  }
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

function fetchHtmlTitleWithCurl(rawUrl: string): Promise<string> {
  return new Promise(resolve => {
    const url = String(rawUrl || '').trim()

    if (!url) {
      return resolve('')
    }

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
      if (bytes >= TITLE_BYTE_BUDGET) {
        return
      }
      const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)
      const remaining = TITLE_BYTE_BUDGET - bytes
      const next = buffer.length > remaining ? buffer.subarray(0, remaining) : buffer
      chunks.push(next)
      bytes += next.length
    })

    child.on('error', () => resolve(''))
    child.on('close', () => {
      if (!chunks.length) {
        return resolve('')
      }
      resolve(parseHtmlTitle(Buffer.concat(chunks).toString('utf8')))
    })
  })
}

function getLinkTitleSession() {
  if (linkTitleSession || !app.isReady()) {
    return linkTitleSession
  }
  linkTitleSession = session.fromPartition('hermes:link-titles', { cache: false })
  linkTitleSession.webRequest.onBeforeRequest((details, callback) => {
    callback({ cancel: RENDER_TITLE_BLOCKED_RESOURCES.has(details.resourceType) })
  })
  guardLinkTitleSession(linkTitleSession)

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
    if (!app.isReady()) {
      return resolve('')
    }

    const partitionSession = getLinkTitleSession()

    if (!partitionSession) {
      return resolve('')
    }

    let settled = false
    let window = null
    let hardTimer = null
    let graceTimer = null

    const finish = title => {
      if (settled) {
        return
      }
      settled = true

      if (hardTimer) {
        clearTimeout(hardTimer)
      }

      if (graceTimer) {
        clearTimeout(graceTimer)
      }
      const value = (title || '').replace(/\s+/g, ' ').trim()

      try {
        if (window && !window.isDestroyed()) {
          window.destroy()
        }
      } catch {
        // BrowserWindow may already be torn down; ignore.
      }

      resolve(value)
    }

    try {
      window = createLinkTitleWindow(BrowserWindow, partitionSession)
    } catch {
      return finish('')
    }

    const finishWithTitle = () => finish(readLinkTitleWindowTitle(window))

    const scheduleGrace = () => {
      if (graceTimer) {
        clearTimeout(graceTimer)
      }
      graceTimer = setTimeout(finishWithTitle, RENDER_TITLE_GRACE_MS)
    }

    hardTimer = setTimeout(finishWithTitle, RENDER_TITLE_TIMEOUT_MS)

    window.webContents.setUserAgent(TITLE_USER_AGENT)
    window.webContents.on('page-title-updated', scheduleGrace)
    window.webContents.on('did-finish-load', scheduleGrace)
    window.webContents.on('did-fail-load', (_event, _code, _desc, _validatedURL, isMainFrame) => {
      if (isMainFrame) {
        finish('')
      }
    })

    window
      .loadURL(rawUrl, {
        httpReferrer: 'https://www.google.com/',
        userAgent: TITLE_USER_AGENT
      })
      .catch(() => finish(''))
  })
}

function fetchHtmlTitleWithRenderer(rawUrl: string): Promise<string> {
  return new Promise(resolve => {
    renderTitleQueue.push({ resolve, url: rawUrl })
    dequeueRenderTitle()
  })
}

// Strips known error/captcha titles (e.g. "GetYourGuide – Error", "Just a
// moment...") so they don't get cached as the resolved title.
function usableTitle(value: string): string {
  return value && !TITLE_ERROR_RE.test(value) ? value : ''
}

function fetchLinkTitle(rawUrl) {
  const url = String(rawUrl || '').trim()
  const key = canonicalTitleCacheKey(url)

  if (!key) {
    return Promise.resolve('')
  }

  if (titleCache.has(key)) {
    return Promise.resolve(titleCache.get(key))
  }

  if (titleInflight.has(key)) {
    return titleInflight.get(key)
  }

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
  if (!rawUrl) {
    throw new Error('Missing URL')
  }

  if (rawUrl.startsWith('data:')) {
    const match = rawUrl.match(/^data:([^;,]+)?(;base64)?,(.*)$/s)

    if (!match) {
      throw new Error('Invalid data URL')
    }
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
  const { buffer } = (await resourceBufferFromUrl(rawUrl)) as any
  const image = nativeImage.createFromBuffer(buffer)

  if (image.isEmpty()) {
    throw new Error('Could not read image')
  }
  clipboard.writeImage(image)
}

async function saveImageFromUrl(rawUrl) {
  const { buffer, mimeType } = (await resourceBufferFromUrl(rawUrl)) as any
  const fallbackName = filenameFromUrl(rawUrl, `image${extensionForMimeType(mimeType) || '.png'}`)

  const result = await dialog.showSaveDialog(mainWindow, {
    title: 'Save Image',
    defaultPath: fallbackName
  })

  if (result.canceled || !result.filePath) {
    return false
  }
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
  if (!mainWindow || mainWindow.isDestroyed()) {
    return
  }
  const { webContents } = mainWindow

  if (!webContents || webContents.isDestroyed()) {
    return
  }
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

    if (timer) {
      clearTimeout(timer)
    }
    timer = setTimeout(() => {
      timer = null

      if (!fileExists(filePath)) {
        return
      }
      sendPreviewFileChanged({ id, path: filePath, url: pathToFileURL(filePath).toString() })
    }, PREVIEW_WATCH_DEBOUNCE_MS)
  })

  previewWatchers.set(id, {
    close: () => {
      if (timer) {
        clearTimeout(timer)
      }
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
  if (!IS_MAC) {
    return null
  }

  return mainWindow?.getWindowButtonPosition?.() || WINDOW_BUTTON_POSITION
}

function getNativeOverlayWidth() {
  return computeNativeOverlayWidth({ isWindows: IS_WINDOWS, isWsl: IS_WSL, isMac: IS_MAC })
}

function getWindowState() {
  return {
    isFullscreen: Boolean(mainWindow?.isFullScreen?.()),
    nativeOverlayWidth: getNativeOverlayWidth(),
    windowButtonPosition: getWindowButtonPosition()
  }
}

function sendBackendExit(payload) {
  // Intentional soft re-home (gateway mode apply) kills the child on purpose —
  // don't surface the "backend stopped" error toast / boot-failure path.
  if (softRehomeInProgress) {
    return
  }

  if (!mainWindow || mainWindow.isDestroyed()) {
    return
  }
  const { webContents } = mainWindow

  if (!webContents || webContents.isDestroyed()) {
    return
  }
  webContents.send('hermes:backend-exit', payload)
}

function sendClosePreviewRequested() {
  if (!mainWindow || mainWindow.isDestroyed()) {
    return
  }
  const { webContents } = mainWindow

  if (!webContents || webContents.isDestroyed()) {
    return
  }
  webContents.send('hermes:close-preview-requested')
}

// Tell the renderer the machine just woke. Sleep silently drops the
// renderer's WebSocket to the local backend; the renderer reconnects on this
// signal so the chat composer doesn't stay stuck on "Starting Hermes...".
function sendPowerResume() {
  if (!mainWindow || mainWindow.isDestroyed()) {
    return
  }
  const { webContents } = mainWindow

  if (!webContents || webContents.isDestroyed()) {
    return
  }
  webContents.send('hermes:power-resume')
}

let powerResumeRegistered = false

function registerPowerResumeListeners() {
  if (powerResumeRegistered) {
    return
  }
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
  if (!mainWindow || mainWindow.isDestroyed()) {
    return
  }
  const { webContents } = mainWindow

  if (!webContents || webContents.isDestroyed()) {
    return
  }
  webContents.send('hermes:open-updates')

  if (!mainWindow.isVisible()) {
    mainWindow.show()
  }
  mainWindow.focus()
}

function sendWindowStateChanged(nextIsFullscreen?: boolean) {
  if (!mainWindow || mainWindow.isDestroyed()) {
    return
  }
  const { webContents } = mainWindow

  if (!webContents || webContents.isDestroyed()) {
    return
  }
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

    if (!isInspectShortcut) {
      return
    }
    event.preventDefault()
    toggleDevTools(window)
  })
}

function installPreviewShortcut(window) {
  window.webContents.on('before-input-event', (event, input) => {
    const key = String(input.key || '').toLowerCase()
    const isPreviewCloseShortcut = key === 'w' && (IS_MAC ? input.meta : input.control) && !input.alt && !input.shift

    if (!isPreviewCloseShortcut || !previewShortcutActive) {
      return
    }

    event.preventDefault()
    sendClosePreviewRequested()
  })
}

// Zoom level is persisted in the renderer's own localStorage (per-origin,
// survives reloads/restarts) rather than a main-process JSON file. The main
// process owns setZoomLevel, so we mirror each change into localStorage and
// read it back on did-finish-load to re-apply after reloads or crash recovery.
import {
  clampZoomLevel,
  installZoomReassertOnWindowEvents,
  percentToZoomLevel,
  ZOOM_STORAGE_KEY,
  zoomLevelToPercent
} from './zoom'

function setAndPersistZoomLevel(window, zoomLevel) {
  if (!window || window.isDestroyed()) {
    return
  }
  const next = clampZoomLevel(zoomLevel)
  window.webContents.setZoomLevel(next)
  // Keep any open settings UI in sync, including changes made via the
  // keyboard shortcuts or the View menu.
  window.webContents.send('hermes:zoom:changed', { level: next, percent: zoomLevelToPercent(next) })
  window.webContents
    .executeJavaScript(
      `try { localStorage.setItem(${JSON.stringify(ZOOM_STORAGE_KEY)}, ${JSON.stringify(String(next))}) } catch {}`
    )
    .catch(error => rememberLog(`[zoom] persist failed: ${error?.message || error}`))
}

function restorePersistedZoomLevel(window) {
  if (!window || window.isDestroyed()) {
    return
  }
  window.webContents
    .executeJavaScript(
      `(() => { try { return localStorage.getItem(${JSON.stringify(ZOOM_STORAGE_KEY)}) } catch { return null } })()`
    )
    .then(stored => {
      if (stored == null || !window || window.isDestroyed()) {
        return
      }
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

    if (!mod || input.alt || input.shift) {
      return
    }

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
      if (template.length) {
        template.push({ type: 'separator' })
      }
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
      if (template.length) {
        template.push({ type: 'separator' })
      }

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
      if (template.length) {
        template.push({ type: 'separator' })
      }

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
    if (permission === 'media' || permission === ('audioCapture' as any) /* todo: is this needed? */) {
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
  if (oauthSession || !app.isReady()) {
    return oauthSession
  }
  oauthSession = session.fromPartition(OAUTH_SESSION_PARTITION)

  return oauthSession
}

// Bare + prefixed variants of the session cookies live in
// connection-config.ts (cookiesHaveSession / cookiesHaveLiveSession). See
// that module for details.

async function hasOauthSessionCookie(baseUrl) {
  const sess = getOauthSession()

  if (!sess) {
    return false
  }
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

  if (!sess) {
    return false
  }
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

  if (!sess) {
    return
  }

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

// Open a gateway login window in the OAuth session partition, resolving once
// the access-token cookie appears (login done) or rejecting if the user closes
// the window first. The window navigates through the IDP and back to
// /auth/callback, which sets the session cookies on the partition; we poll the
// cookie jar rather than try to read the HttpOnly value.
//
// `silent` selects the URL the window loads, which decides interactive-vs-silent:
//   - silent=false (default): load ``/login`` — the public interstitial that
//     renders the "Log in with X" provider chooser. This is the interactive
//     remote-gateway login the settings UI drives.
//   - silent=true: load the PROTECTED root ``/`` instead. ``/login`` is a public
//     route, so loading it NEVER triggers the gate's auto-SSO and always shows
//     the chooser. Loading a protected page with no session cookie makes the
//     gate run ``_auto_sso_response``: single registered provider + a live
//     portal session in this partition → a silent 302 through
//     ``/auth/login`` → portal ``/oauth/authorize`` (auto-approves org members)
//     → ``/auth/callback``, which sets the gateway cookie with NO interactive
//     prompt. This is the per-agent cloud cascade (decisions.md Q5).
function openOauthLoginWindow(baseUrl, { silent = false } = {}) {
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
    let revealTimer = null

    const finish = err => {
      if (settled) {
        return
      }
      settled = true

      if (pollTimer) {
        clearInterval(pollTimer)
      }

      if (revealTimer) {
        clearTimeout(revealTimer)
      }

      try {
        if (win && !win.isDestroyed()) {
          win.destroy()
        }
      } catch {
        // window already torn down
      }

      if (err) {
        reject(err)
      } else {
        resolve({ baseUrl, ok: true })
      }
    }

    const checkCookie = async () => {
      if (settled) {
        return
      }

      if (await hasOauthSessionCookie(baseUrl)) {
        finish(null)
      }
    }

    try {
      win = new BrowserWindow({
        width: 520,
        height: 720,
        title: silent ? 'Connecting to Hermes Cloud agent…' : 'Sign in to Hermes gateway',
        autoHideMenuBar: true,
        // Silent cascade: start HIDDEN. The auto-SSO 302 chain completes in
        // well under a second, so the window normally never needs to show. We
        // only reveal it as a fallback if the cascade DOESN'T complete quickly
        // (e.g. the portal session lapsed and the gate fell through to the
        // interactive chooser) — see the reveal timer below.
        show: !silent,
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

    // Silent-mode reveal fallback: if the cascade hasn't settled shortly, the
    // auto-SSO didn't go through silently (no portal session, multi-provider,
    // loop-guard tripped, etc.) and the window is now showing an interactive
    // page. Reveal it so the user can complete sign-in manually rather than
    // staring at nothing. Cleared on finish().
    if (silent && win) {
      revealTimer = setTimeout(() => {
        try {
          if (!settled && win && !win.isDestroyed() && !win.isVisible()) {
            win.show()
          }
        } catch {
          // window torn down
        }
      }, 2500)
    }

    win.on('closed', () => {
      if (!settled) {
        finish(new Error('Login window closed before authentication completed.'))
      }
    })

    // ``next`` is intentionally omitted: the gateway lands on ``/`` after
    // login, which is a valid authenticated page that sets the cookies. We
    // only care that the cookie jar is populated.
    //
    // silent=true loads the protected root so the gate auto-SSOs (no chooser);
    // silent=false loads the public ``/login`` chooser for interactive sign-in.
    const normalizedBase = normalizeRemoteBaseUrl(baseUrl)
    const loginUrl = silent ? `${normalizedBase}/` : `${normalizedBase}/login`
    win.loadURL(loginUrl).catch(error => {
      finish(error instanceof Error ? error : new Error(String(error)))
    })
  })
}

// JSON request routed through the OAuth session partition so the HttpOnly
// session cookie is attached automatically by Electron's net stack. Used for
// authed REST against a gated gateway, including minting WS tickets.
function fetchJsonViaOauthSession(url, options: any = {}) {
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
    } as any)

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
        if (timedOut) {
          return
        }
        clearTimeout(timer)
        const text = Buffer.concat(chunks).toString('utf8')
        const statusCode = res.statusCode || 500

        if (statusCode >= 400) {
          const err = new Error(`${statusCode}: ${text || ''}`) as any
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
      if (timedOut) {
        return
      }
      clearTimeout(timer)
      reject(error)
    })

    if (body) {
      request.write(body)
    }
    request.end()
  })
}

// Mint a single-use WS ticket for a gated gateway. Returns the ticket string.
// Throws (with statusCode 401) if the session cookie is missing/expired —
// callers treat that as "needs re-login".
async function mintGatewayWsTicket(baseUrl) {
  const body = (await fetchJsonViaOauthSession(`${baseUrl}/api/auth/ws-ticket`, {
    method: 'POST',
    timeoutMs: 8_000
  })) as any

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

// --- Hermes Cloud discovery + silent per-agent sign-in (cloud-auto-discovery
// Phase 3) ---------------------------------------------------------------
//
// The "cloud" connection mode lets a user sign in to the Nous portal ONCE in
// the OAuth session partition, then (a) discover their hosted agents and (b)
// connect to any of them with no second interactive sign-in. Both ride the one
// portal session cookie living in `persist:hermes-remote-oauth`:
//   - discovery  → GET {portal}/api/agents over the partition-bound net; the
//     portal session cookie authenticates it (NAS Phase 2.5 accepts the cookie).
//   - cascade    → opening an agent's own /login in the same partition hits the
//     portal's silent auto-approve (org member, existing session) and 302s back
//     with that agent's session cookie — no prompt. Each agent still completes
//     its own PKCE exchange; SSO removes the human click, not a security check.

// Canonical Nous portal base URL, overridable for staging/dev. Mirrors the CLI
// convention (hermes_cli/auth.py DEFAULT_NOUS_PORTAL_URL + the same env names)
// so a single override flips every Hermes surface to the same portal.
const DEFAULT_NOUS_PORTAL_URL = 'https://portal.nousresearch.com'

function resolvePortalBaseUrl() {
  const raw = process.env.HERMES_PORTAL_BASE_URL || process.env.NOUS_PORTAL_BASE_URL || DEFAULT_NOUS_PORTAL_URL

  return String(raw).trim().replace(/\/+$/, '')
}

// Whether the OAuth partition currently holds a live Nous portal session — the
// credential that powers both discovery and the silent cascade. The portal
// authenticates via PRIVY, not the Hermes gateway session cookies, so this
// checks for the `privy-token` cookie on the portal host (NOT
// hasLiveOauthSession, which looks for hermes_session_at/rt that the portal
// never sets). See connection-config.ts cookiesHavePrivySession.
async function hasLivePortalSession() {
  const sess = getOauthSession()

  if (!sess) {
    return false
  }

  const portalBaseUrl = resolvePortalBaseUrl()
  const parsed = new URL(portalBaseUrl)

  try {
    const cookies = await sess.cookies.get({ url: portalBaseUrl })

    return cookiesHavePrivySession(cookies)
  } catch {
    try {
      const cookies = await sess.cookies.get({ domain: parsed.hostname })

      return cookiesHavePrivySession(cookies)
    } catch {
      return false
    }
  }
}

// Drive a one-time interactive portal sign-in in the OAuth partition. Unlike
// openOauthLoginWindow (which targets a gateway's /login), this lands on the
// portal itself so the resulting session cookie is portal-scoped — the cookie
// that authenticates discovery AND is reused for every silent per-agent
// cascade. Resolves once the portal session cookie appears.
function openPortalLoginWindow() {
  const portalBaseUrl = resolvePortalBaseUrl()

  return new Promise((resolve, reject) => {
    if (!app.isReady()) {
      reject(new Error('Desktop is not ready to start a Hermes Cloud sign-in.'))

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
      if (settled) {
        return
      }
      settled = true

      if (pollTimer) {
        clearInterval(pollTimer)
      }

      try {
        if (win && !win.isDestroyed()) {
          win.destroy()
        }
      } catch {
        // window already torn down
      }

      if (err) {
        reject(err)
      } else {
        resolve({ portalBaseUrl, ok: true })
      }
    }

    const checkCookie = async () => {
      if (settled) {
        return
      }

      // A live portal (Privy) session cookie means sign-in completed.
      if (await hasLivePortalSession()) {
        finish(null)
      }
    }

    try {
      win = new BrowserWindow({
        width: 520,
        height: 720,
        title: 'Sign in to Hermes Cloud',
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

    win.webContents.on('did-navigate', () => void checkCookie())
    win.webContents.on('did-redirect-navigation', () => void checkCookie())
    win.webContents.on('did-frame-navigate', () => void checkCookie())
    pollTimer = setInterval(() => void checkCookie(), 750)

    win.on('closed', () => {
      if (!settled) {
        finish(new Error('Sign-in window closed before authentication completed.'))
      }
    })

    // Land on the portal root; any authenticated portal page sets the session
    // cookie. We only care that the partition cookie jar is populated.
    win.loadURL(portalBaseUrl).catch(error => {
      finish(error instanceof Error ? error : new Error(String(error)))
    })
  })
}

// Discover the hosted (Hermes Cloud) agents the signed-in user can see. Calls
// the NAS trimmed-summary endpoint over the partition-bound net, so the portal
// session cookie is attached automatically (no bearer needed — NAS accepts the
// cookie). Returns { agents } on success, or { needsOrgSelection: true, orgs }
// when the user belongs to multiple orgs and hasn't picked one yet (NAS 409
// org_selection_required). Pass `org` (a slug/id from a prior org list) to
// scope discovery to that org. Throws a needsCloudLogin-tagged error when no
// portal session is present.
async function discoverCloudAgents(org?: string) {
  const portalBaseUrl = resolvePortalBaseUrl()

  if (!(await hasLivePortalSession())) {
    const err = new Error(
      'You are not signed in to Hermes Cloud. Open Settings → Gateway, choose Hermes Cloud, and sign in.'
    ) as any
    err.needsCloudLogin = true
    throw err
  }

  const orgQuery = org ? `?org=${encodeURIComponent(org)}` : ''
  let body

  try {
    body = (await fetchJsonViaOauthSession(`${portalBaseUrl}/api/agents${orgQuery}`, {
      method: 'GET',
      timeoutMs: 15_000
    })) as any
  } catch (error) {
    // A 401 means the portal session lapsed between the liveness check and the
    // call — surface it as a re-login, not a generic failure.
    if (error && error.statusCode === 401) {
      const err = new Error('Your Hermes Cloud session has expired. Open Settings → Gateway and sign in again.') as any
      err.needsCloudLogin = true
      err.cause = error
      throw err
    }

    // A 409 means we're a multi-org user who hasn't picked an org. The body
    // carries the user's org list; surface it so the renderer shows a picker
    // and re-calls discovery with the chosen org. (fetchJsonViaOauthSession
    // throws on >=400 with err.statusCode + err.message "409: <json body>".)
    if (error && error.statusCode === 409) {
      const orgs = parseOrgSelectionError(error)

      if (orgs) {
        return { needsOrgSelection: true, orgs }
      }
    }

    throw error
  }

  return { agents: trimCloudAgents(body), org: trimCloudOrg(body?.org) }
}

// Project a NAS response org ({ id, slug, name, isPersonal }) to the trimmed
// shape the renderer persists, or null when absent/malformed.
function trimCloudOrg(org) {
  if (!org || typeof org !== 'object' || typeof org.id !== 'string') {
    return null
  }

  return {
    id: org.id,
    slug: typeof org.slug === 'string' ? org.slug : null,
    name: typeof org.name === 'string' ? org.name : org.id,
    isPersonal: Boolean(org.isPersonal),
    role: typeof org.role === 'string' ? org.role : 'MEMBER'
  }
}

// Extract the org list from a 409 org_selection_required error body. The error
// message is "409: <raw json>" (see fetchJsonViaOauthSession); parse defensively
// and return null if it isn't the shape we expect (caller then rethrows).
function parseOrgSelectionError(error) {
  const msg = String(error?.message || '')
  const jsonStart = msg.indexOf('{')

  if (jsonStart < 0) {
    return null
  }

  let parsed

  try {
    parsed = JSON.parse(msg.slice(jsonStart))
  } catch {
    return null
  }

  if (parsed?.error !== 'org_selection_required' || !Array.isArray(parsed.orgs)) {
    return null
  }

  return parsed.orgs
    .filter(o => o && typeof o === 'object' && typeof o.id === 'string')
    .map(o => ({
      id: o.id,
      slug: typeof o.slug === 'string' ? o.slug : null,
      name: typeof o.name === 'string' ? o.name : o.id,
      isPersonal: Boolean(o.isPersonal),
      role: typeof o.role === 'string' ? o.role : 'MEMBER'
    }))
}

// Project NAS's agent rows to the trimmed DTO the renderer consumes.
function trimCloudAgents(body) {
  const agents = Array.isArray(body?.agents) ? body.agents : []

  return agents
    .filter(a => a && typeof a === 'object' && typeof a.id === 'string')
    .map(a => ({
      id: a.id,
      name: typeof a.name === 'string' ? a.name : a.id,
      status: typeof a.status === 'string' ? a.status : 'unknown',
      dashboardUrl: typeof a.dashboardUrl === 'string' ? a.dashboardUrl : null,
      dashboardGatewayState: typeof a.dashboardGatewayState === 'string' ? a.dashboardGatewayState : 'unknown'
    }))
}

// Silent per-agent sign-in: open the selected agent dashboard's /login in the
// SAME OAuth partition. Because the user already holds a live portal session
// there, the agent's /oauth/authorize auto-approves (org member) and 302s back,
// setting that agent's gateway session cookie WITHOUT a second interactive
// prompt. Reuses openOauthLoginWindow — the window self-closes the instant the
// agent's session cookie lands (a silent flow finishes in well under a second;
// if the portal session were absent it would fall through to an interactive
// login, which the discovery gate already prevents). Returns once the agent's
// gateway session cookie is present.
async function cloudAgentSilentSignIn(dashboardUrl) {
  const baseUrl = normalizeRemoteBaseUrl(dashboardUrl)

  // Pre-req: a live portal session must exist, or this would surface an
  // interactive prompt rather than a silent cascade. Discovery already gates on
  // this, but a selection can arrive after the session lapsed.
  if (!(await hasLivePortalSession())) {
    const err = new Error('Your Hermes Cloud session has expired. Sign in to Hermes Cloud again.') as any
    err.needsCloudLogin = true
    throw err
  }

  await openOauthLoginWindow(baseUrl, { silent: true })

  return { baseUrl, connected: await hasOauthSessionCookie(baseUrl) }
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
function sanitizeConnectionProfiles(raw: Record<string, any>) {
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

    const cleaned: {
      mode: 'remote' | 'local' | 'cloud'
      url?: string
      authMode?: string
      token?: object
      org?: string
    } = {
      mode: modeIsRemoteLike(entry.mode) ? entry.mode : 'local'
    }
    const url = String(entry.url || '').trim()

    if (url) {
      cleaned.url = url
    }

    cleaned.authMode = normAuthMode(entry.authMode)

    if ((entry as any).token && typeof entry.token === 'object') {
      cleaned.token = entry.token
    }

    // Preserve the Hermes Cloud org tag on cloud-mode entries so Settings can
    // reopen into the same org for a per-profile cloud connection.
    if (cleaned.mode === 'cloud') {
      const org = String(entry.org || '').trim()

      if (org) {
        cleaned.org = org
      }
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
        mode: modeIsRemoteLike(parsed.mode) ? parsed.mode : 'local',
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
  // The env override forces a plain remote connection. Otherwise reflect the
  // saved mode, preserving 'cloud' (a Hermes Cloud connection — Q6) so the UI
  // reopens into the cloud picker; any non-remote-like value collapses to local.
  const savedMode = key ? scoped?.mode : config.mode
  const mode = envOverride ? 'remote' : modeIsRemoteLike(savedMode) ? savedMode : 'local'

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
    // The persisted Hermes Cloud org (slug/id) for a cloud connection, or '' for
    // remote/local. Lets Settings → Gateway reopen into the same org.
    cloudOrg: mode === 'cloud' ? String(block.org || '') : '',
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
// `org` (optional) is the Hermes Cloud org slug/id the instance was discovered
// under — persisted so Settings can reopen into the same org; omitted from the
// block when empty so plain remote connections stay unchanged.
function buildRemoteBlock(remoteUrl, authMode, token, org?: string) {
  if (authMode !== 'oauth' && !decryptDesktopSecret(token)) {
    throw new Error('Remote gateway session token is required.')
  }

  const block: { url: string; authMode: string; token: object; org?: string } = {
    url: normalizeRemoteBaseUrl(remoteUrl),
    authMode,
    token
  }
  const orgValue = typeof org === 'string' ? org.trim() : ''

  if (orgValue) {
    block.org = orgValue
  }

  return block
}

function coerceDesktopConnectionConfig(input: any = {}, existing = readDesktopConnectionConfig(), options: any = {}) {
  const persistToken = options.persistToken !== false
  const key = connectionScopeKey(input.profile)
  // 'cloud' and 'remote' both persist a remote-shaped block; 'cloud' is
  // remembered as its own provenance (Q6) and resolves to remote downstream.
  // Anything else collapses to local.
  const mode = modeIsRemoteLike(input.mode) ? input.mode : 'local'
  const remoteLike = modeIsRemoteLike(mode)

  // The block being edited: a per-profile entry or the global remote block.
  const rawExistingBlock = key ? existing.profiles?.[key] || {} : existing.remote || {}
  // Leaving a CLOUD connection unselects it: a cloud block's url/org/token
  // describe a discovered Hermes Cloud instance, NOT a user-owned remote gateway,
  // so switching to local or remote must NOT inherit them (otherwise the stale
  // cloud URL lingers and re-selecting Cloud looks "already connected"). When the
  // saved block was cloud and the new mode is not cloud, start from an empty
  // block. (remote↔local toggles still preserve a real remote URL as before.)
  const existingMode = key ? existing.profiles?.[key]?.mode : existing.mode
  const leavingCloud = existingMode === 'cloud' && mode !== 'cloud'
  const existingBlock = leavingCloud ? {} : rawExistingBlock
  const remoteUrl = String(input.remoteUrl ?? existingBlock.url ?? '').trim()
  // authMode: explicit input wins; otherwise inherit the saved value, default 'token'.
  const authMode = resolveAuthMode(input.remoteAuthMode, existingBlock.authMode)
  // Cloud org: only meaningful for 'cloud' mode. Explicit input wins; otherwise
  // inherit the saved org. A plain 'remote' connection never carries an org
  // (switching cloud→remote drops it), so it stays unset unless mode is cloud.
  const cloudOrg = mode === 'cloud' ? String(input.cloudOrg ?? existingBlock.org ?? '').trim() : ''
  const incomingToken = typeof input.remoteToken === 'string' ? input.remoteToken.trim() : ''

  const nextToken = incomingToken
    ? persistToken
      ? encryptDesktopSecret(incomingToken)
      : { encoding: 'plain', value: incomingToken }
    : existingBlock.token

  if (key) {
    // Per-profile scope: a remote/cloud entry pins this profile to its own
    // backend; a local entry clears the override so the profile inherits the
    // default. The mode tag (remote vs cloud) is preserved on the entry.
    const profiles = { ...(existing.profiles || {}) }

    if (remoteLike) {
      profiles[key] = { mode, ...buildRemoteBlock(remoteUrl, authMode, nextToken, cloudOrg) }
    } else {
      delete profiles[key]
    }

    return {
      mode: modeIsRemoteLike(existing.mode) ? existing.mode : 'local',
      remote: existing.remote || {},
      profiles
    }
  }

  const nextRemote = remoteLike
    ? buildRemoteBlock(remoteUrl, authMode, nextToken, cloudOrg)
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
      ) as any

      err.needsOauthLogin = true
      throw err
    }

    let ticket

    try {
      ticket = await mintGatewayWsTicket(baseUrl)
    } catch (error) {
      const err = new Error(
        'Your remote gateway session has expired. ' + 'Open Settings → Gateway and click "Sign in" again.'
      ) as any

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

  // 3. Global remote (or cloud — cloud resolves to a remote backend, Q6).
  if (!modeIsRemoteLike(config.mode)) {
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
// Remote/Cloud, or the env override): a SINGLE remote backend serves every
// profile via ?profile=. Cloud counts — it resolves to a remote backend (Q6).
// Distinct from per-profile overrides — here there's one host for all.
function globalRemoteActive() {
  if (process.env.HERMES_DESKTOP_REMOTE_URL) {
    return true
  }

  return modeIsRemoteLike(readDesktopConnectionConfig().mode)
}

// GET a profile's resolved backend (remote pool or local primary), parsed JSON.
async function fetchJsonForProfile(profile, path) {
  return requestJsonForProfile(profile, path, 'GET')
}

// Issue an arbitrary method against a profile's resolved backend, parsed JSON.
async function requestJsonForProfile(profile: string, path: string, method: Method, body?: string) {
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
  } catch (error: any) {
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
      const body = (await fetchPublicJson(`${baseUrl}/api/auth/providers`, { timeoutMs: 8_000 })) as any

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

async function testDesktopConnectionConfig(input: any = {}) {
  const config = coerceDesktopConnectionConfig(input, readDesktopConnectionConfig(), { persistToken: false })
  const key = connectionScopeKey(input.profile)
  // The block under test: a per-profile entry or the global remote. Coerce has
  // already normalized the URL and resolved token inheritance for the scope.
  const block = key ? config.profiles?.[key] || null : config.remote

  const wantRemote =
    modeIsRemoteLike(block?.mode) || (!key && modeIsRemoteLike(config.mode)) || (modeIsRemoteLike(input.mode) && block)

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

  const status = (await fetchJson(`${baseUrl}/api/status`, token, { timeoutMs: 8_000 })) as any

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

function stopBackendChild(child) {
  if (!child || child.killed) {
    return
  }

  try {
    if (IS_WINDOWS && Number.isInteger(child.pid)) {
      forceKillProcessTree(child.pid)
    } else {
      child.kill('SIGTERM')
    }
  } catch {
    // Already gone.
  }
}

// Soft gateway-mode apply: tear down the primary without resetting boot UI or
// reloading the renderer. The shell stays up; the renderer wipes session lists
// (so skeletons retrigger) and re-dials. Distinct from hard re-home (profile
// switch / crash recovery), which still resets boot progress + reloads.
function resetHermesConnection({ soft = false } = {}) {
  connectionPromise = null
  backendStartFailure = null

  stopBackendChild(hermesProcess)

  hermesProcess = null

  if (!soft) {
    resetBootProgressForReconnect()
  }
}

// Re-home the primary backend: reset connection state, then wait for the live
// dashboard process to actually exit (SIGKILL after 5s) so the next
// startHermes() spawns fresh instead of racing the dying one. Shared by the
// connection-config and profile switch flows.
async function teardownPrimaryBackendAndWait({ soft = false } = {}) {
  // Capture the reference before resetHermesConnection() nulls hermesProcess.
  const dying = hermesProcess && !hermesProcess.killed ? hermesProcess : null

  if (soft) {
    softRehomeInProgress = true
  }

  try {
    resetHermesConnection({ soft })
    await waitForBackendExit(dying)
  } finally {
    if (soft) {
      softRehomeInProgress = false
    }
  }
}

function sendConnectionApplied() {
  if (!mainWindow || mainWindow.isDestroyed()) {
    return
  }

  const { webContents } = mainWindow

  if (!webContents || webContents.isDestroyed()) {
    return
  }

  webContents.send('hermes:connection:applied')
}

async function waitForBackendExit(child, timeoutMs = 5000) {
  if (!child) {
    return
  }

  if (child.exitCode !== null || child.signalCode !== null) {
    return
  }

  await new Promise<void>(resolve => {
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

  const stopping = poolStops.get(key)

  if (stopping) {
    await stopping
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

  if (!key) {
    return
  }
  const entry = backendPool.get(key)

  if (entry) {
    entry.lastActiveAt = Date.now()
  }
}

// Evict least-recently-used pool backends until at most `keep` remain — but only
// ever evict backends without a live renderer socket (stale beyond the keepalive
// window). When every backend is actively kept alive we let the pool exceed the
// soft cap rather than kill a running session.
function evictLruPoolBackends(keep) {
  if (backendPool.size <= keep) {
    return
  }
  const now = Date.now()

  const evictable = [...backendPool.entries()]
    .filter(([, entry]) => now - (entry.lastActiveAt || 0) > POOL_KEEPALIVE_FRESH_MS)
    .sort((a, b) => (a[1].lastActiveAt || 0) - (b[1].lastActiveAt || 0))

  let removable = backendPool.size - Math.max(0, keep)

  for (const [profile] of evictable) {
    if (removable <= 0) {
      break
    }
    rememberLog(`Evicting idle profile backend "${profile}" (LRU cap ${POOL_MAX_BACKENDS})`)
    stopPoolBackend(profile)
    removable -= 1
  }
}

function startPoolIdleReaper() {
  if (poolIdleReaper) {
    return
  }
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

  if (typeof poolIdleReaper.unref === 'function') {
    poolIdleReaper.unref()
  }
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
  const backendArgs = ['--profile', profile, 'serve', '--host', '127.0.0.1', '--port', '0']
  const backend = await ensureRuntime(resolveHermesBackend(backendArgs))
  // Route old runtimes (no `serve`) through the legacy `dashboard --no-open`.
  backend.args = getBackendArgsForRuntime(backend)
  const hermesCwd = resolveHermesCwd()
  const webDist = resolveWebDist()
  const readyFile = backend.readyFile ? makeDashboardReadyFile() : null

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
        // Pin the gateway's tool/terminal cwd to the same directory we chose for
        // the child process. Inherited TERMINAL_CWD (or a stale config bridge)
        // can still point at the install dir even when spawn cwd is home.
        TERMINAL_CWD: hermesCwd,
        HERMES_DASHBOARD_SESSION_TOKEN: token,
        // Marks this dashboard backend as desktop-spawned so it runs the cron
        // scheduler tick loop (the gateway isn't running under the app).
        HERMES_DESKTOP: '1',
        HERMES_WEB_DIST: webDist,
        ...(readyFile ? { HERMES_DESKTOP_READY_FILE: readyFile } : {})
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
  const port = await Promise.race([waitForDashboardPortAnnouncement(child, { readyFile }), startFailed])

  if (readyFile) {
    fs.unlink(readyFile, () => {})
  }

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

const poolStops = new Map()

async function stopPoolBackend(profile) {
  const stopping = poolStops.get(profile)

  if (stopping) {
    return stopping
  }
  const entry = backendPool.get(profile)

  if (!entry) {
    return
  }
  const stop = (async () => {
    stopBackendChild(entry.process)
    await waitForBackendExit(entry.process)
  })().finally(() => {
    if (backendPool.get(profile) === entry) {
      backendPool.delete(profile)
    }
    poolStops.delete(profile)
  })

  poolStops.set(profile, stop)
  return stop
}

function teardownPoolBackendAndWait(profile) {
  return stopPoolBackend(profile)
}

function stopAllPoolBackends() {
  return Promise.all([...backendPool.keys()].map(profile => stopPoolBackend(profile)))
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

// Returns the profile name whose backend was torn down, or null when the
// request is not a profile-delete.  The caller uses this to skip ensureBackend
// for the just-torn-down profile — otherwise ensureBackend respawns a pool
// backend whose ensure_hermes_home() recreates the deleted profile directory.
async function prepareProfileDeleteRequest(request) {
  const profile = profileNameFromDeleteRequest(request)

  if (!profile || profile === 'default' || !PROFILE_NAME_RE.test(profile)) {
    return null
  }

  if (profile === primaryProfileKey()) {
    writeActiveDesktopProfile('default')
    await teardownPrimaryBackendAndWait()

    return profile
  }

  await teardownPoolBackendAndWait(profile)

  return profile
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

  if (backendStartFailure) {
    throw backendStartFailure
  }

  if (connectionPromise) {
    return connectionPromise
  }

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

    // Mutual exclusion with an in-app update (#50238). If this instance was
    // relaunched while the Tauri updater is still applying an update, spawning
    // a local backend now re-locks the venv shim and gets killed by the
    // updater's straggler cleanup — looping. Park until the update finishes (or
    // is detected stale), THEN start the backend. Local backends only; remote
    // connections returned above and never touch the install tree.
    await waitForUpdateToFinish()

    const token = crypto.randomBytes(32).toString('base64url')
    // --port 0: the OS assigns an ephemeral port; the child announces it on stdout.
    const backendArgs = ['serve', '--host', '127.0.0.1', '--port', '0']
    // Pin the desktop's chosen profile via the global --profile flag. This is
    // deterministic (it wins over the sticky ~/.hermes/active_profile file) and
    // resolves HERMES_HOME the same way `hermes -p <name>` does on the CLI. An
    // unset preference keeps the legacy launch so existing installs are
    // unaffected.
    const activeProfile = readActiveDesktopProfile()

    if (activeProfile) {
      backendArgs.unshift('--profile', activeProfile)
    }

    await advanceBootProgress('backend.runtime', 'Resolving Hermes runtime', 28)
    const backend = await ensureRuntime(resolveHermesBackend(backendArgs))
    // Route old runtimes (no `serve`) through the legacy `dashboard --no-open`.
    backend.args = getBackendArgsForRuntime(backend)
    const hermesCwd = resolveHermesCwd()
    const webDist = resolveWebDist()
    const readyFile = backend.readyFile ? makeDashboardReadyFile() : null

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
          TERMINAL_CWD: hermesCwd,
          HERMES_DASHBOARD_SESSION_TOKEN: token,
          // Marks this dashboard backend as desktop-spawned so it runs the cron
          // scheduler tick loop (the gateway isn't running under the app).
          HERMES_DESKTOP: '1',
          HERMES_WEB_DIST: webDist,
          ...(readyFile ? { HERMES_DESKTOP_READY_FILE: readyFile } : {})
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
    const port = await Promise.race([
      waitForDashboardPortAnnouncement(hermesProcess, { readyFile }),
      backendStartFailed
    ])

    if (readyFile) {
      fs.unlink(readyFile, () => {})
    }

    const baseUrl = `http://127.0.0.1:${port}`
    await advanceBootProgress('backend.wait', 'Waiting for Hermes backend to become ready', 90)
    await Promise.race([waitForHermes(baseUrl, token), backendStartFailed])
    backendReady = true
    backendStartFailure = null

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
    backendStartFailure = error instanceof Error ? error : new Error(message)
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
//
// `zoom` is opt-out for the pet overlay: it sizes its own OS window to fit the
// sprite in unzoomed CSS px (overlayWindowSize -> setBounds) and has its own
// Alt+wheel scale, so inheriting the global UI zoom would render the mascot
// larger than its window and crop it. Chat windows keep zoom on.
function wireCommonWindowHandlers(win, { zoom = true }: { zoom?: boolean } = {}) {
  installPreviewShortcut(win)
  installDevToolsShortcut(win)
  if (zoom) {
    installZoomShortcuts(win)
    // Re-apply persisted zoom on show/restore (Windows drops webContents zoom on
    // minimize/restore) and on first load (reloads / crash recovery).
    installZoomReassertOnWindowEvents(win, () => restorePersistedZoomLevel(win))
    win.webContents.once('did-finish-load', () => restorePersistedZoomLevel(win))
  }
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
// builder live in session-windows.ts so they stay unit-testable.
const sessionWindows = createSessionWindowRegistry()

function focusWindow(win) {
  if (!win || win.isDestroyed()) {
    return
  }

  if (win.isMinimized()) {
    win.restore()
  }

  if (!win.isVisible()) {
    win.show()
  }
  win.focus()
}

function spawnSecondaryWindow({
  sessionId,
  watch,
  newSession
}: { sessionId?: string; watch?: boolean; newSession?: boolean } = {}) {
  const icon = getAppIconPath()

  const win = new BrowserWindow({
    width: SESSION_WINDOW_MIN_WIDTH,
    height: SESSION_WINDOW_MIN_HEIGHT,
    minWidth: SESSION_WINDOW_MIN_WIDTH,
    minHeight: SESSION_WINDOW_MIN_HEIGHT,
    title: 'Hermes',
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
    webPreferences: chatWindowWebPreferences(PRELOAD_PATH)
  })

  if (IS_MAC) {
    win.setWindowButtonPosition?.(WINDOW_BUTTON_POSITION)
  }

  win.once('ready-to-show', () => {
    if (!win.isDestroyed()) {
      win.show()
    }
  })

  win.on('enter-full-screen', () => sendWindowStateChanged(true))
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

// The pet overlay: a single transparent, frameless, always-on-top window that
// hosts ONLY the floating mascot. Shift-clicking the in-window pet "pops it out"
// here so it can leave the app's bounds and stay visible while Hermes is
// minimized (Codex-style task-completion glance). It carries no gateway
// connection of its own — the main renderer is the single source of truth and
// pushes pet state over IPC (hermes:pet-overlay:state); the overlay just renders
// it. Control flows back (pop-in, composer submit) via hermes:pet-overlay:control.
let petOverlayWindow = null

function petOverlayUrl() {
  if (DEV_SERVER) {
    return `${DEV_SERVER.endsWith('/') ? DEV_SERVER.slice(0, -1) : DEV_SERVER}/?win=overlay#/`
  }

  return `${pathToFileURL(resolveRendererIndex()).toString()}?win=overlay#/`
}

function spawnPetOverlayWindow(bounds) {
  const win = new BrowserWindow({
    width: Math.max(80, Math.round(bounds?.width || 220)),
    height: Math.max(80, Math.round(bounds?.height || 220)),
    x: Number.isFinite(bounds?.x) ? Math.round(bounds.x) : undefined,
    y: Number.isFinite(bounds?.y) ? Math.round(bounds.y) : undefined,
    frame: false,
    transparent: true,
    resizable: false,
    movable: true,
    minimizable: false,
    maximizable: false,
    fullscreenable: false,
    // Windows/Linux need this so the helper window does not get its own
    // taskbar/alt-tab entry. On macOS, cmd-tab is app-level and this can make
    // the whole app look like it vanished when the only newly-created visible
    // window is a frameless overlay. Use NSPanel + Mission Control hiding below
    // instead, leaving the main Hermes app as the Dock/cmd-tab anchor.
    skipTaskbar: !IS_MAC,
    hasShadow: false,
    alwaysOnTop: true,
    // macOS panels are non-activating helper windows and can float over full
    // screen spaces without becoming the app's main switcher window.
    type: IS_MAC ? 'panel' : undefined,
    hiddenInMissionControl: IS_MAC,
    // Non-activating: the overlay must never become the app's key/main window,
    // or it (a frameless, taskbar-skipping panel) becomes the app's switcher
    // anchor and the Hermes icon drops out of cmd/alt-tab — especially when the
    // main window is minimized. We flip this on only while the composer needs
    // the keyboard (see hermes:pet-overlay:set-focusable).
    focusable: false,
    show: false,
    // Fully transparent — the renderer paints only the sprite + bubble.
    backgroundColor: '#00000000',
    webPreferences: {
      preload: PRELOAD_PATH,
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
      devTools: true,
      // Keep the sprite animating + bubble updating while the main window is
      // minimized/blurred — the whole point of the overlay.
      backgroundThrottling: false
    }
  })

  // Float above other apps and follow the user across desktops so the pet is
  // always reachable. `floating` + `type: panel` is the macOS NSPanel path; the
  // more aggressive `screen-saver` level can interfere with normal app/window
  // switching semantics.
  win.setAlwaysOnTop(true, IS_MAC ? 'floating' : 'screen-saver')
  win.setHiddenInMissionControl?.(true)

  try {
    // Electron docs: macOS may transform process type on each
    // setVisibleOnAllWorkspaces() call unless skipTransformProcessType=true,
    // which briefly hides the Dock/cmd-tab presence. Keep Hermes in the normal
    // ForegroundApplication class so shift-clicking the pet never drops the app
    // out of app switchers.
    win.setVisibleOnAllWorkspaces(
      true,
      IS_MAC ? { visibleOnFullScreen: true, skipTransformProcessType: true } : undefined
    )
  } catch {
    // Not supported everywhere — best effort.
  }

  // Pet overlay opts out of global UI zoom (see wireCommonWindowHandlers): it
  // owns its window-fit + scale, and inheriting zoom would crop the sprite.
  wireCommonWindowHandlers(win, { zoom: false })

  win.once('ready-to-show', () => {
    if (!win.isDestroyed()) {
      win.showInactive()
    }
  })

  win.on('closed', () => {
    if (petOverlayWindow === win) {
      petOverlayWindow = null
    }

    // If the overlay went away on its own (e.g. ⌘W), tell the main renderer to
    // pop the pet back in so it doesn't stay hidden. Harmless echo when we're
    // the ones who closed it (popInPet already cleared the active flag).
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('hermes:pet-overlay:control', { type: 'pop-in' })
    }
  })

  win.loadURL(petOverlayUrl())

  return win
}

function openPetOverlay(bounds) {
  if (petOverlayWindow && !petOverlayWindow.isDestroyed()) {
    if (bounds) {
      petOverlayWindow.setBounds({
        x: Math.round(bounds.x),
        y: Math.round(bounds.y),
        width: Math.max(80, Math.round(bounds.width)),
        height: Math.max(80, Math.round(bounds.height))
      })
    }

    petOverlayWindow.showInactive()

    return petOverlayWindow
  }

  petOverlayWindow = spawnPetOverlayWindow(bounds)

  return petOverlayWindow
}

function closePetOverlay() {
  if (petOverlayWindow && !petOverlayWindow.isDestroyed()) {
    petOverlayWindow.close()
  }

  petOverlayWindow = null
}

function createWindow() {
  const icon = getAppIconPath()
  const savedWindowState = readWindowState()
  mainWindow = new BrowserWindow({
    ...computeWindowOptions(savedWindowState, screen.getAllDisplays()),
    minWidth: WINDOW_MIN_WIDTH,
    minHeight: WINDOW_MIN_HEIGHT,
    title: 'Hermes',
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
    // windows, stalling the live answer until refocus. See session-windows.ts.
    webPreferences: chatWindowWebPreferences(PRELOAD_PATH)
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
        applyTitleBarOverlay(mainWindow)
      })
    }
  }

  if (savedWindowState?.isMaximized) {
    mainWindow.maximize()
  }

  mainWindow.once('ready-to-show', () => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.show()
    }
  })

  mainWindow.on('will-enter-full-screen', () => sendWindowStateChanged(true))
  mainWindow.on('enter-full-screen', () => sendWindowStateChanged(true))
  mainWindow.on('will-leave-full-screen', () => sendWindowStateChanged(false))
  mainWindow.on('leave-full-screen', () => sendWindowStateChanged(false))

  // Reopen where the user left off. resized/moved settle once per drag; close is
  // the cross-platform backstop, flushed synchronously before the window is gone.
  mainWindow.on('resized', schedulePersistWindowState)
  mainWindow.on('moved', schedulePersistWindowState)
  mainWindow.on('maximize', schedulePersistWindowState)
  mainWindow.on('unmaximize', schedulePersistWindowState)
  mainWindow.on('close', () => schedulePersistWindowState.flush())

  // The overlay rides the main window — closing the app's primary window must
  // tear it down too (otherwise it strands as an orphan that blocks
  // window-all-closed from quitting on Windows/Linux).
  mainWindow.on('closed', () => closePetOverlay())

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
        if (!mainWindow || mainWindow.isDestroyed()) {
          return
        }

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

    if (level !== 3) {
      return
    }

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
    // Zoom restore is handled by wireCommonWindowHandlers (shared with session
    // windows); no need to reapply it here.
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

// --- Text size (zoom) -------------------------------------------------------
// The settings UI drives the same clamped zoom scale as the Ctrl/Cmd
// shortcuts and the View menu. Reads and writes target the asking window.
ipcMain.handle('hermes:zoom:get', event => {
  const window = BrowserWindow.fromWebContents(event.sender)
  const level = window && !window.isDestroyed() ? window.webContents.getZoomLevel() : 0

  return { level, percent: zoomLevelToPercent(level) }
})
ipcMain.on('hermes:zoom:set-percent', (event, percent) => {
  const window = BrowserWindow.fromWebContents(event.sender)

  if (!window || window.isDestroyed()) {
    return
  }
  setAndPersistZoomLevel(window, percentToZoomLevel(Number(percent)))
})

// --- Pet overlay (pop-out mascot) -----------------------------------------
// `request` is `{ bounds, screen }`. A fresh pop-out passes viewport-space
// bounds (screen=false): convert to screen space by adding the main window's
// content origin so the pet lands where it sat in-window. A remembered/dragged
// spot passes screen-space bounds (screen=true) and is used as-is. We return the
// resolved screen bounds so the renderer can persist exactly where it opened.
ipcMain.handle('hermes:pet-overlay:open', async (_event, request) => {
  const bounds = request && request.bounds ? request.bounds : request
  const isScreen = Boolean(request && request.screen)
  let screenBounds = bounds

  try {
    if (bounds && !isScreen && mainWindow && !mainWindow.isDestroyed()) {
      const content = mainWindow.getContentBounds()
      screenBounds = {
        x: content.x + (bounds.x || 0),
        y: content.y + (bounds.y || 0),
        width: bounds.width,
        height: bounds.height
      }
    }
  } catch {
    // Fall back to raw bounds if the window geometry is unavailable.
  }

  openPetOverlay(screenBounds)

  return { ok: true, bounds: screenBounds }
})
ipcMain.handle('hermes:pet-overlay:close', async () => {
  closePetOverlay()

  return { ok: true }
})
// Drag/resize: the overlay reports new absolute screen bounds (it already knows
// the pointer's screen coords). Drag keeps the size constant; the wheel-to-scale
// gesture grows/shrinks it so the sprite is never cropped by the window edge.
// The window is created non-resizable (no stray edge-drag on the transparent
// frameless panel), which on Windows/Linux also blocks programmatic setBounds
// sizing — so briefly flip resizable on whenever the size actually changes.
ipcMain.on('hermes:pet-overlay:set-bounds', (_event, bounds) => {
  if (!petOverlayWindow || petOverlayWindow.isDestroyed() || !bounds) {
    return
  }

  const win = petOverlayWindow
  const width = Math.max(80, Math.round(bounds.width))
  const height = Math.max(80, Math.round(bounds.height))
  const [curW, curH] = win.getSize()
  const resizing = width !== curW || height !== curH

  if (resizing && !win.isResizable()) {
    win.setResizable(true)
  }

  win.setBounds({ x: Math.round(bounds.x), y: Math.round(bounds.y), width, height })

  if (resizing) {
    win.setResizable(false)
  }
})
// Click-through: the overlay window is a full rectangle but only the pet pixels
// should be interactive. The renderer toggles this as the cursor enters/leaves
// the sprite so transparent margins pass clicks to whatever is behind.
ipcMain.on('hermes:pet-overlay:ignore-mouse', (_event, ignore) => {
  if (petOverlayWindow && !petOverlayWindow.isDestroyed()) {
    petOverlayWindow.setIgnoreMouseEvents(Boolean(ignore), { forward: true })
  }
})
// The overlay is a non-activating panel (focusable:false) so it never steals
// the app's cmd/alt-tab anchor from the main window. But the pop-up composer
// needs the keyboard, so the renderer asks us to flip it focusable + focus it
// while the composer is open, then back to non-activating when it closes.
ipcMain.on('hermes:pet-overlay:set-focusable', (_event, focusable) => {
  if (!petOverlayWindow || petOverlayWindow.isDestroyed()) {
    return
  }

  petOverlayWindow.setFocusable(Boolean(focusable))

  if (focusable) {
    petOverlayWindow.focus()
  }
})
// Main renderer → overlay: forward the latest pet state for the overlay to render.
ipcMain.on('hermes:pet-overlay:state', (_event, payload) => {
  if (petOverlayWindow && !petOverlayWindow.isDestroyed()) {
    petOverlayWindow.webContents.send('hermes:pet-overlay:state', payload)
  }
})
// Overlay → main renderer: control messages (pop back in, composer submit).
ipcMain.on('hermes:pet-overlay:control', (_event, payload) => {
  if (!mainWindow || mainWindow.isDestroyed()) {
    return
  }

  // Double-click toggles the app window: hide it away if it's up front, bring it
  // back if it's minimized/buried. Pure window control — nothing for the
  // renderer to do, so don't forward it.
  if (payload && payload.type === 'toggle-app') {
    if (mainWindow.isMinimized() || !mainWindow.isVisible()) {
      mainWindow.show()
      mainWindow.focus()
    } else {
      mainWindow.minimize()
    }

    return
  }

  // The mail icon means "take me to the app": raise the main window (it may be
  // minimized or buried) before the renderer navigates to the latest thread.
  if (payload && payload.type === 'open-app') {
    if (mainWindow.isMinimized()) {
      mainWindow.restore()
    }

    mainWindow.show()
    mainWindow.focus()
  }

  mainWindow.webContents.send('hermes:pet-overlay:control', payload)
})
ipcMain.handle('hermes:bootstrap:reset', async () => {
  // Renderer's "Reload and retry" path. Clear the latched failure and
  // reset connection state so the next startHermes() call restarts the
  // full backend flow (including a fresh runBootstrap pass).
  rememberLog('[bootstrap] reset requested by renderer; clearing latched failure')
  await teardownPrimaryBackendAndWait()
  bootstrapFailure = null
  backendStartFailure = null
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
  backendStartFailure = null
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

// --- Hermes Cloud (cloud-auto-discovery Phase 3) ---
// One portal login in the OAuth partition powers both discovery and the silent
// per-agent cascade. See the discovery/cascade helpers above.
ipcMain.handle('hermes:cloud:status', async () => ({
  portalBaseUrl: resolvePortalBaseUrl(),
  signedIn: await hasLivePortalSession()
}))
ipcMain.handle('hermes:cloud:login', async () => {
  await openPortalLoginWindow()

  return { ok: true, signedIn: await hasLivePortalSession() }
})
ipcMain.handle('hermes:cloud:logout', async () => {
  await clearOauthSession(resolvePortalBaseUrl())

  return { ok: true, signedIn: await hasLivePortalSession() }
})
ipcMain.handle('hermes:cloud:discover', async (_event, org) => {
  // Returns { agents } or { needsOrgSelection: true, orgs }. `org` (optional)
  // scopes discovery to a chosen org for multi-org users.
  return discoverCloudAgents(typeof org === 'string' && org ? org : undefined)
})
ipcMain.handle('hermes:cloud:agent-sign-in', async (_event, dashboardUrl) => {
  // Silent per-agent sign-in via the shared portal session. Returns the agent's
  // gateway baseUrl + whether its session cookie landed; the renderer then
  // saves a cloud-mode connection pointed at this dashboardUrl.
  return cloudAgentSilentSignIn(dashboardUrl)
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
    // Global / primary connection: soft re-home. Tear down the window backend
    // without resetting boot UI or reloading — the shell stays, the renderer
    // wipes session lists (skeletons) and re-dials on hermes:connection:applied.
    await teardownPrimaryBackendAndWait({ soft: true })
    sendConnectionApplied()
  }

  return sanitizeDesktopConnectionConfig(config, payload?.profile)
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

      if (body) {
        delete body.profile
      }

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

  return { ...(data as any), sessions: rowsOf(data) }
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

  const base = (await fetchJson(`${primary.baseUrl}/api/profiles/sessions?${searchParams}`, primary.token, {
    method: 'GET',
    timeoutMs: DEFAULT_FETCH_TIMEOUT_MS
  }).catch(() => ({ sessions: [], total: 0, profile_totals: {} }))) as any

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

  return { ...(base as any), sessions: merged.slice(offset, offset + limit), total, profile_totals: profileTotals }
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

  const tornDownProfile = await prepareProfileDeleteRequest(request)

  const profile = request?.profile
  // After tearing down a backend for profile deletion, route to the primary
  // backend instead of spawning a fresh pool backend.  A freshly spawned
  // backend calls ensure_hermes_home() which recreates the profile directory,
  // defeating the deletion and leaving a zombie process.
  const routeProfile = tornDownProfile ? null : profile
  const connection = await ensureBackend(routeProfile)
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
  if (connection.authMode === 'oauth') {
    return fetchJsonViaOauthSession(url, {
      method: request?.method,
      body: request?.body,
      timeoutMs
    })
  }

  return fetchJson(url, connection.token, {
    method: request?.method,
    body: request?.body,
    timeoutMs
  })
})

ipcMain.handle('hermes:notify', (_event, payload) => {
  if (!Notification.isSupported()) {
    return false
  }
  // Action buttons render only on signed macOS builds; elsewhere they're dropped
  // and the body click still works.
  const actions = Array.isArray(payload?.actions) ? payload.actions : []

  const notification = new Notification({
    title: payload?.title || 'Hermes',
    body: payload?.body || '',
    silent: Boolean(payload?.silent),
    actions: actions.map(action => ({ type: 'button', text: String(action?.text || '') }))
  })

  notification.on('click', () => {
    if (!mainWindow || mainWindow.isDestroyed()) {
      return
    }
    focusWindow(mainWindow)

    if (payload?.sessionId) {
      mainWindow.webContents.send('hermes:focus-session', payload.sessionId)
    }
  })
  notification.on('action', (_actionEvent, index) => {
    if (!mainWindow || mainWindow.isDestroyed()) {
      return
    }
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

ipcMain.handle('hermes:selectPaths', async (_event, options: any = {}) => {
  const properties = options?.directories ? ['openDirectory'] : ['openFile']

  if (options?.multiple !== false) {
    properties.push('multiSelections')
  }

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
    properties: properties as any,
    filters: Array.isArray(options?.filters) ? options.filters : undefined
  })

  if (result.canceled) {
    return []
  }

  return result.filePaths
})

ipcMain.handle('hermes:writeClipboard', (_event, text) => {
  clipboard.writeText(String(text || ''))

  return true
})

ipcMain.handle('hermes:saveImageFromUrl', (_event, url) => saveImageFromUrl(String(url || '')))

ipcMain.handle('hermes:saveImageBuffer', async (_event, payload) => {
  const data = payload?.data

  if (!data) {
    throw new Error('saveImageBuffer: missing data')
  }

  const buffer = Buffer.isBuffer(data) ? data : Buffer.from(data)

  return writeComposerImage(buffer, payload?.ext || '.png')
})

ipcMain.handle('hermes:saveClipboardImage', async () => {
  const image = clipboard.readImage()

  if (image && !image.isEmpty()) {
    return writeComposerImage(image.toPNG(), '.png')
  }

  // WSL2/WSLg doesn't bridge clipboard *images* from the Windows host to the
  // Linux clipboard Electron reads, so a host screenshot looks empty above.
  // Pull it straight off the Windows clipboard via PowerShell as a fallback.
  if (IS_WSL) {
    const png = readWslWindowsClipboardImage()

    if (png) {
      return writeComposerImage(png, '.png')
    }
  }

  return ''
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
  applyTitleBarOverlay(mainWindow)
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

ipcMain.handle('hermes:openPreviewInBrowser', async (_event, url) => {
  if (!(await openPreviewInBrowser(url))) {
    throw new Error('Invalid preview URL')
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

// Reveal a path in the OS file manager (Finder / Explorer / Files).
ipcMain.handle('hermes:fs:reveal', async (_event, targetPath) => {
  const target = String(targetPath || '').trim()

  if (!target) {
    return false
  }

  try {
    shell.showItemInFolder(target)

    return true
  } catch {
    return false
  }
})

// Rename a file/folder in place. The renderer passes the existing path + a new
// base name; the destination is resolved in the SAME parent dir so a rename can
// never move the item elsewhere or traverse out. Rejects on a name collision.
ipcMain.handle('hermes:fs:rename', async (_event, targetPath, newName) => {
  const src = String(targetPath || '').trim()
  const name = String(newName || '').trim()

  if (!src || !name || name === '.' || name === '..' || name.includes('/') || name.includes('\\')) {
    throw new Error('Invalid rename')
  }

  const dst = path.join(path.dirname(src), name)

  if (dst === src) {
    return { path: dst }
  }

  if (fs.existsSync(dst)) {
    throw new Error(`"${name}" already exists`)
  }

  await fs.promises.rename(src, dst)

  return { path: dst }
})

// Write a small UTF-8 text file (e.g. a project's IDEA.md at creation). The path
// is hardened (resolveRequestedPathForIpc) and the parent must already exist —
// this never creates directory trees or escapes the allowed roots, and content
// is size-capped so it can't be abused as a bulk-write primitive.
ipcMain.handle('hermes:fs:writeText', async (_event, filePath, content) => {
  const raw = String(filePath || '').trim()

  if (!raw) {
    throw new Error('Invalid path')
  }

  const text = String(content ?? '')

  if (text.length > 1_000_000) {
    throw new Error('Content too large')
  }

  const resolved = resolveRequestedPathForIpc(expandUserPath(raw), { purpose: 'Write text file' })

  if (!directoryExists(path.dirname(resolved))) {
    throw new Error('Parent directory does not exist')
  }

  await fs.promises.writeFile(resolved, text, 'utf8')

  return { path: resolved }
})

// Move a file/folder to the OS trash (recoverable) — the VS Code "Delete"
// default. `shell.trashItem` routes to Finder/Explorer/Files trash per platform.
ipcMain.handle('hermes:fs:trash', async (_event, targetPath) => {
  const target = String(targetPath || '').trim()

  if (!target) {
    throw new Error('Invalid delete')
  }

  await shell.trashItem(target)

  return true
})

// Git-driven worktree management ("Start work" flow). Errors surface to the
// renderer as rejected promises so it can toast a friendly message.
ipcMain.handle('hermes:git:worktreeList', async (_event, repoPath) => listWorktrees(repoPath, resolveGitBinary()))

ipcMain.handle('hermes:git:worktreeAdd', async (_event, repoPath, options) =>
  addWorktree(repoPath, options || {}, resolveGitBinary())
)

ipcMain.handle('hermes:git:worktreeRemove', async (_event, repoPath, worktreePath, options) =>
  removeWorktree(repoPath, worktreePath, options || {}, resolveGitBinary())
)

ipcMain.handle('hermes:git:branchSwitch', async (_event, repoPath, branch) =>
  switchBranch(repoPath, branch, resolveGitBinary())
)

ipcMain.handle('hermes:git:branchList', async (_event, repoPath) => listBranches(repoPath, resolveGitBinary()))

// Compact repo status (branch, ahead/behind, change counts + files) for the
// composer coding rail. Returns null on a non-repo / remote backend so the rail
// hides cleanly rather than erroring.
ipcMain.handle('hermes:git:repoStatus', async (_event, repoPath) => repoStatus(repoPath, resolveGitBinary()))

// Codex-style review pane: list changed files for a scope, fetch one file's
// unified diff, and stage / unstage / revert. Reads return empty on failure;
// mutations reject so the renderer can toast.
ipcMain.handle('hermes:git:review:list', async (_event, repoPath, scope, baseRef) =>
  reviewList(repoPath, scope, baseRef, resolveGitBinary())
)
ipcMain.handle('hermes:git:review:diff', async (_event, repoPath, filePath, scope, baseRef, staged) =>
  reviewDiff(repoPath, filePath, scope, baseRef, staged, resolveGitBinary())
)
// Working-tree-vs-HEAD diff for one file (the preview's "show the diff" view).
ipcMain.handle('hermes:git:fileDiff', async (_event, repoPath, filePath) =>
  fileDiffVsHead(repoPath, filePath, resolveGitBinary())
)
ipcMain.handle('hermes:git:review:stage', async (_event, repoPath, filePath) =>
  reviewStage(repoPath, filePath ?? null, resolveGitBinary())
)
ipcMain.handle('hermes:git:review:unstage', async (_event, repoPath, filePath) =>
  reviewUnstage(repoPath, filePath ?? null, resolveGitBinary())
)
ipcMain.handle('hermes:git:review:revert', async (_event, repoPath, filePath) =>
  reviewRevert(repoPath, filePath ?? null, resolveGitBinary())
)
ipcMain.handle('hermes:git:review:revParse', async (_event, repoPath, ref) =>
  reviewRevParse(repoPath, ref, resolveGitBinary())
)
ipcMain.handle('hermes:git:review:commit', async (_event, repoPath, message, push) =>
  reviewCommit(repoPath, message, Boolean(push), resolveGitBinary())
)
ipcMain.handle('hermes:git:review:commitContext', async (_event, repoPath) =>
  reviewCommitContext(repoPath, resolveGitBinary())
)
ipcMain.handle('hermes:git:review:push', async (_event, repoPath) => reviewPush(repoPath, resolveGitBinary()))
ipcMain.handle('hermes:git:review:shipInfo', async (_event, repoPath) => reviewShipInfo(repoPath, resolveGhBinary()))
ipcMain.handle('hermes:git:review:createPr', async (_event, repoPath) =>
  reviewCreatePr(repoPath, resolveGitBinary(), resolveGhBinary())
)

// Repo-first project discovery: scan bounded roots for git repos (pure fs walk,
// no native addon). Never throws to the renderer — failures yield an empty list.
ipcMain.handle('hermes:git:scanRepos', async (_event, roots, options) => {
  try {
    return await scanGitRepos(roots || [], options || {})
  } catch {
    return []
  }
})

ipcMain.handle('hermes:terminal:start', async (event, payload = {}) => {
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
    applicationVersion: resolveHermesVersion(),
    copyright: 'Copyright © 2026 Nous Research'
  })
  app.showAboutPanel()
}

ipcMain.handle('hermes:version', async () => ({
  appVersion: resolveHermesVersion(),
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
      if (settled) {
        return
      }
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
        if (code !== 0) {
          return done(fallback())
        }

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
  isQuittingForHandoff = true
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
const HERMES_PROTOCOL = 'hermes'
let _pendingDeepLink = null
let _rendererReadyForDeepLink = false

function _extractDeepLink(argv) {
  if (!Array.isArray(argv)) {
    return null
  }

  return argv.find(a => typeof a === 'string' && a.startsWith(`${HERMES_PROTOCOL}://`)) || null
}

function handleDeepLink(url) {
  if (!url || typeof url !== 'string') {
    return
  }
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
    if (mainWindow.isMinimized()) {
      mainWindow.restore()
    }
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

// Single-instance lock: deep links on a running app (Win/Linux) arrive as a
// second-instance argv. Without the lock a second `hermes://` launch spawns a
// whole new app instead of routing into the running one.
const _gotSingleInstanceLock = app.requestSingleInstanceLock()

if (!_gotSingleInstanceLock) {
  app.quit()
} else {
  app.on('second-instance', (_event, argv) => {
    const url = _extractDeepLink(argv)

    if (url) {
      handleDeepLink(url)
    } else if (mainWindow) {
      if (mainWindow.isMinimized()) {
        mainWindow.restore()
      }
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
  if (IS_MAC) {
    Menu.setApplicationMenu(buildApplicationMenu())
  } else {
    Menu.setApplicationMenu(null)
  }

  installMediaPermissions()
  registerMediaProtocol()
  installEmbedReferer()
  registerDeepLinkProtocol()
  ensureWslWindowsFonts()
  configureSpellChecker()
  registerPowerResumeListeners()
  createWindow()

  // Win/Linux cold start: the launching hermes:// URL is in our own argv.
  const _coldStartLink = _extractDeepLink(process.argv)

  if (_coldStartLink) {
    handleDeepLink(_coldStartLink)
  }

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

let backendQuitReady = false
let backendQuitPromise = null

app.on('before-quit', event => {
  if (backendQuitReady) {
    return
  }
  event.preventDefault()

  if (backendQuitPromise) {
    return
  }

  // The always-on-top overlay isn't a "real" app window; close it so a stray
  // pet can't keep the process alive or float over a quit app.
  closePetOverlay()

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

  const dying = hermesProcess && !hermesProcess.killed ? hermesProcess : null
  stopBackendChild(dying)
  hermesProcess = null

  backendQuitPromise = Promise.all([waitForBackendExit(dying), stopAllPoolBackends()]).finally(() => {
    backendQuitReady = true
    app.quit()
  })
})

app.on('window-all-closed', () => {
  // macOS convention: keep the process alive in the Dock when the user closes
  // the last window. But when we're handing off to a detached updater / swap /
  // uninstall script, the process MUST exit so the script can replace or remove
  // the bundle and relaunch — without this the script's PID-wait spins to its
  // full timeout and the user is left with an invisible app (or an uninstall
  // that appears to do nothing).
  if (process.platform !== 'darwin' || isQuittingForHandoff) {
    app.quit()
  }
})
