import fs from 'node:fs'
import path from 'node:path'

import { appIconCandidates } from './app-icon'
import { hermesManagedNodePathEntries, normalizeHermesHomeRoot } from './backend-env'
import { createDesktopInstallHomeRuntime } from './desktop-install-home-runtime'
import { createDesktopLogRuntime, rotateLogIfNeededSync } from './desktop-log-runtime'
import { DESKTOP_PROFILE_NAME_RE } from './desktop-profile'
import { guestOnboardingEnabled, skipIntroEnabled } from './guest-onboarding'
import { CHROMIUM_LOG_FILENAME, enableLinuxCrashDiagnostics, linuxCrashDiagnostics } from './linux-crash-diagnostics'
import { createNativeAppearanceController } from './native-appearance-controller'
import { planLaunchSwitches, readDesktopLaunchConfig } from './renderer-heap-flags'
import { readWindowsUserEnvVar } from './windows-user-env'

// Runs after pre-ready platform policy and before privileged scheme registration.
// Install identity, launch switches, logging, appearance, and native shell name
// share this order because Chromium and the first window snapshot these values.
export function createDesktopStartupContext(deps: any) {
  const {
    app, BrowserWindow, Menu, crashReporter, nativeTheme,
    APP_ROOT, USER_DATA_OVERRIDE, IS_WINDOWS, IS_MAC, IS_WSL,
    IS_PACKAGED, DARWIN_MAJOR, GLASS_SUPPORTED,
    directoryExists, unpackedPathFor
  } = deps

  const SOURCE_REPO_ROOT = path.resolve(APP_ROOT, '../..')

  const { loadInstallStamp, resolveHermesHome } = createDesktopInstallHomeRuntime({
    APP_ROOT,
    USER_DATA_OVERRIDE,
    IS_WINDOWS,
    app,
    directoryExists,
    normalizeHermesHomeRoot,
    readWindowsUserEnvVar
  })

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

  const HERMES_HOME = resolveHermesHome()

  // #77311: `desktop.electron_flags` and the renderer heap ceiling
  // (`desktop.renderer_max_old_space_mb`) used to reach Chromium only through
  // the `hermes desktop` launcher's argv, so a packaged app opened from its
  // Start-menu / .desktop entry ran with no `--js-flags` at all. Apply them here
  // from config.yaml, before `ready` — Chromium copies `js-flags` to renderer
  // processes only from the browser's pre-launch command line.
  {
    let desktopLaunchYaml = ''

    try {
      desktopLaunchYaml = fs.readFileSync(path.join(HERMES_HOME, 'config.yaml'), 'utf8')
    } catch {
      void 0 // first run: no config yet → Chromium defaults
    }

    for (const planned of planLaunchSwitches(readDesktopLaunchConfig(desktopLaunchYaml), process.argv.slice(1))) {
      if (planned.value === undefined) {
        app.commandLine.appendSwitch(planned.name)
      } else {
        app.commandLine.appendSwitch(planned.name, planned.value)
      }

      console.log(
        `[hermes] desktop launch switch from config.yaml: --${planned.name}${planned.value === undefined ? '' : `=${planned.value}`}`
      )
    }
  }

  function pathWithHermesManagedNode(...entries) {
    const managed = hermesManagedNodePathEntries(HERMES_HOME).filter(directoryExists)

    return [...managed, ...entries, process.env.PATH].filter(Boolean).join(path.delimiter)
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
  // v2 multi-connection registry (named agent sources). Lives BESIDE
  // connection.json — v1 stays on disk untouched so older builds sharing the
  // profile keep working; the registry imports from it once and then owns its
  // own file. Same secret posture as connection.json (encrypted tokens, 0600).
  const DESKTOP_CONNECTIONS_REGISTRY_PATH = path.join(app.getPath('userData'), 'connections.json')
  const DESKTOP_INSTALLATION_PATH = path.join(app.getPath('userData'), 'desktop-installation.json')
  const DESKTOP_UPDATE_CONFIG_PATH = path.join(app.getPath('userData'), 'updates.json')
  const DESKTOP_UPDATE_CHECK_CACHE_PATH = path.join(app.getPath('userData'), 'update-check-cache.json')
  const DESKTOP_WINDOW_STATE_PATH = path.join(app.getPath('userData'), 'window-state.json')
  const DESKTOP_BACKEND_OWNERSHIP_PATH = path.join(app.getPath('userData'), 'backend-ownership.json')
  const DESKTOP_MANAGED_SSH_RECOVERY_PATH = path.join(app.getPath('userData'), 'managed-ssh-update-recovery.json')
  // active-profile.json records which Hermes profile the desktop launches its
  // local backend as. When set, startHermes() passes `hermes --profile <name>
  // dashboard …`, which deterministically pins HERMES_HOME (see
  // _apply_profile_override in hermes_cli/main.py) and bypasses the sticky
  // ~/.hermes/active_profile file. Unset (null) preserves the legacy behavior:
  // no --profile flag, so the backend honors active_profile / default.
  const DESKTOP_PROFILE_CONFIG_PATH = path.join(app.getPath('userData'), 'active-profile.json')
  // Mirrors hermes_cli.profiles._PROFILE_ID_RE so we never hand the backend a
  // value its profile resolver would reject and exit on.
  const PROFILE_NAME_RE = DESKTOP_PROFILE_NAME_RE
  // Branch we track for self-update. The GUI work has merged to main, so this
  // tracks main. User can also override at runtime via
  // hermesDesktop.updates.setBranch().
  const DEFAULT_UPDATE_BRANCH = 'main'
  // desktop.log lives under HERMES_HOME/logs/ so it sits next to agent.log,
  // errors.log, gateway.log produced by hermes_logging.setup_logging — one log
  // directory per user, regardless of which UI surface produced the line.
  const DESKTOP_LOG_PATH = path.join(HERMES_HOME, 'logs', 'desktop.log')

  // Native appearance receives rememberLog during module evaluation, so the
  // logging runtime must be created before that controller.
  const { hermesLog, flushDesktopLogBufferSync, rememberLog, startChromiumLogWatcher, stopDesktopLogFlushTimer } =
    createDesktopLogRuntime(DESKTOP_LOG_PATH)
  // Bound desktop.log on disk. It is an append-only forensic log, so a boot loop
  // (version-skew crash -> backend exits instantly -> renderer keeps hitting
  // Retry) appends the full bootstrap transcript every attempt and grows without
  // bound — we have seen it reach ~326 GB and exhaust the disk, which then breaks
  // update/install (no room for git/venv/npm temp files). The cap, the cascade
  // and the discard ceiling live in log-rotation.ts, shared with the Chromium
  // log below.

  // #100573: keep the FATAL line and a local minidump for the next Linux SIGTRAP.
  // Both must be wired before `app` is ready; the log-file switch is inherited by
  // every child process, so a zygote or GPU CHECK lands in the same file.
  // Chromium opens an explicit --log-file with APPEND_TO_OLD_LOG_FILE, so this
  // one accumulates across launches exactly like desktop.log: bound it the same
  // way, and never let optional diagnostics fail the shell's startup.
  const CRASH_DIAGNOSTICS_LOGS_DIR = path.dirname(DESKTOP_LOG_PATH)

  const CRASH_DIAGNOSTICS = linuxCrashDiagnostics(CRASH_DIAGNOSTICS_LOGS_DIR)
  const CHROMIUM_LOG_PATH = path.join(CRASH_DIAGNOSTICS_LOGS_DIR, CHROMIUM_LOG_FILENAME)

  enableLinuxCrashDiagnostics(CRASH_DIAGNOSTICS, CRASH_DIAGNOSTICS_LOGS_DIR, {
    ensureLogsDir: dir => fs.mkdirSync(dir, { recursive: true }),
    reclaimChromiumLog: file => rotateLogIfNeededSync(file),
    appendSwitch: (name, value) => app.commandLine.appendSwitch(name, value),
    startCrashReporter: options => crashReporter.start(options)
  })

  const BOOT_FAKE_MODE = process.env.HERMES_DESKTOP_BOOT_FAKE === '1'
  const BOOT_FAKE_ERROR = process.env.HERMES_DESKTOP_BOOT_FAKE_ERROR || ''
  // Automated teardown (Playwright's app.close(), harness scripts) quits with
  // nobody to answer a modal, so the active-work confirmation would hang the
  // caller instead of letting the process exit. Force quits set this.
  const SKIP_QUIT_CONFIRM = process.env.HERMES_DESKTOP_SKIP_QUIT_CONFIRM === '1'
  // Nous free tier gate, decided ONCE here and stamped onto every backend spawn
  // (desktopBackendSpawnEnv) and the renderer (hermes:launch-flags).
  const GUEST_ONBOARDING = guestOnboardingEnabled()
  const SKIP_INTRO = skipIntroEnabled()

  const BOOT_FAKE_STEP_MS = (() => {
    const raw = Number.parseInt(String(process.env.HERMES_DESKTOP_BOOT_FAKE_STEP_MS || ''), 10)

    if (!Number.isFinite(raw) || raw <= 0) {
      return 650
    }

    return Math.max(120, raw)
  })()

  const APP_NAME = process.env.HERMES_DESKTOP_APP_NAME || 'Hermes'
  const HUD_WINDOW_TITLE = `${APP_NAME} HUD`
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
  // The apple-touch PNG bakes in the macOS-style ~10% margin, which is correct
  // for the dock but renders visibly smaller than neighboring taskbar icons on
  // Windows, where icons are full-bleed. Windows prefers the full-bleed
  // assets/icon.ico (shipped to resources/ via extraResources) and only falls
  // back to the padded PNG if the ico is missing.
  // The ladder is BUILT once here but each window factory RE-RESOLVES through
  // resolveAppIcon (decoding probe): existence alone is not proof the bytes
  // decode, and an undecodable icon must never take the main process down.
  const APP_ICON_PATHS = appIconCandidates({
    isWindows: IS_WINDOWS,
    appRoot: APP_ROOT,
    resourcesPath: process.resourcesPath,
    unpackedPathFor
  })

  const appearance = createNativeAppearanceController({
    userDataDir: app.getPath('userData'),
    nativeTheme,
    getAllWindows: () => BrowserWindow.getAllWindows(),
    log: rememberLog,
    isMac: IS_MAC,
    isWindows: IS_WINDOWS,
    isWsl: IS_WSL,
    darwinMajor: DARWIN_MAJOR,
    glassSupported: GLASS_SUPPORTED,
    titlebarHeight: TITLEBAR_HEIGHT
  })

  app.setName(APP_NAME)

  // No application menu until the first window exists. Electron would otherwise
  // install its default menu at `will-finish-launching` (before `ready`), and a
  // key equivalent routed through that menu's delegate with no window open
  // segfaults the macOS shell — the updater relaunch races the user's keystroke
  // (#115332). Must run at module scope: on macOS a later `null` never removes
  // an installed menu. The real menu lands in installApplicationMenuAfterFirstWindow.
  Menu.setApplicationMenu(null)

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

  return {
    SOURCE_REPO_ROOT, INSTALL_STAMP, loadInstallStamp, HERMES_HOME, pathWithHermesManagedNode,
    ACTIVE_HERMES_ROOT, VENV_ROOT, BOOTSTRAP_COMPLETE_MARKER,
    BOOTSTRAP_MARKER_SCHEMA_VERSION,
    DESKTOP_CONNECTION_CONFIG_PATH, DESKTOP_CONNECTIONS_REGISTRY_PATH,
    DESKTOP_INSTALLATION_PATH, DESKTOP_UPDATE_CONFIG_PATH,
    DESKTOP_UPDATE_CHECK_CACHE_PATH, DESKTOP_WINDOW_STATE_PATH,
    DESKTOP_BACKEND_OWNERSHIP_PATH, DESKTOP_MANAGED_SSH_RECOVERY_PATH,
    DESKTOP_PROFILE_CONFIG_PATH, PROFILE_NAME_RE, DEFAULT_UPDATE_BRANCH,
    DESKTOP_LOG_PATH, hermesLog, flushDesktopLogBufferSync, rememberLog,
    startChromiumLogWatcher, stopDesktopLogFlushTimer,
    CRASH_DIAGNOSTICS, CHROMIUM_LOG_PATH,
    BOOT_FAKE_MODE, BOOT_FAKE_ERROR, SKIP_QUIT_CONFIRM,
    GUEST_ONBOARDING, SKIP_INTRO, BOOT_FAKE_STEP_MS,
    APP_NAME, HUD_WINDOW_TITLE, WINDOW_BUTTON_POSITION,
    APP_ICON_PATHS, appearance
  }
}
