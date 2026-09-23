import { execFileSync, spawn } from 'node:child_process'
import crypto from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import tls from 'node:tls'
import { pathToFileURL } from 'node:url'

import {
  app,
  BrowserWindow,
  clipboard,
  crashReporter,
  dialog,
  net as electronNet,
  webContents as electronWebContents,
  ipcMain,
  Menu,
  nativeTheme,
  powerMonitor,
  powerSaveBlocker,
  protocol,
  safeStorage,
  screen,
  session,
  shell,
  systemPreferences
} from 'electron'

import { destroyKeepaliveAgents, readStatusCode } from './api-transport'
import { resolveAppIcon } from './app-icon'
import { installApplicationMenuAfterFirstWindow } from './application-menu-startup'
import { stopBackendChild as stopBackendChildImpl } from './backend-child'
import {
  createBackendOutputTail,
  execText,
  formatBackendExitLine,
  probeStartMarker,
  processStartMarker
} from './backend-claim'
import { createBackendConnectionState } from './backend-connection-state'
import { BackendDialClaims } from './backend-dial-claim'
import { createBackendExitRecoveryLatch } from './backend-exit-recovery'
import { isReauthRequiredError } from './backend-health'
import { createBackendShutdownCoordinator } from './backend-ownership'
import { waitForDashboardPortAnnouncement } from './backend-ready'
import { recycleOwnedBackend } from './backend-recycle'
import {
  isHostKeyChangedBootFailure,
  isRetryableRemoteBootFailure,
  shouldLatchBackendStartFailure,
  shouldLatchHostKeyChangedFailure,
  shouldLatchRemoteReauthFailure
} from './backend-start-failure'
import {
  isWindowsBinaryPathInWsl,
  isWslEnvironment
} from './bootstrap-platform'
import { detectBundleSwap } from './bundle-swap'
import { registerChatOnboardingWindow } from './chat-onboarding-window'
import { installCommandScreenshot } from './command-screenshot'
import {
  connectionInstallIds,
  sshInventoryAttemptedAt,
  sshRosterCache
} from './connection-caches'
import {
  buildGatewayWsUrlWithTicket,
  gatewayWsUrlIpcResult,
  normalizeSshConfig
} from './connection-config'
import {
  backendScopeKey,
  backendScopePrefix,
  buildAgentRoster,
  rememberSshEnumeration,
  resolveRegistryLocalRoute,
  shouldDeferLocalEnumeration,
  shouldRetrySshInventory,
  updateEligibility
} from './connection-registry'
import { createContentFileRuntime } from './content-file-runtime'
import { describeCrashReason, installCrashForensics } from './crash-forensics'
import { adoptServedDashboardToken } from './dashboard-token'
import { createDesktopAppLifecycleRuntime } from './desktop-app-lifecycle-runtime'
import { createDesktopBackendOwnershipRuntime } from './desktop-backend-ownership-runtime'
import { registerDesktopBootstrapIpc } from './desktop-bootstrap-ipc'
import { createDesktopBootstrapMarkerRuntime } from './desktop-bootstrap-marker-runtime'
import { createDesktopConnectionAdmissionRuntime } from './desktop-connection-admission-runtime'
import { registerDesktopConnectionApiIpc } from './desktop-connection-api-ipc'
import { createDesktopConnectionAssembly } from './desktop-connection-assembly'
import { registerDesktopConnectionAuthIpc } from './desktop-connection-auth-ipc'
import { registerDesktopConnectionDialIpc } from './desktop-connection-dial-ipc'
import { registerDesktopConnectionFleetIpc } from './desktop-connection-fleet-runtime'
import { registerDesktopConnectionRegistryIpc } from './desktop-connection-registry-ipc'
import { createDesktopExternalOpenRuntime } from './desktop-external-open-runtime'
import { registerDesktopFileIpc } from './desktop-file-ipc'
import { createDesktopGatewayReadinessRuntime } from './desktop-gateway-readiness-runtime'
import { createDesktopHeldQuitRuntime } from './desktop-held-quit-runtime'
import { createDesktopHostAttachRuntime } from './desktop-host-attach-runtime'
import {
  createDesktopMediaProtocolRuntime,
  ensureWslWindowsFonts as ensureWslWindowsFontsImpl,
  makeDashboardReadyFile as makeDashboardReadyFileImpl,
  recentHermesLog as recentHermesLogImpl,
  writeFileAtomic
} from './desktop-host-utilities'
import { createDesktopLocalRuntime } from './desktop-local-runtime'
import { createDesktopNativeChromeRuntime } from './desktop-native-chrome-runtime'
import { createDesktopNativePreferencesRuntime, registerDesktopF12PreferenceIpc } from './desktop-native-preferences-runtime'
import { createDesktopNativeWindowServicesRuntime } from './desktop-native-window-services-runtime'
import { registerBackendPoolIpc, registerDesktopOperationsIpc, registerWorkspaceAndLogIpc } from './desktop-operational-ipc'
import { registerDesktopPageInteractionIpc } from './desktop-page-interaction-ipc'
import { createDesktopPetOverlayRuntime } from './desktop-pet-overlay-runtime'
import { installDesktopPlatformPreflightRuntime } from './desktop-platform-preflight-runtime'
import { createDesktopPluginCompatNoticeRuntime } from './desktop-plugin-compat-notice-runtime'
import { registerDesktopPluginProfileRoutesIpc } from './desktop-plugin-profile-routes-ipc'
import { createDesktopPoolBackendRuntime } from './desktop-pool-backend-runtime'
import { createDesktopPoolPolicyRuntime } from './desktop-pool-policy-runtime'
import { createDesktopPowerRuntime } from './desktop-power-runtime'
import { createDesktopPrimaryBackendRuntime } from './desktop-primary-backend-runtime'
import { createDesktopPrimaryTeardownRuntime } from './desktop-primary-teardown-runtime'
import { createDesktopPrimaryWindowRuntime } from './desktop-primary-window-runtime'
import {
  type DesktopProfileRoute
} from './desktop-profile'
import { createDesktopProfileMutationRuntime } from './desktop-profile-mutation-runtime'
import { registerDesktopProfileRoutingIpc } from './desktop-profile-routing-ipc'
import { registerDesktopQuickEntryIpc } from './desktop-quick-entry-ipc'
import { registerDesktopQuitRuntime } from './desktop-quit-runtime'
import { resolveDesktopRemoteRoute } from './desktop-remote-route'
import { createDesktopRendererAssetsRuntime } from './desktop-renderer-assets-runtime'
import { createDesktopRuntimeDiscovery } from './desktop-runtime-discovery'
import { createDesktopSecondaryWindowRuntime } from './desktop-secondary-window-runtime'
import { createDesktopShellOverlayRuntime } from './desktop-shell-overlay-runtime'
import { createDesktopShellRuntime } from './desktop-shell-runtime'
import { createDesktopStartupContext } from './desktop-startup-context'
import { createDesktopUpdateCheckRuntime } from './desktop-update-check-runtime'
import { createDesktopWindowEventsRuntime } from './desktop-window-events-runtime'
import { registerDesktopWindowIpcRuntime } from './desktop-window-ipc-runtime'
import { createDesktopWindowWiringRuntime } from './desktop-window-wiring-runtime'
import { createDesktopWorkspaceCwdRuntime } from './desktop-workspace-cwd-runtime'
import { installEmbedReferer } from './embed-referer'
import { createAmbientClaimArbiter } from './event-dedupe'
import { createExecutableDiscoveryRuntime } from './executable-discovery-runtime'
import {
  buildTerminalScript,
  resolveTerminalLaunch,
  terminalScriptEnv,
  terminalScriptExtension,
  tuiResumeArgs
} from './external-terminal'
import {
  installFindShortcut,
  installFoundInPageForwarder,
  performFindAfterIndexingStarted,
  stopFind
} from './find-in-page'
import { createFirstRunBootRuntime } from './first-run-boot-runtime'
import { registerFsIpc } from './fs-ipc'
import { createGatewayFileRuntime } from './gateway-file-runtime'
import { createGatewayJsonRuntime } from './gateway-json-runtime'
import { probeGatewayWebSocket } from './gateway-ws-probe'
import { registerGitIpc } from './git-ipc'
import { desktopBackendSpawnEnv } from './guest-onboarding'
import { readAndConsumeHandoffResult } from './handoff-result'
import {
  enableBasicPasswordStoreEncryption,
  resolveReadableFileForIpc,
  resolveRequestedPathForIpc
} from './hardening'
import { assertNoSecondLocalBackend } from './host-backend-singleton'
import { registerHudIpc } from './hud-ipc'
import { installHudModifierTap } from './hud-modifier'
import { createIntroRevealWindowController } from './intro-reveal-window'
import { createLinkMetadataRuntime } from './link-metadata-runtime'
import { notifyLauncherWindowRevealed } from './linux-launcher-ready'
import { createLocalBackendLifecycle, waitForTeardown } from './local-backend-lifecycle'
import { ensureMainWindow } from './main-window-lifecycle'
import { createManagedSshLifecycleRuntime } from './managed-ssh-lifecycle-runtime'
import {
  assertManagedUpdatePreflightClear,
  executeManagedRemoteUpdate,
  managedSshScopeRole,
  refusedManagedSshUpdate,
  runManagedSshUpdate,
  waitForManagedRemoteClearance,
  waitForManagedSshBootstrapFence,
  waitForManagedUpdateOperations
} from './managed-ssh-update'
import { registerMcpOauthCallbackIpc } from './mcp-oauth-callback-ipc'
import { createMediaProtocolHandler, MEDIA_PROTOCOL } from './media-protocol'
import { fetchLocalMedia } from './media-range'
import { registerNativeNotifications } from './notification-ipc'
import { parentWatchdogEnv } from './parent-process-identity'
import { registerPetOverlayIpc } from './pet-overlay-ipc'
import {
  pendingNotice as pendingPluginCompatNotice,
  recordDismissed as recordPluginCompatDismissed
} from './plugin-compat-notice'
import { createPoolRetirer } from './pool-retire'
import { createPoolRetirementClient } from './pool-retire-http'
import {
  BackgroundSlotRetryDeferredError,
  isBackgroundSlotWaitTimeout,
  type LocalBackendSpawnPriority,
  registerLocalBackendExitFinalizer,
  releaseLocalBackendSlotAfterExit
} from './pool-spawn-coordinator'
import { createPoolStopper } from './pool-stop'
import { createKeepAwake } from './power-save'
import { createPreviewTargetRuntime, registerPreviewTargetIpc } from './preview-target-runtime'
import {
  createPrimaryRemoteConnection,
  FirstRunSetupResetError,
  runPrimaryBackendStartup
} from './primary-backend-startup'
import {
  assertLocalProfileCanStart,
  localProfilePoolKeys,
  ProfileDeletionGate
} from './profile-delete-routing'
import { sanitizeQuickEntrySettings } from './quick-entry'
import { createQuitFinalization } from './quit-finalization'
import { type ActiveWork, mergeActiveWork, normalizeActiveWork } from './quit-guard'
import { backendQuitNeedsWait, createQuitTeardownCoordinator } from './quit-teardown'
import * as remoteLifecycle from './remote-lifecycle'
import {
  attachPowerResumeRemoteRevalidation,
  RemoteLivenessTracker,
  RemoteRevalidationCoordinator
} from './remote-liveness'
import { rosterSourceEnumerationTimeoutMs } from './remote-oauth-ticket'
import {
  createRegistryGatewayWsUrlHandler
} from './remote-ws-headers'
import { missingRendererAssets } from './renderer-bundle'
import { loadRendererLoadErrorPage } from './renderer-load-error-page'
import { attachRendererConsoleCapture, formatRendererBoundaryReport } from './renderer-log'
import { fetchRosterSourceData } from './roster-source-fetch'
import { chatWindowWebPreferences } from './session-windows'
import { ensureLoginShellPath } from './shell-path'
import { createSshProbeConnection } from './ssh-connection'
import { createStreamThrottle } from './stream-throttle'
import { registerTerminalIpc } from './terminal-ipc'
import { nativeOverlayWidth as computeNativeOverlayWidth } from './titlebar-overlay-width'
import { glassSupportedOn, translucencySupportedOn } from './translucency'
import { waitForUpdateClearance } from './update-gate'
import { createUpdateGateRuntime, UPDATE_WAIT_POLL_MS, UPDATE_WAIT_TIMEOUT_MS } from './update-gate-runtime'
import { createUpdateHandoffRuntime } from './update-handoff-runtime'
import { readLiveUpdateMarker } from './update-marker'
import { fetchMarketplaceThemes, searchMarketplaceThemes } from './vscode-marketplace'
import { createWakeIndicatorWindowController } from './wake-indicator-window'
import { readWindowBelow } from './window-below'
import { bindWindowChromeEvents } from './window-chrome-events'
import {
  WindowConnectionRouteRegistry
} from './window-connection-route'
import { registerWindowControlIpc, windowControlState } from './window-controls'
import { createWindowOpenHandler } from './window-open-policy'
import { installWindowRendererLifecycle } from './window-renderer-lifecycle'
import { createWindowRevealController } from './window-reveal'
import {
  bindGeometryPersistence,
  computeWindowOptions,
  debounce,
  sanitizeWindowState,
  MIN_HEIGHT as WINDOW_MIN_HEIGHT,
  MIN_WIDTH as WINDOW_MIN_WIDTH
} from './window-state'
import { hiddenWindowsChildOptions } from './windows-child-options'
import {
  detectRemotePlatform,
  probeWindowsRemote,
  terminateOwnedWindowsDashboardForUpdate
} from './windows-remote-lifecycle'
import {
  alreadyHasNoSandbox,
  buildNoSandboxRelaunchArgs,
  fallbackMarker,
  markerAfterSuccessfulBoot,
  shouldRelaunchForRendererSandboxCrashLoop,
  writeSandboxMarker
} from './windows-sandbox-fallback'
import { installWindowsSystemCaTrust } from './windows-system-ca'
import { setActiveGatewayProfile, setWslBridgeProfileState } from './wsl-path-bridge'
import {
  DEFAULT_ZOOM_LEVEL,
  installZoomReassertOnNavigation,
  installZoomReassertOnWindowEvents,
  percentToZoomLevel,
  zoomLevelToPercent,
  zoomWiringForWindowKind
} from './zoom'

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
// Glass: macOS vibrancy, or Windows 11 22H2+ system backdrop. Computed once
// so the renderer, the persisted default, and every chat window agree.
const GLASS_SUPPORTED = glassSupportedOn(process.platform, os.release())
// Clear rides setOpacity, a documented no-op on Linux, so neither mode works
// there and Settings drops the row entirely.
const TRANSLUCENCY_SUPPORTED = translucencySupportedOn(process.platform)
const APP_ROOT = app.getAppPath()

// Device-local preference: block F12 from opening DevTools.
// Set dynamically via IPC from the renderer Settings → Advanced.
let f12Blocked = false

// Preload must be plain JS — Electron's sandbox can't run .ts, and tsx's
// ESM loader is broken on Electron 40's Node (ERR_INVALID_RETURN_PROPERTY_VALUE).
// Dev (`npm run dev`) and prod both load the esbuild output from dist/.
const PRELOAD_PATH = path.join(APP_ROOT, 'dist', 'electron-preload.js')
const PREVIEW_GUEST_PRELOAD_PATH = path.join(APP_ROOT, 'dist', 'preview-guest-preload.js')

const { sandboxState } = installDesktopPlatformPreflightRuntime({
  app,
  ipcMain,
  devServer: DEV_SERVER,
  exitAfterBackendShutdown,
  isPackaged: IS_PACKAGED,
  isWindows: IS_WINDOWS,
  isWsl: IS_WSL
})

const startup = createDesktopStartupContext({
  app, BrowserWindow, Menu, crashReporter, nativeTheme,
  APP_ROOT, USER_DATA_OVERRIDE, IS_WINDOWS, IS_MAC, IS_WSL,
  IS_PACKAGED, DARWIN_MAJOR, GLASS_SUPPORTED,
  directoryExists, unpackedPathFor
})

const {
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
} = startup

// Custom scheme for streaming audio/video into the renderer. Local paths read
// from this machine; remote paths are proxied through the configured gateway
// with main-process authentication. This avoids whole-file data URLs and keeps
// playback seekable and Range-aware. Must be registered before app readiness.
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
  return desktopMediaProtocolRuntime.registerMediaProtocol()
}

let mainWindow = null
const backendConnectionState = createBackendConnectionState<ReturnType<typeof spawn>, any>()

const localBackendLifecycle = createLocalBackendLifecycle<ReturnType<typeof spawn>>({
  stopChild: child => {
    if (child.exitCode === null && child.signalCode === null) {
      stopBackendChildImpl(child, { forceKillProcessTree, isWindows: IS_WINDOWS })
    }
  },
  waitForExit: child => waitForBackendExit(child),
  cancelSetup: () => {
    firstRunBoot.resetExistingSetupGateForRetry()
    bootstrapAbortController?.abort()
  }
})

function spawnOwnedBackend(...args: Parameters<typeof spawn>) {
  const child = localBackendLifecycle.spawn(() => spawn(...args))
  child.once('exit', () => localBackendLifecycle.release(child))
  child.once('error', () => {
    if (!child.pid) {
      localBackendLifecycle.release(child)
    }
  })

  return child
}

const remoteLiveness = new RemoteLivenessTracker()
const remoteRevalidation = new RemoteRevalidationCoordinator()
const registryDispatchRevalidation = new RemoteRevalidationCoordinator()
// Single-owner reconnect/dial claim (#90812): reconnectGateway()'s in-flight
// lock is per-renderer, so two windows racing one wake can both invoke the
// backend ensure IPC and double-dial a pooled SSH backend. Main owns backend
// lifecycles, so concurrent dials for one (connectionId, profile) scope
// coalesce here — the second caller awaits the first spawn's result.
const backendDialClaims = new BackendDialClaims()
// True while connection-config:apply soft-rehomes the primary — suppresses the
// backend-exit toast so an intentional kill doesn't look like a crash.
// Primary-slot bookkeeping for the exit supervisor (#112344). `primaryStartsInFlight`
// counts startHermes() calls that have not settled; `primaryRecoverySuppressed`
// is set by every intentional invalidate of the slot and cleared by the next
// startHermes(), so the dying child's stale exit never respawns behind a
// re-home, a quit, or a latched boot failure.
let primaryStartsInFlight = 0
let primaryRecoverySuppressed = false
const primaryExitRecovery = createBackendExitRecoveryLatch()
// Additional per-profile backends, keyed by profile name. The PRIMARY backend
// (the desktop's launch profile) stays managed by backendConnectionState +
// startHermes(); this pool only holds EXTRA profile
// backends spawned lazily when a session belongs to a different profile. A user
// with no named profiles never populates this map, so their experience is
// byte-for-byte the single-backend behavior.
const backendPool = new Map() // profile -> { process, port, token, connectionPromise, lastActiveAt }
const profileDeletionGate = new ProfileDeletionGate()

// The pool map and idle timer remain main-process state shared with shutdown.
let poolIdleReaper: ReturnType<typeof setInterval> | null = null

const {
  localBackendSpawnCoordinator,
  backgroundSlotRetryBackoff,
  POOL_SLOT_WAIT_MS,
  spawnPriorityFrom,
  takeForegroundSpawn,
  promotePoolEntry,
  logPoolSpawnFailure,
  applySpawnPriority,
  poolMaxBackends,
  setPoolLimits,
  touchPoolBackend,
  evictLruPoolBackends,
  startPoolIdleReaper,
  getPoolLimits
} = createDesktopPoolPolicyRuntime({
  app,
  backendPool,
  fs,
  getPoolRetirer: () => poolRetirer,
  getPoolIdleReaper: () => poolIdleReaper,
  setPoolIdleReaper: timer => {
    poolIdleReaper = timer
  },
  rememberLog,
  stopPoolBackend: key => stopPoolBackend(key)
})

// Auto-reload budget for renderer crashes, shared by EVERY window (primary,
// secondary session, instance) so a crash loop anywhere is suppressed after
// the same budget instead of reloading per-window forever. A deterministic
// startup crash would otherwise loop forever (reload → crash → reload),
// pinning CPU and spamming logs. Allow a few reloads per rolling window, then
// stop and leave the dead window so the user can read the error / quit.
const RENDERER_RELOAD_WINDOW_MS = 60_000
const RENDERER_RELOAD_MAX = 3
const rendererReloadTimesRef: { current: number[] } = { current: [] }
// Latched bootstrap failure: when the first-launch install fails, we hold
// onto the error so subsequent startHermes() calls (e.g. the renderer's
// ensureGatewayOpen retrying after the WS won't open) return the same error
// instead of re-running install.ps1 in a hot loop. Cleared explicitly by
// the renderer's "Reload and retry" path or by quitting the app.
let bootstrapFailure = null
// Latched non-bootstrap backend spawn failure — stops getConnection() from
// respawning hermes serve backend children in a tight loop while boot is broken.
let backendStartFailure = null
// Latched CONFIRMED remote reauth failure. Remote failures deliberately do not
// latch via backendStartFailure (they're usually transient and must stay
// retryable), but a rejected session cannot self-heal — and the non-latching
// path actively breaks recovery: each retry re-emits running:true and hides
// the boot-failure overlay, so the "Sign in" button flickers away before it
// can be clicked. Cleared on every recovery path and on a confirmed sign-in.
let remoteReauthFailure = null
// Active first-launch install, so the renderer's Cancel button (and app quit)
// can abort the in-flight install.sh/ps1 instead of leaving it running.
let bootstrapAbortController = null
// Explicit "the user asked for a repair" flag. Repair used to signal intent by
// deleting the bootstrap marker, which stranded healthy installs whose only
// problem was a transient backend error (#72166). Intent now lives here, so
// repair can force the installer without destroying provenance about how the
// install was created. Cleared once the reinstall is under way.
let bootstrapRepairRequested = false
// Counter for in-flight repair attempts. Reset on a clean boot completion
// (see runBootstrap -> ensureRuntime resolve path). Each successive repair
// in the same failure episode increments this; once it crosses
// MAX_BOOTSTRAP_REPAIR_SOFT_ATTEMPTS the guard escalates from "soft restart"
// to "hard reinstall" so a transient backend stall (issue #74874) stops
// looping the user through a destructive venv reinstall.
let bootstrapRepairAttempt = 0
const MAX_BOOTSTRAP_REPAIR_SOFT_ATTEMPTS = 3
let previewShortcutActive = false

const firstRunBoot = createFirstRunBootRuntime({
  activeRoot: ACTIVE_HERMES_ROOT,
  fakeMode: BOOT_FAKE_MODE,
  fakeStepMs: BOOT_FAKE_STEP_MS,
  getMainWindow: () => mainWindow,
  getRemoteReauthFailure: () => (remoteReauthFailure ? remoteReauthFailure.message : null),
  log: rememberLog,
  platform: process.platform
})

installCrashForensics({ flush: flushDesktopLogBufferSync, log: rememberLog })

// A rejected loadURL leaves a blank window and, unhandled, no trace anywhere
// the user can send us. `label` names the surface so the log says which one.
function loadWindowUrl(win, url, label) {
  win.loadURL(url).catch(error => rememberLog(`${label} failed to load: ${describeCrashReason(error)}`))
}

const { openExternalUrl, openPreviewInBrowser } = createDesktopExternalOpenRuntime({
  IS_WSL,
  shell,
  spawn,
  pathToFileURL,
  resolveRequestedPathForIpc,
  rememberLog
})

function ensureWslWindowsFonts() {
  return ensureWslWindowsFontsImpl({ isWsl: IS_WSL, fs, path, app, spawn, rememberLog })
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

// The apply path keeps the same dwell interval while the startup gate lives in
// update-gate-runtime.ts.
const UPDATE_HANDOFF_DWELL_MS = 2500

const updateGateRuntime = createUpdateGateRuntime({
  hermesHome: HERMES_HOME,
  isPackaged: IS_PACKAGED,
  installStamp: INSTALL_STAMP,
  loadInstallStamp,
  getUpdateInFlight: () => updateInFlight,
  getHandoffActive: () => isQuittingForHandoff,
  readLiveUpdateMarker,
  readAndConsumeHandoffResult,
  waitForUpdateClearance,
  detectBundleSwap,
  buildNoSandboxRelaunchArgs,
  app,
  dialog,
  shell,
  localBackendLifecycle,
  firstRunBoot,
  rememberLog,
  sendOpenUpdatesRequested,
  exitAfterBackendShutdown
})

const { updateGateDeps, waitForUpdateToFinish } = updateGateRuntime

function unpackedPathFor(filePath) {
  return filePath.replace(/app\.asar(?=$|[\\/])/, 'app.asar.unpacked')
}

const {
  findOnPath,
  isCommandScript,
  unwrapWindowsVenvHermesCommand,
  getBackendArgsForRuntime,
  looksLikeDesktopAppBinary,
  isHermesSourceRoot,
  findPythonForRoot,
  findSystemPython,
  findGitBash,
  getVenvPython,
  venvRootForPython
} = createDesktopRuntimeDiscovery({
  hermesHome: HERMES_HOME,
  isWindows: IS_WINDOWS,
  isWsl: IS_WSL,
  fileExists,
  directoryExists,
  rememberLog
})

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

function makeDashboardReadyFile() {
  return makeDashboardReadyFileImpl(app.getPath('userData'))
}

const { resolveGitBinary, resolveGhBinary } = createExecutableDiscoveryRuntime({
  isWindows: IS_WINDOWS,
  fileExists,
  findOnPath,
  getHomePath: () => app.getPath('home'),
  execFileSync
})

function recentHermesLog() {
  return recentHermesLogImpl(hermesLog)
}

const { readWindowState, schedulePersistWindowState, readZoomState, writeZoomState, getAppIconPath,
  registerNativeWindowServicesIpc } = createDesktopNativeWindowServicesRuntime({
  APP_ICON_PATHS,
  BrowserWindow,
  DESKTOP_WINDOW_STATE_PATH,
  IS_MAC,
  app,
  debounce,
  getMainWindow: () => mainWindow,
  ipcMain,
  readWindowBelow,
  rememberLog,
  resolveAppIcon,
  sanitizeWindowState,
  systemPreferences,
  writeFileAtomic
})

// ─── Self-update (git-pull against the running backend's hermes root) ──────

const {
  checkUpdates,
  emitUpdateProgress,
  firstLine,
  readDesktopUpdateConfig,
  resolveHealedBranch,
  resolveUpdateRoot,
  runGit,
  writeDesktopUpdateConfig
} = createDesktopUpdateCheckRuntime({
  ACTIVE_HERMES_ROOT,
  BrowserWindow,
  DEFAULT_UPDATE_BRANCH,
  DESKTOP_UPDATE_CHECK_CACHE_PATH,
  DESKTOP_UPDATE_CONFIG_PATH,
  IS_PACKAGED,
  IS_WINDOWS,
  SOURCE_REPO_ROOT,
  directoryExists,
  isHermesSourceRoot,
  rememberLog,
  resolveGitBinary,
  writeFileAtomic
})

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

// Quit-guard latches: one while the confirmation is on screen (a second
// Cmd-Q must not stack dialogs), one after the user has said "quit anyway"
// (the app.quit() that follows re-enters before-quit and must pass through).

// Keep shared main-process state and backend teardown call sites wired to the
// update handoff runtime. Backend ownership below uses its process-tree stop.
const updateHandoffRuntime = createUpdateHandoffRuntime({
  hermesHome: HERMES_HOME,
  isWindows: IS_WINDOWS,
  isMac: IS_MAC,
  isPackaged: IS_PACKAGED,
  updateHandoffDwellMs: UPDATE_HANDOFF_DWELL_MS,
  defaultUpdateBranch: DEFAULT_UPDATE_BRANCH,
  app,
  backendConnectionState,
  backendPool,
  directoryExists,
  emitUpdateProgress,
  fileExists,
  getUpdateInFlight: () => updateInFlight,
  setUpdateInFlight: value => {
    updateInFlight = value
  },
  setHandoffActive: value => {
    isQuittingForHandoff = value
  },
  globalRemoteActive: () => globalRemoteActive(),
  localBackendLifecycle,
  pathWithHermesManagedNode,
  readDesktopUpdateConfig,
  rememberLog,
  resolveHealedBranch,
  resolveHermesBackend: backendArgs => resolveHermesBackend(backendArgs),
  resolveUpdateRoot,
  runGit,
  startHermes,
  stopAllPoolBackends
})

const { forceKillProcessTree, releaseBackendLock, applyUpdates, handOffWindowsBootstrapRecovery } = updateHandoffRuntime

const desktopShellRuntime = createDesktopShellRuntime({
  ACTIVE_HERMES_ROOT,
  APP_NAME,
  HERMES_HOME,
  INSTALL_STAMP,
  IS_PACKAGED,
  IS_WINDOWS,
  VENV_ROOT,
  app,
  buildNoSandboxRelaunchArgs,
  exitAfterBackendShutdown,
  fileExists,
  findSystemPython,
  fs,
  getVenvPython,
  hiddenWindowsChildOptions,
  isHermesSourceRoot,
  loadInstallStamp,
  os,
  path,
  process,
  releaseBackendLock,
  rememberLog,
  resolveUpdateRoot,
  runGit,
  setQuittingForHandoff: () => {
    isQuittingForHandoff = true
  },
  spawn
})

// Seed the native About panel with the live Hermes version. This is refreshed
// on every open via the explicit "About" menu handler (refreshAboutPanel), so
// an in-place `hermes update` mid-session is reflected without an app restart;
// the seed here just covers the first open and any non-menu invocation path.
app.setAboutPanelOptions({
  applicationName: APP_NAME,
  applicationVersion: desktopShellRuntime.resolveHermesVersion(),
  copyright: 'Copyright © 2026 Nous Research'
})

const { claimBackendChild, desktopParentStartMarker, reapOrphanedBackendsOnce, releaseBackendChild } =
  createDesktopBackendOwnershipRuntime({
    ownershipPath: DESKTOP_BACKEND_OWNERSHIP_PATH,
    isWindows: IS_WINDOWS,
    execText,
    processStartMarker,
    probeStartMarker,
    forceKillProcessTree,
    stopBackendChild,
    waitForBackendExit,
    rememberLog
  })

const { activeRuntimeState, writeBootstrapMarker } = createDesktopBootstrapMarkerRuntime({
  ACTIVE_HERMES_ROOT,
  VENV_ROOT,
  BOOTSTRAP_COMPLETE_MARKER,
  BOOTSTRAP_MARKER_SCHEMA_VERSION,
  app,
  getVenvPython,
  isHermesSourceRoot,
  fileExists,
  writeFileAtomic
})

const { resolveWebDist, resolveRendererIndexWithMissing, resolveRendererIndex } =
  createDesktopRendererAssetsRuntime({
    APP_ROOT,
    IS_PACKAGED,
    directoryExists,
    fileExists,
    missingRendererAssets,
    rememberLog,
    unpackedPathFor
  })

// True when `dir` lives inside the packaged app bundle / install tree.
// Packaged Electron's process.cwd() (and npm's INIT_CWD when dev tooling
// leaked into a release build) often resolve here — e.g. win-unpacked on
// Windows — which is exactly where PR #37536 item 16 said we must NOT run.
const { resolveHermesCwd, sanitizeWorkspaceCwd, readDefaultProjectDir, writeDefaultProjectDir } =
  createDesktopWorkspaceCwdRuntime({
    app,
    APP_ROOT,
    SOURCE_REPO_ROOT,
    IS_PACKAGED,
    directoryExists,
    rememberLog
  })

const desktopLocalRuntime = createDesktopLocalRuntime({
  hermesHome: HERMES_HOME,
  activeRoot: ACTIVE_HERMES_ROOT,
  venvRoot: VENV_ROOT,
  sourceRepoRoot: SOURCE_REPO_ROOT,
  installStamp: INSTALL_STAMP,
  isWindows: IS_WINDOWS,
  isPackaged: IS_PACKAGED,
  isWsl: IS_WSL,
  fileExists,
  findPythonForRoot,
  venvRootForPython,
  getVenvPython,
  findSystemPython,
  isHermesSourceRoot,
  activeRuntimeState,
  findOnPath,
  isWindowsBinaryPathInWsl,
  looksLikeDesktopAppBinary,
  unwrapWindowsVenvHermesCommand,
  isCommandScript,
  rememberLog,
  localBackendLifecycle,
  firstRunBoot,
  handOffWindowsBootstrapRecovery,
  writeBootstrapMarker,
  resolveGitBinary,
  findGitBash,
  state: {
    get bootstrapFailure() {
      return bootstrapFailure
    },
    set bootstrapFailure(value) {
      bootstrapFailure = value
    },
    get bootstrapAbortController() {
      return bootstrapAbortController
    },
    set bootstrapAbortController(value) {
      bootstrapAbortController = value
    },
    get bootstrapRepairRequested() {
      return bootstrapRepairRequested
    },
    set bootstrapRepairRequested(value) {
      bootstrapRepairRequested = value
    },
    get bootstrapRepairAttempt() {
      return bootstrapRepairAttempt
    },
    set bootstrapRepairAttempt(value) {
      bootstrapRepairAttempt = value
    }
  }
})

const { resolveHermesBackend, ensureRuntime } = desktopLocalRuntime

const { fetchJson, fetchPublicJson } = createGatewayJsonRuntime({ headersForRemoteRequest })

const { mimeTypeForPath, extensionForMimeType, saveImageFromUrl, writeComposerImage } = createContentFileRuntime({
  app,
  dialog,
  getMainWindow: () => mainWindow,
  resolveReadableFileForIpc
})

const { fetchLinkTitle, resolveFaviconCached } = createLinkMetadataRuntime()

const previewTargetRuntime = createPreviewTargetRuntime({
  app,
  directoryExists,
  fileExists,
  getMainWindow: () => mainWindow,
  hermesHome: HERMES_HOME,
  mimeTypeForPath,
  resolveHermesCwd
})

const { expandUserPath } = previewTargetRuntime

const { gatewayAuthProviders, waitForHermes } = createDesktopGatewayReadinessRuntime({
  fetchJson,
  fetchPublicJson,
  fetchJsonViaOauthSession: (url, options) => fetchJsonViaOauthSession(url, options),
  ensureNativeAccessToken: baseUrl => ensureNativeAccessToken(baseUrl)
})

const {
  getWindowState,
  sendClosePreviewRequested,
  sendPreviewNavCommand,
  installBrowserNavGestures,
  sendOpenFolderRequested,
  sendWindowStateChanged
} = createDesktopWindowEventsRuntime({
  DARWIN_MAJOR,
  IS_MAC,
  IS_WINDOWS,
  IS_WSL,
  WINDOW_BUTTON_POSITION,
  computeNativeOverlayWidth,
  electronWebContents,
  getMainWindow: () => mainWindow,
  windowControlState
})

function sendBackendExit(payload) {
  // Intentional soft re-home (gateway mode apply) kills the child on purpose —
  // don't surface the "backend stopped" error toast / boot-failure path.
  if (primaryTeardown.isSoftRehomeInProgress()) {
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

const { registerPowerResumeListeners } = createDesktopPowerRuntime({
  BrowserWindow,
  attachPowerResumeRemoteRevalidation,
  getMainWindow: () => mainWindow,
  ipcMain,
  powerMonitor,
  rememberLog,
  revalidateSuspectPoolAfterResume
})

const { showPluginCompatNoticeOnce } = createDesktopPluginCompatNoticeRuntime({
  HERMES_HOME,
  app,
  dialog,
  getHermesProtocol: () => HERMES_PROTOCOL,
  getMainWindow: () => mainWindow,
  handleDeepLink,
  pendingPluginCompatNotice,
  recordPluginCompatDismissed,
  rememberLog
})

function sendOpenUpdatesRequested() {
  // The renderer mounts its open-updates listener in the same effect pass that
  // signals deep-link readiness. Before that (e.g. a boot-time dialog answered
  // before the window is up) queue the request; 'hermes:deep-link-ready' flushes it.
  if (!_rendererReadyForDeepLink || !mainWindow || mainWindow.isDestroyed()) {
    _pendingOpenUpdates = true

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

const {
  buildApplicationMenu,
  installDevToolsShortcut,
  installPreviewShortcut,
  setAndPersistZoomLevel,
  restorePersistedZoomLevel,
  installZoomShortcuts,
  lastContextMenuPoint,
  installContextMenuBridge,
  installDownloadHandling,
  installMediaPermissions
} = createDesktopNativeChromeRuntime({
  APP_NAME,
  IS_MAC,
  closeHudWindow: () => shellOverlayRuntime.closeHudWindow(),
  extensionForMimeType,
  getCreateInstanceWindow: () => createInstanceWindow,
  getF12Blocked: () => f12Blocked,
  getHudWindow: () => shellOverlayRuntime.getHudWindow(),
  getMainWindow: () => mainWindow,
  readZoomState,
  rememberLog,
  sendClosePreviewRequested,
  sendOpenFolderRequested,
  sendOpenUpdatesRequested,
  sendPreviewNavCommand,
  showAboutPanelFresh,
  writeZoomState
})

const connections = createDesktopConnectionAssembly({
  app, BrowserWindow, electronNet, session, safeStorage,
  DESKTOP_CONNECTION_CONFIG_PATH, DESKTOP_CONNECTIONS_REGISTRY_PATH,
  DESKTOP_INSTALLATION_PATH, DESKTOP_MANAGED_SSH_RECOVERY_PATH,
  DESKTOP_PROFILE_CONFIG_PATH, HERMES_HOME, PROFILE_NAME_RE, GUEST_ONBOARDING,
  fetchJson, fetchPublicJson, rememberLog, writeFileAtomic,
  ensureBackend: profile => ensureBackend(profile),
  ensureRegistryBackend: (connectionId, profile) => ensureRegistryBackend(connectionId, profile),
  stopRegistryConnectionBackends, primaryProfileKey,
  getIsolatedBackend: () => ISOLATED_BACKEND,
  backendDialClaims, waitForHermes,
  getWindowConnectionRoute: webContentsId => windowConnectionRoutes.get(webContentsId),
  disposeTerminalSessionsForSshScope: scope => terminalIpc.disposeTerminalSessionsForSshScope(scope),
  managedSshConfig: (source, profile) => managedSshConfig(source, profile),
  getMainWindow: () => mainWindow,
  startHermes
})

const {
  getOauthSessionForUrl, hasOauthSessionCookie, hasLiveOauthSession,
  clearOauthSession, openOauthLoginWindow, fetchJsonViaOauthSession,
  nativeAccessTokenCoordinator, ensureNativeAccessToken, hasNativeSession,
  postJsonNoAuth, mintGatewayWsTicket, freshGatewayWsUrl,
  resolvePortalBaseUrl, hasLivePortalSession, openPortalLoginWindow,
  discoverCloudAgents, cloudAgentSilentSignIn,
  postJsonForBackend, getJsonForBackend, fetchJsonForBackend,
  sendConnectionApplied, broadcastConnectionsChanged,
  secretStoragePolicy, applySecretStorageEncryption,
  encryptDesktopSecret, decryptDesktopSecret, decryptRemoteHeaders,
  rememberRemoteWsHeaders, installRemoteHeaderRules,
  readDesktopConnectionConfig, writeDesktopConnectionConfig,
  readDesktopConnectionsRegistry, writeDesktopConnectionsRegistry,
  sanitizeConnectionsRegistry, saveRegistryConnection,
  migrateLegacyEncryptedSecretsOnce,
  desktopProfilePreferences, validateDesktopProfileRoute,
  readActiveDesktopProfile, writeActiveDesktopProfile,
  migrateActiveProfileIfMissing, profileRouteOptions,
  sanitizeDesktopConnectionConfig, coerceDesktopConnectionConfig,
  buildRemoteConnection, sshConnections, sshIsolatedKeepalives,
  managedConnectionUpdateGate, managedConnectionUpdates,
  managedConnectionRecoveries, managedPrimaryRestoreOwners,
  managedUpdateQuitState, assertCanMutateManagedPrimaryRouting,
  readManagedSshRecoveryRecords, persistManagedSshRecovery,
  markManagedSshRecoveryLaunching, clearManagedSshRecovery,
  sshBootstrapCoordinator, sshTeardowns, sshScopeKey, sshRememberLog,
  teardownSshConnection, activeSshTerminalTarget, ensureTerminalBackend,
  resetPreviewReach, reachablePreviewUrl, effectiveSshConfigFingerprint,
  bootstrapSshConnection, resolveRemoteBackend, profileHasRemoteOverride,
  configuredRemoteProfileNames, primaryBackendIsRemote,
  fetchJsonForProfile, requestJsonForProfile, probeRemoteAuthMode,
  testDesktopConnectionConfig, fetchConnectionStatus
} = connections

// These two callbacks are handed to earlier runtimes before the connection
// assembly is initialized. They are not invoked during module evaluation.
function headersForRemoteRequest(requestUrl: string) {
  return connections.headersForRemoteRequest(requestUrl)
}

function globalRemoteActive() {
  return connections.globalRemoteActive()
}

const primaryTeardown = createDesktopPrimaryTeardownRuntime({
  firstRunBoot, localBackendLifecycle, rememberLog,
  clearFailures: () => { backendStartFailure = null; remoteReauthFailure = null },
  remoteLiveness,
  suppressPrimaryRecovery: () => { primaryRecoverySuppressed = true },
  backendConnectionState, forceKillProcessTree, IS_WINDOWS,
  readActiveDesktopProfile, backendPool, sshConnections,
  sshBootstrapCoordinator,
  stopPoolBackend: key => stopPoolBackend(key),
  teardownSshConnection
})

const { primaryProfilePin, resetHermesConnection, invalidatePrimaryConnection,
  teardownPrimaryBackendAndWait } = primaryTeardown

// Earlier factories capture these declarations before primary teardown exists.
function stopBackendChild(child) {
  return primaryTeardown.stopBackendChild(child)
}

function waitForBackendExit(child, timeoutMs = 5000) {
  return primaryTeardown.waitForBackendExit(child, timeoutMs)
}

function primaryProfileKey() {
  return primaryTeardown.primaryProfileKey()
}

async function stopRegistryConnectionBackends(connectionId) {
  return primaryTeardown.stopRegistryConnectionBackends(connectionId)
}

// Compose the single pool owner after SSH/bootstrap state and spawn dependencies.
const { releaseLocalBackendSlot, teardownFailedLocalBackend, spawnPoolBackend, poolStopper, stopPoolBackend, poolRetirer } =
  createDesktopPoolBackendRuntime({
    adoptServedDashboardToken,
    assertLocalProfileCanStart,
    assertNoSecondLocalBackend,
    backendPool,
    backgroundSlotRetryBackoff,
    BackgroundSlotRetryDeferredError,
    BrowserWindow,
    claimBackendChild,
    createBackendOutputTail,
    createPoolRetirer,
    createPoolRetirementClient,
    createPoolStopper,
    crypto,
    desktopBackendSpawnEnv,
    desktopParentStartMarker,
    directoryExists,
    ensureRuntime,
    fetchJson,
    formatBackendExitLine,
    fs,
    getBackendArgsForRuntime,
    getWindowState,
    GUEST_ONBOARDING,
    HERMES_HOME,
    hermesLog,
    hiddenWindowsChildOptions,
    isBackgroundSlotWaitTimeout,
    localBackendLifecycle,
    localBackendSpawnCoordinator,
    makeDashboardReadyFile,
    parentWatchdogEnv,
    path,
    POOL_SLOT_WAIT_MS,
    poolMaxBackends,
    probeGatewayWebSocket,
    profileDeletionGate,
    profileRouteOptions,
    reapOrphanedBackendsOnce,
    registerLocalBackendExitFinalizer,
    releaseBackendChild,
    releaseLocalBackendSlotAfterExit,
    rememberLog,
    resolveHermesBackend,
    resolveHermesCwd,
    resolveRemoteBackend,
    resolveWebDist,
    spawnOwnedBackend,
    spawnPriorityFrom,
    sshBootstrapCoordinator,
    sshRememberLog,
    stopBackendChild,
    takeForegroundSpawn,
    teardownSshConnection,
    UPDATE_WAIT_POLL_MS,
    UPDATE_WAIT_TIMEOUT_MS,
    updateGateDeps,
    waitForBackendExit,
    waitForDashboardPortAnnouncement,
    waitForHermes,
    waitForUpdateClearance
  })



// All admission callers above this point are deferred functions or callbacks.
// The pool stopper and retirer are now initialized, so preserve their live
// identities in the admission runtime.
const { ensureBackend, ensureRegistryBackend, connectRegistryBackend } = createDesktopConnectionAdmissionRuntime({
  backendPool,
  bootstrapSshConnection,
  buildRemoteConnection,
  decryptDesktopSecret,
  effectiveSshConfigFingerprint,
  evictLruPoolBackends,
  fetchJsonForBackend,
  getWindowState,
  globalRemoteActive,
  hermesLog,
  localBackendLifecycle,
  logPoolSpawnFailure,
  managedConnectionUpdateGate,
  poolMaxBackends,
  poolRetirer,
  poolStopper,
  primaryProfileKey,
  profileDeletionGate,
  profileHasRemoteOverride,
  profileRouteOptions,
  promotePoolEntry,
  readDesktopConnectionsRegistry,
  registryDispatchRevalidation,
  rememberLog,
  setWslBridgeProfileState,
  spawnPoolBackend,
  spawnPriorityFrom,
  sshBootstrapCoordinator,
  startHermes,
  startPoolIdleReaper,
  stopPoolBackend,
  teardownFailedLocalBackend,
  teardownSshConnection,
  waitForHermes
})

const desktopMediaProtocolRuntime = createDesktopMediaProtocolRuntime({
  createMediaProtocolHandler,
  protocol,
  MEDIA_PROTOCOL,
  ensureNativeAccessToken,
  fetchLocalMedia,
  electronNet,
  getOauthSessionForUrl,
  resolveReadableFileForIpc,
  backendDialClaims,
  backendScopeKey,
  ensureRegistryBackend,
  ensureBackend
})

const { managedSshConfig, updateManagedSshConnection, resumeManagedSshRecoveries } =
  createManagedSshLifecycleRuntime({
    assertManagedUpdatePreflightClear,
    backendConnectionState,
    backendPool,
    backendScopeKey,
    backendScopePrefix,
    clearManagedSshRecovery,
    connectRegistryBackend,
    createSshProbeConnection,
    detectRemotePlatform,
    encryptDesktopSecret,
    executeManagedRemoteUpdate,
    managedConnectionRecoveries,
    managedConnectionUpdateGate,
    managedConnectionUpdates,
    managedPrimaryRestoreOwners,
    managedSshScopeRole,
    markManagedSshRecoveryLaunching,
    normalizeSshConfig,
    persistManagedSshRecovery,
    primaryProfileKey,
    probeWindowsRemote,
    readDesktopConnectionConfig,
    readDesktopConnectionsRegistry,
    readManagedSshRecoveryRecords,
    remoteLifecycle,
    resolveDesktopRemoteRoute,
    runManagedSshUpdate,
    sshBootstrapCoordinator,
    sshConnections,
    sshIsolatedKeepalives,
    sshRememberLog,
    sshScopeKey,
    startHermes,
    startPoolIdleReaper,
    terminalIpc: { disposeTerminalSessionsForSshScope: scope => terminalIpc.disposeTerminalSessionsForSshScope(scope) },
    terminateOwnedWindowsDashboardForUpdate,
    waitForHermes,
    waitForManagedRemoteClearance,
    waitForManagedSshBootstrapFence
  })

const { saveGatewayFile } = createGatewayFileRuntime({
  dialog,
  electronNet,
  ensureBackend,
  ensureRegistryBackend,
  ensureNativeAccessToken,
  fetchJsonForBackend,
  getMainWindow: () => mainWindow,
  getOauthSessionForUrl,
  profileRouteOptions
})

localBackendLifecycle.signal.addEventListener('abort', poolRetirer.dispose, { once: true })

async function teardownPoolBackendAndWait(profile) {
  await Promise.all(localProfilePoolKeys(profile).map(key => stopPoolBackend(key)))
}

async function stopAllPoolBackends() {
  const entries = [...backendPool.values()]
  await poolStopper.stopAll()
  entries.forEach(releaseLocalBackendSlot)
}

const backendShutdown = createBackendShutdownCoordinator(async () => {
  const localShutdown = localBackendLifecycle.shutdown()
  const primary = backendConnectionState.invalidate()

  stopBackendChild(primary)
  const pooledStops = stopAllPoolBackends()

  if (poolIdleReaper) {
    clearInterval(poolIdleReaper)
    poolIdleReaper = null
  }

  await waitForTeardown([localShutdown, waitForBackendExit(primary), pooledStops], 7_000)
})

const quitTeardown = createQuitTeardownCoordinator(() => app.quit())

const quitFinalization = createQuitFinalization({
  isWindows: IS_WINDOWS,
  hardExit: code => {
    rememberLog(`[quit] forcing Windows process exit after Electron quit finalization stalled`)
    app.exit(code)
  }
})

async function teardownSshForQuit() {
  const scopes = [...sshConnections.keys()]

  for (const scope of scopes) {
    void teardownSshConnection(scope || null).catch(error => rememberLog(`SSH teardown failed: ${error.message}`))
  }

  await sshTeardowns.finish(sshBootstrapCoordinator.promises(), () => sshBootstrapCoordinator.forceCleanupAll())
}

async function exitAfterBackendShutdown(code) {
  await backendShutdown.run()
  app.exit(code)
}

const { prepareProfileDeleteRequest, prepareProfileRenameRequest } = createDesktopProfileMutationRuntime({
  profileNameRe: PROFILE_NAME_RE,
  primaryProfileKey,
  writeActiveDesktopProfile,
  teardownPrimaryBackendAndWait,
  teardownPoolBackendAndWait,
  getMainWindow: () => mainWindow,
  startHermes
})

// ── Attach-first: one backend per HOST (multiplex-only) ───────────────────
// Escape hatch: a dedicated, private backend for this app instead of the host's.
const ISOLATED_BACKEND = process.env.HERMES_DESKTOP_ISOLATED_BACKEND === '1'

const {
  stopAttachedBackendMonitor,
  startAttachedBackendMonitor,
  attachToRunningHostBackend,
  releaseHostSpawnReservation
} = createDesktopHostAttachRuntime({
  HERMES_HOME,
  ISOLATED_BACKEND,
  rememberLog,
  waitForHermes,
  invalidatePrimaryConnection,
  scheduleUnexpectedPrimaryRecovery
})

const primaryBackendRuntime = createDesktopPrimaryBackendRuntime({
  BOOT_FAKE_ERROR,
  DESKTOP_LOG_PATH,
  FirstRunSetupResetError,
  GUEST_ONBOARDING,
  HERMES_HOME,
  adoptServedDashboardToken,
  attachToRunningHostBackend,
  backendConnectionState,
  backendShutdown,
  claimBackendChild,
  createBackendOutputTail,
  createPrimaryRemoteConnection,
  crypto,
  desktopBackendSpawnEnv,
  desktopParentStartMarker,
  ensureLoginShellPath,
  ensureRuntime,
  firstLine,
  firstRunBoot,
  formatBackendExitLine,
  fs,
  getBackendArgsForRuntime,
  getWindowState,
  hermesLog,
  hiddenWindowsChildOptions,
  invalidatePrimaryConnection,
  isHostKeyChangedBootFailure,
  isReauthRequiredError,
  isRetryableRemoteBootFailure,
  localBackendLifecycle,
  makeDashboardReadyFile,
  managedPrimaryRestoreOwners,
  migrateActiveProfileIfMissing,
  parentWatchdogEnv,
  primaryBackendIsRemote,
  primaryExitRecovery,
  primaryProfileKey,
  primaryProfilePin,
  probeGatewayWebSocket,
  readActiveDesktopProfile,
  readStatusCode,
  reapOrphanedBackendsOnce,
  recentHermesLog,
  releaseBackendChild,
  releaseHostSpawnReservation,
  rememberLog,
  resolveHermesBackend,
  resolveHermesCwd,
  resolveRemoteBackend,
  resolveWebDist,
  runPrimaryBackendStartup,
  sendBackendExit,
  setActiveGatewayProfile,
  setWslBridgeProfileState,
  shouldLatchBackendStartFailure,
  shouldLatchHostKeyChangedFailure,
  shouldLatchRemoteReauthFailure,
  showPluginCompatNoticeOnce,
  spawnOwnedBackend,
  startAttachedBackendMonitor,
  stopAttachedBackendMonitor,
  stopBackendChild,
  waitForBackendExit,
  waitForDashboardPortAnnouncement,
  waitForHermes,
  waitForUpdateToFinish,
  state: {
    get isPrimaryInstance() {
      return isPrimaryInstance
    },
    get isQuittingForHandoff() {
      return isQuittingForHandoff
    },
    get primaryStartsInFlight() {
      return primaryStartsInFlight
    },
    set primaryStartsInFlight(value) {
      primaryStartsInFlight = value
    },
    get primaryRecoverySuppressed() {
      return primaryRecoverySuppressed
    },
    set primaryRecoverySuppressed(value) {
      primaryRecoverySuppressed = value
    },
    get bootstrapFailure() {
      return bootstrapFailure
    },
    set bootstrapFailure(value) {
      bootstrapFailure = value
    },
    get backendStartFailure() {
      return backendStartFailure
    },
    set backendStartFailure(value) {
      backendStartFailure = value
    },
    get remoteReauthFailure() {
      return remoteReauthFailure
    },
    set remoteReauthFailure(value) {
      remoteReauthFailure = value
    },
    get bootstrapRepairAttempt() {
      return bootstrapRepairAttempt
    },
    set bootstrapRepairAttempt(value) {
      bootstrapRepairAttempt = value
    }
  }
})

function startHermes(options: { supervisorRecovery?: boolean } = {}) {
  return primaryBackendRuntime.startHermes(options)
}

function scheduleUnexpectedPrimaryRecovery(
  options: { code?: number | null; error?: string | null; ready?: boolean; signal?: string | null } = {}
) {
  return primaryBackendRuntime.scheduleUnexpectedPrimaryRecovery(options)
}

const { wireCommonWindowHandlers, installPreviewGuestPreload, wireWindowReveal } =
  createDesktopWindowWiringRuntime({
    DEV_SERVER,
    PREVIEW_GUEST_PRELOAD_PATH,
    app,
    createWindowOpenHandler,
    createWindowRevealController,
    installBrowserNavGestures,
    installContextMenuBridge,
    installDevToolsShortcut,
    installFindShortcut,
    installPreviewShortcut,
    installZoomReassertOnNavigation,
    installZoomReassertOnWindowEvents,
    installZoomShortcuts,
    openExternalUrl,
    rememberLog,
    restorePersistedZoomLevel
  })

const { minimizeToTray, focusWindow, createSessionWindow, createBrowserWindow, createInstanceWindow } =
  createDesktopSecondaryWindowRuntime({
    DEV_SERVER,
    IS_MAC,
    PRELOAD_PATH,
    RENDERER_RELOAD_MAX,
    RENDERER_RELOAD_WINDOW_MS,
    WINDOW_BUTTON_POSITION,
    appearance,
    createWindow,
    ensureMainWindow,
    getAppIconPath,
    getIsQuittingForHandoff: () => isQuittingForHandoff,
    getMainWindow: () => mainWindow,
    getStreamThrottle: () => streamThrottle,
    getWindowConnectionRoutes: () => windowConnectionRoutes,
    loadWindowUrl,
    primaryProfileKey,
    readWindowState,
    recordWindowConnectionRoute,
    rememberLog,
    rendererReloadTimesRef,
    resolveRendererIndex,
    sendWindowStateChanged,
    validateDesktopProfileRoute,
    wireCommonWindowHandlers,
    wireWindowReveal
  })

// A macOS-only ambient wake cue. It is deliberately a gateway-less helper
// window: the active renderer owns voice state and sends only the visual phase.
const wakeIndicatorController = createWakeIndicatorWindowController({
  devServer: DEV_SERVER,
  isMac: IS_MAC,
  loadWindowUrl,
  log: rememberLog,
  preloadPath: PRELOAD_PATH,
  rendererIndex: resolveRendererIndex,
  wireWindow: window => wireCommonWindowHandlers(window, zoomWiringForWindowKind('wakeIndicator'))
})

const introRevealController = createIntroRevealWindowController({
  devServer: DEV_SERVER,
  enabled: GUEST_ONBOARDING,
  isMac: IS_MAC,
  loadWindowUrl,
  log: rememberLog,
  mainWindow: () => mainWindow,
  preloadPath: PRELOAD_PATH,
  rendererIndex: resolveRendererIndex,
  showMain: () => {
    mainWindow.show()
    mainWindow.focus()
  },
  wireWindow: window => wireCommonWindowHandlers(window, zoomWiringForWindowKind('petOverlay'))
})

registerChatOnboardingWindow({
  enabled: GUEST_ONBOARDING,
  mainWindow: () => mainWindow
})

const { getPetOverlayWindow, openPetOverlay, closePetOverlay } = createDesktopPetOverlayRuntime({
  DEV_SERVER,
  IS_MAC,
  PRELOAD_PATH,
  getMainWindow: () => mainWindow,
  loadWindowUrl,
  rememberLog,
  resolveRendererIndex,
  wireCommonWindowHandlers,
  wireWindowReveal
})

const shellOverlayRuntime = createDesktopShellOverlayRuntime({
  DEV_SERVER,
  HUD_WINDOW_TITLE,
  IS_MAC,
  PRELOAD_PATH,
  bindGeometryPersistence,
  focusWindow,
  getMainWindow: () => mainWindow,
  getStreamThrottle: () => streamThrottle,
  loadWindowUrl,
  rememberLog,
  resolveRendererIndex,
  wireCommonWindowHandlers,
  wireWindowReveal,
  writeFileAtomic
})

const {
  applyQuickEntrySettings,
  closeHudWindow,
  closeQuickEntryWindow,
  hideQuickEntryWindow,
  openHudWindow,
  readQuickEntrySettings,
  resetHudWindowLayout,
  writeQuickEntrySettings
} = shellOverlayRuntime

const primaryWindowRuntime = createDesktopPrimaryWindowRuntime({
  app,
  BrowserWindow,
  screen,
  DEV_SERVER,
  IS_MAC,
  IS_WINDOWS,
  PRELOAD_PATH,
  RENDERER_RELOAD_MAX,
  RENDERER_RELOAD_WINDOW_MS,
  WINDOW_BUTTON_POSITION,
  WINDOW_MIN_HEIGHT,
  WINDOW_MIN_WIDTH,
  alreadyHasNoSandbox,
  appearance,
  attachRendererConsoleCapture,
  backendShutdown,
  bindGeometryPersistence,
  bindWindowChromeEvents,
  buildNoSandboxRelaunchArgs,
  chatWindowWebPreferences,
  clearRendererReadyForDeepLink: () => {
    _rendererReadyForDeepLink = false
  },
  closePetOverlay,
  computeWindowOptions,
  connectDesktopProfileRoute,
  desktopProfilePreferences,
  exitAfterBackendShutdown,
  fallbackMarker,
  firstRunBoot,
  getAppIconPath,
  getIsQuittingForHandoff: () => isQuittingForHandoff,
  getMainWindow: () => mainWindow,
  getStreamThrottle: () => streamThrottle,
  installWindowRendererLifecycle,
  introRevealController,
  loadRendererLoadErrorPage,
  loadWindowUrl,
  markerAfterSuccessfulBoot,
  minimizeToTray,
  notifyLauncherWindowRevealed,
  readWindowState,
  recordWindowConnectionRoute,
  rememberLog,
  rendererReloadTimesRef,
  resolveRendererIndex,
  resolveRendererIndexWithMissing,
  sandboxState,
  schedulePersistWindowState,
  sendWindowStateChanged,
  setMainWindow: window => {
    mainWindow = window
  },
  shouldRelaunchForRendererSandboxCrashLoop,
  startHermes,
  wakeIndicatorController,
  wireCommonWindowHandlers,
  wireWindowReveal,
  writeSandboxMarker,
  zoomWiringForWindowKind
})

// The secondary-window runtime receives this declaration before the primary
// runtime is initialized; the callback is invoked only after app startup.
function createWindow() {
  return primaryWindowRuntime.createWindow()
}

const windowConnectionRoutes = new WindowConnectionRouteRegistry()

const {
  connectDesktopProfileRoute: connectDesktopProfileRouteImpl,
  recordWindowConnectionRoute: recordWindowConnectionRouteImpl,
  revalidateSuspectPoolAfterResume: revalidateSuspectPoolAfterResumeImpl
} = registerDesktopConnectionDialIpc({
  ipcMain,
  windowConnectionRoutes,
  applySpawnPriority,
  backendConnectionState,
  backendDialClaims,
  backendPool,
  ensureBackend,
  ensureRegistryBackend,
  fetchJsonForBackend,
  primaryProfileKey,
  readDesktopConnectionsRegistry,
  rememberLog,
  remoteLiveness,
  remoteRevalidation,
  resetHermesConnection,
  resetPreviewReach,
  spawnPriorityFrom,
  sshBootstrapCoordinator,
  sshScopeKey,
  stopPoolBackend,
  teardownSshConnection
})

// These declarations are passed to earlier runtime factories before IPC setup.
function connectDesktopProfileRoute(
  route: DesktopProfileRoute,
  spawnPriority: LocalBackendSpawnPriority = 'foreground'
) {
  return connectDesktopProfileRouteImpl(route, spawnPriority)
}

function recordWindowConnectionRoute(sender: Electron.WebContents, route: unknown) {
  return recordWindowConnectionRouteImpl(sender, route)
}

function revalidateSuspectPoolAfterResume() {
  return revalidateSuspectPoolAfterResumeImpl()
}

registerBackendPoolIpc({
  ipcMain,
  touchPoolBackend,
  getPoolLimits,
  setPoolLimits,
  gatewayWsUrlIpcResult,
  freshGatewayWsUrl
})
registerDesktopWindowIpcRuntime({
  BrowserWindow,
  DEFAULT_ZOOM_LEVEL,
  HERMES_HOME,
  app,
  buildTerminalScript,
  createBrowserWindow,
  createInstanceWindow,
  createSessionWindow,
  findOnPath,
  ipcMain,
  percentToZoomLevel,
  registerWindowControlIpc,
  rememberLog,
  resolveHermesBackend,
  resolveTerminalLaunch,
  sanitizeWorkspaceCwd,
  setAndPersistZoomLevel,
  terminalScriptEnv,
  terminalScriptExtension,
  tuiResumeArgs,
  wakeIndicatorController,
  zoomLevelToPercent
})

// --- Pet overlay (pop-out mascot) — see pet-overlay-ipc.ts. ---------------
registerPetOverlayIpc({
  getMainWindow: () => mainWindow,
  getPetOverlayWindow,
  openPetOverlay,
  closePetOverlay
})

// --- HUD mode (chrome-free floating chat) — see hud-ipc.ts. ---------------
const hudIpc = registerHudIpc({
  isMac: IS_MAC,
  getTranslucencyState: appearance.getTranslucencyState,
  getHudWindow: shellOverlayRuntime.getHudWindow,
  openHudWindow,
  closeHudWindow,
  resetHudLayout: resetHudWindowLayout,
  setHudSessionId: shellOverlayRuntime.setHudSessionId
})

registerDesktopBootstrapIpc({
  ipcMain,
  recycleOwnedBackend,
  sendConnectionApplied,
  primaryProfileKey,
  teardownPoolBackendAndWait,
  teardownPrimaryBackendAndWait,
  teardownSshConnection,
  rememberLog,
  clearFailures: () => {
    bootstrapFailure = null
    backendStartFailure = null
    remoteReauthFailure = null
  },
  firstRunBoot,
  incrementRepairAttempt: () => ++bootstrapRepairAttempt,
  maxBootstrapRepairSoftAttempts: MAX_BOOTSTRAP_REPAIR_SOFT_ATTEMPTS,
  getPrimaryBackendProcess: () => backendConnectionState.getProcess(),
  setBootstrapRepairRequested: value => { bootstrapRepairRequested = value },
  resetHermesConnection,
  getBootstrapAbortController: () => bootstrapAbortController
})
registerDesktopPluginProfileRoutesIpc({
  ipcMain,
  readDesktopConnectionConfig,
  sanitizeDesktopConnectionConfig,
  readDesktopConnectionsRegistry,
  enumerateRegistryAgentSources: registry => enumerateRegistryAgentSources(registry),
  buildAgentRoster
})
registerDesktopConnectionRegistryIpc({
  ipcMain,
  testDesktopConnectionConfig,
  secretStoragePolicy,
  applySecretStorageEncryption,
  sanitizeConnectionsRegistry,
  saveRegistryConnection,
  managedConnectionUpdateGate,
  readDesktopConnectionsRegistry,
  writeDesktopConnectionsRegistry,
  stopRegistryConnectionBackends,
  broadcastConnectionsChanged,
  desktopProfilePreferences,
  assertCanMutateManagedPrimaryRouting,
  probeSshProfileInventory: connection => probeSshProfileInventory(connection),
  startHermes,
  decryptRemoteHeaders,
  decryptDesktopSecret,
  fetchConnectionStatus,
  rememberConnectionInstallId: (connectionId, statusBody) => rememberConnectionInstallId(connectionId, statusBody),
  mintGatewayWsTicket
})

// ── Union agent roster + registry ws-url + fan-out updates (phase 3-5) ─────

// Enumerate every registered connection's profiles concurrently and flatten
// into the union roster. Eager REST enumeration, lazy sockets: local + already
// -dialed sources answer instantly; unreachable ones return an error entry
// instead of failing the whole roster. ssh sources that have never been dialed
// are SKIPPED (connect-on-demand — dialing every ssh box just to list agents
// would spawn tunnels the user never asked for); once dialed, their pooled
// descriptor serves the enumeration like any remote. Last-known SSH profile
// lists are reused so switching the window back to local does not empty Bot Mode.
// These three live in ./connection-caches, which states (and tests) the invariant they share:
// each is keyed by connection id and is only valid while that id names the same machine, so
// removing a connection or re-pointing it must evict them (`evictConnectionCaches`).
const { rememberConnectionInstallId, probeSshProfileInventory, enumerateRegistryAgentSources } =
  registerDesktopConnectionFleetIpc({
    applyUpdates,
    backendDialClaims,
    backendPool,
    backendScopeKey,
    buildAgentRoster,
    buildGatewayWsUrlWithTicket,
    connectionInstallIds,
    createRegistryGatewayWsUrlHandler,
    createSshProbeConnection,
    ensureRegistryBackend,
    fetchRosterSourceData,
    gatewayWsUrlIpcResult,
    getJsonForBackend,
    globalRemoteActive,
    ipcMain,
    managedConnectionUpdateGate,
    managedConnectionUpdates,
    mintGatewayWsTicket,
    normalizeSshConfig,
    postJsonForBackend,
    primaryProfileKey,
    profileHasRemoteOverride,
    readDesktopConnectionsRegistry,
    refusedManagedSshUpdate,
    rememberRemoteWsHeaders,
    rememberSshEnumeration,
    remoteLifecycle,
    resolveRegistryLocalRoute,
    rosterSourceEnumerationTimeoutMs,
    shouldDeferLocalEnumeration,
    shouldRetrySshInventory,
    sshInventoryAttemptedAt,
    sshRememberLog,
    sshRosterCache,
    updateEligibility,
    updateManagedSshConnection
  })

registerDesktopConnectionAuthIpc({
  ipcMain,
  probeRemoteAuthMode,
  nativeAccessTokenCoordinator,
  fetchPublicJson,
  gatewayAuthProviders,
  postJsonNoAuth,
  rememberLog,
  clearOauthSession,
  hasLiveOauthSession,
  hasNativeSession,
  hasOauthSessionCookie,
  openOauthLoginWindow,
  resolvePortalBaseUrl,
  hasLivePortalSession,
  openPortalLoginWindow,
  discoverCloudAgents,
  cloudAgentSilentSignIn,
  assertCanMutateManagedPrimaryRouting,
  coerceDesktopConnectionConfig,
  writeDesktopConnectionConfig,
  sanitizeDesktopConnectionConfig,
  readDesktopConnectionConfig,
  readDesktopConnectionsRegistry,
  testDesktopConnectionConfig,
  writeDesktopConnectionsRegistry,
  sshBootstrapCoordinator,
  primaryProfileKey,
  sendConnectionApplied,
  firstRunBoot,
  teardownPrimaryBackendAndWait,
  stopPoolBackend,
  teardownSshConnection,
  clearRemoteReauthFailure: () => {
    remoteReauthFailure = null
  },
  clearLocalBootstrapFailure: () => {
    bootstrapFailure = null
  },
  shell
})
registerDesktopProfileRoutingIpc({
  ipcMain,
  desktopProfilePreferences,
  readActiveDesktopProfile,
  writeActiveDesktopProfile,
  assertCanMutateManagedPrimaryRouting,
  teardownPrimaryBackendAndWait,
  getMainWindow: () => mainWindow
})

ipcMain.on('hermes:previewShortcutActive', (_event, active) => {
  previewShortcutActive = Boolean(active)
})

registerNativeWindowServicesIpc()

registerDesktopConnectionApiIpc({
  ipcMain,
  backendDialClaims,
  backendPool,
  configuredRemoteProfileNames,
  desktopProfilePreferences,
  ensureBackend,
  ensureRegistryBackend,
  fetchJsonForBackend,
  fetchJsonForProfile,
  getJsonForBackend,
  globalRemoteActive,
  poolStopper,
  prepareProfileDeleteRequest,
  prepareProfileRenameRequest,
  PROFILE_NAME_RE,
  profileDeletionGate,
  profileHasRemoteOverride,
  profileRouteOptions,
  readDesktopConnectionsRegistry,
  rememberLog,
  requestJsonForProfile,
  spawnPriorityFrom,
  sshBootstrapCoordinator,
  teardownSshConnection
})
// Main serializes cross-window ambient claims (see event-dedupe.ts for why a
// spoken reply holds its claim far longer than a beep).
const ownsAmbientCue = createAmbientClaimArbiter()
ipcMain.handle('hermes:ambient:claim', (_event, key) => ownsAmbientCue(String(key ?? '')))

const nativeNotifications = registerNativeNotifications({ getMainWindow: () => mainWindow, focusWindow })

registerDesktopFileIpc({
  app,
  clipboard,
  dialog,
  electronWebContents,
  getMainWindow: () => mainWindow,
  HERMES_HOME,
  ipcMain,
  IS_WINDOWS,
  IS_WSL,
  lastContextMenuPoint,
  mimeTypeForPath,
  rememberLog,
  saveGatewayFile,
  saveImageFromUrl,
  writeComposerImage
})

registerPreviewTargetIpc(ipcMain, previewTargetRuntime)

// Each renderer reports the turns it has in flight; the quit guard reads the
// merged picture. Keyed by webContents id so a closed window stops counting.
const activeWorkByWebContents = new Map<number, ActiveWork>()

// The same merged picture drives background throttling: chat windows run
// unthrottled while any turn is in flight (streaming must paint while hidden)
// and fall back to Chromium's default throttling at idle. See stream-throttle.ts.
const streamThrottle = createStreamThrottle()

function updateStreamThrottleFromActiveWork() {
  streamThrottle.update(mergeActiveWork(activeWorkByWebContents.values()).count > 0)
}

ipcMain.on('hermes:active-work', (event, payload) => {
  const id = event.sender.id

  if (!activeWorkByWebContents.has(id)) {
    event.sender.once('destroyed', () => {
      activeWorkByWebContents.delete(id)
      updateStreamThrottleFromActiveWork()
    })
  }

  activeWorkByWebContents.set(id, normalizeActiveWork(payload))
  updateStreamThrottleFromActiveWork()
})

const { keepAwake, readPersistedKeepAwake } = createDesktopNativePreferencesRuntime({
  GLASS_SUPPORTED,
  GUEST_ONBOARDING,
  SKIP_INTRO,
  TRANSLUCENCY_SUPPORTED,
  app,
  appearance,
  createKeepAwake,
  destroyKeepaliveAgents,
  hudIpc,
  ipcMain,
  nativeNotifications,
  powerSaveBlocker,
  quitFinalization,
  rememberLog,
  sshIsolatedKeepalives
})

registerDesktopQuickEntryIpc({
  applyQuickEntrySettings,
  getMainWindow: () => mainWindow,
  hideQuickEntryWindow,
  ipcMain,
  readQuickEntrySettings,
  rememberLog,
  sanitizeQuickEntrySettings,
  shellOverlayRuntime,
  writeQuickEntrySettings
})

const { readPersistedDisableF12 } = registerDesktopF12PreferenceIpc({
  app,
  f12State: {
    get blocked() {
      return f12Blocked
    },
    set blocked(value) {
      f12Blocked = value
    }
  },
  ipcMain,
  rememberLog
})

registerDesktopPageInteractionIpc({
  BrowserWindow,
  installFoundInPageForwarder,
  ipcMain,
  openExternalUrl,
  openPreviewInBrowser,
  performFindAfterIndexingStarted,
  reachablePreviewUrl,
  stopFind
})

const terminalIpc = registerWorkspaceAndLogIpc({
  ipcMain,
  app,
  dialog,
  shell,
  readDefaultProjectDir,
  resolveHermesCwd,
  sanitizeWorkspaceCwd,
  writeDefaultProjectDir,
  mkdirSync: fs.mkdirSync,
  desktopLogPath: DESKTOP_LOG_PATH,
  fileExists,
  appendFile: (filePath, content) => fs.promises.appendFile(filePath, content),
  hermesLog,
  rememberLog,
  formatRendererBoundaryReport,
  flushDesktopLogBufferSync,
  fetchLinkTitle,
  resolveFaviconCached,
  registerFsIpc,
  registerGitIpc,
  registerMcpOauthCallbackIpc,
  registerTerminalIpc,
  hermesHome: HERMES_HOME,
  readActiveDesktopProfile,
  expandUserPath,
  resolveRequestedPathForIpc,
  directoryExists,
  resolveGitBinary,
  resolveGhBinary,
  isWindows: IS_WINDOWS,
  findOnPath,
  activeSshTerminalTarget,
  ensureTerminalBackend,
  getSshConnectionState: scope => sshConnections.get(scope)
})

// Keep this callback hoisted for native chrome, which captures it earlier.
function showAboutPanelFresh() {
  return desktopShellRuntime.showAboutPanelFresh()
}

registerDesktopOperationsIpc({
  ipcMain,
  checkUpdates,
  applyUpdates,
  readDesktopUpdateConfig,
  writeDesktopUpdateConfig,
  defaultUpdateBranch: DEFAULT_UPDATE_BRANCH,
  desktopShellRuntime,
  fetchMarketplaceThemes,
  searchMarketplaceThemes
})

const HERMES_PROTOCOL = DEV_SERVER ? 'hermes-dev' : 'hermes'
let _rendererReadyForDeepLink = false
// Set by sendOpenUpdatesRequested() when the renderer cannot hear it yet.
let _pendingOpenUpdates = false

function handleDeepLink(url) {
  desktopAppLifecycle.handleDeepLink(url)
}

const desktopAppLifecycle = createDesktopAppLifecycleRuntime({
  app,
  ipcMain,
  Menu,
  screen,
  session,
  safeStorage,
  path,
  tls,
  pathToFileURL,
  CHROMIUM_LOG_PATH,
  CRASH_DIAGNOSTICS,
  DEV_SERVER,
  HERMES_PROTOCOL,
  IS_MAC,
  backendShutdown,
  buildApplicationMenu,
  createWindow,
  ensureLoginShellPath,
  ensureMainWindow,
  ensureWslWindowsFonts,
  enableBasicPasswordStoreEncryption,
  focusWindow,
  getIsQuittingForHandoff: () => isQuittingForHandoff,
  getMainWindow: () => mainWindow,
  getPendingOpenUpdates: () => _pendingOpenUpdates,
  getRendererReadyForDeepLink: () => _rendererReadyForDeepLink,
  installDownloadHandling,
  installApplicationMenuAfterFirstWindow,
  installCommandScreenshot,
  installEmbedReferer,
  installHudModifierTap,
  installMediaPermissions,
  installPreviewGuestPreload,
  installRemoteHeaderRules,
  installWindowsSystemCaTrust,
  keepAwake,
  migrateLegacyEncryptedSecretsOnce,
  minimizeToTray,
  openHudWindow,
  primaryBackendIsRemote,
  primaryProfileKey,
  readPersistedDisableF12,
  readPersistedKeepAwake,
  readQuickEntrySettings,
  registerMediaProtocol,
  registerPowerResumeListeners,
  rememberLog,
  resolveRendererIndex,
  resumeManagedSshRecoveries,
  sendOpenUpdatesRequested,
  setActiveGatewayProfile,
  setF12Blocked: blocked => {
    f12Blocked = blocked
  },
  setPendingOpenUpdates: pending => {
    _pendingOpenUpdates = pending
  },
  setRendererReadyForDeepLink: ready => {
    _rendererReadyForDeepLink = ready
  },
  setWslBridgeProfileState,
  startChromiumLogWatcher,
  wakeIndicatorController,
  applyQuickEntrySettings
})

const isPrimaryInstance = desktopAppLifecycle.isPrimaryInstance

// Register after lifecycle setup so the held quit precedes normal teardown.
const heldQuitForActiveWork = createDesktopHeldQuitRuntime({
  app,
  BrowserWindow,
  dialog,
  activeWorkByWebContents,
  minimizeToTray,
  getIsQuittingForHandoff: () => isQuittingForHandoff,
  skipQuitConfirm: SKIP_QUIT_CONFIRM
})

registerDesktopQuitRuntime({
  IS_WINDOWS,
  app,
  backendConnectionState,
  backendQuitNeedsWait,
  backendShutdown,
  closePetOverlay,
  closeQuickEntryWindow,
  flushDesktopLogBufferSync,
  getBootstrapAbortController: () => bootstrapAbortController,
  getIsQuittingForHandoff: () => isQuittingForHandoff,
  getWindowsSandboxFallbackSticky: () => sandboxState.fallbackSticky,
  heldQuitForActiveWork,
  introRevealController,
  localBackendLifecycle,
  managedConnectionRecoveries,
  managedConnectionUpdates,
  managedUpdateQuitState,
  markerAfterSuccessfulBoot,
  minimizeToTray,
  poolStopper,
  previewTargetRuntime,
  quitTeardown,
  shellOverlayRuntime,
  sshBootstrapCoordinator,
  sshConnections,
  sshTeardowns,
  stopDesktopLogFlushTimer,
  teardownSshForQuit,
  terminalIpc,
  waitForManagedUpdateOperations,
  wakeIndicatorController,
  writeSandboxMarker
})
