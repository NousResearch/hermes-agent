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

import { classifyActiveRuntime } from './active-runtime-state'
import { destroyKeepaliveAgents, readStatusCode } from './api-transport'
import { appIconCandidates, resolveAppIcon } from './app-icon'
import { installApplicationMenuAfterFirstWindow } from './application-menu-startup'
import { stopBackendChild as stopBackendChildImpl, waitForBackendExit as waitForBackendExitImpl } from './backend-child'
import {
  createBackendOutputTail,
  execText,
  formatBackendExitLine,
  probeStartMarker,
  processStartMarker
} from './backend-claim'
import { dashboardFallbackArgs } from './backend-command'
import { createBackendConnectionState } from './backend-connection-state'
import { BackendDialClaims } from './backend-dial-claim'
import { buildDesktopBackendEnv, hermesManagedNodePathEntries, normalizeHermesHomeRoot } from './backend-env'
import { createBackendExitRecoveryLatch } from './backend-exit-recovery'
import { isReauthRequiredError, waitForHermesReady } from './backend-health'
import { createBackendShutdownCoordinator } from './backend-ownership'
import { canImportHermesCli, PROBE_TIMEOUT_MS } from './backend-probes'
import { waitForDashboardPortAnnouncement } from './backend-ready'
import { recycleOwnedBackend } from './backend-recycle'
import { createBackendServeSupportResolver } from './backend-serve-support'
import {
  isHostKeyChangedBootFailure,
  isRetryableRemoteBootFailure,
  shouldLatchBackendStartFailure,
  shouldLatchHostKeyChangedFailure,
  shouldLatchRemoteReauthFailure
} from './backend-start-failure'
import {
  detectRemoteDisplay,
  isWindowsBinaryPathInWsl,
  isWslEnvironment,
  resolveLinuxPasswordStore
} from './bootstrap-platform'
import { decideBootstrapRepair } from './bootstrap-repair-guard'
import { detectBundleSwap } from './bundle-swap'
import { registerChatOnboardingWindow } from './chat-onboarding-window'
import { installCommandScreenshot } from './command-screenshot'
import { teardownSshState } from './connection-apply'
import {
  connectionInstallIds,
  sshInventoryAttemptedAt,
  sshRosterCache
} from './connection-caches'
import {
  buildGatewayWsUrlWithTicket,
  connectionScopeKey,
  gatewayWsUrlIpcResult,
  normalizeSshConfig,
  profileRemoteOverride,
  profileSshOverride,
  resolveRemoteSshDashboardProfile
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
import { adoptServedDashboardToken, resolveServedDashboardToken } from './dashboard-token'
import { createDesktopAppLifecycleRuntime } from './desktop-app-lifecycle-runtime'
import { createDesktopBackendOwnershipRuntime } from './desktop-backend-ownership-runtime'
import { createDesktopConnectionAdmissionRuntime } from './desktop-connection-admission-runtime'
import { registerDesktopConnectionApiIpc } from './desktop-connection-api-ipc'
import { registerDesktopConnectionAuthIpc } from './desktop-connection-auth-ipc'
import { createDesktopConnectionAuthRuntime } from './desktop-connection-auth-runtime'
import { createDesktopConnectionDescriptorRuntime } from './desktop-connection-descriptor-runtime'
import { registerDesktopConnectionDialIpc } from './desktop-connection-dial-ipc'
import { registerDesktopConnectionFleetIpc } from './desktop-connection-fleet-runtime'
import { createDesktopConnectionProbeRuntime } from './desktop-connection-probe-runtime'
import { registerDesktopConnectionRegistryIpc } from './desktop-connection-registry-ipc'
import { createDesktopConnectionStorageRuntime } from './desktop-connection-storage-runtime'
import { createDesktopExternalOpenRuntime } from './desktop-external-open-runtime'
import { registerDesktopFileIpc } from './desktop-file-ipc'
import { loadOrCreateInstallationId, sshOwnershipId } from './desktop-installation'
import { createDesktopLocalRuntime } from './desktop-local-runtime'
import { createDesktopLogRuntime, rotateLogIfNeededSync } from './desktop-log-runtime'
import { createDesktopNativeChromeRuntime } from './desktop-native-chrome-runtime'
import { createDesktopNativePreferencesRuntime, registerDesktopF12PreferenceIpc } from './desktop-native-preferences-runtime'
import { createDesktopOauthSessionRuntime } from './desktop-oauth-session-runtime'
import { createDesktopPetOverlayRuntime } from './desktop-pet-overlay-runtime'
import { createDesktopPluginCompatNoticeRuntime } from './desktop-plugin-compat-notice-runtime'
import { createDesktopPowerRuntime } from './desktop-power-runtime'
import { createDesktopPrimaryBackendRuntime } from './desktop-primary-backend-runtime'
import { createDesktopPrimaryWindowRuntime } from './desktop-primary-window-runtime'
import {
  createDesktopProfilePreferences,
  DESKTOP_PROFILE_NAME_RE,
  type DesktopProfileRoute
} from './desktop-profile'
import { registerDesktopQuickEntryIpc } from './desktop-quick-entry-ipc'
import { registerDesktopQuitRuntime } from './desktop-quit-runtime'
import { resolveDesktopRemoteRoute, v1SshTerminalPoolKey } from './desktop-remote-route'
import { createDesktopRendererAssetsRuntime } from './desktop-renderer-assets-runtime'
import { createDesktopSecondaryWindowRuntime } from './desktop-secondary-window-runtime'
import { createDesktopShellOverlayRuntime } from './desktop-shell-overlay-runtime'
import { createDesktopShellRuntime } from './desktop-shell-runtime'
import { createDesktopSshBootstrapRuntime } from './desktop-ssh-bootstrap-runtime'
import { createDesktopSshSessionRuntime } from './desktop-ssh-session-runtime'
import { resolveRemovableAppPath } from './desktop-uninstall'
import { createDesktopUpdateCheckRuntime } from './desktop-update-check-runtime'
import { createDesktopWindowEventsRuntime } from './desktop-window-events-runtime'
import { registerDesktopWindowIpcRuntime } from './desktop-window-ipc-runtime'
import { createDesktopWindowWiringRuntime } from './desktop-window-wiring-runtime'
import { describeDevCdpDecision, resolveDevCdpPort } from './dev-cdp'
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
import { findGitBash as _findGitBash } from './find-git-bash'
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
import { desktopBackendSpawnEnv, guestOnboardingEnabled, skipIntroEnabled } from './guest-onboarding'
import { readAndConsumeHandoffResult } from './handoff-result'
import {
  enableBasicPasswordStoreEncryption,
  resolveReadableFileForIpc,
  resolveRequestedPathForIpc
} from './hardening'
import {
  type AttachedBackend,
  attachOrReserveSpawn,
  spawnLedgerPath,
  type SpawnReservation
} from './host-backend-attach'
import { assertNoSecondLocalBackend } from './host-backend-singleton'
import { registerHudIpc } from './hud-ipc'
import { installHudModifierTap } from './hud-modifier'
import { createIntroRevealWindowController } from './intro-reveal-window'
import { createLinkMetadataRuntime } from './link-metadata-runtime'
import { CHROMIUM_LOG_FILENAME, enableLinuxCrashDiagnostics, linuxCrashDiagnostics } from './linux-crash-diagnostics'
import { notifyLauncherWindowRevealed } from './linux-launcher-ready'
import { createLocalBackendLifecycle, waitForTeardown } from './local-backend-lifecycle'
import { ensureMainWindow } from './main-window-lifecycle'
import { createManagedSshLifecycleRuntime } from './managed-ssh-lifecycle-runtime'
import { createManagedSshRecoveryJournal } from './managed-ssh-recovery-journal'
import {
  assertManagedUpdatePreflightClear,
  executeManagedRemoteUpdate,
  ManagedConnectionUpdateGate,
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
import { createNativeAppearanceController } from './native-appearance-controller'
import { resolveReadinessProbeAuth } from './native-auth-decisions'
import { registerNativeNotifications } from './notification-ipc'
import { requestWithOauthFallback } from './oauth-rest-request'
import { parentWatchdogEnv } from './parent-process-identity'
import { registerPetOverlayIpc } from './pet-overlay-ipc'
import {
  pendingNotice as pendingPluginCompatNotice,
  recordDismissed as recordPluginCompatDismissed
} from './plugin-compat-notice'
import {
  buildRegistryProfileRoutes,
  isLocalEnumerationFailure,
  localRouteFallbackProfiles,
  undialedSshRouteSeeds
} from './plugin-profile-routes'
import { clampPoolLimits, parsePoolLimits, POOL_LIMITS_DEFAULTS } from './pool-limits'
import { createPoolRetirer } from './pool-retire'
import { createPoolRetirementClient } from './pool-retire-http'
import {
  BackgroundSlotRetryBackoff,
  BackgroundSlotRetryDeferredError,
  isBackgroundSlotRetryDeferred,
  isBackgroundSlotWaitTimeout,
  LocalBackendSpawnCoordinator,
  type LocalBackendSpawnPriority,
  type LocalBackendSpawnRequest,
  registerLocalBackendExitFinalizer,
  releaseLocalBackendSlotAfterExit
} from './pool-spawn-coordinator'
import { createPoolStopper } from './pool-stop'
import { poolTouchKeys } from './pool-touch-scope'
import { createKeepAwake } from './power-save'
import { createPreviewTargetRuntime, registerPreviewTargetIpc } from './preview-target-runtime'
import {
  createPrimaryRemoteConnection,
  FirstRunSetupResetError,
  runPrimaryBackendStartup
} from './primary-backend-startup'
import { PrimaryProfilePin } from './primary-profile-pin'
import {
  assertLocalProfileCanStart,
  decideProfileDeleteAction,
  localProfilePoolKeys,
  ProfileDeletionGate,
  profileNameFromDeleteRequest
} from './profile-delete-routing'
import { migrateActiveProfileIfMissing as migrateActiveProfileIfMissingPure } from './profile-migration'
import { prepareProfileRenameLifecycle } from './profile-rename-routing'
import { sanitizeQuickEntrySettings } from './quick-entry'
import { createQuitFinalization } from './quit-finalization'
import { type ActiveWork, mergeActiveWork, normalizeActiveWork, quitPromptFor } from './quit-guard'
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
import { planLaunchSwitches, readDesktopLaunchConfig } from './renderer-heap-flags'
import { loadRendererLoadErrorPage } from './renderer-load-error-page'
import { attachRendererConsoleCapture, formatRendererBoundaryReport } from './renderer-log'
import { fetchRosterSourceData } from './roster-source-fetch'
import { GIT_UNUSABLE } from './select-runnable-binary'
import { chatWindowWebPreferences } from './session-windows'
import { ensureLoginShellPath } from './shell-path'
import { createBootstrapCoordinator } from './ssh-bootstrap-coordinator'
import { createSshProbeConnection, pickLocalPort, redactSecrets, SshConnection } from './ssh-connection'
import { createSshIsolatedKeepaliveRegistry } from './ssh-isolated-keepalive'
import { createSshTeardownTracker } from './ssh-teardown'
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
  registrySshPoolScopeByConnectionId,
  registrySshScopeForWindowRoute,
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
import { buildPathExtCandidates, getVenvSitePackagesEntries, resolveVenvHermesCommand } from './windows-hermes-path'
import {
  connectWindowsRemote,
  detectRemotePlatform,
  probeWindowsRemote,
  terminateOwnedWindowsDashboardForUpdate
} from './windows-remote-lifecycle'
import {
  alreadyHasNoSandbox,
  buildNoSandboxRelaunchArgs,
  decideWindowsSandboxLaunch,
  fallbackMarker,
  grantAllApplicationPackagesAcl,
  markerAfterSuccessfulBoot,
  readSandboxMarker,
  type SandboxFallbackReason,
  shouldAttemptAclRepair,
  shouldRelaunchForGpuSandboxCrash,
  shouldRelaunchForRendererSandboxCrashLoop,
  writeSandboxMarker
} from './windows-sandbox-fallback'
import { installWindowsSystemCaTrust } from './windows-system-ca'
import { readWindowsUserEnvVar } from './windows-user-env'
import { isPackagedInstallPath as isPackagedInstallPathUnderRoots } from './workspace-cwd'
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

// Renderer debugging port. On for dev-server runs (`hgui` / `npm run dev`) so
// the CDP tooling in scripts/ can attach; never for a packaged build — see
// electron/dev-cdp.ts. Must run before app `ready` like the switches above;
// Chromium binds it at launch.
const DEV_CDP = resolveDevCdpPort({ env: process.env, isPackaged: IS_PACKAGED, devServer: DEV_SERVER })

if (DEV_CDP.port) {
  app.commandLine.appendSwitch('remote-debugging-port', String(DEV_CDP.port))
  // Loopback only. Chromium already defaults to 127.0.0.1, but say it out loud
  // so a future edit can't widen it by omission.
  app.commandLine.appendSwitch('remote-debugging-address', '127.0.0.1')
  console.log(
    `[hermes] renderer debugging on http://127.0.0.1:${DEV_CDP.port} — anything that can reach it ` +
      'can run code in the renderer. HERMES_DESKTOP_CDP_PORT=off to disable.'
  )
} else {
  const why = describeDevCdpDecision(DEV_CDP)

  if (why) {
    console.warn(`[hermes] ${why}`)
  }
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

// Linux: point Chromium at the session's keychain backend so safeStorage can
// encrypt remote gateway tokens (hardening.ts refuses to persist them without
// it). The value arrives via HERMES_DESKTOP_PASSWORD_STORE, bridged by the
// `hermes desktop` launcher from detection or `desktop.password_store` in
// config.yaml. Must run before app `ready` — the switch only applies pre-launch.
const PASSWORD_STORE = resolveLinuxPasswordStore()

if (PASSWORD_STORE.warning) {
  console.warn(`[hermes] ${PASSWORD_STORE.warning}`)
}

if (PASSWORD_STORE.store) {
  app.commandLine.appendSwitch('password-store', PASSWORD_STORE.store)
  console.log(`[hermes] using password-store backend: ${PASSWORD_STORE.store}`)
}

// Windows sandbox / GPU breakpoint crash recovery (#38216).
//
// Some hosts (AMD RX 6000 drivers, orphan AppContainer SIDs under %LOCALAPPDATA%,
// missing S-1-15-2-2 ACEs) kill Chromium's sandboxed GPU/renderer children with
// 0x80000003. After enough GPU deaths the browser process FATAL-exits before the
// UI is usable. Must run before app `ready` so `--no-sandbox` applies to child
// processes. The sticky marker recovers Start Menu / shortcut launches that
// never go through `hermes desktop`; it is version-scoped so an app update
// re-probes the sandbox instead of degrading forever.
//
// `windowsSandboxFallbackActive` = this process runs without the Chromium
// sandbox (any cause, including a manual --no-sandbox flag) — guards the
// relaunch handlers. `windowsSandboxFallbackSticky` = the fallback machinery
// engaged and the marker must stay `fallback` after a successful boot; a
// manual flag alone is honored but never made sticky.
let windowsSandboxFallbackActive = false
let windowsSandboxFallbackSticky = false
let windowsSandboxFallbackReason: SandboxFallbackReason = 'boot-loop'
let windowsNoSandboxRelaunchAttempted = false

if (IS_WINDOWS) {
  const windowsUserData = app.getPath('userData')
  const priorMarker = readSandboxMarker(windowsUserData)

  // Best-effort ACL repair, only when the last boot aborted or the fallback is
  // engaged — icacls /T recurses the whole install tree, so healthy launches
  // skip it (the installer already granted the ACE at install time). Repair
  // targets the install dir only: granting AppContainer read on userData would
  // expose Hermes sessions/config to every packaged app on the machine.
  if (shouldAttemptAclRepair(priorMarker)) {
    const exeDir = path.dirname(process.execPath)
    const acl = grantAllApplicationPackagesAcl(exeDir, { execFileSync })

    if (acl.ok) {
      console.log(`[hermes] granted ALL APPLICATION PACKAGES RX on ${exeDir} (#38216)`)
    } else if (acl.error && acl.error !== 'missing-target-or-exec') {
      console.warn(`[hermes] AppContainer ACL grant failed on ${exeDir}: ${acl.error}`)
    }
  }

  const sandboxDecision = decideWindowsSandboxLaunch({
    argv: process.argv,
    env: process.env,
    marker: priorMarker,
    appVersion: app.getVersion()
  })

  windowsSandboxFallbackActive = sandboxDecision.enable
  windowsSandboxFallbackSticky = sandboxDecision.nextMarker.state === 'fallback'

  if (sandboxDecision.nextMarker.state === 'fallback' && sandboxDecision.nextMarker.reason) {
    windowsSandboxFallbackReason = sandboxDecision.nextMarker.reason
  }

  if (sandboxDecision.enable && sandboxDecision.reason !== 'already-enabled') {
    app.commandLine.appendSwitch('no-sandbox')
    process.env.ELECTRON_DISABLE_SANDBOX = '1'
    console.log(
      `[hermes] Windows sandbox fallback enabled (${sandboxDecision.reason}); launching with --no-sandbox (#38216)`
    )
  }

  writeSandboxMarker(windowsUserData, sandboxDecision.nextMarker)

  // Catch the first GPU breakpoint death and relaunch before Chromium's
  // "GPU process isn't usable" FATAL abort ends the process with no recovery.
  app.on('child-process-gone', (_event, details) => {
    if (
      !shouldRelaunchForGpuSandboxCrash({
        details,
        alreadyNoSandbox: windowsSandboxFallbackActive || alreadyHasNoSandbox(process.argv, process.env),
        relaunchAttempted: windowsNoSandboxRelaunchAttempted
      })
    ) {
      return
    }

    windowsNoSandboxRelaunchAttempted = true
    windowsSandboxFallbackActive = true
    windowsSandboxFallbackSticky = true
    windowsSandboxFallbackReason = 'gpu-breakpoint'

    try {
      writeSandboxMarker(app.getPath('userData'), fallbackMarker('gpu-breakpoint', app.getVersion()))
    } catch {
      void 0
    }

    console.warn(
      `[hermes] Windows GPU sandbox crashed (exit=${details?.exitCode}); relaunching once with --no-sandbox (#38216)`
    )

    try {
      app.relaunch({ args: buildNoSandboxRelaunchArgs(process.argv.slice(1)) })
      void exitAfterBackendShutdown(0)
    } catch (error) {
      console.error(`[hermes] --no-sandbox relaunch failed: ${error?.message || error}`)
    }
  })
}

ipcMain.handle('hermes:get-remote-display-reason', () => REMOTE_DISPLAY_REASON)

// Keep the renderer's PROCESS priority normal while its windows are hidden —
// a deprioritized renderer streams a live answer visibly slower once the
// window is minimized. This switch only affects scheduling priority; it does
// not exempt timers from throttling and costs nothing at idle.
//
// The timer/rAF throttling story is deliberately NOT handled here anymore.
// The old process-wide `disable-background-timer-throttling` /
// `disable-backgrounding-occluded-windows` switches (plus a static
// `backgroundThrottling: false` on every chat window) pinned every renderer's
// `document.visibilityState` to 'visible' forever — which silently turned all
// the renderer's visibility-gated backstop polls and clock ticks into
// always-on timers. A completely idle, minimized Hermes burned ~20% CPU
// around the clock. Throttling is now a runtime dial scoped to streaming:
// see createStreamThrottle() — chat windows are unthrottled while any turn is
// in flight (so a live answer keeps painting while blurred, occluded, or
// minimized, exactly as before) and return to Chromium's default throttling
// once the work settles.
app.commandLine.appendSwitch('disable-renderer-backgrounding')

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
  const handler = createMediaProtocolHandler({
    ensureRemoteBearer: baseUrl => ensureNativeAccessToken(baseUrl),
    // Answer local files ourselves: Electron's file:// loader ignores Range and
    // returns the whole body as 200 without Accept-Ranges, which makes <video>
    // unseekable (seekable=[0,0]).
    fetchLocal: fetchLocalMedia,
    fetchRemote: (url, headers, method) =>
      electronNet.fetch(url, {
        bypassCustomProtocolHandlers: true,
        credentials: 'omit',
        headers,
        method
      }),
    fetchRemoteWithCookies: (url, headers, method) => {
      const oauthSession = getOauthSessionForUrl(url)

      if (!oauthSession) {
        throw new Error('OAuth session partition is unavailable.')
      }

      return oauthSession.fetch(url, {
        bypassCustomProtocolHandlers: true,
        credentials: 'include',
        headers,
        method
      })
    },
    resolveLocalFile: async filePath => {
      const { resolvedPath } = await resolveReadableFileForIpc(filePath, { purpose: 'Media stream' })

      return resolvedPath
    },
    // Claim-guarded (#90812): a media stream load can race a renderer's own
    // reconnect dial for the same (connectionId, profile) scope; coalescing
    // here avoids bootstrapping a second SSH tunnel / remote dashboard.
    resolveRemoteConnection: ({ connectionId, profile }) =>
      backendDialClaims.run(backendScopeKey(connectionId, profile), () =>
        connectionId ? ensureRegistryBackend(connectionId, profile) : ensureBackend(profile)
      )
  })

  protocol.handle(MEDIA_PROTOCOL, handler)
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
let softRehomeInProgress = false
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
// Keep the pool light: cap concurrent profile backends (LRU eviction) and reap
// idle ones. A user idles at exactly the primary backend; pool backends only
// exist while a non-primary profile is actively being chatted through.
// Pool sizing is a device preference (Settings → Advanced → pool rows), not a
// launch constant: mutable at runtime, persisted in userData, applied live.
// The legacy HERMES_DESKTOP_POOL_* env vars remain the initial-value fallback
// for scripted/headless setups; after launch the stored preference wins.
const POOL_LIMITS_PATH = path.join(app.getPath('userData'), 'pool-limits.json')

function readPersistedPoolLimits() {
  try {
    const limits = parsePoolLimits(fs.readFileSync(POOL_LIMITS_PATH, 'utf8'))
    rememberLog(
      `[pool-limits] loaded from ${POOL_LIMITS_PATH}: maxBackends=${limits.maxBackends}, idleMs=${limits.idleMs}`
    )

    return limits
  } catch {
    // No persisted file yet — fall back to the legacy env vars so scripted
    // setups keep working. Log which source won: a silently-ignored env var
    // here costs a scripted-setup user a debugging session.
    const fromEnv = clampPoolLimits({
      maxBackends: Number(process.env.HERMES_DESKTOP_POOL_MAX) || undefined,
      idleMs: Number(process.env.HERMES_DESKTOP_POOL_IDLE_MS) || undefined
    })

    if (fromEnv.maxBackends !== POOL_LIMITS_DEFAULTS.maxBackends || fromEnv.idleMs !== POOL_LIMITS_DEFAULTS.idleMs) {
      rememberLog(
        `[pool-limits] no saved file; using env-var overrides: maxBackends=${fromEnv.maxBackends}, idleMs=${fromEnv.idleMs}`
      )
    } else {
      rememberLog('[pool-limits] no saved file and no env overrides; using defaults')
    }

    return fromEnv
  }
}

function persistPoolLimits(limits) {
  try {
    fs.mkdirSync(path.dirname(POOL_LIMITS_PATH), { recursive: true })
    // Atomic write: write to a temp file in the same directory, then rename.
    // A crash mid-write would otherwise leave truncated JSON and silently
    // lose the user's saved sizing.
    const tmpPath = `${POOL_LIMITS_PATH}.tmp`
    fs.writeFileSync(tmpPath, JSON.stringify(limits, null, 2), 'utf8')
    fs.renameSync(tmpPath, POOL_LIMITS_PATH)
  } catch (error) {
    rememberLog(`[pool-limits] write failed: ${error.message}`)
  }
}

let poolLimits = readPersistedPoolLimits()
// Hard cap on local backends that are starting OR running (the LRU eviction
// above is soft — it spares keepalive-fresh entries). Follows the live
// preference: setPoolLimits() pushes a new max into the coordinator.
const localBackendSpawnCoordinator = new LocalBackendSpawnCoordinator(poolLimits.maxBackends)
const backgroundSlotRetryBackoff = new BackgroundSlotRetryBackoff()
// How long a spawn may wait for a free local slot. Must stay under the
// renderer's BACKEND_BOOT_WAIT_TIMEOUT_MS (45s, src/lib/with-timeout.ts) so
// the queued ticket fails before the renderer does and the user sees why.
const POOL_SLOT_WAIT_MS = 30_000

function spawnPriorityFrom(value: unknown): LocalBackendSpawnPriority {
  return value === 'foreground' ? 'foreground' : 'background'
}

// Foreground intent for a dial whose pool entry does not exist yet: a user
// click that joins an in-flight backendDialClaims claim never re-enters
// ensureBackend(), and the claim owner may still be awaiting poolStopper /
// registry resolution before backendPool.set(). The local spawn takes the mark
// right before its slot request; the IPC handler that set it clears it once
// the claim settles, so a dial that never reaches a slot request (primary
// route, remote scope, a guard rejection) cannot leave it for a later
// hydration spawn of the same key to pick up.
const pendingForegroundSpawns = new Set<string>()

function takeForegroundSpawn(...poolKeys: string[]): boolean {
  let marked = false

  for (const poolKey of poolKeys) {
    marked = pendingForegroundSpawns.delete(poolKey) || marked
  }

  return marked
}

// Upgrade a pooled entry (running, spawning, or queued for a slot) to
// foreground so a queued slot wait can take the reserved foreground slot.
function promotePoolEntry(entry: any): void {
  entry.spawnPriority = 'foreground'
  entry.localBackendSpawnRequest?.promote?.('foreground')
}

// Land a spawn failure in desktop.log. Background slot waits back off per
// profile under a saturated pool, so a roster refresh cannot create a retry
// storm while a user-triggered foreground open still gets its reserved slot.
function logPoolSpawnFailure(label: string, error: unknown): void {
  if (isBackgroundSlotRetryDeferred(error)) {
    return
  }

  if (isBackgroundSlotWaitTimeout(error)) {
    rememberLog(`Profile backend ${label} slot wait timed out (background); retry is backing off`)
  } else {
    rememberLog(
      `Hermes backend for profile ${label} failed to start: ${error instanceof Error ? error.message : String(error)}`
    )
  }
}

// Apply foreground intent to the dial claim for `scopeKey`: an entry already
// in the pool is promoted directly, otherwise the intent is marked for the
// spawn the claim owner is about to start. Returns the cleanup that clears a
// mark the dial never consumed.
function applySpawnPriority(scopeKey: string, spawnPriority: LocalBackendSpawnPriority): () => void {
  // The renderer's socket-close event may beat its parking IPC. Main owns
  // this fence too, so that race cannot resurrect the retired generation.
  for (const key of poolTouchKeys(scopeKey)) {
    poolRetirer.assertCanOpen(key, spawnPriority)
  }

  if (spawnPriority !== 'foreground') {
    return () => undefined
  }

  const existing = backendPool.get(scopeKey)

  if (existing) {
    promotePoolEntry(existing)
  } else {
    pendingForegroundSpawns.add(scopeKey)
  }

  return () => void pendingForegroundSpawns.delete(scopeKey)
}

function poolMaxBackends() {
  return poolLimits.maxBackends
}

function poolIdleMs() {
  return poolLimits.idleMs
}

/**
 * Apply new limits live: persist, then converge the running pool — evict
 * LRU backends down to the new max, and let the (already running) idle
 * reaper handle a shortened idle window on its next tick. Returns the
 * limits actually in force (post-clamp).
 */
function setPoolLimits(raw) {
  poolLimits = clampPoolLimits(raw)
  persistPoolLimits(poolLimits)
  localBackendSpawnCoordinator.setLimit(poolLimits.maxBackends)
  void evictLruPoolBackends(poolMaxBackends())
  startPoolIdleReaper()

  return { ...poolLimits }
}

// A backend touched within this window has a live renderer socket (the keepalive
// pings every 60s for every open profile). LRU eviction must spare these — a
// concurrent multi-profile session keeps several backends "fresh" at once, and
// killing one to honor the soft cap would abort a running agent.
//
// The window is intentionally MUCH wider than the 60s ping cadence:
//   * 1 missed ping    = +60s of apparent silence
//   * WSL2 IPC stall  = the renderer's `hermes:backend:touch` roundtrips
//                       through 9p; a single brief 9p hiccup can stretch a
//                       ping to ~30s of observed silence (#95189: gateways
//                       exited every ~2 min on WSL2 because the previous
//                       90s window left no headroom — one delayed ping
//                       pushed a live backend past the threshold and the
//                       cap-driven eviction killed the active profile's
//                       backend mid-session, re-minting runtime ids and
//                       re-allocating pooled gateway secondaries ~700×/day).
//   * 3× ping + 60s headroom = ~4 min, comfortable margin for two missed
//     pings + WSL2 IPC stall. The hard ceiling for the cap-eligible set is
//     pool idle window above (default 10 min) — this constant only governs the
//     "is this backend plausibly still alive" question for LRU eviction,
//     not when the idle reaper definitively tears a backend down.
const POOL_KEEPALIVE_FRESH_MS = Math.max(
  120_000,
  Number(process.env.HERMES_DESKTOP_POOL_KEEPALIVE_FRESH_MS) || 4 * 60_000
)

let poolIdleReaper = null
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
  const extensions = buildPathExtCandidates(process.env.PATHEXT, IS_WINDOWS)

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

async function unwrapWindowsVenvHermesCommand(command, backendArgs) {
  return resolveVenvHermesCommand(command, backendArgs, {
    isWindows: IS_WINDOWS,
    isCommandScript,
    fileExists,
    directoryExists,
    canImportHermesCli,
    getVenvPython,
    getVenvSitePackagesEntries,
    buildDesktopBackendEnv,
    hermesHome: HERMES_HOME,
    resolvePath: (...segments) => path.resolve(...segments),
    dirname: p => path.dirname(p),
    basename: p => path.basename(p),
    rememberLog
  })
}

// Does the resolved runtime understand the `serve` subcommand? The desktop
// spawns `hermes serve`; runtimes older than serve only have `dashboard`. We
// detect support so getBackendArgsForRuntime() can route old runtimes through
// the legacy `dashboard --no-open` form instead of crashing on an unknown
// subcommand (would brick every user mid-upgrade — #54568 follow-up).
// Fast-path / probe / cache strategy: see backend-serve-support.ts header.
const backendSupportsServe = createBackendServeSupportResolver(HERMES_HOME, rememberLog)

// Given a resolved backend whose args target `serve`, return the args the
// runtime actually understands: unchanged when `serve` is supported, or
// rewritten to `dashboard --no-open` for older runtimes.
async function getBackendArgsForRuntime(backend) {
  return (await backendSupportsServe(backend)) ? backend.args : dashboardFallbackArgs(backend.args)
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

async function findPythonForRoot(root) {
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

async function findSystemPython() {
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
        const out = await execText(
          'reg',
          ['query', `${hive}\\SOFTWARE\\Python\\PythonCore\\${version}\\InstallPath`, '/ve', '/reg:64'],
          { timeout: 5_000 }
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
        const out = await execText(pyExe, [`-${version}`, '-c', 'import sys; print(sys.executable)'], {
          timeout: PROBE_TIMEOUT_MS
        })

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

// findGitBash — locate bash.exe on Windows. Resolves HERMES_GIT_BASH_PATH
// first (mirrors tools/environments/local.py:_find_bash), then PortableGit,
// standard install locations, and finally PATH.
function findGitBash() {
  return _findGitBash({
    isWindows: IS_WINDOWS,
    env: process.env,
    fileExists,
    findOnPath
  })
}

function getVenvPython(venvRoot) {
  return path.join(venvRoot, IS_WINDOWS ? path.join('Scripts', 'python.exe') : path.join('bin', 'python'))
}

// Map a selected interpreter back to the venv that OWNS it (the directory
// above bin/ or Scripts/), but only when that venv lives inside `root`.
// Returns null for system pythons — they own no site-packages we should mount.
//
// This exists because findPythonForRoot() probes `.venv` before `venv`, and a
// checkout can legitimately have BOTH (dev tooling venv + the CLI install
// venv, possibly on different Python versions). The interpreter and the
// site-packages placed on PYTHONPATH must come from the SAME venv: pairing a
// .venv 3.12 python with venv/lib/python3.11/site-packages makes the backend
// die on its first native import (pydantic_core) before the gateway binds —
// the renderer then reports "Gateway offline" on every profile.
function venvRootForPython(python: string, root: string) {
  const parent = path.dirname(python)
  const binName = path.basename(parent).toLowerCase()

  if (binName !== 'bin' && binName !== 'scripts') {
    return null
  }

  const candidate = path.dirname(parent)
  const relative = path.relative(root, candidate)

  if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) {
    return null
  }

  return candidate
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

function makeDashboardReadyFile() {
  const dir = path.join(app.getPath('userData'), 'backend-ready')
  fs.mkdirSync(dir, { recursive: true })

  return path.join(dir, `dashboard-${process.pid}-${Date.now()}-${crypto.randomBytes(6).toString('hex')}.json`)
}

const { resolveGitBinary, resolveGhBinary } = createExecutableDiscoveryRuntime({
  isWindows: IS_WINDOWS,
  fileExists,
  findOnPath,
  getHomePath: () => app.getPath('home'),
  execFileSync
})

function recentHermesLog() {
  return hermesLog.slice(-20).join('\n')
}

// Atomic file write: temp + rename (atomic on all platforms). Prevents
// partial writes on crash/power loss that corrupt JSON config files.
function writeFileAtomic(targetPath, data, encoding?: BufferEncoding) {
  const tmp = targetPath + '.tmp'
  fs.writeFileSync(tmp, data, encoding)
  fs.renameSync(tmp, targetPath)
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

// move/resize fire many times mid-drag; debounce to one write.
const schedulePersistWindowState = debounce(persistWindowState, 250)

// Zoom's primary store is a main-process JSON file. The renderer localStorage
// mirror lives under Electron's cache/storage folders, which crash recovery
// can move or recreate — wiping the zoom setting exactly when the user just
// recovered from a crash (#56726). JSON survives; localStorage is kept as a
// secondary mirror so pre-JSON installs migrate transparently on first read.
const DESKTOP_ZOOM_STATE_PATH = path.join(app.getPath('userData'), 'zoom-state.json')

function readZoomState() {
  try {
    const raw = JSON.parse(fs.readFileSync(DESKTOP_ZOOM_STATE_PATH, 'utf8'))
    const level = Number(raw?.zoomLevel)

    return Number.isFinite(level) ? level : null
  } catch {
    return null
  }
}

function writeZoomState(zoomLevel) {
  try {
    fs.mkdirSync(path.dirname(DESKTOP_ZOOM_STATE_PATH), { recursive: true })
    writeFileAtomic(DESKTOP_ZOOM_STATE_PATH, JSON.stringify({ zoomLevel }, null, 2))
  } catch (error) {
    rememberLog(`[zoom] json persist failed: ${error?.message || error}`)
  }
}

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
let quitPromptOpen = false
let quitConfirmedWithActiveWork = false

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

function readJson(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'))
  } catch {
    return null
  }
}

// Bootstrap-complete marker helpers. The marker is written by whichever
// installer ran: install.ps1, install.sh, the Rust bootstrap installer, or the
// first-launch bootstrap runner. It is provenance ("a bootstrap finished
// here"), NOT the launch gate -- activeRuntimeState() decides that, because a
// healthy runtime can predate the marker or outlive a repair that cleared it.
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
async function isActiveRuntimeUsable() {
  const venvPython = getVenvPython(VENV_ROOT)

  return (
    isHermesSourceRoot(ACTIVE_HERMES_ROOT) &&
    fileExists(venvPython) &&
    // Explicit await: a bare promise as the last `&&` operand only works via
    // async-return flattening; any operand appended after it would make the
    // expression truthy regardless of the probe result.
    (await canImportHermesCli(venvPython, {
      env: {
        PYTHONPATH: [ACTIVE_HERMES_ROOT, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter)
      }
    }))
  )
}

async function activeRuntimeState() {
  // We DELIBERATELY do NOT verify that the checkout is currently at the
  // pinned commit -- users update via the in-app update path or `hermes
  // update`, which moves HEAD legitimately. The marker only attests "a
  // desktop-managed bootstrap ran here at least once"; runtime usability is
  // what decides whether we can actually launch.
  return classifyActiveRuntime(readBootstrapMarker(), BOOTSTRAP_MARKER_SCHEMA_VERSION, await isActiveRuntimeUsable())
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

function requestOptionsWithHeaders(options: any = {}, headers = {}) {
  return {
    ...options,
    headers: {
      ...headers,
      ...(options.headers || {})
    }
  }
}

// Best-effort read of a gateway's advertised auth providers, cached per base
// URL for the life of the process. Used by the oauth pre-flight guard to tell
// a password-provider gateway (which cannot satisfy the bearer/cookie checks
// by design) from a real OAuth one. Any failure returns [] so callers keep the
// strict guard — backends predating /api/auth/providers are unaffected.
const gatewayAuthProvidersCache = new Map<string, any[]>()

async function gatewayAuthProviders(baseUrl, headers = {}) {
  const cached = gatewayAuthProvidersCache.get(baseUrl)

  if (cached) {
    return cached
  }

  let providers = []

  try {
    const body = (await fetchPublicJson(
      `${baseUrl}/api/auth/providers`,
      requestOptionsWithHeaders({ timeoutMs: 8_000 }, headers)
    )) as any

    if (Array.isArray(body?.providers)) {
      providers = body.providers
        .filter(p => p && typeof p === 'object')
        .map(p => ({ name: String(p.name || ''), supportsPassword: Boolean(p.supports_password) }))
        .filter(p => p.name)
    }

    gatewayAuthProvidersCache.set(baseUrl, providers)
  } catch {
    // Optional metadata — an unreadable list keeps the strict guard.
  }

  return providers
}

// Build the readiness probe for a connection's auth mode. A gated gateway
// must be probed with the SAME credentials the rest of the connection uses:
// an anonymous probe 401s forever against a live session, and it can never
// see the 404 that identifies a backend predating /api/health (the auth gate
// answers before the SPA catch-all). `probeIsCredentialed` tells
// waitForHermesReady how to read a 401 — rejected session vs gated route.
async function buildReadinessHealthProbe(baseUrl, authMode, token) {
  if (authMode === 'oauth') {
    return {
      probeHealth: (url: string, options: any = {}) =>
        requestWithOauthFallback(baseUrl, {
          ensureNativeAccessToken,
          requestWithBearer: bearer => fetchJson(url, null, { ...options, bearer }),
          requestWithCookie: () => fetchJsonViaOauthSession(url, options)
        }),
      probeIsCredentialed: true
    }
  }

  const probeAuth = resolveReadinessProbeAuth(authMode, null, token)

  if (probeAuth.kind === 'token' && probeAuth.token) {
    return {
      probeHealth: (url, options: any = {}) => fetchJson(url, probeAuth.token, options),
      probeIsCredentialed: true
    }
  }

  return { probeHealth: fetchPublicJson, probeIsCredentialed: false }
}

async function waitForHermes(baseUrl, token, signal?, authMode?, headers = {}) {
  const { probeHealth, probeIsCredentialed } = await buildReadinessHealthProbe(baseUrl, authMode, token)

  return waitForHermesReady(baseUrl, {
    token,
    signal,
    fetchPublicJson,
    fetchJson: probeIsCredentialed
      ? (url, _token, options = {}) => probeHealth(url, requestOptionsWithHeaders(options, headers))
      : fetchJson,
    probeHealth: (url, options = {}) => probeHealth(url, requestOptionsWithHeaders(options, headers)),
    probeIsCredentialed
  })
}

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

const { registerPowerResumeListeners } = createDesktopPowerRuntime({
  BrowserWindow,
  attachPowerResumeRemoteRevalidation,
  getMainWindow: () => mainWindow,
  ipcMain,
  powerMonitor,
  rememberLog,
  revalidateSuspectPoolAfterResume
})

function getAppIconPath() {
  // Fail-soft: skip candidates that exist but don't decode (truncated PNG in a
  // packaged app.asar previously crashed createWindow mid-session). Missing
  // every candidate is fine — the window then uses the platform default icon.
  try {
    return resolveAppIcon(APP_ICON_PATHS)
  } catch {
    return undefined
  }
}

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

const {
  getOauthSession,
  getOauthSessionForUrl,
  warmOauthCookieStore,
  hasOauthSessionCookie,
  hasLiveOauthSession,
  clearOauthSession,
  openOauthLoginWindow,
  fetchJsonViaOauthSession
} = createDesktopOauthSessionRuntime({
  app,
  BrowserWindow,
  electronNet,
  session,
  readDesktopConnectionsRegistry,
  readDesktopConnectionConfig,
  installRemoteHeaderRulesOnSession,
  headersForRemoteRequest,
  rememberLog,
  installWindowRendererLifecycle
})

const {
  _nativeTokenStoreIo,
  nativeAccessTokenCoordinator,
  ensureNativeAccessToken,
  hasNativeSession,
  postJsonNoAuth,
  mintGatewayWsTicket,
  freshGatewayWsUrl,
  resolvePortalBaseUrl,
  hasLivePortalSession,
  hasPortalAccessToken,
  renewPortalAccessSilently,
  openPortalLoginWindow,
  discoverCloudAgents,
  cloudAgentSilentSignIn
} = createDesktopConnectionAuthRuntime({
  app,
  BrowserWindow,
  fetchJson,
  fetchJsonViaOauthSession,
  encryptDesktopSecret: (value, options) => encryptDesktopSecret(value, options),
  decryptDesktopSecret: secret => decryptDesktopSecret(secret),
  ensureBackend: profile => ensureBackend(profile),
  getOauthSession,
  warmOauthCookieStore,
  hasOauthSessionCookie,
  openOauthLoginWindow,
  rememberRemoteWsHeaders: (url, headers) => rememberRemoteWsHeaders(url, headers),
  rememberLog
})

// ---------------------------------------------------------------------------
// Opt-in keychain encryption (secret-storage-policy.ts owns the decision).
// Default OFF: no safeStorage call is ever made, so a broken/locked macOS
// login keychain can never throw its password dialog on launch. Settings →
// Gateway exposes the toggle; flipping it re-encrypts (or decrypts) the
// stored secrets in place.
// ---------------------------------------------------------------------------
const {
  secretStoragePolicy,
  applySecretStorageEncryption,
  probeSecureTokenStorage,
  encryptDesktopSecret,
  decryptDesktopSecret,
  decryptRemoteHeaders,
  encryptIncomingRemoteHeaders,
  rememberRemoteWsHeaders,
  headersForRemoteRequest: headersForRemoteRequestImpl,
  installRemoteHeaderRulesOnSession: installRemoteHeaderRulesOnSessionImpl,
  installRemoteHeaderRules,
  readDesktopConnectionConfig: readDesktopConnectionConfigImpl,
  writeDesktopConnectionConfig,
  readDesktopConnectionsRegistry: readDesktopConnectionsRegistryImpl,
  writeDesktopConnectionsRegistry,
  sanitizeConnectionsRegistry,
  sanitizeRegistryConnection,
  saveRegistryConnection,
  migrateLegacyEncryptedSecretsOnce
} = createDesktopConnectionStorageRuntime({
  app,
  safeStorage,
  session,
  connectionConfigPath: DESKTOP_CONNECTION_CONFIG_PATH,
  connectionsRegistryPath: DESKTOP_CONNECTIONS_REGISTRY_PATH,
  profileNameRe: PROFILE_NAME_RE,
  nativeTokenStoreIo: _nativeTokenStoreIo,
  rememberLog,
  assertCanMutateRegistryConnection: id => managedConnectionUpdateGate.assertCanMutate(id),
  stopRegistryConnectionBackends,
  broadcastConnectionsChanged
})

// These callbacks are passed into earlier runtimes during module evaluation.
// Keep their declarations hoisted; the storage runtime is ready before any
// callback is invoked by the app lifecycle.
function headersForRemoteRequest(requestUrl: string) {
  return headersForRemoteRequestImpl(requestUrl)
}

function installRemoteHeaderRulesOnSession(sess: Electron.Session) {
  return installRemoteHeaderRulesOnSessionImpl(sess)
}

function readDesktopConnectionConfig() {
  return readDesktopConnectionConfigImpl()
}

function readDesktopConnectionsRegistry() {
  return readDesktopConnectionsRegistryImpl()
}

// Last-used profile and explicit app-wide default share the existing desktop
// preference file, but only the explicit action changes the default route.
const desktopProfilePreferences = createDesktopProfilePreferences(DESKTOP_PROFILE_CONFIG_PATH, {
  validateRoute: validateDesktopProfileRoute,
  onDefaultChanged: route => {
    for (const win of BrowserWindow.getAllWindows()) {
      if (!win.webContents.isDestroyed()) {
        win.webContents.send('hermes:profile:default:changed', route)
      }
    }
  }
})

function validateDesktopProfileRoute(route: DesktopProfileRoute) {
  if (
    route.connectionId &&
    !readDesktopConnectionsRegistry().connections.some((source: { id: string }) => source.id === route.connectionId)
  ) {
    throw new Error(`No connection with id "${route.connectionId}".`)
  }
}

function readActiveDesktopProfile() {
  return desktopProfilePreferences.readActive()
}

function writeActiveDesktopProfile(name) {
  return desktopProfilePreferences.remember(name)
}

// True when the given pid belongs to a running process whose command line
// contains "hermes", avoiding false positives from stale gateway.pid files
// whose PID was recycled by the OS to an unrelated process.
function isHermesProcess(pid) {
  try {
    process.kill(pid, 0) // signal 0 = existence check, no signal sent
  } catch {
    return false
  }

  // On macOS / Linux, check the command line to avoid PID recycling false positives.
  try {
    const cmdline = fs.readFileSync(`/proc/${pid}/cmdline`, 'utf8')

    return cmdline.includes('hermes')
  } catch {
    // /proc not available (macOS) — fall back to ps. Use -o args= to inspect
    // the full command line, not just the process name.  -o comm= would return
    // "python3" for any Python process, creating false positives.
    try {
      const { execSync } = require('child_process')
      const out = execSync(`ps -p ${pid} -o args=`, { encoding: 'utf8', timeout: 2000 })

      return out.includes('hermes')
    } catch {
      return false
    }
  }
}

// Seed active-profile.json from the best available signal when the file does
// not yet exist.  Runs exactly once (no-op once the file exists).  Priority:
//   1. Legacy ~/.hermes/active_profile (explicit CLI choice via hermes profile use)
//   2. Running gateway (gateway.pid with verified liveness + hermes identity)
//   3. state.db heuristics (hybrid recency×size score picks the primary workspace)
// The stored JSON includes _migrated:true so the renderer can optionally surface
// a one-time notification that the profile was auto-detected.
//
// Decision logic lives in profile-migration.ts (pure + unit-tested). This wrapper
// just wires Electron/Node fs into a MigrationDeps bag and delegates.
function migrateActiveProfileIfMissing() {
  migrateActiveProfileIfMissingPure(DESKTOP_PROFILE_CONFIG_PATH, {
    legacyActivePath: path.join(HERMES_HOME, 'active_profile'),
    hermesHome: HERMES_HOME,
    profilesRoot: path.join(HERMES_HOME, 'profiles'),
    existsSync: p => fs.existsSync(p),
    readFileSync: (p, enc) => fs.readFileSync(p, enc),
    statSync: p => fs.statSync(p),
    readdirSync: (p, opts) => fs.readdirSync(p, opts as { withFileTypes: true }),
    isHermesProcess,
    now: () => Date.now(),
    writeJson: (target, decision) => {
      // Mirror writeActiveDesktopProfile's atomic-write + parent-dir-create
      // semantics so the migration produces a file indistinguishable from a
      // user-driven profile switch.
      fs.mkdirSync(path.dirname(target), { recursive: true })
      writeFileAtomic(target, JSON.stringify(decision, null, 2))
    },
    isValidProfileName: p => PROFILE_NAME_RE.test(p)
  })
}

const { sanitizeDesktopConnectionConfig, coerceDesktopConnectionConfig, buildRemoteConnection } =
  createDesktopConnectionDescriptorRuntime({
    readDesktopConnectionConfig,
    decryptDesktopSecret,
    decryptRemoteHeaders,
    encryptDesktopSecret,
    probeSecureTokenStorage,
    hasNativeSession,
    hasLiveOauthSession,
    mintGatewayWsTicket,
    rememberRemoteWsHeaders
  })

const sshConnections = new Map<string, any>()

const sshIsolatedKeepalives = createSshIsolatedKeepaliveRegistry({
  log: chunk => sshRememberLog(chunk)
})

const desktopInstallationId = loadOrCreateInstallationId(DESKTOP_INSTALLATION_PATH)

// Managed SSH update lifecycle (#93042): while an update owns a registered
// SSH connection, the gate pauses new dials and dial-material mutations for
// that connection id; the durable recovery journal below survives a crash
// mid-transaction so the next launch can restore every drained scope.
const managedConnectionUpdateGate = new ManagedConnectionUpdateGate(
  connectionId =>
    readManagedSshRecoveryRecords().find(record => record.connectionId === connectionId)?.correlationId || null
)

const managedConnectionUpdates = new Map<string, Promise<any>>()
const managedConnectionRecoveries = new Map<string, Promise<void>>()
const managedPrimaryRestoreOwners = new Map<string, { correlationId: string; profile: string; source: any }>()
let managedUpdateQuitWait: Promise<void> | null = null
let managedUpdateQuitWaitDone = false

function assertCanMutateManagedPrimaryRouting() {
  const durableIds = readManagedSshRecoveryRecords().map(record => record.connectionId)

  const ids = new Set([
    ...managedConnectionUpdates.keys(),
    ...managedConnectionRecoveries.keys(),
    ...managedPrimaryRestoreOwners.keys(),
    ...durableIds
  ])

  if (ids.size > 0) {
    const error: any = new Error(
      `Primary connection routing cannot change while managed SSH update recovery is pending for ${[...ids].join(', ')}.`
    )

    error.code = 'managed-update-in-progress'
    throw error
  }
}

const {
  readManagedSshRecoveryRecords,
  persistManagedSshRecovery,
  markManagedSshRecoveryLaunching,
  clearManagedSshRecovery
} = createManagedSshRecoveryJournal(DESKTOP_MANAGED_SSH_RECOVERY_PATH)


const sshBootstrapCoordinator = createBootstrapCoordinator()
const sshTeardowns = createSshTeardownTracker()

const {
  sshScopeKey,
  sshRememberLog,
  teardownSshConnection,
  activeSshTerminalTarget,
  ensureTerminalBackend,
  resetPreviewReach,
  reachablePreviewUrl,
  effectiveSshConfigFingerprint,
  bootstrapSshConnection
} = createDesktopSshSessionRuntime({
  GUEST_ONBOARDING,
  SshConnection,
  adoptServedDashboardToken,
  backendDialClaims,
  backendScopeKey,
  buildRemoteConnection,
  connectWindowsRemote,
  connectionScopeKey,
  createDesktopSshBootstrapRuntime,
  desktopInstallationId,
  detectRemotePlatform,
  ensureBackend: profile => ensureBackend(profile),
  ensureRegistryBackend: (connectionId, profile) => ensureRegistryBackend(connectionId, profile),
  execText,
  fetchJson,
  managedConnectionUpdateGate,
  persistSshConnectionToken: (profile, source, token, id) => persistSshConnectionToken(profile, source, token, id),
  pickLocalPort,
  primaryProfileKey,
  readDesktopConnectionConfig,
  readDesktopConnectionsRegistry,
  redactSecrets,
  registrySshPoolScopeByConnectionId,
  registrySshScopeForWindowRoute,
  rememberLog,
  remoteLifecycle,
  resolveDesktopRemoteRoute,
  resolveRemoteSshDashboardProfile,
  sshBootstrapCoordinator,
  sshConnections,
  sshIsolatedKeepalives,
  sshOwnershipId,
  sshTeardowns,
  teardownSshState,
  terminalIpc: { disposeTerminalSessionsForSshScope: scope => terminalIpc.disposeTerminalSessionsForSshScope(scope) },
  terminateOwnedWindowsDashboardForUpdate,
  v1SshTerminalPoolKey,
  waitForHermes,
  windowConnectionRoutes: { get: webContentsId => windowConnectionRoutes.get(webContentsId) }
})

const {
  persistSshConnectionToken: persistSshConnectionTokenImpl,
  resolveRemoteBackend,
  profileHasRemoteOverride,
  configuredRemoteProfileNames,
  globalRemoteActive: globalRemoteActiveImpl,
  registryPrimaryIsRemote,
  primaryBackendIsRemote,
  fetchJsonForProfile,
  requestJsonForProfile,
  probeRemoteAuthMode,
  testDesktopConnectionConfig,
  fetchConnectionStatus
} = createDesktopConnectionProbeRuntime({
  bootstrapSshConnection,
  buildRemoteConnection,
  coerceDesktopConnectionConfig,
  decryptDesktopSecret,
  decryptRemoteHeaders,
  encryptDesktopSecret,
  ensureBackend: profile => ensureBackend(profile),
  fetchJsonForBackend,
  fetchPublicJson,
  managedConnectionUpdateGate,
  managedPrimaryRestoreOwners,
  managedSshConfig: (source, profile) => managedSshConfig(source, profile),
  mintGatewayWsTicket,
  primaryProfileKey,
  readDesktopConnectionConfig,
  readDesktopConnectionsRegistry,
  sshRememberLog,
  startHermes,
  writeDesktopConnectionConfig,
  writeDesktopConnectionsRegistry
})

// Earlier runtimes receive these callbacks before the probe runtime is built.
function persistSshConnectionToken(profile, source, token, registryConnectionId = '') {
  return persistSshConnectionTokenImpl(profile, source, token, registryConnectionId)
}

function globalRemoteActive() {
  return globalRemoteActiveImpl()
}

function resetBootProgressForReconnect() {
  firstRunBoot.updateBootProgress(
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
  void localBackendLifecycle.stop(child).catch(error => rememberLog(`Backend teardown failed: ${error.message}`))
}

// Soft gateway-mode apply: tear down the primary without resetting boot UI or
// reloading the renderer. The shell stays up; the renderer wipes session lists
// (so skeletons retrigger) and re-dials. Distinct from hard re-home (profile
// switch / crash recovery), which still resets boot progress + reloads.
function resetHermesConnection({ soft = false } = {}) {
  backendStartFailure = null
  remoteReauthFailure = null
  remoteLiveness.clear()
  // The next startHermes() re-reads active-profile.json for its launch profile.
  primaryProfilePin.clear()
  const hermesProcess = invalidatePrimaryConnection()
  stopBackendChild(hermesProcess)

  if (!soft) {
    resetBootProgressForReconnect()
  }
}

// Every deliberate emptying of the primary slot goes through here so the
// dying child's stale exit reads as intentional (see primaryRecoverySuppressed).
function invalidatePrimaryConnection() {
  primaryRecoverySuppressed = true

  return backendConnectionState.invalidate()
}

// Re-home the primary backend: reset connection state, then wait for the live
// dashboard process to actually exit (SIGKILL after 5s) so the next
// startHermes() spawns fresh instead of racing the dying one. Shared by the
// connection-config and profile switch flows.
async function teardownPrimaryBackendAndWait({ soft = false } = {}) {
  // Capture the reference before resetHermesConnection() invalidates it.
  const hermesProcess = backendConnectionState.getProcess()
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

// Registry lifecycle push: a connection was removed or materially edited, so
// every window must tear down (and, for edits, re-dial) its secondary sockets
// scoped to that connection. Without this, a removed remote/cloud source keeps
// its renderer WebSocket open and streaming as a ghost, and an edited one
// keeps talking to the OLD endpoint until idle-reap.
function broadcastConnectionsChanged(payload: { connectionId: string; reason: 'removed' | 'saved' | 'updated' }) {
  for (const win of BrowserWindow.getAllWindows()) {
    const { webContents } = win

    if (webContents && !webContents.isDestroyed()) {
      webContents.send('hermes:connections:changed', payload)
    }
  }
}

const backendExitWaits = new Map<any, Promise<void>>()

function waitForBackendExit(child, timeoutMs = 5000) {
  const existing = backendExitWaits.get(child)

  if (existing) {
    return existing
  }

  const waiting = waitForBackendExitImpl(child, { forceKillProcessTree, isWindows: IS_WINDOWS }, timeoutMs)
  backendExitWaits.set(child, waiting)
  void waiting.then(
    () => backendExitWaits.delete(child),
    () => backendExitWaits.delete(child)
  )

  return waiting
}

// The profile the primary (window) backend was actually LAUNCHED as. Pinned by
// startHermes() and cleared when the primary is torn down; while a primary is
// live this must NOT follow active-profile.json (see primary-profile-pin.ts).
const primaryProfilePin = new PrimaryProfilePin()

function primaryProfileKey() {
  return primaryProfilePin.resolve(readActiveDesktopProfile)
}

// Options describing the current connection setup for `resolveProfileBackendRoute`.
function profileRouteOptions(profile, request?) {
  const config = readDesktopConnectionConfig()
  const sshOverride = profileSshOverride(config, profile)
  const key = connectionScopeKey(profile) || primaryProfileKey()

  return {
    // A desktop profile can be only a client-side routing alias. Keep backend
    // endpoint filters in the SSH target's namespace (e.g. mara → default).
    backendProfile: sshOverride?.remoteProfile,
    globalRemote: globalRemoteActive(),
    primaryProfile: primaryProfileKey(),
    profileRemoteOverride: Boolean(profileRemoteOverride(config, profile) || sshOverride),
    // The primary profile's own backend resolves to a remote host (its
    // per-profile override, env, or global). Unknown sub-profiles on that
    // gateway must route THROUGH it, not spawn local backends (#88296).
    primaryRemoteActive: primaryBackendIsRemote(),
    // A stored per-profile entry (local or remote) — pins this profile to
    // its own backend; absent entries inherit the primary's remote.
    ownEntry: Boolean((config.profiles || {})[key]),
    isolatedBackend: ISOLATED_BACKEND,
    requestMethod: request?.method,
    requestPath: request?.path
  }
}

// Managed SSH restore borrows the same gate, pools, and coordinator that startup
// and before-quit use. It is composed after connection admission is ready.

// Stop every pooled backend and ssh scope owned by a registry connection —
// called when the connection is removed from the registry.
async function stopRegistryConnectionBackends(connectionId) {
  const prefix = backendScopePrefix(connectionId)

  for (const key of [...backendPool.keys()]) {
    if (String(key).startsWith(prefix)) {
      stopPoolBackend(key)
    }
  }

  const sshScopes = new Set([
    ...[...sshConnections.keys()].filter(scope => String(scope).startsWith(prefix)),
    ...[...sshBootstrapCoordinator.active].map(entry => entry.scope).filter(scope => String(scope).startsWith(prefix))
  ])

  await Promise.all(
    [...sshScopes].map(async scope => {
      await sshBootstrapCoordinator.cancelAndWait(scope)
      await teardownSshConnection(scope)
    })
  )
}

// Mark a pool profile as recently used so the idle reaper spares it. The
// renderer calls this when it opens a profile's chat WS and periodically while
// streaming, since the main process can't see the direct renderer↔backend WS.
// It also reports whether a prompt turn currently leases the backend: a
// foreground dial that must retire a resident skips leased ones early. That
// flag is an optimisation, never the proof — the backend probe is (see
// pool-retire.ts). Shape from #104871 by @bounce12340.
function touchPoolBackend(profile, options: { activeTurn?: boolean } = {}) {
  for (const key of poolTouchKeys(profile)) {
    const entry = backendPool.get(key)

    if (entry) {
      entry.lastActiveAt = Date.now()

      if (typeof options.activeTurn === 'boolean') {
        entry.activeTurn = options.activeTurn
      }

      return
    }
  }
}

// Evict least-recently-used SPAWNED pool backends until at most `keep` remain —
// but only ever evict backends without a live renderer socket (stale beyond the
// keepalive window). When every backend is actively kept alive we let the pool
// exceed the soft cap rather than kill a running session. Process-less
// descriptor entries (remote/cloud registry sources, per-profile remote
// overrides — `entry.process === null`) are excluded from the cap entirely:
// they hold no local process, so counting them used to let a roster refresh
// across N registered remote connections LRU-evict a REAL local backend that
// was merely idle past the keepalive window. Descriptors are still reclaimed
// by the idle reaper.
async function evictLruPoolBackends(keep) {
  return poolRetirer.evictTo(Math.max(0, keep), POOL_KEEPALIVE_FRESH_MS)
}

function startPoolIdleReaper() {
  if (poolIdleReaper) {
    return
  }

  poolIdleReaper = setInterval(() => {
    const now = Date.now()

    for (const [profile, entry] of [...backendPool.entries()]) {
      if (now - (entry.lastActiveAt || 0) > poolIdleMs()) {
        // Remote descriptors hold no child/slot. Local children require the
        // same admission authority as foreground and LRU reclamation.
        const retiring = entry.process ? poolRetirer.retireIdle(profile, poolIdleMs()) : stopPoolBackend(profile)

        void retiring.catch(error => rememberLog(`Pool idle retirement failed: ${String(error)}`))
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

function releaseLocalBackendSlot(entry: any) {
  if (!entry) {
    return
  }

  const release = entry.releaseLocalBackendSlot
  const request = entry.localBackendSpawnRequest as LocalBackendSpawnRequest | null
  entry.releaseLocalBackendSlot = null
  entry.localBackendSlotKey = null
  entry.localBackendSpawnRequest = null

  if (release) {
    release()
  } else {
    request?.cancel()
  }
}

// `releaseSlot` must be false once `entry.process` exists: the lease has to
// stay held until the child has actually exited (pool-spawn-coordinator
// invariant), and teardownFailedLocalBackend releases it after that exit. A
// pre-spawn release here would turn that post-exit release into a no-op and
// let a successor spawn while the superseded child is still alive.
function assertPoolEntryStillOwned(poolKey: string, entry: any, { releaseSlot = true } = {}) {
  if (localBackendLifecycle.signal.aborted || backendPool.get(poolKey) !== entry) {
    if (releaseSlot) {
      releaseLocalBackendSlot(entry)
    }

    throw new Error(`Profile backend start for "${poolKey}" was cancelled before it became ready.`)
  }
}

const failedLocalBackendTeardowns = new WeakMap<object, Promise<void>>()

function teardownFailedLocalBackend(poolKey: string, entry: any): Promise<void> {
  const existing = failedLocalBackendTeardowns.get(entry)

  if (existing) {
    return existing
  }

  if (backendPool.get(poolKey) === entry) {
    backendPool.delete(poolKey)
  }

  const child = entry.process

  const teardown = releaseLocalBackendSlotAfterExit(
    () => releaseLocalBackendSlot(entry),
    async () => {
      stopBackendChild(child)
      await waitForBackendExit(child)
      releaseBackendChild(child)
    }
  )

  // Keep the settled promise in the WeakMap for the lifetime of this entry.
  // Error + exit + outer catch may all request cleanup; none may run it twice.
  failedLocalBackendTeardowns.set(entry, teardown)

  return teardown
}

// Spawn an additional dashboard backend pinned to a named profile. Mirrors the
// local-spawn portion of startHermes() but without the boot-progress UI,
// bootstrap, or remote handling (those belong to the primary backend only).
// `opts.forceLocal` skips remote resolution entirely (the registry 'local'
// entry means THIS machine regardless of the v1 routing table); `opts.poolKey`
// is the backendPool key when it differs from the profile name (composite
// registry scopes) so the exit/error cleanup evicts the right entry.
function spawnPoolBackend(
  profile,
  entry,
  opts: { forceLocal?: boolean; poolKey?: string; unscopableRequest?: boolean } = {}
) {
  return localBackendLifecycle.start(() => runPoolBackendStart(profile, entry, opts))
}

async function runPoolBackendStart(
  profile,
  entry,
  opts: { forceLocal?: boolean; poolKey?: string; unscopableRequest?: boolean } = {}
) {
  const poolKey = opts.poolKey || profile

  await reapOrphanedBackendsOnce()
  profileDeletionGate.assertCanStart(profile)

  // A profile may point at its OWN remote backend (connection.json
  // `profiles[name]`), or inherit the app-wide remote (env / global settings).
  // In either case there is no local child to spawn — we just verify the
  // remote is reachable and hand back its connection descriptor. The pool
  // entry keeps `entry.process === null`, which stopPoolBackend/evict already
  // tolerate.
  const remote = opts.forceLocal ? null : await resolveRemoteBackend(profile, { poolKey })
  profileDeletionGate.assertCanStart(profile)

  if (remote) {
    await waitForHermes(remote.baseUrl, remote.token, undefined, remote.authMode, remote.headers)

    // Recorded on the entry so revalidation can probe this descriptor without
    // awaiting connectionPromise, which may still be pending for a sibling.
    entry.remoteBaseUrl = remote.baseUrl

    return {
      ...remote,
      profile,
      logs: hermesLog.slice(-80),
      ...getWindowState()
    }
  }

  // Everything below starts a LOCAL `hermes serve` child. Multiplex-only says
  // the host has exactly one, and routing (resolveProfileBackendRoute case 6)
  // keeps local profiles off this path — this is the backstop that makes the
  // pool spawn path genuinely unreachable rather than merely unused.
  // Same options object the router reads, so the guard cannot drift from it
  // (profileRouteOptions folds the per-profile SSH override into
  // profileRemoteOverride; a hand-rolled term here missed that).
  const guardRoute = profileRouteOptions(profile)

  assertNoSecondLocalBackend(poolKey, {
    isolated: guardRoute.isolatedBackend,
    primaryRemoteActive: guardRoute.primaryRemoteActive,
    profileRemoteOverride: opts.forceLocal ? false : guardRoute.profileRemoteOverride,
    unscopableRequest: opts.unscopableRequest
  })

  // Bound the slot wait BELOW the renderer's backend-boot budget (45s): once
  // the renderer has given up on this spawn, a ticket still queued for the
  // pool-idle window (10 min) would hold the pool key hostage and every
  // later click on the profile would join that stale wait. Failing here
  // surfaces the "all N slots busy" reason instead of a generic boot timeout.
  // The caller stamped entry.spawnPriority from its own request; a foreground
  // dial that joined the claim before this entry existed left a mark instead.
  if (takeForegroundSpawn(poolKey, profile)) {
    entry.spawnPriority = 'foreground'
  }

  const spawnPriority: LocalBackendSpawnPriority = spawnPriorityFrom(entry.spawnPriority)

  assertPoolEntryStillOwned(poolKey, entry)

  if (spawnPriority === 'background' && !backgroundSlotRetryBackoff.canAttempt(poolKey)) {
    throw new BackgroundSlotRetryDeferredError(profile)
  }

  // The arbiter subscribes to the actual coordinator queue, so a later
  // foreground promotion receives reclamation too, not just fresh starts.
  const spawnRequest = localBackendSpawnCoordinator.request(poolKey, {
    timeoutMs: POOL_SLOT_WAIT_MS,
    priority: spawnPriority
  })

  entry.localBackendSlotKey = poolKey
  entry.localBackendSpawnRequest = spawnRequest

  if (spawnRequest.queued) {
    rememberLog(
      `Profile backend "${profile}" waiting for a free local slot (${localBackendSpawnCoordinator.activeCount}/${poolMaxBackends()} busy, ${localBackendSpawnCoordinator.queuedCount} queued)`
    )
  }

  const cancelRequest = () => spawnRequest.cancel()
  localBackendLifecycle.signal.addEventListener('abort', cancelRequest, { once: true })

  try {
    entry.releaseLocalBackendSlot = await spawnRequest.acquired
    backgroundSlotRetryBackoff.clear(poolKey)
  } catch (error) {
    if (isBackgroundSlotWaitTimeout(error)) {
      backgroundSlotRetryBackoff.recordFailure(poolKey)
    }

    throw error
  } finally {
    localBackendLifecycle.signal.removeEventListener('abort', cancelRequest)
  }

  if (entry.localBackendSpawnRequest === spawnRequest) {
    entry.localBackendSpawnRequest = null
  }

  assertPoolEntryStillOwned(poolKey, entry)

  const token = crypto.randomBytes(32).toString('base64url')

  // Same update mutual exclusion as the primary window's waitForLocalStart
  // (#73822): pool backends spawn from the same venv, so an ungated respawn
  // during applyUpdates' critical section re-locks the venv and trips the
  // venv-blocker preflight. No boot-progress UI here — pool backends boot
  // silently for background profiles — so we only log while parked.
  {
    let poolAnnounced = false

    const clearance = await waitForUpdateClearance(updateGateDeps(), {
      signal: localBackendLifecycle.signal,
      isCancelled: () => backendPool.get(poolKey) !== entry,
      onWaitTick: reason => {
        if (!poolAnnounced) {
          poolAnnounced = true
          rememberLog(`[updates] update in progress (${reason}); deferring pool backend start for profile "${profile}"`)
        }
      },
      pollMs: UPDATE_WAIT_POLL_MS,
      timeoutMs: UPDATE_WAIT_TIMEOUT_MS
    })

    if (clearance === 'cancelled') {
      assertPoolEntryStillOwned(poolKey, entry)
    }
  }

  profileDeletionGate.assertCanStart(profile)
  assertPoolEntryStillOwned(poolKey, entry)

  // --profile wins over the inherited HERMES_HOME env (see _apply_profile_override
  // step 3 in hermes_cli/main.py), so the child re-homes to this profile.
  // --port 0: the OS assigns an ephemeral port; the child announces it on stdout.
  const backendArgs = ['--profile', profile, 'serve', '--host', '127.0.0.1', '--port', '0']

  const backend = await ensureRuntime(await resolveHermesBackend(backendArgs), () =>
    assertPoolEntryStillOwned(poolKey, entry)
  )

  // Route old runtimes (no `serve`) through the legacy `dashboard --no-open`.
  backend.args = await getBackendArgsForRuntime(backend)
  assertPoolEntryStillOwned(poolKey, entry)
  const hermesCwd = resolveHermesCwd()
  const webDist = resolveWebDist()
  const readyFile = backend.readyFile ? makeDashboardReadyFile() : null

  // Guard BEFORE the "Starting" line: a profile that only exists on a remote
  // backend (remote-primary desktop asked for a forced-local child) rejects
  // here, and logging "Starting" first left an orphaned line with no READY
  // and no exit — the exact undiagnosable burst signature in remote-gateway
  // user bundles (Aug 2026, Dash's report).
  assertLocalProfileCanStart(profile, profileDeletionGate, key =>
    directoryExists(path.join(HERMES_HOME, 'profiles', key))
  )
  rememberLog(`Starting Hermes backend for profile "${profile}" via ${backend.label}`)

  const parentStartMarker = await desktopParentStartMarker()
  const backendNonce = crypto.randomBytes(16).toString('hex')
  const parentIdentityEnv = parentWatchdogEnv(process.pid, parentStartMarker, backendNonce)
  assertPoolEntryStillOwned(poolKey, entry)

  const child = spawnOwnedBackend(
    backend.command,
    backend.args,
    hiddenWindowsChildOptions({
      cwd: hermesCwd,
      env: desktopBackendSpawnEnv(
        {
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
          // Exact parent identity lets the backend self-exit after an unclean
          // Desktop death without mistaking a reused PID for its owner. If the
          // optional marker probe fails, retain legacy PID-only tracking.
          ...parentIdentityEnv,
          HERMES_WEB_DIST: webDist,
          ...(readyFile ? { HERMES_DESKTOP_READY_FILE: readyFile } : {})
        },
        GUEST_ONBOARDING
      ),
      shell: backend.shell,
      stdio: ['ignore', 'pipe', 'pipe']
    })
  )

  entry.process = child
  entry.token = token
  registerLocalBackendExitFinalizer(backendPool, poolKey, entry, () => releaseLocalBackendSlot(entry))
  // Buffer stdout+stderr from the instant of spawn (#93608): an early crash's
  // traceback must survive into the claim error and the before-ready exit
  // message instead of a bare exit code. rememberLog attaches later, after
  // the claim, and would miss anything printed before it.
  const outputTail = createBackendOutputTail()
  outputTail.attach(child)

  let ready = false
  let rejectStart = null

  const startFailed = new Promise((_resolve, reject) => {
    rejectStart = reject
  })

  // Exit/error can now arrive while the ownership claim is still pending.
  startFailed.catch(() => {})

  child.once('error', error => {
    rememberLog(`Hermes backend for profile "${profile}" failed to start: ${error.message}`)
    void teardownFailedLocalBackend(poolKey, entry).catch(cleanupError => {
      rememberLog(
        `Hermes backend for profile "${profile}" cleanup failed: ${cleanupError instanceof Error ? cleanupError.message : String(cleanupError)}`
      )
    })
    rejectStart?.(error)
  })
  child.once('exit', (code, signal) => {
    rememberLog(formatBackendExitLine(`Hermes backend for profile "${profile}" exited`, code, signal, outputTail))
    releaseBackendChild(child)

    if (!ready) {
      rejectStart?.(
        new Error(
          `Hermes backend for profile "${profile}" exited before it became ready (${signal || code}).${outputTail.describe()}`
        )
      )
    }
  })

  // Start watching for the READY announcement BEFORE any await (#60323):
  // stdout is already flowing into the tail, and Node streams never replay
  // consumed chunks to late listeners — a sentinel printed while
  // claimBackendChild runs would otherwise be lost forever, timing out a
  // healthy backend. The tail-buffer accessor covers any residual gap.
  const portAnnouncement = waitForDashboardPortAnnouncement(child, {
    bufferedOutput: () => outputTail.text(),
    describeOutputTail: () => outputTail.describe(),
    readyFile
  })

  portAnnouncement.catch(() => {})
  await claimBackendChild(child, `${backend.command} ${backend.args.join(' ')}`, profile, backendNonce, outputTail)
  assertPoolEntryStillOwned(poolKey, entry, { releaseSlot: false })

  child.stdout.on('data', rememberLog)
  child.stderr.on('data', rememberLog)

  const port = await Promise.race([portAnnouncement, startFailed])
  assertPoolEntryStillOwned(poolKey, entry, { releaseSlot: false })

  if (readyFile) {
    fs.unlink(readyFile, () => {})
  }

  entry.port = port

  const baseUrl = `http://127.0.0.1:${port}`
  await Promise.race([waitForHermes(baseUrl, token), startFailed])
  assertPoolEntryStillOwned(poolKey, entry, { releaseSlot: false })
  ready = true

  const authToken = await adoptServedDashboardToken(baseUrl, token, {
    childAlive: () => child.exitCode === null && !child.killed,
    label: `Hermes backend for profile "${profile}"`,
    rememberLog
  })

  assertPoolEntryStillOwned(poolKey, entry, { releaseSlot: false })

  entry.token = authToken

  // Verify the WebSocket session token before declaring backend ready.
  // HTTP /api/status can pass while WS auth fails (separate transport, separate guards).
  const wsUrl = `ws://127.0.0.1:${port}/api/ws?token=${encodeURIComponent(authToken)}`
  const wsProbe = await probeGatewayWebSocket(wsUrl, { WebSocketImpl: globalThis.WebSocket })
  assertPoolEntryStillOwned(poolKey, entry, { releaseSlot: false })

  if (!wsProbe.ok) {
    throw new Error(
      `Hermes backend for profile "${profile}" is HTTP-reachable but the WebSocket (/api/ws) rejected the session token: ${wsProbe.reason}`
    )
  }

  return {
    baseUrl,
    mode: 'local',
    source: 'local',
    authMode: 'token',
    token: authToken,
    profile,
    wsUrl,
    logs: hermesLog.slice(-80),
    ...getWindowState()
  }
}

// Bounded, deduplicated pool teardown (see pool-stop.ts): every stop path —
// idle reaper, LRU eviction, profile delete/rename, quit — shares one
// in-flight stop per key and retains the process handle until the bounded
// SIGTERM -> SIGKILL escalation in waitForBackendExit() resolves. Previously
// SIGTERM + immediate entry delete dropped the handle and a slow child
// survived detached under PID 1.
const poolStopper = createPoolStopper({
  pool: backendPool,
  stopChild: child => stopBackendChild(child),
  waitForExit: child => waitForBackendExit(child),
  // Remote / SSH-isolated pool entries keep `process: null`. Child exit is
  // immediate; hold the same in-flight fence through bootstrap drain + SSH
  // teardown so a reconnect cannot publish into a dying scope (#106935).
  afterStop: async key => {
    try {
      await sshBootstrapCoordinator.cancelAndWait(key, () => teardownSshConnection(key))
    } catch (err) {
      // The idle reaper calls stopPoolBackend un-awaited; a failed SSH teardown
      // must not surface as an unhandled rejection or block the pool fence.
      sshRememberLog(`[ssh-teardown] ${key}: ${String(err)}`)
    }
  }
})

function stopPoolBackend(profile: string): Promise<void> {
  const entry = backendPool.get(profile)

  const stopping = releaseLocalBackendSlotAfterExit(
    () => releaseLocalBackendSlot(entry),
    () => poolStopper.stop(profile)
  )

  // Fire-and-forget callers still need diagnostics; awaiters receive the
  // rejection, while physical ownership and the exit finalizer remain live.
  void stopping.catch(error => {
    rememberLog(`Profile backend "${profile}" stop failed: ${error instanceof Error ? error.message : String(error)}`)
  })

  return stopping
}

// Tell every window the pooled backend under `poolKey` is being retired so the
// renderer parks that scope (wantOpen=false) instead of redialing into the
// slot it just vacated. Fired BEFORE the SIGTERM (pool-retire.ts contract).
function broadcastPoolBackendRetiring(poolKey: string) {
  for (const win of BrowserWindow.getAllWindows()) {
    const { webContents } = win

    if (webContents && !webContents.isDestroyed()) {
      webContents.send('hermes:pool:retiring', { poolKey })
    }
  }
}

const poolRetirer = createPoolRetirer({
  pool: backendPool,
  coordinator: localBackendSpawnCoordinator,
  ...createPoolRetirementClient(fetchJson),
  stopBackend: stopPoolBackend,
  onRetiring: broadcastPoolBackendRetiring,
  log: rememberLog
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

// Returns the profile name whose backend was torn down, or null when the
// request is not a profile-delete.  The caller uses this to skip ensureBackend
// for the just-torn-down profile — otherwise ensureBackend respawns a pool
// backend whose ensure_hermes_home() recreates the deleted profile directory.
//
// The routing *decision* (which branch fires, what profile name gets
// returned) lives in the pure decideProfileDeleteAction() in
// profile-delete-routing.ts; this function only performs the side effects
// that decision calls for.
async function prepareProfileDeleteRequest(request) {
  const profile = profileNameFromDeleteRequest(request)

  const decision = decideProfileDeleteAction(profile, {
    isDefaultProfile: p => p === 'default',
    isValidProfileName: p => PROFILE_NAME_RE.test(p),
    primaryProfileKey
  })

  if (decision.action === 'noop') {
    return null
  }

  if (decision.action === 'teardown-primary') {
    writeActiveDesktopProfile('default')
    await Promise.all([teardownPrimaryBackendAndWait(), teardownPoolBackendAndWait(decision.profile)])

    return decision.profile
  }

  await teardownPoolBackendAndWait(decision.profile)

  return decision.profile
}

async function prepareProfileRenameRequest(request) {
  return prepareProfileRenameLifecycle(request, {
    isValidProfileName: profile => PROFILE_NAME_RE.test(profile),
    primaryProfileKey,
    reloadPrimaryWindow: () => {
      mainWindow?.reload()
    },
    restartPrimaryBackend: async () => {
      await startHermes()
    },
    teardownPoolBackendAndWait,
    teardownPrimaryBackendAndWait,
    writeActiveDesktopProfile: profile => {
      writeActiveDesktopProfile(profile)
    }
  })
}

// ── Attach-first: one backend per HOST (multiplex-only) ───────────────────
// Escape hatch: a dedicated, private backend for this app instead of the host's.
const ISOLATED_BACKEND = process.env.HERMES_DESKTOP_ISOLATED_BACKEND === '1'
const ATTACHED_LIVENESS_POLL_MS = 15_000
let attachedBackendMonitor: NodeJS.Timeout | null = null
let hostSpawnReservation: SpawnReservation | null = null

function stopAttachedBackendMonitor() {
  if (attachedBackendMonitor) {
    clearInterval(attachedBackendMonitor)
    attachedBackendMonitor = null
  }
}

/**
 * An attached backend has no child process, so `child.exit` can never drive
 * recovery. Poll its readiness instead; a backend that dies under us
 * invalidates the connection and hands the respawn to the same supervisor path
 * a dead child would (which re-runs discovery and spawns, since the host now
 * has no backend).
 */
function startAttachedBackendMonitor(attached: AttachedBackend) {
  stopAttachedBackendMonitor()

  attachedBackendMonitor = setInterval(() => {
    void waitForHermes(attached.baseUrl, attached.token, undefined, 'token', {}).catch(() => {
      stopAttachedBackendMonitor()
      rememberLog(`[attach] attached backend on ${attached.baseUrl} (pid ${attached.pid}) is gone; recovering`)
      invalidatePrimaryConnection()
      scheduleUnexpectedPrimaryRecovery({ error: 'The Hermes backend this app attached to exited.', ready: true })
    })
  }, ATTACHED_LIVENESS_POLL_MS)

  attachedBackendMonitor.unref?.()
}

/** Discover and attach to the host's running backend; null means "spawn one". */
function attachToRunningHostBackend(): Promise<AttachedBackend | null> {
  const options = { isolated: ISOLATED_BACKEND, ledgerPath: spawnLedgerPath(HERMES_HOME, path.join) }

  return attachOrReserveSpawn(options, hostBackendAttachDeps(), hostSpawnGateDeps())
    .then(outcome => {
      if ('attached' in outcome) {
        releaseHostSpawnReservation()

        return outcome.attached
      }

      hostSpawnReservation = outcome.reservation

      return null
    })
    .catch(error => {
      // Discovery must never be able to block boot: fall through to spawning.
      rememberLog(`[attach] host backend discovery failed (${error.message}); spawning our own`)

      return null
    })
}

function hostBackendAttachDeps() {
  return {
    log: rememberLog,
    readLedger: (target: string) => {
      try {
        return fs.readFileSync(target, 'utf8')
      } catch {
        return null
      }
    },
    probeWebSocket: (wsUrl: string) => probeGatewayWebSocket(wsUrl, { WebSocketImpl: globalThis.WebSocket }),
    resolveServedToken: (baseUrl: string) => resolveServedDashboardToken(baseUrl, ''),
    waitForReady: (baseUrl: string, token: string) => waitForHermes(baseUrl, token, undefined, 'token', {})
  }
}

function hostSpawnGatePath() {
  return path.join(HERMES_HOME, 'desktop-backend-spawn.json')
}

function hostSpawnGateDeps() {
  return {
    now: () => Date.now(),
    read: () => {
      try {
        const record = JSON.parse(fs.readFileSync(hostSpawnGatePath(), 'utf8'))
        const owner = Number(record?.pid)

        if (!Number.isInteger(owner) || owner <= 0) {
          return null
        }

        // A gate whose owner is gone is no gate at all.
        try {
          process.kill(owner, 0)
        } catch {
          return null
        }

        return { ownerAlive: true, startedAt: Number(record?.startedAt) || 0 }
      } catch {
        return null
      }
    },
    take: () => {
      const gatePath = hostSpawnGatePath()

      try {
        fs.writeFileSync(gatePath, JSON.stringify({ pid: process.pid, startedAt: Date.now() }), { mode: 0o600 })
      } catch {
        // A gate we cannot write is a race we cannot win; spawning anyway is
        // exactly today's behaviour, so never fail boot over it.
      }

      return () => {
        try {
          fs.unlinkSync(gatePath)
        } catch {
          // Already gone / never written.
        }
      }
    },
    sleep: (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms))
  }
}

function releaseHostSpawnReservation() {
  hostSpawnReservation?.release()
  hostSpawnReservation = null
}

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
  sandboxState: {
    get fallbackActive() {
      return windowsSandboxFallbackActive
    },
    set fallbackActive(value) {
      windowsSandboxFallbackActive = value
    },
    get fallbackSticky() {
      return windowsSandboxFallbackSticky
    },
    set fallbackSticky(value) {
      windowsSandboxFallbackSticky = value
    },
    get fallbackReason() {
      return windowsSandboxFallbackReason
    },
    set fallbackReason(value: SandboxFallbackReason) {
      windowsSandboxFallbackReason = value
    },
    get noSandboxRelaunchAttempted() {
      return windowsNoSandboxRelaunchAttempted
    },
    set noSandboxRelaunchAttempted(value) {
      windowsNoSandboxRelaunchAttempted = value
    }
  },
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

ipcMain.handle('hermes:backend:touch', async (_event, profile, options) => {
  touchPoolBackend(profile, options)

  return { ok: true }
})
// Pool sizing (Settings → Advanced): device-local, live-applied. Main is
// authoritative (it owns the pool and the persisted copy); the returned
// limits are what actually took effect post-clamp.
ipcMain.handle('hermes:pool-limits:get', async () => ({ ...poolLimits }))
ipcMain.handle('hermes:pool-limits:set', async (_event, raw) => {
  const next = setPoolLimits({
    maxBackends: typeof raw?.maxBackends === 'number' ? raw.maxBackends : poolLimits.maxBackends,
    idleMs: typeof raw?.idleMs === 'number' ? raw.idleMs : poolLimits.idleMs
  })

  return { ok: true, limits: next }
})
ipcMain.handle('hermes:gateway:ws-url', async (_event, profile) => {
  return gatewayWsUrlIpcResult(() => freshGatewayWsUrl(profile))
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

ipcMain.handle('hermes:backend:recycle', async (_event, profile) => {
  // Models-page recovery after a code-skew 503 (#97046): kill the owned
  // SSH serve (if any) before the local child so reconnect cannot reuse a
  // stale lockfile. Soft primary teardown keeps the renderer shell mounted.
  await recycleOwnedBackend({
    notifyApplied: sendConnectionApplied,
    primaryProfile: primaryProfileKey(),
    profile: typeof profile === 'string' ? profile : '',
    teardownPool: teardownPoolBackendAndWait,
    teardownPrimary: () => teardownPrimaryBackendAndWait({ soft: true }),
    teardownSsh: value => teardownSshConnection(value || null)
  })

  return { ok: true }
})
ipcMain.handle('hermes:bootstrap:reset', async () => {
  // Renderer's "Reload and retry" path. Clear the latched failure and
  // reset connection state so the next startHermes() call restarts the
  // full backend flow (including a fresh runBootstrap pass).
  rememberLog('[bootstrap] reset requested by renderer; clearing latched failure')
  await teardownPrimaryBackendAndWait()
  bootstrapFailure = null
  backendStartFailure = null
  remoteReauthFailure = null
  firstRunBoot.getFirstRunSetupGate().resetForRetry()
  firstRunBoot.resetBootstrapSnapshot()

  return { ok: true }
})
ipcMain.handle('hermes:bootstrap:repair', async () => {
  // Forceful repair: force the next startHermes() through the full installer
  // (refreshing a broken/partial venv) and clear any latched failure + live
  // connection. The renderer reloads afterwards to re-drive the boot flow.
  //
  // We do NOT delete the bootstrap marker here. Repair is also reachable from
  // transient backend errors on a perfectly healthy install, and deleting the
  // marker in that case stranded the app in first-run setup with no way back
  // (#72166). The explicit flag carries the intent instead.
  bootstrapRepairAttempt += 1

  // Probe the live backend process so the guard can distinguish "venv is
  // genuinely broken" (force reinstall) from "backend is just transiently
  // stalled under GIL pressure" (#74874 — `event loop stalled` followed by
  // `ws ready frame send failed`, then renderer keeps reporting dead).
  const primaryProc = backendConnectionState.getProcess()

  const primaryBackendAlive = Boolean(
    primaryProc &&
    (primaryProc as { exitCode?: number | null }).exitCode === null &&
    (primaryProc as { signalCode?: string | null }).signalCode === null
  )

  const repairDecision = decideBootstrapRepair({
    attempt: bootstrapRepairAttempt,
    maxSoftAttempts: MAX_BOOTSTRAP_REPAIR_SOFT_ATTEMPTS,
    primaryBackendAlive
  })

  rememberLog(
    `[bootstrap] repair requested by renderer; forcing reinstall + clearing latched failure ` +
      `(attempt=${repairDecision.attempt}/${MAX_BOOTSTRAP_REPAIR_SOFT_ATTEMPTS}, ` +
      `primaryBackendAlive=${primaryBackendAlive}, ` +
      `hardReinstall=${repairDecision.hardReinstall}): ${repairDecision.reason}`
  )

  // The guard may decide the install is healthy enough that a restart
  // (without touching the venv) is the right answer. Translate that into
  // the existing flag: if the guard said "soft restart", we skip the
  // "bypass active runtime" path inside startHermes() and fall through
  // to the normal restart branch, which just kills the current child
  // and respawns it against the same venv. See #74874 — this is what
  // breaks the infinite reinstall loop the user hit.
  bootstrapRepairRequested = repairDecision.hardReinstall
  bootstrapFailure = null
  backendStartFailure = null
  remoteReauthFailure = null
  firstRunBoot.getFirstRunSetupGate().resetForRepair()
  resetHermesConnection()

  return { ok: true }
})
ipcMain.handle('hermes:bootstrap:continue-local', async () => {
  rememberLog('[bootstrap] local install selected by renderer; continuing first-launch bootstrap')
  firstRunBoot.continueFirstRunLocalBootstrap()

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
ipcMain.handle('hermes:boot-progress:get', async () => firstRunBoot.getBootProgressState())
ipcMain.handle('hermes:bootstrap:get', async () => firstRunBoot.getBootstrapState())
ipcMain.handle('hermes:connection-config:get', async (_event, profile) =>
  sanitizeDesktopConnectionConfig(readDesktopConnectionConfig(), profile)
)
ipcMain.handle('hermes:plugin-profile-routes', async (_event, rawProfileNames) => {
  const fallbackProfileNames = Array.isArray(rawProfileNames)
    ? rawProfileNames
        .filter(name => typeof name === 'string')
        .map(name => name.trim())
        .filter(Boolean)
        .slice(0, 256)
    : []

  const registry = readDesktopConnectionsRegistry()
  const enumerations = await enumerateRegistryAgentSources(registry)
  let agents = buildAgentRoster(enumerations, { primaryConnectionId: registry.primary })

  // Roster enumeration deliberately does not dial connect-on-demand SSH
  // sources. Publish one credential-free seed route so a plugin can be the
  // first caller that opens the tunnel.
  const sshSeeds = undialedSshRouteSeeds(agents, registry.connections)

  if (sshSeeds.length > 0) {
    agents = [
      ...agents,
      ...sshSeeds.map(seed => {
        const source = registry.connections.find(connection => connection.id === seed.connectionId)!

        return {
          connectionId: source.id,
          connectionKind: source.kind,
          connectionLabel: source.label,
          handle: seed.profile,
          profile: seed.profile
        }
      })
    ]
  }

  // A local enumeration can fail while remote/cloud sources succeed. Preserve
  // cached v1 profile names as explicitly-local rows so those valid routes do
  // not disappear and duplicate names remain source-qualified.
  const localSource = registry.connections.find(source => source.kind === 'local')

  const localEnumeration = localSource
    ? enumerations.find(({ connection }) => connection.id === localSource.id)
    : undefined

  const localFallbackProfiles = localSource
    ? localRouteFallbackProfiles(
        agents,
        localSource.id,
        fallbackProfileNames,
        isLocalEnumerationFailure(localEnumeration?.error)
      )
    : []

  if (localSource && localFallbackProfiles.length > 0) {
    agents = [
      ...agents,
      ...localFallbackProfiles.map(profile => ({
        connectionId: localSource.id,
        connectionKind: localSource.kind,
        connectionLabel: localSource.label,
        handle: profile,
        profile
      }))
    ]
  }

  return buildRegistryProfileRoutes({ agents, sources: registry.connections })
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

// Convenience wrappers around the bearer-aware descriptor request path.
// Native OAuth sessions are cookieless, so these must not bypass
// fetchJsonForBackend and fall straight through to the cookie partition.
async function postJsonForBackend(descriptor, path, body, opts: any = {}) {
  return fetchJsonForBackend(descriptor, path, { ...opts, body: body ?? {}, method: 'POST' })
}

// GET twin of postJsonForBackend.
async function getJsonForBackend(descriptor, path, opts: any = {}) {
  return fetchJsonForBackend(descriptor, path, opts)
}

// Any-method REST call against a resolved backend descriptor — the descriptor
// analogue of the hermes:api handler's own auth split: OAuth backends prefer a
// native bearer (cookieless RFC 8252 flow) and fall back to the OAuth cookie
// partition; token/local descriptors use the static session-token header.
async function fetchJsonForBackend(
  descriptor,
  path,
  opts: { method?: string; body?: unknown; upload?: unknown; timeoutMs?: number } = {}
) {
  const url = `${descriptor.baseUrl}${path}`

  if (descriptor.authMode === 'oauth') {
    // The OAuth cookie path rides electron.net with JSON headers; multipart
    // isn't wired there. Fail loudly rather than corrupting the upload.
    if (opts.upload) {
      throw new Error('File uploads are not supported against OAuth-gated remote backends yet.')
    }

    const options = {
      method: opts.method,
      body: opts.body,
      timeoutMs: opts.timeoutMs,
      headers: descriptor.headers
    }

    return requestWithOauthFallback(descriptor.baseUrl, {
      ensureNativeAccessToken,
      requestWithBearer: bearer => fetchJson(url, null, { ...options, bearer }),
      requestWithCookie: () => fetchJsonViaOauthSession(url, options)
    })
  }

  return fetchJson(url, descriptor.token, {
    method: opts.method,
    body: opts.body,
    upload: opts.upload,
    timeoutMs: opts.timeoutMs,
    headers: descriptor.headers
  })
}

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
ipcMain.handle('hermes:profile:default:get', async () => desktopProfilePreferences.getDefault())
ipcMain.handle('hermes:profile:default:set', async (_event, route) => desktopProfilePreferences.setDefault(route))
ipcMain.handle('hermes:profile:get', async () => ({ profile: readActiveDesktopProfile() }))
// Persistence-only sibling of hermes:profile:set: records the profile the
// Desktop last used WITHOUT tearing down the backend or reloading the window.
// An explicit default route wins at launch and is never replaced here.
ipcMain.handle('hermes:profile:remember', async (_event, name) => ({
  profile: writeActiveDesktopProfile(name)
}))
ipcMain.handle('hermes:profile:set', async (_event, name) => {
  assertCanMutateManagedPrimaryRouting()
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

// read_window_below tool: which OS window is directly underneath this one.
// Metadata only (app, title, bounds) — never pixels. On macOS, other apps'
// window titles are gated behind the Screen Recording permission; pass titles
// through only when it is ALREADY granted, and never prompt for it here.
ipcMain.handle('hermes:window:readBelow', async event => {
  const win = BrowserWindow.fromWebContents(event.sender)

  if (!win || win.isDestroyed()) {
    return null
  }

  const titlesAvailable = IS_MAC ? systemPreferences.getMediaAccessStatus?.('screen') === 'granted' : true

  const [x, y] = win.getPosition()
  const [width, height] = win.getSize()

  return readWindowBelow(process.pid, { x, y, width, height }, titlesAvailable)
})

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

ipcMain.handle('hermes:openExternal', (_event, url) => {
  if (!openExternalUrl(url)) {
    throw new Error('Invalid external URL')
  }
})

// ── Find-in-page (Ctrl/Cmd+F) ─────────────────────────────────────────────
// The desktop supports multiple BrowserWindows (one primary plus any
// per-session secondary windows spawned via `hermes:window:openSession`).
// Find must run against the requesting window, not a global — otherwise
// Cmd+F pressed in a secondary session window would search the primary
// and the match counter would report matches the user can't see. Resolve
// the sender through `BrowserWindow.fromWebContents(event.sender)` and
// forward `found-in-page` results back to that same sender.

// Lazily-installed forwarder per sender webContents. We track one
// uninstall fn per webContents id and prune entries when the sender goes
// away — Electron does not auto-detach webContents listeners on close,
// so the map is the cleanup path.
const foundInPageForwarders = new Map<number, () => void>()

function ensureFoundInPageForwarder(sender: Electron.WebContents): void {
  if (foundInPageForwarders.has(sender.id)) {
    return
  }

  const uninstall = installFoundInPageForwarder(sender)
  foundInPageForwarders.set(sender.id, uninstall)

  sender.once('destroyed', () => {
    foundInPageForwarders.get(sender.id)?.()
    foundInPageForwarders.delete(sender.id)
  })
}

ipcMain.handle('hermes:find-in-page', async (event, query, options) => {
  const win = BrowserWindow.fromWebContents(event.sender)

  if (!win || win.isDestroyed()) {
    return { count: 0 }
  }

  ensureFoundInPageForwarder(event.sender)
  await performFindAfterIndexingStarted(win.webContents, query, options)

  // The match count still arrives asynchronously via `found-in-page`; this
  // reply only acknowledges that Chromium has begun returning this request.
  return { count: 0 }
})

ipcMain.handle('hermes:stop-find-in-page', event => {
  const win = BrowserWindow.fromWebContents(event.sender)

  if (!win || win.isDestroyed()) {
    return
  }

  stopFind(win.webContents)
})

// The renderer can't know whether a loopback URL is reachable — only main
// knows which transport backs this gateway. Ask before loading one.
ipcMain.handle('hermes:preview:reach', async (event, url) => reachablePreviewUrl(event.sender.id, String(url || '')))

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

ipcMain.handle('hermes:resolveFavicon', (_event, url) => resolveFaviconCached(url))

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

// Renderer error-boundary catches (#79428 defect B): the component stack only
// exists in renderer memory, so the boundary posts it here and we persist it
// via the desktop.log pipeline. `on`, not `handle` — the sender may be mid-
// crash and must not await. Flush immediately: a crashing window can be gone
// before the debounced flush timer fires.
ipcMain.on('hermes:logs:renderer-error', (_event, report) => {
  const { label, boundary, message, componentStack } = report && typeof report === 'object' ? report : {}
  rememberLog(formatRendererBoundaryReport(label, boundary, message, componentStack))
  flushDesktopLogBufferSync()
})

// Local filesystem + plugin-root IPC (readDir/reveal/rename/trash/…) — see fs-ipc.ts.
registerFsIpc({
  hermesHome: HERMES_HOME,
  readActiveDesktopProfile,
  expandUserPath,
  resolveRequestedPathForIpc,
  directoryExists,
  resolveGitBinary
})

// Git-driven features (worktrees, review pane, repo scan) — see git-ipc.ts.
registerGitIpc({ resolveGitBinary, resolveGhBinary })

// Client-side loopback callback for MCP OAuth against remote backends — see
// mcp-oauth-callback-ipc.ts.
registerMcpOauthCallbackIpc()

// Embedded terminal PTY host (hermes:terminal:*) — see terminal-ipc.ts.
const terminalIpc = registerTerminalIpc({
  isWindows: IS_WINDOWS,
  findOnPath,
  rememberLog,
  activeSshTerminalTarget,
  ensureBackend: webContentsId => ensureTerminalBackend(webContentsId),
  getSshConnectionState: scope => sshConnections.get(scope)
})

const disposeTerminalSession = terminalIpc.disposeTerminalSession

ipcMain.handle('hermes:updates:check', async (_event, opts) =>
  checkUpdates({ force: Boolean(opts?.force) }).catch(error => ({
    supported: true,
    branch: readDesktopUpdateConfig().branch,
    error: error?.kind === GIT_UNUSABLE ? GIT_UNUSABLE : 'check-failed',
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

// Keep this callback hoisted for native chrome, which captures it earlier.
function showAboutPanelFresh() {
  return desktopShellRuntime.showAboutPanelFresh()
}

ipcMain.handle('hermes:version', async () => desktopShellRuntime.getVersionInfo())
ipcMain.handle('hermes:app:relaunch', async () => desktopShellRuntime.relaunchAfterBundleSwap())
ipcMain.handle('hermes:machine:profile', async () => desktopShellRuntime.getMachineProfile())
ipcMain.handle('hermes:uninstall:summary', async () => desktopShellRuntime.getUninstallSummary())
ipcMain.handle('hermes:uninstall:run', async (_event, payload) => {
  const mode = payload && typeof payload === 'object' ? payload.mode : payload

  return desktopShellRuntime.runDesktopUninstall(String(mode || ''))
})

// Download a VS Code Marketplace extension and return the raw color-theme JSON
// it contributes. No theme code is executed — we only read JSON from the .vsix.
ipcMain.handle('hermes:vscode-theme:fetch', async (_event, id) => fetchMarketplaceThemes(String(id || '')))

// Search the Marketplace for color-theme extensions (empty query = top installs).
ipcMain.handle('hermes:vscode-theme:search', async (_event, query) => searchMarketplaceThemes(String(query || ''), 20))

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

// Ask before a quit kills a turn in flight. True when the quit was intercepted
// and the confirmation is on screen; "Quit Anyway" re-enters before-quit with
// the latch set and falls straight through to the teardown below.
function heldQuitForActiveWork(event: Electron.Event): boolean {
  if (SKIP_QUIT_CONFIRM || quitConfirmedWithActiveWork || isQuittingForHandoff) {
    return false
  }

  if (quitPromptOpen) {
    event.preventDefault()

    return true
  }

  const prompt = quitPromptFor(mergeActiveWork(activeWorkByWebContents.values()), isQuittingForHandoff)

  // A tray quit with live work still needs the ordinary visible confirmation.
  if (prompt && minimizeToTray.status().available) {
    minimizeToTray.restore()
  }

  // A hidden aux window must never parent the quit prompt: the dialog would
  // be invisible and the held quit unanswerable (#116376 §E).
  const parent = BrowserWindow.getFocusedWindow() ?? BrowserWindow.getAllWindows().find(window => window.isVisible())

  if (!prompt || !parent || parent.isDestroyed()) {
    return false
  }

  event.preventDefault()
  quitPromptOpen = true

  void dialog
    .showMessageBox(parent, {
      buttons: ['Keep Running', 'Quit Anyway'],
      cancelId: 0,
      defaultId: 0,
      detail: prompt.detail,
      message: prompt.message,
      type: 'question'
    })
    .then(({ response }) => {
      quitPromptOpen = false

      if (response === 1) {
        quitConfirmedWithActiveWork = true
        app.quit()
      }
    })
    .catch(() => {
      // A dialog we can't show must not become a quit we can't perform.
      quitPromptOpen = false
      quitConfirmedWithActiveWork = true
      app.quit()
    })

  return true
}

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
  getWindowsSandboxFallbackSticky: () => windowsSandboxFallbackSticky,
  heldQuitForActiveWork,
  introRevealController,
  localBackendLifecycle,
  managedConnectionRecoveries,
  managedConnectionUpdates,
  managedUpdateQuitState: {
    get wait() {
      return managedUpdateQuitWait
    },
    set wait(value) {
      managedUpdateQuitWait = value
    },
    get done() {
      return managedUpdateQuitWaitDone
    },
    set done(value) {
      managedUpdateQuitWaitDone = value
    }
  },
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
