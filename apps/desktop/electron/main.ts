import { execFileSync, spawn } from 'node:child_process'
import crypto from 'node:crypto'
import fs from 'node:fs'
import http from 'node:http'
import https from 'node:https'
import os from 'node:os'
import path from 'node:path'
import tls from 'node:tls'
import { pathToFileURL } from 'node:url'

import {
  app,
  BrowserWindow,
  clipboard,
  dialog,
  net as electronNet,
  webContents as electronWebContents,
  globalShortcut,
  ipcMain,
  Menu,
  nativeTheme,
  Notification,
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
import { destroyKeepaliveAgents, downloadAgentFor, jsonAgentFor, withRetry } from './api-transport'
import { stopBackendChild as stopBackendChildImpl, stopBackendTreesForUpdate } from './backend-child'
import {
  type BackendOutputTail,
  claimDecision,
  createBackendOutputTail,
  execText,
  isPidOnlyStartMarker,
  pidOnlyStartMarker,
  probeStartMarker,
  processStartMarker
} from './backend-claim'
import { dashboardFallbackArgs, sourceDeclaresServe } from './backend-command'
import { createBackendConnectionState } from './backend-connection-state'
import { buildDesktopBackendEnv, hermesManagedNodePathEntries, normalizeHermesHomeRoot } from './backend-env'
import {
  isReauthRequiredError,
  makeNousCloudBackendDownError,
  makeUnsignedOauthError,
  waitForHermesReady
} from './backend-health'
import { backendCommandMatches, createBackendOwnership, createBackendShutdownCoordinator } from './backend-ownership'
import {
  canImportHermesCli,
  execProbeSync,
  PROBE_TIMEOUT_MS,
  shouldTrustHermesOverride,
  verifyHermesCli
} from './backend-probes'
import { waitForDashboardPortAnnouncement } from './backend-ready'
import { isPidAliveWindows, waitForBackendRelease } from './backend-release-gate'
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
import { runBootstrap } from './bootstrap-runner'
import {
  BROWSER_WINDOW_HEIGHT,
  BROWSER_WINDOW_MIN_HEIGHT,
  BROWSER_WINDOW_MIN_WIDTH,
  BROWSER_WINDOW_WIDTH,
  buildBrowserWindowUrl
} from './browser-windows'
import { detectBundleSkew } from './bundle-skew'
import { applyConnectionChange } from './connection-apply'
import {
  apiRequestRegistryConnectionId,
  authModeFromStatus,
  buildGatewayWsUrl,
  buildGatewayWsUrlWithTicket,
  connectionScopeKey,
  cookiesHaveLiveSession,
  cookiesHavePrivyAccessToken,
  cookiesHavePrivySession,
  cookiesHaveSession,
  gatewayTicketFailure,
  gatewayWsUrlIpcResult,
  hostLabelFromBaseUrl,
  localProfileEntry,
  modeIsRemoteLike,
  normalizeRemoteBaseUrl,
  normalizeRemoteHeaders,
  normalizeSshConfig,
  normAuthMode,
  pathForRegistryBackendRequest,
  pathWithGlobalRemoteProfile,
  profileHasRemoteConnection,
  profileRemoteOverride,
  profileSshOverride,
  type RegistryBackendRequestScope,
  remoteRequestMatchesBaseUrl,
  resolveAuthMode,
  resolveProfileApiRequest,
  resolveProfileBackendRoute,
  resolveRemoteSshDashboardProfile,
  resolveTestWsUrl,
  savedProfileSsh,
  tokenPreview,
  withTransientRetries
} from './connection-config'
import { applyConnectionConfigAtomically } from './connection-config-apply'
import {
  backendScopeKey,
  backendScopePrefix,
  buildAgentRoster,
  connectionDialFieldsChanged,
  mergeConnectionInput,
  migrateV1ToRegistry,
  normalizeConnectionInput,
  normalizeRegistry,
  reconcileAppliedGlobalConnection,
  reconcileRegistryDrift,
  rememberSshEnumeration,
  removeConnection,
  resolvedConnectionId,
  resolveRegistryLocalRoute,
  setConnectionLaunchMode,
  setLastUsedConnection,
  setPrimaryConnection,
  shouldDeferLocalEnumeration,
  shouldRetrySshInventory,
  updateEligibility,
  upsertConnection
} from './connection-registry'
import { describeCrashReason, installCrashForensics } from './crash-forensics'
import { adoptServedDashboardToken } from './dashboard-token'
import { loadOrCreateInstallationId, sshOwnershipId } from './desktop-installation'
import { formatDesktopLogLine } from './desktop-log-line'
import { resolveDesktopRemoteRoute } from './desktop-remote-route'
import {
  buildPosixCleanupScript,
  buildWindowsCleanupScript,
  modeRemovesAgent,
  modeRemovesUserData,
  resolveRemovableAppPath,
  shouldRemoveAppBundle,
  uninstallArgsForMode
} from './desktop-uninstall'
import { describeDevCdpDecision, resolveDevCdpPort } from './dev-cdp'
import { installEmbedReferer } from './embed-referer'
import { createEventDeduper } from './event-dedupe'
import {
  buildTerminalScript,
  resolveTerminalLaunch,
  terminalScriptEnv,
  terminalScriptExtension,
  tuiResumeArgs
} from './external-terminal'
import { type FaviconIo, resolveFavicon } from './favicon'
import { findGitBash as _findGitBash } from './find-git-bash'
import {
  installFindShortcut,
  installFoundInPageForwarder,
  performFindAfterIndexingStarted,
  stopFind
} from './find-in-page'
import { createFirstRunSetupGate } from './first-run-setup-gate'
import { registerFsIpc } from './fs-ipc'
import {
  filenameFromContentDisposition,
  gatewayFilePath,
  gatewayFileRequestPaths,
  isNotFoundError,
  parseDataUrlToBuffer,
  pumpStreamToFile,
  resolveGatewayFileBackend
} from './gateway-file-download'
import { probeGatewayWebSocket } from './gateway-ws-probe'
import { registerGitIpc } from './git-ipc'
import { clearStaleGitLocks } from './gitlock'
import { readAndConsumeHandoffResult } from './handoff-result'
import {
  ATTACHMENT_UPLOAD_DEFAULT_MAX_BYTES,
  clampDataUrlReadMaxMb,
  DATA_URL_READ_DEFAULT_MAX_MB,
  dataUrlReadMaxBytesFromMb,
  DEFAULT_FETCH_TIMEOUT_MS,
  enableBasicPasswordStoreEncryption,
  encryptDesktopSecret as encryptDesktopSecretStrict,
  readFileDataUrlForIpc,
  resolvePersistedRemoteToken,
  resolveReadableFileForIpc,
  resolveRequestedPathForIpc,
  resolveTimeoutMs,
  SAFE_STORAGE_ENCODING,
  TEXT_PREVIEW_SOURCE_MAX_BYTES,
  tightenSecretFileMode,
  writeSecretFileAtomic
} from './hardening'
import { cursorPointInWindow } from './hud-cursor'
import { startHudGameOverlayWatch } from './hud-game-overlay'
import { applyHudResetBounds, defaultHudBounds } from './hud-geometry'
import { registerHudIpc } from './hud-ipc'
import { applyHudElectronOverlay, promoteHudOverlay } from './hud-overlay'
import { snapHudBounds } from './hud-snap'
import { createHudSnapShortcut } from './hud-snap-shortcut'
import { buildHudWindowUrl } from './hud-url'
import { resolveHudWindowing } from './hud-windowing'
import { createLinkTitleWindow, guardLinkTitleSession, readLinkTitleWindowTitle } from './link-title-window'
import { ensureMainWindow } from './main-window-lifecycle'
import { createMediaProtocolHandler, MEDIA_PROTOCOL } from './media-protocol'
import {
  oauthGuardMayHardFail,
  oauthSessionIsLive,
  resolveGatedDownloadAuth,
  resolveJsonBody,
  resolveOauthRestAuth,
  resolveReadinessProbeAuth
} from './native-auth-decisions'
import {
  nativeRefreshUrl,
  type NativeTokenSet,
  parseTokenResponse,
  resolveLoginStrategy,
  tokenNeedsRefresh
} from './native-oauth'
import { runNativeLogin } from './native-oauth-login'
import { loadNativeTokenSet, type NativeTokenStoreIo, persistNativeTokenSet } from './native-token-store'
import { serializeJsonBody, setJsonRequestHeaders } from './oauth-net-request'
import { createParentStartMarkerResolver, parentWatchdogEnv } from './parent-process-identity'
import { registerPetOverlayIpc } from './pet-overlay-ipc'
import {
  buildRegistryProfileRoutes,
  localRouteFallbackProfiles,
  registryGatewayWsUrl,
  undialedSshRouteSeeds
} from './plugin-profile-routes'
import { selectPoolEvictions } from './pool-eviction'
import { createPoolStopper } from './pool-stop'
import { poolTouchKeys } from './pool-touch-scope'
import { createKeepAwake } from './power-save'
import { PreviewReachRegistry } from './preview-reach'
import {
  createPrimaryRemoteConnection,
  FirstRunSetupResetError,
  runPrimaryBackendStartup
} from './primary-backend-startup'
import { rehomePrimaryConnection } from './primary-connection-rehome'
import {
  assertLocalProfileCanStart,
  decideProfileDeleteAction,
  dispatchConnectionScopedProfileDelete,
  localProfilePoolKeys,
  ProfileDeletionGate,
  profileNameFromDeleteRequest,
  resolveRouteProfile
} from './profile-delete-routing'
import { prepareProfileRenameLifecycle, profileRenameFromRequest } from './profile-rename-routing'
import {
  buildSidebarSessionSliceParams,
  fetchPrimaryProfileSessions,
  fetchRegistrySessionRows,
  fetchRemoteProfileSessions,
  findRemoteOwnerProfileForSession,
  mergeProfileSessionWindow,
  type RegistrySessionSource,
  spliceRegistrySessionRows,
  tagRegistrySessionResponse
} from './profile-session-routing'
import { createQuickEntryShortcut, quickEntryWindowBounds, sanitizeQuickEntrySettings } from './quick-entry'
import { type ActiveWork, mergeActiveWork, normalizeActiveWork, quitPromptFor } from './quit-guard'
import * as remoteLifecycle from './remote-lifecycle'
import {
  RemoteLivenessTracker,
  RemoteRevalidationCoordinator,
  revalidatePooledRemoteBackends,
  revalidateRemoteConnection
} from './remote-liveness'
import { missingRendererAssets } from './renderer-bundle'
import { attachRendererConsoleCapture, formatRendererBoundaryReport } from './renderer-log'
import {
  classifyStoredSecret,
  readSecretStoragePolicy,
  SECRET_STORAGE_POLICY_FILE,
  type SecretStoragePolicy,
  writeSecretStoragePolicy
} from './secret-storage-policy'
import {
  buildInstanceWindowUrl,
  buildSessionWindowUrl,
  chatWindowWebPreferences,
  createSessionWindowRegistry,
  instanceWindowBounds,
  SESSION_WINDOW_MIN_HEIGHT,
  SESSION_WINDOW_MIN_WIDTH
} from './session-windows'
import { ensureLoginShellPath } from './shell-path'
import { createBootstrapCoordinator, sshConfigFingerprint } from './ssh-bootstrap-coordinator'
import { collectSshConfigHosts, parseSshGOutput } from './ssh-config'
import { createSshProbeConnection, pickLocalPort, redactSecrets, SshConnection } from './ssh-connection'
import { createStreamThrottle } from './stream-throttle'
import { registerTerminalIpc } from './terminal-ipc'
import { nativeOverlayWidth as computeNativeOverlayWidth, macTitleBarOverlayHeight } from './titlebar-overlay-width'
import {
  backgroundMaterialFor,
  defaultTranslucencyState,
  glassActive,
  glassSupportedOn,
  normalizeState as normalizeTranslucency,
  opacityNeedsSetting,
  translucencySupportedOn,
  vibrancyFor as vibrancyForTranslucency,
  windowBackingOptions,
  windowOpacityFor,
  windowOpacityOptions
} from './translucency'
import {
  compareApiUrl,
  parseCompareBehindCount,
  resolveBehindCount,
  resolveCommitLogSelection,
  shouldCountCommits
} from './update-count'
import { waitForUpdateClearance } from './update-gate'
import { readLiveUpdateMarker, updateHandoffConflict, writeUpdateMarker } from './update-marker'
import { isOfficialSshRemote, OFFICIAL_REPO_HTTPS_URL } from './update-remote'
import {
  collectRelaunchArgs,
  observeUpdaterHandoff,
  resolvePosixScriptHandoff,
  resolveStagedUpdaterBinary,
  resolveUpdateScriptHandoff,
  sandboxFallbackFromEnv,
  spawnUpdaterProcess,
  stagedUpdaterSupportsPrewrittenMarker,
  wrapHandoffForDetachedConsole
} from './updater-process'
import {
  formatBlockerMessage,
  formatProbeFailedMessage,
  scanVenvBlockers,
  stopSafeVenvBlockers
} from './venv-blocker-scan'
import { fetchMarketplaceThemes, searchMarketplaceThemes } from './vscode-marketplace'
import { createWakeIndicatorWindowController } from './wake-indicator-window'
import { enumerateWindowsFrontToBack, enumerationFailed, readWindowBelow } from './window-below'
import { registrySshScopeForWindowRoute, WindowConnectionRouteRegistry } from './window-connection-route'
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
  buildPathExtCandidates,
  chooseUpdaterArgs,
  getVenvSitePackagesEntries,
  resolveVenvHermesCommand
} from './windows-hermes-path'
import { connectWindowsRemote, detectRemotePlatform, helper } from './windows-remote-lifecycle'
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
import { readWslWindowsClipboardImage } from './wsl-clipboard-image'
import { resolvePickerDefaultPath, setActiveGatewayProfile, setWslBridgeProfileState } from './wsl-path-bridge'

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
const DESKTOP_WINDOW_STATE_PATH = path.join(app.getPath('userData'), 'window-state.json')
const DESKTOP_BACKEND_OWNERSHIP_PATH = path.join(app.getPath('userData'), 'backend-ownership.json')
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
const BOOT_FAKE_ERROR = process.env.HERMES_DESKTOP_BOOT_FAKE_ERROR || ''
// Automated teardown (Playwright's app.close(), harness scripts) quits with
// nobody to answer a modal, so the active-work confirmation would hang the
// caller instead of letting the process exit. Force quits set this.
const SKIP_QUIT_CONFIRM = process.env.HERMES_DESKTOP_SKIP_QUIT_CONFIRM === '1'

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
const APP_ICON_PATHS = [
  ...(IS_WINDOWS
    ? [path.join(process.resourcesPath ?? '', 'icon.ico'), path.join(APP_ROOT, 'assets', 'icon.ico')]
    : []),
  path.join(APP_ROOT, 'public', 'apple-touch-icon.png'),
  path.join(APP_ROOT, 'dist', 'apple-touch-icon.png'),
  path.join(unpackedPathFor(APP_ROOT), 'dist', 'apple-touch-icon.png')
]

let rendererTitleBarTheme = null

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
// default). Two modes share the lever (see electron/translucency.ts and
// store/translucency): 'clear' maps it to the native window opacity so the
// desktop shows through the whole window; 'glass' keeps the window opaque
// and lets the renderer thin its surfaces over a platform material instead
// — a matte blur with full-contrast text. macOS uses vibrancy; Windows 11
// uses DWM acrylic/mica/tabbed. Persisted so a cold launch applies it at
// window creation, before the renderer reports its value.
// macOS + Windows only; `setOpacity` is a no-op on Linux.
const TRANSLUCENCY_CONFIG_PATH = path.join(app.getPath('userData'), 'translucency.json')

function readPersistedTranslucency() {
  try {
    return normalizeTranslucency(JSON.parse(fs.readFileSync(TRANSLUCENCY_CONFIG_PATH, 'utf8')), GLASS_SUPPORTED)
  } catch {
    // Nothing persisted yet — a first launch. Glass ships on, so the FIRST
    // window has to be created with the glass backing already: a window born
    // opaque cannot reliably be swapped to glass afterwards (see
    // windowBackingOptions). nativeTheme is the only appearance signal main
    // has this early; the renderer's first resolved send corrects it.
    return defaultTranslucencyState(nativeTheme.shouldUseDarkColors ? 'dark' : 'light', GLASS_SUPPORTED, IS_WINDOWS)
  }
}

function writePersistedTranslucency(state) {
  try {
    fs.mkdirSync(path.dirname(TRANSLUCENCY_CONFIG_PATH), { recursive: true })
    fs.writeFileSync(TRANSLUCENCY_CONFIG_PATH, JSON.stringify(state, null, 2), 'utf8')
  } catch (error) {
    rememberLog(`[translucency] write failed: ${error.message}`)
  }
}

let translucencyState = readPersistedTranslucency()

// Chat windows whose webContents backing follows translucency (primary,
// instance peers, session windows). The HUD / pet overlay / quick entry /
// wake indicator are `transparent: true` windows that own their backgrounds —
// painting a themed backing onto them would turn them into opaque rectangles.
const translucencyBackedWindows = new WeakSet()

// Set a live window's native opacity, but only when the state asks it to fade
// — or when the window is already faded and is on its way back to opaque. The
// window's own opacity is the record of whether that door was ever opened; see
// opacityNeedsSetting for why it matters that it stays shut.
function applyWindowOpacity(win) {
  const opacity = windowOpacityFor(translucencyState)

  if (typeof win.setOpacity === 'function' && opacityNeedsSetting(opacity, win.getOpacity?.() ?? 1)) {
    win.setOpacity(opacity)
  }
}

// Re-apply translucency to a live window (runtime toggle, no recreation).
// Opacity goes through applyWindowOpacity, which knows when the call is worth
// making at all. The backing swap is the glass half: Chromium composites the
// page against the window backing BEFORE the OS composites the window, so
// glass needs the backing dropped for the platform material to reach it, and
// every other state needs the opaque themed backing (anti-flash, and it is
// what makes clear mode fade to the desktop instead of to black).
//
// `changed` says which native properties actually need touching. Dragging the
// intensity slider emits ~100 updates, and in glass mode NONE of them change
// anything native — the tint is painted by the renderer and windowOpacityFor
// answers off `fade`, not `intensity`, there. Re-issuing setVibrancy on every
// tick restarts its 150ms animation before macOS can settle the material,
// which reads as jank and flattens the frost levels into each other. Windows
// setBackgroundMaterial is instantaneous but still skipped on tint-only ticks.
// The glass Fade lever is the one glass drag that does reach main, and it
// costs exactly what a Clear drag costs: one setOpacity.
//
// CAUTION (measured, macOS 26 / Electron 40): a runtime
// setBackgroundColor('#00000000') is silently LOST on a window whose
// compositor hasn't been up for a few seconds — including calls from
// 'ready-to-show' and 'did-finish-load'. Cold launches therefore must not
// rely on this path: windows are BORN with the right backing
// (windowBackingOptions at each creation site). This path only has to cover
// live toggles from Settings, where the window is long settled.
function applyWindowTranslucency(win, changed = { backing: true, material: true, opacity: true }) {
  if (!win || win.isDestroyed()) {
    return
  }

  try {
    // Backing swap + material are scoped to registered chat windows (see
    // translucencyBackedWindows above).
    if (translucencyBackedWindows.has(win)) {
      if (changed.backing && typeof win.setBackgroundColor === 'function') {
        win.setBackgroundColor(glassActive(translucencyState) ? '#00000000' : getWindowBackgroundColor())
      }

      if (changed.material) {
        // Glass frost level = the platform material. Animate the macOS hop so
        // a deliberate frost switch feels continuous — which only works if we
        // don't re-issue it on unrelated updates. Windows has no equivalent
        // animation option; setBackgroundMaterial is instantaneous.
        if (IS_MAC && typeof win.setVibrancy === 'function') {
          win.setVibrancy(vibrancyForTranslucency(translucencyState), { animationDuration: 150 })
        }

        if (IS_WINDOWS && GLASS_SUPPORTED && typeof win.setBackgroundMaterial === 'function') {
          win.setBackgroundMaterial(backgroundMaterialFor(translucencyState))
        }
      }
    }

    if (changed.opacity) {
      applyWindowOpacity(win)
    }
  } catch (error) {
    rememberLog(`[translucency] apply failed: ${error.message}`)
  }
}

// Constructor options every chat window shares for its translucency surface:
// the platform material, the webContents backing, and a native opacity only if
// the state actually fades — all under the CURRENT state. Glass omits
// backgroundColor so the material shows from the first frame (Electron hands a
// translucent window a transparent default backing, and runtime swaps are lost
// early in a window's life — see applyWindowTranslucency); otherwise the opaque
// themed anti-flash backing.
//
// Call sites also register the window in translucencyBackedWindows so a live
// toggle can re-apply. The HUD, pet overlay, quick entry and wake indicator
// are `transparent: true` windows that own their backgrounds and are
// deliberately not chat windows.
function chatWindowSurfaceOptions() {
  return {
    vibrancy: IS_MAC ? vibrancyForTranslucency(translucencyState) : undefined,
    // Pin the material to its ACTIVE appearance: several NSVisualEffectView
    // materials collapse to a shared inactive look when the window blurs
    // (measured on macOS 26: sidebar, popover and under-window composited
    // pixel-identically once unfocused), which would quietly erase the
    // user's frost choice whenever they click elsewhere. Only observable
    // under glass — everywhere else the page buries the material.
    visualEffectState: IS_MAC ? ('active' as const) : undefined,
    // NOT `transparent: true` on Windows. The backdrop material already makes
    // the window translucent on its own: `IsTranslucent` answers yes off
    // `background_material_` alone, which is what gives the page its transparent
    // default backing, and `SetBackgroundMaterial` flips widget translucency
    // live, so a Clear→Glass toggle needs no recreate either way. Its one gate
    // is a frameless window, and `titleBarStyle: 'hidden'` already makes
    // `has_frame()` false here.
    //
    // What `transparent` adds on top is permanent and unwanted: it pins the
    // widget to kTranslucent for the window's whole life, so even glass-OFF
    // windows pay a DirectComposition redraw per frame (electron#39895), and it
    // opts into the documented transparent-window limits — including that a
    // RESIZABLE transparent window is unsupported and breaks (electron#48421).
    // Every chat window is resizable.
    backgroundMaterial: IS_WINDOWS && GLASS_SUPPORTED ? backgroundMaterialFor(translucencyState) : undefined,
    ...windowOpacityOptions(translucencyState),
    ...windowBackingOptions(translucencyState, getWindowBackgroundColor())
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
  '.pdf': 'application/pdf',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.wav': 'audio/wav',
  '.webm': 'video/webm',
  '.webp': 'image/webp'
}

const PREVIEW_HTML_EXTENSIONS = new Set(['.html', '.htm'])
const PREVIEW_PDF_EXTENSIONS = new Set(['.pdf'])
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
    ensureRemoteBearer: baseUrl => ensureNativeAccessToken(baseUrl).catch(() => null),
    fetchLocal: (resolvedPath, headers, method) =>
      electronNet.fetch(pathToFileURL(resolvedPath).toString(), {
        bypassCustomProtocolHandlers: true,
        credentials: 'omit',
        headers,
        method
      }),
    fetchRemote: (url, headers, method) =>
      electronNet.fetch(url, {
        bypassCustomProtocolHandlers: true,
        credentials: 'omit',
        headers,
        method
      }),
    fetchRemoteWithCookies: (url, headers, method) => {
      const oauthSession = getOauthSession()

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
    resolveRemoteConnection: ({ connectionId, profile }) =>
      connectionId ? ensureRegistryBackend(connectionId, profile) : ensureBackend(profile)
  })

  protocol.handle(MEDIA_PROTOCOL, handler)
}

let mainWindow = null
const backendConnectionState = createBackendConnectionState<ReturnType<typeof spawn>, any>()
const remoteLiveness = new RemoteLivenessTracker()
const remoteRevalidation = new RemoteRevalidationCoordinator()
// True while connection-config:apply soft-rehomes the primary — suppresses the
// backend-exit toast so an intentional kill doesn't look like a crash.
let softRehomeInProgress = false
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
const POOL_MAX_BACKENDS = Math.max(1, Number(process.env.HERMES_DESKTOP_POOL_MAX) || 3)
const POOL_IDLE_MS = Math.max(60_000, Number(process.env.HERMES_DESKTOP_POOL_IDLE_MS) || 10 * 60_000)
// A backend touched within this window has a live renderer socket (the keepalive
// pings every 60s for every open profile). LRU eviction must spare these — a
// concurrent multi-profile session keeps several backends "fresh" at once, and
// killing one to honor the soft cap would abort a running agent.
const POOL_KEEPALIVE_FRESH_MS = 90_000
let poolIdleReaper = null
let backendOrphanReapPromise = null
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
let connectionConfigCache = null
let connectionConfigCacheMtime = null
let connectionRegistryCache = null
let connectionRegistryCacheMtime = null
let remoteHeaderRulesInstalled = false
const remoteWsHeadersByUrl = new Map<string, Record<string, string>>()
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
  isCloudBackendDown: false,
  message: 'Waiting to start Hermes backend',
  phase: 'idle',
  progress: 0,
  retryable: false,
  running: false,
  statusCode: null,
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

  // One timestamp per chunk: lines arriving in the same event happened
  // at the same moment.  ISO-8601 UTC, matching agent.log/gateway.log.
  const stamp = new Date().toISOString()
  const lines = text.split(/\r?\n/).map(line => formatDesktopLogLine(line, stamp))
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

installCrashForensics({ flush: flushDesktopLogBufferSync, log: rememberLog })

// A rejected loadURL leaves a blank window and, unhandled, no trace anywhere
// the user can send us. `label` names the surface so the log says which one.
function loadWindowUrl(win, url, label) {
  win.loadURL(url).catch(error => rememberLog(`${label} failed to load: ${describeCrashReason(error)}`))
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
  setupChoice: null,
  unsupportedPlatform: null
}

let firstRunSetupGate = null

function broadcastBootstrapEvent(ev) {
  if (ev.type === 'manifest') {
    bootstrapState.manifest = ev
    bootstrapState.active = true
    bootstrapState.setupChoice = null
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
    bootstrapState.setupChoice = null
  } else if (ev.type === 'unsupported-platform') {
    bootstrapState.active = false
    bootstrapState.setupChoice = null
    bootstrapState.unsupportedPlatform = {
      platform: ev.platform,
      activeRoot: ev.activeRoot,
      installCommand: ev.installCommand,
      docsUrl: ev.docsUrl
    }
  } else if (ev.type === 'setup-choice') {
    bootstrapState.active = false
    bootstrapState.error = null
    bootstrapState.manifest = null
    bootstrapState.stages = {}
    bootstrapState.setupChoice = ev.active
      ? {
          platform: ev.platform,
          activeRoot: ev.activeRoot
        }
      : null
    bootstrapState.unsupportedPlatform = null
  } else if (ev.type === 'dismissed') {
    resetBootstrapSnapshot()
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

function resetBootstrapSnapshot() {
  bootstrapState = {
    active: false,
    manifest: null,
    stages: {},
    error: null,
    log: [],
    startedAt: null,
    completedAt: null,
    setupChoice: null,
    unsupportedPlatform: null
  }
}

function promptFirstRunSetupChoice(backend) {
  broadcastBootstrapEvent({
    type: 'setup-choice',
    active: true,
    platform: backend.platform || process.platform,
    activeRoot: backend.activeRoot || ACTIVE_HERMES_ROOT
  })
}

function hideFirstRunSetupChoice() {
  if (bootstrapState.setupChoice) {
    broadcastBootstrapEvent({ type: 'setup-choice', active: false })
  }
}

function getFirstRunSetupGate() {
  if (!firstRunSetupGate) {
    firstRunSetupGate = createFirstRunSetupGate({
      hideChoice: hideFirstRunSetupChoice,
      log: rememberLog,
      onStuck: (_backend, stuckAfterMs) => {
        updateBootProgress(
          {
            error: null,
            message: `Still waiting for first-run setup choice after ${Math.round(stuckAfterMs / 1000)} seconds`,
            phase: 'bootstrap.choice',
            progress: 12,
            running: true
          },
          { allowDecrease: true }
        )
      },
      promptChoice: promptFirstRunSetupChoice
    })
  }

  return firstRunSetupGate
}

async function waitForFirstRunSetupChoice(backend) {
  const gate = getFirstRunSetupGate()

  if (!gate.shouldGate(backend)) {
    return 'continue-local'
  }

  updateBootProgress(
    {
      error: null,
      message: 'Waiting for first-run setup choice',
      phase: 'bootstrap.choice',
      progress: 12,
      running: true
    },
    { allowDecrease: true }
  )

  return gate.wait(backend)
}

function continueFirstRunLocalBootstrap() {
  getFirstRunSetupGate().continueLocal()
}

function abandonFirstRunSetupChoiceForRemoteApply() {
  const gate = getFirstRunSetupGate()

  if (!gate.hasWaiter()) {
    return false
  }

  const resumedGatedConnection = gate.abandonForRemoteApply()

  if (resumedGatedConnection) {
    broadcastBootstrapEvent({ type: 'dismissed' })
  }

  return resumedGatedConnection
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
    // `retryable` rides with `error`: it survives updates that preserve the
    // error and resets alongside a new/cleared error unless explicitly set.
    retryable:
      update.retryable === undefined
        ? update.error === undefined && Boolean(bootProgressState.retryable)
        : Boolean(update.retryable),
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

// Gate deps shared by the primary-window boot path and the pool-backend
// spawn path. Consulting BOTH the on-disk marker and the in-process
// updateInFlight flag is load-bearing (#73822): applyUpdates kills its own
// backend BEFORE the Windows venv-blocker scan but only writes the marker
// AFTER it, so a marker-only gate lets the renderer's ~1s reconnect respawn
// a backend inside the update's own critical section — which the scan then
// reports as a blocker, aborting every update attempt.
function updateGateDeps() {
  return {
    hasLiveMarker: () => Boolean(readLiveUpdateMarker(HERMES_HOME)),
    isUpdateInFlight: () => updateInFlight
  }
}

// Block until no live update is in progress (or we hit the wait timeout).
// Emits a boot-progress phase so the renderer shows "Update in progress…"
// rather than a frozen splash. Returns true if it parked at all.
async function waitForUpdateToFinish() {
  let announced = false

  const outcome = await waitForUpdateClearance(updateGateDeps(), {
    onWaitTick: async reason => {
      if (!announced) {
        announced = true
        rememberLog(`[updates] update in progress (${reason}); deferring backend start until it finishes`)
      }

      await advanceBootProgress(
        'backend.update-wait',
        'An update is finishing — Hermes will start automatically when it completes…',
        12
      )
    },
    pollMs: UPDATE_WAIT_POLL_MS,
    timeoutMs: UPDATE_WAIT_TIMEOUT_MS
  })

  // The detached hand-off script (scripts/desktop-update/windows.ps1) runs hidden;
  // its result file is the ONLY way the user learns a detached update
  // failed. Consume it exactly once, here, right where boot passes the
  // update gate — success gets a log line, failure gets a real dialog
  // (previously a failed detached update was indistinguishable from
  // "nothing happened").
  try {
    const result = readAndConsumeHandoffResult(HERMES_HOME)

    if (result && result.ok && result.manual) {
      // Update landed but the user must act (reopen/reinstall/sandbox). On
      // machines with no shim browser and no notifier this dialog is the
      // FIRST time the message is visible — it must not be a log line.
      rememberLog(`[updates] detached update finished with manual action (branch ${result.branch}): ${result.message}`)
      dialog.showMessageBox({
        type: 'warning',
        title: 'Hermes update',
        message: 'The update finished, but needs one more step',
        detail: result.message
      })
    } else if (result && result.ok) {
      rememberLog(`[updates] detached update finished OK (branch ${result.branch})`)
    } else if (result) {
      rememberLog(`[updates] detached update FAILED (exit ${result.exitCode}): ${result.message}`)
      dialog.showErrorBox(
        'Hermes update did not finish',
        `${result.message}\n\nDetails: ${path.join(HERMES_HOME, 'logs', 'desktop-update-handoff.log')}`
      )
    }
  } catch (err) {
    rememberLog(`[updates] could not read hand-off result: ${err.message}`)
  }

  if (outcome === 'clear') {
    return false
  }

  if (outcome === 'timeout') {
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

function unwrapWindowsVenvHermesCommand(command, backendArgs) {
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
      // Same cold-Windows Python-startup class as the runtime probes
      // (#61764/#72632/#72707): `serve --help` imports at least as much as
      // `hermes --version` (~10.5s measured cold), and a false negative here
      // is cached for the process lifetime, silently routing a modern
      // runtime through the legacy `dashboard` form. Share the probe budget
      // and its timeout-only retry instead of a thinner local bound.
      execProbeSync(backend.command, [...prefix, 'serve', '--help'], {
        cwd: backend.root || undefined,
        env: { ...process.env, HERMES_HOME, ...(backend.env || {}) },
        timeout: PROBE_TIMEOUT_MS,
        stdio: 'ignore',
        // `.cmd`/`.bat` shim backends carry shell: true in their descriptor
        // (see resolveHermesBackend step 4); execFileSync of a .cmd without
        // shell throws EINVAL on modern Node, which the catch below would
        // mis-cache as "serve unsupported" for the process lifetime.
        shell: Boolean(backend.shell),
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
          // Registry reads are near-instant; the bound only exists so a
          // pathologically wedged reg.exe can't hang the synchronous boot
          // resolver forever (this ran unbounded before).
          hiddenWindowsChildOptions({ encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'], timeout: 5_000 })
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
            stdio: ['ignore', 'pipe', 'ignore'],
            // Bare interpreter startup — much lighter than the hermes-import
            // probes, but still python.exe under cold cache / AV scan, so
            // share the probe budget rather than running unbounded (this
            // synchronous exec previously had no timeout at all).
            timeout: PROBE_TIMEOUT_MS
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

    // Passive SSH-official checks only know tip SHAs (ls-remote) — never
    // fabricate a "1 commit behind". Recover the exact count via the GitHub
    // compare API when possible; otherwise behind stays null ("update
    // available, count unknown") and updateAvailable carries the signal.
    // ahead_by === 0 with differing tips means the remote tip is reachable
    // from our HEAD — a local carried commit sitting AHEAD, not behind:
    // flagging that as an update nudges the user into wiping their work.
    const tipsEqual = Boolean(currentSha && currentSha === targetSha)

    const sshBehind = tipsEqual
      ? 0
      : await fetchCompareBehindCount({ currentSha, originUrl: OFFICIAL_REPO_HTTPS_URL, targetSha })

    const upToDate = tipsEqual || sshBehind === 0

    return {
      supported: true,
      branch,
      currentBranch,
      behind: upToDate ? 0 : sshBehind,
      updateAvailable: !upToDate,
      currentSha,
      targetSha,
      commits: [],
      dirty: dirtyStr.length > 0,
      hermesRoot: updateRoot,
      fetchedAt: Date.now()
    }
  }

  // Self-heal abandoned git lock files before fetching. A stale
  // .git/shallow.lock from a crashed/interrupted fetch otherwise fails every
  // later fetch ("Unable to create '.git/shallow.lock': File exists") and this
  // check reports 'fetch-failed' forever — git never removes these itself.
  await clearStaleGitLocks(updateRoot)

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

  const [currentSha, targetSha, dirtyStr, currentBranch, shallowStr] = await Promise.all([
    git(['rev-parse', 'HEAD']),
    git(['rev-parse', `origin/${branch}`]),
    git(['status', '--porcelain']),
    git(['rev-parse', '--abbrev-ref', 'HEAD']),
    git(['rev-parse', '--is-shallow-repository'])
  ])

  const isShallow = shallowStr === 'true'

  // A shallow graph cannot provide a trustworthy exact count, even when it has
  // a visible merge-base. Skip the ancestry walk and use the SHA fallback.
  const countStr = shouldCountCommits({ isShallow }) ? await git(['rev-list', `HEAD..origin/${branch}`, '--count']) : ''

  // A positive directional ancestry result remains trustworthy in a shallow
  // graph and prevents a local commit on top of origin from looking outdated.
  const targetIsAncestorOfHead =
    isShallow &&
    currentSha !== targetSha &&
    (await runGit(['merge-base', '--is-ancestor', `origin/${branch}`, 'HEAD'], { cwd: updateRoot })).code === 0

  let behind = resolveBehindCount({
    countStr,
    currentSha,
    targetSha,
    isShallow,
    targetIsAncestorOfHead
  })

  // Recover the exact count a shallow clone can't compute: the GitHub compare
  // API knows the full graph regardless of local clone depth. Best-effort —
  // offline, rate-limited, or non-GitHub origins keep the honest null
  // ("update available", no fabricated number).
  if (behind === null) {
    behind = await fetchCompareBehindCount({ currentSha, originUrl, targetSha })
  }

  // behind === null means "update available, exact count unknown" (shallow
  // clone): still list what origin offers — resolveCommitLogSelection keeps
  // the shallow log to the fetched tip so the range walk can't enumerate the
  // contaminated ancestry — so "See what's new" stays useful and honest.
  const commits = behind !== 0 ? await readCommitLog(updateRoot, branch, isShallow) : []

  return {
    supported: true,
    branch,
    currentBranch,
    behind,
    updateAvailable: behind === null || behind > 0,
    currentSha,
    targetSha,
    commits,
    dirty: dirtyStr.length > 0,
    hermesRoot: updateRoot,
    fetchedAt: Date.now()
  }
}

// Best-effort exact behind-count for graphs the local clone can't measure.
// Delegates URL building + response parsing to update-count.ts (pure, unit
// tested); this wrapper only does the bounded network call. Any failure —
// offline, 4xx/5xx, rate limit, shape surprise — returns null so callers keep
// the honest "update available, count unknown" state.
async function fetchCompareBehindCount({ currentSha, originUrl, targetSha }) {
  const url = compareApiUrl({ currentSha, originUrl, targetSha })

  if (!url) {
    return null
  }

  try {
    const payload = await new Promise((resolve, reject) => {
      const req = https.get(
        url,
        {
          headers: {
            Accept: 'application/vnd.github+json',
            // GitHub requires a UA on api.github.com; requests without one 403.
            'User-Agent': 'hermes-desktop-update-check'
          },
          timeout: 10_000
        },
        res => {
          const chunks = []
          res.on('error', reject)
          res.on('data', chunk => chunks.push(chunk))
          res.on('end', () => {
            if ((res.statusCode || 500) >= 400) {
              reject(new Error(`compare API ${res.statusCode}`))

              return
            }

            try {
              resolve(JSON.parse(Buffer.concat(chunks).toString('utf8')))
            } catch (error) {
              reject(error)
            }
          })
        }
      )

      req.on('timeout', () => req.destroy(new Error('compare API timeout')))
      req.on('error', reject)
    })

    return parseCompareBehindCount(payload)
  } catch {
    return null
  }
}

async function readCommitLog(cwd, branch, isShallow) {
  const SEP = '\x1f'
  const REC = '\x1e'
  const { limit, revision } = resolveCommitLogSelection({ branch, isShallow })

  const { stdout } = await runGit(
    ['log', revision, `--pretty=format:%H${SEP}%s${SEP}%an${SEP}%at${REC}`, '-n', String(limit)],
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

// Quit-guard latches: one while the confirmation is on screen (a second
// Cmd-Q must not stack dialogs), one after the user has said "quit anyway"
// (the app.quit() that follows re-enters before-quit and must pass through).
let quitPromptOpen = false
let quitConfirmedWithActiveWork = false

// Resolve the staged updater binary the desktop may hand an update to. On
// Windows that binary owns ALL repo mutation — running `hermes update` +
// rebuilding the desktop — so the desktop never touches its own bits while
// running. macOS/Linux stage the same binary but deliberately do not use it;
// see resolveStagedUpdaterBinary for the policy and for #74836. Returns null
// whenever no hand-off applies; callers degrade gracefully.
function resolveUpdaterBinary() {
  return resolveStagedUpdaterBinary(HERMES_HOME, { fileExists, isWindows: IS_WINDOWS })
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

function writeBackendOwnership(contents) {
  fs.mkdirSync(path.dirname(DESKTOP_BACKEND_OWNERSHIP_PATH), { recursive: true })
  const tempPath = `${DESKTOP_BACKEND_OWNERSHIP_PATH}.${process.pid}.tmp`

  try {
    fs.writeFileSync(tempPath, contents, { encoding: 'utf8', mode: 0o600 })
    fs.renameSync(tempPath, DESKTOP_BACKEND_OWNERSHIP_PATH)
  } finally {
    try {
      fs.rmSync(tempPath, { force: true })
    } catch {
      void 0
    }
  }
}

// execText and processStartMarker moved to backend-claim.ts (#93608) so the
// claim/probe policy is testable — including on Windows CI with real
// PowerShell — without booting Electron. main.ts calls through the module.

async function backendCommandForPid(pid) {
  try {
    const command = IS_WINDOWS ? 'powershell.exe' : 'ps'

    const args = IS_WINDOWS
      ? [
          '-NoProfile',
          '-NonInteractive',
          '-Command',
          `(Get-CimInstance Win32_Process -Filter 'ProcessId = ${pid}').CommandLine`
        ]
      : ['-p', String(pid), '-o', 'command=']

    return (await execText(command, args)) || null
  } catch {
    return null
  }
}

async function processIdentityMatches(identity) {
  // Degraded PID-only identity (#93608): the start-marker probe failed while
  // the child was verifiably alive, so only PID liveness can be checked here.
  // backendIdentityMatches layers the command-line check on top before
  // anything destructive relies on the answer.
  if (isPidOnlyStartMarker(identity.startMarker)) {
    try {
      process.kill(identity.pid, 0)

      return true
    } catch (error) {
      const code = (error as NodeJS.ErrnoException | null)?.code

      return code === 'ESRCH' || code === 'ENOENT' ? false : code === 'EPERM' ? true : undefined
    }
  }

  try {
    return (await processStartMarker(identity.pid)) === identity.startMarker
  } catch (error) {
    return error?.code === 'ENOENT' || error?.code === 'ESRCH' ? false : undefined
  }
}

async function backendIdentityMatches(identity) {
  const processMatches = await processIdentityMatches(identity)

  if (processMatches !== true) {
    return processMatches
  }

  const command = await backendCommandForPid(identity.pid)

  return command === null ? undefined : backendCommandMatches(command)
}

// True when the recorded parent Electron is still running (same PID AND start
// marker); false when it is gone or its PID was reused; undefined when the
// ownership record predates parent tracking. Undefined deliberately falls back
// to the pre-parent reap behaviour so legacy orphan cleanup keeps working.
async function backendParentMatches(entry) {
  if (!Number.isInteger(entry.parentPid) || typeof entry.parentStartMarker !== 'string' || !entry.parentStartMarker) {
    return undefined
  }

  try {
    return (await processStartMarker(entry.parentPid)) === entry.parentStartMarker
  } catch (error) {
    return error?.code === 'ENOENT' || error?.code === 'ESRCH' ? false : undefined
  }
}

async function stopOwnedBackend(identity) {
  if ((await processIdentityMatches(identity)) !== true) {
    return
  }

  if (IS_WINDOWS) {
    forceKillProcessTree(identity.pid)
  } else {
    try {
      process.kill(-identity.pid, 'SIGTERM')
    } catch {
      try {
        process.kill(identity.pid, 'SIGTERM')
      } catch {
        return
      }
    }

    const deadline = Date.now() + 1500

    while (Date.now() < deadline) {
      if ((await processIdentityMatches(identity)) !== true) {
        return
      }

      await new Promise(resolve => setTimeout(resolve, 50))
    }

    // Revalidate immediately before escalation so PID reuse cannot target a
    // replacement process.
    if ((await processIdentityMatches(identity)) === true) {
      try {
        process.kill(-identity.pid, 'SIGKILL')
      } catch {
        process.kill(identity.pid, 'SIGKILL')
      }
    }
  }

  await new Promise(resolve => setTimeout(resolve, 50))
  const remaining = await processIdentityMatches(identity)

  if (remaining !== false) {
    throw new Error(`Backend PID ${identity.pid} did not stop cleanly.`)
  }
}

const backendOwnership = createBackendOwnership({
  matchesIdentity: backendIdentityMatches,
  matchesParent: backendParentMatches,
  stop: stopOwnedBackend,
  store: {
    read: () => {
      try {
        return fs.readFileSync(DESKTOP_BACKEND_OWNERSHIP_PATH, 'utf8')
      } catch {
        return null
      }
    },
    write: writeBackendOwnership,
    // A corrupt ownership file is moved aside instead of being rewritten
    // away by the reap sweep — its records are the only pointer to any
    // still-running backends it described (#89298).
    quarantine: () => {
      const parked = `${DESKTOP_BACKEND_OWNERSHIP_PATH}.corrupt`

      try {
        fs.renameSync(DESKTOP_BACKEND_OWNERSHIP_PATH, parked)
        rememberLog(`Backend ownership file was unreadable; moved to ${parked}`)
      } catch {
        // Nothing to move (or no permission) — the sweep already skipped.
      }
    }
  }
})

const desktopParentStartMarker = createParentStartMarkerResolver({
  load: () => processStartMarker(process.pid),
  onError: error => {
    const detail = error instanceof Error ? error.message : String(error)

    rememberLog(
      `Could not resolve the Desktop process start marker; starting the backend with PID-only parent tracking: ${detail}`
    )
  }
})

async function claimBackendChild(child, command, profile, nonce, outputTail: BackendOutputTail | null = null) {
  // Probe/claim policy lives in backend-claim.ts (#93608): a marker probe
  // that fails against a LIVE child degrades to PID-only identity — matching
  // createParentStartMarkerResolver — instead of killing a healthy backend
  // over a flaky Get-Process (PS 5.1 cold starts, #87169). Only a child that
  // actually died keeps the fail-closed throw, now carrying its stderr tail.
  const probe = await probeStartMarker(child.pid)
  const decision = claimDecision(child.exitCode === null && !child.killed, probe)

  if (decision.action === 'fail') {
    stopBackendChild(child)
    await waitForBackendExit(child)
    throw new Error(
      `Hermes backend (PID ${child.pid}) died before its identity could be recorded: ${decision.reason}${outputTail?.describe() ?? ''}`
    )
  }

  let startMarker

  if (decision.action === 'degrade') {
    startMarker = pidOnlyStartMarker(child.pid)
    rememberLog(
      `WARNING: process start marker probe failed for live Hermes backend PID ${child.pid}; ` +
        `claiming with PID-only identity instead of stopping it: ${decision.reason}`
    )
  } else {
    startMarker = decision.startMarker
  }

  try {
    const identity = await backendOwnership.claim({
      command,
      nonce,
      pid: child.pid,
      profile,
      startMarker,
      // Record the spawning Electron so reapOrphans can tell an orphaned
      // backend (parent gone) from one owned by a live instance — a live
      // parent's backend is never reaped (#87295).
      parentPid: process.pid,
      parentStartMarker: await desktopParentStartMarker()
    })

    child.hermesBackendIdentity = identity

    return identity
  } catch (error) {
    stopBackendChild(child)
    await waitForBackendExit(child)
    throw new Error(
      `Could not persist ownership for the Hermes backend: ${error.message}${outputTail?.describe() ?? ''}`
    )
  }
}

function releaseBackendChild(child) {
  const identity = child?.hermesBackendIdentity

  if (!identity) {
    return
  }

  try {
    backendOwnership.release(identity)
  } catch (error) {
    rememberLog(`Could not release backend ownership for PID ${identity.pid}: ${error.message}`)
  }
}

function reapOrphanedBackendsOnce() {
  if (!backendOrphanReapPromise) {
    backendOrphanReapPromise = backendOwnership
      .reapOrphans()
      .then(pids => {
        if (pids.length) {
          rememberLog(`Reaped orphaned desktop backend PID(s): ${pids.join(', ')}`)
        }
      })
      .catch(error => {
        backendOrphanReapPromise = null
        throw error
      })
  }

  return backendOrphanReapPromise
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

  const hermesProcess = backendConnectionState.getProcess()

  // Seed the release gate with every PID we are about to signal: the
  // supervised primary backend and all pool backends. The gate waits for
  // these to actually LEAVE the process table, not just for the shim to
  // unlock — the shim probe only covers venv\Scripts\hermes.exe, but the
  // backend is `python.exe -m hermes_cli.main serve`, which need not hold
  // the shim at all (#74805 first-attempt race).
  const initialPids = []

  if (hermesProcess && Number.isInteger(hermesProcess.pid)) {
    initialPids.push(hermesProcess.pid)
  }

  for (const entry of backendPool.values()) {
    if (entry.process && Number.isInteger(entry.process.pid)) {
      initialPids.push(entry.process.pid)
    }
  }

  stopBackendTreesForUpdate(hermesProcess, {
    forceKillProcessTree,
    stopAllPoolBackends
  })

  const shim = venvHermesShimPath(updateRoot)

  const gate = await waitForBackendRelease(
    initialPids,
    {
      isShimLocked: () => Boolean(isShimLocked(shim)),
      isPidAlive: isPidAliveWindows,
      collectStragglerPids: () => {
        const stragglers = []

        const currentHermesProcess = backendConnectionState.getProcess()

        if (currentHermesProcess && Number.isInteger(currentHermesProcess.pid)) {
          stragglers.push(currentHermesProcess.pid)
        }

        for (const entry of backendPool.values()) {
          if (entry.process && Number.isInteger(entry.process.pid)) {
            stragglers.push(entry.process.pid)
          }
        }

        return stragglers
      },
      killProcessTree: forceKillProcessTree,
      sleep: (ms: number) => new Promise(r => setTimeout(r, ms)),
      now: () => Date.now(),
      log: rememberLog
    },
    tag
  )

  if (gate.unlocked) {
    return { unlocked: true }
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
async function applyUpdates(opts: { stopSafeBlockers?: boolean } = {}) {
  if (updateInFlight) {
    throw new Error('An update is already in progress.')
  }

  updateInFlight = true

  try {
    const updater = resolveUpdaterBinary()

    if (!updater && !IS_WINDOWS) {
      // macOS/Linux: hand off to the repo-owned posix script — same shape as
      // Windows (quit → detached orchestrator → `hermes update` → relaunch),
      // minus the venv-lock gauntlet POSIX doesn't need. The old in-app
      // updater (applyUpdatesPosixInApp) is gone with everything it dragged
      // in: the HERMES_DESKTOP_CHILD_PID reaper-exclusion dance (#37532),
      // the in-window rebuild retry, and the relaunch-outcome matrix — the
      // script owns swap/relaunch, and the app is DEAD during the update so
      // there is nothing to reap around. Checkouts that predate the script
      // get the manual `hermes update` card once; their next update pulls it.
      return await applyUpdatesPosixHandoff(opts)
    }

    if (!updater) {
      // No staged updater binary — this is a CLI-installed user (they ran
      // `hermes desktop`, never the Tauri installer that self-copies
      // hermes-setup.exe into HERMES_HOME). On Windows the repo hand-off
      // script serves them just as well as installer users — it only needs
      // PowerShell and the checkout — so fall through to the normal hand-off
      // when the script exists. Only when the checkout predates the script do
      // we surface the manual one-liner.
      const updateRoot = resolveUpdateRoot()

      if (!resolveUpdateScriptHandoff(updateRoot)) {
        // They DO have a working `hermes` on PATH / in the venv, so the
        // correct path is the one-liner in their native medium. We show the
        // EXACT command, branch-pinned to the checkout they're on — bare
        // `hermes update` defaults to main and would silently switch a
        // bb/gui (or any non-main) install off-branch. Mirror the GUI
        // button's contract: append --branch <current> for non-main
        // checkouts, keep it bare for main so the card stays clean.
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

      rememberLog('[updates] no staged updater; using repo hand-off script for CLI install')
    }

    const handoffConflict = updateHandoffConflict(HERMES_HOME)

    if (handoffConflict) {
      // A different updater already owns the marker — most often a previous
      // "Update" click whose updater is still alive and parked mid-run.
      // Spawning another here would overwrite its claim and let two updaters
      // mutate the checkout at once (#75778); refuse instead.
      rememberLog(`[updates] refusing hand-off: ${handoffConflict.message}`)
      emitUpdateProgress({ stage: 'error', message: handoffConflict.message, percent: null })

      return { ok: false, error: 'update-already-running', message: handoffConflict.message }
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

    // ── Pre-flight state.db integrity guard (#68474) ─────────────────
    // Emergency backup and header verification before the update touches
    // anything.  Runs while the backend is still alive.
    preflightStateDb(HERMES_HOME, rememberLog)

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

    // Preflight: after releasing our own backends, check for remaining
    // Hermes processes running from this venv.  The updater normally refuses
    // when it detects a holder, but because the updater is spawned detached
    // with stdio:ignore, the user never sees that refusal and the update
    // silently fails.  This preflight detects holders early and gives the
    // user an actionable error.  Windows-only; the .pyd lock hazard is a
    // Windows phenomenon.  ALL failures (blocked, missing python, timeout,
    // malformed output, missing psutil) abort the handoff — never proceed
    // to the detached updater when the venv state is unknown.
    if (IS_WINDOWS) {
      let scanOutcome = await scanVenvBlockers(updateRoot)

      if (scanOutcome.kind === 'blocked' && opts.stopSafeBlockers) {
        const stopResult = await stopSafeVenvBlockers(updateRoot, scanOutcome.result)
        rememberLog(
          `[updates] user-approved blocker cleanup: stopped=${stopResult.stopped.join(',') || 'none'} failed=${stopResult.failed.join(',') || 'none'}`
        )
        // Let verified process-tree termination finish unwinding wrapper shells,
        // then make the scanner — not the stale renderer payload — authoritative.
        await new Promise(resolve => setTimeout(resolve, 300))
        scanOutcome = await scanVenvBlockers(updateRoot)
      }

      // Re-scan before aborting on 'blocked' (#74805). Process-table teardown
      // is asynchronous on Windows: even after releaseBackendLock's PID-exit
      // wait, a grandchild the desktop never tracked (or a process an AV /
      // NTFS filter driver is holding in teardown) can stay enumerable for a
      // few more seconds and read as a holder. Each scan already costs
      // seconds (spawns a venv python + psutil sweep), so two retries with a
      // short dwell give the table time to settle without meaningfully
      // delaying the abort path when a REAL holder (a user terminal, second
      // window) is present — that holder is still there on the third scan.
      for (let attempt = 0; scanOutcome.kind === 'blocked' && attempt < 2; attempt++) {
        rememberLog(
          `[updates] venv-blocker scan reported ${scanOutcome.result.processes.length} holder(s); re-scanning after settle (attempt ${attempt + 2}/3)`
        )
        await new Promise(resolve => setTimeout(resolve, 1500))
        scanOutcome = await scanVenvBlockers(updateRoot)
      }

      if (scanOutcome.kind === 'blocked') {
        const message = formatBlockerMessage(scanOutcome.result)

        rememberLog(`[updates] venv-blocked: ${scanOutcome.result.processes.length} process(es) hold the install`)
        emitUpdateProgress({ stage: 'error', message, percent: null })
        startHermes().catch(() => {})

        return { ok: false, error: 'venv-blocked', message, blockers: scanOutcome.result.processes }
      }

      if (scanOutcome.kind === 'probe-failure') {
        const message = formatProbeFailedMessage()

        rememberLog(`[updates] venv-blocker probe failed: ${scanOutcome.error}`)
        emitUpdateProgress({ stage: 'error', message, percent: null })
        startHermes().catch(() => {})

        return { ok: false, error: 'venv-probe-failed', message }
      }
    }

    // Detached so the updater outlives this process — it needs us GONE before
    // `hermes update` will run (the venv shim is locked while we live).
    //
    // Prefer the repo-owned hand-off script over the staged Tauri binary.
    // The staged binary is frozen (no self-update path) and historically runs
    // months-stale updater logic — pre-#67369 cache resolver, pre-#74782
    // marker adoption — producing failures that were fixed on main long ago
    // (2026-08-09 incident). scripts/desktop-update/windows.ps1 ships WITH the
    // checkout, so each `hermes update` refreshes the code that drives the
    // next one. Checkouts that predate the script fall back to the binary
    // path unchanged.
    const scriptHandoff = resolveUpdateScriptHandoff(updateRoot)
    let child

    if (scriptHandoff) {
      const updateStartedAt = Math.floor(Date.now() / 1000)

      // A bare detached+hidden powershell spawn silently dies before -File
      // processing (console-subsystem init failure — see
      // wrapHandoffForDetachedConsole). Route through `cmd start` so the
      // script gets its own minimized console and survives our exit. The
      // wrapper cmd.exe exits immediately, so child.pid is NOT the script's
      // pid — the script claims the update marker itself with its own $PID
      // as its first action, and a relaunched Desktop parks on that.
      const wrapped = wrapHandoffForDetachedConsole(scriptHandoff, [
        '-InstallRoot',
        updateRoot,
        '-Branch',
        branch,
        '-DesktopPid',
        String(process.pid),
        '-RelaunchExe',
        process.execPath
      ])

      child = spawnUpdaterProcess(wrapped.command, wrapped.args, {
        cwd: HERMES_HOME,
        env: {
          ...process.env,
          HERMES_HOME,
          HERMES_UPDATE_STARTED_AT: String(updateStartedAt),
          PATH: pathWithHermesManagedNode(venvBin)
        },
        detached: true,
        stdio: 'ignore'
      })

      // Bridge marker: child.pid is the short-lived cmd.exe WRAPPER, not the
      // script (see wrapHandoffForDetachedConsole). Write it anyway to cover
      // the first moments of the hand-off — the script's step 0 overwrites it
      // with its own live $PID, and if the script never starts the wrapper's
      // dead pid makes the marker read as stale and self-delete (no wedge).
      // The `hermes update` child adopts the SCRIPT's claim via
      // update_lock.py's process-ancestry rule; no mtime heuristics needed.
      if (Number.isInteger(child.pid)) {
        writeUpdateMarker(HERMES_HOME, child.pid, { startedAt: updateStartedAt })
      }

      rememberLog(
        `[updates] launched repo hand-off script: ${scriptHandoff.scriptPath} (branch ${branch}); exiting desktop to release venv shim`
      )
    } else {
      child = spawnUpdaterProcess(updater, updaterArgs, {
        cwd: HERMES_HOME,
        env: {
          ...process.env,
          HERMES_HOME,
          PATH: pathWithHermesManagedNode(venvBin)
        },
        detached: true,
        stdio: 'ignore'
      })

      // Write the update-in-progress marker IMMEDIATELY — before the 2.5s
      // quit dwell. The Tauri updater won't write its own marker for several
      // seconds (window init + manifest), and during that gap our renderer
      // can reconnect and spawn a fresh backend that re-locks .pyd files in
      // the venv. By writing the marker ourselves the renderer's
      // waitForUpdateToFinish() gate sees a live update and parks instead.
      // The updater overwrites this with its own PID later; same format.
      //
      // SKIPPED for pre-#74782 staged updaters: those have no self-PID
      // exclusion, so they read this very marker as a foreign live owner and
      // abort with "Another Hermes update is already running (PID <itself>)" —
      // an unbreakable loop, because the update that would replace the stale
      // binary is the one being refused. Losing the anti-respawn hardening is
      // strictly better than never updating again, and the updater still writes
      // its own marker moments later.
      if (Number.isInteger(child.pid) && stagedUpdaterSupportsPrewrittenMarker(updater)) {
        writeUpdateMarker(HERMES_HOME, child.pid)
      } else if (Number.isInteger(child.pid)) {
        rememberLog(
          `[updates] skipping marker pre-write: staged updater predates self-adopt (${updater}); it would refuse its own claim`
        )
      }

      rememberLog(
        `[updates] launched updater: ${updater} ${updaterArgs.join(' ')}; exiting desktop to release venv shim`
      )
    }

    // Linger on the "updating — don't reopen" overlay long enough for the user
    // to actually read it (and to bridge the gap until the updater's own window
    // appears), THEN quit to release the venv shim. The updater rebuilds and
    // relaunches us when it's done. (#50419 — a 600ms quit looked like a crash
    // and lured users into the #50238 relaunch loop.)
    //
    // The dwell doubles as the hand-off settle window (#66753): watch the
    // detached child for an async spawn `error` (ENOENT/EACCES) or an early
    // non-zero/signal exit. On failure, DON'T quit — the user would be left
    // with no app, no updater, and no evidence. Restart our backend and
    // surface the error instead. The pre-written marker names the dead child
    // pid, so readLiveUpdateMarker self-heals it; no cleanup needed.
    const dwellStartedAt = Date.now()
    const handoffOutcome = await observeUpdaterHandoff(child, UPDATE_HANDOFF_DWELL_MS)

    if (!handoffOutcome.ok) {
      const message = `Update failed to start: ${handoffOutcome.message}. Hermes will keep running — try again, or run \`hermes update\` from a terminal.`

      rememberLog(`[updates] hand-off not viable, aborting quit: ${handoffOutcome.message}`)
      emitUpdateProgress({ stage: 'error', message, percent: null })
      startHermes().catch(() => {})

      return { ok: false, error: 'updater-spawn-failed', message }
    }

    isQuittingForHandoff = true
    setTimeout(
      () => {
        app.quit()
      },
      Math.max(0, UPDATE_HANDOFF_DWELL_MS - (Date.now() - dwellStartedAt))
    )

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

  const handoffConflict = updateHandoffConflict(HERMES_HOME)

  if (handoffConflict) {
    // Same hazard as applyUpdates (#75778): a live foreign updater already
    // owns the marker. Spawning another here would overwrite its claim and
    // race a second updater over the same install tree. The live updater
    // is already working on this exact install and will restart us when
    // it finishes, so treat this the same as a successful hand-off instead
    // of clobbering it with our own.
    rememberLog(`[bootstrap] refusing recovery hand-off: ${handoffConflict.message}`)
    isQuittingForHandoff = true
    setTimeout(() => {
      app.quit()
    }, UPDATE_HANDOFF_DWELL_MS)

    return true
  }

  const updateRoot = resolveUpdateRoot()
  const { branch: configuredBranch } = readDesktopUpdateConfig()

  const branch = directoryExists(path.join(updateRoot, '.git'))
    ? await resolveHealedBranch(updateRoot, configuredBranch || DEFAULT_UPDATE_BRANCH)
    : configuredBranch || DEFAULT_UPDATE_BRANCH

  const venvBin = path.join(updateRoot, 'venv', IS_WINDOWS ? 'Scripts' : 'bin')
  const venvHermes = path.join(venvBin, IS_WINDOWS ? 'hermes.exe' : 'hermes')
  const venvPython = path.join(venvBin, IS_WINDOWS ? 'python.exe' : 'python')

  // The updater invokes the venv's Hermes launcher, which in turn requires the
  // venv interpreter. A bootstrap-complete marker proves only that setup once
  // finished; it can outlive a manually removed or quarantined venv. Sending a
  // marker-only install through --update dead-ends at "Could not find the hermes
  // CLI" instead of rebuilding the runtime, so only a runnable pair gets the
  // gentle update path. Partial or missing runtimes go through full repair.
  const updaterArgs = chooseUpdaterArgs(
    {
      hasBootstrapMarker: fileExists(path.join(updateRoot, '.hermes-bootstrap-complete')),
      hasVenvHermes: fileExists(venvHermes),
      hasVenvPython: fileExists(venvPython)
    },
    branch
  )

  await releaseBackendLockForUpdate(updateRoot)

  const child = spawnUpdaterProcess(updater, updaterArgs, {
    cwd: HERMES_HOME,
    env: {
      ...process.env,
      HERMES_HOME,
      PATH: pathWithHermesManagedNode(venvBin)
    },
    detached: true,
    stdio: 'ignore'
  })

  // Same marker pre-write as applyUpdates — see comment there. The recovery
  // hand-off has the same window where the renderer can respawn a backend
  // before the updater writes its own marker, and the same stale-updater
  // exclusion: a pre-#74782 binary would refuse its own pre-written claim and
  // strand the very recovery meant to heal the install.
  if (Number.isInteger(child.pid) && stagedUpdaterSupportsPrewrittenMarker(updater)) {
    writeUpdateMarker(HERMES_HOME, child.pid)
  } else if (Number.isInteger(child.pid)) {
    rememberLog(
      `[bootstrap] skipping marker pre-write: staged updater predates self-adopt (${updater}); it would refuse its own claim`
    )
  }

  rememberLog(
    `[bootstrap] handed off ${reason} recovery to updater: ${updater} ${updaterArgs.join(' ')}; exiting desktop to release app.asar`
  )
  // Same dwell as the in-app update hand-off (#50419): give the updater's
  // window time to appear before we vanish, so the recovery doesn't look like
  // a crash and provoke a mid-recovery relaunch. The dwell doubles as the
  // hand-off settle window (#66753): a spawn error or early updater death
  // returns false so the caller falls through to its next recovery path
  // instead of quitting into nothing.
  const dwellStartedAt = Date.now()
  const handoffOutcome = await observeUpdaterHandoff(child, UPDATE_HANDOFF_DWELL_MS)

  if (!handoffOutcome.ok) {
    rememberLog(`[bootstrap] recovery hand-off not viable, staying alive: ${handoffOutcome.message}`)

    return false
  }

  isQuittingForHandoff = true
  setTimeout(
    () => {
      app.quit()
    },
    Math.max(0, UPDATE_HANDOFF_DWELL_MS - (Date.now() - dwellStartedAt))
  )

  return true
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

// ── Pre-flight state.db integrity guard (#68474) ─────────────────────
// Take an emergency snapshot of state.db and verify the live copy is
// intact before any update process mutates the install.  Runs in the
// desktop Electron process itself, before the backend is killed and
// before the updater is spawned — a separate safety net from the
// Python-level pre-update snapshot inside `hermes update`.
function preflightStateDb(hermesHome, rememberLog) {
  const stateDbPath = path.join(hermesHome, 'state.db')

  if (!fileExists(stateDbPath)) {
    rememberLog('[updates] state.db pre-flight: not found (fresh install?)')

    return
  }

  try {
    const stat = fs.statSync(stateDbPath)

    if (stat.size > 100) {
      const fd = fs.openSync(stateDbPath, 'r')
      const header = Buffer.alloc(16)

      fs.readSync(fd, header, 0, 16, 0)
      fs.closeSync(fd)

      const expectedHeader = Buffer.from('SQLite format 3\0')
      const headerOk = header.equals(expectedHeader)

      rememberLog(
        `[updates] state.db pre-flight: size=${stat.size}, ` +
          `headerOk=${headerOk}, headerHex=${header.toString('hex')}`
      )

      if (!headerOk) {
        rememberLog(
          '[updates] state.db header is INVALID before update — ' +
            'this indicates pre-existing corruption or a concurrent write issue'
        )
      }

      // Emergency timestamped backup, separate from the Python-level snapshot.
      const ts = new Date().toISOString().replace(/[:.]/g, '-')

      const emergencyPath = path.join(hermesHome, `state.db.pre-update-emergency-${ts}.bak`)

      try {
        fs.copyFileSync(stateDbPath, emergencyPath)
        const emergStat = fs.statSync(emergencyPath)

        rememberLog(`[updates] emergency state.db backup: ${emergencyPath} ` + `(${emergStat.size} bytes)`)

        // Prune to the 2 most recent emergency backups.
        try {
          const homeDir = fs.readdirSync(hermesHome)

          const backups = homeDir
            .filter(
              f =>
                f.startsWith('state.db.pre-update-emergency-') &&
                f.endsWith('.bak') &&
                f !== path.basename(emergencyPath)
            )
            .sort()
            .reverse()

          for (const old of backups.slice(2)) {
            try {
              fs.unlinkSync(path.join(hermesHome, old))
            } catch {
              void 0
            }
          }
        } catch {
          void 0
        }
      } catch (copyErr) {
        rememberLog(`[updates] emergency state.db backup failed: ${copyErr.message}`)
      }
    } else {
      rememberLog(`[updates] state.db too small (${stat.size} bytes) for a valid SQLite database`)
    }
  } catch (statErr) {
    rememberLog(`[updates] could not stat state.db before update: ${statErr.message}`)
  }
}

// macOS/Linux update hand-off: spawn the repo-owned posix orchestrator
// (scripts/desktop-update/posix.sh) detached and QUIT. The script waits us
// out, runs `hermes update`, swaps/relaunches the app bundle, and writes
// .hermes-update-result.json for the relaunched Desktop to surface. It shows
// its own tiny shim window (or nothing, headless) — this process only needs
// to leave. Checkouts that predate the script get the manual card once.
async function applyUpdatesPosixHandoff(opts: any) {
  const updateRoot = resolveUpdateRoot()
  const handoff = resolvePosixScriptHandoff(updateRoot)

  if (!handoff) {
    emitUpdateProgress({ stage: 'manual', message: 'hermes update', percent: null })

    return { ok: true, manual: true, command: 'hermes update', hermesRoot: updateRoot }
  }

  const handoffConflict = updateHandoffConflict(HERMES_HOME)

  if (handoffConflict) {
    // Same hazard as the Windows path (#75778): a live foreign updater
    // already owns the marker — refuse rather than double-mutate the tree.
    rememberLog(`[updates] refusing posix hand-off: ${handoffConflict.message}`)
    emitUpdateProgress({ stage: 'error', message: handoffConflict.message, percent: null })

    return { ok: false, error: 'update-already-running', message: handoffConflict.message }
  }

  // ── Pre-flight state.db integrity guard (#68474) ──
  preflightStateDb(HERMES_HOME, rememberLog)

  // Branch-pin so a non-main checkout doesn't get switched to main (and
  // self-heal to main when the pinned branch no longer exists on origin).
  let branch = 'main'

  try {
    const head = await runGit(['rev-parse', '--abbrev-ref', 'HEAD'], { cwd: updateRoot })
    const current = (head.stdout || '').trim()

    if (head.code === 0 && current && current !== 'HEAD') {
      branch = await resolveHealedBranch(updateRoot, current)
    }
  } catch {
    // best effort
  }

  const args = [...handoff.args, '--install-root', updateRoot, '--branch', branch, '--desktop-pid', String(process.pid)]
  const updateStartedAt = Math.floor(Date.now() / 1000)

  // Relaunch target: the running .app bundle on mac (script swaps the
  // rebuilt bundle over it), the running binary elsewhere. The script's gate
  // (an exact port of update-relaunch.ts's decideRelaunchOutcome) relaunches
  // only a binary the rebuild replaced with a launchable sandbox helper —
  // replaying the original launch context (filtered args, cwd, sandbox
  // opt-out) so a deep-link or --no-sandbox launch survives the update.
  const targetApp = IS_MAC ? runningAppBundle() : process.execPath

  if (targetApp) {
    args.push('--relaunch-target', targetApp)
  }

  const relaunchArgs = collectRelaunchArgs(process.argv.slice(1))

  if (!IS_MAC) {
    args.push('--relaunch-cwd', process.cwd())

    if (sandboxFallbackFromEnv(process.env, relaunchArgs)) {
      args.push('--sandbox-fallback')
    }

    if (relaunchArgs.length) {
      args.push('--', ...relaunchArgs)
    }
  }

  const child = spawnUpdaterProcess(handoff.command, args, {
    cwd: HERMES_HOME,
    env: {
      ...process.env,
      HERMES_HOME,
      HERMES_UPDATE_STARTED_AT: String(updateStartedAt),
      PATH: pathWithHermesManagedNode(path.join(updateRoot, 'venv', 'bin'))
    },
    detached: true,
    stdio: 'ignore'
  })

  // Bridge marker (same contract as the Windows hand-off): cover the gap
  // until the script claims the marker with its own pid as step 0. If the
  // script never starts, the dead pid reads as stale and self-deletes.
  if (Number.isInteger(child.pid)) {
    writeUpdateMarker(HERMES_HOME, child.pid, { startedAt: updateStartedAt })
  }

  rememberLog(`[updates] launched posix hand-off: ${handoff.scriptPath} (branch ${branch}); quitting to hand off`)
  emitUpdateProgress({
    stage: 'restart',
    message:
      'Updating Hermes — this window will close. Don’t reopen Hermes yourself; it restarts automatically when the update finishes.',
    percent: 100
  })

  // Settle window (#66753): the reported macOS failure mode is exactly this
  // path — the app quits, bash/posix.sh dies early (or was never spawnable),
  // and the user is left with no app, no updater, and no relaunch. Watch the
  // child through the dwell; on spawn error or early death, stay alive and
  // surface the failure instead of quitting into nothing.
  const dwellStartedAt = Date.now()
  const handoffOutcome = await observeUpdaterHandoff(child, UPDATE_HANDOFF_DWELL_MS)

  if (!handoffOutcome.ok) {
    const message = `Update failed to start: ${handoffOutcome.message}. Hermes will keep running — try again, or run \`hermes update\` from a terminal.`

    rememberLog(`[updates] posix hand-off not viable, aborting quit: ${handoffOutcome.message}`)
    emitUpdateProgress({ stage: 'error', message, percent: null })

    return { ok: false, error: 'updater-spawn-failed', message }
  }

  isQuittingForHandoff = true
  setTimeout(
    () => {
      app.quit()
    },
    Math.max(0, UPDATE_HANDOFF_DWELL_MS - (Date.now() - dwellStartedAt))
  )

  return { ok: true, handedOff: true, updater: handoff.scriptPath }
}

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

function activeRuntimeState() {
  // We DELIBERATELY do NOT verify that the checkout is currently at the
  // pinned commit -- users update via the in-app update path or `hermes
  // update`, which moves HEAD legitimately. The marker only attests "a
  // desktop-managed bootstrap ran here at least once"; runtime usability is
  // what decides whether we can actually launch.
  return classifyActiveRuntime(readBootstrapMarker(), BOOTSTRAP_MARKER_SCHEMA_VERSION, isActiveRuntimeUsable())
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
  const asarIndex = path.join(APP_ROOT, 'dist', 'index.html')
  const webDistIndex = path.join(resolveWebDist(), 'index.html')

  // A packaged build ships dist/ twice: inside app.asar AND — because
  // asarUnpack lists dist/** — beside it in app.asar.unpacked. Prefer the
  // unpacked tree, matching the resolveWebDist()/unpackedPathFor precedent:
  // it is the copy the embedded dashboard serves and the copy a repair
  // rewrites, while pointing the window at the asar-internal index.html is
  // exactly how lazy chunks end up fetched from a path that cannot serve
  // them (#93479). Every window loader shares this resolver (main, overlay,
  // quick), so the ordering fix covers all of them. Dev is unchanged:
  // unpackedPathFor is a no-op outside an asar, so both candidates collapse
  // to APP_ROOT/dist and the original order is preserved.
  const candidates = IS_PACKAGED ? [webDistIndex, asarIndex] : [asarIndex, webDistIndex]
  const present = [...new Set(candidates)].filter(fileExists)

  // index.html and the hashed chunks it names are one generation. An update
  // that replaces only one of the two shipped copies (app.asar vs
  // app.asar.unpacked) leaves a TORN copy: the window loads, then dies on the
  // first lazy import with "Failed to fetch dynamically imported module" and
  // every restart reloads the same torn copy. Prefer a copy whose modules are
  // all present, so the intact generation heals the boot by itself.
  for (const candidate of present) {
    const missing = missingRendererAssets(candidate)

    if (missing.length === 0) {
      return candidate
    }

    rememberLog(
      `[renderer] skipping torn renderer bundle at ${candidate}: ` +
        `${missing.length} module file(s) named by index.html are missing ` +
        `(${missing.slice(0, 3).join(', ')}${missing.length > 3 ? ', …' : ''})`
    )
  }

  if (present.length > 0) {
    // Every copy is torn. Load the first one anyway — the boundary's error is
    // still better than a blank window — but say what is wrong and how to fix
    // it, because no amount of restarting repairs a torn bundle.
    rememberLog(
      `[renderer] every renderer bundle is incomplete (${present.join(', ')}). ` +
        `The last update replaced the app while its files were locked. ` +
        `Repair with: hermes desktop --force-build`
    )

    return present[0]
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

  // The venv whose interpreter we selected is the venv whose site-packages
  // belong on PYTHONPATH — findPythonForRoot may have picked `.venv` over
  // `venv`, and mixing the two crashes the backend on its first native
  // import (see venvRootForPython). Fall back to root/venv only for a
  // system python, where the historical layout is the best guess.
  const venvRoot = venvRootForPython(python, root) ?? path.join(root, 'venv')
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

  // 3. ACTIVE_HERMES_ROOT — the canonical install at
  //    %LOCALAPPDATA%\\hermes\\hermes-agent (Windows) or ~/.hermes/hermes-agent.
  //    A valid bootstrap marker proves Desktop finished the first-run install
  //    flow, but marker provenance is NOT the same thing as runtime usability:
  //    the CLI can create the exact same repo+venv layout, and older desktop
  //    builds could leave a healthy install behind without the marker. If the
  //    active runtime is usable, launch it directly; only fall through to
  //    bootstrap when the runtime itself is unusable.
  const activeRuntime = activeRuntimeState()

  if (activeRuntime.shouldUseActiveRuntime && !bootstrapRepairRequested) {
    if (!activeRuntime.hasValidMarker) {
      rememberLog(
        `[bootstrap] Active Hermes runtime at ${ACTIVE_HERMES_ROOT} is usable but the bootstrap marker is missing or stale; skipping first-run bootstrap.`
      )
    }

    return createActiveBackend(backendArgs)
  }

  if (bootstrapRepairRequested) {
    rememberLog('[bootstrap] repair requested; bypassing the usable active runtime to re-run the installer')
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

      // HERMES_DESKTOP_HERMES is an explicit deployment override (used by
      // the Nix wrapper), not a discovered PATH candidate. It must not fall
      // through to the install-script bootstrap if the optional probe times
      // out under load; the pinned backend is the only valid runtime there.
      if (shouldTrustHermesOverride(hermesOverride) || verifyHermesCli(hermesCommand, { shell: shellForProbe })) {
        // `unwrapped` above already answered "is this a Windows venv shim?" —
        // it was null (not a shim, or its import probe failed). Do NOT re-run
        // unwrapWindowsVenvHermesCommand here: the second call repeats the
        // same un-memoized import probe, costing up to another full probe
        // timeout on the boot path for an answer we already have.
        return {
          label: `existing Hermes CLI at ${hermesCommand}`,
          command: hermesCommand,
          args: backendArgs,
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

    // The repair request has been honoured by reaching the installer; clear it
    // so a later boot isn't forced through bootstrap again.
    bootstrapRepairRequested = false
    bootstrapRepairAttempt = 0

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
    // normal flow because activeRuntimeState() requires isHermesSourceRoot()
    // plus an importable hermes_cli before it hands back the active runtime.
    // If we hit this, the user (or a deleted venv) broke the invariant; tell
    // them to re-run the install.
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

// Assemble a single-file multipart/form-data body (FastAPI `UploadFile`
// endpoints, e.g. kanban attachments). Hand-rolled because node's http has no
// FormData and the payload is one file — a dependency would be overkill.
function multipartBody(upload) {
  const boundary = `----hermes-${crypto.randomBytes(12).toString('hex')}`
  const filename = String(upload.filename || 'file').replace(/["\r\n]/g, '_')

  const body = Buffer.concat([
    Buffer.from(
      `--${boundary}\r\n` +
        `Content-Disposition: form-data; name="file"; filename="${filename}"\r\n` +
        `Content-Type: ${upload.contentType || 'application/octet-stream'}\r\n\r\n`
    ),
    Buffer.from(upload.bytes),
    Buffer.from(`\r\n--${boundary}--\r\n`)
  ])

  return { body, contentType: `multipart/form-data; boundary=${boundary}` }
}

function fetchJson(url, token, options: any = {}) {
  // Retry policy lives in api-transport.ts: idempotent verbs retry on any
  // transient transport error; POST/PUT/DELETE only when the request provably
  // never reached the server (see shouldRetryRequest) — never double-submit.
  return withRetry(
    (requestState: any) =>
      new Promise((resolve, reject) => {
        const { body, contentType } = options.upload
          ? multipartBody(options.upload)
          : {
              body: options.body === undefined ? undefined : Buffer.from(JSON.stringify(options.body)),
              contentType: 'application/json'
            }

        const parsed = new URL(url)
        const client = parsed.protocol === 'https:' ? https : http
        const agent = jsonAgentFor(parsed.protocol)
        const timeoutMs = resolveTimeoutMs(options.timeoutMs, DEFAULT_FETCH_TIMEOUT_MS)

        if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
          reject(new Error(`Unsupported Hermes backend URL protocol: ${parsed.protocol}`))

          return
        }

        const req = client.request(
          parsed,
          {
            agent,
            method: options.method || 'GET',
            headers: {
              ...headersForRemoteRequest(url),
              ...(options.headers || {}),
              'Content-Type': contentType,
              'X-Hermes-Session-Token': token,
              // RFC 8252 native flow authenticates the gated gateway with a bearer
              // token instead of the loopback session-token header. When
              // ``options.bearer`` is set we send Authorization: Bearer <token>;
              // the gateway's OAuth gate verifies it via the provider stack with
              // no cookie involved.
              ...(options.bearer ? { Authorization: `Bearer ${options.bearer}` } : {}),
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

        // From here the request goes on the wire: a later transport error can no
        // longer prove the server didn't process it, so non-idempotent verbs must
        // not be retried past this point.
        requestState.bodySent = true

        if (body) {
          req.write(body)
        }

        req.end()
      }),
    { method: options.method || 'GET' }
  )
}

// Token-auth download that streams the response body straight to a
// user-selected destination (via finalizeGatewayDownload) instead of buffering
// the whole file in memory. The connect timeout is cleared once headers arrive
// so a slow save dialog or a large stream doesn't trip it. `options.bearer`
// switches the header to Authorization (RFC 8252 native flow), matching fetchJson.
function downloadViaTokenToFile(url, token, ctx, options: any = {}) {
  return new Promise((resolve, reject) => {
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

    const client = parsed.protocol === 'https:' ? https : http
    const agent = downloadAgentFor(parsed.protocol)
    const timeoutMs = resolveTimeoutMs(options.timeoutMs, DEFAULT_FETCH_TIMEOUT_MS)

    const req = client.request(
      parsed,
      {
        agent,
        method: 'GET',
        headers: options.bearer ? { Authorization: `Bearer ${options.bearer}` } : { 'X-Hermes-Session-Token': token }
      },
      res => {
        // Headers arrived — the connection phase is done. Drop the idle timeout
        // so it can't abort mid-stream or while the save dialog is open.
        req.setTimeout(0)
        finalizeGatewayDownload(res, res.statusCode || 500, res.headers || {}, {
          ...ctx,
          abort: () => {
            try {
              req.destroy()
            } catch {
              // already finished
            }
          }
        }).then(resolve, reject)
      }
    )

    req.on('error', reject)
    req.setTimeout(timeoutMs, () => {
      req.destroy(new Error(`Timed out connecting to Hermes backend after ${timeoutMs}ms`))
    })
    req.end()
  })
}

function fetchPublicJson(url, options: any = {}) {
  // Credential-free JSON GET/POST for public gateway endpoints
  // (``/api/status``, ``/api/auth/providers``). Unlike ``fetchJson`` it sends
  // NO ``X-Hermes-Session-Token`` header — used by the auth-mode probe before
  // any credentials exist, and any time we must not leak a token to an
  // endpoint that doesn't need one.
  return withRetry(
    (requestState: any) =>
      new Promise((resolve, reject) => {
        const body = options.body === undefined ? undefined : Buffer.from(JSON.stringify(options.body))
        let parsed

        try {
          parsed = new URL(url)
        } catch (error) {
          reject(new Error(`Invalid URL: ${error.message}`))

          return
        }

        const client = parsed.protocol === 'https:' ? https : http
        const agent = jsonAgentFor(parsed.protocol)
        const timeoutMs = resolveTimeoutMs(options.timeoutMs, DEFAULT_FETCH_TIMEOUT_MS)

        if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
          reject(new Error(`Unsupported Hermes backend URL protocol: ${parsed.protocol}`))

          return
        }

        const req = client.request(
          parsed,
          {
            agent,
            method: options.method || 'GET',
            headers: {
              ...headersForRemoteRequest(url),
              ...(options.headers || {}),
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

        // Past this point the request is on the wire — see fetchJson.
        requestState.bodySent = true

        if (body) {
          req.write(body)
        }

        req.end()
      }),
    { method: options.method || 'GET' }
  )
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

// ─── Favicon resolution ──────────────────────────────────────────────────────
// The ladder itself is electron/favicon.ts; this is its I/O, its cache, and
// the one rule that belongs to the app rather than the algorithm: one icon
// per host. A connector's mark doesn't vary by path, and hosting the cache on
// the host key means Linear's docs page and Linear's MCP endpoint cost one
// lookup between them.

const FAVICON_CACHE_PATH = path.join(app.getPath('userData'), 'favicon-cache.json')
const FAVICON_CACHE_LIMIT = 400
const FAVICON_TTL_MS = 30 * 24 * 60 * 60 * 1000
// A miss is cheap to re-check and expensive to be wrong about (a site that
// was behind a captcha yesterday has a logo today), so it expires fast.
const FAVICON_MISS_TTL_MS = 12 * 60 * 60 * 1000
const FAVICON_TIMEOUT_MS = 6000
const FAVICON_MAX_BYTES = 256 * 1024
const FAVICON_WRITE_DEBOUNCE_MS = 3000

let faviconCache: Map<string, { at: number; icon: string }> | null = null
let faviconWriteTimer: null | ReturnType<typeof setTimeout> = null
const faviconInflight = new Map<string, Promise<string>>()

function faviconCacheKey(rawUrl: string): string {
  try {
    return new URL(rawUrl).hostname.replace(/^www\./i, '').toLowerCase()
  } catch {
    return ''
  }
}

function loadFaviconCache(): Map<string, { at: number; icon: string }> {
  if (faviconCache) {
    return faviconCache
  }

  faviconCache = new Map()

  try {
    const raw = JSON.parse(fs.readFileSync(FAVICON_CACHE_PATH, 'utf8'))

    for (const [host, entry] of Object.entries(raw?.icons ?? {})) {
      const at = Number((entry as { at?: number })?.at)
      const icon = String((entry as { icon?: string })?.icon ?? '')

      if (Number.isFinite(at) && Date.now() - at < (icon ? FAVICON_TTL_MS : FAVICON_MISS_TTL_MS)) {
        faviconCache.set(host, { at, icon })
      }
    }
  } catch {
    // No cache yet, or it's unreadable — resolving again is the whole cost.
  }

  return faviconCache
}

function saveFaviconCacheSoon() {
  if (faviconWriteTimer) {
    return
  }

  faviconWriteTimer = setTimeout(() => {
    faviconWriteTimer = null

    try {
      const icons = Object.fromEntries(loadFaviconCache())

      fs.writeFileSync(FAVICON_CACHE_PATH, JSON.stringify({ icons }), 'utf8')
    } catch {
      // Cache is an optimization; failing to persist it costs one refetch.
    }
  }, FAVICON_WRITE_DEBOUNCE_MS)

  faviconWriteTimer.unref?.()
}

async function faviconFetch(url: string, accept: string) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), FAVICON_TIMEOUT_MS)

  try {
    return await electronNet.fetch(url, {
      // Same browser-shaped identity the title fetcher uses: a plain Electron
      // UA gets a challenge page from anything behind a bot wall.
      headers: { Accept: accept, 'Accept-Language': 'en-US,en;q=0.7', 'User-Agent': TITLE_USER_AGENT },
      redirect: 'follow',
      signal: controller.signal
    })
  } finally {
    clearTimeout(timer)
  }
}

const faviconIo: FaviconIo = {
  fetchImage: async url => {
    const response = await faviconFetch(url, 'image/avif,image/webp,image/svg+xml,image/*;q=0.8,*/*;q=0.5')

    if (!response.ok) {
      return null
    }

    const buffer = await response.arrayBuffer()

    if (buffer.byteLength === 0 || buffer.byteLength > FAVICON_MAX_BYTES) {
      return null
    }

    return { bytes: new Uint8Array(buffer), mime: response.headers.get('content-type') ?? '' }
  },
  fetchText: async url => {
    const response = await faviconFetch(url, 'text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.5')

    return response.ok ? (await response.text()).slice(0, TITLE_BYTE_BUDGET * 2) : ''
  }
}

function resolveFaviconCached(rawUrl: string): Promise<string> {
  const key = faviconCacheKey(String(rawUrl || '').trim())

  if (!key) {
    return Promise.resolve('')
  }

  const cache = loadFaviconCache()
  const hit = cache.get(key)

  if (hit && Date.now() - hit.at < (hit.icon ? FAVICON_TTL_MS : FAVICON_MISS_TTL_MS)) {
    return Promise.resolve(hit.icon)
  }

  const inflight = faviconInflight.get(key)

  if (inflight) {
    return inflight
  }

  const pending = resolveFavicon(rawUrl, faviconIo)
    .catch(() => '')
    .then(icon => {
      if (cache.size >= FAVICON_CACHE_LIMIT) {
        cache.delete(cache.keys().next().value)
      }

      cache.set(key, { at: Date.now(), icon })
      saveFaviconCacheSoon()
      faviconInflight.delete(key)

      return icon
    })

  faviconInflight.set(key, pending)

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

async function saveImageFromUrl(rawUrl) {
  const { buffer, mimeType } = (await resourceBufferFromUrl(rawUrl)) as any
  const extension = extensionForMimeType(mimeType) || '.png'
  // Generated-image URLs (fal.media etc.) usually end in an extensionless
  // content hash. Keep the name but always guarantee an extension — without
  // one Windows saves an unopenable "All Files" blob (#image18 report).
  const baseName = filenameFromUrl(rawUrl, `image${extension}`)
  const fallbackName = path.extname(baseName) ? baseName : `${baseName}${extension}`

  let downloadsDir = ''

  try {
    downloadsDir = app.getPath('downloads')
  } catch {
    // Leave the dialog at its last-used location when the OS has no
    // Downloads directory to offer.
  }

  const result = await dialog.showSaveDialog(mainWindow, {
    title: 'Save Image',
    defaultPath: downloadsDir ? path.join(downloadsDir, fallbackName) : fallbackName,
    filters: [
      { name: 'Images', extensions: ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'svg'] },
      { name: 'All Files', extensions: ['*'] }
    ]
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
  const isPdf = PREVIEW_PDF_EXTENSIONS.has(ext) || mimeType === 'application/pdf'
  const previewKind = isHtml ? 'html' : isImage ? 'image' : isPdf ? 'pdf' : metadata.binary ? 'binary' : 'text'

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

function requestOptionsWithHeaders(options: any = {}, headers = {}) {
  return {
    ...options,
    headers: {
      ...headers,
      ...(options.headers || {})
    }
  }
}

/** Watch a DIRECTORY for entry churn (folders appearing/vanishing) — the
 *  disk-plugin door's "new plugin folder" signal, replacing the renderer's 5s
 *  readdir poll. Same registry + change channel as the preview file watchers
 *  (the renderer reconciles on any tick; per-file edits stay on their own
 *  watches), so stopPreviewFileWatch/closePreviewWatchers manage these too. */
function watchDirectory(rawDir) {
  const watchDir = path.resolve(String(rawDir || ''))

  if (!fs.existsSync(watchDir) || !fs.statSync(watchDir).isDirectory()) {
    throw new Error(`Not a directory: ${watchDir}`)
  }

  const id = crypto.randomBytes(12).toString('base64url')
  let timer = null

  const watcher = fs.watch(watchDir, () => {
    if (timer) {
      clearTimeout(timer)
    }

    timer = setTimeout(() => {
      timer = null
      sendPreviewFileChanged({ id, path: watchDir, url: pathToFileURL(watchDir).toString() })
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

  return { id, path: watchDir }
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
  const nativeAt = authMode === 'oauth' ? await ensureNativeAccessToken(baseUrl).catch(() => null) : null
  const probeAuth = resolveReadinessProbeAuth(authMode, nativeAt, token)

  if (probeAuth.kind === 'bearer') {
    return {
      // fetchJson takes the bearer via `options.bearer` — a raw `headers`
      // option is ignored, so passing one here would silently probe
      // uncredentialed and reintroduce the 401 loop.
      probeHealth: (url, options: any = {}) => fetchJson(url, null, { ...options, bearer: probeAuth.token }),
      probeIsCredentialed: true
    }
  }

  if (probeAuth.kind === 'cookie') {
    return {
      probeHealth: (url, options: any = {}) => fetchJsonViaOauthSession(url, options),
      probeIsCredentialed: true
    }
  }

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

function getWindowButtonPosition(win = mainWindow) {
  if (!IS_MAC) {
    return null
  }

  // Fullscreen hides the traffic lights — treat as no left-side controls so the
  // renderer drops the traffic-light dodge inset and Y nudge.
  if (win?.isFullScreen?.()) {
    return null
  }

  return win?.getWindowButtonPosition?.() || WINDOW_BUTTON_POSITION
}

function getNativeOverlayWidth() {
  return computeNativeOverlayWidth({ isWindows: IS_WINDOWS, isWsl: IS_WSL, isMac: IS_MAC })
}

function getWindowState(win = mainWindow) {
  return {
    isFullscreen: Boolean(win?.isFullScreen?.()),
    isMinimized: Boolean(win?.isMinimized?.()),
    isVisible: Boolean(win?.isVisible?.()),
    nativeOverlayWidth: getNativeOverlayWidth(),
    windowButtonPosition: getWindowButtonPosition(win),
    darwinMajor: IS_MAC ? DARWIN_MAJOR : 0
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

/**
 * Run a browser gesture on the guest page the user is actually in, if any.
 *
 * A `<webview>` guest is its own out-of-process webContents: pointer and focus
 * events inside the page never reach the host document, so NOTHING in the
 * renderer — not `document.activeElement`, not the layout tree's hover/focus
 * ladder — can see that the user is in there. Main can: Electron tracks the
 * focused webContents across processes, which is the definition of a runtime
 * fact it owns.
 *
 * Returns false when focus is in the app's own chrome, where the renderer is
 * the one that knows which pane is active.
 */
function commandFocusedGuest(command: 'back' | 'forward' | 'reload'): boolean {
  const focused = electronWebContents.getFocusedWebContents()

  if (!focused || focused.isDestroyed() || focused.getType() !== 'webview') {
    return false
  }

  const history = focused.navigationHistory

  if (command === 'reload') {
    focused.reload()
  } else if (command === 'back') {
    if (!history.canGoBack()) {
      return true
    }

    history.goBack()
  } else {
    if (!history.canGoForward()) {
      return true
    }

    history.goForward()
  }

  return true
}

/**
 * Ask the renderer to run a browser-navigation gesture on its focused preview
 * pane. `reload` also has an app-level fallback (reload the window); `back` and
 * `forward` mean nothing outside the browser, so the renderer just ignores them.
 */
function sendPreviewNavCommand(command: 'back' | 'forward' | 'reload') {
  // The user is inside the page itself — main is the only party that can see
  // that, so act here and never round-trip.
  if (commandFocusedGuest(command)) {
    return
  }

  if (!mainWindow || mainWindow.isDestroyed()) {
    return
  }

  const { webContents } = mainWindow

  if (!webContents || webContents.isDestroyed()) {
    return
  }

  webContents.send('hermes:preview-nav', command)
}

/**
 * The native back/forward gestures, which never reach the renderer on their own.
 *
 * - macOS: a two/three-finger swipe. Chromium's own overscroll navigation is
 *   off in an Electron window, so the OS gesture surfaces as this event and
 *   nothing consumes it. Requires "Swipe between pages" in System Settings.
 * - Windows/Linux: the dedicated back/forward buttons on a mouse, delivered as
 *   `WM_APPCOMMAND`.
 */
function installBrowserNavGestures(window) {
  window.on('swipe', (_event, direction) => {
    if (direction === 'left' || direction === 'right') {
      // Swipe LEFT moves the page left, revealing what's behind it — that's
      // back. Matches Safari, Chrome, and Finder.
      sendPreviewNavCommand(direction === 'left' ? 'back' : 'forward')
    }
  })

  window.on('app-command', (event, command) => {
    if (command !== 'browser-backward' && command !== 'browser-forward') {
      return
    }

    // Claim it either way: unhandled, Chromium walks the HOST document's
    // history, which would navigate the app shell itself.
    event.preventDefault()
    sendPreviewNavCommand(command === 'browser-backward' ? 'back' : 'forward')
  })
}

function sendOpenFolderRequested() {
  if (!mainWindow || mainWindow.isDestroyed()) {
    return
  }

  const webContents = mainWindow.webContents

  if (!webContents || webContents.isDestroyed()) {
    return
  }

  webContents.send('hermes:open-folder-requested')
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

// Mirror of powerMonitor's AC/battery state, broadcast to every window so
// renderer backstop polls can slow down on battery (see store/power.ts).
// `null` until the first powerMonitor read after app ready.
let onBatteryPower: boolean | null = null

// Renderer-side battery gating seeds from this and stays current via the
// 'hermes:power-battery' push below.
ipcMain.handle('hermes:power-battery:get', () => onBatteryPower === true)

function broadcastBatteryState(next: boolean) {
  if (onBatteryPower === next) {
    return
  }

  onBatteryPower = next

  for (const win of BrowserWindow.getAllWindows()) {
    const { webContents } = win

    if (webContents && !webContents.isDestroyed()) {
      webContents.send('hermes:power-battery', next)
    }
  }
}

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
    powerMonitor.on('on-battery', () => broadcastBatteryState(true))
    powerMonitor.on('on-ac', () => broadcastBatteryState(false))
    onBatteryPower = powerMonitor.isOnBatteryPower()
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

// Push titlebar/fullscreen chrome state to a window's renderer. Defaults to the
// primary, but any full chat window (primary or a secondary "instance" peer)
// passes itself so its own fullscreen toggle drives its own traffic-light inset.
function sendWindowStateChanged(nextIsFullscreen?: boolean, target = mainWindow) {
  if (!target || target.isDestroyed()) {
    return
  }

  const { webContents } = target

  if (!webContents || webContents.isDestroyed()) {
    return
  }

  const state = getWindowState(target)

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
      // No accelerator: ⌘⇧N is a rebindable renderer keybind (session.newWindow);
      // a menu accelerator would fight the rebind panel and (on macOS) be
      // swallowed before the renderer sees it. Here purely for discoverability.
      { click: () => createInstanceWindow(), label: 'New Window' },
      // Same no-accelerator rationale: ⌘O is the rebindable renderer keybind
      // (workspace.openFolder). Clicking runs the same open-folder-as-project
      // flow through the renderer.
      { click: () => sendOpenFolderRequested(), label: 'Open Folder…' },
      { type: 'separator' },
      IS_MAC
        ? {
            // NO accelerator: on macOS a registered ⌘W is consumed by the OS
            // menu before the web contents ever sees it (and registerAccelerator
            // false is a no-op on mac — electron#18295). Leaving it off lets the
            // `before-input-event` handler below intercept ⌘W and route it to the
            // renderer's close-active-tab. Clicking the item still closes the tab
            // (or window) via the same request.
            click: () => sendClosePreviewRequested(),
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
      // ⌘⇧V is only wired up by this item existing: an accelerator with no menu
      // entry is never translated into an editor command, so the chord was a
      // no-op in every input in the app. The composer inserts plain text on
      // every paste anyway, so this is the same result as ⌘V there — it's the
      // terminal, preview, and other editable surfaces that need the strip.
      { role: 'pasteAndMatchStyle' },
      { role: 'delete' },
      { role: 'selectAll' }
    ]
  })
  template.push({
    label: 'View',
    submenu: [
      // Not `role: 'reload'`: that hard-reloads the RENDERER (every pane, the
      // whole shell) and a focused in-app browser needs ⌘R to mean "reload
      // this page", the way it does in every other browser. ⇧⌘R
      // (`forceReload`) below stays the unconditional escape hatch.
      //
      // No accelerator: ⌘R is claimed in `installPreviewShortcut`, which works
      // on every platform (this menu exists only on macOS). Declaring it here
      // too would fire the item and the input hook for one keypress.
      { click: () => sendPreviewNavCommand('reload'), label: 'Reload' },
      { role: 'forceReload' },
      {
        label: 'Toggle Developer Tools',
        accelerator: process.platform === 'darwin' ? 'Alt+Cmd+I' : 'Ctrl+Shift+I',
        click: (_menuItem, browserWindow) => toggleDevTools(browserWindow || mainWindow)
      },
      { type: 'separator' },
      {
        label: 'Actual Size',
        accelerator: 'CommandOrControl+0',
        click: () => {
          setAndPersistZoomLevel(mainWindow, DEFAULT_ZOOM_LEVEL)
        }
      },
      {
        label: 'Zoom In',
        accelerator: 'CommandOrControl+Plus',
        click: () => {
          if (mainWindow && !mainWindow.isDestroyed()) {
            setAndPersistZoomLevel(mainWindow, mainWindow.webContents.getZoomLevel() + ZOOM_STEP)
          }
        }
      },
      {
        label: 'Zoom Out',
        accelerator: 'CommandOrControl+-',
        click: () => {
          if (mainWindow && !mainWindow.isDestroyed()) {
            setAndPersistZoomLevel(mainWindow, mainWindow.webContents.getZoomLevel() - ZOOM_STEP)
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
  // Only Ctrl+Shift+I (or Cmd+Opt+I on Mac) opens DevTools.
  // F12 is explicitly blocked so Chromium's built-in handler doesn't open it.
  window.webContents.on('before-input-event', (event, input) => {
    const key = input.key.toLowerCase()

    // F12 opens DevTools by default; block only when the user disabled it.
    if (input.key === 'F12') {
      if (f12Blocked) {
        event.preventDefault()

        return
      }
      // Not blocked — fall through to open DevTools.
    }

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
    const accel = (IS_MAC ? input.meta : input.control) && !input.alt
    const isCloseTabShortcut = key === 'w' && accel && !input.shift

    // Always claim ⌘W here (the File>Close item deliberately has no
    // accelerator, so nothing else does). The renderer decides tab-vs-window
    // — no `previewShortcutActive` gate, so it works for every closeable tab.
    if (isCloseTabShortcut) {
      event.preventDefault()
      sendClosePreviewRequested()

      return
    }

    // ⌘R rides here rather than on the View menu item for the same reason:
    // the application menu only exists on macOS (it is set to null elsewhere,
    // see #77845), so a menu accelerator would leave Windows and Linux with no
    // way to reload a page at all. ⇧⌘R is left alone — that is `forceReload`,
    // the unconditional whole-window escape hatch.
    if (key === 'r' && accel && !input.shift) {
      event.preventDefault()
      sendPreviewNavCommand('reload')
    }
  })
}

// Zoom level is persisted in the renderer's own localStorage (per-origin,
// survives reloads/restarts) rather than a main-process JSON file. The main
// process owns setZoomLevel, so we mirror each change into localStorage and
// read it back on did-finish-load to re-apply after reloads or crash recovery.
import {
  applyZoomLevel,
  DEFAULT_ZOOM_LEVEL,
  installZoomReassertOnNavigation,
  installZoomReassertOnWindowEvents,
  percentToZoomLevel,
  ZOOM_STEP,
  ZOOM_STORAGE_KEY,
  zoomLevelToPercent,
  zoomWiringForWindowKind
} from './zoom'

function setAndPersistZoomLevel(window, zoomLevel) {
  if (!window || window.isDestroyed()) {
    return
  }

  // Apply + notify in one funnel so the settings UI stays in sync, including
  // changes made via the keyboard shortcuts or the View menu.
  const next = applyZoomLevel(window.webContents, zoomLevel)

  // Primary store: main-process JSON (survives crash recovery — #56726).
  writeZoomState(next)
  // Secondary mirror: renderer localStorage (legacy store; kept in sync so a
  // downgrade or JSON read failure still finds a sane value).
  window.webContents
    .executeJavaScript(
      `try { localStorage.setItem(${JSON.stringify(ZOOM_STORAGE_KEY)}, ${JSON.stringify(String(next))}) } catch {
      void 0
    }`
    )
    .catch(error => rememberLog(`[zoom] persist failed: ${error?.message || error}`))
}

function restorePersistedZoomLevel(window) {
  if (!window || window.isDestroyed()) {
    return
  }

  // Prefer the JSON file — it survives crash recovery wiping Electron's
  // cache/storage folders (#56726). applyZoomLevel notifies the renderer so
  // the Appearance UI Scale control stays in sync.
  const saved = readZoomState()

  if (saved != null) {
    // Drift-guard: skip when this window already shows the persisted level.
    // Blindly re-applying on every resize/move would race the compositor's
    // surface reconfigure during a Wayland resize storm (Cosmic tiled mode
    // fires one whenever a new session window opens — #84818) and keep the
    // renderer notification stream churning for no gain. The settle-verify
    // chain in installZoomReassertOnWindowEvents re-applies only when the
    // window actually drifted from the persisted level.
    const current = window.webContents?.getZoomLevel?.()

    if (current != null && Math.abs(current - saved) < 1e-9) {
      return
    }

    applyZoomLevel(window.webContents, saved)

    return
  }

  // No JSON yet: paint the shipped default immediately so a fresh install
  // doesn't flash Chromium 100%, then try localStorage for pre-JSON installs
  // and overwrite if a legacy value is there.
  applyZoomLevel(window.webContents, DEFAULT_ZOOM_LEVEL)

  window.webContents
    .executeJavaScript(
      `(() => { try { return localStorage.getItem(${JSON.stringify(ZOOM_STORAGE_KEY)}) } catch { return null } })()`
    )
    .then(stored => {
      if (!window || window.isDestroyed()) {
        return
      }

      const level = stored == null ? DEFAULT_ZOOM_LEVEL : Number(stored)
      const applied = applyZoomLevel(window.webContents, level)
      writeZoomState(applied)
    })
    .catch(error => rememberLog(`[zoom] restore failed: ${error?.message || error}`))
}

function installZoomShortcuts(window) {
  // Override Ctrl/Cmd + +/-/0 with half Chromium's default zoom step (ZOOM_STEP
  // is 0.1 vs Chromium's 0.2). The menu items handle this on macOS (where the
  // menu is always present), but on Linux/Windows the menu is null and
  // Chromium's default handler would use the full 0.2 step, so we intercept
  // here for consistency. Ctrl/Cmd+0 resets to DEFAULT_ZOOM_LEVEL, not Chromium 0.
  window.webContents.on('before-input-event', (event, input) => {
    const mod = IS_MAC ? input.meta : input.control

    if (!mod || input.alt) {
      return
    }

    const key = input.key

    if (key === '0') {
      if (input.shift) {
        return // Ctrl/Cmd+Shift+0 is not a zoom chord — leave it alone
      }

      event.preventDefault()
      setAndPersistZoomLevel(window, DEFAULT_ZOOM_LEVEL)
    } else if (key === '=' || key === '+') {
      // Zoom-in must accept the shift modifier: on US layouts Plus is
      // physically Shift+=, so Cmd+Plus arrives as Cmd+Shift+'+' (or '='
      // depending on platform). The old blanket shift guard silently
      // dropped keyboard zoom-in on macOS (#43517).
      event.preventDefault()
      setAndPersistZoomLevel(window, window.webContents.getZoomLevel() + ZOOM_STEP)
    } else if (key === '-') {
      if (input.shift) {
        return // Shift+'-' is '_' territory on most layouts, not zoom-out
      }

      event.preventDefault()
      setAndPersistZoomLevel(window, window.webContents.getZoomLevel() - ZOOM_STEP)
    }
  })

  // Ctrl/Cmd + mouse wheel — the standard desktop/browser zoom gesture
  // (#40295). Chromium surfaces it as the main-process 'zoom-changed' event
  // (wheel events are DOM-side, so before-input-event never sees them).
  // Route through the same persist+notify funnel as the keyboard shortcuts
  // so wheel zoom survives restarts and the settings Scale control stays in
  // sync, and use the same half step for consistency.
  window.webContents.on('zoom-changed', (event, zoomDirection) => {
    event.preventDefault()
    const delta = zoomDirection === 'in' ? ZOOM_STEP : -ZOOM_STEP
    setAndPersistZoomLevel(window, window.webContents.getZoomLevel() + delta)
  })
}

/**
 * The custom (renderer) context menu's main-process half.
 *
 * The app popups no native menus: the renderer owns the menu UI so labels
 * are translated with the rest of the app. Main keeps only what Chromium
 * reports here and the renderer cannot see:
 *  - spell-check facts (misspelled word + suggestions) — forwarded so the
 *    renderer appends them to its already-open menu,
 *  - the gesture coordinates — kept for copyImageAt, which needs them.
 */
const lastContextMenuPoint = new Map<number, { x: number; y: number }>()

function installContextMenuBridge(window: BrowserWindow) {
  window.webContents.on('context-menu', (_event, params) => {
    lastContextMenuPoint.set(window.webContents.id, { x: params.x, y: params.y })

    const suggestions = Array.isArray(params.dictionarySuggestions) ? params.dictionarySuggestions : []

    if (params.isEditable && params.misspelledWord) {
      window.webContents.send('hermes:context-menu-spellcheck', {
        misspelledWord: params.misspelledWord,
        suggestions
      })
    }
  })
}

// Microphone and camera capture. The voice composer drives mic access and
// renderer features (e.g. desktop plugins) can drive camera access, both
// through getUserMedia, which Chromium gates behind these two session hooks.
//
// The naive `details.mediaTypes.includes('audio')` check works on macOS but
// breaks on Windows: Chromium frequently fires the request with an empty or
// undefined `mediaTypes`, so a strict check denies it and getUserMedia throws
// NotAllowedError. We therefore allow the capture permissions and treat absent
// metadata as allowed.
//
// Granting here is not the last gate: the OS still applies its own capture
// permission (macOS TCC prompts on first use, per the NSMicrophone/NSCamera
// usage strings), so the user keeps a real allow/deny and can revoke it in
// System Settings afterwards.
function isMediaCapturePermission(permission, details) {
  if (permission === 'audioCapture' || permission === 'videoCapture') {
    return true
  }

  if (permission !== 'media') {
    return false
  }

  const mediaTypes = details?.mediaTypes

  // Windows: mediaTypes is often empty for a capture request. Don't deny on
  // missing metadata.
  if (!Array.isArray(mediaTypes) || mediaTypes.length === 0) {
    return true
  }

  return mediaTypes.includes('audio') || mediaTypes.includes('video')
}

// Chromium-initiated downloads (renderer anchor/blob downloads, drag-outs)
// land here. Without a handler the OS save dialog opens with the process cwd
// as the default directory (win-unpacked in packaged installs) and whatever
// extensionless name the anchor carried. Route every download to the user's
// Downloads directory and guarantee a MIME-derived extension.
function installDownloadHandling() {
  session.defaultSession.on('will-download', (_event, item) => {
    const suggested = item.getFilename() || 'download'
    const hasExtension = Boolean(path.extname(suggested))
    const extension = hasExtension ? '' : extensionForMimeType(item.getMimeType())
    const filename = `${suggested}${extension}`

    try {
      item.setSaveDialogOptions({
        title: 'Save File',
        defaultPath: path.join(app.getPath('downloads'), filename),
        filters:
          extension || /^image\//i.test(item.getMimeType() || '')
            ? [
                { name: 'Images', extensions: ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'svg'] },
                { name: 'All Files', extensions: ['*'] }
              ]
            : undefined
      })
    } catch {
      // No Downloads directory to offer — keep Chromium's default prompt.
    }
  })
}

function installMediaPermissions() {
  // Async request handler: the prompt-style path (most platforms).
  session.defaultSession.setPermissionRequestHandler((_webContents, permission, callback, details) => {
    callback(isMediaCapturePermission(permission, details))
  })

  // Synchronous check handler: Chromium consults this for getUserMedia on
  // Windows in addition to (or instead of) the request handler. Without it,
  // the check defaults to false and capture is denied before the request
  // handler ever runs.
  session.defaultSession.setPermissionCheckHandler((_webContents, permission) => {
    return (
      permission === 'media' ||
      permission === ('audioCapture' as any) /* todo: is this needed? */ ||
      permission === ('videoCapture' as any)
    )
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

// Cold-start cookie-jar warm-up. A `persist:` partition materialized via
// session.fromPartition() loads its on-disk cookie store LAZILY: the very first
// cookies.get() on a fresh cold start can resolve BEFORE the jar has finished
// hydrating from disk and return an empty array — even though the user is
// signed in. That false-negative used to make hasLiveOauthSession() report
// "not signed in", which on the initial boot path (startHermes → the renderer's
// single-shot boot() with no retry) surfaced as the "Hermes couldn't start"
// OAuth overlay that vanishes the instant the user clicks Retry.
//
// We force the store to hydrate once, up front: flushStorageData() then a
// throwaway cookies.get(). The promise is memoized so every caller awaits the
// same single warm-up. Best-effort — any error resolves so we fall back to the
// live read (which then does its own bounded re-check).
let oauthCookieWarmup: Promise<void> | null = null

function warmOauthCookieStore() {
  if (oauthCookieWarmup) {
    return oauthCookieWarmup
  }

  oauthCookieWarmup = (async () => {
    const sess = getOauthSession()

    if (!sess) {
      // App not ready yet — don't memoize a no-op; let a later call retry.
      oauthCookieWarmup = null

      return
    }

    try {
      // flushStorageData() forces Chromium to reconcile the in-memory cookie
      // monster with the on-disk SQLite store; the subsequent get() then reads
      // a populated jar rather than racing the lazy first-access load.
      sess.flushStorageData?.()
      await sess.cookies.get({})
    } catch {
      // Best effort; the real read below re-checks with bounded retries.
    }
  })()

  return oauthCookieWarmup
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

  const readLive = async () => {
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

  // First read against the (possibly still-hydrating) jar.
  if (await readLive()) {
    return true
  }

  // Cold-start false-negative guard. A `persist:` partition's cookie store
  // loads lazily, so the FIRST read on a fresh boot can come back empty even
  // for a signed-in user — the exact race that produced the transient "Hermes
  // couldn't start / not signed in" overlay that Retry always cleared. Before
  // trusting a negative, force the store to hydrate and re-read a couple of
  // times with a short backoff. A genuinely signed-out user still resolves
  // false quickly (≤ ~180ms); a signed-in user racing the load now wins.
  await warmOauthCookieStore()

  for (const delayMs of [30, 60, 90]) {
    if (await readLive()) {
      return true
    }

    await new Promise(resolve => setTimeout(resolve, delayMs))
  }

  return readLive()
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
    // Log-only lifecycle diagnostics: a crashed sign-in renderer is invisible
    // to the window's promise path (it never settles), so without this the
    // failure leaves no trace in desktop.log (#81290 follow-up).
    installWindowRendererLifecycle(win, { kind: 'oauth', callbacks: { log: rememberLog } })
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

    for (const [name, value] of Object.entries({ ...headersForRemoteRequest(url), ...(options.headers || {}) })) {
      request.setHeader(name, String(value))
    }

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

// ---------------------------------------------------------------------------
// RFC 8252 native-app tokens (system-browser + loopback + PKCE).
//
// Unlike the cookie flow, the native flow hands the desktop opaque bearer
// tokens it holds itself: the access token authenticates REST via
// ``Authorization: Bearer`` (which the gateway gate now accepts) and mints WS
// tickets the same way, so NO browser session cookie or embedded webview is
// involved. Tokens are persisted encrypted at rest via Electron ``safeStorage``
// (OS keychain) keyed by gateway base URL, and refreshed via
// ``/auth/native/refresh`` before expiry. This is the desktop half of the
// feature; the server half lives in hermes_cli/dashboard_auth/native_flow.py.
// ---------------------------------------------------------------------------

// In-memory cache of decrypted native tokens, keyed by normalized base URL.
// Backed by the encrypted on-disk store so it survives restarts.
const _nativeTokens = new Map<string, NativeTokenSet>()

function _nativeTokenStorePath() {
  // Co-located with the connection config under userData; one JSON file mapping
  // baseUrl → { encoding, value } safeStorage payloads.
  return path.join(app.getPath('userData'), 'native-oauth-tokens.json')
}

// The electron-coupled half of the token store: safeStorage encryption plus the
// userData file. native-token-store.ts owns the serialization/parse round trip
// so it can be tested without an Electron runtime.
function _nativeTokenStoreIo(): NativeTokenStoreIo {
  return {
    encrypt: encryptDesktopSecret,
    decrypt: decryptDesktopSecret,
    readStoreText: () => fs.readFileSync(_nativeTokenStorePath(), 'utf8'),
    writeStoreText: (text: string) => {
      fs.mkdirSync(path.dirname(_nativeTokenStorePath()), { recursive: true })
      fs.writeFileSync(_nativeTokenStorePath(), text, { mode: 0o600 })
    },
    rememberLog
  }
}

function _persistNativeTokens(baseUrl: string, tokens: NativeTokenSet | null) {
  persistNativeTokenSet(baseUrl, tokens, _nativeTokenStoreIo())
}

function _loadNativeTokens(baseUrl: string): NativeTokenSet | null {
  const cached = _nativeTokens.get(baseUrl)

  if (cached) {
    return cached
  }

  const tokens = loadNativeTokenSet(baseUrl, _nativeTokenStoreIo())

  if (tokens) {
    _nativeTokens.set(baseUrl, tokens)
  }

  return tokens
}

function _storeNativeTokens(baseUrl: string, tokens: NativeTokenSet) {
  _nativeTokens.set(baseUrl, tokens)
  _persistNativeTokens(baseUrl, tokens)
}

function _clearNativeTokens(baseUrl: string) {
  _nativeTokens.delete(baseUrl)
  _persistNativeTokens(baseUrl, null)
}

// True when we hold native bearer tokens for this gateway (the native-flow
// analogue of hasLiveOauthSession's cookie check).
function hasNativeSession(baseUrl: string): boolean {
  return _loadNativeTokens(baseUrl) !== null
}

// POST JSON WITHOUT the OAuth cookie partition — used for the native token +
// refresh exchanges, which are cookieless by design. Thin wrapper over
// fetchJson (no token) so it shares timeout/JSON handling.
function postJsonNoAuth(url: string, body: unknown, opts: any = {}) {
  // resolveJsonBody passes the object through UNCHANGED — fetchJson owns
  // JSON.stringify. Pre-stringifying here double-encodes the body (a JSON
  // string inside a JSON string), which the gateway's Pydantic model rejects
  // with a 422 "Input should be a valid dictionary" (the native
  // /auth/native/token + /auth/native/refresh legs both go through here).
  return fetchJson(url, null, { method: 'POST', body: resolveJsonBody(body), ...opts })
}

// Return a valid native access token for baseUrl, refreshing via
// /auth/native/refresh if the stored one is at/near expiry. Returns null when
// there are no tokens or the refresh is terminally rejected (caller re-logins).
async function ensureNativeAccessToken(baseUrl: string): Promise<string | null> {
  const tokens = _loadNativeTokens(baseUrl)

  if (!tokens) {
    return null
  }

  if (!tokenNeedsRefresh(tokens, Math.floor(Date.now() / 1000))) {
    return tokens.accessToken
  }

  if (!tokens.refreshToken) {
    // Access token expired and no RT to rotate — force re-login.
    _clearNativeTokens(baseUrl)

    return null
  }

  try {
    const body = await postJsonNoAuth(
      nativeRefreshUrl(baseUrl),
      { refresh_token: tokens.refreshToken, provider: tokens.provider },
      { timeoutMs: 10_000 }
    )

    const rotated = parseTokenResponse(body)
    _storeNativeTokens(baseUrl, rotated)

    return rotated.accessToken
  } catch (error: any) {
    // A 401 means the RT is dead (session_expired) — drop tokens so the UI
    // prompts a fresh native login. A 503/transient keeps them for a retry.
    if (error && error.statusCode === 401) {
      _clearNativeTokens(baseUrl)

      return null
    }

    throw error
  }
}

// OAuth-session download that streams the response body straight to a
// user-selected destination (via finalizeGatewayDownload). The connect timeout
// is cleared once the response headers arrive.
function downloadViaOauthSessionToFile(url, ctx, options: any = {}) {
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

    const timeoutMs = resolveTimeoutMs(options.timeoutMs, DEFAULT_FETCH_TIMEOUT_MS)

    const request = electronNet.request({
      method: 'GET',
      url,
      session: sess,
      useSessionCookies: true,
      redirect: 'follow'
    } as any)

    let settled = false

    const timer = setTimeout(() => {
      if (settled) {
        return
      }

      settled = true

      try {
        request.abort()
      } catch {
        // already finished
      }

      reject(new Error(`Timed out connecting to Hermes backend after ${timeoutMs}ms`))
    }, timeoutMs)

    request.on('response', res => {
      if (settled) {
        return
      }

      // Response headers arrived — cancel the connect timeout so it can't abort
      // the stream while the save dialog is open or bytes are still flowing.
      settled = true
      clearTimeout(timer)
      finalizeGatewayDownload(res, res.statusCode || 500, res.headers || {}, {
        ...ctx,
        abort: () => {
          try {
            request.abort()
          } catch {
            // already finished
          }
        }
      }).then(resolve, reject)
    })
    request.on('error', error => {
      if (settled) {
        return
      }

      settled = true
      clearTimeout(timer)
      reject(error)
    })
    request.end()
  })
}

// Shared tail for both transports: validate status, pick a filename, prompt the
// save dialog, then stream the (still-unconsumed) response body to the chosen
// destination. On an HTTP error the status code is attached so saveGatewayFile
// can trigger the 404-only compatibility fallback.
async function finalizeGatewayDownload(res, statusCode, headers, ctx: any = {}) {
  if (statusCode >= 400) {
    const message = await readGatewayErrorText(res)
    const error: any = new Error(`${statusCode}: ${message}`)
    error.statusCode = statusCode
    throw error
  }

  const disposition = headers['content-disposition'] || headers['Content-Disposition']
  const filename = filenameFromContentDisposition(disposition) || ctx.suggested || ctx.fallbackName

  const result = await dialog.showSaveDialog(mainWindow, {
    defaultPath: filename,
    title: 'Save File'
  })

  if (result.canceled || !result.filePath) {
    ctx.abort?.()

    return { canceled: true, saved: false }
  }

  try {
    await pumpStreamToFile(res, result.filePath, {
      createWriteStream: (destPath: string) => fs.createWriteStream(destPath),
      unlink: (destPath: string) => fs.promises.unlink(destPath)
    })
  } catch (error) {
    ctx.abort?.()
    throw error
  }

  return { path: result.filePath, saved: true }
}

// Read a bounded amount of an error response body for the thrown message.
function readGatewayErrorText(res): Promise<string> {
  return new Promise(resolve => {
    const chunks = []
    let total = 0

    res.on('data', chunk => {
      if (total >= 500) {
        return
      }

      const buffer = Buffer.from(chunk)

      total += buffer.length
      chunks.push(buffer)
    })
    res.on('end', () => resolve(Buffer.concat(chunks).toString('utf8').slice(0, 500)))
    res.on('error', () => resolve(Buffer.concat(chunks).toString('utf8').slice(0, 500)))
  })
}

interface GatewayFileConnection extends RegistryBackendRequestScope {
  authMode?: 'oauth' | 'token'
  baseUrl: string
  token?: null | string
}

interface GatewayFileSaveContext {
  fallbackName: string
  suggested: string
}

interface GatewayFileSavePayload {
  connectionId?: unknown
  path?: unknown
  profile?: unknown
  suggestedName?: unknown
}

async function gatedFileAuth(connection: GatewayFileConnection) {
  const nativeAt =
    connection.authMode === 'oauth' ? await ensureNativeAccessToken(connection.baseUrl).catch(() => null) : null

  return resolveGatedDownloadAuth(connection.authMode, nativeAt, connection.token)
}

function gatewayFileRequestPath(
  connection: GatewayFileConnection,
  connectionId: null | string,
  profile: null | string,
  requestPath: string
) {
  return connectionId
    ? pathForRegistryBackendRequest(requestPath, profile, connection)
    : pathWithGlobalRemoteProfile(requestPath, profile, profileRouteOptions(profile))
}

async function saveGatewayFile(payload: GatewayFileSavePayload = {}) {
  const filePath = gatewayFilePath(payload.path)

  if (!filePath) {
    throw new Error('Missing gateway file path')
  }

  const { connection, connectionId, profile } = await resolveGatewayFileBackend<GatewayFileConnection>(payload, {
    ensureLegacy: ensureBackend,
    ensureRegistry: ensureRegistryBackend
  })

  const suggested = String(payload.suggestedName || '').trim()
  const fallbackName = path.basename(filePath) || suggested || 'download'
  const ctx = { suggested, fallbackName }

  const requestPaths = gatewayFileRequestPaths(filePath, requestPath =>
    gatewayFileRequestPath(connection, connectionId, profile, requestPath)
  )

  const url = `${connection.baseUrl}${requestPaths.download}`

  try {
    const auth = await gatedFileAuth(connection)

    if (auth.kind === 'bearer') {
      return await downloadViaTokenToFile(url, auth.token, ctx, { bearer: auth.token })
    }

    if (auth.kind === 'cookie') {
      return await downloadViaOauthSessionToFile(url, ctx)
    }

    return await downloadViaTokenToFile(url, auth.token, ctx)
  } catch (error) {
    // Desktop and the remote gateway update independently. A gateway predating
    // /api/fs/download 404s here; fall back (ONLY on 404) to the older capped
    // data-URL route so downloads keep working against older backends.
    if (isNotFoundError(error)) {
      return await saveGatewayFileViaDataUrl(connection, requestPaths.dataUrl, ctx)
    }

    throw error
  }
}

// Compatibility fallback: fetch the file through the capped
// `/api/fs/read-data-url` route, decode it, and save. Bounded by the gateway's
// data-URL cap, so it only serves smaller files — enough to keep older gateways
// working until they gain the streaming route.
async function saveGatewayFileViaDataUrl(
  connection: GatewayFileConnection,
  requestPath: string,
  ctx: GatewayFileSaveContext
) {
  const url = `${connection.baseUrl}${requestPath}`
  const auth = await gatedFileAuth(connection)
  let json: unknown

  if (auth.kind === 'bearer') {
    json = await fetchJson(url, null, { bearer: auth.token })
  } else if (auth.kind === 'cookie') {
    json = await fetchJsonViaOauthSession(url)
  } else {
    json = await fetchJson(url, auth.token)
  }

  const dataUrl =
    json && typeof json === 'object' && 'dataUrl' in json && typeof json.dataUrl === 'string' ? json.dataUrl : ''

  if (!dataUrl) {
    throw new Error('Gateway returned no file data')
  }

  const buffer = parseDataUrlToBuffer(dataUrl)
  const filename = ctx.suggested || ctx.fallbackName

  const result = await dialog.showSaveDialog(mainWindow, {
    defaultPath: filename,
    title: 'Save File'
  })

  if (result.canceled || !result.filePath) {
    return { canceled: true, saved: false }
  }

  await fs.promises.writeFile(result.filePath, buffer)

  return { path: result.filePath, saved: true }
}

// Mint a single-use WS ticket for a gated gateway. Returns the ticket string.
// Prefers a native bearer token (cookieless RFC 8252 flow) when present,
// falling back to the OAuth cookie partition otherwise.
// Throws (with statusCode 401) if the session cookie is missing/expired —
// callers treat that as "needs re-login".
// Transient transport blips (brief host unreachable, 5xx, timeouts) are retried
// a few times before failing — those 1-3s flaps were promoting into the
// full-screen "couldn't start" lockout on reconnect.
async function mintGatewayWsTicket(baseUrl, headers = {}) {
  return withTransientRetries(async () => {
    // Native flow: mint the ticket with the bearer token, no cookie involved.
    const nativeAt = await ensureNativeAccessToken(baseUrl).catch(() => null)

    if (nativeAt) {
      const body = (await fetchJson(`${baseUrl}/api/auth/ws-ticket`, null, {
        method: 'POST',
        timeoutMs: 8_000,
        bearer: nativeAt,
        headers
      })) as any

      const ticket = body?.ticket

      if (!ticket || typeof ticket !== 'string') {
        throw new Error('Gateway did not return a WS ticket.')
      }

      return ticket
    }

    const body = (await fetchJsonViaOauthSession(`${baseUrl}/api/auth/ws-ticket`, {
      method: 'POST',
      timeoutMs: 8_000,
      headers
    })) as any

    const ticket = body?.ticket

    if (!ticket || typeof ticket !== 'string') {
      throw new Error('Gateway did not return a WS ticket.')
    }

    return ticket
  })
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
    const ticket = await mintGatewayWsTicket(connection.baseUrl, connection.headers)
    const wsUrl = buildGatewayWsUrlWithTicket(connection.baseUrl, ticket)

    rememberRemoteWsHeaders(wsUrl, connection.headers)

    return wsUrl
  }

  // Local/token: the cached wsUrl already carries the (long-lived) token.
  rememberRemoteWsHeaders(connection.wsUrl, connection.headers)

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
//
// Mirrors hasLiveOauthSession's cold-start guard (#73495): a `persist:`
// partition's cookie store hydrates lazily, so the FIRST read on a fresh boot
// can come back empty even for a signed-in user. The renderer checks Cloud
// status exactly once on entering cloud mode, so a single false-negative here
// used to clear the discovered agent list and demand a re-login that a plain
// retry would have avoided. Warm the store and re-read with a short backoff
// before trusting a negative.
async function hasLivePortalSession() {
  const sess = getOauthSession()

  if (!sess) {
    return false
  }

  const portalBaseUrl = resolvePortalBaseUrl()
  const parsed = new URL(portalBaseUrl)

  const readPortal = async () => {
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

  if (await readPortal()) {
    return true
  }

  await warmOauthCookieStore()

  for (const delayMs of [30, 60, 90]) {
    if (await readPortal()) {
      return true
    }

    await new Promise(resolve => setTimeout(resolve, delayMs))
  }

  return readPortal()
}

// Whether the jar holds the short-lived Privy ACCESS token — the exact cookie
// `/api/agents` validates. hasLivePortalSession() answers "signed in at all?"
// (renewal material counts); this answers "can discovery succeed right now?".
async function hasPortalAccessToken() {
  const sess = getOauthSession()

  if (!sess) {
    return false
  }

  const portalBaseUrl = resolvePortalBaseUrl()
  const parsed = new URL(portalBaseUrl)

  try {
    const cookies = await sess.cookies.get({ url: portalBaseUrl })

    return cookiesHavePrivyAccessToken(cookies)
  } catch {
    try {
      const cookies = await sess.cookies.get({ domain: parsed.hostname })

      return cookiesHavePrivyAccessToken(cookies)
    } catch {
      return false
    }
  }
}

// Bounded silent renewal of the short-lived Privy access token (#73495).
//
// After a Desktop restart the long-lived `privy-session` / `privy-refresh-token`
// cookies routinely survive while the ~1h `privy-token` access cookie has
// expired. Discovery then 401s and the only offered recovery used to be a full
// interactive re-login — even though the persisted refresh material can mint a
// fresh access token with no user action: loading any portal page runs the
// Privy client, which rotates a new `privy-token` from the refresh session.
//
// This drives exactly that, headlessly: a hidden window on the portal root in
// the OAuth partition, polled until the access cookie lands, torn down on a
// bounded timeout. Never shown — if renewal can't complete silently the caller
// falls back to the interactive needsCloudLogin path. The in-flight promise is
// shared so concurrent discovery + cascade calls ride one renewal.
let portalAccessRenewal: Promise<boolean> | null = null

function renewPortalAccessSilently() {
  if (portalAccessRenewal) {
    return portalAccessRenewal
  }

  portalAccessRenewal = (async () => {
    if (!app.isReady()) {
      return false
    }

    const sess = getOauthSession()

    if (!sess) {
      return false
    }

    // No renewal material at all → nothing to renew; interactive login is
    // genuinely required.
    if (!(await hasLivePortalSession())) {
      return false
    }

    if (await hasPortalAccessToken()) {
      return true
    }

    const portalBaseUrl = resolvePortalBaseUrl()

    return await new Promise<boolean>(resolve => {
      let settled = false
      let win = null
      let pollTimer = null
      let deadlineTimer = null

      const finish = (ok: boolean) => {
        if (settled) {
          return
        }

        settled = true

        if (pollTimer) {
          clearInterval(pollTimer)
        }

        if (deadlineTimer) {
          clearTimeout(deadlineTimer)
        }

        try {
          if (win && !win.isDestroyed()) {
            win.destroy()
          }
        } catch {
          // window already torn down
        }

        rememberLog(`[cloud] silent portal access renewal ${ok ? 'succeeded' : 'did not complete'}`)
        resolve(ok)
      }

      const checkCookie = async () => {
        if (settled) {
          return
        }

        if (await hasPortalAccessToken()) {
          finish(true)
        }
      }

      try {
        win = new BrowserWindow({
          width: 520,
          height: 720,
          show: false,
          title: 'Renewing Hermes Cloud session…',
          autoHideMenuBar: true,
          webPreferences: {
            contextIsolation: true,
            nodeIntegration: false,
            sandbox: true,
            session: sess,
            webSecurity: true
          }
        })
      } catch {
        finish(false)

        return
      }

      win.webContents.on('did-navigate', () => void checkCookie())
      win.webContents.on('did-redirect-navigation', () => void checkCookie())
      win.webContents.on('did-frame-navigate', () => void checkCookie())
      installWindowRendererLifecycle(win, { kind: 'portal-renew', callbacks: { log: rememberLog } })
      pollTimer = setInterval(() => void checkCookie(), 500)
      // Hard deadline: this window is never revealed, so an unrenewable session
      // (revoked refresh token, portal down) must resolve false rather than
      // hang the discovery call behind an invisible window.
      deadlineTimer = setTimeout(() => finish(false), 12_000)

      win.on('closed', () => finish(false))

      win.loadURL(portalBaseUrl).catch(() => finish(false))
    })
  })().finally(() => {
    portalAccessRenewal = null
  }) as Promise<boolean>

  return portalAccessRenewal
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
    // Log-only lifecycle diagnostics, same rationale as the OAuth window:
    // a crashed portal sign-in renderer never settles the promise, so the
    // failure would otherwise leave no trace in desktop.log (#81290
    // follow-up).
    installWindowRendererLifecycle(win, { kind: 'portal', callbacks: { log: rememberLog } })
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

  // Renewable session present but the short-lived access token `/api/agents`
  // validates is gone (typical after a restart — `privy-token` is ~1h,
  // `privy-session`/`privy-refresh-token` last ~30 days). Renew silently up
  // front instead of letting the request 401 into a re-login demand (#73495).
  if (!(await hasPortalAccessToken())) {
    await renewPortalAccessSilently()
  }

  const orgQuery = org ? `?org=${encodeURIComponent(org)}` : ''
  let body

  const fetchAgents = () =>
    fetchJsonViaOauthSession(`${portalBaseUrl}/api/agents${orgQuery}`, {
      method: 'GET',
      timeoutMs: 15_000
    })

  try {
    body = (await fetchAgents()) as any
  } catch (initialError) {
    let error = initialError as any

    // A 401 with renewal material still in the jar: attempt ONE bounded silent
    // renewal and retry, so a lapsed access token doesn't surface as a full
    // interactive re-login while a 30-day refresh session sits unused. Only a
    // rejected/failed renewal (or a second 401 on genuinely fresh access)
    // falls through to needsCloudLogin.
    if (error && error.statusCode === 401 && (await renewPortalAccessSilently())) {
      try {
        body = (await fetchAgents()) as any
      } catch (retryError) {
        error = retryError
      }
    }

    if (body === undefined) {
      // A 401 means the portal session lapsed (and silent renewal could not
      // recover it) — surface it as a re-login, not a generic failure.
      if (error && error.statusCode === 401) {
        const err = new Error(
          'Your Hermes Cloud session has expired. Open Settings → Gateway and sign in again.'
        ) as any

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

  // The cascade rides the portal's auto-approve, which needs the short-lived
  // access state just like discovery. If only renewal material survived the
  // restart, mint a fresh access token first so the hidden cascade window
  // auto-SSOs instead of stalling on an interactive chooser (#73495).
  if (!(await hasPortalAccessToken())) {
    await renewPortalAccessSilently()
  }

  await openOauthLoginWindow(baseUrl, { silent: true })

  return { baseUrl, connected: await hasOauthSessionCookie(baseUrl) }
}

// ---------------------------------------------------------------------------
// Opt-in keychain encryption (secret-storage-policy.ts owns the decision).
// Default OFF: no safeStorage call is ever made, so a broken/locked macOS
// login keychain can never throw its password dialog on launch. Settings →
// Gateway exposes the toggle; flipping it re-encrypts (or decrypts) the
// stored secrets in place.
// ---------------------------------------------------------------------------
const SECRET_STORAGE_POLICY_PATH = path.join(app.getPath('userData'), SECRET_STORAGE_POLICY_FILE)

const _secretStoragePolicyIo = {
  readText: () => fs.readFileSync(SECRET_STORAGE_POLICY_PATH, 'utf8'),
  writeText: (text: string) => writeSecretFileAtomic(SECRET_STORAGE_POLICY_PATH, text, { encoding: 'utf8' })
}

let _secretStoragePolicy: SecretStoragePolicy | null = null

function secretStoragePolicy(): SecretStoragePolicy {
  if (!_secretStoragePolicy) {
    _secretStoragePolicy = readSecretStoragePolicy(_secretStoragePolicyIo)
  }

  return _secretStoragePolicy
}

function setSecretStoragePolicy(next: SecretStoragePolicy) {
  _secretStoragePolicy = { on: next.on === true, migrated: next.migrated === true }
  writeSecretStoragePolicy(_secretStoragePolicy, _secretStoragePolicyIo)
}

/**
 * Keychain availability as the renderer should see it. With encryption
 * opted out this must NOT probe safeStorage — isEncryptionAvailable() is
 * itself a keychain touch that raises the macOS dialog this feature exists
 * to avoid. We report `true` so no plain-text warning banners fire: storing
 * plaintext is the user's chosen (default) mode, not a degraded state.
 */
function probeSecureTokenStorage(): boolean {
  if (!secretStoragePolicy().on) {
    return true
  }

  try {
    return Boolean(safeStorage.isEncryptionAvailable())
  } catch {
    return false
  }
}

/**
 * Rewrite every stored desktop secret (v1 connection.json token/headers +
 * per-profile overrides, v2 registry connections, native OAuth token store)
 * through `reencode`. Returns true when any store was rewritten. Shared by
 * the one-shot legacy migration and the Settings encryption toggle.
 */
function rewriteAllStoredSecrets(shouldRewrite: (secret: any) => boolean, reencode: (secret: any) => any): boolean {
  let touched = false

  const rewriteBlock = (block: any) => {
    if (!block || typeof block !== 'object') {
      return block
    }

    const next = { ...block, ...(block.token ? { token: reencode(block.token) } : {}) }

    if (block.headers && typeof block.headers === 'object') {
      next.headers = Object.fromEntries(Object.entries(block.headers).map(([k, v]) => [k, reencode(v)]))
    }

    return next
  }

  const blockNeedsRewrite = (o: any) =>
    shouldRewrite(o?.token) ||
    Object.values(o?.headers && typeof o.headers === 'object' ? o.headers : {}).some(shouldRewrite)

  // v1 connection.json.
  const config = readDesktopConnectionConfig()

  if (blockNeedsRewrite(config.remote) || Object.values(config.profiles || {}).some(blockNeedsRewrite)) {
    touched = true
    writeDesktopConnectionConfig({
      ...config,
      remote: rewriteBlock(config.remote),
      profiles: Object.fromEntries(Object.entries(config.profiles || {}).map(([k, v]) => [k, rewriteBlock(v)]))
    })
  }

  // v2 connections.json registry.
  const registry = readDesktopConnectionsRegistry()

  if (registry.connections?.some(blockNeedsRewrite)) {
    touched = true
    writeDesktopConnectionsRegistry({ ...registry, connections: registry.connections.map(rewriteBlock) })
  }

  // Native OAuth token store: baseUrl → blob.
  const io = _nativeTokenStoreIo()

  try {
    const store = JSON.parse(io.readStoreText())

    if (store && typeof store === 'object' && !Array.isArray(store)) {
      const entries = Object.entries(store)

      if (entries.some(([, v]) => shouldRewrite(v))) {
        touched = true
        io.writeStoreText(JSON.stringify(Object.fromEntries(entries.map(([k, v]) => [k, reencode(v)]))))
      }
    }
  } catch {
    // Missing/corrupt native token store: nothing to rewrite.
  }

  return touched
}

/**
 * One-shot legacy migration: builds before the opt-in policy wrote every
 * secret as a safeStorage blob. With encryption now defaulting OFF, decrypt
 * each stored blob once and rewrite it as plain so no future launch touches
 * the keychain. Marked `migrated` whether or not every blob decrypts — a
 * broken keychain costs at most ONE prompt (this pass), never one per
 * launch; blobs that would not decrypt are left in place and simply read as
 * absent from then on (classifyStoredSecret → 'drop'), so opting encryption
 * back ON later can still recover them on a healthy keychain.
 *
 * Runs before createWindow() so every later read sees the final encodings.
 */
function migrateLegacyEncryptedSecretsOnce() {
  const policy = secretStoragePolicy()

  if (policy.on || policy.migrated) {
    return
  }

  const needsMigration = (secret: any) => classifyStoredSecret(secret, policy) === 'migrate'

  const reencode = (secret: any) => {
    if (!needsMigration(secret)) {
      return secret
    }

    const plaintext = decryptDesktopSecret(secret)

    // Undecryptable now (locked/absent keychain): keep the blob for a
    // potential future opt-in, but post-migration reads treat it as unset.
    return plaintext ? { encoding: 'plain', value: plaintext } : secret
  }

  let touchedKeychain = false

  try {
    touchedKeychain = rewriteAllStoredSecrets(needsMigration, reencode)
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error)

    rememberLog(`[secret-storage] legacy migration pass failed: ${detail}`)
  }

  setSecretStoragePolicy({ on: false, migrated: true })

  if (touchedKeychain) {
    rememberLog('[secret-storage] migrated legacy keychain-encrypted secrets to opt-out storage (one-shot pass)')
  }
}

/**
 * Settings → Gateway toggle: flip keychain-backed encryption and re-encode
 * every stored secret to match. Turning ON encrypts plain blobs through
 * strict safeStorage (throws loudly when the keychain is unusable — the
 * toggle stays off and the renderer shows the error). Turning OFF decrypts
 * back to plain; this is user-initiated, so a keychain prompt here is
 * expected and acceptable.
 */
function applySecretStorageEncryption(on: boolean) {
  const enable = on === true

  if (secretStoragePolicy().on === enable) {
    return { on: enable }
  }

  if (enable) {
    const needsEncrypt = (secret: any) => secret?.encoding === 'plain' && Boolean(secret.value)

    // Probe FIRST so an unusable keychain fails before any store is touched.
    if (
      !(() => {
        try {
          return Boolean(safeStorage.isEncryptionAvailable())
        } catch {
          return false
        }
      })()
    ) {
      throw new Error(
        'OS keychain encryption is unavailable on this machine, so stored gateway secrets cannot be encrypted.'
      )
    }

    setSecretStoragePolicy({ on: true, migrated: true })

    try {
      rewriteAllStoredSecrets(needsEncrypt, secret =>
        needsEncrypt(secret) ? encryptDesktopSecretStrict(String(secret.value), safeStorage) : secret
      )
    } catch (error) {
      // Encryption failed midway: revert the policy so reads keep working
      // against whatever encodings are on disk (mixed stores read fine —
      // decryptDesktopSecret handles both encodings under either policy).
      setSecretStoragePolicy({ on: false, migrated: true })
      throw error
    }

    return { on: true }
  }

  // Turning OFF: decrypt everything back to plain while the keychain is
  // still readable, then flip the policy.
  const needsDecrypt = (secret: any) => secret?.encoding === SAFE_STORAGE_ENCODING

  rewriteAllStoredSecrets(needsDecrypt, (secret: any) => {
    if (!needsDecrypt(secret)) {
      return secret
    }

    const plaintext = decryptDesktopSecret(secret)

    return plaintext ? { encoding: 'plain', value: plaintext } : secret
  })

  setSecretStoragePolicy({ on: false, migrated: true })

  return { on: false }
}

function encryptDesktopSecret(value, options = {}) {
  if (!secretStoragePolicy().on) {
    const raw = String(value || '')

    return raw ? { encoding: 'plain', value: raw } : null
  }

  return encryptDesktopSecretStrict(value, safeStorage, options)
}

function decryptDesktopSecret(secret) {
  if (!secret || typeof secret !== 'object') {
    return ''
  }

  const value = String(secret.value || '')

  if (!value) {
    return ''
  }

  if (secret.encoding === SAFE_STORAGE_ENCODING) {
    // Legacy blob under an opted-out policy: once the one-shot migration pass
    // has run, never touch safeStorage again — a dead keychain would otherwise
    // prompt on every read. Before that pass, decryption is allowed so the
    // migration itself (and this launch's reads) can recover the value.
    if (classifyStoredSecret(secret, secretStoragePolicy()) === 'drop') {
      return ''
    }

    try {
      return safeStorage.decryptString(Buffer.from(value, 'base64'))
    } catch {
      return ''
    }
  }

  // Any other encoding (a hand-edited config, or one written by a pre-release
  // build) is returned verbatim on purpose: this fallback is what lets such a
  // config connect at all. Not a plaintext-writing path — nothing in this file
  // persists a token this way.
  return value
}

function decryptRemoteHeaders(headers) {
  const normalized = normalizeRemoteHeaders(headers)
  const out = {}

  for (const [name, secret] of Object.entries(normalized)) {
    const value = decryptDesktopSecret(secret)

    if (value) {
      out[name] = value
    }
  }

  return out
}

/**
 * Turn an editor payload of remote gateway headers into stored secret
 * envelopes. The payload map is authoritative (a name missing from it is
 * cleared); per-name values are:
 *   - non-empty string  → new plaintext value, encrypted like a token
 *   - null              → keep the currently stored envelope for that name
 *                         (the editor shows a set-but-hidden secret)
 *   - envelope object   → stored verbatim (hand-edited import path)
 * Name filtering (forbidden/managed headers) happens in
 * normalizeRemoteHeaders at the registry/config layer.
 */
function encryptIncomingRemoteHeaders(raw, existing, options: { allowPlainText?: boolean } = {}) {
  const out = {}
  const stored = normalizeRemoteHeaders(existing)

  for (const [name, value] of Object.entries(raw || {})) {
    const key = String(name || '').trim()

    if (!key) {
      continue
    }

    if (typeof value === 'string') {
      const trimmed = value.trim()

      if (trimmed) {
        out[key] = encryptDesktopSecret(trimmed, { allowPlainText: options.allowPlainText === true })
      }

      continue
    }

    if (value === null) {
      if (stored[key]) {
        out[key] = stored[key]
      }

      continue
    }

    if (value && typeof value === 'object') {
      out[key] = value
    }
  }

  return out
}

function rememberRemoteWsHeaders(wsUrl, headers = {}) {
  if (!wsUrl || Object.keys(headers).length === 0) {
    return
  }

  remoteWsHeadersByUrl.set(String(wsUrl), headers as Record<string, string>)

  while (remoteWsHeadersByUrl.size > 100) {
    const oldest = remoteWsHeadersByUrl.keys().next().value

    if (!oldest) {
      break
    }

    remoteWsHeadersByUrl.delete(oldest)
  }
}

function headersForRemoteRequest(requestUrl) {
  const exactWsHeaders = remoteWsHeadersByUrl.get(String(requestUrl))

  if (exactWsHeaders && Object.keys(exactWsHeaders).length > 0) {
    return exactWsHeaders
  }

  const config = readDesktopConnectionConfig()

  if (modeIsRemoteLike(config.mode) && config.remote?.url) {
    const headers = decryptRemoteHeaders(config.remote.headers)

    if (Object.keys(headers).length > 0 && remoteRequestMatchesBaseUrl(requestUrl, config.remote.url)) {
      return headers
    }
  }

  return {}
}

function installRemoteHeaderRules() {
  if (remoteHeaderRulesInstalled) {
    return
  }

  remoteHeaderRulesInstalled = true
  session.defaultSession.webRequest.onBeforeSendHeaders((details, callback) => {
    const headers = headersForRemoteRequest(details.url)

    if (Object.keys(headers).length === 0) {
      callback({})

      return
    }

    callback({ requestHeaders: { ...details.requestHeaders, ...headers } })
  })
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

    if (entry.mode === 'ssh') {
      const ssh = normalizeSshConfig(entry)

      if (ssh) {
        if (entry.token && typeof entry.token === 'object') {
          ssh.token = entry.token
        }

        out[name] = ssh
      }

      continue
    }

    const cleaned: {
      mode: 'remote' | 'local' | 'cloud'
      url?: string
      authMode?: string
      token?: object
      headers?: object
      org?: string
      savedSsh?: object
    } = {
      mode: modeIsRemoteLike(entry.mode) ? entry.mode : 'local'
    }

    if (cleaned.mode === 'local') {
      const savedSsh = normalizeSshConfig(entry.savedSsh)

      if (savedSsh) {
        cleaned.savedSsh = savedSsh
      }
    }

    const url = String(entry.url || '').trim()

    if (url) {
      cleaned.url = url
    }

    cleaned.authMode = normAuthMode(entry.authMode)

    if ((entry as any).token && typeof entry.token === 'object') {
      cleaned.token = entry.token
    }

    const headers = normalizeRemoteHeaders((entry as any).headers)

    if (Object.keys(headers).length > 0) {
      cleaned.headers = headers
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
    // Tighten an install written before this file was owner-only. Every write
    // now goes out at 0600, but a file already on disk keeps its old 0644 bits
    // until something chmods it, and waiting for the user's next Settings save
    // would leave it group/other-readable indefinitely. Runs on a cache miss
    // only (once per launch, plus after an external edit); chmod moves ctime,
    // not mtime, so it cannot invalidate the cache it sits inside.
    //
    // Deliberately BEFORE JSON.parse, not after: a truncated or hand-mangled
    // connection.json still contains the token bytes, and parse throws into the
    // catch below, which swallows the error and falls back to local mode. With
    // the tighten after the parse, exactly the file that is both corrupt AND
    // world-readable would be the one file never tightened — and nothing would
    // ever retry it, because the fallback config is not written back. The chmod
    // needs only the path, so it has no reason to wait for valid JSON.
    tightenSecretFileMode(DESKTOP_CONNECTION_CONFIG_PATH)

    const parsed = JSON.parse(raw)

    // NOT done here: migrating a legacy non-safeStorage token payload to
    // ciphertext at rest. Deferred deliberately — it has to honor the opt-in
    // plaintext choice PR #62319 adds (re-encrypting it converts a portable
    // credential into a keychain-bound one and can lose the token), write
    // through sanitizeConnectionProfiles below rather than persisting raw
    // `parsed`, and tell the user to ROTATE, since every existing backup copy
    // still holds the old secret. Do not add it without those three.

    if (parsed && typeof parsed === 'object') {
      const remote = parsed.remote && typeof parsed.remote === 'object' ? parsed.remote : {}
      // authMode lives on the remote sub-object: 'oauth' (cookie + ws-ticket)
      // or 'token' (legacy static session token). Default to 'token' for
      // backward compatibility with configs written before OAuth support.
      remote.authMode = remote.authMode === 'oauth' ? 'oauth' : 'token'
      config = {
        mode: parsed.mode === 'ssh' ? 'ssh' : modeIsRemoteLike(parsed.mode) ? parsed.mode : 'local',
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
  // Owner-only, not writeFileAtomic: this is the single choke point for every
  // connection.json write (the IPC save/apply handlers and
  // persistSshConnectionToken all land here), and the file carries the
  // safeStorage-encrypted gateway token plus its URL and SSH host/user/keyPath.
  // safeStorage keeps the token opaque; 0600 keeps the whole record — and the
  // fields that are NOT encrypted — off other local accounts, matching
  // native-oauth-tokens.json and desktop-installation.json.
  writeSecretFileAtomic(DESKTOP_CONNECTION_CONFIG_PATH, JSON.stringify(config, null, 2))
  connectionConfigCache = config
  connectionConfigCacheMtime = fs.statSync(DESKTOP_CONNECTION_CONFIG_PATH).mtimeMs
}

// ── v2 connection registry (multi-source) ──────────────────────────────────

/**
 * Read the v2 registry, importing from v1 connection.json exactly once (when
 * connections.json does not exist yet). Same mtime-cache + tighten-mode
 * discipline as readDesktopConnectionConfig; a corrupt registry degrades to
 * local-only via normalizeRegistry rather than throwing at boot.
 *
 * An EXISTING registry is additionally reconciled against v1 when the two have
 * drifted — see reconcileRegistryDrift. The one-shot migration cannot cover a
 * user who registered nothing and then pointed Settings -> Gateway at a remote,
 * and until that heals, every launch re-homes them onto a local backend.
 */
function readDesktopConnectionsRegistry() {
  let mtime = null

  try {
    mtime = fs.statSync(DESKTOP_CONNECTIONS_REGISTRY_PATH).mtimeMs
  } catch {
    mtime = null
  }

  if (connectionRegistryCache && connectionRegistryCacheMtime === mtime) {
    return connectionRegistryCache
  }

  let registry

  if (mtime === null) {
    // First run on this build: import the v1 single-connection config. The v1
    // file is NOT modified or deleted — older builds keep reading it. The
    // migration is deterministic over the v1 input, so even if two processes
    // race the first run (updater relaunch, second window), both derive the
    // same registry and the later atomic write is a no-op content-wise.
    registry = migrateV1ToRegistry(readDesktopConnectionConfig())

    try {
      writeDesktopConnectionsRegistry(registry)
    } catch {
      // Write failed (full disk, read-only userData). Keep the migrated
      // registry in memory so list/save keep working this session instead of
      // hard-failing every hermes:connections:* call.
      connectionRegistryCache = registry
      connectionRegistryCacheMtime = null
    }

    return connectionRegistryCache
  }

  try {
    // Same rationale as connection.json: tighten BEFORE parse so a corrupt
    // file that still holds token bytes gets its mode fixed anyway.
    tightenSecretFileMode(DESKTOP_CONNECTIONS_REGISTRY_PATH)
    registry = normalizeRegistry(JSON.parse(fs.readFileSync(DESKTOP_CONNECTIONS_REGISTRY_PATH, 'utf8')))
  } catch {
    registry = normalizeRegistry(null)
  }

  // Heal v1 -> v2 drift: the v1 global route names a remote this registry has
  // never heard of, so the live descriptor resolves to no connectionId and the
  // launch pick sends the window somewhere else. Persist so the repair is a
  // one-time event rather than a recomputation on every read; a failed write
  // still returns the healed registry for this session.
  const reconciled = reconcileRegistryDrift(registry, readDesktopConnectionConfig())

  if (reconciled.changed) {
    registry = reconciled.registry

    try {
      writeDesktopConnectionsRegistry(registry)

      return connectionRegistryCache
    } catch {
      connectionRegistryCache = registry
      connectionRegistryCacheMtime = null

      return registry
    }
  }

  connectionRegistryCache = registry
  connectionRegistryCacheMtime = mtime

  return registry
}

function writeDesktopConnectionsRegistry(registry) {
  fs.mkdirSync(path.dirname(DESKTOP_CONNECTIONS_REGISTRY_PATH), { recursive: true })
  // Owner-only for the same reason as connection.json: entries carry
  // safeStorage-encrypted tokens plus URLs and SSH host/user/keyPath.
  writeSecretFileAtomic(DESKTOP_CONNECTIONS_REGISTRY_PATH, JSON.stringify(registry, null, 2))
  connectionRegistryCache = registry
  connectionRegistryCacheMtime = fs.statSync(DESKTOP_CONNECTIONS_REGISTRY_PATH).mtimeMs
}

/**
 * Renderer-facing view of a registry entry: token bytes never cross the IPC
 * boundary — the renderer gets a preview + set flag, mirroring
 * sanitizeDesktopConnectionConfig.
 */
function sanitizeRegistryConnection(entry) {
  const { token, headers, ...rest } = entry
  const decrypted = decryptDesktopSecret(token)
  // Last-known stable backend identity (from roster enumeration / Test) so
  // Settings can hint "Same backend as <label>" on connections that are two
  // addresses for one box. Display-only; absent until a probe has seen it.
  const knownInstallId = connectionInstallIds.get(entry.id)?.id

  return {
    ...rest,
    tokenSet: Boolean(decrypted),
    tokenPreview: tokenPreview(decrypted),
    ...(knownInstallId ? { installId: knownInstallId } : {}),
    // Header VALUES are secrets (Cloudflare Access client secrets etc.) and
    // never cross the IPC boundary — the renderer only needs the names to
    // render the edit form.
    headerNames: headers && typeof headers === 'object' ? Object.keys(headers) : []
  }
}

function sanitizeConnectionsRegistry(registry = readDesktopConnectionsRegistry()) {
  // Same keyring signal the v1 sanitize exposes: lets the Connections panel
  // offer the plain-text opt-in on keyring-less Linux instead of failing.
  // Policy-aware: never touches safeStorage while encryption is opted out.
  const secureTokenStorage = probeSecureTokenStorage()

  return {
    version: registry.version,
    primary: registry.primary,
    launchMode: registry.launchMode,
    lastUsed: registry.lastUsed,
    secureTokenStorage,
    connections: registry.connections.map(sanitizeRegistryConnection)
  }
}

/**
 * Save (create or edit) a registry connection from a renderer payload.
 * Edits merge over the stored entry (mergeConnectionInput) so fields the
 * editor doesn't carry — cloud `org`, ssh `remoteHermesPath`/`remoteProfile` —
 * survive a rename. Token handling mirrors coerceDesktopConnectionConfig: an
 * incoming plaintext token is encrypted (honoring the same allowPlainTextToken
 * opt-in seam as Settings → Gateway); an absent token field inherits the
 * stored envelope on edit; switching auth away from 'token' clears it
 * (normalizeConnectionInput drops tokens on non-token entries).
 */
async function saveRegistryConnection(input: any = {}) {
  const registry = readDesktopConnectionsRegistry()
  const existing = input.id ? registry.connections.find(c => c.id === input.id) : null
  const incomingToken = typeof input.token === 'string' ? input.token.trim() : ''

  const token = resolvePersistedRemoteToken({
    incomingToken,
    persistToken: true,
    existingToken: existing?.token,
    allowPlainText: input.allowPlainTextToken,
    encryptSecret: encryptDesktopSecret
  })

  // Extra gateway headers arrive as plaintext strings from the editor (or
  // envelopes from a hand-edited import). Encrypt plaintext values the same
  // way tokens are stored; a null/empty value drops that header. An absent
  // `headers` field inherits the stored set via mergeConnectionInput.
  const headers =
    input.headers && typeof input.headers === 'object'
      ? encryptIncomingRemoteHeaders(input.headers, existing?.headers, {
          allowPlainText: input.allowPlainTextToken
        })
      : input.headers

  const merged = mergeConnectionInput({ ...input, token, headers }, existing)
  const entry = normalizeConnectionInput(merged, registry)

  // Token-auth remotes must actually have a token to be dialable. OAuth and
  // cloud entries authenticate via cookies/native tokens instead.
  if (entry.kind === 'remote' && entry.authMode !== 'oauth' && !decryptDesktopSecret(entry.token)) {
    throw new Error('Remote gateway session token is required.')
  }

  writeDesktopConnectionsRegistry(upsertConnection(registry, entry))

  // A dial-material edit (endpoint/auth/ssh routing — NOT a label rename)
  // leaves pooled backends under `conn:<id>::*` and renderer sockets pointing
  // at the OLD target while the UI shows the new one. Recycle them: stop this
  // connection's pooled backends/tunnels and tell renderers to dispose+redial
  // their secondaries for this connection id.
  if (existing && connectionDialFieldsChanged(existing, entry)) {
    await stopRegistryConnectionBackends(entry.id)
    broadcastConnectionsChanged({ connectionId: entry.id, reason: 'updated' })
  }

  return sanitizeRegistryConnection(entry)
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
  const savedMode = key ? scoped?.mode : config.mode
  const ssh = savedMode === 'ssh' ? normalizeSshConfig(block) : null

  const savedSsh = savedMode === 'local' ? (key ? savedProfileSsh(config, key) : normalizeSshConfig(block)) : null

  const remoteToken = decryptDesktopSecret(block.token)
  const authMode = normAuthMode(block.authMode)
  const remoteUrl = envOverride ? String(process.env.HERMES_DESKTOP_REMOTE_URL || '') : String(block.url || '')
  const mode = envOverride ? 'remote' : savedMode === 'ssh' ? 'ssh' : modeIsRemoteLike(savedMode) ? savedMode : 'local'

  // Whether the OS keyring (safeStorage) can encrypt the saved token. When
  // false the renderer knows to offer the plain-text opt-in in Settings →
  // Gateway. With keychain encryption opted out (the default) this reports
  // true WITHOUT touching safeStorage — probing is itself a keychain touch
  // that raises the macOS password dialog (see probeSecureTokenStorage).
  const secureTokenStorage = probeSecureTokenStorage()

  // Whether the currently saved token is stored in plain text (the keyring-less
  // opt-in path). The env override supplies its token from the environment, not
  // the saved block, so it never reports as plain text here.
  const remoteTokenPlainText = !envOverride && block.token?.encoding === 'plain'

  let remoteOauthConnected = false

  if (authMode === 'oauth' && remoteUrl) {
    try {
      // Display signal: treat a live RT cookie as "connected" even if the AT
      // cookie has lapsed — the gateway refreshes the AT on the next request,
      // so the session is still usable. A stored native bearer token (cookieless
      // RFC 8252 flow) counts as connected too — otherwise a completed native
      // sign-in shows "not connected" in Settings. The authoritative liveness
      // check is the ws-ticket mint in resolveRemoteBackend at actual connect time.
      remoteOauthConnected = oauthSessionIsLive(hasNativeSession(remoteUrl), await hasLiveOauthSession(remoteUrl))
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
    // Whether the OS keyring can encrypt a token; drives the plain-text opt-in
    // affordance in Settings → Gateway on keyring-less Linux.
    secureTokenStorage,
    // Whether the saved token is currently persisted in plain text.
    remoteTokenPlainText,
    sshHost: (ssh || savedSsh)?.host || '',
    sshUser: (ssh || savedSsh)?.user || '',
    sshPort: (ssh || savedSsh)?.port || null,
    sshKeyPath: (ssh || savedSsh)?.keyPath || '',
    sshRemoteHermesPath: (ssh || savedSsh)?.remoteHermesPath || '',
    sshRemoteProfile: (ssh || savedSsh)?.remoteProfile || '',
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
function buildRemoteBlock(remoteUrl, authMode, token, org?: string, headers?: object) {
  if (authMode !== 'oauth' && !decryptDesktopSecret(token)) {
    throw new Error('Remote gateway session token is required.')
  }

  const block: { url: string; authMode: string; token: object; headers?: object; org?: string } = {
    url: normalizeRemoteBaseUrl(remoteUrl),
    authMode,
    token
  }

  const remoteHeaders = normalizeRemoteHeaders(headers)

  if (Object.keys(remoteHeaders).length > 0) {
    block.headers = remoteHeaders
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
  const mode = input.mode === 'ssh' ? 'ssh' : modeIsRemoteLike(input.mode) ? input.mode : 'local'
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
  const leavingSsh = rawExistingBlock.mode === 'ssh' && mode !== 'ssh' && mode !== 'local'
  const existingBlock = leavingCloud || leavingSsh ? {} : rawExistingBlock
  const remoteUrl = String(input.remoteUrl ?? existingBlock.url ?? '').trim()
  // authMode: explicit input wins; otherwise inherit the saved value, default 'token'.
  const authMode = resolveAuthMode(input.remoteAuthMode, existingBlock.authMode)
  // Cloud org: only meaningful for 'cloud' mode. Explicit input wins; otherwise
  // inherit the saved org. A plain 'remote' connection never carries an org
  // (switching cloud→remote drops it), so it stays unset unless mode is cloud.
  const cloudOrg = mode === 'cloud' ? String(input.cloudOrg ?? existingBlock.org ?? '').trim() : ''
  const incomingToken = typeof input.remoteToken === 'string' ? input.remoteToken.trim() : ''

  const remoteHeaders =
    input.remoteHeaders && typeof input.remoteHeaders === 'object' ? input.remoteHeaders : existingBlock.headers

  // Persist decision lives in hardening.resolvePersistedRemoteToken so the
  // IPC-propagation seam (allowPlainTextToken → encryptDesktopSecret opt-in) is
  // covered by a focused regression test. Pass allowPlainText through RAW — the
  // helper coerces with `=== true`, so a truthy-non-true value never enables
  // plain-text storage, and that strictness is asserted in exactly one place.
  const nextToken = resolvePersistedRemoteToken({
    incomingToken,
    persistToken,
    existingToken: existingBlock.token,
    allowPlainText: input.allowPlainTextToken,
    encryptSecret: encryptDesktopSecret
  })

  if (mode === 'ssh') {
    const sshBlock = buildSshBlock(input, savedProfileSsh(existing, key) || rawExistingBlock)

    if (key) {
      const profiles = { ...(existing.profiles || {}), [key]: sshBlock }

      return {
        mode: existing.mode === 'ssh' || modeIsRemoteLike(existing.mode) ? existing.mode : 'local',
        remote: existing.remote || {},
        profiles
      }
    }

    return { mode: 'ssh', remote: sshBlock, profiles: existing.profiles || {} }
  }

  if (key) {
    // Per-profile scope: a remote/cloud entry pins this profile to its own
    // backend; a local entry clears the override so the profile inherits the
    // default. The mode tag (remote vs cloud) is preserved on the entry.
    const profiles = { ...(existing.profiles || {}) }

    if (remoteLike) {
      profiles[key] = {
        mode,
        ...buildRemoteBlock(remoteUrl, authMode, nextToken, cloudOrg, remoteHeaders)
      }
    } else {
      const localEntry = localProfileEntry(rawExistingBlock)

      if (localEntry) {
        profiles[key] = localEntry
      } else {
        delete profiles[key]
      }
    }

    return {
      mode: existing.mode === 'ssh' || modeIsRemoteLike(existing.mode) ? existing.mode : 'local',
      remote: existing.remote || {},
      profiles
    }
  }

  const nextRemote = remoteLike
    ? buildRemoteBlock(remoteUrl, authMode, nextToken, cloudOrg, remoteHeaders)
    : existingMode === 'ssh'
      ? rawExistingBlock
      : { url: remoteUrl ? normalizeRemoteBaseUrl(remoteUrl) : remoteUrl, authMode, token: nextToken }

  // Preserve per-profile overrides when saving the global connection.
  return { mode, remote: nextRemote, profiles: existing.profiles || {} }
}

// Build an SSH connection block from a save payload, preserving an
// already-adopted dashboard token from the existing block (the token is minted
// + reconciled at bootstrap, never user-entered). `mode: 'ssh'` is stamped so
// normalizeSshConfig/profileSshOverride recognize it.
function buildSshBlock(input: any, existingBlock: any = {}) {
  // `??` (not `||`) so an explicit '' (user CLEARED the field) wins over the
  // saved value; only a truly absent (undefined) field inherits.
  const merged = normalizeSshConfig({
    mode: 'ssh',
    host: input.sshHost ?? existingBlock.host,
    user: input.sshUser ?? existingBlock.user,
    port: input.sshPort ?? existingBlock.port,
    keyPath: input.sshKeyPath ?? existingBlock.keyPath,
    remoteHermesPath: input.sshRemoteHermesPath ?? existingBlock.remoteHermesPath,
    remoteProfile: input.sshRemoteProfile ?? existingBlock.remoteProfile
  })

  if (!merged) {
    throw new Error('SSH host is required.')
  }

  // Carry forward an already-adopted dashboard token unless the host changed
  // (a different host invalidates the old dashboard's token).
  if (existingBlock.token && existingBlock.host === merged.host) {
    merged.token = existingBlock.token
  }

  return merged
}

// Build a remote backend connection descriptor from an already-resolved remote
// config. Handles both auth models (OAuth ws-ticket vs static session token)
// and is shared by the per-profile, env, and global resolution paths. `token`
// is the DECRYPTED static token (or null in OAuth mode). `source` is a label
// for diagnostics ('profile' | 'env' | 'settings').
async function buildRemoteConnection(
  rawUrl,
  authMode,
  token,
  source,
  remoteHost?,
  remoteKind = 'url',
  remoteIdentity?,
  headers?
) {
  const baseUrl = normalizeRemoteBaseUrl(rawUrl)
  const remoteHeaders = decryptRemoteHeaders(headers)
  // For token/oauth remotes the meaningful host is the real backend URL; for
  // SSH remotes the caller passes the entered/resolved host explicitly (the
  // baseUrl is a 127.0.0.1 tunnel and would be useless in the pill).
  const host = remoteHost || hostLabelFromBaseUrl(baseUrl)

  if (authMode === 'oauth') {
    // OAuth gateway: auth comes from EITHER a native bearer token (cookieless
    // RFC 8252 flow) OR the session cookies in the OAuth partition. Liveness is
    // NOT "is the access-token cookie present?" — Portal issues a 24h rotating
    // refresh token (hermes #37247), and the gateway middleware transparently
    // rotates a fresh ~15-min access token from it on the next authenticated
    // request. So a session with an expired AT cookie but a live RT cookie is
    // still perfectly connectable. We early-out only when NEITHER a native
    // token NOR any cookie is present, then mint a ws-ticket (which itself
    // prefers the native bearer) as the authoritative liveness check.
    //
    // The native-token check is essential: the native login stores bearer
    // tokens (no cookie is ever set), so gating solely on hasLiveOauthSession
    // here would reject a freshly-completed native sign-in and loop the UI back
    // into "not signed in" even though mintGatewayWsTicket would succeed with
    // the stored bearer.
    if (
      !oauthSessionIsLive(hasNativeSession(baseUrl), await hasLiveOauthSession(baseUrl)) &&
      oauthGuardMayHardFail(await gatewayAuthProviders(baseUrl, remoteHeaders))
    ) {
      throw makeUnsignedOauthError()
    }

    let ticket

    try {
      ticket = await mintGatewayWsTicket(baseUrl, remoteHeaders)
    } catch (error) {
      // For a Nous-managed Cloud agent, a 502/503/504 from the WS-ticket mint
      // means the backend server itself is down — the actionable Cloud-down
      // error. This boundary runs BEFORE the readiness loop, so without this
      // the ticket wrapper below would swallow the server-fault classification
      // and the renderer would never see isCloudBackendDown. Preserve the
      // existing 401/403 reauth and generic transport behavior for everything
      // else (#85335).
      const cloudError = makeNousCloudBackendDownError(baseUrl, error)

      if (cloudError !== null) {
        throw cloudError
      }

      throw gatewayTicketFailure(
        error,
        'Your remote gateway session has expired. Open Settings → Gateway and click "Sign in" again.',
        'Could not reach the remote Hermes gateway while refreshing its WebSocket ticket. Try reconnecting.'
      )
    }

    const wsUrl = buildGatewayWsUrlWithTicket(baseUrl, ticket)

    rememberRemoteWsHeaders(wsUrl, remoteHeaders)

    return {
      baseUrl,
      mode: 'remote',
      source,
      authMode: 'oauth',
      remoteHost: host || undefined,
      remoteIdentity,
      remoteKind,
      headers: remoteHeaders,
      // No static token in OAuth mode; REST is cookie-authed via the partition.
      token: null,
      wsUrl
    }
  }

  if (!token) {
    throw new Error(
      'Remote Hermes gateway is selected, but no session token is saved. ' +
        'Open Settings → Gateway and save a token, or switch back to Local.'
    )
  }

  const wsUrl = buildGatewayWsUrl(baseUrl, token)

  rememberRemoteWsHeaders(wsUrl, remoteHeaders)

  return {
    baseUrl,
    mode: 'remote',
    source,
    authMode: 'token',
    remoteHost: host || undefined,
    remoteIdentity,
    remoteKind,
    headers: remoteHeaders,
    token,
    wsUrl
  }
}

const sshConnections = new Map<string, any>()
const desktopInstallationId = loadOrCreateInstallationId(DESKTOP_INSTALLATION_PATH)

const sshBootstrapCoordinator = createBootstrapCoordinator()

let sshQuitTeardownDone = false
let backendQuitTeardownDone = false

function sshScopeKey(profile) {
  return connectionScopeKey(profile) || ''
}

function sshOwnershipKey(profile) {
  return sshOwnershipId(desktopInstallationId, sshScopeKey(profile))
}

function sshRememberLog(chunk) {
  rememberLog(redactSecrets(String(chunk == null ? '' : chunk)))
}

async function sshProbeReuseProof(baseUrl, token, spawnNonce) {
  try {
    const proof: any = await fetchJson(`${baseUrl}/api/ssh/ownership`, token)

    return proof?.ok === true && proof.sshOwnerNonce === spawnNonce && proof.protocolVersion === 1
      ? 'authenticated-ok'
      : 'authenticated-stale'
  } catch (error: any) {
    if (/^(401|403|404):/.test(String(error?.message || ''))) {
      return 'authenticated-stale'
    }

    throw error
  }
}

async function teardownSshConnection(profile) {
  const scope = sshScopeKey(profile)
  const state = sshConnections.get(scope)

  if (!state) {
    return
  }

  sshConnections.delete(scope)

  terminalIpc.disposeTerminalSessionsForSshScope(scope)

  try {
    if (state.localPort && state.remotePort) {
      await state.ssh.cancelForward(state.localPort, state.remotePort)
    }
  } catch {
    // best effort
  }

  try {
    await state.ssh.close()
  } catch {
    // best effort
  }
}

// CRITICAL: this must mirror resolveRemoteBackend's precedence, not just return
// any cached SSH state. A per-profile token/OAuth override wins over a global
// SSH connection — so if the active profile resolves to a NON-SSH backend, the
// terminal must NOT fall through to a global SSH host.
function activeSshTerminalTarget(webContentsId?: number) {
  const windowRoute = typeof webContentsId === 'number' ? windowConnectionRoutes.get(webContentsId) : null

  if (windowRoute?.registryScoped && windowRoute.connectionId) {
    const scope = registrySshScopeForWindowRoute(windowRoute, readDesktopConnectionsRegistry())

    if (!scope) {
      return null
    }

    const state = sshConnections.get(scope)

    return state && state.ssh ? { ssh: state.ssh, scope } : 'pending'
  }

  const profile = windowRoute?.profile ?? primaryProfileKey()
  const config = readDesktopConnectionConfig()

  const route = resolveDesktopRemoteRoute({
    config,
    env: {
      token: process.env.HERMES_DESKTOP_REMOTE_TOKEN,
      url: process.env.HERMES_DESKTOP_REMOTE_URL
    },
    profile,
    registry: readDesktopConnectionsRegistry()
  })

  if (!route || route.kind !== 'ssh') {
    return null
  }

  const scope = route.connectionId
    ? backendScopeKey(route.connectionId, profile)
    : sshScopeKey(route.source === 'profile' ? profile : null)

  const state = sshConnections.get(scope)

  return state && state.ssh ? { ssh: state.ssh, scope } : 'pending'
}

async function ensureTerminalBackend(webContentsId: number) {
  const windowRoute = windowConnectionRoutes.get(webContentsId)

  if (windowRoute?.registryScoped && windowRoute.connectionId) {
    return ensureRegistryBackend(windowRoute.connectionId, windowRoute.profile)
  }

  return ensureBackend(windowRoute?.profile ?? primaryProfileKey())
}

// Loopback reach for the browser pane. Scoped to the SSH connection that
// authorized it: a different host (or none) must never inherit live forwards
// into somebody else's machine.
const previewReachByWebContents = new Map<number, { registry: PreviewReachRegistry; scope: string }>()

async function resetPreviewReach(webContentsId?: number) {
  if (typeof webContentsId === 'number') {
    const current = previewReachByWebContents.get(webContentsId)

    previewReachByWebContents.delete(webContentsId)

    if (current) {
      await current.registry.closeAll()
    }

    return
  }

  const open = [...previewReachByWebContents.values()]

  previewReachByWebContents.clear()
  await Promise.allSettled(open.map(entry => entry.registry.closeAll()))
}

/**
 * Rewrite a gateway-loopback URL into one this machine can actually load.
 *
 * Returns the URL unchanged when no rewrite is needed or possible — a local
 * backend (the address is already true), a non-loopback host, or a url/cloud
 * remote with no tunnel to borrow. Callers must not treat an unchanged URL as
 * failure; the pane explains an unreachable one on its own.
 */
async function reachablePreviewUrl(webContentsId: number, rawUrl: string): Promise<string> {
  let target = activeSshTerminalTarget(webContentsId)

  if (target === 'pending') {
    await ensureTerminalBackend(webContentsId).catch(() => undefined)
    target = activeSshTerminalTarget(webContentsId)
  }

  if (!target || target === 'pending') {
    // No SSH transport behind this renderer's gateway. Another window's
    // forward must never be reused for this preview.
    await resetPreviewReach(webContentsId)

    return rawUrl
  }

  const { scope, ssh } = target as { scope: string; ssh: any }
  let reach = previewReachByWebContents.get(webContentsId)

  if (!reach || reach.scope !== scope) {
    await resetPreviewReach(webContentsId)
    reach = { registry: new PreviewReachRegistry(), scope }
    previewReachByWebContents.set(webContentsId, reach)
  }

  try {
    const rewritten = await reach.registry.resolve(rawUrl, {
      cancel: (localPort, remotePort) => ssh.cancelForward(localPort, remotePort),
      forward: (localPort, remotePort, remoteHost) => ssh.forward(localPort, remotePort, remoteHost),
      isCurrent: () => sshConnections.get(scope)?.ssh === ssh,
      // pickLocalPort predates the typed surface here and infers `unknown`.
      pickLocalPort: () => pickLocalPort() as Promise<number>
    })

    return rewritten || rawUrl
  } catch (error: any) {
    sshRememberLog(`preview reach failed for ${rawUrl}: ${error?.message || error}`)

    return rawUrl
  }
}

function effectiveSshConfigFingerprint(sshConfig) {
  const ssh =
    process.platform === 'win32'
      ? path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'OpenSSH', 'ssh.exe')
      : 'ssh'

  const args = ['-G']

  if (sshConfig.port) {
    args.push('-p', String(sshConfig.port))
  }

  if (sshConfig.keyPath) {
    args.push('-i', sshConfig.keyPath)
  }

  args.push('--', sshConfig.user ? `${sshConfig.user}@${sshConfig.host}` : sshConfig.host)
  const output = execFileSync(ssh, args, { encoding: 'utf8', timeout: 10_000, windowsHide: true })

  return crypto.createHash('sha256').update(output).digest('hex')
}

async function bootstrapSshConnection(profile, sshConfig, reuseToken, source) {
  const scope = sshScopeKey(profile)
  const effectiveConfigFingerprint = effectiveSshConfigFingerprint(sshConfig)
  const resolvedConfig = { ...sshConfig, effectiveConfigFingerprint }
  const fingerprint = sshConfigFingerprint(scope, resolvedConfig)

  return sshBootstrapCoordinator.start(scope, fingerprint, lease =>
    bootstrapSshConnectionInner(profile, resolvedConfig, reuseToken, source, fingerprint, lease)
  )
}

async function bootstrapSshConnectionInner(profile, sshConfig, reuseToken, source, fingerprint, lease) {
  const scope = sshScopeKey(profile)
  const hostLabel = sshConfig.user ? `${sshConfig.user}@${sshConfig.host}` : sshConfig.host
  const existing = sshConnections.get(scope)

  if (existing && existing.fingerprint !== fingerprint) {
    await teardownSshConnection(profile)
  }

  let ssh = sshConnections.get(scope)?.ssh

  if (ssh && !(await ssh.isAlive())) {
    try {
      await ssh.close()
    } catch {
      void 0
    }

    ssh = null
    sshConnections.delete(scope)
  }

  const created = !ssh

  let removeForceCleanup = () => {}

  if (created) {
    ssh = new SshConnection(
      { host: sshConfig.host, user: sshConfig.user, port: sshConfig.port, keyPath: sshConfig.keyPath },
      {
        rememberLog: sshRememberLog,
        ownershipId: sshOwnershipKey(profile),
        scope,
        effectiveConfigFingerprint: sshConfig.effectiveConfigFingerprint
      }
    )
    removeForceCleanup = lease.onForceCleanup(() => ssh.close())
    await ssh.open({ signal: lease.signal })
  }

  let result

  try {
    const platform = await detectRemotePlatform(ssh, sshConfig.remoteHermesPath || '')
    const lifecycle = platform.os === 'Windows' ? connectWindowsRemote : remoteLifecycle.connect
    result = await lifecycle({
      ssh,
      profile: resolveRemoteSshDashboardProfile(sshConfig.remoteProfile, profile),
      remoteHermesPath: sshConfig.remoteHermesPath || '',
      ownershipId: sshOwnershipKey(profile),
      reuseToken: reuseToken || '',
      forward: (localPort, remotePort) => ssh.forward(localPort, remotePort),
      cancelForward: (localPort, remotePort) => ssh.cancelForward(localPort, remotePort),
      pickLocalPort,
      waitForHermes: (baseUrl, token) => waitForHermes(baseUrl, token, lease.signal, 'token'),
      probeReuseProof: sshProbeReuseProof,
      adoptServedToken: adoptServedDashboardToken,
      rememberLog: sshRememberLog,
      signal: lease.signal
    })
  } catch (error: any) {
    if (created) {
      try {
        await ssh.close()
      } catch {
        void 0
      }
    } else {
      // The cached master was reused but the lifecycle probe against it
      // failed ("Could not verify the existing SSH backend"). Keeping the
      // stale entry means every subsequent boot re-attempts through the same
      // wedged master/tunnel and fails identically until the user re-enters
      // the connection details (whose changed fingerprint forces a teardown).
      // Tear it down now so the next attempt — automatic retry included —
      // bootstraps a fresh master, which is exactly what manual re-entry
      // did (#82679).
      try {
        await teardownSshConnection(profile)
      } catch {
        void 0
      }
    }

    const err = new Error(error.message) as any
    err.sshError = error.kind || 'unknown'
    err.isSshBootstrap = true
    throw err
  }

  try {
    lease.assertCurrent()
  } catch (error) {
    try {
      await ssh.cancelForward(result.localPort, result.remotePort)
      await ssh.close()
    } catch {
      void 0
    }

    throw error
  }

  persistSshConnectionToken(profile, source, result.token)

  removeForceCleanup()
  sshConnections.set(scope, {
    ssh,
    fingerprint,
    localPort: result.localPort,
    remotePort: result.remotePort,
    pid: result.pid,
    host: sshConfig.host,
    hostLabel,
    hermesVersion: result.hermesVersion || '',
    remotePlatform: result.platform?.os || '',
    reused: result.reused
  })

  sshRememberLog(
    `[ssh] connection ${result.reused ? 'REUSED' : 'spawned'} dashboard: ` +
      `${result.hermesVersion || 'hermes (version unknown)'} at ${result.hermesPath || '?'}`
  )

  const connection = await buildRemoteConnection(
    result.baseUrl,
    'token',
    result.token,
    source,
    hostLabel,
    'ssh',
    result.ownershipId
  )

  return { ...connection, remoteHermesVersion: result.hermesVersion || '' }
}

function persistSshConnectionToken(profile, source, token) {
  try {
    // Registry-scoped ssh backend (source "registry:<connectionId>"): the
    // served token belongs on the registry entry, not v1 connection.json.
    if (typeof source === 'string' && source.startsWith('registry:')) {
      const id = source.slice('registry:'.length)
      const registry = readDesktopConnectionsRegistry()
      const entry = registry.connections.find(c => c.id === id)

      if (entry && entry.kind === 'ssh') {
        writeDesktopConnectionsRegistry(upsertConnection(registry, { ...entry, token: encryptDesktopSecret(token) }))
      }

      return
    }

    const config = readDesktopConnectionConfig()
    const encrypted = encryptDesktopSecret(token)

    if (source === 'profile') {
      const key = connectionScopeKey(profile)

      if (key && config.profiles?.[key]?.mode === 'ssh') {
        config.profiles[key].token = encrypted
        writeDesktopConnectionConfig(config)
      }
    } else if (config.mode === 'ssh' && config.remote) {
      config.remote.token = encrypted
      writeDesktopConnectionConfig(config)
    }
  } catch (error: any) {
    sshRememberLog(`[ssh] could not persist served token: ${error.message}`)
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

  const route = resolveDesktopRemoteRoute({
    config,
    env: {
      token: process.env.HERMES_DESKTOP_REMOTE_TOKEN,
      url: process.env.HERMES_DESKTOP_REMOTE_URL
    },
    profile,
    registry: readDesktopConnectionsRegistry()
  })

  if (!route) {
    return null
  }

  let connection

  if (route.kind === 'ssh') {
    connection = await bootstrapSshConnection(
      route.source === 'profile' ? profile : null,
      route.ssh,
      decryptDesktopSecret(route.token),
      route.source
    )
  } else {
    const token =
      route.authMode === 'oauth' ? null : route.source === 'env' ? route.token : decryptDesktopSecret(route.token)

    connection = await buildRemoteConnection(
      route.url,
      route.authMode,
      token,
      route.source,
      undefined,
      route.kind === 'cloud' ? 'cloud' : 'url',
      undefined,
      route.headers
    )
  }

  return route.connectionId ? { ...connection, connectionId: route.connectionId } : connection
}

// A remote profile's sessions live on its remote host's state.db, not on a local
// file the primary can open — so reads for it must route to the remote backend,
// not the local-disk fast path. These three helpers drive that (see
// interceptSessionReadForRemote).
function profileHasRemoteOverride(profile) {
  return profileHasRemoteConnection(readDesktopConnectionConfig(), profile)
}

function configuredRemoteProfileNames() {
  const config = readDesktopConnectionConfig()

  return Object.keys(config.profiles || {}).filter(name => profileHasRemoteConnection(config, name))
}

// True when the app is in app-global remote mode (Settings → "All profiles" →
// Remote/Cloud, or the env override): a SINGLE remote backend serves every
// profile via ?profile=. Cloud counts — it resolves to a remote backend (Q6).
// Distinct from per-profile overrides — here there's one host for all.
function globalRemoteActive() {
  if (process.env.HERMES_DESKTOP_REMOTE_URL) {
    return true
  }

  const mode = readDesktopConnectionConfig().mode

  return modeIsRemoteLike(mode) || mode === 'ssh'
}

// True when the PRIMARY profile's backend resolves to a remote/cloud host —
// i.e. resolveRemoteBackend(primaryProfileKey()) would return a descriptor
// rather than null. Mirrors that function's precedence (per-profile override →
// env → global) so a startHermes() failure can be classified as remote (never
// latch — transient, must stay retryable) vs local (latch to break install
// loops) BEFORE the throwing resolve/mint runs.
function primaryBackendIsRemote() {
  return Boolean(profileHasRemoteOverride(primaryProfileKey())) || globalRemoteActive()
}

// GET a profile's resolved backend (remote pool or local primary), parsed JSON.
async function fetchJsonForProfile(profile, path) {
  return requestJsonForProfile(profile, path, 'GET')
}

// Issue an arbitrary method against a profile's resolved backend, parsed JSON.
async function requestJsonForProfile(profile: string, path: string, method: string, body?: string) {
  const conn = await ensureBackend(profile)
  const url = `${conn.baseUrl}${path}`
  const opts = { method, body, timeoutMs: DEFAULT_FETCH_TIMEOUT_MS }

  if (conn.authMode === 'oauth') {
    // Native RFC 8252 flow: authenticate with the bearer token (cookieless)
    // when we hold one for this gateway; otherwise use the cookie partition.
    const nativeAt = await ensureNativeAccessToken(conn.baseUrl).catch(() => null)

    if (nativeAt) {
      return fetchJson(url, null, { ...opts, bearer: nativeAt, headers: conn.headers })
    }

    return fetchJsonViaOauthSession(url, { ...opts, headers: conn.headers })
  }

  return fetchJson(url, conn.token, { ...opts, headers: conn.headers })
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
  if (input.mode === 'ssh') {
    const sshConfig = normalizeSshConfig({
      mode: 'ssh',
      host: input.sshHost,
      user: input.sshUser,
      port: input.sshPort,
      keyPath: input.sshKeyPath,
      remoteHermesPath: input.sshRemoteHermesPath
    })

    if (!sshConfig) {
      return { reachable: false, sshError: 'unreachable', error: 'SSH host is required.' }
    }

    const ssh = createSshProbeConnection(
      { host: sshConfig.host, user: sshConfig.user, port: sshConfig.port, keyPath: sshConfig.keyPath },
      { rememberLog: sshRememberLog }
    )

    try {
      // One bounded retry on TIMEOUT only: a cold Windows backend's first
      // PowerShell exec can exceed the budget (observed live), and a timeout is
      // indeterminate — unlike auth/host-key/unreachable, which are verdicts.
      let attempt = 0

      for (;;) {
        try {
          await ssh.open()
          const platform: any = await detectRemotePlatform(ssh, sshConfig.remoteHermesPath || '')
          let hermesPath
          let hermesVersion
          let supported

          if (platform.os === 'Windows') {
            const runtime = platform
            hermesPath = runtime.hermesPath
            const inspection = await helper(ssh, runtime, 'inspect', [runtime.hermesPath])
            hermesVersion = inspection.version
            supported = inspection.supported
          } else {
            hermesPath = await remoteLifecycle.locateHermes(ssh, sshConfig.remoteHermesPath || '')
            hermesVersion = await remoteLifecycle.probeHermesVersion(ssh, hermesPath)
            supported = await remoteLifecycle.remoteSupportsSshOwnership(ssh, hermesPath)
          }

          if (!supported) {
            return {
              reachable: false,
              sshError: 'update-required',
              error: 'Update Hermes on the remote host before connecting with Desktop SSH.'
            }
          }

          return {
            reachable: true,
            sshError: null,
            error: null,
            remotePlatform: `${platform.os}/${platform.arch}`,
            remoteHermesPath: hermesPath,
            remoteHermesVersion: hermesVersion,
            host: sshConfig.user ? `${sshConfig.user}@${sshConfig.host}` : sshConfig.host
          }
        } catch (error: any) {
          if (error?.kind === 'timeout' && attempt === 0) {
            attempt += 1
            sshRememberLog('[ssh] test probe timed out once; retrying')

            continue
          }

          throw error
        }
      }
    } catch (error: any) {
      return { reachable: false, sshError: error.kind || 'unknown', error: error.message }
    } finally {
      try {
        await ssh.close()
      } catch {
        void 0
      }
    }
  }

  const config = coerceDesktopConnectionConfig(input, readDesktopConnectionConfig(), { persistToken: false })
  const key = connectionScopeKey(input.profile)
  // The block under test: a per-profile entry or the global remote. Coerce has
  // already normalized the URL and resolved token inheritance for the scope.
  const block = key ? config.profiles?.[key] || null : config.remote

  const wantRemote =
    modeIsRemoteLike(block?.mode) || (!key && modeIsRemoteLike(config.mode)) || (modeIsRemoteLike(input.mode) && block)

  // Test ``/api/status`` through the connection's real auth path. Self-hosted
  // gateways may protect it, and an anonymous success/failure would not prove
  // that the OAuth cookie/native bearer or configured token is reusable. For
  // a remote config we normalize the URL from the input; for local we fall
  // back to the resolved/started backend.
  let baseUrl
  let token = null
  let authMode = 'token'
  let testHeaders = {}

  if (wantRemote && block?.url) {
    baseUrl = normalizeRemoteBaseUrl(block.url)
    authMode = normAuthMode(block.authMode)
    testHeaders = decryptRemoteHeaders(block.headers)

    if (authMode !== 'oauth') {
      token = decryptDesktopSecret(block.token)
    }
  } else {
    const remote = (await resolveRemoteBackend(key)) || (await startHermes())
    baseUrl = remote.baseUrl
    token = remote.token
    authMode = normAuthMode(remote.authMode)
    testHeaders = remote.headers || {}
  }

  const status = (await fetchConnectionStatus(baseUrl, authMode, token, testHeaders)) as any

  // The HTTP status check above proves the backend is reachable, but the chat
  // surface only works once the renderer's live WebSocket to ``/api/ws``
  // connects — a separate transport with separate server-side guards (Host/
  // Origin, ws-ticket/token auth). Validating only the HTTP side produced a
  // false-positive "reachable" while the real boot still failed with "Could not
  // connect to Hermes gateway". Mirror the renderer's connect here so the test
  // reflects the full path the app actually uses.
  const wsUrl = await resolveTestWsUrl(baseUrl, authMode, token, {
    mintTicket: url => mintGatewayWsTicket(url, testHeaders)
  })

  // Skip the WS leg only when the runtime genuinely lacks a WebSocket (so an
  // older Electron/Node never fails the test spuriously); Electron's main
  // process ships a global WebSocket on every supported version.
  if (wsUrl && typeof globalThis.WebSocket === 'function') {
    const probe = await probeGatewayWebSocket(wsUrl, { WebSocketImpl: globalThis.WebSocket, headers: testHeaders })

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

async function fetchConnectionStatus(baseUrl, authMode, token, headers = {}) {
  const url = `${baseUrl}/api/status`

  if (authMode === 'oauth') {
    // Native PKCE bearer first, OAuth session cookies second — the same two
    // credentials real traffic uses, in the same order. A refresh failure is
    // NOT a silent downgrade to an anonymous probe: the cookie path is still
    // an authenticated request, and if neither credential works the probe
    // fails, which is the correct answer for a gateway we cannot reach with
    // the credentials we hold.
    const nativeAt = await ensureNativeAccessToken(baseUrl).catch(() => null)

    if (nativeAt) {
      return fetchJson(url, null, { timeoutMs: 8_000, bearer: nativeAt, headers })
    }

    return fetchJsonViaOauthSession(url, { timeoutMs: 8_000, headers })
  }

  return fetchJson(url, token, { timeoutMs: 8_000, headers })
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
  stopBackendChildImpl(child, { forceKillProcessTree, isWindows: IS_WINDOWS })
}

// Soft gateway-mode apply: tear down the primary without resetting boot UI or
// reloading the renderer. The shell stays up; the renderer wipes session lists
// (so skeletons retrigger) and re-dials. Distinct from hard re-home (profile
// switch / crash recovery), which still resets boot progress + reloads.
function resetHermesConnection({ soft = false } = {}) {
  backendStartFailure = null
  remoteReauthFailure = null
  remoteLiveness.clear()
  const hermesProcess = backendConnectionState.invalidate()
  stopBackendChild(hermesProcess)

  if (!soft) {
    resetBootProgressForReconnect()
  }
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
function broadcastConnectionsChanged(payload: { connectionId: string; reason: 'removed' | 'updated' }) {
  for (const win of BrowserWindow.getAllWindows()) {
    const { webContents } = win

    if (webContents && !webContents.isDestroyed()) {
      webContents.send('hermes:connections:changed', payload)
    }
  }
}

async function waitForBackendExit(child, timeoutMs = 5000) {
  if (!child || child.exitCode !== null || child.signalCode !== null) {
    return
  }

  const exited = () => child.exitCode !== null || child.signalCode !== null

  const wait = delay =>
    new Promise<void>(resolve => {
      if (exited()) {
        resolve()

        return
      }

      const timer = setTimeout(resolve, delay)
      child.once('exit', () => {
        clearTimeout(timer)
        resolve()
      })
    })

  await wait(timeoutMs)

  if (exited()) {
    return
  }

  try {
    if (IS_WINDOWS && Number.isInteger(child.pid)) {
      forceKillProcessTree(child.pid)
    } else if (Number.isInteger(child.pid)) {
      try {
        process.kill(-child.pid, 'SIGKILL')
      } catch {
        child.kill('SIGKILL')
      }
    } else {
      child.kill('SIGKILL')
    }
  } catch {
    return
  }

  // Await the escalation as well; do not let shutdown or failed adoption race
  // a still-running backend.
  await wait(1000)
}

// The profile the primary (window) backend runs as. readActiveDesktopProfile()
// returns the desktop's stored preference, or null when unset (legacy launch
// that defers to active_profile / default).
function primaryProfileKey() {
  return readActiveDesktopProfile() || 'default'
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
    requestMethod: request?.method,
    requestPath: request?.path
  }
}

// Resolve a backend connection for the given profile, per the routing table in
// resolveProfileBackendRoute(). An empty / unknown profile resolves to the
// primary, so legacy callers are unchanged.
async function ensureBackend(profile) {
  const key = profile && String(profile).trim() ? String(profile).trim() : primaryProfileKey()

  profileDeletionGate.assertCanStart(key)

  const route = resolveProfileBackendRoute(key, profileRouteOptions(key))

  if (route.backend === 'primary') {
    const connection = await startHermes()
    setWslBridgeProfileState(key, connection.mode !== 'remote')

    // A shared backend still owes the caller its profile scope, so renderer-side
    // WebSocket, filesystem, and cache routing target the selected profile.
    // `sharedPrimary` marks this as the shared-primary route: pooled backends
    // also carry `profile`, so only this descriptor gets the flag.
    return route.descriptorProfile
      ? { ...connection, profile: route.descriptorProfile, sharedPrimary: true }
      : connection
  }

  // A backend for this key may still be dying (idle reap, LRU eviction, a
  // just-finished delete). Wait for its bounded exit before reusing or
  // spawning, so two children never share one profile's HERMES_HOME.
  const stopping = poolStopper.inFlight(key)

  if (stopping) {
    await stopping
  }

  const existing = backendPool.get(key)

  if (existing) {
    existing.lastActiveAt = Date.now()
    const connection = await existing.connectionPromise
    setWslBridgeProfileState(key, connection.mode !== 'remote')

    return connection
  }

  evictLruPoolBackends(POOL_MAX_BACKENDS - 1)

  const entry = {
    process: null,
    port: null,
    token: null,
    connectionPromise: null,
    lastActiveAt: Date.now(),
    remoteBaseUrl: null
  }

  entry.connectionPromise = spawnPoolBackend(key, entry).catch(async error => {
    // Land the failure in desktop.log: without this a spawn that dies before
    // its child exists (guard rejection, runtime resolution) leaves no trace
    // beyond renderer-side rejections users never see in a bundle.
    rememberLog(
      `Hermes backend for profile "${key}" failed to start: ${error instanceof Error ? error.message : String(error)}`
    )

    if (backendPool.get(key) === entry) {
      backendPool.delete(key)
    }

    stopBackendChild(entry.process)
    await waitForBackendExit(entry.process)
    throw error
  })
  backendPool.set(key, entry)
  startPoolIdleReaper()

  const connection = await entry.connectionPromise
  setWslBridgeProfileState(key, connection.mode !== 'remote')

  return connection
}

// ── Registry-scoped backends (multi-connection, PR 2 of the campaign) ──────
// Resolve a backend for (connectionId, profile) against the v2 registry.
// The LOCAL connection routes through ensureBackend() when the v1 route is
// itself local (so every single-source path stays byte-identical), and forces
// a genuinely-local child when the v1 mode says remote; non-local connections
// pool under the composite key from backendScopeKey() and reuse the same pool
// entry lifecycle (LRU, idle reaper, touch) as per-profile local backends.
async function ensureRegistryBackend(connectionId, profile) {
  const registry = readDesktopConnectionsRegistry()
  const id = String(connectionId || '').trim() || registry.primary
  const source = registry.connections.find(c => c.id === id)

  if (!source) {
    throw new Error(`No connection with id "${id}".`)
  }

  if (source.kind === 'local') {
    // The registry's 'local' entry means THIS machine's runtime — always.
    // ensureBackend() follows the v1 routing table, which resolves to a
    // REMOTE descriptor when the v1 global mode is remote (or the profile
    // has its own remote override). A migrated remote-mode user would then
    // see the roster's "This device" rows enumerate + dial the remote box
    // (every profile duplicated, -slug handles forced). Delegate only when
    // the v1 route is genuinely local; otherwise spawn/reuse a forced-local
    // child pooled under the composite 'conn:local::<profile>' key so it
    // can't collide with the v1 remote descriptor cached at the bare key.
    const profileKey = String(profile ?? '').trim() || 'default'

    profileDeletionGate.assertCanStart(profileKey)

    const localRoute = resolveRegistryLocalRoute(profileKey, {
      globalRemote: globalRemoteActive(),
      profileRemoteOverride: Boolean(profileHasRemoteOverride(profileKey))
    })

    if (localRoute.delegate) {
      return ensureBackend(profile)
    }

    const stoppingLocal = poolStopper.inFlight(localRoute.poolKey)

    if (stoppingLocal) {
      await stoppingLocal
    }

    const existingLocal = backendPool.get(localRoute.poolKey)

    if (existingLocal) {
      existingLocal.lastActiveAt = Date.now()

      return existingLocal.connectionPromise
    }

    evictLruPoolBackends(POOL_MAX_BACKENDS - 1)

    const localEntry = {
      process: null,
      port: null,
      token: null,
      connectionPromise: null,
      lastActiveAt: Date.now(),
      remoteBaseUrl: null
    }

    localEntry.connectionPromise = spawnPoolBackend(profileKey, localEntry, {
      forceLocal: true,
      poolKey: localRoute.poolKey
    }).catch(async error => {
      // Same trace rule as the v1 pool path: a forced-local child whose spawn
      // rejects before the child exists must still land in desktop.log.
      rememberLog(
        `Hermes backend for profile "${profileKey}" (forced-local) failed to start: ${error instanceof Error ? error.message : String(error)}`
      )

      if (backendPool.get(localRoute.poolKey) === localEntry) {
        backendPool.delete(localRoute.poolKey)
      }

      stopBackendChild(localEntry.process)
      await waitForBackendExit(localEntry.process)
      throw error
    })
    backendPool.set(localRoute.poolKey, localEntry)
    startPoolIdleReaper()

    return localEntry.connectionPromise
  }

  const key = backendScopeKey(id, profile)
  const existing = backendPool.get(key)

  if (existing) {
    existing.lastActiveAt = Date.now()

    return existing.connectionPromise
  }

  evictLruPoolBackends(POOL_MAX_BACKENDS - 1)

  const entry = {
    process: null,
    port: null,
    token: null,
    connectionPromise: null,
    lastActiveAt: Date.now(),
    remoteBaseUrl: null
  }

  entry.connectionPromise = connectRegistryBackend(source, profile, key, entry).catch(error => {
    if (backendPool.get(key) === entry) {
      backendPool.delete(key)
    }

    throw error
  })
  backendPool.set(key, entry)
  startPoolIdleReaper()

  return entry.connectionPromise
}

// Dial a non-local registry connection for one profile. Never spawns a local
// child (entry.process stays null — stopPoolBackend/evict already tolerate
// that shape from remote per-profile overrides).
async function connectRegistryBackend(source, profile, key, poolEntry) {
  const profileKey = String(profile ?? '').trim() || 'default'

  if (source.kind === 'ssh') {
    // The composite key doubles as the ssh scope so each (connection, profile)
    // pair owns its own tunnel + remote dashboard; the profile that re-homes
    // the REMOTE process is the entry's remoteProfile or the requested one —
    // never the composite string.
    const sshConfig = normalizeSshConfig({
      mode: 'ssh',
      host: source.host,
      user: source.user,
      port: source.port,
      keyPath: source.keyPath,
      remoteHermesPath: source.remoteHermesPath,
      remoteProfile: source.remoteProfile || (profileKey === 'default' ? '' : profileKey)
    })

    if (!sshConfig) {
      throw new Error(`SSH connection "${source.label}" has no host configured.`)
    }

    const connection = await bootstrapSshConnection(
      key,
      sshConfig,
      decryptDesktopSecret(source.token),
      `registry:${source.id}`
    )

    poolEntry.remoteBaseUrl = connection.baseUrl

    return {
      ...connection,
      profile: profileKey,
      connectionId: source.id,
      // The remote process runs as this profile; the desktop-side profile key
      // is only the routing label. hermes:api uses it to translate explicit
      // self-profile query filters into the backend's namespace.
      remoteProfile: sshConfig.remoteProfile || '',
      logs: hermesLog.slice(-80),
      ...getWindowState()
    }
  }

  // remote / cloud: one gateway host serves every profile of that source,
  // scoped per request — the descriptor carries the profile + connectionId so
  // renderer-side WS minting and REST scoping target the right agent.
  const token = source.authMode === 'oauth' ? null : decryptDesktopSecret(source.token)

  const connection = await buildRemoteConnection(
    source.url,
    normAuthMode(source.authMode),
    token,
    `registry:${source.id}`,
    undefined,
    source.kind === 'cloud' ? 'cloud' : 'url',
    undefined,
    source.headers
  )

  await waitForHermes(connection.baseUrl, connection.token, undefined, connection.authMode, connection.headers)
  poolEntry.remoteBaseUrl = connection.baseUrl

  return {
    ...connection,
    profile: profileKey,
    connectionId: source.id,
    // One host, many profiles: REST paths must carry ?profile= (same contract
    // as the global-remote shared-primary route).
    sharedRemote: true,
    logs: hermesLog.slice(-80),
    ...getWindowState()
  }
}

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
function touchPoolBackend(profile) {
  for (const key of poolTouchKeys(profile)) {
    const entry = backendPool.get(key)

    if (entry) {
      entry.lastActiveAt = Date.now()

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
function evictLruPoolBackends(keep) {
  const evictions = selectPoolEvictions(backendPool.entries(), Math.max(0, keep), Date.now(), POOL_KEEPALIVE_FRESH_MS)

  for (const profile of evictions) {
    rememberLog(`Evicting idle profile backend "${profile}" (LRU cap ${POOL_MAX_BACKENDS})`)
    stopPoolBackend(profile)
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
// `opts.forceLocal` skips remote resolution entirely (the registry 'local'
// entry means THIS machine regardless of the v1 routing table); `opts.poolKey`
// is the backendPool key when it differs from the profile name (composite
// registry scopes) so the exit/error cleanup evicts the right entry.
async function spawnPoolBackend(profile, entry, opts: { forceLocal?: boolean; poolKey?: string } = {}) {
  const poolKey = opts.poolKey || profile

  await reapOrphanedBackendsOnce()
  profileDeletionGate.assertCanStart(profile)

  // A profile may point at its OWN remote backend (connection.json
  // `profiles[name]`), or inherit the app-wide remote (env / global settings).
  // In either case there is no local child to spawn — we just verify the
  // remote is reachable and hand back its connection descriptor. The pool
  // entry keeps `entry.process === null`, which stopPoolBackend/evict already
  // tolerate.
  const remote = opts.forceLocal ? null : await resolveRemoteBackend(profile)
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

  const token = crypto.randomBytes(32).toString('base64url')

  // Same update mutual exclusion as the primary window's waitForLocalStart
  // (#73822): pool backends spawn from the same venv, so an ungated respawn
  // during applyUpdates' critical section re-locks the venv and trips the
  // venv-blocker preflight. No boot-progress UI here — pool backends boot
  // silently for background profiles — so we only log while parked.
  {
    let poolAnnounced = false

    await waitForUpdateClearance(updateGateDeps(), {
      onWaitTick: reason => {
        if (!poolAnnounced) {
          poolAnnounced = true
          rememberLog(`[updates] update in progress (${reason}); deferring pool backend start for profile "${profile}"`)
        }
      },
      pollMs: UPDATE_WAIT_POLL_MS,
      timeoutMs: UPDATE_WAIT_TIMEOUT_MS
    })
  }

  profileDeletionGate.assertCanStart(profile)

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
        // Exact parent identity lets the backend self-exit after an unclean
        // Desktop death without mistaking a reused PID for its owner. If the
        // optional marker probe fails, retain legacy PID-only tracking.
        ...parentIdentityEnv,
        HERMES_WEB_DIST: webDist,
        ...(readyFile ? { HERMES_DESKTOP_READY_FILE: readyFile } : {})
      },
      shell: backend.shell,
      stdio: ['ignore', 'pipe', 'pipe']
    })
  )

  entry.process = child
  entry.token = token
  // Buffer stdout+stderr from the instant of spawn (#93608): an early crash's
  // traceback must survive into the claim error and the before-ready exit
  // message instead of a bare exit code. rememberLog attaches later, after
  // the claim, and would miss anything printed before it.
  const outputTail = createBackendOutputTail()
  outputTail.attach(child)
  await claimBackendChild(child, `${backend.command} ${backend.args.join(' ')}`, profile, backendNonce, outputTail)

  child.stdout.on('data', rememberLog)
  child.stderr.on('data', rememberLog)

  let ready = false
  let rejectStart = null

  const startFailed = new Promise((_resolve, reject) => {
    rejectStart = reject
  })

  child.once('error', error => {
    rememberLog(`Hermes backend for profile "${profile}" failed to start: ${error.message}`)
    releaseBackendChild(child)
    backendPool.delete(poolKey)
    rejectStart?.(error)
  })
  child.once('exit', (code, signal) => {
    rememberLog(`Hermes backend for profile "${profile}" exited (${signal || code})`)
    releaseBackendChild(child)
    backendPool.delete(poolKey)

    if (!ready) {
      rejectStart?.(
        new Error(
          `Hermes backend for profile "${profile}" exited before it became ready (${signal || code}).${outputTail.describe()}`
        )
      )
    }
  })

  // Discover the ephemeral port the child bound to
  const port = await Promise.race([
    waitForDashboardPortAnnouncement(child, { describeOutputTail: () => outputTail.describe(), readyFile }),
    startFailed
  ])

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

  // Verify the WebSocket session token before declaring backend ready.
  // HTTP /api/status can pass while WS auth fails (separate transport, separate guards).
  const wsUrl = `ws://127.0.0.1:${port}/api/ws?token=${encodeURIComponent(authToken)}`
  const wsProbe = await probeGatewayWebSocket(wsUrl, { WebSocketImpl: globalThis.WebSocket })

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
  waitForExit: child => waitForBackendExit(child)
})

function stopPoolBackend(profile) {
  return poolStopper.stop(profile)
}

async function teardownPoolBackendAndWait(profile) {
  await Promise.all(localProfilePoolKeys(profile).map(key => poolStopper.stop(key)))
}

function stopAllPoolBackends() {
  return poolStopper.stopAll()
}

const backendShutdown = createBackendShutdownCoordinator(async () => {
  const primary = backendConnectionState.invalidate()

  stopBackendChild(primary)
  const pooledStops = stopAllPoolBackends()

  if (poolIdleReaper) {
    clearInterval(poolIdleReaper)
    poolIdleReaper = null
  }

  await Promise.all([waitForBackendExit(primary), pooledStops])
})

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

async function startHermes() {
  // Only the single-instance lock holder may reap/spawn/claim the desktop
  // backend. A lock-losing instance must stay inert even if some path reaches
  // here (e.g. the deferred-quit window before `ready`): its reapOrphans()
  // otherwise SIGTERMs the running instance's live backend (#87295).
  if (!isPrimaryInstance) {
    rememberLog('[boot] non-primary instance: skipping backend machinery')
    throw new Error('Hermes Desktop is already running in another window.')
  }

  await reapOrphanedBackendsOnce()

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

  // A confirmed remote reauth rejection is terminal until the user signs in.
  // Short-circuiting here keeps the boot-failure overlay latched and its
  // "Sign in" button clickable, instead of re-driving boot on every retry.
  if (remoteReauthFailure) {
    throw remoteReauthFailure
  }

  // E2E: simulate a boot failure without breaking the real backend. The boot
  // progresses a few steps, then fails with the given error message.
  if (BOOT_FAKE_ERROR) {
    await advanceBootProgress('backend.resolve', 'Resolving Hermes backend', 8)
    const error = new Error(BOOT_FAKE_ERROR) as any
    error.isBootstrapFailure = true
    bootstrapFailure = error
    throw error
  }

  const existingConnectionPromise = backendConnectionState.getPromise()

  if (existingConnectionPromise) {
    return existingConnectionPromise
  }

  const connectionAttempt = backendConnectionState.startAttempt()
  const primaryProfile = primaryProfileKey()

  // Legacy path callers without an explicit profile belong to the primary
  // window backend. Profile-scoped callers still pass their key directly.
  setActiveGatewayProfile(primaryProfile)

  // Classify this boot BEFORE the throwing resolve/mint runs: a remote failure
  // must NOT latch (it's transient — see shouldLatchBackendStartFailure), while
  // a local failure latches to break install-restart loops.
  let attemptedRemote = primaryBackendIsRemote()

  const connectionPromise = (async () => {
    const connectRemote = async remote => {
      // resolveRemote() may take arbitrarily long (settings resolve / ws-ticket
      // mint). If a newer attempt started meanwhile (e.g. the user switched
      // remotes and Apply invalidated this attempt), bail before probing.
      if (!backendConnectionState.isCurrentAttempt(connectionAttempt)) {
        throw new Error('Hermes backend start was superseded by a newer connection attempt.')
      }

      await advanceBootProgress('backend.remote', `Connecting to remote Hermes backend at ${remote.baseUrl}`, 24)
      await waitForHermes(remote.baseUrl, remote.token, undefined, remote.authMode, remote.headers)

      // Second async boundary: the health probe itself can outlive the
      // attempt. A late success here must not publish a stale descriptor.
      if (!backendConnectionState.isCurrentAttempt(connectionAttempt)) {
        throw new Error('Hermes backend start was superseded by a newer connection attempt.')
      }

      updateBootProgress({
        phase: 'backend.ready',
        message: 'Remote Hermes backend is ready',
        progress: 94,
        running: true,
        error: null
      })

      return createPrimaryRemoteConnection(remote, hermesLog.slice(-80), getWindowState())
    }

    await advanceBootProgress('backend.resolve', 'Resolving Hermes backend', 8)
    // Resolve for the desktop's primary profile so a per-profile remote
    // override on the active profile is honored (falls back to env / global).

    // GUI launches (Finder/Dock, desktop launchers) inherit a minimal PATH
    // that skips the user's shell profiles. Merge the login-shell PATH into
    // process.env BEFORE resolving the runtime or spawning the backend, so
    // both the Electron-side resolvers and the whole backend subtree (tool
    // availability checks, stdio MCP servers) can find Homebrew-, nvm-, and
    // ~/.local/bin-installed CLIs. Single-flight with the whenReady warmup;
    // failure-hardened — a broken shell profile never blocks boot.
    const loginShellPath = await ensureLoginShellPath()

    if (loginShellPath.applied) {
      rememberLog('[env] merged login-shell PATH into process.env for backend spawn')
    } else if (loginShellPath.reason && !['win32', 'unchanged'].includes(loginShellPath.reason)) {
      rememberLog(`[env] login-shell PATH resolution unavailable (${loginShellPath.reason}); keeping inherited PATH`)
    }

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

    const setup = await runPrimaryBackendStartup({
      connectRemote,
      ensureLocalRuntime: ensureRuntime,
      prepareLocalBackend: async () => {
        await advanceBootProgress('backend.runtime', 'Resolving Hermes runtime', 28)

        return resolveHermesBackend(backendArgs)
      },
      resolveRemote: () => {
        // Classify immediately before each throwing resolve. This callback runs
        // both for an already-saved remote and after first-run remote Apply.
        attemptedRemote = primaryBackendIsRemote()

        return resolveRemoteBackend(primaryProfile)
      },
      waitForDecision: waitForFirstRunSetupChoice,
      // Mutual exclusion with an in-app update (#50238). Remote connections
      // return before this waiter; local starts park until the updater exits.
      waitForLocalStart: waitForUpdateToFinish
    })

    if (setup.kind === 'remote') {
      // Paths from the remote backend belong to a host the Windows desktop
      // cannot open via wsl.exe — disable WSL path bridging so native dialogs
      // and file panels don't spawn wsl.exe (or the interactive install prompt
      // on WSL-less machines) for unresolvable paths. (#66433)
      setWslBridgeProfileState(primaryProfile, false)

      return setup.connection
    }

    // Local WSL backend — paths are bridgeable.
    setWslBridgeProfileState(primaryProfile, true)

    const backend = setup.backend
    // Route old runtimes (no `serve`) through the legacy `dashboard --no-open`.
    backend.args = getBackendArgsForRuntime(backend)
    const hermesCwd = resolveHermesCwd()
    const webDist = resolveWebDist()
    const readyFile = backend.readyFile ? makeDashboardReadyFile() : null

    await advanceBootProgress('backend.spawn', `Starting Hermes backend via ${backend.label}`, 84)
    rememberLog(`Starting Hermes backend via ${backend.label}`)

    const profile = primaryProfileKey()
    const parentStartMarker = await desktopParentStartMarker()
    const backendNonce = crypto.randomBytes(16).toString('hex')
    const parentIdentityEnv = parentWatchdogEnv(process.pid, parentStartMarker, backendNonce)

    const hermesProcess = spawn(
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
          // Exact parent identity lets the backend self-exit after an unclean
          // Desktop death without mistaking a reused PID for its owner. If the
          // optional marker probe fails, retain legacy PID-only tracking.
          ...parentIdentityEnv,
          HERMES_WEB_DIST: webDist,
          ...(readyFile ? { HERMES_DESKTOP_READY_FILE: readyFile } : {})
        },
        shell: backend.shell,
        stdio: ['ignore', 'pipe', 'pipe']
      })
    )

    // Buffer stdout+stderr from the instant of spawn (#93608): an early
    // crash's traceback must survive into the claim error and the
    // before-ready exit message shown by the boot UI. rememberLog attaches
    // later, after the claim, and would miss anything printed before it.
    const primaryOutputTail = createBackendOutputTail()
    primaryOutputTail.attach(hermesProcess)
    await claimBackendChild(
      hermesProcess,
      `${backend.command} ${backend.args.join(' ')}`,
      profile,
      backendNonce,
      primaryOutputTail
    )
    const processOwner = backendConnectionState.attachProcess(connectionAttempt, hermesProcess)

    if (!processOwner) {
      stopBackendChild(hermesProcess)
      await waitForBackendExit(hermesProcess)
      releaseBackendChild(hermesProcess)
      throw new Error('Hermes backend start was superseded by a newer connection attempt.')
    }

    hermesProcess.stdout.on('data', rememberLog)
    hermesProcess.stderr.on('data', rememberLog)
    let backendReady = false
    let rejectBackendStart = null

    const backendStartFailed = new Promise((_resolve, reject) => {
      rejectBackendStart = reject
    })

    hermesProcess.once('error', error => {
      releaseBackendChild(hermesProcess)

      if (!backendConnectionState.clearForCurrentProcess(processOwner)) {
        rememberLog(`Ignoring stale Hermes backend error: ${error.message}`)
        rejectBackendStart?.(new Error('Hermes backend start was superseded by a newer connection attempt.'))

        return
      }

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
      sendBackendExit({ code: null, signal: null, error: error.message })
      rejectBackendStart?.(error)
    })
    hermesProcess.once('exit', (code, signal) => {
      releaseBackendChild(hermesProcess)

      if (!backendConnectionState.clearForCurrentProcess(processOwner)) {
        rememberLog(`Ignoring stale Hermes backend exit (${signal || code})`)

        if (!backendReady) {
          rejectBackendStart?.(new Error('Hermes backend start was superseded by a newer connection attempt.'))
        }

        return
      }

      rememberLog(`Hermes backend exited (${signal || code})`)
      sendBackendExit({ code, signal })

      if (!backendReady) {
        const message = `Hermes backend exited before it became ready (${signal || code}).${primaryOutputTail.describe()}`
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
      waitForDashboardPortAnnouncement(hermesProcess, {
        describeOutputTail: () => primaryOutputTail.describe(),
        readyFile
      }),
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
      childAlive: () => hermesProcess.exitCode === null && !hermesProcess.killed,
      rememberLog
    })

    // Verify the WebSocket session token before declaring backend ready.
    const wsUrl = `ws://127.0.0.1:${port}/api/ws?token=${encodeURIComponent(authToken)}`
    const wsProbe = await probeGatewayWebSocket(wsUrl, { WebSocketImpl: globalThis.WebSocket })

    if (!wsProbe.ok) {
      throw new Error(
        `Local Hermes backend is HTTP-reachable but the WebSocket (/api/ws) rejected the session token: ${wsProbe.reason}`
      )
    }

    updateBootProgress({
      phase: 'backend.ready',
      message: 'Hermes backend is ready. Finalizing desktop startup',
      progress: 94,
      running: true,
      error: null
    })

    // A successful boot (including a soft restart that the repair-guard
    // chose over a hard reinstall, see #74874) means any in-flight repair
    // attempt counter has been honoured — reset it so the next genuine
    // failure starts fresh from attempt 1 instead of inheriting the
    // accumulated count of the resolved episode.
    bootstrapRepairAttempt = 0

    return {
      baseUrl,
      mode: 'local',
      source: 'local',
      authMode: 'token',
      token: authToken,
      wsUrl,
      logs: hermesLog.slice(-80),
      ...getWindowState()
    }
  })().catch(async error => {
    if (!backendConnectionState.clearPromiseForAttempt(connectionAttempt)) {
      throw error
    }

    const failedProcess = backendConnectionState.invalidate()
    stopBackendChild(failedProcess)
    await waitForBackendExit(failedProcess)

    if (error instanceof FirstRunSetupResetError) {
      throw error
    }

    const message = error instanceof Error ? error.message : String(error)
    const hostKeyChanged = isHostKeyChangedBootFailure(error)

    // Carry structured Cloud-down metadata through the boot-progress / IPC
    // boundary when present, so the renderer overlay can key on it rather than
    // re-classifying the message string. main owns classification; the renderer
    // only consumes the structured result (#85335).
    const isCloudBackendDown = Boolean(error && typeof error === 'object' && (error as any).isCloudBackendDown === true)

    const statusCode = Number(
      error && typeof error === 'object' && Number.isInteger((error as any).statusCode)
        ? (error as any).statusCode
        : NaN
    )

    // Only latch LOCAL boot failures. A remote failure (lapsed session / mint
    // timeout / host briefly unreachable across sleep) is transient and has no
    // child 'exit' handler to clear the cache — latching it would wedge the app
    // on "session expired" until a full restart, defeating reconnect, the
    // "Sign out & sign in" reload, and the wake-recovery revalidate path.
    if (shouldLatchBackendStartFailure({ attemptedRemote })) {
      backendStartFailure = error instanceof Error ? error : new Error(message)
    }

    // A host-key CHANGE is the terminal exception among remote failures: SSH
    // fails closed until the user verifies the change and clears the stale
    // known_hosts entry, so retrying re-drives the identical doomed boot (one
    // bundle showed 157 consecutive failures over 2.5h). Latch it like a local
    // failure — reset/repair/apply-config clear the latch after the user fixes
    // known_hosts.
    if (shouldLatchHostKeyChangedFailure({ attemptedRemote, isReauth: false, isHostKeyChanged: hostKeyChanged })) {
      backendStartFailure = error instanceof Error ? error : new Error(message)
    }

    // A confirmed reauth rejection latches separately: it can't self-heal, and
    // leaving it unlatched hides the overlay's "Sign in" button on every retry.
    if (shouldLatchRemoteReauthFailure({ attemptedRemote, isReauth: isReauthRequiredError(error) })) {
      remoteReauthFailure = error instanceof Error ? error : new Error(message)
    }

    updateBootProgress(
      {
        error: message,
        isCloudBackendDown: isCloudBackendDown || undefined,
        message: `Desktop boot failed: ${message}`,
        phase: 'backend.error',
        // Renderer contract for the self-heal loop (#82679): a transient
        // REMOTE failure (dropped SSH/HTTP registered connection, mint
        // timeout) is retryable — the renderer re-attempts the boot with
        // bounded backoff. Local failures, confirmed reauth rejections, and
        // host-key changes are not: those end in the recovery overlay /
        // sign-in affordance.
        retryable: isRetryableRemoteBootFailure({
          attemptedRemote,
          isReauth: isReauthRequiredError(error),
          isHostKeyChanged: hostKeyChanged
        }),
        running: false,
        statusCode: Number.isInteger(statusCode) ? statusCode : undefined
      },
      { allowDecrease: true }
    )
    throw error
  })

  backendConnectionState.setPromise(connectionAttempt, connectionPromise)

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
  installBrowserNavGestures(win)

  // Claim Ctrl/Cmd+F in the main process — on Pop!_OS / GNOME-based Linux
  // distros the Ctrl+F keydown does not reach the renderer's `view.findInPage`
  // binding (#81727). Routing it through `before-input-event` forwards the
  // intent at the earliest observable point. macOS / Windows keep the
  // renderer's own rebindable keybind, so the hook is Linux-only: installing
  // it elsewhere would make Ctrl/Cmd+F un-rebindable and double-open.
  if (process.platform === 'linux') {
    installFindShortcut(win)
  }

  if (zoom) {
    installZoomShortcuts(win)
    // Re-apply persisted zoom on show/restore/resize/cross-display move
    // (Chromium can drop webContents zoom after these window transitions), on
    // EVERY full load — not once, since crash recovery reloads and would
    // outlive a spent `once` listener (#46429) — and after in-page navigation,
    // where Chromium applies the target hash route's own per-URL zoom record
    // (see installZoomReassertOnNavigation; #48658, #38854, #79863).
    const reassertZoom = () => restorePersistedZoomLevel(win)

    installZoomReassertOnWindowEvents(win, reassertZoom)
    installZoomReassertOnNavigation(win.webContents, reassertZoom)
  }

  installContextMenuBridge(win)
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

// Every window we open starts with `show: false` so the renderer's first themed
// paint lands before it appears, and `ready-to-show` is what reveals it.
// Electron 40 can drop that event entirely (electron/electron#51972) on
// Linux/Wayland, remote displays and VMs, leaving the window hidden forever even
// though the renderer finished loading. Keep the themed path as the preferred
// reveal, then fall back a few seconds after the renderer loads. `show` and
// `onRevealed` carry the caller's reveal action and post-visible work; whichever
// path wins runs them exactly once.
function wireWindowReveal(win, { show, onRevealed }: { show?: () => void; onRevealed?: () => void } = {}) {
  const controller = createWindowRevealController(
    {
      isDestroyed: () => win.isDestroyed(),
      isVisible: () => win.isVisible(),
      show: show ?? (() => win.show())
    },
    { onRevealed }
  )

  win.once('ready-to-show', controller.reveal)
  win.webContents.once('did-finish-load', controller.scheduleFallback)
  win.on('closed', controller.dispose)

  return controller
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

function spawnSecondaryWindow({ sessionId, watch }: { sessionId?: string; watch?: boolean } = {}) {
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
    ...chatWindowSurfaceOptions(),
    icon,
    // Don't show until the renderer's first themed paint is ready. macOS
    // `vibrancy` ignores `backgroundColor` and paints a translucent OS
    // material (which follows the OS appearance, not the app theme), so a
    // dark-themed app on a light-mode Mac flashes white until the renderer
    // covers it. ready-to-show fires after the boot-time paint in
    // themes/context.tsx, so the window appears already themed.
    show: false,
    webPreferences: chatWindowWebPreferences(PRELOAD_PATH)
  })

  // Chat-surface registration: applyWindowTranslucency swaps this window's
  // backing between opaque-themed and alpha-0 when glass toggles.
  translucencyBackedWindows.add(win)

  if (IS_MAC) {
    win.setWindowButtonPosition?.(WINDOW_BUTTON_POSITION)
  }

  wireWindowReveal(win)

  win.on('enter-full-screen', () => sendWindowStateChanged(true))
  win.on('leave-full-screen', () => sendWindowStateChanged(false))

  streamThrottle.register(win)
  wireCommonWindowHandlers(win, zoomWiringForWindowKind('chat'))
  attachRendererConsoleCapture(win, 'session-window', rememberLog)

  // Renderer lifecycle diagnostics + recovery (#81290): a dead session-window
  // renderer used to log nothing and stay black; now it logs with its window
  // kind and reloads under the shared crash-loop budget, exactly like the
  // primary window, without touching any other window.
  installWindowRendererLifecycle(win, {
    kind: 'secondary',
    callbacks: {
      log: rememberLog,
      reload: () => {
        win.webContents.reload()
      }
    },
    reloadWindowMs: RENDERER_RELOAD_WINDOW_MS,
    reloadMax: RENDERER_RELOAD_MAX,
    recentReloadTimesRef: rendererReloadTimesRef
  })

  loadWindowUrl(
    win,
    buildSessionWindowUrl(sessionId, {
      devServer: DEV_SERVER,
      rendererIndexPath: DEV_SERVER ? undefined : resolveRendererIndex(),
      watch
    }),
    'Session window'
  )

  return win
}

// Open (or focus) a standalone window for a single chat session.
function createSessionWindow(sessionId, { watch = false } = {}) {
  return sessionWindows.openOrFocus(sessionId, () => spawnSecondaryWindow({ sessionId, watch }))
}

// Popped-out in-app Browser: same webview + address bar as a docked Browser
// tab, in its own OS window. One window per tab id (re-open focuses); closing
// it tells the other renderers so they can dock the tab again.
const browserWindows = createSessionWindowRegistry()

function notifyBrowserPopoutClosed(tabId) {
  if (typeof tabId !== 'string' || !tabId) {
    return
  }

  for (const other of BrowserWindow.getAllWindows()) {
    if (!other.isDestroyed()) {
      other.webContents.send('hermes:browser-popout:closed', tabId)
    }
  }
}

function spawnBrowserWindow(tabId) {
  const icon = getAppIconPath()

  const win = new BrowserWindow({
    width: BROWSER_WINDOW_WIDTH,
    height: BROWSER_WINDOW_HEIGHT,
    minWidth: BROWSER_WINDOW_MIN_WIDTH,
    minHeight: BROWSER_WINDOW_MIN_HEIGHT,
    title: 'Hermes',
    titleBarStyle: 'hidden',
    titleBarOverlay: getTitleBarOverlayOptions(),
    trafficLightPosition: IS_MAC ? WINDOW_BUTTON_POSITION : undefined,
    ...chatWindowSurfaceOptions(),
    icon,
    show: false,
    webPreferences: chatWindowWebPreferences(PRELOAD_PATH)
  })

  translucencyBackedWindows.add(win)

  if (IS_MAC) {
    win.setWindowButtonPosition?.(WINDOW_BUTTON_POSITION)
  }

  wireWindowReveal(win)

  win.on('enter-full-screen', () => sendWindowStateChanged(true))
  win.on('leave-full-screen', () => sendWindowStateChanged(false))

  streamThrottle.register(win)
  wireCommonWindowHandlers(win, zoomWiringForWindowKind('chat'))
  attachRendererConsoleCapture(win, 'browser-window', rememberLog)

  installWindowRendererLifecycle(win, {
    kind: 'browser',
    callbacks: {
      log: rememberLog,
      reload: () => {
        win.webContents.reload()
      }
    },
    reloadWindowMs: RENDERER_RELOAD_WINDOW_MS,
    reloadMax: RENDERER_RELOAD_MAX,
    recentReloadTimesRef: rendererReloadTimesRef
  })

  win.on('closed', () => notifyBrowserPopoutClosed(tabId))

  loadWindowUrl(
    win,
    buildBrowserWindowUrl(tabId, {
      devServer: DEV_SERVER,
      rendererIndexPath: DEV_SERVER ? undefined : resolveRendererIndex()
    }),
    'Browser window'
  )

  return win
}

function createBrowserWindow(tabId) {
  return browserWindows.openOrFocus(tabId, () => spawnBrowserWindow(tabId))
}

// Additional full "instance" windows — peers of the primary that render the
// COMPLETE app (sidebar, routing, its own draft) against the shared backend, so
// a user can run multiple GUI windows at once (⌘⇧N / the "New Window" palette
// command). Unlike the compact session windows they carry no `?win` flag; a
// separate `peer=1` marker prevents them from replaying app-launch source
// restoration after joining that shared backend. The primary mainWindow stays
// the notification / deep-link / pet-overlay anchor and
// is NOT tracked here. The set holds a strong reference so an open peer isn't
// garbage-collected, and drops it on close.
const instanceWindows = new Set<any>()

// Cascade a new instance off whichever window spawned it so it doesn't land
// exactly on top of its source. Falls back to the persisted primary geometry
// when there's no live source window (e.g. all windows closed on macOS). The
// pure cascade math lives in session-windows.ts (instanceWindowBounds).
function nextInstanceBounds() {
  const source = BrowserWindow.getFocusedWindow() || mainWindow
  const fallback = computeWindowOptions(readWindowState(), screen.getAllDisplays())
  const base = source && !source.isDestroyed() ? source.getBounds() : null

  return instanceWindowBounds(base, fallback)
}

// Open a new full-chrome instance window. Mirrors createWindow()'s window
// options (shared chatWindowWebPreferences + streamThrottle registration so a
// streamed answer never stalls in the background) but is a peer, not the
// primary: it never overwrites the mainWindow global, doesn't start the backend
// (the renderer's getConnection() joins the already-running one), and loads the
// plain renderer URL so the full app renders.
function createInstanceWindow() {
  const icon = getAppIconPath()

  const win = new BrowserWindow({
    ...nextInstanceBounds(),
    minWidth: WINDOW_MIN_WIDTH,
    minHeight: WINDOW_MIN_HEIGHT,
    title: 'Hermes',
    titleBarStyle: 'hidden',
    titleBarOverlay: getTitleBarOverlayOptions(),
    trafficLightPosition: IS_MAC ? WINDOW_BUTTON_POSITION : undefined,
    ...chatWindowSurfaceOptions(),
    icon,
    show: false,
    webPreferences: chatWindowWebPreferences(PRELOAD_PATH)
  })

  instanceWindows.add(win)

  // Chat-surface registration: see applyWindowTranslucency.
  translucencyBackedWindows.add(win)

  if (IS_MAC) {
    win.setWindowButtonPosition?.(WINDOW_BUTTON_POSITION)
  }

  wireWindowReveal(win)

  // Per-window fullscreen chrome: send this window its own titlebar inset so its
  // traffic lights hide/show independently of the primary.
  win.on('enter-full-screen', () => sendWindowStateChanged(true, win))
  win.on('leave-full-screen', () => sendWindowStateChanged(false, win))

  streamThrottle.register(win)
  wireCommonWindowHandlers(win, zoomWiringForWindowKind('chat'))

  // Renderer lifecycle diagnostics + recovery (#81290), same policy as the
  // primary and session windows: a crashed instance renderer logs with its
  // window kind and reloads under the shared crash-loop budget.
  installWindowRendererLifecycle(win, {
    kind: 'instance',
    callbacks: {
      log: rememberLog,
      reload: () => {
        win.webContents.reload()
      }
    },
    reloadWindowMs: RENDERER_RELOAD_WINDOW_MS,
    reloadMax: RENDERER_RELOAD_MAX,
    recentReloadTimesRef: rendererReloadTimesRef
  })

  win.on('closed', () => {
    instanceWindows.delete(win)
  })

  attachRendererConsoleCapture(win, 'instance', rememberLog)
  loadWindowUrl(
    win,
    buildInstanceWindowUrl({
      devServer: DEV_SERVER,
      rendererIndexPath: DEV_SERVER ? undefined : resolveRendererIndex()
    }),
    'Instance window'
  )

  return win
}

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

  // Pet overlay opts out of global UI zoom (see zoomWiringForWindowKind): it
  // owns its window-fit + scale, and inheriting zoom would crop the sprite.
  wireCommonWindowHandlers(win, zoomWiringForWindowKind('petOverlay'))

  wireWindowReveal(win, { show: () => win.showInactive() })

  // Log-only renderer lifecycle (#81290): a dead overlay must never resurrect
  // itself over the app, but its loss belongs in desktop.log.
  installWindowRendererLifecycle(win, { kind: 'overlay', callbacks: { log: rememberLog } })

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

  attachRendererConsoleCapture(win, 'pet-overlay', rememberLog)
  loadWindowUrl(win, petOverlayUrl(), 'Pet overlay')

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

// ── HUD mode ────────────────────────────────────────────────────────────────
//
// The chrome-free floating chat: a transparent, frameless, always-on-top
// window showing only the composer and its scrollback, so Hermes can be driven
// while the user works in another app.
//
// Unlike the pet overlay / quick entry, this is a FULL app renderer with its
// own gateway — the same thing createInstanceWindow() spawns, reshaped. That
// is deliberate: the HUD renders the real chat surface, so its composer is the
// app's composer (slash commands, attachments, queue, voice) instead of a
// lookalike that drifts. Entering HUD mode hides the main window; leaving
// restores it.
let hudWindow = null

// Whether the main window was visible when HUD mode was entered, so exiting
// puts the desktop back as it was rather than raising a window the user had
// already minimized.
let hudRestoreMainWindow = false

// The session the HUD is currently on, reported by its renderer whenever the
// selection changes. Leaving HUD mode is a HANDOFF, not just a window close:
// the gateway binds a session's event stream to exactly one socket, so the
// turn the HUD started is streaming to the HUD's socket and the app window
// hears nothing. The app has to re-resume that session to take the stream
// back, and it can only do that if it knows which session to ask for — the
// HUD may have switched sessions, or started a new one the app has never
// seen. Main is the only party that outlives the HUD's renderer, so it holds
// the id and hands it over in the close broadcast.
let hudSessionId = null

// The profile the live HUD renderer booted against (rides hudUrl's query
// string). A renderer adopts its backend once at boot, so a retarget onto a
// session from a DIFFERENT profile cannot be a same-window `goto` — the HUD
// must be respawned against the new profile's backend (see openHudWindow).
let hudProfile = null

// A wide, short bar parked near the bottom of the active display — the shape
// of a game chat frame, and where one belongs. Defaults only: once the user
// moves or resizes the HUD, hud-state.json wins (same pattern as the main
// window's window-state.json).
const HUD_STATE_PATH = path.join(app.getPath('userData'), 'hud-state.json')

function readHudState() {
  try {
    const raw = JSON.parse(fs.readFileSync(HUD_STATE_PATH, 'utf8'))

    if (
      [raw?.x, raw?.y, raw?.width, raw?.height].every(v => Number.isFinite(v)) &&
      raw.width >= 380 &&
      raw.height >= 160
    ) {
      return raw
    }
  } catch {
    // First run / unreadable — fall through to defaults.
  }

  return null
}

function persistHudState() {
  if (!hudWindow || hudWindow.isDestroyed()) {
    return
  }

  try {
    const { x, y, width, height } = hudWindow.getNormalBounds()
    fs.mkdirSync(path.dirname(HUD_STATE_PATH), { recursive: true })
    writeFileAtomic(HUD_STATE_PATH, JSON.stringify({ x, y, width, height }, null, 2))
  } catch (err) {
    rememberLog(`[hud-state] persist failed: ${err?.message || err}`)
  }
}

function resetHudWindowLayout(): boolean {
  if (!hudWindow || hudWindow.isDestroyed()) {
    return false
  }

  const win = hudWindow
  const display = screen.getDisplayNearestPoint(screen.getCursorScreenPoint())
  const bounds = defaultHudBounds(display?.workArea)

  if (!applyHudResetBounds(win, bounds)) {
    rememberLog('[hud-state] reset layout failed while applying native bounds')

    return false
  }

  persistHudState()

  return true
}

const schedulePersistHudState = debounce(persistHudState, 250)

// How often Linux gets told where the cursor is. Fast enough that the bar is
// solid before a click lands after the pointer arrives, cheap enough to leave
// running for as long as the HUD is open — it is one `getCursorScreenPoint()`
// and, when the answer has not changed, nothing else.
const HUD_CURSOR_POLL_MS = 60

// Snap-to-pointer — global ⌘⇧G while the HUD is open (tap, not hold).
const HUD_SNAP_ANCHOR_Y = 48

function hudWindowing() {
  return resolveHudWindowing(process.platform, process.env, process.argv)
}

function applyHudSnapToPointer() {
  if (!hudWindow || hudWindow.isDestroyed() || !hudWindowing().clientPlacement) {
    return
  }

  const cursor = screen.getCursorScreenPoint()
  const bounds = hudWindow.getBounds()
  const display = screen.getDisplayNearestPoint(cursor)
  const workArea = display?.workArea ?? bounds
  const anchor = { x: Math.round(bounds.width / 2), y: HUD_SNAP_ANCHOR_Y }

  const origin = snapHudBounds(
    cursor,
    anchor,
    { width: bounds.width, height: bounds.height },
    hudWindow.webContents.getZoomFactor(),
    workArea
  )

  // setBounds — NOT setPosition alone: on Windows, a transparent frameless
  // window silently grows ~1px per setPosition call (see move-by handler).
  // On native Wayland the compositor ignores the position half; the snap
  // shortcut is therefore a documented no-op there.
  hudWindow.setBounds({
    x: origin.x,
    y: origin.y,
    width: bounds.width,
    height: bounds.height
  })
}

const hudSnapShortcut = createHudSnapShortcut(globalShortcut, applyHudSnapToPointer)

function registerHudSnapShortcut() {
  if (!hudSnapShortcut.register()) {
    rememberLog('[hud] snap shortcut unavailable — CommandOrControl+Shift+G may be owned by another app')
  }
}

/**
 * Feed the HUD renderer the cursor position on Linux.
 *
 * Everywhere else the renderer learns this from mousemove, which keeps arriving
 * while the window ignores the mouse because we pass `{ forward: true }`. That
 * option is macOS/Windows only. Without it a Linux HUD stops hearing the
 * pointer the moment it turns click-through, so it can never notice the pointer
 * coming back and stays transparent — the bar is there, and clicking it hits
 * whatever is behind. Main can still see the cursor, so it says so.
 *
 * Deliberately the same decision, just a different source for one input: the
 * renderer runs its usual hit test on the point it is handed. Re-deciding
 * anything here would put a second, drifting copy of the click-through rules in
 * the main process.
 */
function startHudCursorFeed(win: BrowserWindow) {
  const windowing = hudWindowing()

  if (!windowing.cursorFeed) {
    if (!windowing.ignoreMouse) {
      try {
        win.setIgnoreMouseEvents(false)
      } catch {
        // best effort
      }
    }

    return
  }

  let last: string | null = null

  const timer = setInterval(() => {
    if (win.isDestroyed() || !win.isVisible()) {
      return
    }

    const point = cursorPointInWindow(screen.getCursorScreenPoint(), win.getBounds(), win.webContents.getZoomFactor())

    // Off-window is a real answer (it is what hands the mouse back), so it is
    // sent — once. Only an unchanged answer is dropped, to keep an idle cursor
    // from waking the renderer 16 times a second.
    const key = point ? `${Math.round(point.x)},${Math.round(point.y)}` : 'out'

    if (key === last) {
      return
    }

    last = key
    win.webContents.send('hermes:hud:cursor', point)
  }, HUD_CURSOR_POLL_MS)

  win.on('closed', () => clearInterval(timer))
}

/**
 * Watch for a fullscreen app under the HUD (the Discord-style game overlay
 * cue) and feed the answer to its renderer. Pure detection lives in
 * hud-game-overlay.ts; enumeration is the same front-to-back walk the
 * read_window_below tool uses. The renderer answers with the low-opacity
 * treatment (`data-hud-game`), so main stays out of the styling business.
 */
function startHudGameOverlayFeed(win: BrowserWindow) {
  const titlesAvailable = IS_MAC ? systemPreferences.getMediaAccessStatus?.('screen') === 'granted' : true

  let last = { active: false, app: '' }

  const push = (state: { active: boolean; app: string }) => {
    if (!win.isDestroyed()) {
      win.webContents.send('hermes:hud:game-overlay', state)
    }
  }

  // Replay the latest state to every load of this window. The watch pushes only
  // on CHANGE and its first tick fires the moment the window is created — well
  // before the renderer has mounted its listener — so a HUD opened over a game
  // that is already fullscreen would hear the one and only message before it
  // could receive it, then sit at "no game" forever while main was certain it
  // had reported one. (Same reason quick entry caches its last state push.)
  // did-finish-load also covers HMR full reloads during development.
  win.webContents.on('did-finish-load', () => push(last))

  // The watch gives up after two failed enumerations and never says so, which
  // is how a HUD that cannot see the screen at all — no game cue, and
  // read_window_below failing beside it — leaves nothing in the log to explain
  // itself. Report the reason once; the null keeps the watch's contract.
  let reported = false

  const enumerate = async () => {
    const windows = await enumerateWindowsFrontToBack(process.pid, titlesAvailable)

    if (!enumerationFailed(windows)) {
      return windows
    }

    if (!reported) {
      reported = true
      console.warn(`[hermes] HUD cannot enumerate windows: ${windows.reason}`)
    }

    return null
  }

  const dispose = startHudGameOverlayWatch({
    enumerate,
    displayBounds: () => screen.getDisplayMatching(win.getBounds()).bounds,
    selfPid: process.pid,
    send: state => {
      last = state
      push(state)
    }
  })

  win.on('closed', dispose)
}

function hudBounds() {
  // Remembered spot first — validated against the LIVE displays so a HUD
  // parked on an unplugged monitor comes back on-screen instead of lost.
  const saved = readHudState()

  if (saved) {
    const onScreen = screen.getAllDisplays().some(d => {
      const a = d.workArea

      return (
        saved.x < a.x + a.width - 40 &&
        saved.x + saved.width > a.x + 40 &&
        saved.y < a.y + a.height - 40 &&
        saved.y + saved.height > a.y + 40
      )
    })

    if (onScreen) {
      return saved
    }
  }

  const display = screen.getDisplayNearestPoint(screen.getCursorScreenPoint())
  const area = display?.workArea

  return defaultHudBounds(area)
}

function hudUrl(sessionId, profile) {
  // The profile rides the query string next to `win=hud` (BEFORE the '#', so
  // HashRouter never sees it). The HUD renderer's gateway boot reads it and
  // adopts that backend instead of the primary — without it, a HUD opened on a
  // non-primary profile's conversation resolves the session id against the
  // wrong backend and falls back to the default profile's last session.
  return buildHudWindowUrl(sessionId, {
    devServer: DEV_SERVER,
    profile,
    rendererIndexPath: DEV_SERVER ? undefined : resolveRendererIndex()
  })
}

// Tell every window whether the HUD is up, so a toggle in any of them reads
// the truth even when the HUD is closed from its own side (⌘W / its exit row).
// Carries the HUD's session so the app window can re-home onto it on the way
// out (see hudSessionId).
function broadcastHudState(open) {
  const payload = { open, sessionId: hudSessionId }

  for (const win of BrowserWindow.getAllWindows()) {
    if (!win.isDestroyed()) {
      win.webContents.send('hermes:hud:changed', payload)
    }
  }
}

function spawnHudWindow(sessionId, profile) {
  const win = new BrowserWindow({
    ...hudBounds(),
    minWidth: 380,
    minHeight: 160,
    title: HUD_WINDOW_TITLE,
    frame: false,
    transparent: true,
    // NOT resizable. A transparent frameless window on Windows keeps a
    // system-level edge resize hot-zone while `resizable` is on — the OS
    // interprets pointer capture near the edge as a resize gesture, so the
    // window grows a few px every drag (worse at >100% DPI scaling). The
    // composer drag calls setPosition, which must move the window, not resize
    // it. Resizing is done by the renderer's edge/corner handles through
    // `hermes:hud:set-bounds`, which flips resizable on for the call — the
    // same pattern the pet overlay uses for its wheel-scale.
    resizable: false,
    movable: true,
    minimizable: false,
    maximizable: false,
    fullscreenable: false,
    // Same rationale as the pet overlay: on Windows/Linux keep the helper out
    // of the taskbar/alt-tab list; on macOS use an NSPanel so the frameless
    // window never becomes the app's cmd-tab anchor.
    skipTaskbar: !IS_MAC,
    hasShadow: false,
    alwaysOnTop: true,
    type: IS_MAC ? 'panel' : undefined,
    // Clips the vibrancy layer to the HUD's silhouette rather than a hard
    // rectangle — the frost stops where the window's corners do.
    roundedCorners: true,
    // Vibrancy must keep rendering while the window is BLURRED: streaming under
    // another app is the whole feature, and the default 'followWindow' kills
    // the frost the moment something else takes focus.
    visualEffectState: 'active',
    hiddenInMissionControl: IS_MAC,
    show: false,
    backgroundColor: '#00000000',
    // The full chat webPreferences — this window streams a real transcript, so
    // it needs everything a chat window needs (preload bridge, autoplay for
    // voice, the shared throttling contract).
    webPreferences: chatWindowWebPreferences(PRELOAD_PATH)
  })

  applyHudElectronOverlay(win, process.platform)
  win.setHiddenInMissionControl?.(true)

  // Linux intentionally starts on ONE virtual desktop. During a renderer
  // grab, hermes:hud:workspace-transfer temporarily makes the X11 window
  // sticky; releasing it assigns the HUD to KDE's then-current desktop.

  // Streaming into a window that is ALWAYS blurred (the user is in another
  // app) is the entire feature, so it gets the same stream-aware unthrottling
  // every chat window does.
  streamThrottle.register(win)
  wireCommonWindowHandlers(win, zoomWiringForWindowKind('chat'))

  // Remember where the user parks and sizes it (debounced — these fire many
  // times mid-drag).
  bindGeometryPersistence(win, schedulePersistHudState)

  startHudCursorFeed(win)
  startHudGameOverlayFeed(win)

  wireWindowReveal(win, {
    show: () => {
      win.show()
      win.focus()
    },
    onRevealed: () => {
      // Step the app aside: the HUD IS the surface now.
      if (hudRestoreMainWindow && mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.hide()
      }

      // Compositor overlay adapters (Hyprland float+pin today). Electron
      // alwaysOnTop is already set; this is the dialect some WMs actually hear.
      void promoteHudOverlay({ title: HUD_WINDOW_TITLE })
    }
  })

  win.on('closed', () => {
    if (hudWindow === win) {
      hudWindow = null
    }

    // Closed from its own side (⌘W) — closeHudWindow()'s dispose() never ran,
    // so the global snap shortcut would otherwise stay registered (and stuck
    // taken) with no HUD left to apply it to. dispose() is idempotent, so
    // this is safe even if closeHudWindow() already released it.
    hudSnapShortcut.dispose()

    // Put the app back so the user is never left with no surface, and
    // correct every window's toggle.
    restoreMainWindowFromHud()
    broadcastHudState(false)
  })

  attachRendererConsoleCapture(win, 'hud', rememberLog)
  // Log-only lifecycle (#81290): the HUD is a compact auxiliary surface the
  // user can re-toggle; a dead renderer should be diagnosable, not resurrected.
  installWindowRendererLifecycle(win, { kind: 'hud', callbacks: { log: rememberLog } })
  loadWindowUrl(win, hudUrl(sessionId, profile), 'HUD')

  return win
}

// Put the app window back the way HUD mode found it.
function restoreMainWindowFromHud() {
  if (!hudRestoreMainWindow) {
    return
  }

  hudRestoreMainWindow = false

  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.show()
  }
}

function openHudWindow(sessionId, profile) {
  const profileKey = typeof profile === 'string' && profile.trim() ? profile.trim() : null

  if (hudWindow && !hudWindow.isDestroyed()) {
    // Pointed at another PROFILE: the live renderer is bound to the old
    // profile's backend, and a renderer adopts its backend exactly once at
    // boot — an in-place goto would resolve the id against the wrong backend
    // (the #82285 fallback). Respawn against the right one.
    if (profileKey && hudProfile !== profileKey) {
      const win = hudWindow
      hudWindow = null
      win.removeAllListeners('closed')
      win.destroy()

      hudSessionId = sessionId || null
      hudProfile = profileKey
      hudWindow = spawnHudWindow(sessionId, profileKey)
      broadcastHudState(true)
      registerHudSnapShortcut()

      return hudWindow
    }

    // Already up, but pointed somewhere else — switch it rather than just
    // raising it. Asking for HUD mode from another tab means "put THIS
    // conversation in the HUD", and a plain focus leaves the wrong one there.
    if (sessionId && sessionId !== hudSessionId) {
      hudSessionId = sessionId
      hudWindow.webContents.send('hermes:hud:goto', sessionId)
      // Keep every window's idea of where the HUD is pointed in step, so the
      // toggle keeps reading "switch" vs "dismiss" correctly.
      broadcastHudState(true)
    }

    focusWindow(hudWindow)

    return hudWindow
  }

  hudRestoreMainWindow = Boolean(mainWindow && !mainWindow.isDestroyed() && mainWindow.isVisible())
  hudSessionId = sessionId || null
  hudProfile = profileKey
  hudWindow = spawnHudWindow(sessionId, profileKey)
  broadcastHudState(true)
  registerHudSnapShortcut()

  return hudWindow
}

function closeHudWindow() {
  hudSnapShortcut.dispose()

  const win = hudWindow
  hudWindow = null

  if (win && !win.isDestroyed()) {
    // Null'd first so the 'closed' handler doesn't broadcast a second time.
    win.removeAllListeners('closed')
    win.close()
  }

  restoreMainWindowFromHud()
  broadcastHudState(false)

  if (mainWindow && !mainWindow.isDestroyed()) {
    focusWindow(mainWindow)
  }
}

// ── Quick Entry ─────────────────────────────────────────────────────────────
//
// A global shortcut summons a small frameless always-on-top composer from
// anywhere, so a prompt can be fired without raising the whole app. The window
// carries NO gateway connection: it hands its text to us, we forward it to the
// PRIMARY renderer, and that renderer submits through the same prompt path the
// normal composer uses (see store/quick-entry + hooks/use-quick-entry-bridge).
//
// Main owns the OS registration and the persisted preference (it must restore
// the shortcut on a cold launch without the renderer ever visiting Settings),
// same authority split as keep-awake. Registration failure is surfaced, never
// swallowed: a chord another app already owns comes back as `error: 'taken'`.
const QUICK_ENTRY_CONFIG_PATH = path.join(app.getPath('userData'), 'quick-entry.json')

let quickEntryWindow = null

// Latest state push from the primary renderer (connection + recent sessions),
// replayed to a quick window that spawns after the push happened.
let quickEntryLastState = null

function readQuickEntrySettings() {
  try {
    return sanitizeQuickEntrySettings(JSON.parse(fs.readFileSync(QUICK_ENTRY_CONFIG_PATH, 'utf8')))
  } catch {
    // Missing / unreadable / malformed → shipped defaults (enabled, default chord).
    return sanitizeQuickEntrySettings(undefined)
  }
}

function writeQuickEntrySettings(settings) {
  try {
    fs.mkdirSync(path.dirname(QUICK_ENTRY_CONFIG_PATH), { recursive: true })
    fs.writeFileSync(QUICK_ENTRY_CONFIG_PATH, JSON.stringify(settings, null, 2), 'utf8')
  } catch (error) {
    rememberLog(`[quick-entry] write failed: ${error.message}`)
  }
}

function quickEntryUrl() {
  if (DEV_SERVER) {
    return `${DEV_SERVER.endsWith('/') ? DEV_SERVER.slice(0, -1) : DEV_SERVER}/?win=quick#/`
  }

  return `${pathToFileURL(resolveRendererIndex()).toString()}?win=quick#/`
}

function spawnQuickEntryWindow() {
  const cursor = screen.getCursorScreenPoint()
  const display = screen.getDisplayNearestPoint(cursor)
  const bounds = quickEntryWindowBounds(display?.workArea)

  const win = new BrowserWindow({
    ...bounds,
    frame: false,
    transparent: true,
    resizable: false,
    movable: true,
    minimizable: false,
    maximizable: false,
    fullscreenable: false,
    // Same rationale as the pet overlay: on Windows/Linux keep the helper out
    // of the taskbar/alt-tab list; on macOS use an NSPanel so the frameless
    // capture window never becomes the app's cmd-tab anchor.
    skipTaskbar: !IS_MAC,
    hasShadow: true,
    alwaysOnTop: true,
    type: IS_MAC ? 'panel' : undefined,
    hiddenInMissionControl: IS_MAC,
    show: false,
    backgroundColor: '#00000000',
    webPreferences: {
      preload: PRELOAD_PATH,
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
      devTools: true
    }
  })

  win.setAlwaysOnTop(true, IS_MAC ? 'floating' : 'screen-saver')
  win.setHiddenInMissionControl?.(true)

  try {
    win.setVisibleOnAllWorkspaces(
      true,
      IS_MAC ? { visibleOnFullScreen: true, skipTransformProcessType: true } : undefined
    )
  } catch {
    // Not supported everywhere — best effort.
  }

  // Opts out of global UI zoom for the same reason as the pet overlay: it sizes
  // its own OS window and a zoomed composer would overflow it.
  wireCommonWindowHandlers(win, zoomWiringForWindowKind('quickEntry'))

  // Log-only renderer lifecycle (#81290): a dead quick-entry window must never
  // resurrect itself over the app, but its loss belongs in desktop.log.
  installWindowRendererLifecycle(win, { kind: 'quick', callbacks: { log: rememberLog } })

  // Hide on blur. The window must never hold the user's focus captive — losing
  // focus is the cheapest, least surprising dismiss (matches Spotlight).
  win.on('blur', () => {
    if (!win.isDestroyed()) {
      win.hide()
    }
  })

  win.on('closed', () => {
    if (quickEntryWindow === win) {
      quickEntryWindow = null
    }
  })

  // Replay the last known gateway state as soon as the page can hear it — a
  // freshly spawned quick window must not sit "disconnected" when the primary
  // renderer already reported a live gateway.
  win.webContents.on('did-finish-load', () => {
    if (!win.isDestroyed() && quickEntryLastState) {
      win.webContents.send('hermes:quick-entry:state', quickEntryLastState)
    }
  })

  attachRendererConsoleCapture(win, 'quick-entry', rememberLog)
  loadWindowUrl(win, quickEntryUrl(), 'Quick entry')

  return win
}

// Move the (already-open) window to the display the cursor is on, so the chord
// summons it where the user is looking rather than where they last were.
function repositionQuickEntryWindow(win) {
  try {
    const display = screen.getDisplayNearestPoint(screen.getCursorScreenPoint())
    win.setBounds(quickEntryWindowBounds(display?.workArea))
  } catch (error) {
    rememberLog(`[quick-entry] reposition failed: ${error.message}`)
  }
}

function showQuickEntryWindow() {
  if (!quickEntryWindow || quickEntryWindow.isDestroyed()) {
    // Reveal the window this call created, not whatever `quickEntryWindow`
    // points at by the time the event lands.
    const win = spawnQuickEntryWindow()
    quickEntryWindow = win

    wireWindowReveal(win, {
      show: () => {
        win.show()
        win.focus()
      }
    })

    return
  }

  repositionQuickEntryWindow(quickEntryWindow)
  quickEntryWindow.show()
  quickEntryWindow.focus()
  // Re-summoned: tell the renderer to clear any stale draft and refocus.
  quickEntryWindow.webContents.send('hermes:quick-entry:shown')
}

function hideQuickEntryWindow() {
  if (quickEntryWindow && !quickEntryWindow.isDestroyed()) {
    quickEntryWindow.hide()
  }
}

// The chord toggles: pressing it while the composer is up puts it away, so one
// gesture does exactly one thing in both directions.
function toggleQuickEntryWindow() {
  if (quickEntryWindow && !quickEntryWindow.isDestroyed() && quickEntryWindow.isVisible()) {
    hideQuickEntryWindow()

    return
  }

  showQuickEntryWindow()
}

const quickEntryShortcut = createQuickEntryShortcut(globalShortcut, toggleQuickEntryWindow)

function applyQuickEntrySettings(settings) {
  const state = quickEntryShortcut.apply(settings)

  if (!settings.enabled) {
    // Turning the feature off must not leave an orphan always-on-top window.
    if (quickEntryWindow && !quickEntryWindow.isDestroyed()) {
      quickEntryWindow.close()
    }

    quickEntryWindow = null
  }

  if (state.error === 'taken') {
    rememberLog(`[quick-entry] shortcut ${state.shortcut} is already taken by another application`)
  } else if (state.error === 'invalid') {
    rememberLog(`[quick-entry] shortcut ${state.shortcut} is not a valid accelerator`)
  }

  return { ...state, enabled: settings.enabled }
}

function closeQuickEntryWindow() {
  quickEntryShortcut.dispose()

  if (quickEntryWindow && !quickEntryWindow.isDestroyed()) {
    quickEntryWindow.close()
  }

  quickEntryWindow = null
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
    ...chatWindowSurfaceOptions(),
    icon,
    // Hidden until the first themed paint so macOS `vibrancy` (which ignores
    // `backgroundColor` and follows the OS appearance) can't flash a light
    // material before the renderer paints the app theme. See createSessionWindow.
    show: false,
    // Shared with the secondary session windows (chatWindowWebPreferences);
    // stream-aware throttling is applied per-window via streamThrottle so a
    // live answer keeps painting while the window is blurred or minimized,
    // without pinning visibilityState to 'visible' at idle. See
    // session-windows.ts and stream-throttle.ts.
    webPreferences: chatWindowWebPreferences(PRELOAD_PATH)
  })

  const createdMainWindow = mainWindow

  // Chat-surface registration: see applyWindowTranslucency.
  translucencyBackedWindows.add(mainWindow)

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
        for (const win of BrowserWindow.getAllWindows()) {
          applyTitleBarOverlay(win)
        }
      })
    }
  }

  if (savedWindowState?.isMaximized) {
    mainWindow.maximize()
  }

  const revealController = wireWindowReveal(createdMainWindow, {
    onRevealed: () => {
      // Persist geometry as soon as the window is visible so a crash before the
      // first clean resize/move/close still captures the restored bounds (#56726).
      schedulePersistWindowState()

      // #38216: clear the mid-boot marker only after a window is actually usable.
      // Keep sticky `fallback` when we launched with --no-sandbox so the next
      // Start Menu click does not re-enter the GPU FATAL crash loop. The marker
      // records the app version so the next update re-probes the sandbox.
      if (IS_WINDOWS) {
        try {
          writeSandboxMarker(
            app.getPath('userData'),
            markerAfterSuccessfulBoot({
              fallbackActive: windowsSandboxFallbackSticky,
              reason: windowsSandboxFallbackReason,
              appVersion: app.getVersion()
            })
          )
        } catch (error) {
          rememberLog(`[sandbox] marker update after main-window reveal failed: ${error?.message || error}`)
        }
      }
    }
  })

  // Under Playwright testing, instantly show the window: `ready-to-show`
  // doesn't fire in some testing envs, and the suite can't wait out the
  // production fallback.
  if (process.env.TEST_WORKER_INDEX !== undefined) {
    revealController.reveal()
  }

  mainWindow.on('will-enter-full-screen', () => sendWindowStateChanged(true))
  mainWindow.on('enter-full-screen', () => sendWindowStateChanged(true))
  mainWindow.on('will-leave-full-screen', () => sendWindowStateChanged(false))
  mainWindow.on('leave-full-screen', () => sendWindowStateChanged(false))
  mainWindow.on('minimize', () => sendWindowStateChanged())
  mainWindow.on('restore', () => sendWindowStateChanged())
  mainWindow.on('hide', () => sendWindowStateChanged())
  mainWindow.on('show', () => sendWindowStateChanged())

  // Reopen where the user left off. close is the backstop, flushed
  // synchronously before the window is gone.
  bindGeometryPersistence(mainWindow, schedulePersistWindowState)
  mainWindow.on('maximize', schedulePersistWindowState)
  mainWindow.on('unmaximize', schedulePersistWindowState)
  mainWindow.on('close', () => schedulePersistWindowState.flush())

  // the closed wrapper remains truthy, so clear only the window this callback owns.
  mainWindow.on('closed', () => {
    closePetOverlay()
    wakeIndicatorController.close()

    if (mainWindow === createdMainWindow) {
      mainWindow = null
      // the replacement renderer must register before queued links can be delivered.
      _rendererReadyForDeepLink = false
    }
  })

  streamThrottle.register(mainWindow)
  wireCommonWindowHandlers(mainWindow, zoomWiringForWindowKind('chat'))

  // Per-window renderer lifecycle diagnostics + recovery (#81290). The reload
  // policy (crashed/oom → bounded reload via the shared rolling budget, then
  // the #38216 Windows sandbox relaunch check on suppression) is the same
  // policy this window used before it moved into the shared helper, so a
  // crashed peer renderer now logs and recovers exactly like the primary one.
  installWindowRendererLifecycle(mainWindow, {
    kind: 'main',
    callbacks: {
      log: rememberLog,
      reload: () => {
        mainWindow.webContents.reload()
      },
      onCrashLoopSuppressed: details => {
        // #38216 renderer flavor (same recovery as #56726, credit @Sahil-SS9):
        // a deterministic Windows renderer crash loop with the sandbox
        // breakpoint signature gets one --no-sandbox relaunch instead of a
        // dead window. Gated on the exit code so unrelated crash loops don't
        // silently drop the sandbox.
        if (
          !shouldRelaunchForRendererSandboxCrashLoop({
            reason: details?.reason,
            exitCode: details?.exitCode,
            alreadyNoSandbox: windowsSandboxFallbackActive || alreadyHasNoSandbox(process.argv, process.env),
            relaunchAttempted: windowsNoSandboxRelaunchAttempted
          })
        ) {
          return
        }

        windowsNoSandboxRelaunchAttempted = true
        windowsSandboxFallbackActive = true
        windowsSandboxFallbackSticky = true
        windowsSandboxFallbackReason = 'renderer-crash-loop'

        try {
          writeSandboxMarker(app.getPath('userData'), fallbackMarker('renderer-crash-loop', app.getVersion()))
        } catch {
          void 0
        }

        rememberLog('[renderer] Windows sandbox crash loop detected; relaunching once with --no-sandbox (#38216)')

        try {
          app.relaunch({ args: buildNoSandboxRelaunchArgs(process.argv.slice(1)) })
          void exitAfterBackendShutdown(0)
        } catch (err) {
          rememberLog(`[renderer] --no-sandbox relaunch failed: ${err?.message || err}`)
        }
      }
    },
    reloadWindowMs: RENDERER_RELOAD_WINDOW_MS,
    reloadMax: RENDERER_RELOAD_MAX,
    recentReloadTimesRef: rendererReloadTimesRef
  })

  // Electron always passes the event first. The canonical (Electron 36+) shape
  // is (event, messageDetails); the deprecated positional shape is
  // (event, level, message, line, sourceId). Handled in renderer-log.ts, which
  // every renderer-content window shares (#79428: crashes in secondary/HUD/
  // quick-entry windows used to vanish without a trace).
  attachRendererConsoleCapture(mainWindow, 'main', rememberLog)

  loadWindowUrl(mainWindow, DEV_SERVER || pathToFileURL(resolveRendererIndex()).toString(), 'Renderer')

  // Start the Python backend NOW, in parallel with the renderer load — not on
  // did-finish-load. The backend cold boot (spawn → port announce → /api/status)
  // is the dominant startup cost, and serializing it behind Chromium's load
  // added the whole renderer load time to first-usable-composer. The promise is
  // shared (backendConnectionState), so the renderer's getConnection() joins
  // this in-flight boot instead of duplicating it; early boot-progress events
  // the renderer misses are recovered by its getBootProgress() pull on mount.
  startHermes().catch(error => rememberLog(error.stack || error.message))

  mainWindow.webContents.once('did-finish-load', () => {
    // Zoom restore is handled by wireCommonWindowHandlers (shared with session
    // windows); no need to reapply it here.
    broadcastBootProgress()
    sendWindowStateChanged()
  })
}

ipcMain.handle('hermes:connection', async (_event, profile) => {
  const connection = await ensureBackend(profile)
  const connectionId = resolvedConnectionId(readDesktopConnectionsRegistry(), connection)

  return connectionId ? { ...connection, connectionId } : connection
})
// Registry-scoped variant: resolve a backend for (connectionId, profile).
// connectionId '' / 'local' / the registry primary all behave sensibly; the
// local kind delegates to ensureBackend when the v1 route is local, and
// forces a genuinely-local child when the v1 global mode is remote (the
// registry 'local' entry always means this machine).
ipcMain.handle('hermes:connection:for', async (_event, payload) => {
  const { connectionId, profile } = payload && typeof payload === 'object' ? (payload as any) : ({} as any)
  const registry = readDesktopConnectionsRegistry()
  const id = String(connectionId || '').trim() || registry.primary
  const connection = await ensureRegistryBackend(id, profile)

  return { ...connection, connectionId: id, registryScoped: true }
})

const windowConnectionRoutes = new WindowConnectionRouteRegistry()
const windowConnectionRouteOwners = new Set<number>()

ipcMain.on('hermes:connection:active-route', (event, route) => {
  const id = event.sender.id
  const previous = windowConnectionRoutes.get(id)
  const next = windowConnectionRoutes.set(id, route)

  if (
    previous?.connectionId !== next?.connectionId ||
    previous?.profile !== next?.profile ||
    previous?.registryScoped !== next?.registryScoped
  ) {
    void resetPreviewReach(id)
  }

  if (!windowConnectionRouteOwners.has(id)) {
    windowConnectionRouteOwners.add(id)
    event.sender.once('destroyed', () => {
      windowConnectionRoutes.delete(id)
      windowConnectionRouteOwners.delete(id)
      void resetPreviewReach(id)
    })
  }
})
// Reconnect-after-wake recovery. A REMOTE primary backend has no child process,
// so the 'exit'/'error' handlers that would clear a dead connection promise never
// fire — once the remote becomes unreachable across a sleep/wake the renderer
// re-dials the same dead descriptor forever and the composer stays stuck on
// "Starting Hermes…". Before the renderer's backoff loop reconnects, it asks us
// to confirm the cached PRIMARY backend is still reachable; if a remote one is
// not, we drop the cache so the next getConnection() rebuilds it. Local backends
// self-heal via their child 'exit' handler, so we never touch them here.
ipcMain.handle('hermes:connection:revalidate', async () => {
  const connectionPromise = backendConnectionState.getPromise()

  if (!connectionPromise) {
    await revalidatePool()

    return { ok: true, rebuilt: false }
  }

  // Main and every session pop-out have their own renderer reconnect loop but
  // share this primary connection. Coalesce simultaneous requests so one outage
  // produces one failure observation rather than exhausting the whole streak.
  return remoteRevalidation.run(connectionPromise, async () => {
    const [result] = await Promise.all([
      revalidateRemoteConnection({
        connectionPromise,
        currentConnectionPromise: () => backendConnectionState.getPromise(),
        log: rememberLog,
        probe: (connection, path, options) => fetchJsonForBackend(connection, path, options),
        resetConnection: () => resetHermesConnection({ soft: true }),
        tracker: remoteLiveness
      }),
      revalidatePool()
    ])

    // A rebuilt SSH connection must also tear down its tunnel/master before the
    // renderer re-dials (which only happens after this handler resolves), so the
    // fresh bootstrap can't reattach to a dying transport.
    if (result.rebuilt) {
      const conn = await connectionPromise.catch(() => null)

      if (conn?.remoteKind === 'ssh') {
        const profile = primaryProfileKey()
        await sshBootstrapCoordinator.cancelAndWait(sshScopeKey(profile))
        await teardownSshConnection(profile)
      }
    }

    return result
  })
})

// Pooled remote descriptors get the same treatment as the primary: they have no
// child process to signal their host's death, and the renderer's keepalive touch
// spares them from the idle reaper, so nothing else can retire a dead one.
function revalidatePool() {
  return revalidatePooledRemoteBackends({
    entries: backendPool.entries(),
    log: rememberLog,
    probe: (connection, path, options) => fetchJsonForBackend(connection, path, options),
    stopBackend: stopPoolBackend,
    tracker: remoteLiveness
  })
}

ipcMain.handle('hermes:backend:touch', async (_event, profile) => {
  touchPoolBackend(profile)

  return { ok: true }
})
ipcMain.handle('hermes:gateway:ws-url', async (_event, profile) => {
  return gatewayWsUrlIpcResult(() => freshGatewayWsUrl(profile))
})
ipcMain.handle('hermes:window:openSession', async (_event, sessionId, opts) => {
  if (typeof sessionId !== 'string' || !sessionId.trim()) {
    return { ok: false, error: 'invalid-session-id' }
  }

  createSessionWindow(sessionId.trim(), { watch: opts?.watch === true })

  return { ok: true }
})
ipcMain.handle('hermes:window:openInstance', async () => {
  createInstanceWindow()

  return { ok: true }
})
ipcMain.handle('hermes:window:openBrowser', async (_event, tabId) => {
  if (typeof tabId !== 'string' || !tabId.trim()) {
    return { ok: false, error: 'invalid-tab-id' }
  }

  createBrowserWindow(tabId.trim())

  return { ok: true }
})

// Hand a session to the user's OWN terminal emulator, running the TUI against
// it (`hermes --tui --resume <id>`). Not the in-app terminal pane: the point is
// to continue the chat in the terminal they already live in.
//
// The desktop's runtime is usually a venv Python invoked as
// `python -m hermes_cli.main`, so we resolve the SAME backend the app itself
// launches and carry its argv + PYTHONPATH into a launcher script rather than
// hoping a `hermes` exists on the user's interactive PATH. Resolution only —
// never ensureRuntime(), which would kick off a first-run install from a menu
// click; an unresolved runtime is reported instead.
ipcMain.handle('hermes:window:openInTerminal', async (_event, sessionId, opts) => {
  if (typeof sessionId !== 'string' || !sessionId.trim()) {
    return { ok: false, error: 'invalid-session-id' }
  }

  try {
    const profile = typeof opts?.profile === 'string' ? opts.profile.trim() : ''
    const backend = resolveHermesBackend(tuiResumeArgs(sessionId.trim(), profile || undefined))

    if (!backend.command) {
      return { ok: false, error: 'Hermes is not installed yet' }
    }

    const { cwd } = sanitizeWorkspaceCwd(opts?.cwd)
    const scriptDir = path.join(app.getPath('userData'), 'open-in-terminal')
    fs.mkdirSync(scriptDir, { recursive: true })

    const scriptPath = path.join(
      scriptDir,
      `hermes-${crypto.randomBytes(6).toString('hex')}${terminalScriptExtension()}`
    )

    fs.writeFileSync(
      scriptPath,
      buildTerminalScript({
        args: backend.args,
        command: backend.command,
        cwd,
        env: terminalScriptEnv(backend.env, HERMES_HOME)
      }),
      { mode: 0o700 }
    )

    const launch = resolveTerminalLaunch({ findOnPath, scriptPath })

    if (!launch) {
      return { ok: false, error: 'No terminal emulator found' }
    }

    rememberLog(`[terminal] opening session ${sessionId} via ${launch.command}`)

    // Detached + unref'd: the terminal window outlives the desktop app, and
    // never inherits our stdio (a closed pipe would kill the TUI).
    const child = spawn(launch.command, launch.args, { detached: true, stdio: 'ignore' })
    child.unref()

    return { ok: true }
  } catch (error) {
    rememberLog(`[terminal] open in terminal failed: ${error.message}`)

    return { ok: false, error: error.message }
  }
})
ipcMain.handle('hermes:wake-indicator:get', () => wakeIndicatorController.getState())
ipcMain.on('hermes:wake-indicator:set', (_event, state) => {
  wakeIndicatorController.setState(state)
})

// --- Text size (zoom) -------------------------------------------------------
// The settings UI drives the same clamped zoom scale as the Ctrl/Cmd
// shortcuts and the View menu. Reads and writes target the asking window.
ipcMain.handle('hermes:zoom:get', event => {
  const window = BrowserWindow.fromWebContents(event.sender)

  const level = window && !window.isDestroyed() ? window.webContents.getZoomLevel() : DEFAULT_ZOOM_LEVEL

  return { level, percent: zoomLevelToPercent(level) }
})
ipcMain.on('hermes:zoom:set-percent', (event, percent) => {
  const window = BrowserWindow.fromWebContents(event.sender)

  if (!window || window.isDestroyed()) {
    return
  }

  setAndPersistZoomLevel(window, percentToZoomLevel(Number(percent)))
})

// --- Pet overlay (pop-out mascot) — see pet-overlay-ipc.ts. ---------------
registerPetOverlayIpc({
  getMainWindow: () => mainWindow,
  getPetOverlayWindow: () => petOverlayWindow,
  openPetOverlay,
  closePetOverlay
})

// --- HUD mode (chrome-free floating chat) — see hud-ipc.ts. ---------------
const hudIpc = registerHudIpc({
  isMac: IS_MAC,
  getTranslucencyState: () => translucencyState,
  getHudWindow: () => hudWindow,
  openHudWindow,
  closeHudWindow,
  resetHudLayout: resetHudWindowLayout,
  setHudSessionId: value => {
    hudSessionId = value
  }
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
  getFirstRunSetupGate().resetForRetry()
  resetBootstrapSnapshot()

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
  getFirstRunSetupGate().resetForRepair()
  resetHermesConnection()

  return { ok: true }
})
ipcMain.handle('hermes:bootstrap:continue-local', async () => {
  rememberLog('[bootstrap] local install selected by renderer; continuing first-launch bootstrap')
  continueFirstRunLocalBootstrap()

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
    ? localRouteFallbackProfiles(agents, localSource.id, fallbackProfileNames, Boolean(localEnumeration?.error))
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
ipcMain.handle('hermes:ssh-config:hosts', async () => ({ hosts: collectSshConfigHosts() }))
ipcMain.handle('hermes:ssh-config:resolve', async (_event, host) => {
  const value = String(host || '').trim()

  if (!value) {
    throw new Error('SSH host is required.')
  }

  const ssh =
    process.platform === 'win32'
      ? path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'OpenSSH', 'ssh.exe')
      : 'ssh'

  return new Promise((resolve, reject) => {
    const child = spawn(ssh, ['-G', '--', value], hiddenWindowsChildOptions({ stdio: ['ignore', 'pipe', 'pipe'] }))
    let stdout = ''
    let stderr = ''

    const timer = setTimeout(() => {
      child.kill()
      reject(new Error('SSH config resolution timed out.'))
    }, 10_000)

    child.stdout.on('data', chunk => {
      stdout += String(chunk)
    })
    child.stderr.on('data', chunk => {
      stderr += String(chunk)
    })
    child.once('error', error => {
      clearTimeout(timer)
      reject(error)
    })
    child.once('close', code => {
      clearTimeout(timer)

      if (code !== 0) {
        reject(new Error(stderr.trim() || 'Could not resolve SSH host.'))
      } else {
        resolve(parseSshGOutput(stdout))
      }
    })
  })
})
ipcMain.handle('hermes:connection-config:test', async (_event, payload) => testDesktopConnectionConfig(payload))

// ── Opt-in keychain encryption for stored secrets ───────────────────────────
// get returns the current policy without touching safeStorage; set flips it
// and re-encodes every stored secret (see applySecretStorageEncryption).
ipcMain.handle('hermes:secret-storage:get', async () => ({ on: secretStoragePolicy().on }))
ipcMain.handle('hermes:secret-storage:set', async (_event: any, on: any) => applySecretStorageEncryption(on === true))

// ── v2 connection registry IPC (multi-source) ───────────────────────────────
// Storage-level CRUD for named agent sources. Routing/pooling consumption of
// the registry lands separately; these handlers only manage the persisted
// list, so they are safe to ship ahead of the switchover.
ipcMain.handle('hermes:connections:list', async () => sanitizeConnectionsRegistry())
ipcMain.handle('hermes:connections:save', async (_event, payload) => {
  const saved = await saveRegistryConnection(payload)

  return { ok: true, connection: saved, registry: sanitizeConnectionsRegistry() }
})
ipcMain.handle('hermes:connections:remove', async (_event, id) => {
  const key = String(id || '')
  const registry = removeConnection(readDesktopConnectionsRegistry(), key)
  writeDesktopConnectionsRegistry(registry)
  // Tear down anything the removed connection still had running: pooled
  // backends under its composite keys and any ssh tunnel scopes it owned.
  await stopRegistryConnectionBackends(key)
  // And the renderer side: without this push, secondaries scoped to the
  // removed connection keep their WebSocket open (remote/cloud have no local
  // process to kill) and stream ghost events until page reload.
  broadcastConnectionsChanged({ connectionId: key, reason: 'removed' })

  return { ok: true, registry: sanitizeConnectionsRegistry(registry) }
})
ipcMain.handle('hermes:connections:set-primary', async (_event, id) => {
  const registry = setPrimaryConnection(readDesktopConnectionsRegistry(), String(id || ''))
  writeDesktopConnectionsRegistry(registry)

  return { ok: true, registry: sanitizeConnectionsRegistry(registry) }
})
ipcMain.handle('hermes:connections:set-launch-mode', async (_event, mode) => {
  const registry = setConnectionLaunchMode(readDesktopConnectionsRegistry(), String(mode || ''))
  writeDesktopConnectionsRegistry(registry)

  return { ok: true, registry: sanitizeConnectionsRegistry(registry) }
})
ipcMain.handle('hermes:connections:set-last-used', async (_event, id) => {
  const registry = setLastUsedConnection(readDesktopConnectionsRegistry(), String(id || ''))
  writeDesktopConnectionsRegistry(registry)

  return { ok: true, registry: sanitizeConnectionsRegistry(registry) }
})
ipcMain.handle('hermes:connections:test', async (_event, id) => {
  const registry = readDesktopConnectionsRegistry()
  const entry = registry.connections.find(c => c.id === String(id || ''))

  if (!entry) {
    throw new Error(`No connection with id "${String(id || '')}".`)
  }

  // The ssh probe path in testDesktopConnectionConfig never consults v1
  // connection state, so mapping the entry onto it is safe.
  if (entry.kind === 'ssh') {
    const result = await testDesktopConnectionConfig({
      mode: 'ssh',
      sshHost: entry.host,
      sshUser: entry.user,
      sshPort: entry.port,
      sshKeyPath: entry.keyPath,
      sshRemoteHermesPath: entry.remoteHermesPath
    })

    if (result?.reachable) {
      sshInventoryAttemptedAt.delete(entry.id)
      sshRosterCache.delete(entry.id)
      await probeSshProfileInventory(entry)
    }

    return result
  }

  // Remote/cloud/local probe built DIRECTLY from the registry entry. Routing
  // through coerceDesktopConnectionConfig would use v1 connection.json as the
  // `existing` base: an entry with a broken/absent token would inherit the v1
  // global remote's token and send it to THIS entry's URL (cross-host
  // credential transmission + a false "reachable"), and testing the local
  // entry would probe whatever v1's global mode points at instead of the
  // app-managed local backend.
  let baseUrl
  let token = null
  let authMode = 'token'
  let testHeaders = {}

  if (entry.kind === 'local') {
    const local = await startHermes()
    baseUrl = local.baseUrl
    token = local.token
    authMode = normAuthMode(local.authMode)
  } else {
    baseUrl = normalizeRemoteBaseUrl(entry.url)
    authMode = normAuthMode(entry.authMode)
    testHeaders = decryptRemoteHeaders(entry.headers)

    if (authMode !== 'oauth') {
      token = decryptDesktopSecret(entry.token)

      if (!token) {
        throw new Error('This connection has no saved session token. Edit the connection and paste one.')
      }
    }
  }

  const status = (await fetchConnectionStatus(baseUrl, authMode, token, testHeaders)) as any

  // The Test button is the cheapest moment to (re)learn this backend's stable
  // identity for the same-backend roster collapse + Settings hint.
  rememberConnectionInstallId(entry.id, status)

  // Same HTTP+WS two-leg check as testDesktopConnectionConfig: HTTP alone is
  // a false positive when the WebSocket leg is blocked.
  const wsUrl = await resolveTestWsUrl(baseUrl, authMode, token, {
    mintTicket: url => mintGatewayWsTicket(url, testHeaders)
  })

  if (wsUrl && typeof globalThis.WebSocket === 'function') {
    const probe = await probeGatewayWebSocket(wsUrl, { WebSocketImpl: globalThis.WebSocket, headers: testHeaders })

    if (!probe.ok) {
      throw new Error(
        `Reached the gateway over HTTP, but the live WebSocket (/api/ws) connection failed: ${probe.reason} ` +
          'The HTTP check can pass while the WebSocket is blocked by a proxy, firewall, or gateway auth/origin guard.'
      )
    }
  }

  return { ok: true, baseUrl, version: status?.version || null }
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
const sshRosterCache = new Map<string, string[]>()
const sshInventoryAttemptedAt = new Map<string, number>()
const SSH_INVENTORY_RETRY_MS = 60_000

// Stable backend identity per registered connection: the `install_id` its
// /api/status reports (absent on backends older than the field). Enumeration
// runs on the ~5s Bot Mode roster poll and only hits /api/profiles, so the
// status probe is cached per connection with a TTL to avoid doubling roster
// traffic; the Test button refreshes it eagerly. A missing id simply bypasses
// the same-backend roster collapse — fully backward compatible.
const connectionInstallIds = new Map<string, { id?: string; ts: number }>()
const INSTALL_ID_TTL_MS = 5 * 60_000
const INSTALL_ID_NEGATIVE_TTL_MS = 60_000

function rememberConnectionInstallId(connectionId: string, statusBody: any) {
  const raw = statusBody && typeof statusBody === 'object' ? statusBody.install_id : undefined
  const id = typeof raw === 'string' && raw.trim() ? raw.trim() : undefined
  connectionInstallIds.set(connectionId, { id, ts: Date.now() })

  return id
}

async function probeConnectionInstallId(connectionId: string, descriptor: any): Promise<string | undefined> {
  const cached = connectionInstallIds.get(connectionId)

  if (cached && Date.now() - cached.ts < (cached.id ? INSTALL_ID_TTL_MS : INSTALL_ID_NEGATIVE_TTL_MS)) {
    return cached.id
  }

  try {
    const status: any = await getJsonForBackend(descriptor, '/api/status', { timeoutMs: 8_000 })

    return rememberConnectionInstallId(connectionId, status)
  } catch {
    // Keep any previously-known id (identity is stable; a transient fetch
    // failure must not flap the roster collapse), but do not cache a MISS
    // over it.
    if (cached?.id) {
      return cached.id
    }

    connectionInstallIds.set(connectionId, { id: undefined, ts: Date.now() })

    return undefined
  }
}

async function probeSshProfileInventory(connection) {
  if (
    !shouldRetrySshInventory(
      sshRosterCache.has(connection.id),
      sshInventoryAttemptedAt.get(connection.id),
      Date.now(),
      SSH_INVENTORY_RETRY_MS
    )
  ) {
    return
  }

  sshInventoryAttemptedAt.set(connection.id, Date.now())

  const sshConfig = normalizeSshConfig({
    mode: 'ssh',
    host: connection.host,
    user: connection.user,
    port: connection.port,
    keyPath: connection.keyPath,
    remoteHermesPath: connection.remoteHermesPath
  })

  if (!sshConfig) {
    return
  }

  const ssh = createSshProbeConnection(
    { host: sshConfig.host, user: sshConfig.user, port: sshConfig.port, keyPath: sshConfig.keyPath },
    { rememberLog: sshRememberLog }
  )

  try {
    await ssh.open()
    const profiles = await remoteLifecycle.listRemoteHermesProfiles(ssh)

    if (profiles.length > 0) {
      sshRosterCache.set(connection.id, profiles)
    }
  } catch (error: any) {
    sshRememberLog(`[ssh] profile inventory failed for ${connection.id}: ${error?.message || error}`)
  } finally {
    try {
      await ssh.close()
    } catch {
      void 0
    }
  }
}

async function enumerateRegistryAgentSources(registry = readDesktopConnectionsRegistry()) {
  // One dead source must not wedge the whole roster: ensureRegistryBackend on
  // an unreachable remote can block up to the 45s readiness timeout, and the
  // Bot Mode poll runs every 5s — each poll queued behind the dead dial, so
  // the renderer painted stale rows for the entire outage (and the roster IPC
  // hung >30s in live repro). Bound each source's enumeration; a timeout is
  // reported like any other unreachable source and retried on the next poll.
  const perSourceTimeoutMs = 10_000

  const withEnumerationDeadline = async <T>(work: Promise<T>): Promise<T> => {
    let timer: ReturnType<typeof setTimeout> | null = null

    try {
      return await Promise.race([
        work,
        new Promise<never>((_resolve, reject) => {
          timer = setTimeout(() => reject(new Error('roster enumeration timed out')), perSourceTimeoutMs)
        })
      ])
    } finally {
      if (timer !== null) {
        clearTimeout(timer)
      }
    }
  }

  return Promise.all(
    registry.connections.map(async connection => {
      let raw: { connection: typeof connection; error?: string; installId?: string; profiles: null | string[] }

      try {
        // SSH roster listing must never spawn a dashboard. A stale
        // sshConnections key used to fall into ensureRegistryBackend and
        // respawn Spark/Mini every Bot Mode poll (~5s), then the mux died
        // (ECONNRESET / liveness probe drop).
        if (connection.kind === 'ssh') {
          await probeSshProfileInventory(connection)
          raw = { connection, profiles: null, error: 'connect-on-demand' }
        } else {
          // Same connect-on-demand courtesy for the forced-local path: when
          // the primary route is remote, enumerating "This device" would
          // SPAWN a local backend this user has never asked for — a phantom
          // `default` agent that also forces -device handle disambiguation
          // onto the real one (remote-gateway-only desktops showed their main
          // agent twice, Aug 17 2026). Enumerate the local source only when
          // it is the delegate route (local-primary desktops, unchanged
          // behavior) or a forced-local child is ALREADY pooled (the user
          // opened one).
          if (connection.kind === 'local') {
            const localRoute = resolveRegistryLocalRoute('default', {
              globalRemote: globalRemoteActive(),
              profileRemoteOverride: Boolean(profileHasRemoteOverride(primaryProfileKey()))
            })

            if (shouldDeferLocalEnumeration(localRoute, backendPool.keys(), connection.id)) {
              return { connection, profiles: null, error: 'connect-on-demand' }
            }
          }

          const descriptor: any = await withEnumerationDeadline(
            Promise.resolve(ensureRegistryBackend(connection.id, null))
          )

          const body: any = await getJsonForBackend(descriptor, '/api/profiles', { timeoutMs: 8_000 })

          // Cached with a TTL, so the 5s roster poll usually pays zero extra
          // requests for the backend-identity probe.
          const installId = await probeConnectionInstallId(connection.id, descriptor)

          const profiles = Array.isArray(body?.profiles)
            ? body.profiles.map(p => String(p?.name || '').trim()).filter(Boolean)
            : []

          // The root HERMES_HOME is an agent too; enumerations that omit it
          // (older backends list only named profiles) still get a default row.
          if (!profiles.includes('default')) {
            profiles.unshift('default')
          }

          raw = { connection, profiles, ...(installId ? { installId } : {}) }
        }
      } catch (error: any) {
        raw = { connection, profiles: null, error: String(error?.message || error) }
      }

      if (raw.profiles && raw.profiles.length > 0) {
        sshRosterCache.set(connection.id, raw.profiles)
      }

      const remembered = rememberSshEnumeration(raw, sshRosterCache.get(connection.id), connection.kind)

      return { connection, ...remembered, ...(raw.installId ? { installId: raw.installId } : {}) }
    })
  )
}

ipcMain.handle('hermes:agents:roster', async () => {
  const registry = readDesktopConnectionsRegistry()
  const enumerations = await enumerateRegistryAgentSources(registry)

  return {
    agents: buildAgentRoster(enumerations, { primaryConnectionId: registry.primary }),
    // The active gateway owns the renderer's profiles.list — union agents
    // that report THIS connection are the same identities, not extra rows.
    // Expose the primary id so the plugin merger can annotate them in place
    // instead of appending duplicates (remote-only desktops doubled every
    // bot otherwise; see #88344).
    primaryConnectionId: registry.primary,
    sources: enumerations.map(({ connection, error, installId, profiles }) => ({
      connectionId: connection.id,
      label: connection.label,
      kind: connection.kind,
      reachable: profiles !== null,
      ...(installId ? { installId } : {}),
      ...(error ? { error } : {})
    }))
  }
})

// Registry-scoped fresh WS URL: the (connectionId, profile) analogue of
// hermes:gateway:ws-url. Same single-use-ticket discipline for OAuth sources.
ipcMain.handle('hermes:gateway:ws-url-for', async (_event, payload) => {
  const { connectionId, profile } = payload && typeof payload === 'object' ? (payload as any) : ({} as any)

  return gatewayWsUrlIpcResult(async () => {
    const connection: any = await ensureRegistryBackend(connectionId, profile)

    if (connection.authMode === 'oauth') {
      const ticket = await mintGatewayWsTicket(connection.baseUrl, connection.headers)
      const wsUrl = buildGatewayWsUrlWithTicket(connection.baseUrl, ticket)

      rememberRemoteWsHeaders(wsUrl, connection.headers)

      return registryGatewayWsUrl(connection, wsUrl)
    }

    rememberRemoteWsHeaders(connection.wsUrl, connection.headers)

    return registryGatewayWsUrl(connection, connection.wsUrl)
  })
})

// Fan out `hermes update` to every eligible registered connection at once.
// Cloud entries are excluded (platform-managed); each dispatch reports
// independently so one dead LAN box can't wedge the batch. Local reuses the
// app's own update pipeline; remote/ssh POST the backend's own
// /api/hermes/update endpoint (the dashboard updater), which runs
// `hermes update` on THAT machine.
ipcMain.handle('hermes:connections:update-all', async (_event, payload) => {
  const registry = readDesktopConnectionsRegistry()

  // Optional renderer-side exclusions: the everything-update flow dispatches
  // the ACTIVE backend through its own detailed-progress path and chains the
  // local client apply LAST (it relaunches the app), so it excludes those ids
  // here to avoid double-dispatch. No payload keeps the Settings button's
  // original all-rows behavior byte-identical.
  const excludeIds = new Set<string>(
    Array.isArray((payload as any)?.excludeIds) ? (payload as any).excludeIds.map((id: unknown) => String(id)) : []
  )

  const results = await Promise.all(
    registry.connections
      .filter(connection => !excludeIds.has(connection.id))
      .map(async connection => {
        const base = { connectionId: connection.id, label: connection.label, kind: connection.kind }
        const eligibility = updateEligibility(connection)

        if (!eligibility.eligible) {
          return { ...base, ok: false, skipped: true, reason: eligibility.reason }
        }

        try {
          if (connection.kind === 'local') {
            // The app-managed runtime updates through the same pipeline as the
            // Settings → Updates button (marker + venv gate + relaunch flow).
            const result: any = await applyUpdates({})

            return { ...base, ok: result?.ok !== false, detail: result?.message || 'update started' }
          }

          const descriptor: any = await ensureRegistryBackend(connection.id, null)

          const body: any = await postJsonForBackend(descriptor, '/api/hermes/update', {}, { timeoutMs: 15_000 })

          if (body?.ok === false) {
            // The backend refused (docker/nix/externally-managed installs) —
            // surface ITS message, per-row, instead of failing the batch.
            return {
              ...base,
              ok: false,
              skipped: true,
              reason: body?.error || 'backend-refused',
              detail: body?.message
            }
          }

          return { ...base, ok: true, detail: body?.message || 'update started' }
        } catch (error: any) {
          return { ...base, ok: false, error: String(error?.message || error) }
        }
      })
  )

  return { ok: true, results }
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

    const nativeAt = await ensureNativeAccessToken(descriptor.baseUrl).catch(() => null)

    if (nativeAt) {
      return fetchJson(url, null, {
        method: opts.method,
        body: opts.body,
        timeoutMs: opts.timeoutMs,
        bearer: nativeAt,
        headers: descriptor.headers
      })
    }

    return fetchJsonViaOauthSession(url, {
      method: opts.method,
      body: opts.body,
      timeoutMs: opts.timeoutMs,
      headers: descriptor.headers
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

ipcMain.handle('hermes:connection-config:probe', async (_event, rawUrl) => probeRemoteAuthMode(rawUrl))
ipcMain.handle('hermes:connection-config:oauth-login', async (_event, rawUrl) => {
  // Capability-gated login (RFC 8252). Probe the gateway's public /api/status:
  //   - advertises "native_pkce" in auth_flows → run the system-browser +
  //     loopback + PKCE flow. No embedded webview, tokens held by the app
  //     (encrypted keychain), REST/WS authenticated by bearer — no cookies.
  //   - older gateway without native_pkce → fall back to the legacy embedded
  //     BrowserWindow cookie flow, preserving compatibility.
  // This is the "observable ladder + compatibility fallback tied to an
  // identified older runtime" the desktop guide requires.
  const baseUrl = normalizeRemoteBaseUrl(rawUrl)

  let statusBody: any = null

  try {
    statusBody = await fetchPublicJson(`${baseUrl}/api/status`, { timeoutMs: 8_000 })
  } catch {
    // Can't read status — fall through to the embedded flow, which has its
    // own error handling and works against any gated gateway.
  }

  const strategy = resolveLoginStrategy(statusBody)

  if (strategy === 'native') {
    try {
      const tokens = await runNativeLogin(baseUrl, {
        openExternal: url => shell.openExternal(url),
        postJson: (url, body, opts) => postJsonNoAuth(url, body, opts),
        rememberLog
      })

      _storeNativeTokens(baseUrl, tokens)
      // Confirmed sign-in — release the reauth latch so the next
      // startHermes() re-dials instead of replaying the stale rejection.
      remoteReauthFailure = null

      return { ok: true, baseUrl, connected: true }
    } catch (error) {
      rememberLog(
        `[native-oauth] native login failed (${
          error instanceof Error ? error.message : String(error)
        }); falling back to embedded flow`
      )
      // Fall through to the embedded flow so a native-flow hiccup (blocked
      // loopback, user closed the browser) still lets the user sign in.
    }
  }

  // Legacy embedded-webview cookie flow.
  await openOauthLoginWindow(baseUrl)

  const connected = await hasOauthSessionCookie(baseUrl)

  // Only a CONFIRMED sign-in releases the latch. A cancelled/closed login
  // window must leave it set, or the overlay's "Sign in" button starts
  // flickering again on the next retry.
  if (connected) {
    remoteReauthFailure = null
  }

  return { ok: true, baseUrl, connected }
})
ipcMain.handle('hermes:connection-config:oauth-logout', async (_event, rawUrl) => {
  const baseUrl = rawUrl ? normalizeRemoteBaseUrl(rawUrl) : ''
  await clearOauthSession(baseUrl || undefined)

  // Also drop any native (RFC 8252) bearer tokens for this gateway so a
  // logout clears BOTH auth shapes.
  if (baseUrl) {
    _clearNativeTokens(baseUrl)
  }

  // Report against the SAME liveness notion the Settings indicator uses
  // (AT-or-RT cookie, or a native token) so a logout that left any session
  // behind is reflected as still-connected rather than silently signed-out.
  const connected = baseUrl ? (await hasLiveOauthSession(baseUrl)) || hasNativeSession(baseUrl) : false

  return { ok: true, connected }
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
  const previousConfig = readDesktopConnectionConfig()
  const previousRegistry = readDesktopConnectionsRegistry()
  const config = coerceDesktopConnectionConfig(payload, previousConfig)

  const key = connectionScopeKey(payload?.profile)
  const scope = key || ''
  const nextRegistry = key ? previousRegistry : reconcileAppliedGlobalConnection(previousRegistry, config)

  await applyConnectionConfigAtomically({
    previousConfig,
    previousRegistry,
    nextConfig: config,
    nextRegistry,
    // Exercise the same authenticated REST + real WebSocket legs before either
    // config file changes. A rejected OAuth session or blocked /api/ws leaves
    // the previous primary/current connection intact.
    preflight: !key && modeIsRemoteLike(config.mode) ? () => testDesktopConnectionConfig(payload) : undefined,
    writeConfig: writeDesktopConnectionConfig,
    writeRegistry: writeDesktopConnectionsRegistry,
    apply: () =>
      applyConnectionChange({
        cancelAndWait: value => sshBootstrapCoordinator.cancelAndWait(value),
        isPrimary: !key || key === primaryProfileKey(),
        rehomePrimary: () =>
          rehomePrimaryConnection({
            clearLocalBootstrapFailure: () => {
              // A remote connection bypasses local runtime/bootstrap failures. Clear
              // the local-install latch so unsupported/failure escape paths can re-home.
              bootstrapFailure = null
            },
            mode: config.mode,
            notifyConnectionApplied: sendConnectionApplied,
            resumeFirstRunRemote: abandonFirstRunSetupChoiceForRemoteApply,
            teardownPrimaryBackend: teardownPrimaryBackendAndWait
          }),
        scope,
        sendApplied: sendConnectionApplied,
        stopPool: stopPoolBackend,
        teardownPrimary: () => teardownPrimaryBackendAndWait({ soft: true }),
        teardownSsh: value => teardownSshConnection(value || null)
      })
  })

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
    const registrySources = await pooledRegistrySessionSources()

    if (remoteProfiles.length === 0 && registrySources.length === 0) {
      return undefined // no remote profiles and no connected registry gateways → local fast path
    }

    const requested = (searchParams.get('profile') || 'all').trim() || 'all'

    if (requested !== 'all') {
      return profileHasRemoteOverride(requested) ? remoteSessionList(requested, searchParams) : undefined
    }

    return mergeRemoteProfileSessions(searchParams, remoteProfiles)
  }

  // Batched sidebar slices. With no remote profiles the local batched endpoint
  // (one DB open per profile) serves it directly — take the fast path. When
  // remotes exist, fan the three slices back out to the per-slice
  // /api/profiles/sessions path (which already merges remote rows correctly) and
  // reassemble; local profiles fall back to three primary reads there, but
  // remote correctness is preserved.
  if (method === 'GET' && pathname === '/api/profiles/sessions/sidebar') {
    const remoteProfiles = configuredRemoteProfileNames()
    const registrySources = await pooledRegistrySessionSources()

    if (remoteProfiles.length === 0 && registrySources.length === 0) {
      return undefined // local fast path → batched endpoint's single DB open
    }

    const { recents: recentsSp, cron: cronSp, messaging: messagingSp } = buildSidebarSessionSliceParams(searchParams)

    const [recents, cron, messaging] = await Promise.all([
      fetchProfilesSessionSlice(recentsSp, remoteProfiles),
      fetchProfilesSessionSlice(cronSp, remoteProfiles),
      fetchProfilesSessionSlice(messagingSp, remoteProfiles)
    ])

    return {
      recents: {
        sessions: rowsOf(recents),
        total: Number(recents?.total) || 0,
        profile_totals: recents?.profile_totals || {}
      },
      cron: { sessions: rowsOf(cron) },
      messaging: {
        sessions: rowsOf(messaging),
        total: Number(messaging?.total) || rowsOf(messaging).length
      },
      errors: []
    }
  }

  // Per-session read/mutation. Owner is in ?profile= (reads) or request.profile
  // (mutations). Two remote shapes:
  //  - per-profile override: route to that profile's own remote, sans profile
  //    param (it serves its own state.db natively).
  //  - global remote mode: ONE backend serves every profile via ?profile=, so
  //    route there and KEEP the profile param so it opens the right state.db.
  if (/^\/api\/sessions\/[^/]+(\/messages)?$/.test(pathname)) {
    let profile = (searchParams.get('profile') || request.profile || '').trim()

    if (!profile) {
      // No explicit owner hint (#85834). The list endpoints above already know
      // which remote profile owns each row (remoteSessionList tags s.profile),
      // but a caller without a hint used to fall straight through to the LOCAL
      // backend and 404 on its state.db even though the session lives on a
      // remote. Consult the same remote lists to find the owner; only fall
      // through when the id is genuinely unknown remotely.
      const sessionId = decodeURIComponent(pathname.split('/')[3] || '')
      profile = (await remoteOwnerProfileForSession(sessionId)) || ''

      if (!profile) {
        return undefined
      }
    }

    // Preserve every non-profile query param (limit/offset/order pagination —
    // stripping them made getAllSessionMessages loop the same default page
    // against paginating remote backends).
    const passthroughParams = new URLSearchParams(searchParams)
    passthroughParams.delete('profile')
    const passthroughQuery = passthroughParams.toString()

    if (profileHasRemoteOverride(profile)) {
      if (method === 'GET') {
        return fetchJsonForProfile(profile, passthroughQuery ? `${pathname}?${passthroughQuery}` : pathname)
      }

      const body = request.body && typeof request.body === 'object' ? { ...request.body } : request.body

      if (body) {
        delete body.profile
      }

      return requestJsonForProfile(profile, pathname, method, body)
    }

    if (globalRemoteActive()) {
      // Single global backend: keep ?profile= so it opens the right state.db.
      passthroughParams.set('profile', profile)
      const path = `${pathname}?${passthroughParams.toString()}`

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
  const data = await fetchRemoteProfileSessions(profile, searchParams, fetchJsonForProfile)

  for (const s of rowsOf(data)) {
    s.profile = profile
    s.is_default_profile = false
  }

  return { ...(data as any), sessions: rowsOf(data) }
}

// #85834: find which remote profile owns a session id when the caller gave no
// profile hint (pure lookup lives in profile-session-routing.ts). Results are
// memoized briefly so a burst of hint-less reads (transcript + messages)
// costs one sweep across the configured remotes.
const remoteOwnerBySessionId = new Map<string, { at: number; profile: null | string }>()
const REMOTE_OWNER_CACHE_TTL_MS = 30_000

async function remoteOwnerProfileForSession(sessionId: string) {
  if (!sessionId) {
    return null
  }

  const remoteProfiles = configuredRemoteProfileNames()

  if (remoteProfiles.length === 0) {
    return null
  }

  const cached = remoteOwnerBySessionId.get(sessionId)

  if (cached && Date.now() - cached.at < REMOTE_OWNER_CACHE_TTL_MS) {
    return cached.profile
  }

  const owner = await findRemoteOwnerProfileForSession(sessionId, remoteProfiles, (profile, params) =>
    remoteSessionList(profile, params)
  ).catch(() => null)

  remoteOwnerBySessionId.set(sessionId, { at: Date.now(), profile: owner })

  return owner
}

// Resolve one /api/profiles/sessions slice with remote profiles spliced in —
// the same branch logic as the GET /api/profiles/sessions intercept, but always
// returns data (never `undefined`) so a batched caller can compose slices. A
// specific local profile reads from the local primary; a remote-override profile
// reads from its remote; 'all' merges every remote into the primary aggregate.
async function fetchProfilesSessionSlice(searchParams, remoteProfiles) {
  const requested = (searchParams.get('profile') || 'all').trim() || 'all'

  if (requested !== 'all') {
    if (profileHasRemoteOverride(requested)) {
      return remoteSessionList(requested, searchParams)
    }

    return fetchPrimaryProfileSessions(searchParams, fetchJsonForProfile)
  }

  return mergeRemoteProfileSessions(searchParams, remoteProfiles)
}

// Unified list: primary's local aggregate, with each remote profile's stale local
// rows/totals swapped for the remote's real ones, re-sorted by recency and
// re-windowed to the requested page. A dead remote contributes nothing rather
// than breaking the sidebar. Connected registry gateways' sessions are spliced
// in too (#88880) — the unified Sessions list shows EVERY connected gateway's
// chats, tagged with connection_id + profile so opens route correctly.
async function mergeRemoteProfileSessions(searchParams, remoteProfiles) {
  const limit = Math.max(1, Number(searchParams.get('limit')) || 20)
  const offset = Math.max(0, Number(searchParams.get('offset')) || 0)
  const order = searchParams.get('order') === 'created' ? 'started_at' : 'last_active'

  const base = (await fetchPrimaryProfileSessions(searchParams, fetchJsonForProfile)) as any

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

  // Registry gateways (v2 connections): splice every CONNECTED gateway's rows
  // into the unified list. Only already-pooled backends are read — a sidebar
  // refresh must never dial or spawn a backend (the Bot Mode roster-respawn
  // trap). Reads omit include_hidden, so Bot Mode's hidden canonical chats
  // stay out of the global list, same as local sessions.
  const registrySources = await pooledRegistrySessionSources()

  if (registrySources.length) {
    const registryRows = await fetchRegistrySessionRows(registrySources, remoteParams, (descriptor, path) =>
      getJsonForBackend(descriptor, path, { timeoutMs: 10_000 })
    )

    const { added } = spliceRegistrySessionRows(merged, registryRows, profileTotals)
    total += added
  }

  const recency = s => s?.[order] ?? s?.started_at ?? 0
  merged.sort((a, b) => recency(b) - recency(a))

  return {
    ...(base as any),
    sessions: mergeProfileSessionWindow(merged, offset, limit),
    total,
    profile_totals: profileTotals
  }
}

// Every CONNECTED registry gateway as a session source: resolved descriptors
// straight from the backend pool, never dialing. SSH sources contribute one
// backend per pooled (connection, profile) scope; remote/cloud sources are one
// shared host (any pooled scope's descriptor serves the cross-profile read).
// The primary local connection is excluded — the primary aggregate already
// carries local rows.
async function pooledRegistrySessionSources(): Promise<RegistrySessionSource[]> {
  const registry = readDesktopConnectionsRegistry()
  const sources: RegistrySessionSource[] = []

  for (const connection of registry.connections) {
    if (connection.kind === 'local') {
      continue
    }

    const prefix = backendScopePrefix(connection.id)

    const pooled = [...backendPool.entries()].filter(
      ([key, entry]) => key.startsWith(prefix) && entry.connectionPromise
    )

    if (pooled.length === 0) {
      continue
    }

    const backends: Array<{ descriptor: unknown; profileLabel: null | string }> = []

    for (const [key, entry] of connection.kind === 'ssh' ? pooled : pooled.slice(0, 1)) {
      try {
        // Already-resolved for a connected backend; a still-dialing entry is
        // skipped via the timeout guard rather than blocking the sidebar.
        const descriptor = await Promise.race([
          entry.connectionPromise,
          new Promise((_, reject) => setTimeout(() => reject(new Error('pending')), 2_000))
        ])

        backends.push({
          descriptor,
          profileLabel: connection.kind === 'ssh' ? key.slice(prefix.length) || 'default' : null
        })
      } catch {
        // Dead or still-connecting backend — contributes nothing this refresh.
      }
    }

    if (backends.length) {
      sources.push({ backends, connectionId: connection.id, kind: connection.kind })
    }
  }

  return sources
}

async function dispatchRegistryApiRequest(
  request,
  registryConnectionId,
  routeProfile = request?.profile,
  requestProfile = request?.profile
) {
  const connection: any = await ensureRegistryBackend(registryConnectionId, routeProfile)

  const requestPath = pathForRegistryBackendRequest(request.path, requestProfile, connection)

  const response = await fetchJsonForBackend(connection, requestPath, {
    method: request?.method,
    body: request?.body,
    upload: request?.upload,
    timeoutMs: resolveTimeoutMs(request?.timeoutMs, DEFAULT_FETCH_TIMEOUT_MS)
  })

  return (request?.method || 'GET').toUpperCase() === 'GET'
    ? tagRegistrySessionResponse(requestPath, response, registryConnectionId)
    : response
}

function registryConnectionKind(connectionId) {
  const registry = readDesktopConnectionsRegistry()
  const source = registry.connections.find(connection => connection.id === connectionId)

  if (!source) {
    throw new Error(`No connection with id "${connectionId}".`)
  }

  return source.kind
}

async function teardownConnectionScopedProfileBackend(connectionId, profile) {
  const key = backendScopeKey(connectionId, profile)
  await Promise.all([
    poolStopper.stop(key),
    sshBootstrapCoordinator.cancelAndWait(key).then(() => teardownSshConnection(key))
  ])
}

async function handleHermesApiRequest(request) {
  // Registry-pinned request (request.connectionId): the renderer is working
  // against a REGISTERED gateway connection, so the data — cron jobs and their
  // run sessions included — lives in THAT host's state.db, not any local
  // profile's. Resolve the backend through the registry (same pool the job
  // list and WS traffic use) instead of the legacy profile route; a shared
  // remote/cloud host serves every profile via ?profile=, so scope the path.
  // An absent/empty id falls through to the byte-identical v1 route below.
  // Explicit `local` stays registry-pinned so it cannot inherit a v1 remote.
  const registryConnectionId = apiRequestRegistryConnectionId(request)

  if (registryConnectionId) {
    return dispatchRegistryApiRequest(request, registryConnectionId)
  }

  // Remote-profile session requests would otherwise hit the local primary off
  // each profile's on-disk state.db — fine for local profiles, but a remote
  // profile's sessions live on its remote host, so the UI's IDs 404 (or mutations
  // no-op) the moment they run there. Route reads + mutations to the remote.
  const rerouted = await interceptSessionRequestForRemote(request)

  if (rerouted !== undefined) {
    return rerouted
  }

  const profileRename = await prepareProfileRenameRequest(request)
  const tornDownProfile = await prepareProfileDeleteRequest(request)

  const profile = request?.profile
  // After tearing down a backend for profile deletion, route to the primary
  // backend instead of spawning a fresh pool backend.  A freshly spawned
  // backend calls ensure_hermes_home() which recreates the profile directory,
  // defeating the deletion and leaving a zombie process.
  //
  // Safe local-profile REST calls also stay on the primary dashboard and carry
  // ?profile=. Endpoints that cannot honor that scope retain their pooled
  // backend so a destructive call can never fall through to the primary home.
  //
  // A profile rename tears down the old-name backend the same way; for a
  // primary rename the lifecycle has already made `default` the temporary
  // primary until the PATCH settles, so the request routes there.
  const apiRoute = resolveProfileApiRequest(profile, request.path, profileRouteOptions(profile, request))

  const routeProfile = profileRename
    ? profileRename.routeProfile
    : resolveRouteProfile(tornDownProfile, apiRoute.backendProfile)

  let response

  try {
    const connection = await ensureBackend(routeProfile)
    const timeoutMs = resolveTimeoutMs(request?.timeoutMs, DEFAULT_FETCH_TIMEOUT_MS)

    const url = `${connection.baseUrl}${apiRoute.requestPath}`

    // OAuth gateways authenticate REST via EITHER a native bearer token
    // (cookieless RFC 8252 flow) OR the HttpOnly session cookie held in the OAuth
    // partition. Prefer the native bearer when present (mirroring
    // mintGatewayWsTicket): the native flow never sets a cookie, so routing an
    // oauth-mode REST call through the cookie-only path returns 401 no_cookie even
    // though a valid bearer is held. Cookie mode rides Electron's net stack bound
    // to the OAuth partition so the cookie attaches automatically. Token/local
    // modes keep using the static session-token header.
    if (connection.authMode === 'oauth') {
      // The OAuth path rides electron.net with JSON headers; multipart isn't
      // wired there. Fail loudly rather than corrupting the upload.
      if (request?.upload) {
        throw new Error('File uploads are not supported against OAuth-gated remote backends yet.')
      }

      // Native bearer first (cookieless). ensureNativeAccessToken transparently
      // refreshes a near-expiry AT via /auth/native/refresh; a null return means
      // no native session (resolveOauthRestAuth then selects the cookie path).
      const nativeAt = await ensureNativeAccessToken(connection.baseUrl).catch(() => null)
      const restAuth = resolveOauthRestAuth(nativeAt)

      if (restAuth.kind === 'bearer') {
        response = await fetchJson(url, null, {
          method: request?.method,
          body: request?.body,
          timeoutMs,
          bearer: restAuth.token
        })
      } else {
        response = await fetchJsonViaOauthSession(url, {
          method: request?.method,
          body: request?.body,
          timeoutMs
        })
      }
    } else {
      response = await fetchJson(url, connection.token, {
        method: request?.method,
        body: request?.body,
        upload: request?.upload,
        timeoutMs
      })
    }
  } catch (error) {
    // A failed rename PATCH must not strand the app on the temporary primary:
    // restore the original active profile and restart its backend.
    if (profileRename) {
      try {
        await profileRename.rollback()
      } catch (rollbackError) {
        rememberLog(`Failed to restore primary profile after rename error: ${String(rollbackError)}`)
      }
    }

    throw error
  }

  await profileRename?.complete()

  return response
}

ipcMain.handle('hermes:api', async (_event, request) => {
  // Hold the deletion gate for BOTH profile deletes and renames: a concurrent
  // renderer reconnect entering ensureBackend() mid-mutation would otherwise
  // respawn the old-name backend and recreate its HERMES_HOME (#45474).
  const deletingProfile = profileNameFromDeleteRequest(request)
  const mutatingProfile = deletingProfile || profileRenameFromRequest(request)?.oldName || null
  const registryConnectionId = apiRequestRegistryConnectionId(request)

  if (deletingProfile && registryConnectionId) {
    return dispatchConnectionScopedProfileDelete(request, {
      acquire: profile => profileDeletionGate.acquire(profile),
      connectionKind: connectionId => registryConnectionKind(connectionId),
      dispatch: routeProfile =>
        dispatchRegistryApiRequest(request, registryConnectionId, routeProfile, deletingProfile),
      isDefaultProfile: profile => profile === 'default',
      isValidProfileName: profile => PROFILE_NAME_RE.test(profile),
      prepareLocal: localRequest => prepareProfileDeleteRequest(localRequest).then(() => undefined),
      teardownConnection: (connectionId, profile) => teardownConnectionScopedProfileBackend(connectionId, profile)
    })
  }

  if (!mutatingProfile) {
    return handleHermesApiRequest(request)
  }

  const releaseProfileDeletion = profileDeletionGate.acquire(mutatingProfile)

  return handleHermesApiRequest(request).finally(releaseProfileDeletion)
})

// One deduper per cross-window cue — the choke point every window shares. Main
// handles IPC serially, so the first window to claim a key wins with no race.
const isDuplicateNotification = createEventDeduper()
const claimedAmbientCue = createEventDeduper()

// A window asks "do I own this ambient cue (turn-end sound / spoken reply)?".
// The first caller within the window gets true; peers get false and stay quiet.
ipcMain.handle('hermes:ambient:claim', (_event, key) => !claimedAmbientCue(String(key ?? '')))

ipcMain.handle('hermes:notify', (_event, payload) => {
  if (!Notification.isSupported()) {
    return false
  }

  // Multiple full windows each run their own renderer throttle, so the same
  // kind+session can arrive here twice. Collapse it at this single choke point.
  // Return true (not false): a notification for the event IS being shown by the
  // first caller, so the settings "send test" success probe stays honest.
  if (isDuplicateNotification(`${payload?.kind ?? ''}:${payload?.sessionId ?? payload?.tag ?? ''}`)) {
    return true
  }

  // Action buttons render only on signed macOS builds; elsewhere they're dropped
  // and the body click still works.
  const actions = Array.isArray(payload?.actions) ? payload.actions : []
  const icon = typeof payload?.icon === 'string' && payload.icon.trim() ? payload.icon.trim() : undefined

  const notification = new Notification({
    title: payload?.title || 'Hermes',
    body: payload?.body || '',
    silent: Boolean(payload?.silent),
    ...(icon ? { icon } : {}),
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

    // Plugin / session-less activation — serializable path (+ optional notifyId
    // for renderer callbacks). Same vocabulary as hermes://index-network/….
    if (payload?.activate || payload?.notifyId) {
      mainWindow.webContents.send('hermes:notification-activate', {
        activate: payload?.activate,
        notifyId: payload?.notifyId,
        tag: payload?.tag
      })
    }
  })
  notification.on('action', (_actionEvent, index) => {
    if (!mainWindow || mainWindow.isDestroyed()) {
      return
    }

    const action = actions[index]

    if (!action?.id) {
      return
    }

    // Approvals keep the existing session-scoped channel.
    if (payload?.sessionId && !payload?.notifyId && !payload?.activate) {
      mainWindow.webContents.send('hermes:notification-action', { sessionId: payload.sessionId, actionId: action.id })

      return
    }

    focusWindow(mainWindow)
    mainWindow.webContents.send('hermes:notification-activate', {
      actionId: action.id,
      activate: action.activate || payload?.activate,
      notifyId: payload?.notifyId,
      tag: payload?.tag
    })
  })
  notification.show()

  return true
})

// Data-URL file load cap (composer attach + local previews). Main owns the
// persisted MB value so every IPC read honours Settings → Chat without the
// renderer having to pass maxBytes on each call. Default is 16 MB; clamp
// lives in hardening.ts.
const DATA_URL_READ_MAX_CONFIG_PATH = path.join(app.getPath('userData'), 'data-url-read-max.json')

function readPersistedDataUrlReadMaxMb() {
  try {
    return clampDataUrlReadMaxMb(JSON.parse(fs.readFileSync(DATA_URL_READ_MAX_CONFIG_PATH, 'utf8')).maxMb)
  } catch {
    return DATA_URL_READ_DEFAULT_MAX_MB
  }
}

let dataUrlReadMaxMb = readPersistedDataUrlReadMaxMb()

function persistDataUrlReadMaxMb(maxMb) {
  const next = clampDataUrlReadMaxMb(maxMb)
  dataUrlReadMaxMb = next

  try {
    fs.mkdirSync(path.dirname(DATA_URL_READ_MAX_CONFIG_PATH), { recursive: true })
    fs.writeFileSync(DATA_URL_READ_MAX_CONFIG_PATH, JSON.stringify({ maxMb: next }, null, 2), 'utf8')
  } catch (error) {
    rememberLog(`[data-url-read-max] write failed: ${error.message}`)
  }

  return next
}

ipcMain.handle('hermes:data-url-read-max:get', () => ({
  maxMb: dataUrlReadMaxMb,
  // Keep the default bytes constant visible for tests / diagnostics.
  defaultMaxMb: DATA_URL_READ_DEFAULT_MAX_MB,
  maxBytes: dataUrlReadMaxBytesFromMb(dataUrlReadMaxMb)
}))

ipcMain.handle('hermes:data-url-read-max:set', (_event, maxMb) => {
  const next = persistDataUrlReadMaxMb(maxMb)

  return {
    maxMb: next,
    defaultMaxMb: DATA_URL_READ_DEFAULT_MAX_MB,
    maxBytes: dataUrlReadMaxBytesFromMb(next)
  }
})

ipcMain.handle('hermes:readFileDataUrl', async (_event, filePath) => {
  return readFileDataUrlForIpc(filePath, {
    maxBytes: dataUrlReadMaxBytesFromMb(dataUrlReadMaxMb),
    mimeType: mimeTypeForPath(resolveRequestedPathForIpc(filePath, { purpose: 'File preview' })),
    purpose: 'File preview'
  })
})

// Remote attachment transfer is independent of the preview / Settings path.
// Keep a finite cap so Electron + base64 memory stays bounded while archives
// can exceed the default 16 MiB preview ceiling (and still fit the gateway
// WebSocket frame limit after base64 expansion).
ipcMain.handle('hermes:readFileDataUrlForAttach', async (_event, filePath) => {
  return readFileDataUrlForIpc(filePath, {
    maxBytes: ATTACHMENT_UPLOAD_DEFAULT_MAX_BYTES,
    mimeType: mimeTypeForPath(resolveRequestedPathForIpc(filePath, { purpose: 'Attachment upload' })),
    purpose: 'Attachment upload'
  })
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

// Runtime desktop plugins load their FULL source through this door.
// `hermes:readFileText` is the *preview* read and silently truncates at
// TEXT_PREVIEW_MAX_BYTES (512 KiB) — for a plugin that means evaluating half a
// file. Dedicated generous cap, full read, and a hard EFBIG (via maxBytes)
// instead of truncation when the source exceeds it.
const PLUGIN_SOURCE_MAX_BYTES = 16 * 1024 * 1024

ipcMain.handle('hermes:readPluginSource', async (_event: unknown, filePath: unknown) => {
  const { resolvedPath, stat } = await resolveReadableFileForIpc(filePath, {
    maxBytes: PLUGIN_SOURCE_MAX_BYTES,
    purpose: 'Plugin source'
  })

  return {
    byteSize: stat.size,
    path: resolvedPath,
    text: await fs.promises.readFile(resolvedPath, 'utf8'),
    truncated: false
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
      // On a Windows host with a WSL backend the cwd may be a POSIX/WSL path;
      // bridge it to a UNC/drive form the native dialog can actually open.
      const bridged = IS_WINDOWS
        ? resolvePickerDefaultPath(String(options.defaultPath), undefined, options?.profile)
        : String(options.defaultPath)

      resolvedDefaultPath = bridged ? path.resolve(bridged) : undefined
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

// Native save-location picker (profile export etc.) — the write itself happens
// elsewhere (the backend, for profile archives); this only picks the path.
ipcMain.handle('hermes:selectSavePath', async (_event, options: any = {}) => {
  const result = await dialog.showSaveDialog(mainWindow, {
    title: options?.title || 'Save',
    defaultPath: options?.defaultPath ? String(options.defaultPath) : undefined,
    filters: Array.isArray(options?.filters) ? options.filters : undefined
  })

  if (result.canceled || !result.filePath) {
    return null
  }

  return result.filePath
})

// Paired reader for the GUI terminal's paste chord: the renderer's
// navigator.clipboard.readText() throws "Document is not focused" whenever a
// portaled overlay has focus, and there's no way to route a read through the
// canvas. The main process has no such gate.
ipcMain.handle('hermes:readClipboard', () => clipboard.readText())

ipcMain.handle('hermes:saveGatewayFile', (_event, payload) => saveGatewayFile(payload))

ipcMain.handle('hermes:saveImageFromUrl', (_event, url) => saveImageFromUrl(String(url || '')))

// The custom context menu's edit verbs. They act on the SENDER's focused
// element, so the renderer restores focus to the editable before invoking.
ipcMain.handle('hermes:context-menu:edit', (event, command) => {
  const contents = event.sender

  if (command === 'copy') {
    contents.copy()
  } else if (command === 'cut') {
    contents.cut()
  } else if (command === 'paste') {
    contents.paste()
  } else if (command === 'selectAll') {
    contents.selectAll()
  }
})

// Copy the image under the sender's LAST context-menu gesture. Chromium only
// exposes image bytes through copyImageAt, and only main saw the coordinates.
ipcMain.handle('hermes:context-menu:copy-image', event => {
  const point = lastContextMenuPoint.get(event.sender.id)

  if (point) {
    event.sender.copyImageAt(point.x, point.y)
  }
})

ipcMain.handle('hermes:context-menu:spellcheck', (event, action) => {
  const kind = action?.kind
  const word = String(action?.word || '')

  if (!word) {
    return
  }

  if (kind === 'replace') {
    event.sender.replaceMisspelling(word)
  } else if (kind === 'add') {
    event.sender.session.addWordToSpellCheckerDictionary(word)
  }
})

// Guest dictionary add: the webview TAG exposes replaceMisspelling but no
// session API, so the renderer names the guest by webContents id.
ipcMain.handle('hermes:context-menu:guest-add-word', (_event, payload) => {
  const word = String(payload?.word || '')
  const guest = electronWebContents.fromId(Number(payload?.webContentsId))

  if (word && guest && !guest.isDestroyed()) {
    guest.session.addWordToSpellCheckerDictionary(word)
  }
})

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

ipcMain.handle('hermes:watchDirectory', (_event, dir) => watchDirectory(String(dir || '')))

ipcMain.handle('hermes:stopPreviewFileWatch', (_event, id) => stopPreviewFileWatch(String(id || '')))

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

ipcMain.on('hermes:titlebar-theme', (_event, payload) => {
  if (!payload || !isHexColor(payload.background) || !isHexColor(payload.foreground)) {
    return
  }

  rendererTitleBarTheme = {
    background: payload.background,
    foreground: payload.foreground
  }

  // Repaint the native (Windows/Linux) titlebar overlay on every open chat
  // window, not just the primary — instance peers and session windows share the
  // one app theme. applyTitleBarOverlay no-ops on the frameless pet overlay.
  for (const win of BrowserWindow.getAllWindows()) {
    applyTitleBarOverlay(win)
  }
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

// See-through window translucency. Persist + re-apply to every open window at
// runtime (no recreation, so caching/sessions are untouched).
//
// The intensity slider is a HOT path: ~100 updates per drag. Two things make
// that cheap. Native work is diffed, so an intensity-only change under glass
// touches nothing (it's painted by the renderer). And the disk write is
// coalesced onto a trailing timer, because writePersistedTranslucency is a
// synchronous writeFileSync and doing one per tick blocks the main process
// mid-drag. Only a cold launch reads that file, so it just has to be correct
// once the hand comes off the slider.
let translucencyWriteTimer = null

function scheduleTranslucencyWrite() {
  if (translucencyWriteTimer) {
    clearTimeout(translucencyWriteTimer)
  }

  translucencyWriteTimer = setTimeout(() => {
    translucencyWriteTimer = null
    writePersistedTranslucency(translucencyState)
  }, 250)
}

// Flush a pending write before the process can exit, so a quit landing inside
// the debounce window doesn't lose the setting.
app.on('before-quit', () => {
  if (translucencyWriteTimer) {
    clearTimeout(translucencyWriteTimer)
    translucencyWriteTimer = null
    writePersistedTranslucency(translucencyState)
  }
})

// Close the pooled keep-alive sockets on quit so lingering connections can't
// hold the event loop open or leak FDs past app teardown.
app.on('will-quit', () => {
  destroyKeepaliveAgents()
})

// Answered synchronously so preload can publish the verdict before the
// renderer's first script — see the note there on why it cannot decide this
// itself. Registered at module scope, which runs long before any window.
ipcMain.on('hermes:translucency:support', event => {
  event.returnValue = { glass: GLASS_SUPPORTED, translucency: TRANSLUCENCY_SUPPORTED }
})

ipcMain.on('hermes:translucency', (_event, payload) => {
  const next = normalizeTranslucency(payload, GLASS_SUPPORTED)
  const previous = translucencyState

  if (
    next.intensity === previous.intensity &&
    next.fade === previous.fade &&
    next.mode === previous.mode &&
    next.material === previous.material &&
    next.scope === previous.scope
  ) {
    return
  }

  translucencyState = next

  // Which native properties actually moved. `scope` is renderer-only (which
  // surfaces thin), so it never appears here.
  const changed = {
    // The backing follows whether glass is ON, not the intensity behind it.
    backing: glassActive(previous) !== glassActive(next),
    material: vibrancyForTranslucency(previous) !== vibrancyForTranslucency(next),
    opacity: windowOpacityFor(previous) !== windowOpacityFor(next)
  }

  scheduleTranslucencyWrite()

  // The HUD's frost reads the same setting but answers on its own terms (see
  // hudFrostFor) — and it is a transparent window, so it is deliberately not
  // in the chat fan-out below. It self-diffs, so an unrelated change costs
  // nothing native.
  hudIpc.applyHudFrost()

  if (changed.backing || changed.material || changed.opacity) {
    for (const win of BrowserWindow.getAllWindows()) {
      applyWindowTranslucency(win, changed)
    }
  }
})

// Keep-awake: hold the machine awake for long/overnight runs. Main owns the one
// blocker and its persisted state so a cold launch restores it (applied on
// ready — powerSaveBlocker needs the app ready). The renderer toggles it from
// Settings → Advanced over IPC. See store/keep-awake.
const KEEP_AWAKE_CONFIG_PATH = path.join(app.getPath('userData'), 'keep-awake.json')
const keepAwake = createKeepAwake(powerSaveBlocker)

function readPersistedKeepAwake() {
  try {
    return JSON.parse(fs.readFileSync(KEEP_AWAKE_CONFIG_PATH, 'utf8')).on === true
  } catch {
    return false
  }
}

ipcMain.on('hermes:keep-awake', (_event, on) => {
  const enabled = Boolean(on)
  keepAwake.set(enabled)

  try {
    fs.mkdirSync(path.dirname(KEEP_AWAKE_CONFIG_PATH), { recursive: true })
    fs.writeFileSync(KEEP_AWAKE_CONFIG_PATH, JSON.stringify({ on: enabled }, null, 2), 'utf8')
  } catch (error) {
    rememberLog(`[keep-awake] write failed: ${error.message}`)
  }
})

// Quick Entry: the renderer reads the live registration state on settings mount
// and writes the preference back. Main is authoritative — it owns the OS
// accelerator — so both handlers return the state that ACTUALLY resulted,
// including `registered: false` + `error: 'taken'` when another app owns the
// chord. See electron/quick-entry.ts + store/quick-entry.
ipcMain.handle('hermes:quick-entry:settings:get', async () => {
  const settings = readQuickEntrySettings()
  const state = quickEntryShortcut.current()

  // Ground truth is what the last apply produced; the shortcut we report is the
  // live one (a saved-but-rejected chord still shows what the user asked for).
  return {
    enabled: settings.enabled,
    error: state.error,
    registered: state.registered,
    shortcut: settings.enabled ? state.shortcut : settings.shortcut
  }
})

ipcMain.handle('hermes:quick-entry:settings:set', async (_event, patch) => {
  const current = readQuickEntrySettings()

  const next = sanitizeQuickEntrySettings({
    enabled: patch?.enabled === undefined ? current.enabled : patch.enabled === true,
    shortcut: typeof patch?.shortcut === 'string' && patch.shortcut.trim() ? patch.shortcut : current.shortcut
  })

  writeQuickEntrySettings(next)

  return applyQuickEntrySettings(next)
})

// Quick window → main → PRIMARY renderer. We never submit here: the renderer
// owns the one prompt-submit path, and forwarding keeps it that way. The
// payload is `{ target, text }` — target routing (current chat / a picked
// session / new) is the renderer's job too.
ipcMain.on('hermes:quick-entry:submit', (_event, payload) => {
  hideQuickEntryWindow()

  const text = typeof payload?.text === 'string' ? payload.text.trim() : ''

  if (!text) {
    return
  }

  if (!mainWindow || mainWindow.isDestroyed()) {
    rememberLog('[quick-entry] dropped a submit: no primary window to route it to')

    return
  }

  // Deliberately does NOT raise/focus the main window — the user asked to fire
  // a prompt from wherever they were, not to be yanked into the app.
  mainWindow.webContents.send('hermes:quick-entry:submit', {
    target: typeof payload?.target === 'string' && payload.target ? payload.target : 'current',
    text
  })
})

// Primary renderer → main → quick window: gateway connection state + the
// recent-session list for the target picker. Cached so a quick window spawned
// AFTER the last push still boots from truth instead of "disconnected".
ipcMain.on('hermes:quick-entry:state', (_event, payload) => {
  quickEntryLastState = payload ?? null

  if (quickEntryWindow && !quickEntryWindow.isDestroyed()) {
    quickEntryWindow.webContents.send('hermes:quick-entry:state', payload)
  }
})

ipcMain.on('hermes:quick-entry:dismiss', () => hideQuickEntryWindow())

// Disable F12 DevTools: maintained in the main process so a cold launch
// restores it before any window is shown (applied on ready). The renderer
// toggles it from Settings → Advanced over IPC. See store/disable-f12.
const DISABLE_F12_CONFIG_PATH = path.join(app.getPath('userData'), 'disable-f12.json')

function readPersistedDisableF12() {
  try {
    return JSON.parse(fs.readFileSync(DISABLE_F12_CONFIG_PATH, 'utf8')).on === true
  } catch {
    return false
  }
}

ipcMain.on('hermes:devtools:disable-f12', (_event, on) => {
  f12Blocked = Boolean(on)

  try {
    fs.mkdirSync(path.dirname(DISABLE_F12_CONFIG_PATH), { recursive: true })
    fs.writeFileSync(DISABLE_F12_CONFIG_PATH, JSON.stringify({ on: f12Blocked }, null, 2), 'utf8')
  } catch (error) {
    rememberLog(`[disable-f12] write failed: ${error.message}`)
  }
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

// Renderer-bundle skew: `hermes update` moves the SOURCE TREE, but the UI
// (including bundled plugins like Bot Mode) is compiled into this binary at
// build time. A terminal-side update — or an in-app update whose bundle-swap
// leg failed — leaves the new runtime running under an old renderer, so About
// shows the new version while the sidebar is missing that version's desktop
// features. Compare the build stamp's commit against the tree, scoped to
// apps/desktop/, and warn when the running renderer is provably behind.
// Fail-quiet: dev runs (no stamp), non-git builds, and shallow-clone gaps all
// report in-sync rather than risk a false "your install is torn" warning.
async function detectRendererSkew() {
  return detectBundleSkew(INSTALL_STAMP, runGit, resolveUpdateRoot())
}

// Re-resolve the live Hermes version and push it into the native About panel
// just before showing it, so an in-place `hermes update` is reflected without
// an app restart. macOS only — `showAboutPanel()` is a no-op elsewhere, and the
// other platforms don't use this menu item.
function showAboutPanelFresh() {
  void detectRendererSkew().then(skew => {
    app.setAboutPanelOptions({
      applicationName: APP_NAME,
      applicationVersion: skew.outOfSync
        ? `${resolveHermesVersion()} — app build out of date, update the desktop app`
        : resolveHermesVersion(),
      copyright: 'Copyright © 2026 Nous Research'
    })
    app.showAboutPanel()
  })
}

ipcMain.handle('hermes:version', async () => {
  const skew = await detectRendererSkew()

  return {
    appVersion: resolveHermesVersion(),
    electronVersion: process.versions.electron,
    nodeVersion: process.versions.node,
    platform: process.platform,
    hermesRoot: resolveUpdateRoot(),
    bundleOutOfSync: skew.outOfSync,
    bundleCommitsBehind: skew.desktopCommitsBehind
  }
})

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
// hermes:// deep links (e.g. hermes://blueprint/morning-brief?time=08:00,
// hermes://mcp/install?name=NAME&config=B64 — the vendor "Add to Hermes"
// button, or hermes://plugin/install?repo=owner/repo). Dev
// (`HERMES_DESKTOP_DEV_SERVER`) registers hermes-dev:// instead — bare
// Electron or a stale OS handler often owns hermes:// on dev machines.
// Parsing is generic ({kind, name, params}); the renderer routes per kind
// and anything install-shaped requires explicit user confirmation there.
// A docs/dashboard "Send to App" button opens this URL; we route it into the
// running app. Three delivery paths: macOS 'open-url',
// Win/Linux running-app 'second-instance' (argv), Win/Linux cold-start argv.
// ---------------------------------------------------------------------------
const HERMES_PROTOCOL = DEV_SERVER ? 'hermes-dev' : 'hermes'
/** Schemes accepted when parsing inbound URLs (dev accepts both). */
const DEEPLINK_SCHEMES = DEV_SERVER ? ['hermes-dev', 'hermes'] : ['hermes']
let _pendingDeepLink = null
let _rendererReadyForDeepLink = false

function _extractDeepLink(argv) {
  if (!Array.isArray(argv)) {
    return null
  }

  return argv.find(a => typeof a === 'string' && DEEPLINK_SCHEMES.some(s => a.startsWith(`${s}://`))) || null
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

  const scheme = parsed.protocol.replace(/:$/, '')

  if (!DEEPLINK_SCHEMES.includes(scheme)) {
    rememberLog(`[deeplink] ignoring scheme ${scheme} (expected ${DEEPLINK_SCHEMES.join(' or ')})`)

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
      // relaunch us with the URL. argv[1] is usually "." when launched via
      // `electron .` from apps/desktop — resolve against cwd.
      const entry = path.resolve(process.argv[1])
      app.setAsDefaultProtocolClient(HERMES_PROTOCOL, process.execPath, [entry])
    } else {
      app.setAsDefaultProtocolClient(HERMES_PROTOCOL)
    }

    rememberLog(`[deeplink] registered ${HERMES_PROTOCOL}:// handler`)
  } catch (err) {
    rememberLog(`[deeplink] protocol registration failed: ${err.message}`)
  }
}

// Single-instance lock: deep links on a running app (Win/Linux) arrive as a
// second-instance argv. Without the lock a second `hermes://` launch spawns a
// whole new app instead of routing into the running one.
const _gotSingleInstanceLock = app.requestSingleInstanceLock()
const isPrimaryInstance = _gotSingleInstanceLock

if (!isPrimaryInstance) {
  // Hard-exit, not app.quit(): the before-quit teardown coordinator defers a
  // plain quit (event.preventDefault + async backend shutdown), and in that
  // window `ready` still fires — the lock-losing instance then runs the full
  // startup (shortcut registration, createWindow → startHermes), whose
  // reapOrphans() SIGTERMs the running instance's live backend (#87295).
  // app.exit() terminates immediately, before `ready`, so a second launch
  // routes into the running window and never touches backend machinery.
  app.exit(0)
} else {
  app.on('second-instance', (_event, argv) => {
    const url = _extractDeepLink(argv)

    if (url) {
      handleDeepLink(url)
    }

    ensureMainWindow(mainWindow, {
      isReady: app.isReady(),
      createWindow,
      focusWindow,
      // deep-link delivery focuses a live window after its renderer is ready.
      focusExisting: !url
    })
  })
}

// macOS delivers deep links via 'open-url' — register early (can fire before
// whenReady; handleDeepLink queues until the renderer is ready).
app.on('open-url', (event, url) => {
  event.preventDefault()
  handleDeepLink(url)
})

app.whenReady().then(() => {
  // Warm the login-shell PATH resolution immediately so it usually completes
  // before the backend start path awaits the same single-flight promise.
  void ensureLoginShellPath()

  const systemCa = installWindowsSystemCaTrust(tls)

  if (systemCa.applied) {
    rememberLog(
      `[tls] trusting ${systemCa.systemCertificateCount} Windows system CA certificate(s) for backend connections`
    )
  } else if (systemCa.error) {
    rememberLog(`[tls] could not load Windows system CA certificates: ${systemCa.error}`)
  }

  // Keyring-less Linux `--password-store=basic` support. This must run before
  // createWindow() and anything that could touch safeStorage; the narrow
  // platform/switch/guard semantics live in the extracted helper.
  enableBasicPasswordStoreEncryption({
    platform: process.platform,
    passwordStoreSwitch: app.commandLine.getSwitchValue('password-store'),
    safeStorageApi: safeStorage
  })

  // Keychain encryption is opt-in (default OFF). One-shot: rewrite any
  // legacy safeStorage-encrypted secrets as plain so no later launch ever
  // touches the OS keychain unless the user turns encryption on in
  // Settings → Gateway. Must run before createWindow() and the first
  // connection resolution.
  migrateLegacyEncryptedSecretsOnce()

  if (IS_MAC) {
    Menu.setApplicationMenu(buildApplicationMenu())
  } else {
    Menu.setApplicationMenu(null)
  }

  installMediaPermissions()
  installDownloadHandling()
  registerMediaProtocol()
  installEmbedReferer()
  installRemoteHeaderRules()
  registerDeepLinkProtocol()

  ensureWslWindowsFonts()
  configureSpellChecker()
  registerPowerResumeListeners()
  keepAwake.set(readPersistedKeepAwake())
  f12Blocked = readPersistedDisableF12()
  // Seed this before the first window exists: a picker can open before
  // startHermes() finishes resolving the configured backend.
  const primaryProfile = primaryProfileKey()

  setActiveGatewayProfile(primaryProfile)
  setWslBridgeProfileState(primaryProfile, !primaryBackendIsRemote())
  // Quick Entry's global chord — registered on ready so a cold launch restores
  // it without the renderer visiting Settings. A failed registration is logged
  // here and surfaced in Settings via the IPC state (never silent).
  applyQuickEntrySettings(readQuickEntrySettings())

  if (IS_MAC) {
    const reposition = () => wakeIndicatorController.reposition()

    screen.on('display-added', reposition)

    screen.on('display-metrics-changed', reposition)

    screen.on('display-removed', reposition)
  }

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

// Ask before a quit kills a turn in flight. True when the quit was intercepted
// and the confirmation is on screen; "Quit Anyway" re-enters before-quit with
// the latch set and falls straight through to the teardown below.
function heldQuitForActiveWork(event: Electron.Event): boolean {
  if (SKIP_QUIT_CONFIRM || quitConfirmedWithActiveWork || quitPromptOpen) {
    return false
  }

  const prompt = quitPromptFor(mergeActiveWork(activeWorkByWebContents.values()), isQuittingForHandoff)
  const parent = BrowserWindow.getFocusedWindow() ?? BrowserWindow.getAllWindows()[0]

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

app.on('before-quit', event => {
  // Runs ahead of every teardown below, so "Keep Running" leaves the app
  // exactly as it was.
  if (heldQuitForActiveWork(event)) {
    return
  }

  if (!backendQuitTeardownDone) {
    event.preventDefault()
    void backendShutdown.run().finally(() => {
      backendQuitTeardownDone = true
      app.quit()
    })
  }

  if ((sshConnections.size > 0 || sshBootstrapCoordinator.promises().length > 0) && !sshQuitTeardownDone) {
    event.preventDefault()
    sshBootstrapCoordinator.cancelAll()
    const scopes = [...sshConnections.keys()]

    const pending = Promise.allSettled([
      ...scopes.map(scope => teardownSshConnection(scope || null)),
      ...sshBootstrapCoordinator.promises()
    ])

    void Promise.race([pending, new Promise(resolve => setTimeout(resolve, 4_000))]).then(async () => {
      await sshBootstrapCoordinator.forceCleanupAll()
      sshQuitTeardownDone = true
      app.quit()
    })
  }

  // Clean quit mid-boot should not trip next-launch --no-sandbox (#38216).
  // FATAL GPU aborts skip before-quit, leaving the `booting` marker in place.
  // Keyed on sticky (not active): a manual --no-sandbox run still records a
  // clean quit, while an engaged fallback keeps its sticky marker.
  if (IS_WINDOWS && !windowsSandboxFallbackSticky) {
    try {
      writeSandboxMarker(app.getPath('userData'), markerAfterSuccessfulBoot({ fallbackActive: false }))
    } catch {
      void 0
    }
  }

  // The always-on-top overlay isn't a "real" app window; close it so a stray
  // pet can't keep the process alive or float over a quit app.
  closePetOverlay()
  wakeIndicatorController.close()

  // Same for the HUD — an always-on-top panel outliving the app would leave a
  // floating composer with nothing behind it. Close it directly rather than via
  // closeHudWindow(): that also re-shows the main window, which is wrong on the
  // way out (and `hudRestoreMainWindow` may still be armed from entering HUD).
  hudSnapShortcut.dispose()

  if (hudWindow && !hudWindow.isDestroyed()) {
    hudWindow.removeAllListeners('closed')
    hudWindow.destroy()
  }

  hudWindow = null

  // Same for the Quick Entry composer — and release its global accelerator so a
  // quitting Hermes never keeps another app's chord hostage.
  closeQuickEntryWindow()

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
  terminalIpc.disposeAllTerminalSessions()

  void backendShutdown.run()
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
