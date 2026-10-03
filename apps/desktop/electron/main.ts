Warning: truncated output (original token count: 188688)
Total output lines: 19870

import { type ChildProcess, execFileSync, spawn } from 'node:child_process'
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
  crashReporter,
  dialog,
  net as electronNet,
  webContents as electronWebContents,
  globalShortcut,
  ipcMain,
  type IpcMainEvent,
  type IpcMainInvokeEvent,
  Menu,
  type MenuItemConstructorOptions,
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
import type { Session } from 'electron'

import { type ActiveRuntimeState, classifyActiveRuntime } from './active-runtime-state'
import { HERMES_API_EXPECTED_404 } from './api-expected-404'
import {
  destroyKeepaliveAgents,
  htmlResponseError,
  httpStatusError,
  jsonAgentFor,
  readJsonErrorBody,
  readStatusCode,
  withRetry
} from './api-transport'
import { appIconCandidates, resolveAppIcon, shouldOverrideDockIcon } from './app-icon'
import { stageAppInstallerFile } from './app-installer-file'
import {
  appVersionInfo,
  type AppVersionInfo,
  assertSourceUpdateChannel,
  nativeAboutVersion,
  packagedReleaseChannel
} from './app-version'
import { runAppInstallerChecker } from './appinstaller-checker'
import { installApplicationMenuAfterFirstWindow } from './application-menu-startup'
import { stopBackendChild as stopBackendChildImpl, waitForBackendExit } from './backend-child'
import {
  type BackendOutputTail,
  claimDecision,
  createBackendOutputTail,
  execText,
  formatBackendExitLine,
  isPidOnlyStartMarker,
  pidOnlyStartMarker,
  probeStartMarker,
  processStartMarker,
  REAP_PROBE_TIMEOUT_MS
} from './backend-claim'
import { dashboardFallbackArgs, serveBackendArgs } from './backend-command'
import { createBackendConnectionState } from './backend-connection-state'
import { BackendDialClaims } from './backend-dial-claim'
import type { HostBackendRecord } from './backend-discovery'
import { buildDesktopBackendEnv, profileBackendParentEnv } from './backend-env'
import { createBackendExitRecoveryLatch } from './backend-exit-recovery'
import { isReauthRequiredError, waitForHermesReady } from './backend-health'
import {
  backendCommandMatches,
  type BackendOwnershipEntry,
  createBackendOwnership,
  createBackendShutdownCoordinator
} from './backend-ownership'
import { canImportHermesCli, PROBE_TIMEOUT_MS, shouldTrustHermesOverride, verifyHermesCli } from './backend-probes'
import { waitForDashboardPortAnnouncement } from './backend-ready'
import { recycleOwnedBackend } from './backend-recycle'
import { isPidAliveWindows, waitForBackendRelease } from './backend-release-gate'
import { createInstalledRuntimeGate } from './backend-resolution'
import { createBackendServeSupportResolver } from './backend-serve-support'
import {
  isHostKeyChangedBootFailure,
  isRetryableRemoteBootFailure,
  isSshAuthFailedBootFailure,
  isSshClientFailedBootFailure,
  shouldHoldBootProgressForReauth,
  shouldLatchBackendStartFailure,
  shouldLatchHostKeyChangedFailure,
  shouldLatchRemoteReauthFailure,
  shouldLatchSshAuthFailure,
  shouldLatchSshClientFailure,
  sshClientFailedError
} from './backend-start-failure'
import { describeBootstrapFailure } from './bootstrap-failure-copy'
import {
  detectRemoteDisplay,
  isWindowsBinaryPathInWsl,
  isWslEnvironment,
  resolveLinuxPasswordStore
} from './bootstrap-platform'
import { decideBootstrapRepair } from './bootstrap-repair-guard'
import { runBootstrap } from './bootstrap-runner'
import { bootstrapSnapshot } from './bootstrap-state'
import {
  BROWSER_WINDOW_HEIGHT,
  BROWSER_WINDOW_MIN_HEIGHT,
  BROWSER_WINDOW_MIN_WIDTH,
  BROWSER_WINDOW_WIDTH,
  buildBrowserWindowUrl
} from './browser-windows'
import { createBundleSkewChecker } from './bundle-skew'
import { detectBundleSwap, readBundleSwapStamp } from './bundle-swap'
import { registerChatOnboardingWindow } from './chat-onboarding-window'
import { provisionCliLinks } from './cli-provision'
import { closeStopFailureMessage, finishWindowsCloseStop, type RuntimeLock } from './close-stop-kill'
import { shouldAttemptCloudBootCascade } from './cloud-boot-cascade'
import { discoverWithTeamFallback } from './cloud-discovery'
import { createCloudSessionRecovery } from './cloud-session-recovery'
import { installCommandScreenshot } from './command-screenshot'
import { composerImageTimestamp } from './composer-image-name'
import { writeComposerPaste } from './composer-paste'
import { applyConnectionChange, teardownSshState } from './connection-apply'
import {
  connectionInstallIds,
  evictConnectionCaches,
  rosterSourceErrors,
  sshInventoryAttemptedAt,
  sshRosterCache
} from './connection-caches'
import {
  apiRequestRegistryConnectionId,
  authModeFromStatus,
  buildGatewayWsUrl,
  buildGatewayWsUrlWithTicket,
  connectionScopeKey,
  cookiesHaveLiveSession,
  cookiesHaveSession,
  gatewayWsUrlIpcResult,
  hostLabelFromBaseUrl,
  isGatewayAuthRejection,
  localProfileEntry,
  modeIsRemoteLike,
  normalizeRemoteBaseUrl,
  normalizeRemoteHeaders,
  normalizeRemoteProfileName,
  normalizeSshConfig,
  normAuthMode,
  pathForRegistryBackendRequest,
  pathWithGlobalRemoteProfile,
  profileHasRemoteConnection,
  profileRemoteOverride,
  type ProfileRouteOptions,
  profileSshOverride,
  type RegistryBackendRequestScope,
  resolveAuthMode,
  resolveProfileApiRequest,
  resolveProfileBackendRoute,
  resolveRemoteSshDashboardProfile,
  resolveTestWsUrl,
  sanitizeRemoteHeaderValue,
  savedProfileSsh,
  tokenPreview,
  unscopableMutatingRequest,
  withTransientRetries
} from './connection-config'
import { applyConnectionConfigAtomically } from './connection-config-apply'
import {
  backendScopeKey,
  backendScopePrefix,
  buildAgentRoster,
  connectionDialFieldsChanged,
  connectionIdForPendingLogin,
  mergeConnectionInput,
  migrateV1ToRegistry,
  normalizeConnectionInput,
  normalizeRegistry,
  parseBackendScopeKey,
  reconcileAppliedGlobalConnection,
  reconcileRegistryDrift,
  registryDialConnectionId,
  rememberSshEnumeration,
  removeConnection,
  type ResolvedConnectionDescriptor,
  resolvedConnectionId,
  resolveRegistryLocalRoute,
  reuseMatchingPrimaryRemoteBackend,
  reuseMatchingPrimarySshBackend,
  setConnectionLaunchMode,
  setLastUsedConnection,
  setPrimaryConnection,
  type SharedRegistryProfileScope,
  shouldDeferLocalEnumeration,
  shouldRetrySshInventory,
  updateEligibility,
  upsertConnection
} from './connection-registry'
import type { RegistryConnection } from './connection-registry'
import type { RosterProfileMetadata } from './connection-registry'
import { liveWindowState, overlayWindowState } from './connection-window-state'
import { describeCrashReason, installCrashForensics } from './crash-forensics'
import {
  adoptServedDashboardToken,
  isAttachedBackendTokenDrifted,
  resolveServedDashboardToken
} from './dashboard-token'
import { resolveDashboardWebDist } from './dashboard-web-dist'
import { resolveDesktopHermesHome, resolveDesktopUserData } from './data-paths'
import { loadOrCreateInstallationId, sshOwnershipId } from './desktop-installation'
import { formatDesktopLogLine, formatLogStamp } from './desktop-log-line'
import {
  createDesktopProfilePreferences,
  DESKTOP_PROFILE_NAME_RE,
  type DesktopProfileRoute,
  resolveDesktopConnectionRequest,
  resolveDesktopWindowLaunch
} from './desktop-profile'
import { registryPrimaryBootRoute, resolveDesktopRemoteRoute, v1SshTerminalPoolKey } from './desktop-remote-route'
import { type DesktopSharedMetrics, registerDesktopSharedMetrics } from './desktop-shared-metrics'
import {
  buildPosixCleanupScript,
  buildWindowsCleanupScript,
  type DesktopUninstallResult,
  modeRemovesAgent,
  modeRemovesUserData,
  registerDesktopUninstallIpc,
  resolveRemovableAppPath,
  shouldRemoveAppBundle,
  uninstallArgsForMode,
  type UninstallSummaryDetails
} from './desktop-uninstall'
import { describeDevCdpDecision, resolveDevCdpPort } from './dev-cdp'
import { preReadyDockLaunchSteps } from './dock-launch-order'
import { embedHostOrigin } from './embed-host'
import { installEmbedReferer } from './embed-referer'
import { createAmbientClaimArbiter } from './event-dedupe'
import { openExternalUrl as externalOpen, type ExternalOpenDeps, reportPreOpenStatFailure } from './external-open'
import {
  buildTerminalScript,
  resolveTerminalLaunch,
  terminalScriptEnv,
  terminalScriptExtension,
  tuiResumeArgs
} from './external-terminal'
import { f12ShortcutDecision, toF12KeyboardEventPayload } from './f12-shortcut'
import { resolveFeatureFlags } from './feature-flags'
import {
  installFindShortcut,
  installFoundInPageForwarder,
  performFindAfterIndexingStarted,
  stopFind
} from './find-in-page'
import { createFirstRunSetupGate } from './first-run-setup-gate'
import { registerFsIpc } from './fs-ipc'
import type {
  GatewayFileSaveContext,
  GatewayFileSaveDeps,
  GatewayFileSaveResult,
  GatewaySaveDialogOptions,
  GatewaySaveDialogResult
} from './gateway-file-download'
import {
  gatewayFilePath,
  gatewayFileRequestPaths,
  resolveGatewayFileBackend,
  saveGatewayDownload
} from './gateway-file-download'
import { downloadViaOauthSessionToFile, downloadViaTokenToFile } from './gateway-file-download-transport'
import { stopGatewayBeforeUpdate } from './gateway-stop-before-update'
import { probeGatewayWebSocket, spawnedBackendProbeOptions } from './gateway-ws-probe'
import { windowsGitCandidates } from './git-binary-candidates'
import { registerGitIpc } from './git-ipc'
import { desktopBackendSpawnEnv, guestOnboardingEnabled } from './guest-onboarding'
import { readAndConsumeHandoffResult } from './handoff-result'
import {
  assertExistingPathForOpen,
  ATTACHMENT_UPLOAD_DEFAULT_MAX_BYTES,
  clampDataUrlReadMaxMb,
  DATA_URL_READ_DEFAULT_MAX_MB,
  dataUrlReadMaxBytesFromMb,
  DEFAULT_FETCH_TIMEOUT_MS,
  enableBasicPasswordStoreEncryption,
  encryptDesktopSecret as encryptDesktopSecretStrict,
  homeRelativeAttachmentCandidates,
  isMissingFileError,
  missingFileResult,
  readFileDataUrlForIpc,
  resolvePersistedRemoteToken,
  resolveReadableFileForIpc,
  resolveRemoteTokenPlainText,
  resolveRequestedPathForIpc,
  resolveTimeoutMs,
  SAFE_STORAGE_ENCODING,
  TEXT_PREVIEW_SOURCE_MAX_BYTES,
  tightenSecretFileMode,
  writeSecretFileAtomic
} from './hardening'
import {
  type AttachedBackend,
  attachOrReserveSpawn,
  HOST_SPAWN_GATE_STALE_MS,
  spawnLedgerPath,
  type SpawnReservation
} from './host-backend-attach'
import { assertNoSecondLocalBackend, assertNotPassiveSpawn } from './host-backend-singleton'
import { lookupPublishedSessionToken } from './host-published-token'
import { claimHostSpawnGate } from './host-spawn-gate'
import { HERMES_HUB_FALLBACK_ORIGIN, HERMES_HUB_ORIGIN, isHermesHubClipboardWrite } from './hub-iframe-policy'
import { requestHudClose } from './hud-close'
import { cursorPointInWindow } from './hud-cursor'
import { startHudGameOverlayWatch } from './hud-game-overlay'
import { applyHudResetBounds, defaultHudBounds } from './hud-geometry'
import { registerHudIpc } from './hud-ipc'
import { installHudModifierTap } from './hud-modifier'
import { applyHudElectronOverlay, promoteHudOverlay } from './hud-overlay'
import { snapHudBounds } from './hud-snap'
import { createHudSnapShortcut } from './hud-snap-shortcut'
import { buildHudWindowUrl } from './hud-url'
import { linuxOzoneBackend, resolveHudWindowing } from './hud-windowing'
import { INSTALL_STAMP, installShape } from './install-stamp'
import type { InstallStamp } from './install-stamp'
import { resolveLocalRuntimeVersion } from './local-runtime-version'
import { applyLaunchProfileOverride } from './launch-profile'
import { fetchLinkTitle, resolveFaviconCached } from './link-metadata'
import { CHROMIUM_LOG_FILENAME, enableLinuxCrashDiagnostics, linuxCrashDiagnostics } from './linux-crash-diagnostics'
import {
  decideLinuxGpuLaunch,
  disableGpuSwitchNeededForReason,
  LINUX_GPU_SILENT_RETRY_GRACE_S,
  linuxGpuChildDeathPath,
  linuxGpuFallbackMarker,
  linuxGpuMarkerAfterSuccessfulBoot,
  readLinuxGpuMarker,
  shouldEngageSilentGpuRetryFallback,
  writeLinuxGpuMarker
} from './linux-gpu-fallback'
import { notifyLauncherWindowRevealed } from './linux-launcher-ready'
import {
  decideNvidiaEglFallback,
  nvidiaEglFallbackMarker,
  nvidiaEglMarkerAfterSuccessfulBoot,
  parseNvidiaDriverMajor,
  parseNvidiaDriverVersion,
  readNvidiaEglMarker,
  shouldRelaunchForNvidiaGpuDeath,
  writeNvidiaEglMarker
} from './linux-nvidia-egl-fallback'
import { createLocalBackendLifecycle, waitForTeardown } from './local-backend-lifecycle'
import { resolveIpcFileReadPath, resolveMediaStreamFile, resolvePreviewTargetPath } from './local-read-path'
import { localSkinProfileKey, readLocalSkinPayload } from './local-skin'
import { ACTIVE_LOG_POLL_MS, planLogRotation, reclaimActiveLogIfOversized } from './log-rotation'
import { registerMachineProfile } from './machine-profile'
import { createMainProcessLagWatchdog } from './main-process-lag-watchdog'
import { activateWindow, ensureMainWindow, shouldQuitOnLastChatClosed } from './main-window-lifecycle'
import {
  assertManagedUpdatePreflightClear,
  executeManagedRemoteUpdate,
  fenceManagedSshBootstrapPublication,
  ManagedConnectionUpdateGate,
  managedSshDrainBlocker,
  managedSshRecoveryScopes,
  managedSshScopeRole,
  managedSshTokenPersistencePlan,
  managedSshUpdateAllRow,
  MAX_MANAGED_SSH_RECOVERY_ATTEMPTS,
  recoverManagedSshScopes,
  refusedManagedSshUpdate,
  type RemoteUpdateTarget,
  runManagedSshUpdate,
  validateCorrelationId,
  waitForManagedRemoteClearance,
  waitForManagedSshBootstrapFence,
  waitForManagedUpdateOperations
} from './managed-ssh-update'
import { registerMcpOauthCallbackIpc } from './mcp-oauth-callback-ipc'
import { isMediaCapturePermission } from './media-capture-permission'
import { createMediaProtocolHandler, MEDIA_PROTOCOL } from './media-protocol'
import { fetchLocalMedia } from './media-range'
import { createMinimizeToTray } from './minimize-to-tray'
import {
  createNativeAccessTokenCoordinator,
  type NativeAccessTokenOptions,
  NativeAuthChangedError
} from './native-access-token'
import { oauthSessionIsLive, resolveJsonBody, resolveReadinessProbeAuth } from './native-auth-decisions'
import {
  nativeRefreshUrl,
  type NativeTokenSet,
  parseTokenResponse,
  resolveLoginStrategy,
  tokenNeedsRefresh
} from './native-oauth'
import { runNativeLogin } from './native-oauth-login'
import { loadNativeTokenSet, type NativeTokenStoreIo, persistNativeTokenSet } from './native-token-store'
import { execGit, killTimedGitChildren, setNoConsoleGitRoots } from './no-console-git'
import { registerNativeNotifications } from './notification-ipc'
import { isExpectedOauthNavigationAbort } from './oauth-navigation'
import { serializeJsonBody, setJsonRequestHeaders } from './oauth-net-request'
import { LEGACY_OAUTH_PARTITION, resolveOauthPartition } from './oauth-partition'
import {
  canShowInteractiveOauthLogin,
  mintGatewayWsTicket as mintOauthGatewayWsTicket,
  requestWithOauthFallback,
  retryCookie401WithLogin,
  withoutInteractiveOauthLogin
} from './oauth-rest-request'
import { wireOauthSessionResponse } from './oauth-session-response'
import { listWindowsProcesses, reapPackageRootedProcesses } from './package-process-reap'
import { createParentStartMarkerResolver, parentWatchdogEnv } from './parent-process-identity'
import { bundledPayload, installIdForRoot, type PayloadInfo, payloadPythonPath } from './payload-backend'
import { petOverlayClickThrough, shouldPopInOnOverlayClosed } from './pet-overlay'
import { placePetOverlay, registerPetOverlayIpc } from './pet-overlay-ipc'
import {
  buildRegistryProfileRoutes,
  isLocalEnumerationFailure,
  localRouteFallbackProfiles,
  undialedSshRouteSeeds
} from './plugin-profile-routes'
import { clampPoolLimits, parsePoolLimits, POOL_LIMITS_DEFAULTS, POOL_LIMITS_MIN } from './pool-limits'
import { createPoolRetirer } from './pool-retire'
import { createPoolRetirementClient } from './pool-retire-http'
import {
  assertPoolEntryStillOwned,
  BackgroundSlotRetryBackoff,
  BackgroundSlotRetryDeferredError,
  isBackgroundSlotRetryDeferred,
  isBackgroundSlotWaitTimeout,
  LocalBackendSpawnCoordinator,
  type LocalBackendSpawnPriority,
  registerLocalBackendExitFinalizer,
  releaseLocalBackendSlot,
  releaseLocalBackendSlotAfterExit
} from './pool-spawn-coordinator'
import { createPoolStopper } from './pool-stop'
import { poolTouchKeys } from './pool-touch-scope'
import { createPortalSession } from './portal-session'
import {
  createKeepAwake,
  type KeepAwakeMode,
  keepAwakeWanted,
  parseKeepAwakeMode,
  readKeepAwakeMode
} from './power-save'
import { readPreUpdateBackupEnabled } from './pre-update-backup-config'
import { capturePreviewContents } from './preview-capture'
import { onPreviewWatchOwnerDestroyed, sendPreviewFileChangedToOwner } from './preview-file-watch'
import { hasClosePreviewFlag, previewGuestInputAction } from './preview-guest-escape'
import { commandFocusedGuest, notePreviewGuestHidden } from './preview-guest-offscreen'
import { PreviewReachRegistry } from './preview-reach'
import { previewHttpUrlTarget } from './preview-url-target'
import {
  createPrimaryRemoteConnection,
  FirstRunSetupResetError,
  runPrimaryBackendStartup
} from './primary-backend-startup'
import { rehomePrimaryConnection } from './primary-connection-rehome'
import { PrimaryProfilePin, resolveLaunchProfile } from './primary-profile-pin'
import { applyDesktopIdentity, PRODUCT_IDENTITY } from './product-identity'
import {
  assertLocalProfileCanStart,
  decideProfileDeleteAction,
  dispatchConnectionScopedProfileDelete,
  localProfilePoolKeys,
  ProfileDeletionGate,
  profileNameFromDeleteRequest,
  resolveRouteProfile
} from './profile-delete-routing'
import { migrateActiveProfileIfMissing as migrateActiveProfileIfMissingPure } from './profile-migration'
import { prepareProfileRenameLifecycle, profileRenameFromRequest } from './profile-rename-routing'
import {
  assembleSidebarSessionSlices,
  buildSidebarSessionSliceParams,
  fetchPrimaryProfileSessions,
  fetchRegistrySessionRows,
  fetchRemoteProfileSessions,
  findRemoteOwnerProfileForSession,
  hasPinnedRegistrySessionSource,
  isAllProfilesSessionListRequest,
  mergeProfileSessionWindow,
  pathWithRemoteOwnerScope,
  type RegistrySessionSource,
  remoteProfileQueryScope,
  shouldIncludeLocalRegistrySessionSource,
  spliceRegistrySessionRows,
  tagRegistrySessionResponse,
  tagRemoteSessionRows
} from './profile-session-routing'
import {
  createQuickEntryShortcut,
  createQuickEntrySubmitRelay,
  quickEntryWindowBounds,
  sanitizeQuickEntrySettings
} from './quick-entry'
import { createQuitFinalization } from './quit-finalization'
import {
  type ActiveWork,
  backendOwnedByApp,
  mergeActiveWork,
  normalizeActiveWork,
  quitPromptFor,
  shouldGuardWindowClose
} from './quit-guard'
import {
  backendQuitNeedsWait,
  backendTeardownOptions,
  createQuitTeardownCoordinator,
  type QuitTeardownTask
} from './quit-teardown'
import * as remoteLifecycle from './remote-lifecycle'
import {
  attachPowerResumeRemoteRevalidation,
  ensureHealthyPooledRemoteBackendForDispatch,
  REMOTE_POOLED_LIVENESS_FAILURE_WINDOW_MS,
  RemoteLivenessTracker,
  RemoteRevalidationCoordinator,
  revalidatePooledRemoteBackends,
  revalidateRemoteConnection,
  revalidateSuspectPooledRemoteBackends
} from './remote-liveness'
import { resolveRemoteOauthTicket, rosterSourceEnumerationTimeoutMs } from './remote-oauth-ticket'
import { createRemoteOwnerCache } from './remote-owner-cache'
import { remoteSessionCookies } from './remote-session-cookies'
import {
  attachRemoteRequestHeaderListener,
  collectRemoteHeaderSources,
  createRegistryGatewayWsUrlHandler,
  createRemoteWsHeaderStore,
  oauthLoginLoadUrlOptions,
  resolveRemoteRequestHeaders
} from './remote-ws-headers'
import { enableRendererAccessibility } from './renderer-accessibility'
import { missingRendererAssets, presentRendererIndexes } from './renderer-bundle'
import { planLaunchSwitches, readDesktopLaunchConfig } from './renderer-heap-flags'
import { loadRendererLoadErrorPage } from './renderer-load-error-page'
import { attachRendererConsoleCapture, formatRendererBoundaryReport } from './renderer-log'
import { fetchRosterSourceData } from './roster-source-fetch'
import { rosterSourceStatus } from './roster-source-status'
import {
  classifyStoredSecret,
  readSecretStoragePolicy,
  SECRET_STORAGE_POLICY_FILE,
  type SecretStoragePolicy,
  writeSecretStoragePolicy
} from './secret-storage-policy'
import { selectPathsDialogProperties } from './select-paths-dialog'
import { selectRunnableBinary } from './select-runnable-binary'
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
import { removeStaleSingletonLock } from './singleton-lock'
import { createSourcePythonBackend, resolveSourceInstallationBackend, type SourceBackend } from './source-backend'
import { resolveSourcePython } from './source-python'
import { resolveSshBinary } from './ssh-binary'
import { createBootstrapCoordinator, sshConfigFingerprint } from './ssh-bootstrap-coordinator'
import { collectSshConfigHosts, parseSshGOutput } from './ssh-config'
import { createSshProbeConnection, pickLocalPort, redactSecrets, SshConnection } from './ssh-connection'
import { createSshIsolatedKeepaliveRegistry } from './ssh-isolated-keepalive'
import { createSshTeardownTracker } from './ssh-teardown'
import { createStreamThrottle } from './stream-throttle'
import { installSystemCaTrust } from './system-ca'
import { registerTerminalIpc } from './terminal-ipc'
import { nativeOverlayWidth as computeNativeOverlayWidth, titleBarOverlayOptions } from './titlebar-overlay-width'
import {
  backgroundMaterialFor,
  defaultTranslucencyState,
  glassActive,
  glassSupportedOn,
  normalizeState as normalizeTranslucency,
  opacityNeedsSetting,
  translucencySupportedOn,
  vibrancyFor as vibrancyForTranslucency,
  windowBackgroundMaterialOptions,
  windowBackingOptions,
  windowOpacityFor,
  windowOpacityOptions
} from './translucency'
import { updateGateReason, waitForUpdateClearance } from './update-gate'
import { readLiveUpdateMarker, updateHandoffConflict, writeUpdateMarker } from './update-marker'
import { updateConnectionsBeforeLocal } from './update-order'
import {
  resolveUpdaterMechanism,
  type UpdaterApplyResultWire,
  type UpdaterStatusWire,
  type UpdaterStrategy
} from './updater'
import {
  observeUpdaterHandoff,
  resolveInstallationLauncher,
  resolveStagedUpdaterBinary,
  resolveVenvDir,
  spawnUpdaterProcess,
  stagedUpdaterSupportsPrewrittenMarker,
  userLauncherInstallRoot
} from './updater-process'
import { AppInstallerStrategy } from './updater/app-installer'
import { createChannelAppInstallerStrategy } from './updater/app-installer'
import { ChannelResolver, type ChannelTarget } from './updater/channel'
import { inspectRunningChannelApp } from './updater/channel-native'
import { ChannelStrategy } from './updater/channel-strategy'
import { verifyPreparedChannelInstaller } from './updater/channel-windows-host'
import { createCheckoutStrategy } from './updater/checkout'
import { readSourceUpdate, type SourceUpdate } from './updater/checkout-source'
import { ExternalStrategy } from './updater/external'
import { readUpdatesFeedBaseFromConfig, resolveFeedBaseUrl } from './updater/feed-config'
import { createChannelMacStrategy, createMacStrategy } from './updater/mac-client'
import { UpdateOperation } from './updater/operation'
import {
  type ConsumedRelaunch,
  consumePendingRelaunch,
  registerUpdateRelaunch,
  type RelaunchRegistration
} from './updater/relaunch'
import { relaunchWaiterScript, startRelaunchWaiter } from './updater/relaunch-waiter'
import { preflightStateDb } from './updater/state-db-preflight'
import { createStoreStrategy } from './updater/store-client'
import { isExternalVenvHolder, isHermesOwnedVenvDaemon } from './venv-holder-select'
import { fetchMarketplaceThemes, searchMarketplaceThemes } from './vscode-marketplace'
import { createWakeIndicatorWindowController } from './wake-indicator-window'
import { guardedWatch } from './watch-storm-breaker'
import { windowAcceleratorAction } from './window-accelerator'
import { enumerateWindowsFrontToBack, enumerationFailed, readWindowBelow } from './window-below'
import { bindWindowChromeEvents } from './window-chrome-events'
import {
  appliedPrimaryWindowRoute,
  registrySshPoolScopeByConnectionId,
  registrySshScopeForWindowRoute,
  WindowConnectionRouteRegistry
} from './window-connection-route'
import { registerWindowControlIpc, windowControlState } from './window-controls'
import { revealAction, shouldFocusToTakeKeyboard } from './window-focus-policy'
import { windowMenuTemplate } from './window-menu'
import { createWindowOpenHandler } from './window-open-policy'
import { installWindowRendererLifecycle } from './window-renderer-lifecycle'
import { wireWindowReveal } from './window-reveal'
import {
  bindGeometryPersistence,
  computeWindowOptions,
  debounce,
  firstLaunchSize,
  sanitizeWindowState,
  MIN_HEIGHT as WINDOW_MIN_HEIGHT,
  MIN_WIDTH as WINDOW_MIN_WIDTH
} from './window-state'
import { hiddenWindowsChildOptions, windowsShellCommand } from './windows-child-options'
import { buildPathExtCandidates, chooseUpdaterArgs, resolveVenvHermesCommand } from './windows-hermes-path'
import {
  connectWindowsRemote,
  detectRemotePlatform,
  helper,
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
import {
  alreadyHasDisableGpu,
  buildDisableGpuRelaunchArgs,
  decideWindowsGpuStackCookieLaunch,
  gpuStackCookieFallbackMarker,
  isHermesDesktopGpuOverrideOff,
  markerAfterSuccessfulGpuStackCookieBoot,
  readGpuStackCookieMarker,
  shouldRelaunchForRendererStackCookieCrashLoop,
  shouldSurfaceErrorForRendererStackCookieCrashLoop,
  writeGpuStackCookieMarker
} from './windows-stack-cookie-fallback'
import { readWindowsUserEnvVar } from './windows-user-env'
import { isPackagedInstallPath as isPackagedInstallPathUnderRoots } from './workspace-cwd'
import { readWslWindowsClipboardImage } from './wsl-clipboard-image'
import { resolvePickerDefaultPath, setActiveGatewayProfile, setWslBridgeProfileState } from './wsl-path-bridge'

const IDENTITY_APP_NAME: string | null = applyDesktopIdentity(app)
const USER_DATA_OVERRIDE: string | undefined = process.env.HERMES_DESKTOP_USER_DATA_DIR

if (USER_DATA_OVERRIDE || process.env.HERMES_DATA_DIR_SUFFIX) {
  const resolvedUserData: string = resolveDesktopUserData(app.getPath('userData'))
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

  // #97616: disableHardwareAcceleration() alone does NOT stop a GPU child
  // from spawning (it then dies error_code=1002 on AMD/Mesa). For the explicit
  // HERMES_DESKTOP_DISABLE_GPU override, fully spawn-block it. Remote-display
  // detections keep their long-standing compositing-only behavior.
  if (disableGpuSwitchNeededForReason(REMOTE_DISPLAY_REASON)) {
    app.commandLine.appendSwitch('disable-gpu')
  }

  console.log(
    `[hermes] remote display detected (${REMOTE_DISPLAY_REASON}); disabling GPU hardware acceleration to prevent flicker`
  )
}

// #108047: a local Windows renderer crash loop with STATUS_STACK_BUFFER_OVERRUN
// (0xC0000409) is recovered by disabling GPU — NOT by dropping the sandbox
// (that path stays owned by STATUS_BREAKPOINT / #38216). Must run before app
// `ready`. Skip applying switches when the remote-display block above already
// did; still honor a sticky per-version marker so Start Menu launches recover.
let windowsGpuStackCookieFallbackActive = false
let windowsGpuStackCookieFallbackSticky = false
let windowsGpuStackCookieRelaunchAttempted = false

if (IS_WINDOWS) {
  const windowsGpuUserData = app.getPath('userData')

  const gpuStackCookieDecision = decideWindowsGpuStackCookieLaunch({
    argv: process.argv,
    marker: readGpuStackCookieMarker(windowsGpuUserData),
    env: process.env,
    appVersion: app.getVersion()
  })

  windowsGpuStackCookieFallbackActive = gpuStackCookieDecision.enable
  windowsGpuStackCookieFallbackSticky = gpuStackCookieDecision.nextMarker.state === 'fallback'

  try {
    writeGpuStackCookieMarker(windowsGpuUserData, gpuStackCookieDecision.nextMarker)
  } catch {
    void 0
  }

  if (gpuStackCookieDecision.enable && !REMOTE_DISPLAY_REASON) {
    app.disableHardwareAcceleration()
    app.commandLine.appendSwitch('disable-gpu-compositing')
    console.log(
      `[hermes] Windows GPU stack-cookie fallback enabled (${gpuStackCookieDecision.reason}); disabling GPU hardware acceleration (0xC0000409 / #108047)`
    )
  }
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
// display already forced software (SSH'd-into-WSL), and on Wayland ozone: WSL has
// no DRM render node, so forced GPU compositing segfaults the GPU process there.
if (
  IS_WSL &&
  !REMOTE_DISPLAY_REASON &&
  fs.existsSync('/dev/dxg') &&
  linuxOzoneBackend(process.env, process.argv) !== 'wayland'
) {
  app.commandLine.appendSwitch('ignore-gpu-blocklist')
  app.commandLine.appendSwitch('enable-gpu-rasterization')
  app.commandLine.appendSwitch('enable-zero-copy')
  console.log('[hermes] WSL GPU passthrough (/dev/dxg) detected; enabling GPU acceleration')
}

// #40077 / #124255: NVIDIA driver 580.x breaks ANGLE's EGL probing (Invalid
// visual ID), killing the GPU process at startup. Route ANGLE through its
// SwiftShader backend when the breakage is WITNESSED, not assumed: the same
// point release breaks hosts where NVIDIA drives the display and renders fine
// on hybrid hosts whose session EGL lands on the iGPU, so a driver-series gate
// burns 4-9 CPU cores on healthy hosts (#124255). The gate is now behavioral —
// boot with hardware GL and a marker; a GPU-process death before the first
// window flips the marker sticky (per app + full driver version) and relaunches
// once with SwiftShader. Deliberately NOT disableHardwareAcceleration(): on
// 580.173.02 + Electron 40 that SIGKILLs the renderer (see the closed #40119).
// Must run before app `ready` — the switch only applies pre-launch. Override
// with HERMES_DESKTOP_NVIDIA_SWIFTSHADER (1/true → force on, 0/false → never).
const NVIDIA_PROC_VERSION = (() => {
  try {
    return fs.readFileSync('/proc/driver/nvidia/version', 'utf8')
  } catch {
    return ''
  }
})()

const NVIDIA_DRIVER_MAJOR = parseNvidiaDriverMajor(NVIDIA_PROC_VERSION)
const NVIDIA_DRIVER_VERSION = parseNvidiaDriverVersion(NVIDIA_PROC_VERSION)

let nvidiaEglFallbackActive = false
let nvidiaEglRelaunchAttempted = false

const NVIDIA_EGL_FALLBACK = decideNvidiaEglFallback({
  driverMajor: NVIDIA_DRIVER_MAJOR,
  driverVersion: NVIDIA_DRIVER_VERSION,
  marker: readNvidiaEglMarker(app.getPath('userData')),
  appVersion: app.getVersion(),
  env: process.env,
  platform: process.platform,
  isWsl: IS_WSL,
  remoteDisplayReason: REMOTE_DISPLAY_REASON
})

nvidiaEglFallbackActive = NVIDIA_EGL_FALLBACK.enable

// Persist the launch decision before GPU children start: a `booting` marker
// left behind by a launch that never reached first paint is itself evidence
// of a GPU death (the "GPU process isn't usable" FATAL abort wins the race
// against our relaunch handler), and the next launch engages from it.
if (NVIDIA_DRIVER_MAJOR !== null) {
  try {
    writeNvidiaEglMarker(app.getPath('userData'), NVIDIA_EGL_FALLBACK.nextMarker)
  } catch {
    void 0
  }
}

if (NVIDIA_EGL_FALLBACK.enable) {
  app.commandLine.appendSwitch('use-angle', 'swiftshader')
  console.log(
    `[hermes] NVIDIA EGL fallback enabled (${NVIDIA_EGL_FALLBACK.reason}); routing ANGLE ` +
      'through SwiftShader. Witnessed GPU-process death probe (#40077, #124255); an app or ' +
      'driver update re-probes hardware GL once. HERMES_DESKTOP_NVIDIA_SWIFTSHADER=0 to opt out.'
  )
}

// The behavioral half of the gate: a GPU-process death on a Linux NVIDIA host
// that booted with hardware GL is the #40077 signature. Catch it before
// Chromium's "GPU process isn't usable" FATAL abort ends the process, flip the
// marker sticky, and relaunch once with SwiftShader. `killed` counts (the
// #40077 GPU process died to Chromium's health-check SIGTERM, exit_code=15).
if (NVIDIA_DRIVER_MAJOR !== null && process.platform === 'linux') {
  app.on('child-process-gone', (_event, details) => {
    if (
      !shouldRelaunchForNvidiaGpuDeath({
        details,
        fallbackActive: nvidiaEglFallbackActive,
        relaunchAttempted: nvidiaEglRelaunchAttempted
      })
    ) {
      return
    }

    nvidiaEglRelaunchAttempted = true

    try {
      writeNvidiaEglMarker(
        app.getPath('userData'),
        nvidiaEglFallbackMarker(app.getVersion(), NVIDIA_DRIVER_VERSION ?? String(NVIDIA_DRIVER_MAJOR))
      )
    } catch {
      void 0
    }

    console.warn(
      `[hermes] NVIDIA GPU process died (reason=${details?.reason}, exit=${details?.exitCode}); ` +
        'relaunching once with --use-angle=swiftshader (#40077, #124255)'
    )

    try {
      app.relaunch({
        args: [...process.argv.slice(1), '--use-angle=swiftshader']
      })
      void exitAfterBackendShutdown(0)
    } catch (error) {
      console.error(`[hermes] NVIDIA SwiftShader relaunch failed: ${error?.message || error}`)
    }
  })
}

// #124843: on Mesa/Wayland the Chromium GPU child can fail init
// (error_code=1002) and retry inside a sub-zygote forever — ~350% CPU, no
// gpu-process, no crash. Bound it: one relaunch into software rendering,
// then a sticky per-version marker so the next boot goes straight there.
// Reactive only — healthy Mesa/Wayland stacks keep full acceleration. Must
// run before app `ready`. Override with HERMES_DESKTOP_DISABLE_GPU
// (1/true → always software, 0/false → keep GPU on).
let linuxGpuFallbackActive = false
let linuxGpuFallbackSticky = false
let linuxGpuRelaunchAttempted = false

const LINUX_GPU_SOFTWARE_ACTIVE =
  Boolean(REMOTE_DISPLAY_REASON) || NVIDIA_EGL_FALLBACK.enable || alreadyHasDisableGpu(process.argv, process.env)

if (process.platform === 'linux') {
  const linuxGpuUserData = app.getPath('userData')

  const linuxGpuDecision = decideLinuxGpuLaunch({
    argv: process.argv,
    env: process.env,
    marker: readLinuxGpuMarker(linuxGpuUserData),
    appVersion: app.getVersion(),
    remoteDisplayReason: REMOTE_DISPLAY_REASON,
    nvidiaFallbackActive: NVIDIA_EGL_FALLBACK.enable
  })

  linuxGpuFallbackActive = linuxGpuDecision.enable
  linuxGpuFallbackSticky = linuxGpuDecision.nextMarker.state === 'fallback'

  try {
    writeLinuxGpuMarker(linuxGpuUserData, linuxGpuDecision.nextMarker)
  } catch {
    void 0
  }

  if (linuxGpuDecision.enable && linuxGpuDecision.reason !== 'already-enabled' && !LINUX_GPU_SOFTWARE_ACTIVE) {
    app.disableHardwareAcceleration()
    app.commandLine.appendSwitch('disable-gpu-compositing')
    console.log(
      `[hermes] Linux GPU software fallback enabled (${linuxGpuDecision.reason}); disabling GPU ` +
        'hardware acceleration after a GPU-child init failure (#124843). HERMES_DESKTOP_DISABLE_GPU=0 to opt out.'
    )
  }
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

// #121954: the two-strike boot-abort ladder now also covers Linux. On Linux
// hosts where the sandboxed GPU child cannot start (dies pre-main on an
// FD-ownership violation), Chromium prints "GPU process isn't usable.
// Goodbye." and aborts — a 100% crash loop; the host isolation matrix in
// #121954 shows only `--no-sandbox` reaches the UI. Same sticky per-version
// recovery as #38216: two consecutive mid-boot aborts engage `--no-sandbox`,
// an app update re-probes the sandbox once. Windows-only extras (ACL repair,
// renderer crash-loop relaunch) stay inside the IS_WINDOWS branch.
if (IS_WINDOWS || process.platform === 'linux') {
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
      `[hermes] sandbox fallback enabled (${sandboxDecision.reason}); launching with --no-sandbox (#38216, #121954)`
    )
  }

  writeSandboxMarker(windowsUserData, sandboxDecision.nextMarker)

  // One coalesced Linux GPU-child recovery (#86073, #124843, #121954): the
  // sandbox signature is tried first (that host's matrix shows --disable-gpu
  // still crashes), then the software ladder — including the relapse after a
  // --no-sandbox boot died again. One death, one bounded relaunch; Windows
  // keeps its breakpoint-signature fast path unchanged.
  app.on('child-process-gone', (_event, details) => {
    if (IS_WINDOWS) {
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
        `[hermes] GPU child died with the sandbox signature (exit=${details?.exitCode}); relaunching once with --no-sandbox (#38216)`
      )

      try {
        app.relaunch({ args: buildNoSandboxRelaunchArgs(process.argv.slice(1)) })
        void exitAfterBackendShutdown(0)
      } catch (error) {
        console.error(`[hermes] --no-sandbox relaunch failed: ${error?.message || error}`)
      }

      return
    }

    const alreadySoftware =
      LINUX_GPU_SOFTWARE_ACTIVE || linuxGpuFallbackActive || alreadyHasDisableGpu(process.argv, process.env)

    const path = linuxGpuChildDeathPath({
      details,
      alreadyNoSandbox: windowsSandboxFallbackActive || alreadyHasNoSandbox(process.argv, process.env),
      alreadySoftware,
      sandboxRelaunchAttempted: windowsNoSandboxRelaunchAttempted,
      softwareRelaunchAttempted: linuxGpuRelaunchAttempted
    })

    if (path === 'no-sandbox') {
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
        `[hermes] Linux GPU child died with the sandbox signature (exit=${details?.exitCode}); relaunching once with --no-sandbox (#121954)`
      )

      try {
        app.relaunch({ args: buildNoSandboxRelaunchArgs(process.argv.slice(1)) })
        void exitAfterBackendShutdown(0)
      } catch (error) {
        console.error(`[hermes] --no-sandbox relaunch failed: ${error?.message || error}`)
      }

      return
    }

    if (path === 'disable-gpu') {
      linuxGpuRelaunchAttempted = true
      linuxGpuFallbackActive = true
      linuxGpuFallbackSticky = true

      const reason =
        String(details?.reason || '').toLowerCase() === 'launch-failure' ? 'gpu-launch-failure' : 'gpu-crash'

      try {
        writeLinuxGpuMarker(app.getPath('userData'), linuxGpuFallbackMarker(reason, app.getVersion()))
      } catch {
        void 0
      }

      console.warn(
        `[hermes] Linux GPU child gone (reason=${details?.reason}); relaunching once with --disable-gpu (#124843)`
      )

      try {
        app.relaunch({ args: buildDisableGpuRelaunchArgs(process.argv.slice(1)) })
        void exitAfterBackendShutdown(0)
      } catch (error) {
        console.error(`[hermes] --disable-gpu relaunch failed: ${error?.message || error}`)
      }
    }
  })
}

ipcMain.handle('hermes:get-remote-display-reason', () => REMOTE_DISPLAY_REASON)
ipcMain.handle('hermes:embed-host:origin', () => embedHostOrigin())

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

// Runtime identity comes only from the baked artifact stamp. Dev runs have none.
if (INSTALL_STAMP) {
  console.log(
    `[hermes] install stamp: ${INSTALL_STAMP.commit ? INSTALL_STAMP.commit.slice(0, 12) : 'no-commit'}${INSTALL_STAMP.branch ? ` (${INSTALL_STAMP.branch})` : ''}${INSTALL_STAMP.dirty ? ' [DIRTY]' : ''} from ${INSTALL_STAMP.source || 'unknown'}`
  )
} else if (IS_PACKAGED) {
  // Dev builds without a stamp are normal; packaged builds without one
  // mean the bootstrap won't know what to clone. Surface clearly.
  console.error(
    '[hermes] WARNING: no install-stamp.json found in packaged build. First-launch bootstrap will not have a pinned ref to install.'
  )
}

const DESKTOP_PROFILE_CONFIG_PATH: string = path.join(app.getPath('userData'), 'active-profile.json')

// Only the lock-owning destination may adopt a workspace or start a backend.
// #78101: on Linux/X11 a zombie/defunct Electron process leaves the
// SingletonLock symlink behind with a PID that still answers kill(pid, 0),
// so Chromium's own liveness probe keeps refusing every later launch and the
// app silently exits. Clear a provably-dead owner and retry once; always log
// when the lock is legitimately lost so the exit is diagnosable.
function acquireSingleInstanceLock(): boolean {
  if (app.requestSingleInstanceLock()) {
    return true
  }

  const stalePid = removeStaleSingletonLock(app.getPath('userData'))

  if (stalePid !== null) {
    console.error(`[hermes] removed stale SingletonLock (owner ${stalePid} dead); retrying launch`)

    return app.requestSingleInstanceLock()
  }

  return false
}

const isPrimaryInstance: boolean = acquireSingleInstanceLock()

if (!isPrimaryInstance) {
  console.error('[hermes] another Hermes Desktop instance holds the single-instance lock; exiting')
  app.exit(0)
}

// `hermes desktop` shortens TMPDIR only so the lock above can bind its socket (#124688). The
// backend and every other child get the real one (the profile scratch dir) back.
if (process.env.HERMES_DESKTOP_TMPDIR) {
  process.env.TMPDIR = process.env.HERMES_DESKTOP_TMPDIR
  delete process.env.HERMES_DESKTOP_TMPDIR
}

const HERMES_HOME: string = resolveDesktopHermesHome({
  home: app.getPath('home'),
  directoryExists,
  readWindowsHome: (): string | null => readWindowsUserEnvVar('HERMES_HOME')
})

// #77311: `desktop.electron_flags` and the renderer heap ceiling
// (`desktop.renderer_max_old_space_mb`) used to reach Chromium only through
// the `hermes desktop` launcher's argv, so a packaged app opened from its
// Start-menu / .desktop entry ran with no `--js-flags` at all. Apply them here
// from config.yaml, before `ready` — Chromium copies `js-flags` to renderer
// processes only from the browser's pre-launch command line.
// `desktop.ssh_path` (#103288) rides the same pre-window read: an explicit
// Windows ssh client for when the in-box OpenSSH is missing or broken.
let desktopSshPathOverride = ''

{
  let desktopLaunchYaml: string = ''

  try {
    desktopLaunchYaml = fs.readFileSync(path.join(HERMES_HOME, 'config.yaml'), 'utf8')
  } catch {
    void 0 // first run: no config yet → Chromium defaults
  }

  const desktopLaunchConfig = readDesktopLaunchConfig(desktopLaunchYaml)
  desktopSshPathOverride = desktopLaunchConfig.sshPath || ''

  // `desktop.renderer_accessibility: false` must reach packaged launches too,
  // not only the `hermes desktop` launcher's env bridge (#118271).
  if (
    desktopLaunchConfig.rendererAccessibility === false &&
    process.env.HERMES_DESKTOP_RENDERER_ACCESSIBILITY === undefined
  ) {
    process.env.HERMES_DESKTOP_RENDERER_ACCESSIBILITY = '0'
  }

  for (const planned of planLaunchSwitches(desktopLaunchConfig, process.argv.slice(1))) {
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

// ACTIVE_HERMES_ROOT — the canonical mutable Hermes install. Same path
// install.ps1 / install.sh use, so a desktop-only user and a CLI-only user end
// up with identical layouts and can share one install.
const ACTIVE_HERMES_ROOT = path.join(HERMES_HOME, 'hermes-agent')
setNoConsoleGitRoots([!IS_PACKAGED ? SOURCE_REPO_ROOT : null, ACTIVE_HERMES_ROOT])
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
const DESKTOP_MANAGED_SSH_RECOVERY_PATH = path.join(app.getPath('userData'), 'managed-ssh-update-recovery.json')
// active-profile.json records which Hermes profile the desktop launches its
// local backend as. When set, startHermes() passes `hermes --profile <name>
// dashboard …`, which deterministically pins HERMES_HOME (see
// _apply_profile_override in hermes_cli/main.py) and bypasses the sticky
// ~/.hermes/active_profile file. Unset (null) preserves the legacy behavior:
// no --profile flag, so the backend honors active_profile / default.

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
const DESKTOP_LOG_FLUSH_MS = 120
const DESKTOP_LOG_BUFFER_MAX_CHARS = 64 * 1024
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
// One launch decision must reach both the renderer and every backend spawn.
const GUEST_ONBOARDING: boolean = guestOnboardingEnabled()

const BOOT_FAKE_STEP_MS = (() => {
  const raw = Number.parseInt(String(process.env.HERMES_DESKTOP_BOOT_FAKE_STEP_MS || ''), 10)

  if (!Number.isFinite(raw) || raw <= 0) {
    return 650
  }

  return Math.max(120, raw)
})()

const APP_NAME: string = IDENTITY_APP_NAME || process.env.HERMES_DESKTOP_APP_NAME || 'Hermes'
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
    ...windowBackgroundMaterialOptions(translucencyState, IS_WINDOWS, GLASS_SUPPORTED),
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

// WSLg returns false: the RDP host paints nothing for a frameless window and
// Electron's own overlay drifts its hit-region under RAIL, so the renderer
// paints its own min/max/close (wslg-window-controls.tsx) over the
// hermes:window-control IPC channel. See titleBarOverlayOptions.
function getTitleBarOverlayOptions(win?) {
  return titleBarOverlayOptions({
    platform: IS_MAC ? 'mac' : IS_WINDOWS ? 'windows' : IS_WSL ? 'wslg' : 'linux',
    darwinMajor: DARWIN_MAJOR,
    titlebarHeight: TITLEBAR_HEIGHT,
    color: TITLEBAR_OVERLAY_COLOR,
    foreground:
      rendererTitleBarTheme && isHexColor(rendererTitleBarTheme.foreground) ? rendererTitleBarTheme.foreground : null,
    dark: nativeTheme.shouldUseDarkColors,
    // The native WCO buttons don't scale with the page; scale the overlay so
    // its height tracks the zoomed renderer titlebar (#81086).
    zoomFactor: win?.webContents?.getZoomFactor?.()
  })
}

// Push refreshed overlay options to a live window after a theme/appearance
// change. No-op only on plain (non-WSL) Linux, where getTitleBarOverlayOptions()
// returns false; the try/catch additionally guards builds where
// setTitleBarOverlay isn't supported.
function applyTitleBarOverlay(win) {
  const options = getTitleBarOverlayOptions(win)

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
  app.setAppUserModelId(IDENTITY_APP_NAME ? PRODUCT_IDENTITY.appId : 'com.nousresearch.hermes')
}

// Seed the native About panel with the best-known Hermes version. This is
// refreshed on every open via showAboutPanelFresh, so an in-place
// `hermes update` mid-session is reflected without an app restart; the seed
// covers the first open and any non-menu invocation path. Never seed empty:
// an empty applicationVersion falls back to the bundle version, which is the
// 0.0.0 placeholder on local builds (#124581).
app.setAboutPanelOptions({
  applicationName: APP_NAME,
  applicationVersion: nativeAboutVersion(appVersionInfo(INSTALL_STAMP, '', app.getVersion())),
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

function registerMediaProtocol(): void {
  const handler: ReturnType<typeof createMediaProtocolHandler> = createMediaProtocolHandler({
    ensureRemoteBearer: (baseUrl: string): Promise<string | null> => ensureNativeAccessToken(baseUrl),
    // Electron's file:// loader ignores Range, which prevents video seeking.
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
      // On a Windows host with a WSL backend the media path arrives as a
      // WSL/POSIX path (`/home/...`, `/mnt/c/...`) the Windows fs can't open
      // as-is; bridge it to a UNC/drive form first, same as directory reads.
      // The protocol handler already percent-decoded the pathname, so this
      // boundary bridges only — re-decoding/stripping would corrupt the path.
      const { resolvedPath } = await resolveReadableFileForIpc(resolveMediaStreamFile(filePath), {
        purpose: 'Media stream'
      })

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

const localBackendLifecycle = createLocalBackendLifecycle<ChildProcess>({
  stopChild: (child: ChildProcess): void => {
    if (child.exitCode === null && child.signalCode === null) {
      stopBackendChildImpl(child, { forceKillProcessTree, isWindows: IS_WINDOWS })
    }
  },
  waitForExit: (child: ChildProcess): Promise<void> =>
    waitForBackendExit(child, { forceKillProcessTree, isWindows: IS_WINDOWS }),
  cancelSetup: (): void => {
    firstRunSetupGate?.resetForRetry()
    bootstrapAbortController?.abort()
  }
})

function spawnOwnedBackend(...args: Parameters<typeof spawn>): ChildProcess {
  const child = localBackendLifecycle.spawn((): ChildProcess => spawn(...args))
  child.once('exit', (): boolean => localBackendLifecycle.release(child))
  child.once('error', (): void => {
    if (!child.pid) {
      localBackendLifecycle.release(child)
    }
  })

  return child
}

const remoteLiveness = new RemoteLivenessTracker()

// Pooled remotes are probed on the renderer reconnect cadence (minutes apart),
// not the primary's sub-minute retry loop, so they need a failure window wider
// than that cadence or a dead pooled descriptor's streak resets on every tick
// and it is never dropped (#94381).
const pooledRemoteLiveness = new RemoteLivenessTracker(undefined, REMOTE_POOLED_LIVENESS_FAILURE_WINDOW_MS)

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

// rememberLog() state. Declared here, before the top-level
// readPersistedPoolLimits() call below, because that call logs during module
// evaluation; declaring these later crashed launch with `undefined.push` in
// the packaged build (esbuild lowers the TDZ to undefined instead of throwing).
const hermesLog: string[] = []
let desktopLogBuffer = ''
let desktopLogFlushTimer = null
let desktopLogFlushPromise = Promise.resolve()

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

// Background hydration backs off after a slot timeout. Foreground opens bypass the cooldown.
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
  void evictLruPoolBackends(poolMaxBackends()).catch((error: Error): void =>
    rememberLog(`Pool LRU eviction failed: ${String(error)}`)
  )
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

// Pinned-tier TTL (#105239): the renderer's 60s keepalive (touchPoolBackend)
// refreshes lastActiveAt for every OPEN chat, so the idle reaper's only clock
// never fires for the pinned tier — every profile whose chat was ever opened
// held its ~120 MB serve child until app quit (126 processes / 7.5 GB on the
// reporter's machine, all parented to Hermes.exe). A keepalive proves the
// chat is open, not that anything streamed: retire a local child whose last
// streamed turn is older than this window. Re-focusing the chat re-ensures it
// idempotently (ensureBackend/ensureRegistryBackend reuse), and mid-stream
// safety is unchanged — activeTurn entries are excluded by the retirer.
const POOL_PINNED_IDLE_MS = Math.max(
  POOL_LIMITS_MIN.idleMs,
  Number(process.env.HERMES_DESKTOP_POOL_PINNED_IDLE_MS) || 60 * 60_000
)

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
const remoteHeaderSessions = new WeakSet<object>()
const remoteWsHeaderStore = createRemoteWsHeaderStore()
const previewWatchers = new Map()
let previewShortcutActive = false
const f12ShortcutActiveWindows = new Set<number>()
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

// Chromium owns its --log-file for the life of the process, so the startup
// reclaim above cannot bound a shell that stays up for days writing errors.
// Poll and truncate in place; renaming would leave Chromium appending to the
// renamed inode. Unref'd so it never holds the process open.
function startChromiumLogWatcher(file) {
  const io = {
    size: f => {
      try {
        return fs.statSync(f).size
      } catch {
        return null // Not created yet — nothing has been logged.
      }
    },
    truncate: f => fs.truncateSync(f, 0)
  }

  const timer = setInterval(() => {
    try {
      if (reclaimActiveLogIfOversized(file, io)) {
        rememberLog(`[diagnostics] truncated oversized Chromium log ${file}`)
      }
    } catch {
      // Best-effort — an unbounded log beats a crashed shell.
    }
  }, ACTIVE_LOG_POLL_MS)

  timer.unref?.()
}

function rotateLogIfNeededSync(base) {
  let size

  try {
    size = fs.statSync(base).size
  } catch {
    return // No live file yet — the append (re)creates it.
  }

  for (const [op, src, dst] of planLogRotation(size, base)) {
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

  for (const [op, src, dst] of planLogRotation(size, DESKTOP_LOG_PATH)) {
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
    rotateLogIfNeededSync(DESKTOP_LOG_PATH)
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
  // at the same moment.  Local time, same shape as agent.log/gui.log.
  const stamp = formatLogStamp(new Date())
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

// Main-process stalls leave renderer-scoped lifecycle logging unable to run.
// When the loop resumes, retain the delayed timer's timing in desktop.log so a
// Windows AppHang report can be correlated without changing tray semantics.
const mainProcessLagWatchdog = createMainProcessLagWatchdog({
  cadenceMs: 1_000,
  thresholdMs: 2_000,
  now: Date.now,
  log: rememberLog,
  setInterval,
  clearInterval
})

installCrashForensics({ flush: flushDesktopLogBufferSync, log: rememberLog })

// A rejected loadURL leaves a blank window and, unhandled, no trace anywhere
// the user can send us. `label` names the surface so the log says which one.
function loadWindowUrl(win, url, label) {
  win.loadURL(url).catch(error => rememberLog(`${label} failed to load: ${describeCrashReason(error)}`))
}

const EXTERNAL_OPEN_DEPS: ExternalOpenDeps = {
  isWsl: IS_WSL,
  spawn: (cmd, args, opts) => spawn(cmd, args, opts),
  openExternal: url => shell.openExternal(url),
  openFile: openExternalFile,
  openLocalPath: openLocalFilesystemPath,
  notifyFailure: broadcastOpenFailed,
  log: rememberLog
}

// Deps for the pre-open stat guard (see openExternalFile): a miss is
// broadcast with the 'missing-file' code so the dialog shows file-not-found
// copy; other stat failures only log.
const GUARD_REPORT_DEPS = {
  log: rememberLog,
  reportMissing: (rawUrl: string, message: string) => broadcastOpenFailed(rawUrl, message, 'missing-file')
}

// The single route every external URL open funnels through (external-open.ts).
// main.ts only binds the electron deps; all open/fallback logic lives in the
// module so it unit-tests without loading electron.
function openExternalUrl(rawUrl: string) {
  return externalOpen(String(rawUrl || '').trim(), EXTERNAL_OPEN_DEPS)
}

async function openPreviewInBrowser(rawUrl: string) {
  const result = await externalOpen(String(rawUrl || '').trim(), EXTERNAL_OPEN_DEPS)

  // Only an invalid/unsupported URL is a hard "no" for the caller; an open
  // failure already raised the fallback modal with the URL.
  return !(result.ok === false && result.reason === 'invalid')
}

// `file://` URLs come from the artifacts panel (the renderer can't open them
// itself because Chromium blocks that navigation). Reveal the file in the
// system file manager instead of dispatching to the OS file association:
// on Windows, archive artifacts (.gz/.tar) have no usable association, and
// handing the path back to the OS shell bounces the open through the default
// (Chromium) handler, which re-downloads the file — an infinite download loop
// (issue #53170). Reveal-in-folder never re-opens the file, so it can't loop.
//
// A short per-path dedupe window additionally absorbs renderer-side double
// clicks and retry storms so repeated open requests can't pile up windows.
const FILE_REVEAL_DEDUPE_MS = 1_500
const recentFileReveals = new Map<string, number>()

export function _resetFileRevealDedupeForTest() {
  recentFileReveals.clear()
}

async function openExternalFile(rawUrl: string) {
  let localPath: string

  try {
    localPath = resolveRequestedPathForIpc(rawUrl, { purpose: 'Open external file' })
  } catch {
    return
  }

  // A missing file must never reach the reveal fallback: on macOS revealing a
  // non-existent path is silently a no-op, so the click would do nothing at
  // all. Say "missing" before the OS is asked. Only ENOENT/ENOTDIR count as
  // missing — any other stat failure (EACCES on a locked volume, ELOOP) is
  // logged and still reaches the OS below, so an existing-but-locked file
  // keeps its real error instead of a fabricated miss. Classification lives
  // in external-open.ts so it unit-tests without electron. Misses are
  // reported here and not rethrown: external-open.ts documents that openFile
  // handles its own reporting, and rethrowing would stack a second dialog.
  try {
    assertExistingPathForOpen(localPath, 'Open external file')
  } catch (error) {
    if (reportPreOpenStatFailure(error, rawUrl, GUARD_REPORT_DEPS)) {
      return
    }
  }

  const now = Date.now()
  const lastReveal = recentFileReveals.get(localPath)

  if (lastReveal !== undefined && now - lastReveal < FILE_REVEAL_DEDUPE_MS) {
    rememberLog(`[file] duplicate reveal request within ${FILE_REVEAL_DEDUPE_MS}ms; ignored: ${localPath}`)

    return
  }

  recentFileReveals.set(localPath, now)

  // Prune stale entries so the map can't grow without bound over a long session.
  for (const [path, ts] of recentFileReveals) {
    if (now - ts >= FILE_REVEAL_DEDUPE_MS && path !== localPath) {
      recentFileReveals.delete(path)
    }
  }

  try {
    shell.showItemInFolder(localPath)
  } catch (error) {
    rememberLog(`[file] reveal in folder failed: ${error instanceof Error ? error.message : String(error)}`)
  }
}

// The `hermes:openExternal` route for a BARE local filesystem path — a chat
// media link, a markdown href, or an artifacts-panel value carrying
// `C:\…`, `~/…`, `/…` or a UNC path instead of a `file://` URL. `new URL()`
// cannot express those (hermes-agent 80946): resolve through the same audited
// `resolveRequestedPathForIpc` the file route uses, then OPEN with the OS
// handler and fall back to reveal-in-folder, mirroring openExternalFile's
// missing-file guard so a dead path reports "File not found" instead of a
// silent no-op. Resolves false only when the path could not be resolved at
// all; open failures are logged (with the path, so "Failed to open path" is
// diagnosable) and still count as handled.
async function openLocalFilesystemPath(rawPath: string): Promise<boolean> {
  let localPath: string

  try {
    localPath = resolveRequestedPathForIpc(String(rawPath || ''), { purpose: 'Open external file' })
  } catch {
    return false
  }

  try {
    assertExistingPathForOpen(localPath, 'Open external file')
  } catch (error) {
    if (reportPreOpenStatFailure(error, rawPath, GUARD_REPORT_DEPS)) {
      return true
    }
  }

  try {
    const errorMessage = await shell.openPath(localPath)

    if (!errorMessage) {
      return true
    }

    // Include the path so "Failed to open path" is diagnosable (hermes-agent
    // 84361), then reveal: on Windows archive artifacts have no usable
    // association, and the reveal never re-opens so it can't loop (#53170).
    rememberLog(`[file] openPath failed: ${errorMessage}; path=${localPath}; revealing in folder instead`)

    try {
      shell.showItemInFolder(localPath)
    } catch (revealError) {
      rememberLog(
        `[file] showItemInFolder failed: ${revealError instanceof Error ? revealError.message : String(revealError)}; path=${localPath}`
      )
    }

    return true
  } catch (error) {
    rememberLog(
      `[file] openPath rejected: ${error instanceof Error ? error.message : String(error)}; path=${localPath}`
    )

    return true
  }
}

// An open failure is surfaced to the renderer as a modal carrying the URL, so
// a dead system-browser click (e.g. no https handler registered on Linux) is
// never silent. `code` tags the failure class (e.g. 'missing-file') so the
// dialog can show accurate localized copy instead of the generic one.
// Broadcast to every window — the trigger has no single sender.
function broadcastOpenFailed(url: string, message: string, code?: 'missing-file') {
  rememberLog(`[open-failed] ${url}: ${message}`)

  for (const win of BrowserWindow.getAllWindows()) {
    win.webContents.send('hermes:external-open-failed', { url, message, ...(code ? { code } : {}) })
  }
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
          activeRoot: ev.activeRoot,
          local: ev.local || 'none',
          bundled: installShape() === 'bundled'
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
  return bootstrapSnapshot(bootstrapState)
}

function resetBootstrapSnapshot(): void {
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
    activeRoot: backend.activeRoot || ACTIVE_HERMES_ROOT,
    local: backend.local || 'none',
    bundled: installShape() === 'bundled'
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

// The latched reauth failure whose hold has already been logged, so a burst of
// dropped updates from one in-flight sibling attempt logs once, not per event.
let bootProgressHeldFor: Error | null = null

function updateBootProgress(update, options: { allowDecrease?: boolean } = {}) {
  // A latched CONFIRMED reauth rejection owns the boot surface until a
  // recovery path clears it. Updates that are not a re-emit of that failure —
  // a running:true phase or cleared error from an attempt already in flight
  // when the latch closed, or an unrelated sibling failure that would flip
  // retryable back on — must not reach the renderer, or the overlay's Sign in
  // button flickers away again (#95701).
  if (shouldHoldBootProgressForReauth(remoteReauthFailure ? remoteReauthFailure.message : null, update)) {
    if (bootProgressHeldFor !== remoteReauthFailure) {
      bootProgressHeldFor = remoteReauthFailure
      rememberLog('[boot] remote reauth latched: holding the recovery overlay against a stale boot-progress update')
    }

    return
  }

  bootProgressHeldFor = null

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

function isGitCheckout(root: string): boolean {
  return fs.existsSync(path.join(root, '.git'))
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
// spawn path. Consulting the on-disk marker, the in-process updateInFlight
// flag, AND the successful detached hand-off state is load-bearing (#73822):
// applyUpdates stops its own backend before committing the update hand-off.
// A marker-only gate lets the renderer's reconnect respawn a backend during
// that critical section, racing the update and leaving a live process on the
// runtime being replaced.
// The hand-off state closes the later Windows `cmd start` wrapper gap: the
// wrapper exits 0 before the real PowerShell script claims the marker, and
// `finally` clears updateInFlight immediately after the hand-off is accepted.
function updateGateDeps() {
  return {
    hasLiveMarker: () => Boolean(readLiveUpdateMarker(HERMES_HOME)),
    isUpdateInFlight: () => updateInFlight,
    isHandoffActive: () => isQuittingForHandoff,
    // The latest receipt is cross-process truth: a `hermes update` that failed
    // records outcome "failed" even when its marker write/release raced a
    // crash (#122206). Only a TERMINAL failure counts — "running" must keep
    // parking, and "partial" kept the install usable.
    hasFailedReceipt: () => {
      const receipt = readLatestSyncReceipt()

      return receipt?.outcome === 'failed'
    }
  }
}

// One-shot guard for the automatic bundle-swap relaunch below: the relaunched
// instance carries this flag so a stamp that still mismatches (unreadable
// resources, exotic packaging) can never produce a relaunch loop.
const BUNDLE_SWAP_RELAUNCH_FLAG = '--hermes-bundle-swap-relaunched'

// How long the parked instance waits for its own scheduled exit to land before
// giving up and booting the stale build anyway. Better a torn renderer with a
// banner than a window that never comes back.
const BUNDLE_SWAP_RELAUNCH_FAILSAFE_MS = 15_000

// The detached updater swaps the packaged bundle on disk AFTER `hermes update`
// exits (posix.sh mac_swap / windows.ps1). An instance reopened mid-update —
// the #50238 gesture the gate above exists for — was launched from the
// PRE-swap bundle, and the updater's `open` leg then merely focuses us (single
// instance), so no process ever loads the new build. Letting boot proceed here
// runs the new runtime under the old renderer: exactly the skew
// detectRendererSkew() warns about, except the Updates card already says
// "latest", so the warning's own remedy has nothing to run.
//
// This is the earliest point where the swap is PROVABLE — it happens while we
// are parked on the gate, so checking any sooner (at `ready`, before the gate)
// only ever compares a stamp with itself. Relaunching here also keeps the
// boot-progress window up for the whole wait instead of leaving the user with
// no window at all.
//
// Returns true when the relaunch was scheduled; the caller must park rather
// than continue booting, because the process exits underneath it.
function relaunchIntoSwappedBundle() {
  if (!IS_PACKAGED || process.argv.includes(BUNDLE_SWAP_RELAUNCH_FLAG)) {
    return false
  }

  if (!detectBundleSwap(INSTALL_STAMP, readBundleSwapStamp(process.resourcesPath))) {
    return false
  }

  rememberLog('[updates] app bundle was swapped during the update; relaunching into the new build')

  try {
    app.relaunch({
      args: [...buildNoSandboxRelaunchArgs(process.argv.slice(1)), BUNDLE_SWAP_RELAUNCH_FLAG]
    })
  } catch (err) {
    rememberLog(`[updates] bundle-swap relaunch failed: ${err?.message || err}; continuing with the current build`)

    return false
  }

  void exitAfterBackendShutdown(0)

  return true
}

// Block until no live update is in progress (or we hit the wait timeout).
// Emits a boot-progress phase so the renderer shows "Update in progress…"
// rather than a frozen splash. Returns true if it parked at all.
async function waitForUpdateToFinish() {
  let announced = false
  let parkedOnFailedReceipt = false

  const outcome = await waitForUpdateClearance(updateGateDeps(), {
    signal: localBackendLifecycle.signal,
    abandonOn: reason => {
      // The update that owns the gate already recorded a terminal failure
      // (#122206): parking the full 20-minute budget on a receipt that says
      // "failed" strands the window behind a dead updater (486 silent polls
      // measured). Stop waiting; the failure dialog below carries the
      // recovery guidance and the backend's own launch path finishes only
      // what is safely retryable, bounded by venv_sync's completion-retry
      // backoff.
      if (reason === 'failed-receipt') {
        parkedOnFailedReceipt = true
        rememberLog('[updates] latest update receipt records a failure; not parking the boot on it')

        return true
      }

      return false
    },
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
      const handoffLogPath = path.join(HERMES_HOME, 'logs', 'desktop-update-handoff.log')

      // Async so boot is not blocked behind the dialog; the response handlers
      // reuse the menu's open-updates path (queued until the renderer is ready)
      // and the same reveal primitive as 'hermes:logs:reveal'.
      void dialog
        .showMessageBox({
          type: 'error',
          title: 'Hermes update',
          message: "Hermes couldn't finish updating",
          detail:
            "You're still on the previous version and can keep using it. Try the update again, or open the update log to report the problem.\n\n" +
            `Details: ${result.message}`,
          buttons: ['Try again', 'Open log', 'Close'],
          defaultId: 0,
          cancelId: 2,
          noLink: true
        })
        .then(({ response }) => {
          if (response === 0) {
            sendOpenUpdatesRequested()
          } else if (response === 1) {
            shell.showItemInFolder(handoffLogPath)
          }
        })
    }
  } catch (err) {
    rememberLog(`[updates] could not read hand-off result: ${err.message}`)
  }

  if (outcome === 'cancelled') {
    localBackendLifecycle.assertCanStart()
  }

  if (outcome === 'clear') {
    return false
  }

  if (outcome === 'timeout') {
    rememberLog('[updates] update still in progress after wait timeout; starting backend anyway')
  } else if (parkedOnFailedReceipt) {
    // The gate closed on a terminal failure, not a live update: no swap to
    // relaunch into (the update never succeeded), so boot the current build
    // and let the failure dialog above carry the recovery guidance.
    rememberLog('[updates] proceeding with backend start despite the failed update receipt')
  } else if (relaunchIntoSwappedBundle()) {
    await advanceBootProgress('backend.update-restart', 'Restarting Hermes to load the updated app…', 14)
    // Park while the scheduled exit lands so this stale build never starts a
    // backend; the failsafe below only runs if the exit somehow does not.
    await new Promise(resolve => setTimeout(resolve, BUNDLE_SWAP_RELAUNCH_FAILSAFE_MS))
    rememberLog(
      `[updates] relaunch did not land within ${BUNDLE_SWAP_RELAUNCH_FAILSAFE_MS}ms; continuing with the current build`
    )
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

async function unwrapWindowsVenvHermesCommand(command, backendArgs) {
  return resolveVenvHermesCommand(command, backendArgs, {
    isWindows: IS_WINDOWS,
    isCommandScript,
    fileExists,
    directoryExists,
    canImportHermesCli,
    getVenvPython,
    buildDesktopBackendEnv,
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

  // existsSync for the archive: Electron's asar shim stats app.asar itself as a directory (so
  // fileExists was always false) and constructs the deprecated fs.Stats doing it (#96857).
  return (
    fs.existsSync(path.join(resourcesDir, 'app.asar')) || directoryExists(path.join(resourcesDir, 'app.asar.unpacked'))
  )
}

function isHermesSourceRoot(root) {
  return directoryExists(root) && fileExists(path.join(root, 'hermes_cli', 'main.py'))
}

async function findPythonForRoot(root: string): Promise<string | null> {
  return resolveSourcePython(root, {
    override: process.env.HERMES_DESKTOP_PYTHON,
    isWindows: IS_WINDOWS,
    fileExists
  })
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

function makeDashboardReadyFile() {
  const dir = path.join(app.getPath('userData'), 'backend-ready')
  fs.mkdirSync(dir, { recursive: true })

  return path.join(dir, `dashboard-${process.pid}-${Date.now()}-${crypto.randomBytes(6).toString('hex')}.json`)
}

// resolveGitBinary — locate git.exe on Windows. A fresh installer-driven
// install only has PortableGit under %LOCALAPPDATA%\hermes\git (never on
// PATH), so a bare spawn('git') ENOENTs and self-update checks fail with
// "Couldn't check for updates". PortableGit first, then UGit's bundled copy
// (see ./git-binary-candidates), then standard Git-for-Windows locations,
// then PATH. Cached after first probe.
let _gitBinaryCache = null

// A binary can exist on disk and still be unlaunchable — on macOS an
// Intel-only build ahead on PATH (e.g. a pre-Rosetta-removal Homebrew)
// fails at spawn time with errno -86 (EBADARCH), which callers then report
// as an update-server/network problem. Probing `git --version` before
// committing to a candidate skips such entries; the existence-only
// fallback keeps behaviour unchanged where the probe itself cannot run.
function binaryRuns(candidate) {
  try {
    execFileSync(candidate, ['--version'], {
      stdio: 'ignore',
      timeout: 5000,
      windowsHide: true
    })

    return true
  } catch {
    return false
  }
}

function findPathCandidates(command) {
  const pathEntries = String(process.env.PATH || '')
    .split(path.delimiter)
    .filter(Boolean)

  const candidates = []

  for (const entry of pathEntries) {
    const candidate = path.join(entry, command)

    if (fileExists(candidate)) {
      candidates.push(candidate)
    }
  }

  return candidates
}

function resolveGitBinary() {
  if (_gitBinaryCache) {
    return _gitBinaryCache
  }

  if (!IS_WINDOWS) {
    // Every PATH hit, probed — the first entry that merely exists can be
    // unlaunchable while a working system git sits later on the same PATH.
    const selected = selectRunnableBinary({
      candidates: findPathCandidates('git'),
      fileExists,
      binaryRuns
    })

    _gitBinaryCache = selected || 'git'

    return _gitBinaryCache
  }

  const localAppData = process.env.LOCALAPPDATA || ''

  // Fixed candidates + the UGit-bundled glob (a UGit install moves with every
  // app-* version, so it can only be found by enumerating the dir). UGit's
  // git.exe is usually on PATH, but an Explorer-launched Electron inherits the
  // login-time environment block and can miss it (#61494).
  const candidates = windowsGitCandidates(
    {
      localAppData,
      programFiles: process.env['ProgramFiles'] || 'C:\\Program Files',
      programFilesX86: process.env['ProgramFiles(x86)'] || 'C:\\Program Files (x86)'
    },
    { existsSync: fileExists, readdirSync: dir => fs.readdirSync(dir) }
  )

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
    // PATH hits go through the same probe: a bare findOnPath fallback would
    // re-select an unlaunchable first hit when none of the fixed locations exist.
    candidates.push(...findPathCandidates('gh'))
  }

  // Same selection rule as git: an existing-but-unlaunchable candidate (e.g.
  // an Intel-only build from a stale Homebrew) must not shadow a working one
  // further down the list, and PATH is only consulted when none of the
  // explicit candidates is usable.
  const selected = selectRunnableBinary({
    candidates,
    fileExists,
    binaryRuns
  })

  _ghBinaryCache = selected || findOnPath('gh') || 'gh'

  return _ghBinaryCache
}

function recentHermesLog() {
  return hermesLog.slice(-20).join('\n')
}

// ─── Self-update (git-pull against the running backend's hermes root) ──────

function readDesktopUpdateConfig(): { branch: string; branchExplicit: boolean } {
  try {
    const parsed: { branch?: unknown } | null = JSON.parse(fs.readFileSync(DESKTOP_UPDATE_CONFIG_PATH, 'utf8'))
    const branch: string = typeof parsed?.branch === 'string' ? parsed.branch.trim() : ''

    return { branch: branch || DEFAULT_UPDATE_BRANCH, branchExplicit: branch.length > 0 }
  } catch {
    return { branch: DEFAULT_UPDATE_BRANCH, branchExplicit: false }
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
// lands back where the user actually sized the window. While fullscreen,
// getNormalBounds() reports the fullscreen bounds with isMaximized=false — the
// broken transition behind #94319 — so record that provenance and let recovery
// on the next launch recognize the snapshot instead of guessing from geometry.
function persistWindowState() {
  if (!mainWindow || mainWindow.isDestroyed() || mainWindow.isMinimized()) {
    return
  }

  try {
    const { x, y, width, height } = mainWindow.getNormalBounds()
    fs.mkdirSync(path.dirname(DESKTOP_WINDOW_STATE_PATH), { recursive: true })
    writeFileAtomic(
      DESKTOP_WINDOW_STATE_PATH,
      JSON.stringify(
        {
          x,
          y,
          width,
          height,
          isMaximized: mainWindow.isMaximized(),
          boundsCapturedFullScreen: mainWindow.isFullScreen()
        },
        null,
        2
      )
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

  return candidates.find(isGitCheckout) || candidates[0] || ACTIVE_HERMES_ROOT
}

function emitUpdateProgress(payload) {
  const merged = { stage: 'idle', message: '', percent: null, error: null, ...payload, at: Date.now() }
  rememberLog(`[updates] ${merged.stage}: ${merged.message || merged.error || ''}`)
  desktopMetrics.noteUpdateProgress(merged.stage)

  for (const window of BrowserWindow.getAllWindows()) {
    window.webContents.send('hermes:updates:progress', merged)
  }
}

async function checkUpdates(opts: { force?: boolean } = {}): Promise<UpdaterStatusWire> {
  // A packaged install delegates to the update owner named by its stamp.
  let strategy: UpdaterStrategy | null = null

  try {
    strategy = await resolvePackagedUpdateStrategy()

    if (strategy) {
      return await strategy.check(opts)
    }
  } catch (error) {
    return {
      supported: true,
      mechanism: strategy?.mechanism,
      error: 'check-failed',
      message: error instanceof Error ? error.message : String(error),
      fetchedAt: Date.now()
    }
  }

  // Checkout install: dispatch through the strategy layer — one mechanism,
  // one stamp, no direct body path. The flow lives in updater/checkout.ts;
  // this is the only production door to the checkout arms.
  return resolveCheckoutUpdateStrategy().check(opts)
}

let updateInFlight = false

// ── bundled / App Installer helpers ─────────────────────────────────────────

/**
 * Keep the native updater instance alive across check, download and install.
 * Its identity comes from the packaged app, not its optional Python payload.
 */
const updateOperation: UpdateOperation = new UpdateOperation(createPackagedUpdateStrategy)

const desktopMetrics: DesktopSharedMetrics = registerDesktopSharedMetrics()

function resolvePackagedUpdateStrategy(): Promise<UpdaterStrategy | null> {
  return updateOperation.resolve()
}

async function createPackagedUpdateStrategy(): Promise<UpdaterStrategy | null> {
  const mechanism = resolveUpdaterMechanism({
    platform: process.platform,
    updateMechanism: INSTALL_STAMP?.updateMechanism,
    source: INSTALL_STAMP?.source
  })

  if (mechanism === 'windows-handoff' || mechanism === 'posix-handoff') {
    return null
  }

  if (INSTALL_STAMP?.channelBuild && (mechanism === 'electron-updater' || mechanism === 'app-installer')) {
    const build = INSTALL_STAMP.channelBuild
    const installed = await inspectRunningChannelApp(build)

    // Platform/arch/signing are the channel's own preconditions. The Python
    // payload is not: electron-updater swaps the .app without it, and the
    // darwin Light channel build ships none (write-build-stamp.mjs). The
    // strategy that consumes the payload (app-installer) demands it itself.
    if (
      (process.platform !== 'darwin' && process.platform !== 'win32') ||
      (process.arch !== 'arm64' && process.arch !== 'x64')
    ) {
      throw new Error('Channel updates require a supported packaged application')
    }

    return new ChannelStrategy({
      build,
      mechanism,
      resolver: new ChannelResolver({
        build,
        platform: process.platform,
        arch: process.arch,
        signer: installed.signer
      }),
      nativeFactory: (target: ChannelTarget): UpdaterStrategy => createNativePackagedStrategy(mechanism, target)
    })
  }

  return createNativePackagedStrategy(mechanism)
}

function createNativePackagedStrategy(
  mechanism: UpdaterStrategy['mechanism'],
  target?: ChannelTarget
): UpdaterStrategy {
  if (mechanism === 'electron-updater') {
    const deps: Parameters<typeof createMacStrategy>[0] = {
      channel: resolveUpdaterChannelFromStamp(),
      light: isLightVariant(),
      feedBaseUrl: resolveDesktopFeedBaseUrl(),
      appVersion: app.getVersion(),
      log: rememberLog,
      emitProgress: emitUpdateProgress,
      beforeInstall: teardownBundledBackend,
      onInstallFailure: restoreBundledBackend
    }

    return target ? createChannelMacStrategy(deps, target) : createMacStrategy(deps)
  }

  if (mechanism === 'app-installer') {
    const payload: PayloadInfo = requireBundledPayload(mechanism)

    const deps: ConstructorParameters<typeof AppInstallerStrategy>[0] = {
      python: payload.storePython,
      // The checker is bundled core code, run with the payload python.
      module: 'hermes_cli.windows_appinstaller_update',
      run: (python, module) =>
        runAppInstallerChecker(python, module, {
          env: { ...process.env, PYTHONPATH: payloadPythonPath(payload) },
          onStderr: stderr => console.error(`[app-installer] checker stderr: ${stderr.slice(0, 400)}`)
        }),
      channel: resolveUpdaterChannelFromStamp(),
      light: isLightVariant(),
      feedBaseUrl: resolveDesktopFeedBaseUrl(),
      installer: {
        prepare: url => stageAppInstallerFile(url, path.join(app.getPath('userData'), 'updates')),
        open: file => shell.openPath(file)
      },
      teardownBundledBackend,
      restoreBundledBackend,
      emitUpdateProgress,
      appVersion: INSTALL_STAMP?.channelBuild?.windowsVersion ?? app.getVersion(),
      quit: () => app.quit(),
      registerPendingRelaunch: (fromVersion: string): Promise<RelaunchRegistration> =>
        registerUpdateRelaunch(app, fromVersion, {
          // The relaunch mechanism: a detached waiter, external to the dying
          // process. It snapshots the OLD package, signals the handshake,
          // waits for this process to exit and for the OS to swap the
          // installed package version, then activates the NEW package.
          // Electron's app.relaunch would re-execute the OLD package's
          // binary and can hold the swap open, so it is not used. The
          // script is staged to a temp dir and resolved absolutely so
          // nothing inherited from the package holds the swap open.
          relaunch: () =>
            startRelaunchWaiter({
              processId: process.pid,
              processStartTimeMs: Math.round(Date.now() - process.uptime() * 1000),
              identityName: PRODUCT_IDENTITY.msixAppIdWithOrg,
              scriptPath: relaunchWaiterScript(process.resourcesPath)
            })
        })
    }

    return target
      ? createChannelAppInstallerStrategy(deps, target, verifyPreparedChannelInstaller)
      : new AppInstallerStrategy(deps)
  }

  if (mechanism === 'microsoft-store') {
    const payload: PayloadInfo = requireBundledPayload(mechanism)

    return createStoreStrategy({
      python: payload.storePython,
      module: 'hermes_cli.windows_store_update',
      pythonPath: payloadPythonPath(payload),
      env: process.env,
      windowHandle: () => (BrowserWindow.getFocusedWindow() ?? mainWindow)?.getNativeWindowHandle() ?? null,
      appVersion: app.getVersion(),
      teardown: teardownBundledBackend,
      restore: restoreBundledBackend,
      emitProgress: emitUpdateProgress,
      quit: () => app.quit(),
      registerPendingRelaunch: (fromVersion: string): Promise<RelaunchRegistration> =>
        registerUpdateRelaunch(app, fromVersion, {
          relaunch: () =>
            startRelaunchWaiter({
              processId: process.pid,
              processStartTimeMs: Math.round(Date.now() - process.uptime() * 1000),
              identityName: PRODUCT_IDENTITY.storeMsix!.identityName,
              scriptPath: relaunchWaiterScript(process.resourcesPath),
              timeoutSeconds: 1860
            })
        })
    })
  }

  return new ExternalStrategy(INSTALL_STAMP)
}

/**
 * The bundled Python payload for a strategy whose update check runs inside
 * it. A stamp that names such a mechanism without a payload is a broken
 * install; say so instead of failing on `payload.storePython`.
 */
function requireBundledPayload(mechanism: UpdaterStrategy['mechanism']): PayloadInfo {
  const payload: PayloadInfo | null = bundledPayload(process.resourcesPath)

  if (!payload) {
    throw new Error(`${mechanism} updates require the bundled application payload, which this install has none of`)
  }

  return payload
}

/**
 * The checkout updater strategy (windows-handoff / posix-handoff). The flow
 * lives in updater/checkout.ts; this is the composition root that hands it
 * the app shell's impure edges. The public checkUpdates/applyUpdates
 * entrypoints dispatch only through this strategy — there is no other
 * production path to the checkout arms.
 */
function resolveCheckoutUpdateStrategy(): UpdaterStrategy {
  return createCheckoutStrategy({
    hermesHome: HERMES_HOME,
    isWindows: IS_WINDOWS,
    isMac: IS_MAC,
    defaultUpdateBranch: DEFAULT_UPDATE_BRANCH,
    updateHandoffDwellMs: UPDATE_HANDOFF_DWELL_MS,
    readSourceUpdate: async (updateRoot: string, opts: { force?: boolean }): Promise<SourceUpdate | null> =>
      readSourceUpdate({
        python: await findPythonForRoot(updateRoot),
        git: resolveGitBinary(),
        updateRoot,
        hermesHome: HERMES_HOME,
        branchConfigPath: DESKTOP_UPDATE_CONFIG_PATH,
        force: opts.force
      }),
    resolveUpdateRoot,
    resolveUpdaterBinary,
    remoteGatewayActive: globalRemoteActive,

    emitUpdateProgress,
    rememberLog,
    startHermes,
    stopBackendsForUpdate,
    repairMacUpdaterHelper,
    preflightStateDb: async (home: string, log: (message: string) => void): Promise<void> => {
      const root: string = resolveUpdateRoot()

      // `updates.pre_update_backup: off` is the CLI's opt-out for the whole
      // pre-update backup family; the Desktop's emergency snapshot honours it
      // too (4de06d1dbf7b). An unreadable answer keeps the snapshot.
      if (
        !(await readPreUpdateBackupEnabled(
          resolveHermesBackend(['config', 'get', 'updates.pre_update_backup', '--json']),
          home
        ))
      ) {
        log('[updates] emergency state.db backup disabled by updates.pre_update_backup')

        return
      }

      // PM-managed checkouts carry no venv of their own: the installation
      // launcher owns interpreter and generation selection there — same
      // contract as readSourceUpdate and the hand-off script.
      const managed: boolean = directoryExists(path.join(root, 'pm'))

      const launcher: string | null = managed ? resolveInstallationLauncher(root, IS_WINDOWS, HERMES_HOME) : null

      if (managed && !launcher) {
        const message =
          `state.db pre-flight failed: the installation launcher under ${root} is missing. ` +
          'Update cancelled before backend shutdown. Repair this installation before retrying.'

        log(`[updates] ${message}`)
        throw new Error(message)
      }

      preflightStateDb({
        python: managed ? null : await findPythonForRoot(root),
        launcher,
        script: path.join(root, 'hermes_cli', 'backup_sqlite.py'),
        home,
        log
      })
    },
    runningAppBundle,
    markQuittingForHandoff: () => {
      isQuittingForHandoff = true
    },
    quit: () => app.quit()
  })
}

/**
 * The App Installer feed base URL for a bundled MSIX install: config.yaml's
 * `updates.desktop_feed_base_url`, else Windows' registered App Installer
 * source.
 */
function resolveDesktopFeedBaseUrl(): string {
  return resolveFeedBaseUrl(readUpdatesFeedBaseFromConfig(path.join(HERMES_HOME, 'config.yaml')))
}

/** The updater channel from the baked install stamp ('canary' vs 'stable'). */
function resolveUpdaterChannelFromStamp(): string {
  return packagedReleaseChannel(INSTALL_STAMP) ?? 'stable'
}

/** True when this artifact is the light (remote-only) variant. */
function isLightVariant(): boolean {
  return INSTALL_STAMP?.payload === 'light'
}

/** Invalidate connections and wait for every owned backend before the swap. */
async function teardownBundledBackend(): Promise<void> {
  isQuittingForHandoff = true

  const results = await Promise.allSettled([
    teardownPrimaryBackendAndWait(backendTeardownOptions('reconnect')),
    stopAllPoolBackends()
  ])

  const errors = results.filter(result => result.status === 'rejected').map(result => result.reason)

  if (errors.length) {
    // An incomplete shutdown blocks replacement until a retry proves exit.
    backendStartFailure = new AggregateError(errors, 'Backend shutdown failed')
    throw backendStartFailure
  }
}

async function restoreBundledBackend(): Promise<void> {
  try {
    // A failed stop retains its process handle. Retry before enabling a new start.
    await teardownBundledBackend()
  } finally {
    isQuittingForHandoff = false
    updateInFlight = false
  }

  backendStartFailure = null
  await startHermes()
}

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
  const venvDir = resolveVenvDir(updateRoot)

  return IS_WINDOWS ? path.join(venvDir, 'Scripts', 'hermes.exe') : path.join(venvDir, 'bin', 'hermes')
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

// Kill only Hermes-OWNED venv daemons (the memory plugin's hindsight daemon:
// exe under venv\Scripts AND cmdline referencing hindsight_api.main). The
// daemon is spawned DETACHED, so it outlives the backend tree-kill and keeps
// venv files mapped. External holders (a user terminal running `hermes`,
// unrelated scripts) are NOT killed. The uninstall lock probe refuses a
// held installation. Selection lives in the pure
// venv-holder-select module (ordinal path-prefix, no PowerShell -like
// wildcard hazards) so it's testable without Electron.
function scanWindowsProcesses(): Array<{ ProcessId?: unknown; ExecutablePath?: string; CommandLine?: string }> {
  const out = execFileSync(
    'powershell',
    [
      '-NoProfile',
      '-Command',
      'Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.CommandLine } | Select-Object ProcessId, ExecutablePath, CommandLine | ConvertTo-Json -Compress'
    ],
    hiddenWindowsChildOptions({ encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'], timeout: 15_000 })
  )

  const parsed = JSON.parse(String(out || '[]'))

  return Array.isArray(parsed) ? parsed : [parsed]
}

function killHermesOwnedVenvDaemons(updateRoot) {
  if (!IS_WINDOWS) {
    return
  }

  const scriptsDir = path.join(resolveVenvDir(updateRoot), 'Scripts')

  let holders = []

  try {
    holders = scanWindowsProcesses().filter(p => isHermesOwnedVenvDaemon(p?.ExecutablePath, p?.CommandLine, scriptsDir))
  } catch {
    // Best-effort: the uninstall lock probe remains the backstop.
    return
  }

  for (const holder of holders) {
    const pid = Number(holder?.ProcessId)

    if (Number.isInteger(pid) && pid > 0) {
      rememberLog(`[updates] stopping Hermes-owned venv daemon (hindsight) PID ${pid} before hand-off`)

      try {
        forceKillProcessTree(pid)
      } catch (error) {
        // Update hand-off only. Close/stop must not swallow this; see
        // windowsCloseStopOwnedBackends.
        rememberLog(`[updates] taskkill PID ${pid} failed: ${error instanceof Error ? error.message : String(error)}`)
      }
    }
  }
}

// Kill EXTERNAL Hermes processes that hold this install's venv shim (#62311):
// the gateway Startup item and dashboard Scheduled Task are launched outside
// this app (Task Scheduler / autostart), so the backend teardown above never
// sees them — yet they map venv files and made every update hand-off abort
// with "venv shim still locked". Selection is deliberately narrow
// (isExternalVenvHolder: exe under venv\Scripts AND unambiguously a Hermes
// program) — unrelated processes that merely mention the install root or use
// the venv interpreter for their own scripts are never killed; the shim-lock
// probe still aborts the hand-off for those. Called before the release gate
// AND inside each gate pass, so a respawning autostart holder is re-killed
// instead of winning the 15s race.
function killExternalVenvHolders(updateRoot) {
  if (!IS_WINDOWS) {
    return
  }

  const scriptsDir = path.join(resolveVenvDir(updateRoot), 'Scripts')

  let holders = []

  try {
    holders = scanWindowsProcesses().filter(p => isExternalVenvHolder(p?.ExecutablePath, p?.CommandLine, scriptsDir))
  } catch {
    // Best-effort: the shim-lock probe remains the backstop.
    return
  }

  for (const holder of holders) {
    const pid = Number(holder?.ProcessId)

    if (Number.isInteger(pid) && pid > 0) {
      rememberLog(
        `[updates] stopping external Hermes venv holder (autostart gateway/dashboard) PID ${pid} before hand-off`
      )

      try {
        forceKillProcessTree(pid)
      } catch (error) {
        rememberLog(`[updates] taskkill PID ${pid} failed: ${error instanceof Error ? error.message : String(error)}`)
      }
    }
  }
}

// Force-kill the entire process TREE rooted at each PID. Node's child.kill()
// only signals the direct child, so on Windows a backend `hermes.exe` that
// spawned its own grandchildren (a `hermes` REPL, a pty terminal session, the
// gateway) would survive and keep the venv shim locked. taskkill /T /F reaps
// the whole tree synchronously. The command is not widened: one owned PID,
// /T /F, nothing else. Failures propagate — close/stop must not discard them.
// Windows-only: this is called solely from the Windows shim-unlock and
// close/stop paths, and the backend is NOT spawned detached (so it's not a
// process-group leader — a POSIX negative-pgid kill would be meaningless
// here anyway). POSIX teardown stays with the existing before-quit SIGTERM.
function forceKillProcessTree(pid) {
  if (!IS_WINDOWS) {
    return
  }

  if (!Number.isInteger(pid) || pid <= 0) {
    return
  }

  execFileSync('taskkill', ['/PID', String(pid), '/T', '/F'], hiddenWindowsChildOptions({ stdio: 'ignore' }))
}

function holderPidsFromLockFile(lockPath: string): Pick<RuntimeLock, 'holderPids' | 'held'> {
  try {
    const raw = fs.readFileSync(lockPath, 'utf8').trim()

    if (!raw) {
      return { holderPids: [] }
    }

    let pid = Number(raw)

    if (!Number.isInteger(pid)) {
      const parsed = JSON.parse(raw) as { pid?: unknown }
      pid = Number(typeof parsed === 'object' && parsed ? parsed.pid : parsed)
    }

    return Number.isInteger(pid) && pid > 0 ? { holderPids: [pid] } : { holderPids: [] }
  } catch (error) {
    const code = (error as NodeJS.ErrnoException | null)?.code

    // Can't read the file: a live holder may be why. Do not clear it.
    if (code === 'EBUSY' || code === 'EPERM' || code === 'EACCES') {
      return { holderPids: [], held: true }
    }

    return { holderPids: [] }
  }
}

function collectCloseStopLocks(): RuntimeLock[] {
  const roots = [HERMES_HOME]
  const profilesRoot = path.join(HERMES_HOME, 'profiles')

  try {
    for (const name of fs.readdirSync(profilesRoot)) {
      roots.push(path.join(profilesRoot, name))
    }
  } catch {
    // No profiles directory — the default home lock is enough.
  }

  const locks: RuntimeLock[] = []

  for (const root of roots) {
    const lockPath = path.join(root, 'gateway.lock')

    try {
      if (!fs.statSync(lockPath).isFile()) {
        continue
      }
    } catch {
      continue
    }

    locks.push({ path: lockPath, ...holderPidsFromLockFile(lockPath) })
  }

  return locks
}

// Captured before teardown drops the handles. Node keeps each process handle
// open until exit is observed, so a PID read from a still-running child here
// cannot have been reused by an unrelated process.
function collectOwnedBackendChildren(): ChildProcess[] {
  const children = [backendConnectionState.getProcess(), ...[...backendPool.values()].map(entry => entry?.process)]

  return children.filter(
    (child): child is ChildProcess => Boolean(child) && Number.isInteger(child.pid) && child.pid > 0
  )
}

// Close/stop, after the graceful teardown, pool stop and straggler reap: the
// same tree-kill for any owned child that is still running, an inventory of
// those PIDs, and a clear of only the locks no live holder owns. Never throws;
// returns the failure for the caller to surface once cleanup is done.
function windowsCloseStopOwnedBackends(children: ChildProcess[]): Error | null {
  if (!IS_WINDOWS) {
    return null
  }

  try {
    const running = children.filter(child => child.exitCode === null && child.signalCode === null)

    const result = finishWindowsCloseStop(
      running.map(child => child.pid as number),
      collectCloseStopLocks(),
      {
        killTree: forceKillProcessTree,
        isPidAlive: isPidAliveWindows,
        clearLock: lockPath => {
          fs.rmSync(lockPath, { force: true })
        }
      }
    )

    for (const failure of result.taskkillFailures) {
      rememberLog(`[close-stop] taskkill PID ${failure.pid} failed: ${failure.error}`)
    }

    for (const failure of result.lockErrors) {
      rememberLog(`[close-stop] could not clear unheld lock ${failure.path}: ${failure.error}`)
    }

    if (result.clearedLocks.length) {
      rememberLog(`[close-stop] cleared unheld lock(s): ${result.clearedLocks.join(', ')}`)
    }

    return result.liveFailure ? new Error(closeStopFailureMessage(result)) : null
  } catch (error) {
    return error instanceof Error ? error : new Error(String(error))
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

async function processIdentityMatches(identity, timeoutMs: number = 30_000) {
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
    return (await processStartMarker(identity.pid, timeoutMs)) === identity.startMarker
  } catch (error) {
    return error?.code === 'ENOENT' || error?.code === 'ESRCH' ? false : undefined
  }
}

async function backendIdentityMatches(identity) {
  const processMatches = await processIdentityMatches(identity, REAP_PROBE_TIMEOUT_MS)

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
    return (await processStartMarker(entry.parentPid, REAP_PROBE_TIMEOUT_MS)) === entry.parentStartMarker
  } catch (error) {
    return error?.code === 'ENOENT' || error?.code === 'ESRCH' ? false : undefined
  }
}

async function stopOwnedBackend(identity) {
  const matches = await processIdentityMatches(identity, REAP_PROBE_TIMEOUT_MS)

  if (matches === false) {
    return
  }

  if (matches !== true) {
    // Identity probe failed (not confirmed gone): preserve the record so a
    // later launch retries the stop instead of dropping it and leaking the
    // backend. reapOrphans keeps the entry when stop() throws.
    throw new Error(`Could not verify backend PID ${identity.pid} before stopping it.`)
  }

  if (IS_WINDOWS) {
    try {
      forceKillProcessTree(identity.pid)
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error)
      const stillThere = await processIdentityMatches(identity, REAP_PROBE_TIMEOUT_MS)

      if (stillThere !== false) {
        throw new Error(`taskkill failed for backend PID ${identity.pid}: ${detail}`)
      }

      rememberLog(`taskkill reported failure for already-gone backend PID ${identity.pid}: ${detail}`)
    }
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
      if ((await processIdentityMatches(identity, REAP_PROBE_TIMEOUT_MS)) !== true) {
        return
      }

      await new Promise(resolve => setTimeout(resolve, 50))
    }

    // Revalidate immediately before escalation so PID reuse cannot target a
    // replacement process.
    if ((await processIdentityMatches(identity, REAP_PROBE_TIMEOUT_MS)) === true) {
      try {
        process.kill(-identity.pid, 'SIGKILL')
      } catch {
        process.kill(identity.pid, 'SIGKILL')
      }
    }
  }

  await new Promise(resolve => setTimeout(resolve, 50))
  const remaining = await processIdentityMatches(identity, REAP_PROBE_TIMEOUT_MS)

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

async function claimBackendChild(
  child: ChildProcess & { hermesBackendIdentity?: BackendOwnershipEntry },
  command: string,
  profile: string,
  nonce: string,
  outputTail: BackendOutputTail | null = null
): Promise<BackendOwnershipEntry> {
  // Probe/claim policy lives in backend-claim.ts (#93608): a marker probe
  // that fails against a LIVE child degrades to PID-only identity — matching
  // createParentStartMarkerResolver — instead of killing a healthy backend
  // over a flaky Get-Process (PS 5.1 cold starts, #87169). Only a child that
  // actually died keeps the fail-closed throw, now carrying its stderr tail.
  const probe = await probeStartMarker(child.pid)
  const decision = claimDecision(child.exitCode === null && !child.killed, probe)

  if (decision.action === 'fail') {
    await localBackendLifecycle.stop(child)
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
    await localBackendLifecycle.stop(child)
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

// Stop app-owned Windows backends before replacing application outputs.
// PM generations can retain live readers. Gateway draining/restart belongs to
// `hermes update`; neither venv scans nor a second fleet stop belong here.
async function stopBackendsForUpdate(): Promise<void> {
  if (IS_WINDOWS) {
    await Promise.all([teardownPrimaryBackendAndWait(backendTeardownOptions('reconnect')), stopAllPoolBackends()])
  }
}

// Uninstall still deletes the installation and its historical venv. Unlike
// generation updates, deletion must wait for those old files to be released.
async function releaseBackendLock(updateRoot: string, tag: string): Promise<{ unlocked: boolean }> {
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

  // No backend comes back after an uninstall: stay silent, like a quit.
  await Promise.all([teardownPrimaryBackendAndWait(backendTeardownOptions('quit')), stopAllPoolBackends()])

  // Uninstall deletes the whole runtime. Drain separately-running gateways
  // through the CLI, rather than targeting a gateway worker by PID.
  stopGatewayBeforeUpdate(venvHermesShimPath(updateRoot), HERMES_HOME)

  // Reap Hermes-OWNED venv daemons the tree-kill above cannot reach: the
  // memory plugin's hindsight daemon is spawned DETACHED (it outlives the
  // backend) yet runs off venv\Scripts\pythonw.exe, keeping venv files
  // mapped past the backend teardown (#75477/#75478). Narrowly scoped
  // (venv-holder-select) — external holders are never killed here.
  killHermesOwnedVenvDaemons(updateRoot)

  // External autostart Hermes processes (gateway Startup item, dashboard
  // Scheduled Task) also hold the venv shim and are invisible to the backend
  // teardown (#62311). Kill them before the gate AND re-scan inside each gate
  // pass, so a respawning holder loses the race instead of the update.
  killExternalVenvHolders(updateRoot)

  const shim = venvHermesShimPath(updateRoot)

  const gate = await waitForBackendRelease(
    initialPids,
    {
      isShimLocked: () => Boolean(isShimLocked(shim)),
      isPidAlive: isPidAliveWindows,
      collectStragglerPids: () => {
        // Re-kill resurgent external holders (autostart gateway/dashboard) on
        // every pass — #62311 — before collecting the desktop-owned stragglers.
        killExternalVenvHolders(updateRoot)
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
      killProcessTree: pid => {
        try {
          forceKillProcessTree(pid)
        } catch (error) {
          rememberLog(`[${tag}] taskkill PID ${pid} failed: ${error instanceof Error ? error.message : String(error)}`)
        }
      },
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
async function applyUpdates(): Promise<UpdaterApplyResultWire> {
  return updateOperation.apply(async (): Promise<UpdaterApplyResultWire> => {
    updateInFlight = true
    let handedOff: boolean = false

    try {
      // The local handoff asks the window to exit and the update scripts only
      // wait so long for that PID — never start that deadline while quit would
      // still be gated on a managed SSH update or its recovery transaction
      // (before-quit joins the same operations; the updater must not race them).
      await waitForManagedUpdateOperations(() => [
        ...managedConnectionUpdates.values(),
        ...managedConnectionRecoveries.values()
      ])

      const packaged: UpdaterStrategy | null = await resolvePackagedUpdateStrategy()
      const strategy: UpdaterStrategy = packaged ?? resolveCheckoutUpdateStrategy()
      const result: UpdaterApplyResultWire = await desktopMetrics.trackUpdateApply(packaged, strategy)
      handedOff = result.handedOff === true

      return result
    } finally {
      if (!handedOff) {
        updateInFlight = false
      }
    }
  })
}

async function handOffWindowsBootstrapRecovery(reason) {
  if (!IS_WINDOWS || !IS_PACKAGED) {
    return false
  }

  // A bundled install does not own %LOCALAPPDATA%\hermes — the updater
  // would try to heal a tree the app never created. The payload IS the
  // runtime; recovery means reinstalling the app, not spawning the
  // updater. (ensureRuntime's bundled guard also short-circuits before
  // this call; this is the belt-and-suspenders check.)
  if (installShape() === 'bundled') {
    rememberLog('[bootstrap] refusing updater recovery hand-off on a bundled install; reinstall the app')

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

  // Recovery can run without Python. Keep the chosen branch; do not guess a replacement.
  const branch: string = configuredBranch || DEFAULT_UPDATE_BRANCH

  const updaterArgs: string[] = chooseUpdaterArgs({ runtimeUsable: await isSourceRuntimeUsable(updateRoot) }, branch)

  await stopBackendsForUpdate()

  const child = spawnUpdaterProcess(updater, updaterArgs, {
    cwd: HERMES_HOME,
    env: {
      ...process.env,
      HERMES_HOME,
      HERMES_INSTALL_ROOT: updateRoot
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

// macOS/Linux update hand-off: spawn the repo-owned posix orchestrator
// (scripts/desktop-update/posix.sh) detached and QUIT. The script waits us
// out, runs `hermes update`, swaps/relaunches the app bundle, and writes
// .hermes-update-result.json for the relaunched Desktop to surface. It shows
// its own tiny shim window (or nothing, headless) — this process only needs
// to leave. Checkouts that predate the script get the manual card once.
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
async function isSourceRuntimeUsable(root: string): Promise<boolean> {
  return (await resolveSourceInstallationBackend(root, [], { hermesHome: HERMES_HOME })) !== null
}

function isActiveRuntimeUsable(): Promise<boolean> {
  return isSourceRuntimeUsable(ACTIVE_HERMES_ROOT)
}

function activeRuntimeState(backend: SourceBackend | null): ActiveRuntimeState {
  // We DELIBERATELY do NOT verify that the checkout is currently at the
  // pinned commit -- users update via the in-app update path or `hermes
  // update`, which moves HEAD legitimately. The marker only attests "a
  // desktop-managed bootstrap ran here at least once"; runtime usability is
  // what decides whether we can actually launch.
  const state: ActiveRuntimeState = classifyActiveRuntime(
    readBootstrapMarker(),
    BOOTSTRAP_MARKER_SCHEMA_VERSION,
    backend !== null
  )

  // The canonical install stamp (written next to the runtime by the bootstrap)
  // tells the UI where this runtime came from. Prefer it over the marker so
  // the Runtime row reflects the actual install provenance.
  state.canonicalInstallStamp = readCanonicalInstallStamp()

  return state
}

/** Read the checkout-owned install stamp in the canonical runtime root
 *  (`ACTIVE_HERMES_ROOT/install-stamp.json`), written by the Python
 *  completion tail. Returns null when absent or unreadable. */
function readCanonicalInstallStamp() {
  try {
    const raw = fs.readFileSync(path.join(ACTIVE_HERMES_ROOT, 'install-stamp.json'), 'utf8')
    const parsed = JSON.parse(raw)

    if (parsed && typeof parsed === 'object' && typeof parsed.source === 'string') {
      return parsed
    }

    return null
  } catch {
    return null
  }
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

  // The checkout's own install stamp is written by the Python completion tail
  // (hermes_cli/source_completion.py) during the products stage, from the
  // checkout itself. The desktop never synthesizes it: the checkout is the
  // authority for its runtime identity, and a desktop-written copy would
  // clobber the completion tail's richer provenance.

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

  // existsSync, not directoryExists: a stat inside app.asar constructs the deprecated fs.Stats (#96857).
  if (IS_PACKAGED && /app\.asar(?=$|[\\/])/.test(fallback) && !fs.existsSync(fallback)) {
    rememberLog(
      `[web-dist] dashboard frontend dir resolved to an asar-internal path that ` +
        `is not a real directory: ${fallback}. Static routes will 404. ` +
        `Ensure dist/** is unpacked (asarUnpack) or set HERMES_DESKTOP_WEB_DIST.`
    )
  }

  return fallback
}

// Same resolution as resolveRendererIndex, but also hands back the missing
// asset list already computed for the copy it chose. The primary-window path
// needs BOTH, and re-deriving the list means walking the whole renderer
// generation a second time: missingRendererAssets follows index.html's
// modulepreload refs and then every chunk's inline __vite__mapDeps table, so
// on a release build it reads ~28 MiB across ~160 files synchronously on the
// main thread — measured ~49 ms per walk, twice before loadWindowUrl().
// Callers that only need the path keep using resolveRendererIndex below.
function resolveRendererIndexWithMissing(): { index: string; missing: string[] } {
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
  const present = presentRendererIndexes(candidates)

  // index.html and the hashed chunks it names are one generation. An update
  // that replaces only one of the two shipped copies (app.asar vs
  // app.asar.unpacked) leaves a TORN copy: the window loads, then dies on the
  // first lazy import with "Failed to fetch dynamically imported module" and
  // every restart reloads the same torn copy. Prefer a copy whose modules are
  // all present, so the intact generation heals the boot by itself.
  // Remember the FIRST candidate's list: if every copy turns out to be torn we
  // load present[0], and its list is already in hand — recomputing it there
  // would reintroduce the very second walk this function exists to avoid.
  let firstMissing: string[] | null = null

  for (const candidate of present) {
    const missing = missingRendererAssets(candidate)

    if (missing.l…88688 tokens truncated…ease the global snap shortcut, put the app back so the user is never
    // left with no surface, and correct every window's toggle.
    hudSnapShortcut.dispose()
    restoreMainWindowFromHud()
    broadcastHudState(false)
  })

  attachRendererConsoleCapture(win, 'hud', rememberLog)
  // Log-only lifecycle (#81290): the HUD is a compact auxiliary surface the
  // user can re-toggle; a dead renderer should be diagnosable, not resurrected.
  installWindowRendererLifecycle(win, { kind: 'hud', callbacks: { log: rememberLog } })
  // Same timing as a session window: the profile query is known now, while the
  // renderer cannot announce its route until after preload has already run.
  recordWindowConnectionRoute(win.webContents, {
    connectionId: null,
    profile: localSkinProfileKey(profile ?? primaryProfileKey()),
    registryScoped: false
  })
  loadWindowUrl(win, hudUrl(sessionId, profile), 'HUD')

  return win
}

// Put the app window back, and give it the keyboard. `focusWindow`, not a bare
// `show()`: show() alone leaves a minimized window minimized, and on macOS a
// shown-but-not-key window means the user is looking at the app with the
// caret still belonging to whatever the HUD was floating over.
function restoreMainWindowFromHud() {
  if (!hudRestoreMainWindow) {
    return
  }

  hudRestoreMainWindow = false
  focusWindow(mainWindow)
}

// Take the HUD window down. The 'closed' handler stays attached so ONE path
// owns the teardown (snap shortcut, main-window restore, close broadcast)
// whether the window went via the exit button, ⌘W, a profile respawn, or the
// grace deadline — detaching it before close() was how a renderer that never
// answered the close left an always-on-top HUD nobody could dismiss and no
// broadcast to correct the toggles.
function destroyHudWindow(win: BrowserWindow) {
  if (hudWindow === win) {
    hudWindow = null
  }

  requestHudClose(win)
}

function openHudWindow(sessionId, profile) {
  const profileKey = typeof profile === 'string' && profile.trim() ? profile.trim() : null

  if (hudWindow && !hudWindow.isDestroyed()) {
    // Pointed at another PROFILE: the live renderer is bound to the old
    // profile's backend, and a renderer adopts its backend exactly once at
    // boot — an in-place goto would resolve the id against the wrong backend
    // (the #82285 fallback). Respawn against the right one. The old window's
    // 'closed' handler sees `hudWindow` already pointing at the replacement,
    // so it neither restores main nor broadcasts a false "closed".
    if (profileKey && hudProfile !== profileKey) {
      const previous = hudWindow

      hudSessionId = sessionId || null
      hudProfile = profileKey
      hudWindow = spawnHudWindow(sessionId, profileKey)
      previous.destroy()
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

  hudRestoreMainWindow = Boolean(mainWindow && !mainWindow.isDestroyed())
  hudSessionId = sessionId || null
  hudProfile = profileKey
  hudWindow = spawnHudWindow(sessionId, profileKey)
  broadcastHudState(true)
  registerHudSnapShortcut()

  return hudWindow
}

function closeHudWindow() {
  const win = hudWindow

  if (win && !win.isDestroyed()) {
    destroyHudWindow(win)

    return
  }

  // No live HUD (a renderer that died, a toggle racing the close): still
  // release what an open HUD holds, so the toggles read right.
  hudWindow = null
  hudSnapShortcut.dispose()
  restoreMainWindowFromHud()
  broadcastHudState(false)
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
    // macOS derives a transparent window's native shadow from its alpha
    // content, but the boot HTML paints an OPAQUE background before the
    // renderer forces transparency (quick-entry-root.tsx) — the OS then
    // caches a full-frame shadow that renders as a stray detached blur blob
    // behind the card (#99172). The card draws its own CSS box-shadow, so the
    // native one only double-paints; the other transparent overlays (pet,
    // HUD) already run shadowless. Other platforms keep it.
    hasShadow: !IS_MAC,
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
    ...computeWindowOptions(
      savedWindowState ?? firstLaunchSize(screen.getPrimaryDisplay().workArea),
      screen.getAllDisplays()
    ),
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
  minimizeToTray.registerWindow(createdMainWindow, { closeToTray: true })
  registerChatWindow(createdMainWindow)
  const defaultRoute = desktopProfilePreferences.getDefault()

  if (defaultRoute) {
    recordWindowConnectionRoute(mainWindow.webContents, {
      ...defaultRoute,
      registryScoped: defaultRoute.connectionId !== null
    })
  }

  // Chat-surface registration: see applyWindowTranslucency.
  translucencyBackedWindows.add(mainWindow)

  if (IS_MAC) {
    mainWindow.setWindowButtonPosition?.(WINDOW_BUTTON_POSITION)

    // Packaged builds keep the bundle icon so macOS can style it (#73195).
    if (icon && shouldOverrideDockIcon({ platform: process.platform, isPackaged: app.isPackaged })) {
      // The window icon is full-bleed for Linux; the Dock wants the mac grid.
      app.dock?.setIcon(resolveAppIcon([path.join(APP_ROOT, 'assets', 'icon-mac.png')]) ?? icon)
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

      // #111906: the Linux launcher holds back its .desktop entry write until the
      // window is on screen (a STARTING gnome-shell app must not see its entry change).
      notifyLauncherWindowRevealed()

      // #124255: the first revealed window means the GPU survived this boot.
      // Keep a sticky SwiftShader marker when we launched with the fallback;
      // otherwise mark the probe healthy so future launches trust hardware GL.
      if (NVIDIA_DRIVER_MAJOR !== null) {
        try {
          writeNvidiaEglMarker(
            app.getPath('userData'),
            nvidiaEglMarkerAfterSuccessfulBoot({
              fallbackActive: nvidiaEglFallbackActive,
              appVersion: app.getVersion(),
              driverVersion: NVIDIA_DRIVER_VERSION
            })
          )
        } catch {
          void 0
        }
      }

      // #38216/#121954: clear the mid-boot marker only after a window is
      // actually usable. Keep sticky `fallback` when we launched with
      // --no-sandbox so the next launcher click does not re-enter the GPU
      // FATAL crash loop. The marker records the app version so the next
      // update re-probes the sandbox.
      if (IS_WINDOWS || process.platform === 'linux') {
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

        try {
          writeGpuStackCookieMarker(
            app.getPath('userData'),
            markerAfterSuccessfulGpuStackCookieBoot({
              fallbackActive: windowsGpuStackCookieFallbackSticky,
              appVersion: app.getVersion()
            })
          )
        } catch (error) {
          rememberLog(`[gpu] stack-cookie marker update after main-window reveal failed: ${error?.message || error}`)
        }
      }

      // #124843: clear the mid-boot marker only after a window is actually
      // usable. Keep sticky `fallback` when we launched with software
      // rendering so the next launch skips the GPU-child retry loop.
      if (process.platform === 'linux') {
        try {
          writeLinuxGpuMarker(
            app.getPath('userData'),
            linuxGpuMarkerAfterSuccessfulBoot({
              fallbackActive: linuxGpuFallbackSticky,
              appVersion: app.getVersion()
            })
          )
        } catch (error) {
          rememberLog(`[gpu] linux marker update after main-window reveal failed: ${error?.message || error}`)
        }

        // #124843 silent-retry manifestation: the boot "succeeded" (window
        // revealed, marker ok) while a sub-zygote retries GPU init forever —
        // no child-process-gone event ever fires, so the reactive ladder
        // above never engages. After a grace window, a boot that should have
        // a GPU child but has none is that retry loop: engage the sticky
        // software fallback so the NEXT launch skips it (this boot's switches
        // already applied pre-ready and cannot change now).
        if (!linuxGpuFallbackSticky) {
          const checkSilentGpuRetry = (): void => {
            const alreadySoftware =
              LINUX_GPU_SOFTWARE_ACTIVE ||
              linuxGpuFallbackActive ||
              alreadyHasDisableGpu(process.argv, process.env) ||
              isHermesDesktopGpuOverrideOff(process.env)

            const gpuChildPresent = app
              .getAppMetrics()
              .some(metric => String(metric?.type || '').toLowerCase() === 'gpu')

            if (
              shouldEngageSilentGpuRetryFallback({
                gpuChildPresent,
                graceElapsed: true,
                alreadySoftware
              })
            ) {
              linuxGpuFallbackActive = true
              linuxGpuFallbackSticky = true

              try {
                writeLinuxGpuMarker(
                  app.getPath('userData'),
                  linuxGpuFallbackMarker('gpu-launch-failure', app.getVersion())
                )
              } catch {
                void 0
              }

              console.warn(
                '[hermes] Linux: no GPU child after window reveal — GPU init is retrying silently; software fallback engaged for the next launch (#124843)'
              )
            }
          }

          setTimeout(checkSilentGpuRetry, LINUX_GPU_SILENT_RETRY_GRACE_S).unref()
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

  bindWindowChromeEvents(mainWindow, sendWindowStateChanged)

  // Reopen where the user left off. close is the backstop, flushed
  // synchronously before the window is gone.
  bindGeometryPersistence(mainWindow, schedulePersistWindowState)
  mainWindow.on('maximize', schedulePersistWindowState)
  mainWindow.on('unmaximize', schedulePersistWindowState)
  mainWindow.on('close', (event: Electron.Event) => {
    schedulePersistWindowState.flush()

    // A prevented close (tray absorb, active-work "Keep Running") leaves the
    // window alive, so the app is not quitting: keep the latch clear so a
    // later close still quits cleanly (#130810).
    if (event.defaultPrevented) {
      return
    }

    // On Windows/Linux, closing the primary window IS quitting (the
    // window-all-closed handler calls app.quit()). Latch the quit flag here,
    // before 'closed' fires closePetOverlay() — otherwise the overlay's
    // 'closed' handler echoes pop-in and wipes the persisted popped-out state
    // the next boot needs (#55920).
    if (!IS_MAC) {
      appQuitting = true
    }
  })

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
  // policy (crashed/oom/killed → bounded reload via the shared rolling budget, then
  // the #38216 Windows sandbox relaunch check on suppression) is the same
  // policy this window used before it moved into the shared helper, so a
  // crashed peer renderer now logs and recovers exactly like the primary one.
  const mainContentsId = mainWindow.webContents.id
  installWindowRendererLifecycle(mainWindow, {
    kind: 'main',
    callbacks: {
      log: rememberLog,
      reload: () => {
        mainWindow.webContents.reload()
      },
      onCrashLoopSuppressed: details => {
        // #108047: STATUS_STACK_BUFFER_OVERRUN crash loops get a one-shot GPU
        // disable relaunch. Checked BEFORE the sandbox path so 0xC0000409 never
        // piggybacks --no-sandbox. If GPU fallback cannot run, surface the
        // visible error page instead of leaving a blank window.
        const stackCookieCrashLoop = {
          reason: details?.reason,
          exitCode: details?.exitCode,
          alreadyGpuDisabled:
            Boolean(REMOTE_DISPLAY_REASON) ||
            windowsGpuStackCookieFallbackActive ||
            alreadyHasDisableGpu(process.argv, process.env),
          relaunchAttempted: windowsGpuStackCookieRelaunchAttempted,
          gpuOverrideOff: isHermesDesktopGpuOverrideOff(process.env)
        }

        if (shouldRelaunchForRendererStackCookieCrashLoop(stackCookieCrashLoop)) {
          windowsGpuStackCookieRelaunchAttempted = true
          windowsGpuStackCookieFallbackActive = true
          windowsGpuStackCookieFallbackSticky = true

          try {
            writeGpuStackCookieMarker(
              app.getPath('userData'),
              gpuStackCookieFallbackMarker('renderer-crash-loop', app.getVersion())
            )
          } catch {
            void 0
          }

          rememberLog(
            '[renderer] Windows stack-cookie crash loop (0xC0000409); relaunching once with GPU disabled (#108047)'
          )

          try {
            app.relaunch({ args: buildDisableGpuRelaunchArgs(process.argv.slice(1)) })
            void exitAfterBackendShutdown(0)
          } catch (err) {
            rememberLog(`[renderer] GPU-disable relaunch failed: ${err?.message || err}`)
          }

          return
        }

        if (shouldSurfaceErrorForRendererStackCookieCrashLoop(stackCookieCrashLoop)) {
          rememberLog(
            '[renderer] Windows stack-cookie crash loop (0xC0000409) with GPU fallback unavailable; surfacing error page (#108047)'
          )
          void loadRendererLoadErrorPage(mainWindow, {
            errorCode: details?.exitCode,
            errorDescription:
              'The desktop renderer crashed repeatedly (Windows STATUS_STACK_BUFFER_OVERRUN / 0xC0000409). GPU fallback could not recover the window.',
            repairHint: 'hermes desktop --force-build',
            reloadUrl: DEV_SERVER || pathToFileURL(resolveRendererIndex()).toString()
          })

          return
        }

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
      },
      // #95575: a renderer that repeatedly fails to load (torn bundle after
      // an update, file locked by AV, missing index.html) used to sit on a
      // white screen with only a desktop.log line. Once the bounded reload
      // budget is exhausted, put the VISIBLE error page in the window so the
      // user sees what is wrong and how to repair it.
      onFailedLoadBudgetExhausted: details => {
        rememberLog(
          `[renderer:main] load-failure budget exhausted; loading visible error page` +
            `${details?.errorCode === undefined ? '' : ` code=${String(details.errorCode)}`}`
        )
        void loadRendererLoadErrorPage(mainWindow, {
          errorCode: details?.errorCode,
          url: details?.url,
          errorDescription: 'The desktop renderer failed to load repeatedly after the update.',
          repairHint: 'hermes desktop --force-build',
          reloadUrl: DEV_SERVER || pathToFileURL(resolveRendererIndex()).toString()
        })
      },
      // #116472: the OS/Chromium can kill a renderer while the window is live (memory
      // reclaim, an external SIGTERM/SIGKILL). The lifecycle reloads that under the shared
      // budget (#85048); once the budget is spent, or for an unrecoverable reason, surface
      // the reason + a recovery button instead of a silent dead window.
      onRendererTerminated: details => {
        // An intentional quit/handoff also tears the renderer down; never pop a
        // recovery page for it (its window may still be alive when this fires).
        if (isQuittingForHandoff || backendShutdown.hasStarted() || mainWindow.isDestroyed()) {
          return
        }

        const reason = details?.reason ? String(details.reason) : 'unknown'
        const exit = details?.exitCode === undefined ? '' : `, exit code ${String(details.exitCode)}`
        rememberLog(`[renderer:main] renderer terminated while live (reason=${reason}${exit}); surfacing recovery page`)
        void loadRendererLoadErrorPage(mainWindow, {
          title: 'Hermes desktop UI was terminated',
          errorDescription:
            `The desktop UI process was terminated unexpectedly (reason: ${reason}${exit}). ` +
            'Your sessions and the background gateway are unaffected — reload to continue.',
          reloadUrl: DEV_SERVER || pathToFileURL(resolveRendererIndex()).toString()
        })
      }
    },
    isIntentionalTeardown: rendererTeardownInProgress,
    reloadWindowMs: RENDERER_RELOAD_WINDOW_MS,
    reloadMax: RENDERER_RELOAD_MAX,
    recentReloadTimesRef: rendererReloadTimesRef,
    reloadOnFailedLoad: true,
    onRendererGone: reason => desktopMetrics.recordRendererGone(mainContentsId, reason)
  })

  // Electron always passes the event first. The canonical (Electron 36+) shape
  // is (event, messageDetails); the deprecated positional shape is
  // (event, level, message, line, sourceId). Handled in renderer-log.ts, which
  // every renderer-content window shares (#79428: crashes in secondary/HUD/
  // quick-entry windows used to vanish without a trace).
  attachRendererConsoleCapture(mainWindow, 'main', rememberLog)

  // #95575: a torn renderer bundle (update replaced the app while its files
  // were locked) loads fine and then dies on the first lazy import — a white
  // screen with no error surface. resolveRendererIndex already logs the torn
  // copies; here we refuse to load one into the PRIMARY window and put the
  // visible repair page in it instead. The Reload button re-attempts the
  // bundle in case the file lock cleared since boot.
  const resolvedRenderer = DEV_SERVER ? null : resolveRendererIndexWithMissing()
  const rendererIndex = resolvedRenderer?.index ?? null
  const tornAssets = resolvedRenderer?.missing ?? []

  if (!DEV_SERVER && rendererIndex && tornAssets.length > 0) {
    rememberLog(
      `[renderer] primary window: chosen renderer bundle ${rendererIndex} is incomplete ` +
        `(${tornAssets.length} missing asset(s)); loading visible repair page instead of a white screen`
    )
    void loadRendererLoadErrorPage(mainWindow, {
      errorCode: 'ERR_FILE_NOT_FOUND',
      errorDescription: `The desktop renderer bundle is incomplete after the last update (${tornAssets.length} missing file(s)).`,
      missingAssets: tornAssets,
      repairHint: 'hermes desktop --force-build',
      reloadUrl: pathToFileURL(rendererIndex).toString()
    })
  } else {
    loadWindowUrl(
      mainWindow,
      DEV_SERVER || pathToFileURL(rendererIndex || resolveRendererIndex()).toString(),
      'Renderer'
    )
  }

  // Start the Python backend NOW, in parallel with the renderer load — not on
  // did-finish-load. The backend cold boot (spawn → port announce → /api/status)
  // is the dominant startup cost, and serializing it behind Chromium's load
  // added the whole renderer load time to first-usable-composer. The promise is
  // shared (backendConnectionState), so the renderer's getConnection() joins
  // this in-flight boot instead of duplicating it; early boot-progress events
  // the renderer misses are recovered by its getBootProgress() pull on mount.
  const startup = defaultRoute ? connectDesktopProfileRoute(defaultRoute) : startHermes()
  startup.catch(error => rememberLog(error.stack || error.message))

  mainWindow.webContents.once('did-finish-load', () => {
    // Zoom restore is handled by wireCommonWindowHandlers (shared with session
    // windows); no need to reapply it here.
    broadcastBootProgress()
    sendWindowStateChanged()
  })
}

ipcMain.handle('hermes:connection', async (event, profile, extra) => {
  const route = resolveDesktopConnectionRequest(
    profile,
    windowConnectionRoutes.get(event.sender.id),
    primaryProfileKey()
  )

  return connectDesktopProfileRoute(route, spawnPriorityFrom(extra?.priority), event.sender)
})

async function connectDesktopProfileRoute(
  route: DesktopProfileRoute,
  spawnPriority: LocalBackendSpawnPriority = 'foreground',
  sender?: Electron.WebContents
) {
  // Coalesce concurrent renderer dials for one profile scope (#90812): the
  // renderer-side reconnect lock is per-window, so two windows waking at once
  // both land here. The claim key mirrors ensureBackend()'s own profile
  // normalization so every spelling of the primary coalesces onto one dial.
  const scopeKey = backendScopeKey(route.connectionId, route.profile)
  const clearSpawnPriority = applySpawnPriority(scopeKey, spawnPriority)

  let connection

  try {
    connection = await backendDialClaims.run(scopeKey, () =>
      route.connectionId
        ? ensureRegistryBackend(route.connectionId, route.profile, '', { spawnPriority })
        : ensureBackend(route.profile, { spawnPriority })
    )
  } finally {
    clearSpawnPriority()
  }

  // Every republish carries LIVE window state (#102451): the backend pool entry
  // (and the getWindowState() snapshot startHermes baked into it) outlives
  // reloads, reconnects and sleep/wake, so a reply built only from the cached
  // descriptor overwrites the renderer's live fullscreen flag with the
  // mint-time snapshot. Reading the caller's state HERE keeps registry-scoped,
  // primary-resolved and bare replies consistent with the
  // hermes:window-state-changed live-push path.
  const windowState = liveWindowState(sender, {
    fromWebContents: BrowserWindow.fromWebContents,
    getWindowState,
    fallback: mainWindow
  })

  if (route.connectionId) {
    return overlayWindowState({ ...connection, connectionId: route.connectionId, registryScoped: true }, windowState)
  }

  const connectionId = resolvedConnectionId(readDesktopConnectionsRegistry(), connection)

  return connectionId
    ? overlayWindowState({ ...connection, connectionId }, windowState)
    : overlayWindowState(connection, windowState)
}

// Registry-scoped variant: resolve a backend for (connectionId, profile).
// An empty connection id is not registry.primary — that substitution dials
// another SSH host when a scoped caller drops the id. 'local' and an explicit
// primary id still resolve to those sources. The local kind delegates to
// ensureBackend when the v1 route is local, and forces a genuinely-local
// child when the v1 global mode is remote (the registry 'local' entry always
// means this machine) unless the profile is remote-only.
ipcMain.handle('hermes:connection:for', async (event, payload) => {
  const { connectionId, profile, priority } = payload && typeof payload === 'object' ? (payload as any) : ({} as any)
  const registry = readDesktopConnectionsRegistry()
  const id = registryDialConnectionId(connectionId, registry.primary)
  const spawnPriority = spawnPriorityFrom(priority)

  return connectDesktopProfileRoute(
    { connectionId: id, profile: String(profile ?? '').trim() || 'default' },
    spawnPriority,
    event.sender
  )
})

const windowConnectionRoutes = new WindowConnectionRouteRegistry()
const windowConnectionRouteOwners = new Set<number>()

function recordWindowConnectionRoute(sender: Electron.WebContents, route: unknown) {
  const id = sender.id
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
    sender.once('destroyed', () => {
      windowConnectionRoutes.delete(id)
      windowConnectionRouteOwners.delete(id)
      void resetPreviewReach(id)
    })
  }
}

ipcMain.on('hermes:connection:active-route', (event, route) => recordWindowConnectionRoute(event.sender, route))
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
        resetConnection: () => resetHermesConnectionState({ soft: true }),
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
    tracker: pooledRemoteLiveness
  })
}

// Re-dial one retired pool key through the SAME claim-guarded ensure path a
// renderer dial takes (#90812), so a resume-driven rebuild and a concurrent
// renderer reconnect coalesce onto one spawn instead of racing.
function redialPoolBackendAfterResume(poolKey: string) {
  const { connectionId, profile } = parseBackendScopeKey(poolKey)

  return backendDialClaims.run(poolKey, () =>
    connectionId ? ensureRegistryBackend(connectionId, profile) : ensureBackend(profile)
  )
}

// Identity for coalescing post-resume sweeps in the shared revalidation
// coordinator: overlapping resume/unlock/network-restore kicks join the one
// in-flight sweep instead of stacking probes.
const suspectPoolSweepScope = {}

// Sleep/wake recovery for POOLED remote/SSH backends (#93910). The primary
// renderer socket already has wake-path probe/reconnect nudges, but pooled
// descriptors (Bots pane, secondary connections) kept serving dead SSH
// tunnels after macOS resume: no child 'exit' fires for a remote, and the
// background failure-streak policy takes several rounds to drop one. On
// resume every pooled remote is suspect — probe each once (bounded), tear
// down the dead ones (pool entry + SSH bootstrap + tunnel/master) and rebuild
// them through the claim-guarded dial path.
function revalidateSuspectPoolAfterResume() {
  return remoteRevalidation.run(suspectPoolSweepScope, () =>
    revalidateSuspectPooledRemoteBackends({
      entries: backendPool.entries(),
      log: rememberLog,
      probe: (connection, path, options) => fetchJsonForBackend(connection, path, options),
      rebuild: poolKey => redialPoolBackendAfterResume(poolKey),
      retire: async poolKey => {
        await stopPoolBackend(poolKey)
        // The pool key doubles as the SSH scope for registry SSH backends and
        // resolves through sshScopeKey() for bare-profile remotes; both
        // teardown calls no-op when the scope holds no SSH state.
        await sshBootstrapCoordinator.cancelAndWait(poolKey)
        await teardownSshConnection(poolKey)
      },
      tracker: remoteLiveness
    })
  )
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
ipcMain.handle('hermes:window:openSession', async (_event, sessionId, opts) => {
  if (typeof sessionId !== 'string' || !sessionId.trim()) {
    return { ok: false, error: 'invalid-session-id' }
  }

  createSessionWindow(sessionId.trim(), {
    connectionId: typeof opts?.connectionId === 'string' ? opts.connectionId : null,
    profile: typeof opts?.profile === 'string' ? opts.profile : null,
    watch: opts?.watch === true
  })

  return { ok: true }
})
ipcMain.handle('hermes:window:openInstance', async (event, options) => {
  createInstanceWindow(options, BrowserWindow.fromWebContents(event.sender))

  return { ok: true }
})
registerWindowControlIpc(ipcMain, sender => BrowserWindow.fromWebContents(sender))
ipcMain.handle('hermes:window:openBrowser', async (_event, tabId) => {
  if (typeof tabId !== 'string' || !tabId.trim()) {
    return { ok: false, error: 'invalid-tab-id' }
  }

  createBrowserWindow(tabId.trim())

  return { ok: true }
})

// Cross-window renderer relay. The Browser pop-out, the primary window, and
// session tiles are separate renderers; packaged `file://` windows must not
// depend on BroadcastChannel origin semantics, so main relays opaque payloads
// between them over IPC. Destination validation and reply correlation stay
// renderer-side, so every feature riding this relay still fails closed instead
// of falling through to whatever window happens to be active later.
ipcMain.on('hermes:window:relay', (event, payload) => {
  for (const other of BrowserWindow.getAllWindows()) {
    if (!other.isDestroyed() && other.webContents.id !== event.sender.id) {
      other.webContents.send('hermes:window:relay', payload)
    }
  }
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
    const backend = await resolveHermesBackend(tuiResumeArgs(sessionId.trim(), profile || undefined))

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
  getFirstRunSetupGate().resetForRetry()
  resetBootstrapSnapshot()

  return { ok: true }
})
ipcMain.handle('hermes:bootstrap:repair', async (): Promise<{ ok: boolean; bundled?: boolean; error?: string }> => {
  // A bundled install's payload is immutable and sealed at build time —
  // "repair" would re-run the installer against a separate
  // %LOCALAPPDATA%\hermes tree the app doesn't own. The only repair for a
  // damaged bundle is reinstalling the app itself. Refuse without touching
  // bootstrapRepairRequested so a stale renderer can't drive an install.
  if (installShape() === 'bundled') {
    rememberLog('[bootstrap] repair refused on a bundled install; repair means reinstalling the app')

    return { ok: false, error: 'bundled-immutable' }
  }

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
  await teardownPrimaryBackendAndWait()

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
ipcMain.handle('hermes:local-backend:probe', async () => {
  // Resolution only. ensureRuntime/runBootstrap must not start from a hover
  // or a click that has not confirmed the install.
  const backend = await resolveHermesBackend([])

  return { bootstrapNeeded: backend?.kind === 'bootstrap-needed' }
})
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

  return buildRegistryProfileRoutes({
    agents,
    primaryConnectionId: registry.primary,
    sources: registry.connections
  })
})
ipcMain.handle('hermes:ssh-config:hosts', async () => ({ hosts: collectSshConfigHosts() }))
ipcMain.handle('hermes:ssh-config:resolve', async (_event, host) => {
  const value = String(host || '').trim()

  if (!value) {
    throw new Error('SSH host is required.')
  }

  const ssh = desktopSshBinary()

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
  managedConnectionUpdateGate.assertCanMutate(key)
  const registry = removeConnection(readDesktopConnectionsRegistry(), key)
  writeDesktopConnectionsRegistry(registry)
  // Tear down anything the removed connection still had running: pooled
  // backends under its composite keys and any ssh tunnel scopes it owned.
  await stopRegistryConnectionBackends(key)
  // …and everything cached ABOUT it. Ids are recycled label slugs, so re-adding "Mac mini"
  // gets `mac-mini` back — with the removed machine's profile list still cached under it.
  evictConnectionCaches(key)
  // And the renderer side: without this push, secondaries scoped to the
  // removed connection keep their WebSocket open (remote/cloud have no local
  // process to kill) and stream ghost events until page reload.
  broadcastConnectionsChanged({ connectionId: key, reason: 'removed' })
  desktopProfilePreferences.connectionRemoved(key)

  return { ok: true, registry: sanitizeConnectionsRegistry(registry) }
})
ipcMain.handle('hermes:connections:set-primary', async (_event, id) => {
  assertCanMutateManagedPrimaryRouting()
  const registry = setPrimaryConnection(readDesktopConnectionsRegistry(), String(id || ''))
  writeDesktopConnectionsRegistry(registry)

  return { ok: true, registry: sanitizeConnectionsRegistry(registry) }
})
ipcMain.handle('hermes:connections:set-launch-mode', async (_event, mode) => {
  assertCanMutateManagedPrimaryRouting()
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
// Connection caches live in ./connection-caches, which states (and tests) the invariant they share:
// each is keyed by connection id and is only valid while that id names the same machine, so
// removing a connection or re-pointing it must evict them (`evictConnectionCaches`).
const SSH_INVENTORY_RETRY_MS = 60_000

// Stable backend identity per registered connection: the `install_id` its
// /api/status reports (absent on backends older than the field). Enumeration
// runs on the ~5s Bot Mode roster poll and only hits /api/profiles, so the
// status probe is cached per connection with a TTL to avoid doubling roster
// traffic; the Test button refreshes it eagerly. A missing id simply bypasses
// the same-backend roster collapse — fully backward compatible.
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
    { rememberLog: sshRememberLog, sshBinary: desktopSshBinary() }
  )

  try {
    await ssh.open()
    const profiles = await remoteLifecycle.listRemoteHermesProfiles(ssh)

    if (profiles.length > 0) {
      sshRosterCache.set(connection.id, profiles)
    }

    // Backend identity, on the session we already have open: without it an ssh connection has no
    // install id at all, so two addresses for one machine never collapse into one roster row
    // (#88828 wired this for remote/local only, through /api/status).
    connectionInstallIds.set(connection.id, {
      id: await remoteLifecycle.readRemoteInstallId(ssh),
      ts: Date.now()
    })
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
  const withEnumerationDeadline = async <T>(work: Promise<T>, timeoutMs: number): Promise<T> => {
    let timer: ReturnType<typeof setTimeout> | null = null

    try {
      return await Promise.race([
        work,
        new Promise<never>((_resolve, reject) => {
          timer = setTimeout(() => reject(new Error('roster enumeration timed out')), timeoutMs)
        })
      ])
    } finally {
      if (timer !== null) {
        clearTimeout(timer)
      }
    }
  }

  return withoutInteractiveOauthLogin(() =>
    Promise.all(
      registry.connections.map(async connection => {
        let sourceFailureDetail = ''

        let raw: {
          connection: typeof connection
          error?: string
          needsSignIn?: boolean
          installId?: string
          profiles: null | string[]
          profileMetadata?: Record<string, RosterProfileMetadata>
        }

        try {
          // SSH roster listing must never spawn a dashboard. A stale
          // sshConnections key used to fall into ensureRegistryBackend and
          // respawn Spark/Mini every Bot Mode poll (~5s), then the mux died
          // (ECONNRESET / liveness probe drop).
          if (connection.kind === 'ssh') {
            await probeSshProfileInventory(connection)
            // The inventory probe learns the backend's install id on its own session; carrying it
            // here is what lets two ssh addresses for one machine collapse to one row.
            raw = {
              connection,
              profiles: null,
              error: 'connect-on-demand',
              installId: connectionInstallIds.get(connection.id)?.id
            }
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

            // Claim-guarded (#90812): this ~5s roster poll can race a renderer's
            // own reconnect dial for the same connection; coalescing avoids
            // bootstrapping a second SSH tunnel / remote dashboard.
            const descriptor: any = await withEnumerationDeadline(
              Promise.resolve(
                backendDialClaims.run(backendScopeKey(connection.id, null), () =>
                  ensureRegistryBackend(connection.id, null)
                )
              ),
              rosterSourceEnumerationTimeoutMs(connection)
            )

            const { body, installId } = await fetchRosterSourceData(
              () => getJsonForBackend(descriptor, '/api/profiles', { timeoutMs: 8_000 }),
              () => probeConnectionInstallId(connection.id, descriptor)
            )

            // The install-id probe is TTL-cached, so the 5s roster poll usually
            // pays zero extra requests; on a miss it runs beside /api/profiles.

            const profiles = Array.isArray(body?.profiles)
              ? body.profiles.map(p => String(p?.name || '').trim()).filter(Boolean)
              : []

            const profileMetadata = Array.isArray(body?.profiles)
              ? Object.fromEntries(
                  body.profiles
                    .map(profile => {
                      const name = String(profile?.name || '').trim()

                      if (!name) {
                        return null
                      }

                      const metadata: RosterProfileMetadata = {}

                      if (typeof profile?.display_name === 'string' && profile.display_name.trim()) {
                        metadata.display_name = profile.display_name.trim()
                      }

                      // `/api/profiles` names the Bot Mode title `bot_title`. Carried even
                      // when empty: "this backend has no title" is what lets the renderer
                      // drop a stale local one instead of painting it on this bot.
                      if (typeof profile?.bot_title === 'string') {
                        metadata.title = profile.bot_title.trim()
                      }

                      if (profile?.ui_meta && typeof profile.ui_meta === 'object') {
                        metadata.ui_meta = profile.ui_meta
                      }

                      if (typeof profile?.has_avatar === 'boolean') {
                        metadata.has_avatar = profile.has_avatar
                      }

                      return [name, metadata] as const
                    })
                    .filter((entry): entry is readonly [string, RosterProfileMetadata] => Boolean(entry))
                )
              : undefined

            // The root HERMES_HOME is an agent too; enumerations that omit it
            // (older backends list only named profiles) still get a default row.
            if (!profiles.includes('default')) {
              profiles.unshift('default')
            }

            raw = {
              connection,
              profiles,
              ...(installId ? { installId } : {}),
              ...(profileMetadata ? { profileMetadata } : {})
            }
          }
        } catch (error: any) {
          sourceFailureDetail = [error?.statusCode, error?.cause?.message].filter(Boolean).join(' | ')
          raw = {
            connection,
            profiles: null,
            error: redactSecrets(String(error?.message || error)),
            needsSignIn:
              isReauthRequiredError(error) || (connection.authMode === 'oauth' && isGatewayAuthRejection(error))
          }
        }

        if (raw.error && raw.error !== 'connect-on-demand') {
          const diagnostic = redactSecrets([raw.error, sourceFailureDetail].filter(Boolean).join(' | '))
            .replace(/[\r\n]+/g, ' ')
            .slice(0, 800)

          if (rosterSourceErrors.get(connection.id) !== diagnostic) {
            rememberLog(`[fleet-roster] ${connection.id}: ${diagnostic}`)
            rosterSourceErrors.set(connection.id, diagnostic)
          }
        } else if (raw.profiles && rosterSourceErrors.delete(connection.id)) {
          rememberLog(`[fleet-roster] ${connection.id}: connection recovered`)
        }

        if (raw.profiles && raw.profiles.length > 0) {
          sshRosterCache.set(connection.id, raw.profiles)
        }

        const remembered = rememberSshEnumeration(raw, sshRosterCache.get(connection.id), connection.kind)

        return {
          connection,
          ...remembered,
          ...(raw.needsSignIn ? { needsSignIn: true } : {}),
          ...(raw.installId ? { installId: raw.installId } : {}),
          ...(raw.profileMetadata ? { profileMetadata: raw.profileMetadata } : {})
        }
      })
    )
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
    sources: enumerations.map(({ connection, error, installId, profiles, needsSignIn }) => ({
      connectionId: connection.id,
      label: connection.label,
      kind: connection.kind,
      ...rosterSourceStatus({ profiles, error, needsSignIn }),
      ...(installId ? { installId } : {})
    }))
  }
})

// Registry-scoped fresh WS URL: the (connectionId, profile) analogue of
// hermes:gateway:ws-url. Same single-use-ticket discipline for OAuth sources.
const registryGatewayWsUrlHandler = createRegistryGatewayWsUrlHandler({
  ensureBackend: ensureRegistryBackend,
  mintTicket: mintGatewayWsTicket,
  buildTicketUrl: buildGatewayWsUrlWithTicket,
  rememberHeaders: rememberRemoteWsHeaders
})

ipcMain.handle('hermes:gateway:ws-url-for', async (_event, payload) => {
  return gatewayWsUrlIpcResult(() => registryGatewayWsUrlHandler(payload))
})

// Transactional update for a Desktop-managed SSH install. Unlike the generic
// fleet fan-out below, this path owns the remote serve lifecycle: it gates new
// dials, drains only exact Desktop-owned processes, runs the launcher outside
// those serves, proves the correlated receipt, and restores every prior scope.
async function requestManagedSshUpdate(rawId) {
  const connectionId = String(rawId || '').trim()
  const existing = managedConnectionUpdates.get(connectionId)

  if (existing) {
    return existing
  }

  const correlationId = crypto.randomUUID()
  const registry = readDesktopConnectionsRegistry()
  const source = registry.connections.find(connection => connection.id === connectionId)

  if (!source) {
    return refusedManagedSshUpdate(connectionId, correlationId, `No connection with id "${connectionId}".`)
  }

  if (source.kind !== 'ssh') {
    return refusedManagedSshUpdate(
      connectionId,
      correlationId,
      'Only registered Desktop-managed SSH connections can use this update lifecycle.'
    )
  }

  if (!managedConnectionUpdateGate.claim(connectionId, correlationId)) {
    return refusedManagedSshUpdate(connectionId, correlationId, 'A managed update is already in progress.')
  }

  const operation = (async () => {
    try {
      return await updateManagedSshConnection(source, correlationId)
    } catch (error: any) {
      return refusedManagedSshUpdate(connectionId, correlationId, String(error?.message || error))
    } finally {
      managedConnectionUpdateGate.release(connectionId, correlationId)
      managedConnectionUpdates.delete(connectionId)
    }
  })()

  managedConnectionUpdates.set(connectionId, operation)

  return operation
}

ipcMain.handle('hermes:connections:update-managed', async (_event, rawId) => requestManagedSshUpdate(rawId))

// Fan out `hermes update` to every eligible registered connection at once.
// Cloud entries are excluded (platform-managed); each dispatch reports
// independently so one dead LAN box can't wedge the batch. Local reuses the
// app's own update pipeline; Desktop-managed SSH uses the transactional
// drain/update/restore lifecycle; URL remotes POST their backend updater.
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

  // Remote entries settle before the local handoff runs: the local updater
  // waits on the window PID exiting, so a still-running managed SSH update
  // would eat into (or outlive) that deadline. Order of results is preserved.
  const results = await updateConnectionsBeforeLocal(
    registry.connections.filter(connection => !excludeIds.has(connection.id)),
    async (connection: RegistryConnection) => {
      const base = { connectionId: connection.id, label: connection.label, kind: connection.kind }
      const eligibility = updateEligibility(connection)

      if (!eligibility.eligible) {
        return { ...base, ok: false, skipped: true, reason: eligibility.reason }
      }

      try {
        if (connection.kind === 'local') {
          // The app-managed runtime updates through the same pipeline as the
          // Settings → Updates button (marker + venv gate + relaunch flow).
          const result: any = await applyUpdates()

          return { ...base, ok: result?.ok !== false, detail: result?.message || 'update started' }
        }

        if (connection.kind === 'ssh') {
          return managedSshUpdateAllRow(base, await requestManagedSshUpdate(connection.id))
        }

        // Claim-guarded (#90812): coalesce with a concurrent renderer dial
        // for the same connection instead of bootstrapping a second backend.
        const descriptor: any = await backendDialClaims.run(backendScopeKey(connection.id, null), () =>
          ensureRegistryBackend(connection.id, null)
        )

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
    }
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

    const options = {
      method: opts.method,
      body: opts.body,
      timeoutMs: opts.timeoutMs,
      headers: descriptor.headers
    }

    return requestWithOauthFallback(descriptor.baseUrl, {
      ensureNativeAccessToken,
      requestWithBearer: (bearer: string) => fetchJson(url, null, { ...options, bearer }),
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

ipcMain.handle('hermes:connection-config:probe', async (_event, rawUrl) => probeRemoteAuthMode(rawUrl))
ipcMain.handle('hermes:connection-config:oauth-login', async (_event, rawUrl, rawOpts) => {
  // Capability-gated login (RFC 8252). Probe the gateway's public /api/status
  // for supported auth_flows and /api/auth/providers for provider capabilities:
  //   - all providers support password → always use the embedded login window
  //     (password providers require the dashboard login form; native PKCE
  //     can never complete for that provider shape)
  //   - advertises "native_pkce" AND at least one non-password provider →
  //     run the system-browser + loopback + PKCE flow
  //   - older gateway with no provider metadata → fall back to the auth_flows
  //     check (existing compatibility)
  //   - a failed native login reports the error rather than auto-falling back
  //     to the embedded flow — one sign-in action opens at most one window.
  const baseUrl = normalizeRemoteBaseUrl(rawUrl)
  // Order login attempts without interrupting rotation of the existing session.
  const authIsCurrent: () => boolean = nativeAccessTokenCoordinator.beginLogin(baseUrl)

  // A registry-editor sign-in can run BEFORE the draft connection is saved:
  // settle the id the save will reuse (returned so the renderer pins it into
  // the draft) so the login window writes its session cookies into the
  // per-connection jar the saved connection will actually read — not the
  // legacy shared jar an unmatched URL would fall back to, where the session
  // is both unreadable by the new connection and able to evict a same-host
  // primary's cookie (#92183 isolation hole). '' → URL-matched behavior.
  let loginConnectionId = ''

  try {
    loginConnectionId = connectionIdForPendingLogin({
      connectionId: rawOpts?.connectionId,
      label: rawOpts?.label,
      registry: readDesktopConnectionsRegistry()
    })
  } catch {
    loginConnectionId = ''
  }

  // The draft's intended entry shape (kind/authMode the save will persist).
  // The identity branch in oauth-partition.ts gates the pre-save private jar
  // on it: only a cookie-auth remote draft earns its own jar up front; a
  // cloud or token draft signs in on the legacy jar — the jar the saved
  // entry reads — so login and read can never disagree. Invalid or missing
  // values fail closed to the legacy jar in the resolver.
  const pendingKind = typeof rawOpts?.kind === 'string' ? rawOpts.kind : ''
  const pendingAuthMode = typeof rawOpts?.authMode === 'string' ? rawOpts.authMode : ''

  let statusBody: any = null

  try {
    statusBody = await fetchPublicJson(`${baseUrl}/api/status`, { timeoutMs: 8_000 })
  } catch {
    // Can't read status — fall through to the embedded flow, which has its
    // own error handling and works against any gated gateway.
  }

  const authRequired = statusBody && authModeFromStatus(statusBody) === 'oauth'
  const providers = authRequired ? await gatewayAuthProviders(baseUrl) : []

  const strategy = resolveLoginStrategy(statusBody, { providers })

  // A newer login/logout can finish during the status/provider probes. Do not
  // open a browser or login window for an attempt that no longer owns auth.
  if (!authIsCurrent()) {
    throw new NativeAuthChangedError()
  }

  if (strategy === 'native') {
    try {
      const tokens = await runNativeLogin(baseUrl, {
        // Route the browser-open through the single external-open path so it
        // gets the WSL handling and the open-failure modal. Fail fast (throw)
        // so runNativeLogin reports the real reason instead of waiting out the
        // loopback timeout.
        openExternal: async url => {
          const result = await openExternalUrl(url)

          if (result.ok === false) {
            throw new Error(
              result.reason === 'failed' && result.message
                ? result.message
                : 'Could not open the system browser for native sign-in'
            )
          }
        },
        postJson: (url, body, opts) => postJsonNoAuth(url, body, opts),
        rememberLog
      })

      if (!authIsCurrent()) {
        throw new NativeAuthChangedError()
      }

      nativeAccessTokenCoordinator.storeTokens(baseUrl, tokens)
      // Confirmed sign-in — release the reauth latch so the next
      // startHermes() re-dials instead of replaying the stale rejection.
      remoteReauthFailure = null

      return { ok: true, baseUrl, connected: true, connectionId: loginConnectionId || undefined }
    } catch (error) {
      rememberLog(`[native-oauth] native login failed (${error instanceof Error ? error.message : String(error)})`)

      return {
        ok: false,
        error: error instanceof Error ? error.message : String(error),
        connected: false,
        connectionId: loginConnectionId || undefined
      }
    }
  }

  // Legacy embedded-webview cookie flow.
  await openOauthLoginWindow(baseUrl, {
    connectionId: loginConnectionId,
    pendingAuthMode,
    pendingKind
  })

  const connected = await hasOauthSessionCookie(baseUrl, {
    connectionId: loginConnectionId,
    pendingAuthMode,
    pendingKind
  })

  // Only a CONFIRMED sign-in releases the latch. A cancelled/closed login
  // window must leave it set, or the overlay's "Sign in" button starts
  // flickering again on the next retry.
  if (!authIsCurrent()) {
    throw new NativeAuthChangedError()
  }

  if (connected) {
    // A confirmed cookie login supersedes any older native identity.
    nativeAccessTokenCoordinator.clearTokens(baseUrl)
    remoteReauthFailure = null
  }

  return { ok: true, baseUrl, connected, connectionId: loginConnectionId || undefined }
})
ipcMain.handle('hermes:connection-config:oauth-logout', async (_event, rawUrl) => {
  const baseUrl = normalizeRemoteBaseUrl(rawUrl)

  // Also drop any native (RFC 8252) bearer tokens for this gateway so a
  // logout clears BOTH auth shapes.
  // Clear before awaiting cookie I/O: a pending login/refresh cannot restore
  // logout, and a later login must not be erased when cookie clearing settles.
  nativeAccessTokenCoordinator.clearTokens(baseUrl)
  await clearOauthSession(baseUrl)

  // Report against the SAME liveness notion the Settings indicator uses
  // (AT-or-RT cookie, or a native token) so a logout that left any session
  // behind is reflected as still-connected rather than silently signed-out.
  const connected = (await hasLiveOauthSession(baseUrl)) || hasNativeSession(baseUrl)

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
  assertCanMutateManagedPrimaryRouting()
  const config = coerceDesktopConnectionConfig(payload)
  writeDesktopConnectionConfig(config)

  return sanitizeDesktopConnectionConfig(config, payload?.profile)
})
ipcMain.handle('hermes:connection-config:apply', async (_event, payload) => {
  assertCanMutateManagedPrimaryRouting()
  const previousConfig = readDesktopConnectionConfig()
  const previousRegistry = readDesktopConnectionsRegistry()
  const config = coerceDesktopConnectionConfig(payload, previousConfig)

  const key = connectionScopeKey(payload?.profile)
  const scope = key || ''
  const nextRegistry = key ? previousRegistry : reconcileAppliedGlobalConnection(previousRegistry, config)

  // Primary apply: the applied window's recorded route still names the source
  // it just LEFT, and a profile-less re-dial is answered from that record
  // (resolveDesktopConnectionRequest), so the renderer kept dialing the gateway
  // the user switched away from and every later reconnect re-asked the same
  // stale question (#92352). Re-point the record at the newly applied primary
  // in the same step as the notify: both run only after the config/registry
  // write has committed, so a rolled-back apply can never leave a record
  // naming a source that did not land. A profile-scoped apply (v1 per-profile
  // override) leaves the registry primary untouched, so it keeps the bare notify.
  const applyPrimaryRoute = () => {
    const win = mainWindow

    if (win && !win.isDestroyed() && win.webContents && !win.webContents.isDestroyed()) {
      const previous = windowConnectionRoutes.get(win.webContents.id)

      recordWindowConnectionRoute(
        win.webContents,
        appliedPrimaryWindowRoute(nextRegistry, previous?.profile ?? primaryProfileKey())
      )
    }

    sendConnectionApplied()
  }

  const notifyApplied = key ? sendConnectionApplied : applyPrimaryRoute

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
            notifyConnectionApplied: notifyApplied,
            resumeFirstRunRemote: abandonFirstRunSetupChoiceForRemoteApply,
            teardownPrimaryBackend: teardownPrimaryBackendAndWait
          }),
        scope,
        sendApplied: notifyApplied,
        stopPool: stopPoolBackend,
        teardownPrimary: () => teardownPrimaryBackendAndWait({ soft: true }),
        teardownSsh: value => teardownSshConnection(value || null)
      })
  })

  return sanitizeDesktopConnectionConfig(config, payload?.profile)
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

// The host renderer reports which of its Browser guests are off screen (a
// hidden session's kept-alive page), so focused-guest gestures skip them.
ipcMain.on('hermes:preview-guest-hidden', (event, payload) => {
  notePreviewGuestHidden(
    event.sender,
    electronWebContents.fromId(Number(payload?.webContentsId)),
    Boolean(payload?.hidden)
  )
})

ipcMain.on('hermes:f12ShortcutActive', (event, active) => {
  if (active) {
    f12ShortcutActiveWindows.add(event.sender.id)
  } else {
    f12ShortcutActiveWindows.delete(event.sender.id)
  }
})

app.on('web-contents-created', (_event, contents) => {
  contents.once('destroyed', () => f12ShortcutActiveWindows.delete(contents.id))
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
async function interceptSessionRequestForRemote(request, registryConnectionId = null) {
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

    const registrySources = await pooledRegistrySessionSources(
      shouldIncludeLocalRegistrySessionSource(registryConnectionId, !globalRemoteActive())
    )

    if (remoteProfiles.length === 0 && registrySources.length === 0) {
      return undefined // no remote profiles and no connected registry gateways → local fast path
    }

    if (
      !hasPinnedRegistrySessionSource(registryConnectionId, request?.profile, registrySources, !globalRemoteActive())
    ) {
      // Do not manufacture a partial all-gateways response while the selected
      // registry backend is still dialing or has just gone idle. The caller
      // falls back to the direct pinned route, which is slower but complete
      // for the gateway the renderer actually selected.
      return undefined
    }

    const requested = (searchParams.get('profile') || 'all').trim() || 'all'

    if (requested !== 'all') {
      return profileHasRemoteOverride(requested) ? remoteSessionList(requested, searchParams) : undefined
    }

    return mergeRemoteProfileSessions(searchParams, remoteProfiles, registrySources)
  }

  // Batched sidebar slices. With no remote profiles the local batched endpoint
  // (one DB open per profile) serves it directly — take the fast path. When
  // remotes exist, fan the three slices back out to the per-slice
  // /api/profiles/sessions path (which already merges remote rows correctly) and
  // reassemble; local profiles fall back to three primary reads there, but
  // remote correctness is preserved.
  if (method === 'GET' && pathname === '/api/profiles/sessions/sidebar') {
    const remoteProfiles = configuredRemoteProfileNames()

    const registrySources = await pooledRegistrySessionSources(
      shouldIncludeLocalRegistrySessionSource(registryConnectionId, !globalRemoteActive())
    )

    if (remoteProfiles.length === 0 && registrySources.length === 0) {
      return undefined // local fast path → batched endpoint's single DB open
    }

    if (
      !hasPinnedRegistrySessionSource(registryConnectionId, request?.profile, registrySources, !globalRemoteActive())
    ) {
      return undefined
    }

    const { recents: recentsSp, cron: cronSp, messaging: messagingSp } = buildSidebarSessionSliceParams(searchParams)

    const [recents, cron, messaging] = await Promise.all([
      fetchProfilesSessionSlice(recentsSp, remoteProfiles, registrySources),
      fetchProfilesSessionSlice(cronSp, remoteProfiles, registrySources),
      fetchProfilesSessionSlice(messagingSp, remoteProfiles, registrySources)
    ])

    return assembleSidebarSessionSlices(recents, cron, messaging)
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
      // #64999: the override's remote can be a multi-profile backend — an
      // unscoped read opens its launch-profile state.db, so a resume 4007s
      // even though the row exists under its real owner. Scope the read the
      // same way the list fetch does; a legacy single-profile scope ('')
      // keeps the path bare.
      const ownerScope =
        remoteProfileQueryScope(profile, profileSshOverride(readDesktopConnectionConfig(), profile)?.remoteProfile) ||
        profile

      if (method === 'GET') {
        return fetchJsonForProfile(
          profile,
          pathWithRemoteOwnerScope(passthroughQuery ? `${pathname}?${passthroughQuery}` : pathname, ownerScope)
        )
      }

      const body = request.body && typeof request.body === 'object' ? { ...request.body } : request.body

      if (body && ownerScope) {
        ;(body as Record<string, unknown>).profile = ownerScope
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

// A remote profile's session list. The fetch itself is profile-scoped
// (fetchRemoteProfileSessions, #64999); the remote's own stamps carry the
// authoritative identity — never relabel rows with the Desktop scope name.
async function remoteSessionList(profile, searchParams) {
  const sshOverride = profileSshOverride(readDesktopConnectionConfig(), profile)

  const data = await fetchRemoteProfileSessions(profile, searchParams, fetchJsonForProfile, {
    remoteProfileAlias: sshOverride?.remoteProfile
  })

  const rows = tagRemoteSessionRows(
    rowsOf(data),
    remoteProfileQueryScope(profile, sshOverride?.remoteProfile) || profile
  )

  return { ...(data as any), sessions: rows }
}

// #85834: find which remote profile owns a session id when the caller gave no
// profile hint (pure lookup lives in profile-session-routing.ts; the bounded
// memo lives in remote-owner-cache.ts — #58485: it only ever INSERTS, so the
// raw Map grew one entry per session id ever resolved, unbounded, on the main
// process heap).
const remoteOwnerCache = createRemoteOwnerCache()

async function remoteOwnerProfileForSession(sessionId: string) {
  if (!sessionId) {
    return null
  }

  const remoteProfiles = configuredRemoteProfileNames()

  if (remoteProfiles.length === 0) {
    return null
  }

  const cached = remoteOwnerCache.fresh(sessionId)

  if (cached) {
    return cached.profile
  }

  const owner = await findRemoteOwnerProfileForSession(sessionId, remoteProfiles, (profile, params) =>
    remoteSessionList(profile, params)
  ).catch(() => null)

  remoteOwnerCache.remember(sessionId, owner)

  return owner
}

// Resolve one /api/profiles/sessions slice with remote profiles spliced in —
// the same branch logic as the GET /api/profiles/sessions intercept, but always
// returns data (never `undefined`) so a batched caller can compose slices. A
// specific local profile reads from the local primary; a remote-override profile
// reads from its remote; 'all' merges every remote into the primary aggregate.
async function fetchProfilesSessionSlice(searchParams, remoteProfiles, registrySources = null) {
  const requested = (searchParams.get('profile') || 'all').trim() || 'all'

  if (requested !== 'all') {
    if (profileHasRemoteOverride(requested)) {
      return remoteSessionList(requested, searchParams)
    }

    return fetchPrimaryProfileSessions(searchParams, fetchJsonForProfile)
  }

  return mergeRemoteProfileSessions(searchParams, remoteProfiles, registrySources)
}

// Unified list: primary's local aggregate, with each remote profile's stale local
// rows/totals swapped for the remote's real ones, re-sorted by recency and
// re-windowed to the requested page. A dead remote contributes nothing rather
// than breaking the sidebar. Connected registry gateways' sessions are spliced
// in too (#88880) — the unified Sessions list shows EVERY connected gateway's
// chats, tagged with connection_id + profile so opens route correctly.
async function mergeRemoteProfileSessions(searchParams, remoteProfiles, registrySourcesOverride = null) {
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
  const registrySources = registrySourcesOverride || (await pooledRegistrySessionSources())

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
// The primary local connection is excluded for the legacy unpinned path — the
// primary aggregate carries local rows there. A registry-pinned aggregate opts
// in so forced-local backends remain visible when the legacy primary is remote.
async function pooledRegistrySessionSources(includeLocal = false): Promise<RegistrySessionSource[]> {
  const registry = readDesktopConnectionsRegistry()
  const sources: RegistrySessionSource[] = []

  for (const connection of registry.connections) {
    if (connection.kind === 'local' && !includeLocal) {
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

    const perProfile = connection.kind === 'ssh' || (includeLocal && connection.kind === 'local')

    for (const [key, entry] of perProfile ? pooled : pooled.slice(0, 1)) {
      try {
        // Already-resolved for a connected backend; a still-dialing entry is
        // skipped via the timeout guard rather than blocking the sidebar.
        const descriptor = await Promise.race([
          entry.connectionPromise,
          new Promise((_, reject) => setTimeout(() => reject(new Error('pending')), 2_000))
        ])

        backends.push({
          descriptor,
          profileLabel: perProfile ? key.slice(prefix.length) || 'default' : null
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
  // Claim-guarded (#90812): every registry-scoped REST call funnels through
  // here, so it can race a renderer's own WS reconnect dial for the same
  // (connectionId, profile) scope; coalescing avoids bootstrapping a second
  // SSH tunnel / remote dashboard. A passive read never dials, so it stays
  // OUT of the claim: an interactive open coalescing onto an in-flight
  // passive read would otherwise inherit its "no warm backend" rejection.
  const spawnPriority = spawnPriorityFrom(request?.priority)

  const connection: any = request?.passive
    ? await ensureRegistryBackend(registryConnectionId, routeProfile, '', { passive: true })
    : await backendDialClaims.run(backendScopeKey(registryConnectionId, routeProfile), () =>
        ensureRegistryBackend(registryConnectionId, routeProfile, '', { spawnPriority })
      )

  const requestPath = pathForRegistryBackendRequest(request.path, requestProfile, connection)

  const response = await fetchJsonForBackend(connection, requestPath, {
    method: request?.method,
    body: request?.body,
    upload: request?.upload,
    timeoutMs: resolveTimeoutMs(request?.timeoutMs, DEFAULT_FETCH_TIMEOUT_MS)
  })

  desktopProfilePreferences.afterProfileRequest(registryConnectionId, request, response, connection.mode)

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

// A 404 raised by `fetchJson` — the shape is `404: <body>` (see fetchJson).
// Session lookups are a probe ladder: "not on this profile" is a normal rung
// outcome, not a failure.
function isNotFoundApiError(error) {
  return /(?:^|\s)404\b/.test(String((error as any)?.message ?? error))
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
    if (isAllProfilesSessionListRequest(request?.method, request?.path)) {
      const aggregate = await interceptSessionRequestForRemote(request, registryConnectionId)

      if (aggregate !== undefined) {
        return aggregate
      }
    }

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
  const spawnPriority = spawnPriorityFrom(request?.priority)
  // After tearing down a backend for profile deletion, route to the primary
  // backend instead of spawning a fresh pool backend.  A freshly spawned
  // backend calls ensure_hermes_home() which recreates the profile directory,
  // defeating the deletion and leaving a zombie process.
  //
  // Local-profile REST calls stay on the primary dashboard and carry ?profile=
  // (or name the profile in the path / PATCH body). A request that MUTATES
  // state the server cannot scope at all retains its pooled backend, whose
  // HERMES_HOME is then the scope, so a destructive call can never fall
  // through to the primary home — `resolveProfileBackendRoute` case 6.
  //
  // A profile rename tears down the old-name backend the same way; for a
  // primary rename the lifecycle has already made `default` the temporary
  // primary until the PATCH settles, so the request routes there.
  const apiRoute = resolveProfileApiRequest(profile, request.path, profileRouteOptions(profile, request))

  const routeProfile = profileRename
    ? profileRename.routeProfile
    : resolveRouteProfile(tornDownProfile, apiRoute.backendProfile)

  let response
  let connection

  try {
    connection = await ensureBackend(routeProfile, {
      passive: request?.passive,
      request: { method: request?.method, path: request?.path },
      spawnPriority
    })
    const timeoutMs = resolveTimeoutMs(request?.timeoutMs, DEFAULT_FETCH_TIMEOUT_MS)

    response = await fetchJsonForBackend(connection, apiRoute.requestPath, {
      method: request?.method,
      body: request?.body,
      upload: request?.upload,
      timeoutMs
    })
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

  try {
    desktopProfilePreferences.afterProfileRequest(null, request, response, connection.mode)
  } finally {
    await profileRename?.complete()
  }

  return response
}

// Format an api-request failure for desktop.log. The renderer only ever sees
// the invoke rejection; the real stack lives here in main, so persist it
// before rethrowing. Clamp: paths and error detail must not bloat the log.
function formatApiRequestFailure(
  request: { method?: string; path?: string } | null | undefined,
  error: unknown
): string {
  const method = String(request?.method ?? 'GET').toUpperCase()
  const path = String(request?.path ?? '(no path)').slice(0, 500)
  const detail = error instanceof Error ? (error.stack ?? error.message) : String(error)

  return `[hermes:api ${method} ${path}] ${detail}`.slice(0, 6000)
}

ipcMain.handle('hermes:api', async (_event, request) => {
  // Hold the deletion gate for BOTH profile deletes and renames: a concurrent
  // renderer reconnect entering ensureBackend() mid-mutation would otherwise
  // respawn the old-name backend and recreate its HERMES_HOME (#45474).
  const deletingProfile = profileNameFromDeleteRequest(request)
  const mutatingProfile = deletingProfile || profileRenameFromRequest(request)?.oldName || null
  const registryConnectionId = apiRequestRegistryConnectionId(request)

  try {
    if (deletingProfile && registryConnectionId) {
      return await dispatchConnectionScopedProfileDelete(request, {
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
      return await handleHermesApiRequest(request)
    }

    const releaseProfileDeletion = profileDeletionGate.acquire(mutatingProfile)

    return await handleHermesApiRequest(request).finally(releaseProfileDeletion)
  } catch (error) {
    // Electron logs "Error occurred in handler for 'hermes:api'" with a full
    // stack for EVERY rejected invoke, and there is no opt-out on the handler.
    // Session resolution is a deliberate probe ladder (`resolveStoredSession`:
    // cache → active backend → each other profile) and the renderer already
    // handles a miss by falling to the next rung — so an expected 404 is not an
    // error condition. Left rejecting, it printed a multi-line stack per probe
    // on every startup and session switch: pure noise that buries genuine
    // handler failures.
    //
    // So don't reject for that one case — RESOLVE with a sentinel and let
    // preload (our own code, the other side of the same seam) turn it back into
    // a rejection with the identical `404: <body>` message. The renderer
    // contract is unchanged; only Electron's logging is bypassed. Every other
    // failure still rejects and still logs in full.
    if (isNotFoundApiError(error)) {
      return { [HERMES_API_EXPECTED_404]: String((error as any)?.message ?? error) }
    }

    // Persist the failure (full stack) before the rejection crosses to the
    // renderer, where the invoke wrapper strips it to a one-line message.
    rememberLog(formatApiRequestFailure(request, error))
    flushDesktopLogBufferSync()

    throw error
  }
})

// Speech claims outlive instant cues so throttled peer windows cannot replay a reply.
const ownsAmbientCue: ReturnType<typeof createAmbientClaimArbiter> = createAmbientClaimArbiter()
ipcMain.handle('hermes:ambient:claim', (_event: IpcMainInvokeEvent, key: unknown): boolean =>
  ownsAmbientCue(String(key ?? ''))
)

const nativeNotifications = registerNativeNotifications({
  getMainWindow: (): BrowserWindow | null => mainWindow,
  focusWindow
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
  // Backend-reported paths are WSL/POSIX (`/home/...`, `/mnt/c/...`); on a
  // Windows host bridge them to a UNC/drive form, same as directory reads.
  const bridgedPath = resolveIpcFileReadPath(filePath)

  try {
    return await readFileDataUrlForIpc(bridgedPath, {
      maxBytes: dataUrlReadMaxBytesFromMb(dataUrlReadMaxMb),
      mimeType: mimeTypeForPath(resolveRequestedPathForIpc(bridgedPath, { purpose: 'File preview' })),
      purpose: 'File preview'
    })
  } catch (error) {
    if (isMissingFileError(error)) {
      return missingFileResult(filePath, error)
    }

    throw error
  }
})

// Remote attachment transfer is independent of the preview / Settings path.
// Keep a finite cap so Electron + base64 memory stays bounded while archives
// can exceed the default 16 MiB preview ceiling (and still fit the gateway
// WebSocket frame limit after base64 expansion).
ipcMain.handle('hermes:readFileDataUrlForAttach', async (_event, filePath) => {
  const bridgedPath = resolveIpcFileReadPath(filePath)

  try {
    return await readFileDataUrlForIpc(bridgedPath, {
      maxBytes: ATTACHMENT_UPLOAD_DEFAULT_MAX_BYTES,
      mimeType: mimeTypeForPath(resolveRequestedPathForIpc(bridgedPath, { purpose: 'Attachment upload' })),
      purpose: 'Attachment upload'
    })
  } catch (error) {
    if (isMissingFileError(error)) {
      return missingFileResult(filePath, error)
    }

    throw error
  }
})

ipcMain.handle('hermes:readFileText', async (_event, filePath) => {
  try {
    const { resolvedPath, stat } = await resolveReadableFileForIpc(resolveIpcFileReadPath(filePath), {
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
  } catch (error) {
    // A preview probing a file that is gone (deleted, moved, or cleared from
    // /tmp since the tab/transcript reference was written) is an expected
    // outcome. Return a structured error instead of rejecting — Electron logs
    // every rejected handler with a stack trace even though the renderer shows
    // "preview unavailable" either way.
    if (isMissingFileError(error)) {
      return missingFileResult(filePath, error)
    }

    throw error
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
  const properties = selectPathsDialogProperties(options || {})

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

ipcMain.handle('hermes:capturePreview', async (_event, payload) => {
  const guest = electronWebContents.fromId(Number(payload?.webContentsId))

  return capturePreviewContents(guest, payload?.rect, payload?.viewport)
})

ipcMain.handle('hermes:saveImageBuffer', async (_event, payload) => {
  const data = payload?.data

  if (!data) {
    throw new Error('saveImageBuffer: missing data')
  }

  const buffer = Buffer.isBuffer(data) ? data : Buffer.from(data)

  return writeComposerImage(buffer, payload?.ext || '.png', payload?.name)
})

ipcMain.handle('hermes:savePastedText', async (_event, payload) => {
  const text = typeof payload?.text === 'string' ? payload.text : ''

  if (!text) {
    throw new Error('savePastedText: missing text')
  }

  return writeComposerPaste(HERMES_HOME, text)
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

ipcMain.handle('hermes:watchPreviewFile', (event, url) => watchPreviewFile(event.sender, String(url || '')))

ipcMain.handle('hermes:watchDirectory', (event, dir) => watchDirectory(event.sender, String(dir || '')))

ipcMain.handle('hermes:stopPreviewFileWatch', (_event, id) => stopPreviewFileWatch(String(id || '')))

// Each renderer reports the turns it has in flight; the quit guard reads the
// merged picture. Keyed by webContents id so a closed window stops counting.
const activeWorkByWebContents = new Map<number, ActiveWork>()

// Synchronous, webContents-independent cache of the most recent active-work
// summary we heard from *any* renderer. The per-webContents map above is
// dropped the moment a webContents is destroyed (a stream can reload its
// webContents mid-turn), so at quit time it can read empty even though a turn
// is live. This cached value survives that and is what the quit guard falls
// back to. It is only ever refreshed by real publishes, so an idle app
// (count=0) clears it — no false positives after a turn ends.
let lastActiveWorkSeen: ActiveWork = { count: 0, titles: [] }

// Every window that hosts a chat surface (primary, session, instance). The
// last-window close guard below is installed centrally for all of them.
const chatWindows = new Set<BrowserWindow>()

function hasOtherChatWindows(window: BrowserWindow): boolean {
  return [...chatWindows].some(candidate => candidate !== window && !candidate.isDestroyed())
}

// The same merged picture drives background throttling: chat windows run
// unthrottled while any turn is in flight (streaming must paint while hidden)
// and fall back to Chromium's default throttling at idle. See stream-throttle.ts.
const streamThrottle = createStreamThrottle(undefined, undefined, {
  // #94865 is specific to native Wayland fullscreen surfaces. Reuse the same
  // Ozone resolver as the rest of Desktop so XWayland/macOS/Windows retain the
  // normal idle throttling contract.
  keepFullscreenPainting: process.platform === 'linux' && linuxOzoneBackend(process.env, process.argv) === 'wayland'
})

function isAnyTurnInFlight() {
  return mergeActiveWork(activeWorkByWebContents.values()).count > 0
}

function updateStreamThrottleFromActiveWork() {
  const working = isAnyTurnInFlight()

  streamThrottle.update(working)
  // The 'while-working' keep-awake mode rides the same merged signal: no
  // listener or timer of its own (see the keep-awake block below).
  applyKeepAwake(working)
}

ipcMain.on('hermes:active-work', (event, payload) => {
  const id = event.sender.id

  if (!activeWorkByWebContents.has(id)) {
    const forget = () => {
      // Whichever fires first detaches the other, so crash/reload cycles on
      // one webContents don't stack listeners.
      event.sender.off('destroyed', forget)
      event.sender.off('render-process-gone', forget)
      activeWorkByWebContents.delete(id)
      updateStreamThrottleFromActiveWork()
    }

    event.sender.once('destroyed', forget)
    // A dead renderer keeps its webContents, and some exits never reload, so
    // its last count would pin throttling and the 'while-working' blocker
    // until quit. A reloaded renderer re-reports its live turns on mount.
    event.sender.once('render-process-gone', forget)
  }

  const work = normalizeActiveWork(payload)
  activeWorkByWebContents.set(id, work)
  lastActiveWorkSeen = work
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
  killTimedGitChildren()
  sshIsolatedKeepalives.stopAll()
  destroyKeepaliveAgents()
  nativeNotifications.dispose()
  quitFinalization.arm()
})

app.on('quit', () => {
  quitFinalization.cancel()
})

// Answered synchronously so preload can publish the verdict before the
// renderer's first script — see the note there on why it cannot decide this
// itself. Registered at module scope, which runs long before any window.
ipcMain.on('hermes:translucency:support', event => {
  event.returnValue = { glass: GLASS_SUPPORTED, translucency: TRANSLUCENCY_SUPPORTED }
})

// Feature-flag facts the renderer needs before first paint (same sendSync
// pattern as translucency). Resolved in feature-flags.ts from the launch
// argv and the artifact's channel: `--local` (from `hermes desktop --local`
// or directly on Hermes.exe, a shortcut edit) gates the local-models GUI on
// stable builds, and canary builds get the same surfaces by default. Launch
// flags survive self-relaunches because collectRelaunchArgs only strips
// internal flags.
ipcMain.on('hermes:feature-flags', (event: IpcMainEvent): void => {
  event.returnValue = {
    ...resolveFeatureFlags({
      argv: process.argv,
      canary: resolveUpdaterChannelFromStamp() === 'canary'
    }),
    guestOnboarding: GUEST_ONBOARDING
  }
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
// blocker and the persisted mode so a cold launch restores it (applied on
// ready — powerSaveBlocker needs the app ready). The renderer picks the mode
// from Settings → Advanced over IPC. In 'while-working' the blocker follows the
// merged active-work picture that already drives stream throttling, so it is
// held exactly while a turn is in flight and released at idle. See
// store/keep-awake + power-save.ts.
const KEEP_AWAKE_CONFIG_PATH = path.join(app.getPath('userData'), 'keep-awake.json')
const keepAwake = createKeepAwake(powerSaveBlocker)
let keepAwakeMode: KeepAwakeMode = 'off'

function readPersistedKeepAwakeMode(): KeepAwakeMode {
  try {
    return readKeepAwakeMode(JSON.parse(fs.readFileSync(KEEP_AWAKE_CONFIG_PATH, 'utf8')))
  } catch {
    return 'off'
  }
}

// Reconcile the blocker with the mode and the live turn picture. Called from
// every mode change and every active-work report — both happen after app
// ready, when `keepAwake` and `keepAwakeMode` above are initialised.
function applyKeepAwake(working = isAnyTurnInFlight()) {
  keepAwake.set(keepAwakeWanted(keepAwakeMode, working))
}

ipcMain.on('hermes:keep-awake', (_event, value) => {
  // Accepts the mode string, or the boolean the pre-mode toggle sent.
  const mode = parseKeepAwakeMode(value)

  if (mode === null) {
    return
  }

  keepAwakeMode = mode
  applyKeepAwake()

  try {
    fs.mkdirSync(path.dirname(KEEP_AWAKE_CONFIG_PATH), { recursive: true })
    fs.writeFileSync(KEEP_AWAKE_CONFIG_PATH, JSON.stringify({ mode }, null, 2), 'utf8')
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
const quickEntrySubmitRelay = createQuickEntrySubmitRelay({
  // A late ack for a timed-out submit proves the outcome. Forward it to the
  // capture window so it can reconcile the unknown state instead of the user
  // resending a prompt that may already be delivered.
  onLateResult: (correlationId, result) => {
    if (quickEntryWindow && !quickEntryWindow.isDestroyed()) {
      quickEntryWindow.webContents.send('hermes:quick-entry:late-result', { correlationId, result })
    }
  },
  onSuccess: () => {
    hideQuickEntryWindow()

    if (process.platform === 'darwin') {
      app.dock?.show()
    }

    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.show()
      mainWindow.focus()
    }
  }
})

// Main owns the request lifecycle so the capture window can keep text until
// the primary renderer confirms delivery (#85590).
ipcMain.handle('hermes:quick-entry:submit', (event, payload) => {
  if (!quickEntryWindow || event.sender !== quickEntryWindow.webContents) {
    return { code: 'forbidden', message: 'Quick Entry sender is not authorized.', ok: false, retryable: false }
  }

  const text =
    typeof payload === 'string' ? payload.trim() : typeof payload?.text === 'string' ? payload.text.trim() : ''

  if (!text) {
    return { code: 'empty', message: 'Enter a prompt before submitting.', ok: false, retryable: false }
  }

  if (!mainWindow || mainWindow.isDestroyed()) {
    return { code: 'no-primary', message: 'The primary Hermes window is unavailable.', ok: false, retryable: true }
  }

  const target =
    typeof payload === 'object' && typeof payload?.target === 'string' && payload.target ? payload.target : 'current'

  return quickEntrySubmitRelay.begin(correlationId => {
    mainWindow.webContents.send('hermes:quick-entry:submit', { correlationId, target, text })
  })
})

// Main cannot invoke the primary renderer, so the primary returns by id. Stale
// or duplicate acknowledgements are intentionally ignored (#85590).
ipcMain.on('hermes:quick-entry:ack', (event, payload) => {
  if (!mainWindow || event.sender !== mainWindow.webContents) {
    return
  }

  quickEntrySubmitRelay.acknowledge(payload?.correlationId, payload?.result)
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

ipcMain.handle('hermes:openExternal', async (_event, url) => {
  const result = await openExternalUrl(url)

  if (result.ok === false && result.reason === 'invalid') {
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

// The preload reads this small, sanitized payload synchronously so the renderer
// can register the local skin before its first theme paint. It stays independent
// of the selected gateway, which can be an offline remote primary.
ipcMain.on('hermes:skin:local', event => {
  // The window route is more specific than the global next-launch preference:
  // a peer can be booting another profile while that preference changes.
  event.returnValue = readLocalSkinPayload(
    HERMES_HOME,
    windowConnectionRoutes.get(event.sender.id)?.profile,
    primaryProfileKey()
  )
})

// Renderer error toasts (notifyError): the toast shows the summarized copy,
// so the caller posts the full error here for desktop.log. Fire-and-forget,
// like renderer-error — the toast must never depend on this round-trip.
// Clamp: the line is renderer-supplied.
ipcMain.on('hermes:logs:renderer-line', (_event, line) => {
  const text = typeof line === 'string' ? line.slice(0, 6000) : ''

  if (!text) {
    return
  }

  rememberLog(text)
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
  sshBinary: desktopSshBinary,
  ensureBackend: webContentsId => ensureTerminalBackend(webContentsId),
  getSshConnectionState: scope => sshConnections.get(scope)
})

const disposeTerminalSession = terminalIpc.disposeTerminalSession

ipcMain.handle(
  'hermes:updates:check',
  async (_event: Electron.IpcMainInvokeEvent, opts?: { force?: boolean }): Promise<UpdaterStatusWire> =>
    checkUpdates({ force: Boolean(opts?.force) }).catch((error: Error): UpdaterStatusWire => ({
      supported: true,
      branch: readDesktopUpdateConfig().branch,
      error: 'check-failed',
      message: error?.message || String(error),
      fetchedAt: Date.now()
    }))
)

ipcMain.handle('hermes:updates:apply', async (_event, payload) =>
  applyUpdates().catch(error => ({
    ok: false,
    error: 'apply-failed',
    message: error?.message || String(error)
  }))
)

ipcMain.handle('hermes:updates:branch:get', async () => readDesktopUpdateConfig())

ipcMain.handle(
  'hermes:updates:branch:set',
  async (_event: Electron.IpcMainInvokeEvent, name: unknown): Promise<{ branch: string }> => {
    assertSourceUpdateChannel(INSTALL_STAMP)
    const branch: string = typeof name === 'string' && name.trim() ? name.trim() : DEFAULT_UPDATE_BRANCH
    writeDesktopUpdateConfig({ branch })

    return { branch }
  }
)

async function resolveDesktopClientVersion(): Promise<string> {
  // Fixed channel/bundled artifacts already carry their client identity in
  // the install stamp; only source/bootstrap installs need a local probe.
  if (INSTALL_STAMP?.payload && INSTALL_STAMP.payload !== 'bootstrap') {
    return ''
  }

  return resolveLocalRuntimeVersion(resolveUpdateRoot(), HERMES_HOME)
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
const checkRendererSkew = createBundleSkewChecker(
  INSTALL_STAMP,
  (args, options) => execGit(resolveGitBinary(), args, options),
  { isUpdating: () => updateGateReason(updateGateDeps()) !== null }
)

async function detectRendererSkew() {
  return checkRendererSkew(resolveUpdateRoot())
}

// Re-resolve the live Hermes version and push it into the native About panel
// just before showing it, so an in-place `hermes update` is reflected without
// an app restart. macOS only — `showAboutPanel()` is a no-op elsewhere, and the
// other platforms don't use this menu item.
function showAboutPanelFresh(): void {
  void Promise.all([detectRendererSkew(), resolveDesktopClientVersion()]).then(([skew, version]) => {
    const info: AppVersionInfo = appVersionInfo(INSTALL_STAMP, version, app.getVersion())
    // The product name already identifies canary and commit builds. Never pass
    // through empty/placeholder: the panel would render the bundle's 0.0.0 (#124581).
    const display: string = nativeAboutVersion(info)
    app.setAboutPanelOptions({
      applicationName: APP_NAME,
      applicationVersion: skew.outOfSync ? `${display} — app build out of date, update the desktop app` : display,
      copyright: 'Copyright © 2026 Nous Research'
    })
    app.showAboutPanel()
  })
}

ipcMain.handle('hermes:version', async (_event, _scope?: { connectionId?: string; profile?: string }) => {
  const [skew, version] = await Promise.all([detectRendererSkew(), resolveDesktopClientVersion()])

  return {
    ...appVersionInfo(INSTALL_STAMP, version, app.getVersion()),
    electronVersion: process.versions.electron,
    nodeVersion: process.versions.node,
    platform: process.platform,
    hermesRoot: resolveUpdateRoot(),
    hermesHome: HERMES_HOME,
    bundleOutOfSync: skew.outOfSync,
    bundleCommitsBehind: skew.desktopCommitsBehind,
    // The install id: sha16 of the canonical install-root path — the key of
    // this install's per-install channel record and its installs/<sha16>/
    // state folder. Same value `hermes update --install-id` prints; About
    // renders it as `sha16 (path)`.
    installId: installIdForRoot(resolveUpdateRoot(), canonicalizeInstallPath),
    // The artifact kind of THIS app plus whether the runtime checkout came
    // from a bootstrap installer script — About's Distribution row
    // disambiguates installer shells, bundles, and script installs from these.
    payload: INSTALL_STAMP?.payload,
    installedByScript: isInstallerCreatedCheckout(),
    // What this build carries and where an external backend runs from.
    // Bundled artifacts always run their payload; light artifacts have no
    // runtime and only reach remote backends. External builds classify from
    // the install stamp (git/docker/nix), 'unknown' when it can't be told.
    hermesRuntime: resolveHermesRuntime(),
    // True when the bundle on disk is not the one this process loaded — a
    // plain app restart (no rebuild, no installer) clears the skew above.
    // Packaged only: a dev `--build-only` rewrites build/install-stamp.json
    // under a running `npm start`, which is a rebuild the developer asked for,
    // not a torn install to offer a restart for.
    bundleSwapPending: IS_PACKAGED && detectBundleSwap(INSTALL_STAMP, readBundleSwapStamp(process.resourcesPath))
  }
})

// The About page's "Restart Hermes" button (shown when bundleSwapPending):
// load the already-swapped bundle without asking the user to quit manually.
// app.relaunch() re-executes by path, so the fresh process picks up whatever
// bundle now lives there.
ipcMain.handle('hermes:app:relaunch', async () => {
  rememberLog('[updates] renderer requested an app relaunch (swapped bundle pending)')
  app.relaunch({ args: buildNoSandboxRelaunchArgs(process.argv.slice(1)) })
  void exitAfterBackendShutdown(0)
})

/** The latest pm/venv/plugin-operation receipt — the machine-readable
 *  surface every medium reads (CLI: `hermes pm status`). Returned as one
 *  parsed JSON object: { kind, outcome, venv_rebuild, plugin_bisect,
 *  plugin_checks, ... } or null when no operation has run yet. The file
 *  lives at <HERMES_HOME>/logs/update_receipts/latest.json — written by
 *  pm syncs (bisect disables, failed rebuilds), plugin update checks,
 *  and (embedded) updates. */
function readLatestSyncReceipt(): Record<string, unknown> | null {
  const receiptPath = path.join(HERMES_HOME, 'logs', 'update_receipts', 'latest.json')

  try {
    const text = fs.readFileSync(receiptPath, 'utf8')
    // tolerate a BOM (hermes writes plain, but editors touch configs)
    const stripped = text.charCodeAt(0) === 0xfeff ? text.slice(1) : text

    return JSON.parse(stripped)
  } catch {
    return null
  }
}

ipcMain.handle('hermes:sync-status', () => readLatestSyncReceipt())

// Python's Path.resolve() equivalent for install-id derivation: realpath when
// the path exists, plain resolve otherwise. Must stay byte-compatible with
// boot_bootstrap._install_key or the CLI and the app would compute two
// different ids for one install.
function canonicalizeInstallPath(p: string): string {
  try {
    return fs.realpathSync(p)
  } catch {
    return path.resolve(p)
  }
}

/** True when the runtime checkout was created by a bootstrap installer
 *  (install.sh / install.ps1 / the desktop first-launch bootstrap): those all
 *  finish by writing `.hermes-bootstrap-complete` into the checkout root, and
 *  `hermes update` preserves the file, so a manual clone never grows one. A
 *  missing checkout (sealed bundled payloads, remote backends) is not
 *  installer-created. */
export function isInstallerCreatedCheckout(root: string | null = ACTIVE_HERMES_ROOT): boolean {
  if (!root) {
    return false
  }

  try {
    return fs.existsSync(path.join(root, path.basename(BOOTSTRAP_COMPLETE_MARKER)))
  } catch {
    return false
  }
}

/** Classify what this build carries (embedded / light / external). The stamp's
 *  `payload` decides the first two; an external build classifies its root via
 *  the canonical-root checkout stamp plus the bootstrap marker, or the app
 *  stamp's `source`, so About's Runtime row names
 *  git/docker/nix/desktop-bootstrap instead of a bare "external". */
function resolveHermesRuntime() {
  const stamp = INSTALL_STAMP as InstallStamp | null

  if (stamp?.payload === 'light') {
    return { type: 'light' }
  }

  if (stamp?.payload === 'bundled') {
    return { type: 'embedded' }
  }

  const root = resolveUpdateRoot()
  const canonicalStamp = readCanonicalInstallStamp()

  // A desktop first-launch bootstrap is attested by the bootstrap-complete
  // marker, not by the stamp: the stamp is the checkout's own identity
  // (source: git, written by the Python completion tail).
  if (canonicalStamp?.updateMechanism === 'self' && readBootstrapMarker()) {
    return { type: 'desktop-bootstrap', root }
  }

  const source = stamp?.source

  if (source === 'git') {
    return { type: 'git', root }
  }

  if (source === 'nix') {
    return { type: 'nix', root: null }
  }

  if (source === 'docker') {
    return { type: 'docker', root: null }
  }

  return { type: 'unknown' }
}

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
// The IPC boundary applies the baked install policy before either callback.
// Only self-managed installs use the Python summary or the cleanup script.

function uninstallVenvPython(): string {
  return getVenvPython(VENV_ROOT)
}

function fallbackUninstallSummary(): UninstallSummaryDetails {
  return {
    hermes_home: HERMES_HOME,
    agent_installed: isHermesSourceRoot(ACTIVE_HERMES_ROOT) && fileExists(uninstallVenvPython()),
    gui_installed: true,
    source_built_artifacts: [],
    packaged_app_paths: [],
    userdata_dir: app.getPath('userData'),
    userdata_exists: true,
    platform: process.platform,
    probe: 'fallback'
  }
}

async function probeUninstallSummary(): Promise<UninstallSummaryDetails> {
  const py: string = uninstallVenvPython()
  const agentRoot: string = ACTIVE_HERMES_ROOT

  if (!fileExists(py)) {
    return fallbackUninstallSummary()
  }

  return new Promise<UninstallSummaryDetails>((resolve: (value: UninstallSummaryDetails) => void): void => {
    let stdout: string = ''
    let settled: boolean = false

    const done: (value: UninstallSummaryDetails) => void = (value: UninstallSummaryDetails): void => {
      if (settled) {
        return
      }

      settled = true
      resolve(value)
    }

    try {
      const child: ChildProcess = spawn(
        py,
        ['-m', 'hermes_cli.main', 'uninstall', '--gui-summary'],
        hiddenWindowsChildOptions({
          cwd: agentRoot,
          env: { ...process.env, HERMES_HOME, NO_COLOR: '1' },
          stdio: ['ignore', 'pipe', 'ignore']
        })
      )

      child.stdout.on('data', (chunk: Buffer): void => {
        stdout += chunk.toString()
      })
      child.on('error', (): void => done(fallbackUninstallSummary()))
      child.on('exit', (code: number | null): void => {
        if (code !== 0) {
          return done(fallbackUninstallSummary())
        }

        try {
          const line: string = stdout.trim().split('\n').filter(Boolean).pop() || '{}'
          const parsed: UninstallSummaryDetails = JSON.parse(line)
          // The app bundle the renderer would be removing on *this* machine,
          // resolved from the running exe (the Python probe only knows the
          // standard locations, not where THIS build actually runs from).
          parsed.running_app_path = resolveRemovableAppPath(process.execPath, process.platform, process.env)
          done(parsed)
        } catch {
          done(fallbackUninstallSummary())
        }
      })
      setTimeout((): void => done(fallbackUninstallSummary()), 8000)
    } catch {
      done(fallbackUninstallSummary())
    }
  })
}

async function runDesktopUninstall(mode: string): Promise<DesktopUninstallResult> {
  let uninstallArgs: string[]

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
    const sysPy = await findSystemPython()

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

registerDesktopUninstallIpc({
  ipcMain,
  stamp: INSTALL_STAMP,
  fallbackSummary: fallbackUninstallSummary,
  probeSummary: probeUninstallSummary,
  runUninstall: runDesktopUninstall
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
// Set by sendOpenUpdatesRequested() when the renderer cannot hear it yet.
let _pendingOpenUpdates = false

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

  // Route the Windows Copilot hardware key (registered by the MSIX
  // copilotkeyprovider fragment). quick-entry is the eventual summon; for
  // now it falls through to the renderer's deep-link listener. stop and
  // unknown copilot-key paths are activation noise and must not reach the
  // renderer (a tap fires start+stop nearly together; acting on stop would
  // undo the summon).
  if (kind === 'copilot-key' && name !== 'start') {
    rememberLog(`[deeplink] ignoring copilot-key path: ${name}`)

    return
  }

  // hermes://close-preview — the out-of-band exit hatch for a preview pane
  // that fullscreened itself and now owns all input (#97213). Handled here
  // rather than in the renderer because the whole point is to work when the
  // renderer cannot hear anything: exit the fullscreen window and close the
  // pane from the main process.
  if (kind === 'close-preview') {
    if (mainWindow && !mainWindow.isDestroyed()) {
      if (mainWindow.isMinimized()) {
        mainWindow.restore()
      }

      // #130810: a tray-hidden primary reports invisible but not minimized;
      // without an explicit show the focus below lands on a hidden window.
      if (!mainWindow.isVisible()) {
        mainWindow.show()
      }

      mainWindow.focus()

      if (mainWindow.isFullScreen()) {
        mainWindow.setFullScreen(false)
      }

      sendClosePreviewRequested()
    }

    return
  }

  if (!_rendererReadyForDeepLink || !mainWindow || mainWindow.isDestroyed()) {
    _pendingDeepLink = payload

    return
  }

  try {
    // #130810 un-hide a tray-hidden window; #83998 no foreground re-pump.
    activateWindow(mainWindow)

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

  if (_pendingOpenUpdates) {
    _pendingOpenUpdates = false
    sendOpenUpdatesRequested()
  }

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

// macOS: register the deep link before the lock. Launch Services relaunches
// the app when the default protocol client changes; if the lock is already
// held, that relaunch flashes a second Dock icon and then app.exit(0)s.
// Win/Linux have no Dock and still register on ready.
const preReadyDockSteps = preReadyDockLaunchSteps(process.platform)

if (preReadyDockSteps.includes('register-deep-link')) {
  registerDeepLinkProtocol()
}

// Single-instance lock: deep links on a running app (Win/Linux) arrive as a
// second-instance argv. Without the lock a second `hermes://` launch spawns a
// whole new app instead of routing into the running one.
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
  // Cold-start --profile must win over the stored preference before
  // startHermes() reads active-profile.json. Only the instance that will
  // boot writes: a second launch must not retarget the running app. A missing
  // or invalid flag is a no-op, so the stored profile stays.
  try {
    applyLaunchProfileOverride(process.argv, name => {
      writeActiveDesktopProfile(name)
    })
  } catch (error) {
    console.error('[hermes] failed to persist --profile launch override:', error)
  }

  app.on('second-instance', (_event, argv) => {
    // --close-preview: the same escape hatch as hermes://close-preview, for
    // environments where spawning a URL is harder than a flag (kiosk launchers,
    // SSH-started sessions). Checked before deep links so a carried `hermes://`
    // URL still routes normally when no flag is present.
    if (hasClosePreviewFlag(argv)) {
      handleDeepLink('hermes://close-preview')
    }

    const url = _extractDeepLink(argv)

    if (url) {
      handleDeepLink(url)
    }

    // #130810: a second Start-menu / shortcut / Hermes.exe launch must never
    // silently exit. Log it so a future silent exit stays diagnosable, then
    // restore a live primary with activation (restore + show + focus), or
    // re-create it when it was destroyed.
    rememberLog(`[second-instance] relaunch (deepLink=${url ? 'yes' : 'no'})`)
    ensureMainWindow(mainWindow, {
      isReady: app.isReady(),
      createWindow,
      focusWindow: activateWindow,
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
  // Post-update relaunch detection (App Installer arm): when the previous
  // version wrote the one-shot pending-relaunch marker before quitting into
  // an OS package swap, consume it here — the renderer toasts "Hermes
  // updated to vX.Y.Z" once its bridge is up. Same-version markers (update
  // never landed) are deleted silently.
  const relaunchInfo: ConsumedRelaunch = consumePendingRelaunch(app, app.getVersion())

  if (relaunchInfo.wasUpdateRelaunch) {
    rememberLog(`[updates] post-update relaunch detected (from ${relaunchInfo.fromVersion})`)
  }

  // Warm the login-shell PATH resolution immediately so it usually completes
  // before the backend start path awaits the same single-flight promise.
  void ensureLoginShellPath()

  if (CRASH_DIAGNOSTICS) {
    startChromiumLogWatcher(CHROMIUM_LOG_PATH)
  }

  const systemCa = installSystemCaTrust(tls)

  if (systemCa.applied) {
    rememberLog(`[tls] trusting ${systemCa.systemCertificateCount} OS CA certificate(s) for backend connections`)
  } else if (systemCa.error) {
    rememberLog(`[tls] could not load OS system CA certificates: ${systemCa.error}`)
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

  // Expose the renderer's accessibility tree to the OS (#118271, Windows
  // twin #92607): dictation tools that insert text through the accessibility
  // APIs don't register as screen readers, so Chromium never builds the tree
  // and the composer stays invisible to them. Must run after `ready` (the
  // API's requirement). Opt out with desktop.renderer_accessibility: false
  // (bridged as HERMES_DESKTOP_RENDERER_ACCESSIBILITY=0); the platform/env
  // decision lives in the extracted helper.
  enableRendererAccessibility({ appApi: app })

  installMediaPermissions()
  installDownloadHandling()
  registerMediaProtocol()
  installEmbedReferer()
  installRemoteHeaderRules()

  if (!preReadyDockSteps.includes('register-deep-link')) {
    registerDeepLinkProtocol()
  }

  installPreviewGuestEscapeHatch()
  installPreviewGuestPreload()

  ensureWslWindowsFonts()
  configureSpellChecker()
  registerPowerResumeListeners()
  keepAwakeMode = readPersistedKeepAwakeMode()
  applyKeepAwake()
  void minimizeToTray.start()
  mainProcessLagWatchdog.start()
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
  installCommandScreenshot({ rendererUrl: DEV_SERVER || pathToFileURL(resolveRendererIndex()).toString() })
  installHudModifierTap({
    rendererUrl: DEV_SERVER || pathToFileURL(resolveRendererIndex()).toString(),
    summon: () => {
      if (!isQuittingForHandoff && !backendShutdown.hasStarted()) {
        openHudWindow(null, null)
      }
    }
  })

  if (IS_MAC) {
    const reposition = () => wakeIndicatorController.reposition()

    screen.on('display-added', reposition)

    screen.on('display-metrics-changed', reposition)

    screen.on('display-removed', reposition)
  }

  // The popped-out pet must never be stranded on a disconnected display: when
  // the topology changes, pull an off-screen overlay back onto the display
  // that holds the main window (and persist the corrected spot). Unlike the
  // wake indicator this applies on every platform — the pet overlay exists
  // everywhere, and rehomePetOverlay is a cheap no-op while the pet is in the
  // window or still on-screen.
  screen.on('display-added', rehomePetOverlay)

  screen.on('display-metrics-changed', rehomePetOverlay)

  screen.on('display-removed', rehomePetOverlay)

  // A hard crash can interrupt the in-memory restore loop after exact remote
  // serves were drained. The owner-only recovery journal survives that crash;
  // its worker waits for the install marker to clear, then reopens every scope
  // captured by the original transaction before removing the journal entry.
  void resumeManagedSshRecoveries()
  installApplicationMenuAfterFirstWindow({
    isMac: IS_MAC,
    buildMenu: buildApplicationMenu,
    setApplicationMenu: menu => Menu.setApplicationMenu(menu),
    createWindow
  })

  // Win/Linux cold start: the launching hermes:// URL is in our own argv.
  const _coldStartLink = _extractDeepLink(process.argv)

  if (_coldStartLink) {
    handleDeepLink(_coldStartLink)
  }

  app.on('activate', () => {
    // Recreate the primary window if it's gone. Guard on mainWindow directly
    // (not just total window count) so a dock click still restores the main
    // window when only secondary session windows remain open.
    // #130810: a dock/taskbar activate is explicit, so restore with
    // activation (restore + show + focus), not the ambient showInactive path.
    if (!mainWindow || mainWindow.isDestroyed()) {
      createWindow()
    } else {
      activateWindow(mainWindow)
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

// Does quitting take the agent down with the app? Reads the primary profile's
// route through the same resolver resolveRemoteBackend uses, plus every backend
// the quit teardown below will stop (spawned children, SSH-managed servers).
// A route we can't resolve counts as owned: the lost-work warning is the safe
// side to be wrong on.
function quitStopsBackendWork(): boolean {
  let primaryRouteKind: 'cloud' | 'remote' | 'ssh' | null

  try {
    primaryRouteKind =
      resolveDesktopRemoteRoute({
        config: readDesktopConnectionConfig(),
        env: {
          token: process.env.HERMES_DESKTOP_REMOTE_TOKEN,
          url: process.env.HERMES_DESKTOP_REMOTE_URL
        },
        profile: primaryProfileKey(),
        registry: readDesktopConnectionsRegistry()
      })?.kind ?? null
  } catch {
    return true
  }

  const ownedBackendCount =
    (backendConnectionState.getProcess() ? 1 : 0) +
    [...backendPool.values()].filter(entry => entry?.process).length +
    sshConnections.size

  return backendOwnedByApp({ ownedBackendCount, primaryRouteKind })
}

// Ask before a quit kills a turn in flight. True when the quit was intercepted
// and the confirmation is on screen; the confirm button re-enters before-quit with
// the latch set and falls straight through to the teardown below.
function heldQuitForActiveWork(event: Electron.Event): boolean {
  if (SKIP_QUIT_CONFIRM || quitConfirmedWithActiveWork || isQuittingForHandoff) {
    return false
  }

  if (quitPromptOpen) {
    event.preventDefault()

    return true
  }

  // The per-webContents map can read empty at quit time even though a turn
  // is live (a stream can reload its webContents mid-turn, dropping the entry
  // before the guard runs), so merge in the last summary any renderer sent.
  const work = mergeActiveWork([...activeWorkByWebContents.values(), lastActiveWorkSeen])

  const prompt = quitPromptFor(work, isQuittingForHandoff, quitStopsBackendWork())

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
      buttons: [...prompt.buttons],
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

// Intercept the close of the LAST chat window while the window and its
// active-work report are still alive. On Windows/Linux the primary quit
// gesture is the title-bar close button: closing the final window destroys
// its webContents (clearing the active-work map) BEFORE window-all-closed
// reactively calls app.quit() — by the time before-quit runs, heldQuitForActiveWork
// finds nothing and the app exits silently (#96139). Running the same guard
// here, on the close event itself, catches it in time; "Quit Anyway" re-enters
// before-quit with the latch set and falls through.
function registerChatWindow(window: BrowserWindow) {
  chatWindows.add(window)
  window.on('close', (event: Electron.Event) => {
    // The tray's close-to-tray handler runs first (registered first) and
    // absorbs the close into a hide — the work keeps running, so there is
    // nothing to confirm.
    if (event.defaultPrevented) {
      return
    }

    const work = mergeActiveWork(activeWorkByWebContents.values())

    if (!shouldGuardWindowClose(work, isQuittingForHandoff, IS_MAC, hasOtherChatWindows(window))) {
      return
    }

    heldQuitForActiveWork(event)
  })
  window.once('closed', () => {
    chatWindows.delete(window)
    quitIfNoSurfaceLeft()
  })
}

// Last-surface fallback (#130810), run when a chat window OR a popped-out
// Browser window closes: popped-out Browser windows are user-visible surfaces
// too, so they keep the app alive — and closing the last one must quit.
function quitIfNoSurfaceLeft() {
  if (
    shouldQuitOnLastChatClosed({
      platform: process.platform,
      isQuittingForHandoff,
      remainingChatWindows: chatWindows.size + browserWindows.size,
      quitInProgress
    })
  ) {
    app.quit()
  }
}

app.on('before-quit', event => {
  // Runs ahead of every teardown below, so "Keep Running" leaves the app
  // exactly as it was: a held quit is not a quit in progress, and the
  // overlay-suppression latch must not leak into the next close (#130810).
  if (heldQuitForActiveWork(event)) {
    appQuitting = false
    quitInProgress = false

    return
  }

  // Quit is really proceeding: latch both before ANY teardown below closes
  // the pet overlay — its 'closed' handler must not echo pop-in during quit,
  // or the persisted popped-out state is wiped and the overlay never restores
  // (#55920).
  appQuitting = true
  quitInProgress = true

  minimizeToTray.beginQuit()
  mainProcessLagWatchdog.stop()

  // A detached remote updater can outlive this Electron process. Do not tear
  // down its SSH observer/restore transaction at the generic SSH shutdown
  // deadline: join it first (BEFORE sealing the bootstrap coordinator, whose
  // shutdown would refuse the restore dials), then re-enter before-quit for
  // normal teardown. A crash still fails closed on next launch via the remote
  // install-marker preflight in both POSIX and Windows lifecycle
  // implementations.
  if (
    !managedUpdateQuitWaitDone &&
    (managedUpdateQuitWait || managedConnectionUpdates.size > 0 || managedConnectionRecoveries.size > 0)
  ) {
    event.preventDefault()

    if (!managedUpdateQuitWait) {
      managedUpdateQuitWait = waitForManagedUpdateOperations(() => [
        ...managedConnectionUpdates.values(),
        ...managedConnectionRecoveries.values()
      ]).finally(() => {
        managedUpdateQuitWaitDone = true
        app.quit()
      })
    }

    return
  }

  // A prevented first quit leaves the renderer alive while teardown runs.
  // Seal the SSH coordinator before touching connections so reconnect
  // callbacks cannot recreate a backend for a registration whose app is
  // already quitting (#91668).
  sshBootstrapCoordinator.shutdown()

  const backendNeedsWait = backendQuitNeedsWait({
    connectionPending: backendConnectionState.getPendingPromise() !== null || localBackendLifecycle.hasPending(),
    poolPending: poolStopper.hasPending(),
    processAttached: backendConnectionState.getProcess() !== null,
    shutdownPending: backendShutdown.isPending()
  })

  const sshNeedsWait =
    sshConnections.size > 0 || sshBootstrapCoordinator.promises().length > 0 || sshTeardowns.hasPending()

  const teardownTasks: QuitTeardownTask[] = [
    { run: (): Promise<void> => backendShutdown.run(), waitForCompletion: backendNeedsWait }
  ]

  if (sshNeedsWait) {
    teardownTasks.push({ run: teardownSshForQuit, waitForCompletion: true })
  }

  if (quitTeardown.begin(teardownTasks)) {
    event.preventDefault()
  }

  // Clean quit mid-boot should not trip next-launch --no-sandbox (#38216).
  // FATAL GPU aborts skip before-quit, leaving the `booting` marker in place.
  // Keyed on sticky (not active): a manual --no-sandbox run still records a
  // clean quit, while an engaged fallback keeps its sticky marker.
  if ((IS_WINDOWS || process.platform === 'linux') && !windowsSandboxFallbackSticky) {
    try {
      writeSandboxMarker(app.getPath('userData'), markerAfterSuccessfulBoot({ fallbackActive: false }))
    } catch {
      void 0
    }
  }

  // #124843: a clean quit mid-boot must not trip next-launch --disable-gpu.
  // Keyed on sticky (not active) so an engaged fallback keeps its marker.
  if (process.platform === 'linux' && !linuxGpuFallbackSticky) {
    try {
      writeLinuxGpuMarker(app.getPath('userData'), linuxGpuMarkerAfterSuccessfulBoot({ fallbackActive: false }))
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
