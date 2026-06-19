export {}

declare global {
  interface Window {
    hermesDesktop: {
      // Resolve a backend connection. Omit `profile` (or pass the primary) for
      // the window's backend; pass a named profile to lazily spawn/reuse that
      // profile's backend from the pool.
      getConnection: (profile?: string | null) => Promise<HermesConnection>
      // Reconnect-after-wake recovery: liveness-probe the cached PRIMARY backend
      // and drop it if a remote one has gone unreachable, so the next
      // getConnection() rebuilds a reachable descriptor instead of the renderer
      // re-dialing a dead remote forever. No-op for local backends (they
      // self-heal via the child 'exit' handler). `rebuilt` is true when a stale
      // remote cache was dropped.
      revalidateConnection: () => Promise<{ ok: boolean; rebuilt: boolean }>
      // Keepalive: mark a pool profile backend as recently used so the idle
      // reaper spares it while its chat is active.
      touchBackend: (profile?: string | null) => Promise<{ ok: boolean }>
      getGatewayWsUrl: (profile?: null | string) => Promise<string>
      // Open (or focus) a standalone OS window for a single chat session so
      // the user can work with multiple chats side by side. Returns ok:false
      // with an error code when the sessionId is empty/invalid. `watch` opens
      // a spectator window (lazy resume — no agent build) for live-streaming
      // a running subagent's session.
      openSessionWindow: (sessionId: string, opts?: { watch?: boolean }) => Promise<{ ok: boolean; error?: string }>
      // Open (or focus) a compact secondary window on the new-session draft.
      openNewSessionWindow: () => Promise<{ ok: boolean; error?: string }>
      getBootProgress: () => Promise<DesktopBootProgress>
      getConnectionConfig: (profile?: null | string) => Promise<DesktopConnectionConfig>
      saveConnectionConfig: (payload: DesktopConnectionConfigInput) => Promise<DesktopConnectionConfig>
      applyConnectionConfig: (payload: DesktopConnectionConfigInput) => Promise<DesktopConnectionConfig>
      testConnectionConfig: (payload: DesktopConnectionConfigInput) => Promise<DesktopConnectionTestResult>
      probeConnectionConfig: (remoteUrl: string) => Promise<DesktopConnectionProbeResult>
      oauthLoginConnectionConfig: (remoteUrl: string) => Promise<DesktopOauthLoginResult>
      oauthLogoutConnectionConfig: (remoteUrl?: string) => Promise<DesktopOauthLogoutResult>
      profile: {
        get: () => Promise<DesktopActiveProfile>
        // Persists the desktop's profile choice and relaunches the local
        // backend under the new HERMES_HOME (reloads the window). Pass null to
        // clear the preference.
        set: (name: string | null) => Promise<DesktopActiveProfile>
      }
      api: <T>(request: HermesApiRequest) => Promise<T>
      notify: (payload: HermesNotification) => Promise<boolean>
      requestMicrophoneAccess: () => Promise<boolean>
      readFileDataUrl: (filePath: string) => Promise<string>
      readFileText: (filePath: string) => Promise<HermesReadFileTextResult>
      selectPaths: (options?: HermesSelectPathsOptions) => Promise<string[]>
      writeClipboard: (text: string) => Promise<boolean>
      saveImageFromUrl: (url: string) => Promise<boolean>
      saveImageBuffer: (data: ArrayBuffer | Uint8Array, ext: string) => Promise<string>
      saveClipboardImage: () => Promise<string>
      getPathForFile: (file: File) => string
      normalizePreviewTarget: (target: string, baseDir?: string) => Promise<HermesPreviewTarget | null>
      watchPreviewFile: (url: string) => Promise<HermesPreviewWatch>
      stopPreviewFileWatch: (id: string) => Promise<boolean>
      setTitleBarTheme?: (payload: HermesTitleBarTheme) => void
      setNativeTheme?: (mode: 'dark' | 'light' | 'system') => void
      setTranslucency?: (payload: { intensity: number }) => void
      setPreviewShortcutActive?: (active: boolean) => void
      openExternal: (url: string) => Promise<void>
      fetchLinkTitle: (url: string) => Promise<string>
      sanitizeWorkspaceCwd: (cwd?: null | string) => Promise<{ cwd: string; sanitized: boolean }>
      settings: {
        getDefaultProjectDir: () => Promise<{ defaultLabel: string; dir: null | string; resolvedCwd: string }>
        pickDefaultProjectDir: () => Promise<{ canceled: boolean; dir: null | string }>
        setDefaultProjectDir: (dir: null | string) => Promise<{ dir: null | string }>
      }
      revealLogs: () => Promise<{ ok: boolean; path: string; error?: string }>
      getRecentLogs: () => Promise<{ path: string; lines: string[] }>
      readDir: (path: string) => Promise<HermesReadDirResult>
      gitRoot?: (path: string) => Promise<string | null>
      // Resolve git-worktree identity for a batch of session cwds, reading git's
      // on-disk metadata locally. Returns null per cwd that isn't inside a
      // checkout (or can't be read — e.g. a remote backend's path).
      worktrees?: (cwds: string[]) => Promise<Record<string, HermesWorktreeInfo | null>>
      // Git source control for the desktop Source Control panel. All ops resolve
      // + harden the cwd to a git root in the main process and spawn `git` with
      // an argument array (no shell). `ok:false` carries a human-readable error.
      git?: {
        status: (cwd: string) => Promise<HermesGitStatus>
        diff: (cwd: string, filePath: string, staged: boolean) => Promise<{ ok: boolean; diff?: string; error?: string }>
        stage: (cwd: string, paths: string[]) => Promise<HermesGitResult>
        unstage: (cwd: string, paths: string[]) => Promise<HermesGitResult>
        discard: (cwd: string, paths: string[]) => Promise<HermesGitResult>
        commit: (cwd: string, message: string, options?: { amend?: boolean; all?: boolean }) => Promise<HermesGitResult>
        push: (cwd: string, options?: { setUpstream?: boolean }) => Promise<HermesGitResult>
        pull: (cwd: string) => Promise<HermesGitResult>
        fetch: (cwd: string) => Promise<HermesGitResult>
        branches: (cwd: string) => Promise<HermesGitBranches>
        checkout: (cwd: string, branch: string) => Promise<HermesGitResult>
        createBranch: (cwd: string, name: string, options?: { startPoint?: string }) => Promise<HermesGitResult>
        deleteBranch: (cwd: string, name: string, options?: { force?: boolean }) => Promise<HermesGitResult>
        log: (cwd: string, options?: { limit?: number }) => Promise<HermesGitLog>
        commitDiff: (cwd: string, sha: string) => Promise<{ ok: boolean; diff?: string; error?: string }>
        stashList: (cwd: string) => Promise<HermesGitStashList>
        stashPush: (
          cwd: string,
          options?: { includeUntracked?: boolean; message?: string }
        ) => Promise<HermesGitResult>
        stashAction: (cwd: string, action: 'apply' | 'drop' | 'pop', ref: string) => Promise<HermesGitResult>
        applyHunk: (cwd: string, patch: string, options?: { reverse?: boolean }) => Promise<HermesGitResult>
      }
      terminal: {
        dispose: (id: string) => Promise<boolean>
        onData: (id: string, callback: (payload: string) => void) => () => void
        onExit: (id: string, callback: (payload: HermesTerminalExit) => void) => () => void
        resize: (id: string, size: { cols: number; rows: number }) => Promise<boolean>
        start: (options?: { cols?: number; cwd?: string; rows?: number }) => Promise<HermesTerminalSession>
        write: (id: string, data: string) => Promise<boolean>
      }
      onClosePreviewRequested?: (callback: () => void) => () => void
      onOpenUpdatesRequested?: (callback: () => void) => () => void
      onDeepLink?: (
        callback: (payload: { kind: string; name: string; params: Record<string, string> }) => void
      ) => () => void
      signalDeepLinkReady?: () => Promise<{ ok: boolean }>
      onWindowStateChanged?: (callback: (payload: HermesWindowState) => void) => () => void
      onFocusSession?: (callback: (sessionId: string) => void) => () => void
      onNotificationAction?: (callback: (payload: { actionId: string; sessionId?: string }) => void) => () => void
      onPreviewFileChanged: (callback: (payload: HermesPreviewFileChanged) => void) => () => void
      onBackendExit: (callback: (payload: BackendExit) => void) => () => void
      onPowerResume?: (callback: () => void) => () => void
      onBootProgress: (callback: (payload: DesktopBootProgress) => void) => () => void
      getBootstrapState: () => Promise<DesktopBootstrapState>
      resetBootstrap: () => Promise<{ ok: boolean }>
      repairBootstrap: () => Promise<{ ok: boolean }>
      cancelBootstrap: () => Promise<{ ok: boolean; cancelled: boolean }>
      onBootstrapEvent: (callback: (payload: DesktopBootstrapEvent) => void) => () => void
      getVersion: () => Promise<DesktopVersionInfo>
      updates: {
        check: () => Promise<DesktopUpdateStatus>
        apply: (opts?: DesktopUpdateApplyOptions) => Promise<DesktopUpdateApplyResult>
        getBranch: () => Promise<{ branch: string }>
        setBranch: (name: string) => Promise<{ branch: string }>
        onProgress: (callback: (payload: DesktopUpdateProgress) => void) => () => void
      }
      uninstall: {
        summary: () => Promise<DesktopUninstallSummary>
        run: (mode: DesktopUninstallMode) => Promise<DesktopUninstallResult>
      }
      themes: {
        // Download a VS Code Marketplace extension and return the raw color
        // theme files it contributes. The renderer converts + persists them.
        fetchMarketplace: (id: string) => Promise<DesktopMarketplaceThemeResult>
        // Search the Marketplace for color-theme extensions. An empty query
        // returns the most-installed themes.
        searchMarketplace: (query: string) => Promise<DesktopMarketplaceSearchItem[]>
      }
    }
  }
}

export interface DesktopMarketplaceSearchItem {
  extensionId: string
  displayName: string
  publisher: string
  description: string
  installs: number
}

export interface DesktopMarketplaceThemeFile {
  label: string
  /** VS Code's `uiTheme` for this entry (vs-dark / vs / hc-black). */
  uiTheme?: string
  /** Raw theme JSON (JSONC) text, parsed + converted by the renderer. */
  contents: string
}

export interface DesktopMarketplaceThemeResult {
  extensionId: string
  displayName: string
  themes: DesktopMarketplaceThemeFile[]
}

export interface HermesTerminalSession {
  cwd: string
  id: string
  shell: string
}

export interface HermesTerminalExit {
  code: number | null
  signal: string | null
}

export interface DesktopVersionInfo {
  appVersion: string
  electronVersion: string
  nodeVersion: string
  platform: string
  hermesRoot: string
}

export type DesktopUninstallMode = 'full' | 'gui' | 'lite'

export interface DesktopUninstallSummary {
  hermes_home: string
  agent_installed: boolean
  gui_installed: boolean
  source_built_artifacts: string[]
  packaged_app_paths: string[]
  userdata_dir: string
  userdata_exists: boolean
  platform: string
  running_app_path?: null | string
  probe?: string
}

export interface DesktopUninstallResult {
  ok: boolean
  mode?: DesktopUninstallMode
  willRemoveAppBundle?: boolean
  scriptPath?: string
  error?: string
  message?: string
}

export interface DesktopUpdateCommit {
  sha: string
  summary: string
  author: string
  at: number
}

export interface DesktopUpdateStatus {
  supported: boolean
  branch?: string
  currentBranch?: string
  reason?: string
  message?: string
  error?: string
  behind?: number
  currentSha?: string
  targetSha?: string
  commits?: DesktopUpdateCommit[]
  dirty?: boolean
  fetchedAt?: number
}

export type DesktopUpdateDirtyStrategy = 'abort' | 'stash' | 'force'

export interface DesktopUpdateApplyOptions {
  dirtyStrategy?: DesktopUpdateDirtyStrategy
}

export interface DesktopUpdateApplyResult {
  ok: boolean
  branch?: string
  error?: string
  message?: string
  /** True when no staged updater exists (CLI install) and the user should run
   *  `hermes update` themselves. `command` is the exact line to run. */
  manual?: boolean
  command?: string
  hermesRoot?: string
}

export type DesktopUpdateStage = 'idle' | 'prepare' | 'fetch' | 'pull' | 'pydeps' | 'restart' | 'manual' | 'error'

export interface DesktopUpdateProgress {
  stage: DesktopUpdateStage
  message: string
  percent: number | null
  error: string | null
  at: number
}

export interface HermesConnection {
  baseUrl: string
  isFullscreen: boolean
  mode?: 'local' | 'remote'
  authMode?: 'oauth' | 'token'
  nativeOverlayWidth: number
  source?: 'env' | 'local' | 'settings'
  token: string
  wsUrl: string
  logs: string[]
  // Set for pool (non-primary) backends so the renderer knows which profile a
  // connection belongs to.
  profile?: string
  windowButtonPosition: { x: number; y: number } | null
}

export interface HermesTitleBarTheme {
  background: string
  foreground: string
}

export interface HermesWindowState {
  isFullscreen: boolean
  nativeOverlayWidth: number
  windowButtonPosition: { x: number; y: number } | null
}

export interface DesktopActiveProfile {
  // The desktop's stored profile preference, or null when unset (legacy launch
  // that defers to the sticky active_profile / default).
  profile: string | null
}

export interface DesktopConnectionConfig {
  envOverride: boolean
  mode: 'local' | 'remote'
  // The profile this config describes, or null for the global/default
  // connection. Per-profile entries let a profile point at its own backend.
  profile: null | string
  remoteAuthMode: 'oauth' | 'token'
  remoteOauthConnected: boolean
  remoteTokenPreview: string | null
  remoteTokenSet: boolean
  remoteUrl: string
}

export interface DesktopConnectionConfigInput {
  mode: 'local' | 'remote'
  // When set, the save/apply/test targets this profile's per-profile remote
  // override instead of the global connection.
  profile?: null | string
  remoteAuthMode?: 'oauth' | 'token'
  remoteToken?: string
  remoteUrl?: string
}

export interface DesktopConnectionTestResult {
  baseUrl: string
  ok: boolean
  version: string | null
}

export interface DesktopAuthProvider {
  name: string
  displayName: string
  // True when this provider authenticates with a username + password
  // (the gateway's /login page renders a credential form) rather than an
  // OAuth redirect. The session/cookie/ws-ticket machinery is identical;
  // only the login-page form and the desktop's button copy differ.
  supportsPassword?: boolean
}

export interface DesktopConnectionProbeResult {
  baseUrl: string
  reachable: boolean
  authMode: 'oauth' | 'token' | 'unknown'
  providers: DesktopAuthProvider[]
  version: string | null
  error: string | null
}

export interface DesktopOauthLoginResult {
  ok: boolean
  baseUrl: string
  connected: boolean
}

export interface DesktopOauthLogoutResult {
  ok: boolean
  connected: boolean
}

export interface DesktopBootProgress {
  error: string | null
  fakeMode: boolean
  message: string
  phase: string
  progress: number
  running: boolean
  timestamp: number
}

// First-launch install ("bootstrap") event types -- emitted by
// electron/bootstrap-runner.cjs and observed by the renderer install overlay.
// Mirrors the event shapes emitted by runBootstrap()'s onEvent callback.

export interface DesktopBootstrapStageDescriptor {
  name: string
  title?: string
  category?: string
  needs_user_input?: boolean
}

export type DesktopBootstrapStageState = 'pending' | 'running' | 'succeeded' | 'skipped' | 'failed'

export interface DesktopBootstrapStageResult {
  state: DesktopBootstrapStageState
  durationMs: number | null
  startedAt: number | null
  json: { ok: boolean; skipped?: boolean; reason?: string | null; stage: string } | null
  error: string | null
}

export interface DesktopBootstrapUnsupportedPlatform {
  platform: string
  activeRoot: string
  installCommand: string
  docsUrl: string
}

export interface DesktopBootstrapState {
  active: boolean
  manifest: { type: 'manifest'; stages: DesktopBootstrapStageDescriptor[]; protocolVersion: number | null } | null
  stages: Record<string, DesktopBootstrapStageResult>
  error: string | null
  log: Array<{ ts: number; stage: string | null; line: string; stream?: 'stdout' | 'stderr' }>
  startedAt: number | null
  completedAt: number | null
  unsupportedPlatform: DesktopBootstrapUnsupportedPlatform | null
}

export type DesktopBootstrapEvent =
  | { type: 'manifest'; stages: DesktopBootstrapStageDescriptor[]; protocolVersion: number | null }
  | {
      type: 'stage'
      name: string
      state: DesktopBootstrapStageState
      durationMs?: number
      json?: DesktopBootstrapStageResult['json']
      error?: string | null
    }
  | { type: 'log'; stage?: string | null; line: string; stream?: 'stdout' | 'stderr' }
  | { type: 'complete'; marker: Record<string, unknown> }
  | { type: 'failed'; stage?: string | null; error: string }
  | {
      type: 'unsupported-platform'
      platform: string
      activeRoot: string
      installCommand: string
      docsUrl: string
    }

export interface HermesApiRequest {
  path: string
  method?: string
  body?: unknown
  timeoutMs?: number
  // Route this REST call to a specific profile's backend. Omit for the primary
  // (window) backend. Read-only cross-profile data is served by the primary, so
  // this is only needed for profile-scoped live/settings calls.
  profile?: string | null
}

export interface HermesNotification {
  title?: string
  body?: string
  silent?: boolean
  kind?: string
  sessionId?: string
  actions?: { id: string; text: string }[]
}

export interface HermesPreviewTarget {
  binary?: boolean
  byteSize?: number
  kind: 'file' | 'url'
  label: string
  large?: boolean
  language?: string
  mimeType?: string
  path?: string
  previewKind?: 'binary' | 'html' | 'image' | 'text'
  renderMode?: 'preview' | 'source'
  source: string
  url: string
}

export interface HermesReadFileTextResult {
  binary?: boolean
  byteSize?: number
  language?: string
  mimeType?: string
  path: string
  text: string
  truncated?: boolean
}

export interface HermesPreviewWatch {
  id: string
  path: string
}

export interface HermesGitFile {
  path: string
  origPath?: string
  status: 'added' | 'conflicted' | 'copied' | 'deleted' | 'modified' | 'renamed' | 'typechange' | 'untracked'
  staged: boolean
}

export interface HermesGitStatus {
  ok: boolean
  error?: string
  root?: string
  branch?: null | string
  upstream?: null | string
  ahead?: number
  behind?: number
  staged?: HermesGitFile[]
  unstaged?: HermesGitFile[]
  untracked?: HermesGitFile[]
  conflicted?: HermesGitFile[]
}

export interface HermesGitResult {
  ok: boolean
  error?: string
  output?: string
}

export interface HermesGitBranch {
  name: string
  current: boolean
  upstream: null | string
  ahead: number
  behind: number
  sha: string
  subject: string
}

export interface HermesGitBranches {
  ok: boolean
  error?: string
  current?: null | string
  local?: HermesGitBranch[]
  remote?: HermesGitBranch[]
}

export interface HermesGitCommit {
  sha: string
  shortSha: string
  author: string
  authorEmail: string
  date: string
  relativeDate: string
  subject: string
}

export interface HermesGitLog {
  ok: boolean
  error?: string
  commits?: HermesGitCommit[]
}

export interface HermesGitStash {
  ref: string
  subject: string
}

export interface HermesGitStashList {
  ok: boolean
  error?: string
  stashes?: HermesGitStash[]
}

export interface HermesWorktreeInfo {
  // Main repo root — the shared grouping key for a checkout and all its linked
  // worktrees.
  repoRoot: string
  // This cwd's own worktree root.
  worktreeRoot: string
  // True when this is the repo's primary checkout (.git is a directory).
  isMainWorktree: boolean
  // Current branch (or short detached-HEAD sha), null when unreadable.
  branch: null | string
}

export interface HermesReadDirEntry {
  name: string
  path: string
  isDirectory: boolean
}

export interface HermesReadDirResult {
  entries: HermesReadDirEntry[]
  error?: string
}

export interface HermesPreviewFileChanged {
  id: string
  path: string
  url: string
}

export interface HermesSelectPathsOptions {
  title?: string
  defaultPath?: string
  directories?: boolean
  multiple?: boolean
  filters?: Array<{ name: string; extensions: string[] }>
}

export interface BackendExit {
  code: number | null
  signal: string | null
}
