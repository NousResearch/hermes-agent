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
      account?: {
        status: () => Promise<DesktopAccountStatus>
        me: () => Promise<DesktopAccountMe>
        login: (payload: {
          cloudBaseUrl: string
          email: string
          password: string
        }) => Promise<DesktopAccountLoginResult>
        register: (payload: {
          cloudBaseUrl: string
          email: string
          password: string
        }) => Promise<DesktopAccountLoginResult>
        logout: () => Promise<{ ok: boolean }>
        usage: (opts?: { limit?: number; offset?: number; kind?: string }) => Promise<DesktopAccountUsage>
        wallet: () => Promise<DesktopAccountWallet>
        transactions: (opts?: { limit?: number; offset?: number }) => Promise<DesktopAccountTransactions>
        payConfig: () => Promise<DesktopAccountPayConfig>
        createOrder: (yuan: number) => Promise<DesktopAccountPayOrder>
        mockConfirm: (orderId: string) => Promise<DesktopAccountMockConfirmResult>
        redeem: (code: string) => Promise<DesktopAccountRedeemResult>
        subtree: () => Promise<DesktopAccountSubtree>
        createSubaccount: (payload: {
          email: string
          name?: string
          password: string
        }) => Promise<DesktopAccountSubaccount>
        createRelation: (payload: {
          manager_id: string
          member_id: string
        }) => Promise<DesktopAccountEdge>
        roles: () => Promise<DesktopAccountRoles>
        addRole: (payload: { name: string }) => Promise<DesktopAccountRoles>
        removeRole: (payload: { name: string }) => Promise<DesktopAccountRoles>
        setRole: (payload: { user_id: string; role: string }) => Promise<DesktopAccountRoleResult>
        changePassword: (payload: {
          old_password: string
          new_password: string
        }) => Promise<{ ok: boolean }>
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
      workflow?: {
        start: () => Promise<DesktopWorkflowBackendStatus>
        stop: () => Promise<DesktopWorkflowBackendStatus>
        status: () => Promise<DesktopWorkflowBackendStatus>
        totalMemoryGb: () => Promise<number>
        authStatus: () => Promise<DesktopWorkflowAuthStatus>
        // Fired after the main process restarts the Langflow backend (e.g. on
        // account login/logout) so the canvas can reload with the new token state.
        onRestarted: (callback: (payload: { reason?: string }) => void) => () => void
      }
      knowledge?: {
        inventory: (dirPath: string) => Promise<KnowledgeInventory>
        ingest: (payload: { folderPath: string; name: string }) => Promise<KnowledgeIngestResult>
        list: () => Promise<KnowledgeSource[]>
        remove: (sourceId: string) => Promise<{ ok: boolean; removed?: boolean }>
        sync: (sourceId: string) => Promise<KnowledgeSyncResult>
        onIngestProgress: (callback: (progress: KnowledgeIngestProgress) => void) => () => void
      }
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

export interface DesktopWorkflowBackendStatus {
  error: string | null
  external: boolean
  pid: null | number
  root: string
  state: 'error' | 'exited' | 'ready' | 'starting' | 'stopped'
  url: string
}

export interface DesktopWorkflowAuthStatus {
  loggedIn: boolean
  cloudBaseUrl: string
  cloudReachable?: boolean
  error?: string | null
}

export interface DesktopAccountStatus extends DesktopWorkflowAuthStatus {
  balance?: number
  email?: string
  username?: string
}

export interface DesktopAccountMe {
  user_id: string
  email: string
  name?: null | string
  parent_id?: null | string
  is_admin?: boolean
  balance: number
}

export interface DesktopAccountLoginResult {
  ok: boolean
  error?: string
  balance?: number
  email?: string
  username?: string
}

export interface DesktopAccountUsageItem {
  user_id?: string
  ts?: number
  kind?: string
  credits?: number
  provider?: string | null
  model?: string | null
  note?: string | null
}

export interface DesktopAccountUsage {
  ok: boolean
  error?: string
  items: DesktopAccountUsageItem[]
  total?: number
  balance?: number
  creditRmb?: number
}

export interface DesktopAccountWallet {
  balance: number
  credit_rmb: number
}

export interface DesktopAccountTransactionItem {
  ts?: number
  delta: number
  kind: string
  note?: null | string
  balance_after: number
}

export interface DesktopAccountTransactions {
  items: DesktopAccountTransactionItem[]
  total?: number
  balance?: number
}

export interface DesktopAccountPayConfig {
  enabled: boolean
  mock: boolean
  quick_yuan: number[]
  min_yuan: number
  credits_per_yuan: number
}

export interface DesktopAccountPayOrder {
  order_id: string
  user_id?: string
  aoid?: null | string
  yuan: number
  credits: number
  status?: string
  qr_image_url?: null | string
  mock?: boolean
  expire_in?: number
  created_ts?: number
  paid_ts?: null | number
}

export interface DesktopAccountMockConfirmResult {
  paid: boolean
  credited: boolean
  balance: number
}

export interface DesktopAccountRedeemResult {
  added: number
  balance: number
}

export interface DesktopAccountTreeNode {
  user_id: string
  email: string
  name?: null | string
  parent_id?: null | string
  role?: null | string
}

export interface DesktopAccountRoles {
  roles: string[]
}

export interface DesktopAccountRoleResult {
  user_id: string
  role: null | string
}

export interface DesktopAccountEdge {
  manager_id: string
  member_id: string
  primary?: boolean
}

export interface DesktopAccountSubtree {
  root: string
  nodes: DesktopAccountTreeNode[]
  edges?: DesktopAccountEdge[]
}

export interface DesktopAccountSubaccount {
  user_id: string
  email: string
  name?: null | string
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

export interface KnowledgeInventory {
  ok: boolean
  error?: string
  path?: string
  name?: string
  // 文本类:可全文索引(langflow embed 正文)
  indexable?: { count: number; size: number; types: { ext: string; count: number; size: number }[] }
  // 其余文件:仅文件名进知识库(不解析正文;内容由上游多模态模型按需读取)
  nameOnly?: { count: number; types: { ext: string; count: number }[] }
  skipped?: { hidden: number; noise: number }
  estMinutes?: number
  truncated?: boolean
}

export interface KnowledgeIngestResult {
  ok: boolean
  error?: string
  kb?: string
  indexed?: number
  nameOnly?: number
  truncated?: boolean
}

export interface KnowledgeIngestProgress {
  phase: 'preparing' | 'indexing' | 'names' | 'done' | 'error'
  done?: number
  total?: number
  message?: string
}

// 一条已加入本机知识库的「知识源」(文件夹或文件)+ 它的 KB。manifest 不下发到渲染端。
export interface KnowledgeSource {
  sourceId: string
  kb: string
  type: 'folder' | 'file'
  path: string
  name: string
  indexed: number
  nameOnly: number
  fileCount: number
  truncated: boolean
  lastSyncedTs: number
}

export interface KnowledgeSyncResult {
  ok: boolean
  error?: string
  changed?: boolean
  added?: number
  modified?: number
  removed?: number
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
  // 允许在同一对话框选文件或文件夹(macOS)。优先于 directories。
  both?: boolean
  multiple?: boolean
  filters?: Array<{ name: string; extensions: string[] }>
}

export interface BackendExit {
  code: number | null
  signal: string | null
}
