import type { DesktopBootProgress, DesktopBootstrapState } from '@/global'

function noopUnsubscribe(): () => void {
  return () => undefined
}

function readyBootProgress(): DesktopBootProgress {
  return {
    error: null,
    fakeMode: false,
    message: 'Hermes browser-hosted desktop is ready',
    phase: 'runtime.ready',
    progress: 100,
    running: false,
    timestamp: Date.now()
  }
}

function readyBootstrapState(): DesktopBootstrapState {
  return {
    active: false,
    // A browser host never runs an in-app bundled payload.
    bundled: false,
    completedAt: Date.now(),
    error: null,
    log: [],
    manifest: null,
    setupChoice: null,
    stages: {},
    startedAt: null,
    unsupportedPlatform: null
  }
}

function browserUnsupported(feature: string): Error {
  return new Error(`${feature} is not available in the browser-hosted Desktop`)
}

// Bridge members that need no per-install state: native-only capabilities
// answered with a fixed browser result, a refusal, or a no-op.
export const BROWSER_BRIDGE_STUBS = {
  applyConnectionConfig: async () => {
    throw browserUnsupported('Gateway reconfiguration')
  },
  cancelBootstrap: async () => ({ cancelled: false, ok: true }),
  // Rendering an untrusted link must not contact its destination. PrettyLink
  // uses the URL slug when no title is available; Electron keeps its resolver.
  fetchLinkTitle: async () => '',
  claimAmbientCue: async () => true,
  cloud: {
    agentSignIn: async (dashboardUrl: string) => ({ baseUrl: dashboardUrl, connected: false }),
    discover: async () => ({ agents: [] }),
    login: async () => ({ ok: false, portalBaseUrl: '', signedIn: false }),
    logout: async () => ({ ok: true, portalBaseUrl: '', signedIn: false }),
    status: async () => ({ portalBaseUrl: '', signedIn: false })
  },
  continueBootstrapLocal: async () => ({ ok: true }),
  getBootProgress: async () => readyBootProgress(),
  getBootstrapState: async () => readyBootstrapState(),
  // Electron owns the pool; a browser host must not report a successful native
  // settings write.
  getPoolLimits: async () => {
    throw browserUnsupported('Desktop backend pool sizing')
  },
  setPoolLimits: async () => {
    throw browserUnsupported('Desktop backend pool sizing')
  },
  getRecentLogs: async () => ({ lines: [], path: '' }),
  // Venv/plugin receipts live on the server host; a browser tab has none of its own.
  getSyncStatus: async () => null,
  getRemoteDisplayReason: async () => 'Browser-hosted Desktop uses this server as its backend',
  getVersion: async () => ({
    appVersion: 'browser-hosted',
    electronVersion: '',
    hermesRoot: '',
    nodeVersion: '',
    platform: 'browser'
  }),
  // Browser pages have no native window compositor. Keep these explicit so
  // capability consumers do not fall back to the host OS (for example,
  // Windows) before the browser-host marker is available.
  glassSupported: false,
  translucencySupported: false,
  windowControls: {
    custom: false,
    minimize: () => { throw browserUnsupported('Native window controls') },
    toggleMaximize: () => { throw browserUnsupported('Native window controls') },
    close: () => { throw browserUnsupported('Native window controls') }
  },
  getPathForFile: () => '',
  normalizePreviewTarget: async () => null,
  onBackendExit: noopUnsubscribe,
  onBootProgress: noopUnsubscribe,
  onBootstrapEvent: noopUnsubscribe,
  onFoundInPage: noopUnsubscribe,
  onPreviewFileChanged: noopUnsubscribe,
  probeConnectionConfig: async (remoteUrl: string) => ({
    authMode: 'unknown' as const,
    baseUrl: remoteUrl,
    error: 'Gateway reconfiguration is unavailable in browser-hosted Desktop',
    providers: [],
    reachable: false,
    version: null
  }),
  revealLogs: async () => ({
    error: 'Native log reveal is unavailable in browser-hosted Desktop',
    ok: false,
    path: ''
  }),
  repairBootstrap: async () => ({ ok: true }),
  resetBootstrap: async () => ({ ok: true }),
  revalidateConnection: async () => ({ ok: true, rebuilt: false }),
  saveConnectionConfig: async () => {
    throw browserUnsupported('Gateway reconfiguration')
  },
  sanitizeWorkspaceCwd: async (cwd?: null | string) => ({ cwd: cwd || '', sanitized: false }),
  settings: {
    getDefaultProjectDir: async () => ({ defaultLabel: 'Server workspace', dir: null, resolvedCwd: '' }),
    pickDefaultProjectDir: async () => ({ canceled: true, dir: null }),
    setDefaultProjectDir: async (dir: null | string) => ({ dir })
  },
  sshConfigHosts: async () => ({ hosts: [] }),
  sshResolveHost: async () => ({ hostname: null, identityFile: null, port: null, user: null }),
  setActiveWork: () => undefined,
  setKeepAwake: () => undefined,
  setNativeTheme: () => undefined,
  setPreviewShortcutActive: () => undefined,
  setTitleBarTheme: () => undefined,
  setTranslucency: () => undefined,
  stopFindInPage: async () => undefined,
  stopPreviewFileWatch: async () => true,
  themes: {
    fetchMarketplace: async (id: string) => ({ displayName: id, extensionId: id, themes: [] }),
    searchMarketplace: async () => []
  },
  touchBackend: async () => ({ ok: true }),
  uninstall: {
    run: async () => ({ error: 'Run `hermes uninstall` on the server host', ok: false }),
    summary: async () => ({
      agent_installed: true,
      // Package removal belongs to the server host, never this browser tab.
      code_removal_allowed: false,
      gui_installed: true,
      hermes_home: '',
      packaged_app_paths: [],
      platform: 'browser',
      source_built_artifacts: [],
      userdata_dir: '',
      userdata_exists: true
    })
  },
  updates: {
    apply: async () => ({ command: 'hermes update', manual: true, message: 'Run `hermes update` on the server host', ok: false }),
    check: async () => ({ message: 'Use `hermes update` on the server host', reason: 'browser-hosted', supported: false }),
    getBranch: async () => ({ branch: '' }),
    onProgress: noopUnsubscribe,
    setBranch: async (branch: string) => ({ branch })
  },
  watchPreviewFile: async (url: string) => ({ id: `browser:${url}`, path: url })
} satisfies Partial<Window['hermesDesktop']>
