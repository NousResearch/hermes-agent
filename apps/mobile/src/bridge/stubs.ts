/**
 * stubs.ts — safe no-op implementations for the parts of the window.hermesDesktop
 * contract that don't apply to a remote mobile client: local terminal (xterm),
 * self-update, uninstall, first-launch bootstrap, multi-window, local filesystem,
 * VS Code theme marketplace, etc.
 *
 * These exist so vendored desktop code that reaches for them gets a defined,
 * harmless response instead of `undefined is not a function`. Subscription-style
 * methods return a real unsubscribe.
 */

const noopUnsub = (): void => {}

// A tiny "already booted, nothing to do" progress object.
const readyBoot = {
  error: null,
  fakeMode: false,
  message: 'Connected',
  phase: 'ready',
  progress: 1,
  running: false,
  timestamp: 0,
}

const unsupportedWindow = { ok: false, error: 'unsupported-on-mobile' }

/** Build the stub half of the bridge. Loosely typed; install-bridge merges the
 *  real methods over it and casts the result to the contract. */
export function makeStubs() {
  return {
    revalidateConnection: async () => ({ ok: true, rebuilt: false }),
    touchBackend: async () => ({ ok: true }),

    // Multi-window → handled by in-app routing on mobile.
    openSessionWindow: async () => unsupportedWindow,
    openNewSessionWindow: async () => unsupportedWindow,

    // Boot / bootstrap: there is no local backend to install on a phone.
    getBootProgress: async () => readyBoot,
    onBootProgress: () => noopUnsub,
    getBootstrapState: async () => ({
      active: false,
      manifest: null,
      stages: {},
      error: null,
      log: [],
      startedAt: null,
      completedAt: null,
      unsupportedPlatform: null,
    }),
    resetBootstrap: async () => ({ ok: true }),
    repairBootstrap: async () => ({ ok: true }),
    cancelBootstrap: async () => ({ ok: true, cancelled: false }),
    onBootstrapEvent: () => noopUnsub,

    // Connection-config UI (desktop Settings → Gateway). Mobile owns its own
    // connect/login screens, so these return inert defaults.
    getConnectionConfig: async () => ({
      envOverride: false,
      mode: 'remote' as const,
      profile: null,
      remoteAuthMode: 'oauth' as const,
      remoteOauthConnected: true,
      remoteTokenPreview: null,
      remoteTokenSet: false,
      remoteUrl: '',
    }),
    saveConnectionConfig: async (p: unknown) => p,
    applyConnectionConfig: async (p: unknown) => p,
    testConnectionConfig: async () => ({ baseUrl: '', ok: true, version: null }),
    probeConnectionConfig: async (remoteUrl: string) => ({
      baseUrl: remoteUrl,
      reachable: false,
      authMode: 'oauth' as const,
      providers: [],
      version: null,
      error: 'use the mobile connect screen',
    }),
    oauthLoginConnectionConfig: async (remoteUrl: string) => ({
      ok: false,
      baseUrl: remoteUrl,
      connected: false,
    }),
    oauthLogoutConnectionConfig: async () => ({ ok: true, connected: false }),

    profile: {
      get: async () => ({ profile: null }),
      set: async () => ({ profile: null }),
    },

    requestMicrophoneAccess: async () => true,

    // Local filesystem (desktop-only). Remote files come via /api/files later.
    readFileDataUrl: async () => '',
    readFileText: async (filePath: string) => ({ path: filePath, text: '' }),
    selectPaths: async () => [] as string[],
    readDir: async () => ({ entries: [] }),
    getPathForFile: () => '',
    sanitizeWorkspaceCwd: async (cwd?: null | string) => ({ cwd: cwd ?? '', sanitized: false }),

    // Image capture/save (Phase 4).
    saveImageFromUrl: async () => false,
    saveImageBuffer: async () => '',
    saveClipboardImage: async () => '',

    // Link / preview helpers.
    fetchLinkTitle: async () => '',
    normalizePreviewTarget: async () => null,
    watchPreviewFile: async (url: string) => ({ id: '', path: url }),
    stopPreviewFileWatch: async () => true,
    onPreviewFileChanged: () => noopUnsub,
    onClosePreviewRequested: () => noopUnsub,

    settings: {
      getDefaultProjectDir: async () => ({ defaultLabel: '', dir: null, resolvedCwd: '' }),
      pickDefaultProjectDir: async () => ({ canceled: true, dir: null }),
      setDefaultProjectDir: async (dir: null | string) => ({ dir }),
    },

    revealLogs: async () => ({ ok: false, path: '', error: 'unsupported-on-mobile' }),
    getRecentLogs: async () => ({ path: '', lines: [] as string[] }),

    // Local PTY terminal — not ported to mobile.
    terminal: {
      dispose: async () => true,
      onData: () => noopUnsub,
      onExit: () => noopUnsub,
      resize: async () => true,
      start: async () => ({ cwd: '', id: '', shell: '' }),
      write: async () => true,
    },

    onBackendExit: () => noopUnsub,

    getVersion: async () => ({
      appVersion: '0.0.1',
      electronVersion: '',
      nodeVersion: '',
      platform: 'capacitor',
      hermesRoot: '',
    }),

    updates: {
      check: async () => ({ supported: false, reason: 'managed-on-server' }),
      apply: async () => ({ ok: false, error: 'managed-on-server' }),
      getBranch: async () => ({ branch: '' }),
      setBranch: async (name: string) => ({ branch: name }),
      onProgress: () => noopUnsub,
    },

    uninstall: {
      summary: async () => ({
        hermes_home: '',
        agent_installed: false,
        gui_installed: false,
        source_built_artifacts: [],
        packaged_app_paths: [],
        userdata_dir: '',
        userdata_exists: false,
        platform: 'capacitor',
      }),
      run: async () => ({ ok: false, error: 'unsupported-on-mobile' }),
    },

    themes: {
      fetchMarketplace: async (id: string) => ({ extensionId: id, displayName: '', themes: [] }),
      searchMarketplace: async () => [],
    },
  }
}
