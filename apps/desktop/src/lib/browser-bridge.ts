// Browser fallback for window.hermesDesktop.
//
// When the desktop app renderer runs directly inside a standard web browser
// (such as during frontend dev at http://127.0.0.1:5174/ or on a web host),
// Electron's preload script is absent. This shim exposes the hermesDesktop IPC
// surface in window so that the app's gateway boot sequence, UI panes,
// layout engine, and stores function smoothly without throwing
// "Desktop IPC bridge is unavailable".

import type { HermesConnection, HermesWindowState } from '@/global'

export function installBrowserBridge(): void {
  if (typeof window === 'undefined' || window.hermesDesktop) {
    return
  }

  const listeners = new Map<string, Set<Function>>()

  const on = (event: string, fn: Function) => {
    if (!listeners.has(event)) {
      listeners.set(event, new Set())
    }
    listeners.get(event)!.add(fn)

    return () => {
      listeners.get(event)?.delete(fn)
    }
  }

  const getGatewayUrl = () => {
    try {
      const params = new URLSearchParams(window.location.search)
      const queryGw = params.get('gateway')
      if (queryGw) return queryGw
      const storedGw = localStorage.getItem('hermes-gateway-url')
      if (storedGw) return storedGw
    } catch {
      // localStorage may fail in restricted contexts
    }
    // Default to SamAgent / Hermes local gateway port 8080
    return 'http://127.0.0.1:8080'
  }

  const getWsUrl = (profile?: string | null) => {
    const base = getGatewayUrl()
    const wsBase = base.replace(/^http/i, 'ws')
    const qs = profile && profile !== 'default' ? `?profile=${encodeURIComponent(profile)}` : ''
    return `${wsBase}/ws${qs}`
  }

  const makeConnection = (profile?: string | null, connectionId?: string | null): HermesConnection => {
    const effectiveProfile = profile || 'default'
    return {
      baseUrl: getGatewayUrl(),
      wsUrl: getWsUrl(effectiveProfile),
      token: (typeof localStorage !== 'undefined' ? localStorage.getItem('hermes-token') : null) || '',
      isFullscreen: false,
      isMaximized: true,
      mode: 'local',
      nativeOverlayWidth: 0,
      windowButtonPosition: null,
      logs: [],
      profile: effectiveProfile,
      connectionId: connectionId || undefined
    }
  }

  const defaultConnection = makeConnection('default')

  const bridge = {
    glassSupported: false,
    translucencySupported: false,
    localModelsEnabled: true,
    guestOnboardingEnabled: false,
    localSkin: null,

    getConnection: async (profile?: string | null) => makeConnection(profile),
    getConnectionFor: async (payload?: { profile?: string | null; connectionId?: string | null }) =>
      makeConnection(payload?.profile, payload?.connectionId),
    getGatewayWsUrl: async (profile?: string | null) => ({
      wsUrl: getWsUrl(profile),
      authMode: 'token' as const,
      profile: profile || 'default'
    }),
    getGatewayWsUrlFor: async (payload?: { profile?: string | null; connectionId?: string | null }) => ({
      wsUrl: getWsUrl(payload?.profile),
      authMode: 'token' as const,
      profile: payload?.profile || 'default'
    }),
    getProfileRoutes: async () => [],
    getAgentRoster: async () => ({ agents: [] }),
    revalidateConnection: async () => ({ ok: true, rebuilt: false }),
    touchBackend: async () => ({ ok: true }),
    getPoolLimits: async () => ({ maxBackends: 5, idleMs: 300000 }),
    setPoolLimits: async (limits: { maxBackends?: number; idleMs?: number }) => ({
      ok: true,
      limits: { maxBackends: limits?.maxBackends ?? 5, idleMs: limits?.idleMs ?? 300000 }
    }),

    openSessionWindow: async () => ({ ok: false }),
    openSessionInTerminal: async () => ({ ok: false }),
    openWindow: async () => ({ ok: false }),
    openBrowserWindow: async () => ({ ok: false }),
    onBrowserPopoutClosed: () => () => {},

    onWindowStateChanged: (callback: (payload: HermesWindowState) => void) => on('windowState', callback),
    onBackendExit: (callback: () => void) => on('backendExit', callback),
    onPowerResume: (callback: () => void) => on('powerResume', callback),
    onConnectionApplied: (callback: () => void) => on('connectionApplied', callback),
    onPoolBackendRetiring: (callback: (payload: { poolKey: string }) => void) => on('poolRetiring', callback),

    setActiveConnectionRoute: () => {},
    setActiveWork: () => {},
    getOnBattery: async () => false,
    onBatteryChanged: () => () => {},

    getRecentLogs: async () => ({ lines: [] }),
    revealLogs: async () => ({ ok: true }),

    writeClipboard: async (text: string) => {
      try {
        await navigator.clipboard?.writeText(text)
      } catch {
        // clipboard permission may be denied
      }
    },
    readClipboard: async () => {
      try {
        return (await navigator.clipboard?.readText()) || ''
      } catch {
        return ''
      }
    },

    connections: {
      list: async () => ({ connections: [] }),
      save: async () => ({ ok: true }),
      remove: async () => ({ ok: true }),
      setPrimary: async () => ({ ok: true }),
      setLaunchMode: async () => ({ ok: true }),
      setLastUsed: async () => ({ ok: true }),
      test: async () => ({ ok: true }),
      updateManaged: async () => ({ ok: true }),
      updateAll: async () => ({ ok: true }),
      onChanged: (callback: (payload: any) => void) => on('connectionsChanged', callback)
    },

    profile: {
      getDefault: async () => ({ profile: 'default' }),
      setDefault: async () => ({ ok: true }),
      onDefaultChanged: () => () => {},
      get: async () => ({ profile: 'default' }),
      remember: async () => ({ ok: true }),
      set: async () => ({ ok: true })
    },

    git: {
      repoStatus: async () => null,
      branches: async () => ({ branches: [], current: '' })
    },

    cwd: {
      get: async () => ({ cwd: '' }),
      getDefault: async () => ({ cwd: '' })
    },

    terminal: {
      getBackend: async () => ({ backend: 'local' as const, available: true }),
      isBackendAvailable: async () => true,
      openTerminalWindow: async () => ({ ok: false })
    },

    dataUrlReadMax: {
      get: async () => 10,
      set: async () => ({ ok: true })
    },

    windowControls: {
      custom: false,
      minimize: () => {},
      toggleMaximize: () => {},
      close: () => {}
    },

    wakeIndicator: {
      getState: async () => null,
      setState: () => {},
      onState: () => () => {}
    },

    chatOnboarding: {
      grow: () => {},
      soloBoot: () => {}
    },

    petOverlay: {
      open: async () => ({ ok: false }),
      close: async () => ({ ok: true }),
      setBounds: () => {},
      setIgnoreMouse: () => {},
      setFocusable: () => {},
      pushState: () => {},
      control: () => {},
      onState: () => () => {},
      onControl: () => () => {}
    },

    hud: {
      nativeDrag: false,
      windowing: {
        clientPlacement: true,
        controlDrag: false,
        nativeDrag: false,
        solid: true,
        workspaceTransfer: false
      },
      open: async () => ({ ok: false }),
      close: async () => ({ ok: true }),
      setIgnoreMouse: () => {},
      beginMove: () => {},
      endMove: () => {},
      moveBy: () => {},
      setBounds: () => {},
      resetLayout: async () => ({ ok: true }),
      setFrost: async () => ({ ok: true }),
      setSession: () => {},
      onGoto: () => () => {},
      onChanged: () => () => {},
      onCursor: () => () => {},
      onGameOverlay: () => () => {}
    },

    quickEntry: {
      getSettings: async () => ({ enabled: false }),
      setSettings: async () => ({ ok: true }),
      submit: () => {},
      dismiss: () => {},
      pushState: () => {},
      onState: () => () => {},
      onSubmit: () => () => {},
      onShown: () => () => {}
    },

    cloud: {
      status: async () => ({ loggedIn: false }),
      login: async () => ({ ok: false }),
      logout: async () => ({ ok: true }),
      discover: async () => ({ organizations: [] }),
      agentSignIn: async () => ({ ok: false })
    },

    getBootProgress: async () => null,
    getConnectionConfig: async () => null,
    saveConnectionConfig: async () => ({ ok: true }),
    applyConnectionConfig: async () => ({ ok: true }),
    testConnectionConfig: async () => ({ ok: true }),
    getSecretStorageEncryption: async () => ({ enabled: false }),
    setSecretStorageEncryption: async () => ({ ok: true }),
    sshConfigHosts: async () => [],
    sshResolveHost: async () => null,
    probeConnectionConfig: async () => ({ ok: true }),
    oauthLoginConnectionConfig: async () => ({ ok: false }),
    oauthLogoutConnectionConfig: async () => ({ ok: true }),
    api: async () => ({ ok: false }),
    notify: async () => ({ ok: true }),
    claimStartupLatency: async () => null,
    requestMicrophoneAccess: async () => false,
    readWindowBelow: async () => null,
    readFileDataUrl: async () => '',
    readFileDataUrlForAttach: async () => '',
    readFileText: async () => '',
    readPluginSource: async () => '',
    selectPaths: async () => [],
    selectSavePath: async () => '',
    saveGatewayFile: async () => ({ ok: false }),
    saveImageFromUrl: async () => ({ ok: false }),
    contextMenuEdit: async () => {},
    contextMenuCopyImage: async () => {},
    contextMenuSpellcheck: async () => {},
    contextMenuGuestAddWord: async () => {},
    onContextMenuSpellcheck: () => () => {},
    saveImageBuffer: async () => ({ ok: false }),
    capturePreview: async () => ({ ok: false }),
    savePastedText: async () => ({ ok: false }),
    saveClipboardImage: async () => ({ ok: false }),
    getPathForFile: (file: any) => file?.path || file?.name || '',
    isLocalPath: () => false,
    showItemInFolder: async () => ({ ok: false }),
    openPath: async (p: string) => {
      window.open(p, '_blank')
      return { ok: true }
    },
    openExternal: async (url: string) => {
      window.open(url, '_blank')
      return { ok: true }
    },
    trashFile: async () => ({ ok: false })
  }

  try {
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      enumerable: true,
      value: bridge,
      writable: true
    })
  } catch {
    ;(window as any).hermesDesktop = bridge
  }
}

// Auto-run when module is loaded
installBrowserBridge()
