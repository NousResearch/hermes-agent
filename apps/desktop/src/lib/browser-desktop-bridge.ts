import type {
  DesktopBootProgress,
  DesktopBootstrapState,
  HermesApiRequest,
  HermesConnection,
  HermesTerminalExit,
  HermesTerminalSession
} from '@/global'

interface BrowserBootstrapWindow {
  __HERMES_BASE_PATH__?: string
  __HERMES_SESSION_TOKEN__?: string
  hermesDesktop?: Window['hermesDesktop']
}

const SESSION_HEADER = 'X-Hermes-Session-Token'
const DEFAULT_TIMEOUT_MS = 30_000

const IMAGE_MIME_BY_EXTENSION: Record<string, string> = {
  '.bmp': 'image/bmp',
  '.gif': 'image/gif',
  '.jpeg': 'image/jpeg',
  '.jpg': 'image/jpeg',
  '.png': 'image/png',
  '.tif': 'image/tiff',
  '.tiff': 'image/tiff',
  '.webp': 'image/webp'
}

const IMAGE_EXTENSION_BY_MIME: Record<string, string> = Object.fromEntries(
  Object.entries(IMAGE_MIME_BY_EXTENSION).map(([extension, mime]) => [mime, extension])
)

function normalizedExtension(value: string): string {
  const clean = String(value || '').trim().toLowerCase()

  if (!clean) {return ''}

  return clean.startsWith('.') ? clean : `.${clean}`
}

function bytesToDataUrl(bytes: Uint8Array, mimeType: string): string {
  const chunkSize = 0x8000
  let binary = ''

  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    const chunk = bytes.subarray(offset, Math.min(bytes.length, offset + chunkSize))
    binary += String.fromCharCode(...chunk)
  }

  return `data:${mimeType};base64,${btoa(binary)}`
}

function noopUnsubscribe(): () => void {
  return () => undefined
}

function normalizedBasePath(value: string | undefined): string {
  if (!value) {return ''}
  const leading = value.startsWith('/') ? value : `/${value}`

  return leading.replace(/\/+$/, '')
}

function browserBootstrap(): { basePath: string; token: string } | null {
  const win = window as unknown as BrowserBootstrapWindow
  const token = String(win.__HERMES_SESSION_TOKEN__ || '').trim()

  if (!token) {return null}

  return {
    basePath: normalizedBasePath(win.__HERMES_BASE_PATH__),
    token
  }
}

function endpointUrl(path: string, basePath: string, profile?: null | string): URL {
  const suffix = path.startsWith('/') ? path : `/${path}`
  const url = new URL(`${basePath}${suffix}`, window.location.origin)

  if (profile && !url.searchParams.has('profile')) {
    url.searchParams.set('profile', profile)
  }

  return url
}

function websocketUrl(basePath: string, token: string, profile?: null | string): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const url = new URL(`${protocol}//${window.location.host}${basePath}/api/ws`)
  url.searchParams.set('token', token)

  if (profile) {url.searchParams.set('profile', profile)}

  return url.toString()
}

async function browserApi<T>(bootstrap: { basePath: string; token: string }, request: HermesApiRequest): Promise<T> {
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), request.timeoutMs ?? DEFAULT_TIMEOUT_MS)

  try {
    const headers = new Headers({ [SESSION_HEADER]: bootstrap.token })
    let body: BodyInit | undefined

    if (request.upload) {
      const form = new FormData()

      const blob = new Blob([request.upload.bytes], {
        type: request.upload.contentType || 'application/octet-stream'
      })

      form.append('file', blob, request.upload.filename || 'file')
      body = form
    } else if (request.body !== undefined) {
      headers.set('Content-Type', 'application/json')
      body = JSON.stringify(request.body)
    }

    const response = await fetch(endpointUrl(request.path, bootstrap.basePath, request.profile), {
      body,
      headers,
      method: request.method || 'GET',
      signal: controller.signal
    })

    const text = await response.text()

    if (!response.ok) {
      throw new Error(`${response.status}: ${text || response.statusText}`)
    }

    if (!text) {return null as T}

    if (/^\s*<(?:!doctype|html)/i.test(text)) {
      throw new Error(`Hermes API returned HTML for ${request.path}`)
    }

    return JSON.parse(text) as T
  } finally {
    window.clearTimeout(timeout)
  }
}

function connectionFor(bootstrap: { basePath: string; token: string }, profile?: null | string): HermesConnection {
  const baseUrl = `${window.location.origin}${bootstrap.basePath}`

  return {
    authMode: 'token',
    baseUrl,
    isFullscreen: Boolean(document.fullscreenElement),
    logs: [],
    mode: 'local',
    nativeOverlayWidth: 0,
    profile: profile || undefined,
    source: 'local',
    token: bootstrap.token,
    windowButtonPosition: null,
    wsUrl: websocketUrl(bootstrap.basePath, bootstrap.token, profile)
  }
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

function browserLocalConnectionConfig(profile?: null | string) {
  return {
    cloudOrg: '',
    envOverride: false,
    mode: 'local' as const,
    profile: profile || null,
    remoteAuthMode: 'token' as const,
    remoteOauthConnected: false,
    remoteTokenPlainText: false,
    remoteTokenPreview: null,
    remoteTokenSet: false,
    secureTokenStorage: false,
    remoteUrl: '',
    sshHost: '',
    sshKeyPath: '',
    sshPort: null,
    sshRemoteHermesPath: '',
    sshRemoteProfile: '',
    sshUser: ''
  }
}

function browserUnsupported(feature: string): Error {
  return new Error(`${feature} is not available in the Termux browser-hosted Desktop`)
}

function queryPath(route: string, values: Record<string, boolean | null | string | undefined>) {
  const query = new URLSearchParams()

  for (const [key, value] of Object.entries(values)) {
    if (value !== null && value !== undefined) {query.set(key, String(value))}
  }

  return `${route}?${query.toString()}`
}

/**
 * Install a capability-limited Desktop bridge when the real renderer is served
 * directly by Hermes' loopback web server (the Termux + Termux:X11 path).
 *
 * Electron remains authoritative everywhere it exists: no injected dashboard
 * token means this is a normal Vite/Electron renderer, and an existing preload
 * bridge is never replaced. Browser-hosted mode maps the Desktop's backend
 * contract onto the same-origin /api + /api/ws surface and deliberately exposes
 * only safe browser equivalents for machine-level capabilities.
 */
export function installBrowserDesktopBridge(): boolean {
  const win = window as unknown as BrowserBootstrapWindow

  if (win.hermesDesktop) {return false}

  const bootstrap = browserBootstrap()

  if (!bootstrap) {return false}

  const getConnection = async (profile?: null | string) => connectionFor(bootstrap, profile)

  const openExternal = async (url: string) => {
    window.open(url, '_blank', 'noopener,noreferrer')
  }

  const writeClipboard = async (text: string) => {
    await navigator.clipboard?.writeText(text)

    return true
  }

  const api = <T>(request: HermesApiRequest) => browserApi<T>(bootstrap, request)
  const transientObjectUrls = new Set<string>()

  window.addEventListener(
    'beforeunload',
    () => {
      transientObjectUrls.forEach(url => URL.revokeObjectURL(url))
      transientObjectUrls.clear()
    },
    { once: true }
  )

  const browserProfile = () => new URLSearchParams(window.location.search).get('profile')

  const saveBuffer = async (data: ArrayBuffer | Uint8Array, ext: string) => {
    const source = data instanceof Uint8Array ? data : new Uint8Array(data)
    const bytes = new Uint8Array(source.byteLength)
    bytes.set(source)

    const extension = normalizedExtension(ext)
    const imageMime = IMAGE_MIME_BY_EXTENSION[extension]

    if (imageMime) {
      const uploaded = await api<{ path?: string }>({
        body: {
          data_url: bytesToDataUrl(bytes, imageMime),
          filename: `desktop-upload${extension}`
        },
        method: 'POST',
        path: '/api/chat/image-upload',
        profile: browserProfile()
      })

      return uploaded.path || ''
    }

    const mimeType = extension === '.htm' || extension === '.html' ? 'text/html;charset=utf-8' : 'application/octet-stream'
    const blob = new Blob([bytes.buffer], { type: mimeType })
    const url = URL.createObjectURL(blob)
    transientObjectUrls.add(url)

    return url
  }

  interface BrowserTerminalState {
    closed: boolean
    cwd: string
    dataListeners: Set<(payload: string) => void>
    decoder: TextDecoder
    exit: HermesTerminalExit | null
    exitListeners: Set<(payload: HermesTerminalExit) => void>
    pendingData: string
    shell: string
    socket: WebSocket
  }

  const browserTerminals = new Map<string, BrowserTerminalState>()
  let browserTerminalSequence = 0

  const emitTerminalData = (state: BrowserTerminalState, data: string) => {
    if (!data) {
      return
    }

    if (state.dataListeners.size === 0) {
      state.pendingData = (state.pendingData + data).slice(-256 * 1024)

      return
    }

    state.dataListeners.forEach(listener => listener(data))
  }

  const emitTerminalExit = (state: BrowserTerminalState) => {
    if (state.closed) {
      return
    }

    state.closed = true
    state.exit = { code: null, signal: null }
    state.exitListeners.forEach(listener => listener(state.exit!))
  }

  const startBrowserTerminal = async (options?: {
    cols?: number
    cwd?: string
    rows?: number
  }): Promise<HermesTerminalSession> => {
    let cwd = String(options?.cwd || '').trim()

    if (!cwd) {
      try {
        cwd = (await api<{ cwd?: string }>({ path: '/api/fs/default-cwd', profile: browserProfile() })).cwd || ''
      } catch {
        cwd = ''
      }
    }

    browserTerminalSequence += 1
    const id = `browser-${Date.now().toString(36)}-${browserTerminalSequence.toString(36)}`
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const url = new URL(`${protocol}//${window.location.host}${bootstrap.basePath}/api/pty`)
    url.searchParams.set('token', bootstrap.token)

    const profile = browserProfile()

    if (profile) {
      url.searchParams.set('profile', profile)
    }

    const socket = new WebSocket(url)
    socket.binaryType = 'arraybuffer'

    const state: BrowserTerminalState = {
      closed: false,
      cwd,
      dataListeners: new Set(),
      decoder: new TextDecoder(),
      exit: null,
      exitListeners: new Set(),
      pendingData: '',
      shell: 'hermes-tui',
      socket
    }

    browserTerminals.set(id, state)

    return new Promise<HermesTerminalSession>((resolve, reject) => {
      let settled = false

      const rejectStart = (message: string) => {
        if (settled) {
          return
        }

        settled = true
        browserTerminals.delete(id)
        reject(new Error(message))
      }

      socket.onopen = () => {
        if (settled) {
          return
        }

        settled = true
        resolve({ cwd: state.cwd, id, shell: state.shell })
      }

      socket.onmessage = event => {
        if (typeof event.data === 'string') {
          emitTerminalData(state, event.data)

          return
        }

        const binary = event.data as unknown

        if (
          binary instanceof ArrayBuffer ||
          Object.prototype.toString.call(binary) === '[object ArrayBuffer]'
        ) {
          emitTerminalData(state, state.decoder.decode(binary as ArrayBuffer, { stream: true }))

          return
        }

        if (ArrayBuffer.isView(binary)) {
          emitTerminalData(
            state,
            state.decoder.decode(
              new Uint8Array(binary.buffer, binary.byteOffset, binary.byteLength),
              { stream: true }
            )
          )

          return
        }

        if (binary instanceof Blob) {
          void binary.arrayBuffer().then(buffer =>
            emitTerminalData(state, state.decoder.decode(buffer, { stream: true }))
          )
        }
      }

      socket.onerror = () => rejectStart('Hermes TUI WebSocket failed to connect')

      socket.onclose = event => {
        emitTerminalData(state, state.decoder.decode())

        if (!settled) {
          rejectStart(event.reason || `Hermes TUI closed before startup (${event.code})`)
        }

        emitTerminalExit(state)
      }
    })
  }

  window.addEventListener(
    'beforeunload',
    () => {
      browserTerminals.forEach(state => {
        try {
          state.socket.close(1000, 'page unload')
        } catch {
          // Best effort: the browser may already be tearing down its sockets.
        }
      })
      browserTerminals.clear()
    },
    { once: true }
  )

  const fsGet = <T>(route: string, path: string) =>
    api<T>({ path: queryPath(`/api/fs/${route}`, { path }) })

  const gitGet = <T>(route: string, values: Record<string, boolean | null | string | undefined>) =>
    api<T>({ path: queryPath(`/api/git/${route}`, values) })

  const gitPost = <T>(route: string, body: Record<string, unknown>) =>
    api<T>({ body, method: 'POST', path: `/api/git/${route}` })

  const downloadUrl = async (url: string, filename = '') => {
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = filename
    anchor.rel = 'noopener'
    anchor.style.display = 'none'
    document.body.append(anchor)
    anchor.click()
    anchor.remove()

    return true
  }

  const git = {
    baseBranchList: async (repoPath: string) =>
      (await gitGet<{ branches: unknown[] }>('base-branches', { path: repoPath })).branches,
    branchList: async (repoPath: string) =>
      (await gitGet<{ branches: unknown[] }>('branches', { path: repoPath })).branches,
    branchSwitch: (repoPath: string, branch: string) => gitPost('branch/switch', { branch, path: repoPath }),
    fileDiff: async (repoPath: string, filePath: string) =>
      (await gitGet<{ diff: string }>('file-diff', { file: filePath, path: repoPath })).diff,
    repoStatus: (repoPath: string) => gitGet('status', { path: repoPath }),
    review: {
      commit: (repoPath: string, message: string, push: boolean) =>
        gitPost('review/commit', { message, path: repoPath, push }),
      commitContext: (repoPath: string) => gitGet('review/commit-context', { path: repoPath }),
      createPr: (repoPath: string) => gitPost('review/create-pr', { path: repoPath }),
      fetchPrComment: async () => null,
      diff: async (repoPath: string, filePath: string, scope: string, baseRef?: null | string, staged?: boolean) =>
        (await gitGet<{ diff: string }>('review/diff', {
          base: baseRef,
          file: filePath,
          path: repoPath,
          scope,
          staged
        })).diff,
      list: (repoPath: string, scope: string, baseRef?: null | string) =>
        gitGet('review/list', { base: baseRef, path: repoPath, scope }),
      prList: (repoPath: string, branches: string[], numbers?: number[]) =>
        gitPost('review/pr-list', { branches, numbers: numbers || [], path: repoPath }),
      push: (repoPath: string) => gitPost('review/push', { path: repoPath }),
      revert: (repoPath: string, filePath?: null | string) =>
        gitPost('review/revert', { file: filePath || null, path: repoPath }),
      revParse: async (repoPath: string, ref?: null | string) =>
        (await gitGet<{ sha: null | string }>('review/rev-parse', { path: repoPath, ref })).sha,
      shipInfo: (repoPath: string) => gitGet('review/ship-info', { path: repoPath }),
      stage: (repoPath: string, filePath?: null | string) =>
        gitPost('review/stage', { file: filePath || null, path: repoPath }),
      unstage: (repoPath: string, filePath?: null | string) =>
        gitPost('review/unstage', { file: filePath || null, path: repoPath })
    },
    scanRepos: async () => [],
    worktreeAdd: (repoPath: string, options?: Record<string, unknown>) =>
      gitPost('worktree/add', { path: repoPath, ...options }),
    worktreeList: async (repoPath: string) =>
      (await gitGet<{ worktrees: unknown[] }>('worktrees', { path: repoPath })).worktrees,
    worktreeRemove: (repoPath: string, worktreePath: string, options?: { force?: boolean }) =>
      gitPost('worktree/remove', { force: Boolean(options?.force), path: repoPath, worktreePath })
  } as NonNullable<Window['hermesDesktop']['git']>

  const bridge: Window['hermesDesktop'] = {
    api,
    applyConnectionConfig: async (payload: { mode: string; profile?: null | string }) => {
      if (payload.mode !== 'local') {throw browserUnsupported('Remote/SSH gateway reconfiguration')}

      return browserLocalConnectionConfig(payload.profile)
    },
    cancelBootstrap: async () => ({ cancelled: false, ok: true }),
    fetchLinkTitle: async (url: string) => {
      try {
        const response = await fetch(url)
        const html = await response.text()

        return html.match(/<title[^>]*>([^<]*)<\/title>/i)?.[1]?.trim() || new URL(url).hostname
      } catch {
        return new URL(url).hostname
      }
    },
    findInPage: async (query: string) => {
      const find = (window as Window & { find?: (value: string) => boolean }).find

      return { count: query && find?.call(window, query) ? 1 : 0 }
    },
    getConnectionConfig: async (profile?: null | string) => browserLocalConnectionConfig(profile),
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
    getConnection,
    getRecentLogs: async () => ({ lines: [], path: '' }),
    getGatewayWsUrl: async (profile?: null | string) => ({
      ok: true as const,
      wsUrl: websocketUrl(bootstrap.basePath, bootstrap.token, profile)
    }),
    getRemoteDisplayReason: async () => 'Browser-hosted Termux Desktop uses the local loopback backend',
    getVersion: async () => ({
      appVersion: 'browser-hosted',
      electronVersion: '',
      hermesRoot: '',
      nodeVersion: '',
      platform: 'android-termux'
    }),
    git,
    getPathForFile: () => '',
    gitRoot: async (path: string) => (await fsGet<{ root: null | string }>('git-root', path)).root,
    normalizePreviewTarget: async () => null,
    readDir: (path: string) =>
      fsGet<Awaited<ReturnType<Window['hermesDesktop']['readDir']>>>('list', path),
    readFileDataUrl: async (path: string) => (await fsGet<{ dataUrl: string }>('read-data-url', path)).dataUrl,
    readFileDataUrlForAttach: async (path: string) => (await fsGet<{ dataUrl: string }>('read-data-url', path)).dataUrl,
    readFileText: (path: string) =>
      fsGet<Awaited<ReturnType<Window['hermesDesktop']['readFileText']>>>('read-text', path),
    notify: async ({ title, body }: { title?: string; body?: string }) => {
      if (!('Notification' in window)) {return false}

      if (Notification.permission === 'default') {await Notification.requestPermission()}

      if (Notification.permission !== 'granted') {return false}
      new Notification(title || 'Hermes', { body })

      return true
    },
    onBackendExit: noopUnsubscribe,
    onBootProgress: noopUnsubscribe,
    onBootstrapEvent: noopUnsubscribe,
    onFoundInPage: noopUnsubscribe,
    onPreviewFileChanged: noopUnsubscribe,
    openExternal,
    openPreviewInBrowser: openExternal,
    openSessionWindow: async (sessionId: string, opts?: { watch?: boolean }) => {
      const params = new URLSearchParams({ session: sessionId })

      if (opts?.watch) {params.set('watch', '1')}
      window.open(`${window.location.pathname}${window.location.search}#/?${params}`, '_blank', 'noopener,noreferrer')

      return { ok: true }
    },
    openWindow: async () => {
      window.open(window.location.href, '_blank', 'noopener,noreferrer')

      return { ok: true }
    },
    petOverlay: {
      close: async () => ({ ok: true }),
      control: () => undefined,
      onControl: noopUnsubscribe,
      onState: noopUnsubscribe,
      open: async () => ({ ok: false }),
      pushState: () => undefined,
      setBounds: () => undefined,
      setFocusable: () => undefined,
      setIgnoreMouse: () => undefined
    },
    profile: {
      get: async () => ({ profile: new URLSearchParams(window.location.search).get('profile') }),
      set: async (profile: string | null) => ({ profile })
    },
    quickEntry: {
      dismiss: () => undefined,
      getSettings: async () => ({ enabled: false, error: null, registered: false, shortcut: '' }),
      onShown: noopUnsubscribe,
      onState: noopUnsubscribe,
      onSubmit: noopUnsubscribe,
      pushState: () => undefined,
      setSettings: async (patch: { enabled?: boolean; shortcut?: string }) => ({
        enabled: Boolean(patch.enabled),
        error: null,
        registered: false,
        shortcut: patch.shortcut || ''
      }),
      submit: () => undefined
    },
    oauthLoginConnectionConfig: async (remoteUrl: string) => ({ baseUrl: remoteUrl, connected: false, ok: false }),
    oauthLogoutConnectionConfig: async () => ({ connected: false, ok: true }),
    probeConnectionConfig: async (remoteUrl: string) => ({
      authMode: 'unknown' as const,
      baseUrl: remoteUrl,
      error: 'Remote gateway setup is unavailable in browser-hosted Termux Desktop',
      providers: [],
      reachable: false,
      version: null
    }),
    readClipboard: async () => navigator.clipboard?.readText?.() || '',
    revealLogs: async () => ({
      error: 'Native log reveal is unavailable in browser-hosted Termux Desktop',
      ok: false,
      path: ''
    }),
    repairBootstrap: async () => ({ ok: true }),
    resetBootstrap: async () => ({ ok: true }),
    revalidateConnection: async () => ({ ok: true, rebuilt: false }),
    saveConnectionConfig: async (payload: { mode: string; profile?: null | string }) => {
      if (payload.mode !== 'local') {throw browserUnsupported('Remote/SSH gateway reconfiguration')}

      return browserLocalConnectionConfig(payload.profile)
    },
    saveClipboardImage: async () => {
      const clipboard = navigator.clipboard as Clipboard & {
        read?: () => Promise<ClipboardItems>
      }

      if (!clipboard?.read) {return ''}

      const items = await clipboard.read()

      for (const item of items) {
        const mimeType = item.types.find(type => type.startsWith('image/'))

        if (!mimeType) {continue}

        const blob = await item.getType(mimeType)
        const extension = IMAGE_EXTENSION_BY_MIME[mimeType] || '.png'

        return saveBuffer(await blob.arrayBuffer(), extension)
      }

      return ''
    },
    saveImageBuffer: saveBuffer,
    saveImageFromUrl: (url: string) => downloadUrl(url),
    sanitizeWorkspaceCwd: async (cwd?: null | string) => ({ cwd: cwd || '', sanitized: false }),
    selectPaths: async () => [],
    settings: {
      getDefaultProjectDir: async () => ({ defaultLabel: 'Termux home', dir: null, resolvedCwd: '' }),
      pickDefaultProjectDir: async () => ({ canceled: true, dir: null }),
      setDefaultProjectDir: async (dir: null | string) => ({ dir })
    },
    sshConfigHosts: async () => ({ hosts: [] }),
    sshResolveHost: async () => ({ hostname: null, identityFile: null, port: null, user: null }),
    requestMicrophoneAccess: async () => {
      if (!navigator.mediaDevices?.getUserMedia) {return false}
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      stream.getTracks().forEach(track => track.stop())

      return true
    },
    setActiveWork: () => undefined,
    setKeepAwake: () => undefined,
    setNativeTheme: () => undefined,
    setPreviewShortcutActive: () => undefined,
    setTitleBarTheme: () => undefined,
    setTranslucency: () => undefined,
    stopFindInPage: async () => undefined,
    stopPreviewFileWatch: async () => true,
    testConnectionConfig: async (payload: { mode: string }) =>
      payload.mode === 'local'
        ? { baseUrl: window.location.origin, ok: true, reachable: true, version: null }
        : { error: 'Only the local loopback backend is supported in browser-hosted Termux Desktop', ok: false, reachable: false },
    terminal: {
      cwd: async (id: string) => browserTerminals.get(id)?.cwd || null,
      dispose: async (id: string) => {
        const state = browserTerminals.get(id)

        if (!state) {
          return false
        }

        browserTerminals.delete(id)

        try {
          state.socket.close(1000, 'disposed')
        } catch {
          // Socket may already be closed after the Hermes TUI exited.
        }

        return true
      },
      onData: (id: string, callback: (payload: string) => void) => {
        const state = browserTerminals.get(id)

        if (!state) {
          return noopUnsubscribe()
        }

        state.dataListeners.add(callback)

        if (state.pendingData) {
          const pending = state.pendingData
          state.pendingData = ''
          queueMicrotask(() => callback(pending))
        }

        return () => state.dataListeners.delete(callback)
      },
      onExit: (id: string, callback: (payload: HermesTerminalExit) => void) => {
        const state = browserTerminals.get(id)

        if (!state) {
          return noopUnsubscribe()
        }

        state.exitListeners.add(callback)

        if (state.exit) {
          const exit = state.exit
          queueMicrotask(() => callback(exit))
        }

        return () => state.exitListeners.delete(callback)
      },
      resize: async (id: string, size: { cols: number; rows: number }) => {
        const state = browserTerminals.get(id)

        if (!state || state.socket.readyState !== WebSocket.OPEN) {
          return false
        }

        state.socket.send(`\u001b[RESIZE:${Math.max(1, Math.round(size.cols))};${Math.max(1, Math.round(size.rows))}]`)

        return true
      },
      start: startBrowserTerminal,
      write: async (id: string, data: string) => {
        const state = browserTerminals.get(id)

        if (!state || state.socket.readyState !== WebSocket.OPEN) {
          return false
        }

        state.socket.send(data)

        return true
      }
    },
    themes: {
      fetchMarketplace: async (id: string) => ({ displayName: id, extensionId: id, themes: [] }),
      searchMarketplace: async () => []
    },
    touchBackend: async () => ({ ok: true }),
    uninstall: {
      run: async () => ({ error: 'Run `hermes uninstall` from Termux', ok: false }),
      summary: async () => ({
        agent_installed: true,
        gui_installed: true,
        hermes_home: '',
        packaged_app_paths: [],
        platform: 'android-termux',
        source_built_artifacts: [],
        userdata_dir: '',
        userdata_exists: true
      })
    },
    updates: {
      apply: async () => ({ command: 'hermes update', manual: true, message: 'Run `hermes update` in Termux', ok: false }),
      check: async () => ({ message: 'Use `hermes update` in Termux', reason: 'browser-hosted', supported: false }),
      getBranch: async () => ({ branch: '' }),
      onProgress: noopUnsubscribe,
      setBranch: async (branch: string) => ({ branch })
    },
    watchPreviewFile: async (url: string) => ({ id: `browser:${url}`, path: url }),
    writeClipboard,
    writeTextFile: async (path: string, content: string) => {
      const result = await api<{ path?: string }>({ body: { content, path }, method: 'POST', path: '/api/fs/write-text' })

      return { path: result.path || path }
    },
    zoom: {
      get: async () => ({ level: 0, percent: Math.round(window.devicePixelRatio * 100) || 100 }),
      onChanged: noopUnsubscribe,
      setPercent: () => undefined
    }
  }

  win.hermesDesktop = bridge
  document.documentElement.dataset.hermesDesktopHost = 'browser'

  return true
}
