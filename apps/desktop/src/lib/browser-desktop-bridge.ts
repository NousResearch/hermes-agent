import { buildHermesWebSocketUrl, LOCAL_CONNECTION_ID } from '@hermes/shared'

import type {
  DesktopBootProgress,
  DesktopBootstrapState,
  HermesApiRequest,
  HermesConnection,
  HermesSelectPathsOptions,
  HermesStagedUpload,
  HermesTerminalExit,
  HermesTerminalSession
} from '@/global'
import { bytesToBase64 } from '@/lib/base64'
import { createGitRestBridge } from '@/lib/git-rest'
import { $connection } from '@/store/session'

interface BrowserBootstrapWindow {
  __HERMES_AUTH_REQUIRED__?: boolean
  __HERMES_BASE_PATH__?: string
  __HERMES_SESSION_TOKEN__?: string
  hermesDesktop?: Window['hermesDesktop']
}

interface BrowserBootstrap {
  authRequired: boolean
  basePath: string
  token: string
  stagedUploads: Map<string, HermesStagedUpload>
}

const SESSION_HEADER = 'X-Hermes-Session-Token'
const DEFAULT_TIMEOUT_MS = 30_000
const STAGED_UPLOAD_CACHE_LIMIT = 256
const REAUTH_EVENT = 'hermes:browser-reauth-required'
const TERMINAL_META_PREFIX = '\u0000HERMES_TERMINAL_META:'

class BrowserReauthRequiredError extends Error {
  readonly loginUrl: string
  readonly needsOauthLogin = true

  constructor(message: string, loginUrl: string) {
    super(message)
    this.name = 'BrowserReauthRequiredError'
    this.loginUrl = loginUrl
  }
}

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
  return `data:${mimeType};base64,${bytesToBase64(bytes)}`
}

function noopUnsubscribe(): () => void {
  return () => undefined
}

function normalizedBasePath(value: string | undefined): string {
  if (!value) {return ''}
  const leading = value.startsWith('/') ? value : `/${value}`

  return leading.replace(/\/+$/, '')
}

function browserBootstrap(): BrowserBootstrap | null {
  const win = window as unknown as BrowserBootstrapWindow
  const token = String(win.__HERMES_SESSION_TOKEN__ || '').trim()
  const authRequired = win.__HERMES_AUTH_REQUIRED__ === true

  if (!token && !authRequired) {return null}

  return {
    authRequired,
    basePath: normalizedBasePath(win.__HERMES_BASE_PATH__),
    token,
    stagedUploads: new Map()
  }
}

function endpointUrl(path: string, basePath: string, profile?: null | string): URL {
  const suffix = path.startsWith('/') ? path : `/${path}`
  const normalizedBase = normalizedBasePath(basePath)
  const url = new URL(`${normalizedBase}${suffix}`, window.location.origin)

  if (url.origin !== window.location.origin) {
    throw new Error('Hermes API paths must remain on the Webapp origin')
  }

  if (
    normalizedBase &&
    url.pathname !== normalizedBase &&
    !url.pathname.startsWith(`${normalizedBase}/`)
  ) {
    throw new Error('Hermes API paths must remain inside the configured base path')
  }

  if (profile && !url.searchParams.has('profile')) {
    url.searchParams.set('profile', profile)
  }

  return url
}

function authenticatedEndpointUrl(bootstrap: BrowserBootstrap, path: string, profile?: null | string): URL {
  const url = endpointUrl(path, bootstrap.basePath, profile)

  // Downloads and media elements cannot set the session header. Gated hosts
  // use their same-origin cookie; loopback hosts use the injected token.
  if (bootstrap.token) {url.searchParams.set('token', bootstrap.token)}

  return url
}

function websocketUrl(
  basePath: string,
  path: string,
  credential: { ticket?: string; token?: string },
  profile?: null | string
): string {
  const authParam = credential.ticket
    ? (['ticket', credential.ticket] as const)
    : credential.token
      ? (['token', credential.token] as const)
      : undefined

  return buildHermesWebSocketUrl({
    authParam,
    basePath,
    params: profile ? { profile } : undefined,
    path
  })
}

function sandboxedHtmlBlob(bytes: Uint8Array): Blob {
  const source = bytesToDataUrl(bytes, 'text/html;charset=utf-8')

  const wrapper = [
    '<!doctype html>',
    '<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
    '<style>html,body,iframe{box-sizing:border-box;width:100%;height:100%;margin:0;border:0}</style>',
    '</head><body>',
    `<iframe sandbox="allow-scripts" referrerpolicy="no-referrer" src="${source}"></iframe>`,
    '</body></html>'
  ].join('')

  return new Blob([wrapper], { type: 'text/html;charset=utf-8' })
}

function reauthError(
  bootstrap: BrowserBootstrap,
  response: Response,
  text: string
): BrowserReauthRequiredError | null {
  if (!bootstrap.authRequired || response.status !== 401) {return null}

  let payload: { detail?: unknown; error?: unknown; login_url?: unknown } = {}

  try {
    payload = JSON.parse(text) as typeof payload
  } catch {
    return null
  }

  if (
    !['session_expired', 'unauthenticated'].includes(String(payload.error || '')) ||
    typeof payload.login_url !== 'string'
  ) {
    return null
  }

  let target: URL

  try {
    target = new URL(payload.login_url, window.location.origin)
  } catch {
    return null
  }

  if (target.origin !== window.location.origin) {return null}

  return new BrowserReauthRequiredError(
    `${String(payload.error)}: ${String(payload.detail || 'Unauthorized')}`,
    target.href
  )
}

function navigateToBrowserLogin(error: BrowserReauthRequiredError): void {
  const event = new CustomEvent(REAUTH_EVENT, {
    cancelable: true,
    detail: { loginUrl: error.loginUrl }
  })

  if (window.dispatchEvent(event)) {window.location.assign(error.loginUrl)}
}

interface BrowserFetchRequest {
  body?: BodyInit
  headers?: HeadersInit
  method?: string
  path: string
  profile?: null | string
  timeoutMs?: number
}

async function browserFetch(
  bootstrap: BrowserBootstrap,
  request: BrowserFetchRequest
): Promise<{ response: Response; text: string }> {
  const controller = request.timeoutMs === undefined ? null : new AbortController()
  const timeout = controller ? window.setTimeout(() => controller.abort(), request.timeoutMs) : null
  const headers = new Headers(request.headers)

  if (bootstrap.token) {headers.set(SESSION_HEADER, bootstrap.token)}

  try {
    const response = await fetch(endpointUrl(request.path, bootstrap.basePath, request.profile), {
      body: request.body,
      credentials: 'same-origin',
      headers,
      method: request.method || 'GET',
      signal: controller?.signal
    })

    const text = await response.text()

    if (!response.ok) {
      const authError = reauthError(bootstrap, response, text)

      if (authError) {
        navigateToBrowserLogin(authError)
        throw authError
      }
    }

    return { response, text }
  } finally {
    if (timeout !== null) {window.clearTimeout(timeout)}
  }
}

async function browserApi<T>(bootstrap: BrowserBootstrap, request: HermesApiRequest): Promise<T> {
  const headers = new Headers()
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

  const { response, text } = await browserFetch(bootstrap, {
    body,
    headers,
    method: request.method,
    path: request.path,
    profile: request.profile,
    timeoutMs: request.timeoutMs ?? DEFAULT_TIMEOUT_MS
  })

  if (!response.ok) {
    throw new Error(`${response.status}: ${text || response.statusText}`)
  }

  if (!text) {return null as T}

  if (/^\s*<(?:!doctype|html)/i.test(text)) {
    throw new Error(`Hermes API returned HTML for ${request.path}`)
  }

  return JSON.parse(text) as T
}

async function stageBrowserFile(
  bootstrap: BrowserBootstrap,
  file: File,
  profile?: null | string
): Promise<string> {
  const form = new FormData()
  form.append('file', file, file.name || 'attachment')

  const { response, text } = await browserFetch(bootstrap, {
    body: form,
    method: 'POST',
    path: '/api/chat/file-upload',
    profile
  })

  const payload = JSON.parse(text) as { detail?: string; path?: string; staged_upload?: HermesStagedUpload }

  if (!response.ok || !payload.path) {
    throw new Error(payload.detail || `File upload failed (${response.status})`)
  }

  // Keep the existing string-path bridge for pickers and drops. Composer chips
  // carry this small source descriptor so draft cloning and retries retain it.
  if (payload.staged_upload?.path === payload.path) {
    bootstrap.stagedUploads.set(payload.path, payload.staged_upload)

    // Chips retain their own descriptors. Bound the string-path compatibility
    // cache without consuming lookups shared by multiple attachment occurrences.
    if (bootstrap.stagedUploads.size > STAGED_UPLOAD_CACHE_LIMIT) {
      const oldest = bootstrap.stagedUploads.keys().next().value

      if (oldest !== undefined) { bootstrap.stagedUploads.delete(oldest) }
    }
  }

  return payload.path
}

function selectBrowserFiles(
  bootstrap: BrowserBootstrap,
  options?: HermesSelectPathsOptions,
  fallbackProfile?: null | string
): Promise<string[]> {
  if (options?.directories) {return Promise.resolve([])}

  return new Promise<string[]>((resolve, reject) => {
    const input = document.createElement('input')
    input.type = 'file'
    input.multiple = Boolean(options?.multiple)
    input.style.display = 'none'

    const extensions = (options?.filters || []).flatMap(filter => filter.extensions || [])

    if (extensions.length) {
      input.accept = extensions.map(extension => `.${extension.replace(/^\./, '')}`).join(',')
    }

    const finish = () => input.remove()
    input.addEventListener(
      'cancel',
      () => {
        finish()
        resolve([])
      },
      { once: true }
    )
    input.addEventListener(
      'change',
      () => {
        const files = Array.from(input.files || [])
        finish()

        const stageSelected = async () => {
          const paths: string[] = []
          const profile = options?.profile?.trim() || fallbackProfile || null

          for (const file of files) {
            paths.push(await stageBrowserFile(bootstrap, file, profile))
          }

          return paths
        }

        void stageSelected().then(resolve, reject)
      },
      { once: true }
    )
    document.body.append(input)
    input.click()
  })
}

async function authenticatedWebsocketUrl(
  bootstrap: BrowserBootstrap,
  path: '/api/pty' | '/api/ws',
  profile?: null | string
): Promise<string> {
  if (!bootstrap.authRequired) {
    return websocketUrl(bootstrap.basePath, path, { token: bootstrap.token }, profile)
  }

  const result = await browserApi<{ ticket?: string }>(bootstrap, {
    method: 'POST',
    path: '/api/auth/ws-ticket'
  })

  const ticket = String(result.ticket || '').trim()

  if (!ticket) {throw new Error('Hermes did not return a WebSocket ticket')}

  return websocketUrl(bootstrap.basePath, path, { ticket }, profile)
}

function connectionFor(bootstrap: BrowserBootstrap, profile?: null | string): HermesConnection {
  const baseUrl = `${window.location.origin}${bootstrap.basePath}`

  return {
    authMode: bootstrap.authRequired ? 'oauth' : 'token',
    baseUrl,
    isFullscreen: Boolean(document.fullscreenElement),
    logs: [],
    mode: 'remote',
    nativeOverlayWidth: 0,
    profile: profile || undefined,
    remoteHost: window.location.host,
    remoteKind: 'url',
    sharedPrimary: Boolean(profile),
    source: 'settings',
    token: bootstrap.token,
    windowButtonPosition: null,
    wsUrl: bootstrap.authRequired
      ? ''
      : websocketUrl(bootstrap.basePath, '/api/ws', { token: bootstrap.token }, profile)
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

function browserConnectionConfig(bootstrap: BrowserBootstrap, profile?: null | string) {
  return {
    cloudOrg: '',
    envOverride: false,
    mode: 'remote' as const,
    profile: profile || null,
    remoteAuthMode: bootstrap.authRequired ? ('oauth' as const) : ('token' as const),
    remoteOauthConnected: bootstrap.authRequired,
    remoteTokenPlainText: false,
    remoteTokenPreview: null,
    remoteTokenSet: Boolean(bootstrap.token),
    secureTokenStorage: false,
    remoteUrl: `${window.location.origin}${bootstrap.basePath}`,
    sshHost: '',
    sshKeyPath: '',
    sshPort: null,
    sshRemoteHermesPath: '',
    sshRemoteProfile: '',
    sshUser: ''
  }
}

function browserUnsupported(feature: string): Error {
  return new Error(`${feature} is not available in the browser-hosted Desktop`)
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
 * directly by Hermes' authenticated web server.
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

  const requireBrowserConnection = (connectionId?: null | string) => {
    const requested = connectionId?.trim() || LOCAL_CONNECTION_ID

    if (requested !== LOCAL_CONNECTION_ID) {
      throw new Error(`No connection with id "${requested}"`)
    }
  }

  const getConnectionFor = async (payload: { connectionId?: null | string; profile?: null | string }) => {
    requireBrowserConnection(payload.connectionId)

    return {
      ...connectionFor(bootstrap, payload.profile),
      connectionId: LOCAL_CONNECTION_ID,
      registryScoped: true,
      sharedPrimary: false,
      sharedRemote: Boolean(payload.profile)
    }
  }

  const transientObjectUrls = new Set<string>()
  const nativeWriteClipboard = navigator.clipboard?.writeText?.bind(navigator.clipboard)

  const openExternal = async (url: string) => {
    window.open(url, '_blank', 'noopener,noreferrer')
  }

  const writeClipboard = async (text: string) => {
    if (!nativeWriteClipboard) {return false}

    await nativeWriteClipboard(text)

    return true
  }

  const api = <T>(request: HermesApiRequest) => browserApi<T>(bootstrap, request)

  window.addEventListener(
    'beforeunload',
    () => {
      transientObjectUrls.forEach(url => URL.revokeObjectURL(url))
      transientObjectUrls.clear()
    },
    { once: true }
  )

  const browserProfile = () =>
    $connection.get()?.profile?.trim() || new URLSearchParams(window.location.search).get('profile')

  const rememberBrowserProfile = async (profile: null | string) => {
    const selected = profile?.trim() || null
    const url = new URL(window.location.href)

    if (selected && selected !== 'default') {
      url.searchParams.set('profile', selected)
    } else {
      url.searchParams.delete('profile')
    }

    window.history.replaceState(null, '', `${url.pathname}${url.search}${url.hash}`)

    return { profile: selected }
  }

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

    const blob = extension === '.htm' || extension === '.html'
      ? sandboxedHtmlBlob(bytes)
      : new Blob([bytes.buffer], { type: 'application/octet-stream' })

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
    const profile = browserProfile()
    const url = new URL(await authenticatedWebsocketUrl(bootstrap, '/api/pty', profile))
    url.searchParams.set('mode', 'shell')
    url.searchParams.set('cols', String(Math.max(2, Math.round(options?.cols || 80))))
    url.searchParams.set('rows', String(Math.max(2, Math.round(options?.rows || 24))))

    if (cwd) {url.searchParams.set('cwd', cwd)}
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
      shell: 'host-shell',
      socket
    }

    browserTerminals.set(id, state)

    return new Promise<HermesTerminalSession>((resolve, reject) => {
      let settled = false
      let startupTimer = 0

      const rejectStart = (message: string) => {
        if (settled) {
          return
        }

        settled = true
        window.clearTimeout(startupTimer)
        browserTerminals.delete(id)

        try {
          socket.close(1002, 'terminal startup failed')
        } catch {
          // The socket may already be closing.
        }

        reject(new Error(message))
      }

      const resolveStart = (shell: string) => {
        if (settled) {
          return
        }

        settled = true
        window.clearTimeout(startupTimer)
        state.shell = shell
        resolve({ cwd: state.cwd, id, shell: state.shell })
      }

      startupTimer = window.setTimeout(
        () => rejectStart('Host terminal did not provide startup metadata'),
        10_000
      )

      socket.onmessage = event => {
        if (typeof event.data === 'string') {
          if (event.data.startsWith(TERMINAL_META_PREFIX)) {
            try {
              const payload = JSON.parse(event.data.slice(TERMINAL_META_PREFIX.length)) as { shell?: unknown }
              const shell = typeof payload.shell === 'string' ? payload.shell.trim() : ''

              if (!shell) {
                rejectStart('Host terminal returned an empty shell identity')
              } else {
                resolveStart(shell)
              }
            } catch {
              rejectStart('Host terminal returned invalid startup metadata')
            }

            return
          }

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

      socket.onerror = () => rejectStart('Host terminal WebSocket failed to connect')

      socket.onclose = event => {
        emitTerminalData(state, state.decoder.decode())

        if (!settled) {
          rejectStart(event.reason || `Host terminal closed before startup (${event.code})`)
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
    api<T>({ path: queryPath(`/api/fs/${route}`, { path }), profile: browserProfile() })

  const gitGet = <T>(route: string, values: Record<string, boolean | null | string | undefined>) =>
    api<T>({ path: queryPath(`/api/git/${route}`, values), profile: browserProfile() })

  const gitPost = <T>(route: string, body: Record<string, unknown>) =>
    api<T>({ body, method: 'POST', path: `/api/git/${route}`, profile: browserProfile() })

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

  const git = createGitRestBridge({ get: gitGet, post: gitPost })

  const getGatewayWsUrl = async (profile?: null | string) => {
    try {
      return {
        ok: true as const,
        wsUrl: await authenticatedWebsocketUrl(bootstrap, '/api/ws', profile)
      }
    } catch (error) {
      if (error instanceof BrowserReauthRequiredError) {
        return {
          error: error.message,
          needsOauthLogin: true,
          ok: false as const
        }
      }

      throw error
    }
  }

  const getProfiles = async () => {
    const result = await api<{ profiles?: { name?: string }[] }>({ path: '/api/profiles' })

    return [...new Set((result.profiles || []).map(profile => String(profile.name || '').trim()).filter(Boolean))]
  }

  // Electron owns the pool; a browser host must not report a successful native
  // settings write. Keep this separate for older preload contracts without it.
  const nativePoolLimits = {
    getPoolLimits: async (): Promise<never> => {
      throw browserUnsupported('Desktop backend pool sizing')
    },
    setPoolLimits: async (): Promise<never> => {
      throw browserUnsupported('Desktop backend pool sizing')
    }
  }

  const bridge: Window['hermesDesktop'] = {
    ...nativePoolLimits,
    api,
    applyConnectionConfig: async () => {
      throw browserUnsupported('Gateway reconfiguration')
    },
    cancelBootstrap: async () => ({ cancelled: false, ok: true }),
    fetchLinkTitle: async (url: string) => {
      try {
        const html = await (await fetch(url)).text()

        return html.match(/<title[^>]*>([\s\S]*?)<\/title>/i)?.[1]?.trim() || new URL(url).hostname
      } catch {
        return new URL(url).hostname
      }
    },
    findInPage: async (query: string) => {
      const find = (window as Window & { find?: (value: string) => boolean }).find

      return { count: query && find?.call(window, query) ? 1 : 0 }
    },
    getConnectionConfig: async (profile?: null | string) => browserConnectionConfig(bootstrap, profile),
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
    getConnectionFor,
    getRecentLogs: async () => ({ lines: [], path: '' }),
    getProfileRoutes: async (profiles: string[]) => {
      const available = new Set(await getProfiles())

      return [...new Set(profiles.map(profile => profile.trim() || 'default'))]
        .filter(profile => available.has(profile))
        .map(profile => ({
          connectionId: LOCAL_CONNECTION_ID,
          mode: 'local' as const,
          profile,
          targetProfile: profile
        }))
    },
    getAgentRoster: async () => {
      const profiles = await getProfiles()
      const connectionLabel = window.location.host

      return {
        agents: profiles.map(profile => ({
          connectionId: LOCAL_CONNECTION_ID,
          connectionKind: 'local' as const,
          connectionLabel,
          handle: profile === 'default' ? 'hermes' : profile,
          profile,
          targetProfile: profile
        })),
        primaryConnectionId: LOCAL_CONNECTION_ID,
        sources: [{
          connectionId: LOCAL_CONNECTION_ID,
          kind: 'local' as const,
          label: connectionLabel,
          reachable: true
        }]
      }
    },
    getGatewayWsUrl,
    getGatewayWsUrlFor: async (payload: { connectionId?: null | string; profile?: null | string }) => {
      requireBrowserConnection(payload.connectionId)

      return getGatewayWsUrl(payload.profile)
    },
    getGatewayFileStreamUrl: async payload => {
      requireBrowserConnection(payload.connectionId)

      return authenticatedEndpointUrl(
        bootstrap,
        queryPath('/api/files/stream', { path: payload.path }),
        payload.profile ?? browserProfile()
      ).href
    },
    getRemoteDisplayReason: async () => 'Browser-hosted Desktop uses this server as its backend',
    getVersion: async () => ({
      appVersion: 'browser-hosted',
      electronVersion: '',
      hermesRoot: '',
      nodeVersion: '',
      platform: 'browser'
    }),
    git,
    getPathForFile: () => '',
    stageFileForAttach: (file: File) => stageBrowserFile(bootstrap, file, browserProfile()),
    getStagedFileForAttach: (path: string) => bootstrap.stagedUploads.get(path),
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
    openSessionWindow: async (
      sessionId: string,
      opts?: { profile?: null | string; watch?: boolean }
    ) => {
      const id = sessionId.trim()

      if (!id) {return { error: 'invalid_session_id', ok: false }}
      const target = new URL(window.location.href)
      target.searchParams.set('win', 'secondary')

      const profile = opts?.profile?.trim()

      if (profile === 'default') {
        target.searchParams.delete('profile')
      } else if (profile) {
        target.searchParams.set('profile', profile)
      }

      if (opts?.watch) {
        target.searchParams.set('watch', '1')
      } else {
        target.searchParams.delete('watch')
      }

      target.hash = `/${encodeURIComponent(id)}`
      window.open(target.href, '_blank', 'noopener,noreferrer')

      return { ok: true }
    },
    openWindow: async () => {
      window.open(window.location.href, '_blank', 'noopener,noreferrer')

      return { ok: true }
    },
    profile: {
      get: async () => ({ profile: browserProfile() }),
      // The browser URL is the persisted launch location. Mirror Desktop's
      // persistence-only profile IPC without interrupting the live switch.
      remember: rememberBrowserProfile,
      set: async (profile: string | null) => {
        const result = await rememberBrowserProfile(profile)
        window.location.reload()

        return result
      }
    },
    oauthLoginConnectionConfig: async (remoteUrl: string) => {
      const target = new URL(`${bootstrap.basePath}/login`, window.location.origin)
      target.searchParams.set('next', `${window.location.pathname}${window.location.search}${window.location.hash}`)
      navigateToBrowserLogin(new BrowserReauthRequiredError('Sign in required', target.href))

      return { baseUrl: remoteUrl, connected: false, ok: false }
    },
    oauthLogoutConnectionConfig: async () => {
      const response = await fetch(endpointUrl('/auth/logout', bootstrap.basePath), {
        credentials: 'same-origin',
        method: 'POST'
      })

      if (!response.ok) {
        throw new Error(`Sign out failed (${response.status})`)
      }

      const login = new URL(`${bootstrap.basePath}/login`, window.location.origin)
      navigateToBrowserLogin(new BrowserReauthRequiredError('Signed out', login.href))

      return { connected: false, ok: true }
    },
    probeConnectionConfig: async (remoteUrl: string) => ({
      authMode: 'unknown' as const,
      baseUrl: remoteUrl,
      error: 'Gateway reconfiguration is unavailable in browser-hosted Desktop',
      providers: [],
      reachable: false,
      version: null
    }),
    readClipboard: async () => navigator.clipboard?.readText?.() || '',
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
    saveGatewayFile: async payload => {
      requireBrowserConnection(payload.connectionId)
      const path = queryPath('/api/files/download', { path: payload.path })
      const profile = payload.profile ?? browserProfile()

      // Validate before handing the transfer to the browser so missing or
      // denied files surface in the chat without navigating away. HEAD keeps
      // large downloads out of renderer memory.
      await api({ method: 'HEAD', path, profile })

      return {
        saved: await downloadUrl(authenticatedEndpointUrl(bootstrap, path, profile).href, payload.suggestedName)
      }
    },
    saveImageFromUrl: (url: string) => downloadUrl(url),
    sanitizeWorkspaceCwd: async (cwd?: null | string) => ({ cwd: cwd || '', sanitized: false }),
    selectPaths: options => selectBrowserFiles(bootstrap, options, browserProfile()),
    settings: {
      getDefaultProjectDir: async () => ({ defaultLabel: 'Server workspace', dir: null, resolvedCwd: '' }),
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
    testConnectionConfig: async () => ({
      baseUrl: `${window.location.origin}${bootstrap.basePath}`,
      ok: true,
      reachable: true,
      version: null
    }),
    terminal: {
      // Electron's attach opens its buffered output gate after listeners are
      // registered. Browser terminals already buffer until onData subscribes,
      // so attachment is a liveness acknowledgement for the same contract.
      attach: async (id: string) => {
        const state = browserTerminals.get(id)

        return Boolean(state && !state.closed && state.socket.readyState === WebSocket.OPEN)
      },
      // The host process is remote from this renderer; unlike Electron, the
      // bridge cannot inspect a PTY child's live cwd. Shell OSC 7/9;9 output is
      // still observed by useTerminalSession, so return unknown rather than
      // repeatedly overwriting that authoritative value with the launch cwd.
      cwd: async () => null,
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
      run: async () => ({ error: 'Run `hermes uninstall` on the server host', ok: false }),
      summary: async () => ({
        agent_installed: true,
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
    watchPreviewFile: async (url: string) => ({ id: `browser:${url}`, path: url }),
    writeClipboard,
    writeTextFile: async (path: string, content: string) => {
      const result = await api<{ path?: string }>({
        body: { content, path },
        method: 'POST',
        path: '/api/fs/write-text',
        profile: browserProfile()
      })

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
