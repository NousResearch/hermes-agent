import type { HermesApiRequest, HermesConnection } from '@desktop/global'

import { noEvent, unsupportedAsync } from './unsupported'

type JsonRecord = Record<string, any>
let authRedirecting = false

// Browser-selected files have no usable filesystem path. Keep their bytes in a
// short-lived, renderer-local map and expose opaque virtual paths to the
// existing Desktop composer contract. The gateway only receives bytes through
// its normal image.attach_bytes/file.attach data flow.
const browserFiles = new Map<string, File>()
let browserFileSequence = 0

function browserFilePath(file: File) {
  browserFileSequence += 1
  return `web-upload://${browserFileSequence}/${encodeURIComponent(file.name || 'image')}`
}

function browserFileFromPath(path: string) {
  if (!path.startsWith('web-upload://')) return undefined
  return browserFiles.get(path)
}

function fileAsDataUrl(file: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(reader.error || new Error('Could not read selected file'))
    reader.onload = () => resolve(String(reader.result || ''))
    reader.readAsDataURL(file)
  })
}

async function selectBrowserFiles(options?: { directories?: boolean; multiple?: boolean; filters?: Array<{ extensions?: string[] }> }) {
  if (options?.directories) return []
  const input = document.createElement('input')
  input.type = 'file'
  input.multiple = options?.multiple !== false
  const extensions = options?.filters?.flatMap(filter => filter.extensions || []).filter(Boolean) || []
  if (extensions.length) input.accept = extensions.map(extension => `.${extension.replace(/^\\./, '')}`).join(',')
  input.style.display = 'none'
  document.body.appendChild(input)
  try {
    const selected = await new Promise<File[]>(resolve => {
      input.addEventListener('change', () => resolve(Array.from(input.files || [])), { once: true })
      input.click()
    })
    return selected.map(file => {
      const path = browserFilePath(file)
      browserFiles.set(path, file)
      return path
    })
  } finally {
    input.remove()
  }
}

function withScope(path: string, profile?: null | string, connectionId?: null | string) {
  const url = new URL(path, window.location.origin)
  if (profile?.trim() && !url.searchParams.has('profile')) url.searchParams.set('profile', profile.trim())
  if (connectionId?.trim() && !url.searchParams.has('connectionId')) {
    url.searchParams.set('connectionId', connectionId.trim())
  }
  return `${url.pathname}${url.search}${url.hash}`
}

export async function parseApiResponse<T>(response: Response): Promise<T> {
  const text = await response.text()
  const contentType = response.headers.get('content-type')?.toLowerCase() || ''
  let value: unknown
  let parsedJson = false

  if (text) {
    try {
      value = JSON.parse(text)
      parsedJson = true
    } catch {
      // Auth gateways, reverse proxies, and dev servers can return HTML or
      // plain text for an API request. Never surface their markup as a JSON
      // parser error, and never inject it into a UI error message.
    }
  }

  if (!response.ok) {
    const record = value && typeof value === 'object' ? value as Record<string, unknown> : undefined
    const detail = record?.detail || record?.error || record?.message
    const status = `${response.status}${response.statusText ? ` ${response.statusText}` : ''}`
    const safeText = !parsedJson && text && !contentType.includes('html') ? text.trim().slice(0, 240) : ''
    throw Object.assign(new Error(String(detail || safeText || status)), {
      statusCode: response.status,
      response: value
    })
  }

  if (text && !parsedJson) {
    throw Object.assign(new Error(`Expected a JSON response (${response.status}${response.statusText ? ` ${response.statusText}` : ''})`), {
      statusCode: response.status
    })
  }

  return value as T
}

async function requestJson<T>(request: HermesApiRequest): Promise<T> {
  const path = withScope(request.path, request.profile, (request as HermesApiRequest & { connectionId?: string }).connectionId)
  const headers = new Headers({ Accept: 'application/json' })
  let body: BodyInit | undefined

  if (request.upload) {
    const form = new FormData()
    form.append('file', new File([request.upload.bytes], request.upload.filename, {
      type: request.upload.contentType || 'application/octet-stream'
    }))
    body = form
  } else if (request.body !== undefined) {
    headers.set('Content-Type', 'application/json')
    body = JSON.stringify(request.body)
  }

  const controller = new AbortController()
  const timer = request.timeoutMs ? window.setTimeout(() => controller.abort(), request.timeoutMs) : null

  try {
    const response = await fetch(path, {
      method: request.method || (body === undefined ? 'GET' : 'POST'),
      headers,
      body,
      credentials: 'include',
      signal: controller.signal
    })

    if (response.status === 401) {
      if (!authRedirecting) {
        authRedirecting = true
        const next = `${window.location.pathname}${window.location.search}${window.location.hash}`
        window.location.assign(`/login?next=${encodeURIComponent(next)}`)
      }
      throw Object.assign(new Error('Authentication required'), { statusCode: 401 })
    }

    return await parseApiResponse<T>(response)
  } finally {
    if (timer !== null) window.clearTimeout(timer)
  }
}

function wsOrigin() {
  return `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}`
}

export function isLocalConnection(connectionId?: null | string) {
  const scoped = connectionId?.trim() || ''
  return !scoped || scoped === 'local'
}

export async function gatewayWsUrl(profile?: null | string, connectionId?: null | string) {
  if (isLocalConnection(connectionId)) {
    const url = new URL(`${wsOrigin()}/api/ws`)
    if (profile?.trim()) url.searchParams.set('profile', profile.trim())
    if (connectionId?.trim()) url.searchParams.set('connectionId', connectionId.trim())
    return { ok: true as const, wsUrl: url.toString() }
  }
  try {
    const { ticket } = await requestJson<{ ticket: string }>({
      path: '/api/auth/ws-ticket',
      method: 'POST',
      body: {},
      ...(connectionId ? { connectionId } : {}),
      ...(profile ? { profile } : {})
    } as HermesApiRequest & { connectionId?: string })
    const url = new URL(`${wsOrigin()}/api/ws`)
    url.searchParams.set('ticket', ticket)
    if (profile?.trim()) url.searchParams.set('profile', profile.trim())
    if (connectionId?.trim()) url.searchParams.set('connectionId', connectionId.trim())
    return { ok: true as const, wsUrl: url.toString() }
  } catch (error) {
    return {
      ok: false as const,
      error: error instanceof Error ? error.message : String(error),
      ...((error as { statusCode?: number })?.statusCode === 401 ? { needsOauthLogin: true as const } : {})
    }
  }
}

export function connection(profile?: null | string, connectionId = ''): HermesConnection {
  const selected = profile?.trim() || undefined
  const scoped = connectionId?.trim() || ''
  const local = isLocalConnection(scoped)
  const wsUrl = new URL(`${wsOrigin()}/api/ws`)
  if (scoped) wsUrl.searchParams.set('connectionId', scoped)
  if (selected) wsUrl.searchParams.set('profile', selected)
  return {
    authMode: local ? 'token' : 'oauth',
    baseUrl: window.location.origin,
    connectionId: scoped,
    isFullscreen: false,
    logs: [],
    mode: local ? 'local' : 'remote',
    nativeOverlayWidth: 0,
    profile: selected,
    remoteHost: window.location.host,
    remoteIdentity: scoped || 'local',
    remoteKind: local ? 'local' : 'url',
    source: 'settings',
    token: '',
    windowButtonPosition: null,
    wsUrl: wsUrl.toString()
  }
}

async function profiles(connectionId?: string) {
  const result = await requestJson<any>({
    path: '/api/profiles',
    ...(connectionId ? { connectionId } : {})
  } as HermesApiRequest & { connectionId?: string })
  return Array.isArray(result) ? result : Array.isArray(result?.profiles) ? result.profiles : []
}

async function webConnections() {
  return requestJson<any>({ path: '/api/web-connections' })
}

async function postWeb(path: string, body: JsonRecord) {
  return requestJson<any>({ path, method: 'POST', body })
}

async function roster() {
  const registry = await webConnections()
  const sources = Array.isArray(registry?.connections) ? registry.connections : []
  const groups = await Promise.all(sources.map(async (source: JsonRecord) => {
    const sourceId = String(source.id || '')
    const sourceBase = {
      connectionId: sourceId,
      kind: source.kind || 'remote',
      label: String(source.label || source.id || 'Hermes')
    }
    try {
      const rows = await profiles(sourceId)
      return {
        source: { ...sourceBase, reachable: true },
        agents: rows.map((row: JsonRecord) => {
          const profile = String(row.name || row.profile || 'default')
          const targetProfile = String(source.remoteProfile || profile)
          return {
            connectionId: sourceId,
            connectionKind: source.kind || 'remote',
            connectionLabel: sourceBase.label,
            handle: profile,
            profile,
            targetProfile
          }
        })
      }
    } catch (error) {
      return {
        source: {
          ...sourceBase,
          reachable: false,
          error: error instanceof Error ? error.message : 'Source profile enumeration failed'
        },
        agents: []
      }
    }
  }))
  return {
    agents: groups.flatMap(group => group.agents),
    primaryConnectionId: String(registry?.primary || ''),
    sources: groups.map(group => group.source)
  }
}

function browserNotification(payload: JsonRecord) {
  if (!('Notification' in window)) return false
  const show = () => {
    const notification = new Notification(payload.title || 'Hermes', {
      body: payload.body || '',
      icon: payload.icon,
      silent: Boolean(payload.silent),
      tag: payload.tag
    })
    if (payload.activate || payload.sessionId) {
      notification.onclick = () => window.focus()
    }
    return true
  }
  if (Notification.permission === 'granted') return show()
  if (Notification.permission === 'default') void Notification.requestPermission().then(value => value === 'granted' && show())
  return false
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.click()
  window.setTimeout(() => URL.revokeObjectURL(url), 1_000)
}

function eventGroup(methods: string[]) {
  return Object.fromEntries(methods.map(method => [method, noEvent]))
}

export function createWebHermesDesktop() {
  const native = (name: string) => (..._args: unknown[]) => unsupportedAsync(name)
  const desktop: JsonRecord = {
    glassSupported: false,
    translucencySupported: false,
    getConnection: async (profile?: string | null) => {
      const registry = await webConnections()
      return connection(profile, String(registry?.primary || ''))
    },
    getConnectionFor: async ({ connectionId, profile }: { connectionId?: string | null; profile?: string | null }) => ({
      ...connection(profile, connectionId || ''),
      registryScoped: true
    }),
    getProfileRoutes: async (names: string[]) => {
      const rosterResult = await roster()
      return rosterResult.agents
        .filter((agent: JsonRecord) => names.length === 0 || names.includes(agent.profile))
        .map((agent: JsonRecord) => ({
          connectionId: agent.connectionId,
          mode: 'remote',
          profile: agent.profile,
          targetProfile: agent.profile
        }))
    },
    revalidateConnection: async () => ({ ok: true, rebuilt: false }),
    touchBackend: async () => ({ ok: true }),
    getGatewayWsUrl: gatewayWsUrl,
    getGatewayWsUrlFor: ({ connectionId, profile }: { connectionId?: string | null; profile?: string | null }) => gatewayWsUrl(profile, connectionId),
    getAgentRoster: roster,
    openSessionWindow: (sessionId: string) => {
      window.open(`/#/sessions/${encodeURIComponent(sessionId)}`, '_blank', 'noopener,noreferrer')
      return Promise.resolve({ ok: true })
    },
    openSessionInTerminal: native('openSessionInTerminal'),
    openWindow: () => {
      window.open('/', '_blank', 'noopener,noreferrer')
      return Promise.resolve({ ok: true })
    },
    claimAmbientCue: async () => true,
    wakeIndicator: { getState: native('wakeIndicator.getState'), setState: native('wakeIndicator.setState'), onState: noEvent },
    petOverlay: { ...eventGroup(['onState', 'onControl']), open: native('petOverlay.open'), close: native('petOverlay.close'), setBounds: native('petOverlay.setBounds'), setIgnoreMouse: native('petOverlay.setIgnoreMouse'), setFocusable: native('petOverlay.setFocusable'), pushState: native('petOverlay.pushState'), control: native('petOverlay.control') },
    hud: { ...eventGroup(['onGoto', 'onChanged', 'onCursor', 'onGameOverlay']), open: native('hud.open'), close: native('hud.close'), setIgnoreMouse: native('hud.setIgnoreMouse'), moveBy: native('hud.moveBy'), setBounds: native('hud.setBounds'), setFrost: native('hud.setFrost'), setSession: native('hud.setSession') },
    quickEntry: { ...eventGroup(['onState', 'onSubmit', 'onShown']), getSettings: native('quickEntry.getSettings'), setSettings: native('quickEntry.setSettings'), submit: native('quickEntry.submit'), dismiss: native('quickEntry.dismiss'), pushState: async () => ({ ok: true }) },
    getBootProgress: async () => ({ phase: 'backend.ready', message: 'Hermes backend is ready', progress: 100, running: false, error: null }),
    // Browser deployments keep gateway secrets server-side; there is no OS
    // keychain to toggle in the Web host. Expose the updated Desktop contract
    // without pretending that browser storage is secure token storage.
    getSecretStorageEncryption: async () => ({ on: false }),
    setSecretStorageEncryption: async () => ({ on: false }),
    getConnectionConfig: async (profile?: string | null) => requestJson({
      path: withScope('/api/web-connection-config', profile)
    }),
    saveConnectionConfig: async (payload: JsonRecord) => {
      const result = await postWeb('/api/web-connection-config/save', payload)
      return result?.config || result
    },
    applyConnectionConfig: async (payload: JsonRecord) => {
      const result = await postWeb('/api/web-connection-config/save', { ...payload, apply: true })
      return result?.config || result
    },
    testConnectionConfig: (payload: JsonRecord) => postWeb('/api/web-connection-config/test', payload),
    connections: {
      list: webConnections,
      save: (payload: JsonRecord) => postWeb('/api/web-connections/save', payload),
      remove: (id: string) => postWeb('/api/web-connections/remove', { id }),
      setPrimary: (id: string) => postWeb('/api/web-connections/primary', { id }),
      setLaunchMode: (mode: 'last-used' | 'primary') => postWeb('/api/web-connections/launch-mode', { mode }),
      setLastUsed: (id: string) => postWeb('/api/web-connections/last-used', { id }),
      test: (id: string) => postWeb('/api/web-connections/test', { id }),
      updateAll: async () => ({ ok: false, results: [] }),
      onChanged: noEvent
    },
    sshConfigHosts: async () => ({ hosts: [] }),
    sshResolveHost: async () => ({ hosts: [] }),
    probeConnectionConfig: (remoteUrl: string) => postWeb('/api/web-connection-config/test', { mode: 'remote', remoteUrl }),
    oauthLoginConnectionConfig: async (remoteUrl: string) => {
      // Open synchronously from the click handler so popup blockers allow it;
      // the exact connection row is resolved before navigating the popup.
      const popup = window.open('/login', '_blank', 'popup,width=520,height=720')
      try {
        const registry = await webConnections()
        const wanted = new URL(remoteUrl.trim()).toString().replace(/\/$/, '')
        const match = (Array.isArray(registry?.connections) ? registry.connections : [])
          .find((row: JsonRecord) => {
            if (row.kind !== 'remote' && row.kind !== 'cloud') return false
            try { return new URL(String(row.url || '')).toString().replace(/\/$/, '') === wanted } catch { return false }
          })
        if (!match?.id) return { connected: false }
        const connectionId = String(match.id)
        if (popup) popup.location.href = `/login?connectionId=${encodeURIComponent(connectionId)}`
        for (let attempt = 0; attempt < 240; attempt += 1) {
          await new Promise(resolve => window.setTimeout(resolve, 500))
          const session = await requestJson<{ connected?: boolean }>({
            path: `/api/web-connection-session?connectionId=${encodeURIComponent(connectionId)}`
          })
          if (session?.connected) {
            await postWeb('/api/web-connections/primary', { id: connectionId })
            return { connected: true }
          }
          if (popup?.closed && attempt > 4) break
        }
      } catch {
        // The settings screen displays its normal sign-in failure state.
      }
      return { connected: false }
    },
    oauthLogoutConnectionConfig: async () => ({ ok: true }),
    cloud: {
      status: async () => ({ loggedIn: false }),
      login: async () => ({ ok: false, loggedIn: false }),
      logout: async () => ({ ok: true, loggedIn: false }),
      discover: async () => ({ agents: [] }),
      agentSignIn: async () => ({ ok: false })
    },
    profile: {
      get: async () => ({ profile: localStorage.getItem('hermes-web-profile') || null }),
      set: async (name: string | null) => {
        name ? localStorage.setItem('hermes-web-profile', name) : localStorage.removeItem('hermes-web-profile')
        return { profile: name || null }
      },
      remember: async (name: string | null) => {
        name ? localStorage.setItem('hermes-web-profile', name) : localStorage.removeItem('hermes-web-profile')
        return { profile: name || null }
      }
    },
    api: requestJson,
    notify: async (payload: JsonRecord) => browserNotification(payload),
    requestMicrophoneAccess: async () => { const stream = await navigator.mediaDevices.getUserMedia({ audio: true }); stream.getTracks().forEach(track => track.stop()); return true },
    readWindowBelow: native('readWindowBelow'),
    readFileDataUrl: async (path: string) => {
      const file = browserFileFromPath(path)
      return file ? fileAsDataUrl(file) : (await requestJson<{ dataUrl: string }>({ path: `/api/fs/read-data-url?path=${encodeURIComponent(path)}` })).dataUrl
    },
    readFileDataUrlForAttach: async (path: string) => {
      const file = browserFileFromPath(path)
      return file ? fileAsDataUrl(file) : (await requestJson<{ dataUrl: string }>({ path: `/api/fs/read-data-url?path=${encodeURIComponent(path)}` })).dataUrl
    },
    dataUrlReadMax: { get: async () => ({ maxMb: 16, defaultMaxMb: 16, maxBytes: 16 * 1024 * 1024 }), set: native('dataUrlReadMax.set') },
    readFileText: (path: string) => requestJson({ path: `/api/fs/read-text?path=${encodeURIComponent(path)}` }),
    readPluginSource: async (path: string) => (await requestJson<{ text: string }>({ path: `/api/fs/read-text?path=${encodeURIComponent(path)}` })).text,
    selectPaths: (options?: { directories?: boolean; multiple?: boolean; filters?: Array<{ extensions?: string[] }> }) => selectBrowserFiles(options),
    writeClipboard: (text: string) => navigator.clipboard.writeText(text),
    readClipboard: () => navigator.clipboard.readText(),
    saveGatewayFile: native('saveGatewayFile'),
    saveImageFromUrl: async (url: string) => { const response = await fetch(url); downloadBlob(await response.blob(), 'image'); return { ok: true } },
    contextMenuEdit: native('contextMenuEdit'), contextMenuCopyImage: native('contextMenuCopyImage'), contextMenuSpellcheck: native('contextMenuSpellcheck'), contextMenuGuestAddWord: native('contextMenuGuestAddWord'), onContextMenuSpellcheck: noEvent,
    saveImageBuffer: async (data: ArrayBuffer | Uint8Array, ext: string) => {
      const file = new File([data as unknown as BlobPart], `image.${ext.replace(/^\./, '')}`, { type: `image/${ext.replace(/^\./, '')}` })
      const path = browserFilePath(file)
      browserFiles.set(path, file)
      return path
    },
    saveClipboardImage: native('saveClipboardImage'),
    getPathForFile: () => '',
    normalizePreviewTarget: native('normalizePreviewTarget'), watchPreviewFile: native('watchPreviewFile'), watchDirectory: native('watchDirectory'), stopPreviewFileWatch: native('stopPreviewFileWatch'),
    setActiveWork: () => {}, setTitleBarTheme: () => {}, setNativeTheme: () => {}, setTranslucency: () => {}, setKeepAwake: () => {}, setDisableF12: () => {}, setPreviewShortcutActive: () => {},
    openExternal: async (url: string) => { window.open(url, '_blank', 'noopener,noreferrer'); return { ok: true } },
    openPreviewInBrowser: async (url: string) => { window.open(url, '_blank', 'noopener,noreferrer'); return { ok: true } },
    reachPreviewUrl: async (url: string) => { try { const response = await fetch(url, { method: 'HEAD' }); return { ok: response.ok, status: response.status, url } } catch (error) { return { ok: false, error: String(error), url } } },
    fetchLinkTitle: (url: string) => requestJson({ path: `/api/link-title?url=${encodeURIComponent(url)}` }),
    resolveFavicon: (url: string) => requestJson({ path: `/api/favicon?url=${encodeURIComponent(url)}` }),
    sanitizeWorkspaceCwd: async (cwd: string) => ({ cwd, sanitized: false }),
    settings: {
      getDefaultProjectDir: async () => { const result = await requestJson<{ cwd: string }>({ path: '/api/fs/default-cwd' }); return { dir: result.cwd, defaultLabel: result.cwd } },
      setDefaultProjectDir: native('settings.setDefaultProjectDir'), pickDefaultProjectDir: native('settings.pickDefaultProjectDir')
    },
    zoom: {
      get: async () => { const percent = Number(localStorage.getItem('hermes-web-zoom') || 100); return { level: 0, percent } },
      factor: () => Number(localStorage.getItem('hermes-web-zoom') || 100) / 100,
      setPercent: (percent: number) => { localStorage.setItem('hermes-web-zoom', String(percent)); document.documentElement.style.zoom = `${percent}%` },
      onChanged: noEvent
    },
    revealLogs: native('revealLogs'),
    getRecentLogs: async () => [], reportRendererError: (report: unknown) => console.error('[Hermes renderer]', report),
    readDir: (path: string) => requestJson({ path: `/api/fs/list?path=${encodeURIComponent(path)}` }),
    gitRoot: async (path: string) => (await requestJson<{ root: string | null }>({ path: `/api/fs/git-root?path=${encodeURIComponent(path)}` })).root,
    revealPath: native('revealPath'), openDir: native('openDir'), desktopPluginsRoot: async () => '', logsRoot: native('logsRoot'), agentPluginsRoot: async () => '', renamePath: native('renamePath'),
    writeTextFile: (path: string, content: string) => requestJson({ path: '/api/fs/write-text', method: 'POST', body: { path, content } }),
    trashPath: native('trashPath'),
    git: {
      // Dashboard Git routes return envelopes for list/diff operations; the
      // Electron bridge returns the inner values. Preserve that renderer contract.
      worktreeList: async (path: string) => (await requestJson<{ worktrees: unknown[] }>({ path: `/api/git/worktrees?path=${encodeURIComponent(path)}` })).worktrees || [],
      worktreeAdd: async (repoPath: string, options: JsonRecord = {}) => requestJson({ path: '/api/git/worktree/add', method: 'POST', body: { path: repoPath, ...options } }),
      worktreeRemove: async (repoPath: string, worktreePath: string, options: JsonRecord = {}) => requestJson({ path: '/api/git/worktree/remove', method: 'POST', body: { path: repoPath, worktreePath, force: Boolean(options.force) } }),
      branchSwitch: async (repoPath: string, branch: string) => requestJson({ path: '/api/git/branch/switch', method: 'POST', body: { path: repoPath, branch } }),
      branchList: async (path: string) => (await requestJson<{ branches: unknown[] }>({ path: `/api/git/branches?path=${encodeURIComponent(path)}` })).branches || [],
      baseBranchList: async (path: string) => (await requestJson<{ branches: unknown[] }>({ path: `/api/git/base-branches?path=${encodeURIComponent(path)}` })).branches || [],
      repoStatus: (path: string) => requestJson({ path: `/api/git/status?path=${encodeURIComponent(path)}` }),
      fileDiff: async (repo: string, file: string) => (await requestJson<{ diff: string }>({ path: `/api/git/file-diff?path=${encodeURIComponent(repo)}&file=${encodeURIComponent(file)}` })).diff || '',
      scanRepos: native('git.scanRepos'),
      review: {
        list: async (repoPath: string, scope: string, base?: string | null) => requestJson({ path: `/api/git/review/list?path=${encodeURIComponent(repoPath)}&scope=${encodeURIComponent(scope)}${base ? `&base=${encodeURIComponent(base)}` : ''}` }),
        diff: async (repoPath: string, filePath: string, scope: string, base?: string | null, staged = false) => (await requestJson<{ diff: string }>({ path: `/api/git/review/diff?path=${encodeURIComponent(repoPath)}&file=${encodeURIComponent(filePath)}&scope=${encodeURIComponent(scope)}${base ? `&base=${encodeURIComponent(base)}` : ''}&staged=${staged}` })).diff || '',
        stage: (repoPath: string, file?: string | null) => requestJson({ path: '/api/git/review/stage', method: 'POST', body: { path: repoPath, file: file || null } }),
        unstage: (repoPath: string, file?: string | null) => requestJson({ path: '/api/git/review/unstage', method: 'POST', body: { path: repoPath, file: file || null } }),
        revert: (repoPath: string, file?: string | null) => requestJson({ path: '/api/git/review/revert', method: 'POST', body: { path: repoPath, file: file || null } }),
        revParse: async (repoPath: string, ref?: string | null) => (await requestJson<{ sha: string | null }>({ path: `/api/git/review/rev-parse?path=${encodeURIComponent(repoPath)}${ref ? `&ref=${encodeURIComponent(ref)}` : ''}` })).sha,
        commit: (repoPath: string, message: string, push: boolean) => requestJson({ path: '/api/git/review/commit', method: 'POST', body: { path: repoPath, message, push: Boolean(push) } }),
        commitContext: (repoPath: string) => requestJson({ path: `/api/git/review/commit-context?path=${encodeURIComponent(repoPath)}` }),
        push: (repoPath: string) => requestJson({ path: '/api/git/review/push', method: 'POST', body: { path: repoPath } }),
        shipInfo: (repoPath: string) => requestJson({ path: `/api/git/review/ship-info?path=${encodeURIComponent(repoPath)}` }),
        prList: (repoPath: string, branches: string[], numbers: number[] = []) => requestJson({ path: '/api/git/review/pr-list', method: 'POST', body: { path: repoPath, branches, numbers } }),
        fetchPrComment: native('git.review.fetchPrComment'),
        createPr: (repoPath: string) => requestJson({ path: '/api/git/review/create-pr', method: 'POST', body: { path: repoPath } })
      }
    },
    terminal: new Proxy({}, { get: (_target, property) => property === 'onData' || property === 'onExit' ? noEvent : native(`terminal.${String(property)}`) }),
    ...eventGroup(['onClosePreviewRequested', 'onPreviewNav', 'onOpenFolderRequested', 'onOpenUpdatesRequested', 'onDeepLink', 'onWindowStateChanged', 'onFocusSession', 'onNotificationAction', 'onNotificationActivate', 'onPreviewFileChanged', 'onBackendExit', 'onConnectionApplied', 'onPowerResume', 'onBatteryChanged', 'onBootProgress', 'onBootstrapEvent', 'onFoundInPage', 'onOpenFindBarRequested']),
    signalDeepLinkReady: async () => ({ ok: true }), probePluginRepo: native('probePluginRepo'), installDesktopPlugin: native('installDesktopPlugin'),
    getOnBattery: async () => false,
    getBootstrapState: async () => null, continueBootstrapLocal: native('continueBootstrapLocal'), resetBootstrap: native('resetBootstrap'), repairBootstrap: native('repairBootstrap'), cancelBootstrap: native('cancelBootstrap'),
    getVersion: async () => { const result = await requestJson<{ version?: string }>({ path: '/api/status' }); return result.version || 'web' },
    getRemoteDisplayReason: async () => null,
    uninstall: { summary: native('uninstall.summary'), run: native('uninstall.run') },
    updates: { check: native('updates.check'), apply: native('updates.apply'), getBranch: native('updates.getBranch'), setBranch: native('updates.setBranch'), onProgress: noEvent },
    themes: { fetchMarketplace: native('themes.fetchMarketplace'), searchMarketplace: native('themes.searchMarketplace') },
    findInPage: native('findInPage'), stopFindInPage: native('stopFindInPage')
  }
  return desktop as Window['hermesDesktop']
}

export function installWebHermesDesktop() {
  if (!window.hermesDesktop) window.hermesDesktop = createWebHermesDesktop()
}
