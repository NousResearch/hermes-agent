import type { HermesNotification } from '../electron/notification-types'
import type { DesktopAgentRoster, DesktopConnectionConfig, DesktopConnectionsRegistry, DesktopProfileRoute, HermesPreviewTarget, HermesReadFileTextResult } from '../src/global'

import { selectServerDirectory } from './directory-picker'
import { browserMessage, sharedMessage } from './messages'
import { settings } from './settings'
import { terminal } from './terminal'
import { backendFetch, backendPath, connection, connectionId, gatewayWsUrl, requestJson, setActiveProfile } from './transport'
import { noEvent, unsupportedAsync } from './unsupported'

type JsonRecord = Record<string, unknown>


// Browser-selected files have no usable filesystem path. Keep their bytes in a
// short-lived, renderer-local map and expose opaque virtual paths to the
// existing Desktop composer contract. The gateway only receives bytes through
// its normal image.attach_bytes/file.attach data flow.
const browserFiles = new Map<string, File>()
let browserFileSequence = 0
const MAX_UPLOAD_BYTES = 16 * 1024 * 1024
const MAX_UPLOADS = 32
const uploadExpiry = new Map<string, number>()

export function registerBrowserFile(file: File) {
  const now = Date.now()

  for (const [path, expiry] of uploadExpiry) {
    if (expiry <= now) { browserFiles.delete(path); uploadExpiry.delete(path) }
  }

  if (file.size > MAX_UPLOAD_BYTES) {throw new Error(browserMessage('attachmentSize'))}

  if (browserFiles.size >= MAX_UPLOADS) {throw new Error(browserMessage('attachmentCount'))}
  const path = browserFilePath(file)
  browserFiles.set(path, file)
  uploadExpiry.set(path, now + 30 * 60_000)
  window.setTimeout(() => { browserFiles.delete(path); uploadExpiry.delete(path) }, 30 * 60_000)

  return path
}

function browserFilePath(file: File) {
  browserFileSequence += 1

  return `web-upload://${browserFileSequence}/${encodeURIComponent(file.name || 'image')}`
}

function browserFileFromPath(path: string) {
  if (!path.startsWith('web-upload://')) {return undefined}

  const file = browserFiles.get(path)

  if (!file || (uploadExpiry.get(path) || 0) <= Date.now()) {
    browserFiles.delete(path)
    uploadExpiry.delete(path)
    throw new Error(browserMessage('attachmentExpired'))
  }

  return file
}

function fileAsDataUrl(file: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(reader.error || new Error(sharedMessage('common.error')))
    reader.onload = () => resolve(String(reader.result || ''))
    reader.readAsDataURL(file)
  })
}

async function selectBrowserFiles(options?: { directories?: boolean; multiple?: boolean; filters?: Array<{ extensions?: string[] }> }) {
  if (options?.directories) {return selectServerDirectory()}
  const input = document.createElement('input')
  input.type = 'file'
  input.multiple = options?.multiple !== false
  const extensions = options?.filters?.flatMap(filter => filter.extensions || []).filter(Boolean) || []

  if (extensions.length) {input.accept = extensions.map(extension => `.${extension.replace(/^\\./, '')}`).join(',')}
  input.style.display = 'none'
  document.body.appendChild(input)

  try {
    const selected = await new Promise<File[]>(resolve => {
      input.addEventListener('cancel', () => resolve([]), { once: true })
      input.addEventListener('change', () => resolve(Array.from(input.files || [])), { once: true })
      input.click()
    })

    return selected.map(file => {
      return registerBrowserFile(file)
    })
  } finally {
    input.remove()
  }
}

function registry(): DesktopConnectionsRegistry {
  return { version: 1, primary: connectionId, secureTokenStorage: false,
    connections: [{ id: connectionId, kind: 'remote', label: window.location.host,
      url: window.location.origin, authMode: window.__HERMES_AUTH_REQUIRED__ ? 'oauth' : 'token',
      tokenSet: false, tokenPreview: null }] }
}

function connectionConfig(): DesktopConnectionConfig {
  return { envOverride: false, mode: 'remote', profile: null,
    remoteAuthMode: window.__HERMES_AUTH_REQUIRED__ ? 'oauth' : 'token',
    remoteOauthConnected: Boolean(window.__HERMES_AUTH_REQUIRED__), remoteTokenPreview: null,
    remoteTokenSet: false, secureTokenStorage: false, remoteTokenPlainText: false,
    remoteUrl: window.location.origin, cloudOrg: '', sshHost: '', sshUser: '', sshPort: null,
    sshKeyPath: '', sshRemoteHermesPath: '', sshRemoteProfile: '' }
}

async function roster(): Promise<DesktopAgentRoster> {
  const result = await requestJson<{ profiles: Array<{ name: string }> }>({ path: '/api/profiles' })

  return { agents: result.profiles.map(({ name }) => ({ connectionId, connectionKind: 'remote',
    connectionLabel: window.location.host, profile: name, targetProfile: name, handle: name })),
    sources: [{ connectionId, kind: 'remote', label: window.location.host, reachable: true }] }
}

function browserNotification(payload: HermesNotification) {
  if (!('Notification' in window)) {return false}

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

  if (Notification.permission === 'granted') {return show()}

  if (Notification.permission === 'default') {void Notification.requestPermission().then(value => value === 'granted' && show())}

  return false
}

export async function normalizePreviewTarget(value: string, baseDir?: string): Promise<HermesPreviewTarget | null> {
  if (/^https?:\/\//i.test(value)) {
    const url = new URL(value)

    return { kind: 'url', url: url.toString(), label: url.hostname, source: value }
  }

  if (/^[a-z][a-z0-9+.-]*:/i.test(value) && !/^[A-Za-z]:[\\/]/.test(value)) {return null}
  const path = baseDir && !/^(?:\/|[A-Za-z]:[\\/])/.test(value) ? `${baseDir.replace(/[\\/]$/, '')}/${value}` : value
  const file = await requestJson<HermesReadFileTextResult>({ path: `/api/fs/read-text?path=${encodeURIComponent(path)}` })
  const kind = file.mimeType?.startsWith('image/') ? 'image' : file.mimeType === 'text/html' ? 'html' : file.mimeType === 'application/pdf' ? 'pdf' : file.binary ? 'binary' : 'text'

  return { kind: 'file', path: file.path, url: file.path, source: value, label: file.path.split(/[\\/]/).pop() || file.path,
    previewKind: kind, language: file.language, mimeType: file.mimeType, binary: file.binary, byteSize: file.byteSize }
}

function openExternal(value: string) {
  const url = new URL(value, window.location.origin)

  if (!['http:', 'https:'].includes(url.protocol)) {throw new Error(browserMessage('httpLinks'))}
  window.open(url.toString(), '_blank', 'noopener,noreferrer')
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.click()
  window.setTimeout(() => URL.revokeObjectURL(url), 1_000)
}

function eventGroup<const K extends string>(methods: K[]) {
  return Object.fromEntries(methods.map(method => [method, noEvent])) as Record<K, typeof noEvent>
}

export function createBrowserHost() {
  const native = (name: string) => (..._args: unknown[]) => unsupportedAsync(name)

  const desktop: Window['hermesDesktop'] = {
    windowControls: { custom: false, minimize: () => {}, toggleMaximize: () => {}, close: () => window.close() },
    openBrowserWindow: native('Open this preview in a browser tab'),
    onBrowserPopoutClosed: noEvent,
    glassSupported: false,
    translucencySupported: false,
    getConnection: async (profile) => connection(profile),
    getConnectionFor: async ({ profile, connectionId: id }) => {
      if (id && id !== connectionId) {throw new Error(browserMessage('backendTab'))}

      return connection(profile)
    },
    setActiveConnectionRoute: (route) => setActiveProfile(route?.profile),
    getProfileRoutes: async (names) => (await roster()).agents
      .filter(agent => names.length === 0 || names.includes(agent.profile))
      .map(agent => ({ connectionId, mode: 'remote', profile: agent.profile, targetProfile: agent.profile })),
    revalidateConnection: async () => {
      await requestJson({ path: '/api/status' })

      return { ok: true, rebuilt: false }
    },
    getGatewayWsUrl: gatewayWsUrl,
    getGatewayWsUrlFor: ({ connectionId: id, profile }) => gatewayWsUrl(profile, id),
    getAgentRoster: roster,
    openSessionWindow: async (sessionId, options = {}) => {
      if (!sessionId || (options.connectionId && options.connectionId !== connectionId)) {return { ok: false, error: browserMessage('invalidRoute') }}
      const url = new URL(backendPath('/'), window.location.origin)
      url.searchParams.set('win', 'secondary')
      url.searchParams.set('connectionId', connectionId)
      const profile = options.profile || connection().profile

      if (profile) {url.searchParams.set('profile', profile)}

      if (options.watch) {url.searchParams.set('watch', '1')}
      const { sessionRoute } = await import('../src/app/routes')
      url.hash = sessionRoute(sessionId)
      window.open(url.toString(), '_blank', 'noopener,noreferrer')

      return { ok: true }
    },
    openSessionInTerminal: native('openSessionInTerminal'),
    openWindow: async (options) => {
      if (options?.connectionId && options.connectionId !== connectionId) {return { ok: false, error: browserMessage('backendTab') }}
      const url = new URL(backendPath('/'), window.location.origin)
      const profile = options?.profile || connection().profile

      if (profile) {url.searchParams.set('profile', profile)}
      window.open(url.toString(), '_blank', 'noopener,noreferrer')

      return { ok: true }
    },
    claimAmbientCue: async () => true,
    wakeIndicator: { getState: native('wakeIndicator.getState'), setState: native('wakeIndicator.setState'), onState: noEvent },
    petOverlay: { ...eventGroup(['onState', 'onControl']), open: native('petOverlay.open'), close: native('petOverlay.close'), setBounds: native('petOverlay.setBounds'), setIgnoreMouse: native('petOverlay.setIgnoreMouse'), setFocusable: native('petOverlay.setFocusable'), pushState: native('petOverlay.pushState'), control: native('petOverlay.control') },
    hud: { nativeDrag: false, beginMove: () => {}, endMove: () => {}, resetLayout: native('hud.resetLayout'), ...eventGroup(['onGoto', 'onChanged', 'onCursor', 'onGameOverlay']), open: native('hud.open'), close: native('hud.close'), setIgnoreMouse: native('hud.setIgnoreMouse'), moveBy: native('hud.moveBy'), setBounds: native('hud.setBounds'), setFrost: native('hud.setFrost'), setSession: native('hud.setSession') },
    quickEntry: { ackSubmit: native('quickEntry.ackSubmit'), ...eventGroup(['onState', 'onSubmit', 'onShown', 'onLateResult']), getSettings: native('quickEntry.getSettings'), setSettings: native('quickEntry.setSettings'), submit: native('quickEntry.submit'), dismiss: native('quickEntry.dismiss'), pushState: async () => ({ ok: true }) },
    getBootProgress: async () => ({ phase: 'backend.ready', message: sharedMessage('boot.ready'), progress: 100, running: false, error: null, fakeMode: false, timestamp: Date.now() }),
    // Browser deployments keep gateway secrets server-side; there is no OS
    // keychain to toggle in the Web host. Expose the updated Desktop contract
    // without pretending that browser storage is secure token storage.
    getSecretStorageEncryption: async () => ({ on: false }),
    setSecretStorageEncryption: async () => ({ on: false }),
    getConnectionConfig: async () => connectionConfig(),
    saveConnectionConfig: native('Change backend address by opening its URL'),
    applyConnectionConfig: native('Change backend address by opening its URL'),
    testConnectionConfig: async () => {
      await requestJson({ path: '/api/status' })

      return { ok: true, mode: 'remote' }
    },
    connections: {
      list: async () => registry(),
      save: native('Manage backend bookmarks in your browser'),
      remove: native('Remove backend bookmarks in your browser'),
      setPrimary: async (id) => {
        if (id !== connectionId) {throw new Error(browserMessage('backendTab'))}

        return { ok: true, registry: registry() }
      },
      setLaunchMode: native('Browser startup is managed by your browser'),
      setLastUsed: async () => ({ ok: true, registry: registry() }),
      test: async () => { await requestJson({ path: '/api/status' });

 return { ok: true } },
      updateAll: native('Update the backend on its server'),
      onChanged: noEvent
    },
    sshConfigHosts: async () => ({ hosts: [] }),
    sshResolveHost: native('SSH tunnels are managed on the server'),
    probeConnectionConfig: async () => { await requestJson({ path: '/api/status' });

 return { reachable: true, baseUrl: window.location.origin, authMode: window.__HERMES_AUTH_REQUIRED__ ? 'oauth' : 'token', providers: [], version: null, error: null } },
    oauthLoginConnectionConfig: async () => {
      window.location.assign(backendPath('/login'))

      return { ok: true, connected: false, baseUrl: window.location.origin }
    },
    oauthLogoutConnectionConfig: async () => {
      const response = await backendFetch('/auth/logout', { method: 'POST' })

      if (!response.ok) {throw new Error(sharedMessage('settings.gateway.signOutFailed'))}
      window.location.assign(backendPath('/login'))

      return { ok: true, connected: false }
    },
    cloud: {
      status: async () => ({ portalBaseUrl: '', signedIn: false }),
      login: native('Open your cloud backend URL in this browser'),
      logout: native('Sign out on your cloud backend'),
      discover: async () => ({ agents: [] }),
      agentSignIn: native('Open your cloud backend URL in this browser')
    },
    profile: {
      getDefault: async () => {
        try {
          const value: DesktopProfileRoute | null = JSON.parse(localStorage.getItem('hermes-browser-default-profile') || 'null')

          return value && typeof value.profile === 'string' && value.connectionId === connectionId ? value : null
        } catch { return null }
      },
      setDefault: async (route) => { localStorage.setItem('hermes-browser-default-profile', JSON.stringify(route));

 return route },
      onDefaultChanged: noEvent,
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
    notify: async (payload) => browserNotification(payload),
    requestMicrophoneAccess: async () => { const stream = await navigator.mediaDevices.getUserMedia({ audio: true }); stream.getTracks().forEach(track => track.stop());

 return true },
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
    readPluginSource: (path: string) => requestJson({ path: `/api/fs/read-text?path=${encodeURIComponent(path)}` }),
    selectPaths: (options?: { directories?: boolean; multiple?: boolean; filters?: Array<{ extensions?: string[] }> }) => selectBrowserFiles(options),
    writeClipboard: async (text: string) => { await navigator.clipboard.writeText(text);

 return true },
    readClipboard: () => navigator.clipboard.readText(),
    saveGatewayFile: async ({ path, suggestedName, profile, sessionId }) => {
      const query = new URLSearchParams({ path })

      if (profile) {query.set('profile', profile)}

      if (sessionId) {query.set('session_id', sessionId)}
      const response = await backendFetch(`/api/fs/download?${query}`)

      if (!response.ok) {throw new Error(sharedMessage('fileMenu.downloadFailed'))}
      downloadBlob(await response.blob(), suggestedName || 'download')

      return { saved: true }
    },
    saveImageFromUrl: async (url: string) => { const response = await fetch(url);

 if (!response.ok) {throw new Error(sharedMessage('desktop.imageDownloadFailed'));} downloadBlob(await response.blob(), 'image');

 return true },
    contextMenuEdit: native('contextMenuEdit'), contextMenuCopyImage: native('contextMenuCopyImage'), contextMenuSpellcheck: native('contextMenuSpellcheck'), contextMenuGuestAddWord: native('contextMenuGuestAddWord'), onContextMenuSpellcheck: noEvent,
    saveImageBuffer: async (data: ArrayBuffer | Uint8Array, ext: string) => {
      const file = new File([data as unknown as BlobPart], `image.${ext.replace(/^\./, '')}`, { type: `image/${ext.replace(/^\./, '')}` })

      return registerBrowserFile(file)
    },
    savePastedText: async (text) => {
      const file = new File([text], 'pasted.txt', { type: 'text/plain' })

      return registerBrowserFile(file)
    },
    saveClipboardImage: async () => {
      for (const item of await navigator.clipboard.read()) {
        const type = item.types.find(type => type.startsWith('image/'))

        if (type) {return registerBrowserFile(new File([await item.getType(type)], 'clipboard-image', { type }))}
      }

      throw new Error(sharedMessage('desktop.noClipboardImage'))
    },
    getPathForFile: registerBrowserFile,
    normalizePreviewTarget: normalizePreviewTarget, watchPreviewFile: native('watchPreviewFile'), watchDirectory: native('watchDirectory'), stopPreviewFileWatch: native('stopPreviewFileWatch'),
    setActiveWork: () => {}, setTitleBarTheme: () => {}, setNativeTheme: () => {}, setTranslucency: () => {}, setKeepAwake: () => {}, setDisableF12: () => {}, setPreviewShortcutActive: () => {},
    openExternal: async (url: string) => { openExternal(url) },
    openPreviewInBrowser: async (url: string) => { openExternal(url) },
    reachPreviewUrl: async (url: string) => new URL(url, window.location.origin).toString(),
    // Optional metadata has no backend route; preserve the renderer's fallback.
    fetchLinkTitle: async () => '',
    resolveFavicon: async () => '',
    sanitizeWorkspaceCwd: async (cwd) => ({ cwd: cwd || (await requestJson<{ cwd: string }>({ path: '/api/fs/default-cwd' })).cwd, sanitized: !cwd }),
    settings,
    zoom: {
      get: async () => { const percent = Number(localStorage.getItem('hermes-web-zoom') || 100);

 return { level: 0, percent } },
      factor: () => Number(localStorage.getItem('hermes-web-zoom') || 100) / 100,
      setPercent: (percent: number) => { localStorage.setItem('hermes-web-zoom', String(percent)); document.documentElement.style.zoom = `${percent}%` },
      onChanged: noEvent
    },
    revealLogs: native('revealLogs'),
    getRecentLogs: native('Read backend logs through the Logs settings page'), reportRendererError: (report: unknown) => console.error('[Hermes renderer]', report),
    readDir: (path: string) => requestJson({ path: `/api/fs/list?path=${encodeURIComponent(path)}` }),
    gitRoot: async (path: string) => (await requestJson<{ root: string | null }>({ path: `/api/fs/git-root?path=${encodeURIComponent(path)}` })).root,
    revealPath: native('revealPath'), openDir: native('openDir'), desktopPluginsRoot: async () => '', logsRoot: native('logsRoot'), agentPluginsRoot: async () => '', renamePath: native('renamePath'),
    writeTextFile: (path: string, content: string) => requestJson({ path: '/api/fs/write-text', method: 'POST', body: { path, content } }),
    trashPath: native('trashPath'),
    git: {
      // Dashboard Git routes return envelopes for list/diff operations; the
      // Electron bridge returns the inner values. Preserve that renderer contract.
      worktreeList: async (path: string) => (await requestJson<{ worktrees: Awaited<ReturnType<NonNullable<Window['hermesDesktop']['git']>['worktreeList']>> }>({ path: `/api/git/worktrees?path=${encodeURIComponent(path)}` })).worktrees || [],
      worktreeAdd: async (repoPath: string, options: JsonRecord = {}) => requestJson({ path: '/api/git/worktree/add', method: 'POST', body: { path: repoPath, ...options } }),
      worktreeRemove: async (repoPath: string, worktreePath: string, options: JsonRecord = {}) => requestJson({ path: '/api/git/worktree/remove', method: 'POST', body: { path: repoPath, worktreePath, force: Boolean(options.force) } }),
      branchSwitch: async (repoPath: string, branch: string) => requestJson({ path: '/api/git/branch/switch', method: 'POST', body: { path: repoPath, branch } }),
      branchList: async (path: string) => (await requestJson<{ branches: Awaited<ReturnType<NonNullable<Window['hermesDesktop']['git']>['branchList']>> }>({ path: `/api/git/branches?path=${encodeURIComponent(path)}` })).branches || [],
      baseBranchList: async (path: string) => (await requestJson<{ branches: Awaited<ReturnType<NonNullable<Window['hermesDesktop']['git']>['baseBranchList']>> }>({ path: `/api/git/base-branches?path=${encodeURIComponent(path)}` })).branches || [],
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
        createPr: (repoPath: string) => requestJson({ path: '/api/git/review/create-pr', method: 'POST', body: { path: repoPath } })
      }
    },
    terminal,
    ...eventGroup(['onClosePreviewRequested', 'onPreviewNav', 'onOpenFolderRequested', 'onOpenUpdatesRequested', 'onDeepLink', 'onWindowStateChanged', 'onFocusSession', 'onNotificationAction', 'onNotificationActivate', 'onPreviewFileChanged', 'onBackendExit', 'onConnectionApplied', 'onPowerResume', 'onBatteryChanged', 'onBootProgress', 'onBootstrapEvent', 'onFoundInPage', 'onOpenFindBarRequested']),
    signalDeepLinkReady: async () => ({ ok: true }), probePluginRepo: native('probePluginRepo'), installDesktopPlugin: native('installDesktopPlugin'),
    getOnBattery: async () => false,
    getBootstrapState: native('Bootstrap runs on the server'), continueBootstrapLocal: native('continueBootstrapLocal'), resetBootstrap: native('resetBootstrap'), repairBootstrap: native('repairBootstrap'), cancelBootstrap: native('cancelBootstrap'),
    getSyncStatus: async () => null,
    updateHold: { recheck: native('updateHold.recheck'), quit: native('updateHold.quit'), startAnyway: native('updateHold.startAnyway') },
    getVersion: async () => { const result = await requestJson<{ version?: string }>({ path: '/api/status' });

 return { appVersion: result.version || 'browser', electronVersion: '', nodeVersion: '', platform: 'browser', hermesRoot: '' } },
    getRemoteDisplayReason: async () => null,
    uninstall: { openAppsSettings: native('uninstall.openAppsSettings'), summary: native('uninstall.summary'), run: native('uninstall.run') },
    updates: { check: native('updates.check'), apply: native('updates.apply'), onProgress: noEvent },
    themes: { fetchMarketplace: native('themes.fetchMarketplace'), searchMarketplace: native('themes.searchMarketplace') },
    findInPage: native('findInPage'), stopFindInPage: native('stopFindInPage')
  }

  return desktop
}

export function installBrowserHost() {
  if (!window.hermesDesktop) {window.hermesDesktop = createBrowserHost()}
}
