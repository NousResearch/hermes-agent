// Fake window.hermesDesktop bridge for demo mode. Answers the renderer's REST
// calls (window.hermesDesktop.api) from fixtures and stubs the rest of the
// preload surface so the app boots with no Electron host / backend.
import type { HermesApiRequest } from '../global'

import {
  CONFIG,
  FS_TREE,
  MESSAGES,
  MODEL_INFO,
  MODEL_OPTIONS,
  PROFILES,
  SESSIONS,
  SKILLS,
  STATUS,
  TOOLSETS
} from './fixtures'
import { control } from './gateway'

const noop = () => {}
const off = () => noop

function route(req: HermesApiRequest): unknown {
  const path = (req.path || '').split('?')[0]
  const method = (req.method || 'GET').toUpperCase()

  if (path === '/api/config' || path === '/api/config/defaults') {
    return CONFIG
  }

  if (path === '/api/config/schema') {
    return { fields: {}, category_order: [] }
  }

  if (path === '/api/status') {
    return STATUS
  }

  if (path === '/api/logs') {
    return { file: 'gateway.log', lines: [] }
  }

  if (path === '/api/sessions') {
    return { sessions: SESSIONS, total: SESSIONS.length, offset: 0, limit: 40 }
  }

  if (path === '/api/sessions/search') {
    return { results: [] }
  }

  const messages = path.match(/^\/api\/sessions\/([^/]+)\/messages$/)

  if (messages) {
    const id = decodeURIComponent(messages[1])

    return { session_id: id, messages: MESSAGES[id] || [] }
  }

  if (path === '/api/model/info') {
    return MODEL_INFO
  }

  if (path === '/api/model/options') {
    return MODEL_OPTIONS
  }

  if (path === '/api/model/auxiliary') {
    return { main: { provider: MODEL_OPTIONS.provider, model: MODEL_OPTIONS.model }, tasks: [] }
  }

  if (path === '/api/skills') {
    return SKILLS
  }

  if (path === '/api/tools/toolsets') {
    return TOOLSETS
  }

  if (path === '/api/cron/jobs') {
    return []
  }

  if (path === '/api/profiles') {
    return { profiles: PROFILES }
  }

  const soul = path.match(/^\/api\/profiles\/([^/]+)\/soul$/)

  if (soul) {
    return {
      content: `# ${decodeURIComponent(soul[1])} persona\n\nYou are Hermes — concise, capable, and autonomous.`,
      exists: true
    }
  }

  const setupCommand = path.match(/^\/api\/profiles\/([^/]+)\/setup-command$/)

  if (setupCommand) {
    return { command: `hermes --profile ${decodeURIComponent(setupCommand[1])}` }
  }

  if (path === '/api/messaging/platforms') {
    return { platforms: [] }
  }

  if (path === '/api/providers/oauth') {
    return { providers: [] }
  }

  if (path === '/api/env') {
    return {}
  }

  if (path === '/api/analytics/usage') {
    return { by_model: [], daily: [], period_days: 30, skills: { summary: {}, top_skills: [] }, totals: {} }
  }

  if (method !== 'GET') {
    return { ok: true }
  }

  return {}
}

export const bridge = {
  getConnection: () =>
    Promise.resolve({
      baseUrl: 'http://127.0.0.1:5174',
      isFullscreen: false,
      mode: 'local' as const,
      nativeOverlayWidth: 0,
      source: 'local' as const,
      token: 'demo',
      wsUrl: 'ws://127.0.0.1:5174/__demo_gateway',
      logs: [],
      windowButtonPosition: null
    }),
  getBootProgress: () =>
    Promise.resolve({
      error: null,
      fakeMode: true,
      message: 'Ready',
      phase: 'ready',
      progress: 100,
      running: false,
      timestamp: Date.now()
    }),
  getConnectionConfig: () =>
    Promise.resolve({
      envOverride: false,
      mode: 'local' as const,
      remoteTokenPreview: null,
      remoteTokenSet: false,
      remoteUrl: ''
    }),
  saveConnectionConfig: () =>
    Promise.resolve({
      envOverride: false,
      mode: 'local' as const,
      remoteTokenPreview: null,
      remoteTokenSet: false,
      remoteUrl: ''
    }),
  applyConnectionConfig: () =>
    Promise.resolve({
      envOverride: false,
      mode: 'local' as const,
      remoteTokenPreview: null,
      remoteTokenSet: false,
      remoteUrl: ''
    }),
  testConnectionConfig: () => Promise.resolve({ baseUrl: 'http://127.0.0.1:5174', ok: true, version: '0.0.0-demo' }),
  api<T>(request: HermesApiRequest): Promise<T> {
    return Promise.resolve(route(request) as T)
  },
  notify: () => Promise.resolve(true),
  requestMicrophoneAccess: () => Promise.resolve(false),
  readFileDataUrl: () => Promise.resolve(''),
  readFileText: () => Promise.resolve({ path: '', text: '' }),
  selectPaths: () => Promise.resolve([] as string[]),
  writeClipboard: () => Promise.resolve(true),
  saveImageFromUrl: () => Promise.resolve(true),
  saveImageBuffer: () => Promise.resolve(''),
  saveClipboardImage: () => Promise.resolve(''),
  getPathForFile: () => '',
  normalizePreviewTarget: () => Promise.resolve(null),
  watchPreviewFile: () => Promise.resolve({ id: 'w1', path: '' }),
  stopPreviewFileWatch: () => Promise.resolve(true),
  setTitleBarTheme: noop,
  setPreviewShortcutActive: noop,
  openExternal: () => Promise.resolve(),
  fetchLinkTitle: () => Promise.resolve(''),
  revealLogs: () => Promise.resolve({ ok: true, path: '~/.hermes/logs' }),
  getRecentLogs: () => Promise.resolve({ path: 'gateway.log', lines: [] }),
  readDir: (path: string) => {
    const key = (path || '').replace(/\/+$/, '')

    const entries = (FS_TREE[key] || FS_TREE[path] || []).map(e => ({
      name: e.name,
      isDirectory: e.isDirectory,
      path: `${key}/${e.name}`
    }))

    return Promise.resolve({ entries })
  },
  gitRoot: () => Promise.resolve(null),
  terminal: {
    dispose: () => Promise.resolve(true),
    onData: (_id: string, callback: (payload: string) => void) => {
      control.termCb = callback

      return noop
    },
    onExit: off,
    resize: () => Promise.resolve(true),
    start: () => Promise.resolve({ id: 't1', cwd: '~/code/hermes-agent', shell: 'zsh' }),
    write: () => Promise.resolve(true)
  },
  onClosePreviewRequested: off,
  onOpenUpdatesRequested: off,
  onWindowStateChanged: off,
  onPreviewFileChanged: off,
  onBackendExit: off,
  onBootProgress: off,
  getBootstrapState: () =>
    Promise.resolve({
      active: false,
      manifest: null,
      stages: {},
      error: null,
      log: [],
      startedAt: null,
      completedAt: null,
      unsupportedPlatform: null
    }),
  resetBootstrap: () => Promise.resolve({ ok: true }),
  repairBootstrap: () => Promise.resolve({ ok: true }),
  onBootstrapEvent: off,
  getVersion: () =>
    Promise.resolve({
      appVersion: '0.0.0-demo',
      electronVersion: '',
      nodeVersion: '',
      platform: 'web',
      hermesRoot: '~/.hermes'
    }),
  updates: {
    check: () => Promise.resolve({ supported: false, reason: 'demo' }),
    apply: () => Promise.resolve({ ok: true }),
    getBranch: () => Promise.resolve({ branch: 'demo' }),
    setBranch: (name: string) => Promise.resolve({ branch: name }),
    onProgress: off
  }
} satisfies Window['hermesDesktop']

export function installBridge(): void {
  window.hermesDesktop = bridge
}
