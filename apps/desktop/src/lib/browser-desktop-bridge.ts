import { externalUrlTarget } from '@hermes/shared'

import type { HermesApiRequest } from '@/global'
import { BROWSER_BRIDGE_STUBS } from '@/lib/browser-bridge-stubs'
import { createBrowserClipboardBridge } from '@/lib/browser-clipboard'
import { createBrowserConnectionBridge, requireBrowserConnection } from '@/lib/browser-connection'
import { createBrowserFilesBridge } from '@/lib/browser-files'
import { watchWebappLaunchLink } from '@/lib/browser-launch-session'
import { createBrowserProfileBridge } from '@/lib/browser-profile'
import { createBrowserTerminal } from '@/lib/browser-terminal'
import {
  authenticatedWebsocketUrl,
  browserApi,
  browserBootstrap,
  type BrowserBootstrapWindow
} from '@/lib/browser-transport'
import { createBrowserUploadsBridge } from '@/lib/browser-uploads'
import { createBrowserWindowOpener, sessionWindowTarget } from '@/lib/browser-window'
import { createBrowserZoom } from '@/lib/browser-zoom'
import { $connection } from '@/store/session'

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

  if (bootstrap.privateSession) {
    watchWebappLaunchLink(() => window.location.reload())
  }

  const api = <T>(request: HermesApiRequest) => browserApi<T>(bootstrap, request)

  const currentProfile = () =>
    $connection.get()?.profile?.trim() || new URLSearchParams(window.location.search).get('profile')

  const objectUrls = new Set<string>()

  window.addEventListener(
    'beforeunload',
    () => {
      objectUrls.forEach(url => URL.revokeObjectURL(url))
      objectUrls.clear()
    },
    { once: true }
  )

  const openWindow = createBrowserWindowOpener({
    api,
    basePath: bootstrap.basePath,
    privateSession: bootstrap.privateSession
  })

  // Electron's opener rule: agent- or tool-supplied links must not run script
  // or custom schemes on this origin, whose cookie authenticates the API. A
  // `file:` URL names the server's disk, and this browser cannot open it.
  const openExternal = async (url: string) => {
    const target = externalUrlTarget(url)

    if (target?.kind === 'web') {window.open(target.url, '_blank', 'noopener,noreferrer')}
  }

  const uploads = createBrowserUploadsBridge({ api, bootstrap, currentProfile, objectUrls })

  const bridge: Window['hermesDesktop'] = {
    ...BROWSER_BRIDGE_STUBS,
    ...createBrowserClipboardBridge({ saveBuffer: uploads.saveImageBuffer }),
    ...createBrowserConnectionBridge({ api, bootstrap }),
    ...createBrowserFilesBridge({
      api,
      bootstrap,
      currentProfile,
      objectUrls,
      requireConnection: requireBrowserConnection
    }),
    ...uploads,
    api,
    findInPage: async (query: string) => {
      const find = (window as Window & { find?: (value: string) => boolean }).find

      return { count: query && find?.call(window, query) ? 1 : 0 }
    },
    notify: async ({ title, body }: { title?: string; body?: string }) => {
      if (!('Notification' in window)) {return false}

      if (Notification.permission === 'default') {await Notification.requestPermission()}

      if (Notification.permission !== 'granted') {return false}
      new Notification(title || 'Hermes', { body })

      return true
    },
    openExternal,
    openPreviewInBrowser: openExternal,
    openSessionWindow: async (sessionId, opts) => {
      const target = sessionWindowTarget(window.location.href, sessionId, opts)

      return target ? openWindow(target) : { error: 'invalid_session_id', ok: false }
    },
    requestMicrophoneAccess: async () => {
      if (!navigator.mediaDevices?.getUserMedia) {return false}
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      stream.getTracks().forEach(track => track.stop())

      return true
    },
    terminal: createBrowserTerminal({
      basePath: bootstrap.basePath,
      currentProfile,
      websocketUrl: profile => authenticatedWebsocketUrl(bootstrap, '/api/host-terminal', profile),
      defaultCwd: async profile => (await api<{ cwd?: string }>({ path: '/api/fs/default-cwd', profile })).cwd || ''
    }),
    ...createBrowserProfileBridge({
      basePath: bootstrap.basePath,
      currentProfile,
      openWindow,
      requireConnection: requireBrowserConnection
    }),
    zoom: createBrowserZoom(bootstrap.basePath)
  }

  win.hermesDesktop = bridge
  document.documentElement.dataset.hermesDesktopHost = 'browser'

  return true
}
