import { externalUrlTarget, LOCAL_CONNECTION_ID } from '@hermes/shared'

import type { HermesApiRequest } from '@/global'
import { BROWSER_BRIDGE_STUBS } from '@/lib/browser-bridge-stubs'
import { createBrowserClipboardBridge } from '@/lib/browser-clipboard'
import { createBrowserConnectionBridge } from '@/lib/browser-connection'
import { createBrowserDesktopPluginsBridge } from '@/lib/browser-desktop-plugins'
import { createBrowserFilesBridge } from '@/lib/browser-files'
import { WEBAPP_LAUNCH_REQUIRED } from '@/lib/browser-launch-session'
import { onBrowserFocusSession, showBrowserNotification } from '@/lib/browser-notifications'
import { createBrowserProfileBridge } from '@/lib/browser-profile'
import { createBrowserSettings } from '@/lib/browser-settings'
import { createBrowserTerminal } from '@/lib/browser-terminal'
import { browserApi, browserBootstrap, ticketedWebsocketUrl } from '@/lib/browser-transport'
import { createBrowserUiPreferences } from '@/lib/browser-ui-preferences'
import { createBrowserUploadsBridge } from '@/lib/browser-uploads'
import { announceWebappWindow, createBrowserWindowOpener, sessionWindowTarget } from '@/lib/browser-window'
import { createBrowserZoom } from '@/lib/browser-zoom'
import { failDesktopBoot } from '@/store/boot'
import { $connection } from '@/store/session'
import { windowProfileOverride } from '@/store/windows'

import { resolveGatewayVersion } from '../../electron/gateway-version'
import { tagRegistrySessionResponse } from '../../electron/profile-session-routing'

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
  if (window.hermesDesktop) {
    return false
  }

  const bootstrap = browserBootstrap()

  if (!bootstrap) {
    return false
  }

  // A gated host's opener waits for this window to say it booted (createBrowserWindowOpener).
  if (!bootstrap.privateSession) {
    announceWebappWindow()
  }

  const api = <T>(request: HermesApiRequest) => browserApi<T>(bootstrap, request)

  // The renderer's REST door, owning session rows the way Electron's main process does: every row
  // this tab lists comes from the one gateway serving it, the `local` source the tab's sockets and
  // new chats already name. An untagged listed row beside a `local` new chat read as two owners,
  // which split one gateway's sessions into an unowned group and a `local` gateway group.
  const rendererApi = async <T>(request: HermesApiRequest): Promise<T> => {
    const response = await api<T>(request)

    return (request.method || 'GET').toUpperCase() === 'GET'
      ? (tagRegistrySessionResponse(request.path, response, LOCAL_CONNECTION_ID) as T)
      : response
  }

  const currentProfile = () => $connection.get()?.profile?.trim() || windowProfileOverride()

  const objectUrls = new Set<string>()
  const previewUrls = new Set<string>()

  window.addEventListener(
    'beforeunload',
    () => {
      objectUrls.forEach(url => URL.revokeObjectURL(url))
      objectUrls.clear()
      previewUrls.clear()
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

    if (target?.kind === 'web') {
      window.open(target.url, '_blank', 'noopener,noreferrer')
    }
  }

  const uploads = createBrowserUploadsBridge({ api, bootstrap, currentProfile, objectUrls, previewUrls })
  const files = createBrowserFilesBridge({ api, bootstrap, currentProfile })

  const bridge: Window['hermesDesktop'] = {
    ...BROWSER_BRIDGE_STUBS,
    ...createBrowserClipboardBridge({ saveBuffer: uploads.saveImageBuffer }),
    ...createBrowserConnectionBridge({ api, bootstrap }),
    ...files,
    ...createBrowserDesktopPluginsBridge({ api, files }),
    ...uploads,
    api: rendererApi,
    // This renderer is built from the serving checkout, so its version is the
    // server's, read the way native Desktop reads its runtime's.
    getVersion: async () => ({
      appVersion: await resolveGatewayVersion(path => api({ path, timeoutMs: 5000 })),
      electronVersion: '',
      hermesRoot: '',
      nodeVersion: '',
      platform: 'browser'
    }),
    findInPage: async (query: string) => {
      const find = (window as Window & { find?: (value: string) => boolean }).find

      return { count: query && find?.call(window, query) ? 1 : 0 }
    },
    notify: async payload => showBrowserNotification(payload),
    onFocusSession: onBrowserFocusSession,
    openExternal,
    openPreviewInBrowser: async url => {
      // Only this install's sandbox wrappers may open as same-origin blobs.
      if (previewUrls.has(url)) {
        window.open(url, '_blank', 'noopener,noreferrer')
      } else {
        await openExternal(url)
      }
    },
    openSessionWindow: async (sessionId, opts) => {
      const target = sessionWindowTarget(window.location.href, sessionId, opts)

      return target ? openWindow(target) : { error: 'invalid_session_id', ok: false }
    },
    settings: createBrowserSettings({ basePath: bootstrap.basePath, currentProfile }),
    requestMicrophoneAccess: async () => {
      if (!navigator.mediaDevices?.getUserMedia) {
        return false
      }

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      stream.getTracks().forEach(track => track.stop())

      return true
    },
    terminal: createBrowserTerminal({
      basePath: bootstrap.basePath,
      currentProfile,
      websocketUrl: profile => ticketedWebsocketUrl(bootstrap, '/api/host-terminal', profile),
      defaultCwd: async profile => (await api<{ cwd?: string }>({ path: '/api/fs/default-cwd', profile })).cwd || '',
      // The chat's sign-in view for a launch-link tab. Behind the sign-in gate the next ticket
      // request already takes the tab to the login page.
      onSignedOut: () => {
        if (!bootstrap.authRequired) {
          failDesktopBoot(WEBAPP_LAUNCH_REQUIRED)
        }
      }
    }),
    ...createBrowserProfileBridge({ basePath: bootstrap.basePath, currentProfile, openWindow }),
    uiPreferences: createBrowserUiPreferences(api),
    zoom: createBrowserZoom(bootstrap.basePath)
  }

  window.hermesDesktop = bridge
  document.documentElement.dataset.hermesDesktopHost = 'browser'

  return true
}
