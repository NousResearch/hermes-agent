import { externalUrlTarget, LOCAL_CONNECTION_ID } from '@hermes/shared'

import type { HermesApiRequest } from '@/global'
import { BROWSER_BRIDGE_STUBS } from '@/lib/browser-bridge-stubs'
import { createBrowserClipboardBridge } from '@/lib/browser-clipboard'
import { createBrowserCloudBridge } from '@/lib/browser-cloud'
import { createBrowserConnectionBridge } from '@/lib/browser-connection'
import { createBrowserDesktopPluginsBridge } from '@/lib/browser-desktop-plugins'
import { createBrowserFilesBridge } from '@/lib/browser-files'
import { createHostedRoute } from '@/lib/browser-hosted-route'
import { WEBAPP_LAUNCH_REQUIRED } from '@/lib/browser-launch-session'
import { onBrowserFocusSession, showBrowserNotification } from '@/lib/browser-notifications'
import { createBrowserProfileBridge } from '@/lib/browser-profile'
import { type BrowserSavedConnections, createBrowserSavedConnections } from '@/lib/browser-saved-connections'
import { browserFolderBasePath, createBrowserSettings } from '@/lib/browser-settings'
import { createBrowserTerminal } from '@/lib/browser-terminal'
import { browserApi, type BrowserBootstrap, browserBootstrap, ticketedWebsocketUrl } from '@/lib/browser-transport'
import { createBrowserUiPreferences } from '@/lib/browser-ui-preferences'
import { createBrowserUploadsBridge } from '@/lib/browser-uploads'
import {
  announceWebappWindow,
  createBrowserWindowOpener,
  sessionWindowTarget,
  withoutWindowTarget
} from '@/lib/browser-window'
import { createBrowserZoom } from '@/lib/browser-zoom'
import { signsInToCloudAgents } from '@/lib/hermes-cloud-host'
import { browserHostRunsNoAgent } from '@/lib/platform'
import { failDesktopBoot } from '@/store/boot'
import { $connection } from '@/store/session'
import { windowConnectionOverride, windowProfileOverride } from '@/store/windows'

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
/** The Webapp serves this machine's saved connections; the standalone dashboard does not. */
function savedConnectionsServed(): boolean {
  return (window as Window & { __HERMES_UI_SURFACE__?: string }).__HERMES_UI_SURFACE__ === 'webapp'
}

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

  // This machine's saved gateway connections (the list native Desktop keeps): each other gateway's
  // requests ride `/api/relay/<id>` on this origin (browser-saved-connections.ts). A host that does
  // not serve the list (the standalone dashboard) reaches only its own origin.
  const hostless = savedConnectionsServed() && browserHostRunsNoAgent()

  const relays: BrowserSavedConnections | null = savedConnectionsServed()
    ? createBrowserSavedConnections({ api, bootstrap, hostless })
    : null

  // A host that runs no agent has no `local` gateway: every member below and the connection bridge
  // resolve a request's agent through this one route (browser-hosted-route.ts).
  const hosted =
    hostless && relays
      ? createHostedRoute({
          launchAgent: windowConnectionOverride,
          onUnavailable: error => {
            // A window on another listed agent (it moved there, whatever agent its address still
            // names) keeps its route, and the address names that agent so Retry reopens on it. An
            // address naming the agent that left, or naming none while the window is still on it (it
            // opened on the primary that left), holds that agent's profile, chat and watch flag too;
            // without them, Retry opens the primary on a new chat.
            const named = windowConnectionOverride()
            const current = $connection.get()?.connectionId
            const movedTo = current && current !== error.agentId && relays.gateway(current) ? current : null
            const address = movedTo ? new URL(window.location.href) : null

            if (address && movedTo) {
              // The profile with it: one the address names may be the agent's that left.
              const profile = $connection.get()?.profile?.trim()
              address.searchParams.set('connectionId', movedTo)

              if (profile) {
                address.searchParams.set('profile', profile)
              } else {
                address.searchParams.delete('profile')
              }

              window.history.replaceState(window.history.state, '', `${address.pathname}${address.search}${address.hash}`)
            } else if (!named || named === error.agentId) {
              const cleared = withoutWindowTarget(window.location.href)
              // Retry opens the primary, as the message says: named, so startup's launch preference
              // (last used) does not send the window elsewhere.
              const primary = relays.knownWindowConnection()

              if (primary && primary !== error.agentId) {
                cleared.searchParams.set('connectionId', primary)
              }

              window.history.replaceState(window.history.state, '', `${cleared.pathname}${cleared.search}${cleared.hash}`)
            }

            failDesktopBoot(error.message)
          },
          relays,
          windowAgent: () => $connection.get()?.connectionId
        })
      : null

  const ownConnection = (connectionId?: null | string) =>
    hosted ? hosted.knownAgentFor(connectionId) : connectionId || LOCAL_CONNECTION_ID

  const routeTo = async (connectionId?: null | string) =>
    hosted ? hosted.agentFor(connectionId) : connectionId || LOCAL_CONNECTION_ID

  // Synchronous callers (file links, upload URLs) run after boot has read the list; before that a
  // hostless page has no gateway to name, and saying so beats sending the request to the host.
  const bootstrapFor = (connectionId?: null | string) => {
    const id = ownConnection(connectionId)

    if (!id) {
      throw new Error('This window has not opened a Hermes agent yet')
    }

    return relays?.bootstrapFor(id) ?? bootstrap
  }

  const activeConnectionId = () =>
    ownConnection($connection.get()?.connectionId) ?? LOCAL_CONNECTION_ID

  // A request names its connection; untagged, it goes to the one this window is on.
  const activeApi = async <T>(request: HermesApiRequest) =>
    browserApi<T>(relays?.bootstrapFor(await routeTo(request.connectionId ?? activeConnectionId())) ?? bootstrap, request)

  // The gateway's own routes (health, Portal status): the host's, or on a host with no agent, the
  // window's gateway's.
  const agentApi = hosted ? activeApi : api

  // Members with no connection of their own (host files, uploads) follow the window's connection.
  const activeBootstrap: BrowserBootstrap = {
    get authRequired() {
      return bootstrapFor(activeConnectionId()).authRequired
    },
    get basePath() {
      return bootstrapFor(activeConnectionId()).basePath
    },
    get privateSession() {
      return bootstrapFor(activeConnectionId()).privateSession
    },
    get token() {
      return bootstrapFor(activeConnectionId()).token
    }
  }

  // The renderer's REST door, owning session rows the way Electron's main process does: every row
  // this tab lists comes from the gateway the request went to, the source its sockets and new chats
  // already name (`local` for the one serving this page). An untagged listed row beside a `local`
  // new chat read as two owners, which split one gateway's sessions into an unowned group and a
  // `local` gateway group.
  const rendererApi = async <T>(request: HermesApiRequest): Promise<T> => {
    const owner = await routeTo(request.connectionId)
    const response = await browserApi<T>(relays?.bootstrapFor(owner) ?? bootstrap, request)

    return (request.method || 'GET').toUpperCase() === 'GET'
      ? (tagRegistrySessionResponse(request.path, response, owner) as T)
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

  const uploads = createBrowserUploadsBridge({
    api: activeApi,
    assetsBasePath: bootstrap.basePath,
    bootstrap: activeBootstrap,
    currentProfile,
    objectUrls,
    previewUrls
  })

  const files = createBrowserFilesBridge({
    api: activeApi,
    bootstrap: activeBootstrap,
    bootstrapFor: relays ? bootstrapFor : undefined,
    currentProfile
  })

  const bridge: Window['hermesDesktop'] = {
    ...BROWSER_BRIDGE_STUBS,
    // On a host with no agent the host's list is the account's agents, each signed in already.
    // Settings reads the same check to offer Transfer (`transferAvailable`).
    cloud: createBrowserCloudBridge({ api: agentApi, relays: signsInToCloudAgents() ? relays : null }),
    ...createBrowserClipboardBridge({ saveBuffer: uploads.saveImageBuffer }),
    ...createBrowserConnectionBridge({ api, bootstrap, hosted, relays }),
    ...(relays ? { connections: relays.connections } : {}),
    ...files,
    // The desktop-plugins folder is the serving machine's; a host with no agent has none to offer.
    ...(hosted ? {} : createBrowserDesktopPluginsBridge({ api, files })),
    ...uploads,
    api: rendererApi,
    // This renderer is built from the serving checkout, so its version is the
    // server's, read the way native Desktop reads its runtime's.
    getVersion: async () => ({
      appVersion: await resolveGatewayVersion(path => agentApi({ path, timeoutMs: 5000 })),
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
    // A browser has no idle clock that counts other apps; the reporter falls back to input in this tab.
    presence: {
      get: async () => ({ system: false, idleSeconds: null, locked: false, suspended: false, seq: 0, sampledAt: Date.now() }),
      onChange: () => () => undefined
    },
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
      // On a host with no agent the new window opens on the agent its address names: the chat's.
      // Elsewhere a window boots on this page's gateway, which does not hold another gateway's chats.
      const connectionId = hosted ? await hosted.agentFor(opts?.connectionId) : null

      if (!hosted && opts?.connectionId && opts.connectionId !== LOCAL_CONNECTION_ID) {
        return { error: 'A chat on another gateway opens in this tab, not in a new window.', ok: false }
      }

      const target = sessionWindowTarget(window.location.href, sessionId, { ...opts, connectionId })

      return target ? openWindow(target) : { error: 'invalid_session_id', ok: false }
    },
    settings: createBrowserSettings({ basePath: () => browserFolderBasePath(activeConnectionId()), currentProfile }),
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
      currentConnection: activeConnectionId,
      // The hosted dashboard closes every socket it relays with 4401 within 15 minutes; the terminal redials.
      redialsOn4401: Boolean(hosted),
      pinConnection: hosted ? hosted.agentFor : undefined,
      // The host terminal of the gateway the terminal started on, relayed when it is not the serving one.
      websocketUrl: async (profile, connectionId) => {
        if (!relays || (!hosted && connectionId === LOCAL_CONNECTION_ID)) {
          return ticketedWebsocketUrl(bootstrap, '/api/host-terminal', profile)
        }

        const target = await routeTo(connectionId)

        // A tab restored before this page read its saved connections.
        if (!relays.gateway(target)) {
          await relays.gateways()
        }

        return relays.socketUrl(target, '/api/host-terminal', profile)
      },
      defaultCwd: async (profile, connectionId) =>
        (await activeApi<{ cwd?: string }>({ connectionId, path: '/api/fs/default-cwd', profile })).cwd || '',
      // The chat's sign-in view for a launch-link tab. Behind the sign-in gate the next ticket
      // request already takes the tab to the login page.
      onSignedOut: () => {
        if (!bootstrap.authRequired) {
          failDesktopBoot(WEBAPP_LAUNCH_REQUIRED)
        }
      }
    }),
    ...createBrowserProfileBridge({
      basePath: bootstrap.basePath,
      currentConnection: hosted ? () => ownConnection($connection.get()?.connectionId) : undefined,
      currentProfile,
      openWindow
    }),
    uiPreferences: createBrowserUiPreferences(api),
    zoom: createBrowserZoom(bootstrap.basePath)
  }

  window.hermesDesktop = bridge
  document.documentElement.dataset.hermesDesktopHost = 'browser'

  return true
}
