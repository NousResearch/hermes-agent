import type { DesktopProfileRoute } from '@/global'

import { browserStartupProfileKey, readBrowserStartupProfile, requireProfileRoute } from './browser-startup-profile'

interface BrowserProfileOptions {
  basePath: string
  currentProfile: () => string | null
  /** On a host that runs no agent, the agent the window is on: the address names it beside the
   *  profile, so a reload or a new window opens on that agent, not on the list's primary. */
  currentConnection?: () => null | string | undefined
  openWindow: (target: URL) => Promise<{ ok: true }>
}

// A tab without `?profile=` boots on this origin's saved default, so a named
// profile — Default included — is always written out; only null clears it.
function setUrlProfile(url: URL, profile: string | null): void {
  if (profile) {
    url.searchParams.set('profile', profile)
  } else {
    url.searchParams.delete('profile')
  }
}

function setUrlConnection(url: URL, connectionId: null | string | undefined): void {
  if (connectionId) {
    url.searchParams.set('connectionId', connectionId)
  } else {
    url.searchParams.delete('connectionId')
  }
}

// A route naming an agent (on a host that runs no agent): its profile checked as any route's is.
const agentRoute = (value: unknown) => {
  const { connectionId } = (value ?? {}) as Record<string, unknown>
  const named = typeof connectionId === 'string' && connectionId ? connectionId : null

  return { connectionId: named, profile: requireProfileRoute({ ...(value as object), connectionId: null }).profile }
}

export function createBrowserProfileBridge({
  basePath,
  currentConnection,
  currentProfile,
  openWindow
}: BrowserProfileOptions): Pick<Window['hermesDesktop'], 'openWindow' | 'profile'> {
  // localStorage supplies origin isolation; the base path separates colocated servers.
  const storageKey = browserStartupProfileKey(basePath)
  const listeners = new Set<(route: DesktopProfileRoute | null) => void>()

  const readDefault = (): DesktopProfileRoute | null => readBrowserStartupProfile(basePath)

  const remember = async (profile: string | null) => {
    const selected = profile?.trim() || null
    const url = new URL(window.location.href)
    setUrlProfile(url, selected)

    if (currentConnection) {
      setUrlConnection(url, currentConnection())
    }

    // Keep react-router's entry state (idx/key/usr): only the URL changes.
    window.history.replaceState(window.history.state, '', `${url.pathname}${url.search}${url.hash}`)

    return { profile: selected }
  }

  return {
    openWindow: async options => {
      const url = new URL(window.location.href)

      if (currentConnection) {
        const route = options === undefined ? null : agentRoute(options)
        setUrlProfile(url, route ? route.profile : currentProfile())
        setUrlConnection(url, route?.connectionId ?? currentConnection())
      } else {
        setUrlProfile(url, options === undefined ? currentProfile() : requireProfileRoute(options).profile)
      }

      url.searchParams.delete('win')
      url.searchParams.delete('watch')
      url.hash = '/'

      return openWindow(url)
    },
    profile: {
      get: async () => ({ profile: currentProfile() }),
      getDefault: async () => readDefault(),
      setDefault: async value => {
        const route = requireProfileRoute(value)
        window.localStorage.setItem(storageKey, JSON.stringify(route))
        listeners.forEach(listener => listener(route))

        return route
      },
      onDefaultChanged: callback => {
        listeners.add(callback)

        const onStorage = (event: StorageEvent) => {
          if (event.storageArea === window.localStorage && (event.key === storageKey || event.key === null)) {
            callback(readDefault())
          }
        }

        window.addEventListener('storage', onStorage)

        return () => {
          listeners.delete(callback)
          window.removeEventListener('storage', onStorage)
        }
      },
      remember,
      set: async profile => {
        const result = await remember(profile)
        window.location.reload()

        return result
      }
    }
  }
}
