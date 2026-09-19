import type { DesktopProfileRoute } from '@/global'

interface BrowserProfileOptions {
  basePath: string
  currentProfile: () => string | null
  requireConnection: (connectionId?: string | null) => void
}

function setUrlProfile(url: URL, profile: string | null): void {
  if (profile && profile !== 'default') {
    url.searchParams.set('profile', profile)
  } else {
    url.searchParams.delete('profile')
  }
}

export function createBrowserProfileBridge({
  basePath,
  currentProfile,
  requireConnection
}: BrowserProfileOptions): Pick<Window['hermesDesktop'], 'openWindow' | 'profile'> {
  // localStorage supplies origin isolation; the base path separates colocated servers.
  const storageKey = `hermes:webapp:default-profile:${basePath || '/'}`
  const listeners = new Set<(route: DesktopProfileRoute | null) => void>()

  const requireRoute = (value: unknown): DesktopProfileRoute => {
    if (!value || typeof value !== 'object') {
      throw new Error('A profile route is required.')
    }

    const { connectionId, profile } = value as Record<string, unknown>

    if (typeof profile !== 'string' || !/^[a-z0-9][a-z0-9_-]{0,63}$/.test(profile)) {
      throw new Error('Invalid profile name.')
    }

    if (connectionId !== null && typeof connectionId !== 'string') {
      throw new Error('Invalid connection id.')
    }

    requireConnection(connectionId)

    return { connectionId, profile }
  }

  const readDefault = (): DesktopProfileRoute | null => {
    try {
      return requireRoute(JSON.parse(window.localStorage.getItem(storageKey) || 'null'))
    } catch {
      return null
    }
  }

  const remember = async (profile: string | null) => {
    const selected = profile?.trim() || null
    const url = new URL(window.location.href)
    setUrlProfile(url, selected)
    window.history.replaceState(null, '', `${url.pathname}${url.search}${url.hash}`)

    return { profile: selected }
  }

  return {
    openWindow: async options => {
      const url = new URL(window.location.href)
      setUrlProfile(url, options === undefined ? currentProfile() : requireRoute(options).profile)

      if (options !== undefined) {
        url.searchParams.delete('win')
        url.searchParams.delete('watch')
        url.hash = '/'
      }

      window.open(url.href, '_blank', 'noopener,noreferrer')

      return { ok: true }
    },
    profile: {
      get: async () => ({ profile: currentProfile() }),
      getDefault: async () => readDefault(),
      setDefault: async value => {
        const route = requireRoute(value)
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
