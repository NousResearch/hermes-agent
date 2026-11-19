import { atom } from 'nanostores'

import { logoutBrowserSession } from '@/lib/browser-session'
import { endpointUrl } from '@/lib/browser-transport'
import { clerkErrorMessage } from '@/lib/clerk-session-operations'
import { getUnitalkInstance, initializeUnitalkInstance, UnitalkResponseError } from '@/lib/unitalk-instance-api'

export interface WebappIdentity {
  sessionId: string
  userId: string
  organizationId: null | string
}

export interface WebappInstanceState {
  identity: WebappIdentity | null
  status: 'checking' | 'ready' | 'signed-out' | 'missing' | 'inactive' | 'redirecting' | 'error' |
    'unauthorized' | 'organization-required'
  error: null | string
}

interface WebappInstanceOptions {
  basePath: string
  origin: string
  fetch: typeof fetch
  refreshSession: () => Promise<unknown>
  pause: () => void
  navigate: (url: string) => void
}

function instanceFailureStatus(error: unknown): WebappInstanceState['status'] {
  if (!(error instanceof UnitalkResponseError)) {
    return 'error'
  }

  if (error.status === 401) {
    return 'unauthorized'
  }

  const lookupFailures: Partial<Record<number, WebappInstanceState['status']>> = {
    403: 'organization-required',
    404: 'missing'
  }

  return error.endpoint === '/api/instance' ? lookupFailures[error.status] ?? 'error' : 'error'
}

export function sameWebappIdentity(left: WebappIdentity | null, right: WebappIdentity | null): boolean {
  return (
    left?.sessionId === right?.sessionId &&
    left?.userId === right?.userId &&
    left?.organizationId === right?.organizationId
  )
}

/** One owner serializes cookie-changing requests; superseded lookups can never redirect. */
export function createWebappInstanceSession(options: WebappInstanceOptions) {
  const state = atom<WebappInstanceState>({ error: null, identity: null, status: 'checking' })
  let active: {
    identity: WebappIdentity | null
    controller: AbortController
    promise: Promise<void>
    changing: boolean
  } | null = null
  let queue = Promise.resolve()
  let loggedOut = false

  async function logout() {
    if (!loggedOut) {
      await logoutBrowserSession(options.basePath, options.fetch)
      loggedOut = true
    }
  }

  async function fail(error: unknown, identity: WebappIdentity | null, signal: AbortSignal) {
    if (signal.aborted) {
      return
    }

    let status = instanceFailureStatus(error)
    let message = clerkErrorMessage(error)

    if (status !== 'error') {
      try {
        await logout()
      } catch (logoutError) {
        status = 'error'
        message = `${message}. ${clerkErrorMessage(logoutError)}`
      }
    }

    if (!signal.aborted) {
      state.set({ error: message, identity, status })
    }
  }

  async function hasSession(): Promise<boolean> {
    if (loggedOut) {
      return false
    }

    const response = await options.fetch(endpointUrl('/api/auth/me', options.basePath), {
      cache: 'no-store',
      credentials: 'same-origin',
      headers: { Accept: 'application/json' },
      signal: AbortSignal.timeout(30_000)
    })

    if (response.status === 401 || response.redirected) {
      return false
    }

    if (!response.ok) {
      throw new Error(`Could not check the Hermes session (${response.status})`)
    }

    return true
  }

  async function resolve(identity: WebappIdentity | null, signal: AbortSignal) {
    const publish = (status: WebappInstanceState['status']) => {
      if (!signal.aborted) {
        state.set({ error: null, identity, status })
      }
    }

    if (!identity) {
      await logout()
      publish('signed-out')

      return
    }

    await options.refreshSession()

    if (signal.aborted) {
      return
    }

    const instance = await getUnitalkInstance(options.fetch, signal)

    if (signal.aborted) {
      return
    }

    if (!instance) {
      await logout()
      publish('missing')

      return
    }

    if (instance.ownerId !== identity.organizationId || instance.userId !== identity.userId) {
      throw new Error('The instance does not match the selected account or organization. Try again.')
    }

    if (!instance.ready) {
      await logout()
      publish('inactive')

      return
    }

    const sameHost = new URL(instance.origin).hostname === new URL(options.origin).hostname
    const authenticatedHere = sameHost && (await hasSession())

    if (signal.aborted) {
      return
    }

    if (authenticatedHere) {
      publish('ready')

      return
    }

    await logout()

    if (signal.aborted) {
      return
    }

    publish('redirecting')
    // Cookie writes must finish before the next logout/login, even if intent changes.
    const url = await initializeUnitalkInstance(options.fetch, identity.sessionId, instance)

    if (sameHost) {
      loggedOut = false
    }

    if (signal.aborted) {
      return
    }

    if (sameHost) {
      publish('ready')
    } else {
      options.navigate(url)
    }
  }

  function synchronize(identity: WebappIdentity | null, change?: () => Promise<unknown>): Promise<void> {
    if (!change && active && sameWebappIdentity(active.identity, identity)) {
      return active.promise
    }

    // A previous setActive can publish an intermediate org while the newer action
    // waits its turn. Its hook update must not supersede the user's queued choice.
    if (
      !change &&
      active?.changing &&
      identity &&
      active.identity?.sessionId === identity.sessionId &&
      active.identity?.userId === identity.userId
    ) {
      return active.promise
    }

    active?.controller.abort()
    const controller = new AbortController()
    const { signal } = controller
    state.set({ error: null, identity, status: 'checking' })
    options.pause()

    const promise = queue.then(async () => {
      if (signal.aborted) {
        return
      }

      try {
        await change?.()

        if (active?.controller === controller) {
          active.changing = false
        }

        if (!signal.aborted) {
          await resolve(identity, signal)
        }
      } catch (error) {
        if (active?.controller === controller) {
          active.changing = false
        }

        await fail(error, identity, signal)
      }
    })

    active = { changing: Boolean(change), controller, identity, promise }
    queue = promise

    return promise.then(() => {
      const current = state.get()

      if (change && !signal.aborted && current.status === 'error') {
        throw new Error(current.error ?? 'Could not switch instance')
      }
    })
  }

  return {
    state,
    synchronize,
    retry: (identity: WebappIdentity | null = active?.identity ?? null) => {
      active?.controller.abort()
      active = null

      return synchronize(identity)
    },
    dispose: () => {
      active?.controller.abort()
      active = null
    }
  }
}

export type WebappInstanceSession = ReturnType<typeof createWebappInstanceSession>
