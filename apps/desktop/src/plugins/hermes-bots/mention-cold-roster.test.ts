/**
 * Cross-connection @mentions must resolve in a regular chat even when the
 * Bots pane has never been mounted this launch.
 *
 * `useRoster` is the only caller of `host.agents()` (the union roster across
 * every registered connection), and its only call site is the Bots pane. A
 * launch that never opens that pane therefore leaves the roster cache empty,
 * and both composer surfaces read the cache synchronously. The middleware's
 * cold-cache fallback asked the ACTIVE gateway for `profiles.list`, which by
 * definition cannot see agents on other connections — so a cross-connection
 * mention silently passed through unresolved while a same-gateway mention
 * still worked.
 *
 * The gateway is still CONNECTING when the plugin registers, so the prime has
 * to run on the 'open' transition; priming eagerly at registration fetches
 * against a closed socket and leaves the cache empty (caught in a live app,
 * not by an earlier version of this test).
 */

import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

interface MentionCompletionItem {
  display: string
  insert: string
  meta: string
}

interface ComposerDraft {
  text: string
}

interface Contribution {
  area: string
  data: {
    handler?: (draft: ComposerDraft) => Promise<ComposerDraft>
    provide?: (query: string) => MentionCompletionItem[]
  }
  id: string
}

/** The active (local) gateway's own profiles.list — it has no knowledge of
 *  agents living on other connections. */
const LOCAL_PROFILES = [{ name: 'default' }, { name: 'writer' }]

/** What `host.agents()` reports across every registered connection. */
const UNION = {
  agents: [
    {
      connectionId: 'local',
      connectionKind: 'local',
      connectionLabel: 'This device',
      handle: 'default',
      profile: 'default'
    },
    {
      connectionId: 'local',
      connectionKind: 'local',
      connectionLabel: 'This device',
      handle: 'writer',
      profile: 'writer'
    },
    { connectionId: 'box', connectionKind: 'remote', connectionLabel: 'Box', handle: 'analyst', profile: 'analyst' }
  ],
  primaryConnectionId: 'local',
  sources: [
    { connectionId: 'local', kind: 'local', label: 'This device', reachable: true },
    { connectionId: 'box', kind: 'remote', label: 'Box', reachable: true }
  ]
}

const { cache, gatewayListeners, gatewayState, hostMock } = vi.hoisted(() => {
  const gatewayListeners: Array<(state: string) => void> = []
  // Registration happens while the socket is still connecting — the shape
  // that makes an eager register-time prime a silent no-op.
  const gatewayState = { value: 'connecting' }

  return {
    cache: new Map<string, { key: unknown[]; value: unknown }>(),
    gatewayListeners,
    gatewayState,
    hostMock: {
      agents: vi.fn(async () => UNION),
      notify: vi.fn(),
      profileRoutes: vi.fn(async () => []),
      request: vi.fn(async () => ({ profiles: LOCAL_PROFILES })),
      requestProfile: vi.fn(async () => ({ profiles: LOCAL_PROFILES })),
      state: {
        connectionId: { get: () => 'local', listen: () => () => undefined },
        focusedSessionProfile: { get: () => 'default', listen: () => () => undefined },
        focusedStoredSessionId: { get: () => null, listen: () => () => undefined },
        gateway: {
          get: () => gatewayState.value,
          listen: (fn: (state: string) => void) => {
            gatewayListeners.push(fn)

            return () => undefined
          }
        },
        profile: { get: () => 'default', listen: () => () => undefined }
      }
    }
  }
})

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')

  const stub: unknown = new Proxy(function stubbed() {}, {
    apply: () => stub,
    get: (_target, key) => (key === 'then' ? undefined : stub)
  })

  const keyOf = (key: unknown[]) => JSON.stringify(key)

  // Faithful enough to TanStack Query v5 for these surfaces: getQueryData is
  // exact-key, getQueriesData prefix-matches, fetchQuery runs the queryFn and
  // writes the result under its key.
  const queryClient = {
    fetchQuery: async ({ queryKey, queryFn }: { queryFn: () => Promise<unknown>; queryKey: unknown[] }) => {
      const value = await queryFn()
      cache.set(keyOf(queryKey), { key: queryKey, value })

      return value
    },
    getQueriesData: ({ queryKey }: { queryKey: unknown[] }) =>
      [...cache.values()]
        .filter(entry => queryKey.every((part, index) => entry.key[index] === part))
        .map(entry => [entry.key, entry.value]),
    getQueryData: (key: unknown[]) => cache.get(keyOf(key))?.value,
    invalidateQueries: () => undefined,
    setQueryData: (key: unknown[], value: unknown) => cache.set(keyOf(key), { key, value })
  }

  const known: Record<string, unknown> = {
    atom,
    COMPOSER_AREAS: { atCompletions: 'composer.atCompletions', middleware: 'composer.middleware' },
    host: hostMock,
    PALETTE_AREA: 'palette',
    queryClient
  }

  return new Proxy(known, {
    get: (target, key) =>
      typeof key === 'symbol' || key in target ? target[key as string] : key === 'then' ? undefined : stub,
    has: () => true
  })
})

/** Register the plugin with a COLD cache and a still-connecting gateway. */
async function coldContributions() {
  vi.resetModules()
  cache.clear()
  gatewayListeners.length = 0
  gatewayState.value = 'connecting'

  const plugin = (await import('./plugin')).default
  const registered: Contribution[] = []

  try {
    plugin.register({
      i18n: { register: () => () => undefined },
      onDispose: () => undefined,
      register: (contribution: Contribution) => registered.push(contribution),
      storage: { get: async () => undefined, remove: async () => undefined, set: async () => undefined }
    } as never)
  } catch {
    /* registration walks UI surfaces this stub does not model */
  }

  return {
    handler: registered.find(entry => entry.id === 'mention-middleware')!.data.handler!,
    provide: registered.find(entry => entry.id === 'mention-completions')!.data.provide!
  }
}

/** The socket finishes connecting; flush the prime it triggers. */
async function openGateway() {
  gatewayState.value = 'open'

  for (const fn of gatewayListeners) {
    fn('open')
  }

  await new Promise(resolve => setTimeout(resolve, 0))
}

beforeAll(async () => {
  await import('./plugin')
}, 120_000)

beforeEach(() => {
  vi.clearAllMocks()
})

describe('cross-connection @mentions with the Bots pane never mounted', () => {
  it('identifies an agent on another connection and annotates its relay target', async () => {
    const { handler } = await coldContributions()
    await openGateway()

    const result = await handler({ text: 'ask @analyst to check the logs' })

    expect(result.text).toMatch(/@analyst = agent profile "analyst"/)
    expect(result.text).toMatch(/message_agent target: "analyst@box"/)
  })

  it('offers cross-connection handles in the @ autocomplete', async () => {
    const { provide } = await coldContributions()
    await openGateway()

    expect(provide('').map(item => item.insert)).toContain('@analyst')
  })

  it('still resolves a same-connection mention', async () => {
    const { handler } = await coldContributions()
    await openGateway()

    const result = await handler({ text: 'ask @writer to draft it' })

    expect(result.text).toMatch(/@writer = agent profile "writer"/)
  })

  it('resolves even if the mention arrives before the gateway-open prime', async () => {
    // The middleware fills a cold cache itself rather than falling back to the
    // active gateway's profiles.list, which cannot see other connections.
    const { handler } = await coldContributions()

    const result = await handler({ text: 'ask @analyst to check the logs' })

    expect(result.text).toMatch(/message_agent target: "analyst@box"/)
  })

  it('leaves an unknown handle untouched', async () => {
    const { handler } = await coldContributions()
    await openGateway()

    const result = await handler({ text: 'email someone@example.com and @nobody' })

    expect(result.text).toBe('email someone@example.com and @nobody')
  })
})
