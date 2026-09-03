/**
 * The two composer contributions Bot Mode registers, driven through the real
 * `plugin.register()`: @-mention completions and the mention middleware.
 *
 * Both read the roster IMPERATIVELY (a popover must answer per keystroke, and
 * the middleware runs on submit), so both go through the query cache rather
 * than the hook. `useRoster` keys its query on `[...ROSTER_KEY, connectionId]`
 * — one entry per connection the window has been on — and these readers used
 * to call `getQueryData(ROSTER_KEY)` with the BARE key. That is an exact-key
 * match in TanStack Query, so it matched NOTHING: completions offered no
 * roster handles, and a remote `@name-device` mention passed through the
 * middleware unhandled (the local `profiles.list` fallback only knows bare
 * local names). Reproduced on a two-source install where the same profile
 * name exists locally and on an SSH connection ("Vera"): typing
 * `@default-vera` did nothing.
 *
 * The middleware is IDENTIFICATION-ONLY by design. It annotates who a tag
 * resolves to and hands the agent a `message_agent` target; the renderer
 * composes no CLI handoff and delivers nothing itself.
 */

import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

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

/** One profile name on two sources: locally and on the SSH connection "vera".
 *  The union roster disambiguates the remote row as @default-vera. */
const ROSTER = {
  profiles: [
    { connectionId: 'local', connectionKind: 'local', connectionLabel: 'This device', name: 'default' },
    {
      connectionId: 'vera',
      connectionKind: 'ssh',
      connectionLabel: 'Vera',
      handle: 'default-vera',
      name: 'default',
      remoteSource: true,
      sourceScoped: true
    }
  ]
}

const { cache, hostMock, live } = vi.hoisted(() => {
  const live = { connection: 'local', focused: 'default', profile: 'default' }

  return {
    cache: new Map<string, { key: unknown[]; value: unknown }>(),
    hostMock: {
      agents: vi.fn(
        async (): Promise<{
          agents: Array<Record<string, unknown>>
          primaryConnectionId?: string
          sources: unknown[]
        }> => ({ agents: [], sources: [] })
      ),
      notify: vi.fn(),
      profileRoutes: vi.fn(
        async (): Promise<
          Array<{ connectionId: string; mode: 'local' | 'remote'; profile: string; targetProfile: string }>
        > => []
      ),
      request: vi.fn(async () => ({})),
      requestProfile: vi.fn(
        async (_route: { connectionId: string; targetProfile?: string }): Promise<Record<string, unknown>> => ({})
      ),
      state: {
        connectionId: { get: () => live.connection, listen: () => () => undefined },
        focusedSessionProfile: { get: () => live.focused, listen: () => () => undefined },
        focusedStoredSessionId: { get: () => null, listen: () => () => undefined },
        gateway: { get: () => null, listen: () => () => undefined },
        profile: { get: () => live.profile, listen: () => () => undefined }
      }
    },
    live
  }
})

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')

  const stub: unknown = new Proxy(function stubbed() {}, {
    apply: () => stub,
    get: (_target, key) => (key === 'then' ? undefined : stub)
  })

  // A faithful-enough TanStack Query v5 cache: getQueryData matches ONLY the
  // exact key; getQueriesData prefix-matches a key family.
  const keyOf = (key: unknown[]) => JSON.stringify(key)

  const queryClient = {
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

interface Fixture {
  /** Which connection id the cache entry is filed under — 'local' is the one
   *  the live window would write. */
  cacheKeyConnection?: string
  /** Profile owning the chat on screen; a bot never @s itself. */
  focused?: string
  profiles?: Array<Record<string, unknown>>
}

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(settle => {
    resolve = settle
  })

  return { promise, resolve }
}

/** Register the plugin and hand back its composer contributions. */
async function contributions({
  cacheKeyConnection = 'local',
  focused = 'default',
  profiles = ROSTER.profiles
}: Fixture = {}) {
  vi.resetModules()
  cache.clear()
  live.focused = focused
  // Exactly where useRoster writes it: suffixed with the connection id.
  cache.set(JSON.stringify(['hermes-bots', 'roster', cacheKeyConnection]), {
    key: ['hermes-bots', 'roster', cacheKeyConnection],
    value: { profiles }
  })

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
    // Registration walks UI surfaces the stub does not fully model; the
    // contributions registered before any throw are what these tests drive.
  }

  const completions = registered.find(entry => entry.id === 'mention-completions')
  const middleware = registered.find(entry => entry.id === 'mention-middleware')

  expect(completions?.data.provide).toBeTypeOf('function')
  expect(middleware?.data.handler).toBeTypeOf('function')

  return { handler: middleware!.data.handler!, provide: completions!.data.provide! }
}

// The plugin entry pulls in every Bot Mode surface; transforming that graph
// once costs more than a single test's budget.
beforeAll(async () => {
  await import('./plugin')
}, 120_000)

beforeEach(() => {
  vi.clearAllMocks()
  hostMock.agents.mockResolvedValue({ agents: [], sources: [] })
  hostMock.profileRoutes.mockResolvedValue([])
  hostMock.requestProfile.mockResolvedValue({})
  live.connection = 'local'
  live.focused = 'default'
  live.profile = 'default'
})

afterEach(() => vi.useRealTimers())

describe('@-mention completions', () => {
  it('offers the remote @name-device handle from the suffixed cache entry', async () => {
    const { provide } = await contributions()

    expect(provide('').map(item => item.insert)).toContain('@default-vera')
  })

  it('still resolves when the only cached roster is under another connection id', async () => {
    // The window moved connections; the stale entry is the only one cached.
    const { provide } = await contributions({ cacheKeyConnection: 'vera' })

    expect(provide('').map(item => item.insert)).toContain('@default-vera')
  })

  it('surfaces default as @hermes and prefix-filters on the handle', async () => {
    const { provide } = await contributions({
      focused: 'researcher',
      profiles: [
        { name: 'default' },
        { name: 'researcher' },
        { connectionLabel: 'Homelab', handle: 'writer-homelab', name: 'writer' }
      ]
    })

    expect(provide('').map(item => item.insert)).toEqual(expect.arrayContaining(['@hermes', '@writer-homelab']))
    expect(provide('wri').map(item => item.insert)).toEqual(['@writer-homelab'])
  })

  it('filters self by the FOCUSED chat owner, not the gateway socket profile', async () => {
    // The socket stays on `default` while the user is inside another bot's
    // chat; the bot in front of you must not be offered as a handoff target,
    // and a renamed `default` must stay mentionable from there.
    const { provide } = await contributions({
      focused: 'renametest',
      profiles: [{ display_name: 'Lucy', name: 'default' }, { name: 'renametest' }]
    })

    const inserts = provide('').map(item => item.insert)

    expect(inserts).toContain('@lucy')
    expect(inserts).not.toContain('@renametest')
  })

  it('does not warm an authoritative empty roster', async () => {
    const { provide } = await contributions({ profiles: [] })

    expect(provide('')).toEqual([])
    await Promise.resolve()
    await Promise.resolve()
    expect(hostMock.requestProfile).not.toHaveBeenCalled()
  })

  it('warms a cold union roster for the next keystroke', async () => {
    const { provide } = await contributions({ profiles: [] })
    cache.clear()
    hostMock.requestProfile.mockResolvedValue({ profiles: [{ name: 'default' }] })
    hostMock.agents.mockResolvedValue({
      agents: [
        {
          connectionId: 'local',
          connectionKind: 'local',
          connectionLabel: 'This device',
          handle: 'default',
          profile: 'default'
        },
        {
          connectionId: 'cloud',
          connectionKind: 'cloud',
          connectionLabel: 'Cloud',
          handle: 'writer',
          profile: 'writer'
        }
      ],
      primaryConnectionId: 'local',
      sources: []
    })

    expect(provide('')).toEqual([])
    await vi.waitFor(() => expect(provide('').map(item => item.insert)).toContain('@writer'))
  })

  it('keeps switched source warms on their captured routes', async () => {
    const { provide } = await contributions({ profiles: [] })
    const a = deferred<{ bot_mode_protocol: boolean; profiles: Array<{ name: string }> }>()
    const b = deferred<{ bot_mode_protocol: boolean; profiles: Array<{ name: string }> }>()

    cache.clear()
    hostMock.profileRoutes.mockResolvedValue([
      { connectionId: 'a', mode: 'remote', profile: 'default', targetProfile: 'root-a' },
      { connectionId: 'b', mode: 'remote', profile: 'default', targetProfile: 'root-b' }
    ])
    hostMock.requestProfile.mockImplementation(route => (route.connectionId === 'a' ? a.promise : b.promise))

    live.connection = 'a'
    expect(provide('')).toEqual([])
    live.connection = 'b'
    expect(provide('')).toEqual([])
    await vi.waitFor(() => expect(hostMock.requestProfile).toHaveBeenCalledTimes(2))

    const data = await import('./data')
    const protocolBefore = data.serverInjectsProtocol

    b.resolve({ bot_mode_protocol: true, profiles: [{ name: 'from-b' }] })
    await vi.waitFor(() => expect(cache.get(JSON.stringify(['hermes-bots', 'roster', 'b']))).toBeTruthy())
    expect(data.serverInjectsProtocol).toBe(protocolBefore)
    live.connection = 'a'
    a.resolve({ bot_mode_protocol: false, profiles: [{ name: 'from-a' }] })
    await vi.waitFor(() => expect(cache.get(JSON.stringify(['hermes-bots', 'roster', 'a']))).toBeTruthy())

    expect(data.serverInjectsProtocol).toBe(protocolBefore)

    expect(hostMock.requestProfile.mock.calls.map(([route]) => [route.connectionId, route.targetProfile])).toEqual([
      ['a', 'root-a'],
      ['b', 'root-b']
    ])
    expect((cache.get(JSON.stringify(['hermes-bots', 'roster', 'a']))?.value as { profiles: unknown[] }).profiles[0]).toMatchObject({
      name: 'from-a'
    })
    expect((cache.get(JSON.stringify(['hermes-bots', 'roster', 'b']))?.value as { profiles: unknown[] }).profiles[0]).toMatchObject({
      name: 'from-b'
    })
  })

  it('lets a timed-out warm retry without stale publish or owner deletion', async () => {
    const { provide } = await contributions({ profiles: [] })
    const old = deferred<{ profiles: Array<{ name: string }> }>()
    const current = deferred<{ profiles: Array<{ name: string }> }>()

    cache.clear()
    hostMock.requestProfile.mockReturnValueOnce(old.promise).mockReturnValueOnce(current.promise)
    vi.useFakeTimers()

    expect(provide('')).toEqual([])
    await vi.advanceTimersByTimeAsync(15_000)
    expect(provide('')).toEqual([])
    await vi.advanceTimersByTimeAsync(0)
    expect(hostMock.requestProfile).toHaveBeenCalledTimes(2)

    old.resolve({ profiles: [{ name: 'stale' }] })
    await vi.advanceTimersByTimeAsync(0)
    expect(cache.get(JSON.stringify(['hermes-bots', 'roster', 'local']))).toBeUndefined()

    expect(provide('')).toEqual([])
    await vi.advanceTimersByTimeAsync(0)
    expect(hostMock.requestProfile).toHaveBeenCalledTimes(2)

    current.resolve({ profiles: [{ name: 'current' }] })
    await vi.advanceTimersByTimeAsync(0)
    expect(
      (cache.get(JSON.stringify(['hermes-bots', 'roster', 'local']))?.value as { profiles: unknown[] }).profiles[0]
    ).toMatchObject({ name: 'current' })
  })

  it('does not let a delayed warm replace foreground cache or rebuild alias identity', async () => {
    const { provide } = await contributions({ profiles: [] })
    const warm = deferred<{ profiles: Array<{ name: string }> }>()
    const routing = await import('./routing')
    const cacheKey = JSON.stringify(['hermes-bots', 'roster', 'local'])

    cache.clear()
    routing.indexAliasRoutes([
      { connectionId: 'seed', mode: 'remote', profile: 'keep', targetProfile: 'root' }
    ])
    hostMock.profileRoutes.mockResolvedValue([
      { connectionId: 'local', mode: 'local', profile: 'default', targetProfile: 'root' }
    ])
    hostMock.requestProfile.mockReturnValue(warm.promise)

    try {
      expect(provide('')).toEqual([])
      await vi.waitFor(() => expect(hostMock.requestProfile).toHaveBeenCalledTimes(1))

      const authoritative = { fetchedAt: 999, profiles: [{ name: 'foreground' }] }
      cache.set(cacheKey, {
        key: ['hermes-bots', 'roster', 'local'],
        value: authoritative
      })
      warm.resolve({ profiles: [{ name: 'background' }] })

      await vi.waitFor(() => expect(hostMock.agents).toHaveBeenCalled())
      await vi.waitFor(() => expect(cache.get(cacheKey)?.value).toBe(authoritative))
      expect(routing.aliasIdentityFor({ connectionId: 'local', name: 'root', sourceScoped: true })).toBeNull()
      expect(routing.aliasIdentityFor({ connectionId: 'seed', name: 'root', sourceScoped: true })).toMatchObject({
        name: 'keep'
      })
    } finally {
      routing.indexAliasRoutes([])
    }
  })

  it('never offers the bot whose chat you are already in', async () => {
    const { provide } = await contributions()

    // 'default' on the LIVE connection is the active bot; only the Vera twin
    // is a real handoff target.
    expect(provide('').map(item => item.insert)).toEqual(['@default-vera'])
  })

  it('narrows on the typed prefix and labels the source device', async () => {
    const { provide } = await contributions()

    expect(provide('default-v')[0]).toMatchObject({ insert: '@default-vera', meta: expect.stringContaining('Vera') })
    expect(provide('zzz')).toEqual([])
  })
})

describe('the mention middleware', () => {
  it('identifies a remote @name-device mention without delivering anything', async () => {
    const { handler } = await contributions()
    const result = await handler({ text: '@default-vera what is the disk space on the server?' })

    expect(result.text).toMatch(/@mentions resolved from the Bot Mode roster/)
    expect(result.text).toMatch(/@default-vera/)
    // No CLI handoff is composed and the renderer performs NO delivery — the
    // agent owns messaging via its message_agent tool.
    expect(result.text).not.toMatch(/hermes -p '?default/)
    expect(hostMock.requestProfile).not.toHaveBeenCalled()
  })

  it('hands the agent a relay-resolvable canonical target', async () => {
    const { handler } = await contributions()
    const result = await handler({ text: 'ping @default-vera' })

    // The UI alias ('default-vera') is not a relay identity: resolve_remote_target()
    // accepts only a roster row's handle/profile, optionally @connection-qualified.
    // The annotation must carry the canonical profile@connection form (#97678).
    expect(result.text).toMatch(/message_agent target: "default@vera"/)
    expect(result.text).not.toMatch(/message_agent target: "default-vera/)
    expect(result.text).toMatch(/on Vera/)
  })

  it('annotates the resolvable handle for a local row whose UI alias differs', async () => {
    // The reporter's shape (#97678 / Discord video): the LOCAL twin carries
    // the 'default-this-device' alias when the remote gateway is active.
    // The local resolver only knows bare profile names / 'hermes'.
    const { handler } = await contributions({
      focused: 'ops',
      profiles: [
        { connectionId: 'local', connectionKind: 'local', handle: 'default-this-device', name: 'default' },
        { name: 'ops' }
      ]
    })

    const result = await handler({ text: 'ping @default-this-device' })

    expect(result.text).toMatch(/@default-this-device = agent profile "default"/)
    expect(result.text).toMatch(/message_agent target: "hermes"/)
    expect(result.text).not.toMatch(/message_agent target: "default-this-device/)
  })

  it('passes a draft with no mention straight through', async () => {
    const { handler } = await contributions()
    const draft = { text: 'no tags here' }

    expect(await handler(draft)).toBe(draft)
  })

  it('leaves an unresolvable tag alone rather than annotating a guess', async () => {
    const { handler } = await contributions()
    const draft = { text: 'hey @nobody' }

    expect((await handler(draft)).text).toBe('hey @nobody')
  })

  it('names a local bot without inventing a device', async () => {
    const { handler } = await contributions({ focused: 'research', profiles: [{ name: 'research' }, { name: 'ops' }] })
    const result = await handler({ text: 'please @ops review the diff' })

    expect(result.text).toMatch(/@ops = agent profile "ops"/)
    expect(result.text).toMatch(/message_agent/)
    expect(result.text).not.toMatch(/ — on /)
  })

  it('teaches no shellout and forbids forwarding the user’s text verbatim', async () => {
    // The class behind #91397 / #91304 / #91339: the renderer used to compose
    // a `hermes -p …` handoff, giving the model a second send path and a way
    // to relay the raw draft.
    const { handler } = await contributions({ focused: 'research', profiles: [{ name: 'research' }, { name: 'ops' }] })
    const result = await handler({ text: 'ask @ops to summarize' })

    expect(result.text).not.toMatch(/hermes -p/)
    expect(result.text).not.toMatch(/terminal call/i)
    expect(result.text).not.toMatch(/background=true/)
    expect(result.text).toMatch(/compose your own message/i)
    expect(result.text).toMatch(/never forward/i)
  })

  it('keeps a poisoned bot title inert prose', async () => {
    // Nothing here is a command line, so there is nothing to break out of —
    // the invariant that matters is that no hermes command is ever emitted.
    const { handler } = await contributions({
      focused: 'ops',
      profiles: [
        { name: 'ops' },
        { display_name: 'Evil" ; touch /tmp/pwned ; echo "$(touch /tmp/pwned2)"', name: 'research' }
      ]
    })

    const result = await handler({ text: 'ping @research please' })

    expect(result.text).not.toMatch(/`hermes/)
  })

  it('ignores an email address', async () => {
    const { handler } = await contributions({ focused: 'research', profiles: [{ name: 'research' }, { name: 'ops' }] })
    const untouched = 'mail user@example.com and ping @nosuchbot'

    expect((await handler({ text: untouched })).text).toBe(untouched)
  })
})
