/**
 * The "Open Screen" palette row is the redesign's door to a bot's Screen: its Agents page hosts
 * Bots, so the sidebar has no Screen portal there. The classic shell hosts the Bots pane and the
 * sidebar's Screen portal, and its palette must stay main's — no Screen row.
 */

import { afterEach, beforeAll, expect, it, vi } from 'vitest'

interface Contribution {
  area: string
  id: string
  data?: { detail?: () => string }
}

interface Owner {
  connectionId: string
  profile: string
}

const { focusOwner, hostMock } = vi.hoisted(() => {
  let owner: Owner = { connectionId: 'local', profile: 'default' }
  const ownerListeners = new Set<(next: Owner) => void>()
  const idle = { get: () => null, listen: () => () => undefined }

  return {
    // The focused chat's owner, as the shell publishes it: a fresh object on every focus change.
    focusOwner: (next: Owner) => {
      owner = next
      ownerListeners.forEach(listener => listener(next))
    },
    hostMock: {
      notify: vi.fn(),
      request: vi.fn(async () => ({})),
      requestProfile: vi.fn(async () => ({})),
      agents: vi.fn(async () => ({ agents: [], sources: [] })),
      state: {
        connectionId: { get: () => 'local', listen: () => () => undefined },
        focusedSessionOwner: {
          get: () => owner,
          listen: (listener: (next: Owner) => void) => {
            ownerListeners.add(listener)

            return () => ownerListeners.delete(listener)
          }
        },
        focusedSessionProfile: { get: () => 'default', listen: () => () => undefined },
        focusedStoredSessionId: idle,
        gateway: idle,
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

  const known: Record<string, unknown> = { atom, host: hostMock, PALETTE_AREA: 'palette' }

  return new Proxy(known, {
    get: (target, key) =>
      typeof key === 'symbol' || key in target ? target[key as string] : key === 'then' ? undefined : stub,
    has: () => true
  })
})

// Outside the pane the roster has its own publisher; this test has no gateway to feed it.
vi.mock('./roster-publisher', () => ({ startRosterPublisher: () => () => undefined }))

const live: Contribution[] = []
const disposers: Array<() => void> = []

async function registerPlugin() {
  const plugin = (await import('./plugin')).default

  try {
    plugin.register({
      i18n: { register: () => () => undefined },
      onDispose: (dispose: () => void) => disposers.push(dispose),
      register: (contribution: Contribution) => {
        live.push(contribution)

        return () => live.splice(live.indexOf(contribution), 1)
      },
      storage: { get: async () => undefined, remove: async () => undefined, set: async () => undefined }
    } as never)
  } catch {
    // Registration walks UI surfaces the stub does not model; the Screen row registers before.
  }
}

const screenRow = () => live.find(entry => entry.area === 'palette' && entry.id === 'open-screen')

beforeAll(async () => {
  await import('./plugin')
}, 120_000)

afterEach(async () => {
  disposers.splice(0).forEach(dispose => dispose())
  live.splice(0)
  const { $botsPaneHosted } = await import('./bots-pane-host')
  $botsPaneHosted.set(true)
  const { $lastRoster } = await import('./data')
  $lastRoster.set([])
  focusOwner({ connectionId: 'local', profile: 'default' })
})

it('offers no Screen row in the classic shell, which hosts the Bots pane and its Screen portal', async () => {
  await registerPlugin()
  expect(screenRow()).toBeUndefined()
})

it('offers the Screen row where the Agents page hosts Bots, and follows a live shell flip', async () => {
  const { $botsPaneHosted } = await import('./bots-pane-host')
  $botsPaneHosted.set(false)
  await registerPlugin()
  expect(screenRow()).toBeDefined()

  $botsPaneHosted.set(true)
  expect(screenRow()).toBeUndefined()

  $botsPaneHosted.set(false)
  expect(screenRow()).toBeDefined()
})

it('keeps the Screen row while focus moves within one owner, replaces it for another, and names the machine', async () => {
  const { $botsPaneHosted } = await import('./bots-pane-host')
  const { $lastRoster } = await import('./data')
  $botsPaneHosted.set(false)
  await registerPlugin()
  const first = screenRow()

  focusOwner({ connectionId: 'local', profile: 'default' })
  expect(screenRow()).toBe(first)

  $lastRoster.set([
    { name: 'ops', sourceScoped: true, connectionId: 'studio', connectionKind: 'remote', connectionLabel: 'Studio Mac' }
  ])
  focusOwner({ connectionId: 'studio', profile: 'ops' })
  expect(screenRow()).not.toBe(first)
  expect(screenRow()?.data?.detail?.()).toBe('ops · Studio Mac')
})
