/**
 * Bot Mode MCP setup must act on the selected bot's captured connection and
 * backend profile — never on whatever gateway happens to be in the foreground.
 *
 * The defect class (#Phase 2): `mcpRpc` rode `host.request` (the ambient
 * socket), forwarded a `{connectionId, profile}` OBJECT verbatim as the
 * `profile` wire param, and cached one process-wide support probe, so editing
 * bot A while chatting on B wrote/tested the wrong backend, an older backend's
 * probe hid setup on every other, and a New Bot created remotely lost its
 * connection before credential setup.
 *
 * Pinned here through the real component:
 *   1. probe, catalog add, key write, and test all ride ONE captured
 *      (connection, profile) target — A→B→A keeps each leg on its own
 *      connection with the backend NAME (alias target) on the wire;
 *   2. replacing the target (or unmounting) retires the pending setup —
 *      no stale continuation, no stale onDone;
 *   3. a fulfilled `ok: false` envelope is a failed configuration, transient
 *      probe errors stay retryable, and only a confirmed unknown method hides
 *      setup behind the needs hint.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { translateBots } from './i18n-test-helper'
import { McpSetupButton, type McpSetupTarget } from './mcp-setup'
import type { ProfileRoute } from './types'

const { hostMock } = vi.hoisted(() => ({
  hostMock: {
    completeMcpOAuth: vi.fn(),
    notify: vi.fn(),
    request: vi.fn(),
    requestProfile: vi.fn(),
    state: { connectionId: { get: () => 'local' }, profile: { get: () => 'default' } }
  }
}))

vi.mock('@hermes/plugin-sdk', async () => {
  const { useI18n } = await vi.importActual<typeof HermesSdk>('@hermes/plugin-sdk')

  return {
    Button: (props: React.ComponentProps<'button'>) => <button {...props} />,
    host: hostMock,
    Input: (props: React.ComponentProps<'input'>) => <input {...props} />,
    useI18n,
    usePluginI18n: () => translateBots
  }
})

vi.mock('./shared', () => ({ ID: 'hermes-bots' }))

/** Bot A on connection `conn-a` behind the friendly alias `friendly-bot`; its
 *  backend profile — the one the wire must carry — is `default`. */
const ROUTE_A: ProfileRoute = {
  connectionId: 'conn-a',
  mode: 'remote',
  profile: 'friendly-bot',
  targetProfile: 'default'
}

const TARGET_A: McpSetupTarget = { route: ROUTE_A, profile: 'default' }

/** Bot `work` on connection `conn-b` — same-named profiles elsewhere must not
 *  share its writes. */
const ROUTE_B: ProfileRoute = {
  connectionId: 'conn-b',
  mode: 'remote',
  profile: 'work',
  targetProfile: 'work'
}

const TARGET_B: McpSetupTarget = { route: ROUTE_B, profile: 'work' }

const ENTRY = {
  fromCatalog: true,
  installed: false,
  name: 'github',
  requires: ['GITHUB_TOKEN']
}

const setUpLabel = () => screen.getByText(translateBots('tools.setUp'))
const saveLabel = () => screen.getByText(translateBots('tools.saveTest'))

/** Every RPC the component dispatched, reduced to its routing identity. */
const dispatched = () =>
  hostMock.requestProfile.mock.calls.map(([route, method, params]) => ({
    connectionId: (route as ProfileRoute).connectionId,
    method: method as string,
    profile: (params as { profile?: unknown }).profile,
    route
  }))

beforeEach(() => {
  vi.clearAllMocks()
  hostMock.requestProfile.mockImplementation(async (_route, method) => {
    switch (method) {
      case 'mcp.servers.list':
        return { servers: [] }

      case 'mcp.servers.add':
        return { ok: true, name: 'github' }

      case 'mcp.servers.set_api_key':
        return { ok: true, name: 'github', env_var: 'GITHUB_TOKEN' }

      case 'mcp.servers.test':
        return { ok: true, tools: [] }

      default:
        return {}
    }
  })
})

/** Run one full setup leg: add from catalog, enter the key, save & test. */
async function runSetupLeg() {
  fireEvent.click(setUpLabel())
  await waitFor(() => expect(saveLabel()).toBeTruthy())
  fireEvent.change(screen.getByPlaceholderText('GITHUB_TOKEN'), { target: { value: 'sk-live-1' } })
  fireEvent.click(saveLabel())
  await waitFor(() => expect(hostMock.notify).toHaveBeenCalled())
}

describe('one captured connection/profile target for probe, add, key write, and test', () => {
  it('A→B→A keeps every leg on its own connection with the backend name on the wire', async () => {
    for (const [target, expected] of [
      [TARGET_A, { connectionId: 'conn-a', profile: 'default' }],
      [TARGET_B, { connectionId: 'conn-b', profile: 'work' }],
      [TARGET_A, { connectionId: 'conn-a', profile: 'default' }]
    ] as const) {
      hostMock.requestProfile.mockClear()
      hostMock.request.mockClear()
      hostMock.notify.mockClear()

      const leg = render(<McpSetupButton entry={ENTRY} onDone={() => {}} target={target} />)

      // The support probe rides the same target as the writes.
      await waitFor(() =>
        expect(dispatched().some(call => call.method === 'mcp.servers.list')).toBe(true)
      )
      await runSetupLeg()

      const calls = dispatched()

      expect(calls.map(call => call.method)).toEqual([
        'mcp.servers.list',
        'mcp.servers.add',
        'mcp.servers.set_api_key',
        'mcp.servers.test'
      ])

      for (const call of calls) {
        expect(call.connectionId).toBe(expected.connectionId)
        expect(call.profile).toBe(expected.profile)
        expect(typeof call.profile).toBe('string')
      }

      // The ambient socket is never borrowed for a source-scoped bot.
      expect(hostMock.request).not.toHaveBeenCalled()

      screen.getByText(translateBots('tools.setUpDone'))
      leg.unmount()
    }
  })

  it('New Bot creation carries the selected connection and the created slug', async () => {
    // The create dialog targets the picked connection's default backend door
    // before the profile exists; ensureTarget materializes the slug and hands
    // back the complete setup target.
    const door: ProfileRoute = { connectionId: 'conn-b', mode: 'remote', profile: 'default', targetProfile: 'default' }

    const ensureTarget = vi.fn(
      async (): Promise<McpSetupTarget> => ({ route: door, profile: 'created-slug' })
    )

    render(
      <McpSetupButton
        ensureTarget={ensureTarget}
        entry={ENTRY}
        onDone={() => {}}
        target={{ route: door, profile: 'planned-slug' }}
      />
    )

    await waitFor(() =>
      expect(dispatched().some(call => call.method === 'mcp.servers.list')).toBe(true)
    )
    await runSetupLeg()

    expect(ensureTarget).toHaveBeenCalledOnce()

    const writes = dispatched().filter(call => call.method !== 'mcp.servers.list')

    expect(writes.map(call => call.method)).toEqual([
      'mcp.servers.add',
      'mcp.servers.set_api_key',
      'mcp.servers.test'
    ])

    for (const call of writes) {
      expect(call.connectionId).toBe('conn-b')
      expect(call.profile).toBe('created-slug')
      expect(typeof call.profile).toBe('string')
    }

    expect(hostMock.request).not.toHaveBeenCalled()
  })

  it('an orphaned source-scoped bot reports an unavailable target instead of borrowing the ambient gateway', async () => {
    render(<McpSetupButton entry={ENTRY} onDone={() => {}} target={null} />)

    fireEvent.click(setUpLabel())

    await waitFor(() => expect(screen.getByText(/No target profile/)).toBeTruthy())
    expect(hostMock.request).not.toHaveBeenCalled()
    expect(hostMock.requestProfile).not.toHaveBeenCalled()
  })
})

describe('async target replacement and cancellation', () => {
  function deferred<T>() {
    let resolve!: (value: T) => void

    const promise = new Promise<T>(settle => {
      resolve = settle
    })

    return { promise, resolve }
  }

  it('retiring a replaced target suppresses the stale continuation and onDone', async () => {
    const pendingAdd = deferred<{ ok: boolean; name: string }>()

    hostMock.requestProfile.mockImplementation(async (_route, method) => {
      if (method === 'mcp.servers.list') {
        return { servers: [] }
      }

      if (method === 'mcp.servers.add') {
        return pendingAdd.promise
      }

      return { ok: true }
    })
    const onDone = vi.fn()

    const { rerender } = render(<McpSetupButton entry={ENTRY} onDone={onDone} target={TARGET_A} />)

    await waitFor(() =>
      expect(dispatched().some(call => call.method === 'mcp.servers.list')).toBe(true)
    )
    fireEvent.click(setUpLabel())

    await waitFor(() => expect(dispatched().some(call => call.method === 'mcp.servers.add')).toBe(true))

    // The editor now points at another bot: the pending setup retires.
    rerender(<McpSetupButton entry={ENTRY} onDone={onDone} target={TARGET_B} />)

    pendingAdd.resolve({ ok: true, name: 'github' })
    await waitFor(() => expect(screen.queryByText(translateBots('tools.saveTest'))).toBeNull())

    // The retired flow continued nothing and completed nothing.
    expect(dispatched().some(call => call.method === 'mcp.servers.set_api_key')).toBe(false)
    expect(dispatched().some(call => call.method === 'mcp.servers.test')).toBe(false)
    expect(onDone).not.toHaveBeenCalled()
  })

  it('unmounting mid-flow suppresses stale success without touching the new owner', async () => {
    const pendingAdd = deferred<{ ok: boolean; name: string }>()

    hostMock.requestProfile.mockImplementation(async (_route, method) => {
      if (method === 'mcp.servers.add') {
        return pendingAdd.promise
      }

      return { ok: true }
    })
    const onDone = vi.fn()

    const { unmount } = render(<McpSetupButton entry={ENTRY} onDone={onDone} target={TARGET_A} />)

    fireEvent.click(setUpLabel())
    await waitFor(() => expect(dispatched().some(call => call.method === 'mcp.servers.add')).toBe(true))

    unmount()
    pendingAdd.resolve({ ok: true, name: 'github' })
    await new Promise(settle => setTimeout(settle, 0))

    expect(onDone).not.toHaveBeenCalled()
    expect(dispatched().some(call => call.method === 'mcp.servers.set_api_key')).toBe(false)
  })
})

describe('failure envelopes and probe classification', () => {
  it.each([
    {
      failingMethod: 'mcp.servers.add',
      failure: { ok: false, error: 'rejected by backend' },
      expectTest: false,
      name: 'a fulfilled ok:false add is not successful configuration',
      wireError: /rejected by backend/
    },
    {
      failingMethod: 'mcp.servers.test',
      failure: { ok: false, error: 'boom' },
      expectTest: true,
      name: 'a failed test surfaces the server error and never completes',
      wireError: /boom/
    }
  ])('$name', async ({ failingMethod, failure, expectTest, wireError }) => {
    hostMock.requestProfile.mockImplementation(async (_route, method) => {
      if (method === failingMethod) {
        return failure
      }

      return method === 'mcp.servers.list' ? { servers: [] } : { ok: true, name: 'github' }
    })
    const onDone = vi.fn()

    render(<McpSetupButton entry={ENTRY} onDone={onDone} target={TARGET_A} />)

    await waitFor(() =>
      expect(dispatched().some(call => call.method === 'mcp.servers.list')).toBe(true)
    )
    fireEvent.click(setUpLabel())

    if (expectTest) {
      await waitFor(() => expect(saveLabel()).toBeTruthy())
      fireEvent.change(screen.getByPlaceholderText('GITHUB_TOKEN'), { target: { value: 'sk-1' } })
      fireEvent.click(saveLabel())
    }

    await waitFor(() => expect(screen.getByText(wireError)).toBeTruthy())
    expect(onDone).not.toHaveBeenCalled()
  })

  it('reports partial completion when a later key write fails, without rolling back', async () => {
    hostMock.requestProfile.mockImplementation(async (_route, method) => {
      if (method === 'mcp.servers.list') {
        return { servers: [] }
      }

      if (method === 'mcp.servers.set_api_key') {
        const call = hostMock.requestProfile.mock.calls.filter(([, m]) => m === 'mcp.servers.set_api_key')

        return call.length > 1 ? { ok: false, error: 'denied' } : { ok: true, name: 'github' }
      }

      return { ok: true, name: 'github' }
    })

    render(
      <McpSetupButton entry={{ ...ENTRY, requires: ['K1', 'K2'] }} onDone={() => {}} target={TARGET_A} />
    )

    await waitFor(() =>
      expect(dispatched().some(call => call.method === 'mcp.servers.list')).toBe(true)
    )
    fireEvent.click(setUpLabel())
    await waitFor(() => expect(screen.getByPlaceholderText('K1')).toBeTruthy())
    fireEvent.change(screen.getByPlaceholderText('K1'), { target: { value: 'first' } })
    fireEvent.change(screen.getByPlaceholderText('K2'), { target: { value: 'second' } })
    fireEvent.click(saveLabel())

    await waitFor(() => expect(screen.getByText(/denied/)).toBeTruthy())
    // Partial: one key was accepted before the failure; the test never runs.
    expect(screen.getByText(/1/)).toBeTruthy()
    expect(dispatched().some(call => call.method === 'mcp.servers.test')).toBe(false)
  })

  it('only a confirmed unknown method hides setup behind the needs hint', async () => {
    hostMock.requestProfile.mockRejectedValue(new Error('unknown method: mcp.servers.list'))

    const { unmount } = render(<McpSetupButton entry={ENTRY} onDone={() => {}} target={TARGET_A} />)

    await waitFor(() => expect(screen.getByText(translateBots('tools.needsSetup', 'GITHUB_TOKEN'))).toBeTruthy())
    expect(screen.queryByText(translateBots('tools.setUp'))).toBeNull()
    unmount()

    // A transient failure is NOT permanent unavailability: the button stays
    // and the actions remain retryable.
    hostMock.requestProfile.mockClear()
    hostMock.requestProfile.mockRejectedValue(new Error('gateway unavailable'))

    render(<McpSetupButton entry={ENTRY} onDone={() => {}} target={TARGET_A} />)

    await waitFor(() => expect(screen.getByText(translateBots('tools.setUp'))).toBeTruthy())
  })
})
