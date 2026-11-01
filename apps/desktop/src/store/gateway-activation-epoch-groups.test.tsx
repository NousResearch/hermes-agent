// A real gateway activation can move the activation epoch WITHOUT moving the
// connectionId / profile / gateway atoms (the primary fast path in
// ensureGatewayForProfile). Group-capability consumers key their answer on that
// epoch, so they must re-render and re-probe when it moves; otherwise a healthy
// `groups.capabilities` answer is discarded by its own fence and the UI stays in
// `checking` forever (New Group Chat: two bots ticked, Create disabled, no warning).
// No mock of the epoch: the real beginGatewayActivation runs.
import type * as HermesSdk from '@hermes/plugin-sdk'
import { host } from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { groupCreationSource, useGroupCapability } from '@/plugins/hermes-bots/canonical-group-capabilities'
import { CreateGroupChatDialog } from '@/plugins/hermes-bots/create-dialog'
import { GroupChatWorkspace } from '@/plugins/hermes-bots/group-chat-view'
import { ensureGatewayForProfile, gatewayActivationEpoch } from '@/store/gateway'

const { request } = vi.hoisted(() => ({ request: vi.fn() }))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()
  const { en } = await import('@/i18n/en')
  const { translateBots } = await import('@/plugins/hermes-bots/i18n-test-helper')

  return {
    ...sdk,
    host: {
      ...sdk.host,
      requestProfile: request,
      connections: vi.fn(async () => []),
      state: {
        ...sdk.host.state,
        connectionId: sdk.atom('remote-1'),
        profile: sdk.atom('default'),
        gateway: sdk.atom('open')
      }
    },
    useI18n: () => ({ locale: 'en', t: en }),
    usePluginI18n: () => translateBots
  }
})
vi.mock('@/plugins/hermes-bots/group-chat-parts', async importOriginal => ({
  ...(await importOriginal<any>()),
  GroupImageControls: () => null
}))

const FEATURES = [
  'authority_epoch',
  'coordinator_fencing',
  'room_identity',
  'monotonic_log',
  'idempotent_send',
  'replayable_disband',
  'typed_events',
  'actor_identity',
  'log_replication',
  'authority_takeover'
]

const METHODS = ['capabilities', 'list', 'create', 'state', 'send', 'stop', 'retry', 'approve'].map(m => `groups.${m}`)

const payload = { driver: true, persistent_process: true, protocol_version: 2, features: FEATURES, methods: METHODS }

const roster = [
  { name: 'alpha', connectionId: 'remote-1' },
  { name: 'beta', connectionId: 'remote-1' }
]

const route = { connectionId: 'remote-1', profile: 'default' }
const state = host.state as any
const capCalls = () => request.mock.calls.filter(call => call[1] === 'groups.capabilities').length
const createButton = () => screen.getByRole('button', { name: 'Create Group (2)' }) as HTMLButtonElement

let latest: any

function Hook() {
  latest = useGroupCapability(true)

  return <output>{latest.mode}</output>
}

function deferred() {
  let resolve!: (value: unknown) => void
  const promise = new Promise(r => (resolve = r))

  return { promise, resolve }
}

// The primary fast path bumps the epoch and changes no atom the hooks subscribe to.
const activate = () =>
  act(async () => {
    await ensureGatewayForProfile('default')
  })

const sleep = (ms: number) =>
  act(async () => {
    await new Promise(r => setTimeout(r, ms))
  })

function tickAll() {
  for (const checkbox of screen.getAllByRole('checkbox')) {
    fireEvent.click(checkbox)
  }
}

beforeEach(() => {
  state.connectionId.set('remote-1')
  state.profile.set('default')
  state.gateway.set('open')
  request.mockReset()
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

afterEach(() => {
  cleanup()
  localStorage.clear()
})

it('a primary-route activation moves only the epoch, not the atoms the hooks watch', async () => {
  const epoch = gatewayActivationEpoch()
  const before = [state.connectionId.get(), state.profile.get(), state.gateway.get()]

  await activate()

  expect(gatewayActivationEpoch()).toBe(epoch + 1)
  expect([state.connectionId.get(), state.profile.get(), state.gateway.get()]).toEqual(before)
})

it('an activation while the capability probe is in flight does not strand the hook in checking', async () => {
  const first = deferred()
  request.mockReturnValueOnce(first.promise).mockResolvedValue(payload)
  render(<Hook />)

  await activate()
  await act(async () => {
    first.resolve(payload)
    await first.promise
  })

  await waitFor(() => expect(latest.mode).toBe('canonical'), { timeout: 2000 })
  expect(capCalls()).toBe(2)
})

it('after an activation the hook re-probes once and keeps the stale-answer fence intact', async () => {
  request.mockResolvedValue(payload)
  render(<Hook />)
  await waitFor(() => expect(latest.mode).toBe('canonical'))
  expect(capCalls()).toBe(1)
  const firstEpoch = latest.capability.activationEpoch

  await activate()

  await waitFor(() => expect(capCalls()).toBe(2), { timeout: 2000 })
  await waitFor(() => expect(latest.mode).toBe('canonical'))
  expect(groupCreationSource(route, latest.capability.activationEpoch)()).toBe(true)
  expect(groupCreationSource(route, firstEpoch)()).toBe(false)
  await sleep(80)
  expect(capCalls()).toBe(2)
})

it('New Group Chat enables Create after an activation lands mid-probe', async () => {
  const pending = deferred()
  request.mockReturnValueOnce(pending.promise).mockResolvedValue(payload)
  render(<CreateGroupChatDialog onClose={vi.fn()} open roster={roster} />)
  tickAll()

  await activate()
  await act(async () => {
    pending.resolve(payload)
    await pending.promise
  })

  await waitFor(() => expect(createButton().disabled).toBe(false), { timeout: 2000 })
})

it('New Group Chat stays creatable after an activation that follows a healthy answer', async () => {
  request.mockResolvedValue(payload)
  render(<CreateGroupChatDialog onClose={vi.fn()} open roster={roster} />)
  tickAll()
  await waitFor(() => expect(createButton().disabled).toBe(false))

  await activate()

  await waitFor(() => expect(capCalls()).toBe(2), { timeout: 2000 })
  await waitFor(() => expect(createButton().disabled).toBe(false))
})

// The live repro: a relaunch (or Webapp reload) restores the Bots pane, so the
// roster mounts the closed dialog before the boot activation bumps the epoch.
it('New Group Chat mounted closed before the boot activation enables Create once opened', async () => {
  request.mockResolvedValue(payload)
  const { rerender } = render(<CreateGroupChatDialog onClose={vi.fn()} open={false} roster={roster} />)

  await activate()
  rerender(<CreateGroupChatDialog onClose={vi.fn()} open roster={roster} />)
  tickAll()

  await waitFor(() => expect(createButton().disabled).toBe(false), { timeout: 2000 })
})

it('the existing-room execution gate re-probes after an activation, then settles', async () => {
  request.mockImplementation(async (_route: unknown, method: string) => {
    if (method === 'groups.capabilities') {
      return payload
    }

    return method === 'groups.list' ? { rooms: [], next_offset: null } : {}
  })
  render(<GroupChatWorkspace group="Existing" members={roster as any} />)
  await sleep(200)
  const before = capCalls()

  await activate()
  await sleep(300)

  const after = capCalls()
  expect(after).toBeGreaterThan(before)
  expect(after - before).toBeLessThanOrEqual(2)
  await sleep(300)
  expect(capCalls()).toBe(after)
})
