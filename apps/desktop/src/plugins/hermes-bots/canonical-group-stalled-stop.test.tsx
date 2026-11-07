import { useStore } from '@nanostores/react'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { atom } from 'nanostores'
import type { ComponentProps } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

const request = vi.hoisted(() => vi.fn())
vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock, createGroupGateway } = await import('./group-test-utils')
  const gateway = createGroupGateway()
  const { en } = await import('@/i18n/en')
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')
  const { ConfirmDialog } = await import('@/components/ui/confirm-dialog')
  const { captureGroupRequests } = await import('./canonical-group-contract-test-utils')

  return { ...await pluginSdkMock(gateway.host), atom, useValue: useStore, ConfirmDialog,
    useQuery: (await import('@tanstack/react-query')).useQuery,
    useI18n: () => ({ t: en }),
    usePluginI18n: () => (key: string) => CANONICAL_GROUP_LOCALES.en[key.replace('canonical.', '') as keyof typeof CANONICAL_GROUP_LOCALES.en] ?? key,
    Button: (p: ComponentProps<'button'>) => <button {...p} />,
    Input: (p: ComponentProps<'input'>) => <input {...p} />,
    RowButton: (p: ComponentProps<'button'>) => <button type="button" {...p} />,
    host: { ...gateway.host, requestProfile: captureGroupRequests(request).request } }
})
import { CANONICAL_GROUP_CAPABILITIES } from './canonical-group-contract-test-utils'
import { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'
import { CanonicalGroupWorkspace } from './canonical-group-workspace'
// Captured via AuthorityConnection from HostedRoomService; see PROOF.md for regeneration.
import wire from './fixtures/stalled-stop-responses.json'

const labels = CANONICAL_GROUP_LOCALES.en
const originalDesktop = window.hermesDesktop
beforeEach(() => { Object.defineProperty(window, 'hermesDesktop', { configurable: true, writable: true, value: undefined }) })
afterEach(() => { cleanup(); request.mockReset(); localStorage.clear(); window.hermesDesktop = originalDesktop })

const snapshots = { stalled: wire.stalled, deferred: wire.deferred, indeterminate: wire.indeterminate,
  working: wire.working, settled: wire.settled, empty: wire.empty }

interface Snapshot {
  state: { room: { room_id: string; name: string } }
  log: { events: { seq: number }[] }
}

async function open(kind: keyof typeof snapshots, stop: () => Promise<unknown> = async () => wire.stop, supported = true) {
  let current: Snapshot = snapshots[kind]
  request.mockImplementation(async (_route, method, params) => {
    if (method === 'groups.capabilities') {return { ...CANONICAL_GROUP_CAPABILITIES,
      methods: CANONICAL_GROUP_CAPABILITIES.methods.filter(method => supported || method !== 'groups.stop') } }

    if (method === 'groups.state') {return current.state}

    if (method === 'groups.log') {return { ...current.log, events: current.log.events.filter(event => event.seq > params.since_seq) }}

    if (method === 'groups.stop') {return stop()}

    if (method === 'groups.retry') {throw Object.assign(new Error('invalid_params'), { code: 4001, data: { reason: 'invalid_params' } })}

    if (method === 'groups.send') {current = wire.working;

 return { ...wire.send, client_event_id: params.event_id }}

    throw new Error(`Unexpected RPC: ${method}`)
  })
  const view = render(<CanonicalGroupWorkspace binding={{ connectionId: 'local', profile: 'default', roomId: current.state.room.room_id }} />)
  await screen.findByRole('button', { name: 'Group details' })
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false))

  return { view, update: (next: Snapshot) => { current = next } }
}

const stops = () => request.mock.calls.filter(call => call[1] === 'groups.stop')

it.each(['stalled', 'deferred', 'indeterminate'] as const)('confirms Stop for idle %s work and keeps unconfirmed work visible', async kind => {
  let release!: (result: unknown) => void
  const { update } = await open(kind, () => new Promise(resolve => { release = resolve }))
  const stop = screen.getByRole('button', { name: labels.stopGroup })
  fireEvent.click(stop)
  let dialog = await screen.findByRole('dialog')
  expect(within(dialog).getByText(labels.stopWaitingDescription)).toBeTruthy()
  expect(stops()).toHaveLength(0)
  fireEvent.click(within(dialog).getByRole('button', { name: 'Cancel' }))
  expect(screen.queryByRole('dialog')).toBeNull()
  expect(stops()).toHaveLength(0)
  fireEvent.click(stop)
  dialog = await screen.findByRole('dialog')
  const confirm = within(dialog).getByRole('button', { name: labels.stopGroup })
  fireEvent.click(confirm)
  await waitFor(() => expect(stops()).toHaveLength(1))
  expect((stop as HTMLButtonElement).disabled).toBe(true)
  expect((confirm as HTMLButtonElement).disabled).toBe(true)
  fireEvent.click(confirm)
  fireEvent.click(stop)
  expect(stops()).toHaveLength(1)
  const after = { stalled: wire.afterStop, deferred: wire.afterDeferredStop, indeterminate: wire.afterIndeterminateStop }[kind]
  update(after)
  await act(async () => { release({ stalled: wire.stop, deferred: wire.deferredStop, indeterminate: wire.indeterminateStop }[kind]) })
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())

  // The Stop notice stays while a kept turn waits for Retry or Discard, and goes once nothing waits.
  if (kind === 'stalled') {
    await waitFor(() => expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull())
    expect(screen.queryByText(labels.stopped)).toBeNull()
    expect(screen.queryByRole('button', { name: labels.stopGroup })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull()
    fireEvent.change(screen.getByRole('textbox'), { target: { value: '@ops next question' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send' }))
    await waitFor(() => expect(request.mock.calls.some(call => call[1] === 'groups.send')).toBe(true))
    await screen.findByRole('button', { name: labels.stopGroup })
  } else {
    await screen.findByText(labels.stopped)
    expect(screen.getByRole('button', { name: labels.stopGroup })).toBeTruthy()
    expect(screen.queryByText(labels.cycleStopped)).toBeNull()

    if (kind === 'deferred') {
      expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
      expect(screen.getByRole('button', { name: 'Discard' })).toBeTruthy()
    }
  }
})

it('keeps a failed confirmation open and replays its Stop before fencing newer waiting work', async () => {
  let attempts = 0

  const { update } = await open('stalled', async () => {
    if (++attempts === 1) {throw new Error('Stop timed out')}

    if (attempts === 3) {update(wire.afterStop)}

    return wire.stop
  })

  fireEvent.click(screen.getByRole('button', { name: labels.stopGroup }))
  const dialog = await screen.findByRole('dialog')
  fireEvent.click(within(dialog).getByRole('button', { name: labels.stopGroup }))
  await within(dialog).findByText('Stop timed out')
  expect(screen.queryByText(labels.stopped)).toBeNull()
  fireEvent.click(within(dialog).getByRole('button', { name: labels.stopGroup }))
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  const ids = stops().map(call => call[2].cancel_id)
  expect(ids).toHaveLength(3)
  expect(ids[1]).toBe(ids[0])
  expect(ids[2]).not.toBe(ids[0])
})

it('says in plain words when Retry cannot repair the stall', async () => {
  await open('stalled')
  fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
  expect(await screen.findByText(labels.retryStalled)).toBeTruthy()
  expect(screen.queryByText('invalid_params')).toBeNull()
})

it('stops visibly working bots in one press', async () => {
  const { update } = await open('working', async () => { update(wire.settled);

 return wire.workingStop })

  fireEvent.click(screen.getByRole('button', { name: labels.stopGroup }))
  await waitFor(() => expect(screen.queryByRole('button', { name: labels.stopGroup })).toBeNull())
  expect(stops()).toHaveLength(1)
  expect(screen.queryByRole('dialog')).toBeNull()
})

it.each([
  { kind: 'settled', supported: true }, { kind: 'empty', supported: true }, { kind: 'stalled', supported: false }
] as const)('offers no Stop for $kind work with support=$supported', async ({ kind, supported }) => {
  await open(kind, undefined, supported)
  expect(screen.queryByRole('button', { name: labels.stopGroup })).toBeNull()
})
