import type * as HermesSdk from '@hermes/plugin-sdk'
import { useStore } from '@nanostores/react'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

const { request, notify, connections, osNotify, activation } = vi.hoisted(() => ({
  request: vi.fn(), notify: vi.fn(), connections: vi.fn(), osNotify: vi.fn(), activation: { epoch: 1 }
}))

vi.mock('electron', () => ({ app: {}, ipcMain: {} }))
vi.mock('@hermes/plugin-sdk', async () => {
  const sdk = await vi.importActual<typeof HermesSdk>('@hermes/plugin-sdk')
  const { pluginSdkMock, createGroupGateway, captureGroupRequests } = await import('./group-test-utils')
  const gateway = createGroupGateway()
  const { en } = await import('@/i18n/en')

  return { ...sdk, ...await pluginSdkMock(gateway.host), atom, useValue: useStore, MessageTextContent: sdk.MessageTextContent,
    gatewayActivationEpoch: () => activation.epoch,
    useI18n: () => ({ locale: 'en', t: en }),
    usePluginI18n: () => translateBots,
    host: { ...gateway.host, requestProfile: captureGroupRequests(request).request, connections, notify } }
})
vi.mock('./shared', async importOriginal => ({ ...await importOriginal<object>(), getPluginCtx: () => ({ os: { notify: osNotify } }) }))

import { attemptCanonicalGroupSend, prepareCanonicalGroupSend, rehomeCanonicalGroupSends, settleCanonicalGroupSend } from './canonical-group-send'
import { backup, binding, computer, type Handler, LAPTOP, MINI, offlineStatus, registry, roomState, status, unreachable, VPS }
  from './canonical-group-succession-fixtures'
import { CanonicalGroupWorkspace } from './canonical-group-workspace'
import { translateBots } from './i18n-test-helper'

let handlers: Record<string, Handler> = {}
const calls = (method: string) => request.mock.calls.filter(call => call[1] === method)
const system = { kind: 'system', id: 'room-driver' }
const today = (hours: number) => Math.floor(new Date().setHours(hours, 30, 0, 0) / 1000)
const timeOf = (seconds: number) => new Intl.DateTimeFormat('en', { hour: 'numeric', minute: '2-digit' }).format(new Date(seconds * 1000))

beforeEach(() => {
  activation.epoch++
  handlers = {}
  connections.mockResolvedValue(registry)
  request.mockImplementation(async (route: { connectionId: string }, method: string, params: Record<string, unknown>) => {
    const handler = handlers[route.connectionId]

    if (!handler) {throw new Error(`No connection to ${route.connectionId}`)}

    return handler(method, params ?? {})
  })
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, writable: true, value: undefined })
})
afterEach(() => {cleanup(); request.mockReset(); notify.mockReset(); connections.mockReset(); osNotify.mockReset(); localStorage.clear(); vi.restoreAllMocks()})

function host(methods: Record<string, Handler>, events: unknown[] = []) {
  handlers['mac-mini'] = computer(MINI, { 'groups.state': () => roomState(), 'groups.log': () => ({ events }), ...methods })
}

it('says how ready the group is to move by itself, with the owner’s switch and computer sublabels', async () => {
  let current = status({
    automatic: { state: 'ready', mode: 'careful', standby: { install_id: VPS, name: 'Home VPS' }, voters: 2 },
    backups: [backup(VPS, 'Home VPS', { always_on: true, voter: true }), backup(LAPTOP, 'Laptop', { always_on: false })],
    actions: [{ action: 'automatic', enabled: true }, { action: 'designate', targets: [VPS, LAPTOP] }]
  })

  host({ 'groups.succession.status': () => current,
    'groups.custody.automatic': (_method, params) => ({ room_id: params.room_id, automatic: params.enabled, configuration_seq: 9 }) })
  render(<CanonicalGroupWorkspace binding={binding} />)
  fireEvent.click(await screen.findByRole('button', { name: 'Hosted on Mac mini' }))
  const section = within(await screen.findByRole('region', { name: 'If a computer goes offline' }))
  expect(section.getByText('Keeps running on its own. If Mac mini goes offline, Home VPS takes over after about 3 minutes.')).toBeTruthy()
  expect(section.getByText('With a third always-on computer, takeover is faster and the group can never end up running in two places.')).toBeTruthy()
  expect(section.getByText('When off, you’re asked first, with one tap.')).toBeTruthy()
  expect(screen.getByText('Can take over automatically')).toBeTruthy()
  expect(screen.getByText('Can continue when you choose')).toBeTruthy()

  fireEvent.click(section.getByRole('switch', { name: 'Move automatically if a computer goes offline' }))
  await waitFor(() => expect(calls('groups.custody.automatic').map(call => [call[0].connectionId, call[2].enabled])).toEqual([['mac-mini', false]]))

  for (const [automatic, text] of [
    [{ state: 'ready', mode: 'majority', standby: { install_id: VPS, name: 'Home VPS' } }, 'Keeps running on its own. If Mac mini goes offline, Home VPS takes over within a minute.'],
    [{ state: 'not_ready', reason: 'voters_offline', offline: ['Home VPS'] }, 'Right now Home VPS is offline. If Mac mini goes offline before then, you’ll be asked where to continue.'],
    [{ state: 'unavailable', reason: 'needs_computers', needed: 1 }, 'If Mac mini goes offline, you’ll be asked where to continue it. Add 1 more always-on computer to make this automatic.'],
    [{ state: 'off' }, 'Moves only when you choose.']
  ] as const) {
    current = status({ automatic, actions: [] })
    cleanup()
    activation.epoch++
    render(<CanonicalGroupWorkspace binding={binding} />)
    fireEvent.click(await screen.findByRole('button', { name: 'Hosted on Mac mini' }))
    expect(await screen.findByText(text)).toBeTruthy()
    expect(screen.queryByRole('switch', { name: 'Move automatically if a computer goes offline' })).toBeNull()
  }
})

it('moves the group on purpose with a handover and follows it to the new host', async () => {
  let moved = false
  host({
    'groups.succession.status': () => moved
      ? status({ host: { install_id: VPS, name: 'Home VPS', reachable: true, since: null }, this_install: { install_id: MINI, name: 'Mac mini', role: 'backup' } })
      : status({ automatic: { state: 'ready', mode: 'majority', standby: { install_id: VPS, name: 'Home VPS' } }, actions: [{ action: 'move', targets: [VPS] }] }),
    'groups.succession.move': () => {
      moved = true

      return status({ state: 'moving', moving: { to: { install_id: VPS, name: 'Home VPS' }, step: 'catching_up', reason: 'handover' } })
    }
  })
  handlers.vps = computer(VPS, { 'groups.state': () => roomState({}, 2), 'groups.log': () => ({ events: [] }),
    'groups.succession.status': () => status({ host: { install_id: VPS, name: 'Home VPS', reachable: true, since: null },
      this_install: { install_id: VPS, name: 'Home VPS', role: 'host' } }) })
  render(<CanonicalGroupWorkspace binding={binding} />)
  fireEvent.click(await screen.findByRole('button', { name: 'Hosted on Mac mini' }))
  fireEvent.click(await screen.findByRole('button', { name: 'Move to another computer…' }))
  fireEvent.click(await screen.findByRole('button', { name: 'Home VPS · up to date' }))
  const confirm = within(await screen.findByRole('dialog'))
  expect(confirm.getByText('Move “Harbor launch” to Home VPS?')).toBeTruthy()
  expect(confirm.getByText('It continues there, and Mac mini keeps a copy.')).toBeTruthy()
  expect(confirm.queryByText(/Only continue if/)).toBeNull()
  await act(async () => {fireEvent.click(confirm.getByRole('button', { name: 'Move to Home VPS' }))})
  expect(calls('groups.succession.move').map(call => [call[0].connectionId, call[2].target_install_id])).toEqual([['mac-mini', VPS]])
  await waitFor(() => expect(calls('groups.state').some(call => call[0].connectionId === 'vps')).toBe(true), { timeout: 5000 })
  expect(notify).toHaveBeenCalledWith({ kind: 'success', message: 'Continued on Home VPS.' })
})

it('shows an automatic move without buttons, and a host paused to stay safe with the owner’s override', async () => {
  let online = true
  handlers['mac-mini'] = (method, params) => online ? computer(MINI, { 'groups.state': () => roomState(), 'groups.log': () => ({ events: [] }),
    'groups.succession.status': () => status() })(method, params) : unreachable(method, params)
  handlers.vps = computer(VPS, { 'groups.succession.status': () => offlineStatus({ state: 'moving', actions: [],
    moving: { to: { install_id: VPS, name: 'Home VPS' }, step: 'fencing', reason: 'automatic' } }) })
  render(<CanonicalGroupWorkspace binding={binding} />)
  await screen.findByRole('button', { name: 'Hosted on Mac mini' })
  online = false
  expect(await screen.findByText('Mac mini went offline. Moving to Home VPS…', {}, { timeout: 5000 })).toBeTruthy()
  expect(screen.getByText('Stopping work from Mac mini')).toBeTruthy()
  expect(screen.queryByRole('button', { name: /Continue|Cancel|Other computers/ })).toBeNull()
  cleanup()

  activation.epoch++
  localStorage.clear()
  host({ 'groups.succession.status': () => status({ state: 'paused', actions: [{ action: 'continue_anyway' }],
    paused: { reason: 'lost_majority', waiting_for: [{ install_id: VPS, name: 'Home VPS' }, { install_id: LAPTOP, name: 'Laptop' }] } }),
  'groups.succession.continue_anyway': () => ({}) })
  render(<CanonicalGroupWorkspace binding={binding} />)
  expect(await screen.findByText('Paused to stay safe')).toBeTruthy()
  expect(screen.getByText('Mac mini can’t reach Home VPS and Laptop, so it can’t be sure another computer hasn’t taken over. It resumes as soon as one of them is back.')).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Continue on Mac mini anyway' }))
  const confirm = within(await screen.findByRole('dialog'))
  expect(confirm.getByText('Only do this if Home VPS and Laptop are really offline. If one of them took over, the group would run in two places.')).toBeTruthy()
  await act(async () => {fireEvent.click(confirm.getByRole('button', { name: 'Continue anyway' }))})
  await waitFor(() => expect(calls('groups.succession.continue_anyway').map(call => call[0].connectionId)).toEqual(['mac-mini']))
})

it('words automatic and handover moves, and never claims that no messages were lost', async () => {
  const events = [
    { seq: 1, event_id: 'auto-safe', kind: 'authority.transition', actor: system, created_at: today(9),
      payload: { text: 'English', reason: 'automatic', proof_kind: 'certified', at_risk: 0, to_name: 'Home VPS', from_name: 'Mac mini', offline_since: today(9) } },
    { seq: 2, event_id: 'auto-risk', kind: 'authority.transition', actor: system,
      payload: { text: 'English', reason: 'automatic', proof_kind: 'certified', at_risk: 2, to_name: 'Laptop', from_name: 'Home VPS', offline_since: today(10) } },
    { seq: 3, event_id: 'handover', kind: 'authority.transition', actor: system,
      payload: { text: 'English', reason: 'handover', proof_kind: 'certified', at_risk: 0, to_name: 'Mac mini', from_name: 'Laptop' } },
    { seq: 4, event_id: 'auto-careful', kind: 'authority.transition', actor: system,
      payload: { text: 'English', reason: 'automatic', proof_kind: 'evidence', at_risk: 0, to_name: 'Home VPS', from_name: 'Mac mini', offline_since: today(12) } }
  ]

  host({ 'groups.succession.status': () => status() }, events)
  render(<CanonicalGroupWorkspace binding={binding} />)
  const history = within(screen.getByRole('log'))
  expect(await history.findByText(`This group moved to Home VPS because Mac mini went offline at ${timeOf(today(9))}.`)).toBeTruthy()
  expect(history.queryByText(/No messages were lost/)).toBeNull()
  expect(history.getByText(`This group moved to Laptop because Home VPS went offline at ${timeOf(today(10))}.`)).toBeTruthy()
  expect(history.getByText('This group moved to Mac mini because Laptop was shutting down.')).toBeTruthy()
  expect(history.getByText(`This group moved to Home VPS because Mac mini went offline at ${timeOf(today(12))}.`)).toBeTruthy()
  expect(history.queryByText('English')).toBeNull()
  // A majority move is informational only.
  expect(screen.queryByText(/went silent for 3 minutes/)).toBeNull()
})

it('warns once after a careful move, with going back and asking first offered only to the owner', async () => {
  const transition = { seq: 4, event_id: 'careful-1', kind: 'authority.transition', actor: system, created_at: today(11),
    payload: { text: 'English', reason: 'automatic', proof_kind: 'evidence', at_risk: 0, to_name: 'Home VPS', from_name: 'Mac mini', offline_since: today(11) } }

  let movedIn: unknown = null

  const hostedOnVps = (actions: unknown[]) => status({ host: { install_id: VPS, name: 'Home VPS', reachable: true, since: null },
    this_install: { install_id: VPS, name: 'Home VPS', role: 'host' }, previous_host: { install_id: MINI, name: 'Mac mini', offline_since: today(11) },
    backups: [backup(MINI, 'Mac mini')], actions, moved_in: movedIn })

  let actions: unknown[] = [{ action: 'keep', targets: [MINI] }, { action: 'automatic', enabled: true }]
  handlers.vps = computer(VPS, { 'groups.state': () => roomState({}, 2), 'groups.log': () => ({ events: [transition] }),
    'groups.succession.status': () => hostedOnVps(actions), 'groups.succession.keep': () => ({}),
    'groups.custody.automatic': (_method, params) => ({ room_id: params.room_id, automatic: params.enabled, configuration_seq: 3 }) })
  const vps = { ...binding, connectionId: 'vps' }
  render(<CanonicalGroupWorkspace binding={vps} />)
  expect(await screen.findByText('“Harbor launch” moved to Home VPS')).toBeTruthy()
  expect(screen.getByText('Mac mini went silent for 3 minutes, so Home VPS took over. If Mac mini is actually still running, the group may now be running in both places.')).toBeTruthy()
  await waitFor(() => expect(osNotify).toHaveBeenCalledTimes(1))
  expect(osNotify).toHaveBeenCalledWith({ title: '“Harbor launch” moved to Home VPS',
    body: 'Mac mini went silent for 3 minutes, so Home VPS took over. If Mac mini is actually still running, the group may now be running in both places.' })

  fireEvent.click(await screen.findByRole('button', { name: 'Go back to Mac mini' }))
  const confirm = within(await screen.findByRole('dialog'))
  expect(confirm.getByText(`Home VPS pauses now, and the group continues on Mac mini as soon as it’s reachable. Messages sent on Home VPS since ${timeOf(today(11))} are kept separately.`)).toBeTruthy()
  await act(async () => {fireEvent.click(confirm.getByRole('button', { name: 'Go back to Mac mini' }))})
  await waitFor(() => expect(calls('groups.succession.keep').map(call => [call[0].connectionId, call[2].install_id])).toEqual([['vps', MINI]]))
  await waitFor(() => expect(screen.queryByText('“Harbor launch” moved to Home VPS')).toBeNull())
  cleanup()

  // Acknowledged once, it stays acknowledged; the OS notification is not repeated.
  render(<CanonicalGroupWorkspace binding={vps} />)
  await screen.findByRole('button', { name: 'Hosted on Home VPS' })
  expect(screen.queryByText('“Harbor launch” moved to Home VPS')).toBeNull()
  expect(osNotify).toHaveBeenCalledTimes(1)
  cleanup()

  localStorage.clear()
  actions = [{ action: 'automatic', enabled: true }]
  activation.epoch++
  render(<CanonicalGroupWorkspace binding={vps} />)
  await screen.findByText('“Harbor launch” moved to Home VPS')
  fireEvent.click(await screen.findByRole('button', { name: 'Ask me first next time' }))
  expect(screen.queryByRole('button', { name: 'Go back to Mac mini' })).toBeNull()
  await waitFor(() => expect(calls('groups.custody.automatic').at(-1)?.[2]).toMatchObject({ enabled: false }))
  await waitFor(() => expect(screen.queryByText('“Harbor launch” moved to Home VPS')).toBeNull())
  cleanup()

  // Someone who isn't the owner gets no choices and no OS notification: an info line while the host reports the move.
  localStorage.clear()
  actions = []
  activation.epoch++
  const notified = osNotify.mock.calls.length
  render(<CanonicalGroupWorkspace binding={vps} />)
  await screen.findByRole('button', { name: 'Hosted on Home VPS' })
  expect(screen.queryByText('“Harbor launch” moved to Home VPS')).toBeNull()
  cleanup()

  movedIn = { from: { install_id: MINI, name: 'Mac mini' }, at: today(11), proof_kind: 'evidence' }
  activation.epoch++
  render(<CanonicalGroupWorkspace binding={vps} />)
  const info = await screen.findByText('“Harbor launch” moved to Home VPS')
  expect(info.closest('[data-slot="group-careful-info"]')).toBeTruthy()
  expect(screen.queryByRole('button', { name: /Keep going|Go back|Ask me first/ })).toBeNull()
  expect(osNotify).toHaveBeenCalledTimes(notified)
})

it('names when both computers ran the group after a careful move', async () => {
  const start = today(8), end = today(9)
  host({ 'groups.succession.status': () => status({ state: 'continued_on_two', actions: [{ action: 'keep', targets: [MINI, VPS] }],
    conflict: { hosts: [{ install_id: MINI, name: 'Mac mini' }, { install_id: VPS, name: 'Home VPS' }], start, end } }) })
  render(<CanonicalGroupWorkspace binding={binding} />)
  const moment = (seconds: number) => new Intl.DateTimeFormat('en', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }).format(new Date(seconds * 1000))
  expect(await screen.findByText(`“Harbor launch” ran on both Mac mini and Home VPS while they couldn’t reach each other (${moment(start)}–${moment(end)}). Choose which one to keep. The other’s messages are kept separately.`)).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Keep Mac mini' })).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Keep Home VPS' })).toBeTruthy()
})

it('ends a split it can see by handing the older host the newer chain, quietly, and only shows one that remains', async () => {
  const chain = [
    { seq: 5, event_id: 'config-old', kind: 'custody.configured', actor: system, payload: { voters: [] } },
    { seq: 6, event_id: 'moved', kind: 'authority.transition', actor: system, payload: { to_epoch: 2, reason: 'automatic', proof_kind: 'evidence' } },
    { seq: 7, event_id: 'note', kind: 'message.user', actor: { kind: 'user', id: 'desktop' }, payload: { text: 'after' } },
    { seq: 8, event_id: 'config-new', kind: 'custody.configured', actor: system, payload: { voters: [] } }
  ]

  let stepped = false

  let learn: (params: Record<string, unknown>) => unknown = () => {stepped = true;

 return {}}

  host({ 'groups.succession.status': () => stepped ? status({ this_install: { install_id: MINI, name: 'Mac mini', role: 'backup' },
    host: { install_id: VPS, name: 'Home VPS', reachable: true, since: null } }) : status({ actions: [{ action: 'keep', targets: [MINI, VPS] }] }),
  'groups.succession.learn': (_method, params) => learn(params) })
  handlers.vps = computer(VPS, { 'groups.state': () => roomState({}, 2), 'groups.log': () => ({ events: chain, has_more: false }),
    'groups.succession.status': () => status({ host: { install_id: VPS, name: 'Home VPS', reachable: true, since: null },
      this_install: { install_id: VPS, name: 'Home VPS', role: 'host' } }) })
  handlers.laptop = computer(LAPTOP, { 'groups.succession.status': () => status({ this_install: { install_id: LAPTOP, name: 'Laptop', role: 'backup' } }) })
  render(<CanonicalGroupWorkspace binding={binding} />)
  await waitFor(() => expect(calls('groups.succession.learn')).toHaveLength(1), { timeout: 5000 })
  const learned = calls('groups.succession.learn')[0]
  expect(learned[0].connectionId).toBe('mac-mini')
  expect((learned[2].events as { event_id: string }[]).map(event => event.event_id)).toEqual(['moved', 'config-new'])
  expect(screen.queryByText('This group is running in two places')).toBeNull()
  expect(osNotify).not.toHaveBeenCalled()
  expect(request.mock.calls.some(call => call[0].connectionId === 'unrelated')).toBe(false)
  cleanup()

  // When the older host stays a host, the split is shown loudly, once, with the owner's Keep.
  stepped = false

  learn = () => {throw new Error('refused')}
  activation.epoch++
  render(<CanonicalGroupWorkspace binding={binding} />)
  expect(await screen.findByText('This group is running in two places', {}, { timeout: 5000 })).toBeTruthy()
  expect(screen.getByText('Mac mini and Home VPS are both running “Harbor launch”. Choose one now; the other’s messages are kept separately.')).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Keep Home VPS' })).toBeTruthy()
  await waitFor(() => expect(osNotify).toHaveBeenCalledTimes(1))
  expect(calls('groups.succession.learn')).toHaveLength(2)
})

const journal = () => JSON.parse(localStorage.getItem('hermes.desktop.canonicalGroupSends.v1') || '{}') as Record<string, Record<string, unknown>>
const UNSAVED = 'Not yet saved on another computer'

it('marks a message only the host holds so far, and clears the mark once the host reports it saved elsewhere', async () => {
  const log: Record<string, unknown>[] = []
  let covered = 0
  host({
    'groups.log': (_method, params) => ({ events: log.filter(event => (event.seq as number) > (params.since_seq as number)) }),
    'groups.send': (_method, params) => {
      log.push({ seq: 1, event_id: 'user:hello', room_id: binding.roomId, kind: 'message.user', actor: { kind: 'user', id: 'desktop' },
        payload: { text: 'Hello' } })

      return { accepted: true, client_event_id: params.event_id, event: { event_id: 'user:hello', seq: 1 }, protected: false }
    },
    'groups.succession.status': () => status(),
    'groups.custody.status': () => ({ room_id: binding.roomId, role: 'authority', protected_seq: covered, mode: 'majority' })
  })
  render(<CanonicalGroupWorkspace binding={binding} />)
  const box = await screen.findByRole('textbox') as HTMLTextAreaElement
  await waitFor(() => expect(box.disabled).toBe(false))
  fireEvent.change(box, { target: { value: 'Hello' } })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))

  const history = within(screen.getByRole('log'))
  expect(await history.findByText(UNSAVED)).toBeTruthy()
  expect(Object.values(journal()).map(entry => [entry.acknowledged, entry.unsaved])).toEqual([[true, { seq: 1, event_id: 'user:hello' }]])
  await waitFor(() => expect(box.value).toBe(''))
  expect(box.disabled).toBe(false)

  // A later reading from the running host covers it: the mark and the journal entry go.
  covered = 1
  log.push({ seq: 2, event_id: 'config', room_id: binding.roomId, kind: 'custody.configured', actor: system, payload: {} })
  await waitFor(() => expect(history.queryByText(UNSAVED)).toBeNull(), { timeout: 5000 })
  expect(journal()).toEqual({})
  expect(history.getByText('Hello')).toBeTruthy()
})

it('offers a message the old host held alone to the new host after a move, with the same identity', async () => {
  const entry = await prepareCanonicalGroupSend(binding, { text: 'Before the move' })
  await attemptCanonicalGroupSend(binding, entry)
  await settleCanonicalGroupSend(binding, entry, { protected: false, event: { event_id: 'user:before', seq: 6 } })
  const moved = { ...binding, connectionId: 'vps' }
  await rehomeCanonicalGroupSends(binding, moved)

  handlers.vps = computer(VPS, { 'groups.state': () => roomState({}, 2), 'groups.log': () => ({ events: [] }),
    'groups.succession.status': () => status({ host: { install_id: VPS, name: 'Home VPS', reachable: true, since: null },
      this_install: { install_id: VPS, name: 'Home VPS', role: 'host' } }),
    'groups.send': (_method, params) => ({ accepted: true, client_event_id: params.event_id, event: { event_id: 'user:before', seq: 3 }, protected: true }) })
  render(<CanonicalGroupWorkspace binding={moved} />)
  await waitFor(() => expect(calls('groups.send')).toHaveLength(1), { timeout: 5000 })
  expect(calls('groups.send').map(call => [call[0].connectionId, call[2].event_id, (call[2].payload as { text: string }).text]))
    .toEqual([['vps', entry.params.event_id, 'Before the move']])
  await waitFor(() => expect(journal()).toEqual({}))
  expect(calls('groups.send')).toHaveLength(1)
})
