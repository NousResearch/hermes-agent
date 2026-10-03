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
import { backup, binding, computer, GUEST, type Handler, hex, LAPTOP, MINI, offlineStatus, refusal, registry, roomState, runBy, status,
  unreachable, VPS } from './canonical-group-succession-fixtures'
import { CanonicalGroupWorkspace } from './canonical-group-workspace'
import { translateBots } from './i18n-test-helper'

let handlers: Record<string, Handler> = {}
const calls = (method: string) => request.mock.calls.filter(call => call[1] === method)
const system = { kind: 'system', id: 'room-driver' }
const journal = () => JSON.parse(localStorage.getItem('hermes.desktop.canonicalGroupSends.v1') || '{}') as Record<string, Record<string, unknown>>
const vpsBinding = { ...binding, connectionId: 'vps' }
const onVps = { host: { install_id: VPS, name: 'Home VPS', reachable: true, since: null }, this_install: { install_id: VPS, name: 'Home VPS', role: 'host' } }

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
afterEach(() => {
  cleanup(); request.mockReset(); notify.mockReset(); connections.mockReset(); osNotify.mockReset(); localStorage.clear(); vi.restoreAllMocks()
  vi.useRealTimers()
})

async function ready() {
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false))
}

function write(text: string) {
  fireEvent.change(screen.getByRole('textbox'), { target: { value: text } })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
}

/** The room opens on its host, which then goes offline; the returned call takes it offline. */
async function hostThatGoesOffline(methods: Record<string, Handler> = {}) {
  let online = true
  handlers['mac-mini'] = (method, params) => online ? computer(MINI, {
    'groups.state': () => roomState(), 'groups.log': () => ({ events: [] }), 'groups.succession.status': () => status(), ...methods
  })(method, params) : unreachable(method, params)
  render(<CanonicalGroupWorkspace binding={binding} />)
  await ready()
  await screen.findByRole('button', { name: 'Hosted on Mac mini' })

  return () => {online = false}
}

it('keeps your messages the new host doesn’t have where they were after a move, and sends one again only on a tap', async () => {
  const mine = (seq: number, id: string, text: string, at: number) => ({ seq, event_id: id, room_id: binding.roomId, kind: 'message.user',
    actor: { kind: 'user', id: 'desktop' }, created_at: at, payload: { text } })

  const reply = { seq: 2, event_id: 'reply-1', room_id: binding.roomId, kind: 'message.member', created_at: 1_700_000_020,
    actor: { kind: 'member', id: 'atlas', display_name: 'Atlas Bot' }, payload: { text: 'On it' } }

  const transition = { seq: 3, event_id: 'moved', room_id: binding.roomId, kind: 'authority.transition', actor: system, created_at: 1_700_000_100,
    payload: { reason: 'automatic', proof_kind: 'evidence', at_risk: 1, to_name: 'Home VPS', from_name: 'Mac mini', offline_since: 1_700_000_040 } }

  const first = mine(1, 'user:first', 'First', 1_700_000_010)
  const vpsLog: Record<string, unknown>[] = [first, reply, transition]
  let online = true
  handlers['mac-mini'] = (method, params) => online ? computer(MINI, { 'groups.state': () => roomState(),
    'groups.log': () => ({ events: [first, reply, mine(3, 'user:late', 'Only the old host got this', 1_700_000_030)] }),
    'groups.succession.status': () => status() })(method, params) : unreachable(method, params)
  handlers.vps = computer(VPS, { 'groups.state': () => roomState({}, 2),
    'groups.log': (_method, params) => ({ events: vpsLog.filter(event => (event.seq as number) > (params.since_seq as number)) }),
    'groups.succession.status': () => status({ ...onVps, previous_host: { install_id: MINI, name: 'Mac mini', offline_since: 1_700_000_040 } }),
    'groups.send': (_method, params) => {
      vpsLog.push(mine(vpsLog.length + 1, 'user:again', String((params.payload as { text: string }).text), 1_700_000_200))

      return { accepted: true, client_event_id: params.event_id }
    } })
  render(<CanonicalGroupWorkspace binding={binding} />)
  await within(screen.getByRole('log')).findByText('Only the old host got this')
  online = false

  // The host's connection fails; the backup that now hosts the room answers, and the room follows it there.
  expect(await screen.findByRole('button', { name: 'Hosted on Home VPS' }, { timeout: 5000 })).toBeTruthy()
  const history = within(screen.getByRole('log'))
  const missing = await history.findByText('Didn’t reach Home VPS before Mac mini went offline.')
  const rows = [...screen.getByRole('log').querySelectorAll('article')].map(row => row.textContent ?? '')
  expect(rows.findIndex(row => row.includes('Only the old host got this'))).toBe(rows.findIndex(row => row.includes('On it')) + 1)
  expect(rows.at(-1)).toMatch(/This group moved to Home VPS/)
  expect(calls('groups.send')).toHaveLength(0)

  await ready()
  fireEvent.click(within(missing.closest('[data-slot="missing-message"]') as HTMLElement).getByRole('button', { name: 'Send again' }))
  await waitFor(() => expect(calls('groups.send')).toHaveLength(1))
  expect(calls('groups.send').map(call => [call[0].connectionId, (call[2].payload as { text: string }).text])).toEqual([['vps', 'Only the old host got this']])
  await waitFor(() => expect(history.queryByText('Didn’t reach Home VPS before Mac mini went offline.')).toBeNull(), { timeout: 5000 })
  expect(calls('groups.send')).toHaveLength(1)
})

it('hands a message the new host refuses for good back to the composer, with the reason', async () => {
  const entry = await prepareCanonicalGroupSend(binding, { text: 'Only on the host so far', attachments: [] })
  await attemptCanonicalGroupSend(binding, entry)
  await settleCanonicalGroupSend(binding, entry, { protected: false, event: { event_id: 'user:held', seq: 4 } })
  await rehomeCanonicalGroupSends(binding, vpsBinding)
  handlers.vps = computer(VPS, { 'groups.state': () => roomState({}, 2), 'groups.log': () => ({ events: [] }),
    'groups.succession.status': () => status(onVps), 'groups.send': () => {throw refusal('invalid_params')} })
  render(<CanonicalGroupWorkspace binding={vpsBinding} />)
  const box = await screen.findByRole('textbox') as HTMLTextAreaElement
  await waitFor(() => expect(box.value).toBe('Only on the host so far'))
  expect(screen.getByText('Home VPS didn’t accept this message after the move: something in it, such as an attachment, isn’t available there. Edit it and send it again.')).toBeTruthy()
  await waitFor(() => expect(journal()).toEqual({}))
  await ready()
  expect(calls('groups.send')).toHaveLength(1)
})

it('holds a message a host that paused to stay safe refuses, shows the pause, and delivers it once the group resumes', async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true })
  let paused = false
  const delivered: string[] = []
  handlers['mac-mini'] = computer(MINI, { 'groups.state': () => roomState(), 'groups.log': () => ({ events: [] }),
    'groups.succession.status': () => paused ? status({ state: 'paused', actions: [{ action: 'continue_anyway' }],
      paused: { reason: 'lost_majority', since: 1, waiting_for: [{ install_id: VPS, name: 'Home VPS' }, { install_id: LAPTOP, name: 'Laptop' }] } }) : status(),
    'groups.send': (_method, params) => {
      if (paused) {throw refusal('room_host_paused')}
      delivered.push(String(params.event_id))

      return { accepted: true, client_event_id: params.event_id }
    } })
  render(<CanonicalGroupWorkspace binding={binding} />)
  await ready()
  await screen.findByRole('button', { name: 'Hosted on Mac mini' })

  // The host paused before Desktop read it: the refused message is held, not failed, and the pause shows.
  paused = true
  write('Are you there?')
  expect(await screen.findByText('Paused to stay safe')).toBeTruthy()
  expect(screen.getByText('Mac mini can’t reach Home VPS and Laptop, so it can’t be sure another computer hasn’t taken over. It resumes as soon as one of them is back.')).toBeTruthy()
  expect(screen.getByText('A message you send now will be delivered when the group resumes.')).toBeTruthy()
  expect(screen.queryByRole('alert')).toBeNull()
  const [held] = Object.values(journal())
  expect([held.held, held.attempted]).toEqual([true, false])
  const identity = (held.params as { event_id: string }).event_id

  paused = false
  await act(async () => {await vi.advanceTimersByTimeAsync(31_000)})
  await waitFor(() => expect(delivered).toEqual([identity]))
  await waitFor(() => expect(screen.queryByText('Paused to stay safe')).toBeNull())

  // A pause is read every 30 seconds while the host answers, since a paused host appends nothing.
  paused = true
  await act(async () => {await vi.advanceTimersByTimeAsync(31_000)})
  expect(await screen.findByText('Paused to stay safe')).toBeTruthy()
})

it('says why a host paused itself, in words for each reason it gives, keeping the owner’s Continue anyway', async () => {
  for (const [paused, text, action, title, body] of [
    [{ reason: 'no_lease_layer', waiting_for: [] }, 'Mac mini can’t take part in automatic moves right now, so it paused to stay safe. Its connection to the other computers isn’t ready.',
      { action: 'continue_anyway', turns_off_automatic: true }, 'Continue on Mac mini without automatic moves?',
      'Mac mini can’t take part in automatic moves right now. If you continue, automatic moves are turned off for this group until you turn them back on.'],
    [{ reason: 'something_new', waiting_for: [] }, 'Mac mini paused this group to stay safe. Nothing new runs until it resumes.',
      { action: 'continue_anyway' }, 'Continue on Mac mini anyway',
      'Only do this if no other computer has taken over this group. If one has, the group would run in two places.']
  ] as const) {
    handlers['mac-mini'] = computer(MINI, { 'groups.state': () => roomState(), 'groups.log': () => ({ events: [] }),
      'groups.succession.status': () => status({ state: 'paused', paused, actions: [action] }) })
    render(<CanonicalGroupWorkspace binding={binding} />)
    expect(await screen.findByText(text)).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Continue on Mac mini anyway' }))
    const confirm = within(await screen.findByRole('dialog', { name: title }))
    expect(confirm.getByText(body)).toBeTruthy()
    cleanup()
    activation.epoch++
  }
})

it('holds a message while a planned move waits, even when its host refuses it, and delivers it on the new host', async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true })
  let phase: 'ok' | 'waiting' | 'moved' = 'ok'
  const delivered: [string, string][] = []

  const waiting = () => status({ state: 'moving', actions: [{ action: 'move_now' }],
    moving: { to: { install_id: VPS, name: 'Home VPS' }, step: 'waiting_for_turns', running: 1, reason: 'manual' } })

  handlers['mac-mini'] = computer(MINI, { 'groups.state': () => roomState({ working: true }), 'groups.log': () => ({ events: [] }),
    'groups.succession.status': () => phase === 'ok' ? status() : phase === 'waiting' ? waiting()
      : status({ host: { install_id: VPS, name: 'Home VPS', reachable: true, since: null }, this_install: { install_id: MINI, name: 'Mac mini', role: 'backup' } }),
    // The move started elsewhere: the host already promised the group to Home VPS and stores nothing.
    'groups.send': () => {throw refusal('room_authority_promised', { other: { install_id: VPS, name: 'Home VPS' } })} })
  handlers.vps = computer(VPS, { 'groups.state': () => roomState({}, 2), 'groups.log': () => ({ events: [] }),
    'groups.succession.status': () => status(onVps),
    'groups.send': (_method, params) => {
      delivered.push(['vps', String(params.event_id)])

      return { accepted: true, client_event_id: params.event_id }
    } })
  render(<CanonicalGroupWorkspace binding={binding} />)
  await ready()
  await screen.findByRole('button', { name: 'Hosted on Mac mini' })

  phase = 'waiting'
  write('Before the move')
  expect(await screen.findByText('Moving to Home VPS after the replies in progress finish (1).')).toBeTruthy()
  expect(screen.getByText('A message you send now will be delivered when the group resumes.')).toBeTruthy()
  expect(screen.queryByRole('alert')).toBeNull()
  // Stop still reaches the host that answers.
  expect(screen.getByRole('button', { name: 'Stop' })).toBeTruthy()
  const [held] = Object.values(journal())
  expect([held.held, held.attempted]).toEqual([true, false])
  const identity = (held.params as { event_id: string }).event_id
  const refused = calls('groups.send').length
  await act(async () => {await vi.advanceTimersByTimeAsync(20_000)})
  expect(calls('groups.send')).toHaveLength(refused)

  phase = 'moved'
  await act(async () => {await vi.advanceTimersByTimeAsync(20_000)})
  expect(await screen.findByRole('button', { name: 'Hosted on Home VPS' }, { timeout: 5000 })).toBeTruthy()
  await waitFor(() => expect(delivered).toEqual([['vps', identity]]))
})

it('confirms continuing once, saying what only the old host has, what the target still fetches, and who can’t be reached', async () => {
  const offline = await hostThatGoesOffline()
  let release: () => void = () => undefined
  let cautions: unknown[] = [{ code: 'host_may_be_running' }, { code: 'voters_unreachable', names: ['Laptop'], count: 1 }]
  handlers.vps = computer(VPS, {
    'groups.succession.status': () => offlineStatus(),
    'groups.succession.prepare': () => ({ preview_id: 'preview-1', target: { install_id: VPS, name: 'Home VPS', operator_name: null }, owner: { name: 'Dana' },
      behind_by: 4, at_risk: { count: 2 }, work: { completed: 0, elsewhere: 0, unknown: 0, waiting_for_host: 0 }, unavailable_bots: [], cautions }),
    'groups.succession.promote': () => new Promise(resolve => {release = () => resolve(offlineStatus({ state: 'moving' }))})
  })
  offline()
  fireEvent.click(await screen.findByRole('button', { name: 'Continue on Home VPS' }, { timeout: 5000 }))
  let dialog = within(await screen.findByRole('dialog'))
  expect(dialog.getByText('Home VPS is missing 2 recent messages. They’ll appear if Mac mini comes back.')).toBeTruthy()
  expect(dialog.getByText('Home VPS is catching up 4 messages from another computer.')).toBeTruthy()
  expect(dialog.getByText('Laptop can’t be reached, so this computer can’t confirm Mac mini has stopped. Continue only if Mac mini is really offline.')).toBeTruthy()
  fireEvent.click(dialog.getByRole('button', { name: 'Cancel' }))
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())

  cautions = [{ code: 'voters_unreachable', names: [], count: 2 }]
  fireEvent.click(screen.getByRole('button', { name: 'Continue on Home VPS' }))
  dialog = within(await screen.findByRole('dialog'))
  expect(dialog.getByText('2 computers can’t be reached, so this computer can’t confirm Mac mini has stopped. Continue only if Mac mini is really offline.')).toBeTruthy()

  // A double click sends one promote; the dialog stays busy until it answers.
  const confirm = dialog.getByRole('button', { name: 'Continue on Home VPS' })
  await act(async () => {fireEvent.click(confirm)})
  await act(async () => {fireEvent.click(confirm)})
  expect(calls('groups.succession.promote')).toHaveLength(1)
  await act(async () => {release()})
  expect(calls('groups.succession.promote')).toHaveLength(1)
})

it('moves once on a double click, and lets the owner move now while the move waits for replies in progress', async () => {
  let release: () => void = () => undefined
  let waiting = false

  const waitingStatus = () => status({ state: 'moving', moving: { to: { install_id: VPS, name: 'Home VPS' }, step: 'waiting_for_turns', running: 2, reason: 'manual' },
    actions: [{ action: 'move_now' }] })

  handlers['mac-mini'] = computer(MINI, { 'groups.state': () => roomState(), 'groups.log': () => ({ events: [] }),
    'groups.succession.status': () => waiting ? waitingStatus() : status({ automatic: { state: 'ready', mode: 'majority', standby: { install_id: VPS, name: 'Home VPS' } },
      actions: [{ action: 'move', targets: [VPS] }] }),
    'groups.succession.move': () => new Promise(resolve => {release = () => {waiting = true; resolve(waitingStatus())}}),
    'groups.succession.move_now': () => ({}) })
  render(<CanonicalGroupWorkspace binding={binding} />)
  fireEvent.click(await screen.findByRole('button', { name: 'Hosted on Mac mini' }))
  fireEvent.click(await screen.findByRole('button', { name: 'Move to another computer…' }))
  fireEvent.click(await screen.findByRole('button', { name: 'Home VPS · up to date' }))
  const confirm = within(await screen.findByRole('dialog')).getByRole('button', { name: 'Move to Home VPS' })
  await act(async () => {fireEvent.click(confirm)})
  await act(async () => {fireEvent.click(confirm)})
  expect(calls('groups.succession.move')).toHaveLength(1)
  await act(async () => {release()})

  expect(await screen.findByText('Moving to Home VPS after the replies in progress finish (2).', {}, { timeout: 3000 })).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Move now' }))
  const now = within(await screen.findByRole('dialog', { name: 'Move now?' }))
  expect(now.getByText('Replies still in progress will show as unknown on Home VPS and won’t rerun by themselves.')).toBeTruthy()
  await act(async () => {fireEvent.click(now.getByRole('button', { name: 'Move now' }))})
  await waitFor(() => expect(calls('groups.succession.move_now').map(call => [call[0].connectionId, call[2].room_id])).toEqual([['mac-mini', binding.roomId]]))
  expect(calls('groups.succession.move')).toHaveLength(1)
})

it('says the other computers are deciding while an automatic takeover waits, without offering to continue', async () => {
  const offline = await hostThatGoesOffline()
  handlers.vps = computer(VPS, { 'groups.succession.status': () => offlineStatus({ actions: [], unavailable_reason: 'takeover_waiting' }) })
  offline()
  expect(await screen.findByText('Mac mini went offline. The other computers are deciding which one takes over; this can take a few minutes.', {}, { timeout: 5000 })).toBeTruthy()
  expect(screen.queryByRole('button', { name: /^Continue on/ })).toBeNull()
  expect(screen.queryByText(/No other computer has a full copy/)).toBeNull()
})

it('says it is checking the host while a backup still sees it, instead of asking to connect', async () => {
  const offline = await hostThatGoesOffline()
  handlers.vps = computer(VPS, { 'groups.succession.status': () => status({ this_install: { install_id: VPS, name: 'Home VPS', role: 'backup' } }) })
  offline()
  expect(await screen.findByText('Checking Mac mini…', {}, { timeout: 5000 })).toBeTruthy()
  expect(screen.queryByText(/Connect to Mac mini/)).toBeNull()
})

it('shows which side keeps running a conflict, keeps going there first, and holds what you write on the side that stopped', async () => {
  const conflict = { hosts: [{ install_id: VPS, name: 'Home VPS', since: 2 }, { install_id: MINI, name: 'Mac mini', since: 1 }],
    start: 1_700_000_000, end: 1_700_000_300, running_on: { install_id: VPS, name: 'Home VPS' } }

  handlers.vps = computer(VPS, { 'groups.state': () => roomState({}, 2), 'groups.log': () => ({ events: [] }),
    'groups.succession.status': () => status({ ...onVps, state: 'continued_on_two', conflict, actions: [{ action: 'keep', targets: [VPS, MINI] }],
      backups: [backup(MINI, 'Mac mini')] }) })
  handlers['mac-mini'] = computer(MINI, { 'groups.state': () => roomState({}, 1), 'groups.log': () => ({ events: [] }),
    'groups.succession.learn': () => ({ learned: true }),
    'groups.succession.status': () => status({ state: 'continued_on_two', conflict, actions: [{ action: 'keep', targets: [VPS, MINI] }],
      backups: [backup(VPS, 'Home VPS')] }) })
  render(<CanonicalGroupWorkspace binding={binding} />)
  const running = await screen.findByText('Home VPS is running the group; Mac mini stopped.')
  const banner = running.closest('[data-slot="group-succession-banner"]') as HTMLElement
  expect(within(banner).getAllByRole('button').map(button => button.textContent)).toEqual(['Keep going on Home VPS', 'Keep Mac mini'])
  expect(screen.queryByText('This group is running in two places')).toBeNull()
  expect(calls('groups.succession.learn')).toHaveLength(0)

  await ready()
  write('Still here?')
  expect(await screen.findByText('A message you send now will be delivered when the group resumes.')).toBeTruthy()
  await waitFor(() => expect(Object.values(journal()).map(entry => [entry.held, entry.attempted])).toEqual([[true, false]]))
  expect(calls('groups.send')).toHaveLength(0)
  expect(osNotify).not.toHaveBeenCalled()
})

it('numbers a computer without a name, and shows a change to moving by itself as waiting for the other computers', async () => {
  handlers['mac-mini'] = computer(MINI, { 'groups.state': () => roomState(), 'groups.log': () => ({ events: [] }),
    'groups.succession.status': () => status({ actions: [{ action: 'automatic', enabled: true }],
      automatic: { mode: 'majority', state: 'not_ready', reason: 'voters_offline', standby: null, voters: [], enabled: true, pending: false,
        offline: [{ install_id: `install:${hex('f')}`, name: null }] } }) })
  render(<CanonicalGroupWorkspace binding={binding} />)
  fireEvent.click(await screen.findByRole('button', { name: 'Hosted on Mac mini' }))
  expect(await screen.findByText('Right now Computer 1 is offline. If Mac mini goes offline before then, you’ll be asked where to continue.')).toBeTruthy()
  expect(screen.getByText('Turning off… (waiting for the other computers)')).toBeTruthy()
  expect(screen.getByRole('switch', { name: 'Move automatically if a computer goes offline' }).getAttribute('aria-checked')).toBe('false')
})

it('reads a move less and less often while the computer watching it doesn’t answer', async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true })
  const offline = await hostThatGoesOffline()
  let answering = true
  const moving = offlineStatus({ state: 'moving', moving: { to: { install_id: VPS, name: 'Home VPS' }, step: 'catching_up', reason: 'manual' } })
  handlers.vps = (method, params) => answering ? computer(VPS, {
    'groups.succession.status': () => calls('groups.succession.promote').length ? moving : offlineStatus(),
    'groups.succession.prepare': () => ({ preview_id: 'preview-1', target: { install_id: VPS, name: 'Home VPS', operator_name: null }, owner: { name: 'Dana' },
      behind_by: 0, at_risk: { count: 0 }, work: null, unavailable_bots: [], cautions: [] }),
    'groups.succession.promote': () => offlineStatus({ state: 'moving', moving: { to: { install_id: VPS, name: 'Home VPS' }, step: 'fencing', reason: 'manual' } })
  })(method, params) : unreachable(method, params)
  offline()
  fireEvent.click(await screen.findByRole('button', { name: 'Continue on Home VPS' }, { timeout: 5000 }))
  const dialog = within(await screen.findByRole('dialog'))
  await act(async () => {fireEvent.click(dialog.getByRole('button', { name: 'Continue on Home VPS' }))})
  await screen.findByText('Continuing on Home VPS…')
  answering = false
  const before = calls('groups.succession.status').filter(call => call[0].connectionId === 'vps').length
  await act(async () => {await vi.advanceTimersByTimeAsync(60_000)})
  const reads = calls('groups.succession.status').filter(call => call[0].connectionId === 'vps').length - before
  expect(reads).toBeGreaterThan(0)
  expect(reads).toBeLessThanOrEqual(7)
})

it('says whose computers keep the whole history: in Backup copies, and before another person’s computer is added', async () => {
  const addBackup = vi.fn(async () => ({ ok: true, install_id: `install:${hex('d')}` }))
  window.hermesDesktop = { roomSetup: { create: vi.fn(), recover: vi.fn(), addBackup } } as unknown as typeof window.hermesDesktop
  handlers['mac-mini'] = computer(MINI, { 'groups.capabilities': runBy(MINI, 'Dana'), 'groups.state': () => roomState(), 'groups.log': () => ({ events: [] }),
    // Only two names that are set and differ make someone else's computer; an unnamed operator says nothing.
    'groups.succession.status': () => status({ backups: [backup(VPS, 'Home VPS'), backup(GUEST, 'Guest box', { operator_name: 'Sam', allowed: false, successor: false }),
      backup(LAPTOP, 'Laptop', { operator_name: null })],
      actions: [{ action: 'add_backup' }] }) })
  handlers.unrelated = computer(`install:${hex('d')}`, { 'groups.capabilities': runBy(`install:${hex('d')}`, 'Sam') })
  render(<CanonicalGroupWorkspace binding={binding} />)
  fireEvent.click(await screen.findByRole('button', { name: 'Hosted on Mac mini' }))
  const copies = within(await screen.findByRole('list', { name: 'Backup copies' }))
  expect(copies.getAllByText('Sam’s computer will keep a full copy of this group’s history, including earlier messages.')).toHaveLength(1)

  fireEvent.click(screen.getByRole('button', { name: 'Add a backup computer…' }))
  const add = within(await screen.findByRole('dialog'))
  await act(async () => {fireEvent.click(add.getByRole('button', { name: 'Add: Work box' }))})
  expect(await add.findByText('Sam’s computer will keep a full copy of this group’s history, including earlier messages.')).toBeTruthy()
  expect(addBackup).not.toHaveBeenCalled()
  await act(async () => {fireEvent.click(add.getByRole('button', { name: 'Add: Work box' }))})
  await waitFor(() => expect(addBackup).toHaveBeenCalledTimes(1))
})

it('offers the owner to move back where Bots left behind can take part again', async () => {
  let actions: unknown[] = [{ action: 'move', targets: [MINI] }]
  handlers.vps = computer(VPS, { 'groups.state': () => roomState({}, 2), 'groups.log': () => ({ events: [] }),
    'groups.succession.status': () => status({ ...onVps, previous_host: { install_id: MINI, name: 'Mac mini', offline_since: 1_700_000_000 },
      backups: [backup(MINI, 'Mac mini')], actions,
      unavailable_bots: [{ member_id: 'mira', name: 'Mira Bot', on: { install_id: MINI, name: 'Mac mini', reachable: true } }] }),
    'groups.succession.move': () => status({ ...onVps, state: 'moving', moving: { to: { install_id: MINI, name: 'Mac mini' }, step: 'fencing', reason: 'handover' } }) })
  render(<CanonicalGroupWorkspace binding={vpsBinding} />)
  expect(await screen.findByText('Mira Bot can take part again if the group moves back to Mac mini.')).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: /2 Bots$/ }))
  expect(await screen.findByText('unavailable until the group moves back to Mac mini', { exact: false })).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Move back' }))
  const confirm = within(await screen.findByRole('dialog'))
  expect(confirm.getByText('Move “Harbor launch” to Mac mini?')).toBeTruthy()
  await act(async () => {fireEvent.click(confirm.getByRole('button', { name: 'Move to Mac mini' }))})
  expect(calls('groups.succession.move').map(call => [call[0].connectionId, call[2].target_install_id])).toEqual([['vps', MINI]])
  cleanup()

  // Without the owner's move, there is nothing to offer.
  actions = []
  activation.epoch++
  render(<CanonicalGroupWorkspace binding={vpsBinding} />)
  await screen.findByRole('button', { name: 'Hosted on Home VPS' })
  expect(screen.queryByText(/can take part again/)).toBeNull()
})
