import type * as HermesSdk from '@hermes/plugin-sdk'
import { useStore } from '@nanostores/react'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

const { request, notify, connections, activation } = vi.hoisted(() => ({
  request: vi.fn(), notify: vi.fn(), connections: vi.fn(), activation: { epoch: 1 }
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

import { backup, binding, computer, GUEST, type Handler, hex, LAPTOP, MINI, offlineStatus, refusal, registry, roomState, status,
  unreachable, VPS } from './canonical-group-succession-fixtures'
import { CanonicalGroupWorkspace } from './canonical-group-workspace'
import { translateBots } from './i18n-test-helper'

let handlers: Record<string, Handler> = {}

const calls = (method: string) => request.mock.calls.filter(call => call[1] === method)
const routesOf = (method: string) => calls(method).map(call => call[0].connectionId)

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
afterEach(() => {cleanup(); request.mockReset(); notify.mockReset(); connections.mockReset(); localStorage.clear(); vi.restoreAllMocks()})

async function ready() {
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false))
}

async function hostThatGoesOffline(initial: Record<string, unknown> = {}) {
  let online = true
  handlers['mac-mini'] = (method, params) => online ? computer(MINI, {
    'groups.state': () => roomState(), 'groups.log': () => ({ events: [] }),
    'groups.succession.status': () => status(initial)
  })(method, params) : unreachable(method, params)
  render(<CanonicalGroupWorkspace binding={binding} />)
  await ready()
  await screen.findByRole('button', { name: 'Hosted on Mac mini' })

  return () => {online = false}
}


it('shows nothing about other computers when the host does not advertise continuation, even after it fails', async () => {
  let online = true
  handlers['mac-mini'] = (method, params) => online ? computer(MINI, { 'groups.state': () => roomState(), 'groups.log': () => ({ events: [] }) }, false)(method, params)
    : unreachable(method, params)
  handlers.vps = computer(VPS, { 'groups.succession.status': () => offlineStatus() })
  const view = render(<CanonicalGroupWorkspace binding={binding} />)
  await ready()
  online = false
  await screen.findByText('This group chat is unavailable. Reconnect its host, or update Hermes on that computer.', {}, { timeout: 5000 })
  expect(screen.queryByRole('button', { name: /Hosted on/ })).toBeNull()
  expect(view.container.querySelector('[data-slot="group-succession-banner"]')).toBeNull()
  expect(request.mock.calls.some(call => /^groups\.(succession|custody)\./.test(call[1]))).toBe(false)
  expect(request.mock.calls.every(call => call[0].connectionId === 'mac-mini')).toBe(true)
})

it('shows where the group runs and its backup copies from the host', async () => {
  handlers['mac-mini'] = computer(MINI, {
    'groups.state': () => roomState(), 'groups.log': () => ({ events: [] }),
    'groups.succession.status': () => status({ at_risk: { count: 2 }, backups: [backup(VPS, 'Home VPS'),
      backup(LAPTOP, 'Laptop', { readiness: 'behind', behind_by: 3 }), backup(GUEST, 'Guest box', { readiness: 'offline', last_seen: 1_700_000_000, successor: false, designated: false }),
      backup(`install:${hex('f')}`, 'Old Pi', { readiness: 'unsupported', successor: false }), backup(`install:${hex('9')}`, null, { readiness: 'unknown', successor: false }),
      backup(`install:${hex('8')}`, 'Studio', { readiness: 'needs_reauthorization', successor: false })] })
  })
  const view = render(<CanonicalGroupWorkspace binding={binding} />)
  await ready()
  fireEvent.click(await screen.findByRole('button', { name: 'Hosted on Mac mini' }))
  const panel = within(await screen.findByRole('list', { name: 'Backup copies' }))
  expect(panel.getByText('Home VPS: full copy, up to date')).toBeTruthy()
  expect(panel.getByText('Laptop: full copy, 3 messages behind')).toBeTruthy()
  expect(panel.getByText(/^Guest box: offline since /)).toBeTruthy()
  expect(panel.getByText('Old Pi needs a newer Hermes to keep a copy of this group.')).toBeTruthy()
  expect(panel.getByText('Studio needs to be reconnected: its permission to keep a copy of this group expired.')).toBeTruthy()
  expect(panel.getByText('Another computer: full copy, not confirmed yet')).toBeTruthy()
  expect(screen.getByText('If Mac mini goes offline, you can continue this group on Home VPS and Laptop.')).toBeTruthy()
  expect(screen.getByText('2 recent messages are only on Mac mini so far.')).toBeTruthy()
  // Owner controls come only from the host's actions: none were offered.
  expect(screen.queryByRole('switch')).toBeNull()
  expect(screen.queryByRole('button', { name: 'Add a backup computer…' })).toBeNull()
  expect(routesOf('groups.succession.status')).toEqual(['mac-mini'])
  expect(view.container.querySelector('[data-slot="group-succession-banner"]')).toBeNull()
})

it('asks only the room’s remembered backups when the host fails, and keeps the composer usable while paused', async () => {
  const offline = await hostThatGoesOffline()
  handlers.vps = computer(VPS, { 'groups.succession.status': () => offlineStatus() })
  handlers.laptop = computer(LAPTOP, { 'groups.succession.status': () => offlineStatus({ this_install: { install_id: LAPTOP, name: 'Laptop', role: 'backup' } }) })
  handlers.unrelated = computer(`install:${hex('d')}`, {})
  offline()

  expect(await screen.findByText('Mac mini is offline', {}, { timeout: 5000 })).toBeTruthy()
  expect(screen.getByText('The group is paused. Nothing new will run until Mac mini is back or you continue the group on another computer.')).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Continue on Home VPS' })).toBeTruthy()
  fireEvent.pointerDown(screen.getByRole('button', { name: 'Other computers…' }), { button: 0, ctrlKey: false })
  expect(await screen.findByRole('menuitem', { name: 'Laptop · catching up 3 messages' })).toBeTruthy()
  fireEvent.keyDown(screen.getByRole('menu'), { key: 'Escape' })
  // Only the remembered backup's own connection, confirmed first; never an unrelated one.
  expect(request.mock.calls.some(call => call[0].connectionId === 'unrelated')).toBe(false)
  expect(calls('groups.capabilities').filter(call => call[0].connectionId === 'vps')[0][0]).toMatchObject({ profile: 'default', targetProfile: 'default' })
  expect(routesOf('groups.succession.status')).toContain('vps')

  // Paused: Send keeps the message instead of trying the unreachable host.
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Are you still there?' } })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  expect(await screen.findByText('A message you send now will be delivered when the group resumes.')).toBeTruthy()
  await waitFor(() => expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy())
  expect(calls('groups.send')).toHaveLength(0)
  expect(screen.queryByRole('button', { name: 'Stop' })).toBeNull()
})

it('continues on the chosen computer through its own connection, follows the group there and delivers the held message', async () => {
  const offline = await hostThatGoesOffline()

  const transition = { seq: 7, event_id: 'moved', room_id: binding.roomId, kind: 'authority.transition', actor: { kind: 'system', id: 'room-driver' },
    payload: { text: 'This group now continues on install:b (with the operator’s attestation).', to_name: 'Home VPS', from_name: 'Mac mini',
      offline_since: Math.floor(new Date().setHours(9, 30, 0, 0) / 1000), successor_gateway_id: VPS } }

  let hosting = false
  handlers.vps = computer(VPS, {
    'groups.succession.status': () => hosting ? status({ host: { install_id: VPS, name: 'Home VPS', reachable: true, since: null },
      this_install: { install_id: VPS, name: 'Home VPS', role: 'host' }, previous_host: { install_id: MINI, name: 'Mac mini', offline_since: 1_700_000_000 },
      unavailable_bots: [{ member_id: 'mira', name: 'Mira Bot' }], backups: [backup(LAPTOP, 'Laptop')] }) : offlineStatus(),
    'groups.succession.prepare': (_method, params) => ({ preview_id: 'preview-1', target: { install_id: params.target_install_id, name: 'Home VPS', operator_name: 'Dana' },
      owner: { name: 'Dana' }, behind_by: 2, at_risk: { count: 2 }, work: { completed: 2, elsewhere: 1, unknown: 1, waiting_for_host: 1 },
      unavailable_bots: [{ member_id: 'mira', name: 'Mira Bot' }], cautions: [{ code: 'host_may_be_running' }] }),
    'groups.succession.promote': () => {
      hosting = true

      return status({ host: { install_id: VPS, name: 'Home VPS', reachable: true, since: null }, this_install: { install_id: VPS, name: 'Home VPS', role: 'host' },
        work: { completed: 2, elsewhere: 1, unknown: 1, waiting_for_host: 1 }, previous_host: { install_id: MINI, name: 'Mac mini', offline_since: 1_700_000_000 } })
    },
    'groups.state': () => roomState({ pending_actions: [{ kind: 'discard', member_id: 'atlas', task_id: 'task-old', execution_generation: 2 }],
      tasks: [{ task_id: 'task-file', member_id: 'mira', state: 'waiting_for_host', resource: 'file', host_name: null }] }),
    'groups.log': () => ({ events: [transition] }),
    'groups.send': (_method, params) => ({ accepted: true, client_event_id: params.event_id })
  })
  offline()

  fireEvent.change(await screen.findByRole('textbox'), { target: { value: 'Held while paused' } })
  await screen.findByRole('button', { name: 'Continue on Home VPS' }, { timeout: 5000 })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await screen.findByRole('button', { name: 'Retry' })
  fireEvent.click(screen.getByRole('button', { name: 'Continue on Home VPS' }))

  const dialog = within(await screen.findByRole('dialog'))
  expect(dialog.getByText('Continue this group on Home VPS?')).toBeTruthy()
  expect(dialog.getByText('Home VPS becomes the group’s host. The conversation, members and history stay the same.')).toBeTruthy()
  expect(dialog.getByText('1 Bot runs on Mac mini and stays unavailable until the group moves back to Mac mini: Mira Bot.')).toBeTruthy()
  expect(dialog.getByText('Work in progress: 2 finished, 1 still running on other computers, 1 unknown. Unknown work won’t run again automatically.')).toBeTruthy()
  expect(dialog.getByText('Home VPS is missing 2 recent messages. They’ll appear if Mac mini comes back.')).toBeTruthy()
  expect(dialog.getByText('Home VPS is catching up 2 messages from another computer.')).toBeTruthy()
  expect(dialog.getByText(/^Only continue if Mac mini is really offline\./)).toBeTruthy()
  expect(dialog.queryByText(/will manage this group/)).toBeNull()
  expect(dialog.queryByRole('button', { name: 'Cancel' })).toBeTruthy()
  expect(calls('groups.succession.prepare').map(call => [call[0].connectionId, call[0].profile, call[2].target_install_id])).toEqual([['vps', 'default', VPS]])

  await act(async () => {fireEvent.click(dialog.getByRole('button', { name: 'Continue on Home VPS' }))})
  expect(calls('groups.succession.promote').map(call => [call[0].connectionId, call[2]]))
    .toEqual([['vps', { room_id: binding.roomId, target_install_id: VPS, preview_id: 'preview-1', confirm: true, profile: 'default' }]])
  await waitFor(() => expect(notify).toHaveBeenCalledWith({ kind: 'success', message: 'Continued on Home VPS. 1 task unknown, 1 waiting for Mac mini.' }))

  // The view follows the group: the same room, read from its new host, with the held message delivered once.
  const history = within(screen.getByRole('log'))
  expect(await history.findByText(/^This group now continues on Home VPS\. Mac mini went offline at /, {}, { timeout: 5000 })).toBeTruthy()
  await waitFor(() => expect(calls('groups.send')).toHaveLength(1), { timeout: 5000 })
  expect(calls('groups.send')[0][0].connectionId).toBe('vps')
  expect(calls('groups.send')[0][2].payload).toMatchObject({ text: 'Held while paused' })
  expect(await screen.findByRole('button', { name: 'Hosted on Home VPS' })).toBeTruthy()
  expect(screen.getByText('Unknown: this may have finished on Mac mini. It won’t run again automatically.')).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Skip this reply' })).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull()
  expect(screen.getByText('Waiting for Mac mini: this needs a file that’s only there.')).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: /2 Bots$/ }))
  expect(await screen.findByText('unavailable while Mac mini is offline', { exact: false })).toBeTruthy()
})

it('explains refusals in words: another computer already continuing, a changed preview, and a caller who is not the owner', async () => {
  const offline = await hostThatGoesOffline()

  let promote: () => unknown = () => {throw refusal('room_authority_promised', { other: { install_id: LAPTOP, name: 'Laptop' } })}
  let prepares = 0
  let moving: unknown = null
  handlers.vps = computer(VPS, {
    'groups.succession.status': () => moving ?? offlineStatus(),
    'groups.succession.prepare': () => {
      prepares++

      return { preview_id: `preview-${prepares}`, target: { install_id: VPS, name: 'Home VPS', operator_name: 'Sam' }, owner: { name: 'Dana' },
        behind_by: 0, at_risk: { count: 0 }, work: null, unavailable_bots: [],
        cautions: [{ code: 'host_may_be_running' }, { code: 'participant_not_fenced', names: ['Laptop'], count: 1 }] }
    },
    'groups.succession.promote': () => promote()
  })
  offline()
  fireEvent.click(await screen.findByRole('button', { name: 'Continue on Home VPS' }, { timeout: 5000 }))
  let dialog = within(await screen.findByRole('dialog'))
  expect(dialog.getByText('Sam will manage this group from Home VPS.')).toBeTruthy()
  expect(dialog.getByText('Laptop runs an older Hermes and may still accept work from Mac mini if it is still running.')).toBeTruthy()
  await act(async () => {fireEvent.click(dialog.getByRole('button', { name: 'Continue on Home VPS' }))})
  expect(await dialog.findByText('Couldn’t continue on Home VPS: Laptop is already continuing this group.')).toBeTruthy()

  promote = () => {throw refusal('preview_stale')}
  await act(async () => {fireEvent.click(dialog.getByRole('button', { name: 'Continue on Home VPS' }))})
  expect(await dialog.findByText('This group changed. Check the details again.')).toBeTruthy()
  expect(prepares).toBe(2)
  promote = () => moving = status({ state: 'moving', this_install: { install_id: VPS, name: 'Home VPS', role: 'backup' },
    moving: { to: { install_id: VPS, name: 'Home VPS' }, step: 'catching_up', started_at: 1_700_000_100 } })
  await act(async () => {fireEvent.click(dialog.getByRole('button', { name: 'Continue on Home VPS' }))})
  expect(calls('groups.succession.promote').at(-1)?.[2].preview_id).toBe('preview-2')
  expect(await screen.findByText('Continuing on Home VPS…')).toBeTruthy()
  expect(screen.getByText('Catching up history')).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'Cancel' })).toBeNull()
  cleanup()

  // Prepare refused because this caller does not own the group here.
  handlers = {}
  localStorage.clear()
  const offlineAgain = await hostThatGoesOffline()
  handlers.vps = computer(VPS, { 'groups.succession.status': () => offlineStatus(), 'groups.succession.prepare': () => {throw refusal('not_owner')} })
  offlineAgain()
  fireEvent.click(await screen.findByRole('button', { name: 'Continue on Home VPS' }, { timeout: 5000 }))
  expect(await screen.findByText('Couldn’t continue on Home VPS: Only Dana can continue this group on another computer.')).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'Try again' })).toBeNull()
  expect(screen.queryByRole('dialog')).toBeNull()
})

it('says who can act when nothing can continue here, and offers a computer Desktop can’t reach only as a hint', async () => {
  const offline = await hostThatGoesOffline()
  handlers.vps = computer(VPS, { 'groups.succession.status': () => offlineStatus({ actions: [{ action: 'continue', targets: [GUEST] }],
    backups: [backup(GUEST, 'Guest box')] }) })
  offline()
  expect(await screen.findByText('Connect to Guest box in Hermes Desktop to continue there.', {}, { timeout: 5000 })).toBeTruthy()
  expect(screen.queryByRole('button', { name: /^Continue on/ })).toBeNull()
  cleanup()

  for (const [reason, text] of [['not_owner', 'Only Dana can continue this group on another computer.'],
    ['no_successor', 'No other computer has a full copy of this group yet. It will resume when Mac mini is back.'],
    ['successor_behind_offline', 'The computers that can continue this group are offline right now. It will resume when Mac mini or one of them is back.']]) {
    handlers = {}
    localStorage.clear()
    activation.epoch++
    const goOffline = await hostThatGoesOffline()
    handlers.vps = computer(VPS, { 'groups.succession.status': () => offlineStatus({ actions: [], unavailable_reason: reason }) })
    goOffline()
    expect(await screen.findByText(text, {}, { timeout: 5000 })).toBeTruthy()
    cleanup()
  }
})

it('re-reads status when the room log carries succession.state, and shows a planned restart without offering a move', async () => {
  let events: unknown[] = []
  let current = status()
  handlers['mac-mini'] = computer(MINI, { 'groups.state': () => roomState(), 'groups.log': () => ({ events }),
    'groups.succession.status': () => current })
  render(<CanonicalGroupWorkspace binding={binding} />)
  await ready()
  await screen.findByRole('button', { name: 'Hosted on Mac mini' })
  const before = calls('groups.succession.status').length
  current = status({ state: 'continued_on_two', conflict: { hosts: [{ install_id: MINI, name: 'Mac mini', since: 1 }, { install_id: VPS, name: 'Home VPS', since: 2 }] },
    actions: [] })
  events = [{ seq: 1, event_id: 'state', kind: 'succession.state', actor: { kind: 'system' }, payload: { state: 'continued_on_two' } }]
  expect(await screen.findByText('This group was continued on two computers', {}, { timeout: 5000 })).toBeTruthy()
  expect(calls('groups.succession.status').length).toBeGreaterThan(before)
  expect(screen.getByText('Waiting for Dana to choose which computer keeps the group.')).toBeTruthy()
  expect(within(screen.getByRole('log')).queryByText('succession.state')).toBeNull()
  cleanup()

  handlers = {}
  localStorage.clear()
  const offline = await hostThatGoesOffline()
  handlers.vps = computer(VPS, { 'groups.succession.status': () => offlineStatus({ state: 'host_restarting' }) })
  offline()
  expect(await screen.findByText('Mac mini is restarting. The group will continue in a moment.', {}, { timeout: 5000 })).toBeTruthy()
  expect(screen.queryByRole('button', { name: /^Continue on/ })).toBeNull()
  expect(screen.queryByRole('button', { name: 'Other computers…' })).toBeNull()
})

it('lets the owner keep one of two computers after confirming, on the computer that answered', async () => {
  handlers['mac-mini'] = computer(MINI, { 'groups.state': () => roomState(), 'groups.log': () => ({ events: [] }),
    'groups.succession.status': () => status({ state: 'continued_on_two', conflict: { hosts: [{ install_id: MINI, name: 'Mac mini', since: 1 },
      { install_id: VPS, name: 'Home VPS', since: 2 }] }, actions: [{ action: 'keep', targets: [MINI, VPS] }] }),
    'groups.succession.keep': () => ({}) })
  render(<CanonicalGroupWorkspace binding={binding} />)
  expect(await screen.findByText('Mac mini and Home VPS both continued the group while they couldn’t reach each other. Choose which one to keep. Messages from the other are kept and shown separately.')).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Keep Home VPS' }))
  const dialog = within(await screen.findByRole('dialog'))
  expect(dialog.getByText('Keep Home VPS?')).toBeTruthy()
  expect(dialog.getByText('Messages from Mac mini are kept and shown separately.')).toBeTruthy()
  await act(async () => {fireEvent.click(dialog.getByRole('button', { name: 'Keep Home VPS' }))})
  await waitFor(() => expect(calls('groups.succession.keep').map(call => [call[0].connectionId, call[2].install_id])).toEqual([['mac-mini', VPS]]))
})

it('shows the old host after it returns: separate messages on request and Open on the new host', async () => {
  handlers['mac-mini'] = computer(MINI, { 'groups.state': () => roomState(), 'groups.log': () => ({ events: [] }),
    'groups.succession.status': () => status({ state: 'moved_away', this_install: { install_id: MINI, name: 'Mac mini', role: 'backup' },
      host: { install_id: VPS, name: 'Home VPS', reachable: true, since: null },
      moved: { to: { install_id: VPS, name: 'Home VPS' }, at: 1_700_000_000, separate_events: 2, branch_id: 'branch-1' },
      actions: [{ action: 'open_on', target: VPS }] }),
    'groups.succession.branch_log': () => ({ events: [{ seq: 1, event_id: 'aside', room_id: binding.roomId, kind: 'message.user',
      actor: { kind: 'user', id: 'desktop' }, payload: { text: 'Written while offline' } }], has_more: false }) })
  handlers.vps = computer(VPS, { 'groups.state': () => roomState(), 'groups.log': () => ({ events: [] }),
    'groups.succession.status': () => status({ host: { install_id: VPS, name: 'Home VPS', reachable: true, since: null },
      this_install: { install_id: VPS, name: 'Home VPS', role: 'host' } }) })
  render(<CanonicalGroupWorkspace binding={binding} />)
  expect(await screen.findByText('This group moved to Home VPS')).toBeTruthy()
  expect(screen.getByText('It moved while Mac mini was offline. 2 messages from that time weren’t part of the group; you can still read them.')).toBeTruthy()
  expect(screen.queryByRole('textbox')).toBeNull()
  expect(screen.getByText('This computer now keeps a backup copy. Open the group on Home VPS to keep chatting.')).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Show them' }))
  expect(await within(screen.getByRole('region', { name: 'Messages kept separately' })).findByText('Written while offline')).toBeTruthy()
  expect(calls('groups.succession.branch_log')[0][2]).toEqual({ room_id: binding.roomId, branch_id: 'branch-1', after_seq: 0, limit: 100, profile: 'default' })
  await waitFor(() => expect(screen.getByRole('button', { name: 'Open on Home VPS' })).toBeTruthy())
  fireEvent.click(screen.getByRole('button', { name: 'Open on Home VPS' }))
  await waitFor(() => expect(routesOf('groups.state')).toContain('vps'))
  expect(await screen.findByRole('textbox')).toBeTruthy()
})

it('switches your own computer on in both places, and leaves someone else’s computer to its owner', async () => {
  let current = status({ backups: [backup(LAPTOP, 'Laptop', { allowed: false, successor: false }),
    backup(GUEST, 'Guest box', { allowed: false, successor: false, operator_name: 'Sam' }), backup(VPS, 'Home VPS', { kind: 'backup' })],
  actions: [{ action: 'designate', targets: [LAPTOP, GUEST, VPS] }, { action: 'remove_backup', targets: [VPS] }, { action: 'add_backup' }] })

  handlers['mac-mini'] = computer(MINI, { 'groups.state': () => roomState(), 'groups.log': () => ({ events: [] }),
    'groups.succession.status': () => current,
    'groups.custody.designate': (_method, params) => ({ room_id: params.room_id, install_id: params.install_id, successor: params.successor, configuration_seq: 4 }),
    'groups.custody.remove': (_method, params) => ({ room_id: params.room_id, install_id: params.install_id, configuration_seq: 5 }) })
  handlers.laptop = computer(LAPTOP, { 'groups.custody.allow': (_method, params) => ({ room_id: params.room_id, install_id: LAPTOP, allowed: true, confirmed: false }) })
  const addBackup = vi.fn(async () => ({ ok: true, install_id: `install:${hex('d')}` }))
  window.hermesDesktop = { roomSetup: { create: vi.fn(), recover: vi.fn(), addBackup } } as unknown as typeof window.hermesDesktop
  render(<CanonicalGroupWorkspace binding={binding} />)
  fireEvent.click(await screen.findByRole('button', { name: 'Hosted on Mac mini' }))
  const rows = await screen.findAllByRole('switch', { name: 'Can continue this group' })
  expect(rows.map(row => (row as HTMLButtonElement).disabled)).toEqual([false, true, false])
  expect(screen.getByText('Sam hasn’t allowed this on Guest box.')).toBeTruthy()

  fireEvent.click(rows[0])
  expect(await screen.findByText('Updating…')).toBeTruthy()
  await waitFor(() => expect(calls('groups.custody.designate')).toHaveLength(1))
  expect(calls('groups.custody.allow').map(call => [call[0].connectionId, call[0].profile, call[2]]))
    .toEqual([['laptop', 'default', { room_id: binding.roomId, successor: true, profile: 'default' }]])
  expect(calls('groups.custody.designate').map(call => [call[0].connectionId, call[2].install_id, call[2].successor])).toEqual([['mac-mini', LAPTOP, true]])
  const now = Date.now()
  vi.spyOn(Date, 'now').mockReturnValue(now + 31_000)
  fireEvent.click(screen.getByRole('button', { name: 'Hosted on Mac mini' }))
  fireEvent.click(screen.getByRole('button', { name: 'Hosted on Mac mini' }))
  expect(await screen.findByText('Waiting for Mac mini to confirm.')).toBeTruthy()
  vi.mocked(Date.now).mockRestore()

  current = { ...current, backups: current.backups.map(entry => entry.install_id === LAPTOP ? { ...entry, allowed: true, designated: true, successor: true } : entry) }
  fireEvent.click(screen.getByRole('button', { name: 'Stop keeping a copy on Home VPS' }))
  const confirm = within(await screen.findByRole('dialog'))
  expect(confirm.getByText('Stop keeping a copy on Home VPS?')).toBeTruthy()
  expect(confirm.getByText('Its copy will be deleted.')).toBeTruthy()
  await act(async () => {fireEvent.click(confirm.getByRole('button', { name: 'Stop keeping a copy' }))})
  await waitFor(() => expect(calls('groups.custody.remove').map(call => [call[0].connectionId, call[2].install_id])).toEqual([['mac-mini', VPS]]))
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())

  fireEvent.click(screen.getByRole('button', { name: 'Hosted on Mac mini' }))
  fireEvent.click(await screen.findByRole('button', { name: 'Add a backup computer…' }))
  const add = within(await screen.findByRole('dialog'))
  expect(add.getByText('Keeps a full copy of this group and can continue it if Mac mini goes offline. It doesn’t add a Bot to the group.')).toBeTruthy()
  expect(add.queryByText('Laptop')).toBeNull()
  await act(async () => {fireEvent.click(add.getByRole('button', { name: 'Add: Work box' }))})
  expect(addBackup).toHaveBeenCalledWith({ home: { connectionId: 'mac-mini', profile: 'default' }, roomId: binding.roomId,
    backup: { connectionId: 'unrelated', profile: 'default' }, successor: true })
  // Only the computer being added is asked who runs it, and only that.
  expect(request.mock.calls.filter(call => call[0].connectionId === 'unrelated').map(call => call[1])).toEqual(['groups.capabilities'])
})

it('offers the gentle prompt to the owner when nothing can continue the group', async () => {
  handlers['mac-mini'] = computer(MINI, { 'groups.state': () => roomState(), 'groups.log': () => ({ events: [] }),
    'groups.succession.status': () => status({ backups: [backup(GUEST, 'Guest box', { allowed: false, successor: false })],
      actions: [{ action: 'designate', targets: [GUEST] }, { action: 'add_backup' }] }) })
  window.hermesDesktop = { roomSetup: { create: vi.fn(), recover: vi.fn(), addBackup: vi.fn() } } as unknown as typeof window.hermesDesktop
  render(<CanonicalGroupWorkspace binding={binding} />)
  fireEvent.click(await screen.findByRole('button', { name: 'Hosted on Mac mini' }))
  expect(await screen.findByText('No other computer can continue this group if Mac mini goes offline. Add a backup computer or allow one of the computers above.')).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Add a backup computer…' })).toBeTruthy()
})
