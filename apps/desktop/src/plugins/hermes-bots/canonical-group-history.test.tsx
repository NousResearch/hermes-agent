import type * as HermesSdk from '@hermes/plugin-sdk'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import type { ComponentProps, ReactNode } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { expectDownloaded, observeDownloads } from './canonical-download-test-utils'

const request = vi.hoisted(() => vi.fn())
vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const original = await importOriginal<typeof HermesSdk>()
  const { atom } = await import('nanostores')
  const { en } = await import('@/i18n/en')
  const { captureGroupRequests } = await import('./group-test-utils')

  return { ...original, atom, host: { requestProfile: captureGroupRequests(request).request }, useI18n: () => ({ locale: 'en', t: en }),
    Button: (props: ComponentProps<'button'>) => <button {...props} />,
    Codicon: () => <span />, Tip: ({ children }: { children: ReactNode }) => <>{children}</> }
})
vi.mock('./canonical-group-labels', async () => {
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')

  return { useCanonicalGroupLabels: () => ({ ...CANONICAL_GROUP_LOCALES.en, back: 'Back', refresh: 'Refresh', retry: 'Retry',
    send: 'Send', stop: 'Stop', download: 'Download', discard: 'Discard', cancel: 'Cancel', you: 'You' }) }
})

import { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'
import { CanonicalGroupWorkspace } from './canonical-group-workspace'

const binding = { connectionId: 'original-owner', profile: 'reviewer', roomId: 'room-one' }
const manifest = { attachment_id: 'att_00000000000000000000000000000001', kind: 'file', name: 'notes.txt', mime: 'text/plain', size: 1 }
const originalDesktop = window.hermesDesktop
afterEach(() => { cleanup(); request.mockReset(); vi.restoreAllMocks(); vi.unstubAllGlobals(); localStorage.clear(); window.hermesDesktop = originalDesktop })

it('keeps a committed file downloadable in history after Send clears the composer (F31)', async () => {
  const observed = observeDownloads()
  const save = vi.fn().mockResolvedValue(undefined)
  window.hermesDesktop = { saveImageBuffer: save } as unknown as typeof window.hermesDesktop
  let sent = false
  request.mockImplementation(async (_route, method, params) => {
    if (method === 'groups.state') {return { room: { name: 'Room' }, driver_status: {} }}

    if (method === 'groups.log') {return { events: sent ? [{ seq: 1, room_id: binding.roomId, event_id: 'committed-user-event',
      kind: 'message.user', payload: { text: 'Review these notes', attachments: [manifest] } }] : [] }}

    if (method === 'groups.attachment.upload') {return { ...manifest, sha256: 'receipt-only' }}

    if (method === 'groups.send') {expect(params.payload.attachments).toEqual([manifest]); sent = true;

 return { accepted: true, client_event_id: params.event_id }}

    if (method === 'groups.attachment.download') {return { ...manifest, event_id: params.event_id, data_base64: 'QQ==' }}
    throw new Error(`Unexpected method ${method}`)
  })
  const view = render(<CanonicalGroupWorkspace binding={binding} />)
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false))
  fireEvent.change(view.container.querySelector('input[type=file]')!, { target: { files: [new File(['A'], manifest.name, { type: manifest.mime })] } })
  await waitFor(() => expect(screen.getByText(manifest.name)).toBeTruthy())
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Review these notes' } })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe(''))
  const history = within(screen.getByRole('log'))
  await waitFor(() => expect(history.getByText(manifest.name)).toBeTruthy(), { timeout: 2000 })
  expect(history.queryByRole('button', { name: 'Attach files' })).toBeNull()
  expect(history.queryByRole('button', { name: 'Remove attachment' })).toBeNull()
  expect(view.container.querySelector('form')?.textContent).not.toContain(manifest.name)
  fireEvent.click(history.getByRole('button', { name: 'Download' }))
  await waitFor(() => expect(observed.downloads).toHaveLength(1))
  await expectDownloaded(observed, new Uint8Array([65]), manifest.name, manifest.mime)
  expect(save).not.toHaveBeenCalled()
  const call = request.mock.calls.find(call => call[1] === 'groups.attachment.download')!
  expect(call[0]).toMatchObject({ connectionId: binding.connectionId, targetProfile: binding.profile })
  expect(call[2]).toEqual({ profile: binding.profile, room_id: binding.roomId,
    event_id: 'committed-user-event', attachment_id: manifest.attachment_id })
  expect(request.mock.calls.some(call => call[1] === 'groups.attachment.list')).toBe(false)
})

it('binds user/member history downloads to their real event and refuses missing or foreign room identity', async () => {
  const observed = observeDownloads()
  const save = vi.fn().mockResolvedValue(undefined)
  window.hermesDesktop = { saveImageBuffer: save } as unknown as typeof window.hermesDesktop

  const events = [
    { seq: 1, room_id: binding.roomId, event_id: 'user-event', kind: 'message.user' },
    { seq: 2, room_id: binding.roomId, event_id: 'member-event', kind: 'message.member', actor: { kind: 'member', id: 'helper', profile: 'helper' } },
    { seq: 3, room_id: binding.roomId, kind: 'message.user' },
    { seq: 4, room_id: 'other-room', event_id: 'foreign-event', kind: 'message.member' }
  ].map(event => ({ ...event, payload: { attachments: [{ ...manifest, event_id: 'not-the-event' }] } }))

  request.mockImplementation(async (_route, method, params) => {
    if (method === 'groups.state') {return { room: { name: 'Room' } }}

    if (method === 'groups.log') {return { events }}

    if (method === 'groups.attachment.download') {return { ...manifest, event_id: params.event_id, data_base64: 'QQ==' }}
    throw new Error(`Unexpected method ${method}`)
  })
  render(<CanonicalGroupWorkspace binding={binding} />)
  const history = within(screen.getByRole('log'))
  await waitFor(() => expect(history.getAllByRole('button', { name: 'Download' })).toHaveLength(4))
  const buttons = history.getAllByRole('button', { name: 'Download' }) as HTMLButtonElement[]
  expect(buttons.map(button => button.disabled)).toEqual([false, false, true, true])
  fireEvent.click(buttons[0])
  await waitFor(() => expect(observed.downloads).toHaveLength(1))
  fireEvent.click(buttons[1])
  await waitFor(() => expect(observed.downloads).toHaveLength(2))
  expect(save).not.toHaveBeenCalled()
  const reads = request.mock.calls.filter(call => call[1] === 'groups.attachment.download')
  expect(reads.map(call => call[2].event_id)).toEqual(['user-event', 'member-event'])
  expect(reads.every(call => call[2].room_id === binding.roomId && call[2].profile === binding.profile)).toBe(true)
  expect(history.queryByRole('button', { name: 'Remove attachment' })).toBeNull()
})

it('attributes human and Bot messages to logged identities and leaves gateway events unlabelled', async () => {
  const events = [
    { seq: 1, event_id: 'user', kind: 'message.user', actor: { kind: 'user', id: 'desktop' }, payload: { text: 'hello' } },
    { seq: 2, event_id: 'named', kind: 'message.member', payload: { text: 'hi' },
      actor: { kind: 'member', id: 'm-helper', profile: 'helper', display_name: 'Helper Bot' } },
    { seq: 3, event_id: 'unnamed', kind: 'message.member', actor: { kind: 'member', id: 'm-critic', profile: 'critic' },
      payload: { text: 'noted' } },
    { seq: 4, event_id: 'settled', kind: 'turn.settled', actor: { kind: 'gateway', id: 'gw-1' }, payload: {} },
    { seq: 5, event_id: 'remote-human', kind: 'message.user', actor: { kind: 'user', id: 'telegram:42', display_name: 'Alex' }, payload: { text: 'from Alex' } }
  ]

  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.state') {return { room: { name: 'Room' } }}

    if (method === 'groups.log') {return { events }}
    throw new Error(`Unexpected method ${method}`)
  })
  render(<CanonicalGroupWorkspace binding={binding} />)
  const history = within(screen.getByRole('log'))
  await waitFor(() => expect(history.getByText('Helper Bot')).toBeTruthy())
  expect(history.getByText('m-critic')).toBeTruthy()
  expect(history.queryByText('desktop')).toBeNull()
  expect(history.queryByText('gw-1')).toBeNull()
  expect(history.queryByText('telegram:42')).toBeNull()
  expect(Array.from(screen.getByRole('log').querySelectorAll('strong'), node => node.textContent))
    .toEqual(['You: ', 'Helper Bot: ', 'm-critic: ', 'Alex: '])
})

it('hides empty bookkeeping rows, but keeps unknown kinds and bookkeeping that carries text', async () => {
  const events = [
    { seq: 1, event_id: 'said', kind: 'message.user', actor: { kind: 'user', id: 'desktop' }, payload: { text: 'hello' } },
    { seq: 2, event_id: 'settled', kind: 'turn.settled', actor: { kind: 'gateway', id: 'gw-1' }, payload: {} },
    { seq: 3, event_id: 'activity', kind: 'room.activity', actor: { kind: 'gateway', id: 'gw-1' }, payload: {} },
    { seq: 4, event_id: 'noted', kind: 'turn.settled', actor: { kind: 'gateway', id: 'gw-1' }, payload: { text: 'Stopped by you' } },
    { seq: 5, event_id: 'novel', kind: 'room.future_kind', actor: { kind: 'gateway', id: 'gw-1' }, payload: {} }
  ]

  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.state') {return { room: { name: 'Room' } }}

    if (method === 'groups.log') {return { events }}
    throw new Error(`Unexpected method ${method}`)
  })
  render(<CanonicalGroupWorkspace binding={binding} />)
  const history = within(screen.getByRole('log'))
  await waitFor(() => expect(history.getByText('hello')).toBeTruthy())
  expect(history.getByText('Stopped by you')).toBeTruthy()
  expect(history.getByText('room.future_kind')).toBeTruthy()
  expect(history.queryByText('turn.settled')).toBeNull()
  expect(history.queryByText('room.activity')).toBeNull()
})

it('keeps Layer 7 bookkeeping quiet and words host changes and waiting work from display fields, without new reads', async () => {
  const wentOffline = new Date()
  wentOffline.setHours(9, 30, 0, 0)
  const offlineSince = wentOffline.getTime() / 1000
  const time = new Intl.DateTimeFormat('en', { hour: 'numeric', minute: '2-digit' }).format(wentOffline)
  const english = 'This group now continues on install:0123456789abcdef0123456789abcdef (with the operator’s attestation).'
  const system = { kind: 'system', id: 'room-driver' }

  const events = [
    { seq: 1, event_id: 'said', kind: 'message.user', actor: { kind: 'user', id: 'desktop' }, payload: { text: 'hello' } },
    { seq: 2, event_id: 'admitted', kind: 'task.admitted', actor: system, payload: { task_id: 'task-1', generation: 1 } },
    { seq: 3, event_id: 'custody', kind: 'custody.configured', actor: system, payload: { voters: [{ install_id: 'install:a', role: 'custodian' }] } },
    { seq: 4, event_id: 'state', kind: 'succession.state', actor: system, payload: { state: 'moving' } },
    { seq: 5, event_id: 'moved', kind: 'authority.transition', actor: system,
      payload: { text: english, to_name: 'Home VPS', from_name: 'Mac mini', offline_since: offlineSince } },
    { seq: 6, event_id: 'moved-no-origin', kind: 'authority.transition', actor: system,
      payload: { text: english, to_name: 'Home VPS', from_name: null, offline_since: offlineSince } },
    { seq: 7, event_id: 'moved-unnamed', kind: 'authority.transition', actor: system,
      payload: { text: english, to_name: null, from_name: null, offline_since: null } },
    { seq: 8, event_id: 'moved-older', kind: 'authority.transition', actor: system, payload: { text: 'Older gateway notice.' } },
    { seq: 9, event_id: 'waiting-bot', kind: 'turn.deferred', actor: system,
      payload: { member_id: 'm-atlas', task_id: 'task-2', reason: 'waiting_for_host', resource: 'bot', host_name: 'Mac mini' } },
    { seq: 10, event_id: 'waiting-file', kind: 'turn.deferred', actor: system,
      payload: { member_id: 'm-mira', task_id: 'task-3', reason: 'waiting_for_host', resource: 'file', host_name: null } },
    { seq: 11, event_id: 'other-reason', kind: 'turn.deferred', actor: system, payload: { member_id: 'm-mira', reason: 'approval_pending' } }
  ]

  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.state') {return { room: { name: 'Room' } }}

    if (method === 'groups.log') {return { events }}
    throw new Error(`Unexpected method ${method}`)
  })
  render(<CanonicalGroupWorkspace binding={binding} />)
  const history = within(screen.getByRole('log'))
  await waitFor(() => expect(history.getByText('hello')).toBeTruthy())

  for (const kind of ['task.admitted', 'custody.configured', 'succession.state']) {expect(history.queryByText(kind)).toBeNull()}
  expect(history.getByText(`This group now continues on Home VPS. Mac mini went offline at ${time}.`)).toBeTruthy()
  expect(history.getByText('This group now continues on Home VPS.')).toBeTruthy()
  expect(history.getByText('This group now continues on another computer.')).toBeTruthy()
  expect(history.getByText('Older gateway notice.')).toBeTruthy()
  expect(history.queryByText(english)).toBeNull()
  expect(history.queryByText(/install:/)).toBeNull()
  expect(history.getByText('Waiting for Mac mini: this needs a Bot that’s only there.')).toBeTruthy()
  expect(history.getByText('Waiting for another computer: this needs a file that’s only there.')).toBeTruthy()
  expect(history.getByText('turn.deferred')).toBeTruthy()
  expect(request.mock.calls.every(call => ['groups.state', 'groups.log'].includes(call[1]))).toBe(true)
})

it('keeps every placeholder of the host-change and waiting copy in all nine locales', () => {
  const english = CANONICAL_GROUP_LOCALES.en

  const keys = ['continuedOn', 'continuedOnSince', 'continuedOnUnnamed', 'waitingForHostBot', 'waitingForHostFile',
    'waitingForUnnamedHostBot', 'waitingForUnnamedHostFile'] as const

  const placeholders = (text: string) => (text.match(/\{\w+\}/g) ?? []).sort()

  expect(Object.keys(CANONICAL_GROUP_LOCALES)).toHaveLength(9)

  for (const [locale, messages] of Object.entries(CANONICAL_GROUP_LOCALES)) {
    for (const key of keys) {
      expect(placeholders(messages[key]), `${locale}.${key}`).toEqual(placeholders(english[key]))
    }
  }
})
