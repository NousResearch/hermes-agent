import type * as HermesSdk from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { expectDownloaded, observeDownloads } from './canonical-download-test-utils'

const request = vi.hoisted(() => vi.fn())
vi.mock('@hermes/plugin-sdk', async () => {
  const sdk = await vi.importActual<typeof HermesSdk>('@hermes/plugin-sdk')
  const { en } = await import('@/i18n/en')
  const { captureGroupRequests } = await import('./group-test-utils')

  return { ...sdk, host: { requestProfile: captureGroupRequests(request).request }, useI18n: () => ({ locale: 'en', t: en }) }
})
vi.mock('./canonical-group-labels', async () => {
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')

  return { useCanonicalGroupLabels: () => ({ ...CANONICAL_GROUP_LOCALES.en, back: 'Back', refresh: 'Refresh', retry: 'Retry',
    send: 'Send', stop: 'Stop', download: 'Download', discard: 'Discard', cancel: 'Cancel' }) }
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

 return { accepted: true }}

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

it('blocks Send during a chosen file upload, keeps Stop available, and recovers from a failed upload', async () => {
  let rejectUpload!: (error: Error) => void
  let releaseUpload!: (value: unknown) => void
  const heldUpload = new Promise((resolve, reject) => { releaseUpload = resolve; rejectUpload = reject })
  let uploadResult: Promise<unknown> = heldUpload
  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.state') {return { room: { name: 'Autumn launch' }, driver_status: {} }}

    if (method === 'groups.log') {return { events: [] }}

    if (method === 'groups.attachment.upload') {return uploadResult}

    if (method === 'groups.send') {return { accepted: true }}

    return {}
  })
  const view = render(<CanonicalGroupWorkspace binding={binding} />)
  const input = screen.getByRole('textbox') as HTMLTextAreaElement
  await waitFor(() => expect(input.disabled).toBe(false))
  fireEvent.change(input, { target: { value: 'Please review this file' } })

  const chooseFile = () => fireEvent.change(view.container.querySelector('input[type=file]')!, {
    target: { files: [new File(['A'], manifest.name, { type: manifest.mime })] }
  })

  chooseFile()
  await waitFor(() => expect(request.mock.calls.some(call => call[1] === 'groups.attachment.upload')).toBe(true))
  expect((screen.getByRole('button', { name: 'Send' }) as HTMLButtonElement).disabled).toBe(true)
  expect((screen.getByRole('button', { name: 'Stop' }) as HTMLButtonElement).disabled).toBe(false)
  fireEvent.keyDown(input, { key: 'Enter' })
  fireEvent.submit(view.container.querySelector('form')!)
  expect(request.mock.calls.some(call => call[1] === 'groups.send')).toBe(false)
  await act(async () => rejectUpload(new Error('Upload interrupted')))
  await screen.findByText(/Upload interrupted/)
  expect(input.value).toBe('Please review this file')
  expect((screen.getByRole('button', { name: 'Send' }) as HTMLButtonElement).disabled).toBe(false)
  uploadResult = new Promise(resolve => { releaseUpload = resolve })
  chooseFile()
  await waitFor(() => expect(request.mock.calls.filter(call => call[1] === 'groups.attachment.upload')).toHaveLength(2))
  await act(async () => releaseUpload(manifest))
  await screen.findByText(manifest.name)
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await waitFor(() => expect(request.mock.calls.find(call => call[1] === 'groups.send')?.[2].payload)
    .toMatchObject({ text: 'Please review this file', attachments: [manifest] }))
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

it('attributes rich messages to authoritative friendly members and people without exposing route IDs', async () => {
  const events = [
    { seq: 1, event_id: 'user', kind: 'message.user', actor: { kind: 'user', id: 'desktop' }, payload: { text: 'hello' } },
    { seq: 2, event_id: 'named', kind: 'message.member', payload: { text: 'hi' },
      actor: { kind: 'member', id: 'm-helper', profile: 'helper', display_name: 'Atlas Bot' } },
    { seq: 3, event_id: 'unnamed', kind: 'message.member', actor: { kind: 'member', id: 'm-critic', profile: 'critic' },
      payload: { text: '**Ready** to review\n\n- First item\n- Second item\n\n`example`\n\nMEDIA:/owner/private.png' } },
    { seq: 5, event_id: 'person', kind: 'message.user', actor: { kind: 'user', id: 'other-person', display_name: 'Alex' }, payload: { text: 'Thanks' } },
    { seq: 4, event_id: 'settled', kind: 'turn.settled', actor: { kind: 'gateway', id: 'gw-1' }, payload: {} }
  ]

  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.state') {return { room: { name: 'Room', members: [
      { member_id: 'm-helper', profile: 'helper', handle: 'helper-route', display_name: 'Atlas Bot renamed' },
      { member_id: 'm-critic', profile: 'critic', handle: 'critic-route', display_name: 'Mira Bot' }
    ] } }}

    if (method === 'groups.log') {return { events }}
    throw new Error(`Unexpected method ${method}`)
  })
  render(<CanonicalGroupWorkspace binding={binding} />)
  const history = within(screen.getByRole('log'))
  await waitFor(() => expect(history.getByText('Atlas Bot')).toBeTruthy())
  expect(history.getByText('Mira Bot')).toBeTruthy()
  expect(history.getByText('You')).toBeTruthy()
  expect(history.getByText('Alex')).toBeTruthy()
  expect(history.getByText('Ready').closest('[data-streamdown="strong"]')).toBeTruthy()
  expect(history.getByRole('list')).toBeTruthy()
  expect(history.getByText('example').closest('code')).toBeTruthy()
  expect(history.getByText(/MEDIA:\/owner\/private.png/)).toBeTruthy()
  expect(screen.getByRole('log').querySelector('img')).toBeNull()
  expect(history.queryByText('m-critic')).toBeNull()
  expect(history.queryByText('helper-route')).toBeNull()
  expect(history.queryByText('Atlas Bot renamed')).toBeNull()
  expect(history.queryByText('desktop')).toBeNull()
  expect(history.queryByText('gw-1')).toBeNull()
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
  expect(history.getByText(CANONICAL_GROUP_LOCALES.en.activityUpdated)).toBeTruthy()
  expect(history.getByText('room.future_kind').closest('details')?.open).toBe(false)
  expect(history.queryByText('turn.settled')).toBeNull()
  expect(history.queryByText('room.activity')).toBeNull()
})

it('explains actual failed, deferred and stopped replies while keeping technical details optional', async () => {
  const events = [
    { seq: 1, event_id: 'failure', kind: 'turn.failed', actor: { kind: 'gateway' }, payload: { member_id: 'mira', error: 'tool_process_exit_1' } },
    { seq: 2, event_id: 'deferred', kind: 'turn.deferred', actor: { kind: 'gateway' }, payload: { member_id: 'mira', reason: 'approval_pending' } },
    { seq: 3, event_id: 'stop', kind: 'room.stop_requested', actor: { kind: 'gateway' }, payload: {} }
  ]

  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.state') {return { room: { name: 'Autumn launch', members: [{ member_id: 'mira', profile: 'default', handle: 'peer-mira', display_name: 'Mira Bot' }] } }}

    if (method === 'groups.log') {return { events }}

    return {}
  })
  render(<CanonicalGroupWorkspace binding={binding} />)
  const history = within(screen.getByRole('log'))
  await history.findByText(CANONICAL_GROUP_LOCALES.en.activityFailed.replace('{name}', 'Mira Bot'))
  expect(history.getByText(CANONICAL_GROUP_LOCALES.en.activityDeferred.replace('{name}', 'Mira Bot'))).toBeTruthy()
  expect(history.getByText(CANONICAL_GROUP_LOCALES.en.stopped)).toBeTruthy()
  expect(history.getByText('tool_process_exit_1').closest('details')?.open).toBe(false)
  expect(history.getByText('approval_pending').closest('details')?.open).toBe(false)
  expect(request.mock.calls.every(call => ['groups.state', 'groups.log'].includes(call[1]))).toBe(true)
})
