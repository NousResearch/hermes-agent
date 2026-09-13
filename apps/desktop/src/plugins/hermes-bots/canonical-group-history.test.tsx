import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import type { ComponentProps, ReactNode } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { expectDownloaded, observeDownloads } from './canonical-download-test-utils'

const request = vi.hoisted(() => vi.fn())
vi.mock('@hermes/plugin-sdk', async () => {
  const { en } = await import('@/i18n/en')

  return { host: { requestProfile: request }, useI18n: () => ({ locale: 'en', t: en }),
    Button: (props: ComponentProps<'button'>) => <button {...props} />,
    Codicon: () => <span />, Tip: ({ children }: { children: ReactNode }) => <>{children}</> }
})
vi.mock('./canonical-group-labels', async () => {
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')

  return { useCanonicalGroupLabels: () => ({ ...CANONICAL_GROUP_LOCALES.en, back: 'Back', refresh: 'Refresh', retry: 'Retry',
    send: 'Send', stop: 'Stop', download: 'Download', discard: 'Discard', cancel: 'Cancel' }) }
})

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

it('binds user/member history downloads to their real event and refuses missing or foreign room identity', async () => {
  const observed = observeDownloads()
  const save = vi.fn().mockResolvedValue(undefined)
  window.hermesDesktop = { saveImageBuffer: save } as unknown as typeof window.hermesDesktop

  const events = [
    { seq: 1, room_id: binding.roomId, event_id: 'user-event', kind: 'message.user' },
    { seq: 2, room_id: binding.roomId, event_id: 'member-event', kind: 'message.member', actor: { member_id: 'helper' } },
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
