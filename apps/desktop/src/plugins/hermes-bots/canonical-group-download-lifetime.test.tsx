import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
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

  return { useCanonicalGroupLabels: () => ({ ...CANONICAL_GROUP_LOCALES.en, refresh: 'Refresh', retry: 'Retry',
    send: 'Send', stop: 'Stop', download: 'Download', discard: 'Discard', cancel: 'Cancel' }) }
})
import { CanonicalGroupWorkspace } from './canonical-group-workspace'

const binding = { connectionId: 'owner-a', profile: 'reviewer', roomId: 'room-one' }
const attachment = { attachment_id: 'att_00000000000000000000000000000001', kind: 'file', name: 'retained.txt', mime: 'text/plain', size: 1 }
const originalDesktop = window.hermesDesktop
afterEach(() => { cleanup(); request.mockReset(); vi.restoreAllMocks(); vi.unstubAllGlobals(); localStorage.clear(); window.hermesDesktop = originalDesktop })

function setup() {
  const observed = observeDownloads()
  const cache = vi.fn().mockResolvedValue('/private/composer-images/not-a-user-download')
  window.hermesDesktop = { saveImageBuffer: cache } as unknown as typeof window.hermesDesktop
  let release!: (value: unknown) => void
  const held = new Promise(resolve => { release = resolve })
  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.state') {return { room: { name: 'Room' } }}

    if (method === 'groups.log') {return { events: [{ seq: 1, room_id: binding.roomId, event_id: 'committed',
      kind: 'message.member', payload: { attachments: [attachment] } }] }}

    if (method === 'groups.attachment.download') {return held}
    throw new Error(`Unexpected method ${method}`)
  })

  return { observed, cache, release: () => release({ ...attachment, event_id: 'committed', data_base64: 'QQ==' }) }
}

it('initiates the real user-facing download with exact bytes/name, never the composer cache', async () => {
  const { observed, cache, release } = setup()
  render(<CanonicalGroupWorkspace binding={binding} />)
  const history = within(screen.getByRole('log'))
  fireEvent.click(await history.findByRole('button', { name: 'Download' }))
  await act(async () => release())
  await waitFor(() => expect(observed.downloads).toHaveLength(1), { timeout: 2000 })
  await expectDownloaded(observed, new Uint8Array([65]), attachment.name, attachment.mime)
  expect(cache).not.toHaveBeenCalled()
  expect(request.mock.calls.find(call => call[1] === 'groups.attachment.download')?.[2]).toEqual({
    profile: binding.profile, room_id: binding.roomId, event_id: 'committed', attachment_id: attachment.attachment_id
  })
})

it.each(['unmount', 'hidden', 'profile', 'connection', 'room'])('does not initiate a late download or cache write after %s', async change => {
  const { observed, cache, release } = setup()
  const view = render(<CanonicalGroupWorkspace binding={binding} />)
  fireEvent.click(await within(screen.getByRole('log')).findByRole('button', { name: 'Download' }))
  await waitFor(() => expect(request.mock.calls.some(call => call[1] === 'groups.attachment.download')).toBe(true))

  if (change === 'unmount') {view.unmount()}
  else if (change === 'hidden') {view.rerender(<CanonicalGroupWorkspace binding={binding} visible={false} />)}
  else {
    const key = change === 'connection' ? 'connectionId' : change === 'room' ? 'roomId' : 'profile'
    view.rerender(<CanonicalGroupWorkspace binding={{ ...binding, [key]: 'other' }} />)
  }

  await act(async () => release())
  expect(observed.create).not.toHaveBeenCalled()
  expect(observed.downloads).toHaveLength(0)
  expect(cache).not.toHaveBeenCalled()
})
