import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ComponentProps } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

const request = vi.hoisted(() => vi.fn())
vi.mock('@hermes/plugin-sdk', () => ({
  host: { requestProfile: request },
  Button: (props: ComponentProps<'button'>) => <button {...props} />
}))
vi.mock('./canonical-group-labels', () => ({
  useCanonicalGroupLabels: () => ({ attachFiles: 'Attach files', download: 'Download',
    removeAttachment: 'Remove', uploadFailed: 'Attachment failed' })
}))

import { CanonicalGroupAttachments } from './canonical-group-attachments'

const binding = { connectionId: 'remote-owner', profile: 'reviewer', roomId: 'room' }
const originalDesktop = window.hermesDesktop
afterEach(() => { cleanup(); request.mockReset(); window.hermesDesktop = originalDesktop })

it.each([32, 1_048_576])('uploads all %i bytes using the canonical owner method', async size => {
  const bytes = Uint8Array.from({ length: size }, (_, index) => index % 256)
  const file = new File([bytes], 'report.bin', { type: 'application/octet-stream' })
  // jsdom lacks Blob.arrayBuffer; preserve its real FileReader implementation.
  Object.defineProperty(file, 'arrayBuffer', { value: async () => bytes.buffer })
  const uploaded = { attachment_id: 'file-one', kind: 'file', name: file.name, mime: file.type, size }
  request.mockImplementation(async (route, method, params) => {
    expect(route).toMatchObject({ connectionId: binding.connectionId, targetProfile: binding.profile })
    if (method !== 'groups.attachment.upload') {throw new Error('Unknown gateway method')}
    expect(params).toMatchObject({ profile: binding.profile, room_id: binding.roomId, name: file.name })
    const decoded = Uint8Array.from(atob(params.data_base64), value => value.charCodeAt(0))
    expect(Buffer.from(decoded).equals(Buffer.from(bytes))).toBe(true)

    return { ...uploaded, sha256: 'receipt-digest', state: 'uploaded', created_at: 123,
      idempotent: false, room_id: binding.roomId, authority: { gateway_id: 'owner', epoch: 1 } }
  })
  const changed = vi.fn()
  const submit = vi.fn(event => event.preventDefault())
  const view = render(<form onSubmit={submit}><CanonicalGroupAttachments attachments={[]} binding={binding} disabled={false} onChange={changed} /></form>)
  fireEvent.click(screen.getByRole('button', { name: 'Attach files' }))
  fireEvent.change(view.container.querySelector('input')!, { target: { files: [file] } })
  await waitFor(() => expect(changed).toHaveBeenCalledWith([uploaded]), { timeout: 2000 })
  expect(request).toHaveBeenCalledTimes(1)
  expect(submit).not.toHaveBeenCalled()
})

it('downloads only the selected committed attachment from its captured owner', async () => {
  const attachment = { attachment_id: 'file-one', event_id: 'event-one', kind: 'file', name: 'report.bin', mime: 'application/octet-stream' }
  const save = vi.fn().mockResolvedValue(undefined)
  window.hermesDesktop = { saveImageBuffer: save } as unknown as typeof window.hermesDesktop
  request.mockImplementation(async (route, method, params) => {
    expect(route).toMatchObject({ connectionId: binding.connectionId, targetProfile: binding.profile })
    if (method !== 'groups.attachment.download') {throw new Error('Unknown gateway method')}
    expect(params).toEqual({ profile: binding.profile, room_id: binding.roomId,
      event_id: attachment.event_id, attachment_id: attachment.attachment_id })

    return { ...attachment, data_base64: 'AAEC/w==' }
  })
  const submit = vi.fn(event => event.preventDefault())
  const changed = vi.fn()
  render(<form onSubmit={submit}><CanonicalGroupAttachments attachments={[attachment]} binding={binding} disabled={false} onChange={changed} /></form>)
  fireEvent.click(screen.getByRole('button', { name: 'Download' }))
  await waitFor(() => expect(save).toHaveBeenCalledWith(new Uint8Array([0, 1, 2, 255]), '.bin', attachment.name))
  expect(request).toHaveBeenCalledTimes(1)
  await waitFor(() => expect((screen.getByRole('button', { name: 'Remove' }) as HTMLButtonElement).disabled).toBe(false))
  fireEvent.click(screen.getByRole('button', { name: 'Remove' }))
  expect(changed).toHaveBeenCalledWith([])
  expect(submit).not.toHaveBeenCalled()
})
