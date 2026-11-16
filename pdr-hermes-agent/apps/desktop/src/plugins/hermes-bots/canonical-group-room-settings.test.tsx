import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ComponentProps, ReactNode } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

const request = vi.hoisted(() => vi.fn())

vi.mock('@hermes/plugin-sdk', () => ({
  Button: (props: ComponentProps<'button'>) => <button {...props} />,
  Dialog: ({ children, open }: { children: ReactNode; open: boolean }) => (open ? <div>{children}</div> : null),
  DialogContent: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  DialogFooter: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  DialogHeader: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  DialogTitle: ({ children }: { children: ReactNode }) => <h2>{children}</h2>,
  host: {},
  Input: (props: ComponentProps<'input'>) => <input {...props} />,
  randomUuid: () => 'uuid-1',
  Textarea: (props: ComponentProps<'textarea'>) => <textarea {...props} />
}))
vi.mock('./canonical-groups', async importOriginal => ({ ...(await importOriginal<object>()), canonicalGroupRequest: request }))
const CONTROLS = vi.hoisted(() => ({ cancel: 'Cancel', save: 'Save' }))

vi.mock('./canonical-group-labels', async () => {
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')

  return { useCanonicalGroupLabels: () => ({ ...CANONICAL_GROUP_LOCALES.en, ...CONTROLS }) }
})

import { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'
import { CanonicalGroupRoomSettings } from './canonical-group-room-settings'

const labels = { ...CANONICAL_GROUP_LOCALES.en, ...CONTROLS }
const binding = { roomId: 'room-1', profile: 'default', connectionId: 'local' }

afterEach(() => {
  cleanup()
  request.mockReset()
})

function openSettings() {
  render(<CanonicalGroupRoomSettings binding={binding} context={{ purpose: 'Old purpose', notes: 'Old notes' }} />)
  fireEvent.click(screen.getByRole('button', { name: labels.roomSettings }))

  return screen.getByPlaceholderText(labels.notesPlaceholder) as HTMLTextAreaElement
}

it('shows the purpose and the copyable cron target for this room', () => {
  openSettings()
  expect(screen.getByText('Old purpose')).toBeTruthy()
  expect(screen.getByText('room:room-1@default')).toBeTruthy()
})

it('keeps notes the backend rejected so they can be shortened and saved again', async () => {
  request.mockRejectedValue(new Error('notes exceed 4096 bytes'))
  const notes = openSettings()

  fireEvent.change(notes, { target: { value: 'My long draft' } })
  fireEvent.click(screen.getByText(labels.save))

  await waitFor(() => expect(request).toHaveBeenCalledWith(binding, 'groups.context', { room_id: 'room-1', event_id: 'uuid-1', purpose: 'Old purpose', notes: 'My long draft' }))
  expect(await screen.findByRole('alert')).toBeTruthy()
  expect((screen.getByPlaceholderText(labels.notesPlaceholder) as HTMLTextAreaElement).value).toBe('My long draft')
})

it('an in-flight save cannot be overtaken by edits it would then discard', async () => {
  let fail: (error: Error) => void = () => {}
  request.mockImplementation(() => new Promise((_, reject) => { fail = reject }))
  const notes = openSettings()

  fireEvent.change(notes, { target: { value: 'First draft' } })
  fireEvent.click(screen.getByText(labels.save))
  await waitFor(() => expect(request).toHaveBeenCalledTimes(1))

  // While the save is in flight the draft is frozen: nothing typed now can be lost when it lands.
  expect(notes.disabled).toBe(true)
  expect((screen.getByText(labels.save) as HTMLButtonElement).disabled).toBe(true)

  fail(new Error('nope'))
  await waitFor(() => expect(notes.disabled).toBe(false))
  expect(notes.value).toBe('First draft')
})

it('closes and refreshes the room only after the backend accepted the save', async () => {
  request.mockResolvedValue({})
  const onSaved = vi.fn()
  render(<CanonicalGroupRoomSettings binding={binding} context={{ purpose: '', notes: '' }} onSaved={onSaved} />)
  fireEvent.click(screen.getByRole('button', { name: labels.roomSettings }))
  fireEvent.click(screen.getByText(labels.save))

  await waitFor(() => expect(onSaved).toHaveBeenCalledTimes(1))
  expect(screen.queryByPlaceholderText(labels.notesPlaceholder)).toBeNull()
})

it('blocks saving notes over the 4 KB limit while typing', () => {
  const notes = openSettings()

  fireEvent.change(notes, { target: { value: 'é'.repeat(2049) } }) // 4098 UTF-8 bytes in 2049 characters

  expect(screen.getByTestId('room-notes-bytes').textContent).toBe('4098 / 4096')
  expect((screen.getByText(labels.save) as HTMLButtonElement).disabled).toBe(true)
})
