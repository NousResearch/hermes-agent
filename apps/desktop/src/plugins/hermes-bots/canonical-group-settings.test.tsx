import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ComponentProps } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

vi.mock('@hermes/plugin-sdk', () => ({
  Button: (props: ComponentProps<'button'>) => <button {...props} />,
  ConfirmDialog: () => null,
  GlyphSpinner: () => <span />,
  host: {},
  Input: (props: ComponentProps<'input'>) => <input {...props} />
}))
const CONTROLS = vi.hoisted(() => ({ back: 'Back', cancel: 'Cancel', disband: 'Disband', rename: 'Rename', save: 'Save', stop: 'Stop' }))

vi.mock('./canonical-group-labels', async () => {
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')

  return { useCanonicalGroupLabels: () => ({ ...CANONICAL_GROUP_LOCALES.en, ...CONTROLS }) }
})

import { CanonicalGroupHeader } from './canonical-group-header'
import { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'
import type { CanonicalRoomContext } from './canonical-groups'

const labels = { ...CANONICAL_GROUP_LOCALES.en, ...CONTROLS }
const room = { room_id: 'room-1', name: 'Ops', members: [] }

afterEach(cleanup)

function openSettings(onSaveContext: (context: CanonicalRoomContext) => Promise<boolean>) {
  render(
    <CanonicalGroupHeader
      busy={false}
      canStop={false}
      context={{ purpose: 'Old purpose', notes: 'Old notes' }}
      deliveryTarget="room:room-1"
      onDisband={async () => {}}
      onRename={() => {}}
      onSaveContext={onSaveContext}
      onStop={() => {}}
      phases={new Map()}
      room={room as never}
    />
  )
  fireEvent.click(screen.getByText(labels.roomSettings))

  return screen.getByPlaceholderText(labels.notesPlaceholder) as HTMLTextAreaElement
}

it('keeps notes the backend rejected so they can be shortened and saved again', async () => {
  const save = vi.fn(async () => false)
  const notes = openSettings(save)

  fireEvent.change(notes, { target: { value: 'My long draft' } })
  fireEvent.click(screen.getByText(labels.save))

  await waitFor(() => expect(save).toHaveBeenCalledWith({ purpose: 'Old purpose', notes: 'My long draft' }))
  expect((screen.getByPlaceholderText(labels.notesPlaceholder) as HTMLTextAreaElement).value).toBe('My long draft')
})

it('an in-flight save cannot be overtaken by edits it would then discard', async () => {
  let finish: (saved: boolean) => void = () => {}
  const save = vi.fn(() => new Promise<boolean>(resolve => { finish = resolve }))
  const notes = openSettings(save)

  fireEvent.change(notes, { target: { value: 'First draft' } })
  fireEvent.click(screen.getByText(labels.save))
  await waitFor(() => expect(save).toHaveBeenCalledTimes(1))

  // While the save is in flight the draft is frozen: nothing typed now can be lost when it lands.
  expect(notes.disabled).toBe(true)
  expect((screen.getByText(labels.save) as HTMLButtonElement).disabled).toBe(true)

  finish(false)
  await waitFor(() => expect(notes.disabled).toBe(false))
  expect(notes.value).toBe('First draft')
})

it('blocks saving notes over the 4 KB limit while typing', () => {
  const save = vi.fn(async () => true)
  const notes = openSettings(save)

  fireEvent.change(notes, { target: { value: 'é'.repeat(2049) } }) // 4098 UTF-8 bytes in 2049 characters

  expect(screen.getByTestId('room-notes-bytes').textContent).toBe('4098 / 4096')
  expect((screen.getByText(labels.save) as HTMLButtonElement).disabled).toBe(true)
})
