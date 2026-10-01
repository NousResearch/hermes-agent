import type * as HermesSdk from '@hermes/plugin-sdk'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

vi.mock('@hermes/plugin-sdk', async () => ({ ...await vi.importActual<typeof HermesSdk>('@hermes/plugin-sdk') }))
vi.mock('./canonical-group-labels', () => ({ useCanonicalGroupLabels: () => ({
  everyone: 'Everyone', unknownBot: 'Bot', members: 'Participants', groupMessage: 'Group message', messagePlaceholder: 'Message {name}…'
}) }))

import { CanonicalGroupComposerInput } from './canonical-group-composer'

afterEach(cleanup)

const members = [
  { member_id: 'opaque-owner-one', profile: 'default', handle: 'owner-atlas', display_name: 'Atlas Bot' },
  { member_id: 'opaque-owner-two', profile: 'default', handle: 'peer-mira', display_name: 'Mira Bot' }
]

function Composer({ onSubmit = () => {}, disabled = false }: { onSubmit?: () => void; disabled?: boolean }) {
  const [value, setValue] = useState('')

  return <CanonicalGroupComposerInput disabled={disabled} members={members} name="Autumn launch" onChange={setValue} onSubmit={onSubmit} value={value} />
}

it('finds a friendly Bot name but inserts its exact owner handle, preserving the surrounding draft', () => {
  render(<Composer />)
  const input = screen.getByRole('textbox') as HTMLTextAreaElement
  fireEvent.change(input, { target: { value: 'Please ask @Mir', selectionStart: 15 } })
  expect(screen.getByRole('option', { name: 'Mira Bot' })).toBeTruthy()
  expect(screen.queryByText('opaque-owner-two')).toBeNull()
  fireEvent.keyDown(input, { key: 'Enter' })
  expect(input.value).toBe('Please ask @peer-mira ')
  expect(screen.queryByRole('listbox')).toBeNull()
  expect(input.placeholder).toBe('Message Autumn launch…')
})

it('supports Everyone, keyboard choice, multiline drafts and IME without an accidental send', () => {
  const submit = vi.fn()
  render(<Composer onSubmit={submit} />)
  const input = screen.getByRole('textbox') as HTMLTextAreaElement
  fireEvent.change(input, { target: { value: '@', selectionStart: 1 } })
  fireEvent.keyDown(input, { key: 'Enter', isComposing: true })
  expect(submit).not.toHaveBeenCalled()
  expect(input.value).toBe('@')
  fireEvent.keyDown(input, { key: 'Tab' })
  expect(input.value).toBe('@all ')
  fireEvent.keyDown(input, { key: 'Enter', shiftKey: true })
  expect(submit).not.toHaveBeenCalled()
  fireEvent.keyDown(input, { key: 'Enter' })
  expect(submit).toHaveBeenCalledOnce()
})

it('updates completion when the caret moves and refuses an old selection before a selection event arrives', () => {
  const submit = vi.fn()
  render(<Composer onSubmit={submit} />)
  const input = screen.getByRole('textbox') as HTMLTextAreaElement
  fireEvent.change(input, { target: { value: 'Ask @Mir', selectionStart: 8 } })
  expect(screen.getByRole('option', { name: 'Mira Bot' })).toBeTruthy()
  input.setSelectionRange(0, 0)
  fireEvent.keyDown(input, { key: 'Enter' })
  expect(input.value).toBe('Ask @Mir')
  expect(submit).not.toHaveBeenCalled()
  expect(screen.queryByRole('listbox')).toBeNull()
  input.setSelectionRange(8, 8)
  fireEvent.select(input)
  expect(screen.getByRole('option', { name: 'Mira Bot' })).toBeTruthy()
  input.setSelectionRange(0, 0)
  fireEvent.select(input)
  expect(screen.queryByRole('listbox')).toBeNull()
  fireEvent.keyDown(input, { key: 'Enter' })
  expect(submit).toHaveBeenCalledOnce()
})
