import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, expect, it, vi } from 'vitest'

import { DropdownMenuItem, DropdownMenuSearch } from '@/components/ui/dropdown-menu'
import { useShortViewport } from '@/lib/short-viewport'
import { $touchPointer } from '@/lib/touch-interaction'

import { ComposerControls } from './controls'

let short = false
const listeners = new Set<() => void>()

function KeyboardControls() {
  const keyboardUp = useShortViewport()

  return (
    <ComposerControls
      autoSpeak={false}
      busy={false}
      busyAction="stop"
      canSubmit
      compactModelPill
      conversation={{
        active: false,
        levelStore: atom(0),
        muted: false,
        status: 'idle',
        onEnd: vi.fn(),
        onStart: vi.fn(),
        onStopTurn: vi.fn(),
        onToggleMute: vi.fn()
      }}
      disabled={false}
      hasComposerPayload
      keepModelPicker={keyboardUp}
      minimal={keyboardUp}
      onDictate={vi.fn()}
      onQueue={vi.fn()}
      onToggleAutoSpeak={vi.fn()}
      state={{
        model: {
          canSwitch: true,
          model: 'gpt-6.1-sol-900k',
          provider: 'openai-codex',
          modelMenuContent: (
            <>
              <DropdownMenuSearch placeholder="Find a model" />
              <DropdownMenuItem onSelect={event => event.preventDefault()}>Model settings</DropdownMenuItem>
            </>
          )
        },
        tools: { enabled: false, label: '' },
        voice: { active: false, enabled: false }
      }}
      voiceStatus="idle"
    />
  )
}

afterEach(() => {
  cleanup()
  $touchPointer.set(false)
  listeners.clear()
  short = false
  vi.unstubAllGlobals()
})

it('keeps the same touch model picker mounted through keyboard resize and allows reopening while short', async () => {
  $touchPointer.set(true)
  vi.stubGlobal('matchMedia', (query: string) => ({
    get matches() {
      return query === '(max-height: 520px)' && short
    },
    media: query,
    addEventListener: (_type: string, listener: () => void) => {
      if (query === '(max-height: 520px)') {
        listeners.add(listener)
      }
    },
    removeEventListener: (_type: string, listener: () => void) => listeners.delete(listener)
  }))
  render(<KeyboardControls />)

  const picker = screen.getByRole('button', { name: /^Model ·/ })
  expect(picker.textContent).toContain('GPT-6.1-sol-900k')
  fireEvent.pointerDown(picker, { button: 0, ctrlKey: false, pointerType: 'touch' })
  const search = await screen.findByPlaceholderText('Find a model')
  act(() => search.focus())
  act(() => {
    short = true
    listeners.forEach(listener => listener())
  })

  expect(picker.isConnected).toBe(true)
  expect(picker.getAttribute('aria-expanded')).toBe('true')
  expect(window.document.activeElement).toBe(search)
  expect(picker.textContent).toBe('')
  expect(screen.getByText('Model settings')).toBeTruthy()
  expect(screen.queryByLabelText('Voice dictation')).toBeNull()
  expect(screen.getByRole('button', { name: 'Send', hidden: true })).toBeTruthy()
  fireEvent.keyDown(search, { key: 'Escape' })
  await waitFor(() => expect(screen.queryByRole('menu')).toBeNull())
  expect(screen.getByRole('button', { name: /^Model ·/ })).toBe(picker)
  fireEvent.pointerDown(picker, { button: 0, ctrlKey: false, pointerType: 'touch' })
  expect(await screen.findByText('Model settings')).toBeTruthy()
  act(() => {
    short = false
    listeners.forEach(listener => listener())
  })
  expect(picker.isConnected).toBe(true)
  expect(picker.textContent).toContain('GPT-6.1-sol-900k')
})
