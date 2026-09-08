import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { Dialog } from 'radix-ui'
import { useRef, useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { closeHud } from '@/store/hud'

import { useHudEscape } from './escape'

vi.mock('@/store/hud', () => ({ closeHud: vi.fn() }))

function HudFixture({ autocomplete = false, dialog = false }) {
  const rootRef = useRef<HTMLDivElement>(null)
  const [suggestionsOpen, setSuggestionsOpen] = useState(autocomplete)
  useHudEscape(rootRef)

  return (
    <div ref={rootRef}>
      <input
        aria-label="Composer"
        onKeyDown={event => {
          // Match the composer's autocomplete contract: consume Escape and
          // close suggestions while keyboard focus stays in the editor.
          if (event.key === 'Escape' && suggestionsOpen) {
            event.preventDefault()
            setSuggestionsOpen(false)
          }
        }}
      />
      {suggestionsOpen && <div aria-label="Suggestions" role="listbox" />}
      {dialog && (
        <Dialog.Root>
          <Dialog.Trigger>Open picker</Dialog.Trigger>
          <Dialog.Portal>
            <Dialog.Content aria-describedby={undefined}>
              <Dialog.Title>Picker</Dialog.Title>
              <input aria-label="Picker search" />
            </Dialog.Content>
          </Dialog.Portal>
        </Dialog.Root>
      )}
    </div>
  )
}

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('HUD Escape interaction', () => {
  it('dismisses the HUD once when its focused composer has nothing to dismiss', () => {
    render(<HudFixture />)
    const composer = screen.getByRole('textbox', { name: 'Composer' })
    composer.focus()

    fireEvent.keyDown(composer, { key: 'Escape' })

    expect(closeHud).toHaveBeenCalledTimes(1)
  })

  it('lets autocomplete consume the first Escape and dismisses the HUD on the next', () => {
    render(<HudFixture autocomplete />)
    const composer = screen.getByRole('textbox', { name: 'Composer' })
    composer.focus()

    fireEvent.keyDown(composer, { key: 'Escape' })

    expect(screen.queryByRole('listbox')).toBeNull()
    expect(window.document.activeElement).toBe(composer)
    expect(closeHud).not.toHaveBeenCalled()

    fireEvent.keyDown(composer, { key: 'Escape' })

    expect(closeHud).toHaveBeenCalledTimes(1)
  })

  it('closes a real portalled dialog before allowing Escape to dismiss the HUD', async () => {
    render(<HudFixture dialog />)
    const trigger = screen.getByRole('button', { name: 'Open picker' })
    fireEvent.click(trigger)
    const search = screen.getByRole('textbox', { name: 'Picker search' })
    search.focus()

    fireEvent.keyDown(search, { key: 'Escape' })

    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
    expect(closeHud).not.toHaveBeenCalled()
    await waitFor(() => expect(window.document.activeElement).toBe(trigger))

    fireEvent.keyDown(trigger, { key: 'Escape' })

    expect(closeHud).toHaveBeenCalledTimes(1)
  })

  it('leaves Escape to a focused portal even when it does not preventDefault', () => {
    render(<HudFixture />)
    const portal = window.document.createElement('input')
    window.document.body.appendChild(portal)

    try {
      portal.focus()
      fireEvent.keyDown(portal, { key: 'Escape' })
      expect(closeHud).not.toHaveBeenCalled()
    } finally {
      portal.remove()
    }
  })

  it('ignores other keys and removes its listener on unmount', () => {
    const { unmount } = render(<HudFixture />)
    const composer = screen.getByRole('textbox', { name: 'Composer' })
    composer.focus()
    fireEvent.keyDown(composer, { key: 'Enter' })
    expect(closeHud).not.toHaveBeenCalled()

    unmount()
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(closeHud).not.toHaveBeenCalled()
  })
})
