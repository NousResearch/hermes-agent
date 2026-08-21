import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $petOverlayApproval } from '@/store/pet-overlay'

import { PetBubble, summarizePetApproval } from './pet-bubble'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

afterEach(() => {
  act(() => {
    $petOverlayApproval.set(null)
  })

  if (initialHermesDesktop) {
    desktopWindow.hermesDesktop = initialHermesDesktop
  } else {
    delete desktopWindow.hermesDesktop
  }
})

describe('pet approval summary', () => {
  it('preserves the full multiline command before approval', () => {
    expect(summarizePetApproval('npm run build\nrm -rf dist', 'dangerous command', 'Pending approval')).toBe(
      'npm run build\nrm -rf dist'
    )
  })

  it('falls back to the description when no command is available', () => {
    expect(summarizePetApproval('', 'dangerous command', 'Pending approval')).toBe('dangerous command')
  })

  it('uses the localized fallback when no details are available', () => {
    expect(summarizePetApproval(' ', ' ', 'Pending approval')).toBe('Pending approval')
  })
})

describe('PetBubble approval actions', () => {
  it('does not clear the overlay approval before the renderer confirms success', () => {
    const control = vi.fn()

    desktopWindow.hermesDesktop = {
      petOverlay: { control }
    } as unknown as Window['hermesDesktop']
    $petOverlayApproval.set({ command: 'npm run build\nrm -rf dist', description: 'dangerous', sessionId: 'sess-1' })

    render(
      <I18nProvider configClient={null}>
        <PetBubble />
      </I18nProvider>
    )
    act(() => {
      fireEvent.click(screen.getByRole('button', { name: /Approve once/i }))
    })

    expect(control).toHaveBeenCalledWith({ choice: 'once', sessionId: 'sess-1', type: 'approval' })
    expect($petOverlayApproval.get()).toEqual({
      command: 'npm run build\nrm -rf dist',
      description: 'dangerous',
      sessionId: 'sess-1'
    })
  })

  it('re-enables the buttons when the respond never confirms (failure path)', () => {
    vi.useFakeTimers()
    const control = vi.fn()

    desktopWindow.hermesDesktop = {
      petOverlay: { control }
    } as unknown as Window['hermesDesktop']
    $petOverlayApproval.set({ command: 'rm -rf dist', description: 'dangerous', sessionId: 'sess-2' })

    render(
      <I18nProvider configClient={null}>
        <PetBubble />
      </I18nProvider>
    )
    act(() => {
      fireEvent.click(screen.getByRole('button', { name: /Approve once/i }))
    })

    // In flight: both buttons disabled while the response is pending.
    expect(screen.getAllByRole<HTMLButtonElement>('button').every(button => button.disabled)).toBe(true)
    // Failure path keeps the prompt parked for in-app resolution.
    expect($petOverlayApproval.get()).not.toBeNull()

    act(() => {
      vi.advanceTimersByTime(5000)
    })
    expect(screen.getAllByRole<HTMLButtonElement>('button').every(button => !button.disabled)).toBe(true)
    vi.useRealTimers()
  })
})
