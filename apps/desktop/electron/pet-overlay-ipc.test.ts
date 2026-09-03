import assert from 'node:assert/strict'
import type { BrowserWindow } from 'electron'
import { describe, it, vi } from 'vitest'

const onHandlers = new Map<string, (...args: unknown[]) => unknown>()

vi.mock('electron', () => ({
  ipcMain: {
    handle: vi.fn(),
    on: (channel: string, fn: (...args: unknown[]) => unknown) => {
      onHandlers.set(channel, fn)
    }
  }
}))

const { registerPetOverlayIpc } = await import('./pet-overlay-ipc')

describe('pet-overlay-ipc', () => {
  it('does not steal focus from main window when pet overlay becomes focusable (#102039)', () => {
    let petFocused = false
    let petFocusable = false

    const mockPetOverlayWindow = {
      focus: () => {
        petFocused = true
      },
      isDestroyed: () => false,
      setFocusable: (val: boolean) => {
        petFocusable = val
      }
    } as unknown as BrowserWindow

    let mainFocused = true
    const mockMainWindow = {
      isDestroyed: () => false,
      isFocused: () => mainFocused
    } as unknown as BrowserWindow

    registerPetOverlayIpc({
      closePetOverlay: vi.fn(),
      getMainWindow: () => mockMainWindow,
      getPetOverlayWindow: () => mockPetOverlayWindow,
      openPetOverlay: vi.fn()
    })

    const setFocusableHandler = onHandlers.get('hermes:pet-overlay:set-focusable')
    assert.ok(setFocusableHandler, 'handler registered for hermes:pet-overlay:set-focusable')

    // 1. When main window is focused, pet overlay should become focusable but NOT steal focus
    setFocusableHandler({}, true)
    assert.equal(petFocusable, true)
    assert.equal(petFocused, false, 'must not steal focus when main window is focused')

    // 2. When main window is not focused, pet overlay should focus
    mainFocused = false
    setFocusableHandler({}, true)
    assert.equal(petFocused, true, 'should focus pet overlay when main window is not focused')

    // 3. When focusable is false, focus should not be called
    petFocused = false
    setFocusableHandler({}, false)
    assert.equal(petFocusable, false)
    assert.equal(petFocused, false)
  })
})
