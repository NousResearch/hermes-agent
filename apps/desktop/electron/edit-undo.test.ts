import { describe, expect, it, vi } from 'vitest'

import { installEditUndoShortcut, isEditHistoryChord } from './edit-undo'

const chord = (over: Partial<Electron.Input> = {}): Electron.Input =>
  ({
    alt: false,
    control: false,
    key: 'z',
    meta: false,
    shift: false,
    type: 'keyDown',
    ...over
  }) as Electron.Input

describe('isEditHistoryChord', () => {
  it('claims Cmd+Z / Ctrl+Z as undo', () => {
    expect(isEditHistoryChord(chord({ meta: true }), true)).toBe('undo')
    expect(isEditHistoryChord(chord({ control: true }), false)).toBe('undo')
  })

  it('claims Cmd+Shift+Z and Ctrl+Y as redo', () => {
    expect(isEditHistoryChord(chord({ meta: true, shift: true }), true)).toBe('redo')
    expect(isEditHistoryChord(chord({ control: true, key: 'y' }), false)).toBe('redo')
  })

  it('ignores Alt variants and unrelated keys', () => {
    expect(isEditHistoryChord(chord({ alt: true, meta: true }), true)).toBeNull()
    expect(isEditHistoryChord(chord({ key: 'a', meta: true }), true)).toBeNull()
  })
})

describe('installEditUndoShortcut', () => {
  it('prevents the chord and forwards undo/redo IPC', () => {
    const handlers = new Map<string, (...args: unknown[]) => void>()
    const send = vi.fn()
    const webContents = {
      isDestroyed: () => false,
      off: vi.fn((channel: string) => handlers.delete(channel)),
      on: vi.fn((channel: string, handler: (...args: unknown[]) => void) => {
        handlers.set(channel, handler)
      }),
      send
    }
    const win = { webContents } as unknown as Electron.BrowserWindow

    const uninstall = installEditUndoShortcut(win, () => true)
    const handler = handlers.get('before-input-event')
    expect(handler).toBeTypeOf('function')

    const event = { preventDefault: vi.fn() }
    handler?.(event, chord({ meta: true }))
    expect(event.preventDefault).toHaveBeenCalled()
    expect(send).toHaveBeenCalledWith('hermes:edit-undo')

    send.mockClear()
    event.preventDefault.mockClear()
    handler?.(event, chord({ meta: true, shift: true }))
    expect(send).toHaveBeenCalledWith('hermes:edit-redo')

    uninstall()
    expect(webContents.off).toHaveBeenCalledWith('before-input-event', handler)
  })
})
