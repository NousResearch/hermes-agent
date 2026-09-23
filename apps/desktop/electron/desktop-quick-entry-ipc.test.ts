import { describe, expect, it, vi } from 'vitest'

import { registerDesktopQuickEntryIpc } from './desktop-quick-entry-ipc'

describe('desktop Quick Entry IPC', () => {
  it('forwards a submitted prompt to the primary renderer without raising it', async () => {
    const handles = new Map<string, (...args: any[]) => any>()
    const events = new Map<string, (...args: any[]) => any>()
    const send = vi.fn()
    const show = vi.fn()
    const focus = vi.fn()
    const hideQuickEntryWindow = vi.fn()

    registerDesktopQuickEntryIpc({
      applyQuickEntrySettings: vi.fn(),
      getMainWindow: () => ({ isDestroyed: () => false, webContents: { send }, show, focus }) as any,
      hideQuickEntryWindow,
      ipcMain: {
        handle: (name: string, handler: (...args: any[]) => any) => handles.set(name, handler),
        on: (name: string, handler: (...args: any[]) => any) => events.set(name, handler)
      },
      readQuickEntrySettings: () => ({ enabled: true, shortcut: 'Ctrl+Space' }),
      rememberLog: vi.fn(),
      sanitizeQuickEntrySettings: value => value,
      shellOverlayRuntime: { currentQuickEntryShortcutState: () => ({ registered: true, shortcut: 'Ctrl+Space' }), pushQuickEntryState: vi.fn() },
      writeQuickEntrySettings: vi.fn()
    })

    expect(await handles.get('hermes:quick-entry:settings:get')!()).toMatchObject({ registered: true })

    events.get('hermes:quick-entry:submit')!(null, { target: 'current', text: '  hello  ' })

    expect(hideQuickEntryWindow).toHaveBeenCalledOnce()
    expect(send).toHaveBeenCalledWith('hermes:quick-entry:submit', { target: 'current', text: 'hello' })
    expect(show).not.toHaveBeenCalled()
    expect(focus).not.toHaveBeenCalled()
  })
})
