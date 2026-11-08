import { afterEach, describe, expect, it, vi } from 'vitest'

import { createBrowserClipboardBridge } from './browser-clipboard'
import { installClipboardShim, writeClipboardText } from './clipboard'
import { copySvgAsPng } from './svg-image'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }

function installClipboard(writeText: (text: string) => Promise<void>) {
  Object.defineProperty(navigator, 'clipboard', {
    configurable: true,
    value: { writeText }
  })
}

afterEach(() => {
  Reflect.deleteProperty(desktopWindow, 'hermesDesktop')
  Reflect.deleteProperty(navigator, 'clipboard')
  vi.restoreAllMocks()
  delete document.documentElement.dataset.hermesDesktopHost
})

describe('installClipboardShim', () => {
  it('routes an Electron write through IPC first', async () => {
    const nativeWrite = vi.fn().mockResolvedValue(undefined)
    const ipcWrite = vi.fn().mockResolvedValue(true)
    installClipboard(nativeWrite)
    desktopWindow.hermesDesktop = { writeClipboard: ipcWrite } as unknown as Window['hermesDesktop']

    installClipboardShim()
    await navigator.clipboard.writeText('payload')

    expect(ipcWrite).toHaveBeenCalledWith('payload')
    expect(nativeWrite).not.toHaveBeenCalled()
  })

  it('uses the native write only after the Electron IPC write fails', async () => {
    const nativeWrite = vi.fn().mockResolvedValue(undefined)
    const ipcWrite = vi.fn().mockRejectedValue(new Error('main process gone'))
    installClipboard(nativeWrite)
    desktopWindow.hermesDesktop = { writeClipboard: ipcWrite } as unknown as Window['hermesDesktop']

    installClipboardShim()
    await navigator.clipboard.writeText('payload')

    expect(ipcWrite).toHaveBeenCalledWith('payload')
    expect(nativeWrite).toHaveBeenCalledWith('payload')
  })
})

describe('writeClipboardText', () => {
  it('writes through Electron IPC first, so a copy from an unfocused document still lands', async () => {
    const nativeWrite = vi.fn().mockRejectedValue(new Error('Document is not focused'))
    const ipcWrite = vi.fn().mockResolvedValue(true)
    installClipboard(nativeWrite)
    desktopWindow.hermesDesktop = { writeClipboard: ipcWrite } as unknown as Window['hermesDesktop']

    await writeClipboardText('payload')

    expect(ipcWrite).toHaveBeenCalledTimes(1)
    expect(nativeWrite).not.toHaveBeenCalled()
  })

  it('writes a browser-hosted copy through the native clipboard once', async () => {
    const nativeWrite = vi.fn().mockResolvedValue(undefined)
    installClipboard(nativeWrite)
    document.documentElement.dataset.hermesDesktopHost = 'browser'
    desktopWindow.hermesDesktop = createBrowserClipboardBridge({
      saveBuffer: async () => ''
    }) as Window['hermesDesktop']
    installClipboardShim()

    await writeClipboardText('payload')

    expect(nativeWrite).toHaveBeenCalledTimes(1)
    expect(nativeWrite).toHaveBeenCalledWith('payload')
  })

  it('attempts IPC and native once when an installed Electron shim fails', async () => {
    const nativeWrite = vi.fn().mockRejectedValue(new Error('lost focus'))
    const bridgeWrite = vi.fn().mockResolvedValue(false)
    installClipboard(nativeWrite)
    desktopWindow.hermesDesktop = { writeClipboard: bridgeWrite } as unknown as Window['hermesDesktop']
    installClipboardShim()
    await expect(writeClipboardText('payload')).rejects.toThrow()
    expect(nativeWrite).toHaveBeenCalledTimes(1)
    expect(bridgeWrite).toHaveBeenCalledTimes(1)
  })

  it('does not retry the same native API through the installed browser bridge', async () => {
    const nativeWrite = vi.fn().mockRejectedValue(new Error('permission denied'))
    installClipboard(nativeWrite)
    document.documentElement.dataset.hermesDesktopHost = 'browser'
    desktopWindow.hermesDesktop = createBrowserClipboardBridge({
      saveBuffer: async () => ''
    }) as Window['hermesDesktop']
    installClipboardShim()
    await expect(writeClipboardText('payload')).rejects.toThrow('permission denied')
    expect(nativeWrite).toHaveBeenCalledTimes(1)
  })

  it('reports an unavailable bridge as a failed copy', async () => {
    const bridgeWrite = vi.fn().mockResolvedValue(false)
    desktopWindow.hermesDesktop = { writeClipboard: bridgeWrite } as unknown as Window['hermesDesktop']

    await expect(writeClipboardText('payload')).rejects.toThrow('Clipboard write is unavailable')
  })

  // Safari does not focus a clicked button, so the dialog itself can hold focus when its Copy is clicked.
  it.each(['a button in it', 'the dialog itself'])(
    'copies through a selection where an insecure origin has no Clipboard API, inside a focus trap while %s has focus',
    async holder => {
      document.documentElement.dataset.hermesDesktopHost = 'browser'
      desktopWindow.hermesDesktop = createBrowserClipboardBridge({
        saveBuffer: async () => ''
      }) as Window['hermesDesktop']
      installClipboardShim()
      // A modal dialog that pulls focus back inside itself, as Radix's FocusScope does.
      const dialog = document.body.appendChild(document.createElement('div'))
      dialog.tabIndex = -1
      const trigger = dialog.appendChild(document.createElement('button'))
      const note = dialog.appendChild(document.createElement('p'))
      note.textContent = 'selected before the copy'
      const focused = holder === 'the dialog itself' ? dialog : trigger

      const trap = (event: FocusEvent) => {
        if (!dialog.contains(event.target as Node)) {
          focused.focus()
        }
      }

      document.addEventListener('focusin', trap)
      focused.focus()
      const copied: string[] = []
      let copies = true

      const execCommand = vi.fn((command: string) => {
        const area = document.activeElement

        copied.push(
          area instanceof HTMLTextAreaElement
            ? `${command}:${area.value.slice(area.selectionStart, area.selectionEnd)}`
            : `${command} in ${area?.tagName}`
        )
        // A browser moves the document's selection into the textarea it selects; jsdom does not.
        document.getSelection()!.removeAllRanges()

        return copies
      })

      Object.defineProperty(document, 'execCommand', { configurable: true, value: execCommand })

      try {
        await writeClipboardText('copy me')
        await copySvgAsPng('<svg/>')
        // Copy path / Copy link call the bridge directly.
        expect(await desktopWindow.hermesDesktop!.writeClipboard('/repo/file.ts')).toBe(true)
        expect(copied).toEqual(['copy:copy me', 'copy:<svg/>', 'copy:/repo/file.ts'])
        // A copy the browser refuses fails, and still gives back focus and the page's selection.
        copies = false
        document.getSelection()!.selectAllChildren(note)
        let selected = ''
        // jsdom collapses the selection on every focus(), which a browser does not do for a
        // non-editable element: read it as focus comes back.
        focused.addEventListener('focus', () => (selected = document.getSelection()!.toString()), { once: true })
        await expect(writeClipboardText('refused')).rejects.toThrow('Clipboard API is unavailable')
        expect(copied.at(-1)).toBe('copy:refused')
        expect(document.querySelector('textarea')).toBeNull()
        expect(document.activeElement).toBe(focused)
        expect(selected).toBe('selected before the copy')
      } finally {
        Reflect.deleteProperty(document, 'execCommand')
        document.removeEventListener('focusin', trap)
        document.getSelection()!.removeAllRanges()
        dialog.remove()
      }
    }
  )
})
