import { Terminal } from '@xterm/xterm'
import { describe, expect, it, vi } from 'vitest'

import { installOsc52ClipboardHandler } from './clipboard'

describe('installOsc52ClipboardHandler', () => {
  it('writes clipboard data emitted by a real xterm parser', async () => {
    const terminal = new Terminal({ allowProposedApi: true })
    const writeClipboardText = vi.fn().mockResolvedValue(undefined)
    const registration = installOsc52ClipboardHandler(terminal, writeClipboardText)
    const text = 'Claude Code selection ✓'
    const payload = btoa(String.fromCharCode(...new TextEncoder().encode(text)))

    await new Promise<void>(resolve => terminal.write(`\u001b]52;c;${payload}\u0007`, resolve))

    expect(writeClipboardText).toHaveBeenCalledWith(text)
    registration.dispose()
    terminal.dispose()
  })

  it('registers an OSC 52 handler that disposes cleanly', () => {
    let handler: ((data: string) => boolean) | undefined
    const dispose = vi.fn()
    const writeClipboardText = vi.fn().mockResolvedValue(undefined)

    const terminal = {
      parser: {
        registerOscHandler: vi.fn((_id: number, callback: (data: string) => boolean) => {
          handler = callback

          return { dispose }
        })
      }
    }

    const registration = installOsc52ClipboardHandler(terminal, writeClipboardText)
    const payload = btoa(String.fromCharCode(...new TextEncoder().encode('copied ✓')))

    expect(terminal.parser.registerOscHandler).toHaveBeenCalledWith(52, expect.any(Function))
    expect(handler?.(`c;${payload}`)).toBe(true)
    expect(writeClipboardText).toHaveBeenCalledWith('copied ✓')
    registration.dispose()
    expect(dispose).toHaveBeenCalledOnce()
  })

  it('does not service OSC 52 clipboard reads, clears, or other selection targets', () => {
    let handler: ((data: string) => boolean) | undefined
    const writeClipboardText = vi.fn().mockResolvedValue(undefined)

    const terminal = {
      parser: {
        registerOscHandler: vi.fn((_id: number, callback: (data: string) => boolean) => {
          handler = callback

          return { dispose: vi.fn() }
        })
      }
    }

    installOsc52ClipboardHandler(terminal, writeClipboardText)

    expect(handler?.('c;?')).toBe(false)
    expect(handler?.('c;')).toBe(false)
    expect(handler?.(`p;${btoa('primary')}`)).toBe(false)
    expect(handler?.('c;not base64%%%')).toBe(false)
    expect(writeClipboardText).not.toHaveBeenCalled()
  })
})
