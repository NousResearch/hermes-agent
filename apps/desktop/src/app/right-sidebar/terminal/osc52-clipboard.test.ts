import { Terminal } from '@xterm/xterm'
import { describe, expect, it, vi } from 'vitest'

import { installOsc52ClipboardHandler, OSC52_MAX_TEXT_BYTES, OSC52_MIN_INTERVAL_MS } from './clipboard'

const encode = (text: string) => btoa(String.fromCharCode(...new TextEncoder().encode(text)))
const focused = { isFocused: () => true }

function captureHandler(options: Parameters<typeof installOsc52ClipboardHandler>[2] = focused) {
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

  installOsc52ClipboardHandler(terminal, writeClipboardText, options)

  return { handler: (data: string) => handler?.(data), writeClipboardText }
}

describe('installOsc52ClipboardHandler', () => {
  it('writes clipboard data emitted by a real xterm parser', async () => {
    const terminal = new Terminal({ allowProposedApi: true })
    const writeClipboardText = vi.fn().mockResolvedValue(undefined)
    const registration = installOsc52ClipboardHandler(terminal, writeClipboardText, focused)
    const text = 'Claude Code selection ✓'
    const payload = encode(text)

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

    const registration = installOsc52ClipboardHandler(terminal, writeClipboardText, focused)
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

    installOsc52ClipboardHandler(terminal, writeClipboardText, focused)

    expect(handler?.('c;?')).toBe(false)
    expect(handler?.('c;')).toBe(false)
    expect(handler?.(`p;${btoa('primary')}`)).toBe(false)
    expect(handler?.('c;not base64%%%')).toBe(false)
    expect(writeClipboardText).not.toHaveBeenCalled()
  })

  it('ignores clipboard writes while the window is not focused', () => {
    const { handler, writeClipboardText } = captureHandler({ isFocused: () => false })

    expect(handler(`c;${encode('background write')}`)).toBe(false)
    expect(writeClipboardText).not.toHaveBeenCalled()
  })

  it('rejects an oversized payload before decoding it', () => {
    const { handler, writeClipboardText } = captureHandler()
    const decode = vi.spyOn(globalThis, 'atob')

    expect(handler(`c;${'A'.repeat(Math.ceil((OSC52_MAX_TEXT_BYTES * 4) / 3) + 8)}`)).toBe(false)
    expect(decode).not.toHaveBeenCalled()
    expect(writeClipboardText).not.toHaveBeenCalled()
    decode.mockRestore()
  })

  it('accepts one write per interval from a burst of sequences', async () => {
    let now = 10_000
    const terminal = new Terminal({ allowProposedApi: true })
    const writeClipboardText = vi.fn().mockResolvedValue(undefined)
    const registration = installOsc52ClipboardHandler(terminal, writeClipboardText, { ...focused, now: () => now })
    const burst = Array.from({ length: 500 }, (_, i) => `\u001b]52;c;${encode(`spam ${i}`)}\u0007`).join('')

    await new Promise<void>(resolve => terminal.write(burst, resolve))
    expect(writeClipboardText).toHaveBeenCalledTimes(1)
    expect(writeClipboardText).toHaveBeenCalledWith('spam 0')

    now += OSC52_MIN_INTERVAL_MS
    await new Promise<void>(resolve => terminal.write(`\u001b]52;c;${encode('next copy')}\u0007`, resolve))
    expect(writeClipboardText).toHaveBeenCalledTimes(2)
    expect(writeClipboardText).toHaveBeenLastCalledWith('next copy')
    registration.dispose()
    terminal.dispose()
  })
})
