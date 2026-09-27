import { describe, expect, it, vi } from 'vitest'

import { describeCrashReason, installCrashForensics, isExpectedTransition, markExpectedTransition } from './crash-forensics'

const harness = () => {
  const listeners = new Map<string, (value: unknown) => void>()
  const flush = vi.fn()
  const log = vi.fn()

  installCrashForensics({
    flush,
    log,
    target: { on: (event, listener) => listeners.set(event, listener) }
  })

  return { flush, listeners, log }
}

describe('describeCrashReason', () => {
  it('prefers a stack, then a message, for thrown errors', () => {
    const withStack = new Error('boom')
    withStack.stack = 'Error: boom\n    at somewhere'

    expect(describeCrashReason(withStack)).toBe('Error: boom\n    at somewhere')

    const withoutStack = new Error('boom')
    withoutStack.stack = ''

    expect(describeCrashReason(withoutStack)).toBe('boom')
  })

  it('renders non-error rejections without throwing', () => {
    expect(describeCrashReason('plain string')).toBe('plain string')
    expect(describeCrashReason({ code: 'ECONNRESET' })).toBe('{"code":"ECONNRESET"}')
    expect(describeCrashReason(undefined)).toBe('undefined')

    const circular: Record<string, unknown> = {}
    circular.self = circular

    expect(describeCrashReason(circular)).toBe('[object Object]')
  })
})

describe('installCrashForensics', () => {
  it('records and synchronously flushes an uncaught exception', () => {
    const { flush, listeners, log } = harness()
    const error = new Error('renderer gone')
    error.stack = 'Error: renderer gone\n    at main'

    listeners.get('uncaughtException')?.(error)

    expect(log).toHaveBeenCalledWith(expect.stringContaining('Error: renderer gone\n    at main'))
    expect(flush).toHaveBeenCalledTimes(1)
  })

  it('records and synchronously flushes an unhandled rejection', () => {
    const { flush, listeners, log } = harness()

    listeners.get('unhandledRejection')?.('gateway ticket mint failed')

    expect(log).toHaveBeenCalledWith(expect.stringContaining('gateway ticket mint failed'))
    expect(flush).toHaveBeenCalledTimes(1)
  })

  it('records a marked quit sentinel as a one-line expected transition, not a crash stack', () => {
    const { flush, listeners, log } = harness()

    // The exact shape an intentional quit leaves behind: the AbortController
    // reason from local-backend-lifecycle rejecting an in-flight start.
    const sentinel = markExpectedTransition(new Error('Hermes Desktop is quitting.'))
    sentinel.stack = 'Error: Hermes Desktop is quitting.\n    at Object.run [as shutdown] (file:///app.asar/dist/electron-main.mjs:1374:40)'

    listeners.get('unhandledRejection')?.(sentinel)

    expect(log).toHaveBeenCalledTimes(1)
    const message = log.mock.calls[0]?.[0] as string
    expect(message).toContain('expected shutdown transition')
    expect(message).toContain('Hermes Desktop is quitting.')
    // No stack frames: the line must not read as a crash in desktop.log.
    expect(message).not.toContain('electron-main.mjs')
    expect(flush).toHaveBeenCalledTimes(1)
  })

  it('marks only stamped errors — a same-message unmarked error still renders its stack', () => {
    const { listeners, log } = harness()

    const lookalike = new Error('Hermes Desktop is quitting.')
    lookalike.stack = 'Error: Hermes Desktop is quitting.\n    at main'

    listeners.get('unhandledRejection')?.(lookalike)

    expect(isExpectedTransition(lookalike)).toBe(false)
    expect(log).toHaveBeenCalledWith(expect.stringContaining('at main'))
  })
})
