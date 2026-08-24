import { describe, expect, it, vi } from 'vitest'

import { describeCrashReason, installCrashForensics } from './crash-forensics'

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

    expect(log).toHaveBeenCalledWith('[main] Uncaught exception: Error: renderer gone\n    at main')
    expect(flush).toHaveBeenCalledTimes(1)
  })

  it('records and synchronously flushes an unhandled rejection', () => {
    const { flush, listeners, log } = harness()

    listeners.get('unhandledRejection')?.('gateway ticket mint failed')

    expect(log).toHaveBeenCalledWith('[main] Unhandled rejection: gateway ticket mint failed')
    expect(flush).toHaveBeenCalledTimes(1)
  })

  it('registers both handlers', () => {
    const { listeners } = harness()

    expect([...listeners.keys()].sort()).toEqual(['uncaughtException', 'unhandledRejection'])
  })
})
