import { afterEach, describe, expect, it, vi } from 'vitest'

/**
 * Security verification: ui/message per-card rate limiting.
 *
 * A sandboxed card that fires `ui/message` in a tight loop can trigger
 * unlimited model turns (DoS).  The fix adds a 2-second debounce
 * (MIN_MESSAGE_INTERVAL_MS = 2000) keyed by `debounceKey` so cards don't
 * throttle each other.
 */

const TWO_SECONDS = 2000

describe('mcp-app store security — per-card rate limiting', () => {
  // The store keeps module-level Map state.  Reset before each test.
  afterEach(() => {
    vi.resetModules()
  })

  it('same debounceKey: rapid calls within 2 s are blocked', async () => {
    const { $mcpAppUserMessage, requestMcpAppUserMessage } = await import('./mcp-app')
    let calls = 0
    const unsub = $mcpAppUserMessage.listen(() => { calls++ })

    // Fire 10 calls with the same toolCallId — only the first should pass.
    for (let i = 0; i < 10; i++) {
      requestMcpAppUserMessage(`buy item ${i}`, 'tc_abc')
    }

    expect(calls).toBe(1)
    unsub()
  })

  it('different debounceKeys: independent, neither blocks the other', async () => {
    const { $mcpAppUserMessage, requestMcpAppUserMessage } = await import('./mcp-app')
    let calls = 0
    const unsub = $mcpAppUserMessage.listen(() => { calls++ })

    // Card A and Card B both fire — each should pass independently.
    requestMcpAppUserMessage('from card A', 'tc_a')
    requestMcpAppUserMessage('from card B', 'tc_b')

    expect(calls).toBe(2)
    unsub()
  })

  it('no debounceKey: pass-through (backward compat, no rate limit)', async () => {
    const { $mcpAppUserMessage, requestMcpAppUserMessage } = await import('./mcp-app')
    let calls = 0
    const unsub = $mcpAppUserMessage.listen(() => { calls++ })

    for (let i = 0; i < 10; i++) {
      requestMcpAppUserMessage(`buy item ${i}`)
    }

    // WITHOUT debounceKey, every call passes through.
    expect(calls).toBe(10)
    unsub()
  })

  it('call after real delay passes through for same key', async () => {
    const { $mcpAppUserMessage, requestMcpAppUserMessage } = await import('./mcp-app')
    let calls = 0
    const unsub = $mcpAppUserMessage.listen(() => { calls++ })

    requestMcpAppUserMessage('first', 'tc_delay')
    expect(calls).toBe(1)

    // Wait past the debounce window.
    await new Promise(resolve => setTimeout(resolve, TWO_SECONDS + 100))

    requestMcpAppUserMessage('second', 'tc_delay')
    expect(calls).toBe(2)

    unsub()
  })
})
