import { describe, expect, it } from 'vitest'

import {
  registerAgentTerminalWriter,
  syncAgentTerminalSnapshot,
  writeAgentTerminalChunk
} from './agent-terminal-stream'

// The module keeps process-keyed module-level state (backlog/lastSnapshots), so
// every case uses a UNIQUE procId. A tiny xterm emulator tracks what the live
// terminal would show: `\x1bc<rest>` clears and reseeds, anything else appends.
function mountScreen(procId: string): { screen: () => string } {
  let screen = ''

  registerAgentTerminalWriter(procId, (chunk) => {
    if (chunk.startsWith('\x1bc')) {
      screen = chunk.slice(2)
    } else {
      screen += chunk
    }
  })

  return { screen: () => screen }
}

describe('syncAgentTerminalSnapshot', () => {
  it('appends only the snapshot delta past live backlog, not past the stale last snapshot', () => {
    // Register before the first snapshot so the backlog is empty and the screen
    // tracks exactly the delta writes — matches a live tab that mounts, then
    // receives live chunks and a catch-up snapshot.
    const term = mountScreen('dup-proc')

    syncAgentTerminalSnapshot('dup-proc', 'AB')
    // Live chunks advance the backlog but NOT lastSnapshots.
    writeAgentTerminalChunk('dup-proc', 'C')
    writeAgentTerminalChunk('dup-proc', 'D')
    // Catch-up snapshot is a prefix-superset of BOTH the backlog ('ABCD') and the
    // stale last snapshot ('AB'). The backlog delta ('EF') is the correct one;
    // using the last-snapshot delta ('CDEF') re-appends the already-shown 'CD'.
    syncAgentTerminalSnapshot('dup-proc', 'ABCDEF')

    expect(term.screen()).toBe('ABCDEF')
  })

  it('appends the delta for snapshot-only growth', () => {
    const term = mountScreen('grow-proc')

    syncAgentTerminalSnapshot('grow-proc', 'hello')
    syncAgentTerminalSnapshot('grow-proc', 'hello world')

    expect(term.screen()).toBe('hello world')
  })

  it('is a no-op when a snapshot equals already-streamed live output', () => {
    const term = mountScreen('equal-proc')

    writeAgentTerminalChunk('equal-proc', 'hi')
    syncAgentTerminalSnapshot('equal-proc', 'hi')

    expect(term.screen()).toBe('hi')
  })

  it('resets to the new tail when the snapshot is not a prefix-superset of the backlog', () => {
    const term = mountScreen('reset-proc')

    syncAgentTerminalSnapshot('reset-proc', 'abc')
    syncAgentTerminalSnapshot('reset-proc', 'xyz')

    expect(term.screen()).toBe('xyz')
  })
})
