import { describe, expect, it } from 'vitest'

import {
  registerAgentTerminalWriter,
  syncAgentTerminalSnapshot,
  writeAgentTerminalChunk
} from './agent-terminal-stream'

// Module state is process-keyed and shared across cases: unique procIds per test.
function capture(procId: string) {
  const writes: Array<[string, boolean]> = []
  const stop = registerAgentTerminalWriter(procId, (chunk, pty) => writes.push([chunk, pty]))

  return { stop, writes }
}

describe('agent terminal PTY flag', () => {
  it('a pipe process is written with pipe discipline', () => {
    const { stop, writes } = capture('pty-flag-pipe')

    writeAgentTerminalChunk('pty-flag-pipe', 'line\n')
    stop()

    expect(writes).toEqual([['line\n', false]])
  })

  it('a live PTY chunk switches the tab to PTY discipline', () => {
    const { stop, writes } = capture('pty-flag-live')

    writeAgentTerminalChunk('pty-flag-live', '\x1b[2;3H\x1b[K\n', true)
    stop()

    expect(writes.at(-1)?.[1]).toBe(true)
  })

  it('the flag from a process.list snapshot reaches the backlog replay of a later-opened tab', () => {
    syncAgentTerminalSnapshot('pty-flag-snapshot', 'frame', true)

    const { stop, writes } = capture('pty-flag-snapshot')

    stop()
    expect(writes).toEqual([['frame', true]])
  })

  it('stays PTY once known, even for chunks from a source that does not carry the flag', () => {
    writeAgentTerminalChunk('pty-flag-sticky', 'a', true)

    const { stop, writes } = capture('pty-flag-sticky')

    syncAgentTerminalSnapshot('pty-flag-sticky', 'ab')
    stop()

    expect(writes.every(([, pty]) => pty)).toBe(true)
  })
})
