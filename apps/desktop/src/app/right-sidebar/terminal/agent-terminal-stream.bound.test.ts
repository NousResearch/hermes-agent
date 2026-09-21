import { describe, expect, it } from 'vitest'

import {
  registerAgentTerminalWriter,
  syncAgentTerminalSnapshot,
  writeAgentTerminalChunk
} from './agent-terminal-stream'

// Module-level state is process-keyed and shared across cases, so every test uses
// unique procIds.
const CHUNK = 'x'.repeat(64_000)

function runFinishedProc(id: string, chunks = 6): void {
  for (let i = 0; i < chunks; i++) {
    writeAgentTerminalChunk(id, CHUNK)
  }
}

describe('agent terminal retention is bounded', () => {
  it('a finished process does not pin its backlog for the life of the renderer', () => {
    // 200 background commands, none mounted — the shape that grew to ~93 MB.
    for (let i = 0; i < 200; i++) {
      runFinishedProc(`bound-${i}`)
    }

    // The oldest must be gone: reopening it replays nothing until the registry
    // snapshot re-seeds it.
    let replayed = ''
    const stop = registerAgentTerminalWriter('bound-0', chunk => (replayed += chunk))

    stop()
    expect(replayed).toBe('')
  })

  it('keeps the most recent process replayable', () => {
    for (let i = 0; i < 200; i++) {
      runFinishedProc(`recent-${i}`)
    }

    let replayed = ''
    const stop = registerAgentTerminalWriter('recent-199', chunk => (replayed += chunk))

    stop()
    expect(replayed.length).toBeGreaterThan(0)
  })

  it('never evicts a process whose terminal is mounted', () => {
    // A long-lived mounted tab, then a flood of unrelated finished processes.
    const mounted = 'mounted-proc'

    registerAgentTerminalWriter(mounted, () => {})
    runFinishedProc(mounted)

    for (let i = 0; i < 200; i++) {
      runFinishedProc(`flood-${i}`)
    }

    // Still replayable: on-screen state is not a cache.
    let replayed = ''
    const stop = registerAgentTerminalWriter(mounted, chunk => (replayed += chunk))

    stop()
    expect(replayed.length).toBeGreaterThan(0)
  })

  it('recency is the LAST write, not the first', () => {
    // Small chunks so the CHARACTER ceiling never fires: this case is purely about
    // the entry ceiling and the order eviction walks. `backlog` insertion order is
    // the LRU clock, and a plain re-set leaves a key in its ORIGINAL slot — so
    // without the delete-then-set refresh, touching a process would not protect it.
    const line = 'x'.repeat(64)

    for (let i = 0; i < 20; i++) {
      writeAgentTerminalChunk(`age-${i}`, `age-${i}-MARK ${line}\n`)
    }

    // Touch the OLDEST while it is still resident, then overflow the ceiling.
    writeAgentTerminalChunk('age-0', 'TOUCHED\n')

    for (let i = 20; i < 32; i++) {
      writeAgentTerminalChunk(`age-${i}`, `age-${i}-MARK ${line}\n`)
    }

    const replay = (proc: string) => {
      let out = ''
      const stop = registerAgentTerminalWriter(proc, chunk => (out += chunk))

      stop()

      return out
    }

    // The touched process kept its ORIGINAL content (it was never dropped and
    // re-created); its untouched neighbour from the same era is gone.
    expect(replay('age-0')).toContain('age-0-MARK')
    expect(replay('age-1')).toBe('')
  })

  it('an evicted process does not diff against a tail it no longer has', () => {
    // `lastSnapshots` is the delta fence for `backlog`. If eviction dropped the
    // backlog but KEPT the fence, the next snapshot takes the `startsWith(previous)`
    // path and writes only the delta — so the screen shows `BBB` where the process
    // actually printed `AAABBB`.
    const proc = 'fence-proc'

    writeAgentTerminalChunk(proc, 'AAA')
    syncAgentTerminalSnapshot(proc, 'AAA')

    for (let i = 0; i < 200; i++) {
      runFinishedProc(`fence-flood-${i}`)
    }

    let screen = ''

    const stop = registerAgentTerminalWriter(proc, chunk => {
      screen = chunk.startsWith('\x1bc') ? chunk.slice(2) : screen + chunk
    })

    syncAgentTerminalSnapshot(proc, 'AAABBB')
    stop()
    expect(screen).toContain('AAABBB')
  })

  it('an evicted process re-seeds cleanly from the registry snapshot', () => {
    // Eviction drops backlog AND its lastSnapshots delta fence together, so the
    // next snapshot must reset the screen rather than diff against a missing tail.
    for (let i = 0; i < 200; i++) {
      runFinishedProc(`reseed-${i}`)
    }

    let screen = ''

    const stop = registerAgentTerminalWriter('reseed-0', chunk => {
      screen = chunk.startsWith('\x1bc') ? chunk.slice(2) : screen + chunk
    })

    syncAgentTerminalSnapshot('reseed-0', 'recovered output')
    stop()
    expect(screen).toContain('recovered output')
  })
})
