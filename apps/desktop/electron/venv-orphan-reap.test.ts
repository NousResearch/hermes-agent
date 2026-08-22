'use strict'

/**
 * Tests for apps/desktop/electron/venv-orphan-reap.ts
 *
 * Run with: npx vitest run electron/venv-orphan-reap.test.ts
 * (from apps/desktop; wired into npm test:desktop:platforms)
 */

import assert from 'node:assert/strict'

import { describe, it } from 'vitest'

import { findOrphanedVenvHolders, parseProcessSnapshot, selectOrphanedVenvHolders } from './venv-orphan-reap'

const SHIM = 'C:\\Hermes\\venv\\Scripts\\hermes.exe'

// ---------------------------------------------------------------------------
// selectOrphanedVenvHolders
// ---------------------------------------------------------------------------

describe('selectOrphanedVenvHolders', () => {
  it('selects a shim process whose parent is gone', () => {
    const snapshot = [
      { pid: 19676, parentPid: 2764, executablePath: SHIM },
      { pid: 1000, parentPid: 4, executablePath: 'C:\\Windows\\explorer.exe' }
    ]

    assert.deepEqual(selectOrphanedVenvHolders(snapshot, SHIM, 999), [19676])
  })

  it('leaves a shim process whose parent is still alive', () => {
    // A user terminal running `hermes chat` — the terminal is the live parent,
    // and the user can close it themselves. Killing it would destroy their
    // interactive session without asking.
    const snapshot = [
      { pid: 2764, parentPid: 1000, executablePath: 'C:\\Windows\\System32\\cmd.exe' },
      { pid: 19676, parentPid: 2764, executablePath: SHIM }
    ]

    assert.deepEqual(selectOrphanedVenvHolders(snapshot, SHIM, 999), [])
  })

  it('ignores processes from a different Hermes install', () => {
    const other = 'D:\\Other\\Hermes\\venv\\Scripts\\hermes.exe'
    const snapshot = [{ pid: 4242, parentPid: 1, executablePath: other }]

    assert.deepEqual(selectOrphanedVenvHolders(snapshot, SHIM, 999), [])
  })

  it('matches the shim path case-insensitively (Windows paths are)', () => {
    const snapshot = [{ pid: 7, parentPid: 1, executablePath: SHIM.toUpperCase() }]

    assert.deepEqual(selectOrphanedVenvHolders(snapshot, SHIM, 999), [7])
  })

  it('never selects the calling process', () => {
    const snapshot = [{ pid: 555, parentPid: 1, executablePath: SHIM }]

    assert.deepEqual(selectOrphanedVenvHolders(snapshot, SHIM, 555), [])
  })

  it('treats parentPid 0 as no parent', () => {
    const snapshot = [{ pid: 8, parentPid: 0, executablePath: SHIM }]

    assert.deepEqual(selectOrphanedVenvHolders(snapshot, SHIM, 999), [8])
  })

  it('tolerates a null executablePath (access-denied system rows)', () => {
    const snapshot = [
      { pid: 4, parentPid: 0, executablePath: null },
      { pid: 19676, parentPid: 2764, executablePath: SHIM }
    ]

    assert.deepEqual(selectOrphanedVenvHolders(snapshot, SHIM, 999), [19676])
  })

  it('returns an empty list for an empty snapshot rather than throwing', () => {
    assert.deepEqual(selectOrphanedVenvHolders([], SHIM, 999), [])
  })
})

// ---------------------------------------------------------------------------
// parseProcessSnapshot
// ---------------------------------------------------------------------------

describe('parseProcessSnapshot', () => {
  it('parses a JSON array', () => {
    const raw = JSON.stringify([{ ProcessId: 1, ParentProcessId: 0, ExecutablePath: SHIM }])

    assert.deepEqual(parseProcessSnapshot(raw), [{ pid: 1, parentPid: 0, executablePath: SHIM }])
  })

  it('parses the bare object PowerShell 5.1 emits for a single row', () => {
    // ConvertTo-Json collapses a one-element collection to an object; a parser
    // that assumes an array silently sees zero processes and reaps nothing.
    const raw = JSON.stringify({ ProcessId: 1, ParentProcessId: 0, ExecutablePath: SHIM })

    assert.deepEqual(parseProcessSnapshot(raw), [{ pid: 1, parentPid: 0, executablePath: SHIM }])
  })

  it('returns an empty list for empty output', () => {
    assert.deepEqual(parseProcessSnapshot(''), [])
    assert.deepEqual(parseProcessSnapshot('   '), [])
  })

  it('returns an empty list for malformed JSON rather than throwing', () => {
    assert.deepEqual(parseProcessSnapshot('not json'), [])
  })

  it('skips rows without a usable pid', () => {
    const raw = JSON.stringify([
      { ProcessId: null, ParentProcessId: 0, ExecutablePath: SHIM },
      { ProcessId: 2, ParentProcessId: 1, ExecutablePath: SHIM }
    ])

    assert.deepEqual(parseProcessSnapshot(raw), [{ pid: 2, parentPid: 1, executablePath: SHIM }])
  })
})

// ---------------------------------------------------------------------------
// findOrphanedVenvHolders
// ---------------------------------------------------------------------------

describe('findOrphanedVenvHolders', () => {
  it('returns an empty list off Windows without shelling out', async () => {
    let called = false

    const pids = await findOrphanedVenvHolders(SHIM, {
      isWindows: false,
      selfPid: 999,
      execText: async () => {
        called = true

        return ''
      }
    })

    assert.deepEqual(pids, [])
    assert.equal(called, false)
  })

  it('returns the orphaned holders from a real-shaped snapshot', async () => {
    const raw = JSON.stringify([
      { ProcessId: 19676, ParentProcessId: 2764, ExecutablePath: SHIM },
      { ProcessId: 21976, ParentProcessId: 17460, ExecutablePath: SHIM },
      { ProcessId: 30000, ParentProcessId: 1, ExecutablePath: 'C:\\Windows\\explorer.exe' }
    ])

    const pids = await findOrphanedVenvHolders(SHIM, {
      isWindows: true,
      selfPid: 999,
      execText: async () => raw
    })

    assert.deepEqual(pids, [19676, 21976])
  })

  it('returns an empty list when the probe fails', async () => {
    // Best-effort hardening: a broken probe must never become a new way to
    // wedge an update, so it degrades to the pre-existing owned-PID sweep.
    const pids = await findOrphanedVenvHolders(SHIM, {
      isWindows: true,
      selfPid: 999,
      execText: async () => {
        throw new Error('powershell missing')
      }
    })

    assert.deepEqual(pids, [])
  })

  it('reports a probe failure instead of failing silently', async () => {
    const failures: string[] = []

    await findOrphanedVenvHolders(SHIM, {
      isWindows: true,
      selfPid: 999,
      execText: async () => {
        throw new Error('powershell missing')
      },
      onProbeFailure: message => failures.push(message)
    })

    assert.equal(failures.length, 1)
    assert.match(failures[0], /powershell missing/)
  })

  it('reports unusable probe output rather than reporting zero orphans', async () => {
    // A silent [] here is indistinguishable from "the venv is clear", which is
    // exactly the wedge that made the original failure so hard to read.
    const failures: string[] = []

    const pids = await findOrphanedVenvHolders(SHIM, {
      isWindows: true,
      selfPid: 999,
      execText: async () => 'not json at all',
      onProbeFailure: message => failures.push(message)
    })

    assert.deepEqual(pids, [])
    assert.equal(failures.length, 1)
    assert.match(failures[0], /no usable rows/)
  })

  it('stays silent when the probe genuinely finds no orphans', async () => {
    const failures: string[] = []

    const pids = await findOrphanedVenvHolders(SHIM, {
      isWindows: true,
      selfPid: 999,
      execText: async () => JSON.stringify([{ ProcessId: 4, ParentProcessId: 0, ExecutablePath: 'C:\\a.exe' }]),
      onProbeFailure: message => failures.push(message)
    })

    assert.deepEqual(pids, [])
    assert.deepEqual(failures, [])
  })
})
