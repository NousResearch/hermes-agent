import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it, vi } from 'vitest'

import { retirePooledRemoteAfterDispatchProbe } from './remote-liveness'

const here = path.dirname(fileURLToPath(import.meta.url))
const mainSource = fs.readFileSync(path.join(here, 'main.ts'), 'utf8').replace(/\r\n/g, '\n')

function dispatchProbeRetireBody(source: string): string {
  const helperIdx = source.indexOf('retirePooledRemoteAfterDispatchProbe(')
  const logIdx = source.indexOf('failed its dispatch probe')
  const anchor = helperIdx >= 0 ? helperIdx : logIdx

  expect(anchor).toBeGreaterThan(-1)

  const retireIdx = source.lastIndexOf('retire:', anchor)

  expect(retireIdx).toBeGreaterThan(-1)

  const end = source.indexOf('\n      })', anchor)

  return source.slice(retireIdx, end === -1 ? anchor + 1_200 : end)
}

// The production change lives in a `retire:` callback inside a 14k-line module
// that cannot be imported under test, so the wiring is asserted against the
// source the same way `pool-stop` / `pool-eviction` assert theirs.
describe('dispatch-probe retire wiring in main.ts', () => {
  it('stops the failed pool entry and cancels ssh bootstrap without touching the ssh scope', () => {
    const body = dispatchProbeRetireBody(mainSource)

    expect(body).toContain('stopPoolBackend')
    expect(body).toContain('cancelAndWait')
    expect(body).not.toMatch(/teardownSshConnection\(/)
  })

  // The multi-strike path is what may retire a scope, and it has to keep that
  // power — otherwise a genuinely dead host is never cleaned up.
  it('leaves the post-resume sweep free to tear a dead scope down', () => {
    const resumeIdx = mainSource.indexOf('function revalidateSuspectPoolAfterResume')

    expect(resumeIdx).toBeGreaterThan(-1)
    expect(mainSource.slice(resumeIdx, resumeIdx + 2_000)).toMatch(/teardownSshConnection\(/)
  })
})

describe('retirePooledRemoteAfterDispatchProbe', () => {
  it('stops the pool backend and cancels an in-flight ssh bootstrap', async () => {
    const entry = { connectionPromise: Promise.resolve({}) }
    const stopPoolBackend = vi.fn(async () => undefined)
    const cancelSshBootstrap = vi.fn(async () => undefined)

    await retirePooledRemoteAfterDispatchProbe({
      cancelSshBootstrap,
      currentEntry: () => entry,
      error: new Error('read ECONNRESET'),
      expectedEntry: entry,
      key: 'conn:homelab::atlas',
      log: vi.fn(),
      sourceKind: 'ssh',
      stopPoolBackend
    })

    expect(stopPoolBackend).toHaveBeenCalledOnce()
    expect(stopPoolBackend).toHaveBeenCalledWith('conn:homelab::atlas')
    expect(cancelSshBootstrap).toHaveBeenCalledOnce()
    expect(cancelSshBootstrap).toHaveBeenCalledWith('conn:homelab::atlas')
  })

  it('still stops a non-ssh pool entry, without an ssh bootstrap to cancel', async () => {
    const entry = { connectionPromise: Promise.resolve({}) }
    const stopPoolBackend = vi.fn(async () => undefined)
    const cancelSshBootstrap = vi.fn(async () => undefined)

    await retirePooledRemoteAfterDispatchProbe({
      cancelSshBootstrap,
      currentEntry: () => entry,
      error: new Error('connect ECONNREFUSED'),
      expectedEntry: entry,
      key: 'conn:cloud::default',
      log: vi.fn(),
      sourceKind: 'remote',
      stopPoolBackend
    })

    expect(stopPoolBackend).toHaveBeenCalledWith('conn:cloud::default')
    expect(cancelSshBootstrap).not.toHaveBeenCalled()
  })

  it('does not stop a pool entry that another caller already replaced', async () => {
    const stale = { id: 'stale' }
    const next = { id: 'next' }
    const stopPoolBackend = vi.fn(async () => undefined)
    const cancelSshBootstrap = vi.fn(async () => undefined)

    await retirePooledRemoteAfterDispatchProbe({
      cancelSshBootstrap,
      currentEntry: () => next,
      error: new Error('late probe'),
      expectedEntry: stale,
      key: 'conn:homelab::atlas',
      log: vi.fn(),
      sourceKind: 'ssh',
      stopPoolBackend
    })

    expect(stopPoolBackend).not.toHaveBeenCalled()
    expect(cancelSshBootstrap).not.toHaveBeenCalled()
  })
})
