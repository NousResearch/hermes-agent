import { describe, expect, it } from 'vitest'

import type { HealthEvidence, ScopeEvidence } from '../src/lib/managed-rollout-contract'
import {
  MAX_PROBE_CONCURRENCY,
  buildHealthEvidence,
  isFreshSweep,
  readRemoteLifecycleIdentity,
  runEvidenceSweep
} from './managed-rollout-evidence'

const SHA = 'a'.repeat(40)
const INSTALL = '1'.repeat(32)

function scope(id: string): ScopeEvidence {
  return {
    scopeId: id,
    profile: id,
    restored: true,
    ready: true,
    codeSha: SHA,
    processIdentityVerified: true
  }
}

function healthy(observationId: string, ids: string[] = ['default'], installId = INSTALL): HealthEvidence {
  return buildHealthEvidence({
    observationId,
    observedAt: '2026-09-21T00:00:00.000Z',
    installId,
    checkoutSha: SHA,
    installReady: true,
    markerClear: true,
    receiptCorrelated: true,
    receiptSucceeded: true,
    dependencyReady: true,
    recoveryClear: true,
    scopes: ids.map(scope),
    reasons: []
  })
}

describe('managed rollout evidence', () => {
  it('preserves missing scope knowledge instead of converting it to known empty', () => {
    expect(
      buildHealthEvidence({
        observationId: 'epoch-1',
        observedAt: '2026-09-21T00:00:00.000Z',
        installId: INSTALL,
        checkoutSha: SHA,
        installReady: true,
        markerClear: true,
        receiptCorrelated: true,
        receiptSucceeded: true,
        dependencyReady: true,
        recoveryClear: true,
        scopes: null,
        reasons: []
      }).scopeCapture
    ).toBe('missing')
    expect(
      buildHealthEvidence({
        observationId: 'epoch-1',
        observedAt: '2026-09-21T00:00:00.000Z',
        installId: INSTALL,
        checkoutSha: SHA,
        installReady: true,
        markerClear: true,
        receiptCorrelated: true,
        receiptSucceeded: true,
        dependencyReady: true,
        recoveryClear: true,
        scopes: null,
        reasons: []
      }).scopes
    ).toEqual([])
  })

  it('limits concurrent probes to eight and requires every target in the epoch', async () => {
    let active = 0
    let maxActive = 0
    const targets = Array.from({ length: 10 }, (_, index) => ({
      installId: `${index}`.repeat(32).slice(0, 32),
      requiredScopeIds: ['default'],
      wave: index === 0 ? 0 : 1,
      excluded: false
    }))

    const result = await runEvidenceSweep(
      targets,
      async target => {
        active += 1
        maxActive = Math.max(maxActive, active)
        await new Promise(resolve => setTimeout(resolve, 1))
        active -= 1
        return { health: healthy('epoch-1', ['default'], target.installId) }
      },
      { epochId: 'epoch-1', nowMono: () => 1000, maxConcurrency: MAX_PROBE_CONCURRENCY }
    )

    expect(maxActive).toBeLessThanOrEqual(MAX_PROBE_CONCURRENCY)
    expect(result.ok).toBe(true)
    expect(result.completeScope).toBe(true)
    expect(result.fresh).toBe(true)
    expect(result.observations).toHaveLength(10)
    expect(result.metrics).toMatchObject({
      requestedProbes: 10,
      completedProbes: 10,
      timedOutProbes: 0,
      retryCount: 0
    })
  })

  it('accepts a settled wave and its successor at the 240-probe sweep ceiling', async () => {
    const targets = Array.from({ length: 240 }, (_, index) => ({
      installId: index.toString(16).padStart(32, '0'),
      requiredScopeIds: ['default'],
      wave: index < 120 ? 0 : 1,
      excluded: false
    }))

    const result = await runEvidenceSweep(
      targets,
      async target => ({ health: healthy('epoch-240', ['default'], target.installId) }),
      { epochId: 'epoch-240', nowMono: () => 1000, maxConcurrency: MAX_PROBE_CONCURRENCY }
    )

    expect(result.ok).toBe(true)
    expect(result.observations).toHaveLength(240)
    expect(result.metrics).toEqual({
      requestedProbes: 240,
      completedProbes: 240,
      timedOutProbes: 0,
      retryCount: 0,
      queueDelayMs: 0
    })
  })

  it('rejects stale epoch, missing scope and partial probe evidence', async () => {
    const targets = [
      { installId: INSTALL, requiredScopeIds: ['default'], wave: 0, excluded: false },
      { installId: '2'.repeat(32), requiredScopeIds: ['default'], wave: 1, excluded: false }
    ]

    const stale = await runEvidenceSweep(
      targets,
      async target => ({ health: healthy(target.installId === INSTALL ? 'wrong-epoch' : 'epoch-2') }),
      { epochId: 'epoch-2', nowMono: () => 1000 }
    )
    expect(stale.ok).toBe(false)
    expect(stale.fresh).toBe(false)

    const missing = await runEvidenceSweep(
      targets,
      async target => ({ health: healthy('epoch-2', target.installId === INSTALL ? [] : ['default']) }),
      { epochId: 'epoch-2', nowMono: () => 1000 }
    )
    expect(missing.ok).toBe(false)
    expect(missing.completeScope).toBe(false)

    const partial = await runEvidenceSweep(
      targets,
      async target => {
        if (target.installId === INSTALL) throw new Error('timeout')
        return { health: healthy('epoch-2') }
      },
      { epochId: 'epoch-2', nowMono: () => 1000 }
    )
    expect(partial.ok).toBe(false)
    expect(partial.observations).toHaveLength(1)
  })

  it('bounds a probe that never settles', async () => {
    const result = await runEvidenceSweep(
      [{ installId: INSTALL, requiredScopeIds: [], wave: 0, excluded: false }],
      () => new Promise(() => undefined),
      { epochId: 'epoch-1', nowMono: () => 0, deadlineMs: 5, maxConcurrency: 1 }
    )

    expect(result.ok).toBe(false)
    expect(result.fresh).toBe(false)
    expect(result.errors).toEqual([{ installId: INSTALL, reason: 'probe-timeout' }])
    expect(result.metrics).toMatchObject({
      requestedProbes: 1,
      completedProbes: 0,
      timedOutProbes: 1,
      retryCount: 0
    })
  })

  it('rejects a promotion sweep beyond 120 installations per wave or 240 total probes', async () => {
    const targets = (count: number, wave: number) => Array.from({ length: count }, (_, index) => ({
      installId: `${wave}-${index}`,
      requiredScopeIds: [] as string[],
      wave,
      excluded: false
    }))
    const probe = async () => ({ health: healthy('epoch-1') })
    await expect(runEvidenceSweep(targets(121, 0), probe, { epochId: 'epoch-1' }))
      .rejects.toThrow('sweep-budget-exceeded')
    await expect(runEvidenceSweep([...targets(120, 0), ...targets(120, 1), ...targets(1, 2)], probe,
      { epochId: 'epoch-1' })).rejects.toThrow('sweep-budget-exceeded')
  })

  it('uses the existing lifecycle readers without mutating a remote install', async () => {
    const commands: string[] = []
    const result = await readRemoteLifecycleIdentity({
      exec: async command => {
        commands.push(command)
        if (command.includes('HERMES_HOME')) return '/remote/.hermes\n'
        if (command.includes('install_id')) return `${INSTALL}\n`
        return ''
      }
    })

    expect(result.installId).toBe(INSTALL)
    expect(result.hermesHome).toBe('/remote/.hermes')
    expect(commands.every(command => !/\b(update|install|fetch)\b|rm -f/.test(command))).toBe(true)
  })

  it('expires a completed sweep after the bounded post-sweep freshness window', () => {
    const completed = {
      ok: true,
      epochId: 'epoch-1',
      startedMono: 1000,
      finishedMono: 2000,
      observations: [],
      completeScope: true,
      fresh: true,
      nextAdmissionInstallIds: []
    }

    expect(isFreshSweep(completed, 11_999)).toBe(true)
    expect(isFreshSweep(completed, 12_001)).toBe(false)
  })

  it('fails closed when an epoch exceeds the five-minute sweep deadline', async () => {
    const result = await runEvidenceSweep(
      [{ installId: INSTALL, requiredScopeIds: [], wave: 0, excluded: false }],
      async () => ({ health: healthy('epoch-1', [], INSTALL) }),
      {
        epochId: 'epoch-1',
        nowMono: (() => {
          let calls = 0
          return () => (calls++ === 0 ? 0 : 300_001)
        })()
      }
    )

    expect(result.ok).toBe(false)
    expect(result.fresh).toBe(false)
    expect(result.errors).toEqual([{ installId: INSTALL, reason: 'sweep-deadline-exceeded' }])
    expect(result.metrics).toMatchObject({
      requestedProbes: 1,
      completedProbes: 0,
      timedOutProbes: 1,
      retryCount: 0
    })
  })
})
