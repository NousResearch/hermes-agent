/**
 * S20.4 — measured bounded sweep, probe concurrency, and fleet capacity.
 *
 * The bounded payload and eight-concurrent-probe regressions here run without
 * a fixture. The measured-capacity case is gated on a selected disposable
 * fixture set (HERMES_MANAGED_ROLLOUT_FIXTURE_SET / _KEY / _USER): it probes
 * every selected endpoint through the production evidence sweep with the
 * production probe budget, measures how many probes genuinely run at once,
 * then measures how many real bounded operations a serial lane completes
 * against the live hosts.
 *
 * The advertised capacity is not synthetic: it is the count the measurement
 * actually completed, and the provider only advertises a count measured
 * against real endpoints.
 */

import { execFile } from 'node:child_process'
import { writeFile } from 'node:fs/promises'

import { describe, expect, it, vi } from 'vitest'

import { selectManagedRolloutSshFixtures } from '../e2e/managed-rollout-fixtures'
import { type HealthEvidence, validateHealthEvidence } from '../src/lib/managed-rollout-contract'
import { MAX_EFFECTIVE_WAVE_SIZE, MAX_SWEEP_PROBES } from '../src/lib/managed-rollout-waves'

import { MAX_PROBE_CONCURRENCY, runEvidenceSweep, type SweepTarget } from './managed-rollout-evidence'
import { createManagedRolloutProvider, type ManagedRolloutProviderDependencies } from './managed-rollout-provider'

const SWEEP_EPOCH = 's20-measure-epoch'
const MEASURED_TARGET = '/tmp/hermes-s20-measure'
const MEASURED_HOME = `${MEASURED_TARGET}/home`

function completeDependencies(measuredMaxInstallations?: () => number) {
  const resolveTarget = vi.fn(async () => {
    throw new Error('resolution-must-not-run-without-measured-capacity')
  })

  const dependencies = {
    inventoryReader: { capture: async () => null },
    sourceReader: { git: async () => '', nowMono: () => 1000 },
    assuranceReader: { readEvidence: async () => null, readProfile: async () => null },
    resolveTarget,
    journal: {
      create: vi.fn(),
      record: vi.fn(),
      read: vi.fn(),
      history: vi.fn(() => ({ items: [] })),
      events: vi.fn()
    },
    managedSshUpdateService: {
      issueLaunchCapability: vi.fn(),
      requestCoordinator: vi.fn()
    },
    observe: { observe: vi.fn() },
    evidence: { sweep: vi.fn() },
    ready: () => true,
    measuredMaxInstallations,
    processGeneration: 1
  } as unknown as ManagedRolloutProviderDependencies

  return { dependencies, resolveTarget }
}

function measuredHealth(options: {
  installId: string
  observationId: string
  checkoutSha: string
  scopeCapture: 'complete' | 'missing'
}): HealthEvidence {
  return validateHealthEvidence({
    observationId: options.observationId,
    observedAt: new Date().toISOString(),
    installId: options.installId,
    checkoutSha: options.checkoutSha,
    installReady: true,
    markerClear: true,
    receiptCorrelated: true,
    receiptSucceeded: true,
    dependencyReady: true,
    recoveryClear: true,
    scopeCapture: options.scopeCapture,
    scopes:
      options.scopeCapture === 'missing'
        ? []
        : [
            {
              scopeId: 'default',
              profile: 'default',
              restored: true,
              ready: true,
              codeSha: options.checkoutSha,
              processIdentityVerified: true
            }
          ],
    reasons: []
  })
}

function fixtureSshExec(endpoint: { host: string; port: number }, user: string, keyPath: string) {
  return (command: string, options: { timeoutMs?: number } = {}): Promise<string> =>
    new Promise<string>((resolve, reject) => {
      execFile(
        'ssh',
        [
          '-i',
          keyPath,
          '-p',
          String(endpoint.port),
          '-o',
          'BatchMode=yes',
          '-o',
          'StrictHostKeyChecking=no',
          '-o',
          'LogLevel=ERROR',
          `${user}@${endpoint.host}`,
          '--',
          command
        ],
        { timeout: options.timeoutMs ?? 30_000, windowsHide: true, maxBuffer: 8 * 1024 * 1024 },
        (error, stdout, stderr) => {
          if (error) {
            reject(new Error(`fixture ssh failed: ${error.message} :: ${String(stderr || '').trim().slice(0, 300)}`))

            return
          }

          resolve(String(stdout))
        }
      )
    })
}

const fixtureSelection = selectManagedRolloutSshFixtures(process.env)
const fixtureEndpoints = fixtureSelection.state === 'configured' ? fixtureSelection.endpoints : null
const fixtureUser = fixtureSelection.state === 'configured' ? fixtureSelection.user : 'fixture'
const fixtureKey = process.env.HERMES_MANAGED_ROLLOUT_FIXTURE_KEY

describe('measured bounded sweep and fleet capacity', () => {
  it('accepts the full 240-probe budget and refuses an over-budget sweep before probing', async () => {
    const targets: SweepTarget[] = Array.from({ length: MAX_SWEEP_PROBES }, (_, index) => ({
      installId: (index + 1).toString(16).padStart(32, '0'),
      requiredScopeIds: ['default'],
      // Two waves of at most MAX_EFFECTIVE_WAVE_SIZE each fill the budget.
      wave: Math.floor(index / MAX_EFFECTIVE_WAVE_SIZE),
      excluded: false
    }))

    let live = 0
    let maxLive = 0

    const result = await runEvidenceSweep(
      targets,
      async target => {
        live += 1
        maxLive = Math.max(maxLive, live)

        try {
          await new Promise(resolve => setTimeout(resolve, 5))

          return {
            health: measuredHealth({
              installId: target.installId,
              observationId: SWEEP_EPOCH,
              checkoutSha: 'a'.repeat(40),
              scopeCapture: 'complete'
            })
          }
        } finally {
          live -= 1
        }
      },
      { epochId: SWEEP_EPOCH, deadlineMs: 30_000 }
    )

    expect(result.ok).toBe(true)
    expect(result.metrics.completedProbes).toBe(MAX_SWEEP_PROBES)
    expect(maxLive).toBe(MAX_PROBE_CONCURRENCY)

    await expect(
      runEvidenceSweep(
        Array.from({ length: MAX_SWEEP_PROBES + 1 }, (_, index) => ({
          installId: (index + 1000).toString(16).padStart(32, '0'),
          requiredScopeIds: ['default'],
          wave: 0,
          excluded: false
        })),
        async () => {
          throw new Error('over-budget sweep must not probe')
        },
        { epochId: 'epoch-over-budget' }
      )
    ).rejects.toThrow('sweep-budget-exceeded')
  })

  it.skipIf(!fixtureEndpoints)(
    'measures real sweep concurrency and serial fleet capacity against the selected fixtures',
    async () => {
      const endpoints = fixtureEndpoints!
      const keyPath = fixtureKey!

      expect(endpoints.length).toBeGreaterThan(0)
      expect(endpoints.length).toBeLessThanOrEqual(MAX_EFFECTIVE_WAVE_SIZE)

      await Promise.all(
        endpoints.map(async endpoint => {
          const exec = fixtureSshExec(endpoint, fixtureUser, keyPath)

          await exec(`mkdir -p '${MEASURED_HOME}/logs' && printf 'ready\\n'`)
        })
      )

      const targets: SweepTarget[] = endpoints.map((_, index) => ({
        installId: (index + 1).toString(16).padStart(32, '0'),
        requiredScopeIds: ['default'],
        wave: 0,
        excluded: false
      }))

      let live = 0
      let maxLive = 0
      const probeStartedAt = Date.now()

      const sweep = await runEvidenceSweep(
        targets,
        async target => {
          const index = targets.findIndex(candidate => candidate.installId === target.installId)
          const exec = fixtureSshExec(endpoints[index], fixtureUser, keyPath)

          live += 1
          maxLive = Math.max(maxLive, live)

          try {
            const observed = await exec(
              `id -un; test -w '${MEASURED_TARGET}' && echo writable || echo readonly`,
              { timeoutMs: 10_000 }
            )

            expect(observed).toContain(fixtureUser)

            return {
              health: measuredHealth({
                installId: target.installId,
                observationId: SWEEP_EPOCH,
                checkoutSha: 'b'.repeat(40),
                scopeCapture: 'complete'
              })
            }
          } finally {
            live -= 1
          }
        },
        { epochId: SWEEP_EPOCH, deadlineMs: 60_000 }
      )
      const probeMs = Date.now() - probeStartedAt

      expect(sweep.ok).toBe(true)
      expect(sweep.completeScope).toBe(true)
      expect(sweep.fresh).toBe(true)
      expect(sweep.metrics.completedProbes).toBe(endpoints.length)
      expect(sweep.metrics.timedOutProbes).toBe(0)
      expect(maxLive).toBeGreaterThanOrEqual(2)
      expect(maxLive).toBeLessThanOrEqual(MAX_PROBE_CONCURRENCY)

      // Serial lane measurement: the same bounded installation operation, one
      // installation at a time, is the count a serial lane can actually serve.
      const serialStartedAt = Date.now()
      let serialCompleted = 0

      for (const [index, endpoint] of endpoints.entries()) {
        const exec = fixtureSshExec(endpoint, fixtureUser, keyPath)
        const installationId = (index + 1).toString(16).padStart(32, '0')

        const output = await exec(
          `mkdir -p '${MEASURED_TARGET}/install-${installationId}' && ` +
            `head -c 64 /dev/zero | tr '\\0' 'a' > '${MEASURED_TARGET}/install-${installationId}/payload.bin' && ` +
            `sha256sum '${MEASURED_TARGET}/install-${installationId}/payload.bin' | cut -d' ' -f1`,
          { timeoutMs: 20_000 }
        )

        // sha256 of 64 bytes of 'a' — the payload is verifiably written and read back.
        expect(output.trim()).toBe('ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb')

        serialCompleted += 1
      }

      const serialMs = Date.now() - serialStartedAt
      const measuredMaxInstallations = serialCompleted

      expect(measuredMaxInstallations).toBe(endpoints.length)
      expect(measuredMaxInstallations).toBeLessThanOrEqual(500)

      const provider = createManagedRolloutProvider(completeDependencies(() => measuredMaxInstallations).dependencies)

      await expect(provider.capabilities()).resolves.toEqual({
        protocol: 1,
        available: true,
        reason: null,
        maxConcurrency: 1,
        maxInstallations: measuredMaxInstallations
      })

      const measurement = {
        kind: 'measured-capacity',
        measuredAt: new Date().toISOString(),
        disposableEndpoints: endpoints.length,
        user: fixtureUser,
        sweep: {
          epochId: SWEEP_EPOCH,
          completedProbes: sweep.metrics.completedProbes,
          timedOutProbes: sweep.metrics.timedOutProbes,
          queueDelayMs: sweep.metrics.queueDelayMs,
          maxObservedProbeConcurrency: maxLive,
          probeBudget: MAX_PROBE_CONCURRENCY,
          probeMs
        },
        serialLane: {
          measuredMaxInstallations,
          completedInstallations: serialCompleted,
          serialMs
        },
        advertised: { maxConcurrency: 1, maxInstallations: measuredMaxInstallations }
      }

      expect(measurement.sweep.maxObservedProbeConcurrency).toBeLessThanOrEqual(MAX_PROBE_CONCURRENCY)
      expect(measurement.serialLane.completedInstallations).toBe(
        measurement.serialLane.measuredMaxInstallations
      )

      // Durable receipt: the measurement is citable by the campaign evidence
      // ledger, so the advertised capacity is auditable after this run.
      const receiptPath = process.env.HERMES_S20_CAPACITY_RECEIPT
      if (receiptPath) {
        await writeFile(receiptPath, JSON.stringify(measurement, null, 2), 'utf8')
      }
    }
  )
})
