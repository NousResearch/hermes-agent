import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { launchBuildIdentity } from './launch-build-identity'
import { createLaunchMarkerWriter } from './launch-marker-writer'
import { nvidiaEglMarkerPath, readNvidiaEglMarker, writeNvidiaEglMarker } from './linux-nvidia-egl-fallback'
import { createNvidiaEglRecovery } from './linux-nvidia-egl-recovery'

let userData: string
const driverVersion = '600.1.2'

const build = (commit = 'a'.repeat(40)) =>
  launchBuildIdentity({
    appVersion: '0.0.0',
    installStamp: { commit, builtAt: '2026-10-23T00:00:00Z', dirty: false }
  })

function launch(overrides: Partial<Parameters<typeof createNvidiaEglRecovery>[0]> = {}, primary = true) {
  const writer = createLaunchMarkerWriter()
  const relaunch = vi.fn()
  const log = vi.fn()

  const recovery = createNvidiaEglRecovery({
    userData,
    driverMajor: 600,
    driverVersion,
    appVersion: '0.0.0',
    buildIdentity: build(),
    platform: 'linux',
    env: {},
    queueMarker: write => writer.queue(write),
    relaunch,
    log,
    ...overrides
  })

  writer.flush(primary)

  return { ...recovery, relaunch, log }
}

beforeEach(() => {
  userData = mkdtempSync(path.join(tmpdir(), 'nvidia-recovery-'))
})
afterEach(() => {
  rmSync(userData, { recursive: true, force: true })
})

describe('NVIDIA persisted recovery lifecycle', () => {
  it('a successful native launch stays native through an ordinary restart without an override', () => {
    const first = launch()
    expect(first.decision.enable).toBe(false)
    first.healthy()
    first.cleanExit()
    const restarted = launch()
    expect(restarted.decision.enable).toBe(false)
    restarted.healthy()
    expect(readNvidiaEglMarker(userData)).toEqual({ state: 'ok' })
  })

  it('secondary launches leave the running primary marker unchanged and cannot force the next restart', () => {
    const first = launch()
    first.healthy()
    const before = readFileSync(nvidiaEglMarkerPath(userData), 'utf8')

    for (let i = 0; i < 3; i++) {
      const secondary = launch({}, false)
      secondary.healthy()
      secondary.cleanExit()
      expect(secondary.gpuDied({ type: 'GPU', reason: 'crashed', exitCode: 1 })).toBe(false)
      expect(readFileSync(nvidiaEglMarkerPath(userData), 'utf8')).toBe(before)
    }

    first.cleanExit()
    expect(launch().decision.enable).toBe(false)
  })

  it('a secondary launch during primary startup cannot persist its tentative fallback', () => {
    const first = launch()
    const before = readFileSync(nvidiaEglMarkerPath(userData), 'utf8')
    const secondary = launch({}, false)
    expect(secondary.decision.enable).toBe(true)
    expect(readFileSync(nvidiaEglMarkerPath(userData), 'utf8')).toBe(before)
    first.healthy()
    first.cleanExit()
    expect(launch().decision.enable).toBe(false)
  })

  it('clean restart or update handoff before first paint does not leave a false aborted probe', () => {
    const first = launch()
    expect(readNvidiaEglMarker(userData)?.state).toBe('booting')
    first.cleanExit()
    expect(first.gpuDied({ type: 'GPU', reason: 'killed', exitCode: 0 })).toBe(false)
    expect(first.relaunch).not.toHaveBeenCalled()
    expect(launch().decision.enable).toBe(false)
  })

  it.each(['booting', 'fallback'] as const)(
    'a source update re-probes an old %s record even when app version stays 0.0.0',
    state => {
      writeNvidiaEglMarker(userData, { state, version: '0.0.0', driverVersion, build: build() })
      const updated = launch({ buildIdentity: build('b'.repeat(40)) })
      expect(updated.decision.enable).toBe(false)
      updated.healthy()
      updated.cleanExit()
      expect(launch({ buildIdentity: build('b'.repeat(40)) }).decision.enable).toBe(false)
    }
  )

  it('recovers existing unqualified state, records the new probe, and retains a genuine failure of that probe', () => {
    writeNvidiaEglMarker(userData, { state: 'booting' })
    expect(launch().decision.enable).toBe(false)
    // No healthy/clean-exit event: this process actually aborted during the new probe.
    const retry = launch()
    expect(retry.decision.enable).toBe(true)
    retry.cleanExit()
    expect(launch().decision.enable).toBe(true)
  })

  it('uses the real Electron launch-failed event, bounds relaunch, and preserves its fallback through teardown', () => {
    const first = launch()
    const details = { type: 'GPU' as const, reason: 'launch-failed' as const, exitCode: 1 }
    expect(first.gpuDied(details)).toBe(true)
    expect(first.gpuDied(details)).toBe(false)
    first.healthy()
    first.cleanExit()
    expect(first.relaunch).toHaveBeenCalledTimes(1)
    expect(readNvidiaEglMarker(userData)?.state).toBe('fallback')
    expect(launch().decision.enable).toBe(true)
  })

  it('does not interpret a GPU child killed by an in-progress update as a new failure', () => {
    const first = launch()
    first.healthy()
    expect(first.gpuDied({ type: 'GPU', reason: 'killed', exitCode: 0 }, true)).toBe(false)
    first.cleanExit()
    expect(first.relaunch).not.toHaveBeenCalled()
    expect(launch().decision.enable).toBe(false)
  })

  it.each(['HERMES_DESKTOP_NVIDIA_SWIFTSHADER', 'HERMES_DESKTOP_DISABLE_GPU'])(
    'honors %s=0 throughout GPU-death handling',
    key => {
      const first = launch({ env: { [key]: '0' } })
      expect(first.gpuDied({ type: 'GPU', reason: 'crashed', exitCode: 1 })).toBe(false)
      first.healthy()
      expect(first.relaunch).not.toHaveBeenCalled()
      expect(readNvidiaEglMarker(userData)?.state).toBe('ok')
    }
  )

  it('keeps an explicitly forced SwiftShader launch in software after successful startup and exit', () => {
    const first = launch({ env: { HERMES_DESKTOP_NVIDIA_SWIFTSHADER: '1' } })
    expect(first.decision.enable).toBe(true)
    first.healthy()
    first.cleanExit()
    expect(readNvidiaEglMarker(userData)?.state).toBe('fallback')
  })
})
