import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, describe, expect, it } from 'vitest'

import {
  decideNvidiaEglFallback,
  NVIDIA_GPU_DEATH_REASONS,
  nvidiaEglFallbackMarker,
  nvidiaEglMarkerAfterSuccessfulBoot,
  parseNvidiaDriverMajor,
  parseNvidiaDriverVersion,
  readNvidiaEglMarker,
  shouldRelaunchForNvidiaGpuDeath,
  writeNvidiaEglMarker
} from './linux-nvidia-egl-fallback'

const LINUX = { env: {}, platform: 'linux' as const, isWsl: false, remoteDisplayReason: null }
const PROBE = { ...LINUX, driverMajor: 580, driverVersion: '580.178.04', appVersion: '0.21.5', marker: null }

const tmpDirs: string[] = []

afterEach(() => {
  for (const dir of tmpDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true })
  }
})

function tmpUserData(): string {
  const dir = mkdtempSync(join(tmpdir(), 'nvidia-egl-'))
  tmpDirs.push(dir)

  return dir
}

describe('parseNvidiaDriverMajor / parseNvidiaDriverVersion', () => {
  it('parses the major and full version from /proc/driver/nvidia/version content', () => {
    const text = 'NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.82.09  Mon Jul 21 19:44:16 UTC 2025\n'

    expect(parseNvidiaDriverMajor(text)).toBe(580)
    expect(parseNvidiaDriverVersion(text)).toBe('580.82.09')
  })

  it('returns null for garbage or empty input', () => {
    expect(parseNvidiaDriverMajor('')).toBeNull()
    expect(parseNvidiaDriverMajor('no driver version here')).toBeNull()
    expect(parseNvidiaDriverVersion('')).toBeNull()
    expect(parseNvidiaDriverVersion('no driver version here')).toBeNull()
  })
})

// ─── the behavioral probe (#124255) ───────────────────────────────────────────
//
// The driver-major set gate burned 4-9 CPU cores on healthy 580 hosts whose
// session EGL lands on the iGPU; the same point release kills the GPU process
// on hosts where NVIDIA drives the display. Only a witnessed death may route
// ANGLE through SwiftShader.

describe('decideNvidiaEglFallback — behavioral probe', () => {
  it('boots with hardware GL on a first launch (no marker): the 580 series is probed, not assumed broken', () => {
    const decision = decideNvidiaEglFallback(PROBE)

    expect(decision.enable).toBe(false)
    expect(decision.nextMarker.state).toBe('booting')
  })

  it('a healthy prior boot (ok marker) keeps hardware GL', () => {
    const decision = decideNvidiaEglFallback({ ...PROBE, marker: { state: 'ok' } })

    expect(decision.enable).toBe(false)
    expect(decision.nextMarker.state).toBe('booting')
  })

  it('a witnessed GPU death (fallback marker, same app + driver) engages SwiftShader and stays sticky', () => {
    const marker = nvidiaEglFallbackMarker('0.21.5', '580.178.04')
    const decision = decideNvidiaEglFallback({ ...PROBE, marker })

    expect(decision.enable).toBe(true)
    expect(decision.reason).toContain('witnessed')
    expect(decision.nextMarker).toEqual(marker)
  })

  it('an app update re-probes hardware GL once instead of degrading forever', () => {
    const marker = nvidiaEglFallbackMarker('0.21.4', '580.178.04')
    const decision = decideNvidiaEglFallback({ ...PROBE, marker })

    expect(decision.enable).toBe(false)
    expect(decision.nextMarker.state).toBe('booting')
  })

  it('a driver update re-probes hardware GL once', () => {
    const marker = nvidiaEglFallbackMarker('0.21.5', '580.178.04')
    const decision = decideNvidiaEglFallback({ ...PROBE, marker, driverVersion: '580.182.10' })

    expect(decision.enable).toBe(false)
    expect(decision.nextMarker.state).toBe('booting')
  })

  it('a leftover booting marker engages the fallback: the previous boot died mid-probe', () => {
    // Chromium's "GPU process isn't usable" FATAL abort can kill the process
    // before the child-process-gone handler runs; the unresolved `booting`
    // marker is the evidence it leaves behind.
    const decision = decideNvidiaEglFallback({ ...PROBE, marker: { state: 'booting' } })

    expect(decision.enable).toBe(true)
    expect(decision.reason).toContain('aborted')
    expect(decision.nextMarker.state).toBe('fallback')
  })

  it('the healthy 580 hosts from the issue thread get hardware rendering back', () => {
    // #124255's reporter (Quadro P1000 + Intel UHD 630) and psukez on #123203
    // (940MX + Intel HD 520) both render on hardware with this driver; under
    // the old {580} series gate they were forced onto CPU SwiftShader.
    const healthy = decideNvidiaEglFallback({ ...PROBE, marker: { state: 'ok' } })

    expect(healthy.enable).toBe(false)
  })
})

// ─── the same standing exclusions as before ────────────────────────────────

describe('decideNvidiaEglFallback — exclusions', () => {
  it('stays off on non-linux platforms and without an NVIDIA driver', () => {
    expect(decideNvidiaEglFallback({ ...PROBE, platform: 'darwin' }).enable).toBe(false)
    expect(decideNvidiaEglFallback({ ...PROBE, platform: 'win32' }).enable).toBe(false)
    expect(decideNvidiaEglFallback({ ...PROBE, driverMajor: null, marker: { state: 'booting' } }).enable).toBe(false)
  })

  it('stays off under WSLg and when a remote display already forced software rendering', () => {
    expect(decideNvidiaEglFallback({ ...PROBE, isWsl: true, marker: { state: 'booting' } }).enable).toBe(false)
    expect(
      decideNvidiaEglFallback({ ...PROBE, remoteDisplayReason: 'ssh-session', marker: { state: 'booting' } }).enable
    ).toBe(false)
  })

  it('HERMES_DESKTOP_DISABLE_GPU=0 keeps the GPU on even over a witnessed marker', () => {
    const marker = nvidiaEglFallbackMarker('0.21.5', '580.178.04')

    expect(decideNvidiaEglFallback({ ...PROBE, marker, env: { HERMES_DESKTOP_DISABLE_GPU: '0' } }).enable).toBe(false)
  })

  it('HERMES_DESKTOP_NVIDIA_SWIFTSHADER forces the fallback on without any marker', () => {
    const decision = decideNvidiaEglFallback({
      ...PROBE,
      env: { HERMES_DESKTOP_NVIDIA_SWIFTSHADER: '1' },
      marker: null
    })

    expect(decision.enable).toBe(true)
    expect(decision.reason).toContain('override')
  })

  it('HERMES_DESKTOP_NVIDIA_SWIFTSHADER=0 opts out even over a witnessed marker', () => {
    const marker = nvidiaEglFallbackMarker('0.21.5', '580.178.04')

    expect(
      decideNvidiaEglFallback({ ...PROBE, marker, env: { HERMES_DESKTOP_NVIDIA_SWIFTSHADER: 'off' } }).enable
    ).toBe(false)
  })
})

// ─── the runtime witness ────────────────────────────────────────────────────

describe('shouldRelaunchForNvidiaGpuDeath', () => {
  const BASE = { platform: 'linux' as const, fallbackActive: false, relaunchAttempted: false }

  it('relaunches on a GPU crash', () => {
    expect(shouldRelaunchForNvidiaGpuDeath({ ...BASE, details: { type: 'GPU', reason: 'crashed' } })).toBe(true)
  })

  it('relaunches on a GPU launch failure', () => {
    expect(shouldRelaunchForNvidiaGpuDeath({ ...BASE, details: { type: 'GPU', reason: 'launch-failure' } })).toBe(true)
  })

  it("counts `killed`: the #40077 GPU process died to Chromium's health-check SIGTERM (exit_code=15)", () => {
    expect(NVIDIA_GPU_DEATH_REASONS.has('killed')).toBe(true)
    expect(shouldRelaunchForNvidiaGpuDeath({ ...BASE, details: { type: 'GPU', reason: 'killed' } })).toBe(true)
  })

  it('never relaunches twice in one process, or when the fallback is already active', () => {
    const details = { type: 'GPU', reason: 'crashed' }

    expect(shouldRelaunchForNvidiaGpuDeath({ ...BASE, details, relaunchAttempted: true })).toBe(false)
    expect(shouldRelaunchForNvidiaGpuDeath({ ...BASE, details, fallbackActive: true })).toBe(false)
  })

  it('ignores non-GPU deaths, clean exits, and other platforms', () => {
    expect(shouldRelaunchForNvidiaGpuDeath({ ...BASE, details: { type: 'renderer', reason: 'crashed' } })).toBe(false)
    expect(shouldRelaunchForNvidiaGpuDeath({ ...BASE, details: { type: 'GPU', reason: 'clean-exit' } })).toBe(false)
    expect(shouldRelaunchForNvidiaGpuDeath({ ...BASE, details: null })).toBe(false)
    expect(
      shouldRelaunchForNvidiaGpuDeath({ ...BASE, platform: 'darwin', details: { type: 'GPU', reason: 'crashed' } })
    ).toBe(false)
  })
})

// ─── marker lifecycle ───────────────────────────────────────────────────────

describe('marker persistence', () => {
  it('round-trips through the userData dir', () => {
    const dir = tmpUserData()
    const marker = nvidiaEglFallbackMarker('0.21.5', '580.178.04')

    writeNvidiaEglMarker(dir, marker)

    expect(readNvidiaEglMarker(dir)).toEqual(marker)
  })

  it('returns null for a missing or garbage marker', () => {
    const dir = tmpUserData()

    expect(readNvidiaEglMarker(dir)).toBeNull()
    expect(readNvidiaEglMarker(dir, { readFileSync: () => 'not json' as never })).toBeNull()
  })

  it('a healthy boot marks ok; a fallback boot keeps the sticky marker', () => {
    expect(nvidiaEglMarkerAfterSuccessfulBoot({ fallbackActive: false })).toEqual({ state: 'ok' })

    expect(
      nvidiaEglMarkerAfterSuccessfulBoot({
        fallbackActive: true,
        appVersion: '0.21.5',
        driverVersion: '580.178.04'
      })
    ).toEqual(nvidiaEglFallbackMarker('0.21.5', '580.178.04'))
  })
})
