import { describe, expect, it } from 'vitest'

import {
  decideLinuxGpuLaunch,
  linuxGpuFallbackMarker,
  linuxGpuMarkerAfterSuccessfulBoot,
  parseLinuxGpuMarker,
  shouldRelaunchForLinuxGpuCrash
} from './linux-gpu-fallback'

const LINUX = {
  platform: 'linux' as const,
  argv: [] as string[],
  env: {} as NodeJS.ProcessEnv,
  marker: null,
  appVersion: '0.21.5',
  remoteDisplayReason: null as string | null,
  nvidiaFallbackActive: false
}

describe('parseLinuxGpuMarker', () => {
  it('accepts a well-formed fallback marker', () => {
    expect(
      parseLinuxGpuMarker({ state: 'fallback', reason: 'gpu-launch-failure', version: '0.21.5' })
    ).toEqual({ state: 'fallback', reason: 'gpu-launch-failure', version: '0.21.5' })
  })

  it('rejects garbage', () => {
    expect(parseLinuxGpuMarker(null)).toBeNull()
    expect(parseLinuxGpuMarker({ state: 'nope' })).toBeNull()
    expect(parseLinuxGpuMarker('fallback')).toBeNull()
  })
})

describe('decideLinuxGpuLaunch', () => {
  it('stays off on a clean first boot and records booting', () => {
    const decision = decideLinuxGpuLaunch(LINUX)

    expect(decision.enable).toBe(false)
    expect(decision.nextMarker.state).toBe('booting')
  })

  it('stays off on non-linux platforms', () => {
    const decision = decideLinuxGpuLaunch({ ...LINUX, platform: 'darwin' })

    expect(decision.enable).toBe(false)
  })

  it('re-engages software rendering from a sticky fallback marker', () => {
    const decision = decideLinuxGpuLaunch({
      ...LINUX,
      marker: { state: 'fallback', reason: 'gpu-launch-failure', version: '0.21.5' }
    })

    expect(decision.enable).toBe(true)
    expect(decision.reason).toContain('sticky')
    expect(decision.nextMarker.state).toBe('fallback')
  })

  it('re-probes the GPU once after an app update instead of degrading forever', () => {
    const decision = decideLinuxGpuLaunch({
      ...LINUX,
      marker: { state: 'fallback', reason: 'gpu-launch-failure', version: '0.21.4' }
    })

    expect(decision.enable).toBe(false)
    expect(decision.nextMarker).toMatchObject({ state: 'booting', reprobe: true })
  })

  it('engages after two consecutive aborted boots, not one', () => {
    const first = decideLinuxGpuLaunch({ ...LINUX, marker: { state: 'booting' } })

    expect(first.enable).toBe(false)
    expect(first.nextMarker).toMatchObject({ state: 'booting', bootAborts: 1 })

    const second = decideLinuxGpuLaunch({
      ...LINUX,
      marker: { state: 'booting', bootAborts: 1 }
    })

    expect(second.enable).toBe(true)
    expect(second.reason).toContain('boot-loop')
  })

  it('HERMES_DESKTOP_DISABLE_GPU=0 keeps the GPU on and clears the stale fallback marker', () => {
    const decision = decideLinuxGpuLaunch({
      ...LINUX,
      env: { HERMES_DESKTOP_DISABLE_GPU: '0' },
      marker: { state: 'fallback', reason: 'gpu-launch-failure', version: '0.21.5' }
    })

    expect(decision.enable).toBe(false)
    // The override boot is the recovery path the file documents: it must not leave the sticky
    // marker behind, or the next override-free launch re-engages software rendering from it
    // and only an app-version change would ever clear it.
    expect(decision.nextMarker).toEqual({ state: 'booting' })
  })

  it('keeps the relaunch path sticky when --disable-gpu is already on argv', () => {
    // Same marker, different entry point: the one-shot relaunch writes `fallback` and re-execs
    // with --disable-gpu, so that boot must preserve it — clearing it here would re-arm the
    // GPU-child retry loop this fallback exists to stop.
    const decision = decideLinuxGpuLaunch({
      ...LINUX,
      argv: ['--disable-gpu'],
      marker: { state: 'fallback', reason: 'gpu-launch-failure', version: '0.21.5' }
    })

    expect(decision.reason).toBe('already-enabled')
    expect(decision.nextMarker).toMatchObject({
      state: 'fallback',
      reason: 'gpu-launch-failure',
      version: '0.21.5'
    })
  })

  it('stays off when software rendering is already forced', () => {
    expect(decideLinuxGpuLaunch({ ...LINUX, remoteDisplayReason: 'ssh-session' }).enable).toBe(false)
    expect(decideLinuxGpuLaunch({ ...LINUX, nvidiaFallbackActive: true }).enable).toBe(false)
    expect(
      decideLinuxGpuLaunch({ ...LINUX, argv: ['--disable-gpu'] }).reason
    ).toBe('already-enabled')
  })
})

describe('shouldRelaunchForLinuxGpuCrash', () => {
  it('relaunches once on a GPU launch failure (error_code=1002 class)', () => {
    expect(
      shouldRelaunchForLinuxGpuCrash({
        platform: 'linux',
        details: { type: 'GPU', reason: 'launch-failure' },
        alreadySoftware: false,
        relaunchAttempted: false
      })
    ).toBe(true)
  })

  it('relaunches once on a GPU crash (SIGTRAP class)', () => {
    expect(
      shouldRelaunchForLinuxGpuCrash({
        platform: 'linux',
        details: { type: 'gpu', reason: 'crashed' },
        alreadySoftware: false,
        relaunchAttempted: false
      })
    ).toBe(true)
  })

  it('never relaunches twice in one process', () => {
    expect(
      shouldRelaunchForLinuxGpuCrash({
        platform: 'linux',
        details: { type: 'GPU', reason: 'launch-failure' },
        alreadySoftware: false,
        relaunchAttempted: true
      })
    ).toBe(false)
  })

  it('ignores non-GPU deaths and clean exits', () => {
    const base = { platform: 'linux' as const, alreadySoftware: false, relaunchAttempted: false }

    expect(
      shouldRelaunchForLinuxGpuCrash({ ...base, details: { type: 'renderer', reason: 'crashed' } })
    ).toBe(false)
    expect(
      shouldRelaunchForLinuxGpuCrash({ ...base, details: { type: 'GPU', reason: 'clean-exit' } })
    ).toBe(false)
    expect(shouldRelaunchForLinuxGpuCrash({ ...base, details: null })).toBe(false)
  })

  it('stays off when software rendering is already active or off linux', () => {
    expect(
      shouldRelaunchForLinuxGpuCrash({
        platform: 'linux',
        details: { type: 'GPU', reason: 'crashed' },
        alreadySoftware: true,
        relaunchAttempted: false
      })
    ).toBe(false)
    expect(
      shouldRelaunchForLinuxGpuCrash({
        platform: 'win32',
        details: { type: 'GPU', reason: 'crashed' },
        alreadySoftware: false,
        relaunchAttempted: false
      })
    ).toBe(false)
  })
})

describe('linuxGpuMarkerAfterSuccessfulBoot', () => {
  it('marks a clean boot ok', () => {
    expect(linuxGpuMarkerAfterSuccessfulBoot({ fallbackActive: false })).toEqual({ state: 'ok' })
  })

  it('keeps the sticky fallback after a software-rendered boot', () => {
    expect(
      linuxGpuMarkerAfterSuccessfulBoot({ fallbackActive: true, appVersion: '0.21.5' })
    ).toEqual({ state: 'fallback', reason: 'gpu-launch-failure', version: '0.21.5' })
  })

  it('round-trips the fallback marker helper', () => {
    expect(linuxGpuFallbackMarker('gpu-launch-failure', '0.21.5')).toEqual({
      state: 'fallback',
      reason: 'gpu-launch-failure',
      version: '0.21.5'
    })
  })
})
