import { readFileSync } from 'node:fs'

import { describe, expect, it } from 'vitest'

import {
  decideLinuxGpuLaunch,
  disableGpuSwitchNeededForReason,
  linuxGpuChildDeathPath,
  linuxGpuFallbackMarker,
  linuxGpuMarkerAfterSuccessfulBoot,
  parseLinuxGpuMarker,
  shouldEngageSilentGpuRetryFallback,
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
    expect(parseLinuxGpuMarker({ state: 'fallback', reason: 'gpu-launch-failure', version: '0.21.5' })).toEqual({
      state: 'fallback',
      reason: 'gpu-launch-failure',
      version: '0.21.5'
    })
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
    expect(decideLinuxGpuLaunch({ ...LINUX, argv: ['--disable-gpu'] }).reason).toBe('already-enabled')
  })
})

// #131055: the GPU marker is written by the same pre-lock launch block as the
// sandbox one, so a burst of second-instance launches promoted it to
// `fallback/gpu-launch-failure` on a host that never failed. The writer half is
// covered in launch-marker-writer.test.ts; here the marker must also stop
// outliving the build that promoted it on a source install.
describe('linux GPU marker build identity (#131055)', () => {
  const buildA = '0.0.0+g357f51c49106@2026-09-28T10:11:12Z'
  const buildB = '0.0.0+gaabbccddeeff@2026-10-02T00:00:00Z'

  const promoted = {
    state: 'fallback' as const,
    reason: 'gpu-launch-failure' as const,
    version: '0.0.0',
    build: buildA
  }

  it('stays sticky within one source build', () => {
    const decision = decideLinuxGpuLaunch({ ...LINUX, marker: promoted, appVersion: '0.0.0', buildIdentity: buildA })

    expect(decision.enable).toBe(true)
    expect(decision.reason).toContain('sticky')
  })

  it('re-probes on the next source build even though the version is still 0.0.0', () => {
    const decision = decideLinuxGpuLaunch({ ...LINUX, marker: promoted, appVersion: '0.0.0', buildIdentity: buildB })

    expect(decision.enable).toBe(false)
    expect(decision.nextMarker).toEqual({ state: 'booting', reprobe: true, bootAborts: 0 })
  })

  it('round-trips the build identity through the marker', () => {
    expect(
      parseLinuxGpuMarker({ state: 'fallback', reason: 'gpu-launch-failure', version: '0.0.0', build: buildA })
    ).toEqual({ state: 'fallback', reason: 'gpu-launch-failure', version: '0.0.0', build: buildA })
  })

  it('records the build identity when a marker enters fallback', () => {
    expect(linuxGpuFallbackMarker('boot-loop', '0.0.0', buildA)).toEqual({
      state: 'fallback',
      reason: 'boot-loop',
      version: '0.0.0',
      build: buildA
    })

    // A real release version is the whole identity — no redundant build field.
    expect(linuxGpuFallbackMarker('boot-loop', '0.21.5', '0.21.5')).toEqual({
      state: 'fallback',
      reason: 'boot-loop',
      version: '0.21.5'
    })
  })

  it('keeps the build identity on a sticky marker written by an older build', () => {
    const decision = decideLinuxGpuLaunch({
      ...LINUX,
      marker: { state: 'fallback', reason: 'gpu-launch-failure', version: '0.21.5' },
      appVersion: '0.21.5',
      buildIdentity: '0.21.5'
    })

    expect(decision.nextMarker.build).toBeUndefined()
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

    expect(shouldRelaunchForLinuxGpuCrash({ ...base, details: { type: 'renderer', reason: 'crashed' } })).toBe(false)
    expect(shouldRelaunchForLinuxGpuCrash({ ...base, details: { type: 'GPU', reason: 'clean-exit' } })).toBe(false)
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
    expect(linuxGpuMarkerAfterSuccessfulBoot({ fallbackActive: true, appVersion: '0.21.5' })).toEqual({
      state: 'fallback',
      reason: 'gpu-launch-failure',
      version: '0.21.5'
    })
  })

  it('round-trips the fallback marker helper', () => {
    expect(linuxGpuFallbackMarker('gpu-launch-failure', '0.21.5')).toEqual({
      state: 'fallback',
      reason: 'gpu-launch-failure',
      version: '0.21.5'
    })
  })
})

describe('linuxGpuChildDeathPath', () => {
  const SIGTERM_DEATH = { type: 'GPU', reason: 'crashed', exitCode: 143, signalName: 'SIGTERM' }

  it('prefers the sandbox ladder on the #121954 SIGTERM signature', () => {
    expect(linuxGpuChildDeathPath({ platform: 'linux', details: SIGTERM_DEATH })).toBe('no-sandbox')
  })

  it('falls to software when the sandbox relaunch was already spent (relapse)', () => {
    expect(
      linuxGpuChildDeathPath({
        platform: 'linux',
        details: SIGTERM_DEATH,
        sandboxRelaunchAttempted: true
      })
    ).toBe('disable-gpu')
    expect(
      linuxGpuChildDeathPath({
        platform: 'linux',
        details: SIGTERM_DEATH,
        alreadyNoSandbox: true
      })
    ).toBe('disable-gpu')
  })

  it('sends non-signature GPU failures straight to software', () => {
    expect(
      linuxGpuChildDeathPath({
        platform: 'linux',
        details: { type: 'GPU', reason: 'launch-failure', exitCode: 1002 }
      })
    ).toBe('disable-gpu')
    expect(
      linuxGpuChildDeathPath({ platform: 'linux', details: { type: 'GPU', reason: 'crashed', exitCode: 139 } })
    ).toBe('disable-gpu')
  })

  it('produces at most one decision: both ladders spent or already on means none', () => {
    expect(
      linuxGpuChildDeathPath({
        platform: 'linux',
        details: SIGTERM_DEATH,
        sandboxRelaunchAttempted: true,
        softwareRelaunchAttempted: true
      })
    ).toBeNull()
    expect(
      linuxGpuChildDeathPath({
        platform: 'linux',
        details: { type: 'GPU', reason: 'crashed' },
        alreadySoftware: true
      })
    ).toBeNull()
  })

  it('ignores non-GPU deaths and other platforms', () => {
    expect(linuxGpuChildDeathPath({ platform: 'linux', details: { type: 'Renderer', reason: 'crashed' } })).toBeNull()
    expect(linuxGpuChildDeathPath({ platform: 'darwin', details: SIGTERM_DEATH })).toBeNull()
    expect(linuxGpuChildDeathPath({ platform: 'linux' })).toBeNull()
  })

  it('requires the full SIGTERM signature for the sandbox path, not just exit 143', () => {
    expect(
      linuxGpuChildDeathPath({
        platform: 'linux',
        details: { type: 'GPU', reason: 'crashed', exitCode: 143, signalName: 'SIGKILL' }
      })
    ).toBe('disable-gpu')
    // #121954's contract (shouldRelaunchForGpuSandboxCrash) checks the
    // signal signature only: a GPU child SIGTERMed by Chromium is its own
    // "never became usable" shutdown — the reason string carries no extra
    // signal (an OOM would arrive as SIGKILL, above).
    expect(
      linuxGpuChildDeathPath({
        platform: 'linux',
        details: { type: 'GPU', reason: 'oom', exitCode: 143, signalName: 'SIGTERM' }
      })
    ).toBe('no-sandbox')
  })
})

describe('disableGpuSwitchNeededForReason', () => {
  it('spawn-blocks the GPU process only for the explicit env override', () => {
    expect(disableGpuSwitchNeededForReason('override (HERMES_DESKTOP_DISABLE_GPU)')).toBe(true)
    expect(disableGpuSwitchNeededForReason('ssh-session')).toBe(false)
    expect(disableGpuSwitchNeededForReason('vnc-session')).toBe(false)
    expect(disableGpuSwitchNeededForReason(null)).toBe(false)
  })
})

describe('shouldEngageSilentGpuRetryFallback', () => {
  it('engages only on Linux, after the grace window, with no GPU child, not already software', () => {
    expect(shouldEngageSilentGpuRetryFallback({ platform: 'linux', gpuChildPresent: false, graceElapsed: true })).toBe(
      true
    )
    expect(shouldEngageSilentGpuRetryFallback({ platform: 'linux', gpuChildPresent: true, graceElapsed: true })).toBe(
      false
    )
    expect(shouldEngageSilentGpuRetryFallback({ platform: 'linux', gpuChildPresent: false, graceElapsed: false })).toBe(
      false
    )
    expect(
      shouldEngageSilentGpuRetryFallback({
        platform: 'linux',
        gpuChildPresent: false,
        graceElapsed: true,
        alreadySoftware: true
      })
    ).toBe(false)
    expect(shouldEngageSilentGpuRetryFallback({ platform: 'darwin', gpuChildPresent: false, graceElapsed: true })).toBe(
      false
    )
  })
})

describe('silent-retry marker and grace window in main.ts', () => {
  const mainSource = readFileSync(new URL('./main.ts', import.meta.url), 'utf8')

  it('passes the launch build identity to the silent-retry marker', () => {
    // A source install reports version 0.0.0 forever, so only the build identity
    // can tell a marker this build wrote from one an earlier build wrote. Without
    // it `launchMarkerNeedsReprobe` answers true for every launch, and the
    // once-per-build re-probe this PR adds never happens on a source install.
    expect(mainSource).toContain(
      "linuxGpuFallbackMarker('gpu-launch-failure', app.getVersion(), LAUNCH_BUILD_IDENTITY)"
    )

    // One level of nesting: `app.getVersion()` closes before the call does.
    const callSites = mainSource.match(/linuxGpuFallbackMarker\((?:[^()]|\([^()]*\))*\)/g) ?? []
    expect(callSites).toHaveLength(2)

    for (const call of callSites) {
      expect(call).toContain('LAUNCH_BUILD_IDENTITY')
    }
  })

  it('waits the documented grace window, not 30 milliseconds', () => {
    expect(mainSource).toContain('setTimeout(checkSilentGpuRetry, LINUX_GPU_SILENT_RETRY_GRACE_S * 1000)')
    expect(mainSource).not.toMatch(/setTimeout\(checkSilentGpuRetry, LINUX_GPU_SILENT_RETRY_GRACE_S\)/)
  })

  it('keeps the silent-retry marker sticky within one source build', () => {
    const sourceInstall = '0.0.0'
    const build = `0.0.0+g0e34eb817214@2026-10-04T06:51:49.148Z`
    const written = linuxGpuFallbackMarker('gpu-launch-failure', sourceInstall, build)

    const sameBuild = decideLinuxGpuLaunch({ ...LINUX, appVersion: sourceInstall, buildIdentity: build, marker: written })
    expect(sameBuild.enable).toBe(true)
    expect(sameBuild.reason).toContain('sticky')

    const nextBuild = decideLinuxGpuLaunch({
      ...LINUX,
      appVersion: sourceInstall,
      buildIdentity: `0.0.0+g0e34eb817214@2026-10-05T06:51:49.148Z`,
      marker: written
    })

    expect(nextBuild.enable).toBe(false)
    expect(nextBuild.nextMarker.reprobe).toBe(true)
  })
})
