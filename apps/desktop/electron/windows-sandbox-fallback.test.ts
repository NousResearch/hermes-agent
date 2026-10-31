import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import {
  ALL_APPLICATION_PACKAGES_SID,
  alreadyHasNoSandbox,
  BOOT_ABORTS_BEFORE_FALLBACK,
  buildIcaclsGrantArgs,
  BOOT_INSTANT_TOLERANCE_SECONDS,
  buildNoSandboxRelaunchArgs,
  currentBootIdentity,
  decideWindowsSandboxLaunch,
  fallbackMarker,
  GPU_CHILD_SANDBOX_SIGTERM_EXIT,
  grantAllApplicationPackagesAcl,
  isSameBootWitness,
  isWindowsSandboxBreakpointExit,
  markerAfterSuccessfulBoot,
  parseSandboxMarker,
  readSandboxMarker,
  sandboxMarkerPath,
  shouldAttemptAclRepair,
  shouldRelaunchForGpuSandboxCrash,
  shouldRelaunchForRendererSandboxCrashLoop,
  WINDOWS_SANDBOX_BREAKPOINT_EXIT,
  WINDOWS_SANDBOX_MARKER_FILENAME,
  writeSandboxMarker
} from './windows-sandbox-fallback'

test('isWindowsSandboxBreakpointExit recognizes signed and unsigned STATUS_BREAKPOINT', () => {
  assert.equal(isWindowsSandboxBreakpointExit(WINDOWS_SANDBOX_BREAKPOINT_EXIT), true)
  assert.equal(isWindowsSandboxBreakpointExit(-2147483645), true)
  assert.equal(isWindowsSandboxBreakpointExit(0x80000003), true)
  assert.equal(isWindowsSandboxBreakpointExit(1), false)
  assert.equal(isWindowsSandboxBreakpointExit('nope'), false)
})

test('alreadyHasNoSandbox honors argv and ELECTRON_DISABLE_SANDBOX', () => {
  assert.equal(alreadyHasNoSandbox(['--foo', '--no-sandbox'], {}), true)
  assert.equal(alreadyHasNoSandbox([], { ELECTRON_DISABLE_SANDBOX: '1' }), true)
  assert.equal(alreadyHasNoSandbox([], { ELECTRON_DISABLE_SANDBOX: 'true' }), true)
  assert.equal(alreadyHasNoSandbox(['--disable-gpu'], {}), false)
})

test('decideWindowsSandboxLaunch stays off outside Windows/Linux and on clean markers', () => {
  assert.equal(decideWindowsSandboxLaunch({ platform: 'darwin', marker: { state: 'booting' } }).enable, false)

  const cleanOk = decideWindowsSandboxLaunch({
    platform: 'win32',
    marker: { state: 'ok' },
    argv: [],
    env: {}
  })

  assert.equal(cleanOk.enable, false)
  assert.deepEqual(cleanOk.nextMarker, { state: 'booting' })

  const noMarker = decideWindowsSandboxLaunch({ platform: 'win32', marker: null, argv: [], env: {} })
  assert.equal(noMarker.enable, false)
  assert.deepEqual(noMarker.nextMarker, { state: 'booting' })
})

test('a single mid-boot abort does NOT drop the sandbox (two-strike rule)', () => {
  // First abort: prior launch left `booting` with no abort count. Could be a
  // task-manager kill or power loss — sandbox stays ON, strike recorded.
  const first = decideWindowsSandboxLaunch({
    platform: 'win32',
    marker: { state: 'booting' },
    argv: [],
    env: {}
  })

  assert.equal(first.enable, false)
  assert.deepEqual(first.nextMarker, { state: 'booting', bootAborts: 1 })

  // Second consecutive abort: deterministic crash loop → fallback engages.
  const second = decideWindowsSandboxLaunch({
    platform: 'win32',
    marker: first.nextMarker,
    argv: [],
    env: {},
    appVersion: '1.2.3'
  })

  assert.equal(second.enable, true)
  assert.equal(second.reason, 'boot-loop')
  assert.deepEqual(second.nextMarker, { state: 'fallback', reason: 'boot-loop', version: '1.2.3' })

  assert.equal(BOOT_ABORTS_BEFORE_FALLBACK, 2)
})

test('sticky fallback persists within one app version', () => {
  const decision = decideWindowsSandboxLaunch({
    platform: 'win32',
    marker: { state: 'fallback', reason: 'gpu-breakpoint', version: '1.2.3' },
    argv: [],
    env: {},
    appVersion: '1.2.3'
  })

  assert.equal(decision.enable, true)
  assert.equal(decision.reason, 'sticky-fallback')
  assert.equal(decision.nextMarker.state, 'fallback')
  assert.equal(decision.nextMarker.reason, 'gpu-breakpoint')
})

test('an app update re-probes the sandbox once instead of degrading forever', () => {
  // Version changed since the fallback engaged → probe with sandbox ON.
  const reprobe = decideWindowsSandboxLaunch({
    platform: 'win32',
    marker: { state: 'fallback', reason: 'boot-loop', version: '1.2.3' },
    argv: [],
    env: {},
    appVersion: '1.3.0'
  })

  assert.equal(reprobe.enable, false)
  assert.equal(reprobe.nextMarker.state, 'booting')
  assert.equal(reprobe.nextMarker.reprobe, true)

  // The re-probe boot aborted → straight back to fallback, no second strike.
  const failedReprobe = decideWindowsSandboxLaunch({
    platform: 'win32',
    marker: reprobe.nextMarker,
    argv: [],
    env: {},
    appVersion: '1.3.0'
  })

  assert.equal(failedReprobe.enable, true)
  assert.equal(failedReprobe.reason, 'reprobe-failed')
  assert.equal(failedReprobe.nextMarker.state, 'fallback')
  assert.equal(failedReprobe.nextMarker.version, '1.3.0')

  // A legacy fallback marker without a version stays sticky (no re-probe).
  const legacy = decideWindowsSandboxLaunch({
    platform: 'win32',
    marker: { state: 'fallback' },
    argv: [],
    env: {},
    appVersion: '1.3.0'
  })

  assert.equal(legacy.enable, true)
  assert.equal(legacy.reason, 'sticky-fallback')
})

test('manual --no-sandbox is honored but never made sticky', () => {
  const manual = decideWindowsSandboxLaunch({
    platform: 'win32',
    marker: { state: 'ok' },
    argv: ['--no-sandbox'],
    env: {}
  })

  assert.equal(manual.enable, true)
  assert.equal(manual.reason, 'already-enabled')
  assert.equal(manual.nextMarker.state, 'booting')

  // But a relaunch-written fallback marker is preserved through the flagged boot.
  const relaunched = decideWindowsSandboxLaunch({
    platform: 'win32',
    marker: { state: 'fallback', reason: 'gpu-breakpoint', version: '1.2.3' },
    argv: ['--no-sandbox'],
    env: {},
    appVersion: '1.2.3'
  })

  assert.equal(relaunched.enable, true)
  assert.equal(relaunched.nextMarker.state, 'fallback')
})

test('marker transitions after a successful boot', () => {
  assert.deepEqual(markerAfterSuccessfulBoot({ fallbackActive: false }), { state: 'ok' })
  assert.deepEqual(markerAfterSuccessfulBoot({ fallbackActive: true, reason: 'gpu-breakpoint', appVersion: '1.2.3' }), {
    state: 'fallback',
    reason: 'gpu-breakpoint',
    version: '1.2.3'
  })
})

test('shouldAttemptAclRepair only fires on evidence of trouble', () => {
  assert.equal(shouldAttemptAclRepair(null), false)
  assert.equal(shouldAttemptAclRepair({ state: 'ok' }), false)
  assert.equal(shouldAttemptAclRepair({ state: 'booting' }), true)
  assert.equal(shouldAttemptAclRepair({ state: 'fallback' }), true)
})

test('sandbox marker round-trips through the userData file', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-sandbox-marker-'))

  try {
    assert.equal(sandboxMarkerPath(dir), path.join(dir, WINDOWS_SANDBOX_MARKER_FILENAME))
    assert.equal(readSandboxMarker(dir), null)

    writeSandboxMarker(dir, { state: 'booting', bootAborts: 1 })
    assert.deepEqual(readSandboxMarker(dir), { state: 'booting', bootAborts: 1 })

    writeSandboxMarker(dir, fallbackMarker('renderer-crash-loop', '1.2.3'))
    assert.deepEqual(readSandboxMarker(dir), {
      state: 'fallback',
      reason: 'renderer-crash-loop',
      version: '1.2.3'
    })

    assert.equal(parseSandboxMarker({ state: 'fallback' })?.state, 'fallback')
    assert.equal(parseSandboxMarker({ state: 'nope' }), null)
    // Unknown reason strings and junk fields are dropped, not fatal.
    assert.deepEqual(parseSandboxMarker({ state: 'fallback', reason: 'weird', bootAborts: -3 }), {
      state: 'fallback'
    })
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})

test('buildIcaclsGrantArgs targets ALL APPLICATION PACKAGES with inherited RX', () => {
  assert.deepEqual(buildIcaclsGrantArgs('C:\\Hermes\\win-unpacked'), [
    'C:\\Hermes\\win-unpacked',
    '/grant',
    `*${ALL_APPLICATION_PACKAGES_SID}:(OI)(CI)(RX)`,
    '/T',
    '/C',
    '/Q'
  ])
})

test('grantAllApplicationPackagesAcl is a no-op off Windows and reports exec failures', () => {
  assert.deepEqual(grantAllApplicationPackagesAcl('C:\\x', { platform: 'darwin' }), { ok: false })

  const calls: Array<{ file: string; args: readonly string[] }> = []

  const ok = grantAllApplicationPackagesAcl('C:\\Hermes', {
    platform: 'win32',
    execFileSync(file, args) {
      calls.push({ file, args })

      return Buffer.alloc(0)
    }
  })

  assert.deepEqual(ok, { ok: true })
  assert.equal(calls.length, 1)
  assert.equal(calls[0]?.file, 'icacls')
  assert.deepEqual(calls[0]?.args, buildIcaclsGrantArgs('C:\\Hermes'))

  const failed = grantAllApplicationPackagesAcl('C:\\Hermes', {
    platform: 'win32',
    execFileSync() {
      throw new Error('access denied')
    }
  })

  assert.equal(failed.ok, false)
  assert.match(String(failed.error), /access denied/)
})

test('shouldRelaunchForGpuSandboxCrash only fires once for GPU breakpoint deaths', () => {
  assert.equal(
    shouldRelaunchForGpuSandboxCrash({
      platform: 'win32',
      details: { type: 'GPU', exitCode: WINDOWS_SANDBOX_BREAKPOINT_EXIT },
      alreadyNoSandbox: false,
      relaunchAttempted: false
    }),
    true
  )
  assert.equal(
    shouldRelaunchForGpuSandboxCrash({
      platform: 'win32',
      details: { type: 'GPU', exitCode: WINDOWS_SANDBOX_BREAKPOINT_EXIT },
      alreadyNoSandbox: true,
      relaunchAttempted: false
    }),
    false
  )
  assert.equal(
    shouldRelaunchForGpuSandboxCrash({
      platform: 'win32',
      details: { type: 'GPU', exitCode: WINDOWS_SANDBOX_BREAKPOINT_EXIT },
      alreadyNoSandbox: false,
      relaunchAttempted: true
    }),
    false
  )
  assert.equal(
    shouldRelaunchForGpuSandboxCrash({
      platform: 'win32',
      details: { type: 'renderer', exitCode: WINDOWS_SANDBOX_BREAKPOINT_EXIT },
      alreadyNoSandbox: false,
      relaunchAttempted: false
    }),
    false
  )
  assert.equal(
    shouldRelaunchForGpuSandboxCrash({
      platform: 'linux',
      details: { type: 'GPU', exitCode: WINDOWS_SANDBOX_BREAKPOINT_EXIT },
      alreadyNoSandbox: false,
      relaunchAttempted: false
    }),
    false
  )
})

test('renderer crash-loop relaunch requires the sandbox breakpoint signature', () => {
  assert.equal(
    shouldRelaunchForRendererSandboxCrashLoop({
      platform: 'win32',
      reason: 'crashed',
      exitCode: WINDOWS_SANDBOX_BREAKPOINT_EXIT,
      alreadyNoSandbox: false,
      relaunchAttempted: false
    }),
    true
  )
  // Unrelated renderer crash loops (plain crash, OOM churn) keep the sandbox.
  assert.equal(
    shouldRelaunchForRendererSandboxCrashLoop({
      platform: 'win32',
      reason: 'crashed',
      exitCode: 1,
      alreadyNoSandbox: false,
      relaunchAttempted: false
    }),
    false
  )
  assert.equal(
    shouldRelaunchForRendererSandboxCrashLoop({
      platform: 'win32',
      reason: 'oom',
      exitCode: WINDOWS_SANDBOX_BREAKPOINT_EXIT,
      alreadyNoSandbox: false,
      relaunchAttempted: false
    }),
    false
  )
  assert.equal(
    shouldRelaunchForRendererSandboxCrashLoop({
      platform: 'win32',
      reason: 'crashed',
      exitCode: WINDOWS_SANDBOX_BREAKPOINT_EXIT,
      alreadyNoSandbox: true,
      relaunchAttempted: false
    }),
    false
  )
  assert.equal(
    shouldRelaunchForRendererSandboxCrashLoop({
      platform: 'linux',
      reason: 'crashed',
      exitCode: WINDOWS_SANDBOX_BREAKPOINT_EXIT,
      alreadyNoSandbox: false,
      relaunchAttempted: false
    }),
    false
  )
})

test('buildNoSandboxRelaunchArgs appends a single --no-sandbox flag', () => {
  assert.deepEqual(buildNoSandboxRelaunchArgs(['--foo', '--no-sandbox', 'hermes://x']), [
    '--foo',
    'hermes://x',
    '--no-sandbox'
  ])
})

// #121954: Linux hosts where the sandboxed GPU child cannot start crash-loop
// 100% ("GPU process isn't usable. Goodbye."); only --no-sandbox reaches the
// UI. Same two-strike sticky ladder as #38216, minus the Windows-only extras.
test('linux boot-abort ladder engages --no-sandbox on the second consecutive abort (#121954)', () => {
  const argv: string[] = []
  const env = {}
  const appVersion = '0.21.5'

  const first = decideWindowsSandboxLaunch({
    platform: 'linux',
    marker: { state: 'booting' },
    argv,
    env,
    appVersion
  })

  assert.equal(first.enable, false)
  assert.deepEqual(first.nextMarker, { state: 'booting', bootAborts: 1 })

  const second = decideWindowsSandboxLaunch({
    platform: 'linux',
    marker: first.nextMarker,
    argv,
    env,
    appVersion
  })

  assert.equal(second.enable, true)
  assert.equal(second.reason, 'boot-loop')
  assert.equal(second.nextMarker.state, 'fallback')

  // Sticky within the same app version...
  const sticky = decideWindowsSandboxLaunch({
    platform: 'linux',
    marker: second.nextMarker,
    argv,
    env,
    appVersion
  })

  assert.equal(sticky.enable, true)
  assert.equal(sticky.reason, 'sticky-fallback')

  // ...and an app update re-probes the sandbox once.
  const reprobe = decideWindowsSandboxLaunch({
    platform: 'linux',
    marker: second.nextMarker,
    argv,
    env,
    appVersion: '0.22.0'
  })

  assert.equal(reprobe.enable, false)
  assert.deepEqual(reprobe.nextMarker, { state: 'booting', reprobe: true, bootAborts: 0 })

  const reprobeFailed = decideWindowsSandboxLaunch({
    platform: 'linux',
    marker: reprobe.nextMarker,
    argv,
    env,
    appVersion: '0.22.0'
  })

  assert.equal(reprobeFailed.enable, true)
  assert.equal(reprobeFailed.reason, 'reprobe-failed')
})

test('linux GPU SIGTERM signature triggers the one-shot relaunch; other exits do not (#121954)', () => {
  assert.equal(
    shouldRelaunchForGpuSandboxCrash({
      platform: 'linux',
      details: { type: 'GPU', exitCode: GPU_CHILD_SANDBOX_SIGTERM_EXIT, signalName: 'SIGTERM' },
      alreadyNoSandbox: false,
      relaunchAttempted: false
    }),
    true
  )
  // Non-sandbox GPU deaths (driver faults die 139/SIGSEGV) keep the sandbox.
  assert.equal(
    shouldRelaunchForGpuSandboxCrash({
      platform: 'linux',
      details: { type: 'GPU', exitCode: 139, signalName: 'SIGSEGV' },
      alreadyNoSandbox: false,
      relaunchAttempted: false
    }),
    false
  )
  // Exit 143 without the SIGTERM signal name does not fire.
  assert.equal(
    shouldRelaunchForGpuSandboxCrash({
      platform: 'linux',
      details: { type: 'GPU', exitCode: GPU_CHILD_SANDBOX_SIGTERM_EXIT },
      alreadyNoSandbox: false,
      relaunchAttempted: false
    }),
    false
  )
  // macOS has no recovery path.
  assert.equal(
    shouldRelaunchForGpuSandboxCrash({
      platform: 'darwin',
      details: { type: 'GPU', exitCode: GPU_CHILD_SANDBOX_SIGTERM_EXIT, signalName: 'SIGTERM' },
      alreadyNoSandbox: false,
      relaunchAttempted: false
    }),
    false
  )
})

test('a booting witness from an earlier machine boot is not an aborted boot', () => {
  const decision = decideWindowsSandboxLaunch({
    platform: 'linux',
    marker: { state: 'booting', bootAborts: 1, boot: 'linux:boot-a' },
    bootIdentity: 'linux:boot-b',
    appVersion: '0.0.0'
  })

  assert.equal(decision.enable, false)
  assert.equal(decision.reason, null)
  assert.deepEqual(decision.nextMarker, { state: 'booting', boot: 'linux:boot-b' })
})

test('aborts inside the same boot still reach the two-strike ladder', () => {
  const first = decideWindowsSandboxLaunch({
    platform: 'linux',
    marker: { state: 'booting', boot: 'linux:boot-a' },
    bootIdentity: 'linux:boot-a',
    appVersion: '0.0.0'
  })

  assert.equal(first.enable, false)
  assert.deepEqual(first.nextMarker, { state: 'booting', bootAborts: 1, boot: 'linux:boot-a' })

  const second = decideWindowsSandboxLaunch({
    platform: 'linux',
    marker: first.nextMarker,
    bootIdentity: 'linux:boot-a',
    appVersion: '0.0.0'
  })

  assert.equal(second.enable, true)
  assert.equal(second.reason, 'boot-loop')
})

test('repeated power-loss boots never arm --no-sandbox', () => {
  let marker: Record<string, unknown> | null = { state: 'booting', boot: 'linux:boot-1' }

  for (const boot of ['linux:boot-2', 'linux:boot-3', 'linux:boot-4', 'linux:boot-5']) {
    const decision = decideWindowsSandboxLaunch({
      platform: 'linux',
      marker: marker as never,
      bootIdentity: boot,
      appVersion: '0.0.0'
    })

    assert.equal(decision.enable, false)
    assert.equal(decision.reason, null)
    assert.equal(decision.nextMarker.boot, boot)
    marker = decision.nextMarker as never
  }
})

test('a fallback written by another build re-probes once', () => {
  const decision = decideWindowsSandboxLaunch({
    platform: 'linux',
    marker: {
      state: 'fallback',
      reason: 'boot-loop',
      version: '0.0.0',
      build: '0.0.0+gdeadbeef@2026-10-01T00:00:00+00:00'
    },
    bootIdentity: 'linux:boot-a',
    buildIdentity: '0.0.0+gfeedface@2026-10-29T00:00:00+00:00',
    appVersion: '0.0.0'
  })

  assert.equal(decision.enable, false)
  assert.equal(decision.nextMarker.state, 'booting')
  assert.equal(decision.nextMarker.reprobe, true)
})

test('a fallback from this same build stays sticky', () => {
  const build = '0.0.0+gfeedface@2026-10-29T00:00:00+00:00'
  const decision = decideWindowsSandboxLaunch({
    platform: 'linux',
    marker: { state: 'fallback', reason: 'boot-loop', version: '0.0.0', build },
    bootIdentity: 'linux:boot-a',
    buildIdentity: build,
    appVersion: '0.0.0'
  })

  assert.equal(decision.enable, true)
  assert.equal(decision.reason, 'sticky-fallback')
})

test('an aborted post-update re-probe records the build so the next launch stops probing', () => {
  const build = '0.0.0+gfeedface@2026-10-29T00:00:00+00:00'
  const decision = decideWindowsSandboxLaunch({
    platform: 'linux',
    marker: { state: 'booting', reprobe: true, bootAborts: 0, boot: 'linux:boot-a' },
    bootIdentity: 'linux:boot-a',
    buildIdentity: build,
    appVersion: '0.0.0'
  })

  assert.equal(decision.enable, true)
  assert.equal(decision.reason, 'reprobe-failed')
  assert.equal(decision.nextMarker.build, build)
})

test('a legacy 0.0.0 fallback re-probes when the checkout carries a build identity', () => {
  const decision = decideWindowsSandboxLaunch({
    platform: 'linux',
    marker: { state: 'fallback', reason: 'boot-loop', version: '0.0.0' },
    bootIdentity: 'linux:boot-a',
    buildIdentity: '0.0.0+gfeedface@2026-10-29T00:00:00+00:00',
    appVersion: '0.0.0'
  })

  assert.equal(decision.enable, false)
  assert.equal(decision.nextMarker.state, 'booting')
  assert.equal(decision.nextMarker.reprobe, true)
})

test('currentBootIdentity prefers the kernel boot id and trims it', () => {
  assert.equal(currentBootIdentity({ platform: 'linux', readFileSync: () => ' 0f3c-aa11 \n' }), 'linux:0f3c-aa11')
})

test('currentBootIdentity falls back to the boot instant, stable across relaunches', () => {
  const first = currentBootIdentity({ platform: 'win32', nowMs: 1_700_000_000_000, uptimeSeconds: 1000 })
  const relaunch = currentBootIdentity({ platform: 'win32', nowMs: 1_700_000_060_000, uptimeSeconds: 1060 })

  assert.equal(first, 'at:1699999000')
  assert.equal(relaunch, first)
  assert.equal(
    currentBootIdentity({
      platform: 'linux',
      readFileSync: () => {
        throw new Error('no /proc')
      },
      nowMs: 1_700_000_000_000,
      uptimeSeconds: 1000
    }),
    'at:1699999000'
  )
})

test('write and read round-trip the boot and build witnesses', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-sandbox-witness-'))

  try {
    writeSandboxMarker(dir, { state: 'booting', bootAborts: 1, boot: 'linux:boot-a', build: '0.0.0+gfeedface@x' })

    assert.deepEqual(readSandboxMarker(dir), {
      state: 'booting',
      bootAborts: 1,
      boot: 'linux:boot-a',
      build: '0.0.0+gfeedface@x'
    })
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})

test('a boot-bearing witness still counts when the launch has no identity', () => {
  const decision = decideWindowsSandboxLaunch({
    platform: 'linux',
    marker: { state: 'booting', bootAborts: 1, boot: 'linux:boot-a' },
    appVersion: '1.0.0'
  })

  assert.equal(decision.enable, true)
  assert.equal(decision.reason, 'boot-loop')
})

test('a boot-less witness still counts when the launch has an identity', () => {
  const decision = decideWindowsSandboxLaunch({
    platform: 'linux',
    marker: { state: 'booting', bootAborts: 1 },
    bootIdentity: 'linux:boot-b',
    appVersion: '1.0.0'
  })

  assert.equal(decision.enable, true)
  assert.equal(decision.reason, 'boot-loop')
})

test('an unclean shutdown during a post-update re-probe returns to the fallback', () => {
  const build = '0.0.0+gfeedface@2026-10-29T00:00:00+00:00'
  const dropped = decideWindowsSandboxLaunch({
    platform: 'linux',
    marker: { state: 'booting', reprobe: true, bootAborts: 0, boot: 'linux:boot-a' },
    bootIdentity: 'linux:boot-b',
    buildIdentity: build,
    appVersion: '0.0.0'
  })

  assert.equal(dropped.enable, false)
  assert.equal(dropped.reason, null)
  assert.equal(dropped.nextMarker.reprobe, true)

  const next = decideWindowsSandboxLaunch({
    platform: 'linux',
    marker: dropped.nextMarker,
    bootIdentity: 'linux:boot-b',
    buildIdentity: build,
    appVersion: '0.0.0'
  })

  assert.equal(next.enable, true)
  assert.equal(next.reason, 'reprobe-failed')
  assert.equal(next.nextMarker.build, build)
})

test('a wall-clock step inside one boot is not a stale witness', () => {
  const boot = currentBootIdentity({ platform: 'win32', nowMs: 1_700_000_000_000, uptimeSeconds: 1000 })
  const drifted = currentBootIdentity({ platform: 'win32', nowMs: 1_700_000_002_000, uptimeSeconds: 1000 })
  const nextBoot = currentBootIdentity({ platform: 'win32', nowMs: 1_700_000_600_000, uptimeSeconds: 1000 })

  assert.equal(boot, 'at:1699999000')
  assert.equal(drifted, 'at:1699999002')
  assert.equal(isSameBootWitness(boot, drifted), true)
  assert.equal(isSameBootWitness(boot, nextBoot), false)
  assert.equal(isSameBootWitness('at:1699999000', 'at:1699999003'), false)
  assert.equal(isSameBootWitness('linux:x', 'linux:x'), true)
  assert.equal(isSameBootWitness('linux:x', 'linux:y'), false)
  assert.equal(isSameBootWitness('linux:x', boot), false)
  assert.equal(isSameBootWitness('', boot), false)
  assert.equal(BOOT_INSTANT_TOLERANCE_SECONDS, 2)
})

test('a drifted boot instant still counts the abort instead of dropping it', () => {
  const decision = decideWindowsSandboxLaunch({
    platform: 'win32',
    marker: { state: 'booting', bootAborts: 1, boot: 'at:1699999000' },
    bootIdentity: 'at:1699999002',
    appVersion: '1.0.0'
  })

  assert.equal(decision.enable, true)
  assert.equal(decision.reason, 'boot-loop')
})

test('a drift beyond the tolerance is treated as a new boot', () => {
  const decision = decideWindowsSandboxLaunch({
    platform: 'win32',
    marker: { state: 'booting', bootAborts: 1, boot: 'at:1699999000' },
    bootIdentity: 'at:1699999003',
    appVersion: '1.0.0'
  })

  assert.equal(decision.enable, false)
  assert.equal(decision.nextMarker.boot, 'at:1699999003')
  assert.equal(decision.nextMarker.bootAborts, undefined)
})

test('a booting decision records the build identity it was written by', () => {
  const build = '0.0.0+gfeedface@2026-10-29T00:00:00+00:00'
  const decision = decideWindowsSandboxLaunch({
    platform: 'linux',
    marker: null,
    bootIdentity: 'linux:boot-a',
    buildIdentity: build,
    appVersion: '0.0.0'
  })

  assert.equal(decision.enable, false)
  assert.equal(decision.nextMarker.build, build)

  const sticky = decideWindowsSandboxLaunch({
    platform: 'linux',
    marker: { state: 'fallback', reason: 'boot-loop', version: '0.0.0', build },
    bootIdentity: 'linux:boot-a',
    buildIdentity: build,
    appVersion: '0.0.0'
  })

  assert.equal(sticky.enable, true)
  assert.equal(sticky.nextMarker.build, build)
})

test('a dropped witness round-trips through the marker file and restarts the count', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-sandbox-drop-'))

  try {
    const dropped = decideWindowsSandboxLaunch({
      platform: 'linux',
      marker: { state: 'booting', bootAborts: 1, boot: 'linux:boot-a' },
      bootIdentity: 'linux:boot-b',
      appVersion: '0.0.0'
    })

    writeSandboxMarker(dir, dropped.nextMarker)
    const reloaded = readSandboxMarker(dir)

    assert.deepEqual(reloaded, { state: 'booting', boot: 'linux:boot-b' })

    const next = decideWindowsSandboxLaunch({
      platform: 'linux',
      marker: reloaded,
      bootIdentity: 'linux:boot-b',
      appVersion: '0.0.0'
    })

    assert.equal(next.enable, false)
    assert.equal(next.nextMarker.bootAborts, 1)

    // A re-probe witness written in the boot that was cut short: dropped, but
    // its re-probe intent survives the file round-trip.
    const reprobe = decideWindowsSandboxLaunch({
      platform: 'linux',
      marker: { state: 'booting', reprobe: true, bootAborts: 0, boot: 'linux:boot-a' },
      bootIdentity: 'linux:boot-b',
      buildIdentity: '0.0.0+gfeedface@2026-10-29T00:00:00+00:00',
      appVersion: '0.0.0'
    })

    writeSandboxMarker(dir, reprobe.nextMarker)

    const reloadedReprobe = readSandboxMarker(dir)

    assert.equal(reloadedReprobe?.reprobe, true)

    const afterProbe = decideWindowsSandboxLaunch({
      platform: 'linux',
      marker: reloadedReprobe,
      bootIdentity: 'linux:boot-b',
      buildIdentity: '0.0.0+gfeedface@2026-10-29T00:00:00+00:00',
      appVersion: '0.0.0'
    })

    assert.equal(afterProbe.reason, 'reprobe-failed')
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})
