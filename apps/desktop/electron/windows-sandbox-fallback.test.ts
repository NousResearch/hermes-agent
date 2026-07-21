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
  buildNoSandboxRelaunchArgs,
  decideWindowsSandboxLaunch,
  fallbackMarker,
  grantAllApplicationPackagesAcl,
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

test('decideWindowsSandboxLaunch stays off outside Windows and on clean markers', () => {
  assert.equal(decideWindowsSandboxLaunch({ platform: 'linux', marker: { state: 'booting' } }).enable, false)

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
