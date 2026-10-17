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
  GPU_CHILD_SANDBOX_SIGTERM_EXIT,
  grantAllApplicationPackagesAcl,
  isWindowsSandboxBreakpointExit,
  launchStillOwesReprobe,
  markerAfterSuccessfulBoot,
  parseSandboxMarker,
  readSandboxMarker,
  recordDirectCleanExit,
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

test('a `running` leftover is a mid-session abort, not boot evidence (#112961)', () => {
  // The prior run reached a usable window, then died without before-quit
  // (main-process abort, kill, power loss). No fallback, no strike counted,
  // but the next launch is told.
  const decision = decideWindowsSandboxLaunch({
    platform: 'win32',
    marker: { state: 'running', version: '1.2.3' },
    argv: [],
    env: {}
  })

  assert.equal(decision.enable, false)
  assert.equal(decision.priorSteadyAbort, true)
  assert.deepEqual(decision.nextMarker, { state: 'booting' })

  // Steady-state evidence must not trigger boot-trouble repair either.
  assert.equal(shouldAttemptAclRepair({ state: 'running' }), false)
  assert.equal(parseSandboxMarker({ state: 'running' })?.state, 'running')
})

test('reveal records `running` so a later abort leaves a trace; quit keeps `ok`', () => {
  assert.deepEqual(markerAfterSuccessfulBoot({ fallbackActive: false, steady: true, appVersion: '1.2.3' }), {
    state: 'running',
    version: '1.2.3'
  })
  assert.deepEqual(markerAfterSuccessfulBoot({ fallbackActive: false }), { state: 'ok' })
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

test('recordDirectCleanExit writes `ok` over a `running` leftover so the next launch sees no abort', () => {
  // app.exit() emits no before-quit, so this is the only clean-exit marker
  // write on the relaunch path. Drives the real seam — decision AND write,
  // against a real userData dir — because a test of the decision alone cannot
  // see whether main.ts routes its clean exits here.
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-sandbox-clean-exit-'))

  writeSandboxMarker(dir, markerAfterSuccessfulBoot({ fallbackActive: false, appVersion: '0.21.3', steady: true }))
  assert.deepEqual(readSandboxMarker(dir)?.state, 'running', 'precondition: reveal left a running marker')

  recordDirectCleanExit(dir, { stickyFallback: false })

  assert.deepEqual(readSandboxMarker(dir), { state: 'ok' })

  // The next launch must therefore not report a steady abort.
  const nextLaunch = decideWindowsSandboxLaunch({
    platform: 'win32',
    argv: [],
    env: {},
    marker: readSandboxMarker(dir),
    appVersion: '0.21.3'
  })

  assert.equal(nextLaunch.priorSteadyAbort, undefined, 'a clean direct exit must not read as a steady abort')
})

test('a clean relaunch between arming the re-probe and the real probe keeps the one allowed retry', () => {
  // The reviewer's third finding, one step past a clean-exit assertion: a clean
  // relaunch must not spend the boot-abort budget, but it must NOT also discard
  // the pending `reprobe`. That flag belongs to the NEXT process — the one that
  // actually exercises the sandbox — so dropping it hands a still-broken sandbox
  // a free extra attempt (two aborts to reach fallback instead of one).
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-reprobe-survives-'))

  // Launch N: a sticky fallback from the previous app version arms the re-probe.
  writeSandboxMarker(dir, { state: 'fallback', reason: 'boot-loop', version: '0.21.2' })

  const launchN = decideWindowsSandboxLaunch({
    platform: 'win32',
    argv: [],
    env: {},
    marker: readSandboxMarker(dir),
    appVersion: '0.21.3'
  })

  assert.equal(launchN.enable, false, 'the re-probe launches sandboxed')
  assert.deepEqual(launchN.nextMarker, { state: 'booting', reprobe: true, bootAborts: 0 })
  writeSandboxMarker(dir, launchN.nextMarker)

  // A clean relaunch before this run ever revealed a window.
  recordDirectCleanExit(dir, { stickyFallback: false })

  const afterExit = readSandboxMarker(dir)
  assert.equal(afterExit?.state, 'booting', 'the pending re-probe outlives the clean exit')
  assert.equal(afterExit?.reprobe, true, 'the one allowed sandbox retry is still armed')
  assert.equal(afterExit?.bootAborts, undefined, 'a clean exit records no boot-abort strike')

  // The relaunched process still runs with --no-sandbox (every in-app relaunch
  // does), so it is NOT the process that gets to use the retry — and it must not
  // clear the retry on its way through either.
  const relaunched = decideWindowsSandboxLaunch({
    platform: 'win32',
    argv: ['--no-sandbox'],
    env: {},
    marker: afterExit,
    appVersion: '0.21.3'
  })

  assert.equal(relaunched.reason, 'already-enabled')
  assert.equal(relaunched.nextMarker?.reprobe, true, 'a --no-sandbox launch preserves the pending re-probe')
  writeSandboxMarker(dir, relaunched.nextMarker)

  // Now the first process that actually exercises the sandbox, and it aborts.
  const firstProbe = decideWindowsSandboxLaunch({
    platform: 'win32',
    argv: [],
    env: {},
    marker: readSandboxMarker(dir),
    appVersion: '0.21.3'
  })

  assert.equal(firstProbe.enable, true, 'the first failed post-update probe restores the fallback')
  assert.equal(firstProbe.reason, 'reprobe-failed')
  assert.equal(firstProbe.nextMarker?.state, 'fallback')
})

test('a reveal that ran with --no-sandbox leaves the re-probe owed to the next sandboxed launch', () => {
  // Every in-app relaunch carries --no-sandbox, and such a boot DOES reach a
  // window. That reveal must not consume the retry: this process never
  // exercised the sandbox, so the next sandboxed launch is the one that gets
  // to use it. A reveal that actually ran sandboxed legitimately consumes it.
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-reveal-reprobe-'))

  const armed = { state: 'booting', reprobe: true, bootAborts: 0 } as const

  // The relaunched process reveals a window while running with --no-sandbox.
  const revealed = markerAfterSuccessfulBoot({
    fallbackActive: false,
    steady: true,
    appVersion: '0.21.3',
    pendingReprobe: true
  })

  assert.deepEqual(revealed, { state: 'running', version: '0.21.3', reprobe: true })
  writeSandboxMarker(dir, revealed)

  // Next launch: still steady-state evidence, and the retry moves forward.
  const next = decideWindowsSandboxLaunch({
    platform: 'win32',
    argv: [],
    env: {},
    marker: readSandboxMarker(dir),
    appVersion: '0.21.3'
  })

  assert.equal(next.priorSteadyAbort, true)
  assert.equal(next.nextMarker?.reprobe, true, 'the owed re-probe survives the reveal and the next launch')
  writeSandboxMarker(dir, next.nextMarker)

  // The first launch that really exercises the sandbox aborts: fallback, once.
  const firstProbe = decideWindowsSandboxLaunch({
    platform: 'win32',
    argv: [],
    env: {},
    marker: readSandboxMarker(dir),
    appVersion: '0.21.3'
  })

  assert.equal(firstProbe.enable, true, 'the first failed post-update probe restores the fallback')
  assert.equal(firstProbe.reason, 'reprobe-failed')
  void armed
})

test('a clean quit after a --no-sandbox reveal still owes the re-probe to the next sandboxed launch', () => {
  // The full post-update chain, through the NORMAL quit path rather than the
  // app.exit() relaunch path: arm the probe, relaunch with --no-sandbox, that
  // boot reaches a window (so it writes `running`), and the user then closes it
  // normally. before-quit and exitAfterBackendShutdown both route through
  // recordDirectCleanExit(), so the retry must survive whichever one runs.
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-quit-reprobe-'))

  writeSandboxMarker(dir, { state: 'fallback', reason: 'boot-loop', version: '0.21.2' })
  const armed = decideWindowsSandboxLaunch({
    platform: 'win32', argv: [], env: {}, marker: readSandboxMarker(dir), appVersion: '0.21.3'
  })
  writeSandboxMarker(dir, armed.nextMarker)

  const relaunched = decideWindowsSandboxLaunch({
    platform: 'win32', argv: ['--no-sandbox'], env: {}, marker: readSandboxMarker(dir), appVersion: '0.21.3'
  })
  writeSandboxMarker(dir, relaunched.nextMarker)

  // That boot reaches a window without ever touching the sandbox.
  writeSandboxMarker(dir, markerAfterSuccessfulBoot({
    fallbackActive: false, steady: true, appVersion: '0.21.3', pendingReprobe: true
  }))
  assert.deepEqual(readSandboxMarker(dir), { state: 'running', version: '0.21.3', reprobe: true })

  // A clean quit: this is what the before-quit handler runs.
  recordDirectCleanExit(dir, { stickyFallback: false })

  const afterQuit = readSandboxMarker(dir)
  assert.equal(afterQuit?.reprobe, true, 'quitting normally must not discard the owed re-probe')

  // The next sandboxed launch gets the retry, and its abort is the one that counts.
  const next = decideWindowsSandboxLaunch({
    platform: 'win32', argv: [], env: {}, marker: afterQuit, appVersion: '0.21.3'
  })
  writeSandboxMarker(dir, next.nextMarker)

  const firstProbe = decideWindowsSandboxLaunch({
    platform: 'win32', argv: [], env: {}, marker: readSandboxMarker(dir), appVersion: '0.21.3'
  })

  // ONE abort is enough. Whether it lands on `reprobe-failed` (the retry being
  // spent here) or on an already-sticky fallback (the retry having been spent
  // by the previous launch) is not the property under test; what matters is
  // that the sandbox is not tried a second time before the fallback engages.
  assert.equal(firstProbe.enable, true, 'one failed post-update probe is enough to restore the fallback')
  assert.equal(firstProbe.nextMarker?.state, 'fallback')
})

test('a manual --no-sandbox run on a healthy host must not fabricate a post-update retry', () => {
  // The reveal preserves a pending `reprobe` only when this launch genuinely
  // OWES one. A user typing `hermes --no-sandbox` on a healthy machine runs
  // with the sandbox off too, but nothing armed a retry — preserving one there
  // would make the next ordinary launch report `reprobe-failed` and latch the
  // app into a sticky fallback it never earned.
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-manual-nosandbox-'))

  // Healthy host, no update in flight: the previous run quit cleanly.
  writeSandboxMarker(dir, { state: 'ok' })

  const manual = decideWindowsSandboxLaunch({
    platform: 'win32',
    argv: ['--no-sandbox'],
    env: {},
    marker: readSandboxMarker(dir),
    appVersion: '0.21.3'
  })

  assert.equal(manual.reason, 'already-enabled')
  assert.notEqual(manual.nextMarker?.reprobe, true, 'a manual flag arms no retry')
  writeSandboxMarker(dir, manual.nextMarker)

  // The reveal's `pendingReprobe` argument is derived from the launch decision,
  // exactly as main.ts derives it — so a caller that passed the wrong predicate
  // (e.g. "ran with the sandbox off", which a manual flag also satisfies) makes
  // this fail rather than silently fabricate a retry.
  const revealed = markerAfterSuccessfulBoot({
    fallbackActive: false,
    steady: true,
    appVersion: '0.21.3',
    pendingReprobe: manual.nextMarker?.reprobe === true
  })
  assert.equal(manual.nextMarker?.reprobe, undefined, 'the launch decision must report no owed retry')
  writeSandboxMarker(dir, revealed)
  assert.deepEqual(revealed, { state: 'running', version: '0.21.3' })

  // Clean quit, then the next ordinary launch: sandboxed, no sticky fallback.
  recordDirectCleanExit(dir, { stickyFallback: false })

  const next = decideWindowsSandboxLaunch({
    platform: 'win32',
    argv: [],
    env: {},
    marker: readSandboxMarker(dir),
    appVersion: '0.21.3'
  })

  assert.equal(next.enable, false, 'a healthy host must keep its sandbox')
  assert.equal(next.reason, null)
})

test('a successful post-update re-probe consumes the retry', () => {
  // The version-change arm arms `reprobe` for the sandboxed re-probe launch
  // itself. That launch reaching a window IS the probe succeeding - the retry
  // is spent and must not survive, or the next unrelated abort (a task-manager
  // kill) reports `reprobe-failed` and latches a host whose sandbox just got
  // fixed into sticky --no-sandbox until the following update.
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-probe-succeeds-'))

  writeSandboxMarker(dir, { state: 'fallback', reason: 'boot-loop', version: '0.21.2' })

  const probeLaunch = decideWindowsSandboxLaunch({
    platform: 'win32', argv: [], env: {}, marker: readSandboxMarker(dir), appVersion: '0.21.3'
  })

  // This launch IS the sandboxed re-probe.
  assert.equal(probeLaunch.enable, false, 'the re-probe runs sandboxed')
  assert.deepEqual(probeLaunch.nextMarker, { state: 'booting', reprobe: true, bootAborts: 0 })
  writeSandboxMarker(dir, probeLaunch.nextMarker)

  // The predicate main.ts actually calls, not a re-implementation of it: this
  // launch owed a retry but ran sandboxed, so it spent it.
  const owed = launchStillOwesReprobe(probeLaunch)
  assert.equal(owed, false, 'a sandboxed re-probe spends the retry it was owed')

  writeSandboxMarker(dir, markerAfterSuccessfulBoot({
    fallbackActive: false, steady: true, appVersion: '0.21.3', pendingReprobe: owed
  }))

  assert.deepEqual(readSandboxMarker(dir), { state: 'running', version: '0.21.3' })

  // A main-process abort later leaves that plain `running`, and the next
  // launch must treat it as steady-state evidence, not a failed retry.
  const afterAbort = decideWindowsSandboxLaunch({
    platform: 'win32', argv: [], env: {}, marker: readSandboxMarker(dir), appVersion: '0.21.3'
  })

  assert.equal(afterAbort.priorSteadyAbort, true)
  assert.equal(afterAbort.nextMarker?.reprobe, undefined, 'a spent retry must not come back')
  assert.equal(afterAbort.enable, false, 'a healthy host keeps its sandbox')
})

test('a clean relaunch with no pending re-probe leaves nothing for the next launch to trip on', () => {
  // The ordinary case: no update in flight, so the clean exit is just `ok` and
  // the next launch starts from a plain boot.
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-no-reprobe-'))

  const startup = decideWindowsSandboxLaunch({
    platform: 'win32',
    argv: [],
    env: {},
    marker: null,
    appVersion: '0.21.3'
  })

  assert.deepEqual(startup.nextMarker, { state: 'booting' })
  writeSandboxMarker(dir, startup.nextMarker)

  recordDirectCleanExit(dir, { stickyFallback: false })

  assert.deepEqual(readSandboxMarker(dir), { state: 'ok' })

  const next = decideWindowsSandboxLaunch({
    platform: 'win32',
    argv: [],
    env: {},
    marker: readSandboxMarker(dir),
    appVersion: '0.21.3'
  })

  assert.equal(next.enable, false, 'a clean relaunch must not engage --no-sandbox')
  assert.equal(next.reason, null)
})

test('recordDirectCleanExit leaves the marker untouched only under a sticky fallback', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-sandbox-gated-'))
  writeSandboxMarker(dir, { state: 'running', version: '0.21.3' })

  // An engaged fallback keeps its sticky marker.
  recordDirectCleanExit(dir, { stickyFallback: true })
  assert.equal(readSandboxMarker(dir)?.state, 'running')

  // A Linux run DOES write: the marker ladder is shared with #121954, and a
  // mid-session kill there would otherwise leave a `booting` leftover that
  // counts toward the boot-loop budget. Only the sticky fallback blocks.
  recordDirectCleanExit(dir, { stickyFallback: false })
  assert.equal(readSandboxMarker(dir)?.state, 'ok')
})


test('recordDirectCleanExit takes no platform flag: a caller-supplied one cannot re-gate the write', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-sandbox-noflag-'))
  writeSandboxMarker(dir, { state: 'running', version: '0.21.3' })

  // The signature carries only `stickyFallback`. If a Windows-only gate ever
  // creeps back in, the Linux path goes silent again — a mid-session kill then
  // leaves a `booting` leftover that counts toward the boot-loop budget.
  // Passing a stray `isWindows: false` must therefore be inert, not a mute.
  recordDirectCleanExit(dir, { stickyFallback: false, isWindows: false } as never)

  assert.equal(readSandboxMarker(dir)?.state, 'ok', 'the clean exit must still be recorded')
})
test('buildNoSandboxRelaunchArgs appends a single --no-sandbox flag', () => {
  assert.deepEqual(buildNoSandboxRelaunchArgs(['--foo', '--no-sandbox', 'hermes://x']), [
    '--foo',
    'hermes://x',
    '--no-sandbox'
  ])
})

test('recordDirectCleanExit does not spend the boot-abort budget on a clean pre-window relaunch', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-prewindow-'))

  // Startup already persisted this, before any window exists.
  const startup = decideWindowsSandboxLaunch({
    platform: 'win32',
    argv: [],
    env: {},
    marker: null,
    appVersion: '0.21.3'
  })

  assert.deepEqual(startup.nextMarker, { state: 'booting' })
  writeSandboxMarker(dir, startup.nextMarker)

  // Clean intentional relaunch: the process never revealed a window, no
  // sandbox boot failed, and the fallback is not engaged.
  recordDirectCleanExit(dir, { stickyFallback: false })

  // `booting` is the prior-run-aborted state, so a clean exit must clear it
  // rather than leave a run that never aborted looking like an aborted boot.
  assert.deepEqual(readSandboxMarker(dir), { state: 'ok' })

  // And the next launch must not engage the fallback.
  const next = decideWindowsSandboxLaunch({
    platform: 'win32',
    argv: [],
    env: {},
    marker: readSandboxMarker(dir),
    appVersion: '0.21.3'
  })

  assert.equal(next.enable, false, 'a clean relaunch must not trigger --no-sandbox')
  assert.equal(next.reason, null)
})

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
