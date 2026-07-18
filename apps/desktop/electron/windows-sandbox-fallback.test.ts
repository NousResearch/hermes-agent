import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import {
  ALL_APPLICATION_PACKAGES_SID,
  WINDOWS_SANDBOX_BREAKPOINT_EXIT,
  WINDOWS_SANDBOX_MARKER_FILENAME,
  alreadyHasNoSandbox,
  buildIcaclsGrantArgs,
  buildNoSandboxRelaunchArgs,
  grantAllApplicationPackagesAcl,
  isWindowsSandboxBreakpointExit,
  markerAfterSuccessfulBoot,
  nextSandboxMarkerAfterLaunchDecision,
  parseSandboxMarker,
  readSandboxMarker,
  sandboxMarkerPath,
  shouldEnableWindowsNoSandbox,
  shouldRelaunchForGpuSandboxCrash,
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

test('shouldEnableWindowsNoSandbox stays off outside Windows and on clean markers', () => {
  assert.deepEqual(
    shouldEnableWindowsNoSandbox({ platform: 'linux', marker: { state: 'booting' } }),
    { enable: false, reason: null }
  )
  assert.deepEqual(
    shouldEnableWindowsNoSandbox({ platform: 'win32', marker: { state: 'ok' }, argv: [], env: {} }),
    { enable: false, reason: null }
  )
  assert.deepEqual(
    shouldEnableWindowsNoSandbox({ platform: 'win32', marker: null, argv: [], env: {} }),
    { enable: false, reason: null }
  )
})

test('shouldEnableWindowsNoSandbox recovers from uncleared boot and sticky fallback', () => {
  assert.deepEqual(
    shouldEnableWindowsNoSandbox({
      platform: 'win32',
      marker: { state: 'booting' },
      argv: [],
      env: {}
    }),
    { enable: true, reason: 'uncleared-boot-marker' }
  )
  assert.deepEqual(
    shouldEnableWindowsNoSandbox({
      platform: 'win32',
      marker: { state: 'fallback' },
      argv: [],
      env: {}
    }),
    { enable: true, reason: 'sticky-fallback' }
  )
  assert.deepEqual(
    shouldEnableWindowsNoSandbox({
      platform: 'win32',
      marker: { state: 'ok' },
      argv: ['--no-sandbox'],
      env: {}
    }),
    { enable: true, reason: 'already-enabled' }
  )
})

test('marker transitions preserve sticky fallback after a successful recovered boot', () => {
  assert.deepEqual(nextSandboxMarkerAfterLaunchDecision({ enabledNoSandbox: false }), {
    state: 'booting'
  })
  assert.deepEqual(nextSandboxMarkerAfterLaunchDecision({ enabledNoSandbox: true }), {
    state: 'fallback'
  })
  assert.deepEqual(markerAfterSuccessfulBoot({ fallbackActive: false }), { state: 'ok' })
  assert.deepEqual(markerAfterSuccessfulBoot({ fallbackActive: true }), { state: 'fallback' })
})

test('sandbox marker round-trips through the userData file', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-sandbox-marker-'))

  try {
    assert.equal(sandboxMarkerPath(dir), path.join(dir, WINDOWS_SANDBOX_MARKER_FILENAME))
    assert.equal(readSandboxMarker(dir), null)

    writeSandboxMarker(dir, { state: 'booting' })
    assert.deepEqual(readSandboxMarker(dir), { state: 'booting' })
    assert.equal(parseSandboxMarker({ state: 'fallback' })?.state, 'fallback')
    assert.equal(parseSandboxMarker({ state: 'nope' }), null)
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

test('buildNoSandboxRelaunchArgs appends a single --no-sandbox flag', () => {
  assert.deepEqual(buildNoSandboxRelaunchArgs(['--foo', '--no-sandbox', 'hermes://x']), [
    '--foo',
    'hermes://x',
    '--no-sandbox'
  ])
})
