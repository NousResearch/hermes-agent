import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { shouldRelaunchForRendererSandboxCrashLoop } from './windows-sandbox-fallback'
import {
  alreadyHasDisableGpu,
  buildDisableGpuRelaunchArgs,
  decideWindowsGpuStackCookieLaunch,
  gpuStackCookieFallbackMarker,
  gpuStackCookieMarkerPath,
  isWindowsStackCookieExit,
  readGpuStackCookieMarker,
  shouldRelaunchForRendererStackCookieCrashLoop,
  shouldSurfaceErrorForRendererStackCookieCrashLoop,
  WINDOWS_GPU_STACK_COOKIE_MARKER_FILENAME,
  WINDOWS_STACK_COOKIE_EXIT,
  writeGpuStackCookieMarker
} from './windows-stack-cookie-fallback'

test('isWindowsStackCookieExit recognizes the issue signed exitCode', () => {
  assert.equal(isWindowsStackCookieExit(-1073740791), true)
  assert.equal(isWindowsStackCookieExit(WINDOWS_STACK_COOKIE_EXIT), true)
})

test('isWindowsStackCookieExit recognizes unsigned STATUS_STACK_BUFFER_OVERRUN', () => {
  assert.equal(isWindowsStackCookieExit(0xc0000409), true)
})

test('isWindowsStackCookieExit rejects STATUS_BREAKPOINT (sandbox owns that code)', () => {
  assert.equal(isWindowsStackCookieExit(-2147483645), false)
})

test('isWindowsStackCookieExit rejects unrelated and non-finite values', () => {
  assert.equal(isWindowsStackCookieExit(1), false)
  assert.equal(isWindowsStackCookieExit(Number.NaN), false)
  assert.equal(isWindowsStackCookieExit(Number.POSITIVE_INFINITY), false)
  assert.equal(isWindowsStackCookieExit('nope'), false)
  assert.equal(isWindowsStackCookieExit(undefined), false)
})

test('shouldRelaunchForRendererStackCookieCrashLoop fires for the issue signature', () => {
  assert.equal(
    shouldRelaunchForRendererStackCookieCrashLoop({
      platform: 'win32',
      reason: 'crashed',
      exitCode: -1073740791,
      alreadyGpuDisabled: false,
      relaunchAttempted: false,
      gpuOverrideOff: false
    }),
    true
  )
})

test('shouldRelaunchForRendererStackCookieCrashLoop stays off outside Windows', () => {
  const base = {
    reason: 'crashed' as const,
    exitCode: -1073740791,
    alreadyGpuDisabled: false,
    relaunchAttempted: false,
    gpuOverrideOff: false
  }

  assert.equal(shouldRelaunchForRendererStackCookieCrashLoop({ ...base, platform: 'linux' }), false)
  assert.equal(shouldRelaunchForRendererStackCookieCrashLoop({ ...base, platform: 'darwin' }), false)
})

test('shouldRelaunchForRendererStackCookieCrashLoop stays off for oom or killed', () => {
  const base = {
    platform: 'win32' as const,
    exitCode: -1073740791,
    alreadyGpuDisabled: false,
    relaunchAttempted: false,
    gpuOverrideOff: false
  }

  assert.equal(shouldRelaunchForRendererStackCookieCrashLoop({ ...base, reason: 'oom' }), false)
  assert.equal(shouldRelaunchForRendererStackCookieCrashLoop({ ...base, reason: 'killed' }), false)
})

test('shouldRelaunchForRendererStackCookieCrashLoop stays off when GPU is already disabled', () => {
  assert.equal(
    shouldRelaunchForRendererStackCookieCrashLoop({
      platform: 'win32',
      reason: 'crashed',
      exitCode: -1073740791,
      alreadyGpuDisabled: true,
      relaunchAttempted: false,
      gpuOverrideOff: false
    }),
    false
  )
})

test('shouldRelaunchForRendererStackCookieCrashLoop stays off after a relaunch attempt', () => {
  assert.equal(
    shouldRelaunchForRendererStackCookieCrashLoop({
      platform: 'win32',
      reason: 'crashed',
      exitCode: -1073740791,
      alreadyGpuDisabled: false,
      relaunchAttempted: true,
      gpuOverrideOff: false
    }),
    false
  )
})

test('shouldRelaunchForRendererStackCookieCrashLoop stays off when GPU override is explicitly off', () => {
  assert.equal(
    shouldRelaunchForRendererStackCookieCrashLoop({
      platform: 'win32',
      reason: 'crashed',
      exitCode: -1073740791,
      alreadyGpuDisabled: false,
      relaunchAttempted: false,
      gpuOverrideOff: true
    }),
    false
  )
})

test('decideWindowsGpuStackCookieLaunch enables sticky fallback, re-probes on version change, and fail-opens', () => {
  const sticky = decideWindowsGpuStackCookieLaunch({
    platform: 'win32',
    marker: { state: 'fallback', reason: 'renderer-crash-loop', version: '1.2.3' },
    env: {},
    appVersion: '1.2.3'
  })

  assert.equal(sticky.enable, true)
  assert.equal(sticky.nextMarker.state, 'fallback')

  const linux = decideWindowsGpuStackCookieLaunch({
    platform: 'linux',
    marker: { state: 'fallback', reason: 'renderer-crash-loop', version: '1.2.3' },
    env: {},
    appVersion: '1.2.3'
  })

  assert.equal(linux.enable, false)

  const overrideOff = decideWindowsGpuStackCookieLaunch({
    platform: 'win32',
    marker: { state: 'fallback', reason: 'renderer-crash-loop', version: '1.2.3' },
    env: { HERMES_DESKTOP_DISABLE_GPU: '0' },
    appVersion: '1.2.3'
  })

  assert.equal(overrideOff.enable, false)

  const reprobe = decideWindowsGpuStackCookieLaunch({
    platform: 'win32',
    marker: { state: 'fallback', reason: 'renderer-crash-loop', version: '1.2.3' },
    env: {},
    appVersion: '1.3.0'
  })

  assert.equal(reprobe.enable, false)
  assert.equal(reprobe.nextMarker.state, 'booting')
  assert.equal(reprobe.nextMarker.reprobe, true)

  const legacy = decideWindowsGpuStackCookieLaunch({
    platform: 'win32',
    marker: { state: 'fallback' },
    env: {},
    appVersion: '1.3.0'
  })

  assert.equal(legacy.enable, true)
})

test('alreadyHasDisableGpu honors argv and HERMES_DESKTOP_DISABLE_GPU', () => {
  assert.equal(alreadyHasDisableGpu(['--foo', '--disable-gpu'], {}), true)
  assert.equal(alreadyHasDisableGpu([], { HERMES_DESKTOP_DISABLE_GPU: '1' }), true)
  assert.equal(alreadyHasDisableGpu([], { HERMES_DESKTOP_DISABLE_GPU: 'true' }), true)
  assert.equal(alreadyHasDisableGpu(['--disable-gpu-compositing'], {}), false)
  assert.equal(alreadyHasDisableGpu(['--no-sandbox'], {}), false)
})

test('buildDisableGpuRelaunchArgs appends a single GPU-off pair', () => {
  assert.deepEqual(buildDisableGpuRelaunchArgs(['--foo', '--disable-gpu', 'hermes://x']), [
    '--foo',
    'hermes://x',
    '--disable-gpu',
    '--disable-gpu-compositing'
  ])
})

test('decideWindowsGpuStackCookieLaunch honors argv --disable-gpu even without a marker', () => {
  const viaArgs = decideWindowsGpuStackCookieLaunch({
    platform: 'win32',
    argv: ['--disable-gpu'],
    env: {},
    marker: null,
    appVersion: '1.2.3'
  })

  assert.equal(viaArgs.enable, true)
  assert.equal(viaArgs.reason, 'already-enabled')
  assert.deepEqual(viaArgs.nextMarker, { state: 'booting' })

  const viaArgsKeepsSticky = decideWindowsGpuStackCookieLaunch({
    platform: 'win32',
    argv: ['--disable-gpu'],
    env: {},
    marker: { state: 'fallback', reason: 'renderer-crash-loop', version: '1.2.3' },
    appVersion: '1.2.3'
  })

  assert.equal(viaArgsKeepsSticky.enable, true)
  assert.equal(viaArgsKeepsSticky.nextMarker.state, 'fallback')

  const overrideOffBeatsArgv = decideWindowsGpuStackCookieLaunch({
    platform: 'win32',
    argv: ['--disable-gpu'],
    env: { HERMES_DESKTOP_DISABLE_GPU: '0' },
    marker: { state: 'fallback', reason: 'renderer-crash-loop', version: '1.2.3' },
    appVersion: '1.2.3'
  })

  assert.equal(overrideOffBeatsArgv.enable, false)
})

test('CONTROL: sandbox relaunch stays false for 0xC0000409 (do not piggyback --no-sandbox)', () => {
  assert.equal(
    shouldRelaunchForRendererSandboxCrashLoop({
      platform: 'win32',
      reason: 'crashed',
      exitCode: -1073740791,
      alreadyNoSandbox: false,
      relaunchAttempted: false
    }),
    false
  )
  assert.notEqual(WINDOWS_GPU_STACK_COOKIE_MARKER_FILENAME, 'windows-sandbox-fallback.json')
})

test('shouldSurfaceErrorForRendererStackCookieCrashLoop when GPU fallback cannot run', () => {
  const signature = {
    platform: 'win32' as const,
    reason: 'crashed' as const,
    exitCode: -1073740791
  }

  assert.equal(
    shouldSurfaceErrorForRendererStackCookieCrashLoop({
      ...signature,
      alreadyGpuDisabled: true,
      relaunchAttempted: false,
      gpuOverrideOff: false
    }),
    true
  )
  assert.equal(
    shouldSurfaceErrorForRendererStackCookieCrashLoop({
      ...signature,
      alreadyGpuDisabled: false,
      relaunchAttempted: true,
      gpuOverrideOff: false
    }),
    true
  )
  assert.equal(
    shouldSurfaceErrorForRendererStackCookieCrashLoop({
      ...signature,
      alreadyGpuDisabled: false,
      relaunchAttempted: false,
      gpuOverrideOff: true
    }),
    true
  )
  assert.equal(
    shouldSurfaceErrorForRendererStackCookieCrashLoop({
      ...signature,
      alreadyGpuDisabled: false,
      relaunchAttempted: false,
      gpuOverrideOff: false
    }),
    false
  )
  assert.equal(
    shouldSurfaceErrorForRendererStackCookieCrashLoop({
      ...signature,
      platform: 'linux',
      alreadyGpuDisabled: true,
      relaunchAttempted: false,
      gpuOverrideOff: false
    }),
    false
  )
  assert.equal(
    shouldSurfaceErrorForRendererStackCookieCrashLoop({
      platform: 'win32',
      reason: 'crashed',
      exitCode: -2147483645,
      alreadyGpuDisabled: true,
      relaunchAttempted: false,
      gpuOverrideOff: false
    }),
    false
  )
})

test('GPU stack-cookie marker lives in its own file and round-trips', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-stack-cookie-marker-'))

  try {
    assert.equal(gpuStackCookieMarkerPath(dir), path.join(dir, WINDOWS_GPU_STACK_COOKIE_MARKER_FILENAME))
    assert.equal(WINDOWS_GPU_STACK_COOKIE_MARKER_FILENAME, 'windows-gpu-stack-cookie-fallback.json')
    assert.equal(readGpuStackCookieMarker(dir), null)

    writeGpuStackCookieMarker(dir, gpuStackCookieFallbackMarker('renderer-crash-loop', '1.2.3'))
    assert.deepEqual(readGpuStackCookieMarker(dir), {
      state: 'fallback',
      reason: 'renderer-crash-loop',
      version: '1.2.3'
    })
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})
