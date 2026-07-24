import assert from 'node:assert/strict'

import { describe, test } from 'vitest'

import {
  decideWslgGpuLaunch,
  GPU_CRASHES_BEFORE_FALLBACK,
  gpuCrashEngagesFallback,
  isGpuChildCrash,
  parseWslgGpuMarker,
  readWslgGpuMarker,
  writeWslgGpuMarker,
  wslgGpuFallbackMarker,
  wslgGpuMarkerAfterSuccessfulBoot,
  wslgGpuMarkerPath
} from './wslg-gpu-fallback'

describe('parseWslgGpuMarker', () => {
  test('accepts the three known states and drops junk', () => {
    assert.deepEqual(parseWslgGpuMarker({ state: 'ok' }), { state: 'ok' })
    assert.deepEqual(parseWslgGpuMarker({ state: 'probing', reprobe: true }), { state: 'probing', reprobe: true })
    assert.deepEqual(parseWslgGpuMarker({ state: 'fallback', version: '1.2.3' }), {
      state: 'fallback',
      version: '1.2.3'
    })
  })

  test('rejects unknown/malformed input', () => {
    assert.equal(parseWslgGpuMarker(null), null)
    assert.equal(parseWslgGpuMarker('ok'), null)
    assert.equal(parseWslgGpuMarker({ state: 'weird' }), null)
    assert.equal(parseWslgGpuMarker({}), null)
  })
})

describe('read/write round-trip', () => {
  test('writes then reads back the same marker via injected fs', () => {
    const store = new Map<string, string>()
    const writeFileSync = ((p: string, data: string) => store.set(String(p), String(data))) as never

    const readFileSync = ((p: string) => {
      const v = store.get(String(p))

      if (v === undefined) {
        throw new Error('ENOENT')
      }

      return v
    }) as never

    const mkdirSync = (() => undefined) as never

    writeWslgGpuMarker('/data', { state: 'fallback', version: '9.9.9' }, { mkdirSync, writeFileSync })

    assert.ok(store.has(wslgGpuMarkerPath('/data')))
    assert.deepEqual(readWslgGpuMarker('/data', { readFileSync }), { state: 'fallback', version: '9.9.9' })
  })

  test('missing file reads back null, not a throw', () => {
    const readFileSync = (() => {
      throw new Error('ENOENT')
    }) as never

    assert.equal(readWslgGpuMarker('/data', { readFileSync }), null)
  })

  test('write is best-effort — a throwing fs never propagates', () => {
    const mkdirSync = (() => {
      throw new Error('EACCES')
    }) as never

    assert.doesNotThrow(() => writeWslgGpuMarker('/data', { state: 'ok' }, { mkdirSync }))
  })
})

describe('decideWslgGpuLaunch', () => {
  test('no marker → probe the GPU (enable), next marker is probing', () => {
    const d = decideWslgGpuLaunch({ marker: null, appVersion: '1.0.0' })

    assert.equal(d.enableGpu, true)
    assert.equal(d.reason, null)
    assert.deepEqual(d.nextMarker, { state: 'probing' })
  })

  test('clean ok → probe again', () => {
    const d = decideWslgGpuLaunch({ marker: { state: 'ok' }, appVersion: '1.0.0' })

    assert.equal(d.enableGpu, true)
    assert.deepEqual(d.nextMarker, { state: 'probing' })
  })

  test('sticky fallback on the same version → GPU stays disabled', () => {
    const d = decideWslgGpuLaunch({ marker: { state: 'fallback', version: '1.0.0' }, appVersion: '1.0.0' })

    assert.equal(d.enableGpu, false)
    assert.equal(d.reason, 'sticky-fallback')
    assert.equal(d.nextMarker.state, 'fallback')
    assert.equal(d.nextMarker.version, '1.0.0')
  })

  test('fallback from an older version → re-probe once after an app update', () => {
    const d = decideWslgGpuLaunch({ marker: { state: 'fallback', version: '0.9.0' }, appVersion: '1.0.0' })

    assert.equal(d.enableGpu, true)
    assert.equal(d.reason, 'reprobe-after-update')
    assert.deepEqual(d.nextMarker, { state: 'probing', reprobe: true })
  })

  test('fallback with no recorded version stays sticky (adopts current version)', () => {
    const d = decideWslgGpuLaunch({ marker: { state: 'fallback' }, appVersion: '1.0.0' })

    assert.equal(d.enableGpu, false)
    assert.equal(d.nextMarker.version, '1.0.0')
  })
})

describe('gpuCrashEngagesFallback', () => {
  test('under the threshold → no fallback yet', () => {
    assert.equal(gpuCrashEngagesFallback({ crashCount: GPU_CRASHES_BEFORE_FALLBACK - 1 }), null)
    assert.equal(gpuCrashEngagesFallback({ crashCount: 0 }), null)
  })

  test('at/over the threshold → sticky fallback marker with version', () => {
    const m = gpuCrashEngagesFallback({ crashCount: GPU_CRASHES_BEFORE_FALLBACK, appVersion: '2.0.0' })

    assert.deepEqual(m, { state: 'fallback', version: '2.0.0' })
  })

  test('custom threshold is honored', () => {
    assert.equal(gpuCrashEngagesFallback({ crashCount: 1, threshold: 2 }), null)
    assert.deepEqual(gpuCrashEngagesFallback({ crashCount: 2, threshold: 2 }), { state: 'fallback' })
  })
})

describe('isGpuChildCrash', () => {
  test('a crashed GPU child counts', () => {
    assert.equal(isGpuChildCrash({ type: 'GPU', reason: 'crashed', exitCode: 139 }), true)
    assert.equal(isGpuChildCrash({ type: 'gpu', reason: 'abnormal-exit', exitCode: 139 }), true)
  })

  test('a clean GPU shutdown does not count', () => {
    assert.equal(isGpuChildCrash({ type: 'GPU', reason: 'clean-exit', exitCode: 0 }), false)
  })

  test('non-GPU children never count', () => {
    assert.equal(isGpuChildCrash({ type: 'renderer', reason: 'crashed', exitCode: 139 }), false)
    assert.equal(isGpuChildCrash({ type: 'utility', reason: 'crashed' }), false)
  })

  test('null details never count', () => {
    assert.equal(isGpuChildCrash(null), false)
  })
})

describe('wslgGpuMarkerAfterSuccessfulBoot', () => {
  test('clean boot → ok', () => {
    assert.deepEqual(wslgGpuMarkerAfterSuccessfulBoot({ fallbackActive: false }), { state: 'ok' })
  })

  test('boot with fallback engaged → keep sticky fallback', () => {
    assert.deepEqual(wslgGpuMarkerAfterSuccessfulBoot({ fallbackActive: true, appVersion: '3.0.0' }), {
      state: 'fallback',
      version: '3.0.0'
    })
  })
})

describe('crash-loop lifecycle (integration of the pure pieces)', () => {
  test('probe → crash-loop → next launch disables → app update re-probes', () => {
    const version = '1.0.0'

    // Launch 1: no marker → probe.
    const launch1 = decideWslgGpuLaunch({ marker: null, appVersion: version })
    assert.equal(launch1.enableGpu, true)

    // GPU crashes past the threshold this session → fallback marker.
    const engaged = gpuCrashEngagesFallback({ crashCount: GPU_CRASHES_BEFORE_FALLBACK, appVersion: version })
    assert.deepEqual(engaged, wslgGpuFallbackMarker(version))

    // Launch 2: fallback same version → GPU disabled.
    const launch2 = decideWslgGpuLaunch({ marker: engaged, appVersion: version })
    assert.equal(launch2.enableGpu, false)

    // Launch 3: app updated → re-probe the GPU once.
    const launch3 = decideWslgGpuLaunch({ marker: engaged, appVersion: '1.1.0' })
    assert.equal(launch3.enableGpu, true)
    assert.equal(launch3.nextMarker.reprobe, true)
  })
})
