import assert from 'node:assert/strict'

import { describe, test } from 'vitest'

import {
  buildLunarCityPerfHandshake,
  createChromiumMemoryInfraGpuProbe,
  createLunarCityPerfMainController,
  parseChromiumMemoryInfraGpuAllocation,
  resolveLunarCityPerfLaunch,
  sanitizeAppMetrics
} from './lunar-city-perf-main'

const stamp = {
  builtAt: '2026-08-31T12:00:00.000Z',
  commit: 'a'.repeat(40),
  dirty: false,
  schemaVersion: 1,
  source: 'local'
} as const

describe('Lunar City packaged performance launch gate', () => {
  test('is absent in normal, dev, flag-only, nonce-only, and malformed-nonce launches', () => {
    for (const input of [
      { isPackaged: true, env: {} },
      {
        isPackaged: false,
        env: { HERMES_LUNAR_CITY_PERF_ACCEPTANCE: '1', HERMES_LUNAR_CITY_PERF_NONCE: 'n'.repeat(32) }
      },
      { isPackaged: true, env: { HERMES_LUNAR_CITY_PERF_ACCEPTANCE: '1' } },
      { isPackaged: true, env: { HERMES_LUNAR_CITY_PERF_NONCE: 'n'.repeat(32) } },
      {
        isPackaged: true,
        env: { HERMES_LUNAR_CITY_PERF_ACCEPTANCE: '1', HERMES_LUNAR_CITY_PERF_NONCE: 'short' }
      }
    ]) {
      assert.equal(resolveLunarCityPerfLaunch({ ...input, buildStamp: stamp }), undefined)
    }
  })

  test('accepts only a clean exact packaged build stamp and a unique nonce', () => {
    const launch = resolveLunarCityPerfLaunch({
      buildStamp: stamp,
      env: { HERMES_LUNAR_CITY_PERF_ACCEPTANCE: '1', HERMES_LUNAR_CITY_PERF_NONCE: 'nonce-0123456789abcdef-unique' },
      isPackaged: true
    })

    assert.deepEqual(launch, {
      buildStamp: stamp,
      launchNonce: 'nonce-0123456789abcdef-unique'
    })
    assert.equal(
      resolveLunarCityPerfLaunch({
        buildStamp: { ...stamp, dirty: true },
        env: { HERMES_LUNAR_CITY_PERF_ACCEPTANCE: '1', HERMES_LUNAR_CITY_PERF_NONCE: 'nonce-0123456789abcdef-unique' },
        isPackaged: true
      }),
      undefined
    )
  })
})

test('handshake carries exact lifetime, build, phase, and process provenance', () => {
  assert.deepEqual(
    buildLunarCityPerfHandshake({
      buildStamp: stamp,
      launchNonce: 'nonce-0123456789abcdef-unique',
      mainPid: 41,
      rendererPid: 82,
      rendererStartedAtMs: 1234
    }),
    {
      bridgeVersion: 1,
      buildSha: stamp.commit,
      buildStamp: stamp,
      launchNonce: 'nonce-0123456789abcdef-unique',
      mainPid: 41,
      packaged: true,
      processMetricsSource: 'electron.app.getAppMetrics',
      rendererIdentity: { pid: 82, startedAtMs: 1234 },
      supportedPhases: ['baseline-shell', 'mounted-city']
    }
  )
})

test('app metrics retain only required native CPU and working-set fields', () => {
  assert.deepEqual(
    sanitizeAppMetrics([
      {
        cpu: { idleWakeupsPerSecond: 9, percentCPUUsage: 2.5 },
        memory: { peakWorkingSetSize: 456, privateBytes: 789, workingSetSize: 123 },
        name: 'Tab',
        pid: 55,
        secret: 'must-not-cross-ipc',
        type: 'Tab'
      }
    ]),
    [{ cpu: { percentCPUUsage: 2.5 }, memory: { workingSetSize: 123 }, pid: 55, type: 'Tab' }]
  )
})

describe('chromium-memory-infra-v1 GPU parser', () => {
  test('sums attributable GPU allocator effective sizes without using RSS', () => {
    const result = parseChromiumMemoryInfraGpuAllocation({
      traceEvents: [
        {
          args: {
            dumps: {
              allocators: {
                'gpu/gl/textures': { attrs: { effective_size: { units: 'bytes', value: '100000' } } },
                'gpu/shared_images': { attrs: { effective_size: { units: 'bytes', value: '80000' } } },
                malloc: { attrs: { effective_size: { units: 'bytes', value: 'ffffff' } } }
              }
            }
          },
          cat: 'disabled-by-default-memory-infra',
          id: 'dump-1',
          ph: 'v',
          pid: 82,
          ts: 100
        }
      ]
    })

    assert.deepEqual(result, { gpuMemoryMiB: 1.5, gpuMemorySource: 'chromium-memory-infra-v1' })
  })

  test('reports unavailable for missing or ambiguous allocator attribution', () => {
    assert.deepEqual(parseChromiumMemoryInfraGpuAllocation({ traceEvents: [] }), {
      gpuMemoryMiB: null,
      gpuMemorySource: 'unavailable'
    })
    assert.deepEqual(
      parseChromiumMemoryInfraGpuAllocation({
        traceEvents: [
          {
            args: { dumps: { allocators: { gpu: { attrs: { effective_size: { units: 'objects', value: '2' } } } } } },
            cat: 'disabled-by-default-memory-infra',
            ph: 'v'
          }
        ]
      }),
      { gpuMemoryMiB: null, gpuMemorySource: 'unavailable' }
    )
  })

  test('uses one coherent dump and never double-counts GPU parents with children', () => {
    assert.deepEqual(
      parseChromiumMemoryInfraGpuAllocation({
        traceEvents: [
          {
            args: {
              dumps: {
                allocators: {
                  'gpu/gl': { attrs: { effective_size: { units: 'bytes', value: '200000' } } },
                  'gpu/gl/textures': { attrs: { effective_size: { units: 'bytes', value: '100000' } } },
                  'gpu/shared_images': { attrs: { effective_size: { units: 'bytes', value: '100000' } } }
                }
              }
            },
            cat: 'disabled-by-default-memory-infra',
            id: 'dump-1',
            ph: 'v',
            pid: 82,
            ts: 100
          }
        ]
      }),
      { gpuMemoryMiB: 3, gpuMemorySource: 'chromium-memory-infra-v1' }
    )
  })

  test('rejects multi-process and out-of-order GPU dump attribution', () => {
    const gpuEvent = (id: string, pid: number, ts: number) => ({
      args: { dumps: { allocators: { 'gpu/gl': { attrs: { effective_size: { units: 'bytes', value: '100000' } } } } } },
      cat: 'disabled-by-default-memory-infra',
      id,
      ph: 'v',
      pid,
      ts
    })

    for (const traceEvents of [
      [gpuEvent('a', 82, 100), gpuEvent('b', 83, 110)],
      [gpuEvent('a', 82, 110), gpuEvent('b', 82, 100)]
    ]) {
      assert.deepEqual(parseChromiumMemoryInfraGpuAllocation({ traceEvents }), {
        gpuMemoryMiB: null,
        gpuMemorySource: 'unavailable'
      })
    }
  })

  test('coalesces one dump across timestamped fragments and de-duplicates exact allocator repeats', () => {
    const allocator = { attrs: { effective_size: { units: 'bytes', value: '200000' } } }

    assert.deepEqual(
      parseChromiumMemoryInfraGpuAllocation({
        traceEvents: [
          {
            args: { dumps: { allocators: { 'gpu/gl': allocator } } },
            cat: 'disabled-by-default-memory-infra',
            id: 'dump-1',
            ph: 'v',
            pid: 82,
            ts: 100
          },
          {
            args: { dumps: { allocators: { 'gpu/gl': structuredClone(allocator) } } },
            cat: 'disabled-by-default-memory-infra',
            id: 'dump-1',
            ph: 'v',
            pid: 82,
            ts: 101
          },
          {
            args: {
              dumps: {
                allocators: {
                  'gpu/gl/textures': { attrs: { effective_size: { units: 'bytes', value: '100000' } } }
                }
              }
            },
            cat: 'disabled-by-default-memory-infra',
            id: 'dump-1',
            ph: 'v',
            pid: 82,
            ts: 102
          }
        ]
      }),
      { gpuMemoryMiB: 2, gpuMemorySource: 'chromium-memory-infra-v1' }
    )
  })

  test('rejects conflicting allocator fragments for the same dump identity', () => {
    const event = (value: string, ts: number) => ({
      args: { dumps: { allocators: { 'gpu/gl': { attrs: { effective_size: { units: 'bytes', value } } } } } },
      cat: 'disabled-by-default-memory-infra',
      id: 'dump-1',
      ph: 'v',
      pid: 82,
      ts
    })

    assert.deepEqual(
      parseChromiumMemoryInfraGpuAllocation({ traceEvents: [event('100000', 100), event('200000', 101)] }),
      {
        gpuMemoryMiB: null,
        gpuMemorySource: 'unavailable'
      }
    )
  })
})

test('controller binds bootstrap, metrics, and renderer requests to the launched window', async () => {
  const sent: Array<{ channel: string; payload: unknown }> = []

  const sender = {
    getOSProcessId: () => 82,
    id: 7,
    isDestroyed: () => false,
    send: (channel: string, payload: unknown) => sent.push({ channel, payload })
  }

  const otherSender = { ...sender, id: 8 }
  const event = { frameId: 3, sender }

  const controller = createLunarCityPerfMainController({
    appMetrics: () => [{ cpu: { percentCPUUsage: 2 }, memory: { workingSetSize: 400 }, pid: 82, type: 'Tab' }],
    gpuSnapshot: async () => ({ gpuMemoryMiB: 12, gpuMemorySource: 'chromium-memory-infra-v1' }),
    launch: { buildStamp: stamp, launchNonce: 'nonce-0123456789abcdef-unique' },
    mainPid: 41,
    now: () => 1234,
    ownsSender: candidate => candidate.id === 7,
    requestTimeoutMs: 1_000
  })

  assert.equal(controller.bootstrap({ frameId: 3, sender: otherSender }), undefined)
  const exactHandshake = controller.bootstrap(event)
  assert.deepEqual(exactHandshake, {
    bridgeVersion: 1,
    buildSha: stamp.commit,
    buildStamp: stamp,
    launchNonce: 'nonce-0123456789abcdef-unique',
    mainPid: 41,
    packaged: true,
    processMetricsSource: 'electron.app.getAppMetrics',
    rendererIdentity: { pid: 82, startedAtMs: 1234 },
    supportedPhases: ['baseline-shell', 'mounted-city']
  })
  assert.equal(controller.processMetrics(event), undefined)
  assert.equal(controller.requestRenderer(event, 'snapshot') instanceof Promise, true)
  assert.equal(await controller.requestRenderer(event, 'snapshot'), undefined)
  assert.equal(controller.registerResponder(event, exactHandshake), true)
  assert.equal(controller.registerResponder(event, exactHandshake), false)
  assert.equal(controller.activate(event, exactHandshake), true)
  assert.equal(controller.activate(event, exactHandshake), false)
  assert.deepEqual(controller.processMetrics(event), [
    { cpu: { percentCPUUsage: 2 }, memory: { workingSetSize: 400 }, pid: 82, type: 'Tab' }
  ])
  assert.equal(controller.processMetrics({ frameId: 3, sender: otherSender }), undefined)

  const pending = controller.requestRenderer(event, 'snapshot')
  const request = sent.at(-1)?.payload as { action: string; identity: Record<string, unknown>; requestId: string }
  assert.equal(sent.at(-1)?.channel, 'hermes:lunar-city-perf:request')
  assert.deepEqual(request.identity, {
    bridgeVersion: 1,
    buildSha: stamp.commit,
    frameId: 3,
    launchNonce: 'nonce-0123456789abcdef-unique',
    mainPid: 41,
    rendererGeneration: 1,
    rendererPid: 82,
    rendererStartedAtMs: 1234,
    senderId: 7
  })
  assert.equal(
    controller.resolveRendererResponse({ frameId: 3, sender: otherSender }, { ...request, value: { forged: true } }),
    false
  )
  assert.equal(
    controller.resolveRendererResponse(event, {
      ...request,
      identity: { ...request.identity, rendererGeneration: 2 },
      value: { frameMs: 4 }
    }),
    false
  )
  assert.equal(
    controller.resolveRendererResponse(event, {
      ...request,
      value: { frameMs: 4, rendererGeneration: 1, rendererPid: 82, rendererStartedAtMs: 1234 }
    }),
    true
  )
  assert.deepEqual(await pending, {
    gpuMemoryMiB: 12,
    gpuMemorySource: 'chromium-memory-infra-v1',
    frameMs: 4,
    rendererGeneration: 1,
    rendererPid: 82,
    rendererStartedAtMs: 1234
  })

  controller.dispose()
  assert.equal(await controller.requestRenderer(event, 'snapshot'), undefined)
})

test('controller invalidates authentication and pending work on renderer lifecycle changes', async () => {
  const sent: unknown[] = []

  const sender = {
    getOSProcessId: () => 82,
    id: 7,
    isDestroyed: () => false,
    send: (_channel: string, payload: unknown) => sent.push(payload)
  }

  const event = { frameId: 3, sender }

  const controller = createLunarCityPerfMainController({
    appMetrics: () => [],
    gpuSnapshot: async () => ({ gpuMemoryMiB: null, gpuMemorySource: 'unavailable' }),
    launch: { buildStamp: stamp, launchNonce: 'nonce-0123456789abcdef-unique' },
    mainPid: 41,
    now: () => 1234,
    ownsSender: candidate => candidate.id === 7,
    requestTimeoutMs: 1_000
  })

  const handshake = controller.bootstrap(event)

  assert.equal(controller.registerResponder(event, handshake), true)
  assert.equal(controller.activate(event, handshake), true)
  const pending = controller.requestRenderer(event, 'snapshot')
  controller.invalidateRenderer(sender, 'navigation')

  await assert.rejects(pending, /navigation/u)
  assert.equal(controller.processMetrics(event), undefined)
  assert.equal(await controller.requestRenderer(event, 'snapshot'), undefined)
})

test('controller reports unavailable GPU attribution without substituting RSS', async () => {
  const responses: Array<{ channel: string; payload: unknown }> = []

  const sender = {
    getOSProcessId: () => 82,
    id: 7,
    isDestroyed: () => false,
    send: (channel: string, payload: unknown) => responses.push({ channel, payload })
  }

  const controller = createLunarCityPerfMainController({
    appMetrics: () => [],
    gpuSnapshot: async () => ({ gpuMemoryMiB: null, gpuMemorySource: 'unavailable' }),
    launch: { buildStamp: stamp, launchNonce: 'nonce-0123456789abcdef-unique' },
    mainPid: 41,
    now: () => 1234,
    ownsSender: candidate => candidate.id === 7,
    requestTimeoutMs: 1_000
  })

  const event = { frameId: 3, sender }
  const handshake = controller.bootstrap(event)
  controller.registerResponder(event, handshake)
  controller.activate(event, handshake)
  const pending = controller.requestRenderer(event, 'snapshot')
  const request = responses.at(-1)?.payload as Record<string, unknown>
  controller.resolveRendererResponse(event, {
    ...request,
    value: { rendererGeneration: 1, rendererPid: 82, rendererStartedAtMs: 1234 }
  })

  assert.deepEqual(await pending, {
    gpuMemoryMiB: null,
    gpuMemorySource: 'unavailable',
    rendererGeneration: 1,
    rendererPid: 82,
    rendererStartedAtMs: 1234
  })
})

test('controller binds scenario results to the exact request and owns hidden window transitions', async () => {
  const sent: Array<{ channel: string; payload: unknown }> = []
  const windowActions: unknown[] = []

  const sender = {
    getOSProcessId: () => 82,
    id: 7,
    isDestroyed: () => false,
    send: (channel: string, payload: unknown) => sent.push({ channel, payload })
  }

  const event = { frameId: 3, sender }

  const controller = createLunarCityPerfMainController({
    appMetrics: () => [],
    gpuSnapshot: async () => ({ gpuMemoryMiB: null, gpuMemorySource: 'unavailable' }),
    launch: { buildStamp: stamp, launchNonce: 'nonce-0123456789abcdef-unique' },
    mainPid: 41,
    now: () => 1234,
    ownsSender: candidate => candidate.id === 7,
    scenarioWindowAction: async (_sender, action) => {
      windowActions.push(action)

      return {
        action,
        proof: 1,
        windowState: { minimized: false, visible: false },
        windowTrace: [
          { minimized: false, visible: true },
          { minimized: false, visible: false }
        ]
      }
    }
  })

  const handshake = controller.bootstrap(event)
  assert.equal(controller.registerResponder(event, handshake), true)
  assert.equal(controller.activate(event, handshake), true)

  const hidden = await controller.requestRenderer(event, 'scenario-action', {
    action: 'window-hidden',
    payload: {}
  })

  assert.deepEqual(windowActions, ['window-hidden'])
  assert.equal(sent.length, 0)
  assert.deepEqual(hidden, {
    action: 'window-hidden',
    proof: 1,
    windowState: { minimized: false, visible: false },
    windowTrace: [
      { minimized: false, visible: true },
      { minimized: false, visible: false }
    ],
    bridgeBinding: {
      action: 'scenario-action',
      identity: {
        bridgeVersion: 1,
        buildSha: stamp.commit,
        frameId: 3,
        launchNonce: 'nonce-0123456789abcdef-unique',
        mainPid: 41,
        rendererGeneration: 1,
        rendererPid: 82,
        rendererStartedAtMs: 1234,
        senderId: 7
      },
      payload: { action: 'window-hidden', payload: {} },
      requestId: `lcperf-v1:${stamp.commit}:nonce-0123456789abcdef-unique:41:7:3:82:1:1234:1`
    }
  })

  const orbitPending = controller.requestRenderer(event, 'scenario-action', {
    action: 'orbit',
    payload: { deltaAlpha: 0.5, deltaBeta: 0.1 }
  })

  const request = sent.at(-1)?.payload as Record<string, unknown>
  assert.equal(request.requestId, `lcperf-v1:${stamp.commit}:nonce-0123456789abcdef-unique:41:7:3:82:1:1234:2`)
  assert.equal(controller.resolveRendererResponse(event, { ...request, value: { action: 'orbit', proof: 2 } }), true)
  const orbit = await orbitPending
  assert.deepEqual((orbit as { bridgeBinding: unknown }).bridgeBinding, {
    action: 'scenario-action',
    identity: request.identity,
    payload: { action: 'orbit', payload: { deltaAlpha: 0.5, deltaBeta: 0.1 } },
    requestId: request.requestId
  })
})

test('GPU probe requests only Chromium memory-infra and fails closed on capture errors', async () => {
  let requested: unknown

  const probe = createChromiumMemoryInfraGpuProbe(async config => {
    requested = config

    return {
      traceEvents: [
        {
          args: {
            dumps: { allocators: { 'gpu/gl': { attrs: { effective_size: { units: 'bytes', value: '200000' } } } } }
          },
          cat: 'disabled-by-default-memory-infra',
          id: 'dump-1',
          ph: 'v',
          pid: 82,
          ts: 100
        }
      ]
    }
  })

  assert.deepEqual(await probe(), { gpuMemoryMiB: 2, gpuMemorySource: 'chromium-memory-infra-v1' })
  assert.deepEqual(requested, {
    category: 'disabled-by-default-memory-infra',
    dumpMode: 'detailed',
    periodicIntervalMs: 250
  })

  const unavailable = createChromiumMemoryInfraGpuProbe(async () => {
    throw new Error('trace unavailable')
  })

  assert.deepEqual(await unavailable(), { gpuMemoryMiB: null, gpuMemorySource: 'unavailable' })
})
