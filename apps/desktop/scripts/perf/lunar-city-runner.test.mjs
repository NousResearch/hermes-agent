import assert from 'node:assert/strict'
import { test } from 'node:test'

import {
  cleanupIsolatedRun,
  createIsolatedLaunchPlan,
  inspectPackagedTarget,
  reserveUniqueDebugPort,
  runPackagedLunarCityMeasurement,
  runScenarioThroughBridge,
  validateBridgeHandshake
} from './lunar-city-runner.mjs'
import { deriveRawSamplesFromProvenance } from './lib/lunar-city-provenance.mjs'

const SHA = 'a'.repeat(40)
const BINARY = '/packages/Hermes.app/Contents/MacOS/Hermes'
const STAMP = '/packages/Hermes.app/Contents/Resources/install-stamp.json'
const INFO = '/packages/Hermes.app/Contents/Info.plist'

function cleanStamp(overrides = {}) {
  return {
    schemaVersion: 1,
    commit: SHA,
    branch: null,
    builtAt: '2026-08-31T12:00:00.000Z',
    dirty: false,
    source: 'local',
    ...overrides
  }
}

function processRows(rendererPid, cpu = 1, rendererRssKiB = 204_800) {
  return [
    {
      pid: 10,
      type: 'Browser',
      cpu: { percentCPUUsage: cpu },
      memory: { workingSetSize: 102_400 }
    },
    {
      pid: rendererPid,
      type: 'Tab',
      cpu: { percentCPUUsage: cpu * 2 },
      memory: { workingSetSize: rendererRssKiB }
    }
  ]
}

function rendererMetrics(overrides = {}) {
  return {
    rendererGeneration: 1,
    rendererPid: 20,
    rendererStartedAtMs: 1_000,
    gpuMemoryMiB: 40,
    gpuMemorySource: 'chromium-memory-infra-v1',
    frameMs: 20,
    worldUpdateMs: 3,
    renderFrames: 5,
    drawCalls: 100,
    visibleTriangles: 500_000,
    activeAnimations: 10,
    entities: 100,
    textures: 20,
    listeners: 12,
    timers: 2,
    population: { observed: 100, active: 100, lodMix: { near: 100 }, source: 'fake-backend' },
    populationSourceMix: { 'fake-backend': 100 },
    qualityTier: 'Balanced',
    internalRenderScale: 1,
    targetFps: 15,
    cameraState: 'overview',
    cameraActions: { overview: 1, focus: 0, orbit: 0, zoom: 0, indoor: 0 },
    dialogueState: 'idle',
    dialogueActions: { opened: 0, messagesSent: 0, responsesReceived: 0 },
    lifecycleActions: { contextLosses: 0, recoveries: 0, disposals: 0 },
    qualityActions: { transitions: 0 },
    sceneMount: { id: 'scene-1', generation: 1, startedAtMs: 2_000 },
    lifecycleState: 'mounted',
    environment: {
      electronMode: 'packaged',
      gpuEnabled: true,
      windowState: { minimized: false, visible: true }
    },
    gpuEnabled: true,
    ...overrides
  }
}

function boundAction(action, payload, metrics, handshake, result = {}, sequence = 1) {
  return {
    action,
    proof: 1,
    ...result,
    bridgeBinding: {
      action: 'scenario-action',
      requestId: [
        'lcperf-v1',
        handshake.buildSha,
        handshake.launchNonce,
        handshake.mainPid,
        7,
        3,
        metrics.rendererPid,
        metrics.rendererGeneration,
        metrics.rendererStartedAtMs,
        sequence
      ].join(':'),
      payload: { action, payload },
      identity: {
        bridgeVersion: handshake.bridgeVersion,
        buildSha: handshake.buildSha,
        frameId: 3,
        launchNonce: handshake.launchNonce,
        mainPid: handshake.mainPid,
        rendererGeneration: metrics.rendererGeneration,
        rendererPid: metrics.rendererPid,
        rendererStartedAtMs: metrics.rendererStartedAtMs,
        senderId: 7
      }
    }
  }
}

function phase(name, points) {
  return {
    envelopeVersion: 3,
    phase: name,
    warmupDurationMs: 30_000,
    rendererIdentity: { generation: 1, pid: 20, startedAtMs: 1_000 },
    samples: points.map((point, index) => ({
      timestampMs: index * 1_000,
      processMetrics: processRows(20, point.cpu, point.rssKiB),
      rendererMetrics: rendererMetrics(point.renderer)
    }))
  }
}

function provenance() {
  return {
    provenanceVersion: 3,
    baselineShell: phase('baseline-shell', [
      { cpu: 1, rssKiB: 204_800, renderer: { gpuMemoryMiB: 40 } },
      { cpu: 1, rssKiB: 205_824, renderer: { gpuMemoryMiB: 42 } }
    ]),
    mountedCity: phase('mounted-city', [
      { cpu: 2, rssKiB: 307_200, renderer: { gpuMemoryMiB: 50 } },
      { cpu: 2.5, rssKiB: 309_248, renderer: { gpuMemoryMiB: 52 } }
    ])
  }
}

test('accepts only an existing packaged binary with an exact clean pinned embedded stamp', () => {
  const result = inspectPackagedTarget(
    { binaryPath: BINARY, expectedGitSha: SHA, platform: 'darwin' },
    {
      existsSync: path => path === BINARY || path === STAMP || path === INFO,
      readFileSync: path => {
        assert.equal(path, STAMP)
        return JSON.stringify(cleanStamp())
      },
      statSync: () => ({ isFile: () => true, mode: 0o100755 }),
      readInfoPlist: () => ({
        CFBundleIdentifier: 'com.nousresearch.hermes',
        CFBundleExecutable: 'Hermes',
        CFBundleName: 'Hermes'
      })
    }
  )

  assert.equal(result.binaryPath, BINARY)
  assert.equal(result.resourcesPath, '/packages/Hermes.app/Contents/Resources')
  assert.deepEqual(result.buildStamp, cleanStamp())
})

test('refuses dev Electron and missing or untrustworthy package stamps', () => {
  const deps = {
    existsSync: () => true,
    readFileSync: () => JSON.stringify(cleanStamp()),
    statSync: () => ({ isFile: () => true, mode: 0o100755 }),
    readInfoPlist: () => ({
      CFBundleIdentifier: 'com.nousresearch.hermes',
      CFBundleExecutable: 'Hermes',
      CFBundleName: 'Hermes'
    })
  }
  assert.throws(
    () =>
      inspectPackagedTarget(
        { binaryPath: '/repo/node_modules/electron/dist/Electron.app/Contents/MacOS/Electron', expectedGitSha: SHA },
        deps
      ),
    /dev Electron|packaged Hermes/i
  )

  for (const [label, stamp, pattern] of [
    ['missing', null, /install-stamp.*missing/i],
    ['mismatch', cleanStamp({ commit: 'b'.repeat(40) }), /does not match/i],
    ['dirty', cleanStamp({ dirty: true }), /dirty/i],
    ['fallback', cleanStamp({ source: 'fallback' }), /fallback/i]
  ]) {
    assert.throws(
      () =>
        inspectPackagedTarget(
          { binaryPath: BINARY, expectedGitSha: SHA },
          {
            ...deps,
            existsSync: path => (label === 'missing' ? path === BINARY || path === INFO : true),
            readFileSync: () => JSON.stringify(stamp)
          }
        ),
      pattern,
      label
    )
  }
})

test('rejects non-Hermes macOS bundles, arbitrary executables, and nonexecutable files', () => {
  const deps = {
    existsSync: () => true,
    readFileSync: () => JSON.stringify(cleanStamp()),
    statSync: () => ({ isFile: () => true, mode: 0o100755 }),
    readInfoPlist: () => ({
      CFBundleIdentifier: 'com.nousresearch.hermes',
      CFBundleExecutable: 'Hermes',
      CFBundleName: 'Hermes'
    })
  }
  assert.throws(
    () =>
      inspectPackagedTarget(
        { binaryPath: '/Applications/Calculator.app/Contents/MacOS/Calculator', expectedGitSha: SHA },
        deps
      ),
    /Hermes[.]app.*Hermes|bundle identity/i
  )
  assert.throws(
    () =>
      inspectPackagedTarget(
        { binaryPath: BINARY, expectedGitSha: SHA },
        { ...deps, statSync: () => ({ isFile: () => true, mode: 0o100644 }) }
      ),
    /executable/i
  )
  assert.throws(
    () =>
      inspectPackagedTarget(
        { binaryPath: BINARY, expectedGitSha: SHA },
        {
          ...deps,
          readInfoPlist: () => ({
            CFBundleIdentifier: 'com.apple.calculator',
            CFBundleExecutable: 'Hermes',
            CFBundleName: 'Hermes'
          })
        }
      ),
    /bundle identity/i
  )
})

test('builds an isolated packaged launch without disabling the GPU', () => {
  const previousDevServer = process.env.HERMES_DESKTOP_DEV_SERVER
  const previousRunAsNode = process.env.ELECTRON_RUN_AS_NODE
  const previousBootFake = process.env.HERMES_DESKTOP_BOOT_FAKE
  const previousDisableGpu = process.env.DISABLE_GPU
  const previousApiKey = process.env.OPENAI_API_KEY
  const previousGithubToken = process.env.GITHUB_TOKEN
  process.env.HERMES_DESKTOP_DEV_SERVER = 'http://127.0.0.1:5174'
  process.env.ELECTRON_RUN_AS_NODE = '1'
  process.env.HERMES_DESKTOP_BOOT_FAKE = '1'
  process.env.DISABLE_GPU = '1'
  process.env.OPENAI_API_KEY = 'secret'
  process.env.GITHUB_TOKEN = 'secret'
  const plan = createIsolatedLaunchPlan({
    binaryPath: BINARY,
    debugPort: 49321,
    tempRoot: '/private/tmp/lunar-city-run-7',
    runId: 'run-7',
    launchNonce: 'nonce-7'
  })

  assert.equal(plan.command, BINARY)
  assert.equal(plan.env.HERMES_HOME, '/private/tmp/lunar-city-run-7/hermes-home')
  assert.equal(plan.env.HERMES_DESKTOP_APP_NAME, 'Hermes Lunar City Perf run-7')
  assert.ok(plan.args.includes('--user-data-dir=/private/tmp/lunar-city-run-7/user-data'))
  assert.ok(plan.args.includes('--remote-debugging-port=49321'))
  assert.equal(
    plan.args.some(arg => arg === '--disable-gpu' || arg.startsWith('--disable-gpu=')),
    false
  )
  assert.equal('HERMES_DESKTOP_DEV_SERVER' in plan.env, false)
  assert.equal('ELECTRON_RUN_AS_NODE' in plan.env, false)
  for (const forbidden of ['HERMES_DESKTOP_BOOT_FAKE', 'DISABLE_GPU', 'OPENAI_API_KEY', 'GITHUB_TOKEN']) {
    assert.equal(forbidden in plan.env, false, forbidden)
  }
  assert.equal(plan.env.HERMES_LUNAR_CITY_PERF_NONCE, 'nonce-7')
  assert.equal(plan.env.HERMES_LUNAR_CITY_PERF_ACCEPTANCE, '1')
  if (previousDevServer === undefined) delete process.env.HERMES_DESKTOP_DEV_SERVER
  else process.env.HERMES_DESKTOP_DEV_SERVER = previousDevServer
  if (previousRunAsNode === undefined) delete process.env.ELECTRON_RUN_AS_NODE
  else process.env.ELECTRON_RUN_AS_NODE = previousRunAsNode
  for (const [key, previous] of [
    ['HERMES_DESKTOP_BOOT_FAKE', previousBootFake],
    ['DISABLE_GPU', previousDisableGpu],
    ['OPENAI_API_KEY', previousApiKey],
    ['GITHUB_TOKEN', previousGithubToken]
  ]) {
    if (previous === undefined) delete process.env[key]
    else process.env[key] = previous
  }
})

test('binds an explicit isolated fixture home and connection registry without inheriting secrets', () => {
  const plan = createIsolatedLaunchPlan({
    binaryPath: BINARY,
    debugPort: 49321,
    tempRoot: '/private/tmp/lunar-city-run-7',
    runId: 'run-7',
    launchNonce: 'nonce-7',
    fixture: {
      version: 'lunar-city-population-v3',
      evidenceClass: 'fake-backend-packaged',
      expectedPopulation: 25,
      hermesHome: '/private/tmp/lunar-city-fixture/hermes-home',
      contractPath: '/private/tmp/lunar-city-fixture/population.json',
      root: '/private/tmp/lunar-city-fixture',
      runNonce: 'nonce-7',
      userDataDir: '/private/tmp/lunar-city-fixture/user-data'
    }
  })

  assert.equal(plan.paths.hermesHome, '/private/tmp/lunar-city-fixture/hermes-home')
  assert.equal(plan.paths.userDataDir, '/private/tmp/lunar-city-fixture/user-data')
  assert.equal(plan.fixture.contractPath, '/private/tmp/lunar-city-fixture/population.json')
  assert.equal('HERMES_LUNAR_CITY_FIXTURE_CONTRACT' in plan.env, false)
  assert.equal('OPENAI_API_KEY' in plan.env, false)
  assert.equal('GITHUB_TOKEN' in plan.env, false)
})

test('rejects fixture paths that escape their declared isolated root', () => {
  assert.throws(
    () =>
      createIsolatedLaunchPlan({
        binaryPath: BINARY,
        debugPort: 49321,
        tempRoot: '/private/tmp/lunar-city-run-7',
        runId: 'run-7',
        launchNonce: 'nonce-7',
        fixture: {
          version: 'lunar-city-population-v3',
          evidenceClass: 'fake-backend-packaged',
          expectedPopulation: 25,
          hermesHome: '/Users/operator/.hermes',
          contractPath: '/private/tmp/lunar-city-fixture/population.json',
          root: '/private/tmp/lunar-city-fixture',
          runNonce: 'nonce-7',
          userDataDir: '/private/tmp/lunar-city-fixture/user-data'
        }
      }),
    /escapes.*root|isolated fixture/i
  )
})

test('reserves distinct free loopback debug ports until each reservation is released', async () => {
  const first = await reserveUniqueDebugPort()
  const second = await reserveUniqueDebugPort()
  try {
    assert.notEqual(first.port, second.port)
    assert.ok(first.port >= 1024 && first.port <= 65535)
    assert.ok(second.port >= 1024 && second.port <= 65535)
  } finally {
    await first.release()
    await second.release()
  }
})

test('requires an exact versioned packaged bridge handshake bound to nonce, build, and process lifetime', () => {
  const valid = {
    bridgeVersion: 1,
    launchNonce: 'nonce-7',
    buildSha: SHA,
    packaged: true,
    mainPid: 999,
    rendererIdentity: { pid: 20, startedAtMs: 1_000 },
    supportedPhases: ['baseline-shell', 'mounted-city'],
    processMetricsSource: 'electron.app.getAppMetrics'
  }
  assert.deepEqual(validateBridgeHandshake(valid, { launchNonce: 'nonce-7', buildSha: SHA, mainPid: 999 }), valid)
  for (const [patch, pattern] of [
    [null, /bridge_unavailable/],
    [{ launchNonce: 'incumbent' }, /bridge_mismatch.*nonce/],
    [{ buildSha: 'b'.repeat(40) }, /bridge_mismatch.*SHA/],
    [{ mainPid: 111 }, /bridge_mismatch.*main PID/],
    [{ packaged: false }, /bridge_mismatch.*packaged/]
  ]) {
    assert.throws(
      () =>
        validateBridgeHandshake(patch === null ? null : { ...valid, ...patch }, {
          launchNonce: 'nonce-7',
          buildSha: SHA,
          mainPid: 999
        }),
      pattern
    )
  }
})

test('executes and awaits one real non-disposal bridge action exactly once', async () => {
  const evaluations = []
  const handshake = { bridgeVersion: 1, buildSha: SHA, launchNonce: 'nonce-7', mainPid: 999 }
  const before = rendererMetrics({
    internalRenderScale: 1.25,
    qualityTier: 'Detailed',
    rendererGeneration: 1
  })
  const results = [
    boundAction(
      'quality',
      { tier: 'balanced' },
      before,
      handshake,
      {
        from: { internalRenderScale: 1.25, tier: 'detailed' },
        to: { internalRenderScale: 1, tier: 'balanced' }
      },
      1
    ),
    boundAction(
      'quality',
      { tier: 'efficient' },
      before,
      handshake,
      {
        from: { internalRenderScale: 1, tier: 'balanced' },
        proof: 2,
        to: { internalRenderScale: 0.75, tier: 'efficient' }
      },
      3
    )
  ]
  const observations = [
    rendererMetrics({
      internalRenderScale: 1,
      qualityActions: { transitions: 1 },
      qualityTier: 'Balanced',
      rendererGeneration: 1
    }),
    rendererMetrics({
      internalRenderScale: 0.75,
      qualityActions: { transitions: 2 },
      qualityTier: 'Efficient',
      rendererGeneration: 1
    })
  ]
  const result = await runScenarioThroughBridge(
    {
      eval: async expression => {
        evaluations.push(expression)

        return results.shift()
      }
    },
    'tier-efficient',
    before,
    handshake,
    async () => observations.shift()
  )

  assert.equal(evaluations.length, 2)
  assert.match(evaluations[0], /probe[.]runAction\("quality",\s*\{"tier":"balanced"\}\)/u)
  assert.match(evaluations[1], /probe[.]runAction\("quality",\s*\{"tier":"efficient"\}\)/u)
  assert.equal(result.actions.length, 2)
  assert.equal(result.actions[0].observed.qualityTier, 'Balanced')
  assert.equal(result.actions[1].observed.qualityTier, 'Efficient')
  assert.deepEqual(result.before, before)
})

test('tier-balanced establishes an authoritative Efficient prestate before one Balanced transition', async () => {
  const handshake = { bridgeVersion: 1, buildSha: SHA, launchNonce: 'nonce-7', mainPid: 999 }
  const initial = rendererMetrics({ qualityTier: 'Balanced' })
  const results = [
    boundAction(
      'quality',
      { tier: 'efficient' },
      initial,
      handshake,
      {
        from: { internalRenderScale: 1, tier: 'balanced' },
        to: { internalRenderScale: 0.75, tier: 'efficient' }
      },
      1
    ),
    boundAction(
      'quality',
      { tier: 'balanced' },
      initial,
      handshake,
      {
        from: { internalRenderScale: 0.75, tier: 'efficient' },
        proof: 2,
        to: { internalRenderScale: 1, tier: 'balanced' }
      },
      3
    )
  ]
  const observed = [
    rendererMetrics({
      internalRenderScale: 0.75,
      qualityActions: { transitions: 1 },
      qualityTier: 'Efficient'
    }),
    rendererMetrics({ qualityActions: { transitions: 2 }, qualityTier: 'Balanced' })
  ]
  const execution = await runScenarioThroughBridge(
    { eval: async () => results.shift() },
    'tier-balanced',
    initial,
    handshake,
    async () => observed.shift()
  )
  assert.equal(execution.preparation.result.to.tier, 'efficient')
  assert.equal(execution.before.qualityTier, 'Efficient')
  assert.deepEqual(
    execution.actions.map(entry => entry.result.to.tier),
    ['balanced']
  )
})

test('rejects actionless, duplicated, wrong-target, and wrong-lifetime scenario evidence', async () => {
  await assert.rejects(
    runScenarioThroughBridge({ eval: async () => ({}) }, 'unknown-scenario', rendererMetrics(), {}),
    /unsupported|action plan/i
  )

  const metrics = rendererMetrics({ rendererGeneration: 1 })
  const handshake = { bridgeVersion: 1, buildSha: SHA, launchNonce: 'nonce-7', mainPid: 999 }
  for (const [label, mutate] of [
    ['wrong target', result => (result.bridgeBinding.payload.payload.entityKey = 'wrong-worker')],
    ['wrong lifetime', result => (result.bridgeBinding.identity.rendererGeneration = 2)],
    ['wrong nonce', result => (result.bridgeBinding.identity.launchNonce = 'wrong')]
  ]) {
    await assert.rejects(
      runScenarioThroughBridge(
        {
          eval: async () => {
            const result = {
              action: 'focus',
              proof: 1,
              bridgeBinding: {
                action: 'scenario-action',
                requestId: '7:20:1:1',
                payload: { action: 'focus', payload: { entityKey: metrics.scenarioTargets?.workerEntityKey } },
                identity: {
                  bridgeVersion: 1,
                  buildSha: SHA,
                  frameId: 3,
                  launchNonce: 'nonce-7',
                  mainPid: 999,
                  rendererGeneration: 1,
                  rendererPid: 20,
                  rendererStartedAtMs: 1_000,
                  senderId: 7
                }
              }
            }
            mutate(result)
            return result
          }
        },
        'balanced-worker-focus',
        { ...metrics, scenarioTargets: { workerEntityKey: 'worker-1', leaderId: 'owl' } },
        handshake
      ),
      /proof|binding|identity|payload|target/i,
      label
    )
  }
})

test('provenance rejects spliced actions, duplicate requests, wrong targets, lifetimes, and seeded counters', () => {
  const raw = provenance()
  const handshake = { bridgeVersion: 1, buildSha: SHA, launchNonce: 'nonce-7', mainPid: 999 }
  raw.bridgeHandshake = handshake
  const before = structuredClone(raw.mountedCity.samples[0].rendererMetrics)
  before.scenarioTargets = { workerEntityKey: 'worker-1', leaderId: 'owl' }
  raw.mountedCity.samples.forEach(sample => {
    sample.rendererMetrics.cameraActions.focus = 1
    sample.rendererMetrics.cameraState = 'worker-focus'
  })
  const action = boundAction('focus', { entityKey: 'worker-1' }, before, handshake, { entityKey: 'worker-1' })
  raw.mountedCity.scenarioExecution = {
    actions: [{ action: 'focus', result: action }],
    authority: { frameId: 3, senderId: 7 },
    before,
    initial: before,
    scenario: 'balanced-worker-focus'
  }
  assert.doesNotThrow(() => deriveRawSamplesFromProvenance(raw, { scenario: 'balanced-worker-focus' }))

  for (const [label, mutate] of [
    ['spliced action', value => (value.mountedCity.scenarioExecution.actions[0].action = 'orbit')],
    [
      'duplicate request',
      value =>
        value.mountedCity.scenarioExecution.actions.push(
          structuredClone(value.mountedCity.scenarioExecution.actions[0])
        )
    ],
    [
      'wrong target',
      value =>
        (value.mountedCity.scenarioExecution.actions[0].result.bridgeBinding.payload.payload.entityKey = 'worker-2')
    ],
    [
      'wrong lifetime',
      value => (value.mountedCity.scenarioExecution.actions[0].result.bridgeBinding.identity.rendererGeneration = 2)
    ],
    ['seeded counter', value => (value.mountedCity.scenarioExecution.before.cameraActions.focus = 1)],
    [
      'same-lifetime sender splice',
      value => {
        const binding = value.mountedCity.scenarioExecution.actions[0].result.bridgeBinding
        binding.identity.senderId = 8
        binding.requestId = binding.requestId.replace(':7:3:20:', ':8:3:20:')
      }
    ],
    [
      'same-lifetime frame splice',
      value => {
        const binding = value.mountedCity.scenarioExecution.actions[0].result.bridgeBinding
        binding.identity.frameId = 4
        binding.requestId = binding.requestId.replace(':7:3:20:', ':7:4:20:')
      }
    ],
    [
      'malformed request id',
      value => (value.mountedCity.scenarioExecution.actions[0].result.bridgeBinding.requestId = 'not-canonical')
    ]
  ]) {
    const adversarial = structuredClone(raw)
    mutate(adversarial)
    assert.throws(
      () => deriveRawSamplesFromProvenance(adversarial, { scenario: 'balanced-worker-focus' }),
      /action|proof|identity|payload|target|counter|plan|request|sender|frame/i,
      label
    )
  }
})

test('hidden, minimized, and route-unmounted require authoritative state transitions', () => {
  const handshake = { bridgeVersion: 1, buildSha: SHA, launchNonce: 'nonce-7', mainPid: 999 }
  for (const [scenario, action] of [
    ['hidden', 'window-hidden'],
    ['minimized', 'window-minimized']
  ]) {
    const raw = provenance()
    raw.bridgeHandshake = handshake
    const before = structuredClone(raw.mountedCity.samples[0].rendererMetrics)
    const result = boundAction(action, {}, before, handshake, {
      windowState:
        action === 'window-hidden' ? { minimized: false, visible: false } : { minimized: true, visible: false },
      windowTrace: []
    })
    raw.mountedCity.scenarioExecution = {
      actions: [{ action, result }],
      authority: { frameId: 3, senderId: 7 },
      before,
      initial: before,
      scenario
    }
    assert.throws(() => deriveRawSamplesFromProvenance(raw, { scenario }), /window state|transition/i)
  }

  const route = provenance()
  route.bridgeHandshake = handshake
  const before = structuredClone(route.mountedCity.samples[0].rendererMetrics)
  route.mountedCity.scenarioExecution = {
    actions: [{ action: 'dispose', result: boundAction('dispose', {}, before, handshake) }],
    authority: { frameId: 3, senderId: 7 },
    before,
    initial: before,
    scenario: 'route-unmounted'
  }
  assert.throws(
    () => deriveRawSamplesFromProvenance(route, { scenario: 'route-unmounted' }),
    /disposal|route-unmounted/i
  )
})

test('quality provenance rejects no-op, preseeded counters, wrong plans, and out-of-order requests', () => {
  const raw = provenance()
  const handshake = { bridgeVersion: 1, buildSha: SHA, launchNonce: 'nonce-7', mainPid: 999 }
  raw.bridgeHandshake = handshake
  const before = rendererMetrics({
    internalRenderScale: 1.25,
    qualityActions: { transitions: 1 },
    qualityTier: 'Detailed'
  })
  const balanced = rendererMetrics({
    internalRenderScale: 1,
    qualityActions: { transitions: 2 },
    qualityTier: 'Balanced'
  })
  const efficient = rendererMetrics({
    internalRenderScale: 0.75,
    qualityActions: { transitions: 3 },
    qualityTier: 'Efficient'
  })
  raw.mountedCity.samples.forEach(sample => Object.assign(sample.rendererMetrics, efficient))
  raw.mountedCity.scenarioExecution = {
    actions: [
      {
        action: 'quality',
        observed: balanced,
        result: boundAction(
          'quality',
          { tier: 'balanced' },
          before,
          handshake,
          {
            from: { internalRenderScale: 1.25, tier: 'detailed' },
            proof: 2,
            to: { internalRenderScale: 1, tier: 'balanced' }
          },
          1
        )
      },
      {
        action: 'quality',
        observed: efficient,
        result: boundAction(
          'quality',
          { tier: 'efficient' },
          before,
          handshake,
          {
            from: { internalRenderScale: 1, tier: 'balanced' },
            proof: 3,
            to: { internalRenderScale: 0.75, tier: 'efficient' }
          },
          3
        )
      }
    ],
    authority: { frameId: 3, senderId: 7 },
    before,
    initial: before,
    scenario: 'tier-efficient'
  }
  assert.doesNotThrow(() => deriveRawSamplesFromProvenance(raw, { scenario: 'tier-efficient' }))

  for (const [label, mutate] of [
    [
      'no-op',
      value =>
        (value.mountedCity.scenarioExecution.actions[0].result.from = {
          internalRenderScale: 1,
          tier: 'balanced'
        })
    ],
    ['preseeded counter', value => (value.mountedCity.scenarioExecution.before.qualityActions.transitions = 2)],
    ['wrong plan', value => value.mountedCity.scenarioExecution.actions.reverse()],
    [
      'out-of-order request',
      value =>
        (value.mountedCity.scenarioExecution.actions[1].result.bridgeBinding.requestId = `lcperf-v1:${SHA}:nonce-7:999:7:3:20:1:1000:1`)
    ]
  ]) {
    const adversarial = structuredClone(raw)
    mutate(adversarial)
    assert.throws(
      () => deriveRawSamplesFromProvenance(adversarial, { scenario: 'tier-efficient' }),
      /quality|counter|action|plan|request|proof/i,
      label
    )
  }
})

test('derives CPU, renderer RSS, and GPU deltas from retained baseline and city samples', () => {
  const derived = deriveRawSamplesFromProvenance(provenance())

  assert.deepEqual(derived.rawSamples.cpuDeltaPp, [3, 4.5])
  assert.deepEqual(derived.rawSamples.residentMemoryMiB, [300, 302])
  assert.deepEqual(derived.rawSamples.gpuMemoryDeltaMiB, [9, 11])
  assert.deepEqual(derived.resourceDeltas.rendererRssDeltaMiB, [99.5, 101.5])
  assert.deepEqual(derived.sampleTimestampsMs, [0, 1_000])
  assert.equal(derived.rendererIdentity.pid, 20)
})

test('refuses renderer lifetime changes and RSS masquerading as GPU memory', () => {
  const changed = provenance()
  changed.mountedCity.samples[1].rendererMetrics.rendererPid = 21
  assert.throws(() => deriveRawSamplesFromProvenance(changed), /renderer.*identity|lifetime/i)

  const fakeGpu = provenance()
  fakeGpu.mountedCity.samples[0].rendererMetrics.gpuMemorySource = 'electron-app-metrics-rss'
  assert.throws(() => deriveRawSamplesFromProvenance(fakeGpu), /GPU.*source|RSS.*GPU/i)

  const gpuProcessRss = provenance()
  gpuProcessRss.mountedCity.samples[0].rendererMetrics.gpuMemorySource = 'gpu-process-private-memory'
  assert.throws(() => deriveRawSamplesFromProvenance(gpuProcessRss), /GPU.*source|process.*memory|RSS.*GPU/i)
  const opaque = provenance()
  opaque.mountedCity.samples[0].rendererMetrics.gpuMemorySource = 'babylon-engine-counter'
  assert.throws(() => deriveRawSamplesFromProvenance(opaque), /GPU.*source|attributable/i)
})

test('allows an empty baseline shell but requires a consistent exact mounted population', () => {
  const raw = provenance()
  for (const sample of raw.baselineShell.samples) {
    sample.rendererMetrics.population = { observed: 0, active: 0, lodMix: {}, source: 'unmounted' }
    sample.rendererMetrics.populationSourceMix = {}
  }
  assert.doesNotThrow(() => deriveRawSamplesFromProvenance(raw))
  raw.mountedCity.samples[1].rendererMetrics.population.observed = 99
  assert.throws(() => deriveRawSamplesFromProvenance(raw), /population.*consistent|exact/i)
})

test('rejects action counter resets within one mounted phase', () => {
  const raw = provenance()
  raw.mountedCity.samples[0].rendererMetrics.cameraActions.orbit = 2
  raw.mountedCity.samples[1].rendererMetrics.cameraActions.orbit = 1
  assert.throws(() => deriveRawSamplesFromProvenance(raw), /cameraActions.*decrease|counter.*reset/i)
})

test('refuses an empty city, disabled GPU, and unavailable required metrics', () => {
  const empty = provenance()
  empty.mountedCity.samples[0].rendererMetrics.population = {
    observed: 0,
    active: 0,
    lodMix: {},
    source: 'fake-backend'
  }
  assert.throws(() => deriveRawSamplesFromProvenance(empty), /empty city|populated/i)

  const disabled = provenance()
  disabled.mountedCity.samples[0].rendererMetrics.gpuEnabled = false
  assert.throws(() => deriveRawSamplesFromProvenance(disabled), /GPU.*disabled|GPU.*enabled/i)

  const unavailable = provenance()
  delete unavailable.mountedCity.samples[0].rendererMetrics.worldUpdateMs
  assert.throws(() => deriveRawSamplesFromProvenance(unavailable), /worldUpdateMs.*unavailable|required metric/i)
})

test('orchestrates injected packaged launcher, CDP, process, renderer, and clock probes', async () => {
  const launches = []
  const phases = []
  const scenarioRuns = []
  let now = 0
  let qualityTier = 'Balanced'
  let qualityTransitions = 0
  const result = await runPackagedLunarCityMeasurement(
    {
      binaryPath: BINARY,
      expectedGitSha: SHA,
      sampleCount: 2,
      sampleIntervalMs: 1_000,
      scenario: 'tier-efficient',
      warmupDurationMs: 0
    },
    {
      inspectTarget: () => ({ binaryPath: BINARY, buildStamp: cleanStamp() }),
      makeTempRoot: () => '/private/tmp/lunar-city-run-7',
      makeLaunchNonce: () => 'nonce-7',
      reserveDebugPort: async () => ({ port: 49321, release: async () => {} }),
      launch: plan => {
        launches.push(plan)
        return { pid: 999, kill() {} }
      },
      connectCdp: async ({ port }) => ({
        cdp: { port, close() {} },
        handshake: {
          bridgeVersion: 1,
          launchNonce: 'nonce-7',
          buildSha: SHA,
          packaged: true,
          mainPid: 999,
          rendererIdentity: { pid: 20, startedAtMs: 1_000 },
          supportedPhases: ['baseline-shell', 'mounted-city'],
          processMetricsSource: 'electron.app.getAppMetrics'
        }
      }),
      preparePhase: async (_cdp, name) => phases.push(name),
      processProbe: async () => processRows(20, phases.length === 1 ? 1 : 2),
      rendererProbe: async () =>
        rendererMetrics({
          gpuMemoryMiB: phases.length === 1 ? 40 : 50,
          internalRenderScale: qualityTier === 'Efficient' ? 0.75 : 1,
          qualityActions: { transitions: qualityTransitions },
          qualityTier
        }),
      runScenario: async (_cdp, scenario, before, handshake) => {
        scenarioRuns.push(scenario)
        const detailed = rendererMetrics({
          gpuMemoryMiB: 50,
          internalRenderScale: 1.25,
          qualityActions: { transitions: 1 },
          qualityTier: 'Detailed'
        })
        const balanced = rendererMetrics({
          gpuMemoryMiB: 50,
          qualityActions: { transitions: 2 },
          qualityTier: 'Balanced'
        })
        const efficient = rendererMetrics({
          gpuMemoryMiB: 50,
          internalRenderScale: 0.75,
          qualityActions: { transitions: 3 },
          qualityTier: 'Efficient'
        })
        qualityTier = 'Efficient'
        qualityTransitions = 3

        return {
          actions: [
            {
              action: 'quality',
              observed: balanced,
              result: boundAction(
                'quality',
                { tier: 'balanced' },
                before,
                handshake,
                {
                  from: { internalRenderScale: 1.25, tier: 'detailed' },
                  proof: 2,
                  to: { internalRenderScale: 1, tier: 'balanced' }
                },
                3
              )
            },
            {
              action: 'quality',
              observed: efficient,
              result: boundAction(
                'quality',
                { tier: 'efficient' },
                before,
                handshake,
                {
                  from: { internalRenderScale: 1, tier: 'balanced' },
                  proof: 3,
                  to: { internalRenderScale: 0.75, tier: 'efficient' }
                },
                5
              )
            }
          ],
          authority: { frameId: 3, senderId: 7 },
          before: detailed,
          initial: before,
          preparation: {
            action: 'quality',
            observed: detailed,
            result: boundAction(
              'quality',
              { tier: 'detailed' },
              before,
              handshake,
              {
                from: { internalRenderScale: 1, tier: 'balanced' },
                to: { internalRenderScale: 1.25, tier: 'detailed' }
              },
              1
            )
          },
          scenario
        }
      },
      clock: {
        now: () => now,
        sleep: async milliseconds => {
          now += milliseconds
        }
      },
      cleanup: async () => {},
      captureHostEnvironment: () => ({
        architecture: 'arm64',
        hardwareModel: 'Test Mac',
        os: 'macOS 15.6',
        powerState: 'ac'
      })
    }
  )

  assert.equal(launches.length, 1)
  assert.equal(launches[0].env.HERMES_LUNAR_CITY_PERF_NONCE, 'nonce-7')
  assert.equal(launches[0].env.HERMES_LUNAR_CITY_PERF_ACCEPTANCE, '1')
  assert.deepEqual(phases, ['baseline-shell', 'mounted-city'])
  assert.deepEqual(scenarioRuns, ['tier-efficient'])
  assert.equal(result.rawProvenance.mountedCity.scenarioExecution.scenario, 'tier-efficient')
  assert.equal(result.rawProvenance.mountedCity.scenarioExecution.actions.length, 2)
  assert.equal(
    result.rawProvenance.mountedCity.scenarioExecution.actions[0].result.bridgeBinding.requestId,
    `lcperf-v1:${SHA}:nonce-7:999:7:3:20:1:1000:3`
  )
  assert.equal(result.rawProvenance.mountedCity.scenarioExecution.before.qualityTier, 'Detailed')
  assert.equal(result.rawProvenance.mountedCity.samples.length, 2)
  assert.equal(
    result.rawProvenance.mountedCity.samples.every(sample => sample.rendererMetrics.qualityTier === 'Efficient'),
    true
  )
  assert.equal(result.rawProvenance.provenanceVersion, 3)
  assert.deepEqual(result.rawSamples.cpuDeltaPp, [3, 3])
  assert.equal(result.buildStamp.commit, SHA)
  assert.equal(result.hostEnvironment.hardwareModel, 'Test Mac')
  assert.equal(result.packagedPerformanceEligible, false)
  assert.equal(result.evidenceClass, 'deterministic')
})

test('captures mounted disposal samples followed by exactly one truthful terminal sample', async () => {
  let now = 0
  let phaseName = 'baseline-shell'
  let disposed = false
  let mountedRendererProbes = 0
  let scenarioRuns = 0

  const result = await runPackagedLunarCityMeasurement(
    {
      binaryPath: BINARY,
      expectedGitSha: SHA,
      sampleCount: 2,
      sampleIntervalMs: 1_000,
      scenario: 'disposal',
      warmupDurationMs: 0
    },
    {
      inspectTarget: () => ({ binaryPath: BINARY, buildStamp: cleanStamp() }),
      makeTempRoot: () => '/private/tmp/lunar-city-disposal',
      makeLaunchNonce: () => 'nonce-7',
      reserveDebugPort: async () => ({ port: 49321, release: async () => {} }),
      launch: () => ({ pid: 999, kill() {} }),
      connectCdp: async ({ port }) => ({
        cdp: { port, close() {} },
        handshake: {
          bridgeVersion: 1,
          launchNonce: 'nonce-7',
          buildSha: SHA,
          packaged: true,
          mainPid: 999,
          rendererIdentity: { pid: 20, startedAtMs: 1_000 },
          supportedPhases: ['baseline-shell', 'mounted-city'],
          processMetricsSource: 'electron.app.getAppMetrics'
        }
      }),
      preparePhase: async (_cdp, name) => {
        phaseName = name
      },
      processProbe: async () => processRows(20, phaseName === 'baseline-shell' ? 1 : 2),
      rendererProbe: async () => {
        if (phaseName === 'mounted-city') mountedRendererProbes += 1

        return disposed
          ? rendererMetrics({
              activeAnimations: 0,
              drawCalls: 0,
              entities: 0,
              frameMs: 0,
              lifecycleActions: { contextLosses: 0, recoveries: 0, disposals: 1 },
              lifecycleState: 'disposed',
              listeners: 0,
              population: { observed: 0, active: 0, lodMix: {}, source: 'route-unmounted' },
              populationSourceMix: {},
              renderFrames: 0,
              textures: 0,
              timers: 0,
              visibleTriangles: 0,
              worldUpdateMs: 0
            })
          : rendererMetrics({ gpuMemoryMiB: phaseName === 'baseline-shell' ? 40 : 50 })
      },
      runScenario: async (_cdp, scenario, before, handshake) => {
        scenarioRuns += 1
        disposed = true

        return {
          actions: [{ action: 'dispose', result: boundAction('dispose', {}, before, handshake) }],
          authority: { frameId: 3, senderId: 7 },
          before,
          initial: before,
          scenario
        }
      },
      clock: {
        now: () => now,
        sleep: async milliseconds => {
          now += milliseconds
        }
      },
      cleanup: async () => {},
      captureHostEnvironment: () => ({
        architecture: 'arm64',
        hardwareModel: 'Test Mac',
        os: 'macOS 15.6',
        powerState: 'ac'
      })
    }
  )

  const samples = result.rawProvenance.mountedCity.samples
  assert.equal(scenarioRuns, 1)
  assert.equal(mountedRendererProbes, 3)
  assert.equal(samples.length, 3)
  assert.deepEqual(
    samples.map(sample => sample.rendererMetrics.lifecycleState),
    ['mounted', 'mounted', 'disposed']
  )
  assert.deepEqual(samples.at(-1).rendererMetrics.population, {
    observed: 0,
    active: 0,
    lodMix: {},
    source: 'route-unmounted'
  })
  assert.equal(result.rawProvenance.mountedCity.scenarioExecution.scenario, 'disposal')
  assert.equal(result.rawProvenance.mountedCity.scenarioExecution.actions.length, 1)
  assert.equal(
    result.rawProvenance.mountedCity.scenarioExecution.actions[0].result.bridgeBinding.requestId,
    `lcperf-v1:${SHA}:nonce-7:999:7:3:20:1:1000:1`
  )
  assert.equal(result.rawProvenance.mountedCity.scenarioExecution.before.lifecycleState, 'mounted')
})

test('cleanup attempts close, terminate, wait, kill, and removal despite earlier failures', async () => {
  const calls = []
  await assert.rejects(
    cleanupIsolatedRun(
      {
        cdp: {
          close: () => {
            calls.push('close')
            throw new Error('close failed')
          }
        },
        child: {
          kill: signal => {
            calls.push(signal)
            if (signal === 'SIGTERM') throw new Error('term failed')
          },
          waitForExit: async () => {
            calls.push('wait')
            throw new Error('wait failed')
          }
        },
        tempRoot: '/tmp/run'
      },
      {
        removeTemp: () => {
          calls.push('remove')
        }
      }
    ),
    /cleanup failed/i
  )
  assert.deepEqual(calls, ['close', 'SIGTERM', 'wait', 'SIGKILL', 'remove'])
})
