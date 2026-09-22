#!/usr/bin/env node

import { spawn } from 'node:child_process'
import { execFileSync } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, statSync } from 'node:fs'
import { createServer } from 'node:net'
import { arch, release, tmpdir, type as osType } from 'node:os'
import { basename, dirname, join, normalize, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { CDP } from './lib/cdp.mjs'
import {
  LUNAR_CITY_PHASE_ENVELOPE_VERSION,
  LUNAR_CITY_PROVENANCE_VERSION,
  deriveRawSamplesFromProvenance
} from './lib/lunar-city-provenance.mjs'

const SHA_PATTERN = /^[0-9a-f]{40}$/i
const CANONICAL_ISO_UTC = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/
const BRIDGE_VERSION = 1

function defaultHostEnvironment() {
  if (process.platform !== 'darwin') throw new Error('packaged acceptance host capture currently requires macOS')
  const hardwareModel = execFileSync('/usr/sbin/sysctl', ['-n', 'hw.model'], { encoding: 'utf8' }).trim()
  const power = execFileSync('/usr/bin/pmset', ['-g', 'batt'], { encoding: 'utf8' })
  const powerState = /AC Power/iu.test(power) ? 'ac' : /Battery Power/iu.test(power) ? 'battery' : ''
  if (!hardwareModel || !powerState) throw new Error('authoritative host hardware/power state is unavailable')
  return { architecture: arch(), hardwareModel, os: `${osType()} ${release()}`, powerState }
}

function resourcesPathFor(binaryPath, platform = process.platform) {
  const binary = resolve(binaryPath)
  if (platform === 'darwin') {
    const marker = `${normalize('.app')}/Contents/MacOS/`
    const index = binary.lastIndexOf(marker)
    if (index < 0) throw new Error('packaged Hermes macOS binary must live under Hermes.app/Contents/MacOS')
    return join(binary.slice(0, index + marker.length - 'MacOS/'.length), 'Resources')
  }
  return join(dirname(binary), 'resources')
}

function isDevElectron(binaryPath) {
  const normalized = normalize(binaryPath).toLowerCase()
  return (
    basename(normalized).replace(/\.exe$/u, '') === 'electron' ||
    normalized.includes(`${normalize('node_modules/electron')}/`) ||
    normalized.includes(`${normalize('electron/dist')}/`)
  )
}

function parseBuildStamp(raw) {
  let stamp
  try {
    stamp = JSON.parse(raw)
  } catch {
    throw new Error('packaged install-stamp.json is malformed')
  }
  if (!stamp || typeof stamp !== 'object' || Array.isArray(stamp) || stamp.schemaVersion !== 1) {
    throw new Error('packaged install-stamp.json has an unsupported schema')
  }
  if (!SHA_PATTERN.test(stamp.commit ?? '')) throw new Error('packaged install-stamp commit is not an exact SHA')
  if (!CANONICAL_ISO_UTC.test(stamp.builtAt ?? '')) throw new Error('packaged install-stamp builtAt is invalid')
  if (stamp.dirty !== false) throw new Error('packaged install-stamp is dirty')
  if (stamp.source === 'fallback' || /^0{40}$/u.test(stamp.commit)) {
    throw new Error('packaged install-stamp uses fallback provenance')
  }
  if (!['local', 'ci'].includes(stamp.source)) throw new Error('packaged install-stamp source is not pinned')
  return stamp
}

/** Resolve and verify a pre-existing electron-builder package before launch. */
export function inspectPackagedTarget(
  { binaryPath, expectedGitSha, platform = process.platform },
  deps = {
    existsSync,
    readFileSync,
    statSync,
    readInfoPlist: path =>
      JSON.parse(execFileSync('/usr/bin/plutil', ['-convert', 'json', '-o', '-', path], { encoding: 'utf8' }))
  }
) {
  if (typeof binaryPath !== 'string' || binaryPath.length === 0)
    throw new Error('packaged Hermes binary path is required')
  if (!SHA_PATTERN.test(expectedGitSha ?? '')) throw new Error('expected packaged git SHA must be exact')
  if (isDevElectron(binaryPath))
    throw new Error('dev Electron is forbidden; target an electron-builder packaged Hermes binary')
  if (
    platform === 'darwin' &&
    !normalize(resolve(binaryPath)).endsWith(normalize('/Hermes.app/Contents/MacOS/Hermes'))
  ) {
    throw new Error('packaged Hermes bundle identity requires Hermes.app/Contents/MacOS/Hermes')
  }
  if (!deps.existsSync(binaryPath) || !deps.statSync(binaryPath).isFile()) {
    throw new Error(`packaged Hermes binary is missing: ${binaryPath}`)
  }
  if (platform !== 'win32' && (deps.statSync(binaryPath).mode & 0o111) === 0) {
    throw new Error(`packaged Hermes binary is not executable: ${binaryPath}`)
  }
  const resourcesPath = resourcesPathFor(binaryPath, platform)
  if (platform === 'darwin') {
    const infoPath = join(dirname(dirname(binaryPath)), 'Info.plist')
    if (!deps.existsSync(infoPath)) throw new Error(`packaged Hermes Info.plist is missing: ${infoPath}`)
    const info = deps.readInfoPlist(infoPath)
    if (
      info?.CFBundleIdentifier !== 'com.nousresearch.hermes' ||
      info?.CFBundleExecutable !== 'Hermes' ||
      info?.CFBundleName !== 'Hermes'
    ) {
      throw new Error('packaged Hermes bundle identity does not match electron-builder Hermes')
    }
  }
  const stampPath = join(resourcesPath, 'install-stamp.json')
  if (!deps.existsSync(stampPath)) throw new Error(`packaged install-stamp.json is missing: ${stampPath}`)
  const buildStamp = parseBuildStamp(deps.readFileSync(stampPath, 'utf8'))
  if (buildStamp.commit.toLowerCase() !== expectedGitSha.toLowerCase()) {
    throw new Error(`packaged install-stamp commit ${buildStamp.commit} does not match ${expectedGitSha}`)
  }
  return { binaryPath: resolve(binaryPath), resourcesPath, stampPath, buildStamp }
}

/** Build the direct packaged-binary launch contract without mutating the filesystem. */
export function createIsolatedLaunchPlan({ binaryPath, debugPort, tempRoot, runId, launchNonce, fixture }) {
  if (!Number.isInteger(debugPort) || debugPort < 1024 || debugPort > 65535) {
    throw new Error('isolated CDP debug port must be an integer from 1024 through 65535')
  }
  if (typeof tempRoot !== 'string' || tempRoot.length === 0) throw new Error('isolated temp root is required')
  const safeRunId = String(runId || 'run').replace(/[^a-z0-9._-]/giu, '-')
  let hermesHome = join(tempRoot, 'hermes-home')
  let userDataDir = join(tempRoot, 'user-data')
  let fixtureBinding
  if (fixture) {
    if (
      fixture.version !== 'lunar-city-population-v3' ||
      fixture.evidenceClass !== 'fake-backend-packaged' ||
      ![25, 100, 250].includes(fixture.expectedPopulation)
    ) {
      throw new Error('isolated fixture contract is malformed')
    }
    const fixtureRoot = resolve(fixture.root ?? '')
    if (fixtureRoot === resolve('/') || fixtureRoot.length < 2) throw new Error('isolated fixture root is invalid')
    const withinFixture = (path, field) => {
      const resolved = resolve(path ?? '')
      if (resolved !== fixtureRoot && !resolved.startsWith(`${fixtureRoot}/`)) {
        throw new Error(`isolated fixture ${field} escapes its root`)
      }
      return resolved
    }
    hermesHome = withinFixture(fixture.hermesHome, 'hermesHome')
    userDataDir = withinFixture(fixture.userDataDir, 'userDataDir')
    fixtureBinding = {
      version: fixture.version,
      evidenceClass: fixture.evidenceClass,
      expectedPopulation: fixture.expectedPopulation,
      contractPath: withinFixture(fixture.contractPath, 'contractPath'),
      root: fixtureRoot,
      runNonce: fixture.runNonce
    }
  }
  const launchEnv = {}
  for (const key of [
    'PATH',
    'TMPDIR',
    'TEMP',
    'TMP',
    'SystemRoot',
    'WINDIR',
    'LANG',
    'LC_ALL',
    'SHELL',
    'USER',
    'LOGNAME',
    'DISPLAY',
    'WAYLAND_DISPLAY',
    'XDG_RUNTIME_DIR',
    'DBUS_SESSION_BUS_ADDRESS'
  ]) {
    if (process.env[key] !== undefined) launchEnv[key] = process.env[key]
  }
  return {
    command: resolve(binaryPath),
    args: [`--user-data-dir=${userDataDir}`, `--remote-debugging-port=${debugPort}`],
    cwd: dirname(resolve(binaryPath)),
    env: {
      ...launchEnv,
      HERMES_HOME: hermesHome,
      HERMES_DESKTOP_APP_NAME: `Hermes Lunar City Perf ${safeRunId}`,
      HERMES_DESKTOP_CDP_PORT: String(debugPort),
      HERMES_LUNAR_CITY_PERF_ACCEPTANCE: '1',
      HERMES_LUNAR_CITY_PERF_NONCE: launchNonce
    },
    ...(fixtureBinding ? { fixture: fixtureBinding } : {}),
    paths: { hermesHome, userDataDir, tempRoot }
  }
}

function defaultLaunch(plan) {
  for (const path of [plan.paths.hermesHome, plan.paths.userDataDir]) mkdirSync(path, { recursive: true })
  return spawn(plan.command, plan.args, { cwd: plan.cwd, env: plan.env, stdio: 'inherit' })
}

export function validateBridgeHandshake(handshake, expected) {
  if (!handshake || typeof handshake !== 'object')
    throw new Error('bridge_unavailable: versioned packaged bridge absent')
  if (handshake.bridgeVersion !== BRIDGE_VERSION) throw new Error('bridge_mismatch: bridge version')
  if (typeof handshake.launchNonce !== 'string' || handshake.launchNonce !== expected.launchNonce)
    throw new Error('bridge_mismatch: launch nonce')
  if (handshake.buildSha !== expected.buildSha) throw new Error('bridge_mismatch: build SHA')
  if (handshake.packaged !== true) throw new Error('bridge_mismatch: packaged status')
  if (handshake.mainPid !== expected.mainPid) throw new Error('bridge_mismatch: main PID')
  if (!handshake.rendererIdentity || !Number.isInteger(handshake.rendererIdentity.pid))
    throw new Error('bridge_mismatch: renderer lifetime')
  if (
    !Array.isArray(handshake.supportedPhases) ||
    !['baseline-shell', 'mounted-city'].every(phase => handshake.supportedPhases.includes(phase))
  )
    throw new Error('bridge_mismatch: phase support')
  if (handshake.processMetricsSource !== 'electron.app.getAppMetrics')
    throw new Error('bridge_mismatch: process metrics source')
  return handshake
}

async function defaultConnectCdp({ port, expectedHandshake }) {
  const deadline = Date.now() + 60_000
  let lastMismatch = null
  while (Date.now() < deadline) {
    try {
      const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()
      for (const target of targets.filter(item => item.type === 'page' && item.webSocketDebuggerUrl)) {
        const cdp = await CDP.open(target.webSocketDebuggerUrl)
        try {
          const handshake = await cdp.eval(`(() => {
            const bridge = window.__LUNAR_CITY_PERF__
            return bridge && typeof bridge.handshake === 'function'
              ? bridge.handshake(${JSON.stringify({ bridgeVersion: BRIDGE_VERSION, launchNonce: expectedHandshake.launchNonce })})
              : null
          })()`)
          validateBridgeHandshake(handshake, expectedHandshake)
          return { cdp, handshake }
        } catch (error) {
          if (error instanceof Error && error.message.startsWith('bridge_mismatch:')) lastMismatch = error
          cdp.close()
        }
      }
    } catch {
      // No matching launched target yet.
    }
    await new Promise(resolveSleep => setTimeout(resolveSleep, 250))
  }
  if (lastMismatch) throw lastMismatch
  throw new Error('bridge_unavailable: no CDP target matched launch nonce, build SHA, and main PID')
}

async function defaultPreparePhase(cdp, phase) {
  const method = phase === 'baseline-shell' ? 'prepareBaselineShell' : 'mountCity'
  const result = await cdp.eval(`(() => {
    const probe = window.__LUNAR_CITY_PERF__
    if (!probe || typeof probe.${method} !== 'function') return { ok: false, reason: 'probe-unavailable' }
    return Promise.resolve(probe.${method}()).then(() => ({ ok: true }))
  })()`)
  if (!result?.ok) throw new Error(`required packaged Lunar City performance probe is unavailable for ${phase}`)
}

async function defaultRendererProbe(cdp) {
  const result = await cdp.eval(`(() => {
    const probe = window.__LUNAR_CITY_PERF__
    return probe && typeof probe.snapshot === 'function' ? probe.snapshot() : null
  })()`)
  if (!result) throw new Error('required packaged Lunar City renderer metrics are unavailable')
  return result
}

async function defaultProcessProbe(cdp) {
  const result = await cdp.eval(`(() => {
    const probe = window.__LUNAR_CITY_PERF__
    return probe && typeof probe.processMetrics === 'function' ? probe.processMetrics() : null
  })()`)
  if (!result) throw new Error('required packaged app.getAppMetrics process rows are unavailable')
  return result
}

function scenarioActionPlan(scenario, metrics) {
  const workerEntityKey = metrics?.scenarioTargets?.workerEntityKey
  const leaderId = metrics?.scenarioTargets?.leaderId
  const plans = {
    'balanced-worker-focus': [['focus', { entityKey: workerEntityKey }]],
    'continuous-orbit-zoom': [
      ['orbit', { deltaAlpha: 0.5, deltaBeta: 0.1 }],
      ['zoom', { delta: -2 }]
    ],
    'context-loss-recovery': [['context-loss-restore', {}]],
    'dialogue-camera': [
      ['orbit', { deltaAlpha: 0.5, deltaBeta: 0.1 }],
      ['zoom', { delta: -2 }],
      ['leader-dialogue', { leaderId }]
    ],
    disposal: [['dispose', {}]],
    'route-unmounted': [['dispose', {}]],
    hidden: [['window-hidden', {}]],
    minimized: [['window-minimized', {}]],
    'indoor-occlusion': [['interior', {}]],
    'tier-detailed': [['quality', { tier: 'detailed' }]],
    'tier-efficient': [
      ['quality', { tier: 'balanced' }],
      ['quality', { tier: 'efficient' }]
    ],
    'visible-idle': [['window-visible-cycle', {}]],
    '25-active': [['window-visible-cycle', {}]],
    '100-active': [['window-visible-cycle', {}]],
    '250-lod': [['window-visible-cycle', {}]],
    'balanced-overview': [['window-visible-cycle', {}]],
    'tier-balanced': [['quality', { tier: 'balanced' }]],
    '30-minute-stability': [['window-visible-cycle', {}]]
  }

  const plan = plans[scenario]
  if (!plan?.length) throw new Error(`Unsupported Lunar City scenario action plan: ${scenario}`)
  if (plan.some(([, payload]) => Object.values(payload).some(value => value === undefined))) {
    throw new Error(`Lunar City scenario ${scenario} is missing an exact action target`)
  }

  return plan
}

function sameJson(left, right) {
  return JSON.stringify(left) === JSON.stringify(right)
}

function parseCanonicalRequestId(binding, handshake, metrics) {
  const identity = binding?.identity
  if (
    !Number.isInteger(identity?.senderId) ||
    identity.senderId <= 0 ||
    !Number.isInteger(identity.frameId) ||
    identity.frameId < 0
  ) {
    return undefined
  }
  const prefix = [
    'lcperf-v1',
    handshake?.buildSha,
    handshake?.launchNonce,
    handshake?.mainPid,
    identity.senderId,
    identity.frameId,
    metrics?.rendererPid,
    metrics?.rendererGeneration,
    metrics?.rendererStartedAtMs
  ].join(':')
  if (typeof binding.requestId !== 'string' || !binding.requestId.startsWith(`${prefix}:`)) return undefined
  const sequence = Number(binding.requestId.slice(prefix.length + 1))
  return Number.isSafeInteger(sequence) && sequence > 0 && binding.requestId === `${prefix}:${sequence}`
    ? { frameId: identity.frameId, senderId: identity.senderId, sequence }
    : undefined
}

export async function runScenarioThroughBridge(
  cdp,
  scenario,
  metrics,
  handshake,
  snapshotProbe = defaultRendererProbe
) {
  const actions = []
  const requestIds = new Set()
  let authority
  let lastSequence = 0

  const run = async (action, payload, current) => {
    const result = await cdp.eval(`(() => {
      const probe = window.__LUNAR_CITY_PERF__
      if (!probe || typeof probe.runAction !== 'function') {
        return Promise.reject(new Error('probe-unavailable'))
      }
      return probe.runAction(${JSON.stringify(action)}, ${JSON.stringify(payload)})
    })()`)

    const binding = result?.bridgeBinding
    const identity = binding?.identity
    const canonical = parseCanonicalRequestId(binding, handshake, current)
    if (
      !result ||
      result.action !== action ||
      !Number.isInteger(result.proof) ||
      result.proof <= 0 ||
      binding?.action !== 'scenario-action' ||
      typeof binding.requestId !== 'string' ||
      requestIds.has(binding.requestId) ||
      !sameJson(binding.payload, { action, payload }) ||
      identity?.bridgeVersion !== handshake?.bridgeVersion ||
      identity?.buildSha !== handshake?.buildSha ||
      identity?.launchNonce !== handshake?.launchNonce ||
      identity?.mainPid !== handshake?.mainPid ||
      identity?.rendererPid !== current?.rendererPid ||
      identity?.rendererStartedAtMs !== current?.rendererStartedAtMs ||
      identity?.rendererGeneration !== current?.rendererGeneration ||
      !canonical ||
      canonical.sequence <= lastSequence ||
      (authority && (canonical.senderId !== authority.senderId || canonical.frameId !== authority.frameId))
    ) {
      throw new Error(`Lunar City scenario action ${action} did not return causal proof`)
    }

    authority ??= { frameId: canonical.frameId, senderId: canonical.senderId }
    lastSequence = canonical.sequence
    requestIds.add(binding.requestId)
    const observed = await snapshotProbe(cdp)

    return { action, observed, result }
  }

  let before = structuredClone(metrics)
  let preparation
  const requiredPrestate =
    scenario === 'tier-balanced' ? 'Efficient' : scenario === 'tier-efficient' ? 'Detailed' : null
  if (requiredPrestate && before.qualityTier !== requiredPrestate) {
    const tier = requiredPrestate.toLowerCase()
    preparation = await run('quality', { tier }, before)
    before = structuredClone(preparation.observed)
  }

  for (const [action, payload] of scenarioActionPlan(scenario, metrics)) {
    const entry = await run(action, payload, before)
    actions.push(entry)
    before = structuredClone(entry.observed)
  }

  const authoritativeBefore = preparation ? structuredClone(preparation.observed) : structuredClone(metrics)
  return {
    actions,
    authority,
    before: authoritativeBefore,
    initial: structuredClone(metrics),
    ...(preparation ? { preparation } : {}),
    scenario
  }
}

const defaultClock = {
  now: () => Date.now(),
  sleep: milliseconds => new Promise(resolveSleep => setTimeout(resolveSleep, milliseconds))
}

export async function reserveUniqueDebugPort() {
  const server = createServer()
  await new Promise((resolveListen, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', resolveListen)
  })
  const address = server.address()
  return {
    port: address.port,
    release: () =>
      new Promise((resolveClose, reject) => server.close(error => (error ? reject(error) : resolveClose())))
  }
}

async function waitForChild(child) {
  if (!child) return
  if (typeof child.waitForExit === 'function') return child.waitForExit()
  if (child.exitCode !== null) return
  await new Promise(resolveExit => {
    const timeout = setTimeout(resolveExit, 2_000)
    child.once('exit', () => {
      clearTimeout(timeout)
      resolveExit()
    })
  })
}

export async function cleanupIsolatedRun({ cdp, child, tempRoot }, injected = {}) {
  const errors = []
  const attempt = async action => {
    try {
      await action()
    } catch (error) {
      errors.push(error)
    }
  }
  await attempt(() => cdp?.close())
  await attempt(() => child?.kill('SIGTERM'))
  await attempt(() => waitForChild(child))
  await attempt(() => child?.kill('SIGKILL'))
  await attempt(() => (injected.removeTemp ?? (path => rmSync(path, { recursive: true, force: true })))(tempRoot))
  if (errors.length) throw new AggregateError(errors, 'cleanup failed')
}

async function capturePhase({
  cdp,
  phase,
  warmupDurationMs,
  sampleCount,
  sampleIntervalMs,
  processProbe,
  rendererProbe,
  clock,
  scenarioExecution,
  terminalAction
}) {
  const firstRenderer = await rendererProbe(cdp)
  const rendererIdentity = {
    generation: firstRenderer.rendererGeneration,
    pid: firstRenderer.rendererPid,
    startedAtMs: firstRenderer.rendererStartedAtMs
  }
  const startedAt = clock.now()
  const samples = []
  for (let index = 0; index < sampleCount; index += 1) {
    const rendererMetrics = index === 0 ? firstRenderer : await rendererProbe(cdp)
    const processMetrics = await processProbe(cdp)
    samples.push({ timestampMs: clock.now() - startedAt, processMetrics, rendererMetrics })
    if (index + 1 < sampleCount) await clock.sleep(sampleIntervalMs)
  }

  let execution = scenarioExecution

  if (terminalAction) {
    execution = await terminalAction(firstRenderer)
    const elapsed = clock.now() - startedAt
    const lastTimestamp = samples.at(-1)?.timestampMs ?? -1

    if (elapsed <= lastTimestamp) {
      await clock.sleep(lastTimestamp - elapsed + 1)
    }

    samples.push({
      timestampMs: clock.now() - startedAt,
      processMetrics: await processProbe(cdp),
      rendererMetrics: await rendererProbe(cdp)
    })
  }

  return {
    envelopeVersion: LUNAR_CITY_PHASE_ENVELOPE_VERSION,
    phase,
    warmupDurationMs,
    rendererIdentity,
    samples,
    ...(execution ? { scenarioExecution: execution } : {})
  }
}

/**
 * Launch and sample a packaged app through injectable boundaries. This returns
 * raw evidence only; the receipt validator remains the acceptance authority.
 */
export async function runPackagedLunarCityMeasurement(options, injected = {}) {
  const deps = {
    inspectTarget: inspectPackagedTarget,
    makeTempRoot: () => mkdtempSync(join(tmpdir(), 'hermes-lunar-city-perf-')),
    makeLaunchNonce: randomUUID,
    reserveDebugPort: reserveUniqueDebugPort,
    launch: defaultLaunch,
    connectCdp: defaultConnectCdp,
    preparePhase: defaultPreparePhase,
    processProbe: defaultProcessProbe,
    rendererProbe: defaultRendererProbe,
    runScenario: runScenarioThroughBridge,
    clock: defaultClock,
    cleanup: cleanupIsolatedRun,
    captureHostEnvironment: defaultHostEnvironment,
    ...injected
  }
  const target = deps.inspectTarget({ binaryPath: options.binaryPath, expectedGitSha: options.expectedGitSha })
  const tempRoot = deps.makeTempRoot()
  const runId = options.runId ?? `${process.pid}-${deps.clock.now()}`
  const launchNonce = deps.makeLaunchNonce()
  const reservation = await deps.reserveDebugPort()
  const plan = createIsolatedLaunchPlan({
    binaryPath: target.binaryPath,
    debugPort: reservation.port,
    tempRoot,
    runId,
    launchNonce,
    fixture: options.fixture
  })
  let child
  let cdp
  try {
    await reservation.release()
    child = await deps.launch(plan)
    const connected = await deps.connectCdp({
      port: reservation.port,
      child,
      expectedHandshake: { launchNonce, buildSha: target.buildStamp.commit, mainPid: child.pid }
    })
    validateBridgeHandshake(connected.handshake, {
      launchNonce,
      buildSha: target.buildStamp.commit,
      mainPid: child.pid
    })
    cdp = connected.cdp
    const sampleCount = options.sampleCount ?? 4
    const sampleIntervalMs = options.sampleIntervalMs ?? 10_000
    const warmupDurationMs = options.warmupDurationMs ?? 30_000
    await deps.preparePhase(cdp, 'baseline-shell')
    if (warmupDurationMs > 0) await deps.clock.sleep(warmupDurationMs)
    const baselineShell = await capturePhase({
      cdp,
      phase: 'baseline-shell',
      warmupDurationMs,
      sampleCount,
      sampleIntervalMs,
      processProbe: deps.processProbe,
      rendererProbe: deps.rendererProbe,
      clock: deps.clock
    })
    await deps.preparePhase(cdp, 'mounted-city')
    if (warmupDurationMs > 0) await deps.clock.sleep(warmupDurationMs)
    let scenarioExecution

    if (options.scenario && options.scenario !== 'disposal') {
      scenarioExecution = await deps.runScenario(
        cdp,
        options.scenario,
        await deps.rendererProbe(cdp),
        connected.handshake,
        deps.rendererProbe
      )
    }

    const mountedCity = await capturePhase({
      cdp,
      phase: 'mounted-city',
      warmupDurationMs,
      sampleCount,
      sampleIntervalMs,
      processProbe: deps.processProbe,
      rendererProbe: deps.rendererProbe,
      clock: deps.clock,
      scenarioExecution,
      terminalAction:
        options.scenario === 'disposal'
          ? firstRenderer =>
              deps.runScenario(cdp, options.scenario, firstRenderer, connected.handshake, deps.rendererProbe)
          : undefined
    })
    const rawProvenance = {
      provenanceVersion: LUNAR_CITY_PROVENANCE_VERSION,
      baselineShell,
      bridgeHandshake: connected.handshake,
      mountedCity
    }
    const derived = deriveRawSamplesFromProvenance(rawProvenance, { scenario: options.scenario })
    if (
      derived.rendererIdentity.pid !== connected.handshake.rendererIdentity.pid ||
      derived.rendererIdentity.startedAtMs !== connected.handshake.rendererIdentity.startedAtMs
    ) {
      throw new Error('bridge_mismatch: sampled renderer lifetime differs from handshake')
    }
    return {
      buildStamp: target.buildStamp,
      hostEnvironment: deps.captureHostEnvironment(),
      package: { binaryPath: target.binaryPath },
      ...(plan.fixture ? { fixture: plan.fixture } : {}),
      bridgeHandshake: connected.handshake,
      rawProvenance,
      ...derived,
      evidenceClass: Object.keys(injected).length ? 'deterministic' : 'raw-packaged-capture',
      packagedPerformanceEligible: false
    }
  } finally {
    await deps.cleanup({ child, cdp, tempRoot })
  }
}

function parseArgs(argv) {
  const result = {}
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index]
    if (value === '--binary') result.binaryPath = argv[++index]
    else if (value === '--sha') result.expectedGitSha = argv[++index]
    else if (value === '--scenario') result.scenario = argv[++index]
    else throw new Error(`unknown Lunar City performance argument: ${value}`)
  }
  return result
}

async function main() {
  if (process.argv[2] === '--help') {
    process.stdout.write(
      'usage: node lunar-city-runner.mjs --binary <packaged Hermes> --sha <exact SHA> --scenario <name>\nRaw packaged capture only; does not confer acceptance. No dev builds, inherited credentials or disabled GPU.\n'
    )
    return
  }
  const result = await runPackagedLunarCityMeasurement(parseArgs(process.argv.slice(2)))
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`)
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => {
    process.stderr.write(`[perf:lunar-city] REFUSED: ${error instanceof Error ? error.message : String(error)}\n`)
    process.exitCode = 1
  })
}
