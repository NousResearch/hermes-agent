#!/usr/bin/env node

import { createHash, randomUUID } from 'node:crypto'
import { lstatSync, mkdirSync, readFileSync, realpathSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { basename, join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { runPackagedLunarCityMeasurement } from './lunar-city-runner.mjs'
import { RECEIPT_VERSION, SCENARIOS, SCENARIO_PROFILES, summarizeRawSamples, validateReceipt } from './lunar-city.mjs'

const EXACT_SHA = /^[a-f0-9]{40}$/iu
const FIXTURE_VERSION = 'lunar-city-population-v3'
const EXACT_SCENARIOS = new Set(
  Object.entries(SCENARIO_PROFILES)
    .filter(([, profile]) => profile.populationMode === 'exact')
    .map(([scenario]) => scenario)
)
const REQUIRED_HOST_ENVIRONMENT = Object.freeze(['architecture', 'hardwareModel', 'os', 'powerState'])

export function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`
  if (value && typeof value === 'object') {
    return `{${Object.entries(value)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, child]) => `${JSON.stringify(key)}:${canonicalJson(child)}`)
      .join(',')}}`
  }
  return JSON.stringify(value)
}

function sha256(value) {
  return createHash('sha256').update(value).digest('hex')
}

function rawArtifactBytes(value) {
  return `${JSON.stringify(value, null, 2)}\n`
}

function finitePositive(value, label) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    throw new Error(`${label} must be a positive finite number`)
  }
  return value
}

function validateFixture(fixture, scenario) {
  if (!fixture) {
    if (EXACT_SCENARIOS.has(scenario)) throw new Error(`${scenario} requires canonical v3 fixture evidence`)
    return undefined
  }
  if (
    fixture.version !== FIXTURE_VERSION ||
    fixture.evidenceClass !== 'fake-backend-packaged' ||
    ![25, 100, 250].includes(fixture.expectedPopulation)
  ) {
    throw new Error('fixture connection contract is malformed')
  }
  const root = resolve(fixture.root ?? '')
  if (root === resolve('/') || root.length < 2) throw new Error('fixture root must be isolated')
  for (const field of ['hermesHome', 'contractPath', 'userDataDir']) {
    const path = resolve(fixture[field] ?? '')
    if (path !== root && !path.startsWith(`${root}/`)) throw new Error(`fixture ${field} escapes its isolated root`)
  }
  return Object.freeze({ ...fixture, root })
}

function validateFixtureLocation(fixture) {
  const root = resolve(fixture?.root ?? '')
  const home = realpathSync(homedir())
  const workspace = realpathSync(resolve(process.cwd()))
  if (
    root === resolve('/') ||
    root === home ||
    root.startsWith(`${home}/`) ||
    workspace === root ||
    workspace.startsWith(`${root}/`) ||
    root.startsWith(`${workspace}/`) ||
    root.split('/').filter(Boolean).length < 3
  ) {
    throw new Error('fixture root is a home, workspace, broad ancestor, or otherwise not isolated')
  }
  const canonicalRoot = realpathSync(root)
  if (canonicalRoot !== root || lstatSync(root).isSymbolicLink())
    throw new Error('fixture root is symlinked, not canonical')
  const uid = typeof process.getuid === 'function' ? process.getuid() : undefined
  if (uid !== undefined && statSync(canonicalRoot).uid !== uid)
    throw new Error('fixture root is not owned by current uid')
  const checked = { ...fixture, root: canonicalRoot }
  for (const field of ['hermesHome', 'contractPath', 'userDataDir']) {
    const requested = resolve(fixture[field])
    const canonical = realpathSync(requested)
    if (
      requested !== canonical ||
      lstatSync(requested).isSymbolicLink() ||
      (canonical !== canonicalRoot && relative(canonicalRoot, canonical).startsWith('..'))
    ) {
      throw new Error(`fixture ${field} is symlinked or escapes canonical root`)
    }
    if (uid !== undefined && statSync(canonical).uid !== uid)
      throw new Error(`fixture ${field} is not owned by current uid`)
    checked[field] = canonical
  }
  return checked
}

export function validateIsolatedFixturePaths(fixture) {
  const checked = validateFixtureLocation(fixture)
  const uid = typeof process.getuid === 'function' ? process.getuid() : undefined
  const sentinelPath = join(checked.root, '.lunar-city-fixture-owner.json')
  try {
    lstatSync(sentinelPath)
  } catch {
    throw new Error('fixture ownership sentinel is missing')
  }
  if (lstatSync(sentinelPath).isSymbolicLink()) throw new Error('fixture ownership sentinel is symlinked')
  const sentinel = JSON.parse(readFileSync(sentinelPath, 'utf8'))
  if (sentinel.version !== 1 || sentinel.pid !== process.pid || sentinel.nonce !== fixture.runNonce) {
    throw new Error('fixture ownership sentinel does not match this run')
  }
  if (uid !== undefined && statSync(sentinelPath).uid !== uid)
    throw new Error('fixture sentinel is not owned by current uid')
  return Object.freeze(checked)
}

function currentUid() {
  if (typeof process.getuid !== 'function') throw new Error('gateway ownership requires current uid support')
  return process.getuid()
}

/** Own and observe three gateway children through a narrow injectable process boundary. */
export async function createOwnedGatewayLifecycle({ runNonce, sourceIds }, deps) {
  if (typeof runNonce !== 'string' || runNonce.length === 0) throw new Error('gateway lifecycle run nonce is required')
  if (!Array.isArray(sourceIds) || sourceIds.length !== 3 || new Set(sourceIds).size !== 3)
    throw new Error('gateway lifecycle requires exactly three distinct sources')
  for (const name of ['inspectProcess', 'probePopulation', 'spawnGateway', 'terminateGateway', 'waitGateway']) {
    if (typeof deps?.[name] !== 'function') throw new Error(`gateway lifecycle ${name} boundary is unavailable`)
  }
  const uid = currentUid()
  const spawnedRecords = []
  const acceptedRecords = []
  const identities = []
  const throwAfterCleanup = async primaryError => {
    const terminateResults = await Promise.allSettled(
      spawnedRecords.map(record => deps.terminateGateway(record.handle))
    )
    const waitResults = await Promise.allSettled(spawnedRecords.map(record => deps.waitGateway(record.handle)))
    const inspectResults = await Promise.allSettled(spawnedRecords.map(record => deps.inspectProcess(record.handle)))
    const teardownFailures = []
    for (const [index, record] of spawnedRecords.entries()) {
      const terminateResult = terminateResults[index]
      const waitResult = waitResults[index]
      const inspectResult = inspectResults[index]
      if (terminateResult.status === 'rejected') teardownFailures.push(terminateResult.reason)
      if (waitResult.status === 'rejected') teardownFailures.push(waitResult.reason)
      else if (waitResult.value?.exited !== true)
        teardownFailures.push(new Error(`gateway child process ${record.pid} cleanup wait was unverified`))
      if (inspectResult.status === 'rejected') teardownFailures.push(inspectResult.reason)
      else if (inspectResult.value?.alive === true)
        teardownFailures.push(new Error(`gateway child process ${record.pid} remained alive after cleanup`))
    }
    if (teardownFailures.length > 0) {
      const detail = teardownFailures.map(error => (error instanceof Error ? error.message : String(error))).join('; ')
      throw new AggregateError(
        [primaryError, ...teardownFailures],
        `${primaryError instanceof Error ? primaryError.message : String(primaryError)}; teardown failures: ${detail}`,
        { cause: primaryError }
      )
    }
    throw primaryError
  }
  try {
    for (const sourceId of sourceIds) {
      const handle = await deps.spawnGateway(sourceId, { runNonce })
      const pid = handle?.pid
      const spawnedRecord = { handle, pid, sourceId }
      spawnedRecords.push(spawnedRecord)
      if (!Number.isInteger(pid) || pid <= 0) throw new Error(`gateway ${sourceId} has no positive child PID`)
      if (acceptedRecords.some(existing => existing.pid === pid)) throw new Error('gateway child PID is duplicated')
      const observed = await deps.inspectProcess(handle)
      if (!observed) throw new Error(`gateway child process ${pid} is missing`)
      if (observed.alive !== true) throw new Error(`gateway child process ${pid} is not alive`)
      if (observed.parentPid !== process.pid) throw new Error(`gateway process ${pid} is not an orchestrator child`)
      if (observed.uid !== uid) throw new Error(`gateway process ${pid} does not match current uid`)
      if (typeof observed.startToken !== 'string' || observed.startToken.length === 0)
        throw new Error(`gateway process ${pid} start token is unavailable`)
      acceptedRecords.push(spawnedRecord)
      identities.push({ pid, sourceId, parentPid: observed.parentPid, startToken: observed.startToken, uid })
    }
  } catch (error) {
    await throwAfterCleanup(error)
  }

  const inspectIdentity = async (identity, index) => {
    const observed = await deps.inspectProcess(acceptedRecords[index].handle)
    return Boolean(
      observed?.alive === true &&
      observed.parentPid === identity.parentPid &&
      observed.uid === identity.uid &&
      observed.startToken === identity.startToken
    )
  }
  const assertLive = async () => {
    const statuses = await Promise.all(identities.map(inspectIdentity))
    if (statuses.some(alive => !alive)) throw new Error('owned gateway child identity/liveness changed before capture')
  }
  let observation
  try {
    await assertLive()
    // Population probes receive the exact original handles in source order. Only
    // the container is immutable; handle prototypes/private state are untouched.
    observation = await deps.probePopulation(Object.freeze(acceptedRecords.map(record => record.handle)), { runNonce })
    if (
      observation?.authenticated !== true ||
      !Number.isInteger(observation.observedPopulation) ||
      !observation.sourceMix ||
      !Array.isArray(observation.entityKeys)
    ) {
      throw new Error('owned gateways did not return authenticated population observation')
    }
  } catch (error) {
    await throwAfterCleanup(error)
  }
  let stopPromise
  return {
    assertLive,
    async stop() {
      stopPromise ??= (async () => {
        const livenessResults = await Promise.allSettled(identities.map(inspectIdentity))
        const terminateResults = await Promise.allSettled(
          acceptedRecords.map(record => deps.terminateGateway(record.handle))
        )
        const waitResults = await Promise.allSettled(acceptedRecords.map(record => deps.waitGateway(record.handle)))
        const inspections = await Promise.allSettled(acceptedRecords.map(record => deps.inspectProcess(record.handle)))
        const terminations = acceptedRecords.map((record, index) => {
          const terminateResult = terminateResults[index]
          const waitResult = waitResults[index]
          const inspection = inspections[index]
          const waited = waitResult.status === 'fulfilled' ? waitResult.value : undefined
          const after = inspection.status === 'fulfilled' ? inspection.value : undefined
          if (
            terminateResult.status !== 'fulfilled' ||
            waitResult.status !== 'fulfilled' ||
            inspection.status !== 'fulfilled' ||
            waited?.exited !== true ||
            after?.alive === true
          ) {
            throw new Error(`gateway child process ${record.pid} did not terminate with verified wait evidence`)
          }
          return {
            exitCode: waited.exitCode ?? null,
            signal: waited.signal ?? null,
            verifiedExited: true,
            waited: true
          }
        })
        if (livenessResults.some(result => result.status !== 'fulfilled' || result.value !== true)) {
          throw new Error('owned gateway child identity/liveness changed before stop')
        }
        return {
          authenticated: true,
          entityKeys: [...observation.entityKeys],
          gatewayProcesses: identities.map((identity, index) => ({
            ...identity,
            aliveAtCapture: true,
            termination: terminations[index]
          })),
          observedPopulation: observation.observedPopulation,
          orchestratorPid: process.pid,
          runNonce,
          source: 'owned-authenticated-gateways-v2',
          sourceMix: structuredClone(observation.sourceMix),
          uid
        }
      })()
      return stopPromise
    }
  }
}

function parseCanonicalFixture(bytes) {
  let contract
  try {
    contract = JSON.parse(bytes)
  } catch {
    throw new Error('canonical fixture bytes are malformed')
  }
  if (contract?.version !== FIXTURE_VERSION || !Array.isArray(contract.entities)) {
    throw new Error(`canonical fixture must use ${FIXTURE_VERSION}`)
  }
  const { digest, ...unsigned } = contract
  const calculated = sha256(canonicalJson(unsigned))
  if (typeof digest !== 'string' || digest !== calculated) throw new Error('canonical fixture digest mismatch')
  if (contract.entities.length !== contract.population) throw new Error('canonical fixture population is inconsistent')
  const sourceMix = Object.fromEntries(
    [...new Set(contract.entities.map(entity => entity.connectionId))]
      .sort()
      .map(connectionId => [
        connectionId,
        contract.entities.filter(entity => entity.connectionId === connectionId).length
      ])
  )
  const subagentKeys = contract.entities
    .filter(entity => entity.kind === 'subagent')
    .map(entity => entity.exactKey)
    .sort()
  const entityKeys = contract.entities.map(entity => entity.exactKey).sort()
  if (
    entityKeys.some(key => typeof key !== 'string' || key.length === 0) ||
    new Set(entityKeys).size !== entityKeys.length
  ) {
    throw new Error('canonical fixture entity keys must be unique nonempty strings')
  }
  if (subagentKeys.length === 0) throw new Error('canonical fixture requires at least one subagent entity')
  const sourceIds = Object.keys(sourceMix)
  if (sourceIds.length !== 3) throw new Error('canonical fixture requires exactly three gateway sources')
  return { contract, digest, entityKeys, sourceIds, sourceMix, subagentKeys }
}

export function validateCanonicalFixture({ bytes, proof }) {
  const { contract, digest, entityKeys, sourceIds, sourceMix, subagentKeys } = parseCanonicalFixture(bytes)
  if (
    proof?.authenticated !== true ||
    proof?.source !== 'owned-authenticated-gateways-v2' ||
    proof.observedPopulation !== contract.population ||
    canonicalJson(proof.sourceMix) !== canonicalJson(sourceMix)
  ) {
    throw new Error('canonical fixture lacks matching authenticated gateway population evidence')
  }
  const processRows = proof.gatewayProcesses
  const pids = Array.isArray(processRows) ? processRows.map(row => row?.pid) : []
  if (
    proof.orchestratorPid !== process.pid ||
    proof.uid !== currentUid() ||
    typeof proof.runNonce !== 'string' ||
    proof.runNonce.length === 0 ||
    !Array.isArray(processRows) ||
    processRows.length !== 3 ||
    new Set(pids).size !== 3 ||
    canonicalJson(processRows.map(row => row?.sourceId).sort()) !== canonicalJson(sourceIds) ||
    processRows.some(
      row =>
        !Number.isInteger(row?.pid) ||
        row.pid <= 0 ||
        row.parentPid !== proof.orchestratorPid ||
        row.uid !== proof.uid ||
        typeof row.startToken !== 'string' ||
        row.startToken.length === 0 ||
        row.aliveAtCapture !== true ||
        row.termination?.verifiedExited !== true ||
        row.termination?.waited !== true ||
        !Object.hasOwn(row.termination, 'exitCode') ||
        !Object.hasOwn(row.termination, 'signal') ||
        (row.termination.exitCode !== null && !Number.isInteger(row.termination.exitCode)) ||
        (row.termination.signal !== null && typeof row.termination.signal !== 'string')
    )
  )
    throw new Error('canonical fixture requires fully observed owned and terminated gateway child processes')
  if (!Array.isArray(proof.entityKeys)) throw new Error('canonical fixture observed entity keys must be an array')
  const observedKeys = [...proof.entityKeys].sort()
  if (
    observedKeys.some(key => typeof key !== 'string' || key.length === 0) ||
    new Set(observedKeys).size !== observedKeys.length ||
    observedKeys.some(key => !entityKeys.includes(key)) ||
    subagentKeys.some(key => !observedKeys.includes(key))
  ) {
    throw new Error('canonical fixture observed entity-key set lacks exact authenticated subagent evidence')
  }
  return Object.freeze({
    bytesSha256: sha256(bytes),
    contractBytes: bytes,
    contractDigest: digest,
    expectedPopulation: contract.population,
    sourceMix,
    subagentKeys,
    proof: structuredClone(proof),
    proofDigest: sha256(canonicalJson(proof))
  })
}

/** Resolve an exact clock policy. Overrides may lengthen a run, never shorten it. */
export function resolveScenarioMeasurement(scenario, overrides = {}) {
  const profile = SCENARIO_PROFILES[scenario]
  if (!profile) throw new Error(`unknown Lunar City scenario: ${String(scenario)}`)
  const minimumDurationMs = profile.durationMs
  const minimumWarmupMs = profile.warmupMs
  const durationMs = overrides.durationMs ?? minimumDurationMs
  const warmupDurationMs = overrides.warmupDurationMs ?? minimumWarmupMs
  if (!Number.isInteger(durationMs) || durationMs < minimumDurationMs) {
    throw new Error(`${scenario} duration cannot shorten below ${minimumDurationMs}ms`)
  }
  if (!Number.isInteger(warmupDurationMs) || warmupDurationMs < minimumWarmupMs) {
    throw new Error(`${scenario} warmup cannot shorten below ${minimumWarmupMs}ms`)
  }
  const requestedCadence = overrides.sampleIntervalMs ?? profile.maxCadenceMs
  const maxCadence = profile.maxCadenceMs
  if (!Number.isInteger(requestedCadence) || requestedCadence <= 0 || requestedCadence > maxCadence) {
    throw new Error(`${scenario} cadence must be an integer from 1 through ${maxCadence}ms`)
  }
  // Use a divisor so the final timestamp lands exactly on the required duration.
  const intervalCount = Math.max(2, Math.ceil(durationMs / requestedCadence))
  if (durationMs % intervalCount !== 0) {
    throw new Error(`${scenario} duration must divide evenly into its sampling cadence`)
  }
  const sampleIntervalMs = durationMs / intervalCount
  const fixture = validateFixture(overrides.fixture, scenario)
  if (fixture && profile.expectedObserved !== undefined && fixture.expectedPopulation !== profile.expectedObserved) {
    throw new Error(`${scenario} requires fixture population ${profile.expectedObserved}`)
  }
  return Object.freeze({
    durationMs,
    fixture,
    sampleCount: intervalCount + 1,
    sampleIntervalMs,
    warmupDurationMs
  })
}

function average(values) {
  return values.reduce((total, value) => total + value, 0) / values.length
}

function baselineMetric(capture, selector) {
  const values = capture.rawProvenance?.baselineShell?.samples?.map(selector)
  if (!Array.isArray(values) || values.length === 0 || values.some(value => !Number.isFinite(value))) {
    throw new Error('raw baseline provenance is incomplete')
  }
  return average(values)
}

function validateHostEnvironment(environment) {
  if (!environment || typeof environment !== 'object') throw new Error('captured host environment is required')
  for (const field of REQUIRED_HOST_ENVIRONMENT) {
    if (typeof environment[field] !== 'string' || environment[field].length === 0) {
      throw new Error(`captured host environment ${field} is required`)
    }
  }
  return environment
}

function runtimeEnvironment(capture, fixtureBinding) {
  const raw = capture.mountedClaims?.environment
  const host = validateHostEnvironment(capture.hostEnvironment)
  for (const field of ['chromiumVersion', 'electronVersion']) {
    if (typeof raw?.[field] !== 'string' || raw[field].length === 0) throw new Error(`runtime ${field} is unavailable`)
  }
  const gpuAdapter = raw?.gpuInfo?.gpuDevice?.find(device => typeof device?.deviceString === 'string')?.deviceString
  if (!gpuAdapter) throw new Error('runtime GPU adapter is unavailable')
  const windowSize = { width: raw?.windowBounds?.width, height: raw?.windowBounds?.height }
  finitePositive(windowSize.width, 'runtime window width')
  finitePositive(windowSize.height, 'runtime window height')
  finitePositive(raw?.displayScaleFactor, 'runtime display scale')
  return {
    ...host,
    backendMode: fixtureBinding ? 'fake-backend' : 'unbound',
    chromiumVersion: raw.chromiumVersion,
    cityPopulated: capture.mountedClaims.population.observed > 0,
    displayScale: raw.displayScaleFactor,
    electronMode: raw.electronMode,
    electronVersion: raw.electronVersion,
    gpuAdapter,
    gpuEnabled: raw.gpuEnabled,
    windowSize
  }
}

/** Assemble raw capture into the receipt schema. Eligibility remains validator-owned. */
export function assembleLunarCityReceipt({ capture, evidenceClass, metadata, scenario, timestamp }) {
  if (evidenceClass === 'supervised-live') {
    throw new Error('supervised-live requires a dedicated live provenance path')
  }
  if (evidenceClass !== 'fake-backend-packaged') throw new Error('orchestration requires fake-backend-packaged')
  if (!EXACT_SHA.test(capture?.buildStamp?.commit ?? '')) throw new Error('capture build stamp is not exact')
  const claims = capture.mountedClaims
  if (!claims || !capture.rawProvenance || !capture.rawSamples) throw new Error('raw packaged capture is incomplete')
  const summaries = summarizeRawSamples(capture.rawSamples)
  const fixtureBinding = capture.rawProvenance.acceptanceBindings?.fixture
  if (EXACT_SCENARIOS.has(scenario) && !fixtureBinding) throw new Error(`${scenario} requires fixture binding`)
  if (fixtureBinding) {
    if (
      fixtureBinding.expectedPopulation !== claims.population.observed ||
      canonicalJson(fixtureBinding.sourceMix) !== canonicalJson(claims.populationSourceMix)
    ) {
      throw new Error('mounted population/source mix does not match canonical fixture binding')
    }
  }
  const environment = runtimeEnvironment(capture, fixtureBinding)
  if (![0, 15, 30].includes(claims.targetFps)) throw new Error('runtime scheduler fps cap is unavailable')
  const environmentDigest = sha256(canonicalJson(environment))
  if (capture.rawProvenance.acceptanceBindings?.environmentDigest !== environmentDigest) {
    throw new Error('runtime environment provenance digest mismatch')
  }
  const rawBytes = rawArtifactBytes(capture)
  const operatorMetadataBytes =
    metadata && typeof metadata === 'object' && !Array.isArray(metadata) ? canonicalJson(metadata) : undefined
  if (operatorMetadataBytes) {
    try {
      if (canonicalJson(JSON.parse(operatorMetadataBytes)) !== operatorMetadataBytes) throw new Error('noncanonical')
    } catch {
      throw new Error('operator metadata must be canonical JSON data')
    }
  }
  const receipt = {
    receiptVersion: RECEIPT_VERSION,
    evidenceClass,
    scenario,
    gitSha: capture.buildStamp.commit,
    buildStamp: capture.buildStamp,
    timestamp,
    ...(operatorMetadataBytes ? { operatorMetadata: structuredClone(metadata), operatorMetadataBytes } : {}),
    artifactProvenance: {
      environmentDigest,
      rawArtifactSha256: sha256(rawBytes),
      ...(operatorMetadataBytes ? { operatorMetadataSha256: sha256(operatorMetadataBytes) } : {}),
      ...(fixtureBinding
        ? {
            fixtureBytesSha256: fixtureBinding.bytesSha256,
            fixtureContractDigest: fixtureBinding.contractDigest,
            fixtureProofDigest: fixtureBinding.proofDigest
          }
        : {})
    },
    rawArtifact: capture,
    environment,
    qualityTier: claims.qualityTier,
    internalRenderScale: claims.internalRenderScale,
    warmupDurationMs: claims.warmupDurationMs,
    population: { ...claims.population, sourceMix: claims.populationSourceMix },
    cameraState: claims.cameraState,
    dialogueState: claims.dialogueState,
    lifecycleState: claims.lifecycleState,
    measurement: {
      durationMs: claims.durationMs,
      sampleIntervalMs: capture.sampleTimestampsMs[1] - capture.sampleTimestampsMs[0],
      sampleTimestampsMs: capture.sampleTimestampsMs
    },
    rawSamples: capture.rawSamples,
    rawProvenance: capture.rawProvenance,
    summaries,
    fpsCap: claims.targetFps,
    renderFrames: summaries.maxRenderFrames,
    processCpuBaselinePp: baselineMetric(capture, sample =>
      sample.processMetrics.reduce((total, row) => total + row.cpu.percentCPUUsage, 0)
    ),
    processCpuDeltaPp: summaries.avgCpuDeltaPp,
    gpuMemoryBaselineMiB: baselineMetric(capture, sample => sample.rendererMetrics.gpuMemoryMiB),
    gpuMemoryDeltaMiB: summaries.avgGpuMemoryDeltaMiB,
    residentMemoryDriftMiB: summaries.residentMemoryDriftMiB,
    drawCalls: summaries.maxDrawCalls,
    visibleTriangles: summaries.maxVisibleTriangles,
    activeAnimations: summaries.maxActiveAnimations,
    entities: summaries.maxEntities,
    textures: summaries.maxTextures,
    listeners: summaries.maxListeners,
    timers: summaries.maxTimers,
    disposal: scenario === 'disposal' ? claims.lifecycleState : 'not-applicable',
    recovery: scenario === 'context-loss-recovery' ? claims.lifecycleState : 'not-applicable',
    ...(claims.interactionUsable === true ? { interactionUsable: true } : {})
  }
  // The validator defines canonical outcome fields. The assembler never writes
  // packagedPerformanceEligible; no producer may self-declare acceptance.
  const first = validateReceipt({ ...receipt, pass: true, errors: [] })
  const canonicalErrors = first.errors.filter(error => !/^(?:pass|errors) contradicts canonical outcome/u.test(error))
  return { ...receipt, pass: canonicalErrors.length === 0, errors: canonicalErrors }
}

function defaultWriteJson(path, value) {
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, { encoding: 'utf8', flag: 'wx' })
}

export async function orchestrateLunarCityAcceptance(options, injected = {}) {
  const deps = {
    capture: runPackagedLunarCityMeasurement,
    mkdir: path => mkdirSync(path, { recursive: true }),
    nowIso: () => new Date().toISOString(),
    makeRunNonce: randomUUID,
    inspectProcess: async () => {
      throw new Error('owned authenticated fixture process inspection is unavailable')
    },
    probePopulation: async () => {
      throw new Error('owned authenticated fixture population probe is unavailable')
    },
    readFixture: path => readFileSync(path, 'utf8'),
    spawnGateway: async () => {
      throw new Error('owned authenticated fixture gateway spawn is unavailable')
    },
    terminateGateway: async () => undefined,
    waitGateway: async () => ({ exited: false }),
    validate: validateReceipt,
    writeJson: defaultWriteJson,
    ...injected
  }
  if (!EXACT_SHA.test(options.expectedGitSha ?? '')) throw new Error('expected git SHA must be exact')
  if (!Array.isArray(options.scenarios) || options.scenarios.length === 0)
    throw new Error('at least one scenario is required')
  if (options.evidenceClass === 'supervised-live') {
    throw new Error(
      'supervised-live orchestration is unavailable: the packaged runner has no supported real-gateway preseed API'
    )
  }
  if (options.evidenceClass !== 'fake-backend-packaged') {
    throw new Error('packaged orchestration requires evidence class fake-backend-packaged')
  }
  deps.mkdir(options.outputDirectory)
  const results = []
  for (const scenario of options.scenarios) {
    const policy = resolveScenarioMeasurement(scenario, {
      ...options.measurementOverrides?.[scenario],
      fixture: options.fixture
    })
    let fixtureLifecycle
    let fixtureStopped = false
    let sentinelPath
    try {
      let fixtureBinding
      let fixture = policy.fixture
      if (policy.fixture) {
        fixture = validateFixtureLocation(policy.fixture)
        const runNonce = deps.makeRunNonce()
        const pendingSentinelPath = join(fixture.root, '.lunar-city-fixture-owner.json')
        writeFileSync(pendingSentinelPath, JSON.stringify({ nonce: runNonce, pid: process.pid, version: 1 }), {
          encoding: 'utf8',
          flag: 'wx',
          mode: 0o600
        })
        sentinelPath = pendingSentinelPath
        fixture = validateIsolatedFixturePaths({ ...fixture, runNonce })
        const fixtureBytes = deps.readFixture(fixture.contractPath)
        const { sourceIds } = parseCanonicalFixture(fixtureBytes)
        fixtureLifecycle = await createOwnedGatewayLifecycle(
          { runNonce, sourceIds },
          {
            inspectProcess: deps.inspectProcess,
            probePopulation: deps.probePopulation,
            spawnGateway: deps.spawnGateway,
            terminateGateway: deps.terminateGateway,
            waitGateway: deps.waitGateway
          }
        )
        fixture.rawBytes = fixtureBytes
      }
      await fixtureLifecycle?.assertLive()
      const captured = await deps.capture({
        binaryPath: options.binaryPath,
        expectedGitSha: options.expectedGitSha,
        fixture,
        sampleCount: policy.sampleCount,
        sampleIntervalMs: policy.sampleIntervalMs,
        scenario,
        warmupDurationMs: policy.warmupDurationMs
      })
      await fixtureLifecycle?.assertLive()
      if (fixtureLifecycle) {
        const proof = await fixtureLifecycle.stop()
        fixtureStopped = true
        fixtureBinding = validateCanonicalFixture({ bytes: fixture.rawBytes, proof })
        if (fixtureBinding.expectedPopulation !== fixture.expectedPopulation)
          throw new Error('fixture descriptor population does not match canonical bytes')
      }
      const environment = runtimeEnvironment({ ...captured, hostEnvironment: captured.hostEnvironment }, fixtureBinding)
      const capture = {
        ...captured,
        rawProvenance: {
          ...captured.rawProvenance,
          acceptanceBindings: {
            environmentDigest: sha256(canonicalJson(environment)),
            ...(fixtureBinding ? { fixture: fixtureBinding } : {})
          }
        }
      }
      const prefix = `${scenario}-${capture.buildStamp.commit}`
      deps.writeJson(join(options.outputDirectory, `${prefix}.raw.json`), capture)
      const receipt = assembleLunarCityReceipt({
        capture,
        evidenceClass: options.evidenceClass,
        metadata: options.metadata,
        scenario,
        timestamp: deps.nowIso()
      })
      deps.writeJson(join(options.outputDirectory, `${prefix}.receipt.json`), receipt)
      const validation = deps.validate(receipt)
      results.push({ receipt, scenario, validation })
      if (!validation.ok) throw new Error(`${scenario} receipt validation failed: ${validation.errors.join('; ')}`)
      if (!validation.packagedPerformanceEligible) throw new Error(`${scenario} receipt is not eligible for acceptance`)
    } finally {
      try {
        if (fixtureLifecycle && !fixtureStopped) await fixtureLifecycle.stop()
      } finally {
        if (sentinelPath) rmSync(sentinelPath, { force: true })
      }
    }
  }
  return results
}

function parseJsonFile(path, label) {
  if (!path) throw new Error(`${label} JSON path is required`)
  return JSON.parse(readFileSync(path, 'utf8'))
}

function parseArgs(argv) {
  const result = { evidenceClass: 'fake-backend-packaged', scenarios: [] }
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index]
    if (value === '--binary') result.binaryPath = argv[++index]
    else if (value === '--sha') result.expectedGitSha = argv[++index]
    else if (value === '--output') result.outputDirectory = argv[++index]
    else if (value === '--metadata') result.metadata = parseJsonFile(argv[++index], 'metadata')
    else if (value === '--fixture') result.fixture = parseJsonFile(argv[++index], 'fixture')
    else if (value === '--scenario') result.scenarios.push(argv[++index])
    else if (value === '--all') result.scenarios = [...SCENARIOS]
    else if (value === '--help') result.help = true
    else throw new Error(`unknown Lunar City acceptance argument: ${value}`)
  }
  return result
}

function usage() {
  return `usage: node ${basename(fileURLToPath(import.meta.url))} --binary <packaged Hermes> --sha <exact SHA> --output <dir> [--metadata <supplemental-json>] [--fixture <v3-json>] (--all | --scenario <name>...)\n\nThis command performs capture -> receipt assembly -> validation for fake-backend-packaged evidence. Runtime gates come only from hash-bound package, host, and owned fixture provenance; operator metadata is supplemental. It refuses missing GPU/metrics, skips, ineligible receipts, shortened scenario clocks, dirty/mismatched packages, and unowned or unverified fixtures. Exact-population scenarios are blocked until an owned three-gateway lifecycle can supply authenticated observed subagent evidence. Supervised-live evidence requires a separate provenance path and is rejected here.\n`
}

async function main() {
  const options = parseArgs(process.argv.slice(2))
  if (options.help) {
    process.stdout.write(usage())
    return
  }
  await orchestrateLunarCityAcceptance(options)
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => {
    process.stderr.write(
      `[perf:lunar-city:accept] REFUSED: ${error instanceof Error ? error.message : String(error)}\n`
    )
    process.exitCode = 1
  })
}
