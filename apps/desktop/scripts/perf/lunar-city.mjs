/**
 * Machine-readable Lunar City performance receipt contract.
 *
 * This module deliberately has no Electron or Babylon dependency. It validates
 * receipts produced by the packaged runner and recomputes outcome metrics from
 * the retained raw samples so a summary cannot be edited to hide a regression.
 */

import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'

import { deriveRawSamplesFromProvenance } from './lib/lunar-city-provenance.mjs'

export const RECEIPT_VERSION = 1

export const EVIDENCE_CLASSES = Object.freeze(['deterministic', 'fake-backend-packaged', 'supervised-live'])

export const THRESHOLDS = Object.freeze({
  dormantCpuDeltaPp: 0.5,
  visibleIdleCpuDeltaPp: 3,
  active100CpuDeltaPp: 12,
  active250CpuDeltaPp: 18,
  activeFrameP95Ms: 33.3,
  activeWorldUpdateP95Ms: 6,
  maxGpuMemoryDeltaMiB: 256,
  balancedOverviewDrawCalls: 180,
  balancedOverviewTriangles: 1_500_000,
  balancedWorkerFocusDrawCalls: 220,
  balancedWorkerFocusTriangles: 2_000_000,
  stabilityResidentDriftMiB: 75,
  maxRawSamples: 10_000,
  maxRawValue: 1_000_000_000_000
})

const RAW_FIELDS = Object.freeze([
  'frameMs',
  'worldUpdateMs',
  'cpuDeltaPp',
  'gpuMemoryDeltaMiB',
  'residentMemoryMiB',
  'renderFrames',
  'drawCalls',
  'visibleTriangles',
  'activeAnimations',
  'entities',
  'textures',
  'listeners',
  'timers'
])

const SIGNED_RAW_FIELDS = new Set(['cpuDeltaPp', 'gpuMemoryDeltaMiB'])
const NON_NEGATIVE_RAW_FIELDS = new Set(RAW_FIELDS.filter(field => !SIGNED_RAW_FIELDS.has(field)))
const ALLOWED_LOD_KEYS = new Set(['near', 'mid', 'far'])
const MONOTONIC_FIELDS = Object.freeze([
  ['entities', 'entities'],
  ['textures', 'textures'],
  ['listeners', 'listeners'],
  ['activeAnimations', 'animations'],
  ['timers', 'timers']
])

const SCENARIO_PROFILE_DEFINITIONS = {
  'route-unmounted': {
    dormant: true,
    durationMs: 30_000,
    warmupMs: 30_000,
    cpuLimit: 0.5,
    maxCadenceMs: 15_000,
    minObserved: 0,
    minActive: 0,
    requiredQuality: 'Balanced',
    requiredCamera: 'overview',
    requiredDialogue: 'idle',
    populationMode: 'preserved',
    lodRequired: true
  },
  hidden: {
    dormant: true,
    durationMs: 30_000,
    warmupMs: 30_000,
    cpuLimit: 0.5,
    maxCadenceMs: 15_000,
    minObserved: 0,
    minActive: 0,
    requiredQuality: 'Balanced',
    requiredCamera: 'overview',
    requiredDialogue: 'idle',
    populationMode: 'preserved',
    lodRequired: true
  },
  minimized: {
    dormant: true,
    durationMs: 30_000,
    warmupMs: 30_000,
    cpuLimit: 0.5,
    maxCadenceMs: 15_000,
    minObserved: 0,
    minActive: 0,
    requiredQuality: 'Balanced',
    requiredCamera: 'overview',
    requiredDialogue: 'idle',
    populationMode: 'preserved',
    lodRequired: true
  },
  'visible-idle': {
    durationMs: 60_000,
    warmupMs: 30_000,
    cpuLimit: 3,
    maxCadenceMs: 15_000,
    minObserved: 1,
    minActive: 0,
    requiredQuality: 'Balanced',
    requiredCamera: 'overview',
    requiredDialogue: 'idle',
    populationMode: 'preserved',
    lodRequired: true
  },
  '25-active': {
    durationMs: 30_000,
    warmupMs: 30_000,
    maxCadenceMs: 10_000,
    expectedObserved: 25,
    expectedActive: 25,
    lodTotal: 25,
    requiredQuality: 'Balanced',
    requiredCamera: 'overview',
    requiredDialogue: 'idle',
    populationMode: 'exact',
    minObserved: 25,
    minActive: 25,
    lodRequired: true
  },
  '100-active': {
    durationMs: 30_000,
    warmupMs: 30_000,
    cpuLimit: 12,
    p95FrameLimit: 33.3,
    p95UpdateLimit: 6,
    expectedObserved: 100,
    expectedActive: 100,
    requiredQuality: 'Balanced',
    requiredCamera: 'overview',
    requiredDialogue: 'idle',
    lodTotal: 100,
    maxCadenceMs: 10_000,
    populationMode: 'exact',
    minObserved: 100,
    minActive: 100,
    lodRequired: true
  },
  '250-lod': {
    durationMs: 30_000,
    warmupMs: 30_000,
    cpuLimit: 18,
    p95FrameLimit: 33.3,
    expectedObserved: 250,
    requiredQuality: 'Balanced',
    requiredCamera: 'overview',
    requiredDialogue: 'idle',
    lodTotal: 250,
    requiredLodKeys: ['near', 'far'],
    requiredPositiveLodKeys: ['near', 'far'],
    maxCadenceMs: 10_000,
    populationMode: 'exact',
    minObserved: 250,
    minActive: 0,
    lodRequired: true
  },
  'balanced-overview': {
    durationMs: 30_000,
    warmupMs: 30_000,
    drawCalls: 180,
    triangles: 1_500_000,
    maxCadenceMs: 10_000,
    minObserved: 1,
    minActive: 0,
    requiredQuality: 'Balanced',
    requiredCamera: 'overview',
    requiredDialogue: 'idle',
    populationMode: 'preserved',
    lodRequired: true
  },
  'balanced-worker-focus': {
    durationMs: 30_000,
    warmupMs: 30_000,
    drawCalls: 220,
    triangles: 2_000_000,
    maxCadenceMs: 10_000,
    minObserved: 1,
    minActive: 0,
    requiredQuality: 'Balanced',
    requiredCamera: 'worker-focus',
    requiredDialogue: 'idle',
    populationMode: 'preserved',
    lodRequired: true
  },
  'continuous-orbit-zoom': {
    durationMs: 30_000,
    warmupMs: 30_000,
    requiredCamera: 'orbit-zoom',
    maxCadenceMs: 10_000,
    minObserved: 1,
    minActive: 0,
    requiredQuality: 'Balanced',
    requiredDialogue: 'idle',
    populationMode: 'preserved',
    lodRequired: true
  },
  'indoor-occlusion': {
    durationMs: 30_000,
    warmupMs: 30_000,
    requiredCamera: 'indoor',
    maxCadenceMs: 10_000,
    minObserved: 1,
    minActive: 0,
    requiredQuality: 'Balanced',
    requiredDialogue: 'idle',
    populationMode: 'preserved',
    lodRequired: true
  },
  'dialogue-camera': {
    durationMs: 30_000,
    warmupMs: 30_000,
    requiredDialogue: 'active',
    maxCadenceMs: 10_000,
    minObserved: 1,
    minActive: 0,
    requiredQuality: 'Balanced',
    requiredCamera: 'orbit-zoom',
    populationMode: 'preserved',
    lodRequired: true
  },
  'tier-efficient': {
    durationMs: 30_000,
    warmupMs: 30_000,
    requiredQuality: 'Efficient',
    maxCadenceMs: 10_000,
    minObserved: 1,
    minActive: 0,
    requiredCamera: 'overview',
    requiredDialogue: 'idle',
    populationMode: 'preserved',
    lodRequired: true
  },
  'tier-balanced': {
    durationMs: 30_000,
    warmupMs: 30_000,
    requiredQuality: 'Balanced',
    maxCadenceMs: 10_000,
    minObserved: 1,
    minActive: 0,
    requiredCamera: 'overview',
    requiredDialogue: 'idle',
    populationMode: 'preserved',
    lodRequired: true
  },
  'tier-detailed': {
    durationMs: 30_000,
    warmupMs: 30_000,
    requiredQuality: 'Detailed',
    maxCadenceMs: 10_000,
    minObserved: 1,
    minActive: 0,
    requiredCamera: 'overview',
    requiredDialogue: 'idle',
    populationMode: 'preserved',
    lodRequired: true
  },
  'context-loss-recovery': {
    durationMs: 30_000,
    warmupMs: 30_000,
    requiredRecovery: 'recovered',
    maxCadenceMs: 10_000,
    minObserved: 1,
    minActive: 0,
    requiredQuality: 'Balanced',
    requiredCamera: 'overview',
    requiredDialogue: 'idle',
    populationMode: 'preserved',
    lodRequired: true
  },
  disposal: {
    durationMs: 30_000,
    warmupMs: 30_000,
    requiredDisposal: 'disposed',
    maxCadenceMs: 10_000,
    minObserved: 1,
    minActive: 0,
    requiredQuality: 'Balanced',
    requiredCamera: 'overview',
    requiredDialogue: 'idle',
    populationMode: 'preserved',
    lodRequired: true
  },
  '30-minute-stability': {
    durationMs: 1_800_000,
    warmupMs: 30_000,
    cpuLimit: 12,
    maxCadenceMs: 600_000,
    stability: true,
    expectedObserved: 100,
    expectedActive: 100,
    requiredQuality: 'Balanced',
    requiredCamera: 'overview',
    requiredDialogue: 'idle',
    lodTotal: 100,
    populationMode: 'exact',
    minObserved: 100,
    minActive: 100,
    lodRequired: true
  }
}

export const SCENARIO_PROFILES = Object.freeze(
  Object.fromEntries(
    Object.entries(SCENARIO_PROFILE_DEFINITIONS).map(([scenario, profile]) => [
      scenario,
      {
        durationMs: profile.durationMs,
        warmupMs: profile.warmupMs,
        maxCadenceMs: profile.maxCadenceMs,
        ...profile
      }
    ])
  )
)

export const SCENARIOS = Object.freeze(Object.keys(SCENARIO_PROFILES))

const isRecord = value => value !== null && typeof value === 'object' && !Array.isArray(value)
const isFiniteNumber = value => typeof value === 'number' && Number.isFinite(value)
const CLOCK_TOLERANCE_MS = 1
const CANONICAL_ISO_UTC = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/

function isCanonicalIsoUtc(value) {
  if (typeof value !== 'string' || !CANONICAL_ISO_UTC.test(value)) return false
  const milliseconds = Date.parse(value)
  return Number.isFinite(milliseconds) && new Date(milliseconds).toISOString() === value
}

function equalNumber(left, right) {
  return (
    Object.is(left, right) || Math.abs(left - right) <= Number.EPSILON * Math.max(1, Math.abs(left), Math.abs(right))
  )
}

function canonicalJson(value) {
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

function percentile(values, p) {
  if (!values.length) return 0

  const sorted = [...values].sort((a, b) => a - b)
  return sorted[Math.max(0, Math.min(sorted.length - 1, Math.ceil(sorted.length * p) - 1))]
}

function max(values) {
  if (!values.length) return 0
  let result = values[0]
  for (const value of values) result = Math.max(result, value)
  return result
}

function min(values) {
  if (!values.length) return 0
  let result = values[0]
  for (const value of values) result = Math.min(result, value)
  return result
}

function average(values) {
  if (!values.length) return 0
  let total = 0
  for (const value of values) total += value
  return total / values.length
}

function boundedValues(values) {
  return Array.isArray(values) && values.length <= THRESHOLDS.maxRawSamples ? values : []
}

/** Recompute all receipt summaries from raw samples using a deterministic rank. */
export function summarizeRawSamples(rawSamples) {
  const raw = isRecord(rawSamples) ? rawSamples : {}
  const resident = boundedValues(raw.residentMemoryMiB)
  const cpu = boundedValues(raw.cpuDeltaPp)
  const gpu = boundedValues(raw.gpuMemoryDeltaMiB)
  const frame = boundedValues(raw.frameMs)
  const update = boundedValues(raw.worldUpdateMs)
  const renderFrames = boundedValues(raw.renderFrames)
  const drawCalls = boundedValues(raw.drawCalls)
  const triangles = boundedValues(raw.visibleTriangles)
  const animations = boundedValues(raw.activeAnimations)
  const entities = boundedValues(raw.entities)
  const textures = boundedValues(raw.textures)
  const listeners = boundedValues(raw.listeners)
  const timers = boundedValues(raw.timers)

  return {
    sampleCount: frame.length,
    p95FrameMs: percentile(frame, 0.95),
    p95WorldUpdateMs: percentile(update, 0.95),
    avgCpuDeltaPp: average(cpu),
    maxCpuDeltaPp: max(cpu),
    avgGpuMemoryDeltaMiB: average(gpu),
    maxAbsGpuMemoryDeltaMiB: max(gpu.map(value => Math.abs(value))),
    minGpuMemoryDeltaMiB: min(gpu),
    maxGpuMemoryDeltaMiB: max(gpu),
    residentMemoryDriftMiB: resident.length ? resident[resident.length - 1] - resident[0] : 0,
    maxRenderFrames: max(renderFrames),
    maxDrawCalls: max(drawCalls),
    maxVisibleTriangles: max(triangles),
    maxActiveAnimations: max(animations),
    maxEntities: max(entities),
    maxTextures: max(textures),
    maxListeners: max(listeners),
    maxTimers: max(timers)
  }
}

function canonicalScenario(scenario) {
  return typeof scenario === 'string' && Object.hasOwn(SCENARIO_PROFILES, scenario) ? scenario : null
}

function addRequired(errors, object, field, label = field) {
  if (!(field in object)) errors.push(`${label} is required`)
}

function validateRawSamples(rawSamples, errors) {
  if (!isRecord(rawSamples)) {
    errors.push('rawSamples is required and must be an object')
    return
  }

  for (const field of RAW_FIELDS) {
    const values = rawSamples[field]

    if (!Array.isArray(values)) {
      errors.push(`rawSamples.${field} is required and must be an array`)
      continue
    }
    if (values.length === 0) errors.push(`rawSamples.${field} must not be empty`)
    if (values.length > THRESHOLDS.maxRawSamples) {
      errors.push(`rawSamples.${field} has too many samples (max ${THRESHOLDS.maxRawSamples})`)
      continue
    }
    for (const value of values) {
      if (!isFiniteNumber(value)) errors.push(`rawSamples.${field} contains nonfinite values`)
      else if (Math.abs(value) > THRESHOLDS.maxRawValue) errors.push(`rawSamples.${field} contains unbounded values`)
      else if (NON_NEGATIVE_RAW_FIELDS.has(field) && value < 0)
        errors.push(`rawSamples.${field} contains negative values`)
    }
  }
}

function validateRawProvenance(receipt, errors) {
  if (!Object.hasOwn(receipt, 'rawProvenance')) return
  let derived
  try {
    derived = deriveRawSamplesFromProvenance(receipt.rawProvenance, { scenario: receipt.scenario })
  } catch (error) {
    errors.push(error instanceof Error ? error.message : `raw provenance validation failed: ${String(error)}`)
    return
  }
  const bridgeIdentity = receipt.rawProvenance?.bridgeHandshake?.rendererIdentity
  if (
    bridgeIdentity !== undefined &&
    (!isRecord(bridgeIdentity) ||
      bridgeIdentity.pid !== derived.rendererIdentity.pid ||
      bridgeIdentity.startedAtMs !== derived.rendererIdentity.startedAtMs)
  ) {
    errors.push('rawProvenance bridge renderer lifetime does not match sampled renderer identity')
  }
  const claims = derived.mountedClaims
  const timestamps = receipt.measurement?.sampleTimestampsMs
  if (!Array.isArray(timestamps) || JSON.stringify(timestamps) !== JSON.stringify(derived.sampleTimestampsMs))
    errors.push('measurement timestamps do not match raw mounted sample timestamps')
  if (receipt.measurement?.durationMs !== claims.durationMs)
    errors.push(`measurement duration does not match raw mounted duration ${claims.durationMs}`)
  if (receipt.warmupDurationMs !== claims.warmupDurationMs)
    errors.push('warmup duration does not match raw mounted warmup')
  const receiptPopulationCore = isRecord(receipt.population)
    ? {
        observed: receipt.population.observed,
        active: receipt.population.active,
        lodMix: receipt.population.lodMix,
        source: receipt.population.source
      }
    : receipt.population
  if (JSON.stringify(receiptPopulationCore) !== JSON.stringify(claims.population))
    errors.push('top-level population does not match raw mounted population')
  const packagedClass = receipt.evidenceClass === 'fake-backend-packaged' || receipt.evidenceClass === 'supervised-live'
  if (packagedClass && JSON.stringify(receipt.population?.sourceMix) !== JSON.stringify(claims.populationSourceMix))
    errors.push('top-level population source does not match raw source distribution')
  for (const field of ['qualityTier', 'internalRenderScale', 'cameraState', 'dialogueState']) {
    if (receipt[field] !== claims[field]) errors.push(`${field} does not match raw mounted state`)
  }
  if (receipt.fpsCap !== claims.targetFps) errors.push('fpsCap does not match raw scheduler target FPS')
  if (receipt.lifecycleState !== claims.lifecycleState)
    errors.push('top-level lifecycleState does not match final raw lifecycle state')
  if (receipt.environment?.gpuEnabled !== claims.environment.gpuEnabled)
    errors.push('environment GPU state does not match raw mounted state')
  if (receipt.environment?.electronMode !== claims.environment.electronMode)
    errors.push('environment Electron mode does not match raw mounted state')
  if (receipt.environment?.cityPopulated !== claims.population.observed > 0)
    errors.push('environment city population state does not match raw mounted population')

  const requireAction = (condition, message) => {
    if (!condition) errors.push(message)
  }
  if (receipt.scenario === 'balanced-overview')
    requireAction(claims.cameraActions.overview > 0, 'balanced overview requires raw overview action evidence')
  if (receipt.scenario === 'balanced-worker-focus')
    requireAction(claims.cameraActions.focus > 0, 'worker focus requires raw focus action evidence')
  if (receipt.scenario === 'continuous-orbit-zoom')
    requireAction(
      claims.cameraActions.orbit > 0 && claims.cameraActions.zoom > 0,
      'continuous orbit zoom requires raw orbit and zoom action evidence'
    )
  if (receipt.scenario === 'indoor-occlusion')
    requireAction(claims.cameraActions.indoor > 0, 'indoor occlusion requires raw indoor action evidence')
  if (receipt.scenario === 'dialogue-camera')
    requireAction(
      claims.dialogueActions.opened > 0 &&
        claims.dialogueActions.messagesSent > 0 &&
        claims.dialogueActions.responsesReceived > 0,
      'dialogue camera requires raw dialogue action evidence'
    )
  if (receipt.scenario === 'context-loss-recovery')
    requireAction(
      claims.lifecycleActions.contextLosses > 0 && claims.lifecycleActions.recoveries > 0,
      'context loss recovery requires raw context recovery action evidence'
    )
  if (receipt.scenario === 'disposal')
    requireAction(claims.lifecycleActions.disposals > 0, 'disposal requires raw disposal action evidence')
  const trace = claims.lifecycleTrace
  const mountKey = entry => `${entry.sceneMount.id}:${entry.sceneMount.generation}:${entry.sceneMount.startedAtMs}`
  const unchangedMount = trace.every(entry => mountKey(entry) === mountKey(trace[0]))
  if (receipt.scenario === 'context-loss-recovery') {
    const contextAction = claims.scenarioExecution?.actions?.find(entry => entry.action === 'context-loss-restore')
    const actionTrace = contextAction?.result?.lifecycleTrace
    const lastLoss = Array.isArray(actionTrace) ? actionTrace.lastIndexOf('contextLost') : -1
    const lastRecovery = Array.isArray(actionTrace) ? actionTrace.lastIndexOf('recovered') : -1
    if (
      claims.lifecycleState !== 'recovered' ||
      receipt.recovery !== 'recovered' ||
      lastLoss < 0 ||
      lastRecovery <= lastLoss ||
      !unchangedMount
    )
      errors.push('context recovery claim requires final recovered lifecycle with no later context loss or remount')
  }
  if (receipt.scenario === 'disposal') {
    if (claims.lifecycleState !== 'disposed' || receipt.disposal !== 'disposed' || !unchangedMount)
      errors.push('disposal claim requires final disposed raw lifecycle state with no remount')
  }
  if (receipt.scenario === '30-minute-stability') {
    if (!unchangedMount) errors.push('stability requires one unchanged scene mount identity and generation')
    if (trace.some(entry => entry.state !== 'mounted'))
      errors.push('stability forbids disposal, remount, context loss, or replacement recovery')
    if (
      trace.some(
        entry => entry.counters.contextLosses !== 0 || entry.counters.recoveries !== 0 || entry.counters.disposals !== 0
      )
    )
      errors.push('stability requires zero lifecycle counters across the continuous mount')
    for (const field of ['entities', 'textures', 'listeners', 'timers']) {
      const values = claims.resourceContinuity[field]
      if (values.some(value => value !== values[0])) errors.push(`stability resource continuity failed for ${field}`)
    }
    const resident = claims.resourceContinuity.residentMemoryMiB
    if (resident.some(value => value < resident[0] * 0.5))
      errors.push('stability resident memory reset indicates scene replacement')
  }
  if (!['context-loss-recovery', 'disposal'].includes(receipt.scenario) && claims.lifecycleState !== 'mounted')
    errors.push(`${receipt.scenario} requires final mounted raw lifecycle state`)
  if (!isRecord(receipt.rawSamples)) return
  for (const field of RAW_FIELDS) {
    const retained = receipt.rawSamples[field]
    const expected = derived.rawSamples[field]
    if (
      !Array.isArray(retained) ||
      retained.length !== expected.length ||
      retained.some((value, index) => !equalNumber(value, expected[index]))
    ) {
      errors.push(`rawSamples.${field} does not match raw provenance`)
    }
  }
}

function validateSummaries(rawSamples, summaries, errors) {
  if (!isRecord(summaries)) {
    errors.push('summaries is required and must be an object')
    return {}
  }

  const expected = summarizeRawSamples(rawSamples)
  for (const [field, value] of Object.entries(summaries)) {
    if (!(field in expected)) {
      errors.push(`summaries.${field} is not a recognized summary field`)
    } else if (!isFiniteNumber(value)) {
      errors.push(`summaries.${field} must be finite`)
    } else if (!equalNumber(value, expected[field])) {
      errors.push(`summary ${field} does not match raw samples (${value} !== ${expected[field]})`)
    }
  }
  for (const [field, value] of Object.entries(expected)) {
    if (!(field in summaries)) errors.push(`summaries.${field} is required`)
    else if (!isFiniteNumber(value)) errors.push(`summaries.${field} must be finite`)
  }

  return summaries
}

function numericMetric(receipt, summaries, directField, summaryField, errors, label) {
  if (!(directField in receipt) && (directField === 'processCpuDeltaPp' || directField === 'gpuMemoryDeltaMiB')) {
    errors.push(`${label} unavailable; missing ${label} is not zero`)
    return undefined
  }
  const value = receipt[directField]
  if (directField in receipt && value === undefined) {
    errors.push(`${label} unavailable; missing ${label} is not zero`)
    return undefined
  }
  if (value !== undefined && !isFiniteNumber(value)) errors.push(`${label} must be finite`)
  if (value !== undefined) return value
  return summaries[summaryField]
}

function validateCommonShape(receipt, errors) {
  if (!isRecord(receipt)) {
    errors.push('receipt must be an object')
    return
  }
  addRequired(errors, receipt, 'receiptVersion')
  if (receipt.receiptVersion !== undefined && receipt.receiptVersion !== RECEIPT_VERSION) {
    errors.push(`receiptVersion must equal ${RECEIPT_VERSION}`)
  }
  addRequired(errors, receipt, 'evidenceClass')
  addRequired(errors, receipt, 'scenario')
  addRequired(errors, receipt, 'gitSha')
  addRequired(errors, receipt, 'buildStamp')
  addRequired(errors, receipt, 'timestamp')
  addRequired(errors, receipt, 'pass')
  addRequired(errors, receipt, 'errors')
  addRequired(errors, receipt, 'environment')
  addRequired(errors, receipt, 'qualityTier')
  addRequired(errors, receipt, 'internalRenderScale')
  addRequired(errors, receipt, 'warmupDurationMs')
  addRequired(errors, receipt, 'population')
  addRequired(errors, receipt, 'cameraState')
  addRequired(errors, receipt, 'dialogueState')
  addRequired(errors, receipt, 'lifecycleState')
  addRequired(errors, receipt, 'rawSamples')
  addRequired(errors, receipt, 'summaries')
  for (const field of [
    'fpsCap',
    'renderFrames',
    'processCpuBaselinePp',
    'processCpuDeltaPp',
    'gpuMemoryBaselineMiB',
    'gpuMemoryDeltaMiB',
    'residentMemoryDriftMiB',
    'drawCalls',
    'visibleTriangles',
    'activeAnimations',
    'entities',
    'textures',
    'listeners',
    'timers',
    'disposal',
    'recovery'
  ]) {
    addRequired(errors, receipt, field)
  }

  if (receipt.gitSha !== undefined && (typeof receipt.gitSha !== 'string' || !/^[0-9a-f]{40}$/i.test(receipt.gitSha))) {
    errors.push('gitSha must be an exact 40-character hexadecimal SHA')
  }
  if (
    receipt.buildStamp !== undefined &&
    !(typeof receipt.buildStamp === 'string' ? receipt.buildStamp.length > 0 : isRecord(receipt.buildStamp))
  ) {
    errors.push('buildStamp must be a nonempty string or stamp object')
  }
  if (isRecord(receipt.buildStamp) && Object.keys(receipt.buildStamp).length === 0) {
    errors.push('buildStamp object must not be empty')
  }
  if (receipt.timestamp !== undefined && !isCanonicalIsoUtc(receipt.timestamp)) {
    errors.push('timestamp must be a canonical ISO-8601 UTC timestamp')
  }
  if (receipt.pass !== undefined && typeof receipt.pass !== 'boolean') errors.push('pass must be boolean')
  if (receipt.errors !== undefined && !Array.isArray(receipt.errors)) errors.push('errors must be an array')
  if (receipt.evidenceClass !== undefined && !EVIDENCE_CLASSES.includes(receipt.evidenceClass)) {
    errors.push(`unknown evidence class: ${String(receipt.evidenceClass)}`)
  }
  if (receipt.scenario !== undefined && canonicalScenario(receipt.scenario) === null)
    errors.push('scenario must be a string')
  if (receipt.qualityTier !== undefined && !['Efficient', 'Balanced', 'Detailed'].includes(receipt.qualityTier)) {
    errors.push('qualityTier must be Efficient, Balanced, or Detailed')
  }
  if (
    receipt.lifecycleState !== undefined &&
    !['mounted', 'contextLost', 'recovered', 'disposed'].includes(receipt.lifecycleState)
  )
    errors.push('lifecycleState must be mounted, contextLost, recovered, or disposed')
  if (
    receipt.internalRenderScale !== undefined &&
    (!isFiniteNumber(receipt.internalRenderScale) ||
      receipt.internalRenderScale <= 0 ||
      receipt.internalRenderScale > 1)
  ) {
    errors.push('internalRenderScale must be finite and in (0, 1]')
  }
  if (
    receipt.warmupDurationMs !== undefined &&
    (!isFiniteNumber(receipt.warmupDurationMs) || receipt.warmupDurationMs < 0)
  ) {
    errors.push('warmupDurationMs must be a nonnegative finite number')
  }
  if (receipt.environment !== undefined && !isRecord(receipt.environment)) errors.push('environment must be an object')
  if (receipt.population !== undefined && !isRecord(receipt.population)) errors.push('population must be an object')
  for (const field of [
    'fpsCap',
    'renderFrames',
    'processCpuBaselinePp',
    'processCpuDeltaPp',
    'gpuMemoryBaselineMiB',
    'gpuMemoryDeltaMiB',
    'residentMemoryDriftMiB',
    'drawCalls',
    'visibleTriangles',
    'activeAnimations',
    'entities',
    'textures',
    'listeners',
    'timers'
  ]) {
    const signed = field === 'processCpuDeltaPp' || field === 'gpuMemoryDeltaMiB' || field === 'residentMemoryDriftMiB'
    if (field in receipt && (!isFiniteNumber(receipt[field]) || (!signed && receipt[field] < 0))) {
      errors.push(`${field} must be a nonnegative finite number`)
    } else if (field in receipt && Math.abs(receipt[field]) > THRESHOLDS.maxRawValue) {
      errors.push(`${field} contains an unbounded value`)
    }
  }
}

function validateEnvironment(environment, errors) {
  if (!isRecord(environment)) return
  for (const field of [
    'hardwareModel',
    'architecture',
    'os',
    'electronVersion',
    'chromiumVersion',
    'powerState',
    'gpuAdapter',
    'electronMode',
    'backendMode'
  ]) {
    addRequired(errors, environment, field, `environment.${field}`)
    if (field in environment && (typeof environment[field] !== 'string' || environment[field].length === 0)) {
      errors.push(`environment.${field} must be a nonempty string`)
    }
  }
  addRequired(errors, environment, 'windowSize', 'environment.windowSize')
  addRequired(errors, environment, 'displayScale', 'environment.displayScale')
  addRequired(errors, environment, 'gpuEnabled', 'environment.gpuEnabled')
  addRequired(errors, environment, 'cityPopulated', 'environment.cityPopulated')
  if (isRecord(environment.windowSize)) {
    for (const field of ['width', 'height']) {
      if (!isFiniteNumber(environment.windowSize[field]) || environment.windowSize[field] <= 0) {
        errors.push(`environment.windowSize.${field} must be positive`)
      }
    }
  }
  if (!isFiniteNumber(environment.displayScale) || environment.displayScale <= 0)
    errors.push('environment.displayScale must be positive')
  if (typeof environment.gpuEnabled !== 'boolean') errors.push('environment.gpuEnabled must be boolean')
  if (typeof environment.cityPopulated !== 'boolean') errors.push('environment.cityPopulated must be boolean')
}

function validateBuildStamp(receipt, errors) {
  const stamp = receipt.buildStamp
  if (!isRecord(stamp)) {
    errors.push('buildStamp must be a strict stamp object')
    return
  }
  for (const field of ['schemaVersion', 'commit', 'branch', 'builtAt', 'dirty', 'source']) {
    addRequired(errors, stamp, field, `buildStamp.${field}`)
  }
  if (stamp.schemaVersion !== 1) errors.push('buildStamp.schemaVersion must equal 1')
  if (typeof stamp.commit !== 'string' || !/^[0-9a-f]{40}$/i.test(stamp.commit))
    errors.push('buildStamp.commit must be an exact 40-character hexadecimal SHA')
  if (typeof receipt.gitSha === 'string' && stamp.commit !== receipt.gitSha)
    errors.push('buildStamp.commit must match gitSha')
  if (stamp.branch !== null && typeof stamp.branch !== 'string')
    errors.push('buildStamp.branch must be a string or null')
  if (!isCanonicalIsoUtc(stamp.builtAt)) errors.push('buildStamp.builtAt must be a canonical ISO-8601 UTC timestamp')
  if (typeof stamp.dirty !== 'boolean') errors.push('buildStamp.dirty must be boolean')
  if (!['ci', 'local', 'fallback'].includes(stamp.source))
    errors.push('buildStamp.source must be ci, local, or fallback')
  if (Object.keys(stamp).sort().join(',') !== 'branch,builtAt,commit,dirty,schemaVersion,source')
    errors.push('buildStamp contains unknown or missing fields')
  if (
    (receipt.evidenceClass === 'fake-backend-packaged' || receipt.evidenceClass === 'supervised-live') &&
    stamp.dirty === true
  ) {
    errors.push('packaged performance receipts cannot use a dirty buildStamp')
  }
  if (
    (receipt.evidenceClass === 'fake-backend-packaged' || receipt.evidenceClass === 'supervised-live') &&
    stamp.source === 'fallback'
  ) {
    errors.push('packaged performance receipts require a pinned buildStamp source')
  }
}

function validateArtifactProvenance(receipt, errors) {
  if (receipt.evidenceClass === 'supervised-live') {
    errors.push('supervised-live requires a dedicated live provenance validator')
    return
  }
  if (receipt.evidenceClass !== 'fake-backend-packaged') return
  const artifact = receipt.artifactProvenance
  const raw = receipt.rawArtifact
  if (!isRecord(artifact) || !isRecord(raw)) {
    errors.push('packaged receipt requires hash-bound rawArtifact provenance')
    return
  }
  if (artifact.rawArtifactSha256 !== sha256(`${JSON.stringify(raw, null, 2)}\n`))
    errors.push('raw artifact digest mismatch')
  const hasOperatorMetadata =
    Object.hasOwn(receipt, 'operatorMetadata') ||
    Object.hasOwn(receipt, 'operatorMetadataBytes') ||
    Object.hasOwn(artifact, 'operatorMetadataSha256')
  if (hasOperatorMetadata) {
    let parsedOperatorMetadata
    try {
      parsedOperatorMetadata = JSON.parse(receipt.operatorMetadataBytes)
    } catch {
      errors.push('operator metadata canonical bytes are malformed')
    }
    if (
      typeof receipt.operatorMetadataBytes !== 'string' ||
      !isRecord(receipt.operatorMetadata) ||
      canonicalJson(parsedOperatorMetadata) !== receipt.operatorMetadataBytes ||
      canonicalJson(parsedOperatorMetadata) !== canonicalJson(receipt.operatorMetadata) ||
      artifact.operatorMetadataSha256 !== sha256(receipt.operatorMetadataBytes ?? '')
    ) {
      errors.push('operator metadata bytes/digest mismatch')
    }
  }
  if (canonicalJson(raw.rawProvenance) !== canonicalJson(receipt.rawProvenance))
    errors.push('raw artifact provenance does not match receipt')
  if (canonicalJson(raw.rawSamples) !== canonicalJson(receipt.rawSamples))
    errors.push('raw artifact samples do not match receipt')
  const environmentDigest = sha256(canonicalJson(receipt.environment))
  if (
    artifact.environmentDigest !== environmentDigest ||
    raw.rawProvenance?.acceptanceBindings?.environmentDigest !== environmentDigest
  )
    errors.push('runtime environment provenance digest mismatch')
  const fixture = raw.rawProvenance?.acceptanceBindings?.fixture
  if (SCENARIO_PROFILES[receipt.scenario]?.populationMode === 'exact' && !isRecord(fixture))
    errors.push(`${receipt.scenario} requires canonical v3 fixture binding`)
  if (isRecord(fixture)) {
    let contract
    try {
      contract = JSON.parse(fixture.contractBytes)
    } catch {
      errors.push('fixture canonical bytes are malformed')
    }
    const unsigned = isRecord(contract)
      ? Object.fromEntries(Object.entries(contract).filter(([key]) => key !== 'digest'))
      : null
    const canonicalDigest = unsigned ? sha256(canonicalJson(unsigned)) : null
    const expectedSourceMix = Array.isArray(contract?.entities)
      ? Object.fromEntries(
          [...new Set(contract.entities.map(entity => entity.connectionId))]
            .sort()
            .map(connectionId => [
              connectionId,
              contract.entities.filter(entity => entity.connectionId === connectionId).length
            ])
        )
      : null
    const subagentKeys = Array.isArray(contract?.entities)
      ? contract.entities.filter(entity => entity.kind === 'subagent').map(entity => entity.exactKey)
      : []
    const contractEntityKeys = Array.isArray(contract?.entities) ? contract.entities.map(entity => entity.exactKey) : []
    const observedEntityKeys = fixture.proof?.entityKeys
    const proofRows = fixture.proof?.gatewayProcesses
    const proofPids = Array.isArray(proofRows) ? proofRows.map(row => row?.pid) : []
    const proofSourceIds = Array.isArray(proofRows) ? proofRows.map(row => row?.sourceId).sort() : []
    const expectedSourceIds = expectedSourceMix ? Object.keys(expectedSourceMix).sort() : []
    if (
      typeof fixture.contractBytes !== 'string' ||
      sha256(fixture.contractBytes) !== fixture.bytesSha256 ||
      contract?.version !== 'lunar-city-population-v3' ||
      contract?.digest !== fixture.contractDigest ||
      canonicalDigest !== fixture.contractDigest ||
      contract?.population !== fixture.expectedPopulation ||
      canonicalJson(expectedSourceMix) !== canonicalJson(fixture.sourceMix) ||
      contractEntityKeys.some(key => typeof key !== 'string' || key.length === 0) ||
      new Set(contractEntityKeys).size !== contractEntityKeys.length ||
      subagentKeys.length === 0
    )
      errors.push('fixture canonical bytes/digest/population/source mix mismatch')
    if (
      fixture.proof?.authenticated !== true ||
      fixture.proof?.source !== 'owned-authenticated-gateways-v2' ||
      sha256(canonicalJson(fixture.proof)) !== fixture.proofDigest ||
      fixture.proof?.observedPopulation !== contract?.population ||
      canonicalJson(fixture.proof?.sourceMix) !== canonicalJson(expectedSourceMix) ||
      !Number.isInteger(fixture.proof?.orchestratorPid) ||
      fixture.proof.orchestratorPid <= 0 ||
      !Number.isInteger(fixture.proof?.uid) ||
      fixture.proof.uid < 0 ||
      typeof fixture.proof?.runNonce !== 'string' ||
      fixture.proof.runNonce.length === 0 ||
      !Array.isArray(proofRows) ||
      proofRows.length !== 3 ||
      new Set(proofPids).size !== 3 ||
      canonicalJson(proofSourceIds) !== canonicalJson(expectedSourceIds) ||
      proofRows.some(
        row =>
          !Number.isInteger(row?.pid) ||
          row.pid <= 0 ||
          row.parentPid !== fixture.proof.orchestratorPid ||
          row.uid !== fixture.proof.uid ||
          typeof row.startToken !== 'string' ||
          row.startToken.length === 0 ||
          row.aliveAtCapture !== true ||
          row.termination?.verifiedExited !== true ||
          row.termination?.waited !== true ||
          !Object.hasOwn(row.termination, 'exitCode') ||
          !Object.hasOwn(row.termination, 'signal') ||
          (row.termination.exitCode !== null && !Number.isInteger(row.termination.exitCode)) ||
          (row.termination.signal !== null && typeof row.termination.signal !== 'string')
      ) ||
      !Array.isArray(observedEntityKeys) ||
      observedEntityKeys.some(key => typeof key !== 'string' || key.length === 0) ||
      new Set(observedEntityKeys).size !== observedEntityKeys.length ||
      observedEntityKeys.some(key => !contractEntityKeys.includes(key)) ||
      subagentKeys.some(key => !observedEntityKeys.includes(key))
    )
      errors.push('fixture authenticated subagent/gateway proof mismatch')
    if (
      artifact.fixtureBytesSha256 !== fixture.bytesSha256 ||
      artifact.fixtureContractDigest !== fixture.contractDigest ||
      artifact.fixtureProofDigest !== fixture.proofDigest
    )
      errors.push('fixture artifact provenance digest mismatch')
    if (
      fixture.expectedPopulation !== receipt.population?.observed ||
      canonicalJson(fixture.sourceMix) !== canonicalJson(receipt.population?.sourceMix)
    )
      errors.push('fixture binding does not match mounted population/source mix')
  }
}

function validatePopulation(population, errors) {
  if (!isRecord(population)) return
  for (const field of ['observed', 'active', 'lodMix', 'source']) {
    addRequired(errors, population, field, `population.${field}`)
  }
  for (const field of ['observed', 'active']) {
    if (field in population && (!Number.isInteger(population[field]) || population[field] < 0)) {
      errors.push(`population.${field} must be a nonnegative integer`)
    }
  }
  if (Number.isInteger(population.observed) && Number.isInteger(population.active)) {
    if (population.active > population.observed) errors.push('population.active must not exceed observed')
  }
  if ('lodMix' in population && !isRecord(population.lodMix)) errors.push('population.lodMix must be an object')
  if ('source' in population && typeof population.source !== 'string') errors.push('population.source must be a string')
  if (isRecord(population.lodMix)) {
    let total = 0
    for (const [lod, count] of Object.entries(population.lodMix)) {
      if (!ALLOWED_LOD_KEYS.has(lod)) errors.push(`population.lodMix.${lod} is not an allowed LOD key`)
      if (!Number.isInteger(count) || count < 0) errors.push(`population.lodMix.${lod} must be a nonnegative integer`)
      else total += count
    }
    if (Number.isInteger(population.observed) && total !== population.observed) {
      errors.push(`population.lodMix total ${total} must equal observed ${population.observed}`)
    }
  }
}

function validateMeasurement(receipt, profile, errors) {
  if (!profile) return
  const measurement = receipt.measurement
  if (!isRecord(measurement)) {
    errors.push('measurement is required and must be an object')
    return
  }
  for (const field of ['durationMs', 'sampleIntervalMs', 'sampleTimestampsMs'])
    addRequired(errors, measurement, field, `measurement.${field}`)
  if (
    !isFiniteNumber(measurement.durationMs) ||
    measurement.durationMs < 0 ||
    measurement.durationMs > THRESHOLDS.maxRawValue
  )
    errors.push('measurement.durationMs must be nonnegative and finite')
  if (
    !isFiniteNumber(measurement.sampleIntervalMs) ||
    measurement.sampleIntervalMs <= 0 ||
    measurement.sampleIntervalMs > THRESHOLDS.maxRawValue
  )
    errors.push('measurement.sampleIntervalMs must be positive and finite')
  const timestamps = measurement.sampleTimestampsMs
  if (!Array.isArray(timestamps)) {
    errors.push('measurement.sampleTimestampsMs must be an array')
    return
  }
  if (timestamps.length > THRESHOLDS.maxRawSamples) {
    errors.push(`measurement.sampleTimestampsMs has too many samples (max ${THRESHOLDS.maxRawSamples})`)
    return
  }
  if (timestamps.length < 3) errors.push('measurement sample coverage is inadequate; at least 3 samples are required')
  if (timestamps.length && timestamps[0] !== 0) errors.push('measurement timestamps must begin at zero')
  let previous = null
  for (const timestamp of timestamps) {
    if (!isFiniteNumber(timestamp) || timestamp < 0 || timestamp > THRESHOLDS.maxRawValue)
      errors.push('measurement timestamps must be nonnegative finite values')
    if (previous !== null && timestamp <= previous) errors.push('measurement timestamps must be strictly increasing')
    previous = timestamp
  }
  const raw = receipt.rawSamples
  if (isRecord(raw) && timestamps.length > 0) {
    for (const field of RAW_FIELDS) {
      if (Array.isArray(raw[field]) && raw[field].length !== timestamps.length) {
        errors.push(`rawSamples.${field} must be aligned with measurement timestamps`)
      }
    }
  }
  if (isFiniteNumber(measurement.durationMs) && timestamps.length) {
    const coverageDelta = timestamps[timestamps.length - 1] - measurement.durationMs
    if (Math.abs(coverageDelta) > CLOCK_TOLERANCE_MS) {
      errors.push('measurement timestamps coverage must end at measured duration')
    }
  }
  if (isFiniteNumber(measurement.sampleIntervalMs)) {
    for (let index = 1; index < timestamps.length; index += 1) {
      const delta = timestamps[index] - timestamps[index - 1]
      if (Math.abs(delta - measurement.sampleIntervalMs) > CLOCK_TOLERANCE_MS) {
        errors.push('measurement timestamps do not match declared cadence')
      }
      if (profile.maxCadenceMs && delta > profile.maxCadenceMs + CLOCK_TOLERANCE_MS) {
        errors.push(`measurement timestamp gap exceeds ${profile.maxCadenceMs}ms`)
      }
    }
  }
  if (
    profile.maxCadenceMs &&
    isFiniteNumber(measurement.sampleIntervalMs) &&
    measurement.sampleIntervalMs > profile.maxCadenceMs
  ) {
    errors.push(`measurement cadence exceeds ${profile.maxCadenceMs}ms`)
  }
  if (profile.durationMs && isFiniteNumber(measurement.durationMs) && measurement.durationMs < profile.durationMs) {
    errors.push(`measurement duration must be at least ${profile.durationMs}ms`)
  }
  if (profile.warmupMs && (!isFiniteNumber(receipt.warmupDurationMs) || receipt.warmupDurationMs < profile.warmupMs)) {
    errors.push(`warmup must be at least ${profile.warmupMs}ms for measured visible scenarios`)
  }
}

function validateScenarioInvariants(receipt, profile, scenario, errors) {
  if (!profile) {
    errors.push(`unknown scenario: ${String(receipt.scenario)}`)
    return
  }
  const observed = receipt.population?.observed
  const active = receipt.population?.active
  if (!['preserved', 'exact'].includes(profile.populationMode)) {
    errors.push(`${scenario} has no explicit population invariant`)
  }
  if (profile.lodRequired !== true) errors.push(`${scenario} has no explicit LOD invariant`)
  if (Number.isInteger(observed) && observed < profile.minObserved) {
    errors.push(`${scenario} requires at least ${profile.minObserved} observed inhabitants`)
  }
  if (Number.isInteger(active) && active < profile.minActive) {
    errors.push(`${scenario} requires at least ${profile.minActive} active inhabitants`)
  }
  if (profile.expectedObserved !== undefined && observed !== profile.expectedObserved) {
    errors.push(`${scenario} requires observed population ${profile.expectedObserved}`)
  }
  if (profile.expectedActive !== undefined && active !== profile.expectedActive) {
    errors.push(`${scenario} requires active population ${profile.expectedActive}`)
  }
  if (isRecord(receipt.environment) && Number.isInteger(observed)) {
    const cityPopulated = receipt.environment.cityPopulated === true
    if (observed > 0 !== cityPopulated) {
      errors.push('cityPopulated must match the observed population')
    }
  }
  if (profile.dormant) {
    if (receipt.renderFrames !== 0) errors.push(`${scenario} requires zero render frames`)
  }
  if (profile.requiredQuality && receipt.qualityTier !== profile.requiredQuality) {
    errors.push(`${scenario} requires quality tier ${profile.requiredQuality}`)
  }
  if (profile.requiredCamera && receipt.cameraState !== profile.requiredCamera) {
    errors.push(`${scenario} requires camera state ${profile.requiredCamera}`)
  }
  if (profile.requiredDialogue && receipt.dialogueState !== profile.requiredDialogue) {
    errors.push(`${scenario} requires dialogue state ${profile.requiredDialogue}`)
  }
  if (profile.requiredRecovery && receipt.recovery !== profile.requiredRecovery) {
    errors.push(`${scenario} requires recovery state ${profile.requiredRecovery}`)
  }
  if (profile.requiredDisposal && receipt.disposal !== profile.requiredDisposal) {
    errors.push(`${scenario} requires disposal state ${profile.requiredDisposal}`)
  }
  if (!profile.dormant && receipt.renderFrames === 0) errors.push(`${scenario} requires measured render frames`)
  if (!profile.dormant && profile.lodTotal !== undefined) {
    const lodMix = receipt.population?.lodMix
    let total = 0
    for (const count of Object.values(lodMix ?? {})) {
      if (Number.isInteger(count) && count >= 0) total += count
    }
    if (total !== profile.lodTotal) errors.push(`${scenario} requires LOD total ${profile.lodTotal}`)
  }
  if (profile.requiredLodKeys) {
    const lodMix = isRecord(receipt.population?.lodMix) ? receipt.population.lodMix : {}
    for (const lod of profile.requiredLodKeys) {
      if (!Object.hasOwn(lodMix, lod)) errors.push(`${scenario} requires LOD level ${lod}`)
    }
  }
  if (profile.requiredPositiveLodKeys) {
    const lodMix = isRecord(receipt.population?.lodMix) ? receipt.population.lodMix : {}
    for (const lod of profile.requiredPositiveLodKeys) {
      if (!Number.isInteger(lodMix[lod]) || lodMix[lod] <= 0) {
        errors.push(`${scenario} requires a positive ${lod} LOD representation`)
      }
    }
  }
}

function validateDirectSummaries(receipt, expected, errors) {
  for (const [field, summaryField] of [
    ['renderFrames', 'maxRenderFrames'],
    ['processCpuDeltaPp', 'avgCpuDeltaPp'],
    ['gpuMemoryDeltaMiB', 'avgGpuMemoryDeltaMiB'],
    ['residentMemoryDriftMiB', 'residentMemoryDriftMiB'],
    ['drawCalls', 'maxDrawCalls'],
    ['visibleTriangles', 'maxVisibleTriangles'],
    ['activeAnimations', 'maxActiveAnimations'],
    ['entities', 'maxEntities'],
    ['textures', 'maxTextures'],
    ['listeners', 'maxListeners'],
    ['timers', 'maxTimers']
  ]) {
    if (field in receipt && isFiniteNumber(receipt[field]) && !equalNumber(receipt[field], expected[summaryField])) {
      errors.push(`${field} does not match raw samples (${receipt[field]} !== ${expected[summaryField]})`)
    }
  }
  if (
    'p95FrameMs' in receipt &&
    isFiniteNumber(receipt.p95FrameMs) &&
    !equalNumber(receipt.p95FrameMs, expected.p95FrameMs)
  ) {
    errors.push(`p95FrameMs does not match raw samples (${receipt.p95FrameMs} !== ${expected.p95FrameMs})`)
  }
  if (
    'p95WorldUpdateMs' in receipt &&
    isFiniteNumber(receipt.p95WorldUpdateMs) &&
    !equalNumber(receipt.p95WorldUpdateMs, expected.p95WorldUpdateMs)
  ) {
    errors.push(
      `p95WorldUpdateMs does not match raw samples (${receipt.p95WorldUpdateMs} !== ${expected.p95WorldUpdateMs})`
    )
  }
}

function validateMonotonicStability(rawSamples, errors) {
  for (const [field, label] of MONOTONIC_FIELDS) {
    const values = rawSamples?.[field]
    if (!Array.isArray(values) || values.length < 2) continue
    const growsMonotonically =
      values.every((value, index) => index === 0 || value >= values[index - 1]) &&
      values.some((value, index) => index > 0 && value > values[index - 1])
    if (growsMonotonically) errors.push(`monotonic ${label} growth detected`)
  }
}

export function isAcceptancePopulationState(scenario, cityPopulated, observed) {
  if (scenario === 'route-unmounted') return cityPopulated === false && observed === 0
  return cityPopulated === true && Number.isInteger(observed) && observed > 0
}

function validateAcceptanceGate(receipt, scenario, errors) {
  const environment = receipt.environment
  const stamp = receipt.buildStamp
  const evidenceClass = receipt.evidenceClass
  const profile = SCENARIO_PROFILES[scenario]
  const eligibleClass = evidenceClass === 'fake-backend-packaged'
  if (!eligibleClass) return false
  const packaged = isRecord(environment) && environment.electronMode === 'packaged'
  const gpuEnabled = isRecord(environment) && environment.gpuEnabled === true
  const fixtureBackend = isRecord(environment) && environment.backendMode === 'fake-backend'
  const populated = isRecord(environment) && environment.cityPopulated === true
  const hasPopulation = Number.isInteger(receipt.population?.observed) && receipt.population.observed > 0
  const populationStateEligible = isAcceptancePopulationState(
    scenario,
    environment?.cityPopulated,
    receipt.population?.observed
  )
  const cleanPinnedBuild = isRecord(stamp) && stamp.dirty === false && stamp.source !== 'fallback'
  const hasRawProvenance = Object.hasOwn(receipt, 'rawProvenance')
  const bridge = receipt.rawProvenance?.bridgeHandshake
  const hasBridgeProvenance =
    isRecord(bridge) &&
    bridge.bridgeVersion === 1 &&
    bridge.packaged === true &&
    bridge.buildSha === receipt.gitSha &&
    typeof bridge.launchNonce === 'string' &&
    bridge.launchNonce.length > 0 &&
    Number.isInteger(bridge.mainPid) &&
    isRecord(bridge.rendererIdentity) &&
    Array.isArray(bridge.supportedPhases) &&
    ['baseline-shell', 'mounted-city'].every(phase => bridge.supportedPhases.includes(phase)) &&
    bridge.processMetricsSource === 'electron.app.getAppMetrics'
  const eligible = Boolean(
    profile &&
    packaged &&
    gpuEnabled &&
    fixtureBackend &&
    populationStateEligible &&
    cleanPinnedBuild &&
    hasRawProvenance &&
    hasBridgeProvenance
  )

  if (!eligible) {
    errors.push(`receipt is not eligible for packaged performance acceptance (${scenario ?? 'unknown scenario'})`)
    if (!packaged) errors.push('packaged performance requires packaged Electron, not dev Electron')
    if (!gpuEnabled) errors.push('packaged performance requires GPU enabled')
    if (!fixtureBackend) errors.push('fake packaged performance requires validated fixture backend provenance')
    if (!populationStateEligible) {
      if (scenario === 'route-unmounted') {
        errors.push('route-unmounted acceptance requires a truthful zero-population unmounted city')
      } else {
        if (!populated) errors.push('packaged performance requires a real populated city, not an empty fake boot')
        if (!hasPopulation)
          errors.push('packaged performance requires a populated population snapshot, not an empty fake boot')
      }
    }
    if (!cleanPinnedBuild) errors.push('packaged performance requires a clean pinned buildStamp')
    if (!hasRawProvenance) errors.push('packaged performance requires versioned rawProvenance')
    if (!hasBridgeProvenance) errors.push('packaged performance requires a versioned bound bridge handshake')
  }

  return eligible
}

/** Validate a receipt and return recomputed summaries plus acceptance status. */
export function validateReceipt(receipt) {
  const errors = []
  if (!isRecord(receipt)) {
    validateCommonShape(receipt, errors)
    return { ok: false, packagedPerformanceEligible: false, errors, summary: {} }
  }

  validateCommonShape(receipt, errors)
  validateBuildStamp(receipt, errors)
  validateArtifactProvenance(receipt, errors)
  validateEnvironment(receipt.environment, errors)
  validatePopulation(receipt.population, errors)
  validateMeasurement(receipt, SCENARIO_PROFILES[receipt.scenario], errors)
  validateScenarioInvariants(receipt, SCENARIO_PROFILES[receipt.scenario], canonicalScenario(receipt.scenario), errors)
  validateRawSamples(receipt.rawSamples, errors)
  validateRawProvenance(receipt, errors)
  const summaries = validateSummaries(receipt.rawSamples, receipt.summaries, errors)
  const expected = summarizeRawSamples(receipt.rawSamples)
  validateDirectSummaries(receipt, expected, errors)
  const scenario = canonicalScenario(receipt.scenario)

  const renderFrames = receipt.renderFrames ?? expected.maxRenderFrames
  const cpuDelta = numericMetric(receipt, summaries, 'processCpuDeltaPp', 'avgCpuDeltaPp', errors, 'process CPU delta')
  const gpuDelta = numericMetric(
    receipt,
    summaries,
    'gpuMemoryDeltaMiB',
    'avgGpuMemoryDeltaMiB',
    errors,
    'GPU memory delta'
  )
  const p95Frame = numericMetric(receipt, summaries, 'p95FrameMs', 'p95FrameMs', errors, 'p95 frame time')
  const p95Update = numericMetric(
    receipt,
    summaries,
    'p95WorldUpdateMs',
    'p95WorldUpdateMs',
    errors,
    'p95 world update'
  )
  const drawCalls = receipt.drawCalls ?? expected.maxDrawCalls
  const triangles = receipt.visibleTriangles ?? expected.maxVisibleTriangles
  const residentDrift = receipt.residentMemoryDriftMiB ?? expected.residentMemoryDriftMiB

  for (const [value, label] of [
    [renderFrames, 'renderFrames'],
    [drawCalls, 'drawCalls'],
    [triangles, 'visibleTriangles']
  ]) {
    if (!isFiniteNumber(value) || value < 0) errors.push(`${label} must be a nonnegative finite number`)
  }
  if (cpuDelta === undefined || cpuDelta === null) errors.push('process CPU delta unavailable; missing CPU is not zero')
  if (gpuDelta === undefined || gpuDelta === null) errors.push('GPU memory delta unavailable; missing GPU is not zero')
  if (!isFiniteNumber(cpuDelta)) errors.push('process CPU delta unavailable; missing CPU is not zero')
  if (!isFiniteNumber(gpuDelta)) errors.push('GPU memory delta unavailable; missing GPU is not zero')

  const profile = SCENARIO_PROFILES[scenario]
  if (profile?.dormant) {
    if (renderFrames !== 0) errors.push(`${scenario} requires zero render frames`)
  }
  if (profile?.cpuLimit !== undefined && isFiniteNumber(cpuDelta) && cpuDelta > profile.cpuLimit) {
    errors.push(`CPU delta exceed ${profile.cpuLimit} percentage points (average)`)
  }
  if (scenario === '100-active') {
    if (receipt.fpsCap !== 30) errors.push('100 active requires a 30 FPS cap')
    if (isFiniteNumber(p95Frame) && p95Frame > profile.p95FrameLimit)
      errors.push(`p95 frame exceed ${profile.p95FrameLimit}ms`)
    if (isFiniteNumber(p95Update) && p95Update > profile.p95UpdateLimit)
      errors.push(`p95 world update exceed ${profile.p95UpdateLimit}ms`)
  }
  if (scenario === '250-lod' && isFiniteNumber(p95Frame) && p95Frame > profile.p95FrameLimit) {
    errors.push(`p95 frame exceed ${profile.p95FrameLimit}ms`)
  }
  if (profile?.drawCalls !== undefined && isFiniteNumber(drawCalls) && drawCalls > profile.drawCalls) {
    errors.push(`draw calls exceed ${profile.drawCalls}`)
  }
  if (profile?.triangles !== undefined && isFiniteNumber(triangles) && triangles > profile.triangles) {
    errors.push(`triangles exceed ${profile.triangles}`)
  }
  if (profile?.stability) {
    if (isFiniteNumber(residentDrift) && residentDrift > THRESHOLDS.stabilityResidentDriftMiB)
      errors.push(`resident memory drift exceed ${THRESHOLDS.stabilityResidentDriftMiB} MiB`)
    validateMonotonicStability(receipt.rawSamples, errors)
  }
  const gpuAbsDelta = expected.maxAbsGpuMemoryDeltaMiB
  if (isFiniteNumber(gpuAbsDelta) && gpuAbsDelta > THRESHOLDS.maxGpuMemoryDeltaMiB) {
    errors.push(`GPU memory exceed ${THRESHOLDS.maxGpuMemoryDeltaMiB} MiB`)
  }
  if (
    receipt.qualityTier === 'Efficient' &&
    /integrated/i.test(String(receipt.environment?.gpuAdapter ?? '')) &&
    receipt.interactionUsable !== true
  ) {
    errors.push('Efficient integrated GPU receipt must prove interaction usability')
  }

  const gateEligible = validateAcceptanceGate(receipt, scenario, errors)
  const canonicalErrors = [...errors]
  if ('pass' in receipt && receipt.pass !== (canonicalErrors.length === 0)) {
    errors.push(`pass contradicts canonical outcome (expected ${canonicalErrors.length === 0})`)
  }
  if (Array.isArray(receipt.errors) && JSON.stringify(receipt.errors) !== JSON.stringify(canonicalErrors)) {
    errors.push('errors contradict canonical outcome')
  }
  return {
    ok: errors.length === 0,
    packagedPerformanceEligible: gateEligible && errors.length === 0,
    errors,
    summary: expected
  }
}

/** Validate a JSON receipt file for command-line/package integration. */
export function validateReceiptFile(path) {
  return validateReceipt(JSON.parse(readFileSync(path, 'utf8')))
}

function main() {
  const path = process.argv[2]
  if (path === '--help') {
    console.log(
      'usage: node lunar-city.mjs <receipt.json>\nValidates an existing hash-bound receipt; does not launch a package or create evidence.'
    )
    return
  }
  if (!path) {
    console.error('usage: node lunar-city.mjs <receipt.json>')
    process.exitCode = 2
    return
  }
  const result = validateReceiptFile(path)
  console.log(JSON.stringify(result, null, 2))
  if (!result.ok) process.exitCode = 1
}

if (import.meta.url === `file://${process.argv[1]}`) main()
