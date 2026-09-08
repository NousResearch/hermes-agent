import { resolve } from 'node:path'
import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'

export const OMNIVERSE_RECEIPT_SCHEMA = 'nvidia_omniverse_asset_receipt_v1'
const DEFAULT_MANIFEST_PATH = fileURLToPath(new URL('../../public/lunar-city/v2/world-manifest.v2.json', import.meta.url))

function isRecord(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function nonEmpty(value) {
  return typeof value === 'string' && value.trim().length > 0
}

function finiteNonNegative(value) {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0
}

function isSha256(value) {
  return typeof value === 'string' && /^[0-9a-f]{64}$/i.test(value)
}

function isTimestamp(value) {
  return typeof value === 'string' &&
    /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/.test(value) &&
    !Number.isNaN(Date.parse(value))
}

/**
 * Validate an Omniverse/USD export receipt against Lunar City's existing
 * manifest and quality budgets. This is an asset-pipeline boundary; it does
 * not make the renderer or Electron depend on Omniverse.
 */
export function validateOmniverseExport(receipt, manifest, { mode = 'preview' } = {}) {
  const errors = []
  if (!['preview', 'production'].includes(mode)) {
    errors.push('mode must be preview or production')
  }
  if (!isRecord(receipt) || receipt.schema_name !== OMNIVERSE_RECEIPT_SCHEMA) {
    errors.push('receipt schema_name is invalid')
  }
  if (!isRecord(receipt)) {
    return {
      ok: false,
      errors: Object.freeze(errors),
      classification: 'reference_only',
      runtime_eligible: false,
      manifest_source_sha256: manifest?.source?.sha256 ?? null,
      budget_profile: null,
      mode
    }
  }
  if (receipt?.schema_version !== 1) errors.push('receipt schema_version must equal 1')
  if (!isRecord(manifest) || manifest.version !== 2) errors.push('world manifest version must equal 2')
  if (!nonEmpty(manifest?.assetVersion)) {
    errors.push('world manifest assetVersion is required')
  } else if (receipt?.asset_version !== manifest.assetVersion) {
    errors.push('asset_version does not match world manifest')
  }
  if (!isSha256(manifest?.source?.sha256)) {
    errors.push('world manifest source digest is required')
  } else if (receipt?.manifest_source_sha256 !== manifest.source.sha256) {
    errors.push('manifest source digest does not match world manifest')
  }

  const stage = receipt?.stage
  if (!isRecord(stage) || !nonEmpty(stage.usd_path) || !nonEmpty(stage.identifier)) {
    errors.push('USD stage path and identifier are required')
  }
  if (!isTimestamp(receipt?.exported_at)) errors.push('exported_at must be an RFC 3339 timestamp')
  if (!nonEmpty(receipt?.status) || !['validated', 'accepted', 'best_effort'].includes(receipt.status)) {
    errors.push('receipt status is invalid')
  }
  if (typeof receipt?.reference_only !== 'boolean') errors.push('reference_only must be boolean')
  if (typeof receipt?.production_approved !== 'boolean') errors.push('production_approved must be boolean')
  if (!isRecord(receipt?.provenance)) {
    errors.push('provenance is required')
  } else {
    for (const field of ['source_repo', 'source_revision', 'workflow']) {
      if (!nonEmpty(receipt.provenance[field])) errors.push(`provenance.${field} is required`)
    }
  }

  const budgetProfile = receipt?.budget_profile
  const qualityBudgets = manifest?.qualityBudgets
  const hasOwnBudgetProfile = isRecord(qualityBudgets) && Object.hasOwn(qualityBudgets, budgetProfile)
  const budget = hasOwnBudgetProfile ? qualityBudgets[budgetProfile] : undefined
  const budgetHasValidLimits =
    hasOwnBudgetProfile &&
    isRecord(budget) &&
    ['drawCalls', 'visibleTriangles', 'gpuMiB'].every((field) => finiteNonNegative(budget[field]))
  if (!hasOwnBudgetProfile || !isRecord(budget)) {
    errors.push('budget_profile is not in world manifest qualityBudgets')
  } else if (!budgetHasValidLimits) {
    errors.push('quality budget limits must be finite and non-negative')
  }
  const performance = receipt?.performance
  if (!isRecord(performance)) {
    errors.push('performance metrics are required')
  } else {
    for (const field of ['draw_calls', 'visible_triangles', 'gpu_mib']) {
      if (!finiteNonNegative(performance[field])) errors.push(`performance.${field} must be finite and non-negative`)
    }
    if (budgetHasValidLimits && finiteNonNegative(performance.draw_calls) && performance.draw_calls > budget.drawCalls) {
      errors.push('draw-call budget exceeded')
    }
    if (budgetHasValidLimits && finiteNonNegative(performance.visible_triangles) && performance.visible_triangles > budget.visibleTriangles) {
      errors.push('visible-triangle budget exceeded')
    }
    if (budgetHasValidLimits && finiteNonNegative(performance.gpu_mib) && performance.gpu_mib > budget.gpuMiB) {
      errors.push('GPU memory budget exceeded')
    }
  }

  if (mode === 'production') {
    if (receipt?.reference_only !== false) errors.push('production mode rejects reference-only exports')
    if (receipt?.production_approved !== true) errors.push('production mode requires production_approved')
    if (receipt?.status !== 'accepted') errors.push('production mode requires accepted status')
  }
  if (receipt?.production_approved === true && receipt?.reference_only === true) {
    errors.push('reference-only export cannot be production-approved')
  }
  if (receipt?.reference_only === receipt?.production_approved) {
    errors.push('exactly one of reference_only and production_approved must be true')
  }

  const productionApproved = receipt?.production_approved === true
  const classification = productionApproved ? 'production_candidate' : 'reference_only'
  return {
    ok: errors.length === 0,
    errors: Object.freeze(errors),
    classification,
    runtime_eligible: false,
    manifest_source_sha256: manifest?.source?.sha256 ?? null,
    budget_profile: budgetProfile ?? null,
    mode
  }
}
function option(args, name, fallback = undefined) {
  const index = args.indexOf(name)
  if (index < 0) return fallback
  const value = args[index + 1]
  if (value === undefined || value.startsWith('--')) throw new Error(`${name} requires a value`)
  return value
}

if (resolve(process.argv[1] ?? '') === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2)
  const known = new Set(['--receipt', '--manifest', '--mode'])
  const unknown = args.filter((arg) => arg.startsWith('--') && !known.has(arg))
  if (unknown.length > 0) {
    console.error(`Unknown option: ${unknown[0]}`)
    process.exitCode = 2
  } else {
    let receiptPath
    let manifestPath
    let mode
    try {
      receiptPath = option(args, '--receipt')
      manifestPath = option(args, '--manifest', DEFAULT_MANIFEST_PATH)
      mode = option(args, '--mode', 'preview')
    } catch (error) {
      console.error(error.message)
      process.exitCode = 2
    }
    if (!receiptPath || mode === undefined) {
    console.error('Usage: node validate-omniverse-export.mjs --receipt <receipt.json> [--manifest <manifest.json>] [--mode preview|production]')
    process.exitCode = 2
    } else if (process.exitCode !== 2) {
    const [receipt, manifest] = await Promise.all([
      readFile(receiptPath, 'utf8').then(JSON.parse),
      readFile(manifestPath, 'utf8').then(JSON.parse)
    ])
    const result = validateOmniverseExport(receipt, manifest, { mode })
    console.log(JSON.stringify(result, null, 2))
    if (!result.ok) process.exitCode = 1
    }
  }
}
