import assert from 'node:assert/strict'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'
import { test } from 'vitest'

import { OMNIVERSE_RECEIPT_SCHEMA, validateOmniverseExport } from './validate-omniverse-export.mjs'

const manifest = {
  version: 2,
  assetVersion: '2.0.0',
  source: { sha256: 'a'.repeat(64) },
  qualityBudgets: {
    balancedOverview: { drawCalls: 180, visibleTriangles: 1_500_000, gpuMiB: 256 },
    balancedWorkerFocus: { drawCalls: 220, visibleTriangles: 2_000_000, gpuMiB: 256 }
  }
}

function receipt(overrides = {}) {
  return {
    schema_name: OMNIVERSE_RECEIPT_SCHEMA,
    schema_version: 1,
    asset_version: '2.0.0',
    manifest_source_sha256: 'a'.repeat(64),
    stage: { usd_path: 'omniverse://lunar-city/world.usd', identifier: '/World/LunarCity' },
    exported_at: '2026-09-05T10:00:00Z',
    status: 'validated',
    reference_only: true,
    production_approved: false,
    budget_profile: 'balancedOverview',
    provenance: {
      source_repo: 'NVIDIA/skills',
      source_revision: 'abcdef1234567890',
      workflow: 'omniverse-usd-performance-tuning'
    },
    performance: { draw_calls: 42, visible_triangles: 100_000, gpu_mib: 64 },
    ...overrides
  }
}

test('accepts a bounded reference-only Omniverse export', () => {
  const result = validateOmniverseExport(receipt(), manifest)
  assert.equal(result.ok, true)
  assert.equal(result.classification, 'reference_only')
  assert.equal(result.runtime_eligible, false)
})

test('rejects a digest mismatch and budget overflow', () => {
  const result = validateOmniverseExport(
    receipt({
      manifest_source_sha256: 'b'.repeat(64),
      performance: { draw_calls: 181, visible_triangles: 100_000, gpu_mib: 64 }
    }),
    manifest
  )
  assert.equal(result.ok, false)
  assert.match(result.errors.join('\n'), /source digest/)
  assert.match(result.errors.join('\n'), /draw-call budget/)
})

test('production mode requires a non-reference accepted receipt', () => {
  const result = validateOmniverseExport(
    receipt({ reference_only: false, production_approved: true, status: 'accepted' }),
    manifest,
    { mode: 'production' }
  )
  assert.equal(result.ok, true)
  assert.equal(result.classification, 'production_candidate')
})

test('invalid mode is rejected and malformed receipts return structured errors', () => {
  const invalidMode = validateOmniverseExport(receipt(), manifest, { mode: 'prodction' })
  assert.equal(invalidMode.ok, false)
  assert.match(invalidMode.errors.join('\n'), /mode must be preview or production/)

  const malformed = validateOmniverseExport(null, manifest)
  assert.equal(malformed.ok, false)
  assert.ok(Array.isArray(malformed.errors))
})

test('malformed production receipts return structured errors without throwing', () => {
  const result = validateOmniverseExport(null, manifest, { mode: 'production' })
  assert.equal(result.ok, false)
  assert.ok(Array.isArray(result.errors))
})

test('malformed production receipts stop before direct property access', () => {
  for (const malformed of [null, [], 'receipt']) {
    const result = validateOmniverseExport(malformed, manifest, { mode: 'production' })
    assert.equal(result.ok, false)
    assert.match(result.errors.join('\\n'), /receipt schema_name is invalid/)
  }
})

test('rejects contradictory approval flags', () => {
  const result = validateOmniverseExport(receipt({ production_approved: true }), manifest)
  assert.equal(result.ok, false)
  assert.match(result.errors.join('\n'), /reference-only export/)
})

test('rejects indeterminate approval flags', () => {
  const result = validateOmniverseExport(receipt({ reference_only: false, production_approved: false }), manifest)
  assert.equal(result.ok, false)
  assert.match(result.errors.join('\n'), /exactly one of reference_only and production_approved/)
})

test('rejects production receipts when manifest bindings are missing', () => {
  for (const manifestOverride of [{ assetVersion: undefined }, { source: {} }]) {
    const result = validateOmniverseExport(
      receipt({ reference_only: false, production_approved: true, status: 'accepted' }),
      { ...manifest, ...manifestOverride },
      { mode: 'production' }
    )
    assert.equal(result.ok, false)
  }
})

test('requires an own quality-budget profile with finite limits', () => {
  const inheritedProfile = validateOmniverseExport(
    receipt({
      status: 'accepted',
      reference_only: false,
      production_approved: true,
      budget_profile: '__proto__',
      performance: { draw_calls: 999_999_999, visible_triangles: 999_999_999, gpu_mib: 999_999_999 }
    }),
    manifest,
    { mode: 'production' }
  )
  assert.equal(inheritedProfile.ok, false)
  assert.match(inheritedProfile.errors.join('\n'), /budget_profile is not in world manifest qualityBudgets/)

  const malformedBudget = validateOmniverseExport(
    receipt({ budget_profile: 'malformed' }),
    {
      ...manifest,
      qualityBudgets: {
        malformed: { drawCalls: Number.POSITIVE_INFINITY, visibleTriangles: 2_000_000, gpuMiB: 256 }
      }
    }
  )
  assert.equal(malformedBudget.ok, false)
  assert.match(malformedBudget.errors.join('\n'), /quality budget limits must be finite and non-negative/)
})

test('requires a value when the CLI mode flag is present', () => {
  const workspace = mkdtempSync(join(tmpdir(), 'omniverse-export-'))
  try {
    const receiptPath = join(workspace, 'reference.json')
    const manifestPath = join(workspace, 'manifest.json')
    writeFileSync(receiptPath, JSON.stringify(receipt()))
    writeFileSync(manifestPath, JSON.stringify(manifest))

    const scriptPath = join(dirname(fileURLToPath(import.meta.url)), 'validate-omniverse-export.mjs')
    const result = spawnSync(process.execPath, [scriptPath, '--receipt', receiptPath, '--manifest', manifestPath, '--mode'], {
      encoding: 'utf8'
    })

    assert.equal(result.status, 2)
    assert.match(result.stderr, /Usage: node validate-omniverse-export\.mjs/)
    assert.equal(result.stdout, '')
  } finally {
    rmSync(workspace, { recursive: true, force: true })
  }
})
