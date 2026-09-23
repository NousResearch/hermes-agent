import assert from 'node:assert/strict'
import crypto from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import {
  createManagedRolloutProductionAdapters,
  type ProductionInventoryInspection,
  type ProductionSource
} from './managed-rollout-production-adapters'
import type { RolloutPlan } from '../src/lib/managed-rollout-contract'
import type { TargetResolution } from './managed-rollout-preflight'

const INSTALL_A = 'a'.repeat(32)
const INSTALL_B = 'b'.repeat(32)
const CONNECTION_A = '11111111-1111-4111-8111-111111111111'
const CONNECTION_B = '22222222-2222-4222-8222-222222222222'
const ROOT = '/srv/hermes-agent'
const REPOSITORY = 'github.com/nousresearch/hermes-agent'
const TARGET_SHA = 'c'.repeat(40)
const SOURCE_A = 'd'.repeat(64)
const SOURCE_B = 'e'.repeat(64)

function source(id: string, installId: string): ProductionSource {
  return { id, kind: 'ssh', label: id, host: `${id}.example.test`, user: 'hermes', port: 22, remoteProfile: 'default' }
}

function inspection(sourceValue: ProductionSource, installId: string, sourceFingerprint: string): ProductionInventoryInspection {
  return {
    installId,
    codeRoot: ROOT,
    repositoryId: REPOSITORY,
    headSha: 'f'.repeat(40),
    requiredScopeIds: [`scope:${installId}`],
    source: {
      connectionId: sourceValue.id,
      connectionConfigRevision: `config:${sourceValue.id}`,
      verifiedHostKeyFingerprint: `host:${sourceValue.id}`,
      remoteUser: sourceValue.user,
      port: sourceValue.port,
      configuredProfile: sourceValue.remoteProfile,
      configuredCodePath: '/srv/hermes-agent/.venv/bin/hermes'
    },
    sourceFingerprint
  }
}

function plan(revision: string): RolloutPlan {
  return {
    target: { repositoryId: REPOSITORY, branch: 'main', sha: TARGET_SHA, protocol: 1 },
    inventoryRevision: revision,
    waves: [[INSTALL_A, INSTALL_B]],
    concurrency: 2,
    promotionPolicy: 'auto-if-healthy',
    rows: [
      {
        installId: INSTALL_A,
        connectionId: CONNECTION_A,
        installationFingerprint: '1'.repeat(64),
        sourceFingerprint: SOURCE_A,
        admittedHead: 'f'.repeat(40),
        requiredScopeIds: [`scope:${INSTALL_A}`],
        eligible: true,
        reviewedSource: {
          repositoryRoot: ROOT,
          originUrl: 'https://github.com/NousResearch/hermes-agent.git',
          resolvedRef: 'refs/remotes/origin/main',
          targetSha: TARGET_SHA,
          assuranceProfile: 'profile-v1',
          assuranceEvidenceSha256: '2'.repeat(64),
          assuranceGeneration: 1
        }
      },
      {
        installId: INSTALL_B,
        connectionId: CONNECTION_B,
        installationFingerprint: '3'.repeat(64),
        sourceFingerprint: SOURCE_B,
        admittedHead: 'f'.repeat(40),
        requiredScopeIds: [`scope:${INSTALL_B}`],
        eligible: true,
        reviewedSource: {
          repositoryRoot: ROOT,
          originUrl: 'https://github.com/NousResearch/hermes-agent.git',
          resolvedRef: 'refs/remotes/origin/main',
          targetSha: TARGET_SHA,
          assuranceProfile: 'profile-v1',
          assuranceEvidenceSha256: '4'.repeat(64),
          assuranceGeneration: 1
        }
      }
    ],
    retryOf: null,
    exclusions: []
  }
}

function resolution(): TargetResolution {
  return {
    id: 'resolution-1',
    target: { repositoryId: REPOSITORY, branch: 'main', sha: TARGET_SHA, protocol: 1 },
    fingerprint: '5'.repeat(64),
    cachePath: '/managed-rollout-cache/resolution-1',
    createdAt: Date.parse('2026-09-22T00:00:00.000Z'),
    expiresAt: Date.parse('2026-09-23T00:00:00.000Z')
  }
}

test('captures a coherent inventory revision and routes Git through the inspected source', async () => {
  const sources = [source(CONNECTION_A, INSTALL_A), source(CONNECTION_B, INSTALL_B)]
  const gitCalls: string[] = []
  const adapters = createManagedRolloutProductionAdapters({
    nowMono: () => 1000,
    listSources: () => sources,
    inspectSource: async current => current.id === CONNECTION_A
      ? inspection(current, INSTALL_A, SOURCE_A)
      : inspection(current, INSTALL_B, SOURCE_B),
    git: async (connectionId, args, repositoryRoot) => {
      gitCalls.push(`${connectionId}:${repositoryRoot}:${args.join(' ')}`)
      return 'ok'
    },
    reviewManifestPath: 'unused',
    assuranceRoot: 'unused'
  })

  const snapshot = await adapters.inventoryReader.capture()
  assert.ok(snapshot)
  assert.match(snapshot!.inventoryRevision, /^[0-9a-f]{64}$/)
  assert.equal(snapshot!.observations.length, 2)
  assert.deepEqual(snapshot!.observations[0].aliasConnectionIds, [])
  assert.equal(await adapters.sourceReader.git(['rev-parse', 'HEAD'], ROOT), 'ok')
  assert.equal(gitCalls.length, 1)
  assert.ok(gitCalls[0].includes('rev-parse HEAD'))
})

test('consolidates multiple configured connections to one installation with deterministic alias ownership', async () => {
  const sources = [source(CONNECTION_B, INSTALL_A), source(CONNECTION_A, INSTALL_A)]
  const adapters = createManagedRolloutProductionAdapters({
    nowMono: () => 1000,
    listSources: () => sources,
    inspectSource: async current => inspection(current, INSTALL_A, SOURCE_A),
    git: async () => '',
    reviewManifestPath: 'unused',
    assuranceRoot: 'unused'
  })

  const snapshot = await adapters.inventoryReader.capture()
  assert.ok(snapshot)
  assert.equal(snapshot!.observations.length, 1)
  assert.equal(snapshot!.observations[0].connectionId, CONNECTION_A)
  assert.deepEqual(snapshot!.observations[0].aliasConnectionIds, [CONNECTION_B])
  assert.equal(await adapters.sourceReader.git(['status', '--short'], ROOT), '')
})


test('resolves only a bounded review manifest matching the requested inventory and connections', async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'managed-rollout-review-'))
  const manifestPath = path.join(directory, 'review.json')
  const revision = '6'.repeat(64)
  const payload = { schema: 1, plan: plan(revision), resolution: resolution() }
  fs.writeFileSync(manifestPath, JSON.stringify(payload), { mode: 0o600 })

  try {
    const adapters = createManagedRolloutProductionAdapters({
      nowMono: () => 1000,
      listSources: () => [],
      inspectSource: async () => { throw new Error('not used') },
      git: async () => '',
      reviewManifestPath: manifestPath,
      assuranceRoot: directory
    })
    assert.equal(adapters.ready(), false)
    const result = await adapters.resolveTarget({
      connectionIds: [CONNECTION_A, CONNECTION_B],
      inventoryRevision: revision,
      retryOf: null
    })
    assert.equal(result.resolution.id, 'resolution-1')
    assert.equal(result.plan.rows.length, 2)
    await assert.rejects(
      () => adapters.resolveTarget({ connectionIds: [CONNECTION_A], inventoryRevision: revision, retryOf: null }),
      /review-manifest-connections-mismatch/
    )
  } finally {
    fs.rmSync(directory, { recursive: true, force: true })
  }
})

test('assurance custody is keyed by profile, target, and source fingerprint', async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'managed-rollout-assurance-'))
  const profile = path.join(directory, 'profiles', 'profile-v1.json')
  const evidence = path.join(directory, 'evidence', 'profile-v1', TARGET_SHA, `${SOURCE_A}.json`)
  fs.mkdirSync(path.dirname(profile), { recursive: true })
  fs.mkdirSync(path.dirname(evidence), { recursive: true })
  fs.writeFileSync(profile, JSON.stringify({ generation: 1, requiredControlIds: ['control-a'] }), { mode: 0o600 })
  const raw = new TextEncoder().encode(JSON.stringify({ control: 'a' }))
  fs.writeFileSync(evidence, raw, { mode: 0o600 })

  try {
    const adapters = createManagedRolloutProductionAdapters({
      nowMono: () => 1000,
      listSources: () => [],
      inspectSource: async () => { throw new Error('not used') },
      git: async () => '',
      reviewManifestPath: 'unused',
      assuranceRoot: directory
    })
    assert.deepEqual(await adapters.assuranceReader.readProfile('profile-v1'), {
      generation: 1,
      requiredControlIds: ['control-a']
    })
    assert.deepEqual(
      [...(await adapters.assuranceReader.readEvidence('profile-v1', TARGET_SHA, SOURCE_A))],
      [...raw]
    )
    assert.equal(await adapters.assuranceReader.readEvidence('profile-v1', TARGET_SHA, SOURCE_B), null)
    assert.equal(crypto.createHash('sha256').update(raw).digest('hex').length, 64)
  } finally {
    fs.rmSync(directory, { recursive: true, force: true })
  }
})
