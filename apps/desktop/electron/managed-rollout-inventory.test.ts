import { describe, expect, it } from 'vitest'

import type { RolloutPlan } from '../src/lib/managed-rollout-contract'
import { installationFingerprint, sourceFingerprint } from './managed-rollout-identity'
import { isVerifiedInventoryRow, verifyTrustedInventory } from './managed-rollout-inventory'

const installId = '1'.repeat(32)
const root = '/srv/hermes'
const repositoryId = 'github.com/acme/hermes'
const head = 'a'.repeat(40)
const source = {
  connectionId: 'ssh-a', connectionConfigRevision: 2,
  verifiedHostKeyFingerprint: 'SHA256:host-key', remoteUser: 'operator', port: 22,
  configuredProfile: 'default', configuredCodePath: root
}
const installation = installationFingerprint({ installId, codeRoot: root, repositoryId })
const sshSource = sourceFingerprint({ ...source, installationFingerprint: installation })

function fixture() {
  const plan: RolloutPlan = {
    target: { repositoryId, branch: 'main', sha: head, protocol: 1 },
    inventoryRevision: 'inventory-7', waves: [[installId]], concurrency: 1,
    promotionPolicy: 'manual', retryOf: null, exclusions: [],
    rows: [{
      installId, connectionId: 'ssh-a', installationFingerprint: installation,
      sourceFingerprint: sshSource, admittedHead: head, requiredScopeIds: ['default'], eligible: true,
      reviewedSource: {
        repositoryRoot: root, originUrl: 'https://github.com/acme/hermes.git',
        resolvedRef: 'refs/remotes/origin/main', targetSha: head,
        assuranceProfile: 'managed-ssh-v1', assuranceEvidenceSha256: 'b'.repeat(64), assuranceGeneration: 4
      }
    }]
  }
  const snapshot = {
    inventoryRevision: 'inventory-7', capturedMono: 1_000,
    observations: [{
      installId, connectionId: 'ssh-a', aliasConnectionIds: ['ssh-alias'], codeRoot: root,
      repositoryId, headSha: head, requiredScopeIds: ['default'], source
    }]
  }
  return { plan, snapshot }
}

describe('managed rollout trusted inventory capture', () => {
  it('brands a matching coherent capture and rejects stale revision, identity, source, or alias collision', async () => {
    const { plan, snapshot } = fixture()
    const verified = await verifyTrustedInventory(plan, 1_500, { capture: async () => snapshot })
    expect(isVerifiedInventoryRow(verified.get(installId))).toBe(true)
    expect(verified.get(installId)?.aliasConnectionIds).toEqual(['ssh-alias'])
    await expect(verifyTrustedInventory(plan, 11_001, { capture: async () => snapshot }))
      .rejects.toThrow('inventory-stale')
    await expect(verifyTrustedInventory(plan, 1_500, { capture: async () => ({
      ...snapshot, inventoryRevision: 'inventory-8'
    }) })).rejects.toThrow('inventory-revision-mismatch')
    await expect(verifyTrustedInventory(plan, 1_500, { capture: async () => ({
      ...snapshot, observations: [{ ...snapshot.observations[0], codeRoot: '/srv/other' }]
    }) })).rejects.toThrow('inventory-source-or-scope-mismatch')
    await expect(verifyTrustedInventory(plan, 1_500, { capture: async () => ({
      ...snapshot, observations: [{ ...snapshot.observations[0], source: { ...source, port: 2200 } }]
    }) })).rejects.toThrow('inventory-source-or-scope-mismatch')
    await expect(verifyTrustedInventory(plan, 1_500, { capture: async () => ({
      ...snapshot, observations: [snapshot.observations[0], { ...snapshot.observations[0], connectionId: 'ssh-alias' }]
    }) })).rejects.toThrow('inventory-duplicate-installation')
  })
})
