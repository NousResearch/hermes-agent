/** T7's read-only consumer of a coherent, main-owned installation inventory. */
import type { RolloutPlan, RolloutPlanRow } from '../src/lib/managed-rollout-contract'
import { validateRolloutPlan } from '../src/lib/managed-rollout-contract'
import {
  canonicalCodeRoot, installationFingerprint, sourceFingerprint,
  type SourceFingerprintInput
} from './managed-rollout-identity'

export const INVENTORY_FRESHNESS_MS = 10_000

export interface InventoryObservation {
  installId: string
  connectionId: string
  aliasConnectionIds: readonly string[]
  codeRoot: string
  repositoryId: string
  headSha: string
  requiredScopeIds: readonly string[]
  source: Omit<SourceFingerprintInput, 'installationFingerprint'>
}

export interface TrustedInventorySnapshot {
  inventoryRevision: string
  capturedMono: number
  observations: readonly InventoryObservation[]
}

export interface TrustedInventoryReader {
  /** Capture one coherent revision after read-only SSH identity/scope probes. */
  capture(): Promise<TrustedInventorySnapshot | null>
}

export interface VerifiedInventoryRow {
  readonly installId: string
  readonly connectionId: string
  readonly inventoryRevision: string
  readonly observedMono: number
  readonly repositoryRoot: string
  readonly repositoryId: string
  readonly installationFingerprint: string
  readonly sourceFingerprint: string
  readonly admittedHead: string
  readonly requiredScopeIds: readonly string[]
  readonly aliasConnectionIds: readonly string[]
}

const verifiedRows = new WeakSet<object>()

export function isVerifiedInventoryRow(value: unknown): value is VerifiedInventoryRow {
  return typeof value === 'object' && value !== null && verifiedRows.has(value)
}

function invalid(reason: string): never { throw new Error(reason) }

function unique(values: readonly string[]): boolean {
  return values.every(value => typeof value === 'string' && value.length > 0 && !/[\x00-\x1f\x7f]/.test(value)) &&
    new Set(values).size === values.length
}

function assertMatches(row: RolloutPlanRow, item: InventoryObservation, repositoryId: string): VerifiedInventoryRow {
  if (
    item.installId !== row.installId || item.connectionId !== row.connectionId ||
    item.repositoryId !== repositoryId || !unique(item.aliasConnectionIds) ||
    item.aliasConnectionIds.includes(item.connectionId) || !unique(item.requiredScopeIds)
  ) invalid('inventory-identity-mismatch')
  const root = canonicalCodeRoot(item.codeRoot)
  const installation = installationFingerprint({
    installId: item.installId, codeRoot: root, repositoryId: item.repositoryId
  })
  const source = sourceFingerprint({ ...item.source, installationFingerprint: installation })
  if (
    item.source.connectionId !== row.connectionId ||
    installation !== row.installationFingerprint || source !== row.sourceFingerprint ||
    item.headSha !== row.admittedHead ||
    !row.reviewedSource || canonicalCodeRoot(row.reviewedSource.repositoryRoot) !== root ||
    row.requiredScopeIds === null ||
    JSON.stringify([...item.requiredScopeIds].sort()) !== JSON.stringify([...row.requiredScopeIds].sort())
  ) invalid('inventory-source-or-scope-mismatch')
  return Object.freeze({
    installId: item.installId,
    connectionId: item.connectionId,
    inventoryRevision: '',
    observedMono: 0,
    repositoryRoot: root,
    repositoryId: item.repositoryId,
    installationFingerprint: installation,
    sourceFingerprint: source,
    admittedHead: item.headSha,
    requiredScopeIds: Object.freeze([...item.requiredScopeIds].sort()),
    aliasConnectionIds: Object.freeze([...item.aliasConnectionIds].sort())
  })
}

/** A stale revision or alias collision invalidates the whole capture. */
export async function verifyTrustedInventory(
  planValue: RolloutPlan,
  nowMono: number,
  reader: TrustedInventoryReader
): Promise<ReadonlyMap<string, VerifiedInventoryRow>> {
  const plan = validateRolloutPlan(planValue)
  const snapshot = await reader.capture()
  if (!plan.inventoryRevision || !snapshot || snapshot.inventoryRevision !== plan.inventoryRevision)
    invalid('inventory-revision-mismatch')
  if (
    !Number.isFinite(nowMono) || !Number.isFinite(snapshot.capturedMono) ||
    snapshot.capturedMono < 0 || nowMono < snapshot.capturedMono ||
    nowMono - snapshot.capturedMono > INVENTORY_FRESHNESS_MS
  ) invalid('inventory-stale')
  if (!Array.isArray(snapshot.observations)) invalid('inventory-unavailable')
  const byId = new Map<string, InventoryObservation>()
  const connectionOwners = new Map<string, string>()
  for (const item of snapshot.observations) {
    if (!item || byId.has(item.installId)) invalid('inventory-duplicate-installation')
    byId.set(item.installId, item)
    for (const connectionId of [item.connectionId, ...item.aliasConnectionIds]) {
      const prior = connectionOwners.get(connectionId)
      if (!connectionId || (prior && prior !== item.installId)) invalid('inventory-alias-conflict')
      connectionOwners.set(connectionId, item.installId)
    }
  }
  const result = new Map<string, VerifiedInventoryRow>()
  for (const row of plan.rows) {
    const item = byId.get(row.installId)
    if (!item) invalid('inventory-installation-missing')
    const checked = assertMatches(row, item, plan.target.repositoryId)
    const proof = Object.freeze({
      ...checked, inventoryRevision: snapshot.inventoryRevision, observedMono: snapshot.capturedMono
    })
    verifiedRows.add(proof)
    result.set(row.installId, proof)
  }
  return result
}
