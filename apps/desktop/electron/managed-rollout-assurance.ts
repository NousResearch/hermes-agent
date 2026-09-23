/** T8's read-only consumer of exact Git provenance and #92618 evidence. */
import crypto from 'node:crypto'

import type { ReviewedSourceBinding, RolloutTarget } from '../src/lib/managed-rollout-contract'
import { validateReviewedSourceBinding, validateRolloutTarget } from '../src/lib/managed-rollout-contract'
import { canonicalCodeRoot, canonicalRepositoryId } from './managed-rollout-identity'

const SHA256 = /^[0-9a-f]{64}$/
const REQUIRED_CONTROL_RESULTS = new Set(['pass'])
const verifiedEvidence = new WeakSet<object>()
const verifiedSources = new WeakMap<object, { inventoryRevision: string; verifiedMono: number }>()
export const REVIEWED_SOURCE_FRESHNESS_MS = 10_000

export interface TrustedSourceReader {
  /** Execute a bounded, read-only Git query on the authenticated installation. */
  git(args: readonly string[], repositoryRoot: string): Promise<string | Uint8Array>
  /** Main-process monotonic time captured after the last Git assertion. */
  nowMono(): number
}

export interface TrustedAssuranceReader {
  /** Load the applicable raw evidence envelope from #92618 custody, not from IPC. */
  readEvidence(profile: string, targetSha: string, sourceFingerprint: string): Promise<Uint8Array | null>
  /** Resolve active generation and required IDs from the separately controlled security profile. */
  readProfile(profile: string): Promise<{ generation: number; requiredControlIds: readonly string[] } | null>
}

export interface VerifiedAssuranceEvidence {
  readonly profile: string
  readonly evidenceSha256: string
  readonly generation: number
  readonly repositoryId: string
  readonly targetSha: string
  readonly sourceFingerprint: string
  readonly expiresAt: number
}

export interface AssuranceExpectation {
  profile: string
  repositoryId: string
  targetSha: string
  sourceFingerprint: string
  generation: number
  now: number
}

export interface SourceExpectation {
  target: RolloutTarget
  trustedOriginUrl: string
  repositoryRoot: string
  branch: string
  inventoryRevision: string
}

export class AdmissionEvidenceError extends Error {
  constructor(readonly code: string) {
    super(code)
    this.name = 'AdmissionEvidenceError'
  }
}

function refusal(code: string): never {
  throw new AdmissionEvidenceError(code)
}

function text(value: string | Uint8Array): string {
  return typeof value === 'string' ? value : new TextDecoder('utf-8', { fatal: true }).decode(value)
}

function exactObject(value: unknown, keys: readonly string[]): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value) &&
    Object.keys(value).length === keys.length && keys.every(key => Object.prototype.hasOwnProperty.call(value, key))
}

function validInstant(value: unknown): number | null {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}T/.test(value)) return null
  const instant = Date.parse(value)
  return Number.isFinite(instant) ? instant : null
}

/** A later branch tip C may contain reviewed B; exact object B remains selected. */
export async function verifyReviewedGitSource(
  value: ReviewedSourceBinding,
  expected: SourceExpectation,
  reader: TrustedSourceReader
): Promise<ReviewedSourceBinding> {
  const source = Object.freeze(validateReviewedSourceBinding(value))
  const target = validateRolloutTarget(expected.target)
  const ref = `refs/remotes/origin/${expected.branch}`
  if (source.targetSha !== target.sha || source.resolvedRef !== ref || expected.branch !== target.branch)
    refusal('reviewed-target-or-ref-mismatch')
  if (source.originUrl !== expected.trustedOriginUrl || canonicalRepositoryId(source.originUrl) !== target.repositoryId)
    refusal('reviewed-origin-mismatch')
  if (canonicalCodeRoot(source.repositoryRoot) !== canonicalCodeRoot(expected.repositoryRoot))
    refusal('reviewed-repository-mismatch')

  const query = async (...args: string[]): Promise<string> => text(await reader.git(args, source.repositoryRoot)).trim()
  const [actualRoot, actualOrigin, branch, objectType] = await Promise.all([
    query('rev-parse', '--show-toplevel'),
    query('remote', 'get-url', 'origin'),
    query('symbolic-ref', '--quiet', '--short', 'HEAD'),
    query('cat-file', '-t', source.targetSha)
  ])
  if (canonicalCodeRoot(actualRoot) !== canonicalCodeRoot(source.repositoryRoot)) refusal('reviewed-repository-mismatch')
  if (actualOrigin !== source.originUrl) refusal('reviewed-origin-mismatch')
  if (branch !== expected.branch) refusal('reviewed-branch-mismatch')
  if (objectType !== 'commit') refusal('reviewed-object-unavailable')
  try {
    await reader.git(['merge-base', '--is-ancestor', source.targetSha, ref], source.repositoryRoot)
  } catch {
    refusal('reviewed-object-not-on-origin-ref')
  }
  const { parseProtocolMetadata, PROTOCOL_RESOURCE_PATH } = await import('./managed-rollout-preflight')
  const protocol = await reader.git(['show', `${source.targetSha}:${PROTOCOL_RESOURCE_PATH}`], source.repositoryRoot)
  parseProtocolMetadata(typeof protocol === 'string' ? new TextEncoder().encode(protocol) : protocol)
  const verifiedMono = reader.nowMono()
  if (!Number.isFinite(verifiedMono) || verifiedMono < 0 || !expected.inventoryRevision)
    refusal('reviewed-source-clock-or-revision-invalid')
  verifiedSources.set(source, { inventoryRevision: expected.inventoryRevision, verifiedMono })
  return source
}

export function isVerifiedGitSource(value: unknown): value is ReviewedSourceBinding {
  return typeof value === 'object' && value !== null && verifiedSources.has(value)
}

export function reviewedGitSourceMetadata(value: ReviewedSourceBinding):
  { inventoryRevision: string; verifiedMono: number } | null {
  return verifiedSources.get(value) ?? null
}

/** Check one #92618 consumer envelope. The reader owns independent custody. */
export async function verifyApplicableAssurance(
  expected: AssuranceExpectation,
  reader: TrustedAssuranceReader
): Promise<VerifiedAssuranceEvidence> {
  const [raw, profile] = await Promise.all([
    reader.readEvidence(expected.profile, expected.targetSha, expected.sourceFingerprint),
    reader.readProfile(expected.profile)
  ])
  const requiredControlIds = profile?.requiredControlIds
  if (
    !profile || !Number.isSafeInteger(profile.generation) || profile.generation !== expected.generation ||
    !requiredControlIds || !Array.isArray(requiredControlIds) || requiredControlIds.length === 0 ||
    requiredControlIds.some(id => typeof id !== 'string' || !id || /[\x00-\x1f\x7f]/.test(id)) ||
    new Set(requiredControlIds).size !== requiredControlIds.length
  ) refusal('assurance-profile-unavailable')
  if (!raw || raw.byteLength === 0 || raw.byteLength > 64 * 1024) refusal('assurance-evidence-missing')
  let document: unknown
  try {
    document = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(raw))
  } catch {
    refusal('assurance-evidence-malformed')
  }
  const keys = [
    'schema', 'profile', 'generation', 'repositoryId', 'targetSha',
    'sourceFingerprint', 'observedAt', 'expiresAt', 'controls'
  ]
  if (!exactObject(document, keys) || document.schema !== 1) refusal('assurance-evidence-malformed')
  const observedAt = validInstant(document.observedAt)
  const expiresAt = validInstant(document.expiresAt)
  if (
    document.profile !== expected.profile || document.repositoryId !== expected.repositoryId ||
    document.targetSha !== expected.targetSha || document.sourceFingerprint !== expected.sourceFingerprint ||
    document.generation !== expected.generation || typeof document.generation !== 'number' ||
    !Number.isSafeInteger(document.generation) ||
    !Number.isSafeInteger(expected.now) || observedAt === null || expiresAt === null ||
    observedAt > expected.now || expiresAt <= expected.now || expiresAt <= observedAt
  ) refusal('assurance-evidence-stale-or-mismatched')
  if (!Array.isArray(document.controls) || document.controls.length === 0) refusal('assurance-controls-missing')
  const ids = new Set<string>()
  const results = new Map<string, { required: boolean; result: string }>()
  let requiredCount = 0
  for (const entry of document.controls) {
    if (!exactObject(entry, ['id', 'required', 'result', 'receiptSha256'])) refusal('assurance-control-malformed')
    if (
      typeof entry.id !== 'string' || !entry.id || ids.has(entry.id) ||
      typeof entry.required !== 'boolean' || typeof entry.result !== 'string' ||
      typeof entry.receiptSha256 !== 'string' || !SHA256.test(entry.receiptSha256)
    ) refusal('assurance-control-malformed')
    ids.add(entry.id)
    results.set(entry.id, { required: entry.required, result: entry.result })
    if (entry.required) {
      requiredCount += 1
      if (!REQUIRED_CONTROL_RESULTS.has(entry.result)) refusal('assurance-required-control-not-passed')
    }
  }
  if (requiredCount === 0) refusal('assurance-controls-missing')
  if (requiredControlIds.some(id => !results.get(id)?.required || results.get(id)?.result !== 'pass'))
    refusal('assurance-required-control-not-passed')
  const proof = Object.freeze({
    profile: expected.profile,
    evidenceSha256: crypto.createHash('sha256').update(raw).digest('hex'),
    generation: expected.generation,
    repositoryId: expected.repositoryId,
    targetSha: expected.targetSha,
    sourceFingerprint: expected.sourceFingerprint,
    expiresAt
  })
  verifiedEvidence.add(proof)
  return proof
}

export function isVerifiedAssurance(value: unknown): value is VerifiedAssuranceEvidence {
  return typeof value === 'object' && value !== null && verifiedEvidence.has(value)
}
