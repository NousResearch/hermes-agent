import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

import {
  type RolloutPlan,
  validateRolloutPlan
} from '../src/lib/managed-rollout-contract'

import type { TrustedAssuranceReader, TrustedSourceReader } from './managed-rollout-assurance'
import { canonicalCodeRoot, installationFingerprint, sourceFingerprint } from './managed-rollout-identity'
import type { TrustedInventoryReader, TrustedInventorySnapshot } from './managed-rollout-inventory'
import {
  type TargetResolution,
  validateTargetResolution
} from './managed-rollout-preflight'

const MAX_REVIEW_BYTES = 1024 * 1024
const MAX_ASSURANCE_BYTES = 64 * 1024
const SHA256_RE = /^[0-9a-f]{64}$/
const SHA40_RE = /^[0-9a-f]{40}$/
const SAFE_SEGMENT_RE = /^[A-Za-z0-9._-]+$/

export interface ProductionSource {
  id: string
  kind: string
  label?: string
  host?: string
  user?: string
  port?: number
  remoteProfile?: string
  remoteHermesPath?: string
  [key: string]: unknown
}

export interface ProductionInventoryInspection {
  installId: string
  codeRoot: string
  repositoryId: string
  headSha: string
  requiredScopeIds: readonly string[]
  source: {
    connectionId: string
    connectionConfigRevision: string | number
    verifiedHostKeyFingerprint: string
    remoteUser: string
    port: number
    configuredProfile: string
    configuredCodePath: string
  }
  /** Retained for callers that already calculated it; the adapter recomputes it. */
  sourceFingerprint?: string
}

export interface ManagedRolloutProductionAdapterOptions {
  nowMono: () => number
  listSources: () => readonly ProductionSource[]
  inspectSource: (source: ProductionSource) => Promise<ProductionInventoryInspection | null>
  git: (connectionId: string, args: readonly string[], repositoryRoot: string) => Promise<string | Uint8Array>
  reviewManifestPath: string
  assuranceRoot: string
}

export interface ManagedRolloutProductionAdapters {
  inventoryReader: TrustedInventoryReader
  sourceReader: TrustedSourceReader
  assuranceReader: TrustedAssuranceReader
  resolveTarget: (request: {
    connectionIds: string[]
    inventoryRevision: string
    retryOf: string | null
  }) => Promise<{ plan: RolloutPlan; resolution: TargetResolution }>
  ready: () => boolean
}

function safeFile(pathname: string, maxBytes: number): Buffer | null {
  try {
    const stat = fs.lstatSync(pathname)

    if (!stat.isFile() || stat.isSymbolicLink() || stat.size > maxBytes) {return null}

    return fs.readFileSync(pathname)
  } catch {
    return null
  }
}

function safeSegment(value: string, label: string): string {
  if (!SAFE_SEGMENT_RE.test(value)) {throw new Error(`managed-rollout-${label}-invalid`)}

  return value
}

function safeSha(value: string, label: string, length: number): string {
  const expression = length === 40 ? SHA40_RE : SHA256_RE

  if (!expression.test(value)) {throw new Error(`managed-rollout-${label}-invalid`)}

  return value
}

function exactObject(value: unknown, keys: readonly string[]): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value) &&
    Object.keys(value).length === keys.length && keys.every(key => Object.prototype.hasOwnProperty.call(value, key))
}

function reviewPayload(raw: Buffer): { plan: RolloutPlan; resolution: TargetResolution } {
  let value: unknown

  try {
    value = JSON.parse(raw.toString('utf8'))
  } catch {
    throw new Error('review-manifest-malformed')
  }

  if (!exactObject(value, ['schema', 'plan', 'resolution']) || value.schema !== 1) {
    throw new Error('review-manifest-malformed')
  }

  let plan: RolloutPlan
  let resolution: TargetResolution

  try {
    plan = validateRolloutPlan(value.plan)
    resolution = validateTargetResolution(value.resolution as TargetResolution)
  } catch {
    throw new Error('review-manifest-invalid')
  }

  if (JSON.stringify(plan.target) !== JSON.stringify(resolution.target)) {
    throw new Error('review-manifest-target-mismatch')
  }

  return { plan, resolution }
}

function pathInside(root: string, pathname: string): boolean {
  const resolvedRoot = path.resolve(root)
  const resolvedPath = path.resolve(pathname)

  return resolvedPath === resolvedRoot || resolvedPath.startsWith(`${resolvedRoot}${path.sep}`)
}

function assurancePath(root: string, profile: string, targetSha: string, sourceFingerprintValue: string): string {
  safeSegment(profile, 'assurance-profile')
  safeSha(targetSha, 'target-sha', 40)
  safeSha(sourceFingerprintValue, 'source-fingerprint', 64)
  const pathname = path.join(root, 'evidence', profile, targetSha, `${sourceFingerprintValue}.json`)

  if (!pathInside(root, pathname)) {throw new Error('managed-rollout-assurance-path-invalid')}

  return pathname
}

function profilePath(root: string, profile: string): string {
  safeSegment(profile, 'assurance-profile')
  const pathname = path.join(root, 'profiles', `${profile}.json`)

  if (!pathInside(root, pathname)) {throw new Error('managed-rollout-assurance-path-invalid')}

  return pathname
}

function hasControlCharacter(value: string): boolean {
  return [...value].some(character => {
    const code = character.charCodeAt(0)
    return code <= 0x1f || code === 0x7f
  })
}

function readProfile(root: string, profile: string): { generation: number; requiredControlIds: readonly string[] } | null {
  const raw = safeFile(profilePath(root, profile), MAX_ASSURANCE_BYTES)

  if (!raw) {return null}

  try {
    const value = JSON.parse(raw.toString('utf8')) as Record<string, unknown>

    if (!exactObject(value, ['generation', 'requiredControlIds'])) {return null}

    if (!Number.isSafeInteger(value.generation) || (value.generation as number) < 0) {return null}
    const ids = value.requiredControlIds

    if (!Array.isArray(ids) || ids.length === 0 || ids.some(id => typeof id !== 'string' || !id || hasControlCharacter(id))) {return null}

    if (new Set(ids).size !== ids.length) {return null}

    return { generation: value.generation as number, requiredControlIds: Object.freeze([...ids] as string[]) }
  } catch {
    return null
  }
}

function readEvidence(root: string, profile: string, targetSha: string, sourceFingerprintValue: string): Uint8Array | null {
  return safeFile(assurancePath(root, profile, targetSha, sourceFingerprintValue), MAX_ASSURANCE_BYTES)
}

export function createManagedRolloutProductionAdapters(
  options: ManagedRolloutProductionAdapterOptions
): ManagedRolloutProductionAdapters {
  const repositoryOwners = new Map<string, string>()

  const inventoryReader: TrustedInventoryReader = {
    capture: async (): Promise<TrustedInventorySnapshot | null> => {
      const inspected = await Promise.all(
        options.listSources()
          .filter(source => source.kind === 'ssh')
          .map(async source => ({ source, inspection: await options.inspectSource(source).catch(() => null) }))
      )

      const usable = inspected.filter(
        (item): item is { source: ProductionSource; inspection: ProductionInventoryInspection } => item.inspection !== null
      )

      if (usable.length === 0) {return null}

      const observations = usable.map(({ source, inspection }) => {
        const root = canonicalCodeRoot(inspection.codeRoot)

        const installation = installationFingerprint({
          installId: inspection.installId,
          codeRoot: root,
          repositoryId: inspection.repositoryId
        })

        const computedSourceFingerprint = sourceFingerprint({
          ...inspection.source,
          installationFingerprint: installation
        })

        return {
          source,
          inspection,
          observation: {
            installId: inspection.installId,
            connectionId: source.id,
            aliasConnectionIds: [] as string[],
            codeRoot: root,
            repositoryId: inspection.repositoryId,
            headSha: inspection.headSha,
            requiredScopeIds: [...inspection.requiredScopeIds].sort(),
            source: {
              connectionId: source.id,
              connectionConfigRevision: inspection.source.connectionConfigRevision,
              verifiedHostKeyFingerprint: inspection.source.verifiedHostKeyFingerprint,
              remoteUser: inspection.source.remoteUser,
              port: inspection.source.port,
              configuredProfile: inspection.source.configuredProfile,
              configuredCodePath: inspection.source.configuredCodePath
            },
            computedSourceFingerprint
          }
        }
      })

      repositoryOwners.clear()
      const groups = new Map<string, any[]>()

      for (const item of observations) {
        const key = `${item.observation.installId}:${item.observation.codeRoot}:${item.observation.repositoryId}`
        const entries = groups.get(key) || []
        entries.push(item)
        groups.set(key, entries)
      }

      const consolidated = [...groups.values()].map(entries => {
        entries.sort((left, right) => left.source.id.localeCompare(right.source.id))
        const primary = entries[0]
        primary.observation.aliasConnectionIds = entries.slice(1).map(item => item.source.id)
        const priorOwner = repositoryOwners.get(primary.observation.codeRoot)

        if (priorOwner === undefined) {repositoryOwners.set(primary.observation.codeRoot, primary.source.id)}
        else if (priorOwner !== primary.source.id) {repositoryOwners.set(primary.observation.codeRoot, null)}

        return primary.observation
      })

      const ordered = consolidated.sort((left, right) => left.installId.localeCompare(right.installId))
      const inventoryRevision = crypto.createHash('sha256').update(JSON.stringify(ordered), 'utf8').digest('hex')
      const capturedMono = options.nowMono()

      if (!Number.isFinite(capturedMono) || capturedMono < 0) {return null}

      return { inventoryRevision, capturedMono, observations: ordered }
    }
  }

  const sourceReader: TrustedSourceReader = {
    nowMono: options.nowMono,
    git: async (args, repositoryRoot) => {
      let connectionId = repositoryOwners.get(canonicalCodeRoot(repositoryRoot))

      if (!connectionId) {
        await inventoryReader.capture()
        connectionId = repositoryOwners.get(canonicalCodeRoot(repositoryRoot))
      }

      if (!connectionId) {throw new Error('reviewed-repository-source-unavailable')}

      return options.git(connectionId, args, canonicalCodeRoot(repositoryRoot))
    }
  }

  const assuranceReader: TrustedAssuranceReader = {
    readEvidence: async (profile, targetSha, sourceFingerprintValue) =>
      readEvidence(options.assuranceRoot, profile, targetSha, sourceFingerprintValue),
    readProfile: async profile => readProfile(options.assuranceRoot, profile)
  }

  return {
    inventoryReader,
    sourceReader,
    assuranceReader,
    resolveTarget: async request => {
      if (!Array.isArray(request.connectionIds) || request.connectionIds.length === 0 ||
          new Set(request.connectionIds).size !== request.connectionIds.length ||
          request.connectionIds.some(id => typeof id !== 'string' || !id)) {
        throw new Error('review-manifest-connections-invalid')
      }

      const raw = safeFile(options.reviewManifestPath, MAX_REVIEW_BYTES)

      if (!raw) {throw new Error('review-manifest-unavailable')}
      const reviewed = reviewPayload(raw)

      if (reviewed.plan.inventoryRevision !== request.inventoryRevision) {throw new Error('review-manifest-inventory-mismatch')}
      const rows = reviewed.plan.rows.map(row => row.connectionId)

      if (rows.length !== request.connectionIds.length || !request.connectionIds.every(id => rows.includes(id))) {
        throw new Error('review-manifest-connections-mismatch')
      }

      if (reviewed.plan.retryOf !== request.retryOf) {throw new Error('review-manifest-retry-mismatch')}

      return reviewed
    },
    ready: () => {
      const raw = safeFile(options.reviewManifestPath, MAX_REVIEW_BYTES)

      if (!raw) {return false}

      try {
        const reviewed = reviewPayload(raw)

        return reviewed.plan.rows.every(row => {
          const source = row.reviewedSource

          return Boolean(
            source &&
            readProfile(options.assuranceRoot, source.assuranceProfile) &&
            readEvidence(options.assuranceRoot, source.assuranceProfile, reviewed.plan.target.sha, row.sourceFingerprint)
          )
        })
      } catch {
        return false
      }
    }
  }
}
