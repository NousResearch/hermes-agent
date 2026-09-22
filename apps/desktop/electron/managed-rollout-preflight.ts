import { execFile } from 'node:child_process'
import crypto from 'node:crypto'
import { lstat, mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'

import type {
  PlanChange,
  PromotionPolicy,
  RolloutPlan,
  RolloutPlanRow,
  RolloutTarget
} from '../src/lib/managed-rollout-contract'
import { validateRolloutPlan, validateRolloutTarget } from '../src/lib/managed-rollout-contract'
import { canonicalRepositoryId } from './managed-rollout-identity'
import {
  isVerifiedAssurance, isVerifiedGitSource,
  type VerifiedAssuranceEvidence
} from './managed-rollout-assurance'
import type { ReviewedSourceBinding } from '../src/lib/managed-rollout-contract'

export const PROTOCOL_RESOURCE_PATH = 'hermes_cli/update_rollout_protocol.json'
export const SUPPORTED_PROTOCOL_VERSION = 1
export const MAX_PROTOCOL_BYTES = 4 * 1024
export const MAX_CACHE_BYTES = 512 * 1024 * 1024
export const MAX_CACHE_ACQUISITION_MS = 300 * 1000
export const CACHE_RETENTION_MS = 24 * 60 * 60 * 1000
export const PREFLIGHT_TOKEN_TTL_MS = 15 * 60 * 1000

export type CacheFailureCode =
  | 'cache-invalid-path'
  | 'cache-foreign'
  | 'cache-symlink'
  | 'cache-timeout'
  | 'cache-storage-limit'
  | 'repository-access-denied'
  | 'repository-unavailable'
  | 'target-object-missing'
  | 'target-object-not-commit'
  | 'protocol-missing'
  | 'protocol-invalid'
  | 'protocol-unsupported'

export class ProtocolInspectionError extends Error {
  readonly code: CacheFailureCode | 'protocol-too-large' | 'protocol-invalid-utf8' | 'git-command-failed'

  constructor(code: ProtocolInspectionError['code'], message: string) {
    super(message)
    this.name = 'ProtocolInspectionError'
    this.code = code
  }
}

interface StrictNumber {
  readonly raw: string
  readonly value: number
}

class StrictJsonParser {
  private index = 0

  constructor(private readonly input: string) {}

  parse(): unknown {
    const value = this.parseValue()
    this.skipWhitespace()

    if (this.index !== this.input.length)
      throw new ProtocolInspectionError('protocol-invalid', 'protocol metadata has trailing data')

    return value
  }

  private skipWhitespace(): void {
    while (this.index < this.input.length && /[\u0009\u000a\u000d\u0020]/.test(this.input[this.index])) this.index += 1
  }

  private parseValue(): unknown {
    this.skipWhitespace()
    const character = this.input[this.index]

    if (character === '{') return this.parseObject()
    if (character === '[') return this.parseArray()
    if (character === '"') return this.parseString()
    if (character === 't' && this.input.startsWith('true', this.index)) {
      this.index += 4
      return true
    }
    if (character === 'f' && this.input.startsWith('false', this.index)) {
      this.index += 5
      return false
    }
    if (character === 'n' && this.input.startsWith('null', this.index)) {
      this.index += 4
      return null
    }

    return this.parseNumber()
  }

  private parseObject(): Record<string, unknown> {
    this.index += 1
    const result: Record<string, unknown> = {}
    const keys = new Set<string>()
    this.skipWhitespace()

    if (this.input[this.index] === '}') {
      this.index += 1
      return result
    }

    for (;;) {
      this.skipWhitespace()
      if (this.input[this.index] !== '"')
        throw new ProtocolInspectionError('protocol-invalid', 'object key is not a string')
      const key = this.parseString()
      if (typeof key !== 'string' || keys.has(key))
        throw new ProtocolInspectionError('protocol-invalid', 'duplicate object key')
      keys.add(key)
      this.skipWhitespace()
      if (this.input[this.index] !== ':')
        throw new ProtocolInspectionError('protocol-invalid', 'object key lacks a colon')
      this.index += 1
      result[key] = this.parseValue()
      this.skipWhitespace()
      const delimiter = this.input[this.index]
      if (delimiter === '}') {
        this.index += 1
        return result
      }
      if (delimiter !== ',') throw new ProtocolInspectionError('protocol-invalid', 'object lacks a delimiter')
      this.index += 1
    }
  }

  private parseArray(): unknown[] {
    this.index += 1
    const result: unknown[] = []
    this.skipWhitespace()

    if (this.input[this.index] === ']') {
      this.index += 1
      return result
    }

    for (;;) {
      result.push(this.parseValue())
      this.skipWhitespace()
      const delimiter = this.input[this.index]
      if (delimiter === ']') {
        this.index += 1
        return result
      }
      if (delimiter !== ',') throw new ProtocolInspectionError('protocol-invalid', 'array lacks a delimiter')
      this.index += 1
    }
  }

  private parseString(): string {
    const start = this.index
    this.index += 1

    while (this.index < this.input.length) {
      const character = this.input[this.index]

      if (character === '\\') {
        this.index += 2
        continue
      }
      if (character === '"') {
        this.index += 1
        const raw = this.input.slice(start, this.index)

        try {
          return JSON.parse(raw) as string
        } catch {
          throw new ProtocolInspectionError('protocol-invalid', 'invalid JSON string')
        }
      }
      if (character < ' ') throw new ProtocolInspectionError('protocol-invalid', 'control byte in JSON string')
      this.index += 1
    }

    throw new ProtocolInspectionError('protocol-invalid', 'unterminated JSON string')
  }

  private parseNumber(): StrictNumber {
    const match = /^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?/.exec(this.input.slice(this.index))

    if (!match) throw new ProtocolInspectionError('protocol-invalid', 'invalid JSON value')

    this.index += match[0].length
    const value = Number(match[0])

    if (!Number.isFinite(value)) throw new ProtocolInspectionError('protocol-invalid', 'number is not finite')

    return { raw: match[0], value }
  }
}

export interface ProtocolMetadata {
  protocol: 1
}

export function parseProtocolMetadata(input: string | Uint8Array): ProtocolMetadata {
  const bytes = typeof input === 'string' ? Buffer.from(input, 'utf8') : Buffer.from(input)

  if (bytes.byteLength > MAX_PROTOCOL_BYTES)
    throw new ProtocolInspectionError('protocol-too-large', 'protocol metadata exceeds 4 KiB')

  let text: string

  try {
    text = new TextDecoder('utf-8', { fatal: true }).decode(bytes)
  } catch {
    throw new ProtocolInspectionError('protocol-invalid-utf8', 'protocol metadata is not valid UTF-8')
  }

  let value: unknown

  try {
    value = new StrictJsonParser(text).parse()
  } catch (error) {
    if (error instanceof ProtocolInspectionError) throw error
    throw new ProtocolInspectionError('protocol-invalid', 'protocol metadata is malformed JSON')
  }

  if (
    !value ||
    Array.isArray(value) ||
    typeof value !== 'object' ||
    Object.getPrototypeOf(value) !== Object.prototype
  ) {
    throw new ProtocolInspectionError('protocol-invalid', 'protocol metadata must be an object')
  }

  const record = value as Record<string, unknown>

  if (Object.keys(record).length !== 1 || !Object.prototype.hasOwnProperty.call(record, 'protocol')) {
    throw new ProtocolInspectionError('protocol-invalid', 'protocol metadata must contain exactly protocol')
  }

  const protocol = record.protocol

  if (!protocol || typeof protocol !== 'object' || !('raw' in protocol) || (protocol as StrictNumber).raw !== '1') {
    throw new ProtocolInspectionError('protocol-unsupported', 'protocol metadata does not declare integer protocol 1')
  }

  return { protocol: SUPPORTED_PROTOCOL_VERSION }
}

export interface GitRunOptions {
  cwd: string
  timeoutMs: number
  maxOutputBytes: number
}

export interface GitCommandResult {
  stdout: string | Uint8Array
  stderr?: string | Uint8Array
}

export type GitRunner = (
  args: readonly string[],
  options: GitRunOptions
) => Promise<GitCommandResult | string | Uint8Array>

function asBytes(output: GitCommandResult | string | Uint8Array): Uint8Array {
  if (typeof output === 'string') return Buffer.from(output, 'utf8')
  if (output instanceof Uint8Array) return output
  return typeof output.stdout === 'string' ? Buffer.from(output.stdout, 'utf8') : output.stdout
}

function asText(output: GitCommandResult | string | Uint8Array): string {
  return new TextDecoder('utf-8', { fatal: false }).decode(asBytes(output))
}

function defaultGitRunner(args: readonly string[], options: GitRunOptions): Promise<GitCommandResult> {
  return new Promise((resolve, reject) => {
    execFile(
      'git',
      [...args],
      {
        cwd: options.cwd,
        windowsHide: true,
        timeout: options.timeoutMs,
        maxBuffer: options.maxOutputBytes,
        encoding: 'buffer',
        env: { ...process.env, GIT_TERMINAL_PROMPT: '0', GIT_CONFIG_NOSYSTEM: '1' }
      },
      (error, stdout, stderr) => {
        if (error) {
          const failure = error as unknown as Error & { stderr?: Uint8Array | string }
          failure.stderr = stderr
          reject(failure)
          return
        }

        resolve({ stdout, stderr })
      }
    )
  })
}

const SHA_RE = /^[0-9a-f]{40}$/
const CACHE_ID_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/

function validateSha(sha: string): string {
  if (!SHA_RE.test(sha))
    throw new ProtocolInspectionError('target-object-missing', 'target SHA must be literal lowercase hex')

  return sha
}

function validateCachePart(value: string, label: string): string {
  if (!CACHE_ID_RE.test(value) || value.includes('..'))
    throw new ProtocolInspectionError('cache-invalid-path', `${label} is unsafe`)

  return value
}

function outputErrorCode(error: unknown): CacheFailureCode {
  const value = String((error as { stderr?: unknown })?.stderr || (error as { message?: unknown })?.message || error)

  if (/timed out|timeout|ETIMEDOUT/i.test(value)) return 'cache-timeout'
  if (/authentication|permission denied|could not read from remote|access denied/i.test(value))
    return 'repository-access-denied'
  if (/not found|does not exist|bad object|unknown revision/i.test(value)) return 'target-object-missing'

  return 'repository-unavailable'
}

export interface ExactGitObjectInspection {
  sha: string
  objectType: 'commit'
  protocol: 1
  protocolDigest: string
  protocolPath: string
}

export async function inspectExactGitObject(options: {
  cwd: string
  sha: string
  runGit?: GitRunner
  protocolPath?: string
  timeoutMs?: number
}): Promise<ExactGitObjectInspection> {
  const sha = validateSha(options.sha)
  const protocolPath = options.protocolPath || PROTOCOL_RESOURCE_PATH

  if (protocolPath !== PROTOCOL_RESOURCE_PATH || protocolPath.includes('..') || protocolPath.startsWith('/')) {
    throw new ProtocolInspectionError('protocol-invalid', 'protocol resource path is not canonical')
  }

  const runGit = options.runGit || defaultGitRunner
  const timeoutMs = options.timeoutMs ?? 10_000
  let objectTypeOutput: GitCommandResult | string | Uint8Array

  try {
    objectTypeOutput = await runGit(['cat-file', '-t', sha], {
      cwd: options.cwd,
      timeoutMs,
      maxOutputBytes: 128
    })
  } catch (error) {
    throw new ProtocolInspectionError(outputErrorCode(error), 'exact target object could not be inspected')
  }

  if (asText(objectTypeOutput).trim() !== 'commit') {
    throw new ProtocolInspectionError('target-object-not-commit', 'exact target object is not a commit')
  }

  let metadataOutput: GitCommandResult | string | Uint8Array

  try {
    metadataOutput = await runGit(['show', `${sha}:${protocolPath}`], {
      cwd: options.cwd,
      timeoutMs,
      maxOutputBytes: MAX_PROTOCOL_BYTES
    })
  } catch (error) {
    const code = outputErrorCode(error)
    throw new ProtocolInspectionError(
      code === 'target-object-missing' ? 'protocol-missing' : code,
      'protocol resource is not readable from the exact object'
    )
  }

  const bytes = asBytes(metadataOutput)
  parseProtocolMetadata(bytes)

  return {
    sha,
    objectType: 'commit',
    protocol: SUPPORTED_PROTOCOL_VERSION,
    protocolDigest: crypto.createHash('sha256').update(bytes).digest('hex'),
    protocolPath
  }
}

export async function tryInspectExactGitObject(
  options: Parameters<typeof inspectExactGitObject>[0]
): Promise<{ ok: true; inspection: ExactGitObjectInspection } | { ok: false; code: string; message: string }> {
  try {
    return { ok: true, inspection: await inspectExactGitObject(options) }
  } catch (error) {
    return {
      ok: false,
      code: error instanceof ProtocolInspectionError ? error.code : 'git-command-failed',
      message: error instanceof Error ? error.message : String(error)
    }
  }
}

export interface OwnedCacheRecord {
  path: string
  ownerId: string
  cacheId: string
  sha: string
  createdAt: number
  lastReferencedAt: number
  reviewReferences: number
  activeReferences: number
  unresolvedReferences: number
}

export interface CachePathState {
  exists: boolean
  symlink: boolean
  directory: boolean
}

export interface CacheFileSystem {
  ensureDirectory: (directory: string) => Promise<void>
  writeManifest: (directory: string, record: OwnedCacheRecord) => Promise<void>
  inspectPath: (entry: string) => Promise<CachePathState>
  measureBytes: (directory: string) => Promise<number>
  removePath?: (entry: string) => Promise<void>
  readManifest?: (directory: string) => Promise<Partial<OwnedCacheRecord> | null>
}

async function measureDirectoryBytes(directory: string): Promise<number> {
  const { readdir, stat } = await import('node:fs/promises')
  let total = 0

  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const entryPath = path.join(directory, entry.name)
    if (entry.isSymbolicLink()) throw new ProtocolInspectionError('cache-symlink', 'cache contains a symlink')
    if (entry.isDirectory()) total += await measureDirectoryBytes(entryPath)
    else total += (await stat(entryPath)).size
  }

  return total
}

const defaultCacheFileSystem: CacheFileSystem = {
  ensureDirectory: async directory => {
    await mkdir(directory, { recursive: true })
  },
  writeManifest: async (directory, record) =>
    writeFile(path.join(directory, '.managed-rollout-cache.json'), JSON.stringify(record), {
      encoding: 'utf8',
      mode: 0o600
    }),
  inspectPath: async entry => {
    try {
      const info = await lstat(entry)
      return { exists: true, symlink: info.isSymbolicLink(), directory: info.isDirectory() }
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return { exists: false, symlink: false, directory: false }
      throw error
    }
  },
  measureBytes: measureDirectoryBytes,
  removePath: entry => rm(entry, { recursive: true, force: false }),
  readManifest: async directory => {
    try {
      return JSON.parse(
        await readFile(path.join(directory, '.managed-rollout-cache.json'), 'utf8')
      ) as Partial<OwnedCacheRecord>
    } catch {
      return null
    }
  }
}

function cachePathWithinRoot(cacheRoot: string, candidate: string): boolean {
  const root = path.resolve(cacheRoot)
  const entry = path.resolve(candidate)
  const relative = path.relative(root, entry)

  return relative !== '' && relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative)
}

export function ownedCachePath(cacheRoot: string, ownerId: string, cacheId: string): string {
  validateCachePart(ownerId, 'ownerId')
  validateCachePart(cacheId, 'cacheId')

  return path.join(cacheRoot, `${ownerId}--${cacheId}`)
}

function cacheEligible(cache: OwnedCacheRecord, now: number, ttlMs: number): { eligible: boolean; reason?: string } {
  if (
    !Number.isFinite(now) ||
    !Number.isFinite(ttlMs) ||
    ttlMs < 0 ||
    !Number.isFinite(cache.createdAt) ||
    !Number.isFinite(cache.lastReferencedAt) ||
    ![cache.reviewReferences, cache.activeReferences, cache.unresolvedReferences].every(
      value => Number.isSafeInteger(value) && value >= 0
    )
  ) {
    return { eligible: false, reason: 'cache-metadata-invalid' }
  }
  if (cache.reviewReferences !== 0 || cache.activeReferences !== 0 || cache.unresolvedReferences !== 0) {
    return { eligible: false, reason: 'cache-referenced' }
  }
  if (now < cache.lastReferencedAt || now - cache.lastReferencedAt < ttlMs) {
    return { eligible: false, reason: 'cache-not-expired' }
  }

  return { eligible: true }
}

export async function deleteOwnedCache(
  cache: OwnedCacheRecord,
  options: { cacheRoot: string; ownerId: string; now?: number; ttlMs?: number; fileSystem?: CacheFileSystem }
): Promise<{ deleted: boolean; reason?: string }> {
  const fileSystem = options.fileSystem || defaultCacheFileSystem
  const now = options.now ?? Date.now()
  const ttlMs = options.ttlMs ?? CACHE_RETENTION_MS

  const rootState = await fileSystem.inspectPath(options.cacheRoot)
  if (rootState.symlink || (rootState.exists && !rootState.directory)) {
    return { deleted: false, reason: 'cache-foreign' }
  }

  if (cache.ownerId !== options.ownerId || !cachePathWithinRoot(options.cacheRoot, cache.path)) {
    return { deleted: false, reason: 'cache-foreign' }
  }
  if (path.resolve(cache.path) !== path.resolve(ownedCachePath(options.cacheRoot, cache.ownerId, cache.cacheId))) {
    return { deleted: false, reason: 'cache-path-mismatch' }
  }

  const eligibility = cacheEligible(cache, now, ttlMs)
  if (!eligibility.eligible) return { deleted: false, reason: eligibility.reason }

  const state = await fileSystem.inspectPath(cache.path)
  if (!state.exists) return { deleted: false, reason: 'cache-missing' }
  if (state.symlink || !state.directory) return { deleted: false, reason: 'cache-symlink' }

  if (!fileSystem.readManifest) return { deleted: false, reason: 'cache-foreign' }
  const manifest = await fileSystem.readManifest(cache.path)

  if (
    !manifest ||
    manifest.ownerId !== cache.ownerId ||
    manifest.cacheId !== cache.cacheId ||
    manifest.sha !== cache.sha ||
    manifest.createdAt !== cache.createdAt ||
    manifest.lastReferencedAt !== cache.lastReferencedAt ||
    manifest.reviewReferences !== 0 ||
    manifest.activeReferences !== 0 ||
    manifest.unresolvedReferences !== 0
  ) {
    return { deleted: false, reason: 'cache-foreign' }
  }

  try {
    await fileSystem.measureBytes(cache.path)
  } catch (error) {
    return {
      deleted: false,
      reason:
        error instanceof ProtocolInspectionError && error.code === 'cache-symlink' ? 'cache-symlink' : 'cache-foreign'
    }
  }

  if (!fileSystem.removePath) return { deleted: false, reason: 'cache-removal-unavailable' }
  await fileSystem.removePath(cache.path)

  return { deleted: true }
}

export interface CacheAcquisitionOptions {
  cacheRoot: string
  ownerId: string
  cacheId: string
  repository: string
  sha: string
  maxBytes?: number
  timeoutMs?: number
  now?: () => number
  runGit?: GitRunner
  fileSystem?: CacheFileSystem
}

export type CacheAcquisitionResult =
  | { ok: true; cache: OwnedCacheRecord; inspection: ExactGitObjectInspection }
  | { ok: false; code: CacheFailureCode | string; message: string }

export async function acquireExactGitCache(options: CacheAcquisitionOptions): Promise<CacheAcquisitionResult> {
  const fileSystem = options.fileSystem || defaultCacheFileSystem
  const now = options.now || Date.now
  const configuredMaxBytes = options.maxBytes ?? MAX_CACHE_BYTES
  const configuredTimeoutMs = options.timeoutMs ?? MAX_CACHE_ACQUISITION_MS
  const maxBytes = Math.min(configuredMaxBytes, MAX_CACHE_BYTES)
  const timeoutMs = Math.min(configuredTimeoutMs, MAX_CACHE_ACQUISITION_MS)
  const started = now()
  let cachePath: string | null = null
  let created = false

  try {
    if (!Number.isSafeInteger(started)) throw new ProtocolInspectionError('cache-timeout', 'cache clock is invalid')
    validateCachePart(options.ownerId, 'ownerId')
    validateCachePart(options.cacheId, 'cacheId')
    validateSha(options.sha)
    canonicalRepositoryId(options.repository)
    if (!Number.isSafeInteger(configuredMaxBytes) || configuredMaxBytes < 1) {
      throw new ProtocolInspectionError('cache-storage-limit', 'cache budget is invalid')
    }
    if (!Number.isSafeInteger(configuredTimeoutMs) || configuredTimeoutMs < 1) {
      throw new ProtocolInspectionError('cache-timeout', 'cache timeout is invalid')
    }
    cachePath = ownedCachePath(options.cacheRoot, options.ownerId, options.cacheId)

    if (!cachePathWithinRoot(options.cacheRoot, cachePath))
      throw new ProtocolInspectionError('cache-invalid-path', 'cache path escapes owner root')

    const rootState = await fileSystem.inspectPath(options.cacheRoot)
    if (rootState.symlink || (rootState.exists && !rootState.directory)) {
      throw new ProtocolInspectionError('cache-foreign', 'cache root is not an owned directory')
    }

    const state = await fileSystem.inspectPath(cachePath)
    if (state.symlink || (state.exists && !state.directory))
      throw new ProtocolInspectionError('cache-symlink', 'owned cache is not a regular directory')

    let cache: OwnedCacheRecord = {
      path: cachePath,
      ownerId: options.ownerId,
      cacheId: options.cacheId,
      sha: options.sha,
      createdAt: started,
      lastReferencedAt: started,
      reviewReferences: 0,
      activeReferences: 0,
      unresolvedReferences: 0
    }

    if (!state.exists) {
      await fileSystem.ensureDirectory(cachePath)
      created = true
      await fileSystem.writeManifest(cachePath, cache)
    } else {
      if (!fileSystem.readManifest) {
        throw new ProtocolInspectionError('cache-foreign', 'existing cache has no readable ownership manifest')
      }

      const manifest = await fileSystem.readManifest(cachePath)
      if (
        !manifest ||
        manifest.ownerId !== cache.ownerId ||
        manifest.cacheId !== cache.cacheId ||
        manifest.sha !== cache.sha ||
        typeof manifest.createdAt !== 'number' ||
        !Number.isFinite(manifest.createdAt) ||
        typeof manifest.lastReferencedAt !== 'number' ||
        !Number.isFinite(manifest.lastReferencedAt) ||
        typeof manifest.reviewReferences !== 'number' ||
        !Number.isSafeInteger(manifest.reviewReferences) ||
        manifest.reviewReferences < 0 ||
        typeof manifest.activeReferences !== 'number' ||
        !Number.isSafeInteger(manifest.activeReferences) ||
        manifest.activeReferences < 0 ||
        typeof manifest.unresolvedReferences !== 'number' ||
        !Number.isSafeInteger(manifest.unresolvedReferences) ||
        manifest.unresolvedReferences < 0
      ) {
        throw new ProtocolInspectionError(
          'cache-foreign',
          'existing cache ownership manifest does not match the request'
        )
      }

      cache = {
        ...cache,
        createdAt: manifest.createdAt,
        lastReferencedAt: manifest.lastReferencedAt,
        reviewReferences: manifest.reviewReferences,
        activeReferences: manifest.activeReferences,
        unresolvedReferences: manifest.unresolvedReferences
      }
    }

    const runGit = options.runGit || defaultGitRunner
    const elapsed = () => {
      const current = now()
      if (!Number.isSafeInteger(current) || current < started)
        throw new ProtocolInspectionError('cache-timeout', 'cache clock moved backwards or became invalid')
      return current - started
    }
    const remaining = () => Math.max(1, timeoutMs - elapsed())

    if (elapsed() >= timeoutMs) throw new ProtocolInspectionError('cache-timeout', 'cache acquisition deadline expired')

    try {
      if (created) {
        await runGit(['init', '--bare', '--quiet'], {
          cwd: cachePath,
          timeoutMs: remaining(),
          maxOutputBytes: 4 * 1024
        })
      }
      await runGit(['fetch', '--no-tags', '--depth=1', '--no-write-fetch-head', options.repository, options.sha], {
        cwd: cachePath,
        timeoutMs: remaining(),
        maxOutputBytes: 4 * 1024
      })
    } catch (error) {
      throw new ProtocolInspectionError(outputErrorCode(error), 'authorized repository object acquisition failed')
    }

    if (elapsed() >= timeoutMs) throw new ProtocolInspectionError('cache-timeout', 'cache acquisition deadline expired')

    const inspection = await inspectExactGitObject({ cwd: cachePath, sha: options.sha, runGit, timeoutMs: remaining() })
    const bytes = await fileSystem.measureBytes(cachePath)

    if (bytes > maxBytes)
      throw new ProtocolInspectionError('cache-storage-limit', 'owned cache exceeds its byte budget')

    cache.lastReferencedAt = started + elapsed()
    await fileSystem.writeManifest(cachePath, cache)

    return { ok: true, cache, inspection }
  } catch (error) {
    if (created && cachePath !== null && fileSystem.removePath) {
      try {
        await fileSystem.removePath(cachePath)
      } catch {
        // Preserve the original bounded acquisition failure.
      }
    }

    const code = error instanceof ProtocolInspectionError ? error.code : outputErrorCode(error)

    return {
      ok: false,
      code,
      message: error instanceof Error ? error.message : String(error)
    }
  }
}

export interface TargetResolution {
  id: string
  target: RolloutTarget
  fingerprint: string
  cachePath: string
  createdAt: number
  expiresAt: number
}

const FINGERPRINT_RE = /^[0-9a-f]{64}$/

function validateTargetResolution(value: TargetResolution): TargetResolution {
  if (
    typeof value.id !== 'string' ||
    value.id.length === 0 ||
    value.id.length > 256 ||
    /[\x00\r\n]/.test(value.id) ||
    typeof value.cachePath !== 'string' ||
    value.cachePath.length === 0 ||
    value.cachePath.length > 4096 ||
    /[\x00\r\n]/.test(value.cachePath) ||
    !FINGERPRINT_RE.test(value.fingerprint) ||
    !Number.isSafeInteger(value.createdAt) ||
    !Number.isSafeInteger(value.expiresAt) ||
    value.expiresAt <= value.createdAt
  ) {
    throw new ProtocolInspectionError('protocol-invalid', 'target resolution is malformed')
  }

  return {
    id: value.id,
    target: validateRolloutTarget(value.target),
    fingerprint: value.fingerprint,
    cachePath: value.cachePath,
    createdAt: value.createdAt,
    expiresAt: value.expiresAt
  }
}

export { validateTargetResolution }

export class TargetResolutionStore {
  private readonly values = new Map<string, TargetResolution>()

  add(resolution: TargetResolution): void {
    const validated = validateTargetResolution(resolution)
    this.values.set(validated.id, { ...validated, target: { ...validated.target } })
  }

  get(id: string, now = Date.now()): TargetResolution | null {
    const value = this.values.get(id)

    if (!value || now >= value.expiresAt) {
      if (value) this.values.delete(id)
      return null
    }

    return { ...value, target: { ...value.target } }
  }

  delete(id: string): void {
    this.values.delete(id)
  }
}

export function makeTargetResolution(input: {
  id: string
  target: RolloutTarget
  inspection: ExactGitObjectInspection
  cachePath: string
  createdAt: number
  expiresAt: number
}): TargetResolution {
  const target = validateRolloutTarget(input.target)

  if (
    input.inspection.sha !== target.sha ||
    input.inspection.protocol !== target.protocol ||
    !FINGERPRINT_RE.test(input.inspection.protocolDigest)
  ) {
    throw new ProtocolInspectionError('protocol-unsupported', 'target inspection does not match requested exact target')
  }

  return validateTargetResolution({
    id: input.id,
    target,
    fingerprint: crypto
      .createHash('sha256')
      .update(JSON.stringify([1, target, input.inspection.protocolDigest]))
      .digest('hex'),
    cachePath: input.cachePath,
    createdAt: input.createdAt,
    expiresAt: input.expiresAt
  })
}

function clonePlan(value: RolloutPlan): RolloutPlan {
  const plan = validateRolloutPlan(value)
  const canonicalGithubId = /^github\.com\/[a-z0-9._-]+\/[a-z0-9._-]+$/

  if (!canonicalGithubId.test(plan.target.repositoryId)) {
    try {
      canonicalRepositoryId(plan.target.repositoryId)
    } catch {
      throw new ProtocolInspectionError('repository-access-denied', 'target repository identity is not canonical')
    }
  }

  return JSON.parse(JSON.stringify(plan)) as RolloutPlan
}

export function canonicalPlanTuple(value: RolloutPlan): readonly unknown[] {
  const plan = clonePlan(value)

  return [
    1,
    [plan.target.repositoryId, plan.target.branch, plan.target.sha, plan.target.protocol],
    plan.waves.map(wave => wave.slice()),
    plan.concurrency,
    plan.promotionPolicy,
    plan.rows.map(row => [
      row.installId,
      row.connectionId,
      row.installationFingerprint,
      row.sourceFingerprint,
      row.admittedHead,
      row.requiredScopeIds === null ? null : row.requiredScopeIds.slice().sort(),
      row.eligible,
      row.reviewedSource ?? null
    ]),
    plan.retryOf,
    plan.exclusions.slice().sort()
  ]
}

export function canonicalPlanDigest(value: RolloutPlan): string {
  return crypto
    .createHash('sha256')
    .update(JSON.stringify(canonicalPlanTuple(value)), 'utf8')
    .digest('hex')
}

function rowById(plan: RolloutPlan): Map<string, RolloutPlanRow> {
  return new Map(plan.rows.map(row => [row.installId, row]))
}

function membership(plan: RolloutPlan, installId: string): string | null {
  for (let wave = 0; wave < plan.waves.length; wave += 1) {
    const index = plan.waves[wave].indexOf(installId)
    if (index >= 0) return JSON.stringify({ wave, index })
  }

  return null
}

function diffField(installId: string, field: PlanChange['field'], before: unknown, after: unknown): PlanChange | null {
  const beforeValue =
    before === undefined ? null : before === null ? null : typeof before === 'string' ? before : JSON.stringify(before)
  const afterValue =
    after === undefined ? null : after === null ? null : typeof after === 'string' ? after : JSON.stringify(after)

  return beforeValue === afterValue ? null : { installId, field, before: beforeValue, after: afterValue }
}

const PLAN_CHANGE_FIELDS: readonly PlanChange['field'][] = [
  'membership',
  'identity',
  'source',
  'head',
  'scopes',
  'eligibility'
]

export function diffRolloutPlans(beforeValue: RolloutPlan, afterValue: RolloutPlan): PlanChange[] {
  const before = clonePlan(beforeValue)
  const after = clonePlan(afterValue)
  const beforeRows = rowById(before)
  const afterRows = rowById(after)
  const ids = [...new Set([...beforeRows.keys(), ...afterRows.keys()])].sort()
  const changes: PlanChange[] = []

  for (const installId of ids) {
    const left = beforeRows.get(installId)
    const right = afterRows.get(installId)
    const identityBefore =
      left && right && left.connectionId === right.connectionId
        ? left.installationFingerprint
        : left
          ? JSON.stringify({ connectionId: left.connectionId, installationFingerprint: left.installationFingerprint })
          : undefined
    const identityAfter =
      left && right && left.connectionId === right.connectionId
        ? right.installationFingerprint
        : right
          ? JSON.stringify({ connectionId: right.connectionId, installationFingerprint: right.installationFingerprint })
          : undefined
    const fields: Array<[PlanChange['field'], unknown, unknown]> = [
      ['membership', membership(before, installId), membership(after, installId)],
      ['identity', identityBefore, identityAfter],
      ['source', left ? left.reviewedSource ? [left.sourceFingerprint, left.reviewedSource] : left.sourceFingerprint : undefined,
        right ? right.reviewedSource ? [right.sourceFingerprint, right.reviewedSource] : right.sourceFingerprint : undefined],
      ['head', left?.admittedHead, right?.admittedHead],
      ['scopes', left?.requiredScopeIds, right?.requiredScopeIds],
      ['eligibility', left?.eligible, right?.eligible]
    ]

    for (const field of PLAN_CHANGE_FIELDS) {
      const item = fields.find(candidate => candidate[0] === field)
      if (!item) continue
      const change = diffField(installId, field, item[1], item[2])
      if (change) changes.push(change)
    }
  }

  if (before.target.sha !== after.target.sha) {
    for (const installId of ids) {
      if (!beforeRows.has(installId) && !afterRows.has(installId)) continue
      const existing = changes.find(change => change.installId === installId && change.field === 'head')
      if (existing) continue
      changes.push({ installId, field: 'head', before: before.target.sha, after: after.target.sha })
    }
  }

  return changes.sort(
    (left, right) =>
      left.installId.localeCompare(right.installId) ||
      PLAN_CHANGE_FIELDS.indexOf(left.field) - PLAN_CHANGE_FIELDS.indexOf(right.field)
  )
}

export interface ReviewTokenStoreOptions {
  ttlMs?: number
  tokenFactory?: () => string
}

export interface IssuedReview {
  token: string
  expiresAt: number
  planDigest: string
  canonicalPlan: RolloutPlan
}

export type ReviewRevalidation =
  | { ok: true; token: string; expiresAt: number; planDigest: string; changes: [] }
  | {
      ok: false
      code: 'review-token-invalid' | 'review-token-expired' | 'plan-changed'
      changes: PlanChange[]
      planDigest: string | null
    }

interface ReviewRecord extends IssuedReview {}

export class ReviewTokenStore {
  private readonly records = new Map<string, ReviewRecord>()
  private readonly ttlMs: number
  private readonly tokenFactory: () => string

  constructor(options: ReviewTokenStoreOptions = {}) {
    this.ttlMs = options.ttlMs ?? PREFLIGHT_TOKEN_TTL_MS
    this.tokenFactory = options.tokenFactory || (() => crypto.randomBytes(32).toString('hex'))
    if (!Number.isSafeInteger(this.ttlMs) || this.ttlMs < 1 || this.ttlMs > PREFLIGHT_TOKEN_TTL_MS) {
      throw new Error('review-token-ttl-invalid')
    }
  }

  issue(value: RolloutPlan, now = Date.now()): IssuedReview {
    if (!Number.isSafeInteger(now)) throw new Error('review-token-clock-invalid')
    const canonicalPlan = clonePlan(value)
    const token = this.tokenFactory()

    if (!token || this.records.has(token)) throw new Error('review-token-factory-returned-duplicate')

    const issued: IssuedReview = {
      token,
      expiresAt: now + this.ttlMs,
      planDigest: canonicalPlanDigest(canonicalPlan),
      canonicalPlan
    }

    this.records.set(token, { ...issued, canonicalPlan: clonePlan(canonicalPlan) })

    return { ...issued, canonicalPlan: clonePlan(canonicalPlan) }
  }

  revalidate(token: string, value: RolloutPlan, now = Date.now()): ReviewRevalidation {
    const record = this.records.get(token)

    if (!record) return { ok: false, code: 'review-token-invalid', changes: [], planDigest: null }
    if (!Number.isSafeInteger(now)) return { ok: false, code: 'review-token-invalid', changes: [], planDigest: null }
    if (now >= record.expiresAt) {
      this.records.delete(token)
      return { ok: false, code: 'review-token-expired', changes: [], planDigest: record.planDigest }
    }

    const canonicalPlan = clonePlan(value)
    const planDigest = canonicalPlanDigest(canonicalPlan)

    if (planDigest !== record.planDigest) {
      return {
        ok: false,
        code: 'plan-changed',
        changes: diffRolloutPlans(record.canonicalPlan, canonicalPlan),
        planDigest
      }
    }

    const expiresAt = now + this.ttlMs
    const renewed: ReviewRecord = { ...record, expiresAt }
    this.records.set(token, renewed)

    return { ok: true, token, expiresAt, planDigest, changes: [] }
  }

  revoke(token: string): void {
    this.records.delete(token)
  }
}

export function createPreflightReview(input: {
  plan: RolloutPlan
  resolution: TargetResolution | null
  reviewTokens: ReviewTokenStore
  now?: number
  blockers?: string[]
  verifiedSources?: ReadonlyMap<string, ReviewedSourceBinding>
  verifiedAssurance?: ReadonlyMap<string, VerifiedAssuranceEvidence>
}): {
  token: string | null
  expiresAt: number | null
  planDigest: string | null
  canonicalPlan: RolloutPlan
  changes: PlanChange[]
  blockers: string[]
} {
  const canonicalPlan = clonePlan(input.plan)
  const blockers = [...(input.blockers || [])]
  const now = input.now ?? Date.now()
  let resolution: TargetResolution | null = null

  if (!Number.isSafeInteger(now)) blockers.push('review-clock-invalid')

  if (!input.resolution) blockers.push('target-resolution-unavailable')
  else {
    try {
      resolution = validateTargetResolution(input.resolution)
    } catch {
      blockers.push('target-resolution-invalid')
    }
  }
  if (resolution) {
    if (now >= resolution.expiresAt) blockers.push('target-resolution-expired')
    if (JSON.stringify(resolution.target) !== JSON.stringify(canonicalPlan.target)) blockers.push('target-changed')
  }
  if (canonicalPlan.rows.some(row => !row.eligible)) blockers.push('ineligible-target')
  if (canonicalPlan.rows.some(row => row.requiredScopeIds === null)) blockers.push('scope-evidence-missing')
  for (const row of canonicalPlan.rows) {
    const source = row.reviewedSource
    const verifiedSource = input.verifiedSources?.get(row.installId)
    const assurance = input.verifiedAssurance?.get(row.installId)
    if (!source || !isVerifiedGitSource(verifiedSource) || JSON.stringify(source) !== JSON.stringify(verifiedSource)) {
      blockers.push('reviewed-source-unverified')
      continue
    }
    let canonicalOrigin: string | null = null
    try {
      canonicalOrigin = canonicalRepositoryId(source.originUrl)
    } catch {
      // A credential-bearing or unsupported origin cannot be a reviewed source.
    }
    if (
      source.targetSha !== canonicalPlan.target.sha ||
      source.resolvedRef !== `refs/remotes/origin/${canonicalPlan.target.branch}` ||
      canonicalOrigin !== canonicalPlan.target.repositoryId
    ) blockers.push('reviewed-source-mismatch')
    if (!isVerifiedAssurance(assurance)) {
      blockers.push('assurance-evidence-unverified')
      continue
    }
    if (
      assurance.expiresAt <= now || assurance.profile !== source.assuranceProfile ||
      assurance.evidenceSha256 !== source.assuranceEvidenceSha256 ||
      assurance.generation !== source.assuranceGeneration ||
      assurance.repositoryId !== canonicalPlan.target.repositoryId ||
      assurance.targetSha !== canonicalPlan.target.sha ||
      assurance.sourceFingerprint !== row.sourceFingerprint
    ) blockers.push('assurance-evidence-stale-or-mismatched')
  }

  if (blockers.length) {
    return {
      token: null,
      expiresAt: null,
      planDigest: null,
      canonicalPlan,
      changes: [],
      blockers: [...new Set(blockers)]
    }
  }

  const issued = input.reviewTokens.issue(canonicalPlan, now)

  return {
    token: issued.token,
    expiresAt: issued.expiresAt,
    planDigest: issued.planDigest,
    canonicalPlan: issued.canonicalPlan,
    changes: [],
    blockers: []
  }
}

export function revalidateOpaqueReviewToken(
  reviewTokens: ReviewTokenStore,
  token: string,
  plan: RolloutPlan,
  now = Date.now()
): ReviewRevalidation {
  return reviewTokens.revalidate(token, plan, now)
}

export function sourceIsSupported(origin: string): {
  supported: boolean
  repositoryId: string | null
  reason: string | null
} {
  try {
    return { supported: true, repositoryId: canonicalRepositoryId(origin), reason: null }
  } catch (error) {
    return { supported: false, repositoryId: null, reason: error instanceof Error ? error.message : String(error) }
  }
}

export { CACHE_RETENTION_MS as CACHE_TTL_MS }
