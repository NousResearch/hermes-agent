import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

const STATE_SCHEMA_VERSION = 1
const SAFE_VERSION = /^[0-9A-Za-z](?:[0-9A-Za-z.+-]{0,63})$/
const SHA512_HEX = /^[a-f0-9]{128}$/

interface KnownGoodVersion {
  sha512: string
  version: string
}

interface CandidateVersion {
  attempts: number
  cleanExit?: boolean
  startedAt: number
  version: string
}

type RollbackReason = 'attempt-limit' | 'rejected-version' | 'startup-timeout'

interface RollbackRecord {
  failedVersion: string
  handoffs: number
  reason: RollbackReason
  startedAt: number
  targetVersion: string
}

interface RejectedVersion {
  reason: RollbackReason
  rejectedAt: number
  version: string
}

export interface UpdateHealthState {
  candidate?: CandidateVersion
  knownGood?: KnownGoodVersion
  rejected?: RejectedVersion
  rollback?: RollbackRecord
  schemaVersion: 1
}

export interface RollbackDecision {
  action: 'rollback'
  failedVersion: string
  installerPath: string
  reason: RollbackReason
  targetVersion: string
}

export interface RollbackExhaustedDecision {
  action: 'rollback-exhausted'
  failedVersion: string
  handoffs: number
  targetVersion: string
}

export type UpdateHealthDecision =
  | { action: 'await-health'; attempts: number; version: string }
  | { action: 'disabled' }
  | { action: 'known-good'; version: string }
  | RollbackDecision
  | RollbackExhaustedDecision

interface UpdateHealthManagerOptions {
  cacheDir: string
  currentVersion: string
  enabled: boolean
  hashFile?: (filePath: string) => Promise<string>
  maxAttempts?: number
  maxRollbackHandoffs?: number
  now?: () => number
}

function isSafeVersion(value: unknown): value is string {
  return typeof value === 'string' && SAFE_VERSION.test(value)
}

function isRollbackReason(value: unknown): value is RollbackReason {
  return value === 'attempt-limit' || value === 'rejected-version' || value === 'startup-timeout'
}

function emptyState(): UpdateHealthState {
  return { schemaVersion: STATE_SCHEMA_VERSION }
}

function normalizeState(value: unknown): UpdateHealthState {
  if (
    !value ||
    typeof value !== 'object' ||
    (value as { schemaVersion?: unknown }).schemaVersion !== STATE_SCHEMA_VERSION
  ) {
    return emptyState()
  }

  const raw = value as {
    candidate?: Partial<CandidateVersion>
    knownGood?: Partial<KnownGoodVersion>
    rejected?: Partial<RejectedVersion>
    rollback?: Partial<RollbackRecord>
  }

  const state = emptyState()

  if (raw.knownGood && isSafeVersion(raw.knownGood.version) && typeof raw.knownGood.sha512 === 'string') {
    const sha512 = raw.knownGood.sha512.toLowerCase()

    if (SHA512_HEX.test(sha512)) {
      state.knownGood = { sha512, version: raw.knownGood.version }
    }
  }

  if (
    raw.candidate &&
    isSafeVersion(raw.candidate.version) &&
    Number.isInteger(raw.candidate.attempts) &&
    Number(raw.candidate.attempts) >= 0 &&
    typeof raw.candidate.startedAt === 'number' &&
    Number.isFinite(raw.candidate.startedAt)
  ) {
    state.candidate = {
      attempts: Number(raw.candidate.attempts),
      cleanExit: raw.candidate.cleanExit === true || undefined,
      startedAt: raw.candidate.startedAt,
      version: raw.candidate.version
    }
  }

  if (
    raw.rejected &&
    isSafeVersion(raw.rejected.version) &&
    isRollbackReason(raw.rejected.reason) &&
    typeof raw.rejected.rejectedAt === 'number' &&
    Number.isFinite(raw.rejected.rejectedAt)
  ) {
    state.rejected = {
      reason: raw.rejected.reason,
      rejectedAt: raw.rejected.rejectedAt,
      version: raw.rejected.version
    }
  }

  if (
    raw.rollback &&
    isSafeVersion(raw.rollback.failedVersion) &&
    isSafeVersion(raw.rollback.targetVersion) &&
    typeof raw.rollback.startedAt === 'number' &&
    Number.isFinite(raw.rollback.startedAt)
  ) {
    state.rollback = {
      failedVersion: raw.rollback.failedVersion,
      handoffs:
        Number.isInteger(raw.rollback.handoffs) && Number(raw.rollback.handoffs) >= 1
          ? Number(raw.rollback.handoffs)
          : 1,
      reason: isRollbackReason(raw.rollback.reason) ? raw.rollback.reason : 'attempt-limit',
      startedAt: raw.rollback.startedAt,
      targetVersion: raw.rollback.targetVersion
    }
  }

  return state
}

async function hashFileContents(filePath: string): Promise<string> {
  return await new Promise((resolve, reject) => {
    const hash = crypto.createHash('sha512')
    const stream = fs.createReadStream(filePath)

    stream.on('data', chunk => hash.update(chunk))
    stream.on('error', reject)
    stream.on('end', () => resolve(hash.digest('hex')))
  })
}

function hashesMatch(left: string, right: string): boolean {
  if (!SHA512_HEX.test(left) || !SHA512_HEX.test(right)) {
    return false
  }

  return crypto.timingSafeEqual(Buffer.from(left, 'hex'), Buffer.from(right, 'hex'))
}

export function createUpdateHealthManager({
  cacheDir,
  currentVersion,
  enabled,
  hashFile: hashInstaller = hashFileContents,
  maxAttempts = 2,
  maxRollbackHandoffs = 1,
  now = Date.now
}: UpdateHealthManagerOptions) {
  if (!isSafeVersion(currentVersion)) {
    throw new Error(`Unsafe application version: ${JSON.stringify(currentVersion)}`)
  }

  if (!Number.isInteger(maxAttempts) || maxAttempts < 1) {
    throw new Error('maxAttempts must be a positive integer.')
  }

  if (!Number.isInteger(maxRollbackHandoffs) || maxRollbackHandoffs < 1) {
    throw new Error('maxRollbackHandoffs must be a positive integer.')
  }

  const installersDir = path.resolve(cacheDir, 'installers')
  const statePath = path.resolve(cacheDir, 'health.json')
  let state = readState()
  let transition: 'blocked' | 'confirming' | 'idle' | 'rollback' | 'settled' = 'idle'

  function installerPath(version: string): string {
    if (!isSafeVersion(version)) {
      throw new Error(`Unsafe installer version: ${JSON.stringify(version)}`)
    }

    return path.join(installersDir, `${version}.exe`)
  }

  function readState(): UpdateHealthState {
    if (!enabled) {
      return emptyState()
    }

    try {
      return normalizeState(JSON.parse(fs.readFileSync(statePath, 'utf8')))
    } catch {
      return emptyState()
    }
  }

  function writeState(next: UpdateHealthState) {
    state = normalizeState(next)

    if (!enabled) {
      return
    }

    fs.mkdirSync(cacheDir, { recursive: true })
    const temporaryPath = `${statePath}.${process.pid}.${Date.now()}.tmp`

    try {
      fs.writeFileSync(temporaryPath, `${JSON.stringify(state, null, 2)}\n`, { encoding: 'utf8', mode: 0o600 })
      fs.renameSync(temporaryPath, statePath)
    } finally {
      try {
        fs.rmSync(temporaryPath, { force: true })
      } catch {
        // Best-effort cleanup after an interrupted atomic write.
      }
    }
  }

  function rollbackDecision(reason: RollbackDecision['reason']): RollbackDecision | null {
    if (!state.knownGood || state.knownGood.version === currentVersion) {
      return null
    }

    return {
      action: 'rollback',
      failedVersion: currentVersion,
      installerPath: installerPath(state.knownGood.version),
      reason,
      targetVersion: state.knownGood.version
    }
  }

  return {
    beginStartup(): UpdateHealthDecision {
      if (!enabled) {
        return { action: 'disabled' }
      }

      state = readState()

      if (state.knownGood?.version === currentVersion) {
        const rejected =
          state.rollback?.targetVersion === currentVersion
            ? {
                reason: state.rollback.reason,
                rejectedAt: now(),
                version: state.rollback.failedVersion
              }
            : state.rejected

        if (state.candidate || state.rollback) {
          writeState({ knownGood: state.knownGood, rejected, schemaVersion: STATE_SCHEMA_VERSION })
        }

        return { action: 'known-good', version: currentVersion }
      }

      if (
        state.rollback?.failedVersion === currentVersion &&
        state.rollback.targetVersion === state.knownGood?.version
      ) {
        if (state.rollback.handoffs >= maxRollbackHandoffs) {
          transition = 'blocked'

          return {
            action: 'rollback-exhausted',
            failedVersion: currentVersion,
            handoffs: state.rollback.handoffs,
            targetVersion: state.rollback.targetVersion
          }
        }

        const rollback = rollbackDecision(state.rollback.reason)

        if (rollback) {
          return rollback
        }
      }

      if (state.rejected?.version === currentVersion) {
        const rollback = rollbackDecision('rejected-version')

        if (rollback) {
          return rollback
        }
      }

      const previousCandidate = state.candidate?.version === currentVersion ? state.candidate : undefined
      const attempts = previousCandidate?.cleanExit ? 1 : (previousCandidate?.attempts ?? 0) + 1

      writeState({
        ...state,
        candidate: {
          attempts,
          startedAt: now(),
          version: currentVersion
        },
        rollback: undefined,
        schemaVersion: STATE_SCHEMA_VERSION
      })

      if (attempts >= maxAttempts) {
        const rollback = rollbackDecision('attempt-limit')

        if (rollback) {
          return rollback
        }
      }

      return { action: 'await-health', attempts, version: currentVersion }
    },

    async confirmHealthy(): Promise<{ error?: string; ok: boolean; promoted?: boolean }> {
      if (transition === 'settled') {
        return { ok: true, promoted: false }
      }

      if (transition === 'rollback') {
        return { error: 'Startup rollback is already in progress.', ok: false }
      }

      if (transition === 'blocked') {
        return { error: 'Automatic startup rollback handoff limit is exhausted.', ok: false }
      }

      if (transition === 'confirming') {
        return { error: 'Startup health confirmation is already in progress.', ok: false }
      }

      if (!enabled) {
        transition = 'settled'

        return { ok: true, promoted: false }
      }

      transition = 'confirming'
      state = readState()

      if (state.knownGood?.version === currentVersion && !state.candidate) {
        transition = 'settled'

        return { ok: true, promoted: false }
      }

      const currentInstallerPath = installerPath(currentVersion)

      if (!fs.existsSync(currentInstallerPath)) {
        transition = 'idle'

        return { error: 'Retained installer for the current version is missing.', ok: false }
      }

      let sha512: string

      try {
        sha512 = await hashInstaller(currentInstallerPath)
      } catch {
        transition = 'idle'

        return { error: 'Could not hash the retained installer for the current version.', ok: false }
      }

      const previousKnownGood = state.knownGood?.version
      writeState({
        knownGood: { sha512, version: currentVersion },
        schemaVersion: STATE_SCHEMA_VERSION
      })
      transition = 'settled'

      if (previousKnownGood && previousKnownGood !== currentVersion) {
        try {
          fs.rmSync(installerPath(previousKnownGood), { force: true })
        } catch {
          // The new version is already committed as healthy; stale-cache cleanup is best effort.
        }
      }

      return { ok: true, promoted: true }
    },

    getInstallerPath: installerPath,
    getState: () => state,
    isAwaitingHealth() {
      state = readState()

      return transition === 'idle' && state.candidate?.version === currentVersion
    },
    isVersionRejected(version: string) {
      state = readState()

      return isSafeVersion(version) && state.rejected?.version === version
    },

    async prepareRollbackInstaller(
      decision: UpdateHealthDecision
    ): Promise<{ error?: string; launchPath?: string; ok: boolean }> {
      if (decision.action !== 'rollback') {
        return { error: 'No rollback was requested.', ok: false }
      }

      state = readState()

      if (!state.knownGood || state.knownGood.version !== decision.targetVersion) {
        return { error: 'Known-good rollback metadata is unavailable.', ok: false }
      }

      fs.mkdirSync(cacheDir, { recursive: true })
      const launchPath = path.join(cacheDir, `rollback-runner-${process.pid}-${crypto.randomUUID()}.exe`)

      try {
        fs.copyFileSync(decision.installerPath, launchPath, fs.constants.COPYFILE_EXCL)
        const copiedHash = await hashInstaller(launchPath)

        if (!hashesMatch(copiedHash, state.knownGood.sha512)) {
          fs.rmSync(launchPath, { force: true })

          return { error: 'Known-good installer integrity verification failed.', ok: false }
        }
      } catch {
        try {
          fs.rmSync(launchPath, { force: true })
        } catch {
          // Best-effort cleanup of an incomplete launch copy.
        }

        return { error: 'Known-good installer is missing or unreadable.', ok: false }
      }

      return { launchPath, ok: true }
    },

    recordCleanExit() {
      state = readState()

      if (state.candidate?.version !== currentVersion) {
        return
      }

      writeState({
        ...state,
        candidate: {
          ...state.candidate,
          attempts: 0,
          cleanExit: true
        },
        schemaVersion: STATE_SCHEMA_VERSION
      })
    },

    recordRollbackStarted(decision: UpdateHealthDecision) {
      if (decision.action !== 'rollback') {
        return
      }

      state = readState()

      const previousHandoffs =
        state.rollback?.failedVersion === decision.failedVersion &&
        state.rollback.targetVersion === decision.targetVersion
          ? state.rollback.handoffs
          : 0

      writeState({
        ...state,
        rollback: {
          failedVersion: decision.failedVersion,
          handoffs: previousHandoffs + 1,
          reason: decision.reason,
          startedAt: now(),
          targetVersion: decision.targetVersion
        },
        schemaVersion: STATE_SCHEMA_VERSION
      })
    },

    timeoutDecision(): RollbackDecision | null {
      if (transition !== 'idle') {
        return null
      }

      state = readState()

      if (state.candidate?.version !== currentVersion) {
        return null
      }

      const decision = rollbackDecision('startup-timeout')

      if (decision) {
        transition = 'rollback'
      }

      return decision
    },

    async verifyRollbackInstaller(decision: UpdateHealthDecision): Promise<{ error?: string; ok: boolean }> {
      if (decision.action !== 'rollback') {
        return { error: 'No rollback was requested.', ok: false }
      }

      state = readState()

      if (!state.knownGood || state.knownGood.version !== decision.targetVersion) {
        return { error: 'Known-good rollback metadata is unavailable.', ok: false }
      }

      let actualHash: string

      try {
        actualHash = await hashInstaller(decision.installerPath)
      } catch {
        return { error: 'Known-good installer is missing or unreadable.', ok: false }
      }

      if (!hashesMatch(actualHash, state.knownGood.sha512)) {
        return { error: 'Known-good installer integrity verification failed.', ok: false }
      }

      return { ok: true }
    }
  }
}
