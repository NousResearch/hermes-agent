// Pure migration-decision helpers for active-profile.json first-boot seeding.
// Extracted from main.ts so they're unit-testable without Electron.
//
// The helpers here are deliberately side-effect-free except for the orchestrator
// (`migrateActiveProfileIfMissing`), which only writes the preference file when
// every fs/path concern is funnelled through the injected `MigrationDeps` bag.
// Tests inject synthetic fs maps; production wires real fs/path.

import type { Dirent } from 'node:fs'

// Floor for the size dimension of the hybrid score. A 100-byte file (effectively
// empty) and a 1 KiB file score the same on the size axis — both are too small to
// be the user's primary workspace on their own.
export const PROFILE_SCORE_MIN_SIZE_BYTES = 1024

export interface MigrationDeps {
  legacyActivePath: string
  profilesRoot: string
  existsSync: (path: string) => boolean
  readFileSync: (path: string, encoding: 'utf8') => string
  statSync: (path: string) => { size: number; mtimeMs: number }
  readdirSync: (path: string, options?: { withFileTypes?: boolean }) => Dirent[]
  isHermesProcess: (pid: number) => boolean
  now: () => number
  writeJson: (path: string, payload: MigrationDecision) => void
  isValidProfileName: (name: string) => boolean
}

export interface MigrationDecision {
  profile: string
  /** True when chosen from the state.db heuristic (auto-detected), undefined when explicit. */
  _migrated?: boolean
}

/**
 * Parse the legacy CLI-sticky file. Returns the trimmed name on success, null when
 * missing/unreadable/empty, undefined when present but invalid (so the caller can
 * distinguish "explicitly chose default" from "no legacy file at all"). The
 * `'default'` profile is always rejected here — it's an implicit fallback, never a
 * user-chosen CLI value, and accepting it would suppress the heuristic rung that
 * is the whole point of this migration.
 */
export function readLegacyActiveProfile(
  legacyActivePath: string,
  readFile: MigrationDeps['readFileSync'],
  isValid: MigrationDeps['isValidProfileName']
): string | null | undefined {
  let raw: string
  try {
    raw = readFile(legacyActivePath, 'utf8')
  } catch {
    return null
  }
  const name = raw.trim()
  if (!name) return null
  if (name === 'default') return undefined
  return isValid(name) ? name : undefined
}

/**
 * Return the profile names whose gateway.pid file points to a live hermes process.
 * Tolerates missing/malformed pid files and stale-but-recycled PIDs (the latter is
 * the whole reason we check both liveness AND cmdline identity).
 */
export function findRunningGatewayProfiles(
  profilesRoot: string,
  allProfiles: string[],
  deps: Pick<MigrationDeps, 'existsSync' | 'readFileSync' | 'isHermesProcess'>
): string[] {
  const running: string[] = []
  for (const name of allProfiles) {
    const pidFile = `${profilesRoot}/${name}/gateway.pid`
    if (!deps.existsSync(pidFile)) continue
    let parsed: { pid?: unknown } | null = null
    try {
      parsed = JSON.parse(deps.readFileSync(pidFile, 'utf8'))
    } catch {
      continue
    }
    const pid = Number(parsed?.pid)
    if (!Number.isInteger(pid) || pid < 1) continue
    if (deps.isHermesProcess(pid)) {
      running.push(name)
    }
  }
  return running
}

/**
 * Hybrid recency × size score for a state.db file. Returns null when the file is
 * missing. The formula picks the primary workspace across profiles whose databases
 * have been touched at similar times — a 409 MB DB beats a 28 MB one even with a
 * slightly newer mtime. Floors the recency weight at 0.1 so a profile touched
 * years ago but never deleted still scores > 0 if its DB is large.
 */
export function scoreStateDb(
  dbPath: string,
  now: number,
  stat: MigrationDeps['statSync']
): number | null {
  let s: { size: number; mtimeMs: number }
  try {
    s = stat(dbPath)
  } catch {
    return null
  }
  const daysSinceModified = Math.max(0, (now - s.mtimeMs) / (1000 * 60 * 60 * 24))
  const recencyWeight = Math.max(0.1, 30 - daysSinceModified)
  const sizeWeight = Math.log10(Math.max(PROFILE_SCORE_MIN_SIZE_BYTES, s.size))
  return recencyWeight * sizeWeight
}

/**
 * Pure decision logic. Returns null when nothing migratable was found.
 * - legacy non-null → always prefer (no _migrated flag, user-chosen)
 * - running.length === 1 → prefer that profile (no flag, gateway-owned)
 * - state.db heuristic → set _migrated=true (auto-detected)
 * - best === 'default' → suppress write (single-profile fallback)
 */
export function decideMigration(
  legacyActive: string | null | undefined,
  running: string[],
  candidates: string[],
  deps: MigrationDeps,
  score: (dbPath: string) => number | null
): MigrationDecision | null {
  if (legacyActive) return { profile: legacyActive }
  if (running.length === 1) return { profile: running[0] }

  let best: string | null = null
  let maxScore = -Infinity
  for (const name of candidates) {
    const s = score(`${deps.profilesRoot}/${name}/state.db`)
    if (s == null) continue
    if (s > maxScore) {
      maxScore = s
      best = name
    }
  }
  if (!best || best === 'default') return null
  return { profile: best, _migrated: true }
}

/**
 * List known profile directory names under `profilesRoot`. Accepts `default` and
 * any name passing the injected validator. Returns [] on missing dir or empty.
 */
export function listProfileDirs(deps: MigrationDeps): string[] {
  let entries: Dirent[]
  try {
    entries = deps.readdirSync(deps.profilesRoot, { withFileTypes: true })
  } catch {
    return []
  }
  return entries
    .filter(e => e.isDirectory() && (e.name === 'default' || deps.isValidProfileName(e.name)))
    .map(e => e.name)
}

/**
 * Orchestrator. Idempotent: writes at most once when the preference file is
 * missing. Thin on top of the decision helpers above; the testable surface is
 * `decideMigration` + the individual rung helpers, this function just glues them
 * to the deps bag.
 */
export function migrateActiveProfileIfMissing(
  desktopProfileConfigPath: string,
  deps: MigrationDeps
): boolean {
  if (deps.existsSync(desktopProfileConfigPath)) return false

  const legacyActive = readLegacyActiveProfile(
    deps.legacyActivePath,
    deps.readFileSync,
    deps.isValidProfileName
  )

  const allProfiles = listProfileDirs(deps)
  if (allProfiles.length === 0) return false

  const running = findRunningGatewayProfiles(deps.profilesRoot, allProfiles, deps)
  const candidates = running.length > 1 ? running : allProfiles
  const decision = decideMigration(
    legacyActive,
    running,
    candidates,
    deps,
    dbPath => scoreStateDb(dbPath, deps.now(), deps.statSync)
  )
  if (!decision) return false

  deps.writeJson(desktopProfileConfigPath, decision)
  return true
}