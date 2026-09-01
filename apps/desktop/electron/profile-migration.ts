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
  /** Default profile home (`~/.hermes`). Default's state.db and gateway.pid live here. */
  hermesHome: string
  /** Named-profile root (`~/.hermes/profiles`). Does not contain `default`. */
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
  profile: string | null
  /** True when chosen from the state.db heuristic (auto-detected), undefined when explicit. */
  _migrated?: boolean
}

/**
 * Production layout: default IS `hermesHome`; named profiles are children of
 * `profilesRoot`. There is no `profiles/default` directory on a normal install.
 */
export function profileStateDbPath(name: string, hermesHome: string, profilesRoot: string): string {
  return name === 'default' ? `${hermesHome}/state.db` : `${profilesRoot}/${name}/state.db`
}

export function profileGatewayPidPath(name: string, hermesHome: string, profilesRoot: string): string {
  return name === 'default' ? `${hermesHome}/gateway.pid` : `${profilesRoot}/${name}/gateway.pid`
}

function resolveHermesHome(profilesRoot: string, hermesHome?: string): string {
  if (hermesHome) {
    return hermesHome
  }

  // Tests that predate hermesHome pass only profilesRoot.
  for (const suffix of ['/profiles', '\\profiles']) {
    if (profilesRoot.endsWith(suffix)) {
      return profilesRoot.slice(0, -suffix.length)
    }
  }

  return profilesRoot
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

  if (!name) {
    return null
  }

  if (name === 'default') {
    return undefined
  }

  return isValid(name) ? name : undefined
}

/**
 * Return the profile names whose gateway.pid file points to a live hermes process.
 * Tolerates missing/malformed pid files and stale-but-recycled PIDs (the latter is
 * the whole reason we check both liveness AND cmdline identity).
 *
 * `hermesHome` is optional so existing call sites that only pass `profilesRoot`
 * still work: it is derived as the parent of `…/profiles`.
 */
export function findRunningGatewayProfiles(
  profilesRoot: string,
  allProfiles: string[],
  deps: Pick<MigrationDeps, 'existsSync' | 'readFileSync' | 'isHermesProcess'> & { hermesHome?: string }
): string[] {
  const hermesHome = resolveHermesHome(profilesRoot, deps.hermesHome)
  const running: string[] = []

  for (const name of allProfiles) {
    const pidFile = profileGatewayPidPath(name, hermesHome, profilesRoot)

    if (!deps.existsSync(pidFile)) {
      continue
    }

    let parsed: { pid?: unknown } | null = null

    try {
      parsed = JSON.parse(deps.readFileSync(pidFile, 'utf8'))
    } catch {
      continue
    }

    const pid = Number(parsed?.pid)

    if (!Number.isInteger(pid) || pid < 1) {
      continue
    }

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
export function scoreStateDb(dbPath: string, now: number, stat: MigrationDeps['statSync']): number | null {
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
  if (legacyActive) {
    return { profile: legacyActive }
  }

  if (running.length === 1) {
    return { profile: running[0] }
  }

  let best: string | null = null
  let maxScore = -Infinity

  for (const name of candidates) {
    const s = score(profileStateDbPath(name, deps.hermesHome, deps.profilesRoot))

    if (s == null) {
      continue
    }

    if (s > maxScore) {
      maxScore = s
      best = name
    }
  }

  if (!best || best === 'default') {
    return null
  }

  return { profile: best, _migrated: true }
}

/**
 * List named profile directory names under `profilesRoot`. A directory named
 * `default` is accepted if present (unusual) but production default is not a
 * child of this folder — see `withDefaultCandidate`.
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

/** Default is always a candidate; it is `$HERMES_HOME`, not `$HERMES_HOME/profiles/default`. */
export function withDefaultCandidate(named: string[]): string[] {
  return ['default', ...named.filter(name => name !== 'default')]
}

/**
 * Read an existing active-profile.json. Returns null when missing/malformed.
 * `_migrated: true` means the first-boot heuristic wrote it (safe to re-score).
 * Absence of that flag is a user/CLI choice and must not be overwritten.
 */
export function readExistingPreference(
  desktopProfileConfigPath: string,
  readFile: MigrationDeps['readFileSync']
): { profile: string | null; migrated: boolean } | null {
  let parsed: unknown

  try {
    parsed = JSON.parse(readFile(desktopProfileConfigPath, 'utf8'))
  } catch {
    return null
  }

  if (!parsed || typeof parsed !== 'object') {
    return null
  }

  const rec = parsed as { profile?: unknown; _migrated?: unknown }
  const raw = typeof rec.profile === 'string' ? rec.profile.trim() : ''

  return {
    profile: raw || null,
    migrated: rec._migrated === true
  }
}

/**
 * First-boot seed, plus repair of heuristic-owned files (`_migrated: true`).
 * User-selected files (no `_migrated`) are never overwritten. When a repaired
 * heuristic would now pick default, write `{ profile: null }` so Desktop drops
 * `--profile` instead of pinning `default`.
 */
export function migrateActiveProfileIfMissing(desktopProfileConfigPath: string, deps: MigrationDeps): boolean {
  const existing = deps.existsSync(desktopProfileConfigPath)
    ? readExistingPreference(desktopProfileConfigPath, deps.readFileSync)
    : null

  if (existing && !existing.migrated) {
    return false
  }

  const legacyActive = readLegacyActiveProfile(deps.legacyActivePath, deps.readFileSync, deps.isValidProfileName)

  const allProfiles = withDefaultCandidate(listProfileDirs(deps))
  const running = findRunningGatewayProfiles(deps.profilesRoot, allProfiles, deps)
  const candidates = running.length > 1 ? running : allProfiles

  const decision = decideMigration(legacyActive, running, candidates, deps, dbPath =>
    scoreStateDb(dbPath, deps.now(), deps.statSync)
  )

  // Same as the heuristic rung: pinning `default` into active-profile.json
  // launches `hermes --profile default` and is worse than writing nothing
  // (legacy sticky / implicit default). Covers a lone default gateway.pid.
  if (!decision || decision.profile === 'default') {
    if (existing?.migrated) {
      deps.writeJson(desktopProfileConfigPath, { profile: null })
      return true
    }

    return false
  }

  if (existing?.migrated && existing.profile === decision.profile) {
    return false
  }

  deps.writeJson(desktopProfileConfigPath, decision)

  return true
}
