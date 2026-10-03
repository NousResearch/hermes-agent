/**
 * installs-cli.ts
 *
 * Pure, electron-free helpers for the desktop `hermes installs` UI. These
 * build the argv for the `hermes_cli.main` installs subcommands, validate the
 * install id against a strict pattern, and parse + validate the JSON that
 * `hermes installs list --json` prints.
 *
 * Kept electron-free so Vitest can exercise the IPC handlers and the boot
 * notice decision. main.ts supplies the spawn and the renderer send.
 *
 * The CLI contract (hermes_cli/installs.py):
 *   - `installs list --json` prints one JSON object:
 *     { current, installs[], launchers[], notice: {count, dismissed, partial} }
 *   - `installs remove <id> --yes` exits 0 when it removed, 1 when refused.
 *   - `installs dismiss` hides the notice until the other installs change.
 *
 * Nothing here builds a shell string: the callers pass argv arrays to spawn.
 */

export const INSTALL_ID_PATTERN = /^[0-9a-f]{1,12}$/

const REMOVE_COMMAND = 'remove'

export interface HermesInstallEntry {
  id: string
  root: string
  steward: string
  version: null | string
  sources: string[]
  current: boolean
  package_full_name: null | string
  removable: boolean
  action: string
  refusal: null | string
}

export interface HermesLauncherEntry {
  path: string
  owner: null | string
}

export interface InstallsNoticeState {
  count: number
  dismissed: boolean
  /** The count can be low: on Windows the launch path leaves Store packages out. */
  partial: boolean
}

export interface InstallsListResult {
  current: string
  installs: HermesInstallEntry[]
  launchers: HermesLauncherEntry[]
  notice: InstallsNoticeState
}

/** The argv after the python executable, run as `python -m hermes_cli.main ...`. */
export type InstallsArgs = string[]

/** argv for the full JSON list. */
export function listArgs(): InstallsArgs {
  return ['installs', 'list', '--json']
}

/** argv for a confirmed removal. Returns null on a bad id (never a shell string). */
export function removeArgs(rawId: unknown): null | InstallsArgs {
  const id = validateInstallId(rawId)

  if (!id) {
    return null
  }

  return ['installs', REMOVE_COMMAND, id, '--yes']
}

/** argv that hides the boot notice until the set of other installs changes. */
export function dismissArgs(): InstallsArgs {
  return ['installs', 'dismiss']
}

/** How long the app waits for one command. The CLI gives `Remove-AppxPackage` 300 s, so `remove` waits longer. */
export function runTimeoutMs(args: InstallsArgs): number {
  return args.includes(REMOVE_COMMAND) ? 330_000 : 30_000
}

/**
 * Check an install id from the renderer before it reaches argv. The CLI uses
 * short hex prefixes, so anything else is a typo or an injection attempt.
 * Returns the trimmed id, or null when invalid.
 */
export function validateInstallId(rawId: unknown): null | string {
  if (typeof rawId !== 'string') {
    return null
  }

  const id = rawId.trim()

  return INSTALL_ID_PATTERN.test(id) ? id : null
}

/**
 * Parse the JSON object of `hermes installs list --json` output, skipping any
 * log lines before it, and validate the shape field by field. Returns null on malformed or partial
 * output so a half-migrated CLI never feeds the Settings table broken rows.
 */
export function parseInstallsList(stdout: string): null | InstallsListResult {
  const lines = String(stdout)
    .split('\n')
    .map(entry => entry.trim())

  // The CLI prints with indent=2, so the JSON spans many lines. Find the first
  // line that opens an object and take everything from there.
  const start = lines.findIndex(line => line.startsWith('{'))

  if (start < 0) {
    return null
  }

  const candidate = lines.slice(start).join('\n')

  if (!candidate) {
    return null
  }

  let raw: unknown

  try {
    raw = JSON.parse(candidate)
  } catch {
    return null
  }

  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return null
  }

  const data = raw as Record<string, unknown>

  if (typeof data.current !== 'string') {
    return null
  }

  const installs = parseInstallEntries(data.installs)

  if (!installs) {
    return null
  }

  const launchers = parseLauncherEntries(data.launchers)

  if (!launchers) {
    return null
  }

  const notice = parseNotice(data.notice)

  if (!notice) {
    return null
  }

  return { current: data.current, installs, launchers, notice }
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every(entry => typeof entry === 'string')
}

function isNullString(value: unknown): value is null | string {
  return value === null || typeof value === 'string'
}

function parseInstallEntries(value: unknown): null | HermesInstallEntry[] {
  if (!Array.isArray(value)) {
    return null
  }

  const out: HermesInstallEntry[] = []

  for (const row of value) {
    if (!row || typeof row !== 'object' || Array.isArray(row)) {
      return null
    }

    const entry = row as Record<string, unknown>

    if (
      typeof entry.id !== 'string' ||
      typeof entry.root !== 'string' ||
      typeof entry.steward !== 'string' ||
      !isNullString(entry.version) ||
      !isStringArray(entry.sources) ||
      typeof entry.current !== 'boolean' ||
      !isNullString(entry.package_full_name) ||
      typeof entry.removable !== 'boolean' ||
      typeof entry.action !== 'string' ||
      !isNullString(entry.refusal)
    ) {
      return null
    }

    out.push({
      id: entry.id,
      root: entry.root,
      steward: entry.steward,
      version: entry.version,
      sources: entry.sources,
      current: entry.current,
      package_full_name: entry.package_full_name,
      removable: entry.removable,
      action: entry.action,
      refusal: entry.refusal
    })
  }

  return out
}

function parseLauncherEntries(value: unknown): null | HermesLauncherEntry[] {
  if (!Array.isArray(value)) {
    return null
  }

  const out: HermesLauncherEntry[] = []

  for (const row of value) {
    if (!row || typeof row !== 'object' || Array.isArray(row)) {
      return null
    }

    const entry = row as Record<string, unknown>

    if (typeof entry.path !== 'string' || !isNullString(entry.owner)) {
      return null
    }

    out.push({ path: entry.path, owner: entry.owner })
  }

  return out
}

function parseNotice(value: unknown): null | InstallsNoticeState {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null
  }

  const entry = value as Record<string, unknown>

  if (typeof entry.count !== 'number' || typeof entry.dismissed !== 'boolean' || typeof entry.partial !== 'boolean') {
    return null
  }

  return { count: entry.count, dismissed: entry.dismissed, partial: entry.partial }
}

/** The launcher a new terminal picks up first. Null when none was found. */
export function winningLauncher(list: InstallsListResult): null | HermesLauncherEntry {
  return list.launchers[0] ?? null
}

/**
 * The boot notice shows only when other installs exist and the user has not
 * hidden it. A failed list (null) never shows a notice.
 */
export function shouldShowBootNotice(list: null | InstallsListResult): boolean {
  if (!list) {
    return false
  }

  return list.notice.count > 0 && !list.notice.dismissed
}
