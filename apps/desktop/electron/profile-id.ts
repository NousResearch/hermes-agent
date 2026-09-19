import fs, { type Dirent } from 'node:fs'
import path from 'node:path'

// Profile IDENTITY resolution for backend spawns.
//
// A profile's identity is its DIRECTORY NAME under `<HERMES_HOME>/profiles`
// (mirrors `hermes_cli.profiles._PROFILE_ID_RE`). The `display_name` in
// `profiles/<dir>/profile.yaml` is a LABEL: the desktop renders it, and it must
// never reach the backend as `--profile <label>` or as a pool key — the CLI
// parses that value as a profile name, and a label ('SmartHome') is not one, so
// the child dies with `invalid choice` (exit 2) before it announces a port.
//
// Resolution is therefore a property of the SPAWN ARGUMENT, not of each caller:
// `serveBackendArgs()` below resolves through here, so every spawn site inherits
// the rule. Pure except for `readProfileIdCandidates()`.

/** Canonical profile id shape — a directory name, never a label. */
export const PROFILE_ID_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/

export interface ProfileIdCandidate {
  /** Canonical id: the profile's directory name. */
  name: string
  /** Presentation-only label from profile.yaml (`display_name`). Never an id. */
  displayName?: string
}

function unquoteYamlScalar(raw: string): string {
  const value = String(raw ?? '').trim()
  const quoted =
    value.length >= 2 &&
    ((value.startsWith("'") && value.endsWith("'")) || (value.startsWith('"') && value.endsWith('"')))

  if (quoted) {
    return value.slice(1, -1).trim()
  }

  // Unquoted scalars may carry a trailing comment: `display_name: Kiosk # box`.
  const comment = value.indexOf(' #')

  return (comment === -1 ? value : value.slice(0, comment)).trim()
}

/**
 * Best-effort TOP-LEVEL `display_name` read out of a `profile.yaml` body
 * (mirrors `hermes_cli.profiles.read_profile_meta`'s `data.get("display_name")`).
 * Missing file, missing key, or a nested key we must not treat as a profile
 * label (e.g. a plugin's `ui_meta.display_name`) all collapse to '' — the label
 * is presentation-only, so nothing here may throw on the boot path.
 */
export function parseProfileDisplayName(text: string): string {
  const match = /^display_name[ \t]*:[ \t]*(.*)$/m.exec(String(text ?? ''))

  return match ? unquoteYamlScalar(match[1]) : ''
}

/**
 * Resolve a caller-supplied profile name to the canonical profile id (the
 * directory name) the backend must be started with.
 *
 * Precedence: exact id → unique case-insensitive directory match → unique
 * `display_name` alias → the requested name unchanged. An AMBIGUOUS label never
 * resolves: two profiles sharing a label must not become a coin flip, and an
 * unknown name keeps today's behavior (the backend reports it).
 */
export function resolveProfileId(requested: unknown, candidates: readonly ProfileIdCandidate[] = []): string {
  const name = typeof requested === 'string' ? requested.trim() : ''

  if (!name) {
    return ''
  }

  const list = Array.isArray(candidates) ? candidates : []

  // Already the directory's own name — the everyday case, no alias lookup.
  if (list.some(candidate => candidate?.name === name)) {
    return name
  }

  const folded = name.toLowerCase()
  const byDirectory = list.filter(candidate => String(candidate?.name ?? '').toLowerCase() === folded)

  if (byDirectory.length === 1) {
    return byDirectory[0].name
  }

  const byLabel = list.filter(
    candidate =>
      String(candidate?.displayName ?? '')
        .trim()
        .toLowerCase() === folded
  )

  if (byLabel.length === 1 && byDirectory.length === 0) {
    return byLabel[0].name
  }

  return name
}

/**
 * The profiles that exist on THIS box: directory names plus their labels.
 * Missing/unreadable directories and profile.yaml files are skipped so a broken
 * profile can never break a spawn; the list is advisory (it only ever supplies
 * the id we already know).
 */
export function readProfileIdCandidates(
  profilesRoot: string,
  readFile: (filePath: string) => string = filePath => fs.readFileSync(filePath, 'utf8')
): ProfileIdCandidate[] {
  let entries: Dirent[]

  try {
    entries = fs.readdirSync(profilesRoot, { withFileTypes: true })
  } catch {
    return []
  }

  const candidates: ProfileIdCandidate[] = []

  for (const entry of [...entries].sort((left, right) => left.name.localeCompare(right.name))) {
    if (!entry.isDirectory() || entry.name.startsWith('.')) {
      continue
    }

    let displayName = ''

    try {
      displayName = parseProfileDisplayName(readFile(path.join(profilesRoot, entry.name, 'profile.yaml')))
    } catch {
      // No profile.yaml (or an unreadable one) — the id alone is enough.
    }

    candidates.push(displayName ? { displayName, name: entry.name } : { name: entry.name })
  }

  return candidates
}
