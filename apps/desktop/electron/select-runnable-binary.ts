export interface RunnableBinaryOptions {
  /** Ordered candidate absolute paths, highest preference first. */
  candidates: string[]
  fileExists: (filePath: string) => boolean
  /** Probe: does this candidate actually execute (`<binary> --version`)? */
  binaryRuns: (filePath: string) => boolean
}

/**
 * Pick the first candidate that both exists on disk and runs, falling back to
 * the first that merely exists, then to null.
 *
 * A file can exist and still be unlaunchable: on macOS an Intel-only binary
 * ahead on PATH (Homebrew under Rosetta gone after an OS update) fails at
 * spawn time with errno -86 (EBADARCH, "Bad CPU type in executable"), which
 * callers then misreport as a network/update-server problem because the
 * failure only surfaces when the child is spawned. Existence-only selection
 * (`candidates.find(fileExists)`, first PATH hit) commits to the broken
 * candidate by construction and never looks at the working one right after it.
 *
 * The existence-only fallback keeps behaviour unchanged where the probe itself
 * cannot run (locked-down execution policy, AV interposing on spawn) rather
 * than skipping a binary that would have worked.
 *
 * Resolution order (first match wins):
 *   1. a candidate that exists and runs
 *   2. a candidate that merely exists
 *   3. null — caller falls back to its own PATH/bare-name resolution
 */
export function selectRunnableBinary(opts: RunnableBinaryOptions): string | null {
  const existing = opts.candidates.filter(opts.fileExists)

  return existing.find(opts.binaryRuns) || existing[0] || null
}

/** Marker on the error runGit rejects with when the git binary itself could not be spawned. */
export const GIT_UNUSABLE = 'git-unusable'

const SPAWN_FAILURE_REASONS: Record<string, string> = {
  ENOENT: 'not found',
  EACCES: 'permission denied',
  EBADARCH: 'Bad CPU type in executable'
}

/**
 * Turn a child-process 'error' (the binary never ran, as opposed to running
 * and exiting nonzero) into an actionable local-executable message.
 *
 * Without this the caller shows the update-server/network copy plus the raw
 * `spawn Unknown system error -86` — a network claim for a local Intel-only
 * git that macOS refuses to launch (EBADARCH is errno 86 on Darwin, surfaced by
 * Node as errno -86 with no `code`). The message names the binary that failed
 * so the user can replace it; nothing in this path is a network problem.
 *
 * Returns null for anything that is not a spawn-level failure so those keep
 * their existing wording.
 */
export function describeGitSpawnFailure(error: any, binaryPath: string): string | null {
  const code = error?.code || (error?.errno === -86 ? 'EBADARCH' : '')
  const reason = SPAWN_FAILURE_REASONS[code]

  if (!reason) {
    return null
  }

  return (
    `Git on this computer cannot run (${binaryPath}: ${reason}). ` +
    'Install a Git build for this machine (on macOS: `xcode-select --install`) and check again.'
  )
}

/**
 * macOS ships /usr/bin/git as a shim over the active developer directory, so a
 * broken Xcode toolchain fails git by EXIT CODE with the remedy on stderr —
 * the binary spawned fine, so describeGitSpawnFailure never sees it. Both
 * triggers are common: Xcode auto-updates to a new major and every git command
 * refuses until the new licence is accepted, or the active developer path is
 * missing entirely.
 *
 * Without this the caller cannot tell the difference between "git is broken"
 * and "the network is down" and shows the update-server copy, which sends the
 * user to debug their connection while the actual fix is one local command
 * printed in the stderr we already captured.
 *
 * Matches the remedy git itself prints rather than the prose around it, which
 * is localised and reworded between releases. Returns null for everything else
 * so genuine network and repository failures keep their existing wording.
 */
const LOCAL_TOOLCHAIN_STDERR = /xcodebuild -license|xcode-select|invalid active developer path/i

// The licence notice already names its own remedy; the invalid-developer-path
// one does not, so the user is told what to run rather than only what broke.
const NAMES_A_REMEDY = /xcodebuild -license|xcode-select/i

export function describeGitExitFailure(stderr: string): string | null {
  const notice = (stderr || '').trim()

  if (!LOCAL_TOOLCHAIN_STDERR.test(notice)) {
    return null
  }

  const remedy = NAMES_A_REMEDY.test(notice) ? '' : ' Run `xcode-select --install`, then check again.'

  return `Git on this computer cannot run. ${firstLine(notice)}${remedy}`
}

const firstLine = (text: string): string => text.split('\n').find(Boolean)?.trim() || ''
