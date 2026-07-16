/**
 * Pure logic for interpreting hermes-updater status output and routing
 * the apply decision.
 *
 * Extracted from electron/main.ts so the desktop test rules (no source-regex,
 * extract logic for DI) apply: these functions take plain data and return
 * plain data — no Electron, no filesystem, no spawn. The main process wires
 * them into checkUpdates() and applyUpdates() respectively.
 *
 * Run tests: npx vitest run electron/update-status.test.ts --environment jsdom
 */

// ---------------------------------------------------------------------------
// Types — mirror the shapes in apps/desktop/src/global.d.ts.  We re-declare
// the subset we need locally rather than importing from the renderer's global
// type space (which would pull in the whole d.ts graph).  If global.d.ts adds
// fields the upstream code returns, they pass through the `…rest` spread in
// interpretUpdaterStatus and land on the renderer unchanged.
// ---------------------------------------------------------------------------

export interface DesktopUpdateCommit {
  sha: string
  summary: string
  author: string
  at: number
}

export interface DesktopUpdateStatus {
  supported: boolean
  updateAvailable?: boolean
  branch?: string
  currentBranch?: string
  reason?: string
  message?: string
  error?: string
  behind?: number
  currentSha?: string
  targetSha?: string
  commits?: DesktopUpdateCommit[]
  dirty?: boolean
  fetchedAt?: number
}

// ---------------------------------------------------------------------------
// interpretUpdaterStatus
// ---------------------------------------------------------------------------

/**
 * Shape emitted by `hermes-updater status --check --json`.
 *
 * The updater owns the release channel and version comparison; we only map
 * its verdict into the DesktopUpdateStatus the renderer already knows. The
 * fields below are the contract between the Rust updater and this module —
 * documented in docs/updater-world.md §2.3 and §1.4.
 */
export interface UpdaterStatusJson {
  /** The currently active version string (from current.txt). */
  current_version?: string
  /** The latest available version on the resolved channel. */
  latest_version?: string
  /** True when latest_version > current_version. */
  update_available?: boolean
  /** How many releases behind (always ≥1 when update_available). */
  behind?: number
  /** Release-channel the check used (stable / beta / etc). */
  channel?: string
  /** Human-readable error when the check itself failed. */
  error?: string
  /**
   * Changelog entries from the manifest — release notes the renderer renders
   * as the commit list. Each entry maps to a DesktopUpdateCommit.
   */
  changelog?: UpdaterChangelogEntry[]
  /** The SHA the running slot was built from (may be absent in old bundles). */
  current_sha?: string
  /** The SHA the target release was built from. */
  target_sha?: string
}

export interface UpdaterChangelogEntry {
  version?: string
  summary?: string
  author?: string
  /** Unix timestamp in seconds (git commit convention). */
  at?: number
  sha?: string
}

/**
 * Map `hermes-updater status --check --json` output to DesktopUpdateStatus.
 *
 * - `behind` comes from the updater's release count.
 * - `commits` comes from the manifest's changelog field.
 * - `supported` is always true for a slot install (the updater binary
 *   resolved and answered, which is the precondition for calling this).
 * - On an error from the updater itself, we surface `supported: true,
 *   error: 'fetch-failed'` so the renderer shows a retryable state — the
 *   install is valid, only the network check failed.
 */
export function interpretUpdaterStatus(
  json: unknown,
  opts: { fetchedAt?: number } = {}
): DesktopUpdateStatus {
  const fetchedAt = opts.fetchedAt ?? Date.now()

  // --- Guard: non-object or null ---
  if (typeof json !== 'object' || json === null || Array.isArray(json)) {
    return {
      supported: true,
      error: 'fetch-failed',
      message: 'hermes-updater returned invalid JSON.',
      behind: 0,
      commits: [],
      fetchedAt
    }
  }

  const data = json as UpdaterStatusJson

  // --- Updater-level error (network, release server down, etc.) ---
  if (typeof data.error === 'string' && data.error) {
    return {
      supported: true,
      error: 'fetch-failed',
      message: data.error,
      behind: 0,
      commits: [],
      branch: data.channel,
      fetchedAt
    }
  }

  const updateAvailable = Boolean(data.update_available)
  const behind =
    typeof data.behind === 'number' && data.behind >= 0
      ? data.behind
      : updateAvailable
        ? 1
        : 0

  // Map the changelog entries to DesktopUpdateCommit shape.
  const commits: DesktopUpdateCommit[] = Array.isArray(data.changelog)
    ? data.changelog.map(entry => ({
        sha: typeof entry?.sha === 'string' ? entry.sha : '',
        summary: typeof entry?.summary === 'string' ? entry.summary : '',
        author: typeof entry?.author === 'string' ? entry.author : '',
        at: typeof entry?.at === 'number'
          ? entry.at < 1e12
            ? entry.at * 1000 // seconds → ms (git convention)
            : entry.at // already ms
          : 0
      }))
    : []

  return {
    supported: true,
    updateAvailable,
    behind,
    commits,
    currentSha: data.current_sha,
    targetSha: data.target_sha,
    branch: data.channel,
    fetchedAt
  }
}

// ---------------------------------------------------------------------------
// routeApplyDecision
// ---------------------------------------------------------------------------

/** What kind of install is running? Determines how applyUpdates routes. */
export type InstallType = 'slot' | 'checkout' | 'package'

/** The apply strategy chosen for the current install type. */
export type ApplyRoute =
  | { route: 'updater'; command: string; args: string[] }
  | { route: 'dev-update'; command: string; args: string[] }
  | { route: 'gui-skew'; message: string }

/**
 * Decide how applyUpdates() should hand off based on the install type.
 *
 * - **slot**: spawn `hermes-updater apply --relaunch-app <execPath>
 *   --report json --notify-file <tmp>` detached, then quit.  ALL platforms
 *   converge here.
 * - **checkout**: route to `hermes update` via the existing
 *   runStreamedUpdate streaming (the phase-3 worktree flow).
 * - **package** (AppImage/.deb/.rpm): the backend lives in a slot but the
 *   GUI shell is package-manager-owned — report `supported: false` with the
 *   package-manager message (keeps the existing guiSkew).
 *
 * The returned `command` / `args` for the `updater` route are the stable
 * command and its core args; the caller fills in `execPath` and `tmp`
 * (which require process/app context the pure function doesn't have).
 */
export function routeApplyDecision(installType: InstallType): ApplyRoute {
  switch (installType) {
    case 'slot':
      return {
        route: 'updater',
        command: 'hermes-updater',
        args: ['apply', '--relaunch-app', '{execPath}', '--report', 'json', '--notify-file', '{notifyFile}']
      }

    case 'checkout':
      return {
        route: 'dev-update',
        command: 'hermes',
        args: ['update', '--yes']
      }

    case 'package':
      return {
        route: 'gui-skew',
        message:
          'Backend updated, but the desktop app package was not changed. ' +
          'Update or reinstall the Hermes desktop app to match.'
      }

    default: {
      // Exhaustive guard — if a new InstallType is added without a case,
      // the build fails.
      const _exhaustive: never = installType
      return _exhaustive
    }
  }
}

// ---------------------------------------------------------------------------
// resolveInstallType — pure when given injectable predicates
// ---------------------------------------------------------------------------

/**
 * Determine the install type by probing the filesystem layout.
 *
 * Per docs/updater-world.md §2.5.1:
 *   - managed/slot = a `versions/` directory + `current.txt` at the root
 *   - checkout/ejected = a `.git` directory (or `.git` file for worktrees)
 *   - package = the running binary is an AppImage/.deb/.rpm shell
 *
 * The filesystem probes are injectable so tests don't touch the disk.
 */
export function resolveInstallType(
  hermesHome: string,
  activeHermesRoot: string,
  probes: {
    directoryExists: (p: string) => boolean
    fileExists: (p: string) => boolean
  }
): InstallType {
  const versionsDir = path_join(hermesHome, 'versions')
  const currentTxt = path_join(hermesHome, 'current.txt')

  // Slot: versions/ + current.txt at the hermes-home root.
  if (probes.directoryExists(versionsDir) && probes.fileExists(currentTxt)) {
    return 'slot'
  }

  // Checkout: a .git dir or .git file (worktree) inside the active root.
  const gitDir = path_join(activeHermesRoot, '.git')
  if (probes.directoryExists(gitDir) || probes.fileExists(gitDir)) {
    return 'checkout'
  }

  // Fallback: treat as package-managed (AppImage/.deb/.rpm/dev).
  return 'package'
}

// Minimal path.join to avoid importing 'path' (keeps the module pure for
// testing without Node polyfills).  Only handles forward-slash join.
function path_join(...segments: string[]): string {
  return segments.filter(Boolean).join('/').replace(/\/+/g, '/')
}
