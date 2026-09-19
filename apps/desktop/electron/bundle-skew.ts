/**
 * Renderer-bundle skew detection.
 *
 * The desktop UI (including bundled plugins like Bot Mode) is compiled into
 * the app binary at build time, while `hermes update` only moves the source
 * tree. A user who updates from the terminal — or whose in-app update failed
 * on the bundle-swap leg — ends up running a NEW runtime under an OLD
 * renderer: About proudly reports the new Hermes version while the sidebar
 * is missing the features that version shipped (the "no Bots tab after the
 * Bot Mode update" reports).
 *
 * Detection: the packaged build carries install-stamp.json with the commit
 * it was built from. If commits touching the RUNTIME paths of apps/desktop
 * exist in the source tree AFTER that stamp commit, the running renderer is
 * provably missing desktop changes the installed runtime has:
 *
 *   git merge-base --is-ancestor <stampCommit> HEAD
 *   git rev-list --count <stampCommit>..HEAD -- <RUNTIME_PATHS>
 *
 * Ancestry has to come first, because `A..HEAD` only means "how far HEAD is
 * ahead of A" when A is an ancestor of HEAD. When it is not, the range
 * degenerates to HEAD's own history and the count stops describing skew at
 * all: an update that rewrote the tree into a synthetic root leaves a stamp
 * commit that still resolves but sits on a disconnected graph, so the count
 * is a permanent >= 1 even when apps/desktop is byte-identical (#92233).
 * Resolving the stamp is not enough — an unknown commit already exits
 * non-zero below, but a merely *unrelated* one exits 0 with a positive count.
 *
 * Scoping to runtime paths keeps this quiet for the common cases where the
 * repo advances without user-visible desktop changes: agent-only commits
 * elsewhere in the repo, and docs / e2e spec / dev-script churn under
 * apps/desktop that never reaches the shipped renderer or main process
 * (#99832).
 *
 * Fail-quiet by design: no stamp (dev runs), a fallback all-zero stamp
 * (non-git build), an unknown commit (stamp predates a shallow clone's
 * history), a stamp that is not an ancestor of HEAD, or any git failure all
 * report "not stale". This warning must never false-positive — it tells
 * users their install is torn.
 *
 * Pure + injectable so it is testable without booting Electron or git.
 */

export interface BundleSkewStamp {
  commit: string
  /** write-build-stamp.mjs source tag — 'fallback' means the commit is fake. */
  source?: null | string
}

export interface BundleSkewResult {
  /** Runtime-path commits between the build stamp and HEAD (null = unknowable). */
  desktopCommitsBehind: null | number
  /** True only on positive proof that the renderer predates desktop changes in the tree. */
  outOfSync: boolean
}

export type RunGit = (
  args: string[],
  options: { cwd: string }
) => Promise<{ code: number; stderr: string; stdout: string }>

/**
 * The apps/desktop paths that actually reach the user: renderer sources,
 * main-process sources, the HTML entry, the public/ assets Vite copies into
 * the bundle, app icons, and the packaging config. Docs, e2e specs, scratch
 * scripts, and dev tooling never reach the shipped app, so a delta confined
 * to them is not a torn install in any way the user can see.
 */
export const RUNTIME_PATHS = [
  'apps/desktop/src',
  'apps/desktop/electron',
  'apps/desktop/index.html',
  'apps/desktop/public',
  'apps/desktop/assets',
  'apps/desktop/package.json',
  'apps/desktop/vite.config.ts'
] as const

const NOT_STALE: BundleSkewResult = { desktopCommitsBehind: null, outOfSync: false }

/** Matches write-build-stamp.mjs's all-zero placeholder for non-git builds. */
export function isFallbackCommit(commit: string): boolean {
  return /^0{7,40}$/.test(commit)
}

export async function detectBundleSkew(
  stamp: BundleSkewStamp | null,
  runGit: RunGit,
  repoRoot: string
): Promise<BundleSkewResult> {
  if (!stamp?.commit || stamp.source === 'fallback' || isFallbackCommit(stamp.commit)) {
    return NOT_STALE
  }

  try {
    // Exit 0 = ancestor, 1 = unrelated or diverged, anything else = git could
    // not answer (unknown object, shallow clone, not a repo). Only the first
    // makes the commit count below a statement about skew, and the other two
    // are the same "unknowable" the branches above already answer quietly.
    //
    // Deliberately not falling back to comparing apps/desktop CONTENT here.
    // Differing content would prove the build and the tree disagree, but not
    // which way round: a user sitting on an older checkout than their build
    // would be told "app build out of date" backwards. Ancestry is what makes
    // this a proof that the renderer PREDATES the tree, which is the claim the
    // warning actually makes.
    const ancestry = await runGit(['merge-base', '--is-ancestor', stamp.commit, 'HEAD'], {
      cwd: repoRoot
    })

    if (ancestry.code !== 0) {
      return NOT_STALE
    }

    const result = await runGit(['rev-list', '--count', `${stamp.commit}..HEAD`, '--', ...RUNTIME_PATHS], {
      cwd: repoRoot
    })

    if (result.code !== 0) {
      return NOT_STALE
    }

    const count = Number.parseInt(result.stdout.trim(), 10)

    if (!Number.isFinite(count) || count <= 0) {
      return { desktopCommitsBehind: Number.isFinite(count) ? count : null, outOfSync: false }
    }

    return { desktopCommitsBehind: count, outOfSync: true }
  } catch {
    return NOT_STALE
  }
}
