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
 * it was built from. If commits touching `apps/desktop/` exist in the source
 * tree AFTER that stamp commit, the running renderer is provably missing
 * desktop changes the installed runtime has:
 *
 *   git rev-list --count <stampCommit>..HEAD -- apps/desktop
 *
 * Scoping to `apps/desktop/` keeps this quiet for the common case where the
 * repo advances with agent-only changes — a shell built before those is not
 * stale in any way the user can see.
 *
 * Fail-quiet by design: no stamp (dev runs), a fallback all-zero stamp
 * (non-git build), an unknown commit (stamp predates a shallow clone's
 * history), or any git failure all report "not stale". This warning must
 * never false-positive — it tells users their install is torn.
 *
 * Pure + injectable so it is testable without booting Electron or git.
 */

export interface BundleSkewStamp {
  commit: string
  /** write-build-stamp.mjs source tag — 'fallback' means the commit is fake. */
  source?: null | string
}

export interface BundleSkewResult {
  /** Commits under apps/desktop/ between the build stamp and HEAD (null = unknowable). */
  desktopCommitsBehind: null | number
  /** True only on positive proof that the renderer predates desktop changes in the tree. */
  outOfSync: boolean
}

export type RunGit = (
  args: string[],
  options: { cwd: string }
) => Promise<{ code: number; stderr: string; stdout: string }>

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
    const result = await runGit(['rev-list', '--count', `${stamp.commit}..HEAD`, '--', 'apps/desktop'], {
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
