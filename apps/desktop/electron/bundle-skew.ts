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
  options: { cwd: string; env?: NodeJS.ProcessEnv; timeoutMs?: number }
) => Promise<{ code: number | null; stderr: string; stdout: string }>

/**
 * The paths that actually reach the user: renderer sources, main-process
 * sources, the HTML entry, the public/ assets Vite copies into the bundle, app
 * icons, and the packaging config -- plus apps/shared, which both bundles
 * compile in (the renderer through the `@hermes/shared` alias, the main process
 * by relative import). Docs, e2e specs, scratch scripts, and dev tooling never
 * reach the shipped app, so a delta confined to them is not a torn install in
 * any way the user can see.
 */
export const RUNTIME_PATHS = [
  'apps/desktop/src',
  'apps/desktop/electron',
  'apps/desktop/index.html',
  'apps/desktop/public',
  'apps/desktop/assets',
  'apps/desktop/package.json',
  'apps/desktop/vite.config.ts',
  'apps/shared/src',
  'apps/shared/package.json'
] as const

const NOT_STALE: BundleSkewResult = { desktopCommitsBehind: null, outOfSync: false }

/** About/version polls share one check per checkout, including failed checks. */
export function createBundleSkewChecker(
  stamp: BundleSkewStamp | null,
  runGit: RunGit,
  { isUpdating, probeBuildNeeded, now = Date.now }: { isUpdating: () => boolean; probeBuildNeeded?: BuildNeededProbe; now?: () => number }
): (repoRoot: string) => Promise<BundleSkewResult> {
  const pending = new Map<string, Promise<BundleSkewResult>>()
  const cached = new Map<string, { result: BundleSkewResult; at: number }>()

  return repoRoot => {
    if (isUpdating()) {
      cached.clear()

      return Promise.resolve(NOT_STALE)
    }

    const running = pending.get(repoRoot)

    if (running) {
      return running
    }

    const previous = cached.get(repoRoot)

    if (previous && now() - previous.at < 30_000) {
      return Promise.resolve(previous.result)
    }

    const check = detectBundleSkew(stamp, runGit, repoRoot, probeBuildNeeded)
      .then(result => {
        if (isUpdating()) {
          return NOT_STALE
        }

        cached.set(repoRoot, { result, at: now() })

        return result
      })
      .finally(() => pending.delete(repoRoot))

    pending.set(repoRoot, check)

    return check
  }
}

/** Matches write-build-stamp.mjs's all-zero placeholder for non-git builds. */
export function isFallbackCommit(commit: string): boolean {
  return /^0{7,40}$/.test(commit)
}

/**
 * Asks the checkout whether its desktop bundle needs rebuilding — the same
 * content-hash test `hermes update` runs (`hermes desktop --build-needed`).
 * Null means "could not ask" and never claims skew.
 */
export type BuildNeededProbe = () => Promise<boolean | null>

export async function detectBundleSkew(
  stamp: BundleSkewStamp | null,
  runGit: RunGit,
  repoRoot: string,
  probeBuildNeeded?: BuildNeededProbe
): Promise<BundleSkewResult> {
  // Git can only say something when the stamp commit is in HEAD's history.
  // After a same-branch reset (the nightly rewrite, or `hermes update`
  // resetting a diverged checkout) a bundle whose swap never happened carries
  // a stamp that is simply unrelated to HEAD — the 2026-09-05 shape, where
  // install-stamp named the new commit while app.asar was still the previous
  // build. The checkout's own content-hash stamp knows; ask it when git
  // cannot answer, and stay quiet without a probe (dev, unpackaged).
  const probe = async (): Promise<BundleSkewResult> => {
    if (!probeBuildNeeded) {
      return NOT_STALE
    }

    try {
      const needed = await probeBuildNeeded()

      return needed === true ? { desktopCommitsBehind: null, outOfSync: true } : NOT_STALE
    } catch {
      return NOT_STALE
    }
  }

  if (!stamp?.commit || stamp.source === 'fallback' || isFallbackCommit(stamp.commit)) {
    return probe()
  }

  try {
    // A path-filtered walk of a tree:0 clone can otherwise fetch its entire
    // missing history. Older Git ignores NO_LAZY_FETCH, so also bound and reap
    // the process tree in execGit, not just the promise waiting for it.
    const options = {
      cwd: repoRoot,
      env: { ...process.env, GIT_NO_LAZY_FETCH: '1', GIT_TERMINAL_PROMPT: '0', GCM_INTERACTIVE: 'Never' },
      timeoutMs: 5000
    }

    // Exit 0 = ancestor, 1 = unrelated or diverged, anything else = git could
    // not answer (unknown object, shallow clone, not a repo). Only the first
    // makes the checks below a statement about skew, and the other two are the
    // same "unknowable" the branches above already answer quietly. Ancestry is
    // also what gives a content comparison a direction: without it, differing
    // content could mean the checkout is OLDER than the build.
    const ancestry = await runGit(['merge-base', '--is-ancestor', stamp.commit, 'HEAD'], options)

    if (ancestry.code !== 0) {
      return probe()
    }

    const result = await runGit(['rev-list', '--count', `${stamp.commit}..HEAD`, '--', ...RUNTIME_PATHS], options)

    if (result.code !== 0) {
      // A treeless (tree:0) checkout holds trees only for commits it checked
      // out: the walk above needs every intermediate one and fails, but the
      // stamp and HEAD endpoints are usually local. With ancestry proven, a
      // differing endpoint diff means the renderer predates runtime changes.
      // Exit 1 = differ; anything else (a stamp tree that is missing too) is
      // still unknowable.
      const diff = await runGit(['diff', '--quiet', stamp.commit, 'HEAD', '--', ...RUNTIME_PATHS], options)

      return diff.code === 1 ? { desktopCommitsBehind: null, outOfSync: true } : NOT_STALE
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
