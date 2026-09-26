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
 *   git merge-base --is-ancestor <stampCommit> <headSha>
 *   git rev-list --count <stampCommit>..<headSha> -- <RUNTIME_PATHS>
 *
 * Both calls take one resolved sha, never the symbolic HEAD. The probe reads
 * HEAD once with `git rev-parse`, keys its cache on that sha, and hands the
 * same sha to both calls: should HEAD move mid-probe, the answer still
 * describes the commit its key names instead of a newer one.
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

/**
 * One answer plus whether it is worth remembering.
 *
 * `detectBundleSkew` fails quiet on two very different things: a git that
 * answered "no skew" and a git that could not answer at all (unknown object,
 * shallow clone, not a repo, a throw). They are the same BundleSkewResult, so
 * a cache that keys off the result alone pins the unknowable one as if it were
 * proof. `cacheable` carries the distinction the result type cannot.
 */
interface BundleSkewAnswer {
  cacheable: boolean
  result: BundleSkewResult
}

export interface RunGitOptions {
  cwd: string
  /**
   * Aborting kills the git child. The probe aborts it when its timeout fires,
   * so one hung git cannot outlive the probe (a treeless partial clone can
   * lazy-fetch trees for minutes).
   */
  signal?: AbortSignal
}

export type RunGit = (
  args: string[],
  options: RunGitOptions
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
  repoRoot: string,
  /**
   * The commit to measure against. The probe passes the sha it resolved and
   * keyed on; the default resolves the symbolic HEAD once per git call, which
   * is only safe when nothing else can move HEAD mid-call.
   */
  head = 'HEAD'
): Promise<BundleSkewResult> {
  return (await answerBundleSkew(stamp, runGit, repoRoot, head)).result
}

/**
 * The probe's body, carrying the `cacheable` verdict `detectBundleSkew` must
 * drop to keep its public signature.
 *
 * A trustworthy answer is one git actually produced:
 *   - merge-base exited 0 (an ancestor) and rev-list exited 0 with a finite
 *     count — the number describes skew; or
 *   - merge-base exited exactly 1 — "not an ancestor", which is the real,
 *     settled answer to the #92233 shape and is worth reusing.
 * Everything else is unknowable and must not be remembered: merge-base exit
 * >1, a non-zero rev-list, an unparsable count, and any throw.
 */
async function answerBundleSkew(
  stamp: BundleSkewStamp | null,
  runGit: RunGit,
  repoRoot: string,
  head: string
): Promise<BundleSkewAnswer> {
  if (!stamp?.commit || stamp.source === 'fallback' || isFallbackCommit(stamp.commit)) {
    return { cacheable: false, result: NOT_STALE }
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
    const ancestry = await runGit(['merge-base', '--is-ancestor', stamp.commit, head], {
      cwd: repoRoot
    })

    // Exit 1 answers "not an ancestor" — a real answer, not a failure.
    if (ancestry.code === 1) {
      return { cacheable: true, result: NOT_STALE }
    }

    if (ancestry.code !== 0) {
      return { cacheable: false, result: NOT_STALE }
    }

    const result = await runGit(['rev-list', '--count', `${stamp.commit}..${head}`, '--', ...RUNTIME_PATHS], {
      cwd: repoRoot
    })

    if (result.code !== 0) {
      return { cacheable: false, result: NOT_STALE }
    }

    const count = Number.parseInt(result.stdout.trim(), 10)

    if (!Number.isFinite(count)) {
      return { cacheable: false, result: NOT_STALE }
    }

    if (count <= 0) {
      return { cacheable: true, result: { desktopCommitsBehind: count, outOfSync: false } }
    }

    return { cacheable: true, result: { desktopCommitsBehind: count, outOfSync: true } }
  } catch {
    return { cacheable: false, result: NOT_STALE }
  }
}

/** Bound on ONE probe; on expiry it resolves not-stale and aborts git. */
export const BUNDLE_SKEW_TIMEOUT_MS = 10_000

export interface BundleSkewProbeOptions {
  stamp: BundleSkewStamp | null
  runGit: RunGit
  /** Resolved per call: dev can retarget the source tree at runtime. */
  repoRoot: string | (() => string)
  timeoutMs?: number
}

export type BundleSkewProbe = () => Promise<BundleSkewResult>

/**
 * Wrap detectBundleSkew in the two things an IPC caller needs and it does not
 * have: single-flight and a HEAD-keyed cache.
 *
 * Every caller (window focus, the update poller, checkUpdates, About) used to
 * spawn its own merge-base/rev-list pair, so eight copies could run at once
 * and one treeless-clone lazy fetch held a core for minutes. Concurrent
 * callers now share one run, and a result is reused for as long as HEAD is
 * unchanged — proven with a cheap `git rev-parse HEAD`, the only spawn on a
 * hit. A moved HEAD reruns the probe; a HEAD git cannot resolve is the same
 * "unknowable" the fail-quiet paths answer, and caches nothing so the next
 * call can read it.
 */
export function createBundleSkewProbe({
  stamp,
  runGit,
  repoRoot,
  timeoutMs = BUNDLE_SKEW_TIMEOUT_MS
}: BundleSkewProbeOptions): BundleSkewProbe {
  let cachedKey: string | null = null
  let cachedResult: BundleSkewResult = NOT_STALE
  let inFlight: Promise<BundleSkewResult> | null = null
  // Bumped per run. A run that the timeout gave up on keeps working in the
  // background, and must not write its late answer over a newer run's.
  let generation = 0

  const run = async (): Promise<BundleSkewResult> => {
    if (!stamp?.commit || stamp.source === 'fallback' || isFallbackCommit(stamp.commit)) {
      return NOT_STALE
    }

    // The resolved root is part of the cache key: the source tree can be
    // retargeted at runtime, so the same HEAD under a different root is a
    // different tree and a different answer.
    const cwd = typeof repoRoot === 'function' ? repoRoot() : repoRoot
    const controller = new AbortController()
    const signaled: RunGit = (args, options) => runGit(args, { ...options, signal: controller.signal })
    const myGeneration = ++generation

    let timer: ReturnType<typeof setTimeout> | null = null

    // Never left pending: a run that succeeds before its timeout settles this
    // so the losing half of the race does not live on forever.
    let settleExpired: (result: BundleSkewResult) => void = () => {}

    const expired = new Promise<BundleSkewResult>(resolve => {
      settleExpired = resolve

      timer = setTimeout(() => {
        timer = null
        controller.abort()
        resolve(NOT_STALE)
      }, timeoutMs)
    })

    const work = (async (): Promise<BundleSkewResult> => {
      const head = await signaled(['rev-parse', 'HEAD'], { cwd })

      if (head.code !== 0 || !head.stdout.trim()) {
        return NOT_STALE
      }

      // One sha, resolved once. It is both the cache key and the commit the
      // expensive calls below run against, so a HEAD that moves during the
      // probe cannot make a cached answer describe a commit other than its key.
      const headSha = head.stdout.trim()
      const key = `${cwd}:${stamp.commit}:${headSha}`

      if (key === cachedKey) {
        return cachedResult
      }

      const answer = await answerBundleSkew(stamp, signaled, cwd, headSha)

      // Cache only an answer git actually produced, and only from a run that
      // is still current: a run the timeout aborted answers fail-quiet (not
      // proof), and a late write must never pin that or overwrite a newer
      // run's result.
      if (answer.cacheable && !controller.signal.aborted && myGeneration === generation) {
        cachedKey = key
        cachedResult = answer.result
      }

      return answer.result
    })()

    // Handle the rejection here, before the race: when `expired` wins, nothing
    // else is left to observe a late failure, and an unhandled rejection would
    // take the process with it.
    const settled = work.catch(() => NOT_STALE)

    try {
      return await Promise.race([settled, expired])
    } finally {
      if (timer) {
        clearTimeout(timer)
      }

      // Leave no promise pending forever: the losing `expired` promise is
      // settled too, and resolving an already-settled race is a no-op.
      settleExpired(NOT_STALE)
    }
  }

  return () => {
    if (!inFlight) {
      inFlight = run().finally(() => {
        inFlight = null
      })
    }

    return inFlight
  }
}
