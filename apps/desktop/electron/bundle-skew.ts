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
 * HEAD once with `git rev-parse --is-shallow-repository HEAD`, keys its cache
 * on that sha, and hands the same sha to both calls: should HEAD move
 * mid-probe, the answer still describes the commit its key names instead of a
 * newer one. That one spawn prints the shallow flag on its first line and the
 * sha on its second (measured against git 2.54), so shallowness costs no extra
 * git call. In a shallow clone, exit 1 from merge-base can mean "the history
 * that would prove ancestry is not fetched yet" rather than "unrelated": a
 * fetch or a deepen changes that answer without moving HEAD, so a shallow
 * exit-1 is returned but never cached. Only a 40- or 64-character lowercase
 * hex sha is trusted in the cache key or in a git argument.
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
 * (non-git build, 40 or 64 zeros), an unknown commit (stamp predates a
 * shallow clone's history), a stamp that is not an ancestor of HEAD, or any
 * git failure all report "not stale". This warning must never false-positive —
 * it tells users their install is torn.
 *
 * Pure + injectable so it is testable without booting Electron or git.
 */

import { resolve } from 'node:path'

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
  // 7-40 covers the abbreviated-to-full SHA-1 range; 64 is the SHA-256
  // placeholder a repository on that object format would write.
  return /^(?:0{7,40}|0{64})$/.test(commit)
}

/** A resolved object id: 40 or 64 lowercase hex, never an error message. */
function isResolvedSha(value: string): boolean {
  return /^(?:[0-9a-f]{40}|[0-9a-f]{64})$/.test(value)
}

/** The literal revision git may resolve itself, or a resolved object id. */
function isTrustedHead(value: string): boolean {
  return value === 'HEAD' || isResolvedSha(value)
}

/**
 * Parse the two-line `rev-parse --is-shallow-repository HEAD` output: the
 * shallow flag, then the sha (git 2.54). Null when either line is not what the
 * probe trusts — a missing line, an error message, an abbreviated or uppercase
 * id — so neither the cache key nor a git argument can be built from a guess.
 */
function parseHeadRevision(stdout: string): { headSha: string; shallow: boolean } | null {
  const [flag, sha] = stdout.trim().split(/\r?\n/)

  if ((flag !== 'true' && flag !== 'false') || !sha || !isResolvedSha(sha)) {
    return null
  }

  return { headSha: sha, shallow: flag === 'true' }
}

export async function detectBundleSkew(
  stamp: BundleSkewStamp | null,
  runGit: RunGit,
  repoRoot: string,
  /**
   * The commit to measure against. The probe passes the sha it resolved and
   * keyed on; the default resolves the symbolic HEAD once per git call, which
   * is only safe when nothing else can move HEAD mid-call. Only the literal
   * 'HEAD' or a resolved object id is accepted — any other value returns
   * not-stale without a spawn, because this string fills a git argument.
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
 *   - both git arguments were validated first — only a resolved object id may
 *     fill the stamp slot, only 'HEAD' or a resolved object id the head slot,
 *     so neither can be read as a flag or an error message; and
 *   - merge-base exited 0 (an ancestor) and rev-list exited 0 with a count that
 *     is pure digits — the number describes skew, and it stays true when the
 *     clone is deepened; or
 *   - merge-base exited exactly 1 — "not an ancestor", the real, settled answer
 *     to the #92233 shape, worth reusing in a full clone. In a SHALLOW clone it
 *     is not settled: the history that would prove ancestry may simply not be
 *     fetched yet, and a fetch or a deepen changes the answer without moving
 *     HEAD. `shallow` carries the flag the probe already read, so the shallow
 *     case is answered but never remembered.
 * Everything else is unknowable and must not be remembered: merge-base exit
 * >1, a non-zero rev-list, an unparsable count, and any throw.
 */
async function answerBundleSkew(
  stamp: BundleSkewStamp | null,
  runGit: RunGit,
  repoRoot: string,
  head: string,
  shallow: boolean | null = null
): Promise<BundleSkewAnswer> {
  if (!stamp?.commit || stamp.source === 'fallback' || isFallbackCommit(stamp.commit)) {
    return { cacheable: false, result: NOT_STALE }
  }

  // Only a resolved object id may fill a git argument. An option-like or
  // non-hex stamp could otherwise be read by git as a flag or an error message,
  // so it is rejected here before any git process starts.
  if (!isResolvedSha(stamp.commit)) {
    return { cacheable: false, result: NOT_STALE }
  }

  // `head` fills a git argument too, so it gets the same trust: the literal
  // 'HEAD' git resolves itself, or a resolved object id. Anything else — a
  // branch name, a flag-like string, an error message, an abbreviated or
  // uppercase id — never reaches a git argv, and no process starts.
  if (!isTrustedHead(head)) {
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

    // Exit 1 answers "not an ancestor". In a full clone that is a real answer;
    // in a shallow clone the missing history can produce the same exit, so the
    // shallow case is answered without being cached.
    //
    // `shallow` is null only for the standalone detectBundleSkew, which throws
    // `cacheable` away, so it must not spawn a rev-parse whose only use is the
    // flag it discards. The probe always passes a resolved flag.
    if (ancestry.code === 1) {
      return { cacheable: shallow === false, result: NOT_STALE }
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

    // `rev-list --count` prints one integer. `parseInt` alone accepted a
    // leading integer followed by junk ('2junk' -> 2), which reports a count
    // git never produced as if it were proof, so the digits are matched in
    // full instead. Surrounding whitespace is git's, not a digit.
    const trimmed = result.stdout.trim()

    if (!/^\d+$/.test(trimmed)) {
      return { cacheable: false, result: NOT_STALE }
    }

    const count = Number(trimmed)

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

/** One in-flight run, tagged with the root it belongs to. */
interface InFlightRun {
  promise: Promise<BundleSkewResult>
  root: string
  /**
   * The HEAD sha this run read, once it knows it; null until then. A same-root
   * caller compares its own HEAD against it, so it never joins a run whose
   * answer describes a different commit.
   */
  headSha: string | null
  /**
   * The signal this run's git calls carry. A run replaced by one for another
   * root is aborted through it, so its git child is killed instead of holding
   * a core for a tree no caller is waiting on.
   */
  controller: AbortController
}

/**
 * Wrap detectBundleSkew in the two things an IPC caller needs and it does not
 * have: single-flight and a HEAD-keyed cache.
 *
 * Every caller (window focus, the update poller, checkUpdates, About) used to
 * spawn its own merge-base/rev-list pair, so eight copies could run at once
 * and one treeless-clone lazy fetch held a core for minutes. Concurrent
 * callers for the same root now share one run, and a result is reused for as
 * long as HEAD is unchanged — proven with a cheap `git rev-parse
 * --is-shallow-repository HEAD`, the only spawn on a hit. A moved HEAD reruns
 * the probe; a HEAD git cannot resolve, or resolves to something that is not a
 * sha, is the same "unknowable" the fail-quiet paths answer, and caches nothing
 * so the next call can read it. A root change starts a new run instead of
 * joining one that belongs to the old tree, and aborts the superseded run's
 * controller so its git child is killed rather than left working for a tree no
 * caller waits on.
 */
export function createBundleSkewProbe({
  stamp,
  runGit,
  repoRoot,
  timeoutMs = BUNDLE_SKEW_TIMEOUT_MS
}: BundleSkewProbeOptions): BundleSkewProbe {
  let cachedKey: string | null = null
  let cachedResult: BundleSkewResult = NOT_STALE
  let inFlight: InFlightRun | null = null
  // Bumped per run. A run that the timeout gave up on keeps working in the
  // background, and must not write its late answer over a newer run's.
  let generation = 0

  const run = async (
    cwd: string,
    controller: AbortController,
    onHeadResolved: (headSha: string) => void
  ): Promise<BundleSkewResult> => {
    if (!stamp?.commit || stamp.source === 'fallback' || isFallbackCommit(stamp.commit)) {
      return NOT_STALE
    }

    // Reject a stamp git would not accept as an object id before the HEAD read,
    // so an option-like or non-hex commit never reaches a git argument and no
    // process is spawned.
    if (!isResolvedSha(stamp.commit)) {
      return NOT_STALE
    }

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

        // Resolve before aborting: abort runs its listeners synchronously, and
        // a listener that throws (a kill on an already-reaped child) would
        // otherwise escape this callback as an uncaught exception and leave the
        // probe's promise unsettled forever.
        resolve(NOT_STALE)

        try {
          controller.abort()
        } catch {
          // A throwing abort listener must not wedge the timeout callback.
        }
      }, timeoutMs)
    })

    const work = (async (): Promise<BundleSkewResult> => {
      // One spawn answers both questions: the shallow flag on the first line,
      // the sha on the second (git 2.54).
      const head = await signaled(['rev-parse', '--is-shallow-repository', 'HEAD'], { cwd })

      if (head.code !== 0) {
        return NOT_STALE
      }

      const revision = parseHeadRevision(head.stdout)

      // A sha git did not actually resolve — an error message, an abbreviated or
      // uppercase id, a missing line — cannot key the cache or fill a git
      // argument. Fail quiet and cache nothing, so the next call reads HEAD
      // again instead of pinning a guess.
      if (!revision) {
        return NOT_STALE
      }

      // Publish the sha this run will answer for, so a same-root caller can
      // compare its own HEAD against it before joining.
      onHeadResolved(revision.headSha)

      // One sha, resolved once. It is both the cache key and the commit the
      // expensive calls below run against, so a HEAD that moves during the
      // probe cannot make a cached answer describe a commit other than its key.
      const key = `${cwd}:${stamp.commit}:${revision.headSha}`

      if (key === cachedKey) {
        return cachedResult
      }

      const answer = await answerBundleSkew(stamp, signaled, cwd, revision.headSha, revision.shallow)

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

  /**
   * Read HEAD the same cheap way a cache hit does, so a caller can prove an
   * in-flight run answers for its own commit. Null when git cannot resolve HEAD
   * to a trustworthy sha: there is nothing to compare against, so no in-flight
   * run can be joined on that reading.
   */
  const readHeadSha = async (cwd: string): Promise<string | null> => {
    try {
      const head = await runGit(['rev-parse', '--is-shallow-repository', 'HEAD'], { cwd })

      return head.code === 0 ? (parseHeadRevision(head.stdout)?.headSha ?? null) : null
    } catch {
      return null
    }
  }

  return async (): Promise<BundleSkewResult> => {
    // Normalize the root first. The source tree can be retargeted at runtime, so
    // a run already in flight for another root must not be joined — that would
    // hand this caller the old tree's answer. path.resolve folds a trailing
    // slash or a relative segment, so two spellings of one directory share one
    // cache key and one in-flight entry instead of spawning twice. The
    // generation counter keeps an abandoned run from writing the cache.
    const cwd = resolve(typeof repoRoot === 'function' ? repoRoot() : repoRoot)

    // The entry is captured in a local: the checks below await a git spawn, and
    // the shared `inFlight` could be replaced in the meantime.
    const joining = inFlight

    if (joining && joining.root === cwd) {
      // Join only a run that answers for THIS caller's commit. A run that has
      // already read HEAD answers for that sha; if HEAD moved since, its answer
      // describes an older commit and must not be handed over. A run that has
      // not read HEAD yet reads it after this call, so it cannot be older: join
      // it without a spawn.
      if (joining.headSha === null || (await readHeadSha(cwd)) === joining.headSha) {
        return joining.promise
      }
    }

    // This run supersedes one for a different root, or for a HEAD that moved
    // since that run read it. Its caller still waits on it, but its git is
    // doing work for a tree or a commit nobody else needs: abort its controller
    // so the child is killed (SIGTERM, then SIGKILL, via killChildOnAbort)
    // instead of holding a core for minutes. The generation guard already keeps
    // its late answer out of the cache.
    if (inFlight) {
      try {
        inFlight.controller.abort()
      } catch {
        // Best-effort: an abort listener that throws (a kill on an
        // already-reaped child) must not reject this call or stop the run for
        // the new root from starting.
      }
    }

    const controller = new AbortController()

    const entry: InFlightRun = {
      controller,
      headSha: null,
      promise: Promise.resolve(NOT_STALE),
      root: cwd
    }

    entry.promise = run(cwd, controller, headSha => {
      entry.headSha = headSha
    })

    entry.promise = entry.promise.finally(() => {
      if (inFlight === entry) {
        inFlight = null
      }
    })

    inFlight = entry

    return entry.promise
  }
}
