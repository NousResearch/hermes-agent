import { atom } from 'nanostores'

import type { HermesBranchPullRequest } from '@/global'
import { scanSessionPullRequests, type SessionInfo } from '@/hermes'
import { desktopGit } from '@/lib/desktop-git'
import { Codecs, persistentAtom } from '@/lib/persisted'

/** How a row's PR reads at a glance — and what the sidebar filters on. A
 *  session with no branch, no PR, or an unreachable `gh` is `none`. */
export type PullRequestBucket = 'closed' | 'draft' | 'merged' | 'none' | 'open'

// `gh pr list` is a network call per repo. The sidebar asks on mount, on
// window focus, and whenever the set of repos on screen changes — this keeps
// those from stacking into a burst of identical requests.
const PR_STALE_MS = 60_000

/** Every known PR keyed by `${repoRoot}\n${branch}` — the join a session row
 *  makes with its own `git_repo_root` + `git_branch`. */
export const $pullRequestsByBranch = atom<Record<string, HermesBranchPullRequest>>({})

/** Sessions whose PR isn't on the branch they recorded at start — the checkout
 *  moved mid-conversation, or the work went off to a worktree. Written when the
 *  desktop creates a PR and when one is recovered from a transcript. Holds the
 *  lookup key, not the PR, so state stays live through the same refresh as
 *  everything else. */
export const $prBranchBySession = persistentAtom<Record<string, string>>(
  'hermes.desktop.prBranchBySession',
  {},
  Codecs.stringRecord
)

/** Sessions already scanned for a PR url. A transcript doesn't grow a new PR,
 *  so a miss is permanent and a hit is already in {@link $prBranchBySession} —
 *  either way the session is never scanned again. */
const $prScannedSessions = persistentAtom<string[]>('hermes.desktop.prScannedSessions', [], Codecs.stringArray)

const fetchedAt = new Map<string, number>()
const inFlight = new Set<string>()
let scanUnavailable = false
let scanInFlight = false

// A session sitting on the trunk has no PR of its own, and asking GitHub about
// "main" is how a stranger's fork branch — forks share our branch namespace —
// ends up badged onto it. Never ask.
const TRUNK_BRANCHES = new Set(['dev', 'develop', 'main', 'master', 'trunk'])

export const branchPrKey = (repoRoot: string, branch: string): string => `${repoRoot}\n${branch}`
/** A PR known only by number (recovered from a transcript), keyed so it can
 *  share the one map. GitHub answers by number just as happily as by branch. */
export const numberPrKey = (repoRoot: string, number: number): string => `${repoRoot}\n#${number}`

export function sessionPrKey(session: SessionInfo): null | string {
  const stamped = $prBranchBySession.get()[session.id]

  if (stamped) {
    return stamped
  }

  const root = session.git_repo_root
  const branch = session.git_branch

  return root && branch && !TRUNK_BRANCHES.has(branch.toLowerCase()) ? branchPrKey(root, branch) : null
}

/** Bind a session to the branch it just opened a PR from. */
export function stampSessionPrBranch(sessionId: string, repoRoot: string, branch: string): void {
  if (!sessionId || !repoRoot || !branch) {
    return
  }

  $prBranchBySession.set({ ...$prBranchBySession.get(), [sessionId]: branchPrKey(repoRoot, branch) })
}

/** Recover PRs the branch join can't see, from the sessions' own transcripts.
 *  A session that ran in the main checkout and worked in a worktree recorded
 *  `main` (or nothing) as its branch, but it ran `gh pr create` — whose output
 *  is a bare PR url, the one shape that's a claim rather than a mention. Scans
 *  each session at most once, ever. */
export async function recoverSessionPullRequests(sessions: SessionInfo[]): Promise<void> {
  const scanned = new Set($prScannedSessions.get())
  const roots = new Map<string, string>()

  for (const session of sessions) {
    if (session.git_repo_root && !scanned.has(session.id) && !sessionPrKey(session)) {
      roots.set(session.id, session.git_repo_root)
    }
  }

  if (roots.size === 0 || scanUnavailable || scanInFlight) {
    return
  }

  scanInFlight = true

  try {
    const { pull_requests: found, scanned: asked } = await scanSessionPullRequests([...roots.keys()])
    const stamps = { ...$prBranchBySession.get() }

    for (const [id, pr] of Object.entries(found)) {
      const root = roots.get(id)

      if (root) {
        stamps[id] = numberPrKey(root, pr.number)
      }
    }

    $prBranchBySession.set(stamps)
    $prScannedSessions.set([...new Set([...scanned, ...asked])])
  } catch {
    // An older backend has no such route. Stop asking rather than retrying on
    // every list refresh; the branch join still covers the common case.
    scanUnavailable = true
  } finally {
    scanInFlight = false
  }
}

export function pullRequestBucket(pr: HermesBranchPullRequest | undefined): PullRequestBucket {
  if (!pr) {
    return 'none'
  }

  if (pr.state === 'merged') {
    return 'merged'
  }

  if (pr.state === 'closed') {
    return 'closed'
  }

  return pr.draft ? 'draft' : 'open'
}

/** Pull PRs for the given lookups, grouped by the repo they live in. Each entry
 *  is a branch name, or `#<number>` for a PR recovered from a transcript. Skips
 *  repos fetched recently or still in flight. Goes through the remote-aware git
 *  facade, so a desktop pointed at a remote gateway asks the BACKEND's `gh`
 *  about the backend's checkout. */
export async function refreshPullRequests(lookupsByRepo: Record<string, string[]>, force = false): Promise<void> {
  const review = desktopGit()?.review

  if (!review?.prList) {
    return
  }

  const now = Date.now()

  const stale = Object.keys(lookupsByRepo).filter(
    root => !inFlight.has(root) && (force || now - (fetchedAt.get(root) ?? 0) > PR_STALE_MS)
  )

  await Promise.all(
    stale.map(async root => {
      inFlight.add(root)

      const lookups = lookupsByRepo[root]
      const numbers = lookups.filter(l => l.startsWith('#')).map(l => Number(l.slice(1)))

      try {
        const { prs } = await review.prList(
          root,
          lookups.filter(l => !l.startsWith('#')),
          numbers
        )

        fetchedAt.set(root, Date.now())

        // Replace this repo's slice wholesale: a PR that closed since the last
        // pull has to disappear, not linger as a stale merge of old and new.
        const next = Object.fromEntries(
          Object.entries($pullRequestsByBranch.get()).filter(([key]) => !key.startsWith(`${root}\n`))
        )

        for (const pr of prs) {
          next[branchPrKey(root, pr.branch)] = pr

          // The session that recovered it looks it up by number, and its branch
          // may well be someone else's by now (or deleted).
          if (numbers.includes(pr.number)) {
            next[numberPrKey(root, pr.number)] = pr
          }
        }

        $pullRequestsByBranch.set(next)
      } catch {
        // gh missing, unauthenticated, or off-repo — leave what we had.
        fetchedAt.set(root, Date.now())
      } finally {
        inFlight.delete(root)
      }
    })
  )
}
