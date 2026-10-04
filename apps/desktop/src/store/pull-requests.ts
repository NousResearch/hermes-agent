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
const PR_LOOKUP_CAP = 300

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

// A miss belongs to this transcript revision, not the lifetime of a session.
// Do not reuse the old persisted miss list: it can predate the PR being opened.
const scannedRevisions = new Map<string, string>()
const fetchedAt = new Map<string, number>()
const inFlight = new Map<string, Promise<void>>()
let scanInFlight: Promise<void> | undefined

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

/** Recover the PR actually opened, even when work left the starting checkout. */
export async function recoverSessionPullRequests(sessions: SessionInfo[]): Promise<void> {
  if (scanInFlight) {
    await scanInFlight

    return recoverSessionPullRequests(sessions)
  }

  const revisions = new Map(
    sessions.map(session => [
      session.id,
      JSON.stringify([session.message_count, session.tool_call_count, session.last_active])
    ])
  )

  const ids = [...revisions.keys()].filter(id => scannedRevisions.get(id) !== revisions.get(id))

  if (!ids.length) {
    return
  }

  scanInFlight = (async () => {
    try {
      for (let offset = 0; offset < ids.length; offset += 2000) {
        const { pull_requests: found, scanned: asked } = await scanSessionPullRequests(ids.slice(offset, offset + 2000))
        const stamps = { ...$prBranchBySession.get() }
        let changed = false

        for (const [id, pr] of Object.entries(found)) {
          // Branch stamps are explicit desktop intent; number stamps are recovered.
          // Read after the scan so a PR created while it was pending also wins.
          if (stamps[id] && !/\n#\d+$/.test(stamps[id])) {
            continue
          }

          // The URL owns repository identity. The session may have started outside
          // Git, changed repositories, or opened an upstream PR from a fork.
          const match = /^https:\/\/github\.com\/([\w.-]+\/[\w.-]+)\/pull\/(\d+)\/?$/.exec(pr.url)

          if (!revisions.has(id) || !match || Number(match[2]) !== pr.number) {
            continue
          }

          const key = numberPrKey(`https://github.com/${match[1]}`, pr.number)

          if (stamps[id] !== key) {
            stamps[id] = key
            changed = true
          }
        }

        if (changed) {
          $prBranchBySession.set(stamps)
        }

        for (const id of asked) {
          const revision = revisions.get(id)

          if (revision) {
            scannedRevisions.set(id, revision)
          }
        }
      }
    } catch {
      // A failed read is not an authoritative miss. Retry on the next refresh.
    }
  })()

  try {
    await scanInFlight
  } finally {
    scanInFlight = undefined
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
 *  lookups fetched recently, coalescing in-flight repo reads before rechecking.
 *  Goes through the remote-aware git
 *  facade, so a desktop pointed at a remote gateway asks the BACKEND's `gh`
 *  about the backend's checkout. */
export async function refreshPullRequests(lookupsByRepo: Record<string, string[]>, force = false): Promise<void> {
  const review = desktopGit()?.review

  if (!review?.prList) {
    return
  }

  await Promise.all(
    Object.entries(lookupsByRepo).map(async ([root, requested]) => {
      const pending = inFlight.get(root)

      if (pending) {
        await pending

        return refreshPullRequests({ [root]: requested }, force)
      }

      if (
        !force &&
        requested.every(lookup => Date.now() - (fetchedAt.get(branchPrKey(root, lookup)) ?? -Infinity) < PR_STALE_MS)
      ) {
        return
      }

      const requestedLookups = [...new Set(requested)]

      const task = (async () => {
        for (let offset = 0; offset < requestedLookups.length; offset += PR_LOOKUP_CAP) {
          const lookups = requestedLookups.slice(offset, offset + PR_LOOKUP_CAP)
          const numbers = lookups.filter(l => l.startsWith('#')).map(l => Number(l.slice(1)))

          try {
            const { ghReady, prs } = /^https:\/\/github\.com\/[\w.-]+\/[\w.-]+$/.test(root)
              ? await review.prList(
                  '',
                  [],
                  [],
                  numbers.map(number => `${root}/pull/${number}`)
                )
              : await review.prList(
                  root,
                  lookups.filter(l => !l.startsWith('#')),
                  numbers
                )

            if (ghReady === false) {
              continue
            }

            // A composer subset is authoritative only for the lookups it asked for.
            const next = { ...$pullRequestsByBranch.get() }

            for (const lookup of lookups) {
              delete next[branchPrKey(root, lookup)]
            }

            for (const pr of prs) {
              if (lookups.includes(pr.branch)) {
                next[branchPrKey(root, pr.branch)] = pr
              }

              if (numbers.includes(pr.number)) {
                next[numberPrKey(root, pr.number)] = pr
              }
            }

            $pullRequestsByBranch.set(next)
          } catch {
            // gh missing, unauthenticated, or off-repo — leave confirmed data intact.
          } finally {
            const now = Date.now()

            for (const lookup of lookups) {
              fetchedAt.set(branchPrKey(root, lookup), now)
            }
          }
        }
      })()

      inFlight.set(root, task)

      try {
        await task
      } finally {
        inFlight.delete(root)
      }
    })
  )
}
