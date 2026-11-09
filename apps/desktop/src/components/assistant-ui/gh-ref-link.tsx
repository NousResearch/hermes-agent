import { useStore } from '@nanostores/react'
import type { FC } from 'react'
import { useEffect, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { ghIssueUrl, type GhRef } from '@/lib/gh-refs'

import { refAttrs } from './directive-text'

/** A `#123` reference in assistant markdown (`#gh-ref/` links rewritten in
 *  `preprocessMarkdown`), resolved against the session's repo: plain text until
 *  `git.repo_refs` reports a github.com origin, then a real github.com link.
 *  Reads `useSessionView`, so a split tile resolves its OWN session's repo —
 *  the RPC answers from the backend's authoritative session cwd, and the
 *  per-key promise cache keeps one git spawn per repo across every chip. */
interface GhRepoRefs {
  available: boolean
  host: string
  owner: string
  repo: string
}

const ghRepoRefsPromises = new Map<string, Promise<GhRepoRefs | null>>()

/** Test seam — same convention as `__resetSessionLinkTitleCache`. */
export function __resetGhRepoRefsCache(): void {
  ghRepoRefsPromises.clear()
}

function fetchGhRepoRefs(sessionId: string): Promise<GhRepoRefs | null> {
  const cached = ghRepoRefsPromises.get(sessionId)

  if (cached) {
    return cached
  }

  // Lazy-import so the composer's rich editor (which also imports this module
  // for its chips) never pulls the gateway stack at boot — same story as
  // `openSessionRef` below.
  const promise = import('@/store/gateway')
    .then(({ activeGatewayProfileKey, requestGatewayForProfile }) =>
      requestGatewayForProfile<GhRepoRefs>(
        activeGatewayProfileKey(),
        'git.repo_refs',
        { session_id: sessionId },
        4_000
      )
    )
    .then(result => (result.available ? result : null))
    .catch(() => null)

  ghRepoRefsPromises.set(sessionId, promise)

  return promise
}

export const GhRefLink: FC<{
  label: string
  target: GhRef
}> = ({ label, target }) => {
  const view = useSessionView()
  const runtimeId = useStore(view.$runtimeId)
  const storedId = useStore(view.$storedId)
  const [refs, setRefs] = useState<GhRepoRefs | null>(null)
  const sessionId = runtimeId ?? storedId

  useEffect(() => {
    if (!sessionId) {
      return
    }

    let cancelled = false

    void fetchGhRepoRefs(sessionId).then(resolved => {
      if (!cancelled && resolved) {
        setRefs(resolved)
      }
    })

    return () => {
      cancelled = true
    }
  }, [sessionId])

  if (!refs) {
    return <>{label}</>
  }

  // `owner/repo#N` keeps its own repository; the session's origin only proves the
  // forge is github.com, so a cross-repo ref never links outside a GitHub session.
  const owner = target.owner ?? refs.owner
  const repo = target.repo ?? refs.repo

  return (
    <a
      {...refAttrs('git', 'wrap-anywhere')}
      href={ghIssueUrl(owner, repo, target.number)}
      rel="noopener noreferrer"
      target="_blank"
      title={`${owner}/${repo}#${target.number}`}
    >
      {label}
    </a>
  )
}
