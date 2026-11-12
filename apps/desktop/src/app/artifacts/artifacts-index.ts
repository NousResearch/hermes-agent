import { useStore } from '@nanostores/react'
import { queryOptions, useQuery } from '@tanstack/react-query'
import { useCallback, useEffect } from 'react'

import { getSessionMessages, listAllProfileSessions } from '@/hermes'
import { useI18n } from '@/i18n'
import { queryClient } from '@/lib/query-client'
import { $activeConnectionId } from '@/store/connections'
import { notify, notifyError } from '@/store/notifications'

import { type ArtifactRecord, loadArtifactsForSessions } from './artifact-utils'

/** The recent sessions the Library indexes. */
const INDEXED_SESSIONS = 30

export interface ArtifactIndex {
  artifacts: ArtifactRecord[]
  failures: Awaited<ReturnType<typeof loadArtifactsForSessions>>['failures']
  sessions: number
}

async function indexArtifacts(): Promise<ArtifactIndex> {
  const sessions = (await listAllProfileSessions(INDEXED_SESSIONS, 1)).sessions

  const { artifacts, failures } = await loadArtifactsForSessions(sessions, (session, page) =>
    getSessionMessages(session.id, session.profile, { ...page, includeCompacted: true, order: 'oldest' })
  )

  return {
    artifacts: artifacts.sort((left, right) => right.timestamp - left.timestamp),
    failures,
    sessions: sessions.length
  }
}

/** The Library's index is server truth cached per machine: a revisit (or a click after the idle
 *  prefetch) paints the last index at once and refreshes behind it. */
export const artifactIndexQuery = (connectionId: null | string) =>
  queryOptions({ queryKey: ['artifact-index', connectionId ?? 'local'], queryFn: indexArtifacts })

/** Warm the active machine's index before the Library is opened. */
export function prefetchArtifactIndex(): Promise<void> {
  return queryClient.prefetchQuery(artifactIndexQuery($activeConnectionId.get()))
}

/** An index that failed to load reads as an empty library (the failure is a toast). */
const EMPTY_ARTIFACTS: ArtifactRecord[] = []

/** Sessions an index skipped, as one warning toast (an index can read most sessions and not all). */
function useSkippedSessionsNotice(index: ArtifactIndex | undefined) {
  const { t } = useI18n()
  const failedLoad = t.artifacts.failedLoad
  const failures = index?.failures
  const sessions = index?.sessions ?? 0

  useEffect(() => {
    if (!failures?.length) {
      return
    }

    const isSafeLimit = ({ error }: (typeof failures)[number]) =>
      String(error instanceof Error ? error.message : error).includes('safe-load limit')

    const safeLimitFailures = failures.filter(isSafeLimit).length
    const otherFailures = failures.length - safeLimitFailures

    const detail = [
      safeLimitFailures ? `${safeLimitFailures} exceeded the safe transcript load limit.` : '',
      otherFailures ? `${otherFailures} could not be read.` : ''
    ]
      .filter(Boolean)
      .join(' ')

    notify({
      id: 'artifacts-partial-load',
      kind: 'warning',
      title: failedLoad,
      message: `Skipped ${failures.length} of ${sessions} recent sessions while indexing artifacts.`,
      detail,
      durationMs: 10_000
    })
  }, [failedLoad, failures, sessions])
}

/** The Library's index: null until the first index for this machine exists, then the cached one
 *  while a refresh runs behind it. */
export function useArtifactIndex() {
  const { t } = useI18n()
  const failedLoad = t.artifacts.failedLoad
  // Passed explicitly: the Library also mounts outside the app's query provider (tests, embeds).
  const index = useQuery(artifactIndexQuery(useStore($activeConnectionId)), queryClient)
  const { error, isFetching, refetch } = index

  useEffect(() => {
    if (error) {
      notifyError(error, failedLoad)
    }
  }, [error, failedLoad])

  useSkippedSessionsNotice(index.data)

  const refresh = useCallback(async () => {
    if (!isFetching) {
      await refetch()
    }
  }, [isFetching, refetch])

  return {
    artifacts: index.data?.artifacts ?? (index.isError ? EMPTY_ARTIFACTS : null),
    refreshing: isFetching,
    refresh
  }
}
