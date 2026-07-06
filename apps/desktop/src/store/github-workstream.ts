import { atom } from 'nanostores'

import type { HermesReviewPr, HermesReviewShipInfo } from '@/global'
import { $activeSessionId, $selectedStoredSessionId } from '@/store/session'
import { setWorkstreamLifecycle, workstreamLifecycle } from '@/store/workstream-metadata'

export type GithubWorkstreamPrBySession = Record<string, HermesReviewPr>

export const $githubWorkstreamPrBySession = atom<GithubWorkstreamPrBySession>({})

export function currentGithubWorkstreamSessionId(): null | string {
  return $selectedStoredSessionId.get() || $activeSessionId.get() || null
}

function samePr(a: HermesReviewPr | undefined, b: HermesReviewPr): boolean {
  return Boolean(a && a.number === b.number && a.state === b.state && a.url === b.url)
}

export function githubWorkstreamPr(
  sessionId: null | string,
  prBySession: GithubWorkstreamPrBySession = $githubWorkstreamPrBySession.get()
): HermesReviewPr | null {
  return sessionId ? (prBySession[sessionId] ?? null) : null
}

export function upsertGithubWorkstreamPr(sessionId: string, pr: HermesReviewPr): void {
  if (!sessionId) {
    return
  }

  const lifecycle = workstreamLifecycle(sessionId)

  if (lifecycle === 'closed' || lifecycle === 'safe_delete') {
    setWorkstreamLifecycle(sessionId, 'active')
  }

  const current = $githubWorkstreamPrBySession.get()

  if (samePr(current[sessionId], pr)) {
    return
  }

  $githubWorkstreamPrBySession.set({
    ...current,
    [sessionId]: pr
  })
}

export function syncReviewPrToWorkstream(sessionId: null | string, shipInfo: HermesReviewShipInfo): void {
  if (!shipInfo.pr || !sessionId) {
    return
  }

  upsertGithubWorkstreamPr(sessionId, shipInfo.pr)
}

export function syncReviewPrToCurrentWorkstream(shipInfo: HermesReviewShipInfo): void {
  syncReviewPrToWorkstream(currentGithubWorkstreamSessionId(), shipInfo)
}
