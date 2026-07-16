import { atom } from 'nanostores'

import { requestComposerSubmit } from '@/app/chat/composer/focus'
import { contentFingerprint } from '@/app/review/annotations/anchors'
import { buildAnnotationFeedback } from '@/app/review/annotations/feedback'
import { normalizeOrLocalPreviewTarget } from '@/lib/local-preview'
import { readKey, writeKey } from '@/lib/storage'

import {
  $annotationContext,
  $annotations,
  activateAnnotationContext,
  markAnnotationsSent,
  planReviewContext
} from './annotations'
import { requestPreviewReload, setCurrentSessionPreviewTarget } from './preview'
import { $activeSessionId, $connection, $currentCwd, $selectedStoredSessionId } from './session'

export interface PlanReviewVersion {
  contentHash: string
  createdAt: number
  path: string
}

export interface PlanReview {
  id: string
  scopeId: string
  sessionId: string
  status: 'approved' | 'available' | 'changes_requested' | 'reviewing'
  versions: PlanReviewVersion[]
}

export interface ToolCompletePayload {
  args?: unknown
  name?: string
  result?: unknown
}

const STORAGE_KEY = 'hermes.desktop.planReviews.v2'
const LEGACY_STORAGE_KEY = 'hermes.desktop.planReviews.v1'

function currentPlanScope(): string {
  const connection = $connection.get()

  return JSON.stringify({
    baseUrl: connection?.baseUrl ?? '',
    cwd: $currentCwd.get() ?? '',
    mode: connection?.mode ?? 'local',
    profile: connection?.profile ?? 'default'
  })
}

export function planReviewsForCurrentScope(reviews = $planReviews.get()): PlanReview[] {
  const scopeId = currentPlanScope()

  return reviews.filter(review => review.scopeId === scopeId)
}

function loadPlanReviews(): PlanReview[] {
  try {
    const parsed = JSON.parse(readKey(STORAGE_KEY) ?? readKey(LEGACY_STORAGE_KEY) ?? '[]') as unknown

    if (!Array.isArray(parsed)) {
      return []
    }

    return parsed
      .filter((item): item is PlanReview => {
        if (!item || typeof item !== 'object') {
          return false
        }

        const review = item as Partial<PlanReview>

        return (
          typeof review.id === 'string' &&
          (typeof review.scopeId === 'string' || review.scopeId === undefined) &&
          typeof review.sessionId === 'string' &&
          ['approved', 'available', 'changes_requested', 'reviewing'].includes(review.status ?? '') &&
          Array.isArray(review.versions) &&
          review.versions.every(
            version =>
              typeof version?.contentHash === 'string' &&
              typeof version?.createdAt === 'number' &&
              typeof version?.path === 'string'
          )
        )
      })
      .slice(-50)
      .map(review => ({ ...review, scopeId: review.scopeId ?? 'legacy', versions: review.versions.slice(-20) }))
  } catch {
    return []
  }
}

export const $planReviews = atom<PlanReview[]>(loadPlanReviews())

function setPlanReviews(reviews: PlanReview[]): void {
  const bounded = reviews.slice(-50).map(review => ({ ...review, versions: review.versions.slice(-20) }))

  $planReviews.set(bounded)
  writeKey(STORAGE_KEY, JSON.stringify(bounded))
}

function isPlanPath(path: string): boolean {
  return /(?:^|[/\\])\.hermes[/\\]plans[/\\][^/\\]+\.md$/i.test(path)
}

function toolWriteFailed(result: unknown): boolean {
  if (!result || typeof result !== 'object') {
    return false
  }

  const value = result as Record<string, unknown>

  return value.ok === false || value.success === false || typeof value.error === 'string'
}

export function recordPlanArtifactFromTool(payload: ToolCompletePayload | undefined, sessionId: string | null): void {
  if (payload?.name !== 'write_file' || !sessionId || toolWriteFailed(payload.result)) {
    return
  }

  const args = payload.args && typeof payload.args === 'object' ? (payload.args as Record<string, unknown>) : {}
  const path = typeof args.path === 'string' ? args.path : ''
  const content = typeof args.content === 'string' ? args.content : ''

  if (!path || !content || !isPlanPath(path)) {
    return
  }

  const version: PlanReviewVersion = {
    contentHash: contentFingerprint(content),
    createdAt: Date.now(),
    path
  }

  const reviews = $planReviews.get()
  const scopeId = currentPlanScope()

  const openIndex = reviews.findIndex(
    review => review.scopeId === scopeId && review.sessionId === sessionId && review.status !== 'approved'
  )

  if (openIndex >= 0) {
    const review = reviews[openIndex]

    if (!review.versions.some(item => item.contentHash === version.contentHash)) {
      setPlanReviews(
        reviews.map((item, index) =>
          index === openIndex ? { ...review, status: 'available', versions: [...review.versions, version] } : item
        )
      )
    }

    return
  }

  setPlanReviews([
    ...reviews,
    { id: `${sessionId}:${version.createdAt}`, scopeId, sessionId, status: 'available', versions: [version] }
  ])
}

export async function openPlanReview(reviewId: string): Promise<void> {
  const review = planReviewsForCurrentScope().find(item => item.id === reviewId)
  const version = review?.versions.at(-1)

  if (!review || !version) {
    return
  }

  setPlanReviews($planReviews.get().map(item => (item.id === reviewId ? { ...item, status: 'reviewing' } : item)))

  const target = await normalizeOrLocalPreviewTarget(version.path, $currentCwd.get())

  activateAnnotationContext(planReviewContext(target?.path ?? version.path, version.contentHash, review.sessionId), {
    carryStale: true
  })

  if (target) {
    setCurrentSessionPreviewTarget(target, 'manual', version.path)
    requestPreviewReload()
  }
}

function normalizedPath(path: string): string {
  return path.replace(/\\/g, '/').replace(/^\.\//, '')
}

/** Resolve an open preview back to the plan review that launched it. Matching
 * both content and the normalized path prevents an unrelated Markdown file
 * with the same basename from inheriting plan approval semantics. */
export function contextForPlanArtifact(
  path: string,
  contentHash: string,
  reviews = $planReviews.get(),
  sessionIds = [$activeSessionId.get(), $selectedStoredSessionId.get()]
) {
  const actual = normalizedPath(path)

  const scopeId = currentPlanScope()

  const review = [...reviews].reverse().find(item => {
    if (item.scopeId !== scopeId) {
      return false
    }

    if (item.status !== 'reviewing' || !sessionIds.includes(item.sessionId)) {
      return false
    }

    const version = item.versions.at(-1)
    const expected = normalizedPath(version?.path ?? '')

    return Boolean(
      version && version.contentHash === contentHash && (actual === expected || actual.endsWith(`/${expected}`))
    )
  })

  return review ? planReviewContext(path, contentHash, review.sessionId) : null
}

export function requestPlanChanges(reviewId: string): void {
  const review = planReviewsForCurrentScope().find(item => item.id === reviewId)
  const version = review?.versions.at(-1)
  const context = $annotationContext.get()
  const annotations = $annotations.get().filter(item => item.contextId === context.id)
  const contextPath = normalizedPath(context.artifactPath ?? '')
  const versionPath = normalizedPath(version?.path ?? '')

  const matchingContext =
    context.kind === 'plan' &&
    context.contentHash === version?.contentHash &&
    (contextPath === versionPath || contextPath.endsWith(`/${versionPath}`))

  if (!review || !version || !matchingContext || annotations.length === 0) {
    return
  }

  requestComposerSubmit(
    `${buildAnnotationFeedback(context, annotations)}\n\nRevise the plan and save the next version under .hermes/plans/.`,
    { target: 'main' }
  )
  markAnnotationsSent(annotations.map(item => item.id))
  setPlanReviews(
    $planReviews.get().map(item => (item.id === reviewId ? { ...item, status: 'changes_requested' } : item))
  )
}

export function approvePlanReview(reviewId: string): void {
  const review = planReviewsForCurrentScope().find(item => item.id === reviewId)
  const version = review?.versions.at(-1)

  if (!review || !version) {
    return
  }

  requestComposerSubmit(`Approved. Implement the plan at @file:${version.path}.`, { target: 'main' })
  setPlanReviews($planReviews.get().map(item => (item.id === reviewId ? { ...item, status: 'approved' } : item)))
}
