import { atom } from 'nanostores'

import { $studyOverview } from '@/store/study'
import { $reviewCompletedToday, $reviewDueItems, $reviewStats } from '@/store/study-review'
import type {
  StudyReviewDetail,
  StudyReviewItem,
  StudyReviewResult,
  StudyReviewSubmissionResponse
} from '@/types/hermes'

import type { StudyClient } from '../client'

export type ReviewSessionStatus = 'loading' | 'answering' | 'revealed' | 'submitting' | 'completed' | 'error'

export interface ReviewSessionState {
  activeItem: null | StudyReviewItem
  confidence: null | number
  detail: null | StudyReviewDetail
  elapsedSeconds: number
  error: null | string
  lastResult: null | StudyReviewSubmissionResponse
  response: string
  resumeStatus: 'answering' | 'revealed' | null
  sessionId: string
  startedAt: null | number
  status: ReviewSessionStatus
}

function initialState(): ReviewSessionState {
  return {
    activeItem: null,
    confidence: null,
    detail: null,
    elapsedSeconds: 0,
    error: null,
    lastResult: null,
    response: '',
    resumeStatus: null,
    sessionId: `desktop-review-${Date.now()}`,
    startedAt: null,
    status: 'loading'
  }
}

export const $reviewSession = atom<ReviewSessionState>(initialState())

let sessionEpoch = 0

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function canSubmit(state: ReviewSessionState): boolean {
  return state.status === 'revealed' || (state.status === 'error' && state.resumeStatus === 'revealed')
}

async function refreshReviewProjections(client: StudyClient, projectId: string): Promise<void> {
  const [stats, overview] = await Promise.allSettled([client.getReviewStats(), client.getOverview(projectId)])

  if (stats.status === 'fulfilled') {
    $reviewStats.set(stats.value)
  }

  if (overview.status === 'fulfilled') {
    $studyOverview.set(overview.value)
    $reviewCompletedToday.set(overview.value.completed_today)
  }
}

export async function openReviewItem(
  client: StudyClient,
  item: StudyReviewItem,
  lastResult: null | StudyReviewSubmissionResponse = $reviewSession.get().lastResult
): Promise<void> {
  const epoch = ++sessionEpoch
  const sessionId = $reviewSession.get().sessionId
  $reviewSession.set({
    ...initialState(),
    activeItem: item,
    lastResult,
    sessionId,
    status: 'loading'
  })

  try {
    const detail = await client.getReviewDetail(item.path)

    if (epoch !== sessionEpoch) {
      return
    }

    $reviewSession.set({
      activeItem: item,
      confidence: null,
      detail,
      elapsedSeconds: 0,
      error: null,
      lastResult,
      response: '',
      resumeStatus: null,
      sessionId,
      startedAt: Date.now(),
      status: 'answering'
    })
  } catch (error) {
    if (epoch !== sessionEpoch) {
      return
    }

    $reviewSession.set({
      ...$reviewSession.get(),
      detail: null,
      error: errorMessage(error),
      resumeStatus: null,
      startedAt: null,
      status: 'error'
    })
  }
}

export function startReviewSession(client: StudyClient, item: StudyReviewItem): Promise<void> {
  $reviewSession.set(initialState())

  return openReviewItem(client, item, null)
}

export function answerReview(response: string): void {
  const state = $reviewSession.get()

  if (state.status !== 'answering') {
    return
  }

  $reviewSession.set({ ...state, response })
}

export function setReviewConfidence(confidence: number): void {
  const state = $reviewSession.get()

  if (state.status !== 'answering' || confidence < 1 || confidence > 5) {
    return
  }

  $reviewSession.set({ ...state, confidence })
}

export function revealReviewAnswer(): void {
  const state = $reviewSession.get()

  if (state.status !== 'answering' || !state.response.trim() || state.confidence === null) {
    return
  }

  $reviewSession.set({ ...state, status: 'revealed' })
}

export function tickReviewSession(now = Date.now()): void {
  const state = $reviewSession.get()

  if (state.startedAt === null || (state.status !== 'answering' && state.status !== 'revealed')) {
    return
  }

  const elapsedSeconds = Math.max(0, Math.floor((now - state.startedAt) / 1000))

  if (elapsedSeconds !== state.elapsedSeconds) {
    $reviewSession.set({ ...state, elapsedSeconds })
  }
}

export async function submitReviewResult(
  client: StudyClient,
  options: {
    allItems: StudyReviewItem[]
    items: StudyReviewItem[]
    projectId: string
    result: StudyReviewResult
  }
): Promise<null | number> {
  const state = $reviewSession.get()

  if (!canSubmit(state) || !state.activeItem || !state.response.trim() || state.confidence === null) {
    return null
  }

  const epoch = ++sessionEpoch
  $reviewSession.set({ ...state, error: null, resumeStatus: null, status: 'submitting' })

  try {
    const submitted = await client.submitReview({
      assistance: { hints_used: 0, level: 'independent' },
      diagnoses: [],
      duration_seconds: Math.max(1, state.elapsedSeconds),
      evaluator: { id: 'desktop-review', kind: 'self' },
      note: state.activeItem.path,
      project_id: options.projectId,
      response: state.response.trim(),
      result: options.result,
      self_confidence: state.confidence,
      session_id: state.sessionId,
      transfer_level: 'execution'
    })

    if (epoch !== sessionEpoch) {
      return null
    }

    const remaining = options.items.filter(item => item.path !== state.activeItem?.path)
    const remainingAll = options.allItems.filter(item => item.path !== state.activeItem?.path)
    $reviewDueItems.set(remainingAll)
    $reviewCompletedToday.set(submitted.completed_today)
    void refreshReviewProjections(client, options.projectId)

    if (remaining[0]) {
      await openReviewItem(client, remaining[0], submitted)

      return null
    }

    $reviewSession.set({
      ...$reviewSession.get(),
      error: null,
      lastResult: submitted,
      resumeStatus: null,
      startedAt: null,
      status: 'completed'
    })

    return submitted.completed_today
  } catch (error) {
    if (epoch === sessionEpoch) {
      $reviewSession.set({
        ...$reviewSession.get(),
        error: errorMessage(error),
        resumeStatus: 'revealed',
        status: 'error'
      })
    }

    return null
  }
}

export function cancelReviewSession(): void {
  sessionEpoch += 1
  $reviewSession.set(initialState())
}
