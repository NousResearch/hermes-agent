import { afterEach, describe, expect, it } from 'vitest'

import { $reviewDueItems, resetReviewState } from '@/store/study-review'
import type { StudyReviewItem } from '@/types/hermes'

import { MemoryStudyClient } from '../client'

import {
  $reviewSession,
  answerReview,
  cancelReviewSession,
  revealReviewAnswer,
  setReviewConfidence,
  startReviewSession,
  submitReviewResult,
  tickReviewSession
} from './session'

const item: StudyReviewItem = {
  concepts: ['functions'],
  difficulty: 'medium',
  last_reviewed_at: null,
  next_review_at: null,
  path: 'math/derivative.md',
  review_count: 1,
  review_level: 2,
  subject: 'math',
  tags: ['math'],
  title: 'Derivative signs'
}

function client(): MemoryStudyClient {
  return new MemoryStudyClient({
    reviewDetails: {
      [item.path]: {
        answer_markdown: 'Differentiate and inspect the sign.',
        has_answer: true,
        item: { ...item, frontmatter: {}, patterns: [] },
        prompt_markdown: 'Find the monotonic intervals.'
      }
    },
    submissionResult: {
      attempt: { attempt_id: 'attempt-1', item_id: item.path, result: 'correct', score: 1 },
      completed_today: 1,
      completed_today_increment: 1,
      review: {
        next_review_at: '2026-07-16',
        path: item.path,
        review_count: { new: 2, old: 1 },
        review_level: { new: 3, old: 2 },
        title: item.title
      }
    }
  })
}

afterEach(() => {
  cancelReviewSession()
  resetReviewState()
})

describe('review session state machine', () => {
  it('moves loading → answering → revealed → submitting → completed', async () => {
    const adapter = client()
    $reviewDueItems.set([item])

    const loading = startReviewSession(adapter, item)
    expect($reviewSession.get().status).toBe('loading')
    await loading
    expect($reviewSession.get().status).toBe('answering')

    answerReview('Differentiate first.')
    setReviewConfidence(4)
    const startedAt = $reviewSession.get().startedAt
    tickReviewSession((startedAt ?? 0) + 3000)
    revealReviewAnswer()
    expect($reviewSession.get()).toMatchObject({ elapsedSeconds: 3, status: 'revealed' })

    const submission = submitReviewResult(adapter, {
      allItems: [item],
      items: [item],
      projectId: 'math-project',
      result: 'correct'
    })

    expect($reviewSession.get().status).toBe('submitting')
    expect(await submission).toBe(1)
    expect($reviewSession.get().status).toBe('completed')
    expect($reviewDueItems.get()).toEqual([])
    expect(adapter.submissions[0]).toMatchObject({
      duration_seconds: 3,
      note: item.path,
      response: 'Differentiate first.',
      self_confidence: 4
    })
  })
})
