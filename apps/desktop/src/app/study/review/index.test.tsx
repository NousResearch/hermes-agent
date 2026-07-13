import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n/context'
import { $studySelectedProjectId } from '@/store/study'
import { resetReviewState } from '@/store/study-review'

import { ReviewView } from './index'

const getStudyProfile = vi.fn()
const getStudyReviewDetail = vi.fn()
const getStudyReviewDue = vi.fn()
const getStudyReviewQueue = vi.fn()
const getStudyReviewStats = vi.fn()
const submitStudyReviewAttempt = vi.fn()
const updateStudyProfile = vi.fn()

vi.mock('@/hermes', () => ({
  getStudyProfile: () => getStudyProfile(),
  getStudyReviewDetail: (note: string) => getStudyReviewDetail(note),
  getStudyReviewDue: (params: unknown) => getStudyReviewDue(params),
  getStudyReviewQueue: () => getStudyReviewQueue(),
  getStudyReviewStats: () => getStudyReviewStats(),
  submitStudyReviewAttempt: (submission: unknown) => submitStudyReviewAttempt(submission),
  updateStudyProfile: (profile: unknown) => updateStudyProfile(profile)
}))

const reviewItem = {
  path: 'math/examples/derivative.md',
  subject: '数学',
  title: '导数符号判断',
  review_level: 2,
  review_count: 1,
  last_reviewed_at: null,
  next_review_at: null,
  concepts: ['函数单调性'],
  tags: ['数学'],
  difficulty: 'medium'
}

const englishReviewItem = {
  path: 'english/examples/cloze.md',
  subject: '英语',
  title: '完形填空转折逻辑',
  review_level: 1,
  review_count: 0,
  last_reviewed_at: null,
  next_review_at: null,
  concepts: ['转折关系'],
  tags: ['英语'],
  difficulty: 'easy'
}

function renderReview(onStartAgentReview?: (prompt: string) => void | Promise<void>) {
  return render(
    <I18nProvider configClient={null}>
      <ReviewView onStartAgentReview={onStartAgentReview} />
    </I18nProvider>
  )
}

beforeEach(() => {
  $studySelectedProjectId.set('kaoyan-2027')
  getStudyReviewDue.mockResolvedValue({
    configured: true,
    date: '2026-07-12',
    count: 1,
    subjects: ['数学'],
    due: [reviewItem]
  })
  getStudyReviewStats.mockResolvedValue({
    total: 1,
    by_level: { 2: 1 },
    progress: 0,
    concept_stats: {},
    review_streak: 0,
    due_count: 1,
    cached: false
  })
  getStudyProfile.mockResolvedValue({ daily_review_limit: 20, review_level_filter: null, subject_filter: null })
  getStudyReviewQueue.mockResolvedValue({ new_concepts: [], new_examples: [] })
  getStudyReviewDetail.mockResolvedValue({
    item: { ...reviewItem, frontmatter: {}, patterns: ['含参导数'] },
    prompt_markdown: '# 导数符号判断\n\n求函数的单调区间。',
    answer_markdown: '## 解析\n\n先判断参数符号。',
    has_answer: true
  })
  submitStudyReviewAttempt.mockResolvedValue({
    attempt: { attempt_id: 'att-1', item_id: reviewItem.path, result: 'correct', score: 1 },
    review: {
      path: reviewItem.path,
      title: reviewItem.title,
      next_review_at: '2026-07-14',
      review_level: { old: 2, new: 3 },
      review_count: { old: 1, new: 2 }
    },
    completed_today_increment: 1
  })
})

afterEach(() => {
  cleanup()
  resetReviewState()
  $studySelectedProjectId.set(null)
  vi.clearAllMocks()
})

describe('ReviewView', () => {
  it('runs a deliberate reveal review and records one compound submission', async () => {
    renderReview()

    fireEvent.click(await screen.findByRole('button', { name: /导数符号判断/ }))
    expect(await screen.findByText('求函数的单调区间。')).toBeTruthy()
    expect(screen.queryByText('先判断参数符号。')).toBeNull()

    fireEvent.change(screen.getByLabelText('Your answer'), { target: { value: '先求导，再讨论参数符号。' } })
    fireEvent.click(screen.getByRole('button', { name: 'Confidence 4' }))
    fireEvent.click(screen.getByRole('button', { name: 'Reveal answer' }))

    expect(await screen.findByText('先判断参数符号。')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Correct' }))

    await waitFor(() => {
      expect(submitStudyReviewAttempt).toHaveBeenCalledWith(
        expect.objectContaining({
          project_id: 'kaoyan-2027',
          note: reviewItem.path,
          response: '先求导，再讨论参数符号。',
          result: 'correct',
          self_confidence: 4
        })
      )
    })
    expect(await screen.findByText('Review session complete')).toBeTruthy()
  })

  it('starts explicit study-review skill command for selected tag-filtered items', async () => {
    getStudyReviewDue.mockImplementation((params?: { subject?: string }) =>
      Promise.resolve({
        configured: true,
        date: '2026-07-12',
        count: params?.subject === '数学' ? 1 : 2,
        subjects: ['数学', '英语'],
        due: params?.subject === '数学' ? [reviewItem] : [reviewItem, englishReviewItem]
      })
    )
    const onStartAgentReview = vi.fn()
    renderReview(onStartAgentReview)

    fireEvent.click(await screen.findByRole('button', { name: '数学' }))
    await screen.findByText('1 due reviews')
    fireEvent.click(screen.getByRole('button', { name: 'Select filtered' }))
    fireEvent.click(screen.getByRole('button', { name: 'Review 1 selected with agent' }))

    expect(onStartAgentReview).toHaveBeenCalledTimes(1)
    expect(onStartAgentReview).toHaveBeenCalledWith(
      expect.stringContaining('/study-review project_id="kaoyan-2027" tag="数学" notes="math/examples/derivative.md"')
    )
    expect(onStartAgentReview).toHaveBeenCalledWith(
      expect.stringContaining('导数符号判断 — math/examples/derivative.md')
    )
  })
})
