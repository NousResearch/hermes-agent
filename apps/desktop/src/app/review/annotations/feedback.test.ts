import { beforeEach, describe, expect, it } from 'vitest'

import { createReviewContext, type ReviewAnnotation } from '@/store/annotations'
import { $currentCwd } from '@/store/session'

import { annotationInlineRefs, buildAnnotationFeedback } from './feedback'

const context = createReviewContext({ cwd: '/repo', headSha: 'abc', kind: 'git', reviewScope: 'branch' })

function annotation(side: 'new' | 'old'): ReviewAnnotation {
  return {
    anchor: { excerpt: 'line', kind: 'diff', lineEnd: 8, lineStart: 8, path: 'src/a.ts', side },
    comment: 'check it',
    contextId: context.id,
    createdAt: 1,
    id: side,
    labels: ['bug'],
    status: 'active',
    type: 'concern'
  }
}

describe('annotation feedback', () => {
  beforeEach(() => $currentCwd.set('/repo'))

  it('includes revision context in exported feedback', () => {
    const feedback = buildAnnotationFeedback(context, [annotation('old')])

    expect(feedback).toContain('Diff scope: branch')
    expect(feedback).toContain('Head: abc')
    expect(feedback).toContain('src/a.ts:8 (old side)')
  })

  it('never projects old-side diff lines onto the current file', () => {
    expect(annotationInlineRefs([annotation('old')])).toEqual([])
    expect(annotationInlineRefs([annotation('new')])).toHaveLength(1)
  })
})
