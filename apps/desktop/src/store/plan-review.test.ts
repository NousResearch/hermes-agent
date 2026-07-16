import { beforeEach, describe, expect, it } from 'vitest'

import {
  $planReviews,
  contextForPlanArtifact,
  planReviewsForCurrentScope,
  recordPlanArtifactFromTool
} from './plan-review'
import { $activeSessionId, $currentCwd, $selectedStoredSessionId } from './session'

describe('plan review discovery', () => {
  beforeEach(() => {
    $planReviews.set([])
    $activeSessionId.set(null)
    $currentCwd.set('/repo/one')
    $selectedStoredSessionId.set(null)
  })

  it('records plan writes and groups revisions by session', () => {
    recordPlanArtifactFromTool(
      { args: { content: '# First', path: '.hermes/plans/plan.md' }, name: 'write_file' },
      'session-1'
    )
    recordPlanArtifactFromTool(
      { args: { content: '# Second', path: '.hermes/plans/plan-2.md' }, name: 'write_file' },
      'session-1'
    )

    expect($planReviews.get()).toHaveLength(1)
    expect($planReviews.get()[0].versions).toHaveLength(2)
  })

  it('ignores ordinary file writes and duplicate revisions', () => {
    recordPlanArtifactFromTool({ args: { content: 'x', path: 'README.md' }, name: 'write_file' }, 'session-1')
    recordPlanArtifactFromTool(
      { args: { content: '# Plan', path: '.hermes/plans/plan.md' }, name: 'write_file' },
      'session-1'
    )
    recordPlanArtifactFromTool(
      { args: { content: '# Plan', path: '.hermes/plans/plan.md' }, name: 'write_file' },
      'session-1'
    )

    expect($planReviews.get()[0].versions).toHaveLength(1)
  })

  it('does not offer a plan when the write tool failed', () => {
    recordPlanArtifactFromTool(
      {
        args: { content: '# Plan', path: '.hermes/plans/plan.md' },
        name: 'write_file',
        result: { error: 'permission denied', success: false }
      },
      'session-1'
    )

    expect($planReviews.get()).toEqual([])
  })

  it('does not expose plan reviews in another workspace', () => {
    recordPlanArtifactFromTool(
      { args: { content: '# Plan', path: '.hermes/plans/plan.md' }, name: 'write_file' },
      'session-1'
    )

    $currentCwd.set('/repo/two')

    expect(planReviewsForCurrentScope()).toEqual([])
  })

  it('keeps annotations in plan scope only for the active plan revision', () => {
    recordPlanArtifactFromTool(
      { args: { content: '# Plan', path: '.hermes/plans/plan.md' }, name: 'write_file' },
      'session-1'
    )
    $planReviews.set($planReviews.get().map(review => ({ ...review, status: 'reviewing' })))
    $activeSessionId.set('session-1')

    expect(contextForPlanArtifact('/repo/.hermes/plans/plan.md', 'wrong-hash')).toBeNull()
    const hash = $planReviews.get()[0].versions[0].contentHash
    expect(contextForPlanArtifact('/repo/.hermes/plans/plan.md', hash)).toMatchObject({
      artifactPath: '/repo/.hermes/plans/plan.md',
      kind: 'plan',
      sessionId: 'session-1'
    })

    $activeSessionId.set('other-session')
    expect(contextForPlanArtifact('/repo/.hermes/plans/plan.md', hash)).toBeNull()
  })
})
