import { beforeEach, describe, expect, it } from 'vitest'

import { $reviewShipInfo } from '@/store/review'
import { $activeSessionId, $selectedStoredSessionId } from '@/store/session'
import { $workstreamMetadata, setWorkstreamLifecycle } from '@/store/workstream-metadata'

import {
  $githubWorkstreamPrBySession,
  githubWorkstreamPr,
  syncReviewPrToCurrentWorkstream,
  syncReviewPrToWorkstream,
  upsertGithubWorkstreamPr
} from './github-workstream'

const pr = (over: Partial<NonNullable<ReturnType<typeof githubWorkstreamPr>>> = {}) => ({
  number: 42,
  state: 'OPEN',
  url: 'https://github.com/NousResearch/hermes-agent/pull/42',
  ...over
})

describe('github workstream enrichment', () => {
  beforeEach(() => {
    $activeSessionId.set(null)
    $selectedStoredSessionId.set(null)
    $reviewShipInfo.set({ ghReady: false, pr: null })
    $githubWorkstreamPrBySession.set({})
    $workstreamMetadata.set({})
  })

  it('links review PR state to the selected stored session id', () => {
    $activeSessionId.set('runtime-1')
    $selectedStoredSessionId.set('stored-1')

    syncReviewPrToCurrentWorkstream({ ghReady: true, pr: pr() })

    expect(githubWorkstreamPr('stored-1')).toEqual(pr())
    expect(githubWorkstreamPr('runtime-1')).toBeNull()
  })

  it('falls back to the active runtime session when no stored session is selected', () => {
    $activeSessionId.set('runtime-1')

    syncReviewPrToCurrentWorkstream({ ghReady: true, pr: pr({ number: 7 }) })

    expect(githubWorkstreamPr('runtime-1')).toEqual(pr({ number: 7 }))
  })

  it.each(['closed', 'safe_delete'] as const)('reopens %s workstreams when PR enrichment lands', lifecycle => {
    setWorkstreamLifecycle('stored-1', lifecycle, 123)
    $selectedStoredSessionId.set('stored-1')

    syncReviewPrToCurrentWorkstream({ ghReady: true, pr: pr() })

    expect($workstreamMetadata.get()).toEqual({})
    expect(githubWorkstreamPr('stored-1')).toEqual(pr())
  })

  it('does not reopen restart-required workstreams when PR enrichment lands', () => {
    setWorkstreamLifecycle('stored-1', 'restart_required', 123)
    $selectedStoredSessionId.set('stored-1')

    syncReviewPrToCurrentWorkstream({ ghReady: true, pr: pr() })

    expect($workstreamMetadata.get()).toEqual({
      'stored-1': { lifecycle: 'restart_required', updatedAt: 123 }
    })
  })

  it('does not infer or create a workstream without a selected or active session', () => {
    syncReviewPrToCurrentWorkstream({ ghReady: true, pr: pr() })

    expect($githubWorkstreamPrBySession.get()).toEqual({})
  })

  it('does not sample a newly selected session when async ship info lands', () => {
    $selectedStoredSessionId.set('old-session')
    $selectedStoredSessionId.set('new-session')

    $reviewShipInfo.set({ ghReady: true, pr: pr({ number: 77 }) })

    expect($githubWorkstreamPrBySession.get()).toEqual({})
  })

  it('syncs a PR to the captured source workstream instead of the current selection', () => {
    $selectedStoredSessionId.set('new-session')

    syncReviewPrToWorkstream('old-session', { ghReady: true, pr: pr({ number: 77 }) })

    expect(githubWorkstreamPr('old-session')).toEqual(pr({ number: 77 }))
    expect(githubWorkstreamPr('new-session')).toBeNull()
  })

  it('upserts and reads a PR link by session id', () => {
    upsertGithubWorkstreamPr('stored-1', pr({ number: 88 }))

    expect(githubWorkstreamPr('stored-1')).toEqual(pr({ number: 88 }))
    expect($githubWorkstreamPrBySession.get()).toEqual({
      'stored-1': pr({ number: 88 })
    })
  })
})
