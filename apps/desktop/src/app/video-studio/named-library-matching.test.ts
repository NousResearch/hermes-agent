import { describe, expect, it } from 'vitest'

import type { VideoLibraryClip } from './moneyprinter-client'
import {
  automaticallySelectClips,
  clearLibraryMatches,
  confirmSegmentClip,
  emptyMatchState,
  segmentVideoScript,
  setSegmentCandidates,
  setSegmentError
} from './named-library-matching'

const clip: VideoLibraryClip = {
  asset_id: 'asset-1',
  clip_index: 0,
  created_at: 1,
  description: '后厨煮面工位近景',
  duration_seconds: 5,
  end_seconds: 5,
  file_path: '',
  id: 'clip-1',
  start_seconds: 0,
  status: 'ready',
  tags: [],
  updated_at: 1
}

describe('named library script matching', () => {
  it('automatically selects the best candidate while avoiding adjacent source repetition', () => {
    const repeated = { ...clip, id: 'clip-repeat', asset_id: 'asset-1', score: 0.99 }
    const diverse = { ...clip, id: 'clip-diverse', asset_id: 'asset-2', score: 0.8 }

    expect(
      automaticallySelectClips(
        [
          { id: 'segment-1', text: '第一段' },
          { id: 'segment-2', text: '第二段' }
        ],
        {
          'segment-1': [clip],
          'segment-2': [repeated, diverse]
        }
      )
    ).toEqual({ 'segment-1': 'clip-1', 'segment-2': 'clip-diverse' })
  })

  it('splits editable script into stable non-empty segments', () => {
    expect(segmentVideoScript('后厨现煮。\n\n大块牛肉看得见！')).toEqual([
      { id: 'segment-1', text: '后厨现煮。' },
      { id: 'segment-2', text: '大块牛肉看得见！' }
    ])
  })

  it('splits sentence punctuation when the script has no paragraph breaks', () => {
    expect(segmentVideoScript('后厨现煮。大块牛肉看得见！下个饭点来一碗。')).toEqual([
      { id: 'segment-1', text: '后厨现煮。' },
      { id: 'segment-2', text: '大块牛肉看得见！' },
      { id: 'segment-3', text: '下个饭点来一碗。' }
    ])
  })

  it('clears candidates and confirmations when the library changes', () => {
    const dirty = {
      candidatesBySegment: { 'segment-1': [clip] },
      confirmedBySegment: { 'segment-1': 'clip-1' },
      errorsBySegment: { 'segment-2': '未找到素材' }
    }

    expect(clearLibraryMatches(dirty)).toEqual(emptyMatchState())
  })

  it('does not treat the first candidate as human confirmation', () => {
    const next = setSegmentCandidates(emptyMatchState(), 'segment-1', [clip])

    expect(next.candidatesBySegment['segment-1']).toEqual([clip])
    expect(next.confirmedBySegment).toEqual({})
  })

  it('records explicit confirmation and per-segment errors independently', () => {
    const candidates = setSegmentCandidates(emptyMatchState(), 'segment-1', [clip])
    const confirmed = confirmSegmentClip(candidates, 'segment-1', 'clip-1')
    const failedElsewhere = setSegmentError(confirmed, 'segment-2', '检索失败')

    expect(failedElsewhere.confirmedBySegment).toEqual({ 'segment-1': 'clip-1' })
    expect(failedElsewhere.errorsBySegment).toEqual({ 'segment-2': '检索失败' })
  })
})
