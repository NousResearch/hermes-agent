import { describe, expect, it } from 'vitest'

import {
  cacheFilenameForSelection,
  confirmedTimelineFormPatch,
  timelineVideoSelections
} from './material-cache'

describe('material cache provenance bridge', () => {
  it('keeps shot-plan provenance attached to ordered timeline files', () => {
    const selections = timelineVideoSelections({
      shotPlan: [
        {
          assetId: 'asset-a',
          clipId: 'clip-a',
          libraryId: 'beef-noodle',
          script: '牛骨每天慢火熬制',
          segmentId: 'segment-1',
          sourceSha256: 'abcdef1234567890'
        }
      ],
      tracks: { video: [{ clipId: 'clip-a', file: '/vault/selected/clip.mp4' }] }
    })

    expect(selections).toEqual([
      {
        assetId: 'asset-a',
        clipId: 'clip-a',
        file: '/vault/selected/clip.mp4',
        libraryId: 'beef-noodle',
        script: '牛骨每天慢火熬制',
        segmentId: 'segment-1',
        sourceSha256: 'abcdef1234567890'
      }
    ])
  })

  it('uses library asset clip and source hash in the hidden cache filename', () => {
    expect(
      cacheFilenameForSelection({
        assetId: 'asset-a',
        clipId: 'clip-a',
        file: '/vault/selected/clip.mp4',
        libraryId: 'beef-noodle',
        script: '',
        segmentId: 'segment-1',
        sourceSha256: 'abcdef1234567890'
      })
    ).toBe('beef-noodle-asset-a-clip-a-abcdef123456-clip.mp4')
  })

  it('rejects timeline rows without matching provenance', () => {
    expect(
      timelineVideoSelections({
        shotPlan: [],
        tracks: { video: [{ clipId: 'foreign-clip', file: '/tmp/foreign.mp4' }] }
      })
    ).toEqual([])
  })

  it('forces confirmed timeline clips into deterministic local rendering', () => {
    expect(confirmedTimelineFormPatch(['clip-a.mp4', 'clip-b.mp4'])).toEqual({
      localMaterials: ['clip-a.mp4', 'clip-b.mp4'],
      matchMaterialsToScript: true,
      videoConcatMode: 'sequential',
      videoSource: 'local'
    })
  })
})
