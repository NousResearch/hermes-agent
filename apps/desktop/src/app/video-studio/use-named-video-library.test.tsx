import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type {
  MoneyPrinterResponse,
  VideoLibraryAsset,
  VideoLibraryClip,
  VideoLibraryDescriptor,
  VideoLibraryStatus
} from './moneyprinter-client'
import { useNamedVideoLibrary, type NamedVideoLibraryClient } from './use-named-video-library'

afterEach(() => {
  vi.restoreAllMocks()
})

const libraries: VideoLibraryDescriptor[] = [
  {
    id: 'beef-noodle',
    mode: 'linked',
    name: '牛肉面资产库',
    root: '/vault/牛肉面资产库',
    source_roots: ['/vault/material'],
    taxonomy: 'beef-noodle-v1'
  },
  {
    id: 'second-library',
    mode: 'linked',
    name: '第二资产库',
    root: '/vault/第二资产库',
    source_roots: ['/vault/second'],
    taxonomy: 'general-v1'
  }
]

const clip: VideoLibraryClip = {
  asset_id: 'asset-1',
  clip_index: 0,
  confidence: 0.9,
  created_at: 1,
  description: '后厨煮面工位近景',
  duration_seconds: 5,
  end_seconds: 5,
  file_path: '',
  id: 'clip-1',
  quality_score: 0.8,
  score: 0.98,
  source_file_path: '/vault/material/source.MOV',
  start_seconds: 0,
  status: 'ready',
  tags: [],
  updated_at: 1
}

const status: VideoLibraryStatus = {
  assets: 6,
  clips: 24,
  database_exists: true,
  failed: 0,
  library_id: 'beef-noodle',
  low_confidence: 0,
  root: '/vault/牛肉面资产库',
  unusable: 0
}

function ok<T>(data: T): Promise<MoneyPrinterResponse<T>> {
  return Promise.resolve({ data, error: null, ok: true })
}

function fakeClient(): NamedVideoLibraryClient {
  return {
    addSourceRoot: vi.fn((_libraryId, path) => ok({ library_id: 'beef-noodle', source_roots: [path] })),
    analyzeAsset: vi.fn((_libraryId, assetId) =>
      ok({
        asset: { id: assetId } as VideoLibraryAsset,
        clips: [clip],
        job: { error: '', id: 'job-1', progress: 100, state: 'complete' }
      })
    ),
    createTimeline: vi.fn((_libraryId, clipIds) =>
      ok({ id: 'timeline-1', path: '/vault/牛肉面资产库/timelines/timeline-1.json', timeline: { clipIds } })
    ),
    getLibraryStatus: vi.fn(libraryId => ok({ ...status, library_id: libraryId })),
    importAsset: vi.fn((_libraryId, sourcePath) =>
      ok({ asset: { id: `asset-${sourcePath.split('/').pop()}` } as VideoLibraryAsset })
    ),
    listAssets: vi.fn(() => ok({ assets: [] as VideoLibraryAsset[], total: 0 })),
    listClips: vi.fn((_libraryId, options) => ok({ clips: options?.query ? [clip] : [clip], total: 1 })),
    listLibraries: vi.fn(() => ok({ libraries })),
    migrateLegacyLibrary: vi.fn(() =>
      ok({ failed: 0, imported: 1, library_id: 'beef-noodle', records: [], skipped: 0, total: 1 })
    ),
    scanLibrary: vi.fn(() => ok({ complete: 0, skipped: 11 }))
  }
}

describe('useNamedVideoLibrary', () => {
  it('loads libraries without automatically selecting one', async () => {
    const client = fakeClient()
    const { result } = renderHook(() => useNamedVideoLibrary({ client, script: '后厨现煮。' }))

    await waitFor(() => expect(result.current.libraries).toHaveLength(2))

    expect(result.current.selectedLibraryId).toBe('')
    expect(client.listAssets).not.toHaveBeenCalled()
  })

  it('loads selected data and clears matches when switching libraries', async () => {
    const client = fakeClient()
    const { result } = renderHook(() => useNamedVideoLibrary({ client, script: '后厨现煮。' }))
    await waitFor(() => expect(result.current.libraries).toHaveLength(2))

    act(() => result.current.selectLibrary('beef-noodle'))
    await waitFor(() => expect(result.current.status?.library_id).toBe('beef-noodle'))
    await act(() => result.current.matchSegment('segment-1'))
    act(() => result.current.confirmClip('segment-1', 'clip-1'))
    expect(result.current.matches.confirmedBySegment).toEqual({ 'segment-1': 'clip-1' })

    act(() => result.current.selectLibrary('second-library'))

    expect(result.current.matches.confirmedBySegment).toEqual({})
    await waitFor(() => expect(result.current.status?.library_id).toBe('second-library'))
  })

  it('keeps successful segments when match-all partially fails', async () => {
    const client = fakeClient()
    vi.mocked(client.listClips).mockImplementation((_libraryId, options) => {
      if (options?.query?.includes('第二段')) {
        return ok({ clips: [], total: 0 })
      }
      return ok({ clips: [clip], total: 1 })
    })
    const { result } = renderHook(() => useNamedVideoLibrary({ client, script: '第一段。\n\n第二段。' }))
    await waitFor(() => expect(result.current.libraries).toHaveLength(2))
    act(() => result.current.selectLibrary('beef-noodle'))
    await waitFor(() => expect(result.current.status).not.toBeNull())

    await act(() => result.current.matchAll())

    expect(result.current.matches.candidatesBySegment['segment-1']).toEqual([clip])
    expect(result.current.matches.errorsBySegment['segment-2']).toBe('未找到合适镜头')
  })

  it('creates a timeline with confirmed clips in segment order and the selected library', async () => {
    const client = fakeClient()
    const { result } = renderHook(() => useNamedVideoLibrary({ client, script: '第一段。\n\n第二段。' }))
    await waitFor(() => expect(result.current.libraries).toHaveLength(2))
    act(() => result.current.selectLibrary('beef-noodle'))
    await waitFor(() => expect(result.current.status).not.toBeNull())
    await act(() => result.current.matchAll())
    act(() => {
      result.current.confirmClip('segment-1', 'clip-1')
      result.current.confirmClip('segment-2', 'clip-1')
    })

    await act(() => result.current.createTimeline('9:16'))

    expect(client.createTimeline).toHaveBeenCalledWith('beef-noodle', ['clip-1', 'clip-1'], '9:16', [
      { id: 'segment-1', text: '第一段。' },
      { id: 'segment-2', text: '第二段。' }
    ])
  })

  it('automatically matches every segment and creates a timeline without manual confirmation', async () => {
    const client = fakeClient()
    const secondClip = { ...clip, asset_id: 'asset-2', id: 'clip-2', score: 0.8 }
    vi.mocked(client.listClips).mockImplementation((_libraryId, options) =>
      ok({ clips: options?.query?.includes('第二段') ? [clip, secondClip] : [clip], total: 2 })
    )
    const { result } = renderHook(() => useNamedVideoLibrary({ client, script: '第一段。\n\n第二段。' }))
    await waitFor(() => expect(result.current.libraries).toHaveLength(2))
    act(() => result.current.selectLibrary('beef-noodle'))
    await waitFor(() => expect(result.current.status).not.toBeNull())

    await act(() => result.current.createAutomaticTimeline('9:16'))

    expect(client.createTimeline).toHaveBeenCalledWith('beef-noodle', ['clip-1', 'clip-2'], '9:16', [
      { id: 'segment-1', text: '第一段。' },
      { id: 'segment-2', text: '第二段。' }
    ])
  })

  it('imports and analyzes files only in the selected library', async () => {
    const client = fakeClient()
    const { result } = renderHook(() => useNamedVideoLibrary({ client, script: '后厨现煮。' }))
    await waitFor(() => expect(result.current.libraries).toHaveLength(2))
    act(() => result.current.selectLibrary('beef-noodle'))
    await waitFor(() => expect(result.current.status).not.toBeNull())

    await act(() => result.current.importFiles(['/vault/material/a.mov']))

    expect(client.importAsset).toHaveBeenCalledWith('beef-noodle', '/vault/material/a.mov')
    expect(client.analyzeAsset).toHaveBeenCalledWith('beef-noodle', 'asset-a.mov')
  })

  it('requires a dry-run before confirming a real directory scan', async () => {
    const client = fakeClient()
    const { result } = renderHook(() => useNamedVideoLibrary({ client, script: '后厨现煮。' }))
    await waitFor(() => expect(result.current.libraries).toHaveLength(2))
    act(() => result.current.selectLibrary('beef-noodle'))
    await waitFor(() => expect(result.current.status).not.toBeNull())

    await act(() => result.current.addSourceRoot('/vault/new-material'))

    expect(client.addSourceRoot).toHaveBeenCalledWith('beef-noodle', '/vault/new-material')
    expect(client.scanLibrary).toHaveBeenCalledWith('beef-noodle', true)
    expect(client.scanLibrary).not.toHaveBeenCalledWith('beef-noodle', false)

    await act(() => result.current.confirmScan())
    expect(client.scanLibrary).toHaveBeenCalledWith('beef-noodle', false)
  })

  it('previews existing source roots without writing', async () => {
    const client = fakeClient()
    const { result } = renderHook(() => useNamedVideoLibrary({ client, script: '后厨现煮。' }))
    await waitFor(() => expect(result.current.libraries).toHaveLength(2))
    act(() => result.current.selectLibrary('beef-noodle'))
    await waitFor(() => expect(result.current.status).not.toBeNull())

    await act(() => result.current.previewScan())

    expect(client.scanLibrary).toHaveBeenCalledWith('beef-noodle', true)
    expect(client.scanLibrary).not.toHaveBeenCalledWith('beef-noodle', false)
    expect(result.current.scanPreview).not.toBeNull()
  })

  it('clears management results when switching libraries', async () => {
    const client = fakeClient()
    const { result } = renderHook(() => useNamedVideoLibrary({ client, script: '后厨现煮。' }))
    await waitFor(() => expect(result.current.libraries).toHaveLength(2))
    act(() => result.current.selectLibrary('beef-noodle'))
    await waitFor(() => expect(result.current.status).not.toBeNull())
    await act(() => result.current.addSourceRoot('/vault/new-material'))
    await act(() => result.current.migrateLegacyLibrary())
    expect(result.current.scanPreview).not.toBeNull()
    expect(result.current.migrationResult).not.toBeNull()

    act(() => result.current.selectLibrary('second-library'))

    expect(result.current.scanPreview).toBeNull()
    expect(result.current.migrationResult).toBeNull()
  })
})
