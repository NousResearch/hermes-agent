import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { NamedLibraryMatchState, ScriptSegment } from './named-library-matching'
import { NamedLibraryPanel, type NamedLibraryPanelProps } from './named-library-panel'
import type { VideoLibraryClip, VideoLibraryDescriptor, VideoLibraryStatus } from './moneyprinter-client'

afterEach(() => {
  cleanup()
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
  }
]

const segments: ScriptSegment[] = [{ id: 'segment-1', text: '后厨现煮，热气腾腾。' }]
const clip: VideoLibraryClip = {
  asset_id: 'asset-1',
  clip_index: 0,
  confidence: 0.9,
  created_at: 1,
  description: '后厨煮面工位近景',
  duration_seconds: 5,
  end_seconds: 25,
  file_path: '',
  id: 'clip-1',
  keyframe_path: '/vault/keyframes/clip-1.jpg',
  quality_score: 0.8,
  score: 0.98,
  source_file_path: '/vault/material/source.MOV',
  start_seconds: 20,
  status: 'ready',
  tags: [],
  updated_at: 1
}

const emptyMatches: NamedLibraryMatchState = {
  candidatesBySegment: {},
  confirmedBySegment: {},
  errorsBySegment: {}
}

const baseProps: NamedLibraryPanelProps = {
  error: '',
  libraries,
  loadingLibraries: false,
  loadingLibrary: false,
  matches: emptyMatches,
  matchingAll: false,
  matchingSegmentId: '',
  onConfirmClip: vi.fn(),
  onCreateTimeline: vi.fn(),
  onMatchAll: vi.fn(),
  onMatchSegment: vi.fn(),
  onRefresh: vi.fn(),
  onScan: vi.fn(),
  onSelectLibrary: vi.fn(),
  scanBusy: false,
  segments,
  selectedLibraryId: '',
  status: null,
  timelineBusy: false
}

const readyStatus: VideoLibraryStatus = {
  assets: 6,
  clips: 24,
  database_exists: true,
  failed: 0,
  library_id: 'beef-noodle',
  low_confidence: 0,
  root: '/vault/牛肉面资产库',
  unusable: 0
}

describe('NamedLibraryPanel', () => {
  it('requires manual library selection before matching', () => {
    render(<NamedLibraryPanel {...baseProps} />)

    expect((screen.getByLabelText('视频资产库') as HTMLSelectElement).value).toBe('')
    expect((screen.getByRole('button', { name: '自动匹配全部文案' }) as HTMLButtonElement).disabled).toBe(
      true
    )
    expect(screen.getByText('请先选择资产库')).toBeTruthy()
  })

  it('calls library selection without choosing one automatically', () => {
    render(<NamedLibraryPanel {...baseProps} />)

    fireEvent.change(screen.getByLabelText('视频资产库'), { target: { value: 'beef-noodle' } })

    expect(baseProps.onSelectLibrary).toHaveBeenCalledWith('beef-noodle')
  })

  it('shows candidates without silently confirming one', () => {
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { readFileDataUrl: vi.fn(async () => 'data:image/jpeg;base64,ZmFrZQ==') }
    })
    render(
      <NamedLibraryPanel
        {...baseProps}
        matches={{ ...emptyMatches, candidatesBySegment: { 'segment-1': [clip] } }}
        selectedLibraryId="beef-noodle"
        status={readyStatus}
      />
    )

    expect(screen.getByText('后厨煮面工位近景')).toBeTruthy()
    expect(screen.getByText(/质量 0\.80/)).toBeTruthy()
    expect(screen.getByRole('button', { name: '选用这个镜头' })).toBeTruthy()
    expect(screen.queryByText('已确认')).toBeNull()
  })

  it('keeps single and match-all actions separate', () => {
    const onMatchAll = vi.fn()
    const onMatchSegment = vi.fn()
    render(
      <NamedLibraryPanel
        {...baseProps}
        onMatchAll={onMatchAll}
        onMatchSegment={onMatchSegment}
        selectedLibraryId="beef-noodle"
        status={readyStatus}
      />
    )

    fireEvent.click(screen.getByRole('button', { name: '匹配此段' }))
    fireEvent.click(screen.getByRole('button', { name: '自动匹配全部文案' }))

    expect(onMatchSegment).toHaveBeenCalledWith('segment-1')
    expect(onMatchAll).toHaveBeenCalledOnce()
  })
})
