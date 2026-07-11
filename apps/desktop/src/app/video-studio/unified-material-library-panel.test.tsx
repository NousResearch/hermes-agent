import { cleanup, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { emptyMatchState } from './named-library-matching'
import { UnifiedMaterialLibraryPanel, type UnifiedMaterialLibraryPanelProps } from './unified-material-library-panel'

const baseProps: UnifiedMaterialLibraryPanelProps = {
  error: '',
  libraries: [
    {
      id: 'beef-noodle',
      mode: 'linked',
      name: '牛肉面资产库',
      root: '/vault/牛肉面资产库',
      source_roots: ['/vault/material'],
      taxonomy: 'beef-noodle-v1'
    }
  ],
  loadingLibraries: false,
  loadingLibrary: false,
  managementBusy: false,
  matches: emptyMatchState(),
  matchingAll: false,
  matchingSegmentId: '',
  migrationResult: null,
  onAddFiles: vi.fn(),
  onAutoGenerate: vi.fn(),
  onConfirmClip: vi.fn(),
  onConfirmScan: vi.fn(),
  onCreateTimeline: vi.fn(),
  onMatchAll: vi.fn(),
  onMatchSegment: vi.fn(),
  onMigrateLegacy: vi.fn(),
  onRefresh: vi.fn(),
  onSelectDirectory: vi.fn(),
  onSelectLibrary: vi.fn(),
  onVideoSourceChange: vi.fn(),
  scanBusy: false,
  scanPreview: null,
  segments: [{ id: 'segment-1', text: '牛骨每天慢火熬制' }],
  selectedLibraryId: '',
  status: null,
  timelineBusy: false,
  videoSource: 'local'
}

afterEach(cleanup)

describe('UnifiedMaterialLibraryPanel', () => {
  it('keeps online providers in the unified material source selector', () => {
    render(<UnifiedMaterialLibraryPanel {...baseProps} videoSource="pexels" />)

    const source = screen.getByLabelText('素材来源') as HTMLSelectElement
    expect(Array.from(source.options).map(option => option.value)).toEqual(['local', 'pexels', 'pixabay', 'coverr'])
    expect(screen.getByRole('button', { name: 'AI 自动匹配并生成视频' })).toBeTruthy()
    expect(screen.queryByLabelText('视频资产库')).toBeNull()
  })

  it('shows one material library entry and disables ingestion until selection', () => {
    render(<UnifiedMaterialLibraryPanel {...baseProps} />)

    expect(screen.getAllByText('素材库')).toHaveLength(1)
    expect(screen.queryByText('本地素材')).toBeNull()
    expect(screen.queryByText('视频素材库')).toBeNull()
    expect(screen.queryByText('Obsidian 具名资产库')).toBeNull()
    expect((screen.getByRole('button', { name: '添加素材文件' }) as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByRole('button', { name: '选择素材目录' }) as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByRole('button', { name: '迁移旧素材' }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('distinguishes failed assets from semantic clip failures', () => {
    render(
      <UnifiedMaterialLibraryPanel
        {...baseProps}
        selectedLibraryId="beef-noodle"
        status={{
          assets: 45,
          clips: 150,
          database_exists: true,
          failed: 3,
          failed_assets: 5,
          library_id: 'beef-noodle',
          low_confidence: 2,
          root: '/vault/牛肉面资产库',
          semantic_failed: 3,
          unusable: 1
        }}
      />
    )

    expect(screen.getByText(/失败素材 5/)).toBeTruthy()
    expect(screen.getByText(/语义失败 3/)).toBeTruthy()
  })

  it('shows only confirmed candidates in the render basket', () => {
    render(
      <UnifiedMaterialLibraryPanel
        {...baseProps}
        matches={{
          candidatesBySegment: {
            'segment-1': [
              {
                asset_id: 'asset-1',
                clip_index: 0,
                created_at: 1,
                description: '后厨汤锅热气升腾',
                duration_seconds: 4,
                end_seconds: 8,
                file_path: '/vault/selected/soup.mp4',
                id: 'clip-1',
                source_file_path: '/vault/material/raw.mp4',
                start_seconds: 4,
                status: 'ready',
                tags: [],
                updated_at: 1
              }
            ]
          },
          confirmedBySegment: { 'segment-1': 'clip-1' },
          errorsBySegment: {}
        }}
        selectedLibraryId="beef-noodle"
      />
    )

    const basket = within(screen.getByTestId('selected-shot-basket'))
    expect(basket.getByText('AI 已选镜头（可精修）')).toBeTruthy()
    expect(basket.getByText('后厨汤锅热气升腾')).toBeTruthy()
    expect(basket.getByText('/vault/material/raw.mp4')).toBeTruthy()
  })
})
