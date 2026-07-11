import { useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { cn } from '@/lib/utils'

import type { NamedLibraryMatchState, ScriptSegment } from './named-library-matching'
import type {
  VideoLibraryDescriptor,
  VideoLibraryMigrationResult,
  VideoLibraryStatus
} from './moneyprinter-client'

export interface UnifiedMaterialLibraryPanelProps {
  error: string
  libraries: VideoLibraryDescriptor[]
  loadingLibraries: boolean
  loadingLibrary: boolean
  managementBusy: boolean
  matches: NamedLibraryMatchState
  matchingAll: boolean
  matchingSegmentId: string
  migrationResult: VideoLibraryMigrationResult | null
  onAddFiles: () => void | Promise<void>
  onConfirmClip: (segmentId: string, clipId: string) => void
  onConfirmScan: () => unknown | Promise<unknown>
  onCreateTimeline: () => void | Promise<void>
  onMatchAll: () => void | Promise<void>
  onMatchSegment: (segmentId: string) => unknown | Promise<unknown>
  onMigrateLegacy: () => unknown | Promise<unknown>
  onPreviewScan?: () => unknown | Promise<unknown>
  onRefresh: () => void | Promise<void>
  onSelectDirectory: () => void | Promise<void>
  onSelectLibrary: (libraryId: string) => void
  scanBusy: boolean
  scanPreview: Record<string, unknown> | null
  segments: ScriptSegment[]
  selectedLibraryId: string
  status: VideoLibraryStatus | null
  timelineBusy: boolean
}

function KeyframeImage({ path }: { path?: string }) {
  const [source, setSource] = useState('')

  useEffect(() => {
    let active = true
    setSource('')
    if (!path || !window.hermesDesktop?.readFileDataUrl) return
    void window.hermesDesktop
      .readFileDataUrl(path)
      .then(value => active && setSource(value))
      .catch(() => undefined)
    return () => {
      active = false
    }
  }, [path])

  return source ? (
    <img alt="镜头关键帧" className="h-20 w-14 shrink-0 object-cover" src={source} />
  ) : (
    <div className="flex h-20 w-14 shrink-0 items-center justify-center border border-(--ui-stroke-secondary) bg-(--ui-bg-primary) text-(--ui-text-tertiary)">
      <Codicon name="device-camera-video" size="1rem" />
    </div>
  )
}

function score(value?: number | null): string {
  return typeof value === 'number' ? value.toFixed(2) : '--'
}

export function UnifiedMaterialLibraryPanel(props: UnifiedMaterialLibraryPanelProps) {
  const {
    error,
    libraries,
    loadingLibraries,
    loadingLibrary,
    managementBusy,
    matches,
    matchingAll,
    matchingSegmentId,
    migrationResult,
    onAddFiles,
    onConfirmClip,
    onConfirmScan,
    onCreateTimeline,
    onMatchAll,
    onMatchSegment,
    onMigrateLegacy,
    onPreviewScan,
    onRefresh,
    onSelectDirectory,
    onSelectLibrary,
    scanBusy,
    scanPreview,
    segments,
    selectedLibraryId,
    status,
    timelineBusy
  } = props
  const [migrationOpen, setMigrationOpen] = useState(false)
  const selectedLibrary = useMemo(
    () => libraries.find(library => library.id === selectedLibraryId) || null,
    [libraries, selectedLibraryId]
  )
  const confirmed = segments.flatMap(segment => {
    const clipId = matches.confirmedBySegment[segment.id]
    const clip = (matches.candidatesBySegment[segment.id] || []).find(candidate => candidate.id === clipId)
    return clip ? [{ clip, segment }] : []
  })
  const disabled = !selectedLibraryId || managementBusy

  return (
    <section className="space-y-3 rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-bg-secondary) p-3">
      <div>
        <div className="flex items-center gap-2 text-xs font-semibold text-(--ui-text-primary)">
          <Codicon name="library" size="0.875rem" />
          素材库
        </div>
        <p className="mt-1 text-[0.6875rem] leading-4 text-(--ui-text-tertiary)">
          先手动选择一个资产库。导入、分析、文案匹配和最终混剪只允许使用当前库的镜头。
        </p>
      </div>

      <label className="block space-y-1 text-xs text-(--ui-text-secondary)">
        <span>当前资产库</span>
        <select
          aria-label="视频资产库"
          className="h-8 w-full rounded border border-(--ui-stroke-secondary) bg-(--ui-bg-primary) px-2 text-xs text-(--ui-text-primary)"
          disabled={loadingLibraries}
          onChange={event => onSelectLibrary(event.target.value)}
          value={selectedLibraryId}
        >
          <option value="">请选择资产库</option>
          {libraries.map(library => (
            <option key={library.id} value={library.id}>
              {library.name} · {library.id}
            </option>
          ))}
        </select>
      </label>

      <div className="flex flex-wrap gap-2">
        <Button disabled={disabled} onClick={() => void onAddFiles()} size="xs" type="button">
          添加素材文件
        </Button>
        <Button disabled={disabled} onClick={() => void onSelectDirectory()} size="xs" type="button" variant="outline">
          选择素材目录
        </Button>
        <Button disabled={disabled || scanBusy} onClick={() => void onPreviewScan?.()} size="xs" type="button" variant="outline">
          扫描新增素材
        </Button>
        <Button disabled={disabled || loadingLibrary} onClick={() => void onRefresh()} size="xs" type="button" variant="secondary">
          刷新
        </Button>
        <Button disabled={disabled} onClick={() => setMigrationOpen(true)} size="xs" type="button" variant="ghost">
          迁移旧素材
        </Button>
      </div>

      {selectedLibrary ? (
        <div className="space-y-1 text-[0.6875rem] text-(--ui-text-tertiary)">
          <div>{selectedLibrary.root}</div>
          <div>来源目录：{selectedLibrary.source_roots.join('、')}</div>
          <div>
            素材 {status?.assets ?? '--'} · 镜头 {status?.clips ?? '--'} · 失败 {status?.failed ?? '--'} ·
            低置信度 {status?.low_confidence ?? '--'}
          </div>
        </div>
      ) : (
        <div className="border border-dashed border-(--ui-stroke-secondary) p-3 text-xs text-(--ui-text-tertiary)">
          请先选择资产库，未选择时不允许导入或匹配素材。
        </div>
      )}

      {scanPreview && (
        <div className="flex items-center justify-between gap-3 rounded border border-(--ui-stroke-secondary) p-2 text-xs">
          <span className="text-(--ui-text-secondary)">预扫描完成，请确认后写入当前资产库。</span>
          <Button disabled={scanBusy} onClick={() => void onConfirmScan()} size="xs" type="button">
            {scanBusy ? '扫描中…' : '确认扫描'}
          </Button>
        </div>
      )}

      {migrationResult && (
        <div className="text-xs text-(--ui-text-secondary)">
          旧素材迁移：新增 {migrationResult.imported} / 已存在 {migrationResult.skipped} / 失败 {migrationResult.failed}
        </div>
      )}
      {error && <div className="text-xs text-(--ui-danger-foreground)">{error}</div>}

      <div className="flex justify-end">
        <Button
          disabled={!selectedLibraryId || matchingAll || segments.length === 0}
          onClick={() => void onMatchAll()}
          size="xs"
          type="button"
        >
          {matchingAll ? '匹配中…' : '自动匹配全部文案'}
        </Button>
      </div>

      {segments.map(segment => {
        const candidates = matches.candidatesBySegment[segment.id] || []
        const confirmedClipId = matches.confirmedBySegment[segment.id]
        return (
          <div className="space-y-2 border-t border-(--ui-stroke-secondary) pt-3" key={segment.id}>
            <div className="flex items-start justify-between gap-3">
              <p className="text-xs leading-5 text-(--ui-text-primary)">{segment.text}</p>
              <Button
                disabled={!selectedLibraryId || matchingSegmentId === segment.id}
                onClick={() => void onMatchSegment(segment.id)}
                size="xs"
                type="button"
                variant="secondary"
              >
                {matchingSegmentId === segment.id ? '匹配中…' : '匹配此段'}
              </Button>
            </div>
            {matches.errorsBySegment[segment.id] && (
              <div className="text-xs text-(--ui-danger-foreground)">{matches.errorsBySegment[segment.id]}</div>
            )}
            {candidates.map(clip => {
              const isConfirmed = confirmedClipId === clip.id
              return (
                <article
                  className={cn(
                    'flex gap-3 rounded border p-2',
                    isConfirmed
                      ? 'border-(--ui-accent) bg-(--ui-control-active-background)'
                      : 'border-(--ui-stroke-secondary) bg-(--ui-bg-primary)'
                  )}
                  key={clip.id}
                >
                  <KeyframeImage path={clip.keyframe_path} />
                  <div className="min-w-0 flex-1 space-y-1">
                    <div className="text-xs font-medium text-(--ui-text-primary)">{clip.description}</div>
                    <div className="text-[0.6875rem] text-(--ui-text-tertiary)">
                      {clip.start_seconds.toFixed(2)}–{clip.end_seconds.toFixed(2)} 秒 · 质量 {score(clip.quality_score)} ·
                      置信度 {score(clip.confidence)} · 匹配 {score(clip.score)}
                    </div>
                    <div className="truncate text-[0.625rem] text-(--ui-text-tertiary)">
                      {clip.source_file_path || clip.file_path || '原素材路径未提供'}
                    </div>
                    {!isConfirmed && (
                      <Button onClick={() => onConfirmClip(segment.id, clip.id)} size="xs" type="button" variant="outline">
                        选用这个镜头
                      </Button>
                    )}
                  </div>
                </article>
              )
            })}
          </div>
        )
      })}

      <div className="space-y-2 border-t border-(--ui-stroke-secondary) pt-3" data-testid="selected-shot-basket">
        <div className="text-xs font-semibold text-(--ui-text-primary)">本次已选镜头</div>
        {confirmed.length === 0 ? (
          <div className="text-[0.6875rem] text-(--ui-text-tertiary)">还没有人工确认的镜头。</div>
        ) : (
          confirmed.map(({ clip, segment }) => (
            <div className="text-xs" key={`${segment.id}-${clip.id}`}>
              <div className="text-(--ui-text-primary)">{clip.description}</div>
              <div className="text-[0.625rem] text-(--ui-text-tertiary)">{clip.source_file_path || clip.file_path}</div>
            </div>
          ))
        )}
        <div className="flex justify-end">
          <Button
            disabled={!selectedLibraryId || confirmed.length === 0 || timelineBusy}
            onClick={() => void onCreateTimeline()}
            size="xs"
            type="button"
          >
            {timelineBusy ? '创建中…' : '创建素材时间线'}
          </Button>
        </div>
      </div>

      <ConfirmDialog
        confirmLabel="确认迁移"
        description={`将旧版素材迁移到 ${selectedLibrary?.name || '当前资产库'}。旧文件会保留。`}
        onClose={() => setMigrationOpen(false)}
        onConfirm={async () => {
          await onMigrateLegacy()
        }}
        open={migrationOpen}
        title="迁移旧版素材"
      />
    </section>
  )
}
