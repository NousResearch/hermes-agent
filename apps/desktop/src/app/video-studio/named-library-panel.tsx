import { useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { cn } from '@/lib/utils'

import type { NamedLibraryMatchState, ScriptSegment } from './named-library-matching'
import type { VideoLibraryDescriptor, VideoLibraryStatus } from './moneyprinter-client'

export interface NamedLibraryPanelProps {
  error: string
  libraries: VideoLibraryDescriptor[]
  loadingLibraries: boolean
  loadingLibrary: boolean
  matches: NamedLibraryMatchState
  matchingAll: boolean
  matchingSegmentId: string
  onConfirmClip: (segmentId: string, clipId: string) => void
  onCreateTimeline: () => void | Promise<void>
  onMatchAll: () => void | Promise<void>
  onMatchSegment: (segmentId: string) => unknown | Promise<unknown>
  onRefresh: () => void | Promise<void>
  onScan: () => void | Promise<void>
  onSelectLibrary: (libraryId: string) => void
  scanBusy: boolean
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
      .then(value => {
        if (active) setSource(value)
      })
      .catch(() => undefined)

    return () => {
      active = false
    }
  }, [path])

  if (!source) {
    return (
      <div className="flex h-20 w-14 shrink-0 items-center justify-center border border-(--ui-stroke-secondary) bg-(--ui-bg-primary) text-(--ui-text-tertiary)">
        <Codicon name="device-camera-video" size="1rem" />
      </div>
    )
  }

  return <img alt="镜头关键帧" className="h-20 w-14 shrink-0 object-cover" src={source} />
}

function score(value?: number | null): string {
  return typeof value === 'number' ? value.toFixed(2) : '--'
}

export function NamedLibraryPanel({
  error,
  libraries,
  loadingLibraries,
  loadingLibrary,
  matches,
  matchingAll,
  matchingSegmentId,
  onConfirmClip,
  onCreateTimeline,
  onMatchAll,
  onMatchSegment,
  onRefresh,
  onScan,
  onSelectLibrary,
  scanBusy,
  segments,
  selectedLibraryId,
  status,
  timelineBusy
}: NamedLibraryPanelProps) {
  const confirmedCount = Object.keys(matches.confirmedBySegment).length
  const selectedLibrary = useMemo(
    () => libraries.find(library => library.id === selectedLibraryId) || null,
    [libraries, selectedLibraryId]
  )

  return (
    <section className="space-y-3 rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-bg-secondary) p-3">
      <div>
        <div className="flex items-center gap-2 text-xs font-semibold text-(--ui-text-primary)">
          <Codicon name="library" size="0.875rem" />
          Obsidian 具名资产库
        </div>
        <p className="mt-1 text-[0.6875rem] leading-4 text-(--ui-text-tertiary)">
          每次进入都需要手动选择。候选镜头必须人工确认后才能创建时间线。
        </p>
      </div>

      <label className="block space-y-1 text-xs text-(--ui-text-secondary)">
        <span>视频资产库</span>
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

      {!selectedLibraryId ? (
        <div className="space-y-2 border border-dashed border-(--ui-stroke-secondary) p-3 text-xs text-(--ui-text-tertiary)">
          <div>请先选择资产库</div>
          <Button disabled size="xs" type="button">
            自动匹配全部文案
          </Button>
        </div>
      ) : (
        <>
          <div className="flex flex-wrap items-center gap-2 text-[0.6875rem] text-(--ui-text-tertiary)">
            <span>{selectedLibrary?.root}</span>
            <span>素材 {status?.assets ?? '--'}</span>
            <span>镜头 {status?.clips ?? '--'}</span>
            <span>失败 {status?.failed ?? '--'}</span>
            <span>低置信度 {status?.low_confidence ?? '--'}</span>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button
              disabled={loadingLibrary}
              onClick={() => void onRefresh()}
              size="xs"
              type="button"
              variant="secondary"
            >
              刷新资产库
            </Button>
            <Button
              disabled={scanBusy}
              onClick={() => void onScan()}
              size="xs"
              type="button"
              variant="outline"
            >
              {scanBusy ? '扫描中…' : '扫描新增素材'}
            </Button>
            <Button
              disabled={matchingAll || segments.length === 0}
              onClick={() => void onMatchAll()}
              size="xs"
              type="button"
            >
              {matchingAll ? '匹配中…' : '自动匹配全部文案'}
            </Button>
          </div>
        </>
      )}

      {error && <div className="text-xs text-(--ui-danger-foreground)">{error}</div>}

      {segments.map(segment => {
        const candidates = matches.candidatesBySegment[segment.id] || []
        const confirmedClipId = matches.confirmedBySegment[segment.id]
        const segmentError = matches.errorsBySegment[segment.id]

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

            {segmentError && <div className="text-xs text-(--ui-danger-foreground)">{segmentError}</div>}

            {candidates.map(clip => {
              const confirmed = confirmedClipId === clip.id
              return (
                <article
                  className={cn(
                    'flex gap-3 rounded border p-2',
                    confirmed
                      ? 'border-(--ui-accent) bg-(--ui-control-active-background)'
                      : 'border-(--ui-stroke-secondary) bg-(--ui-bg-primary)'
                  )}
                  key={clip.id}
                >
                  <KeyframeImage path={clip.keyframe_path} />
                  <div className="min-w-0 flex-1 space-y-1">
                    <div className="text-xs font-medium text-(--ui-text-primary)">{clip.description}</div>
                    <div className="text-[0.6875rem] text-(--ui-text-tertiary)">
                      {clip.start_seconds.toFixed(2)}–{clip.end_seconds.toFixed(2)} 秒 · 质量{' '}
                      {score(clip.quality_score)} · 置信度 {score(clip.confidence)} · 匹配 {score(clip.score)}
                    </div>
                    <div className="truncate text-[0.625rem] text-(--ui-text-tertiary)">
                      {clip.source_file_path || clip.file_path || '原素材路径未提供'}
                    </div>
                    {confirmed ? (
                      <span className="text-[0.6875rem] font-medium text-(--ui-accent)">已确认</span>
                    ) : (
                      <Button
                        onClick={() => onConfirmClip(segment.id, clip.id)}
                        size="xs"
                        type="button"
                        variant="outline"
                      >
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

      <div className="flex items-center justify-between border-t border-(--ui-stroke-secondary) pt-3">
        <span className="text-[0.6875rem] text-(--ui-text-tertiary)">已人工确认 {confirmedCount} 个镜头</span>
        <Button
          disabled={!selectedLibraryId || confirmedCount === 0 || timelineBusy}
          onClick={() => void onCreateTimeline()}
          size="xs"
          type="button"
        >
          {timelineBusy ? '创建中…' : '创建素材时间线'}
        </Button>
      </div>
    </section>
  )
}
