import { useCallback, useEffect, useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import type { KnowledgeInventory, KnowledgeSource } from '@/global'
import { useI18n } from '@/i18n'

// 左侧「知识库」入口 = EasyHermes 自建页面(本机/协同知识库,区别于工作流页里给节点
// 运行用的「工作流知识库」)。流程:拖文件夹/文件 → 本地扫描盘点(只统计元数据,不读正文)→
// 报告(可全文索引 / 仅文件名)+ 确认范围 → 入库(文本喂 langflow embed 正文、其余只记
// 文件名)→ 登记为「知识源」(可同步/移除)。多模态内容由上游模型按需读取,本机不解析。

type Phase = 'idle' | 'scanning' | 'review' | 'indexing' | 'done'

function formatSize(bytes: number): string {
  if (bytes >= 1024 * 1024 * 1024) {return `${(bytes / 1024 / 1024 / 1024).toFixed(1)} GB`}

  if (bytes >= 1024 * 1024) {return `${Math.round(bytes / 1024 / 1024)} MB`}

  if (bytes >= 1024) {return `${Math.round(bytes / 1024)} KB`}

  return `${bytes} B`
}

function formatTs(ts: number): string {
  if (!ts) {return '—'}

  try {
    return new Date(ts).toLocaleString(undefined, { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })
  } catch {
    return '—'
  }
}

export function KnowledgeView() {
  const { t } = useI18n()
  const [dragOver, setDragOver] = useState(false)
  const [phase, setPhase] = useState<Phase>('idle')
  const [report, setReport] = useState<KnowledgeInventory | null>(null)
  const [scanError, setScanError] = useState<string | null>(null)
  const [progress, setProgress] = useState<{ done: number; total: number }>({ done: 0, total: 0 })
  const [result, setResult] = useState<{ indexed: number; nameOnly: number } | null>(null)
  const [ingestError, setIngestError] = useState<string | null>(null)

  // 阶段④:本机持久化的知识源列表(红框区)。
  const [sources, setSources] = useState<KnowledgeSource[]>([])
  const [busyId, setBusyId] = useState<string | null>(null)
  const [syncMsg, setSyncMsg] = useState<Record<string, string>>({})

  const loadSources = useCallback(async () => {
    try {
      const list = await window.hermesDesktop?.knowledge?.list?.()
      setSources(Array.isArray(list) ? list : [])
    } catch {
      setSources([])
    }
  }, [])

  useEffect(() => {
    void loadSources()
  }, [loadSources])

  // 监听主进程入库/同步进度(分批 ingest 时回传)。
  useEffect(() => {
    const bridge = window.hermesDesktop

    if (!bridge?.knowledge?.onIngestProgress) {
      return
    }

    return bridge.knowledge.onIngestProgress(p => {
      if (typeof p.done === 'number' && typeof p.total === 'number') {
        setProgress({ done: p.done, total: p.total })
      }
    })
  }, [])

  const scanPath = useCallback(async (rawPath: string) => {
    const srcPath = rawPath.trim()
    const bridge = window.hermesDesktop

    if (!srcPath || !bridge?.knowledge?.inventory) {
      return
    }

    setScanError(null)
    setReport(null)
    setPhase('scanning')

    try {
      const inventory = await bridge.knowledge.inventory(srcPath)

      if (inventory.ok) {
        setReport(inventory)
        setPhase('review')
      } else {
        setScanError(inventory.error || 'read-error')
        setPhase('idle')
      }
    } catch (error) {
      setScanError(error instanceof Error ? error.message : String(error))
      setPhase('idle')
    }
  }, [])

  const handleDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault()
      setDragOver(false)

      const file = event.dataTransfer.files?.[0]
      const bridge = window.hermesDesktop

      if (file && bridge?.getPathForFile) {
        const droppedPath = bridge.getPathForFile(file)

        if (droppedPath) {
          void scanPath(droppedPath)
        }
      }
    },
    [scanPath]
  )

  const handlePick = useCallback(async () => {
    const bridge = window.hermesDesktop

    if (!bridge?.selectPaths) {
      return
    }

    // both:同一对话框选文件夹或文件。
    const paths = await bridge.selectPaths({ both: true, multiple: false })

    if (paths[0]) {
      void scanPath(paths[0])
    }
  }, [scanPath])

  const reset = useCallback(() => {
    setReport(null)
    setScanError(null)
    setIngestError(null)
    setResult(null)
    setProgress({ done: 0, total: 0 })
    setPhase('idle')
  }, [])

  const confirm = useCallback(async () => {
    const bridge = window.hermesDesktop

    if (!report?.path || !bridge?.knowledge?.ingest) {
      return
    }

    setIngestError(null)
    setProgress({ done: 0, total: report.indexable?.count ?? 0 })
    setPhase('indexing')

    try {
      const r = await bridge.knowledge.ingest({ folderPath: report.path, name: report.name ?? '' })

      if (r.ok) {
        setResult({ indexed: r.indexed ?? 0, nameOnly: r.nameOnly ?? 0 })
        setPhase('done')
        void loadSources()
      } else {
        setIngestError(r.error ?? 'ingest-error')
        setPhase('review')
      }
    } catch (error) {
      setIngestError(error instanceof Error ? error.message : String(error))
      setPhase('review')
    }
  }, [report, loadSources])

  const syncSource = useCallback(
    async (sourceId: string) => {
      const bridge = window.hermesDesktop

      if (!bridge?.knowledge?.sync) {
        return
      }

      setBusyId(sourceId)
      setSyncMsg(prev => ({ ...prev, [sourceId]: '' }))

      try {
        const r = await bridge.knowledge.sync(sourceId)

        const msg = !r.ok
          ? r.error ?? 'sync-error'
          : r.changed
            ? t.knowledge.syncedChange(r.added ?? 0, r.modified ?? 0, r.removed ?? 0)
            : t.knowledge.upToDate

        setSyncMsg(prev => ({ ...prev, [sourceId]: msg }))
        await loadSources()
      } catch (error) {
        setSyncMsg(prev => ({ ...prev, [sourceId]: error instanceof Error ? error.message : String(error) }))
      } finally {
        setBusyId(null)
      }
    },
    [loadSources, t.knowledge]
  )

  const removeSource = useCallback(
    async (sourceId: string) => {
      const bridge = window.hermesDesktop

      if (!bridge?.knowledge?.remove) {
        return
      }

      setBusyId(sourceId)

      try {
        await bridge.knowledge.remove(sourceId)
        await loadSources()
      } finally {
        setBusyId(null)
      }
    },
    [loadSources]
  )

  const indexable = report?.indexable
  const nameOnly = report?.nameOnly
  const noise = report?.skipped?.noise ?? 0
  const pct = progress.total > 0 ? Math.min(100, Math.round((progress.done / progress.total) * 100)) : 0

  return (
    <section aria-label={t.knowledge.title} className="flex h-full min-h-0 flex-1 flex-col overflow-hidden bg-background">
      <header className="flex shrink-0 flex-col gap-0.5 border-b border-(--ui-stroke-tertiary) px-5 py-3">
        <h1 className="text-sm font-semibold text-foreground">{t.knowledge.title}</h1>
        <p className="text-xs leading-5 text-muted-foreground">{t.knowledge.subtitle}</p>
      </header>

      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-5">
        {(phase === 'idle' || phase === 'scanning') && (
          <button
            className={`flex w-full flex-col items-center justify-center gap-2 rounded-xl border border-dashed px-6 py-10 text-center transition-colors ${
              dragOver
                ? 'border-foreground bg-(--ui-control-hover-background) text-foreground'
                : 'border-(--ui-stroke-secondary) text-muted-foreground hover:border-foreground/60 hover:text-foreground'
            } ${phase === 'scanning' ? 'pointer-events-none opacity-70' : ''}`}
            disabled={phase === 'scanning'}
            onClick={handlePick}
            onDragLeave={() => setDragOver(false)}
            onDragOver={event => {
              event.preventDefault()
              setDragOver(true)
            }}
            onDrop={handleDrop}
            type="button"
          >
            <Codicon className="text-2xl" name={phase === 'scanning' ? 'loading' : 'cloud-upload'} spinning={phase === 'scanning'} />
            <span className="text-sm font-medium">
              {phase === 'scanning' ? t.knowledge.scanning : dragOver ? t.knowledge.dropActive : t.knowledge.dropHint}
            </span>
          </button>
        )}

        {scanError && (
          <p className="text-center text-xs text-red-500">
            {t.knowledge.scanError}: {scanError}
          </p>
        )}

        {phase === 'review' && report && (
          <div className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-bg-elevated) p-4">
            <div className="flex items-center gap-2 border-b border-(--ui-stroke-tertiary) pb-3">
              <Codicon className="shrink-0 text-sm text-muted-foreground" name="folder" />
              <span className="truncate text-sm font-semibold text-foreground" title={report.path}>
                {report.name}
              </span>
            </div>

            <div className="py-3">
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-xs font-semibold text-foreground">{t.knowledge.indexableLabel}</span>
                <span className="shrink-0 text-xs text-muted-foreground">
                  {t.knowledge.filesCount(indexable?.count ?? 0)} · {formatSize(indexable?.size ?? 0)} ·{' '}
                  {t.knowledge.estMinutes(report.estMinutes ?? 0)}
                </span>
              </div>
              {(indexable?.types?.length ?? 0) > 0 && (
                <div className="mt-2 flex flex-wrap gap-1">
                  {indexable!.types.map(item => (
                    <span
                      className="rounded bg-(--ui-control-hover-background) px-1.5 py-0.5 text-[0.6875rem] font-medium text-foreground"
                      key={item.ext}
                    >
                      {item.ext} · {item.count}
                    </span>
                  ))}
                </div>
              )}
            </div>

            {(nameOnly?.count ?? 0) > 0 && (
              <div className="border-t border-(--ui-stroke-tertiary) py-3">
                <div className="flex items-baseline justify-between gap-2">
                  <span className="text-xs font-semibold text-foreground">{t.knowledge.nameOnlyLabel}</span>
                  <span className="shrink-0 text-xs text-muted-foreground">{t.knowledge.filesCount(nameOnly?.count ?? 0)}</span>
                </div>
                {(nameOnly?.types?.length ?? 0) > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {nameOnly!.types.map(item => (
                      <span
                        className="rounded bg-(--ui-control-hover-background) px-1.5 py-0.5 text-[0.6875rem] font-medium text-(--ui-text-secondary)"
                        key={item.ext}
                      >
                        {item.ext} · {item.count}
                      </span>
                    ))}
                  </div>
                )}
                <p className="mt-2 text-[0.6875rem] leading-5 text-muted-foreground">{t.knowledge.nameOnlyHint}</p>
              </div>
            )}

            {(noise > 0 || report.truncated) && (
              <div className="border-t border-(--ui-stroke-tertiary) pt-3 text-[0.6875rem] leading-5 text-muted-foreground">
                {noise > 0 && <p>{t.knowledge.noiseNote(noise)}</p>}
                {report.truncated && <p className="text-amber-500">{t.knowledge.truncatedNote}</p>}
              </div>
            )}

            {ingestError && <p className="pt-2 text-[0.6875rem] text-red-500">{t.knowledge.ingestError}: {ingestError}</p>}

            <div className="mt-3 flex justify-end gap-2">
              <button
                className="rounded-md px-3 py-1.5 text-xs font-medium text-(--ui-text-secondary) hover:text-foreground"
                onClick={reset}
                type="button"
              >
                {t.knowledge.cancel}
              </button>
              <button
                className="rounded-md bg-foreground px-3 py-1.5 text-xs font-medium text-background hover:opacity-90"
                onClick={confirm}
                type="button"
              >
                {t.knowledge.confirm}
              </button>
            </div>
          </div>
        )}

        {phase === 'indexing' && (
          <div className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-bg-elevated) p-4">
            <div className="flex items-center gap-2">
              <Codicon className="text-sm text-muted-foreground" name="loading" spinning />
              <span className="text-sm font-medium text-foreground">{t.knowledge.indexing}</span>
            </div>
            {progress.total > 0 ? (
              <>
                <p className="mt-2 text-xs text-muted-foreground">{t.knowledge.ingestProgress(progress.done, progress.total)}</p>
                <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-(--ui-control-hover-background)">
                  <div className="h-full rounded-full bg-foreground transition-[width] duration-300" style={{ width: `${pct}%` }} />
                </div>
              </>
            ) : (
              <p className="mt-2 text-xs text-muted-foreground">{t.knowledge.startingBackend}</p>
            )}
          </div>
        )}

        {phase === 'done' && result && (
          <div className="flex flex-col items-center gap-3 rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-bg-elevated) p-6 text-center">
            <Codicon className="text-2xl text-green-500" name="check" />
            <p className="text-sm font-medium text-foreground">{t.knowledge.ingestDone(result.indexed, result.nameOnly)}</p>
            <button
              className="rounded-md bg-foreground px-3 py-1.5 text-xs font-medium text-background hover:opacity-90"
              onClick={reset}
              type="button"
            >
              {t.knowledge.doneBtn}
            </button>
          </div>
        )}

        {/* 知识源列表(红框区):已加入的文件夹/文件 + 同步/移除 */}
        <section className="rounded-xl border border-(--ui-stroke-tertiary)">
          <header className="border-b border-(--ui-stroke-tertiary) px-3 py-2 text-xs font-semibold text-foreground">
            {t.knowledge.sourcesTitle}
          </header>
          {sources.length === 0 ? (
            <p className="px-3 py-10 text-center text-xs text-muted-foreground">{t.knowledge.empty}</p>
          ) : (
            <ul className="divide-y divide-(--ui-stroke-tertiary)">
              {sources.map(s => {
                const busy = busyId === s.sourceId

                return (
                  <li className="flex items-center gap-3 px-3 py-2.5" key={s.sourceId}>
                    <Codicon
                      className="shrink-0 text-base text-muted-foreground"
                      name={s.type === 'file' ? 'file' : 'folder'}
                    />
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-xs font-medium text-foreground" title={s.path}>
                        {s.name}
                      </div>
                      <div className="truncate text-[0.6875rem] text-muted-foreground">
                        {t.knowledge.sourceSummary(s.indexed, s.nameOnly)} · {t.knowledge.lastSync} {formatTs(s.lastSyncedTs)}
                      </div>
                      {syncMsg[s.sourceId] && (
                        <div className="truncate text-[0.6875rem] text-(--ui-text-secondary)">{syncMsg[s.sourceId]}</div>
                      )}
                    </div>
                    <button
                      className="flex shrink-0 items-center gap-1 rounded-md px-2 py-1 text-[0.6875rem] font-medium text-(--ui-text-secondary) hover:bg-(--ui-control-hover-background) hover:text-foreground disabled:opacity-50"
                      disabled={busy}
                      onClick={() => void syncSource(s.sourceId)}
                      type="button"
                    >
                      <Codicon name={busy ? 'loading' : 'sync'} size="0.75rem" spinning={busy} />
                      {busy ? t.knowledge.syncing : t.knowledge.sync}
                    </button>
                    <button
                      aria-label={t.knowledge.remove}
                      className="shrink-0 rounded-md p-1 text-(--ui-text-tertiary) hover:text-red-500 disabled:opacity-50"
                      disabled={busy}
                      onClick={() => void removeSource(s.sourceId)}
                      title={t.knowledge.remove}
                      type="button"
                    >
                      <Codicon name="trash" size="0.8rem" />
                    </button>
                  </li>
                )
              })}
            </ul>
          )}
        </section>
      </div>
    </section>
  )
}
