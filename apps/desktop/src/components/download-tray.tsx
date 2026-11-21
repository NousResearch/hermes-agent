import { useStore } from '@nanostores/react'
import { ChevronDown, ChevronUp, Download, FolderOpen, Pause, Play, X } from 'lucide-react'
import { useEffect, useState } from 'react'

import { useI18n } from '@/i18n'
import {
  $downloads,
  cancelDownload,
  dismissDownload,
  ensureDownloadWatch,
  formatBytes,
  formatEta,
  formatSpeed,
  isDownloadActive,
  pauseDownload,
  resumeDownload
} from '@/store/downloads'

/**
 * Bottom-right download tray.
 *
 * Every row reports what Chromium's native download withheld: a real byte
 * progress bar, the live rate, the ETA, and how many Range connections are
 * running — plus pause/resume on a resumable transfer. Mounted once at the app
 * root; it renders nothing while the list is empty.
 */
export function DownloadTray() {
  const { t } = useI18n()
  const rows = useStore($downloads)
  const [collapsed, setCollapsed] = useState(false)

  useEffect(() => {
    ensureDownloadWatch()
  }, [])

  if (rows.length === 0) {
    return null
  }

  const activeCount = rows.filter(isDownloadActive).length
  const heading = activeCount > 0 ? t.downloads.activeHeading(activeCount) : t.downloads.heading(rows.length)

  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-50 w-[22rem] max-w-[calc(100vw-2rem)]">
      <div className="pointer-events-auto overflow-hidden rounded-xl border border-border bg-background/95 shadow-lg backdrop-blur">
        <button
          className="flex w-full items-center justify-between gap-2 border-b border-border/60 px-3 py-2 text-left"
          onClick={() => setCollapsed(value => !value)}
          type="button"
        >
          <span className="flex items-center gap-2 text-xs font-semibold text-foreground">
            <Download className="size-3.5" />
            {heading}
          </span>
          {collapsed ? <ChevronUp className="size-3.5" /> : <ChevronDown className="size-3.5" />}
        </button>

        {!collapsed && (
          <div className="max-h-[18rem] divide-y divide-border/60 overflow-y-auto">
            {rows.map(row => (
              <DownloadRow key={row.id} row={row} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function DownloadRow({ row }: { row: ReturnType<typeof $downloads.get>[number] }) {
  const { t } = useI18n()
  const ratio = row.totalBytes > 0 ? Math.min(1, row.receivedBytes / row.totalBytes) : 0
  const percent = Math.round(ratio * 100)
  const active = isDownloadActive(row)

  return (
    <div className="px-3 py-2">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <div className="truncate text-[0.7rem] font-medium text-foreground" title={row.savePath}>
            {row.filename}
          </div>
          <div className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[0.62rem] text-muted-foreground">
            <span className="tabular-nums">
              {formatBytes(row.receivedBytes)}
              {row.totalBytes > 0 ? ` / ${formatBytes(row.totalBytes)}` : ''}
            </span>
            {active && (
              <>
                <span className="tabular-nums">{formatSpeed(row.speedBps)}</span>
                <span className="tabular-nums">{formatEta(row.etaMs)}</span>
                <span className="rounded bg-muted px-1 py-px">
                  {row.ranged ? t.downloads.connections(row.connections) : t.downloads.singleStream}
                </span>
              </>
            )}
            {row.status === 'completed' && <span>{t.downloads.completed}</span>}
            {row.status === 'paused' && <span>{t.downloads.paused}</span>}
            {row.status === 'canceled' && <span>{t.downloads.canceled}</span>}
            {row.status === 'failed' && <span className="text-destructive">{row.error || t.downloads.failed}</span>}
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-1">
          {row.status === 'downloading' && (
            <IconButton label={t.downloads.pause} onClick={() => void pauseDownload(row.id)}>
              <Pause className="size-3" />
            </IconButton>
          )}
          {(row.status === 'paused' || row.status === 'failed') && (
            <IconButton label={t.downloads.resume} onClick={() => void resumeDownload(row.id)}>
              <Play className="size-3" />
            </IconButton>
          )}
          {active && (
            <IconButton label={t.downloads.cancel} onClick={() => void cancelDownload(row.id)}>
              <X className="size-3" />
            </IconButton>
          )}
          {row.status === 'completed' && (
            <IconButton
              label={t.downloads.showInFolder}
              onClick={() => void window.hermesDesktop?.openExternal?.(`file://${row.savePath}`)}
            >
              <FolderOpen className="size-3" />
            </IconButton>
          )}
          {!active && (
            <IconButton label={t.downloads.dismiss} onClick={() => void dismissDownload(row.id)}>
              <X className="size-3" />
            </IconButton>
          )}
        </div>
      </div>

      <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-muted" role="progressbar">
        <div
          className={`h-full rounded-full transition-[width] duration-200 ${
            row.status === 'failed' ? 'bg-destructive' : 'bg-primary'
          }`}
          style={{ width: `${row.totalBytes > 0 ? percent : active ? 8 : 100}%` }}
        />
      </div>
      {row.totalBytes > 0 && (
        <div className="mt-0.5 text-right text-[0.58rem] tabular-nums text-muted-foreground">{percent}%</div>
      )}
    </div>
  )
}

function IconButton({
  children,
  label,
  onClick
}: {
  children: React.ReactNode
  label: string
  onClick: () => void
}) {
  return (
    <button
      aria-label={label}
      className="grid size-6 place-items-center rounded-md border border-border/70 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
      onClick={onClick}
      title={label}
      type="button"
    >
      {children}
    </button>
  )
}
