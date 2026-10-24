import { type ReactElement, useCallback, useContext, useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Tip } from '@/components/ui/tooltip'
import type { DesktopStorageEntry, DesktopStorageSummary } from '@/global'
import { useI18n } from '@/i18n'
import { ChevronLeft, Database, FileText, FolderOpen, Loader2, RefreshCw } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { SectionHeading, SettingsBreadcrumbContext } from './primitives'

const UNITS: string[] = ['B', 'KB', 'MB', 'GB', 'TB']

export function formatBytes(bytes: number): string {
  let value: number = bytes
  let unit = 0

  while (value >= 1024 && unit < UNITS.length - 1) {
    value /= 1024
    unit += 1
  }

  return `${unit === 0 ? value : value.toFixed(value >= 10 ? 0 : 1)} ${UNITS[unit]}`
}

export function StorageSection(): ReactElement | null {
  const hasBreadcrumb: boolean = useContext(SettingsBreadcrumbContext)
  const { t } = useI18n()
  const s = t.settings.storageSection
  const bridge: Window['hermesDesktop']['storage'] | undefined = window.hermesDesktop?.storage

  const [summary, setSummary] = useState<DesktopStorageSummary | null>(null)
  const [loading, setLoading] = useState<boolean>(true)
  const [failed, setFailed] = useState<boolean>(false)
  // Folder names below the Hermes home; empty is the home itself.
  const [trail, setTrail] = useState<string[]>([])

  // Finished scans by folder, so going back (or revisiting) is instant. Only
  // Rescan throws them away, because sizes can have changed since.
  const cache = useRef(new Map<string, DesktopStorageSummary>())

  const scan = useCallback(
    (force = false): (() => void) => {
    const key: string = JSON.stringify(trail)
    const cached: DesktopStorageSummary | undefined = force ? undefined : cache.current.get(key)

    if (cached) {
      setSummary(cached)
      setLoading(false)
      setFailed(false)

      return (): void => undefined
    }

    if (force) {
      cache.current.clear()
    }

    let alive = true

    setLoading(true)
    setFailed(false)

    void bridge
      ?.summary(trail)
      .then((result: DesktopStorageSummary): void => {
        cache.current.set(key, result)

        if (alive) {
          setSummary(result)
        }
      })
      .catch((): void => {
        if (alive) {
          setFailed(true)
        }
      })
      .finally((): void => {
        if (alive) {
          setLoading(false)
        }
      })

    return (): void => {
      alive = false
    }
    },
    [bridge, trail]
  )

  useEffect((): (() => void) => scan(), [scan])

  if (!bridge) {
    return null
  }

  const go = (next: string[]): void => {
    // Show a visited folder at once; otherwise drop the old rows so they never
    // appear under the new heading while it loads.
    setSummary(cache.current.get(JSON.stringify(next)) ?? null)
    setTrail(next)
  }

  // Folders drill in; files have nothing to show, so they are selected in the file manager.
  const open = (entry: DesktopStorageEntry): void => {
    if (entry.kind === 'dir') {
      go([...trail, entry.name])
    } else {
      void bridge.reveal(trail, entry.name)
    }
  }

  // Empty folders and files add no information to a size list.
  const entries: DesktopStorageEntry[] = summary?.entries.filter((e: DesktopStorageEntry): boolean => e.bytes > 0) ?? []
  const largest: number = entries[0]?.bytes ?? 0

  return (
    <div className={cn('mx-auto w-full max-w-2xl', !hasBreadcrumb && 'mt-8')}>
      <SectionHeading
        aside={
          <Button disabled={loading} onClick={(): void => void scan(true)} size="sm" variant="outline">
            {loading ? <Loader2 className="animate-spin" /> : <RefreshCw />}
            {s.refresh}
          </Button>
        }
        icon={Database}
        page
        title={s.title}
      />
      {trail.length > 0 && (
        <div className="mb-2 flex items-center gap-2 text-sm">
          <Tip label={s.back}>
            <Button aria-label={s.back} onClick={(): void => go(trail.slice(0, -1))} size="icon-xs" variant="ghost">
              <ChevronLeft />
            </Button>
          </Tip>
          <span className="min-w-0 truncate text-muted-foreground">{[s.home, ...trail].join(' / ')}</span>
        </div>
      )}
      {summary && (
        <p className="mb-3 text-xs text-muted-foreground">
          {s.description(summary.root)} {s.openHint}
        </p>
      )}
      {loading && !summary && <p className="text-sm text-muted-foreground">{s.scanning}</p>}
      {failed && <p className="text-sm text-destructive">{s.failed}</p>}
      {summary && (
        <div className="flex flex-col gap-1 rounded-xl border border-border/60 bg-background/40 px-4 py-3">
          <div className="flex items-baseline justify-between pb-2">
            <span className="text-sm font-medium">{s.total}</span>
            <span className="text-sm font-medium tabular-nums">{formatBytes(summary.totalBytes)}</span>
          </div>
          {entries.length === 0 && <p className="text-xs text-muted-foreground">{s.empty}</p>}
          {entries.map(entry => {
            const Icon = entry.kind === 'dir' ? FolderOpen : FileText

            return (
              <div
                className="flex cursor-pointer flex-col gap-1 rounded-md px-1 py-1 hover:bg-muted/40"
                key={entry.name}
                onClick={(): void => open(entry)}
                onKeyDown={(event): void => {
                  if (event.key === 'Enter') {
                    open(entry)
                  }
                }}
                role="button"
                tabIndex={0}
              >
                <div className="flex items-center justify-between gap-3 text-sm">
                  <span className="flex min-w-0 items-center gap-2">
                    <Tip label={s.openLocation}>
                      <Button
                        aria-label={s.openLocation}
                        onClick={(event): void => {
                          event.stopPropagation()
                          void bridge.reveal(trail, entry.name)
                        }}
                        size="icon-xs"
                        variant="ghost"
                      >
                        <Icon />
                      </Button>
                    </Tip>
                    <span className="truncate" title={entry.kind === 'dir' ? s.folder : s.file}>
                      {entry.name}
                    </span>
                  </span>
                  <span className="shrink-0 tabular-nums text-muted-foreground">{formatBytes(entry.bytes)}</span>
                </div>
                <div className="h-1 overflow-hidden rounded-full bg-muted">
                  <div
                    className="h-full rounded-full bg-primary/60"
                    style={{ width: `${largest > 0 ? Math.max(1, (entry.bytes / largest) * 100) : 0}%` }}
                  />
                </div>
              </div>
            )
          })}
          {summary.truncated && <p className="pt-2 text-xs text-muted-foreground">{s.truncated}</p>}
        </div>
      )}
    </div>
  )
}
