import { useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Dialog, DialogContent, DialogDescription, DialogTitle } from '@/components/ui/dialog'
import { useI18n } from '@/i18n'
import { readDesktopDir, setDesktopFsRemotePicker } from '@/lib/desktop-fs'
import { cn } from '@/lib/utils'

function clean(path: string) {
  return path.replace(/\/+$/, '') || '/'
}

function parentDir(path: string) {
  const value = clean(path)
  if (value === '/') {
    return '/'
  }
  const parent = value.slice(0, value.lastIndexOf('/'))
  return parent || '/'
}

function pathName(path: string) {
  return path.split('/').filter(Boolean).pop() || path
}

interface PendingSelection {
  defaultPath: string
  resolve: (paths: string[]) => void
  title: string
}

export function RemoteFolderPicker() {
  const { t } = useI18n()
  const r = t.rightSidebar
  const [pending, setPending] = useState<PendingSelection | null>(null)
  const [currentPath, setCurrentPath] = useState('/')
  const [entries, setEntries] = useState<Array<{ name: string; path: string }>>([])
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setDesktopFsRemotePicker({
      selectPaths: options =>
        new Promise(resolve => {
          const defaultPath = clean(options?.defaultPath || '/')
          setCurrentPath(defaultPath)
          setPending({ defaultPath, resolve, title: options?.title || r.remotePickerTitle })
        })
    })
    return () => setDesktopFsRemotePicker(null)
  }, [r.remotePickerTitle])

  useEffect(() => {
    if (!pending) {
      return
    }

    let active = true
    setLoading(true)
    setError(null)

    void readDesktopDir(currentPath)
      .then(result => {
        if (!active) {
          return
        }
        if (result.error) {
          setError(result.error)
          setEntries([])
          return
        }
        setEntries(result.entries.filter(entry => entry.isDirectory).map(entry => ({ name: entry.name, path: entry.path })))
      })
      .catch(err => {
        if (active) {
          setError(err instanceof Error ? err.message : String(err))
          setEntries([])
        }
      })
      .finally(() => {
        if (active) {
          setLoading(false)
        }
      })

    return () => {
      active = false
    }
  }, [currentPath, pending])

  const crumbs = useMemo(() => {
    const parts = clean(currentPath).split('/').filter(Boolean)
    const out = [{ label: '/', path: '/' }]
    let acc = ''
    for (const part of parts) {
      acc += `/${part}`
      out.push({ label: part, path: acc })
    }
    return out
  }, [currentPath])

  const close = (paths: string[] = []) => {
    pending?.resolve(paths)
    setPending(null)
    setEntries([])
    setError(null)
  }

  return (
    <Dialog onOpenChange={open => !open && close()} open={Boolean(pending)}>
      <DialogContent className="max-w-lg gap-0 overflow-hidden p-0">
        <div className="border-b border-border/70 px-4 py-3">
          <DialogTitle className="text-sm">{pending?.title || r.remotePickerTitle}</DialogTitle>
          <DialogDescription className="mt-1 text-xs">{r.remotePickerDescription}</DialogDescription>
        </div>

        <div className="flex min-h-[22rem] flex-col">
          <div className="flex flex-wrap items-center gap-1 border-b border-border/50 px-3 py-2 text-xs text-muted-foreground">
            {crumbs.map((crumb, index) => (
              <button
                className={cn('rounded px-1.5 py-0.5 hover:bg-muted hover:text-foreground', index === crumbs.length - 1 && 'text-foreground')}
                key={crumb.path}
                onClick={() => setCurrentPath(crumb.path)}
                type="button"
              >
                {crumb.label}
              </button>
            ))}
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto p-2">
            <FolderRow disabled={currentPath === '/'} name=".." onClick={() => setCurrentPath(parentDir(currentPath))} />
            {loading ? (
              <div className="flex items-center gap-2 px-2 py-3 text-xs text-muted-foreground">
                <Codicon name="loading" size="0.8rem" spinning />
                {r.loadingFiles}
              </div>
            ) : error ? (
              <div className="px-2 py-3 text-xs text-destructive">{r.unreadableBody(error)}</div>
            ) : entries.length === 0 ? (
              <div className="px-2 py-3 text-xs text-muted-foreground">{r.emptyBody}</div>
            ) : (
              entries.map(entry => <FolderRow key={entry.path} name={pathName(entry.path)} onClick={() => setCurrentPath(entry.path)} />)
            )}
          </div>
        </div>

        <div className="flex items-center justify-between gap-2 border-t border-border/70 px-4 py-3">
          <div className="min-w-0 truncate text-xs text-muted-foreground">{currentPath}</div>
          <div className="flex shrink-0 items-center gap-2">
            <Button onClick={() => close()} size="sm" variant="ghost">
              {t.common.cancel}
            </Button>
            <Button onClick={() => close([currentPath])} size="sm">
              {r.remotePickerSelect}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}

function FolderRow({ disabled = false, name, onClick }: { disabled?: boolean; name: string; onClick: () => void }) {
  return (
    <button
      className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-xs text-(--ui-text-secondary) hover:bg-(--ui-row-hover-background) hover:text-foreground disabled:pointer-events-none disabled:opacity-40"
      disabled={disabled}
      onClick={onClick}
      type="button"
    >
      <Codicon name="folder" size="0.875rem" />
      <span className="min-w-0 truncate">{name}</span>
    </button>
  )
}
