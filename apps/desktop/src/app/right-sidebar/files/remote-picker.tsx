import { useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Dialog, DialogContent, DialogDescription, DialogTitle } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { createDesktopDir, readDesktopDir, setDesktopFsRemotePicker } from '@/lib/desktop-fs'
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

function joinPath(dir: string, name: string) {
  const value = clean(dir)

  return value === '/' ? `/${name}` : `${value}/${name}`
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
  const [creating, setCreating] = useState(false)
  const [newName, setNewName] = useState('')
  const [createError, setCreateError] = useState<string | null>(null)
  const [createBusy, setCreateBusy] = useState(false)

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

        setEntries(
          result.entries.filter(entry => entry.isDirectory).map(entry => ({ name: entry.name, path: entry.path }))
        )
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
    cancelCreate()
  }

  function cancelCreate() {
    setCreating(false)
    setNewName('')
    setCreateError(null)
    setCreateBusy(false)
  }

  // Every user navigation drops a half-typed folder name: the input is scoped
  // to the folder it was opened in, so keeping it across folders would create
  // the folder somewhere the user is no longer looking.
  const navigate = (path: string) => {
    cancelCreate()
    setCurrentPath(path)
  }

  const trimmedName = newName.trim()
  const nameValid = Boolean(trimmedName) && trimmedName !== '.' && trimmedName !== '..' && !trimmedName.includes('/')

  const submitCreate = async () => {
    if (!nameValid || createBusy) {
      return
    }

    setCreateBusy(true)
    setCreateError(null)

    try {
      const result = await createDesktopDir(joinPath(currentPath, trimmedName))
      cancelCreate()
      // Land inside the new folder: it is almost always the folder the user
      // wants to pick, and this proves the create round-tripped.
      setCurrentPath(clean(result.path))
    } catch (err) {
      setCreateError(err instanceof Error ? err.message : String(err))
    } finally {
      setCreateBusy(false)
    }
  }

  return (
    <Dialog onOpenChange={open => !open && close()} open={Boolean(pending)}>
      <DialogContent className="flex h-[min(36rem,calc(100vh-4rem))] max-w-lg flex-col gap-0 overflow-hidden p-0">
        <div className="shrink-0 border-b border-border/70 px-4 py-3">
          <DialogTitle className="text-sm">{pending?.title || r.remotePickerTitle}</DialogTitle>
          <DialogDescription className="mt-1 text-xs">{r.remotePickerDescription}</DialogDescription>
        </div>

        <div className="flex min-h-0 flex-1 flex-col">
          <div className="shrink-0 flex flex-wrap items-center gap-1 border-b border-border/50 px-3 py-2 text-xs text-muted-foreground">
            {crumbs.map((crumb, index) => (
              <button
                className={cn(
                  'rounded px-1.5 py-0.5 hover:bg-muted hover:text-foreground',
                  index === crumbs.length - 1 && 'text-foreground'
                )}
                key={crumb.path}
                onClick={() => navigate(crumb.path)}
                type="button"
              >
                {crumb.label}
              </button>
            ))}
            <button
              aria-label={r.remotePickerNewFolder}
              className="ml-auto rounded p-1 hover:bg-muted hover:text-foreground disabled:pointer-events-none disabled:opacity-40"
              disabled={creating}
              onClick={() => {
                setCreating(true)
                setCreateError(null)
              }}
              title={r.remotePickerNewFolder}
              type="button"
            >
              <Codicon name="new-folder" size="0.875rem" />
            </button>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto p-2">
            <FolderRow
              disabled={currentPath === '/'}
              name=".."
              onClick={() => navigate(parentDir(currentPath))}
            />
            {creating ? (
              <div className="flex items-center gap-1.5 px-2 py-1.5">
                <Codicon name="folder" size="0.875rem" />
                <Input
                  autoFocus
                  className="h-6 flex-1 text-xs"
                  disabled={createBusy}
                  onChange={event => {
                    setNewName(event.target.value)
                    setCreateError(null)
                  }}
                  onKeyDown={event => {
                    if (event.key === 'Enter') {
                      event.preventDefault()
                      void submitCreate()
                    } else if (event.key === 'Escape') {
                      event.preventDefault()
                      cancelCreate()
                    }
                  }}
                  placeholder={r.remotePickerNewFolderPlaceholder}
                  value={newName}
                />
                <button
                  aria-label={t.common.confirm}
                  className="rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground disabled:pointer-events-none disabled:opacity-40"
                  disabled={!nameValid || createBusy}
                  onClick={() => void submitCreate()}
                  title={t.common.confirm}
                  type="button"
                >
                  <Codicon name={createBusy ? 'loading' : 'check'} size="0.8rem" spinning={createBusy} />
                </button>
                <button
                  aria-label={t.common.cancel}
                  className="rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
                  onClick={cancelCreate}
                  title={t.common.cancel}
                  type="button"
                >
                  <Codicon name="close" size="0.8rem" />
                </button>
              </div>
            ) : null}
            {createError ? (
              <div className="px-2 pt-1 pb-2 text-xs text-destructive">{r.remotePickerCreateFailed(createError)}</div>
            ) : null}
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
              entries.map(entry => (
                <FolderRow key={entry.path} name={pathName(entry.path)} onClick={() => navigate(entry.path)} />
              ))
            )}
          </div>
        </div>

        <div className="shrink-0 flex items-center justify-between gap-2 border-t border-border/70 px-4 py-3">
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
      className="row-hover flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-xs text-(--ui-text-secondary) hover:text-foreground disabled:pointer-events-none disabled:opacity-40"
      disabled={disabled}
      onClick={onClick}
      type="button"
    >
      <Codicon name="folder" size="0.875rem" />
      <span className="min-w-0 truncate">{name}</span>
    </button>
  )
}
