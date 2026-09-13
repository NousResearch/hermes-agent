import { useEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Dialog, DialogContent, DialogDescription, DialogTitle } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { readDesktopDir, setDesktopFsRemotePicker } from '@/lib/desktop-fs'
import { displayPath, pathLeaf } from '@/lib/display-path'
import { cn } from '@/lib/utils'

interface FolderEntry {
  name: string
  path: string
}

function clean(path: string) {
  const value = path.trim()

  if (!value || value === '/') {
    return '/'
  }

  if (/^[a-z]:[\\/]?$/i.test(value)) {
    return value.length === 2 ? `${value}\\` : value
  }

  return value.replace(/[\\/]+$/, '')
}

function separatorFor(path: string) {
  return path.includes('\\') && !path.includes('/') ? '\\' : '/'
}

function lastSeparator(path: string) {
  return Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'))
}

function isAbsolutePath(path: string) {
  return path.startsWith('/') || path.startsWith('~') || path.startsWith('\\\\') || /^[a-z]:[\\/]/i.test(path)
}

function parentDir(path: string) {
  const value = clean(path)

  if (value === '/') {
    return '/'
  }

  const splitAt = lastSeparator(value)

  if (splitAt < 0) {
    return value
  }

  const parent = value.slice(0, splitAt)

  if (/^[a-z]:$/i.test(parent)) {
    return `${parent}${separatorFor(value)}`
  }

  if (value.startsWith('\\\\')) {
    const rootParts = value.split('\\').filter(Boolean)

    if (rootParts.length <= 2) {
      return value
    }
  }

  return parent || '/'
}

function pathName(path: string) {
  return pathLeaf(path) || path
}

function joinPath(base: string, child: string) {
  const root = clean(base)
  const separator = separatorFor(root)

  return `${root === '/' ? '' : root.replace(/[\\/]+$/, '')}${separator}${child}`
}

function queryCandidatePath(query: string, currentPath: string) {
  const value = clean(query.trim())

  return isAbsolutePath(value) ? value : joinPath(currentPath, value)
}

export function planFolderQuery(query: string, currentPath: string): { browsePath: string; filter: string } {
  const value = query.trim()

  if (!value || lastSeparator(value) < 0) {
    return { browsePath: currentPath, filter: value }
  }

  if (/[\\/]$/.test(value)) {
    return { browsePath: clean(value), filter: '' }
  }

  const splitAt = lastSeparator(value)
  const parent = value.slice(0, splitAt)

  return {
    browsePath: isAbsolutePath(value) ? parent || '/' : joinPath(currentPath, parent),
    filter: value.slice(splitAt + 1)
  }
}

function fuzzyScore(value: string, query: string) {
  const haystack = value.toLocaleLowerCase()
  const needle = query.toLocaleLowerCase()

  if (!needle) {
    return 0
  }

  if (haystack.startsWith(needle)) {
    return 3_000 - haystack.length
  }

  const contiguousAt = haystack.indexOf(needle)

  if (contiguousAt >= 0) {
    return 2_000 - contiguousAt * 10 - haystack.length
  }

  let first = -1
  let previous = -1
  let gaps = 0

  for (const character of needle) {
    const index = haystack.indexOf(character, previous + 1)

    if (index < 0) {
      return null
    }

    if (first < 0) {
      first = index
    } else {
      gaps += index - previous - 1
    }

    previous = index
  }

  return 1_000 - first * 10 - gaps * 5 - haystack.length
}

export function filterAndRankFolders<T extends FolderEntry>(entries: T[], query: string): T[] {
  const needle = query.trim()

  if (!needle) {
    return entries
  }

  return entries
    .map((entry, index) => ({ entry, index, score: fuzzyScore(entry.name, needle) }))
    .filter((match): match is { entry: T; index: number; score: number } => match.score !== null)
    .sort((a, b) => b.score - a.score || a.index - b.index)
    .map(match => match.entry)
}

function directoryEntries(result: Awaited<ReturnType<typeof readDesktopDir>>): FolderEntry[] {
  return result.entries.filter(entry => entry.isDirectory).map(entry => ({ name: entry.name, path: entry.path }))
}

function pathCrumbs(path: string) {
  const value = clean(path)
  let root = '/'
  let rest = value

  if (/^[a-z]:[\\/]/i.test(value)) {
    root = value.slice(0, 3)
    rest = value.slice(3)
  } else if (value.startsWith('\\\\')) {
    const parts = value.split('\\').filter(Boolean)
    root = `\\\\${parts.slice(0, 2).join('\\')}`
    rest = parts.slice(2).join('\\')
  } else if (value.startsWith('~')) {
    root = '~'
    rest = value.slice(1)
  }

  const out = [{ label: root, path: root }]
  let acc = root

  for (const part of rest.split(/[\\/]/).filter(Boolean)) {
    acc = joinPath(acc, part)
    out.push({ label: part, path: acc })
  }

  return out
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
  const [pathQuery, setPathQuery] = useState('/')
  const [queryFilter, setQueryFilter] = useState('')
  const [entries, setEntries] = useState<FolderEntry[]>([])
  const [activeIndex, setActiveIndex] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const directoryRequest = useRef(0)

  useEffect(() => {
    setDesktopFsRemotePicker({
      selectPaths: options =>
        new Promise(resolve => {
          const defaultPath = clean(options?.defaultPath || '/')
          setCurrentPath(defaultPath)
          setPathQuery(defaultPath)
          setQueryFilter('')
          setActiveIndex(0)
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
    const request = ++directoryRequest.current
    setLoading(true)
    setError(null)
    setEntries([])

    void readDesktopDir(currentPath)
      .then(result => {
        if (!active || request !== directoryRequest.current) {
          return
        }

        if (result.error) {
          setError(result.error)
          setEntries([])
        } else {
          setEntries(directoryEntries(result))
        }
      })
      .catch(err => {
        if (active && request === directoryRequest.current) {
          setError(err instanceof Error ? err.message : String(err))
          setEntries([])
        }
      })
      .finally(() => {
        if (active && request === directoryRequest.current) {
          setLoading(false)
        }
      })

    return () => {
      active = false
    }
  }, [currentPath, pending])

  useEffect(() => {
    if (!pending) {
      return
    }

    const typed = pathQuery.trim()

    if (!typed || clean(typed) === clean(currentPath) || lastSeparator(typed) < 0) {
      setQueryFilter(clean(typed) === clean(currentPath) ? '' : typed)
      setActiveIndex(0)
      setError(null)
      setLoading(false)

      return
    }

    let active = true

    const timer = window.setTimeout(() => {
      const request = ++directoryRequest.current
      setLoading(true)
      setError(null)

      void (async () => {
        const candidatePath = queryCandidatePath(typed, currentPath)
        const candidate = await readDesktopDir(candidatePath)

        if (!active || request !== directoryRequest.current) {
          return
        }

        if (!candidate.error) {
          const resolvedPath = candidate.entries[0] ? parentDir(candidate.entries[0].path) : candidatePath
          setCurrentPath(resolvedPath)
          setPathQuery(resolvedPath)
          setQueryFilter('')
          setEntries(directoryEntries(candidate))
          setActiveIndex(0)
          setLoading(false)

          return
        }

        const plan = planFolderQuery(typed, currentPath)
        const parent = await readDesktopDir(plan.browsePath)

        if (!active || request !== directoryRequest.current) {
          return
        }

        if (parent.error) {
          setError(parent.error)
          setEntries([])
          setQueryFilter('')
        } else {
          const resolvedPath = parent.entries[0] ? parentDir(parent.entries[0].path) : plan.browsePath
          setCurrentPath(resolvedPath)
          setPathQuery(queryCandidatePath(typed, currentPath))
          setQueryFilter(plan.filter)
          setEntries(directoryEntries(parent))
          setActiveIndex(0)
        }

        setLoading(false)
      })().catch(err => {
        if (active && request === directoryRequest.current) {
          setError(err instanceof Error ? err.message : String(err))
          setEntries([])
          setLoading(false)
        }
      })
    }, 120)

    return () => {
      active = false
      window.clearTimeout(timer)
    }
  }, [currentPath, pathQuery, pending])

  const crumbs = useMemo(() => pathCrumbs(currentPath), [currentPath])

  const visibleEntries = useMemo(() => filterAndRankFolders(entries, queryFilter), [entries, queryFilter])
  const selectionReady = !loading && !error && clean(pathQuery) === clean(currentPath)

  useEffect(() => {
    setActiveIndex(index => Math.min(index, Math.max(0, visibleEntries.length - 1)))
  }, [visibleEntries.length])

  const navigate = (path: string) => {
    setPathQuery(path)
    setCurrentPath(path)
    setQueryFilter('')
    setEntries([])
    setActiveIndex(0)
  }

  const close = (paths: string[] = []) => {
    pending?.resolve(paths)
    setPending(null)
    setEntries([])
    setError(null)
  }

  return (
    <Dialog onOpenChange={open => !open && close()} open={Boolean(pending)}>
      <DialogContent
        bodyClassName="flex min-h-0 flex-col gap-0 overflow-hidden p-0"
        className="h-[min(36rem,calc(100vh-4rem))] max-w-lg"
      >
        <div className="shrink-0 border-b border-border/70 px-4 py-3">
          <DialogTitle className="text-sm">{pending?.title || r.remotePickerTitle}</DialogTitle>
          <DialogDescription className="mt-1 text-xs">{r.remotePickerDescription}</DialogDescription>
        </div>

        <div className="flex min-h-0 flex-1 flex-col">
          <div className="shrink-0 border-b border-border/50 p-3 pb-2">
            <Input
              aria-activedescendant={
                queryFilter && visibleEntries[activeIndex] ? `remote-folder-option-${activeIndex}` : undefined
              }
              aria-autocomplete="list"
              aria-controls="remote-folder-results"
              aria-expanded={Boolean(queryFilter && visibleEntries.length)}
              aria-label={r.remotePickerPathLabel}
              autoFocus
              className="font-mono"
              onChange={event => setPathQuery(event.target.value)}
              onFocus={event => event.currentTarget.select()}
              onKeyDown={event => {
                if (event.key === 'ArrowDown' && visibleEntries.length) {
                  event.preventDefault()
                  setActiveIndex(index => (index + 1) % visibleEntries.length)
                } else if (event.key === 'ArrowUp' && visibleEntries.length) {
                  event.preventDefault()
                  setActiveIndex(index => (index - 1 + visibleEntries.length) % visibleEntries.length)
                } else if (event.key === 'Enter') {
                  event.preventDefault()

                  if (queryFilter && visibleEntries[activeIndex]) {
                    navigate(visibleEntries[activeIndex].path)
                  } else if (selectionReady) {
                    close([currentPath])
                  }
                }
              }}
              placeholder={r.remotePickerPathPlaceholder}
              prefix={<Codicon name="search" size="0.8rem" />}
              role="combobox"
              size="sm"
              value={pathQuery}
            />
          </div>

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
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto p-2">
            {!queryFilter && (
              <FolderRow
                disabled={parentDir(currentPath) === currentPath}
                name=".."
                onClick={() => navigate(parentDir(currentPath))}
              />
            )}
            {loading ? (
              <div aria-live="polite" className="flex items-center gap-2 px-2 py-3 text-xs text-muted-foreground">
                <Codicon name="loading" size="0.8rem" spinning />
                {r.loadingFiles}
              </div>
            ) : error ? (
              <div aria-live="polite" className="px-2 py-3 text-xs text-destructive">
                {r.unreadableBody(error)}
              </div>
            ) : visibleEntries.length === 0 ? (
              <div aria-live="polite" className="px-2 py-3 text-xs text-muted-foreground">
                {r.emptyBody}
              </div>
            ) : (
              <div id="remote-folder-results" role="listbox">
                {visibleEntries.map((entry, index) => (
                  <FolderRow
                    active={Boolean(queryFilter) && index === activeIndex}
                    key={entry.path}
                    name={pathName(entry.path)}
                    onClick={() => navigate(entry.path)}
                    onMouseEnter={() => setActiveIndex(index)}
                    optionId={`remote-folder-option-${index}`}
                  />
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="shrink-0 flex items-center justify-between gap-2 border-t border-border/70 px-4 py-3">
          <div className="min-w-0 truncate text-xs text-muted-foreground">{displayPath(currentPath)}</div>
          <div className="flex shrink-0 items-center gap-2">
            <Button onClick={() => close()} size="sm" variant="ghost">
              {t.common.cancel}
            </Button>
            <Button disabled={!selectionReady} onClick={() => close([currentPath])} size="sm">
              {r.remotePickerSelect}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}

function FolderRow({
  active = false,
  disabled = false,
  name,
  onClick,
  onMouseEnter,
  optionId
}: {
  active?: boolean
  disabled?: boolean
  name: string
  onClick: () => void
  onMouseEnter?: () => void
  optionId?: string
}) {
  return (
    <button
      aria-selected={optionId ? active : undefined}
      className={cn(
        'row-hover flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-xs text-(--ui-text-secondary) hover:text-foreground disabled:pointer-events-none disabled:opacity-40',
        active && 'bg-accent text-accent-foreground'
      )}
      disabled={disabled}
      id={optionId}
      onClick={onClick}
      onMouseEnter={onMouseEnter}
      role={optionId ? 'option' : undefined}
      type="button"
    >
      <Codicon name="folder" size="0.875rem" />
      <span className="min-w-0 truncate">{name}</span>
    </button>
  )
}
