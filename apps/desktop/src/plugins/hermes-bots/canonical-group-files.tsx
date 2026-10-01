/** Files browser for a gateway room: newest first, search, exact same-name versions, verified Save. */
import {
  Button,
  Codicon,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  EmptyState,
  ErrorState,
  Loader,
  SearchField,
  Tip
} from '@hermes/plugin-sdk'
import { useCallback, useEffect, useRef, useState } from 'react'

import {
  type CanonicalFile,
  CanonicalFilesError,
  type CanonicalFilesPage,
  FILES_MAX_QUERY,
  type FilesFailure,
  filesFailure,
  listCanonicalFiles,
  saveCanonicalFile
} from './canonical-files-client'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import type { CanonicalGroupBinding } from './canonical-groups'

type Labels = ReturnType<typeof useCanonicalGroupLabels>

interface Listing {
  pages: CanonicalFilesPage[]
  index: number
  query: string
  loading: boolean
  failure: FilesFailure | null
}

/** True when `next` comes strictly after `previous` in newest-first, message-order listing. */
function follows(previous: CanonicalFile, next: CanonicalFile) {
  return next.seq < previous.seq || (next.seq === previous.seq && next.index > previous.index)
}

/** One dialog lifetime. A new read retires the one before it; closing retires them all. */
function useCanonicalFiles(binding: CanonicalGroupBinding) {
  const [listing, setListing] = useState<Listing>({ pages: [], index: 0, query: '', loading: true, failure: null })
  // Synchronous ownership of the current read, not a mirror of an external store.
  const model = useRef({ listing, generation: 0, controller: new AbortController(), again: () => {} })

  const publish = useCallback((patch: Partial<Listing>) => {
    model.current.listing = { ...model.current.listing, ...patch }
    setListing(model.current.listing)
  }, [])

  const load = useCallback(async (mode: 'latest' | 'older') => {
    const current = model.current
    const before = current.listing
    const held = before.pages[before.index]
    current.controller.abort()
    const controller = current.controller = new AbortController()
    const generation = ++current.generation
    current.again = () => void load(mode)
    publish({ loading: true, failure: null, ...(mode === 'latest' ? { pages: [], index: 0 } : {}) })

    try {
      const page = await listCanonicalFiles(binding, {
        cursor: mode === 'older' ? held?.nextCursor ?? undefined : undefined,
        query: before.query.trim() || undefined
      }, controller.signal)

      if (generation !== current.generation) {return}

      if (mode === 'older' && held) {
        const last = before.pages.slice(0, before.index + 1).flatMap(earlier => earlier.items).at(-1)

        // A continuation must extend the same snapshot and authority, never repeat or loop.
        if (page.snapshotSeq !== held.snapshotSeq || page.authority !== held.authority ||
          (page.nextCursor !== null && before.pages.some(earlier => earlier.nextCursor === page.nextCursor)) ||
          (last && page.items.length > 0 && !follows(last, page.items[0]))) {
          throw new CanonicalFilesError('cursor')
        }

        publish({ loading: false, pages: [...before.pages.slice(0, before.index + 1), page], index: before.index + 1 })
      } else {
        publish({ loading: false, pages: [page], index: 0 })
      }
    } catch (error) {
      if (generation === current.generation && !controller.signal.aborted) {
        publish({ loading: false, failure: filesFailure(error) })
      }
    }
  }, [binding, publish])

  useEffect(() => {
    const current = model.current
    const timer = setTimeout(() => void load('latest'), listing.query.trim() ? 250 : 0)

    return () => {
      clearTimeout(timer)
      current.generation++
      current.controller.abort()
    }
  }, [listing.query, load])

  const move = (step: number) => {
    const { listing: current } = model.current
    const index = current.index + step

    if (current.loading || index < 0) {return}

    if (current.pages[index]) {publish({ index, failure: null })}
    else if (step > 0 && current.pages[current.index]?.nextCursor) {void load('older')}
  }

  return {
    ...listing,
    page: listing.pages[listing.index] as CanonicalFilesPage | undefined,
    setQuery: (value: string) => {
      const query = [...value].slice(0, FILES_MAX_QUERY).join('')

      if (query !== model.current.listing.query) {publish({ query, pages: [], index: 0, loading: true, failure: null })}
    },
    older: () => move(1),
    newer: () => move(-1),
    latest: () => void load('latest'),
    retry: () => model.current.again()
  }
}

function fileSize(bytes: number, locale?: string) {
  const unit = bytes < 1000 ? 'byte' : bytes < 1_000_000 ? 'kilobyte' : 'megabyte'
  const value = unit === 'byte' ? bytes : bytes / (unit === 'kilobyte' ? 1000 : 1_000_000)

  return new Intl.NumberFormat(locale, { style: 'unit', unit, unitDisplay: 'short',
    maximumFractionDigits: value < 10 ? 1 : 0 }).format(value)
}

function rowProblem(failure: FilesFailure, labels: Labels) {
  return failure === 'verification' ? labels.fileVerificationFailed : failure === 'timeout' ? labels.fileTimeout
    : failure === 'access' ? labels.filesAccess : failure === 'unavailable' ? labels.filesUnavailable
      : labels.fileDownloadFailed
}

function FileRow({ binding, file, labels, active, preciseTime, onFocus }: {
  binding: CanonicalGroupBinding; file: CanonicalFile; labels: Labels; active: boolean; preciseTime: boolean
  onFocus: () => void
}) {
  const [saving, setSaving] = useState<{ pending: boolean; failure: FilesFailure | null }>({ pending: false, failure: null })
  const request = useRef<AbortController | null>(null)

  // Leaving the page or closing the dialog retires a Save that has not delivered its bytes yet.
  useEffect(() => () => request.current?.abort(), [])

  const save = async () => {
    if (request.current) {return}
    const controller = request.current = new AbortController()
    setSaving({ pending: true, failure: null })

    try {
      await saveCanonicalFile(binding, file, controller.signal)

      if (!controller.signal.aborted) {setSaving({ pending: false, failure: null })}
    } catch (error) {
      if (!controller.signal.aborted) {setSaving({ pending: false, failure: filesFailure(error) })}
    } finally {
      if (request.current === controller) {request.current = null}
    }
  }

  const date = new Date(file.sharedAt * 1000)
  const sameDay = date.toDateString() === new Date().toDateString()

  // Same-name versions on one page are told apart by sharer, size and time to the second.
  const time = new Intl.DateTimeFormat(labels.locale, {
    ...(sameDay ? {} : { year: 'numeric', month: 'short', day: 'numeric' } as const),
    hour: 'numeric', minute: '2-digit', ...(preciseTime ? { second: '2-digit' } as const : {})
  }).format(date)

  const sharer = file.sharer.kind === 'user' && file.sharer.label === 'You' ? labels.filesYou : file.sharer.label
  const type = (file.name.includes('.') ? file.name.split('.').pop() : file.mime.split('/')[1])?.toUpperCase() ?? ''
  const details = ` · ${time} · ${type} · ${fileSize(file.size, labels.locale)}`

  return <div aria-label={`${file.name} · ${sharer}${details}`} className="min-w-0 py-1.5 outline-none focus-visible:ring-1"
    data-file-row onFocus={onFocus} role="listitem" tabIndex={active ? 0 : -1}>
    <div className="flex min-w-0 items-center gap-2">
      <Codicon className="shrink-0" name={file.kind === 'pdf' ? 'file-pdf' : file.kind === 'image' ? 'file-media' : 'file'} />
      <div className="min-w-0 flex-1">
        <bdi className="block truncate text-xs font-medium" title={file.name}>{file.name}</bdi>
        <div className="truncate text-[0.65rem]"><bdi>{sharer}</bdi>{details}</div>
      </div>
      <Tip label={`${labels.download}: ${file.name}`}>
        <Button aria-busy={saving.pending} aria-label={`${labels.download}: ${file.name}`} data-file-save
          disabled={saving.pending} onClick={() => void save()} size="icon-xs" type="button" variant="ghost">
          <Codicon name={saving.pending ? 'loading' : 'cloud-download'} spinning={saving.pending} />
        </Button>
      </Tip>
    </div>
    {saving.failure && <div className="mt-1 flex flex-wrap items-center gap-2 text-xs" role="alert">
      <span>{rowProblem(saving.failure, labels)}</span>
      <Button onClick={() => void save()} size="inline" type="button" variant="textStrong">{labels.retry}</Button>
    </div>}
  </div>
}

function FileRows({ binding, items, labels, loading }: {
  binding: CanonicalGroupBinding; items: CanonicalFile[]; labels: Labels; loading: boolean
}) {
  const [active, setActive] = useState(0)

  return <div aria-busy={loading} className="min-h-0 flex-1 overflow-y-auto" onKeyDown={event => {
    const rows = Array.from(event.currentTarget.querySelectorAll<HTMLElement>('[data-file-row]'))
    const row = (event.target as HTMLElement).closest<HTMLElement>('[data-file-row]')

    if (!row) {return}

    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault()
      rows[Math.max(0, Math.min(rows.length - 1, rows.indexOf(row) + (event.key === 'ArrowDown' ? 1 : -1)))]?.focus()
    } else if (event.key === 'Enter' && event.target === row) {
      event.preventDefault()
      row.querySelector<HTMLButtonElement>('[data-file-save]')?.click()
    }
  }} role="list">
    {items.map((file, index) => <FileRow active={active === index} binding={binding} file={file}
      key={`${file.eventId}:${file.attachmentId}`} labels={labels} onFocus={() => setActive(index)}
      preciseTime={items.some(other => other !== file && other.name === file.name)} />)}
  </div>
}

function FilesDialog({ binding, roomName, latestFileSeq, onClose }: {
  binding: CanonicalGroupBinding; roomName: string; latestFileSeq: number; onClose: () => void
}) {
  const labels = useCanonicalGroupLabels()
  const files = useCanonicalFiles(binding)
  const search = useRef<HTMLInputElement>(null)
  const { page, failure } = files
  const newer = files.pages[0] !== undefined && latestFileSeq > files.pages[0].snapshotSeq

  const problem = failure === 'cursor' ? labels.filesExpired : failure === 'access' ? labels.filesAccess
    : failure === 'unavailable' || failure === 'timeout' ? labels.filesUnavailable : labels.filesError

  const recover = failure === 'cursor'
    ? <Button onClick={files.latest} size="inline" type="button" variant="textStrong">{labels.showLatest}</Button>
    : failure !== 'access' && <Button onClick={files.retry} size="inline" type="button" variant="textStrong">{labels.retry}</Button>

  const body = page?.items.length ? <FileRows binding={binding} items={page.items} key={`${files.index}:${page.snapshotSeq}`}
    labels={labels} loading={files.loading} />
    : files.loading ? <Loader className="m-auto size-16" label={labels.filesLoading} type="lemniscate-bloom" />
      : failure ? <ErrorState className="my-auto" title={<p className="text-sm font-medium">{problem}</p>}>{recover}</ErrorState>
        : <div className="my-auto">
          <EmptyState title={page?.nextCursor || files.index > 0 ? labels.filesPageEmpty
            : files.query ? labels.filesNoResults : labels.filesEmpty} />
          {files.query && <div className="flex justify-center">
            <Button onClick={() => files.setQuery('')} size="inline" type="button" variant="textStrong">{labels.filesClearSearch}</Button>
          </div>}
        </div>

  return <Dialog onOpenChange={value => { if (!value) {onClose()} }} open>
    <DialogContent bodyClassName="flex min-h-0 flex-1 flex-col gap-3" className="h-[min(36rem,85vh)] max-w-xl"
      onKeyDown={event => {
        const target = event.target as HTMLElement

        if (event.key === '/' && !target.matches('input,textarea,[contenteditable="true"]') &&
          !event.ctrlKey && !event.metaKey && !event.altKey) {
          event.preventDefault()
          search.current?.focus()
        } else if (event.key === 'ArrowDown' && target === search.current) {
          const row = event.currentTarget.querySelector<HTMLElement>('[data-file-row]')

          if (row) {
            event.preventDefault()
            row.focus()
          }
        }
      }}
      onOpenAutoFocus={event => { event.preventDefault(); search.current?.focus() }}>
      <DialogHeader>
        <DialogTitle>{labels.files}</DialogTitle>
        <DialogDescription><bdi className="block truncate" title={roomName}>{roomName}</bdi></DialogDescription>
      </DialogHeader>
      <SearchField aria-label={labels.searchFiles} containerClassName="w-full" inputClassName="w-full" inputRef={search}
        loading={files.loading && Boolean(page)} onChange={files.setQuery} placeholder={labels.searchFiles} value={files.query} />
      {body}
      <div className="flex min-h-8 flex-wrap items-center justify-end gap-2">
        {page?.items.length && failure ? <div className="mr-auto flex flex-wrap items-center gap-2 text-xs" role="status">
          <span>{problem}</span>{recover}
        </div> : null}
        {newer && <Button disabled={files.loading} onClick={files.latest} size="inline" type="button" variant="textStrong">
          {labels.showLatest}</Button>}
        <Tip label={labels.refreshFiles}>
          <Button aria-label={labels.refreshFiles} disabled={files.loading} onClick={files.latest} size="icon-xs"
            type="button" variant="ghost"><Codicon name="refresh" /></Button>
        </Tip>
        <Tip label={labels.newerFiles}>
          <Button aria-label={labels.newerFiles} disabled={files.index === 0 || files.loading} onClick={files.newer}
            size="icon-xs" type="button" variant="ghost"><Codicon name="chevron-left" /></Button>
        </Tip>
        <Tip label={labels.olderFiles}>
          <Button aria-label={labels.olderFiles} disabled={!page?.nextCursor || files.loading} onClick={files.older}
            size="icon-xs" type="button" variant="ghost"><Codicon name="chevron-right" /></Button>
        </Tip>
      </div>
    </DialogContent>
  </Dialog>
}

/** Offered by the room header when the gateway advertises groups.attachment.list. */
export function CanonicalGroupFiles({ binding, roomName, latestFileSeq }: {
  binding: CanonicalGroupBinding; roomName: string; latestFileSeq: number
}) {
  const labels = useCanonicalGroupLabels()
  const [open, setOpen] = useState(false)

  return <>
    <Button onClick={() => setOpen(true)} size="sm" type="button" variant="ghost"><Codicon name="files" />{labels.files}</Button>
    {open && <FilesDialog binding={binding} latestFileSeq={latestFileSeq} onClose={() => setOpen(false)} roomName={roomName} />}
  </>
}
