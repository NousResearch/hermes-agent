/** Files for a gateway room: its shared versions, newest first, searchable, in exact same-name rows,
 *  each saved only as its exact bytes. */
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
import { type RefObject, useCallback, useEffect, useRef, useState } from 'react'

import {
  type CanonicalFile,
  CanonicalFilesError,
  type CanonicalFilesPage,
  FILES_MAX_QUERY,
  type FilesFailure,
  filesFailure,
  filesFollow,
  listCanonicalFiles,
  saveCanonicalFile
} from './canonical-files-client'
import type { CanonicalGroupEvent } from './canonical-group-history'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import { type CanonicalGroupBinding } from './canonical-groups'

type Labels = ReturnType<typeof useCanonicalGroupLabels>

interface Listing {
  pages: CanonicalFilesPage[]
  index: number
  query: string
  loading: boolean
  failure: FilesFailure | null
}

/** The newest room-log message that shared a file: the list offers Show latest when it is newer. */
export function latestFileSeq(events: CanonicalGroupEvent[]) {
  return events.reduce((latest, event) => event.payload.attachments?.length ? Math.max(latest, event.seq) : latest, 0)
}

/** One dialog lifetime. A new read retires the one before it; closing retires them all. `youLabel`
 *  is the reader's word for their own shares, so a search finds what Files shows. */
function useCanonicalFiles(binding: CanonicalGroupBinding, youLabel: string) {
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
        query: before.query.trim() || undefined,
        youLabel
      }, controller.signal)

      if (generation !== current.generation) {return}

      if (mode === 'older' && held) {
        const last = before.pages.slice(0, before.index + 1).flatMap(earlier => earlier.items).at(-1)

        // A continuation must extend the same snapshot and authority, never repeat or loop.
        if (page.snapshotSeq !== held.snapshotSeq || page.authority !== held.authority ||
          (page.nextCursor !== null && before.pages.some(earlier => earlier.nextCursor === page.nextCursor)) ||
          (last && page.items.length > 0 && !filesFollow(last, page.items[0]))) {
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
  }, [binding, publish, youLabel])

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

  return new Intl.NumberFormat(locale, { style: 'unit', unit, unitDisplay: unit === 'byte' ? 'long' : 'short',
    maximumFractionDigits: value < 10 ? 1 : 0 }).format(value)
}

function fileDetails(file: CanonicalFile, labels: Labels) {
  const date = new Date(file.sharedAt * 1000)
  const sameDay = date.toDateString() === new Date().toDateString()

  // Seconds on every page: two versions can share a name, size and minute.
  const time = new Intl.DateTimeFormat(labels.locale, {
    ...(sameDay ? {} : { year: 'numeric', month: 'short', day: 'numeric' } as const),
    hour: 'numeric', minute: '2-digit', second: '2-digit'
  }).format(date)

  // The gateway labels the room owner's own shares `You`.
  const sharer = file.sharer.kind === 'user' && file.sharer.label === 'You' ? labels.filesYou : file.sharer.label
  const type = (file.name.includes('.') ? file.name.split('.').pop() : file.mime.split('/')[1])?.toUpperCase() ?? ''

  return { sharer, details: ` · ${time} · ${type} · ${fileSize(file.size, labels.locale)}` }
}

function saveProblem(failure: FilesFailure, labels: Labels) {
  return failure === 'verification' ? labels.fileVerificationFailed : failure === 'timeout' ? labels.fileTimeout
    : failure === 'access' ? labels.filesAccess : failure === 'unavailable' ? labels.filesUnavailable
      : labels.fileDownloadFailed
}

/** One version's Save. Leaving the page or closing the dialog retires a Save that has not delivered
 *  its bytes yet; a version the gateway says is gone stays listed with Download disabled. */
function useFileSave(binding: CanonicalGroupBinding, file: CanonicalFile) {
  const [saving, setSaving] = useState<{ pending: boolean; failure: FilesFailure | null }>({ pending: false, failure: null })
  const request = useRef<AbortController | null>(null)

  useEffect(() => () => request.current?.abort(), [])

  const unavailable = !file.available || saving.failure === 'missing'

  const save = async () => {
    if (request.current || unavailable) {return}
    const controller = request.current = new AbortController()
    setSaving({ pending: true, failure: null })

    try {
      await saveCanonicalFile(binding, file, controller.signal)

      if (!controller.signal.aborted) {setSaving({ pending: false, failure: null })}
    } catch (error) {
      if (!controller.signal.aborted) {setSaving({ pending: false, failure: filesFailure(error) })}
    } finally {
      request.current = null
    }
  }

  return { ...saving, unavailable, save }
}

function FileRow({ binding, file, labels, active, onFocus }: {
  binding: CanonicalGroupBinding; file: CanonicalFile; labels: Labels; active: boolean; onFocus: () => void
}) {
  const saving = useFileSave(binding, file)
  const { sharer, details } = fileDetails(file, labels)
  const { unavailable } = saving

  return <div aria-label={`${file.name} · ${sharer}${details}${unavailable ? ` · ${labels.fileVersionUnavailable}` : ''}`}
    className="min-w-0 py-1.5 outline-none focus-visible:ring-1" data-file-row onFocus={onFocus} role="listitem"
    tabIndex={active ? 0 : -1}>
    <div className="flex min-w-0 items-center gap-2">
      <Codicon className="shrink-0" name={file.kind === 'pdf' ? 'file-pdf' : file.kind === 'image' ? 'file-media' : 'file'} />
      <div className="min-w-0 flex-1">
        <bdi className="block truncate text-xs font-medium" title={file.name}>{file.name}</bdi>
        <div className="truncate text-[0.65rem]"><bdi>{sharer}</bdi>{details}</div>
      </div>
      <Tip label={unavailable ? labels.fileVersionUnavailable : `${labels.download}: ${file.name}`}>
        <Button aria-busy={saving.pending} aria-label={`${labels.download}: ${file.name}`} data-file-save
          disabled={saving.pending || unavailable} onClick={() => void saving.save()} size="icon-xs" type="button" variant="ghost">
          <Codicon name={saving.pending ? 'loading' : 'cloud-download'} spinning={saving.pending} />
        </Button>
      </Tip>
    </div>
    {unavailable && <p className="mt-1 text-xs text-(--ui-text-tertiary)" role="status">{labels.fileVersionUnavailable}</p>}
    {saving.failure && !unavailable && <div className="mt-1 flex flex-wrap items-center gap-2 text-xs" role="alert">
      <span>{saveProblem(saving.failure, labels)}</span>
      {saving.failure !== 'access' &&
        <Button onClick={() => void saving.save()} size="inline" type="button" variant="textStrong">{labels.retry}</Button>}
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

    if (row && (event.key === 'ArrowDown' || event.key === 'ArrowUp')) {
      event.preventDefault()
      rows[Math.max(0, Math.min(rows.length - 1, rows.indexOf(row) + (event.key === 'ArrowDown' ? 1 : -1)))]?.focus()
    } else if (row && event.key === 'Enter' && event.target === row) {
      // The row's own Download, so a disabled one (busy or unavailable) stays inert.
      event.preventDefault()
      row.querySelector<HTMLButtonElement>('[data-file-save]')?.click()
    }
  }} role="list">
    {items.map((file, index) => <FileRow active={active === index} binding={binding} file={file}
      key={`${file.eventId}:${file.attachmentId}`} labels={labels} onFocus={() => setActive(index)} />)}
  </div>
}

function filesProblem(failure: FilesFailure | null, labels: Labels) {
  return failure === 'cursor' ? labels.filesExpired : failure === 'access' ? labels.filesAccess
    : failure === 'unavailable' || failure === 'timeout' ? labels.filesUnavailable : labels.filesError
}

type FilesListing = ReturnType<typeof useCanonicalFiles>

/** A refused cursor restarts from the newest files; an access refusal has nothing to retry. */
function FilesRecovery({ files, labels }: { files: FilesListing; labels: Labels }) {
  if (files.failure === 'cursor') {
    return <Button onClick={files.latest} size="inline" type="button" variant="textStrong">{labels.showLatest}</Button>
  }

  return files.failure === 'access' ? null
    : <Button onClick={files.retry} size="inline" type="button" variant="textStrong">{labels.retry}</Button>
}

function FilesBody({ binding, files, labels }: { binding: CanonicalGroupBinding; files: FilesListing; labels: Labels }) {
  const { page, failure } = files

  if (page?.items.length) {
    return <FileRows binding={binding} items={page.items} key={`${files.index}:${page.snapshotSeq}`} labels={labels} loading={files.loading} />
  }

  if (files.loading) {return <Loader className="m-auto size-16" label={labels.filesLoading} type="lemniscate-bloom" />}

  if (failure) {
    return <ErrorState className="my-auto" title={<p className="text-sm font-medium">{filesProblem(failure, labels)}</p>}>
      <FilesRecovery files={files} labels={labels} />
    </ErrorState>
  }

  const empty = page?.nextCursor || files.index > 0 ? labels.filesPageEmpty : files.query ? labels.filesNoResults : labels.filesEmpty

  return <div className="my-auto">
    <EmptyState title={empty} />
    {files.query && <div className="flex justify-center">
      <Button onClick={() => files.setQuery('')} size="inline" type="button" variant="textStrong">{labels.filesClearSearch}</Button>
    </div>}
  </div>
}

function FilesFooter({ files, labels, latest }: { files: FilesListing; labels: Labels; latest: number }) {
  const { page, failure } = files
  const newer = files.pages[0] !== undefined && latest > files.pages[0].snapshotSeq

  return <div className="flex min-h-8 flex-wrap items-center justify-end gap-2">
    {page?.items.length && failure ? <div className="mr-auto flex flex-wrap items-center gap-2 text-xs" role="status">
      <span>{filesProblem(failure, labels)}</span><FilesRecovery files={files} labels={labels} />
    </div> : null}
    {newer && <Button disabled={files.loading} onClick={files.latest} size="inline" type="button" variant="textStrong">
      {labels.showLatest}</Button>}
    {[
      { label: labels.refreshFiles, icon: 'refresh', action: files.latest, disabled: false },
      { label: labels.newerFiles, icon: 'chevron-left', action: files.newer, disabled: files.index === 0 },
      { label: labels.olderFiles, icon: 'chevron-right', action: files.older, disabled: !page?.nextCursor }
    ].map(({ label, icon, action, disabled }) => <Tip key={icon} label={label}>
      <Button aria-label={label} disabled={files.loading || disabled} onClick={action} size="icon-xs"
        type="button" variant="ghost"><Codicon name={icon} /></Button>
    </Tip>)}
  </div>
}

function FilesDialog({ binding, roomName, latest, onClose, opener }: {
  binding: CanonicalGroupBinding; roomName: string; latest: number; onClose: () => void; opener: RefObject<HTMLButtonElement | null>
}) {
  const labels = useCanonicalGroupLabels()
  const files = useCanonicalFiles(binding, labels.filesYou)
  const search = useRef<HTMLInputElement>(null)

  return <Dialog onOpenChange={value => { if (!value) {onClose()} }} open>
    <DialogContent bodyClassName="flex min-h-0 flex-1 flex-col gap-3" className="h-[min(36rem,85vh)] max-w-xl"
      onCloseAutoFocus={event => {
        if (opener.current?.isConnected) {
          event.preventDefault()
          opener.current.focus()
        }
      }}
      onKeyDown={event => {
        const target = event.target as HTMLElement

        if (event.key === '/' && !target.matches('input,textarea,[contenteditable="true"]') &&
          !event.ctrlKey && !event.metaKey && !event.altKey) {
          event.preventDefault()
          search.current?.focus()
        } else if (event.key === 'ArrowDown' && target === search.current && !event.nativeEvent.isComposing) {
          // An input method composing in the search field keeps its own arrow keys.
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
        loading={files.loading && Boolean(files.page)} onChange={files.setQuery} placeholder={labels.searchFiles} value={files.query} />
      <FilesBody binding={binding} files={files} labels={labels} />
      <FilesFooter files={files} labels={labels} latest={latest} />
    </DialogContent>
  </Dialog>
}

function FilesButton({ binding, roomName, latest }: { binding: CanonicalGroupBinding; roomName: string; latest: number }) {
  const labels = useCanonicalGroupLabels()
  const [open, setOpen] = useState(false)
  const opener = useRef<HTMLButtonElement>(null)

  return <>
    <Tip label={labels.files}>
      <Button aria-label={labels.files} onClick={() => setOpen(true)} ref={opener} size="icon-sm" type="button" variant="ghost">
        <Codicon name="files" />
      </Button>
    </Tip>
    {open && <FilesDialog binding={binding} latest={latest} onClose={() => setOpen(false)} opener={opener} roomName={roomName} />}
  </>
}

/** The room header's Files control: offered only when the gateway lists `groups.attachment.list`, and
 *  only while the room is visible (hiding it closes the dialog and retires its reads). */
export function CanonicalGroupFiles({ binding, roomName, latest, visible, offered }: {
  binding: CanonicalGroupBinding; roomName?: string; latest: number; visible: boolean; offered: boolean
}) {
  return offered && visible
    ? <FilesButton binding={binding} latest={latest} roomName={roomName || binding.roomName || ''} /> : null
}
