/** Capability-gated shared-files control and bounded snapshot browser. */

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
  Tip,
  useI18n,
  useValue
} from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'

import { GroupAttachmentDownload } from './group-attachment-download'
import { groupChatHostedGateway } from './group-chat'
import {
  GROUP_FILES_MAX_QUERY_LENGTH,
  GROUP_FILES_PAGE_SIZE,
  type GroupFilesListInput,
  type GroupFilesPage,
  isGroupFilesCursorError,
  listHostedGroupFiles,
  validateGroupFilesContinuation
} from './group-files-client'
import { $hostedRoomCapabilities } from './hosted-room-capability-state'
import { useBots } from './i18n'
import type { GroupChat, GroupMessage } from './types'

export type GroupFilesAvailability = 'available' | 'offline' | 'unavailable'
type GroupFilesLoadState = 'error' | 'loading' | 'ready'
type GroupFilesLoader = (group: string, input?: GroupFilesListInput) => Promise<GroupFilesPage>

export function groupFilesAvailability(
  room: GroupChat,
  capabilities: ReturnType<typeof $hostedRoomCapabilities.get>
): GroupFilesAvailability {
  const authorityId = groupChatHostedGateway(room)
  const capability = capabilities[String(room.hostedConnectionId || '')]

  if (!authorityId || !room.roomId || !capability || room.hostedStatus?.state === 'deleted') {
    return 'unavailable'
  }

  if (capability.kind === 'transient-failure') {
    return 'offline'
  }

  return capability.authorityId === authorityId && capability.limits.attachmentList === true
    ? 'available'
    : 'unavailable'
}

function formatFileSize(bytes: number, locale: string) {
  if (bytes < 1000) {
    return new Intl.NumberFormat(locale, { style: 'unit', unit: 'byte', unitDisplay: 'short' }).format(bytes)
  }

  const units = [
    ['kilobyte', 1_000],
    ['megabyte', 1_000_000]
  ] as const

  const [unit, divisor] = units[bytes < 1_000_000 ? 0 : 1]

  return new Intl.NumberFormat(locale, {
    maximumFractionDigits: bytes < divisor * 10 ? 1 : 0,
    style: 'unit',
    unit,
    unitDisplay: 'short'
  }).format(bytes / divisor)
}

function formatFileType(mime: string, name: string) {
  const subtype = mime.split('/')[1]?.split(/[;+]/)[0]
  const extension = name.includes('.') ? name.split('.').pop() : ''

  return String(extension || subtype || mime).toLocaleUpperCase()
}

interface SharedFilesDialogProps {
  availability: GroupFilesAvailability
  group: string
  latestSeq: number
  loadPage?: GroupFilesLoader
  onClose: () => void
  open: boolean
  roomId: string
}

export function SharedFilesDialog({
  availability,
  group,
  latestSeq,
  loadPage = listHostedGroupFiles,
  onClose,
  open,
  roomId
}: SharedFilesDialogProps) {
  const b = useBots()
  const { locale } = useI18n()
  const [draftQuery, setDraftQuery] = useState('')
  const query = draftQuery.trim()
  const [pages, setPages] = useState<GroupFilesPage[]>([])
  const [pageIndex, setPageIndex] = useState(0)
  const [loadState, setLoadState] = useState<GroupFilesLoadState>('loading')
  const [pageFailed, setPageFailed] = useState(false)
  const [cursorExpired, setCursorExpired] = useState(false)
  const [refresh, setRefresh] = useState(0)
  const requestGeneration = useRef(0)
  const page = pages[pageIndex]

  // eslint-disable-next-line no-restricted-syntax -- invalidates requests for a new dialog scope, not a reactive-value mirror
  useEffect(() => {
    requestGeneration.current += 1
    setDraftQuery('')
    setPages([])
    setPageIndex(0)
    setPageFailed(false)
    setCursorExpired(false)
  }, [group, open, roomId])

  useEffect(
    () => () => {
      requestGeneration.current += 1
    },
    []
  )

  // eslint-disable-next-line no-restricted-syntax -- request generation fences asynchronous results, not a reactive-value mirror
  useEffect(() => {
    const generation = ++requestGeneration.current

    if (!open) {
      return
    }

    if (availability !== 'available') {
      setLoadState('ready')

      return
    }

    setLoadState('loading')
    setPageFailed(false)
    setCursorExpired(false)

    const timer = setTimeout(
      () => {
        void loadPage(group, { limit: GROUP_FILES_PAGE_SIZE, ...(query ? { query } : {}) }).then(
          next => {
            if (requestGeneration.current !== generation) {
              return
            }

            setPages([next])
            setPageIndex(0)
            setLoadState('ready')
          },
          () => {
            if (requestGeneration.current === generation) {
              setLoadState('error')
            }
          }
        )
      },
      query ? 250 : 0
    )

    return () => {
      clearTimeout(timer)
      requestGeneration.current += 1
    }
  }, [availability, draftQuery, group, loadPage, open, query, refresh, roomId])

  const loadOlder = async () => {
    if (!page?.hasMore || !page.nextCursor || loadState === 'loading') {
      return
    }

    if (pages[pageIndex + 1]) {
      setPageIndex(value => value + 1)

      return
    }

    const generation = ++requestGeneration.current
    const expectedIndex = pageIndex
    setLoadState('loading')
    setPageFailed(false)

    try {
      const older = await loadPage(group, {
        cursor: page.nextCursor,
        limit: GROUP_FILES_PAGE_SIZE,
        ...(query ? { query } : {})
      })

      if (requestGeneration.current !== generation) {
        return
      }

      validateGroupFilesContinuation(page, older)
      const seen = new Set(pages.flatMap(loaded => loaded.items.map(item => item.attachment.attachmentId)))

      if (older.items.some(item => seen.has(item.attachment.attachmentId))) {
        throw new Error('Invalid shared-files duplicate page')
      }

      setPages(current => [...current.slice(0, expectedIndex + 1), older])
      setPageIndex(expectedIndex + 1)
      setLoadState('ready')
    } catch (error) {
      if (requestGeneration.current === generation) {
        setLoadState('error')
        setPageFailed(true)
        setCursorExpired(isGroupFilesCursorError(error))
      }
    }
  }

  const showNewer = () => {
    requestGeneration.current += 1
    setPageIndex(value => Math.max(0, value - 1))
    setLoadState('ready')
  }

  const returnToLatest = () => {
    requestGeneration.current += 1
    setPages([])
    setPageIndex(0)
    setPageFailed(false)
    setRefresh(value => value + 1)
  }

  const retry = () => {
    if (availability === 'offline') {
      const generation = ++requestGeneration.current
      setLoadState('loading')
      void loadPage(group, { limit: GROUP_FILES_PAGE_SIZE, ...(query ? { query } : {}) })
        .finally(() => {
          if (requestGeneration.current === generation) {
            setLoadState('ready')
          }
        })
        .catch(() => undefined)

      return
    }

    requestGeneration.current += 1
    setRefresh(value => value + 1)
  }

  const body = (() => {
    if (availability === 'offline') {
      if (loadState === 'loading') {
        return <Loader className="m-auto size-16" label={b.group.sharedFilesLoading} type="lemniscate-bloom" />
      }

      return (
        <ErrorState className="my-auto" title={<p className="text-sm font-medium">{b.group.sharedFilesOffline}</p>}>
          <Button onClick={retry} variant="secondary">
            {b.group.sharedFilesRetry}
          </Button>
        </ErrorState>
      )
    }

    if (availability === 'unavailable') {
      return <EmptyState className="my-auto" title={b.group.sharedFilesUnavailable} />
    }

    if (loadState === 'loading' && !page) {
      return <Loader className="m-auto size-16" label={b.group.sharedFilesLoading} type="lemniscate-bloom" />
    }

    if (loadState === 'error') {
      return (
        <ErrorState
          className="my-auto"
          title={
            <p className="text-sm font-medium">
              {cursorExpired ? b.group.sharedFilesExpired : b.group.sharedFilesError}
            </p>
          }
        >
          <Button onClick={pageFailed ? returnToLatest : retry} variant="secondary">
            {pageFailed ? b.group.returnToLatest : b.group.sharedFilesRetry}
          </Button>
        </ErrorState>
      )
    }

    if (!page?.items.length) {
      const partialPage = Boolean(page && (page.hasMore || pageIndex > 0))

      return (
        <EmptyState
          className="my-auto"
          title={
            partialPage ? b.group.sharedFilesPageEmpty : query ? b.group.sharedFilesNoResults : b.group.sharedFilesEmpty
          }
        />
      )
    }

    return (
      <div aria-busy={loadState === 'loading'} className="min-h-0 flex-1 overflow-y-auto" role="list">
        {page.items.map(item => {
          const attachment = item.attachment
          const timestamp = item.sharedAt * 1000

          const metadata = [
            item.producer.label,
            new Date(timestamp).toLocaleString(locale),
            formatFileType(attachment.mime || '', attachment.name || ''),
            formatFileSize(attachment.size || 0, locale)
          ].join(' · ')

          const message = { eventId: item.eventId, id: item.eventId, roomId } as GroupMessage

          return (
            <div
              className="flex min-h-12 min-w-0 items-center gap-2 py-1.5"
              key={attachment.attachmentId}
              role="listitem"
            >
              <Codicon
                className="shrink-0 text-(--ui-text-tertiary)"
                name={attachment.kind === 'pdf' ? 'file-pdf' : attachment.kind === 'image' ? 'file-media' : 'file'}
              />
              <div className="min-w-0 flex-1">
                <div className="truncate text-xs font-medium" title={attachment.name}>
                  {attachment.name}
                </div>
                <div className="truncate text-[0.65rem] text-(--ui-text-quaternary)" title={metadata}>
                  {metadata}
                </div>
              </div>
              <GroupAttachmentDownload attachment={attachment} group={group} message={message} presentation="icon" />
            </div>
          )
        })}
      </div>
    )
  })()

  const hasNewArrival = Boolean(page && latestSeq > page.snapshotSeq)
  const showSearch = Boolean(query || draftQuery || pages[0]?.items.length)

  return (
    <Dialog onOpenChange={value => !value && onClose()} open={open}>
      <DialogContent bodyClassName="flex min-h-0 flex-col gap-3" className="h-[min(36rem,85vh)] max-w-xl">
        <DialogHeader>
          <DialogTitle>{b.group.sharedFiles}</DialogTitle>
          <DialogDescription className="min-w-0 truncate" title={b.group.sharedFilesDescription(group)}>
            {b.group.sharedFilesDescription(group)}
          </DialogDescription>
        </DialogHeader>
        {showSearch ? (
          <SearchField
            aria-label={b.group.searchSharedFiles}
            containerClassName="w-full"
            inputClassName="w-full"
            loading={loadState === 'loading' && Boolean(page)}
            onChange={value => setDraftQuery([...value].slice(0, GROUP_FILES_MAX_QUERY_LENGTH).join(''))}
            placeholder={b.group.searchSharedFiles}
            value={draftQuery}
          />
        ) : null}
        {body}
        {page && availability === 'available' && loadState !== 'error' ? (
          <div className="flex min-h-6 items-center justify-end gap-1">
            {hasNewArrival ? (
              <Button className="mr-auto" onClick={returnToLatest} size="inline" variant="textStrong">
                {b.group.showLatest}
              </Button>
            ) : null}
            <Tip label={b.group.newerFiles}>
              <Button
                aria-label={b.group.newerFiles}
                disabled={pageIndex === 0 || loadState === 'loading'}
                onClick={showNewer}
                size="icon-xs"
                variant="ghost"
              >
                <Codicon name="chevron-left" />
              </Button>
            </Tip>
            <Tip label={b.group.olderFiles}>
              <Button
                aria-label={b.group.olderFiles}
                disabled={!page.hasMore || loadState === 'loading'}
                onClick={() => void loadOlder()}
                size="icon-xs"
                variant="ghost"
              >
                <Codicon name="chevron-right" />
              </Button>
            </Tip>
          </div>
        ) : null}
      </DialogContent>
    </Dialog>
  )
}

export function SharedFilesControl({ group, room }: { group: string; room: GroupChat }) {
  const b = useBots()
  const capabilities = useValue($hostedRoomCapabilities)
  const availability = groupFilesAvailability(room, capabilities)
  const [open, setOpen] = useState(false)
  const advertised = availability === 'available'

  useEffect(() => setOpen(false), [group, room.roomId, room.hosted, room.hostedEpoch])

  if (!advertised && !open) {
    return null
  }

  return (
    <>
      {advertised ? (
        <Tip label={b.group.sharedFiles}>
          <Button
            aria-label={b.group.sharedFiles}
            className="shrink-0 text-(--ui-text-tertiary) hover:text-foreground"
            onClick={() => setOpen(true)}
            size="icon-sm"
            variant="ghost"
          >
            <Codicon name="files" />
          </Button>
        </Tip>
      ) : null}
      <SharedFilesDialog
        availability={availability}
        group={group}
        latestSeq={Math.max(0, Number(room.hostedSeq || 0))}
        onClose={() => setOpen(false)}
        open={open}
        roomId={String(room.roomId || '')}
      />
    </>
  )
}
