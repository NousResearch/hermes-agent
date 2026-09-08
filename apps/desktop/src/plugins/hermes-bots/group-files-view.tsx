/** Files discovery for hosted rooms and this Desktop's retained classic attachments. */

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
  useValue
} from '@hermes/plugin-sdk'
import { useEffect, useMemo, useRef, useState } from 'react'

import { groupChatHostedGateway } from './group-chat'
import {
  classicLatestFileKey,
  createClassicGroupFilesLoader,
  isClassicFileRoom,
  latestHostedFileSeq
} from './group-files-classic'
import { listHostedGroupFiles } from './group-files-client'
import { GroupFilesRows } from './group-files-rows'
import { $hostedRoomCapabilities } from './hosted-room-capability-state'
import { useBots } from './i18n'
import type { GroupChat } from './types'
import { type GroupFilesAvailability, type GroupFilesLoader, useGroupFiles } from './use-group-files'

export type { GroupFilesAvailability } from './use-group-files'

export function groupFilesAvailability(
  room: GroupChat,
  capabilities: ReturnType<typeof $hostedRoomCapabilities.get>
): GroupFilesAvailability {
  if (isClassicFileRoom(room)) {
    return 'available'
  }

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

interface SharedFilesDialogProps {
  availability: GroupFilesAvailability
  observation?: unknown
  group: string
  latestSeq: number
  latestKey?: string
  classic?: boolean
  loadPage?: GroupFilesLoader
  onClose: () => void
  open: boolean
  roomId: string
}

export function SharedFilesDialog(props: SharedFilesDialogProps) {
  return <FilesDialogBody key={JSON.stringify([props.group, props.roomId, props.classic || false])} {...props} />
}

function FilesDialogBody({
  availability,
  observation,
  group,
  latestSeq,
  latestKey,
  classic = false,
  loadPage = listHostedGroupFiles,
  onClose,
  open,
  roomId
}: SharedFilesDialogProps) {
  const b = useBots()
  const input = useRef<HTMLInputElement>(null)
  const files = useGroupFiles({ availability, observation, group, loadPage, open })
  const page = files.page
  const first = files.pages[0]?.data

  const latest = classic
    ? Boolean(latestKey && first && latestKey !== first.localSnapshotKey)
    : Boolean(first && Math.max(latestSeq, files.latestFileSeq) > first.snapshotSeq)

  const problem = files.cursorExpired
    ? b.group.sharedFilesExpired
    : files.failure === 'access'
      ? b.group.filesAccessUnavailable
      : files.offline
        ? b.group.sharedFilesOffline
        : b.group.sharedFilesError

  const recover = files.cursorExpired ? files.latest : files.retry
  const recoverLabel = files.cursorExpired ? b.group.returnToLatest : b.group.sharedFilesRetry

  const body = files.unavailable ? (
    <EmptyState className="my-auto" title={b.group.sharedFilesUnavailable} />
  ) : page?.items.length ? (
    <GroupFilesRows
      group={group}
      intentSignal={files.deliverySignal}
      items={page.items}
      loading={files.loading}
      onRefresh={files.latest}
      onRoomAccessDenied={files.invalidateAccess}
      roomId={roomId}
    />
  ) : files.loading ? (
    <Loader className="m-auto size-16" label={b.group.sharedFilesLoading} type="lemniscate-bloom" />
  ) : files.failure || files.offline ? (
    <ErrorState className="my-auto" title={<p className="text-sm font-medium">{problem}</p>}>
      <Button onClick={recover} variant="secondary">
        {recoverLabel}
      </Button>
    </ErrorState>
  ) : (
    <div className="my-auto">
      <EmptyState
        title={
          page && (page.hasMore || files.index > 0)
            ? b.group.sharedFilesPageEmpty
            : files.query
              ? b.group.sharedFilesNoResults
              : b.group.sharedFilesEmpty
        }
      />
      {files.query ? (
        <div className="flex justify-center">
          <Button onClick={() => files.setQuery('')} size="inline" variant="textStrong">
            {b.group.filesClearSearch}
          </Button>
        </div>
      ) : null}
    </div>
  )

  return (
    <Dialog
      onOpenChange={value => {
        if (!value) {
          files.cancel()
          onClose()
        }
      }}
      open={open}
    >
      <DialogContent
        bodyClassName="flex min-h-0 flex-1 flex-col gap-3"
        className="h-[min(36rem,85vh)] max-w-xl"
        onKeyDown={event => {
          const target = event.target as HTMLElement
          const editing = target.matches('input,textarea,[contenteditable="true"]')

          if (event.key === '/' && !editing && !event.ctrlKey && !event.metaKey && !event.altKey) {
            event.preventDefault()
            input.current?.focus()
          } else if (event.key === 'ArrowDown' && target === input.current) {
            const row = event.currentTarget.querySelector<HTMLElement>('[data-file-row]')

            if (row) {
              event.preventDefault()
              row.focus()
            }
          }
        }}
        onOpenAutoFocus={event => {
          event.preventDefault()
          input.current?.focus()
        }}
      >
        <DialogHeader>
          <DialogTitle>{b.group.sharedFiles}</DialogTitle>
          <DialogDescription className="min-w-0">
            <bdi className="block truncate" title={group}>
              {group}
            </bdi>
            {classic ? <span className="block break-words">{b.group.filesClassicDescription}</span> : null}
          </DialogDescription>
        </DialogHeader>
        <SearchField
          aria-label={b.group.searchSharedFiles}
          containerClassName="w-full"
          inputClassName="w-full"
          inputRef={input}
          loading={files.loading && Boolean(page)}
          onChange={files.setQuery}
          placeholder={b.group.searchSharedFiles}
          value={files.query}
        />
        {body}
        {page && !files.unavailable ? (
          <div className="flex min-h-8 flex-wrap items-center justify-end gap-2">
            {page.items.length > 0 && (files.failure || files.offline) ? (
              <div className="mr-auto flex min-w-0 flex-wrap items-center gap-2 text-xs" role="status">
                <span>{problem}</span>
                <Button disabled={files.loading} onClick={recover} size="inline" variant="textStrong">
                  {recoverLabel}
                </Button>
              </div>
            ) : files.reconnected ? (
              <span className="mr-auto text-xs text-(--ui-text-tertiary)" role="status">
                {b.group.filesReconnected}
              </span>
            ) : null}
            {latest ? (
              <Button onClick={files.latest} size="inline" variant="textStrong">
                {b.group.showLatest}
              </Button>
            ) : null}
            <Tip label={b.group.newerFiles}>
              <Button
                aria-label={b.group.newerFiles}
                disabled={files.index === 0 || files.loading}
                onClick={files.newer}
                size="icon-xs"
                variant="ghost"
              >
                <Codicon name="chevron-left" />
              </Button>
            </Tip>
            <Tip label={b.group.olderFiles}>
              <Button
                aria-label={b.group.olderFiles}
                disabled={!page.hasMore || files.loading}
                onClick={files.older}
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
  const classic = isClassicFileRoom(room)
  const roomId = String(room.roomId || '')

  const classicLoader = useMemo(
    () => (classic ? createClassicGroupFilesLoader(group, roomId) : null),
    [classic, group, roomId]
  )

  const loader = classicLoader || listHostedGroupFiles
  useEffect(() => {
    if (!open) {
      classicLoader?.clear()
    }

    return () => classicLoader?.clear()
  }, [open, classicLoader])
  useEffect(() => setOpen(false), [group, roomId, room.hosted, room.hostedEpoch, classic])

  return (
    <>
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
      {open ? (
        <SharedFilesDialog
          availability={availability}
          classic={classic}
          group={group}
          key={JSON.stringify([roomId, room.hosted, room.hostedEpoch, classic])}
          latestKey={classic ? classicLatestFileKey(room) : undefined}
          latestSeq={classic ? 0 : latestHostedFileSeq(room)}
          loadPage={loader}
          observation={capabilities[String(room.hostedConnectionId || '')]}
          onClose={() => setOpen(false)}
          open={open}
          roomId={roomId}
        />
      ) : null}
    </>
  )
}
