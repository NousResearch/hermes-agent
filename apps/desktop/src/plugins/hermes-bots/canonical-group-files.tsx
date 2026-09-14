/** Donor Files dialog on canonical groups; no legacy room store or executor. */
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
import { useRef, useState } from 'react'

import type { FilesAuthority } from './canonical-files-client'
import { useCanonicalFilesLabels } from './canonical-files-labels'
import { CanonicalFilesRows } from './canonical-files-rows'
import type { CanonicalGroupBinding } from './canonical-groups'
import { useCanonicalFiles } from './use-canonical-files'

interface FilesProps {
  binding: CanonicalGroupBinding
  name: string
  authority?: FilesAuthority
  latestFileSeq?: number
  accessDenied?: boolean
}

function FilesDialog({
  binding,
  name,
  authority,
  latestFileSeq = 0,
  accessDenied,
  onClose
}: FilesProps & { onClose: () => void }) {
  const b = useCanonicalFilesLabels()
  const input = useRef<HTMLInputElement>(null)
  const files = useCanonicalFiles({ binding, authority, open: true, accessDenied })
  const page = files.page
  const first = files.pages[0]?.data
  const latest = first && Math.max(latestFileSeq, files.latestFileSeq) > first.snapshotSeq
  const expired = files.failure === 'cursor' || files.failure === 'scope'

  const problem = expired
    ? b.sharedFilesExpired
    : files.failure === 'access'
      ? b.filesAccessUnavailable
      : files.failure === 'offline' || files.failure === 'timeout'
        ? b.sharedFilesOffline
        : files.failure === 'unavailable'
          ? b.sharedFilesUnavailable
          : b.sharedFilesError

  const recover = expired ? files.latest : files.retry
  const recoverLabel = expired ? b.returnToLatest : b.sharedFilesRetry

  const body =
    page?.items.length && page.authority ? (
      <CanonicalFilesRows
        authority={page.authority}
        binding={binding}
        items={page.items}
        key={`${page.snapshotSeq}:${files.index}`}
        loading={files.loading}
        onAccessDenied={files.invalidateAccess}
        onRefresh={files.latest}
        signal={files.deliverySignal}
      />
    ) : files.loading ? (
      <Loader className="m-auto size-16" label={b.sharedFilesLoading} type="lemniscate-bloom" />
    ) : files.failure ? (
      <ErrorState className="my-auto" title={<p className="text-sm font-medium">{problem}</p>}>
        <Button disabled={accessDenied} onClick={recover} type="button" variant="secondary">
          {recoverLabel}
        </Button>
      </ErrorState>
    ) : (
      <div className="my-auto">
        <EmptyState
          title={
            page && (page.hasMore || files.index > 0)
              ? b.sharedFilesPageEmpty
              : files.query
                ? b.sharedFilesNoResults
                : b.sharedFilesEmpty
          }
        />
        {files.query && (
          <div className="flex justify-center">
            <Button onClick={() => files.setQuery('')} size="inline" type="button" variant="textStrong">
              {b.filesClearSearch}
            </Button>
          </div>
        )}
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
      open
    >
      <DialogContent
        bodyClassName="flex min-h-0 flex-1 flex-col gap-3"
        className="h-[min(36rem,85vh)] max-w-xl"
        onKeyDown={event => {
          const target = event.target as HTMLElement

          if (
            event.key === '/' &&
            !target.matches('input,textarea,[contenteditable="true"]') &&
            !event.ctrlKey &&
            !event.metaKey &&
            !event.altKey
          ) {
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
          <DialogTitle>{b.sharedFiles}</DialogTitle>
          <DialogDescription>
            <bdi className="block truncate" title={name}>
              {name}
            </bdi>
          </DialogDescription>
        </DialogHeader>
        <SearchField
          aria-label={b.searchSharedFiles}
          containerClassName="w-full"
          inputClassName="w-full"
          inputRef={input}
          loading={files.loading && Boolean(page)}
          onChange={files.setQuery}
          placeholder={b.searchSharedFiles}
          value={files.query}
        />
        {body}
        {page && (
          <div className="flex min-h-8 flex-wrap items-center justify-end gap-2">
            {page.items.length > 0 && files.failure ? (
              <div className="mr-auto flex min-w-0 flex-wrap items-center gap-2 text-xs" role="status">
                <span>{problem}</span>
                <Button disabled={files.loading} onClick={recover} size="inline" type="button" variant="textStrong">
                  {recoverLabel}
                </Button>
              </div>
            ) : files.reconnected ? (
              <span className="mr-auto text-xs text-(--ui-text-tertiary)" role="status">
                {b.filesReconnected}
              </span>
            ) : null}
            {latest && (
              <Button disabled={files.loading} onClick={files.latest} size="inline" type="button" variant="textStrong">
                {b.showLatest}
              </Button>
            )}
            <Tip label={b.filesRefresh}>
              <Button
                aria-label={b.filesRefresh}
                disabled={files.loading}
                onClick={files.latest}
                size="icon-xs"
                type="button"
                variant="ghost"
              >
                <Codicon name="refresh" />
              </Button>
            </Tip>
            <Tip label={b.newerFiles}>
              <Button
                aria-label={b.newerFiles}
                disabled={files.index === 0 || files.loading}
                onClick={files.newer}
                size="icon-xs"
                type="button"
                variant="ghost"
              >
                <Codicon name="chevron-left" />
              </Button>
            </Tip>
            <Tip label={b.olderFiles}>
              <Button
                aria-label={b.olderFiles}
                disabled={!page.hasMore || files.loading}
                onClick={files.older}
                size="icon-xs"
                type="button"
                variant="ghost"
              >
                <Codicon name="chevron-right" />
              </Button>
            </Tip>
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}

function FilesControl(props: FilesProps) {
  const b = useCanonicalFilesLabels()
  const [open, setOpen] = useState(false)

  return (
    <>
      <Tip label={b.sharedFiles}>
        <Button aria-label={b.sharedFiles} onClick={() => setOpen(true)} size="icon-sm" type="button" variant="ghost">
          <Codicon name="files" />
        </Button>
      </Tip>
      {open && <FilesDialog {...props} onClose={() => setOpen(false)} />}
    </>
  )
}

export function CanonicalGroupFiles(props: FilesProps) {
  // Scope changes close the dialog and retire its cached pages and native-save intents.
  return (
    <FilesControl
      {...props}
      key={JSON.stringify([
        props.binding.connectionId,
        props.binding.profile,
        props.binding.roomId,
        props.authority?.gatewayId,
        props.authority?.epoch
      ])}
    />
  )
}
