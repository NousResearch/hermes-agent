import { Button, Codicon, useI18n } from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'

import { GroupAttachmentDownload } from './group-attachment-download'
import type { GroupFileFailure } from './group-file-errors'
import type { GroupFileItem } from './group-files-client'
import { useBots } from './i18n'
import type { GroupMessage } from './types'

function fileSize(bytes: number | undefined, locale: string) {
  if (bytes === undefined) {
    return ''
  }

  const divisor = bytes < 1000 ? 1 : bytes < 1_000_000 ? 1000 : 1_000_000

  return new Intl.NumberFormat(locale, {
    style: 'unit',
    unit: divisor === 1 ? 'byte' : divisor === 1000 ? 'kilobyte' : 'megabyte',
    unitDisplay: 'short',
    maximumFractionDigits: bytes < divisor * 10 ? 1 : 0
  }).format(bytes / divisor)
}

function fileDate(date: Date, locale: string, precise: boolean) {
  const today = date.toDateString() === new Date().toDateString()

  return new Intl.DateTimeFormat(locale, {
    ...(today ? {} : ({ year: 'numeric', month: 'short', day: 'numeric' } as const)),
    hour: 'numeric',
    minute: '2-digit',
    ...(precise ? ({ second: '2-digit' } as const) : {})
  }).format(date)
}

function FileRow({
  item,
  group,
  roomId,
  active,
  onFocus,
  onRefresh,
  onRoomAccessDenied,
  intentSignal,
  preciseTime
}: {
  item: GroupFileItem
  group: string
  roomId: string
  active: boolean
  onFocus: () => void
  onRefresh: () => void
  onRoomAccessDenied: () => void
  intentSignal: AbortSignal
  preciseTime: boolean
}) {
  const b = useBots()
  const { locale } = useI18n()
  const [failure, setFailure] = useState<GroupFileFailure | null>(null)
  const row = useRef<HTMLDivElement>(null)
  const attachment = item.attachment
  const sharedAt = new Date(item.sharedAt * 1000)
  const date = fileDate(sharedAt, locale, preciseTime)
  const extension = attachment.name.includes('.') ? attachment.name.split('.').pop() : ''
  const type = String(extension || attachment.mime?.split('/')[1] || '').toUpperCase()

  const metadata = [item.producer.label, sharedAt.toLocaleString(locale), type, fileSize(attachment.size, locale)]
    .filter(Boolean)
    .join(' · ')

  const message = item.localMessage || ({ eventId: item.eventId, id: item.eventId, roomId } as GroupMessage)
  useEffect(() => setFailure(null), [item])
  const gone = failure === 'gone'

  const failedText = gone
    ? b.group.fileGone
    : failure === 'verification'
      ? b.group.fileVerificationFailed
      : failure === 'timeout'
        ? b.group.fileTimeout
        : b.group.attachmentDownloadFailed

  return (
    <div
      aria-label={`${attachment.name} · ${metadata}`}
      className="min-h-12 min-w-0 py-1.5 outline-none focus-visible:ring-1 focus-visible:ring-(--ui-accent)"
      data-file-row="true"
      onFocus={onFocus}
      ref={row}
      role="listitem"
      tabIndex={active ? 0 : -1}
    >
      <div className="flex min-w-0 items-center gap-2">
        <Codicon
          className="shrink-0 text-(--ui-text-tertiary)"
          name={attachment.kind === 'pdf' ? 'file-pdf' : attachment.kind === 'image' ? 'file-media' : 'file'}
        />
        <div className="min-w-0 flex-1">
          <bdi className="block truncate text-xs font-medium" title={attachment.name}>
            {attachment.name}
          </bdi>
          <div className="truncate text-[0.65rem] text-(--ui-text-quaternary)" title={metadata}>
            <bdi>{item.producer.label}</bdi>
            {` · ${[date, type, fileSize(attachment.size, locale)].filter(Boolean).join(' · ')}`}
          </div>
        </div>
        <GroupAttachmentDownload
          attachment={attachment}
          disabled={gone}
          group={group}
          intentSignal={intentSignal}
          message={message}
          onFailure={failure => {
            if (failure === 'access') {
              onRoomAccessDenied()
            } else {
              setFailure(failure)
            }
          }}
          presentation="icon"
        />
      </div>
      {failure ? (
        <div className="mt-1 flex min-w-0 flex-wrap items-center gap-2 text-xs text-(--ui-text-secondary)" role="alert">
          <span className="min-w-0 break-words">{failedText}</span>
          <Button
            onClick={() => {
              if (gone) {
                onRefresh()
              } else {
                setFailure(null)
                row.current?.querySelector<HTMLButtonElement>('[data-file-download]')?.click()
              }
            }}
            size="inline"
            variant="textStrong"
          >
            {gone ? b.group.filesRefresh : b.group.sharedFilesRetry}
          </Button>
        </div>
      ) : null}
    </div>
  )
}

export function GroupFilesRows({
  items,
  group,
  roomId,
  loading,
  onRefresh,
  onRoomAccessDenied,
  intentSignal
}: {
  items: GroupFileItem[]
  group: string
  roomId: string
  loading: boolean
  onRefresh: () => void
  onRoomAccessDenied: () => void
  intentSignal: AbortSignal
}) {
  const [active, setActive] = useState(0)

  return (
    <div
      aria-busy={loading}
      className="min-h-0 flex-1 overflow-y-auto"
      onKeyDown={event => {
        const rows = Array.from(event.currentTarget.querySelectorAll<HTMLElement>('[data-file-row]'))
        const target = (event.target as HTMLElement).closest<HTMLElement>('[data-file-row]')

        if (!target) {
          return
        }

        const index = rows.indexOf(target)

        if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
          event.preventDefault()
          rows[Math.max(0, Math.min(rows.length - 1, index + (event.key === 'ArrowDown' ? 1 : -1)))]?.focus()
        } else if (event.key === 'Enter' && event.target === target) {
          event.preventDefault()
          target.querySelector<HTMLButtonElement>('[data-file-download]')?.click()
        }
      }}
      role="list"
    >
      {items.map((item, index) => (
        <FileRow
          active={active === index}
          group={group}
          intentSignal={intentSignal}
          item={item}
          key={item.key || `${item.eventId}:${item.attachment.attachmentId}`}
          onFocus={() => setActive(index)}
          onRefresh={onRefresh}
          onRoomAccessDenied={onRoomAccessDenied}
          preciseTime={items.some(other => other !== item && other.attachment.name === item.attachment.name)}
          roomId={roomId}
        />
      ))}
    </div>
  )
}
