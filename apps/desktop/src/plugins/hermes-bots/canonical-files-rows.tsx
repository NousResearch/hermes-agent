/** #104199's version rows, using native Save and canonical owner selection. */
import { Button, Codicon, Tip } from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'

import {
  canonicalFilesFailure,
  type FilesAuthority,
  type FilesFailure,
  saveCanonicalFile
} from './canonical-files-client'
import { useCanonicalFilesLabels } from './canonical-files-labels'
import type { CanonicalGroupBinding } from './canonical-groups'
import type { GroupFileItem } from './group-files-parser'

function fileSize(bytes: number, locale: string) {
  const divisor = bytes < 1000 ? 1 : bytes < 1_000_000 ? 1000 : 1_000_000

  return new Intl.NumberFormat(locale, {
    style: 'unit',
    unit: divisor === 1 ? 'byte' : divisor === 1000 ? 'kilobyte' : 'megabyte',
    unitDisplay: 'short',
    maximumFractionDigits: bytes < divisor * 10 ? 1 : 0
  }).format(bytes / divisor)
}

interface RowProps {
  binding: CanonicalGroupBinding
  authority: FilesAuthority
  item: GroupFileItem
  signal: AbortSignal
  active: boolean
  preciseTime: boolean
  onFocus: () => void
  onRefresh: () => void
  onAccessDenied: (failure: FilesFailure) => void
}

function FileRow({
  binding,
  authority,
  item,
  signal,
  active,
  preciseTime,
  onFocus,
  onRefresh,
  onAccessDenied
}: RowProps) {
  const b = useCanonicalFilesLabels()
  const [failure, setFailure] = useState<FilesFailure | null>(null)
  const [pending, setPending] = useState(false)
  const request = useRef<AbortController | null>(null)
  const row = useRef<HTMLDivElement>(null)
  const file = item.attachment
  const date = new Date(item.sharedAt * 1000)
  const today = date.toDateString() === new Date().toDateString()

  const shownDate = new Intl.DateTimeFormat(b.locale, {
    ...(today ? {} : ({ year: 'numeric', month: 'short', day: 'numeric' } as const)),
    hour: 'numeric',
    minute: '2-digit',
    ...(preciseTime ? ({ second: '2-digit' } as const) : {})
  }).format(date)

  const extension = file.name.includes('.') ? file.name.split('.').pop() : ''
  const type = String(extension || file.mime.split('/')[1]).toUpperCase()
  const metadata = [item.producer.label, date.toLocaleString(b.locale), type, fileSize(file.size, b.locale)].join(' · ')

  useEffect(() => {
    setFailure(null)

    return () => {
      request.current?.abort()
    }
  }, [item, signal])

  const download = async () => {
    if (request.current || signal.aborted) {
      return
    }

    const controller = new AbortController()
    const abort = () => controller.abort()
    signal.addEventListener('abort', abort, { once: true })
    request.current = controller
    setPending(true)
    setFailure(null)

    try {
      await saveCanonicalFile(binding, authority, item, controller.signal)
    } catch (error) {
      if (!controller.signal.aborted) {
        const kind = canonicalFilesFailure(error)

        if (kind === 'access' || kind === 'scope') {
          onAccessDenied(kind)
        } else {
          setFailure(kind)
        }
      }
    } finally {
      signal.removeEventListener('abort', abort)
      request.current = null
      setPending(false)
    }
  }

  const problem =
    failure === 'gone'
      ? b.fileGone
      : failure === 'verification'
        ? b.fileVerificationFailed
        : failure === 'timeout'
          ? b.fileTimeout
          : b.attachmentDownloadFailed

  return (
    <div
      aria-label={`${file.name} · ${metadata}`}
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
          name={file.kind === 'pdf' ? 'file-pdf' : file.kind === 'image' ? 'file-media' : 'file'}
        />
        <div className="min-w-0 flex-1">
          <bdi className="block truncate text-xs font-medium" title={file.name}>
            {file.name}
          </bdi>
          <div className="truncate text-[0.65rem] text-(--ui-text-quaternary)" title={metadata}>
            <bdi>{item.producer.label}</bdi>
            {` · ${shownDate} · ${type} · ${fileSize(file.size, b.locale)}`}
          </div>
        </div>
        <Tip label={`${b.download}: ${file.name}`}>
          <Button
            aria-busy={pending}
            aria-label={`${b.download}: ${file.name}`}
            data-file-download="true"
            disabled={pending || failure === 'gone' || signal.aborted}
            onClick={() => void download()}
            size="icon-xs"
            type="button"
            variant="ghost"
          >
            <Codicon name={pending ? 'loading' : 'cloud-download'} spinning={pending} />
          </Button>
        </Tip>
      </div>
      {failure && (
        <div className="mt-1 flex min-w-0 flex-wrap items-center gap-2 text-xs text-(--ui-text-secondary)" role="alert">
          <span className="min-w-0 break-words">{problem}</span>
          <Button
            onClick={() => {
              if (failure === 'gone') {
                onRefresh()
              } else {
                void download()
              }
            }}
            size="inline"
            type="button"
            variant="textStrong"
          >
            {failure === 'gone' ? b.filesRefresh : b.sharedFilesRetry}
          </Button>
        </div>
      )}
    </div>
  )
}

export function CanonicalFilesRows({
  items,
  binding,
  authority,
  signal,
  loading,
  onRefresh,
  onAccessDenied
}: {
  items: GroupFileItem[]
  binding: CanonicalGroupBinding
  authority: FilesAuthority
  signal: AbortSignal
  loading: boolean
  onRefresh: () => void
  onAccessDenied: (failure: FilesFailure) => void
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
          authority={authority}
          binding={binding}
          item={item}
          key={`${item.eventId}:${item.attachment.attachmentId}`}
          onAccessDenied={onAccessDenied}
          onFocus={() => setActive(index)}
          onRefresh={onRefresh}
          preciseTime={items.some(other => other !== item && other.attachment.name === item.attachment.name)}
          signal={signal}
        />
      ))}
    </div>
  )
}
