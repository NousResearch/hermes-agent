import { memo, useEffect, useState } from 'react'

import { PathText } from '@/components/ui/path-text'
import { PixelIcon } from '@/components/ui/pixel-icon'
import { ExternalLink, hostPathLabel, urlSlugTitleLabel, useLinkTitle } from '@/lib/external-link'
import { resolveMediaDisplaySrc } from '@/lib/media'
import { fmtDayTime } from '@/lib/time'

import { type ArtifactFilter, type ArtifactRecord, paginationItems } from '../artifacts/artifact-utils'
import type { IndexedChat } from '../artifacts/item-index-snapshot'

import { chatOfItem, itemOwner } from './library-actions'
import { type LibraryCopy, useLibraryCopy } from './library-copy'

export const formatItemTime = (timestamp: number) => fmtDayTime.format(new Date(timestamp))

/** The short type tag a row wears (LINK, MD, PNG): the extension, else the kind. */
export function itemTypeTag(item: ArtifactRecord): string {
  if (item.kind === 'link') {
    return 'link'
  }

  const name = item.value.split(/[?#]/)[0] ?? ''
  const ext = /\.([a-z0-9]{1,5})$/i.exec(name)?.[1]

  return ext ?? (item.value.startsWith('data:image/') ? item.value.slice(11).split(/[;,]/)[0] || 'image' : item.kind)
}

/** The image's displayable source, resolved on the machine that owns its chat. */
export function useItemImageSrc(item: ArtifactRecord, chat: IndexedChat | undefined): { failed: boolean; src: string } {
  const [state, setState] = useState({ failed: false, src: '' })
  const connectionId = itemOwner(chat)?.connectionId
  const profile = chat?.profile ?? item.profile

  useEffect(() => {
    let live = true

    setState({ failed: false, src: '' })
    resolveMediaDisplaySrc(item.value, connectionId ? { connectionId, profile } : undefined).then(
      src => live && setState({ failed: !src, src }),
      () => live && setState({ failed: true, src: '' })
    )

    return () => {
      live = false
    }
  }, [connectionId, item.value, profile])

  return state
}

interface ThumbProps {
  item: ArtifactRecord
  chat: IndexedChat | undefined
  className?: string
}

/** An image as a cover-cropped thumbnail; a source that cannot be shown says so in place. */
export const ItemThumb = memo(function ItemThumb({ chat, className, item }: ThumbProps) {
  const { failed, src } = useItemImageSrc(item, chat)
  const [broken, setBroken] = useState(false)
  const copy = useLibraryCopy()

  return (
    <span className={className} data-slot="library-thumb">
      {failed || broken ? (
        <span className="library-thumb-missing">
          <PixelIcon name="image" />
          <span className="sr-only">{copy.imageUnavailable}</span>
        </span>
      ) : (
        src && (
          <img alt="" decoding="async" draggable={false} loading="lazy" onError={() => setBroken(true)} src={src} />
        )
      )}
    </span>
  )
})

interface ImageCardProps {
  item: ArtifactRecord
  chat: IndexedChat | undefined
  onOpen: (item: ArtifactRecord) => void
}

/** 1163:18113: the thumbnail, the file name, then "chat · date". */
export const ImageCard = memo(function ImageCard({ chat, item, onOpen }: ImageCardProps) {
  return (
    <button className="library-card" onClick={() => onOpen(item)} type="button">
      <ItemThumb chat={chat} className="library-card-thumb" item={item} />
      <span className="library-card-meta">
        <span className="library-card-name">{item.label}</span>
        <span className="library-card-sub">
          {chat?.title ?? item.sessionTitle} · {formatItemTime(item.timestamp)}
        </span>
      </span>
    </button>
  )
})

interface ImageGridProps {
  items: readonly ArtifactRecord[]
  chats: Record<string, IndexedChat>
  onOpen: (item: ArtifactRecord) => void
}

export function ImageGrid({ chats, items, onOpen }: ImageGridProps) {
  return (
    <div className="library-grid">
      {items.map(item => (
        <ImageCard chat={chatOfItem(chats, item)} item={item} key={item.id} onOpen={onOpen} />
      ))}
    </div>
  )
}

function LinkName({ item }: { item: ArtifactRecord }) {
  const title = useLinkTitle(item.href)

  return <>{title || urlSlugTitleLabel(item.href)}</>
}

const locationHeader = (filter: ArtifactFilter, copy: LibraryCopy) =>
  filter === 'file' ? copy.columns.path : filter === 'link' ? copy.columns.url : copy.columns.location

interface ItemTableProps {
  items: readonly ArtifactRecord[]
  chats: Record<string, IndexedChat>
  filter: ArtifactFilter
  onOpen: (item: ArtifactRecord) => void
  onOpenChat: (item: ArtifactRecord) => void
}

/** 1163:18155: type tag, name, location (path or URL), then the chat and its date. The name opens
 *  the item (a link through the app's link rules, a file with `onOpen`); the chat cell its chat. */
export function ItemTable({ chats, filter, items, onOpen, onOpenChat }: ItemTableProps) {
  const copy = useLibraryCopy()

  return (
    <div className="library-table-card">
      <table className="library-table">
        <thead>
          <tr>
            <th aria-label={copy.columns.name} className="library-col-tag" />
            <th className="library-col-name">{copy.columns.name}</th>
            <th className="library-col-location">{locationHeader(filter, copy)}</th>
            <th className="library-col-chat">{copy.columns.chat}</th>
          </tr>
        </thead>
        <tbody>
          {items.map(item => {
            const chat = chatOfItem(chats, item)
            const link = item.kind === 'link'

            return (
              <tr key={item.id}>
                <td className="library-col-tag">
                  <span className="library-tag">{itemTypeTag(item)}</span>
                </td>
                <td className="library-col-name">
                  {link ? (
                    <ExternalLink className="library-row-open" href={item.href} title={item.value}>
                      <LinkName item={item} />
                    </ExternalLink>
                  ) : (
                    <button className="library-row-open" onClick={() => onOpen(item)} type="button">
                      {item.label}
                    </button>
                  )}
                </td>
                <td className="library-col-location">
                  {link ? (
                    <span className="library-url">{hostPathLabel(item.value)}</span>
                  ) : (
                    <PathText path={item.value} />
                  )}
                </td>
                <td className="library-col-chat">
                  <button className="library-chat-cell" onClick={() => onOpenChat(item)} type="button">
                    <span className="library-chat-title">{chat?.title ?? item.sessionTitle}</span>
                    <span className="library-chat-date">{formatItemTime(item.timestamp)}</span>
                  </button>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

interface PagerProps {
  filter: ArtifactFilter
  page: number
  pageSize: number
  total: number
  onPage: (page: number) => void
}

/** "1–24 of 161 images ‹ 1 2 3 … 7 ›" (1163:18242). */
export function Pager({ filter, onPage, page, pageSize, total }: PagerProps) {
  const copy = useLibraryCopy()
  const pageCount = Math.max(1, Math.ceil(total / pageSize))
  const start = total ? (page - 1) * pageSize + 1 : 0
  const end = Math.min(total, page * pageSize)

  return (
    <nav aria-label={copy.pages} className="library-pager">
      <span className="library-pager-range">{copy.range(start, end, total, filter)}</span>
      {pageCount > 1 && (
        <>
          <button aria-label={copy.previousPage} disabled={page <= 1} onClick={() => onPage(page - 1)} type="button">
            <PixelIcon name="chevron-left" />
          </button>
          {paginationItems(page, pageCount).map((entry, index) =>
            entry === 'ellipsis' ? (
              <span aria-hidden className="library-pager-gap" key={`gap-${index}`}>
                …
              </span>
            ) : (
              <button
                aria-current={entry === page ? 'page' : undefined}
                aria-label={copy.page(entry)}
                key={entry}
                onClick={() => onPage(entry)}
                type="button"
              >
                {entry}
              </button>
            )
          )}
          <button
            aria-label={copy.nextPage}
            disabled={page >= pageCount}
            onClick={() => onPage(page + 1)}
            type="button"
          >
            <PixelIcon name="chevron-right" />
          </button>
        </>
      )}
    </nav>
  )
}
