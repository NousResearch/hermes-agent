import './chat-items.css'

import { useStore } from '@nanostores/react'
import { type ReactNode, useEffect, useState } from 'react'

import type { ArtifactRecord as IndexedItem } from '@/app/artifacts/artifact-utils'
import type { IndexedChat } from '@/app/artifacts/item-index-snapshot'
import { ArtifactPreview } from '@/app/chat/right-rail/preview-artifact'
import { formatItemTime, ItemThumb, itemTypeTag } from '@/app/library/library-collections'
import { Button } from '@/components/ui/button'
import { PixelIcon } from '@/components/ui/pixel-icon'
import { useI18n } from '@/i18n'
import { ExternalLink, hostPathLabel, urlSlugTitleLabel, useLinkTitle } from '@/lib/external-link'
import { localPreviewTarget } from '@/lib/local-preview'
import type { PixelIconName } from '@/lib/pixel-icon-registry'
import { artifactPreviewTarget, type ArtifactRecord } from '@/store/artifacts'
import { closeChatItems } from '@/store/chat-items'
import { $showsAdvancedChrome } from '@/store/interface-mode'
import { openPreview } from '@/store/preview'
import { $focusedStoredSessionId } from '@/store/session-focus'

import { PageTabs } from '../../redesign/page-header'
import { useChatOwner } from '../use-chat-owner'

import { type ChatItemsCopy, type ChatItemsTab, useChatItemsCopy } from './chat-items-copy'
import { type ChatItems, useChatItems } from './use-chat-items'

/** Rows a section of the All view shows before "See all" (1083:8349). */
const SECTION_ROWS = 3

const ARTIFACT_ICONS: Record<ArtifactRecord['kind'], PixelIconName> = {
  code: 'code',
  html: 'app-windows',
  svg: 'image'
}

interface RowProps {
  icon: ReactNode
  title: ReactNode
  meta: string
  /** A link opens through the app's link rules; anything else is a button. */
  href?: string
  onOpen?: () => void
  trailing?: ReactNode
}

function ItemRow({ href, icon, meta, onOpen, title, trailing }: RowProps) {
  const body = (
    <>
      <span className="chat-items-icon">{icon}</span>
      <span className="chat-items-text">
        <span className="chat-items-title">{title}</span>
        <span className="chat-items-meta">{meta}</span>
      </span>
      <span className="chat-items-trailing">{trailing}</span>
    </>
  )

  return href ? (
    <ExternalLink className="chat-items-row" href={href}>
      {body}
    </ExternalLink>
  ) : (
    <button className="chat-items-row" onClick={onOpen} type="button">
      {body}
    </button>
  )
}

function ArtifactRow({ artifact, onOpen }: { artifact: ArtifactRecord; onOpen: (artifact: ArtifactRecord) => void }) {
  const { t } = useI18n()
  const copy = useChatItemsCopy()

  return (
    <ItemRow
      icon={<PixelIcon name={ARTIFACT_ICONS[artifact.kind]} />}
      meta={`${t.artifactCard.kind[artifact.kind]} · ${copy.version(artifact.versions.length)} · ${formatItemTime(artifact.updatedAt)}`}
      onOpen={() => onOpen(artifact)}
      title={artifact.title}
      trailing={<PixelIcon name="chevron-right" />}
    />
  )
}

function FileRow({ chat, item }: { chat: IndexedChat | undefined; item: IndexedItem }) {
  const open = () => {
    const target = localPreviewTarget(item.value)

    if (target) {
      openPreview(target)
    }
  }

  return (
    <ItemRow
      icon={
        item.kind === 'image' ? (
          <ItemThumb chat={chat} className="chat-items-thumb" item={item} />
        ) : (
          <PixelIcon name="file" />
        )
      }
      meta={`${itemTypeTag(item).toUpperCase()} · ${formatItemTime(item.timestamp)}`}
      onOpen={open}
      title={item.label}
      trailing={<PixelIcon name="chevron-right" />}
    />
  )
}

function LinkTitle({ href }: { href: string }) {
  return <>{useLinkTitle(href) || urlSlugTitleLabel(href)}</>
}

function LinkRow({ item }: { item: IndexedItem }) {
  return (
    <ItemRow
      href={item.href}
      icon={<PixelIcon name="globe" />}
      meta={`${hostPathLabel(item.value).split('/')[0]} · ${formatItemTime(item.timestamp)}`}
      title={<LinkTitle href={item.href} />}
      trailing={<PixelIcon name="external-link" />}
    />
  )
}

interface SectionProps {
  label: string
  count: number
  onSeeAll?: () => void
  children: ReactNode
}

function Section({ children, count, label, onSeeAll }: SectionProps) {
  const copy = useChatItemsCopy()

  return (
    <section className="chat-items-section">
      <div className="chat-items-section-head">
        <h3>
          {label} <span>{count}</span>
        </h3>
        {onSeeAll && count > SECTION_ROWS && (
          <button aria-label={copy.seeAll(label)} className="chat-items-see-all" onClick={onSeeAll} type="button">
            {copy.seeAllShort} {count}
            <PixelIcon name="chevron-right" />
          </button>
        )}
      </div>
      <div className="chat-items-list">{children}</div>
    </section>
  )
}

interface ListProps {
  items: ChatItems
  tab: ChatItemsTab
  simple: boolean
  copy: ChatItemsCopy
  onTab: (tab: ChatItemsTab) => void
  onOpenArtifact: (artifact: ArtifactRecord) => void
}

/** The sections a tab shows: All's three short ones (or simple mode's whole list), or one kind. */
function ItemSections({ copy, items, onOpenArtifact, onTab, simple, tab }: ListProps) {
  const cap = (rows: number) => (tab === 'all' && !simple ? SECTION_ROWS : rows)
  const seeAll = (next: ChatItemsTab) => (tab === 'all' && !simple ? () => onTab(next) : undefined)

  const sections = [
    {
      id: 'artifacts' as const,
      label: simple ? copy.madeForYou : copy.tabs.artifacts,
      count: items.artifacts.length,
      rows: items.artifacts
        .slice(0, cap(items.artifacts.length))
        .map(artifact => <ArtifactRow artifact={artifact} key={artifact.id} onOpen={onOpenArtifact} />)
    },
    {
      id: 'files' as const,
      label: copy.tabs.files,
      count: items.files.length,
      rows: items.files
        .slice(0, cap(items.files.length))
        .map(item => <FileRow chat={items.chat} item={item} key={item.id} />)
    },
    {
      id: 'web' as const,
      label: copy.tabs.web,
      count: items.links.length,
      rows: items.links.slice(0, cap(items.links.length)).map(item => <LinkRow item={item} key={item.id} />)
    }
  ]

  return (
    <>
      {tab === 'artifacts' && <p className="chat-items-intro">{copy.artifactsIntro}</p>}
      {sections
        .filter(section => (tab === 'all' || tab === section.id) && section.count > 0)
        .map(section => (
          <Section count={section.count} key={section.id} label={section.label} onSeeAll={seeAll(section.id)}>
            {section.rows}
          </Section>
        ))}
    </>
  )
}

function EmptyItems({ copy, sessionId }: { copy: ChatItemsCopy; sessionId: null | string }) {
  return (
    <div className="chat-items-empty">
      <span className="chat-items-empty-icon">
        <PixelIcon name="attachment" />
      </span>
      <p className="chat-items-empty-title">{copy.emptyTitle}</p>
      <p className="chat-items-empty-text">{sessionId ? copy.emptyDescription : copy.noChat}</p>
    </div>
  )
}

/** The chat items pane (Figma "Dev · chat panel" 1083:8348, "Non-dev · chat panel" 1095:9308): the
 *  focused chat's artifacts, files and links. Developer mode tabs them (All / Artifacts / Files /
 *  Web); simple mode lists them under one "All items" head and closes from it. An artifact opens
 *  inside the pane (05 "Item from list"); a file opens in the preview, a link through the app's link
 *  rules. Only what the transcript and the item index really hold is shown. */
export function ChatItemsPane() {
  const copy = useChatItemsCopy()
  const advanced = useStore($showsAdvancedChrome)
  const sessionId = useStore($focusedStoredSessionId)
  const owner = useChatOwner(sessionId)
  const items = useChatItems(sessionId, owner)
  const [tab, setTab] = useState<ChatItemsTab>('all')
  const [opened, setOpened] = useState<ArtifactRecord | null>(null)

  useEffect(() => {
    setOpened(null)
    setTab('all')
  }, [sessionId])

  const title = items.chat?.title || owner.row?.title || ''

  if (opened) {
    return (
      <aside aria-label={copy.pane} className="chat-items-pane" data-mode={advanced ? 'developer' : 'simple'}>
        <button className="chat-items-back" onClick={() => setOpened(null)} type="button">
          <PixelIcon name="chevron-left" />
          {copy.allItems}
        </button>
        <div className="chat-items-preview">
          <ArtifactPreview onClose={() => setOpened(null)} target={artifactPreviewTarget(opened)} />
        </div>
      </aside>
    )
  }

  const counts: Record<ChatItemsTab, number> = {
    all: items.total,
    artifacts: items.artifacts.length,
    files: items.files.length,
    web: items.links.length
  }

  return (
    <aside aria-label={copy.pane} className="chat-items-pane" data-mode={advanced ? 'developer' : 'simple'}>
      <header className="chat-items-head">
        {advanced ? (
          <h2 className="chat-items-heading">{title || copy.pane}</h2>
        ) : (
          <>
            <h2 className="chat-items-heading">
              {copy.allItems} <span>{items.total}</span>
            </h2>
            <Button aria-label={copy.close} onClick={closeChatItems} size="icon-sm" variant="ghost">
              <PixelIcon name="close" />
            </Button>
          </>
        )}
      </header>
      {advanced && (
        <PageTabs
          label={copy.tabsLabel}
          onChange={id => setTab(id as ChatItemsTab)}
          tabs={(['all', 'artifacts', 'files', 'web'] as const).map(id => ({
            id,
            label: copy.tabs[id],
            meta: counts[id]
          }))}
          value={tab}
        />
      )}
      <div className="chat-items-body">
        {counts[tab] === 0 ? (
          <EmptyItems copy={copy} sessionId={sessionId} />
        ) : (
          <ItemSections
            copy={copy}
            items={items}
            onOpenArtifact={setOpened}
            onTab={setTab}
            simple={!advanced}
            tab={tab}
          />
        )}
      </div>
    </aside>
  )
}
