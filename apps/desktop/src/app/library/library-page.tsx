import './library.css'

import { type ComponentProps, type ReactNode, useCallback, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router'

import { Button } from '@/components/ui/button'
import { EmptyState } from '@/components/ui/empty-state'
import { PixelIcon } from '@/components/ui/pixel-icon'
import { SearchField } from '@/components/ui/search-field'
import { Skeleton } from '@/components/ui/skeleton'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import type { PixelIconName } from '@/lib/pixel-icon-registry'
import { normalize } from '@/lib/text'
import { fmtClock } from '@/lib/time'

import { ARTIFACT_FILTERS, type ArtifactFilter, type ArtifactRecord } from '../artifacts/artifact-utils'
import type { IndexedChat } from '../artifacts/item-index-snapshot'
import { useRefreshHotkey } from '../hooks/use-refresh-hotkey'
import { useRouteEnumParam } from '../hooks/use-route-enum-param'
import { PageSidebarList, PageSidebarPanel, PageSidebarRow } from '../overlays/page-sidebar'
import { PageHeader } from '../redesign/page-header'

import { chatOfItem, downloadItem, openFileItem, openItemChat } from './library-actions'
import { ImageGrid, ItemTable, Pager } from './library-collections'
import { useLibraryCopy } from './library-copy'
import { ImageLightbox, ImagePreview } from './library-image-view'
import { useItemIndex } from './use-item-index'

const IMAGE_PAGE = 24
const TABLE_PAGE = 100
/** The All view's strips (1163:17585): one row of images, a short table. */
const ALL_IMAGES = 4
const ALL_ROWS = 6

const TAB_ICONS: Record<ArtifactFilter, PixelIconName> = {
  all: 'grid-2x2-2',
  image: 'image',
  file: 'file',
  link: 'link'
}

function matches(item: ArtifactRecord, chat: IndexedChat | undefined, query: string): boolean {
  return (
    !query ||
    item.label.toLowerCase().includes(query) ||
    item.value.toLowerCase().includes(query) ||
    (chat?.title ?? item.sessionTitle).toLowerCase().includes(query)
  )
}

function LibrarySkeleton({ label }: { label: string }) {
  return (
    <div aria-busy="true" aria-label={label} className="library-skeleton" role="status">
      <div className="library-grid">
        {Array.from({ length: ALL_IMAGES }, (_, index) => (
          <Skeleton className="library-skeleton-card" key={index} />
        ))}
      </div>
      <div className="library-table-card">
        {Array.from({ length: ALL_ROWS }, (_, index) => (
          <div className="library-skeleton-row" key={index}>
            <Skeleton className="h-3 w-1/3" />
            <Skeleton className="ml-auto h-3 w-24" />
          </div>
        ))}
      </div>
    </div>
  )
}

function SectionHead({ count, label, onViewAll }: { count: number; label: string; onViewAll: () => void }) {
  const copy = useLibraryCopy()

  return (
    <div className="library-section-head">
      <h2>
        {label} <span>{count}</span>
      </h2>
      <Button aria-label={`${copy.viewAll}: ${label}`} onClick={onViewAll} size="sm" variant="secondary">
        {copy.viewAll}
        <PixelIcon name="chevron-right" />
      </Button>
    </div>
  )
}

interface LibraryBodyProps {
  tab: ArtifactFilter
  images: ArtifactRecord[]
  rows: ArtifactRecord[]
  chats: Record<string, IndexedChat>
  page: number
  onPage: (page: number) => void
  onTab: (tab: ArtifactFilter) => void
  onOpenImage: (item: ArtifactRecord) => void
  onOpenItem: (item: ArtifactRecord) => void
  onOpenChat: (item: ArtifactRecord) => void
}

/** The stage under the header: All's two strips, the Images grid, or the Files / Links table. */
function LibraryBody({
  chats,
  images,
  onOpenChat,
  onOpenImage,
  onOpenItem,
  onPage,
  onTab,
  page,
  rows,
  tab
}: LibraryBodyProps) {
  const copy = useLibraryCopy()

  const table = (items: ArtifactRecord[]) => (
    <ItemTable chats={chats} filter={tab} items={items} onOpen={onOpenItem} onOpenChat={onOpenChat} />
  )

  if (tab === 'all') {
    return (
      <>
        {images.length > 0 && (
          <section className="library-section">
            <SectionHead count={images.length} label={copy.tabs.image} onViewAll={() => onTab('image')} />
            <ImageGrid chats={chats} items={images.slice(0, ALL_IMAGES)} onOpen={onOpenImage} />
          </section>
        )}
        {rows.length > 0 && (
          <section className="library-section">
            <SectionHead count={rows.length} label={copy.sectionFilesLinks} onViewAll={() => onTab('file')} />
            {table(rows.slice(0, ALL_ROWS))}
          </section>
        )}
      </>
    )
  }

  const list = tab === 'image' ? images : rows
  const size = tab === 'image' ? IMAGE_PAGE : TABLE_PAGE
  const shown = list.slice((page - 1) * size, page * size)

  return (
    <section className="library-section">
      <Pager filter={tab} onPage={onPage} page={page} pageSize={size} total={list.length} />
      {tab === 'image' ? <ImageGrid chats={chats} items={shown} onOpen={onOpenImage} /> : table(shown)}
    </section>
  )
}

interface LibraryStateProps {
  body: ReactNode
  error: unknown
  loading: boolean
  /** The search that found nothing ('' when none is typed). */
  noMatches: string
  shown: number
  onRetry: () => void
}

/** What the stage shows under its header: the error (over whatever was found before), the first
 *  load's skeleton, an empty or no-match state, or the items. */
function LibraryState({ body, error, loading, noMatches, onRetry, shown }: LibraryStateProps) {
  const copy = useLibraryCopy()

  return (
    <>
      {Boolean(error) && (
        <div className="library-error" role="alert">
          <EmptyState description={copy.errorDescription} title={copy.errorTitle} />
          <Button onClick={onRetry} variant="secondary">
            {copy.retry}
          </Button>
        </div>
      )}
      {loading ? (
        <LibrarySkeleton label={copy.loading} />
      ) : shown > 0 ? (
        body
      ) : (
        !error && (
          <div className="library-empty">
            {noMatches ? (
              <EmptyState description={copy.noMatchesDescription(noMatches)} title={copy.noMatchesTitle} />
            ) : (
              <EmptyState description={copy.emptyDescription} title={copy.emptyTitle} />
            )}
          </div>
        )
      )}
    </>
  )
}

/** The redesign's Library (Figma 1163:17584): the page panel with the kinds and their counts, the
 *  stage with the current kind, an image's preview on the stage and its full-window view. It paints
 *  from the shared item index (store/item-index.ts), so a revisit shows the last list at once. */
export function LibraryPage(props: ComponentProps<'section'>) {
  const { t } = useI18n()
  const copy = useLibraryCopy()
  const navigate = useNavigate()
  const index = useItemIndex()
  const [tab, setTab] = useRouteEnumParam('tab', ARTIFACT_FILTERS, 'all')
  const [queries, setQueries] = useState<Record<ArtifactFilter, string>>({ all: '', image: '', file: '', link: '' })
  const [pages, setPages] = useState<Record<ArtifactFilter, number>>({ all: 1, image: 1, file: 1, link: 1 })
  const [preview, setPreview] = useState<null | string>(null)
  const [lightbox, setLightbox] = useState(false)
  const searchRef = useRef<HTMLInputElement>(null)
  const enlargeRef = useRef<HTMLButtonElement>(null)
  const { chats, items } = index

  useRefreshHotkey(() => void index.refresh())

  const query = normalize(queries[tab])

  const { images, rows } = useMemo(() => {
    const kept = items.filter(
      item => (tab === 'all' || item.kind === tab) && matches(item, chatOfItem(chats, item), query)
    )

    return { images: kept.filter(item => item.kind === 'image'), rows: kept.filter(item => item.kind !== 'image') }
  }, [chats, items, query, tab])

  const counts = useMemo(() => {
    const count = (kind: ArtifactFilter) =>
      kind === 'all' ? items.length : items.filter(item => item.kind === kind).length

    return Object.fromEntries(ARTIFACT_FILTERS.map(kind => [kind, count(kind)])) as Record<ArtifactFilter, number>
  }, [items])

  const pageCount = Math.max(
    1,
    Math.ceil((tab === 'image' ? images : rows).length / (tab === 'image' ? IMAGE_PAGE : TABLE_PAGE))
  )

  const page = Math.min(pages[tab], pageCount)
  const previewIndex = preview ? images.findIndex(item => item.id === preview) : -1

  const selectTab = useCallback(
    (next: ArtifactFilter) => {
      setPreview(null)
      setTab(next)
    },
    [setTab]
  )

  const openChat = useCallback(
    (item: ArtifactRecord) => openItemChat(item, chatOfItem(chats, item), navigate),
    [chats, navigate]
  )

  const download = useCallback((item: ArtifactRecord) => void downloadItem(item, chatOfItem(chats, item)), [chats])

  const openItem = useCallback((item: ArtifactRecord) => void openFileItem(item, chatOfItem(chats, item)), [chats])

  const loading = items.length === 0 && (index.checkedAt === null || index.refreshing) && !index.error
  const summaryTime = index.checkedAt === null ? null : fmtClock.format(new Date(index.checkedAt))

  const refresh = (
    <Tip label={index.refreshing ? copy.refreshing : copy.refresh}>
      <Button
        aria-label={index.refreshing ? copy.refreshing : copy.refresh}
        className="library-refresh"
        data-busy={index.refreshing || undefined}
        disabled={index.refreshing}
        onClick={() => void index.refresh()}
        size="icon"
        variant="secondary"
      >
        <PixelIcon name={index.refreshing ? 'loader' : 'refresh'} />
      </Button>
    </Tip>
  )

  const search = (
    <SearchField
      aria-label={copy.search[tab]}
      containerClassName="library-search"
      inputRef={searchRef}
      onChange={value => {
        setQueries(current => ({ ...current, [tab]: value }))
        setPages(current => ({ ...current, [tab]: 1 }))
      }}
      placeholder={copy.search[tab]}
      value={queries[tab]}
      variant="box"
    />
  )

  const body = (
    <LibraryBody
      chats={chats}
      images={images}
      onOpenChat={openChat}
      onOpenImage={item => setPreview(item.id)}
      onOpenItem={openItem}
      onPage={next => setPages(current => ({ ...current, [tab]: next }))}
      onTab={selectTab}
      page={page}
      rows={rows}
      tab={tab}
    />
  )

  const viewImage =
    previewIndex >= 0 ? { chats, index: previewIndex, items: images, onDownload: download, onOpenChat: openChat } : null

  return (
    <section {...props} aria-label={t.simplifiedAgents.library} className="library-page" data-page-surface="">
      <PageSidebarPanel
        caption={copy.summary(counts.all, summaryTime)}
        headAction={
          <Tip label={copy.search.all}>
            <Button
              aria-label={copy.search.all}
              className="library-panel-search"
              onClick={() => searchRef.current?.focus()}
              size="icon-sm"
              variant="secondary"
            >
              <PixelIcon name="search" />
            </Button>
          </Tip>
        }
        title={t.simplifiedAgents.library}
      >
        <PageSidebarList>
          {ARTIFACT_FILTERS.map(kind => (
            <PageSidebarRow current={kind === tab} key={kind} onClick={() => selectTab(kind)}>
              <PixelIcon className="page-sidebar-glyph" name={TAB_ICONS[kind]} />
              <span className="library-nav-label">{copy.tabs[kind]}</span>
              <span className="page-sidebar-meta">{loading ? '' : counts[kind]}</span>
            </PageSidebarRow>
          ))}
        </PageSidebarList>
      </PageSidebarPanel>

      <div className="library-stage">
        {viewImage ? (
          <ImagePreview
            {...viewImage}
            backLabel={copy.tabs[tab]}
            enlargeRef={enlargeRef}
            onBack={() => setPreview(null)}
            onEnlarge={() => setLightbox(true)}
            onIndex={next => setPreview(images[next]?.id ?? null)}
          />
        ) : (
          <>
            <PageHeader
              actions={
                <>
                  {search}
                  {refresh}
                </>
              }
              title={copy.tabs[tab]}
            />
            <LibraryState
              body={body}
              error={index.error}
              loading={loading}
              noMatches={query ? queries[tab] : ''}
              onRetry={() => void index.refresh()}
              shown={images.length + rows.length}
            />
          </>
        )}
      </div>

      {viewImage && lightbox && (
        <ImageLightbox
          {...viewImage}
          onClose={() => setLightbox(false)}
          onIndex={next => setPreview(images[next]?.id ?? null)}
          returnFocus={enlargeRef}
        />
      )}
    </section>
  )
}
