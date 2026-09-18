import { useStore } from '@nanostores/react'
import { memo, type ReactNode, useDeferredValue, useEffect, useMemo, useRef, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { ActionStatus } from '@/components/ui/action-status'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Codicon, codiconIcon } from '@/components/ui/codicon'
import { Dialog, DialogContent, DialogTitle } from '@/components/ui/dialog'
import { ErrorState } from '@/components/ui/error-state'
import { RowButton } from '@/components/ui/row-button'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useI18n } from '@/i18n'
import { ExternalLink } from '@/lib/external-link'
import { cn } from '@/lib/utils'

import { DetailColumn, ListColumn, ListStrip, ListStripLabel, MasterDetail, ToolChip } from '../../master-detail'
import { PanelEmpty } from '../../overlays/panel'
import { prettyName } from '../../settings/helpers'
import type { CapabilityView } from '../capability-tabs'
import { $catalogCardView } from '../store'

import { type CatalogEntry, type CatalogKind, useCatalog } from './catalog-data'

interface CatalogBrowserProps {
  kind: CatalogKind
  query?: string
  onQueryChange?: (value: string) => void
  view?: CapabilityView
  isInstalled: (entry: CatalogEntry) => boolean
  onInstall: (entry: CatalogEntry) => void
  isInstalling?: (entry: CatalogEntry) => boolean
  installedEntries?: CatalogEntry[]
  renderInstalledDetail?: (entry: CatalogEntry) => ReactNode
}

const PAGE_SIZE = 60
const ListViewIcon = codiconIcon('list-unordered')
const CardViewIcon = codiconIcon('extensions')

export const CatalogBrowser = memo(function CatalogBrowser({
  kind,
  isInstalled,
  onInstall,
  isInstalling,
  installedEntries,
  renderInstalledDetail,
  view = 'browse',
  query = '',
  onQueryChange
}: CatalogBrowserProps) {
  const { t } = useI18n()
  const c = t.catalog
  const cardView = useStore($catalogCardView)
  const { data, isPending, error, refetch } = useCatalog(kind, view === 'browse')
  const deferredQuery = useDeferredValue(query.trim().toLowerCase())
  const [source, setSource] = useState('all')
  const [category, setCategory] = useState('all')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [detailOpen, setDetailOpen] = useState(false)
  const selectedCardRef = useRef<HTMLButtonElement | null>(null)
  const [limit, setLimit] = useState(PAGE_SIZE)

  const entries = useMemo(() => {
    return view === 'installed' ? (installedEntries ?? []) : (data ?? [])
  }, [data, installedEntries, view])

  const sources = useMemo(() => [...new Set(entries.map(entry => entry.source))], [entries])

  const categories = useMemo(() => {
    const values = new Map<string, string>()

    for (const entry of entries) {
      if (source === 'all' || entry.source === source) {
        values.set(entry.category, entry.categoryLabel)
      }
    }

    return [...values].sort((a, b) => a[1].localeCompare(b[1]))
  }, [entries, source])

  const filtered = useMemo(() => entries.filter(entry =>
    (source === 'all' || entry.source === source) &&
    (category === 'all' || entry.category === category) &&
    (!deferredQuery || entry.search.includes(deferredQuery))
  ), [entries, source, category, deferredQuery])

  const selected = filtered.find(entry => entry.id === selectedId) ?? filtered[0]
  const installedDetail = selected ? renderInstalledDetail?.(selected) : null

  const resetSelection = () => {
    setLimit(PAGE_SIZE)
    setSelectedId(null)
    setDetailOpen(false)
  }

  useEffect(() => {
    setLimit(PAGE_SIZE)
    setSelectedId(null)
    setDetailOpen(false)
  }, [deferredQuery])

  const clearFilters = () => {
    onQueryChange?.('')
    setSource('all')
    setCategory('all')
    resetSelection()
  }

  const installButton = (entry: CatalogEntry) => {
    const installed = isInstalled(entry)
    const installing = isInstalling?.(entry) ?? false

    return (
      <Button disabled={installed || installing || (kind === 'skills' && !entry.installIdentifier)} onClick={() => {
        if (kind === 'plugins') {
          setDetailOpen(false)
        }

        onInstall(entry)
      }} size="sm" variant={installed ? 'secondary' : 'default'}>
        <ActionStatus
          busy={t.skills.hub.installing}
          done={c.installed}
          idle={t.skills.hub.install}
          idleIcon={<Codicon name="cloud-download" />}
          state={installed ? 'done' : installing ? 'saving' : 'idle'}
        />
      </Button>
    )
  }

  const metadata = selected ? [
    [c.author, selected.author],
    [c.source, prettyName(selected.source)],
    [c.category, prettyName(selected.categoryLabel)],
    [c.version, selected.version],
    [c.platforms, selected.platforms.join(', ')],
    [c.requires, selected.requiresHermes ? `Hermes ${selected.requiresHermes}` : ''],
    [c.pinned, selected.sha ? <code title={selected.sha}>{selected.sha.slice(0, 8)}</code> : '']
  ] as [string, ReactNode][] : []

  const DetailTitle = view === 'browse' && cardView ? DialogTitle : 'h3'

  const details = selected ? (
    <>
      <header className="space-y-3">
        {selected.imageUrl && (
          <img
            alt=""
            className="block aspect-[2/1] w-full rounded-md border border-(--ui-border) object-cover"
            decoding="async"
            loading="lazy"
            onError={e => { e.currentTarget.style.display = 'none' }}
            referrerPolicy="no-referrer"
            src={selected.imageUrl}
          />
        )}
        <div className="flex items-start gap-3">
          <Codicon className="mt-1 shrink-0 text-(--ui-text-tertiary)" name={kind === 'plugins' ? 'extensions' : 'book'} size="1.5rem" />
          <div className="min-w-0">
            <DetailTitle className="break-words text-lg font-semibold tracking-tight">{selected.name}</DetailTitle>
            <p className="mt-1 text-xs text-(--ui-text-tertiary)">{selected.author || prettyName(selected.source)}</p>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {installButton(selected)}
          <Badge variant="muted">{prettyName(selected.source)}</Badge>
          {selected.stars !== null && <span className="text-xs text-(--ui-text-tertiary)">☆ {selected.stars.toLocaleString()}</span>}
        </div>
      </header>
      <section className="space-y-2">
        <h4 className="text-xs font-medium">{c.about}</h4>
        <p className="whitespace-pre-wrap break-words text-[length:var(--conversation-caption-font-size)] leading-relaxed text-(--ui-text-secondary)">{selected.description}</p>
        {selected.overview && selected.overview !== selected.description && <p className="whitespace-pre-wrap break-words text-[length:var(--conversation-caption-font-size)] leading-relaxed text-(--ui-text-tertiary)">{selected.overview}</p>}
      </section>
      <dl className="space-y-2 text-[length:var(--conversation-caption-font-size)]">
        {metadata.filter(([, value]) => Boolean(value)).map(([label, value]) => <div className="grid grid-cols-[6rem_minmax(0,1fr)] gap-3" key={label}><dt className="text-(--ui-text-tertiary)">{label}</dt><dd className="m-0 break-words">{value}</dd></div>)}
      </dl>
      {[[c.tools, selected.tools], [c.hooks, selected.hooks], [c.requires, selected.requirements]].map(([label, values]) => (values as string[]).length > 0 && (
        <section className="space-y-2" key={label as string}><h4 className="text-xs font-medium">{label as string}</h4><div className="flex flex-wrap gap-1">{(values as string[]).map(value => <ToolChip key={value}>{value}</ToolChip>)}</div></section>
      ))}
      <div className="flex flex-wrap gap-3 text-xs">
        {selected.sourceUrl && <ExternalLink href={selected.sourceUrl}>{c.repository}</ExternalLink>}
        {selected.docsUrl && <ExternalLink href={selected.docsUrl}>{c.documentation}</ExternalLink>}
      </div>
      <p className="text-[0.65rem] leading-relaxed text-(--ui-text-quaternary)">{c.installHint}</p>
    </>
  ) : null

  return (
    <div className="flex h-full min-h-0 min-w-0 flex-col" data-catalog={kind}>
      {view === 'browse' && <div className="flex shrink-0 flex-wrap items-center gap-x-4 gap-y-2 px-3 py-2">
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <div className="w-36">
            <Select onValueChange={value => { setSource(value); setCategory('all'); resetSelection() }} value={source}>
              <SelectTrigger aria-label={c.source}><SelectValue placeholder={c.allSources} /></SelectTrigger>
              <SelectContent>
                <SelectItem value="all">{c.allSources}</SelectItem>
                {sources.map(value => <SelectItem key={value} value={value}>{prettyName(value)}</SelectItem>)}
              </SelectContent>
            </Select>
          </div>
          <div className="w-40">
            <Select onValueChange={value => { setCategory(value); resetSelection() }} value={category}>
              <SelectTrigger aria-label={c.category}><SelectValue placeholder={c.allCategories} /></SelectTrigger>
              <SelectContent>
                <SelectItem value="all">{c.allCategories}</SelectItem>
                {categories.map(([value, label]) => <SelectItem key={value} value={value || 'uncategorized'}>{prettyName(label)}</SelectItem>)}
              </SelectContent>
            </Select>
          </div>
        </div>
        <SegmentedControl
          className="ml-auto shrink-0"
          iconOnly
          onChange={layout => {
            setDetailOpen(false)
            $catalogCardView.set(layout === 'cards')
          }}
          options={[{ id: 'list', label: c.listView, icon: ListViewIcon }, { id: 'cards', label: c.cardView, icon: CardViewIcon }]}
          value={cardView ? 'cards' : 'list'}
        />
      </div>}
      <div className="min-h-0 flex-1">
        {isPending && view === 'browse' && !entries.length ? <PageLoader label={t.skills.loading} /> : error && !entries.length ? (
          <div className="grid h-full place-items-center p-5">
            <ErrorState description={error.message} title={c.loadFailed}>
              <Button onClick={() => void refetch()} size="sm" variant="secondary">{c.retry}</Button>
            </ErrorState>
          </div>
        ) : !selected ? (
          <PanelEmpty action={<Button onClick={clearFilters} size="sm" variant="secondary">{c.clearFilters}</Button>} description={c.tryAnother} icon="search" title={c.noResults} />
        ) : view === 'browse' && cardView ? (
          <>
            <div className="flex h-full min-h-0 flex-col px-3 pb-3" data-catalog-cards={kind}>
              <ListStrip left={<ListStripLabel>{c.results(filtered.length)}</ListStripLabel>} />
              <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain [scrollbar-gutter:stable]" key={`${source}:${category}:${deferredQuery}`}>
                <div className="grid grid-cols-[repeat(auto-fill,minmax(min(100%,17rem),1fr))] gap-3 py-2">
                  {filtered.slice(0, limit).map(entry => (
                    <article className="row-hover flex min-w-0 flex-col overflow-hidden rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-chat-bubble-background)" data-catalog-card key={entry.id}>
                      <RowButton
                        aria-haspopup="dialog"
                        aria-label={entry.name}
                        className="flex min-w-0 flex-1 cursor-pointer flex-col gap-3 p-4 text-left focus-visible:outline-2 focus-visible:outline-primary focus-visible:-outline-offset-2"
                        onClick={event => {
                          selectedCardRef.current = event.currentTarget
                          setSelectedId(entry.id)
                          setDetailOpen(true)
                        }}
                      >
                        <span className="flex w-full items-center gap-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                          <Codicon name={kind === 'plugins' ? 'extensions' : 'book'} size="1.25rem" />
                          <span className="min-w-0 flex-1 truncate">{prettyName(entry.source)}</span>
                          {entry.stars !== null && <span className="shrink-0">☆ {entry.stars.toLocaleString()}</span>}
                        </span>
                        <span className="line-clamp-2 break-words text-[length:var(--conversation-text-font-size)] font-semibold leading-snug">{entry.name}</span>
                        <span className="line-clamp-3 text-[length:var(--conversation-caption-font-size)] leading-relaxed text-(--ui-text-secondary)">{entry.description}</span>
                        <span className="mt-auto w-full truncate pt-1 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">{entry.author || prettyName(entry.source)}</span>
                      </RowButton>
                      <div className="flex flex-wrap items-center justify-between gap-2 px-4 pb-4">
                        <Badge variant="muted">{prettyName(entry.categoryLabel)}</Badge>
                        {installButton(entry)}
                      </div>
                    </article>
                  ))}
                </div>
                {filtered.length > limit && <Button onClick={() => setLimit(value => value + PAGE_SIZE)} size="sm" variant="text">{c.more}</Button>}
              </div>
            </div>
            <Dialog onOpenChange={setDetailOpen} open={detailOpen}>
              <DialogContent
                aria-describedby={undefined}
                bodyClassName="gap-5"
                className="max-w-2xl"
                onCloseAutoFocus={event => {
                  event.preventDefault()
                  selectedCardRef.current?.focus()
                }}
              >
                {details}
              </DialogContent>
            </Dialog>
          </>
        ) : (
          <div className={cn('h-full min-h-0', detailOpen ? '[&_aside]:max-sm:hidden' : '[&_main]:max-sm:hidden')} data-catalog-list={kind}>
            <MasterDetail resizeId="capabilities-split" split="wide">
              <ListColumn header={<ListStrip left={<ListStripLabel>{c.results(filtered.length)}</ListStripLabel>} />} key={`${source}:${category}:${deferredQuery}`}>
                {filtered.slice(0, limit).map(entry => (
                  <RowButton
                    aria-pressed={entry.id === selected.id}
                    className={cn('row-hover flex w-full min-w-0 items-start gap-3 rounded-md px-2 py-2 text-left', entry.id === selected.id && 'bg-(--ui-row-active-background)')}
                    key={entry.id}
                    onClick={() => { setSelectedId(entry.id); setDetailOpen(true) }}
                  >
                    <Codicon className="mt-0.5 shrink-0 text-(--ui-text-tertiary)" name={kind === 'plugins' ? 'extensions' : 'book'} size="1.1rem" />
                    <span className="min-w-0 flex-1">
                      <span className="flex items-center gap-2">
                        <span className="truncate text-[0.78rem] font-medium">{entry.name}</span>
                        {isInstalled(entry) && <Codicon className="shrink-0 text-(--ui-text-tertiary)" name="check" />}
                      </span>
                      <span className="mt-1 line-clamp-2 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">{entry.description}</span>
                      <span className="mt-1.5 flex items-center gap-2 text-[0.65rem] text-(--ui-text-quaternary)">
                        <span className="truncate">{entry.author || prettyName(entry.source)}</span>
                        {entry.stars !== null && <span className="shrink-0">☆ {entry.stars.toLocaleString()}</span>}
                      </span>
                    </span>
                  </RowButton>
                ))}
                {filtered.length > limit && <Button onClick={() => setLimit(value => value + PAGE_SIZE)} size="sm" variant="text">{c.more}</Button>}
              </ListColumn>
              <DetailColumn footer={c.snapshotHint}>
                <div className="sm:hidden"><Button onClick={() => setDetailOpen(false)} size="sm" variant="text"><Codicon name="arrow-left" />{c.back}</Button></div>
                {installedDetail || details}
              </DetailColumn>
            </MasterDetail>
          </div>
        )}
      </div>
    </div>
  )
})
