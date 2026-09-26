import { type ReactNode, useLayoutEffect, useRef } from 'react'

import { ZoomableImage } from '@/components/chat/zoomable-image'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { DialogTitle } from '@/components/ui/dialog'
import { Reel } from '@/components/ui/reel'
import { RowButton } from '@/components/ui/row-button'
import { useI18n } from '@/i18n'
import { ExternalLink } from '@/lib/external-link'
import { cn } from '@/lib/utils'

import { catalogIcon, CatalogImage } from './catalog-card'
import { type CatalogEntry, type CatalogKind, catalogLabel } from './catalog-data'
import { CatalogDates, CatalogHeaderMeta, CatalogMetadata } from './catalog-metadata'

interface CatalogDetailProps {
  entry: CatalogEntry
  kind: CatalogKind
  action?: ReactNode
  management?: ReactNode
  related: CatalogEntry[]
  onCategory?: (category: string) => void
  onTag?: (tag: string) => void
  onSearch?: (value: string) => void
  onRelated?: (entry: CatalogEntry) => void
  /** Inside the modal: the hero bleeds to the edges and paging scrolls back to the top. */
  dialog: boolean
}

/** The shared catalog inspector used by both Skills and Plugins. No destructuring
 *  defaults: React Compiler 1.0 bails on this component with them. */
export function CatalogDetail({
  entry,
  kind,
  action,
  management,
  related,
  onCategory,
  onTag,
  onSearch,
  onRelated,
  dialog
}: CatalogDetailProps) {
  const { t } = useI18n()
  const c = t.catalog
  const root = useRef<HTMLDivElement>(null)
  const Title = dialog ? DialogTitle : 'h3'

  // In the modal the body (this root's parent) is the scroller; a new entry starts at its top.
  useLayoutEffect(() => {
    if (dialog) {
      root.current?.parentElement?.scrollTo({ top: 0 })
    }
  }, [dialog, entry.id])

  const metadata = [
    [c.platforms, entry.platforms.join(', ')],
    [c.requires, entry.requiresHermes ? `Hermes ${entry.requiresHermes}` : ''],
    [c.pinned, entry.sha ? <code>{entry.sha.slice(0, 8)}</code> : '']
  ] as [string, ReactNode][]

  return (
    <div className="min-w-0 space-y-5" data-catalog-detail={kind} ref={root}>
      {entry.imageUrl && (
        <CatalogImage
          className={cn(
            'aspect-[2/1] object-cover',
            dialog ? '-mx-4 -mt-4 w-[calc(100%+2rem)] max-w-none rounded-t-xl' : 'w-full rounded-md'
          )}
          key={entry.imageUrl}
          src={entry.imageUrl}
        />
      )}
      <header className="space-y-3">
        <div className="flex min-w-0 items-start gap-3">
          <div className="min-w-0 flex-1">
            <div className={cn('flex min-w-0 items-center gap-2', dialog && !entry.imageUrl && 'pr-7')}>
              <Title className="min-w-0 flex-1 break-words text-lg font-semibold tracking-tight">{entry.name}</Title>
              {action && <span className="shrink-0">{action}</span>}
            </div>
            <div className="mt-1 flex min-w-0 items-center gap-2 text-xs text-(--ui-text-tertiary)">
              <span className="min-w-0 truncate">{entry.author || catalogLabel(entry.source)}</span>
              <CatalogHeaderMeta entry={entry} />
            </div>
          </div>
        </div>
        {(entry.sourceUrl || entry.docsUrl) && (
          <div className="flex flex-wrap gap-3 text-xs">
            {entry.sourceUrl && <ExternalLink href={entry.sourceUrl}>{c.repository}</ExternalLink>}
            {entry.docsUrl && <ExternalLink href={entry.docsUrl}>{c.documentation}</ExternalLink>}
          </div>
        )}
      </header>

      <p className="whitespace-pre-wrap break-words text-[length:var(--conversation-caption-font-size)] leading-relaxed text-(--ui-text-secondary)">
        {entry.description}
      </p>
      {entry.overview && entry.overview !== entry.description && (
        <p className="whitespace-pre-wrap break-words text-[length:var(--conversation-caption-font-size)] leading-relaxed text-(--ui-text-tertiary)">
          {entry.overview}
        </p>
      )}

      <div className="space-y-3">
        <CatalogMetadata entry={entry} limit={Infinity} onCategory={onCategory} onSearch={onSearch} onTag={onTag} />
        <CatalogDates entry={entry} />
        <dl className="grid grid-cols-[max-content_minmax(0,1fr)] gap-x-4 gap-y-2 text-[length:var(--conversation-caption-font-size)]">
          {metadata
            .filter(([, value]) => Boolean(value))
            .map(([label, value]) => (
              <div className="contents" key={label}>
                <dt className="whitespace-nowrap text-(--ui-text-tertiary)">{label}</dt>
                <dd className="m-0 break-words">{value}</dd>
              </div>
            ))}
        </dl>
      </div>

      {management}

      {Boolean(entry.screenshots?.length) && (
        <section className="space-y-2">
          <h4 className="text-xs font-medium">{c.screenshots}</h4>
          <Reel className={cn(dialog && '-mx-4 scroll-px-4 px-4')}>
            {entry.screenshots?.map((url, index) => (
              <ZoomableImage
                alt={`${c.screenshots} — ${entry.name} ${index + 1}`}
                className="h-56 w-auto max-w-none cursor-zoom-in rounded-lg border border-(--ui-stroke-tertiary)"
                decoding="async"
                key={url}
                loading="lazy"
                referrerPolicy="no-referrer"
                slot="catalog-screenshot"
                src={url}
              />
            ))}
          </Reel>
        </section>
      )}

      {[
        [c.tools, entry.tools],
        [c.hooks, entry.hooks],
        [c.middleware, entry.middleware ?? []],
        [c.requires, entry.requirements]
      ].map(([label, values]) =>
        (values as string[]).length > 0 ? (
          <section className="space-y-2" key={label as string}>
            <h4 className="text-xs font-medium">{label as string}</h4>
            <div className="flex flex-wrap gap-1">
              {(values as string[]).map(value =>
                onSearch ? (
                  <Button key={value} onClick={() => onSearch(value)} size="xs" variant="ghost">
                    {value}
                  </Button>
                ) : (
                  <Badge key={value} variant="muted">
                    {value}
                  </Badge>
                )
              )}
            </div>
          </section>
        ) : null
      )}

      <p className="text-[0.65rem] leading-relaxed text-(--ui-text-quaternary)">{c.installHint}</p>

      {related.length > 0 && (
        <section className="space-y-2">
          <h4 className="text-xs font-medium">{c.related}</h4>
          <div className="grid grid-cols-3 gap-2">
            {related.map(item => (
              <RowButton
                className="row-hover flex min-w-0 flex-col gap-1 overflow-hidden rounded-md text-left"
                key={item.id}
                onClick={() => onRelated?.(item)}
              >
                {item.imageUrl ? (
                  <CatalogImage className="aspect-[2/1] w-full rounded-t-md object-contain" src={item.imageUrl} />
                ) : (
                  <span className="flex aspect-[2/1] w-full items-center justify-center bg-(--ui-bg-quaternary) text-(--ui-text-tertiary)">
                    <Codicon name={catalogIcon(item.category)} size="1.25rem" />
                  </span>
                )}
                <span className="w-full truncate px-2 pb-2 text-xs font-medium">{item.name}</span>
              </RowButton>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}
