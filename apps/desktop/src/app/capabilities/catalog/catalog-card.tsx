import './catalog.css'

import { type CSSProperties, type ReactNode, useState } from 'react'

import { RowButton } from '@/components/ui/row-button'
import { cn } from '@/lib/utils'

import { type CatalogEntry, catalogLabel } from './catalog-data'
import { CatalogDates, CatalogHeaderMeta, CatalogMetadata, CatalogRating } from './catalog-metadata'

export function catalogIcon(category: string) {
  const icons: Record<string, string> = {
    desktop: 'layout',
    tools: 'tools',
    memory: 'database',
    platform: 'plug',
    models: 'hubot',
    web: 'globe',
    automation: 'run-all',
    voice: 'mic',
    'software-development': 'code',
    'autonomous-ai-agents': 'hubot',
    productivity: 'checklist',
    creative: 'paintcan',
    research: 'beaker',
    science: 'beaker',
    security: 'shield',
    media: 'play',
    apple: 'device-desktop'
  }

  return icons[category] ?? 'extensions'
}

export function CatalogImage({ src, className }: { src: string; className?: string }) {
  const [failed, setFailed] = useState(false)

  return failed ? null : (
    <img
      alt=""
      className={cn('block w-full object-contain', className)}
      decoding="async"
      loading="lazy"
      onError={() => setFailed(true)}
      referrerPolicy="no-referrer"
      src={src}
    />
  )
}

// Shared classes, not per-card inline custom properties: identical declarations
// let the style engine reuse computed styles across cards. catalog.css paints them.
const CARD_ACCENTS = [
  '[--catalog-accent:var(--ui-blue)]',
  '[--catalog-accent:var(--ui-orange)]',
  '[--catalog-accent:var(--ui-purple)]',
  '[--catalog-accent:var(--ui-green)]',
  '[--catalog-accent:var(--ui-red)]',
  '[--catalog-accent:var(--ui-cyan)]',
  '[--catalog-accent:var(--ui-yellow)]'
]

export type CatalogCardVariant = 'default' | 'hero' | 'showcase' | 'compact' | 'horizontal'

interface CatalogCardProps {
  variant?: CatalogCardVariant
  entry: CatalogEntry
  action: ReactNode
  accentIndex: number
  onOpen: (entry: CatalogEntry) => void
  onCategory: (category: string) => void
  onTag: (tag: string) => void
  onSearch: (value: string) => void
  style?: CSSProperties
}

export function CatalogCard({ entry, action, accentIndex, onOpen, onCategory, onTag, onSearch, variant = 'default', style }: CatalogCardProps) {
  return (
    <article
      className={cn(
        'group relative flex min-w-0 flex-col overflow-hidden rounded-lg border',
        CARD_ACCENTS[accentIndex % CARD_ACCENTS.length]
      )}
      data-catalog-card
      data-catalog-editorial-art={(variant !== 'hero' && entry.imageUrl?.includes("official-art")) || undefined}
      data-catalog-variant={variant}
      data-entry-id={entry.id}
      style={style}
    >
      {variant !== 'hero' && entry.imageUrl && ((variant !== 'compact' && variant !== 'showcase') || entry.imageUrl.includes('official-art')) && <CatalogImage className="catalog-card-image aspect-[2/1]" key={entry.imageUrl} src={entry.imageUrl} />}
      {variant !== 'default' && <div className="catalog-card-topline">
        <span className="min-w-0 truncate" title={entry.author || catalogLabel(entry.source)}>{entry.author || catalogLabel(entry.source)}</span>
        {variant !== 'hero' && <span className="relative shrink-0">{action}</span>}
      </div>}
      <div className="catalog-card-copy flex min-h-0 w-full flex-col gap-2 p-3">
        <div className="flex min-w-0 items-center gap-2.5">
          {/* Its ::after stretches over the whole card, so any empty spot opens the
              detail; real controls sit above it (`relative`). One focus stop per card. */}
          <RowButton
            aria-haspopup="dialog"
            aria-label={entry.name}
            className="min-w-0 flex-1 text-left font-semibold leading-snug after:absolute after:inset-0 after:rounded-[inherit] focus-visible:outline-none focus-visible:after:outline-2 focus-visible:after:outline-primary focus-visible:after:-outline-offset-2"
            onClick={() => onOpen(entry)}
          >
            <span className="catalog-card-title line-clamp-2 min-w-0 break-words text-lg">{variant === 'default' ? entry.name : catalogLabel(entry.name)}</span>
          </RowButton>
          {variant === 'default' && action && (
            <span className="relative shrink-0 opacity-[.66] transition-opacity group-hover:opacity-100 group-hover:transition-none group-focus-within:opacity-100 group-focus-within:transition-none">
              {action}
            </span>
          )}
        </div>
        {/* In a fixed-height grid cell the description is what gives way. */}
        <div className="flex min-h-0 min-w-0 flex-col gap-2">
          {variant === 'default' && <span className="flex w-full items-center gap-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
            <span className="min-w-0 flex-1 truncate">{entry.author || catalogLabel(entry.source)}</span>
            <CatalogHeaderMeta entry={entry} />
            <CatalogRating entry={entry} />
          </span>}
          <span className="catalog-card-description line-clamp-4 min-h-0 text-[length:var(--conversation-caption-font-size)] leading-relaxed text-(--ui-text-secondary)">
            {entry.description}
          </span>
        </div>
      </div>
      {variant === 'default' && <div className="catalog-card-metadata px-3 pb-3 [&_button]:relative">
        <CatalogMetadata
          entry={entry}
          onCategory={onCategory}
          onOpen={() => onOpen(entry)}
          onSearch={onSearch}
          onTag={onTag}
        />
      </div>}
      {variant !== 'default' && (
        <div className="catalog-card-footer">
          <CatalogHeaderMeta entry={entry} />
          <span className="flex shrink-0 items-center gap-3">
            <CatalogRating entry={entry} />
            {variant === 'hero' && <span className="catalog-hero-action relative shrink-0">{action}</span>}
          </span>
        </div>
      )}
      {variant === 'default' && <div className="catalog-card-dates mt-auto px-3 pb-3">
        <CatalogDates entry={entry} />
      </div>}
    </article>
  )
}
