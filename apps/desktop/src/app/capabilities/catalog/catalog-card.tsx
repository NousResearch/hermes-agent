import './catalog.css'

import { compactNumber } from '@hermes/shared'
import { type ReactNode, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Codicon } from '@/components/ui/codicon'
import { RowButton } from '@/components/ui/row-button'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { Lock } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { type CatalogEntry, catalogLabel } from './catalog-data'
import { CatalogMetadata } from './catalog-metadata'

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
      className={cn('block w-full object-cover', className)}
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

function isFeaturedSource(source: string) {
  const value = source.trim().toLowerCase()

  return (
    value === 'built-in' ||
    value === 'builtin' ||
    value === 'official' ||
    value === 'bundled' ||
    value === 'community' ||
    value === 'verified'
  )
}

interface CatalogCardProps {
  entry: CatalogEntry
  action: ReactNode
  accentIndex: number
  locked?: boolean
  lockedLabel?: string
  onOpen: (entry: CatalogEntry) => void
  onCategory: (category: string) => void
  onTag: (tag: string) => void
  onSearch: (value: string) => void
}

export function CatalogCard({
  entry,
  action,
  accentIndex,
  locked = false,
  lockedLabel = 'Locked',
  onOpen,
  onCategory,
  onTag,
  onSearch
}: CatalogCardProps) {
  const { t } = useI18n()
  const author = entry.author || catalogLabel(entry.source)
  const version = entry.version ? (entry.version.startsWith('v') ? entry.version : `v${entry.version}`) : null
  const featured = isFeaturedSource(entry.source)
  const sourceKey = entry.source.trim().toLowerCase()
  const sourceLabel = !entry.source
    ? null
    : sourceKey === 'built-in' || sourceKey === 'builtin' || sourceKey === 'bundled'
      ? t.catalog.builtIn
      : catalogLabel(entry.source)

  return (
    <article
      className={cn(
        'group relative flex min-w-0 flex-col overflow-hidden rounded-2xl border border-[#e4eaf4] bg-white dark:border-(--ui-stroke-tertiary) dark:bg-(--ui-chat-bubble-background)',
        CARD_ACCENTS[accentIndex % CARD_ACCENTS.length]
      )}
      data-catalog-card
      data-entry-id={entry.id}
      data-locked={locked ? '' : undefined}
    >
      <div className={cn('flex min-h-0 flex-1 flex-col', locked && 'pointer-events-none select-none blur-[3px]')}>
        {entry.imageUrl ? (
          <CatalogImage className="h-[120px] shrink-0" key={entry.imageUrl} src={entry.imageUrl} />
        ) : null}
        <div className="flex min-h-0 w-full flex-1 flex-col gap-2.5 p-3.5">
          <div className="flex min-w-0 items-start justify-between gap-2.5">
            {/* Its ::after stretches over the whole card, so any empty spot opens the
                detail; real controls sit above it (`relative`). One focus stop per card. */}
            <RowButton
              aria-haspopup="dialog"
              aria-label={entry.name}
              className="min-w-0 flex-1 text-left after:absolute after:inset-0 after:rounded-[inherit] focus-visible:outline-none focus-visible:after:outline-2 focus-visible:after:outline-primary focus-visible:after:-outline-offset-2"
              onClick={() => onOpen(entry)}
            >
              <span className="line-clamp-1 min-w-0 break-words text-[14px] font-bold text-[#0f172a] dark:text-foreground">
                {entry.name}
              </span>
              {author ? (
                <span className="mt-0.5 block truncate text-[12px] font-medium text-[#64748b]">{author}</span>
              ) : null}
            </RowButton>
            {action && !locked ? <span className="relative z-[1] shrink-0 pt-0.5">{action}</span> : null}
          </div>

          <div className="relative z-[1] flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1.5">
            {sourceLabel ? (
              <Badge
                className={
                  featured
                    ? 'rounded-lg border border-[#9db7f0] bg-[#e3ecfd] px-2 py-[3px] text-[11px] font-semibold text-[#2f5bea]'
                    : 'rounded-lg'
                }
                size="xs"
                variant={featured ? 'default' : 'muted'}
              >
                <span className="truncate">{sourceLabel}</span>
              </Badge>
            ) : null}
            {version ? <span className="shrink-0 text-[11px] font-medium tabular-nums text-[#64748b]">{version}</span> : null}
            {entry.stars !== null && entry.stars > 0 ? (
              <span className="flex shrink-0 items-center gap-1 text-[11px] font-semibold text-[#334155]">
                <Codicon name="star-full" size="0.75rem" />
                {compactNumber(entry.stars)}
              </span>
            ) : null}
          </div>

          <p className="line-clamp-2 min-h-0 text-[12px] leading-normal text-[#475569]">{entry.description}</p>

          <div className="mt-auto [&_button]:relative">
            <CatalogMetadata
              entry={entry}
              limit={6}
              onCategory={onCategory}
              onOpen={() => onOpen(entry)}
              onSearch={onSearch}
              onTag={onTag}
            />
          </div>
        </div>
      </div>
      {locked ? (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-background/35 backdrop-blur-[2px]">
          <Tip label={lockedLabel}>
            <span
              aria-label={lockedLabel}
              className="flex size-11 items-center justify-center rounded-full border border-white/20 bg-background/80 text-foreground shadow-lg"
            >
              <Lock className="size-4" />
            </span>
          </Tip>
        </div>
      ) : null}
    </article>
  )
}
