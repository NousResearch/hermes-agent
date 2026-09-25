import { useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { fmtDateTime, relativeTime } from '@/lib/time'

import { type CatalogEntry, catalogLabel } from './catalog-data'

interface CatalogMetadataProps {
  entry: CatalogEntry
  onCategory?: (category: string) => void
  onTag?: (tag: string) => void
  /** Values without a facet (platforms, tools, env vars) browse by search. */
  onSearch?: (value: string) => void
  /** Capability counts open the entry, where each value is listed. */
  onOpen?: () => void
  /** Details show everything; cards can disclose the rest without opening. */
  limit?: number
}

interface CatalogChip {
  id: string
  label: string
  hint?: string
  quiet?: boolean
  onClick?: () => void
}

/** Source, version and stars, trailing the author line. */
export function CatalogHeaderMeta({ entry }: { entry: CatalogEntry }) {
  return (
    <span className="flex min-w-0 shrink-0 items-center gap-1">
      {entry.source && (
        <Badge className="max-w-full" size="xs" variant="muted">
          <span className="truncate">{catalogLabel(entry.source)}</span>
        </Badge>
      )}
      {entry.version && (
        <Badge size="xs" variant="muted">
          v{entry.version}
        </Badge>
      )}
      {entry.stars !== null && entry.stars > 0 && (
        <span className="flex shrink-0 items-center gap-0.5 text-[0.65rem] text-(--ui-text-tertiary)">
          <span aria-hidden>★</span>
          {entry.stars.toLocaleString()}
        </span>
      )}
    </span>
  )
}

/** One metadata projection for Skills and Plugins, in cards, lists and details. */
export function CatalogMetadata({ entry, onCategory, onTag, onSearch, onOpen, limit }: CatalogMetadataProps) {
  const { t } = useI18n()
  const c = t.catalog
  const [expanded, setExpanded] = useState(false)
  const interactive = Boolean(onCategory || onTag)
  const tags = [...new Set(entry.tags)]
  const search = (value: string) => (onSearch ? () => onSearch(value) : undefined)

  const primary: CatalogChip[] = [
    {
      id: 'category',
      label: catalogLabel(entry.categoryLabel),
      onClick: onCategory ? () => onCategory(entry.category) : undefined
    },
    ...entry.platforms.map(value => ({
      id: `platform:${value}`,
      label: value === 'macos' ? 'macOS' : catalogLabel(value),
      quiet: true,
      onClick: search(value)
    }))
  ]

  const capabilities = [
    { id: 'tools', label: c.tools, values: entry.tools },
    { id: 'hooks', label: c.hooks, values: entry.hooks },
    { id: 'middleware', label: c.middleware, values: entry.middleware ?? [] }
  ]

  const primaryLabels = new Set(primary.map(chip => chip.label.trim().toLowerCase()))

  for (const chip of primary) {
    const matchingTag = tags.find(tag => tag.trim().toLowerCase() === chip.label.trim().toLowerCase())

    if (matchingTag && onTag && (chip.quiet || !chip.onClick)) {
      chip.onClick = () => onTag(matchingTag)
    }
  }

  const secondary: CatalogChip[] = [
    ...capabilities
      .filter(group => group.values.length)
      .map(group => ({ id: group.id, label: `${group.label} ${group.values.length}`, quiet: true, onClick: onOpen })),
    ...tags
      .filter(value => !primaryLabels.has(value.trim().toLowerCase()))
      .map(value => ({ id: `tag:${value}`, label: value, onClick: onTag ? () => onTag(value) : undefined })),
    ...entry.requirements.map(value => ({
      id: `env:${value}`,
      label: value,
      hint: c.requires,
      quiet: true,
      onClick: search(value)
    }))
  ]

  const commands = (entry.commands ?? []).map(value => `/${value.replace(/^\//, '')}`)
  const visible = expanded ? secondary : secondary.slice(0, limit ?? 5)
  const remaining = secondary.length - visible.length

  const chip = ({ id, label, hint, quiet, onClick }: CatalogChip) => {
    const content = onClick ? (
      <Button className="max-w-full" onClick={onClick} size="xs" variant={quiet ? 'ghost' : 'chip'}>
        <span className="truncate">{label}</span>
      </Button>
    ) : (
      <Badge className="max-w-full" variant="muted">
        <span className="truncate">{label}</span>
      </Badge>
    )

    return hint ? (
      <Tip key={id} label={hint}>
        {content}
      </Tip>
    ) : (
      <span className="inline-flex max-w-full" key={id}>
        {content}
      </span>
    )
  }

  return (
    <span className="flex flex-col gap-2" data-catalog-metadata>
      <span className="flex flex-wrap items-center gap-1">{primary.filter(item => item.label).map(chip)}</span>
      {secondary.length > 0 && (
        <span className="flex flex-wrap items-center gap-1" data-catalog-tags>
          {visible.map(chip)}
          {remaining > 0 &&
            (interactive ? (
              <Button
                aria-label={`${c.more} (${remaining})`}
                onClick={() => setExpanded(true)}
                size="xs"
                variant="ghost"
              >
                +{remaining}
              </Button>
            ) : (
              <Badge variant="muted">+{remaining}</Badge>
            ))}
        </span>
      )}
      {commands.length > 0 && (
        <span className="truncate font-mono text-[0.65rem] text-(--ui-text-tertiary)" data-catalog-commands>
          {commands.join('  ')}
        </span>
      )}
    </span>
  )
}

export function CatalogDates({ entry }: { entry: CatalogEntry }) {
  const { t } = useI18n()

  const dates = [
    { label: t.catalog.addedDate, value: entry.addedAt },
    { label: t.catalog.updatedDate, value: entry.updatedAt !== entry.addedAt ? entry.updatedAt : undefined }
  ].filter((date): date is { label: string; value: string } => Boolean(date.value))

  if (!dates.length) {
    return null
  }

  return (
    <span className="flex flex-wrap gap-x-2 gap-y-1 text-[0.65rem] text-(--ui-text-tertiary)">
      {dates.map(({ label, value }) => (
        <time dateTime={value} key={label} title={fmtDateTime.format(Date.parse(value))}>
          {label} {relativeTime(Date.parse(value))}
        </time>
      ))}
    </span>
  )
}
