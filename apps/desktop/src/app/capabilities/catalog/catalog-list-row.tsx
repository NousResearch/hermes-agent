import type { ReactNode } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { RowButton } from '@/components/ui/row-button'
import { cn } from '@/lib/utils'

import { type CatalogEntry, type CatalogKind, catalogLabel } from './catalog-data'
import { CatalogHeaderMeta, CatalogMetadata } from './catalog-metadata'

interface CatalogListRowProps {
  /** Rendered after the text, above the row overlay (e.g. a sectioned row's
   *  on/off switch). Absent keeps the plain row other kinds render. */
  action?: ReactNode
  entry: CatalogEntry
  kind: CatalogKind
  selected: boolean
  installed: boolean
  /** Extra left inset for rows nested under a section header. */
  indent?: boolean
  onOpen: (entry: CatalogEntry) => void
  onCategory: (category: string) => void
  onTag: (tag: string) => void
  onSearch: (value: string) => void
}

export function CatalogListRow({
  action,
  entry,
  kind,
  selected,
  installed,
  indent,
  onOpen,
  onCategory,
  onTag,
  onSearch
}: CatalogListRowProps) {
  return (
    <div
      className={cn(
        'row-hover relative flex w-full min-w-0 items-start gap-3 rounded-md px-2 py-2 text-left',
        indent && 'pl-5',
        selected && 'bg-(--ui-row-active-background)'
      )}
      data-entry-id={entry.id}
    >
      <RowButton
        aria-label={entry.name}
        aria-pressed={selected}
        className="absolute inset-0 rounded-md focus-visible:outline-2 focus-visible:outline-primary focus-visible:-outline-offset-2"
        onClick={() => onOpen(entry)}
      />
      <Codicon
        className="mt-0.5 shrink-0 text-(--ui-text-tertiary)"
        name={kind === 'plugins' ? 'extensions' : 'book'}
        size="1.1rem"
      />
      <span className="pointer-events-none min-w-0 flex-1">
        <span className="flex items-center gap-2">
          <span className="truncate text-[0.78rem] font-medium">{entry.name}</span>
          {installed && <Codicon className="shrink-0 text-(--ui-text-tertiary)" name="check" />}
        </span>
        <span className="mt-1 line-clamp-2 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
          {entry.description}
        </span>
        <span className="pointer-events-auto relative mt-2 block">
          <CatalogMetadata entry={entry} limit={2} onCategory={onCategory} onSearch={onSearch} onTag={onTag} />
        </span>
        <span className="mt-1.5 flex items-center gap-2 text-[0.65rem] text-(--ui-text-quaternary)">
          <span className="min-w-0 flex-1 truncate">{entry.author || catalogLabel(entry.source)}</span>
          <CatalogHeaderMeta entry={entry} />
        </span>
      </span>
      {action != null && <span className="pointer-events-auto relative mt-0.5 shrink-0">{action}</span>}
    </div>
  )
}
