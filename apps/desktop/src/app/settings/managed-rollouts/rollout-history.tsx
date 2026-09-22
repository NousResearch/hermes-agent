import { useState } from 'react'

import { useI18n } from '@/i18n'
import { getManagedRolloutMessages } from '@/i18n/managed-rollouts'

export interface RolloutHistoryEntry {
  id: string
  phase: string
  updatedAt: string
  unresolved: number
  archived: boolean
  reason: string | null
}

export function RolloutHistory({
  entries,
  onSelect
}: {
  entries: readonly RolloutHistoryEntry[]
  onSelect: (id: string) => void
}) {
  const { locale, t } = useI18n()
  const copy = getManagedRolloutMessages(t, locale)
  const [selected, setSelected] = useState<string | null>(null)
  const select = (id: string) => {
    setSelected(id)
    onSelect(id)
  }

  return (
    <section aria-label={copy.sections.history} className="grid min-w-0 gap-1">
      {entries.slice(0, 50).map(entry => (
        <button
          aria-label={copy.a11y.historyEntry(entry.id)}
          aria-pressed={selected === entry.id}
          className="grid min-w-0 gap-1 border-b border-(--ui-stroke-secondary) py-2 text-start"
          key={entry.id}
          onClick={() => select(entry.id)}
          type="button"
        >
          <span className="min-w-0 truncate text-sm">
            {copy.labels.historyEntry(entry.id, entry.phase, entry.updatedAt, entry.unresolved, entry.archived)}
          </span>
          {entry.reason ? (
            <span className="min-w-0 truncate text-xs text-(--ui-text-tertiary)">
              {copy.labels.reason(entry.reason)}
            </span>
          ) : null}
        </button>
      ))}
    </section>
  )
}
