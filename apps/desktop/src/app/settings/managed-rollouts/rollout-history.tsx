import { useState } from 'react'

export interface RolloutHistoryEntry { id: string; phase: string; updatedAt: string; unresolved: number; archived: boolean; reason: string | null }

export function RolloutHistory({ entries, onSelect }: { entries: readonly RolloutHistoryEntry[]; onSelect: (id: string) => void }) {
  const [selected, setSelected] = useState<string | null>(null)
  const select = (id: string) => { setSelected(id); onSelect(id) }
  return <section aria-label="Managed rollout history" className="grid gap-1">{entries.slice(0, 50).map(entry => <button aria-pressed={selected === entry.id} className="grid gap-1 border-b border-(--ui-stroke-secondary) py-2 text-left" key={entry.id} onClick={() => select(entry.id)}><span className="text-sm">{entry.id} · {entry.phase}</span><span className="text-xs text-(--ui-text-tertiary)">{entry.updatedAt} · unresolved {entry.unresolved} · {entry.archived ? 'archived' : 'active'}</span>{entry.reason ? <span className="text-xs text-(--ui-text-tertiary)">{entry.reason}</span> : null}</button>)}</section>
}
