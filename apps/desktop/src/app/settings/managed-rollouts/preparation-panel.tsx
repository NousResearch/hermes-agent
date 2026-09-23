import { useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { SearchField } from '@/components/ui/search-field'
import { useI18n } from '@/i18n'
import { getManagedRolloutMessages } from '@/i18n/managed-rollouts'

import { FleetOverview } from './fleet-overview'
import { type ManagedRolloutTarget, targetIdentity } from './target-row'

export function PreparationPanel({ targets, reviewGeneration, disabled = false, reviewDisabled = false, onPrepare, onReview }: { targets: readonly ManagedRolloutTarget[]; reviewGeneration: string; disabled?: boolean; reviewDisabled?: boolean; onPrepare: (targets: ManagedRolloutTarget[]) => Promise<void> | void; onReview?: (targets: ManagedRolloutTarget[]) => void }) {
  const { t } = useI18n()
  const messages = getManagedRolloutMessages(t)
  const [selected, setSelected] = useState<ReadonlySet<string>>(new Set())
  const [confirmed, setConfirmed] = useState(false)
  const [pending, setPending] = useState(false)
  const [query, setQuery] = useState('')
  useEffect(() => { setConfirmed(false); setPending(false); setSelected(new Set()); setQuery('') }, [reviewGeneration])
  const chosen = useMemo(() => targets.filter(target => selected.has(targetIdentity(target))), [selected, targets])

  const visibleTargets = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase()

    return needle ? targets.filter(target => [target.label, target.installId, target.machineId, target.headSha, ...(target.aliases ?? [])].some(value => value?.toLocaleLowerCase().includes(needle))) : targets
  }, [query, targets])

  const toggle = (identity: string) => { setConfirmed(false); setSelected(current => { const next = new Set(current);

 if (next.has(identity)) {next.delete(identity);} else {next.add(identity);}

 return next }) }

  const confirmPreparation = async () => {
    if (!confirmed || chosen.length === 0 || pending || disabled) {return}
    setPending(true)
    setConfirmed(false)

    try { await onPrepare(chosen) } finally { setPending(false) }
  }

  return <section aria-label={messages.sections.preparation} className="grid min-w-0 gap-3">
    <p className="text-xs text-(--ui-text-tertiary)">{messages.descriptions.preparationSeparate}</p>
    <p aria-live="polite" className="text-xs">{messages.labels.selectedCount(chosen.length)}</p>
    <div className="flex min-w-0 flex-wrap items-center gap-2">
      {onReview ? <Button className="motion-reduce:transition-none" disabled={disabled || reviewDisabled || pending || chosen.length === 0} onClick={() => onReview(chosen)} type="button" variant="outline">{messages.actions.reviewSelected}</Button> : null}
      <label className="flex items-center gap-2 text-sm"><input checked={confirmed} onChange={event => setConfirmed(event.target.checked)} type="checkbox" />{messages.a11y.confirmPreparation}</label>
      <Button className="motion-reduce:transition-none" disabled={disabled || !confirmed || chosen.length === 0 || pending} onClick={() => { void confirmPreparation() }} type="button">{pending ? messages.actions.preparing : messages.actions.prepare}</Button>
    </div>
    <SearchField aria-label={messages.labels.searchTargets} containerClassName="w-full" onChange={setQuery} placeholder={messages.labels.searchTargets} value={query} />
    <p className="text-xs text-(--ui-text-tertiary)">{messages.labels.showingTargets(visibleTargets.length, targets.length)}</p>
    <FleetOverview emptyMessage={query ? messages.descriptions.noMatchingTargets : undefined} key={reviewGeneration} onToggle={toggle} selected={selected} targets={visibleTargets} />
  </section>
}
