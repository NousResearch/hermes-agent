import { useEffect, useRef, useState } from 'react'
import { Loader2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { WisdomReviewTables } from '@/components/wisdom-checks'
import {
  getWisdomMediation, resolveWisdomConsent, type ProfileScope,
  type WisdomConsentInteraction, type WisdomMediationActivity
} from '@/hermes'
import { useI18n } from '@/i18n'

export function WisdomMediationCard({ profile, sessionId, passive = false }: {
  profile?: ProfileScope; sessionId?: string; passive?: boolean
}) {
  const { t } = useI18n()
  const copy = t.skills.collective
  const [activity, setActivity] = useState<WisdomMediationActivity | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [expanded, setExpanded] = useState<string | null>(null)
  const [deferred, setDeferred] = useState<Set<string>>(() => new Set())
  const revision = useRef(0)
  const acting = useRef(false)

  useEffect(() => {
    let active = true
    setActivity(null)
    setDeferred(new Set())
    setBusy(null)
    setError(null)
    const refresh = async () => {
      if (acting.current) return
      const request = ++revision.current
      try {
        const next = await getWisdomMediation(profile)
        if (active && request === revision.current) setActivity(next)
      } catch {
        // Keep the last known valid advice through transient outages.
      }
    }
    void refresh()
    const timer = window.setInterval(() => void refresh(), 10_000)
    return () => { active = false; revision.current++; window.clearInterval(timer) }
  }, [profile, sessionId])

  const act = async (interaction: WisdomConsentInteraction, action: 'inspect' | 'defer' | 'confirm') => {
    if (!sessionId || acting.current) return
    acting.current = true
    const request = ++revision.current
    setBusy(interaction.id)
    setError(null)
    try {
      const result = await resolveWisdomConsent(interaction.id, sessionId, action, profile)
      if (request !== revision.current) return
      setActivity(current => current ? {
        ...current, interactions: current.interactions.map(item => item.id === result.id ? result : item)
      } : current)
      if (action === 'inspect') setExpanded(result.id)
      if (action === 'defer') setDeferred(current => new Set([...current, result.id]))
    } catch {
      if (request === revision.current) setError(copy.unavailable)
    } finally {
      acting.current = false
      if (request === revision.current) setBusy(null)
    }
  }

  if (activity?.mode !== 'agent') return null
  const entries = activity.assessments.filter(item => item.advice && (passive || item.owner_session === sessionId))
  if (!entries.length) return null

  return <section aria-label={copy.notifications} className="my-3 min-w-0 border-y border-(--ui-stroke-tertiary) py-3">
    <h2 className="text-sm font-semibold">{copy.title}</h2>
    {error && <p role="alert" className="mt-2 text-sm text-destructive">{error}</p>}
    {entries.map(entry => {
      const interaction = activity.interactions.find(item => item.assessment_id === entry.id)
      if (!passive && interaction && (deferred.has(interaction.id) || interaction.deferred_surfaces?.includes('local'))) return null
      const own = !!sessionId && entry.owner_session === sessionId
      const pending = interaction?.state === 'pending' && interaction.expires_at * 1000 > Date.now()
      return <article key={entry.id} className="min-w-0 border-t border-(--ui-stroke-tertiary) py-3 first:border-0">
        <h3 className="break-words text-sm font-medium">{entry.advice?.title}</h3>
        <p className="mt-1 whitespace-pre-wrap break-words text-sm text-(--ui-text-secondary)">{entry.advice?.explanation}</p>
        {interaction && <>
          <p className="mt-2 text-xs font-medium">
            {interaction.facts.editorial_name || interaction.facts.slug}
            {interaction.facts.version ? ` · v${interaction.facts.version}` : ''}
          </p>
          {interaction.facts.compatibility && <p className="mt-1 text-xs">{interaction.facts.compatibility.outcome.replaceAll('_', ' ')}</p>}
          {interaction.facts.modified && <p className="text-xs text-destructive">{copy.unsavedChanges}</p>}
          {interaction.facts.sensitive_expansion?.map((warning, i) => <p className="text-xs text-destructive" key={i}>{warning}</p>)}
          <details open={expanded === interaction.id} onToggle={event => {
            setExpanded(current => event.currentTarget.open ? interaction.id : current === interaction.id ? null : current)
          }} className="mt-2">
            <summary className="cursor-pointer text-xs">{copy.reviewExact}</summary>
            <WisdomReviewTables security={interaction.facts.security_check} professionalism={interaction.facts.professionalism_check} />
          </details>
          {own && pending ? <div className="mt-3 grid grid-cols-3 items-start gap-2 [&>button]:h-auto [&>button]:min-h-8 [&>button]:min-w-0 [&>button]:whitespace-normal [&>button]:break-words">
            <Button disabled={busy !== null} size="sm" variant="outline" onClick={() => void act(interaction, 'defer')}>{copy.notNow}</Button>
            <Button disabled={busy !== null} size="sm" variant="outline" onClick={() => void act(interaction, 'inspect')}>{copy.reviewFirst}</Button>
            {interaction.actions.includes('confirm') && <Button disabled={busy !== null} size="sm" onClick={() => void act(interaction, 'confirm')}>
              {busy === interaction.id && <Loader2 aria-hidden className="size-3 animate-spin" />}
              {interaction.operation === 'publish' ? copy.yes : interaction.operation === 'install' ? copy.install : t.common.update}
            </Button>}
          </div> : <p className="mt-2 text-xs text-muted-foreground">{copy.draftState(interaction.state)}</p>}
        </>}
      </article>
    })}
  </section>
}
