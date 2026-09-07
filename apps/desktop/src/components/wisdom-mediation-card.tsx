import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { WisdomReviewTables } from '@/components/wisdom-checks'
import {
  getWisdomMediation,
  type ProfileScope,
  resolveWisdomConsent,
  type WisdomConsentInteraction,
  type WisdomMediationActivity
} from '@/hermes'
import { useI18n } from '@/i18n'
import { ChevronLeft, ChevronRight, Loader2 } from '@/lib/icons'

export function WisdomMediationCard({
  profile,
  sessionId,
  passive = false
}: {
  profile?: ProfileScope
  sessionId?: string
  passive?: boolean
}) {
  const { t } = useI18n()
  const copy = t.skills.collective
  const [activity, setActivity] = useState<WisdomMediationActivity | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [expanded, setExpanded] = useState<string | null>(null)
  const [reviews, setReviews] = useState<Record<string, WisdomConsentInteraction['inspection']>>({})
  const [deferred, setDeferred] = useState<Set<string>>(() => new Set())
  const revision = useRef(0)
  const acting = useRef(false)

  useEffect(() => {
    let active = true
    setActivity(null)
    setDeferred(new Set())
    setBusy(null)
    setError(null)
    setReviews({})

    const refresh = async () => {
      if (acting.current) {
        return
      }
      const request = ++revision.current

      try {
        const next = await getWisdomMediation(profile)

        if (active && request === revision.current) {
          setActivity(next)
        }
      } catch {
        // Keep the last known valid advice through transient outages.
      }
    }

    void refresh()
    const timer = window.setInterval(() => void refresh(), 10_000)

    return () => {
      active = false
      revision.current++
      window.clearInterval(timer)
    }
  }, [profile, sessionId])

  const act = async (
    interaction: WisdomConsentInteraction,
    action: 'inspect' | `inspect.${number}` | 'defer' | 'confirm'
  ) => {
    if (!sessionId || acting.current) {
      return
    }
    acting.current = true
    const request = ++revision.current
    setBusy(interaction.id)
    setError(null)

    try {
      const result = await resolveWisdomConsent(interaction.id, sessionId, action, profile)

      if (request !== revision.current) {
        return
      }
      setActivity(current =>
        current
          ? {
              ...current,
              interactions: current.interactions.map(item => (item.id === result.id ? result : item))
            }
          : current
      )

      if (action.startsWith('inspect')) {
        setExpanded(result.id)
      }

      if (result.inspection) {
        setReviews(current => ({ ...current, [result.id]: result.inspection }))
      }

      if (action === 'defer') {
        setDeferred(current => new Set([...current, result.id]))
      }
    } catch {
      if (request === revision.current) {
        setError(copy.unavailable)
      }
    } finally {
      acting.current = false

      if (request === revision.current) {
        setBusy(null)
      }
    }
  }

  if (activity?.mode !== 'agent') {
    return null
  }
  const entries = activity.assessments.filter(item => item.advice && (passive || item.owner_session === sessionId))

  if (!entries.length) {
    return null
  }

  return (
    <section aria-label={copy.notifications} className="my-3 min-w-0 border-y border-(--ui-stroke-tertiary) py-3">
      <h2 className="text-sm font-semibold">{copy.title}</h2>
      {error && (
        <p className="mt-2 text-sm text-destructive" role="alert">
          {error}
        </p>
      )}
      {entries.map(entry => {
        const interaction = activity.interactions.find(item => item.assessment_id === entry.id)

        if (
          !passive &&
          interaction &&
          (deferred.has(interaction.id) || interaction.deferred_surfaces?.includes('local'))
        ) {
          return null
        }
        const own = !!sessionId && entry.owner_session === sessionId
        const pending = interaction?.state === 'pending' && interaction.expires_at * 1000 > Date.now()
        const review = interaction ? reviews[interaction.id] : undefined

        return (
          <article className="min-w-0 border-t border-(--ui-stroke-tertiary) py-3 first:border-0" key={entry.id}>
            <h3 className="break-words text-sm font-medium">{entry.advice?.title}</h3>
            <p className="mt-1 whitespace-pre-wrap break-words text-sm text-(--ui-text-secondary)">
              {entry.advice?.explanation}
            </p>
            {interaction && (
              <>
                <p className="mt-2 text-xs font-medium">
                  {interaction.facts.editorial_name || interaction.facts.slug}
                  {interaction.facts.version ? ` · v${interaction.facts.version}` : ''}
                </p>
                {interaction.facts.compatibility && (
                  <p className="mt-1 text-xs">{interaction.facts.compatibility.outcome.replaceAll('_', ' ')}</p>
                )}
                {interaction.facts.modified && <p className="text-xs text-destructive">{copy.unsavedChanges}</p>}
                {interaction.facts.sensitive_expansion?.map((warning, i) => (
                  <p className="text-xs text-destructive" key={i}>
                    {warning}
                  </p>
                ))}
                {interaction.operation === 'share' && (
                  <p className="mt-2 text-xs text-(--ui-text-secondary)">{copy.sharePreparationNotice}</p>
                )}
                <details
                  className="mt-2"
                  onToggle={event => {
                    const open = event.currentTarget.open
                    setExpanded(current => (open ? interaction.id : current === interaction.id ? null : current))
                  }}
                  open={expanded === interaction.id}
                >
                  <summary className="cursor-pointer text-xs">{copy.reviewExact}</summary>
                  <WisdomReviewTables
                    professionalism={interaction.facts.professionalism_check}
                    security={interaction.facts.security_check}
                  />
                  {interaction.facts.file_names?.map(name => (
                    <p className="break-words font-mono text-xs" key={name}>
                      {name}
                    </p>
                  ))}
                  {review && (
                    <div className="mt-3 min-w-0">
                      <p className="whitespace-pre-wrap break-words text-xs">{review.description}</p>
                      <p className="mt-2 break-words font-mono text-xs">{review.path}</p>
                      <pre className="my-2 max-h-80 overflow-auto whitespace-pre-wrap break-words border border-(--ui-stroke-tertiary) p-2 text-xs">
                        {review.content}
                      </pre>
                      <div className="flex items-center gap-2">
                        <Button
                          aria-label={t.skills.collective.reviewPreviousPage}
                          disabled={busy !== null || review.page === 0}
                          onClick={() => void act(interaction, `inspect.${review.page - 1}`)}
                          size="icon"
                          title={t.skills.collective.reviewPreviousPage}
                          variant="outline"
                        >
                          <ChevronLeft aria-hidden />
                        </Button>
                        <span className="text-xs tabular-nums">
                          {review.page + 1}/{review.page_count}
                        </span>
                        <Button
                          aria-label={t.skills.collective.reviewNextPage}
                          disabled={busy !== null || review.page + 1 >= review.page_count}
                          onClick={() => void act(interaction, `inspect.${review.page + 1}`)}
                          size="icon"
                          title={t.skills.collective.reviewNextPage}
                          variant="outline"
                        >
                          <ChevronRight aria-hidden />
                        </Button>
                      </div>
                    </div>
                  )}
                </details>
                {own && pending ? (
                  <div className="mt-3 grid grid-cols-3 items-start gap-2 [&>button]:h-auto [&>button]:min-h-8 [&>button]:min-w-0 [&>button]:whitespace-normal [&>button]:break-words">
                    <Button
                      disabled={busy !== null}
                      onClick={() => void act(interaction, 'defer')}
                      size="sm"
                      variant="outline"
                    >
                      {copy.notNow}
                    </Button>
                    <Button
                      disabled={busy !== null}
                      onClick={() => void act(interaction, 'inspect')}
                      size="sm"
                      variant="outline"
                    >
                      {copy.reviewFirst}
                    </Button>
                    {interaction.actions.includes('confirm') && (
                      <Button disabled={busy !== null} onClick={() => void act(interaction, 'confirm')} size="sm">
                        {busy === interaction.id && <Loader2 aria-hidden className="size-3 animate-spin" />}
                        {interaction.operation === 'share'
                          ? copy.share
                          : interaction.operation === 'publish'
                            ? copy.approve
                            : interaction.operation === 'install'
                              ? copy.install
                              : t.common.update}
                      </Button>
                    )}
                  </div>
                ) : (
                  <p className="mt-2 text-xs text-muted-foreground">
                    {interaction.result?.packaging_state === 'queued'
                      ? copy.preparingLocal
                      : interaction.result?.packaging_state === 'failed'
                        ? copy.unavailable
                        : copy.draftState(interaction.state)}
                  </p>
                )}
              </>
            )}
          </article>
        )
      })}
    </section>
  )
}
