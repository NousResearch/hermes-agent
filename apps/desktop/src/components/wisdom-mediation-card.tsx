import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { WisdomConsentDetails } from '@/components/wisdom-consent-details'
import { WisdomPublicationReview } from '@/components/wisdom-publication-review'
import { legacySetupReview, WisdomSetupReview } from '@/components/wisdom-setup-review'
import {
  getWisdomMediation,
  prepareWisdomConsentPublication,
  type ProfileScope,
  resolveWisdomConsent,
  type WisdomConsentAction,
  type WisdomConsentInteraction,
  type WisdomMediationActivity
} from '@/hermes'
import { useI18n } from '@/i18n'
import { Loader2 } from '@/lib/icons'

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
  const [publication, setPublication] = useState<{ draft_id: string; interaction_id: string } | null>(null)
  const [reviews, setReviews] = useState<Record<string, WisdomConsentInteraction['inspection']>>({})
  const [deferred, setDeferred] = useState<Set<string>>(() => new Set())
  const [opened, setOpened] = useState<{ entryId: string; interaction: WisdomConsentInteraction } | null>(null)
  const revision = useRef(0)
  const acting = useRef(false)

  useEffect(() => {
    let active = true
    setActivity(null)
    setDeferred(new Set())
    setBusy(null)
    setError(null)
    setReviews({})
    setPublication(null)
    setOpened(null)

    const refresh = async () => {
      if (acting.current) {
        return
      }

      const request = ++revision.current

      try {
        const next = await getWisdomMediation(profile)

        if (active && request === revision.current) {
          setActivity(next)
          setOpened(current => {
            const fresh = next.interactions.find(item => item.id === current?.interaction.id)

            if (!current || !fresh) {
              return null
            }

            // Keep explicit status/recovery review open, but never retain an old approval state.
            return fresh.state !== current.interaction.state || fresh.expires_at !== current.interaction.expires_at
              ? { ...current, interaction: fresh }
              : current
          })
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

  const act = async (interaction: WisdomConsentInteraction, action: WisdomConsentAction) => {
    if (!sessionId || acting.current) {
      return
    }

    acting.current = true
    const request = ++revision.current
    setBusy(interaction.id)
    setError(null)

    try {
      if (['share', 'publish'].includes(interaction.operation) && (action === 'inspect' || action === 'confirm')) {
        const prepared = await prepareWisdomConsentPublication(interaction.id, sessionId, profile)

        if (request === revision.current) {
          setPublication({ ...prepared, interaction_id: interaction.id })
        }

        return
      }

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

      if (result.setup_review) {
        setOpened(current => ({
          entryId: current?.interaction.id === interaction.id ? current.entryId : interaction.assessment_id,
          interaction: result
        }))
      }

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

  if (!activity) {
    return null
  }

  const entries = activity.assessments.filter(item => item.advice && (passive || item.owner_session === sessionId))

  if (!entries.length) {
    return null
  }

  return (
    <section aria-label={copy.notifications} className="my-3 min-w-0 border-y border-(--ui-stroke-tertiary) py-3">
      <h2 className="text-sm font-semibold">{copy.title}</h2>
      {publication && sessionId && (
        <WisdomPublicationReview
          consent={{ interaction_id: publication.interaction_id, session_id: sessionId }}
          draftId={publication.draft_id}
          key={publication.draft_id}
          onClose={() => setPublication(null)}
          profile={profile}
        />
      )}
      {error && (
        <p className="mt-2 text-sm text-destructive" role="alert">
          {error}
        </p>
      )}
      {entries.map(entry => {
        const interaction =
          opened?.entryId === entry.id
            ? opened.interaction
            : activity.interactions.find(item => item.assessment_id === entry.id)

        if (opened && opened.entryId !== entry.id && interaction?.id === opened.interaction.id) {
          return null
        }

        if (
          !passive &&
          interaction &&
          (deferred.has(interaction.id) || interaction.deferred_surfaces?.includes('local'))
        ) {
          return null
        }

        const own = !passive && !!sessionId && entry.owner_session === sessionId
        const pending = interaction?.state === 'pending' && interaction.expires_at * 1000 > Date.now()
        const review = interaction ? reviews[interaction.id] : undefined
        const setupReview = interaction ? (interaction.setup_review ?? legacySetupReview(interaction, copy)) : undefined

        if (publication?.interaction_id === interaction?.id) {
          return null
        }

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
                <WisdomConsentDetails
                  active={own}
                  busy={busy !== null}
                  expanded={expanded === interaction.id}
                  interaction={interaction}
                  onAction={action => void act(interaction, action)}
                  onToggle={open => {
                    setExpanded(current => (open ? interaction.id : current === interaction.id ? null : current))
                  }}
                  review={review}
                />
                {setupReview ? (
                  <WisdomSetupReview
                    active={own}
                    busy={busy !== null}
                    onAction={action => void act(interaction, action)}
                    review={setupReview}
                  />
                ) : own && pending && interaction.operation !== 'setup' ? (
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
