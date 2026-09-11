import { useEffect, useLayoutEffect, useRef, useState } from 'react'

import { TooltipIconButton } from '@/components/assistant-ui/tooltip-icon-button'
import { Button } from '@/components/ui/button'
import { WisdomPublicationReview } from '@/components/wisdom-publication-review'
import {
  deferWisdomCandidate,
  getWisdomEvents,
  prepareWisdomCandidate,
  type ProfileScope,
  type WisdomCandidateEvent
} from '@/hermes'
import { useI18n } from '@/i18n'
import { Volume2, VolumeX } from '@/lib/icons'
import { notifyError } from '@/store/notifications'

const AUTO_REVEAL_PREVIOUS_BOTTOM_GAP_PX = 48

export function WisdomCandidateCard({ profile, sessionId }: { profile?: ProfileScope; sessionId: string }) {
  const { t } = useI18n()
  const copy = t.skills.collective
  const [event, setEvent] = useState<null | WisdomCandidateEvent>(null)
  const [draftId, setDraftId] = useState<string | null>(null)
  const [notificationsMuted, setNotificationsMuted] = useState(false)

  const [busy, setBusy] = useState<null | 'defer' | 'prepare'>(null)

  const [preparationError, setPreparationError] = useState<null | string>(null)
  const cardRef = useRef<HTMLElement>(null)
  const resolvedEventIdsRef = useRef(new Set<string>())
  const eventId = event?.id

  useEffect(() => {
    let active = true
    let refreshSequence = 0
    resolvedEventIdsRef.current.clear()

    const refresh = async () => {
      const sequence = ++refreshSequence

      try {
        const result = await getWisdomEvents(sessionId, profile)

        if (active && sequence === refreshSequence) {
          const pending = result.events.filter(item => !resolvedEventIdsRef.current.has(item.id))
          setEvent(current =>
            current && resolvedEventIdsRef.current.has(current.id)
              ? current
              : (pending.find(item => item.id === current?.id) ?? pending[0] ?? null)
          )
        }
      } catch {
        // Candidate promotion is optional transcript UI. Wisdom availability
        // must never make ordinary chat unusable. Preserve the last confirmed
        // event so a transient refresh cannot make the card flicker away.
      }
    }

    void refresh()
    const timer = window.setInterval(() => void refresh(), 10_000)

    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [profile, sessionId])

  useEffect(() => {
    setDraftId(null)
    setPreparationError(null)
    setNotificationsMuted(false)
  }, [eventId])

  useLayoutEffect(() => {
    const card = cardRef.current

    if (!eventId || !card) {
      return
    }

    const viewport = card.closest<HTMLElement>('[data-slot="aui_thread-viewport"]')

    if (!viewport || typeof card.scrollIntoView !== 'function') {
      return
    }

    const cardHeight = card.getBoundingClientRect().height || card.offsetHeight
    const previousBottomGap = viewport.scrollHeight - cardHeight - viewport.scrollTop - viewport.clientHeight

    if (previousBottomGap <= AUTO_REVEAL_PREVIOUS_BOTTOM_GAP_PX) {
      card.scrollIntoView({ block: 'nearest' })
    }
  }, [eventId])

  if (!event) {
    return null
  }

  const skill = event.payload.skill_name
  const displayName = event.payload.editorial_name?.trim() || skill
  const displayDescription = event.payload.editorial_description?.trim()

  const qualificationNotice =
    event.notice_variant === 'first' ? copy.qualificationFirst(event.organization_name) : copy.qualificationReturning

  const prepareForReview = async () => {
    setBusy('prepare')
    setPreparationError(null)

    try {
      const result = await prepareWisdomCandidate(event.id, profile)

      setDraftId(result.stage === 'review' ? result.review.draft.id : result.prepared.local_draft_id)
    } catch (error) {
      setPreparationError(error instanceof Error ? error.message : String(error))
    } finally {
      setBusy(null)
    }
  }

  const notNow = async () => {
    setBusy('defer')

    try {
      await deferWisdomCandidate(event.id, profile)
      resolvedEventIdsRef.current.add(event.id)
      setDraftId(null)
      setEvent(null)
    } catch (error) {
      notifyError(error, 'Collective Wisdom notification could not be deferred')
    } finally {
      setBusy(null)
    }
  }

  return (
    <section
      aria-label={copy.proposalTitle}
      className="mb-(--conversation-turn-gap) border border-emerald-600/40 bg-(--ui-chat-surface-background)"
      ref={cardRef}
    >
      <header className="flex items-center justify-between border-b border-(--ui-stroke-tertiary) px-4 py-3">
        <div>
          <div className="text-xs font-medium">{copy.proposalTitle}</div>
          <div className="text-[0.68rem] font-medium text-muted-foreground">
            {copy.skillName}: {displayName}
          </div>
        </div>
        <span className="text-[0.62rem] text-muted-foreground">{copy.localSuggestion}</span>
      </header>

      <div className="border-b border-(--ui-stroke-tertiary) px-4 py-3">
        <p className="text-xs leading-5">{qualificationNotice}</p>
        {displayDescription && (
          <p className="mt-1 text-xs text-muted-foreground">
            {copy.whatItDoes}: {displayDescription}
          </p>
        )}
        <p className="mt-2 text-xs font-medium">{copy.sharePrompt}</p>
      </div>

      {!draftId && (
        <div className="p-4">
          {busy === 'prepare' && <p className="mt-3 text-xs">{copy.preparingLocal}</p>}
          {preparationError && (
            <div className="mt-3 text-xs text-destructive" role="alert">
              {preparationError}
            </div>
          )}
          <footer className="mt-3 grid grid-cols-3 items-center gap-2 border-t border-(--ui-stroke-tertiary) pt-3">
            <div className="flex items-center gap-1 justify-self-start">
              <TooltipIconButton
                aria-pressed={notificationsMuted}
                disabled={busy !== null}
                onClick={() => setNotificationsMuted(value => !value)}
                tooltip={notificationsMuted ? copy.unmuteNotificationsSoon : copy.muteNotificationsSoon}
              >
                {notificationsMuted ? <VolumeX className="size-4" /> : <Volume2 className="size-4" />}
              </TooltipIconButton>
              <Button disabled={busy !== null} onClick={() => void notNow()} size="sm" variant="outline">
                {copy.notNow}
              </Button>
            </div>
            <div className="justify-self-center">
              <Button disabled={busy !== null} onClick={() => void prepareForReview()} size="sm">
                {busy === 'prepare' ? copy.preparingLocal : copy.reviewFirst}
              </Button>
            </div>
            <div className="justify-self-end">
              <Button disabled={busy !== null} onClick={() => void prepareForReview()} size="sm">
                {copy.share}
              </Button>
            </div>
          </footer>
        </div>
      )}

      {draftId && (
        <WisdomPublicationReview
          draftId={draftId}
          key={draftId}
          onClose={() => {
            setDraftId(null)

            if (resolvedEventIdsRef.current.has(event.id)) {
              setEvent(null)
            }
          }}
          onSubmitted={() => resolvedEventIdsRef.current.add(event.id)}
          profile={profile}
        />
      )}
    </section>
  )
}
