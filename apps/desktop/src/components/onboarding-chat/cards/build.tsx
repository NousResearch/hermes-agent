/**
 * The three build cards: choosing what to make, handing it to a session of its own, and reporting progress. Unlike the
 * setup cards these read the directive attrs, which the model writes, so each card validates the payload before it
 * renders.
 */

import { useAuiState } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { useEffect, useMemo } from 'react'

import { requestComposerSubmit } from '@/app/chat/composer/focus'
import { useSessionView } from '@/app/chat/session-view'
import { quarantineHandoffReceipt } from '@/app/contrib/handoff-receipt'
import { resolveSessionOwner } from '@/app/session/hooks/use-session-actions/utils'
import type { CardProps } from '@/components/onboarding-chat/cards/frame'
import { Chip } from '@/components/onboarding-chat/chip'
import {
  $handoffError,
  $setupHandoff,
  firstTaskTitle,
  guideHandoffReceiptKey,
  parseHandoffPlan,
  readGuideHandoffReceipt,
  requestSetupHandoff,
  retrySetupHandoff,
  SETUP_PROFILE
} from '@/components/onboarding-chat/setup-profile'
import { Button } from '@/components/ui/button'
import { answeredAfter } from '@/lib/chat-messages/parts'
import { segmentTranscriptDirectives } from '@/lib/transcript-directives'
import { cn } from '@/lib/utils'
import { $onboardingAnswers, markStepCommitted } from '@/store/onboarding-answers'
import { assertSessionOwnerResolved } from '@/store/session-owner-resolution'
import { isSessionOwnerRoute } from '@/store/session-request-router'

/** A tapped option is submitted as the user's own visible message rather than as a hidden [setup] note, so the
 *  model's next message answers a real turn. */
const FALLBACK_OPTION = "Let's figure it out together"

/**
 * The last question card before the handoff. The model asks what the user wants to build first, then places this card
 * with options it wrote from the conversation so far:
 * `::onboarding{step="first" options="A Discord bot|A habit tracker|…"}`.
 */
export function FirstBuildCard({ attrs, locked }: CardProps) {
  const view = useSessionView()
  const storedId = useStore(view.$storedId)
  const target = view.kind === 'tile' ? `tile:${storedId}` : 'main'
  // The pick lives with the other answers, not in component state: the
  // visible submit rebuilds the transcript and a local flag came back null,
  // leaving every chip clickable after one had already been sent. A typed
  // reply in the composer closes the card the same way a chip does.
  const messageId = useAuiState(state => state.message.id)

  const answeredInComposer = answeredAfter(useStore(view.$messages), messageId)

  const committed = useStore($onboardingAnswers).committed.find(step => step.startsWith('first:'))?.slice(6) ?? null
  const picked = committed ?? (answeredInComposer ? '' : null)

  // The 60-character limit keeps an option on one chip. The dedupe is case-insensitive because models repeat
  // themselves. Fewer than 2 usable options falls back to FALLBACK_OPTION, because the model's prose has already
  // told the user to pick one below.
  const seen = new Set<string>()

  const parsed = (attrs.options ?? '')
    .split('|')
    .map(option => option.trim().replace(/\s+/g, ' '))
    .filter(option => {
      const key = option.toLowerCase()

      if (option.length === 0 || option.length > 60 || seen.has(key)) {
        return false
      }

      seen.add(key)

      return true
    })
    .slice(0, 4)

  const options = parsed.length < 2 ? [FALLBACK_OPTION] : parsed

  const pick = (option: string) => {
    if (picked !== null || locked) {
      return
    }

    if (requestComposerSubmit(option, { target })) {
      markStepCommitted(`first:${option}`)
    }
  }

  return (
    <div className="my-3 grid min-w-0 max-w-md gap-4" data-onboarding-card inert={locked || undefined}>
      <div className="flex min-w-0 max-w-full flex-wrap gap-2">
        {options.map(option => (
          <Chip key={option} label={option} on={picked === option} onToggle={() => pick(option)} variant="pill" />
        ))}
      </div>
    </div>
  )
}

/**
 * Moves the first build out of this chat. Setup emits
 * `::onboarding{step="handoff" task="…" brief="…"}` once the task is decided. This card sets the request atom, and
 * the wiring effect then creates a session on the user's default profile, seeds it, and moves the user there.
 *
 * The `first` step already settled what to build, so this card asks nothing and only reports the state of the handoff.
 * The request atom and the accepted receipt stop a re-parse, a re-mount, or a relaunch from starting a second handoff,
 * and a locked (replayed) transcript never starts one.
 */
export function HandoffCard({ attrs, locked }: CardProps) {
  const view = useSessionView()
  const storedId = useStore(view.$storedId)
  const runtimeId = useStore(view.$runtimeId)
  const task = (attrs.task ?? '').trim().slice(0, 60)
  const brief = (attrs.brief ?? '').trim().slice(0, 240)
  const plan = parseHandoffPlan(attrs.plan)
  const state = useStore($setupHandoff)

  const receipt = useMemo(() => {
    try {
      return {
        completed: !!storedId && readGuideHandoffReceipt(storedId).receipt?.status === 'accepted',
        error: null
      }
    } catch (error) {
      return { completed: false, error: String(error) }
    }
  }, [storedId, state?.phase])

  const error = useStore($handoffError) ?? receipt.error
  const completed = receipt.completed

  useEffect(() => {
    if (!task || !brief || locked || !storedId || !runtimeId || $setupHandoff.get() || completed) {
      return
    }

    let cancelled = false
    void resolveSessionOwner(storedId)
      .then(owner => {
        assertSessionOwnerResolved(owner, { method: 'onboarding.handoff', sessionId: storedId })

        if (!cancelled) {
          requestSetupHandoff(task, brief, plan, {
            storedId,
            runtimeId,
            connectionId: isSessionOwnerRoute(owner) ? owner.connectionId : null,
            profile: isSessionOwnerRoute(owner) ? owner.profile : owner || SETUP_PROFILE
          })
        }
      })
      .catch(error => {
        if (!cancelled) {
          $handoffError.set(String(error))
          $setupHandoff.set({ task, brief, plan, phase: 'error' })
        }
      })

    return () => {
      cancelled = true
    }
  }, [brief, locked, plan, task, storedId, runtimeId, completed])

  if (!task || !brief) {
    return null
  }

  const settled = state?.phase === 'done' || (state === null && completed)
  const failed = state?.phase === 'error' || error !== null
  const title = state?.sessionTitle ?? firstTaskTitle(task)

  const retry = async () => {
    if (state?.phase !== 'error') {
      return
    }

    try {
      if (receipt.error && storedId) {
        quarantineHandoffReceipt(guideHandoffReceiptKey(storedId))
      }

      if (!state.guide && storedId && runtimeId) {
        const owner = await resolveSessionOwner(storedId)
        assertSessionOwnerResolved(owner, { method: 'onboarding.handoff', sessionId: storedId })
        $setupHandoff.set({
          ...state,
          guide: {
            storedId,
            runtimeId,
            connectionId: isSessionOwnerRoute(owner) ? owner.connectionId : null,
            profile: isSessionOwnerRoute(owner) ? owner.profile : owner || SETUP_PROFILE
          }
        })
      }

      retrySetupHandoff()
    } catch (error) {
      $handoffError.set(String(error))
    }
  }

  return (
    <div className="my-3 flex max-w-md items-center gap-2 text-sm" data-onboarding-card>
      <StatusDot live={!settled && !failed} />
      <span className="text-(--ui-text-secondary)">
        {failed
          ? (error ?? 'The first build could not be started. Retry to check its session.')
          : settled
            ? `${title} was started — find it in your sessions`
            : `Opening ${title}\u2026`}
      </span>
      {state?.phase === 'error' && (
        <Button disabled={locked} onClick={() => void retry()} size="sm" variant="text">
          Retry first build
        </Button>
      )}
    </div>
  )
}

/** The earlier steps are derived from this transcript on every render, so a re-mount cannot lose or repeat them. */
export function ProgressCard({ attrs, locked }: CardProps) {
  const view = useSessionView()
  const messages = useStore(view.$messages)
  const messageId = useAuiState(state => state.message.id)
  const title = (attrs.title ?? '').trim() || 'Working on it'

  const index = messages.findIndex(message => message.id === messageId)
  const previous = index < 0 ? [] : messages.slice(0, index)

  const steps = previous.flatMap(message => {
    const directives = message.parts.flatMap(part =>
      part.type === 'text' ? (segmentTranscriptDirectives(part.text) ?? []) : []
    )

    const progress = directives
      .filter(
        segment =>
          segment.kind === 'directive' &&
          segment.directive.name === 'onboarding' &&
          segment.directive.attrs.step === 'progress'
      )
      .at(-1)

    return progress?.kind === 'directive'
      ? [{ id: message.id, title: progress.directive.attrs.title?.trim() || 'Working on it' }]
      : []
  })

  return (
    <div className="my-3 grid max-w-md gap-1.5" data-onboarding-card>
      {[...steps, { id: messageId, title }].map(step => {
        const current = step.id === messageId

        return (
          <div className="flex items-center gap-2 text-sm" key={step.id}>
            <StatusDot live={current && locked} muted={!current} />
            <span className={current ? 'text-(--ui-text-secondary)' : 'text-(--ui-text-quaternary)'}>
              {current && locked ? `${step.title}…` : step.title}
            </span>
          </div>
        )
      })}
    </div>
  )
}

function StatusDot({ live, muted = !live }: { live: boolean; muted?: boolean }) {
  return (
    <span
      aria-hidden
      className={cn(
        'inline-block size-1.5 shrink-0 rounded-full',
        muted ? 'bg-(--ui-text-quaternary)' : 'bg-(--ui-accent)',
        live && 'animate-pulse'
      )}
    />
  )
}
