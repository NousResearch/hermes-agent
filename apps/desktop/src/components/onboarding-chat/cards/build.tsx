/**
 * The build beat's three cards: choosing what to make, handing it to a session
 * of its own, and watching it happen. Unlike the setup picks these read the
 * directive's attrs — the payload is model-written, so each one validates
 * before it renders.
 */

import { useAuiState } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { requestComposerSubmit } from '@/app/chat/composer/focus'
import { useSessionView } from '@/app/chat/session-view'
import { $handoffError, retrySetupHandoff } from '@/app/contrib/handoff-receipt'
import { resolveSessionOwner } from '@/app/session/hooks/use-session-actions/utils'
import type { CardProps } from '@/components/onboarding-chat/cards/frame'
import { Chip } from '@/components/onboarding-chat/chip'
import {
  $setupHandoff,
  firstTaskTitle,
  hasCompletedSetupHandoff,
  parseHandoffPlan,
  requestSetupHandoff,
  SETUP_PROFILE
} from '@/components/onboarding-chat/setup-profile'
import { Button } from '@/components/ui/button'
import { segmentTranscriptDirectives } from '@/lib/transcript-directives'
import { cn } from '@/lib/utils'
import { assertSessionOwnerResolved } from '@/store/session-owner-resolution'
import { isSessionOwnerRoute } from '@/store/session-request-router'

/** A tappable option is the user's own reply, so it goes out VISIBLE — the
 *  model's next message answers a real turn, not a hidden [setup] note. */
const FALLBACK_OPTION = "Let's figure it out together"

/**
 * The "first build" card — the close of the get-to-know-you beat. The model
 * asks a thoughtful question about what the user wants to BUILD first, then
 * places this card with the options IT generated from the whole conversation:
 * `::onboarding{step="first" options="A Discord bot|A habit tracker|…"}`.
 */
export function FirstBuildCard({ attrs, locked }: CardProps) {
  const view = useSessionView()
  const storedId = useStore(view.$storedId)
  const target = view.kind === 'tile' ? `tile:${storedId}` : 'main'
  const [picked, setPicked] = useState<null | string>(null)

  // Parse + validate the model's options: up to 4, each short enough to sit on
  // a chip, deduped case-insensitively (models repeat themselves). Garbage in
  // (0-1 usable) must not strand the user — the prose says "pick one below",
  // so fall back to the one option we can always offer.
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
    if (picked || locked) {
      return
    }

    if (requestComposerSubmit(option, { target })) {
      setPicked(option)
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
 * The handoff card — where the first build leaves this chat. Setup emits
 * `::onboarding{step="handoff" task="…" brief="…"}` once the task is decided,
 * and the card performs it: raise the beacon, and the wiring effect opens a
 * session on the user's default profile, seeds it, and moves the user there.
 *
 * Nothing to ask — the build's shape was settled by the `first` step and there
 * is one surface now, so the card just narrates: opening → landed. Both
 * latches (atom + storage) make re-parses, re-mounts, and relaunches inert,
 * and a locked (replayed) transcript never re-fires.
 */
export function HandoffCard({ attrs, locked }: CardProps) {
  const view = useSessionView()
  const storedId = useStore(view.$storedId)
  const runtimeId = useStore(view.$runtimeId)
  const task = (attrs.task ?? '').trim().slice(0, 60)
  const brief = (attrs.brief ?? '').trim().slice(0, 240)
  const plan = parseHandoffPlan(attrs.plan)
  const state = useStore($setupHandoff)
  const error = useStore($handoffError)

  useEffect(() => {
    if (!task || !brief || locked || !storedId || !runtimeId || $setupHandoff.get() || hasCompletedSetupHandoff()) {
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
  }, [brief, locked, plan, task, storedId, runtimeId])

  if (!task || !brief) {
    return null
  }

  const settled = state?.phase === 'done' || (state === null && hasCompletedSetupHandoff())
  const failed = state?.phase === 'error'
  const title = state?.sessionTitle ?? firstTaskTitle(task)

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
      {failed && (
        <Button disabled={locked} onClick={retrySetupHandoff} size="sm" variant="text">
          Retry first build
        </Button>
      )}
    </div>
  )
}

/** Progress comes from this transcript, so virtualization cannot append history. */
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

  steps.push({ id: messageId, title })

  return (
    <div className="my-3 grid max-w-md gap-1.5" data-onboarding-card>
      {steps.map(step => {
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
