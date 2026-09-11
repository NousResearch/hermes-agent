/**
 * The build beat's three cards: choosing what to make, handing it to a session
 * of its own, and watching it happen. Unlike the setup picks these read the
 * directive's attrs — the payload is model-written, so each one validates
 * before it renders.
 */

import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { requestComposerSubmit } from '@/app/chat/composer/focus'
import { $handoffError, retrySetupHandoff } from '@/app/contrib/handoff-receipt'
import type { CardProps } from '@/components/onboarding-chat/cards/frame'
import { Chip } from '@/components/onboarding-chat/chip'
import {
  $setupHandoff,
  firstTaskTitle,
  hasCompletedSetupHandoff,
  parseHandoffPlan,
  requestSetupHandoff
} from '@/components/onboarding-chat/setup-profile'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

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

    if (requestComposerSubmit(option)) {
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
  const task = (attrs.task ?? '').trim().slice(0, 60)
  const brief = (attrs.brief ?? '').trim().slice(0, 240)
  const plan = parseHandoffPlan(attrs.plan)
  const state = useStore($setupHandoff)
  const error = useStore($handoffError)

  useEffect(() => {
    if (task && brief && !locked) {
      requestSetupHandoff(task, brief, plan)
    }
  }, [brief, locked, plan, task])

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

/**
 * The progress card — the build's live status, inline in the transcript. The
 * model re-emits `::onboarding{step="progress" title="…"}` as it works; each
 * emission appends a step row to a session-wide list (module-scope, keyed by
 * nothing — the onboarding thread is the only consumer), the newest row
 * pulsing while the turn streams. Read-only: the user watches the build
 * breathe; permission prompts ride the session concurrently.
 *
 * No fake percentages — the model can't know N-of-M mid-build, so the card is
 * an honest growing step list, not a bar that lies.
 */
const progressSteps: string[] = []

export function ProgressCard({ attrs, locked }: CardProps) {
  const title = (attrs.title ?? '').trim() || 'Working on it'

  // Append on first sight of a new title (re-emits of the same step are the
  // model re-rendering mid-stream, not a new step).
  const [index] = useState(() => {
    if (progressSteps[progressSteps.length - 1] !== title) {
      progressSteps.push(title)
    }

    return progressSteps.length - 1
  })

  return (
    <div className="my-3 grid max-w-md gap-1.5" data-onboarding-card>
      {progressSteps.slice(0, index + 1).map((step, i) => {
        const current = i === index

        return (
          <div className="flex items-center gap-2 text-sm" key={`${i}-${step}`}>
            <StatusDot live={current && !locked} muted={!current} />
            <span className={current ? 'text-(--ui-text-secondary)' : 'text-(--ui-text-quaternary)'}>
              {current && !locked ? `${step}…` : step}
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
