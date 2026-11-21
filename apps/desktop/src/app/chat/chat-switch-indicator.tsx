import { useStore } from '@nanostores/react'
import { type CSSProperties, useEffect, useState } from 'react'

import { useOnboardingChatActive } from '@/components/onboarding/first-run/assembly'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ErrorIcon } from '@/components/ui/error-state'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { useI18n } from '@/i18n'
import { profileColorSoft, resolveProfileColor } from '@/lib/profile-color'
import { $profileColors } from '@/store/profile'
import { dismissProfileSwitchFailure, type ProfileSwitchFailure } from '@/store/profile-switch-state'

// Elapsed-time rungs of a profile switch. Under SLOW the switch is fast enough
// that any chrome would only flash; between SLOW and STALLED a quiet corner
// pill says what is happening; past STALLED an inline strip at the top of the
// thread explains that sends will wait.
export const SWITCH_SLOW_MS = 2_000
export const SWITCH_STALLED_MS = 10_000

/** Readiness contract (also read by scripts/perf/scenarios/profile-switch.mjs):
 *  `pending`/`slow`/`stalled` mean a switch is still in flight. */
export type SwitchIndicatorState = 'failed' | 'idle' | 'pending' | 'slow' | 'stalled' | 'syncing'

type SwitchStage = 'pending' | 'slow' | 'stalled'

// Remember idle as a target transition too: retrying the same failed profile
// must not inherit its previous attempt's stalled stage.
function useSwitchStage(target: null | string): null | SwitchStage {
  const [measured, setMeasured] = useState<{ stage: SwitchStage; target: null | string }>({
    stage: 'pending',
    target
  })

  if (measured.target !== target) {
    setMeasured({ stage: 'pending', target })
  }

  useEffect(() => {
    if (!target) {
      return
    }

    const slow = window.setTimeout(() => setMeasured({ stage: 'slow', target }), SWITCH_SLOW_MS)
    const stalled = window.setTimeout(() => setMeasured({ stage: 'stalled', target }), SWITCH_STALLED_MS)

    return () => {
      window.clearTimeout(slow)
      window.clearTimeout(stalled)
    }
  }, [target])

  if (!target) {
    return null
  }

  return measured.target === target ? measured.stage : 'pending'
}

// Tint over an opaque surface: the pill floats over transcript text, so the
// translucent profile mix needs the chat surface underneath it.
function tintStyle(color: null | string): CSSProperties | undefined {
  if (!color) {
    return undefined
  }

  const soft = profileColorSoft(color)

  return { backgroundImage: `linear-gradient(${soft}, ${soft})` }
}

interface ChatSwitchIndicatorProps {
  /** A rail switch that ended in failure; terminal until retried or dismissed. */
  failure: null | ProfileSwitchFailure
  /** Paint-first wake (#89843): transcript is real, the profile gate is still settling. */
  syncing: null | string
  /** Profile a switch is in flight to. */
  target: null | string
}

// One status surface for every phase of a profile switch. It never covers the
// conversation: the pill is a pointer-transparent corner chip and the strips
// take their own row above the thread. Mounted in flow at the top of the chat
// column so the strips push the thread down rather than painting over it.
export function ChatSwitchIndicator({ failure, syncing, target }: ChatSwitchIndicatorProps) {
  const { t } = useI18n()
  const colors = useStore($profileColors)
  // The first run may swap into the default profile, which the user did not ask for.
  // Switch chrome over the greeting reads as a stall in the moment that has to feel instant.
  const onboarding = useOnboardingChatActive()
  const switching = onboarding ? null : target
  const stage = useSwitchStage(switching)
  const state: SwitchIndicatorState = failure ? 'failed' : (stage ?? (syncing ? 'syncing' : 'idle'))
  const profile = failure?.profile ?? switching ?? syncing
  const color = resolveProfileColor(profile, colors)

  return (
    <div
      aria-live="polite"
      className="relative z-10 shrink-0"
      data-slot="chat-switch-indicator"
      data-switch-state={state}
    >
      {(state === 'slow' || state === 'syncing') && profile && (
        <div
          className="pointer-events-none absolute right-3 top-2 flex items-center gap-1.5 rounded-full border border-(--stroke-nous) bg-(--ui-chat-surface-background) px-2 py-0.5 font-mono text-[0.6875rem] text-(--ui-text-secondary) shadow-nous transition-opacity duration-150 ease-out starting:opacity-0 motion-reduce:transition-none"
          style={tintStyle(color)}
        >
          <GlyphSpinner className="w-3 justify-start text-(--ui-accent)" spinner="braille" />
          {state === 'slow' ? t.composer.switchingProfile(profile) : t.desktop.hydrationSyncing(profile)}
        </div>
      )}

      {state === 'stalled' && profile && (
        <div
          className="flex items-center gap-2 border-b border-(--ui-stroke-tertiary) bg-(--ui-chat-surface-background) px-4 py-1.5 text-xs text-(--ui-text-secondary)"
          style={tintStyle(color)}
        >
          <GlyphSpinner className="w-3 justify-start text-(--ui-accent)" spinner="braille" />
          <span className="font-medium text-(--ui-text-primary)">{t.composer.switchingProfile(profile)}</span>
          <span className="min-w-0 truncate">{t.composer.switchingProfileSlow}</span>
        </div>
      )}

      {failure && (
        <div className="flex items-center gap-2 border-b border-(--ui-stroke-tertiary) bg-(--ui-chat-surface-background) px-4 py-1 text-xs">
          <ErrorIcon size="0.875rem" />
          <span className="min-w-0 truncate font-medium text-(--ui-text-primary)">
            {t.composer.switchProfileFailed(failure.profile)}
          </span>
          <Button onClick={failure.retry} size="xs" variant="textStrong">
            {t.common.retry}
          </Button>
          <Button
            aria-label={t.common.close}
            className="ml-auto"
            onClick={dismissProfileSwitchFailure}
            size="icon-xs"
            variant="ghost"
          >
            <Codicon name="close" />
          </Button>
        </div>
      )}
    </div>
  )
}
