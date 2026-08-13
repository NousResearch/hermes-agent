import { useState } from 'react'

import { composerFloatingPill } from '@/components/chat/composer-dock'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { triggerHaptic } from '@/lib/haptics'
import { brandFor, brandGlyphStyle } from '@/lib/mcp-brands'
import { useSessionSlice } from '@/lib/use-session-slice'
import { cn } from '@/lib/utils'
import { $composerSuggestionsBySession, markSuggestionInvoked, suggestionKey } from '@/store/composer-suggestions'

/**
 * The composer suggestion strip — generic pills fed by the suggestion bus
 * (`store/composer-suggestions.ts`; the MCP connect pills of PR #85036 are
 * provider one of N). Renders beside the micro-action badges in the floating
 * lane above the composer, same pill treatment. Session-scoped like the
 * badges: each composer shows only the suggestions its own session earned.
 *
 * Every pill is a one-click action with a narrated lifecycle: label →
 * workingLabel (click again to request cancel) → doneLabel. The provider's
 * `invoke` owns the work, cancellation, rollback, and error toasts; this
 * component owns only the phase presentation.
 *
 * No dismiss affordance ON PURPOSE. Suggestions are self-limiting — a
 * provider withdraws its offer when the trigger condition stops holding —
 * so a close button would mostly collect accidental permanent opt-outs.
 * The escape hatch is simply not clicking.
 *
 * Same pointer-events rule as the micro-action pills: NEVER
 * `pointer-events-none` — the pop-out drag region sits behind this strip.
 */

type PillPhase = 'done' | 'idle' | 'working'

export function SuggestionPills({ sessionId }: { sessionId: null | string }) {
  const suggestions = useSessionSlice($composerSuggestionsBySession, sessionId)
  const [phases, setPhases] = useState<Record<string, PillPhase>>({})
  // Cancel flags outlive renders but never trigger them (poll-boundary abort).
  const [cancels] = useState(() => new Map<string, boolean>())

  const setPhase = (key: string, phase: PillPhase) => setPhases(current => ({ ...current, [key]: phase }))

  return suggestions.map(suggestion => {
    const key = suggestionKey(suggestion)
    const brand = suggestion.brand ? brandFor(suggestion.brand) : null
    const phase = phases[key] ?? 'idle'

    const label =
      phase === 'working' ? suggestion.workingLabel : phase === 'done' ? suggestion.doneLabel : suggestion.label
    const tip = phase === 'working' ? suggestion.workingTip : phase === 'done' ? suggestion.doneTip : suggestion.tip

    const invoke = async () => {
      cancels.set(key, false)
      setPhase(key, 'working')
      triggerHaptic('selection')
      // Acting on a pill clears its ignored-count in the bus's declined
      // ledger — its later withdrawal is success, not a strike.
      markSuggestionInvoked(sessionId, key)

      try {
        await suggestion.invoke({ cancelled: () => cancels.get(key) === true, sessionId })
        triggerHaptic('submit')
        setPhase(key, 'done')
      } catch {
        // Provider owns error surfacing (and swallows its own cancels);
        // the pill just returns to idle so it can be tried again.
        setPhase(key, 'idle')
      }
    }

    return (
      <Tip key={key} label={tip}>
        <button
          className={cn(composerFloatingPill, 'max-w-56', phase === 'done' && 'cursor-default')}
          onClick={() => {
            if (phase === 'working') {
              // Second click requests cancel (a stuck OAuth tab, etc.).
              cancels.set(key, true)
            } else if (phase === 'idle') {
              void invoke()
            }
          }}
          type="button"
        >
          {phase === 'working' ? (
            <Codicon className="shrink-0 opacity-70" name="loading" size="0.75rem" spinning />
          ) : phase === 'done' ? (
            <Codicon className="shrink-0 text-emerald-400" name="check" size="0.75rem" />
          ) : brand ? (
            <brand.Icon aria-hidden className="size-3 shrink-0" style={brandGlyphStyle(brand)} />
          ) : (
            <Codicon className="shrink-0 opacity-70" name={suggestion.icon ?? 'lightbulb'} size="0.75rem" />
          )}
          <span className="truncate">{label}</span>
        </button>
      </Tip>
    )
  })
}
