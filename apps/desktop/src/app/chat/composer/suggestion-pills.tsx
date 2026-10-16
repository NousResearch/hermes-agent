import { useEffect, useState } from 'react'

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

/**
 * Phase belongs to the pill the user is looking at, so it lives and dies with
 * that pill — the bus owns whether a suggestion exists, this owns only how far
 * along it is. Remounting per session enforces the half of that the strip
 * can't see: one composer stays mounted across a session switch, and phase
 * keys are `provider:id`, so without this "Added GitHub" in one chat renders
 * the next chat's genuine offer as already-done and inert.
 */
export function SuggestionPills({ sessionId }: { sessionId: null | string }) {
  return <SessionSuggestionPills key={sessionId ?? ''} sessionId={sessionId} />
}

function SessionSuggestionPills({ sessionId }: { sessionId: null | string }) {
  const suggestions = useSessionSlice($composerSuggestionsBySession, sessionId)
  const [phases, setPhases] = useState<Record<string, PillPhase>>({})
  // Cancel flags outlive renders but never trigger them (poll-boundary abort).
  const [cancels] = useState(() => new Map<string, boolean>())

  const setPhase = (key: string, phase: PillPhase) => setPhases(current => ({ ...current, [key]: phase }))

  // A pill that leaves the strip takes its phase with it, and cancels whatever
  // it had in flight. Both halves matter:
  //
  //  - Phase, because keys repeat. A provider withdraws and re-offers the same
  //    `provider:id` constantly (the draft loses and regains the trigger word,
  //    a connect fails and the word is still typed), and a `done` left on file
  //    paints the fresh offer as "Added GitHub" — a pill that says it already
  //    worked and ignores clicks, since only `idle` invokes.
  //  - Cancellation, because a withdrawn pill is the user's ONLY cancel
  //    affordance. Clicking a working pill sets this flag; once the pill is
  //    gone nothing can, so an OAuth flow the user walked away from would poll
  //    forever, hold the server's in-progress slot against a retry, and either
  //    land a server they never confirmed or roll one back with no UI to say
  //    so. Withdrawal aborts it at the next poll boundary and lets the
  //    provider roll back.
  //
  // `suggestions` is reference-stable while the offered set is unchanged (the
  // bus preserves identity on no-op writes), so this runs on real churn only.
  useEffect(() => {
    const live = new Set(suggestions.map(suggestionKey))

    for (const key of cancels.keys()) {
      if (!live.has(key)) {
        cancels.set(key, true)
      }
    }

    setPhases(current => {
      const kept = Object.entries(current).filter(([key]) => live.has(key))

      return kept.length === Object.keys(current).length ? current : Object.fromEntries(kept)
    })
  }, [cancels, suggestions])

  // Unmount withdraws everything at once (composer closing, session switch
  // remounting this subtree) — same rule, same reason.
  useEffect(
    () => () => {
      for (const key of cancels.keys()) {
        cancels.set(key, true)
      }
    },
    [cancels]
  )

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
      } finally {
        cancels.delete(key)
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
