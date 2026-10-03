import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import type { TraceTurnSummary } from '@/store/trace'

interface TurnStripProps {
  activeIndex: null | number
  allActive: boolean
  liveIndex: null | number
  onAll: () => void
  onTurn: (index: number) => void
  turns: TraceTurnSummary[]
}

// Turn nav as a row of timeline bars (à la the thread timeline). Each button is
// full header height (the hit target); the bar inside is short when inactive,
// full height when active/live.
export function TurnStrip({ activeIndex, allActive, liveIndex, onAll, onTurn, turns }: TurnStripProps) {
  const { t } = useI18n()
  const a = t.agents

  if (turns.length === 0) {
    return null
  }

  return (
    <div className="flex shrink-0 items-center gap-2 self-stretch overflow-x-auto">
      <Tip label={a.allTurns}>
        <button
          aria-label={a.allTurns}
          className={cn(
            'flex h-full shrink-0 items-center gap-1 rounded px-1 text-[0.6rem] font-medium tracking-wide uppercase transition-colors',
            allActive ? 'text-foreground' : 'text-muted-foreground/45 hover:text-foreground/80'
          )}
          onClick={onAll}
          type="button"
        >
          {a.allTurnsShort}
        </button>
      </Tip>
      <div className="flex h-full items-center gap-px">
        {turns.map(turn => {
          const active = turn.index === activeIndex
          const live = turn.index === liveIndex

          return (
            <Tip key={turn.id} label={a.turnTip(turn.index + 1, turn.label)}>
              <button
                aria-label={a.turnAria(turn.index + 1)}
                className="group flex h-full items-center px-px"
                onClick={() => onTurn(turn.index)}
                onMouseEnter={() => onTurn(turn.index)}
                type="button"
              >
                {/* Fixed-height box so the strip never grows when a bar activates;
                    only the inner fill changes height. */}
                <span className="flex h-4 w-[3px] items-center justify-center">
                  <span
                    className={cn(
                      'w-full rounded-full transition-all duration-100 ease-out group-hover:transition-none',
                      live
                        ? 'h-full animate-pulse bg-emerald-500'
                        : active
                          ? 'h-full bg-foreground'
                          : 'h-1/2 bg-foreground/25 group-hover:h-3/4 group-hover:bg-foreground/50'
                    )}
                  />
                </span>
              </button>
            </Tip>
          )
        })}
      </div>
    </div>
  )
}
