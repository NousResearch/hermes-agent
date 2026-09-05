import { useStore } from '@nanostores/react'

import { cn } from '@/lib/utils'
import { $traceTurns } from '@/store/trace'

interface TurnStripProps {
  activeIndex: null | number
  allActive: boolean
  liveIndex: null | number
  onAll: () => void
  onTurn: (index: number) => void
}

// Turn nav as a row of timeline bars (à la the thread timeline). Each button is
// full header height (the hit target); the bar inside is short when inactive,
// full height when active/live.
export function TurnStrip({ activeIndex, allActive, liveIndex, onAll, onTurn }: TurnStripProps) {
  const turns = useStore($traceTurns)

  if (turns.length === 0) {
    return null
  }

  return (
    <div className="flex shrink-0 self-stretch items-center gap-2 overflow-x-auto pt-7">
      <button
        aria-label="All turns"
        className={cn(
          'flex h-full shrink-0 items-center gap-1 rounded px-1 text-[0.6rem] font-medium tracking-wide uppercase transition-colors',
          allActive ? 'text-foreground' : 'text-muted-foreground/45 hover:text-foreground/80'
        )}
        onClick={onAll}
        title="All turns"
        type="button"
      >
        All
      </button>
      <div className="flex h-full items-center gap-px">
        {turns.map(turn => {
          const active = turn.index === activeIndex
          const live = turn.index === liveIndex

          return (
            <button
              aria-label={`Turn ${turn.index + 1}`}
              className="group flex h-full items-center px-px"
              key={turn.index}
              onClick={() => onTurn(turn.index)}
              onMouseEnter={() => onTurn(turn.index)}
              title={`#${turn.index + 1} · ${turn.label}`}
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
          )
        })}
      </div>
    </div>
  )
}
