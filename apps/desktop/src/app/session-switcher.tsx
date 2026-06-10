import { useStore } from '@nanostores/react'
import { useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'
import { useNavigate } from 'react-router-dom'

import { sessionTitle } from '@/lib/chat-runtime'
import { cn } from '@/lib/utils'
import { $attentionSessionIds, $workingSessionIds } from '@/store/session'
import { $switcherIndex, $switcherOpen, $switcherSessions, closeSwitcher } from '@/store/session-switcher'

import { sessionRoute } from './routes'

// Compact session-switcher HUD — keyboard-driven from `use-keybinds`, rows
// clickable via mousedown (Ctrl+click on macOS). No Dialog: Tab stays global.
export function SessionSwitcher() {
  const open = useStore($switcherOpen)
  const sessions = useStore($switcherSessions)
  const index = useStore($switcherIndex)
  const working = useStore($workingSessionIds)
  const attention = useStore($attentionSessionIds)
  const navigate = useNavigate()

  const activeRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    activeRef.current?.scrollIntoView({ block: 'nearest' })
  }, [index, open])

  if (!open || sessions.length === 0) {
    return null
  }

  const workingIds = new Set(working)
  const attentionIds = new Set(attention)

  const pick = (sessionId: string) => {
    closeSwitcher()
    navigate(sessionRoute(sessionId))
  }

  return createPortal(
    <div className="fixed inset-0 z-[220] flex select-none items-center justify-center">
      <div
        className="absolute inset-0 bg-black/15 backdrop-blur-[1px]"
        onMouseDown={e => {
          e.preventDefault()
          closeSwitcher()
        }}
      />
      <div className="relative max-h-[min(26rem,70vh)] w-[min(20rem,calc(100vw-2rem))] overflow-y-auto rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-chat-bubble-background) p-1 shadow-lg">
        {sessions.map((session, i) => {
          const selected = i === index

          return (
            <div
              className={cn(
                'flex cursor-pointer items-center gap-2 rounded px-1.5 py-1 text-xs leading-tight',
                selected ? 'bg-accent text-accent-foreground' : 'text-(--ui-text-secondary) hover:bg-(--ui-row-hover-background)'
              )}
              key={session.id}
              onMouseDown={e => {
                e.preventDefault()
                pick(session.id)
              }}
              ref={selected ? activeRef : undefined}
            >
              <SwitcherDot attention={attentionIds.has(session.id)} working={workingIds.has(session.id)} />
              <span className="min-w-0 flex-1 truncate">{sessionTitle(session)}</span>
              {i < 9 && (
                <span
                  className={cn(
                    'shrink-0 font-mono text-[0.625rem] tabular-nums',
                    selected ? 'text-accent-foreground/70' : 'text-(--ui-text-quaternary)'
                  )}
                >
                  ⌃{i + 1}
                </span>
              )}
            </div>
          )
        })}
      </div>
    </div>,
    document.body
  )
}

function SwitcherDot({ attention, working }: { attention: boolean; working: boolean }) {
  return (
    <span
      className={cn(
        'size-1 shrink-0 rounded-full',
        attention ? 'bg-amber-400' : working ? 'animate-pulse bg-(--ui-accent)' : 'bg-(--ui-text-quaternary)/50'
      )}
    />
  )
}
