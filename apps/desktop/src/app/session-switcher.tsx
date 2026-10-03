import { useStore } from '@nanostores/react'
import { useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'
import { useNavigate } from 'react-router'

import { sessionTitle } from '@/lib/chat-runtime'
import { cn } from '@/lib/utils'
import { $switcherIndex, $switcherOpen, $switcherSessions, closeSwitcher, switcherRowKey } from '@/store/session-switcher'

import { SessionStatusDot } from './chat/session-status-dot'
import { HUD_ITEM, HUD_POSITION, HUD_SURFACE, HUD_TEXT } from './floating-hud'
import { openSessionFromRow } from './open-session'

// Compact session-switcher HUD — keyboard-driven from `use-keybinds`, rows
// clickable via mousedown (Ctrl+click on macOS). No Dialog: Tab stays global.
export function SessionSwitcher() {
  const open = useStore($switcherOpen)
  const sessions = useStore($switcherSessions)
  const index = useStore($switcherIndex)
  const navigate = useNavigate()

  const activeRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    activeRef.current?.scrollIntoView({ block: 'nearest' })
  }, [index, open])

  if (!open || sessions.length === 0) {
    return null
  }

  // The row is the identity: stored ids are only unique per profile (#92454),
  // so this surface opens the ROW the user clicked, pinning that row's own
  // (connection, profile) as the resume owner exactly as the Sessions sidebar
  // row does. With twins sharing one id, an id-only open could not say which
  // chat it picked.
  const pick = (row: (typeof sessions)[number]) => {
    closeSwitcher()
    openSessionFromRow(row, navigate)
  }

  return createPortal(
    <>
      {/* Transparent click-catcher: click-away closes, but no dim/blur. */}
      <div
        className="fixed inset-0 z-(--z-switcher-backdrop)"
        onMouseDown={e => {
          e.preventDefault()
          closeSwitcher()
        }}
      />
      <div
        className={cn(
          HUD_POSITION,
          HUD_SURFACE,
          'dt-portal-scrollbar z-(--z-switcher) max-h-[min(22rem,64vh)] w-[min(19rem,calc(100vw-2rem))] select-none overflow-y-auto p-1'
        )}
      >
        {sessions.map((session, i) => {
          const selected = i === index

          return (
            <div
              className={cn(
                'row-hover flex items-center rounded leading-tight',
                HUD_ITEM,
                HUD_TEXT,
                selected ? 'bg-accent text-accent-foreground' : 'text-(--ui-text-secondary)'
              )}
              // Key by (profile, id): twins with the same stored id in two
              // profiles are distinct rows (#92454), and a bare-id key collapses
              // them so the highlighted row renders another twin's state.
              key={switcherRowKey(session)}
              onMouseDown={e => {
                e.preventDefault()
                pick(session)
              }}
              ref={selected ? activeRef : undefined}
            >
              <SessionStatusDot className="shrink-0" session={session} storedSessionId={session.id} />
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
    </>,
    document.body
  )
}
