import { useStore } from '@nanostores/react'
import { useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'
import { useNavigate } from 'react-router-dom'

import { useI18n } from '@/i18n'
import { sessionTitle } from '@/lib/chat-runtime'
import { cn } from '@/lib/utils'
import {
  $attentionSessionIds,
  $unreadFinishedSessionIds,
  requestSessionResumeProfile,
  sessionHasUnread,
  sessionNeedsInput,
  sessionScopeKey
} from '@/store/session'
import { $sessionActivityKeys, sessionActivityKey } from '@/store/session-activity'
import { $switcherIndex, $switcherOpen, $switcherSessions, closeSwitcher } from '@/store/session-switcher'

import { HUD_ITEM, HUD_POSITION, HUD_SURFACE, HUD_TEXT } from './floating-hud'
import { sessionRoute } from './routes'

// Compact session-switcher HUD — keyboard-driven from `use-keybinds`, rows
// clickable via mousedown (Ctrl+click on macOS). No Dialog: Tab stays global.
export function SessionSwitcher({ onResume }: { onResume?: (sessionId: string) => void }) {
  const open = useStore($switcherOpen)
  const sessions = useStore($switcherSessions)
  const index = useStore($switcherIndex)
  const working = useStore($sessionActivityKeys)
  const attention = useStore($attentionSessionIds)
  const unread = useStore($unreadFinishedSessionIds)
  const { t } = useI18n()
  const navigate = useNavigate()

  const activeRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    activeRef.current?.scrollIntoView({ block: 'nearest' })
  }, [index, open])

  if (!open || sessions.length === 0) {
    return null
  }

  const workingIds = new Set(working)

  const pick = (sessionId: string, profile?: string) => {
    closeSwitcher()
    requestSessionResumeProfile(sessionId, profile)

    if (onResume) {
      onResume(sessionId)
    } else {
      navigate(sessionRoute(sessionId))
    }
  }

  return createPortal(
    <>
      {/* Transparent click-catcher: click-away closes, but no dim/blur. */}
      <div
        className="fixed inset-0 z-[219]"
        onMouseDown={e => {
          e.preventDefault()
          closeSwitcher()
        }}
      />
      <div
        className={cn(
          HUD_POSITION,
          HUD_SURFACE,
          'dt-portal-scrollbar z-[220] max-h-[min(22rem,64vh)] w-[min(19rem,calc(100vw-2rem))] select-none overflow-y-auto p-1'
        )}
      >
        {sessions.map((session, i) => {
          const selected = i === index
          const sessionWorking = workingIds.has(sessionActivityKey(session.profile, session.id))
          const sessionAttention = sessionNeedsInput(session.id, session.profile)
          const sessionUnread = sessionHasUnread(session.id, session.profile)

          const activityText = sessionAttention
            ? t.profiles.activityNeedsInput
            : sessionUnread
              ? t.profiles.activityUnread
              : sessionWorking
                ? t.profiles.activityRunning
                : null

          return (
            <div
              className={cn(
                'row-hover flex items-center rounded leading-tight',
                HUD_ITEM,
                HUD_TEXT,
                selected ? 'bg-accent text-accent-foreground' : 'text-(--ui-text-secondary)'
              )}
              data-working={sessionWorking ? 'true' : undefined}
              key={sessionScopeKey(session.profile, session.id)}
              onMouseDown={e => {
                e.preventDefault()
                pick(session.id, session.profile)
              }}
              ref={selected ? activeRef : undefined}
            >
              <SwitcherDot attention={sessionAttention} unread={sessionUnread} working={sessionWorking} />
              <span className="min-w-0 flex-1 truncate">{sessionTitle(session)}</span>
              {activityText && <span className="sr-only">{activityText}</span>}
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

function SwitcherDot({ attention, working, unread }: { attention: boolean; working: boolean; unread: boolean }) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        'size-1.5 shrink-0',
        attention
          ? 'rotate-45 rounded-[1px] bg-amber-400'
          : working
            ? 'animate-pulse rounded-full border border-(--ui-accent) bg-transparent motion-reduce:animate-none'
            : unread
              ? 'rounded-[1px] bg-emerald-500'
              : 'rounded-full bg-(--ui-text-quaternary)/50'
      )}
    />
  )
}
