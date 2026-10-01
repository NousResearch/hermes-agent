import { useStore } from '@nanostores/react'
import { type PointerEvent as ReactPointerEvent, useMemo, useState } from 'react'
import { useLocation, useNavigate } from 'react-router'

import { sessionRecency } from '@/app/chat/sidebar/projects/workspace-groups'
import { openSession } from '@/app/open-session'
import { NEW_CHAT_ROUTE, routeSessionId } from '@/app/routes'
import { useI18n } from '@/i18n'
import { sessionTitle } from '@/lib/chat-runtime'
import { MessageCircle, Plus } from '@/lib/icons'
import { relativeTime } from '@/lib/time'
import { $activeSessionId, $sessions, $sessionsLoading } from '@/store/session'

const MAX_ROWS = 30

/**
 * Kirsin header session switcher — a Kirsin-ONLY dropdown (the window adopted
 * the `kirsin` backend, so `$sessions` here is already Kirsin's list; no
 * profile picker, no cross-profile aggregation).
 *
 * The trigger is a quiet message glyph in the header (left of the collapse
 * chevron); it stops header drag so a press opens the menu instead of moving
 * the window, and highlights while the menu is open. Rows resume in place
 * through the same door the sidebar uses (`openSession(id, navigate, 'main')`),
 * ordered by the same `sessionRecency` the sidebar sorts on, and the top row
 * mints a fresh chat (`NEW_CHAT_ROUTE`).
 *
 * Presentation is a data-attribute CSS block in styles.css (the
 * `[data-kirsin-switcher-*]` rules) using the locked Kirsin palette — the same
 * way the header controls are skinned, so it stays consistent with the rest of
 * the window.
 */
export function KirsinSessionSwitcher() {
  const { t } = useI18n()
  const navigate = useNavigate()
  const location = useLocation()
  const sessions = useStore($sessions)
  const loading = useStore($sessionsLoading)
  const activeSessionId = useStore($activeSessionId)
  const [open, setOpen] = useState(false)

  // The Kirsin window is a single surface — the route IS the open chat, so the
  // active row is whichever session the current path resolves to.
  const routedId = routeSessionId(location.pathname) ?? activeSessionId

  // Sidebar parity: most-recently-active first, capped so the menu never grows
  // unbounded (the newest 30 cover a realistic working set; older sessions are
  // still reachable from the main app's sidebar).
  const rows = useMemo(
    () => [...sessions].sort((a, b) => sessionRecency(b) - sessionRecency(a)).slice(0, MAX_ROWS),
    [sessions]
  )

  const stopHeaderDrag = (e: ReactPointerEvent) => e.stopPropagation()

  return (
    <div data-kirsin-switcher>
      <button
        aria-expanded={open}
        aria-label={t.kirsin.sessions}
        data-kirsin-open={open ? 'true' : undefined}
        data-kirsin-switcher-trigger
        onClick={e => {
          e.stopPropagation()
          setOpen(o => !o)
        }}
        onPointerDown={stopHeaderDrag}
        type="button"
      >
        <MessageCircle />
      </button>

      {open && (
        <>
          {/* Click-catcher: closes on any outside press. Sits in-window (not a
              portal) so it stays within the floating panel's bounds. */}
          <div data-kirsin-switcher-catcher onMouseDown={() => setOpen(false)} />
          <div data-kirsin-switcher-menu>
            <button
              data-kirsin-new
              onClick={e => {
                e.stopPropagation()
                setOpen(false)
                navigate(NEW_CHAT_ROUTE)
              }}
              onPointerDown={stopHeaderDrag}
              type="button"
            >
              <Plus className="shrink-0" />
              <span>{t.kirsin.newSession}</span>
            </button>
            <div data-kirsin-switcher-divider />
            <div data-kirsin-switcher-rows>
              {loading && rows.length === 0 ? (
                <div data-kirsin-empty>{t.kirsin.loading}</div>
              ) : rows.length === 0 ? (
                <div data-kirsin-empty>{t.kirsin.noSessions}</div>
              ) : (
                rows.map(session => {
                  const active = session.id === routedId

                  return (
                    <button
                      data-kirsin-active={active ? 'true' : undefined}
                      data-kirsin-row
                      key={session.id}
                      onClick={e => {
                        e.stopPropagation()
                        setOpen(false)
                        openSession(session.id, navigate, 'main')
                      }}
                      onPointerDown={stopHeaderDrag}
                      type="button"
                    >
                      <MessageCircle className="shrink-0" />
                      <span data-kirsin-row-title>{sessionTitle(session)}</span>
                      <span data-kirsin-row-time>
                        {relativeTime(sessionRecency(session) * 1000)}
                      </span>
                      {active && <span data-kirsin-row-dot />}
                    </button>
                  )
                })
              )}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
