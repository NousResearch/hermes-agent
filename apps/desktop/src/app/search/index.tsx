import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { searchSessions, type SessionInfo, type SessionSearchResult } from '@/hermes'
import { useI18n } from '@/i18n'
import { sessionTitle } from '@/lib/chat-runtime'
import { searchResultToSession, sessionMatchesSearch } from '@/lib/session-search'
import { cn } from '@/lib/utils'
import { SESSION_SEARCH_FOCUS_EVENT } from '@/store/layout'
import { $sessions } from '@/store/session'

import { formatAge } from '../chat/sidebar/session-row'
import { PageSearchShell } from '../page-search-shell'
import { sessionRoute } from '../routes'
import type { SetStatusbarItemGroup } from '../shell/statusbar-controls'

// How many recent conversations show under an empty query — the page opens as
// a browsable recents list (Codex-style), not a blank slate.
const RECENT_LIMIT = 30

interface SearchViewProps extends React.ComponentProps<'section'> {
  setStatusbarItemGroup?: SetStatusbarItemGroup
}

/**
 * 搜索 as a main-area page (same shell as 插件/产物): a search field up top,
 * matching conversations below. Local matches over the loaded session page are
 * instant; a debounced server-side FTS pass covers everything beyond it. Rows
 * navigate to the conversation.
 */
export function SearchView({ setStatusbarItemGroup: _setStatusbarItemGroup, ...props }: SearchViewProps) {
  const { t } = useI18n()
  const s = t.sidebar
  const navigate = useNavigate()
  const sessions = useStore($sessions)
  const [query, setQuery] = useState('')
  const [serverMatches, setServerMatches] = useState<SessionSearchResult[]>([])
  const inputRef = useRef<HTMLInputElement>(null)
  const trimmedQuery = query.trim()

  // The page owns the search — focus the field on mount, and again when the
  // focus-search hotkey fires while the page is already open.
  useEffect(() => {
    const focus = () => inputRef.current?.focus({ preventScroll: true })

    focus()
    window.addEventListener(SESSION_SEARCH_FOCUS_EVENT, focus)

    return () => window.removeEventListener(SESSION_SEARCH_FOCUS_EVENT, focus)
  }, [])

  // Full-text search across *all* sessions (not just the loaded page) so long
  // histories stay findable. Debounced; loaded sessions are matched instantly
  // client-side and merged ahead of the server hits.
  useEffect(() => {
    if (!trimmedQuery) {
      setServerMatches([])

      return
    }

    let cancelled = false

    const id = window.setTimeout(() => {
      void searchSessions(trimmedQuery)
        .then(res => {
          if (!cancelled) {
            setServerMatches(res.results)
          }
        })
        .catch(() => undefined)
    }, 200)

    return () => {
      cancelled = true
      window.clearTimeout(id)
    }
  }, [trimmedQuery])

  const recentSessions = useMemo(
    () => [...sessions].sort((a, b) => (b.last_active || b.started_at) - (a.last_active || a.started_at)),
    [sessions]
  )

  const results = useMemo(() => {
    if (!trimmedQuery) {
      return recentSessions.slice(0, RECENT_LIMIT)
    }

    const out = new Map<string, SessionInfo>()
    const lineageSeen = new Set<string>()

    for (const session of recentSessions) {
      if (sessionMatchesSearch(session, trimmedQuery)) {
        out.set(session.id, session)
        lineageSeen.add(session._lineage_root_id ?? session.id)
      }
    }

    for (const match of serverMatches) {
      if (out.has(match.session_id) || lineageSeen.has(match.lineage_root ?? match.session_id)) {
        continue
      }

      out.set(match.session_id, searchResultToSession(match))
    }

    return [...out.values()]
  }, [trimmedQuery, recentSessions, serverMatches])

  return (
    <PageSearchShell
      {...props}
      centered
      onSearchChange={setQuery}
      searchInputRef={inputRef}
      searchPlaceholder={s.searchPlaceholder}
      searchValue={query}
    >
      {/* Shell owns the scroll + centered max-w-2xl column (centered mode); this
          just supplies the column's content and its vertical padding. */}
      <div className="pb-8 pt-1">
        {trimmedQuery && results.length === 0 ? (
          <div className="grid min-h-32 place-items-center px-2 text-center text-sm text-(--ui-text-tertiary)">
            {s.noMatch(trimmedQuery)}
          </div>
        ) : (
          <ul className="flex flex-col gap-px">
            {results.map(session => (
              <SearchResultRow
                key={session.id}
                onOpen={() => navigate(sessionRoute(session.id))}
                session={session}
              />
            ))}
          </ul>
        )}
      </div>
    </PageSearchShell>
  )
}

function SearchResultRow({ onOpen, session }: { onOpen: () => void; session: SessionInfo }) {
  const { t } = useI18n()
  const title = sessionTitle(session)
  const preview = session.preview?.trim() ?? ''
  const age = formatAge(session.last_active || session.started_at, t.sidebar.row)

  return (
    <li>
      <button
        className={cn(
          'flex w-full items-baseline gap-3 rounded-[0.625rem] px-3 py-2 text-left transition-colors duration-100',
          'hover:bg-(--ui-control-hover-background)'
        )}
        onClick={onOpen}
        type="button"
      >
        <span className="min-w-0 flex-1">
          <span className="block truncate text-[0.8125rem] font-medium text-foreground">{title}</span>
          {preview && preview !== title ? (
            <span className="mt-0.5 block truncate text-xs text-(--ui-text-tertiary)">{preview}</span>
          ) : null}
        </span>
        <span className="shrink-0 text-xs tabular-nums text-(--ui-text-quaternary)">{age}</span>
      </button>
    </li>
  )
}
