import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useRef } from 'react'

import { type ConversationMatch, searchConversation } from '@/api/conversation-search'
import { useTranscriptWindow } from '@/components/assistant-ui/thread/transcript-window'
import { Button } from '@/components/ui/button'
import { ErrorState } from '@/components/ui/error-state'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { SearchField } from '@/components/ui/search-field'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { ChevronDown, ChevronUp, Search, X } from '@/lib/icons'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, getSessionOwnerHint } from '@/store/session'

import { clearConversationHighlight, revealConversationMatch } from './conversation-search-jump'
import { createConversationSearch } from './conversation-search-state'
import { useSessionView } from './session-view'

export function ConversationSearch() {
  const { t } = useI18n()
  const copy = t.conversationSearch
  const { searchAvailable } = useTranscriptWindow()
  const view = useSessionView()
  const storedId = useStore(view.$storedId)
  const runtimeId = useStore(view.$runtimeId)
  const connection = useStore($connection)
  const profile = useStore($activeGatewayProfile)
  const connectionId = connection?.connectionId || (connection?.mode === 'local' ? 'local' : '')
  const owner = storedId ? getSessionOwnerHint(storedId, { connectionId, profile }) : undefined
  const ownerConnection = owner?.connectionId || connectionId
  const ownerProfile = owner?.targetProfile || owner?.profile || profile

  const route = useMemo(
    () => ({
      storedId,
      runtimeId,
      searchAvailable,
      scope: { connectionId: ownerConnection || undefined, profile: ownerProfile }
    }),
    [storedId, runtimeId, ownerConnection, ownerProfile, searchAvailable]
  )

  const search = useMemo(
    () => createConversationSearch((query, offset) => searchConversation(route.storedId!, route.scope, query, offset)),
    [route]
  )

  const state = useStore(search.state)
  const trigger = useRef<HTMLButtonElement>(null)
  const input = useRef<HTMLInputElement>(null)
  const jumping = useRef<AbortController | null>(null)
  const surface = () => trigger.current?.closest<HTMLElement>('[data-chat-surface]') ?? null

  useEffect(() => {
    const root = surface()

    return () => {
      search.dispose()
      jumping.current?.abort()
      clearConversationHighlight(root)
    }
  }, [search])
  useEffect(() => {
    jumping.current?.abort()
    clearConversationHighlight(surface())

    if (!state.open || !state.query.trim()) {
      return
    }

    const timer = window.setTimeout(() => {
      void search.search()
    }, 250)

    return () => clearTimeout(timer)
  }, [search, state.open, state.query])

  const jump = async (hit: ConversationMatch | null) => {
    if (!hit) {
      return
    }

    const root = surface()

    if (!root) {
      return
    }

    jumping.current?.abort()
    const controller = new AbortController()
    jumping.current = controller
    const revealed = await revealConversationMatch(root, hit.row_id, controller.signal)

    if (!revealed && !controller.signal.aborted) {
      search.jumpFailed()
    }
  }

  const page = state.page
  const ready = state.phase === 'ready' && Boolean(page?.results.length)
  const atStart = state.selected <= 0 && (!page || page.pagination.offset === 0)
  const atEnd = Boolean(page && state.selected === page.results.length - 1 && !page.pagination.has_more)

  return (
    <Popover onOpenChange={open => search.reset(open)} open={state.open}>
      <Tip label={copy.open}>
        <PopoverTrigger asChild>
          <Button
            aria-label={copy.open}
            className="no-drag pointer-events-auto shrink-0"
            disabled={!storedId || !searchAvailable}
            ref={trigger}
            size="icon-titlebar"
            variant="ghost"
          >
            <Search />
          </Button>
        </PopoverTrigger>
      </Tip>
      <PopoverContent
        align="start"
        className="w-96 max-w-[calc(100vw-1rem)]"
        onOpenAutoFocus={event => {
          event.preventDefault()
          input.current?.focus()
        }}
      >
        <div className="flex items-center gap-1">
          <SearchField
            aria-label={copy.open}
            containerClassName="flex-1"
            inputRef={input}
            loading={state.phase === 'pending'}
            onChange={search.setQuery}
            onKeyDown={event => {
              if (event.key === 'Enter') {
                event.preventDefault()
                void search.step(event.shiftKey ? -1 : 1).then(jump)
              }
            }}
            placeholder={copy.placeholder}
            value={state.query}
          />
          <Button
            aria-label={t.findInPage.previous}
            disabled={!ready || atStart}
            onClick={() => void search.step(-1).then(jump)}
            size="icon-xs"
            variant="ghost"
          >
            <ChevronUp />
          </Button>
          <Button
            aria-label={t.findInPage.next}
            disabled={!ready || atEnd}
            onClick={() => void search.step(1).then(jump)}
            size="icon-xs"
            variant="ghost"
          >
            <ChevronDown />
          </Button>
          <Button aria-label={copy.close} onClick={() => search.reset()} size="icon-xs" variant="ghost">
            <X />
          </Button>
        </div>
        <div aria-live="polite" className="text-xs text-muted-foreground">
          {state.phase === 'pending' && copy.searching}
          {state.phase === 'ready' &&
            page &&
            (page.results.length
              ? copy.matches(page.pagination.offset + 1, page.pagination.offset + page.results.length)
              : copy.empty)}
        </div>
        {state.error && (
          <ErrorState title={copy[state.error]}>
            <Button onClick={() => void search.search()} size="sm" variant="secondary">
              {copy.retry}
            </Button>
          </ErrorState>
        )}
        <div className="max-h-72 overflow-y-auto">
          {page?.results.map((hit, index) => (
            <Button
              className="w-full justify-start text-left whitespace-normal"
              key={hit.row_id}
              onClick={() => void jump(search.select(index))}
              size="sm"
              variant={state.selected === index ? 'secondary' : 'ghost'}
            >
              <span className="line-clamp-3 break-words">{hit.snippet.replace(/>>>|<<</g, '')}</span>
            </Button>
          ))}
        </div>
      </PopoverContent>
    </Popover>
  )
}
