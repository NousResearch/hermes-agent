import { useCallback, useEffect, useRef, useState } from 'react'

import { CompactMarkdown } from '@/components/chat/compact-markdown'
import { Button } from '@/components/ui/button'
import { getSessionMessages } from '@/hermes'
import type { SessionMessage } from '@/types/hermes'

const ARCHIVED_PAGE_SIZE = 100

type ArchivedTranscriptProps = {
  profile?: string | null
  sessionId?: string | null
}

function messageText(message: SessionMessage): string {
  const value = message.content ?? message.text ?? message.context ?? message.name

  if (typeof value === 'string') {
    return value
  }

  if (value === null || value === undefined) {
    return ''
  }

  if (Array.isArray(value)) {
    return value
      .map(item => (typeof item === 'string' ? item : JSON.stringify(item)))
      .filter(Boolean)
      .join('\n')
  }

  if (typeof value === 'object') {
    const record = value as Record<string, unknown>

    for (const key of ['text', 'content', 'message', 'value']) {
      if (typeof record[key] === 'string') {
        return record[key] as string
      }
    }

    try {
      return JSON.stringify(value, null, 2) || ''
    } catch {
      return ''
    }
  }

  return String(value)
}

export function ArchivedTranscript({ profile, sessionId }: ArchivedTranscriptProps) {
  const requestVersion = useRef(0)
  const sessionKey = `${profile ?? ''}:${sessionId ?? ''}`
  const previousSessionKey = useRef(sessionKey)

  if (previousSessionKey.current !== sessionKey) {
    previousSessionKey.current = sessionKey
    requestVersion.current += 1
  }

  const [expanded, setExpanded] = useState(false)
  const [messages, setMessages] = useState<SessionMessage[]>([])
  const [nextOffset, setNextOffset] = useState(0)
  const [hasMore, setHasMore] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadPage = useCallback(
    async (offset: number, append: boolean) => {
      if (!sessionId) {
        return
      }

      const version = ++requestVersion.current
      setLoading(true)
      setError(null)

      try {
        const page = await getSessionMessages(sessionId, profile, {
          limit: ARCHIVED_PAGE_SIZE,
          offset,
          order: 'latest',
          scope: 'compacted'
        })

        if (version !== requestVersion.current) {
          return
        }

        setMessages(previous => (append ? [...page.messages, ...previous] : page.messages))
        setNextOffset(offset + page.messages.length)
        setHasMore(page.pagination?.has_more ?? page.messages.length === ARCHIVED_PAGE_SIZE)
      } catch (cause) {
        if (version !== requestVersion.current) {
          return
        }

        setError(cause instanceof Error ? cause.message : String(cause))
      } finally {
        if (version === requestVersion.current) {
          setLoading(false)
        }
      }
    },
    [profile, sessionId]
  )

  useEffect(() => {
    setExpanded(false)
    setMessages([])
    setNextOffset(0)
    setHasMore(false)
    setLoading(false)
    setError(null)
  }, [profile, sessionId])

  if (!sessionId) {
    return null
  }

  const visibleMessages = messages
    .filter(message => message.display_kind !== 'hidden')
    .filter(message => message.role === 'user' || message.role === 'assistant')
    .map(message => ({ message, text: messageText(message) }))
    .filter(item => item.text.trim())

  const toggleExpanded = () => {
    const nextExpanded = !expanded
    setExpanded(nextExpanded)

    if (nextExpanded && messages.length === 0 && !loading) {
      void loadPage(0, false)
    }
  }

  return (
    <section
      aria-label="Archived conversation history"
      className="shrink-0 border-b border-border/60 bg-muted/15"
      data-testid="archived-transcript"
    >
      <Button
        aria-expanded={expanded}
        className="flex h-9 w-full items-center justify-between rounded-none px-3 text-xs text-muted-foreground hover:bg-muted/30"
        onClick={toggleExpanded}
        variant="ghost"
      >
        <span>Archived history</span>
        <span className="tabular-nums opacity-70">
          {messages.length}
          {hasMore ? '+' : ''} {expanded ? '▴' : '▾'}
        </span>
      </Button>

      {expanded && (
        <div className="max-h-72 overflow-y-auto border-t border-border/40 px-3 py-2">
          {loading && <div className="px-1 py-2 text-xs text-muted-foreground">Loading archived history…</div>}
          {error && <div className="px-1 py-2 text-xs text-destructive">Could not load archived history: {error}</div>}
          {!loading && !error && visibleMessages.length === 0 && (
            <div className="px-1 py-2 text-xs text-muted-foreground">No archived prompts in this session.</div>
          )}

          <div className="space-y-3">
            {visibleMessages.map(({ message, text }, index) => (
              <article
                className="rounded-md border border-border/50 bg-background/50 px-2.5 py-2"
                key={message.id ?? message.row_id ?? `archived-${index}`}
              >
                <div className="mb-1 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
                  {message.role === 'user' ? 'You' : 'Assistant'}
                </div>
                {message.role === 'assistant' ? (
                  <CompactMarkdown text={text} />
                ) : (
                  <div className="whitespace-pre-wrap break-words text-xs leading-relaxed text-foreground/90">
                    {text}
                  </div>
                )}
              </article>
            ))}
          </div>

          {hasMore && !loading && (
            <Button
              className="mt-3 h-7 w-full text-xs"
              onClick={() => void loadPage(nextOffset, true)}
              variant="outline"
            >
              Load older archived messages
            </Button>
          )}
        </div>
      )}
    </section>
  )
}
