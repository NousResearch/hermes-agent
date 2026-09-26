import { useEffect, useRef } from 'react'

import { CodeCardBody } from '@/components/chat/code-card'
import { CopyButton } from '@/components/ui/copy-button'
import { cn } from '@/lib/utils'

interface LogTailProps {
  /** null = still loading (shows the loading glyph); [] = loaded-but-empty
   *  (shows `emptyLabel`); non-empty renders as a tailing terminal log. */
  lines: null | string[]
  emptyLabel: string
  className?: string
}

/** The shared terminal-log surface: CodeCardBody typography, a hover-reveal copy
 *  button, and follow-the-tail scrolling (releases when the user scrolls up).
 *  One component behind every log pane — MCP stdio/agent, hub action logs, etc.
 *  — so they all read, copy, and scroll identically. */
export function LogTail({ className, emptyLabel, lines }: LogTailProps) {
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const stickRef = useRef(true)

  useEffect(() => {
    const el = scrollRef.current

    if (el && stickRef.current) {
      el.scrollTop = el.scrollHeight
    }
  }, [lines])

  return (
    <div className={cn('group/logs relative h-full min-h-0', className)}>
      <CopyButton
        appearance="inline"
        className="absolute right-2.5 top-1.5 z-10 h-5 gap-0 rounded-md px-1 opacity-5 transition-opacity group-hover/logs:opacity-100 hover:opacity-100 focus-visible:opacity-100"
        iconClassName="size-3"
        showLabel={false}
        text={() => (lines ?? []).join('\n')}
      />
      <div
        className="h-full min-h-0 overflow-y-auto [scrollbar-gutter:stable]"
        data-selectable-text="true"
        onScroll={event => {
          const el = event.currentTarget
          stickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24
        }}
        ref={scrollRef}
      >
        {lines === null || lines.length === 0 ? (
          <p className="px-2 py-1.5 font-mono text-[0.7rem] leading-relaxed text-muted-foreground/50">
            {lines === null ? '…' : emptyLabel}
          </p>
        ) : (
          <CodeCardBody>
            <pre className="whitespace-pre-wrap break-words">
              {lines.map((line, index) => (
                <span className={cn('block', line.startsWith('=====') && 'mt-1 text-(--ui-text-tertiary)')} key={index}>
                  {line}
                </span>
              ))}
            </pre>
          </CodeCardBody>
        )}
      </div>
    </div>
  )
}
