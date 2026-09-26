import { type ReactNode, useEffect, useRef } from 'react'

import { Button } from '@/components/ui/button'
import { CopyButton } from '@/components/ui/copy-button'
import { LogView } from '@/components/ui/log-view'
import { Tip } from '@/components/ui/tooltip'
import {
  AlertCircle,
  AlertTriangle,
  ArrowBarToDown,
  ArrowBarToUp,
  CircleIcon,
  Info
} from '@/lib/icons'
import { cn } from '@/lib/utils'

import { commandCenterLogSeverity, highlightLogSegments, type CommandCenterLogSeverity } from './log-lines'

interface CommandCenterLogViewProps {
  bottomLabel: string
  emptyLabel: string
  lines: null | string[]
  query: string
  topLabel: string
}

const SEVERITY_PRESENTATION: Record<
  CommandCenterLogSeverity,
  { icon: typeof Info; iconClassName: string; label: string }
> = {
  CRITICAL: { icon: AlertCircle, iconClassName: 'text-destructive', label: 'CRITICAL' },
  DEBUG: { icon: CircleIcon, iconClassName: 'text-(--ui-text-tertiary)', label: 'DEBUG' },
  ERROR: { icon: AlertCircle, iconClassName: 'text-destructive', label: 'ERROR' },
  INFO: { icon: Info, iconClassName: 'text-[color:var(--ui-blue)]', label: 'INFO' },
  WARNING: { icon: AlertTriangle, iconClassName: 'text-[color:var(--ui-yellow)]', label: 'WARNING' }
}

function HighlightedLine({ line, query }: { line: string; query: string }) {
  return highlightLogSegments(line, query).map((segment, index) =>
    segment.match ? (
      <mark className="rounded-[2px] bg-[color:var(--ui-yellow)]/35 text-foreground" key={index}>
        {segment.text}
      </mark>
    ) : (
      <span key={index}>{segment.text}</span>
    )
  )
}

function NavigationButton({
  children,
  label,
  onClick
}: {
  children: ReactNode
  label: string
  onClick: () => void
}) {
  return (
    <Tip label={label} placement="toolbar">
      <Button aria-label={label} onClick={onClick} size="icon-xs" type="button" variant="ghost">
        {children}
      </Button>
    </Tip>
  )
}

export function CommandCenterLogView({
  bottomLabel,
  emptyLabel,
  lines,
  query,
  topLabel
}: CommandCenterLogViewProps) {
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const stickRef = useRef(true)

  useEffect(() => {
    const el = scrollRef.current

    if (el && stickRef.current && !query.trim()) {
      el.scrollTop = el.scrollHeight
    }
  }, [lines, query])

  const scrollToTop = () => {
    const el = scrollRef.current

    if (!el) {
      return
    }

    el.scrollTo({ top: 0 })
    stickRef.current = false
  }

  const scrollToBottom = () => {
    const el = scrollRef.current

    if (!el) {
      return
    }

    el.scrollTo({ top: el.scrollHeight })
    stickRef.current = true
  }

  return (
    <div className="group/logs relative h-full min-h-0">
      <div className="absolute right-2 top-1 z-10 flex items-center gap-0.5">
        <NavigationButton label={topLabel} onClick={scrollToTop}>
          <ArrowBarToUp />
        </NavigationButton>
        <NavigationButton label={bottomLabel} onClick={scrollToBottom}>
          <ArrowBarToDown />
        </NavigationButton>
        <CopyButton
          appearance="inline"
          className="opacity-20 transition-opacity group-hover/logs:opacity-100 focus-visible:opacity-100"
          showLabel={false}
          text={() => (lines ?? []).join('\n')}
        />
      </div>

      <LogView
        className="h-full min-h-0 [scrollbar-gutter:stable]"
        onScroll={event => {
          const el = event.currentTarget
          stickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24
        }}
        ref={scrollRef}
      >
        {lines === null || lines.length === 0 ? (
          <span className="text-muted-foreground/50">{lines === null ? '…' : emptyLabel}</span>
        ) : (
          lines.map((line, index) => {
            const severity = commandCenterLogSeverity(line)
            const presentation = severity ? SEVERITY_PRESENTATION[severity] : null
            const SeverityIcon = presentation?.icon

            return (
              <span
                className={cn(
                  'grid grid-cols-[1rem_minmax(0,1fr)] items-start gap-1.5 py-px',
                  line.startsWith('=====') && 'mt-1'
                )}
                key={index}
              >
                <span
                  aria-label={presentation?.label}
                  className={cn('mt-[0.12rem] inline-flex items-center justify-center', presentation?.iconClassName)}
                >
                  {SeverityIcon && <SeverityIcon className="size-3.5" />}
                </span>
                <span className="whitespace-pre-wrap break-words pr-16">
                  <HighlightedLine line={line.replace(/[\r\n]+$/, '')} query={query} />
                </span>
              </span>
            )
          })
        )}
      </LogView>
    </div>
  )
}
