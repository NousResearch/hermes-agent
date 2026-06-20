import { Fragment, memo, type ReactNode, useState } from 'react'

import { StatusRow } from '@/components/chat/status-row'
import { TerminalOutput } from '@/components/chat/terminal-output'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { Tip } from '@/components/ui/tooltip'
import { type Translations, useI18n } from '@/i18n'
import { ArrowUpRight, X } from '@/lib/icons'
import type { TodoStatus } from '@/lib/todos'
import { cn } from '@/lib/utils'
import { exitCodeLabel, type ComposerStatusItem } from '@/store/composer-status'

const toolLabel = (name: string) =>
  name
    .split('_')
    .filter(Boolean)
    .map(part => part[0]!.toUpperCase() + part.slice(1))
    .join(' ') || name

// Todo rows speak checkbox, not spinner-and-dot: a dashed ring while the item
// is still open (pending), codicons once it resolves, a live spinner only on
// the in-progress item.
const TODO_GLYPHS: Record<Exclude<TodoStatus, 'in_progress' | 'pending'>, { icon: string; tone: string }> = {
  cancelled: { icon: 'circle-slash', tone: 'text-muted-foreground/45' },
  completed: { icon: 'pass-filled', tone: 'text-emerald-500/80' }
}

// Left slot: braille spinner while running, otherwise a small status dot
// (green = done, red = failed) so the slot is always filled and rows align.
function leadingGlyph(item: ComposerStatusItem, s: Translations['statusStack']): ReactNode {
  if (item.todoStatus === 'pending') {
    return (
      <span
        aria-hidden
        className="box-border size-[0.7rem] rounded-full border border-dashed border-muted-foreground/60"
      />
    )
  }

  if (item.todoStatus && item.todoStatus !== 'in_progress') {
    const glyph = TODO_GLYPHS[item.todoStatus]

    return <Codicon className={glyph.tone} name={glyph.icon} size="0.8rem" />
  }

  if (item.state === 'running') {
    return (
      <GlyphSpinner
        ariaLabel={s.running}
        className="text-[0.9rem] leading-none text-muted-foreground/80"
        spinner="braille"
      />
    )
  }

  return (
    <span
      aria-hidden
      className={cn(
        'size-1.5 rounded-full',
        item.state === 'failed' ? 'bg-destructive/80' : item.state === 'stopped' ? 'bg-muted-foreground/60' : 'bg-emerald-500/70'
      )}
    />
  )
}

interface StatusItemRowProps {
  item: ComposerStatusItem
  /** Clear a finished background task from the stack. */
  onDismiss?: (id: string) => void
  /** Open the subagent's own session window, livestreamed by the gateway's
   *  child-session mirror (Agents view fallback for older gateways). */
  onOpen?: () => void
  /** Cancel a running background task. */
  onStop?: (id: string) => void
}

/**
 * Renders one {@link ComposerStatusItem} into the shared {@link StatusRow}.
 * Memoised + keyed by id so parent re-renders never remount it (the spinner
 * keeps ticking instead of resetting).
 */
export const StatusItemRow = memo(function StatusItemRow({ item, onDismiss, onOpen, onStop }: StatusItemRowProps) {
  const { t } = useI18n()
  const s = t.statusStack
  const [outputOpen, setOutputOpen] = useState(false)
  const failed = item.state === 'failed'
  const stopped = item.state === 'stopped'
  const running = item.state === 'running'

  const action =
    item.type === 'background'
      ? running
        ? onStop && { label: s.stop, onClick: () => onStop(item.id) }
        : onDismiss && { label: s.dismiss, onClick: () => onDismiss(item.id) }
      : null

  const canOpen = item.type === 'subagent' && !!onOpen
  const hasOutput = item.type === 'background' && !!item.output
  const onActivate = canOpen ? onOpen : hasOutput ? () => setOutputOpen(open => !open) : undefined

  return (
    <Fragment>
      <StatusRow
        leading={leadingGlyph(item, s)}
        onActivate={onActivate}
        trailing={
          action ? (
            <Tip label={action.label}>
              <Button
                aria-label={action.label}
                className="-my-1 size-4 rounded-md text-muted-foreground/60 hover:text-foreground/90"
                onClick={event => {
                  event.stopPropagation()
                  action.onClick()
                }}
                size="icon-xs"
                type="button"
                variant="ghost"
              >
                <X size={12} />
              </Button>
            </Tip>
          ) : canOpen ? (
            <ArrowUpRight aria-hidden className="size-3.5 text-muted-foreground/55" />
          ) : undefined
        }
      >
        <span
          className={cn(
            'min-w-0 max-w-[18rem] truncate text-[0.73rem] leading-4',
            failed
              ? 'text-destructive/90'
              : item.todoStatus && item.todoStatus !== 'in_progress'
                ? 'text-muted-foreground/75'
                : 'text-foreground/92'
          )}
        >
          {item.title}
        </span>
        {item.type === 'subagent' && item.currentTool && (
          <span className="shrink-0 truncate text-[0.62rem] leading-4 text-muted-foreground/70">
            {toolLabel(item.currentTool)}
          </span>
        )}
        {(failed || stopped) && typeof item.exitCode === 'number' && item.exitCode !== 0 && (() => {
          const lbl = exitCodeLabel(item.exitCode)
          return (
            <span className={cn(
              'shrink-0 rounded px-1 text-[0.58rem] font-semibold tabular-nums',
              stopped ? 'bg-muted-foreground/15 text-muted-foreground' : 'bg-destructive/15 text-destructive'
            )}>
              {lbl.signal ? lbl.text : s.exit(item.exitCode)}
            </span>
          )
        })()}
        {hasOutput && <DisclosureCaret className="shrink-0 text-muted-foreground/45" open={outputOpen} size="0.8em" />}
      </StatusRow>
      {hasOutput && outputOpen && <TerminalOutput className="mx-auto mb-1 max-w-[90%]" text={item.output!} />}
    </Fragment>
  )
})
