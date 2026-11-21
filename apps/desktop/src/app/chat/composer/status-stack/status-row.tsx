import { Fragment, memo, type MouseEvent as ReactMouseEvent, type ReactNode } from 'react'

import { openAgentTerminal } from '@/app/right-sidebar/terminal/terminals'
import { StatusPendingIcon } from '@/components/chat/status-pending-icon'
import { StatusRow } from '@/components/chat/status-row'
import { Codicon } from '@/components/ui/codicon'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { Tip } from '@/components/ui/tooltip'
import { type Translations, useI18n } from '@/i18n'
import { isDesktopFsRemoteMode } from '@/lib/desktop-fs'
import { localPreviewTarget, openPreviewTargetInBrowser } from '@/lib/local-preview'
import { capitalize } from '@/lib/text'
import type { TodoStatus } from '@/lib/todos'
import { cn } from '@/lib/utils'
import type { ComposerStatusItem } from '@/store/composer-status'
import { notifyError } from '@/store/notifications'
import { openPreview } from '@/store/preview'

const toolLabel = (name: string) => name.split('_').filter(Boolean).map(capitalize).join(' ') || name

// Todo rows speak checkbox, not spinner-and-dot: a dashed ring while the item
// is still open (pending), codicons once it resolves, a live spinner only on
// the in-progress item.
const TODO_GLYPHS: Record<Exclude<TodoStatus, 'in_progress' | 'pending'>, { icon: string; tone: string }> = {
  cancelled: { icon: 'circle-slash', tone: 'text-muted-foreground/45' },
  completed: { icon: 'pass-filled', tone: 'text-emerald-500/80' }
}

// Left slot: braille spinner while running, otherwise a small status dot
// (green = done, red = failed) so the slot is always filled and rows align.
function leadingGlyph(item: ComposerStatusItem, s: Translations['statusStack'], historical = false): ReactNode {
  if (item.type === 'goal') {
    if (item.goalStatus === 'paused') {
      return <Codicon className="text-muted-foreground/60" name="debug-pause" size="0.8rem" />
    }

    if (item.goalStatus === 'done') {
      return <Codicon className="text-emerald-500/80" name="pass-filled" size="0.8rem" />
    }

    return (
      <GlyphSpinner
        ariaLabel={s.running}
        className="text-[0.85rem] leading-none text-emerald-500/80"
        spinner="braille"
      />
    )
  }

  if (item.todoStatus === 'pending' || (historical && item.todoStatus === 'in_progress')) {
    return <StatusPendingIcon />
  }

  if (item.todoStatus && item.todoStatus !== 'in_progress') {
    const glyph = TODO_GLYPHS[item.todoStatus]

    return <Codicon className={glyph.tone} name={glyph.icon} size="0.8rem" />
  }

  if (item.state === 'running') {
    return (
      <GlyphSpinner
        ariaLabel={s.running}
        className="text-[0.85rem] leading-none text-muted-foreground/80"
        spinner="braille"
      />
    )
  }

  return (
    <span
      aria-hidden
      className={cn('size-1.5 rounded-full', item.state === 'failed' ? 'bg-destructive/80' : 'bg-emerald-500/70')}
    />
  )
}

const PORT_CHIP_CLASS = 'shrink-0 rounded bg-muted/70 px-1 text-[0.6rem] leading-4 text-muted-foreground tabular-nums'

/** A port the background process listens on. Locally it opens http://localhost:<port>
 *  (⌘/Ctrl-click: the in-app preview pane, like a detected preview link). A remote
 *  gateway's localhost is not this machine, so there the port is only shown. */
function PortChip({ port, s }: { port: number; s: Translations['statusStack'] }) {
  const { t } = useI18n()

  if (isDesktopFsRemoteMode()) {
    return <span className={PORT_CHIP_CLASS}>{s.port(port)}</span>
  }

  const url = `http://localhost:${port}`

  const open = (event: ReactMouseEvent) => {
    event.stopPropagation()
    const target = localPreviewTarget(url)

    if (!target) {
      return
    }

    if (event.metaKey || event.ctrlKey) {
      openPreview(target)

      return
    }

    openPreviewTargetInBrowser(target).catch(error => notifyError(error, t.preview.unavailable))
  }

  return (
    <Tip label={s.openPort(url)}>
      <button
        aria-label={s.openPort(url)}
        className={cn(PORT_CHIP_CLASS, 'cursor-pointer hover:bg-muted hover:text-foreground')}
        onClick={open}
        onKeyDown={event => event.stopPropagation()}
        type="button"
      >
        {s.port(port)}
      </button>
    </Tip>
  )
}

/** Right-hand badges: the subagent's / goal's current tool, a background process's
 *  listening ports, a failed exit code. */
function RowBadges({ item, s }: { item: ComposerStatusItem; s: Translations['statusStack'] }) {
  const tool =
    item.currentTool &&
    (item.type === 'subagent' ? toolLabel(item.currentTool) : item.type === 'goal' ? item.currentTool : '')

  const failedExit = item.state === 'failed' && typeof item.exitCode === 'number' && item.exitCode !== 0
  const ports = item.type === 'background' && item.state === 'running' ? (item.ports ?? []) : []

  return (
    <>
      {tool && <span className="shrink-0 truncate text-[0.62rem] leading-4 text-muted-foreground/70">{tool}</span>}
      {ports.map(port => (
        <PortChip key={port} port={port} s={s} />
      ))}
      {failedExit && (
        <span className="shrink-0 rounded bg-destructive/15 px-1 text-[0.58rem] font-semibold text-destructive tabular-nums">
          {s.exit(item.exitCode!)}
        </span>
      )}
    </>
  )
}

interface StatusItemRowProps {
  item: ComposerStatusItem
  /** Render a retained, non-running task snapshot. */
  historical?: boolean
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
export const StatusItemRow = memo(function StatusItemRow({
  historical = false,
  item,
  onDismiss,
  onOpen,
  onStop
}: StatusItemRowProps) {
  const { t } = useI18n()
  const s = t.statusStack
  const failed = item.state === 'failed'
  const running = item.state === 'running'

  const action =
    item.type === 'background'
      ? running
        ? onStop && { label: s.stop, onClick: () => onStop(item.id) }
        : onDismiss && { label: s.dismiss, onClick: () => onDismiss(item.id) }
      : null

  const canOpen = item.type === 'subagent' && !!onOpen

  // Background rows link to their read-only terminal tab; subagents open their session.
  const onActivate =
    item.type === 'background' ? () => openAgentTerminal(item.id, item.title) : canOpen ? onOpen : undefined

  return (
    <Fragment>
      <StatusRow
        depth={Math.min(item.depth ?? 0, 4)}
        dismiss={action ? { label: action.label, onDismiss: action.onClick } : undefined}
        leading={leadingGlyph(item, s, historical)}
        onActivate={onActivate}
        trailing={
          canOpen ? (
            <Codicon aria-hidden className="text-muted-foreground/55" name="link-external" size="0.85rem" />
          ) : undefined
        }
      >
        <span
          className={cn(
            'min-w-0 flex-1 truncate text-[0.73rem] leading-4',
            failed
              ? 'text-destructive/90'
              : item.todoStatus && item.todoStatus !== 'in_progress'
                ? 'text-muted-foreground/75'
                : 'text-foreground/92'
          )}
        >
          {item.title}
        </span>
        <RowBadges item={item} s={s} />
      </StatusRow>
    </Fragment>
  )
})
