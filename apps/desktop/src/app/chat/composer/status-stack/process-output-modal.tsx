import { useStore } from '@nanostores/react'
import { useMemo } from 'react'

import { TerminalOutput } from '@/components/chat/terminal-output'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $backgroundStatusBySession, type ComposerStatusItem } from '@/store/composer-status'
import { $activeSessionId } from '@/store/session'

import { OverlayView } from '../../../overlays/overlay-view'

/**
 * Roomy, live-tailing viewer for a single background task's output — opened by
 * clicking a *running* job row in the status stack. Unlike the cramped inline
 * disclosure, this gives a full-height terminal view, and stays live because it
 * re-reads the task from the store (refreshed every 5s by the process.list
 * poll) by id rather than capturing a snapshot.
 */
export function ProcessOutputModal({ itemId, onClose }: { itemId: string; onClose: () => void }) {
  const { t } = useI18n()
  const s = t.statusStack
  const sessionId = useStore($activeSessionId) ?? ''
  const bySession = useStore($backgroundStatusBySession)

  const item: ComposerStatusItem | undefined = useMemo(
    () => (bySession[sessionId] ?? []).find(row => row.id === itemId),
    [bySession, sessionId, itemId]
  )

  // The task vanished from the registry (dismissed / reaped) — close out.
  if (!item) {
    return null
  }

  const running = item.state === 'running'
  const failed = item.state === 'failed'

  return (
    <OverlayView
      closeLabel={t.common?.close ?? 'Close'}
      contentClassName="flex min-h-0 flex-1 flex-col px-4 pt-4 pb-3 sm:px-5"
      onClose={onClose}
      portal
      rootClassName="mx-auto flex h-[80vh] max-h-[80vh] w-full max-w-3xl flex-col"
    >
      <header className="mb-2 flex shrink-0 items-center gap-2">
        <Codicon
          className={cn(
            'shrink-0',
            running ? 'text-emerald-500' : failed ? 'text-destructive' : 'text-muted-foreground/70'
          )}
          name={running ? 'pulse' : failed ? 'error' : 'check'}
          size="0.95rem"
          spinning={running}
        />
        <span className="min-w-0 flex-1 truncate font-mono text-[0.8rem] text-foreground" title={item.title}>
          {item.title}
        </span>
        {typeof item.exitCode === 'number' && (
          <span
            className={cn(
              'shrink-0 rounded px-1.5 text-[0.62rem] font-semibold tabular-nums',
              item.exitCode === 0
                ? 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400'
                : 'bg-destructive/15 text-destructive'
            )}
          >
            {s.exit(item.exitCode)}
          </span>
        )}
        <span className="shrink-0 text-[0.62rem] text-muted-foreground/65">
          {running ? s.running : failed ? s.failed : s.done}
        </span>
      </header>

      <TerminalOutput
        className="min-h-0 flex-1 !max-h-none rounded-md border border-(--ui-stroke-tertiary)"
        text={item.output || (running ? s.waitingForOutput : s.noOutput)}
      />
    </OverlayView>
  )
}
