import { useStore } from '@nanostores/react'
import type { CSSProperties, MutableRefObject, PointerEvent as ReactPointerEvent, RefObject } from 'react'
import { useEffect, useMemo, useRef } from 'react'

import { requestComposerInsert } from '@/app/chat/composer/focus'
import { CopyButton } from '@/components/ui/copy-button'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { PanelBottom, Send, Trash2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notify } from '@/store/notifications'

import type { ConsoleEntry, PreviewConsoleState } from './preview-console-state'

const consoleLevelLabel: Record<number, string> = {
  0: 'log',
  1: 'info',
  2: 'warn',
  3: 'error'
}

const consoleLevelClass: Record<number, string> = {
  0: 'text-foreground',
  1: 'text-sky-700 dark:text-sky-300',
  2: 'text-amber-700 dark:text-amber-300',
  3: 'text-destructive'
}

const CONSOLE_BOTTOM_THRESHOLD = 24
const CONSOLE_HEADER_HEIGHT = 32

export function compactUrl(value: string): string {
  try {
    const url = new URL(value)

    if (url.protocol === 'file:') {
      return decodeURIComponent(url.pathname)
    }

    return `${url.host}${url.pathname}${url.search}`
  } catch {
    return value
  }
}

export function formatLogLine(log: ConsoleEntry): string {
  const head = `[${consoleLevelLabel[log.level] || 'log'}]`
  const tail = log.source ? ` (${compactUrl(log.source)}${log.line ? `:${log.line}` : ''})` : ''

  return `${head} ${log.message}${tail}`.trim()
}

export function formatConsoleEntries(entries: ConsoleEntry[]): string {
  return entries.map(formatLogLine).join('\n')
}

export function isNearConsoleBottom(element: HTMLDivElement | null): boolean {
  if (!element) {
    return true
  }

  return element.scrollHeight - element.scrollTop - element.clientHeight <= CONSOLE_BOTTOM_THRESHOLD
}

export function clampConsoleHeight(value: number): number {
  return Math.max(value, CONSOLE_HEADER_HEIGHT)
}

interface ConsoleRowProps {
  copyText: string
  log: ConsoleEntry
  onSend: () => void
  onToggleSelect: () => void
  selected: boolean
}

function ConsoleRow({ copyText, log, onSend, onToggleSelect, selected }: ConsoleRowProps) {
  const { t } = useI18n()
  const copy = t.preview.console

  return (
    <div
      className={cn(
        'group/row grid grid-cols-[3.25rem_minmax(0,1fr)_auto] items-start gap-2 rounded-md border border-transparent px-1 py-1 transition-colors hover:bg-accent/40',
        selected && 'border-border/60 bg-accent/40'
      )}
    >
      <Tip label={selected ? copy.deselect : copy.select}>
        <button
          className={cn(
            'mt-0.5 text-left uppercase opacity-70 transition-colors hover:opacity-100',
            consoleLevelClass[log.level] ?? consoleLevelClass[0]
          )}
          onClick={onToggleSelect}
          type="button"
        >
          {consoleLevelLabel[log.level] || 'log'}
        </button>
      </Tip>
      <div className="min-w-0" data-selectable-text="true">
        <span className={cn('block wrap-break-word', consoleLevelClass[log.level] ?? consoleLevelClass[0])}>
          {log.message}
        </span>
        {log.source && (
          <span className="block truncate text-muted-foreground/60">
            {compactUrl(log.source)}
            {log.line ? `:${log.line}` : ''}
          </span>
        )}
      </div>
      <span className="opacity-0 transition-opacity group-hover/row:opacity-100">
        <CopyButton
          appearance="inline"
          className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
          errorMessage={copy.copyFailed}
          iconClassName="size-3"
          label={copy.copyEntry}
          showLabel={false}
          text={copyText}
        />
        <Tip label={copy.sendEntry}>
          <button
            className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
            onClick={onSend}
            type="button"
          >
            <Send className="size-3" />
          </button>
        </Tip>
      </span>
    </div>
  )
}

export function PreviewConsoleTitlebarIcon({ consoleState }: { consoleState: PreviewConsoleState }) {
  const { t } = useI18n()
  const logCount = useStore(consoleState.$logCount)

  return (
    <>
      <PanelBottom />
      {logCount > 0 && <span className="sr-only">{t.preview.console.messages(logCount)}</span>}
    </>
  )
}

interface PreviewConsolePanelProps {
  consoleBodyRef: RefObject<HTMLDivElement | null>
  consoleShouldStickRef: MutableRefObject<boolean>
  consoleState: PreviewConsoleState
  startConsoleResize: (event: ReactPointerEvent<HTMLDivElement>) => void
}

export function PreviewConsolePanel({
  consoleBodyRef,
  consoleShouldStickRef,
  consoleState,
  startConsoleResize
}: PreviewConsolePanelProps) {
  const { t } = useI18n()
  const copy = t.preview.console
  const consoleHeight = useStore(consoleState.$height)
  const logs = useStore(consoleState.$logs)
  const selectedLogIds = useStore(consoleState.$selectedLogIds)
  const visibleSelection = useMemo(() => logs.filter(log => selectedLogIds.has(log.id)), [logs, selectedLogIds])
  const sendableLogs = visibleSelection.length > 0 ? visibleSelection : logs
  const stickScrollRafRef = useRef<number | null>(null)

  useEffect(() => {
    if (!consoleShouldStickRef.current) {
      return
    }

    if (stickScrollRafRef.current !== null) {
      window.cancelAnimationFrame(stickScrollRafRef.current)
      stickScrollRafRef.current = null
    }

    stickScrollRafRef.current = window.requestAnimationFrame(() => {
      stickScrollRafRef.current = null
      const consoleBody = consoleBodyRef.current
      consoleBody?.scrollTo({ top: consoleBody.scrollHeight })
    })

    return () => {
      if (stickScrollRafRef.current !== null) {
        window.cancelAnimationFrame(stickScrollRafRef.current)
        stickScrollRafRef.current = null
      }
    }
  }, [consoleBodyRef, consoleHeight, consoleShouldStickRef, logs])

  function sendLogsToComposer(entries: ConsoleEntry[]) {
    if (!entries.length) {
      return
    }

    const block = [copy.promptHeader, '```', ...entries.map(formatLogLine), '```'].join('\n')

    requestComposerInsert(block, { mode: 'block', target: 'main' })
    consoleState.clearSelection()
    notify({
      kind: 'success',
      title: copy.sentTitle,
      message: copy.sentMessage(entries.length)
    })
  }

  return (
    <div
      className="pointer-events-auto absolute inset-x-0 bottom-0 z-20 flex h-(--preview-console-height) min-h-8 flex-col overflow-hidden border-t border-border/60 bg-background"
      style={{ '--preview-console-height': `${consoleHeight}px` } as CSSProperties}
    >
      <div
        aria-label={copy.resize}
        className="group absolute inset-x-0 -top-1 z-1 h-2 cursor-row-resize"
        onDoubleClick={() => consoleState.setHeight(CONSOLE_HEADER_HEIGHT)}
        onPointerDown={startConsoleResize}
        role="separator"
      >
        <span className="absolute left-1/2 top-1/2 h-0.75 w-23 -translate-x-1/2 -translate-y-1/2 rounded-full bg-muted-foreground/80 opacity-0 transition-opacity duration-100 group-hover:opacity-[0.5]" />
      </div>
      <div className="flex h-8 shrink-0 items-center justify-between border-b border-border/50 px-2">
        <div className="flex items-center gap-2 text-[0.6875rem] font-medium text-muted-foreground">
          <PanelBottom className="size-3.5" />
          {copy.title}
          {selectedLogIds.size > 0 && (
            <span className="rounded-full bg-muted px-1.5 py-px text-[0.5625rem] text-muted-foreground">
              {copy.selected(selectedLogIds.size)}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          <button
            className="inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[0.625rem] text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:opacity-40"
            disabled={sendableLogs.length === 0}
            onClick={() => sendLogsToComposer(sendableLogs)}
            type="button"
          >
            <Send className="size-3" />
            {copy.sendToChat}
          </button>
          <CopyButton
            appearance="inline"
            className="inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[0.625rem] text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:opacity-40"
            disabled={sendableLogs.length === 0}
            errorMessage={copy.copyFailed}
            iconClassName="size-3"
            label={visibleSelection.length > 0 ? copy.copySelected : copy.copyAll}
            text={() => formatConsoleEntries(sendableLogs)}
          >
            {copy.copy}
          </CopyButton>
          <button
            className="inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[0.625rem] text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:opacity-40"
            disabled={logs.length === 0}
            onClick={consoleState.clear}
            type="button"
          >
            <Trash2 className="size-3" />
            {copy.clear}
          </button>
        </div>
      </div>
      <div
        className="min-h-0 flex-1 overflow-y-auto px-2 py-1.5 font-mono text-[0.6875rem] leading-relaxed"
        ref={consoleBodyRef}
      >
        {logs.length > 0 ? (
          logs.map(log => {
            const selected = selectedLogIds.has(log.id)

            return (
              <ConsoleRow
                copyText={formatLogLine(log)}
                key={log.id}
                log={log}
                onSend={() => sendLogsToComposer([log])}
                onToggleSelect={() => consoleState.toggleSelection(log.id)}
                selected={selected}
              />
            )
          })
        ) : (
          <div className="py-2 text-muted-foreground/70">{copy.empty}</div>
        )}
      </div>
    </div>
  )
}
