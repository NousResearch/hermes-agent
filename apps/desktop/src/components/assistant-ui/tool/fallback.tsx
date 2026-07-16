'use client'

import { type ToolCallMessagePartProps, useAuiState } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import {
  Children,
  createContext,
  type FC,
  type PropsWithChildren,
  type ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState
} from 'react'

import { AnsiText } from '@/components/assistant-ui/ansi-text'
import { useElapsedSeconds } from '@/components/chat/activity-timer'
import { ActivityTimerText } from '@/components/chat/activity-timer-text'
import { CompactMarkdown } from '@/components/chat/compact-markdown'
import { FileDiffPanel } from '@/components/chat/diff-lines'
import { DisclosureRow } from '@/components/chat/disclosure-row'
import { ZoomableImage } from '@/components/chat/zoomable-image'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { CopyButton } from '@/components/ui/copy-button'
import { FadeText } from '@/components/ui/fade-text'
import { FileTypeIcon } from '@/components/ui/file-type-icon'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { ToolIcon } from '@/components/ui/tool-icon'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { PrettyLink, LinkifiedText as SharedLinkifiedText, urlSlugTitleLabel } from '@/lib/external-link'
import { AlertCircle, CheckCircle2 } from '@/lib/icons'
import { normalize } from '@/lib/text'
import { useEnterAnimation } from '@/lib/use-enter-animation'
import { cn } from '@/lib/utils'
import { recordPreviewArtifact } from '@/store/preview-status'
import { $activeSessionId, $currentCwd } from '@/store/session'
import { $toolInlineDiffs } from '@/store/tool-diffs'
import { $toolRowDismissed, dismissToolRow } from '@/store/tool-dismiss'
import { $toolDisclosureOpen, $toolViewMode, setToolDisclosureOpen } from '@/store/tool-view'

import { PendingToolApproval } from './approval'
import {
  buildToolView,
  clampForDisplay,
  cleanVisibleText,
  countDiffLineStats,
  inlineDiffFromResult,
  isFileEditTool,
  isPreviewableTarget,
  looksRedundant,
  type SearchResultRow,
  selectMessageRunning,
  stripInlineDiffChrome,
  toolCopyPayload,
  type ToolPart,
  toolPartDisclosureId,
  type ToolStatus,
  type ToolTitleAction
} from './fallback-model'

// `true` when a ToolEntry is rendered inside an embedding wrapper that owns
// the per-row chrome (timer / preview). The flat ToolGroupSlot sets this
// false, so every row currently owns its own chrome; kept as a seam for any
// future embedding surface.
const ToolEmbedContext = createContext(false)

// Shared header chrome for tool rows. Both the single-tool DisclosureRow
// and the multi-tool group header pass through these constants so a
// "Patch" row and a "Tool actions · 2 steps" row are visually identical.
const TOOL_HEADER_TITLE_CLASS =
  'text-[length:var(--conversation-tool-font-size)] font-medium leading-(--conversation-line-height) text-(--ui-text-secondary)'

const TOOL_HEADER_DURATION_CLASS = 'shrink-0 text-[0.625rem] tabular-nums text-(--ui-text-tertiary)'

const TOOL_HEADER_SUBTITLE_CLASS =
  'text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)'

const TOOL_HEADER_GLYPH_WRAP_CLASS = 'grid size-3.5 shrink-0 place-items-center self-center'

// Glass-style section label that sits above any pre/JSON/output block.
// Lowercase tracking + tiny size so it reads as a quiet field label rather
// than a chrome heading. Used for "stdout", "stderr", "Search results", etc.
const TOOL_SECTION_LABEL_CLASS = 'mb-1 text-[0.65rem] font-medium uppercase tracking-[0.08em] text-(--ui-text-tertiary)'

// Inset scroll surface for any detail body. The expanded tool row owns the
// border; the payload itself is just clipped raw text.
const TOOL_SECTION_SURFACE_CLASS =
  'max-h-20 max-w-full overflow-auto bg-transparent px-2 py-1.5 text-(--ui-text-secondary)'

const TOOL_EXPANDED_SHELL_CLASS = 'rounded-[0.3125rem] border border-(--ui-stroke-tertiary)'

const TOOL_SECTION_PRE_CLASS = cn(TOOL_SECTION_SURFACE_CLASS, 'font-mono text-[0.7rem] leading-relaxed')

interface ToolStatusCopy {
  statusDone: string
  statusError: string
  statusRecovered: string
  statusRunning: string
}

function rawTechnicalTrace(args: unknown, result: unknown): string {
  const parts = [args, result]
    .filter(value => value !== undefined && value !== null)
    .map(value => {
      if (typeof value === 'string') {
        return value
      }

      try {
        return JSON.stringify(value)
      } catch {
        return String(value)
      }
    })
    .filter(Boolean)

  return clampForDisplay(parts.join('\n'))
}

function statusGlyph(status: ToolStatus, copy: ToolStatusCopy): ReactNode {
  if (status === 'running') {
    return (
      <GlyphSpinner
        ariaLabel={copy.statusRunning}
        className="size-3.5 shrink-0 text-[0.95rem] text-(--ui-text-tertiary)"
        spinner="breathe"
      />
    )
  }

  if (status === 'error') {
    return <AlertCircle aria-label={copy.statusError} className="size-3.5 shrink-0 text-destructive" />
  }

  if (status === 'warning') {
    return (
      <AlertCircle aria-label={copy.statusRecovered} className="size-3.5 shrink-0 text-amber-600 dark:text-amber-400" />
    )
  }

  return (
    <CheckCircle2
      aria-label={copy.statusDone}
      className="size-3.5 shrink-0 text-emerald-600/85 dark:text-emerald-400/85"
    />
  )
}

// Leading glyph for any tool-row header. Status (running/error/warning)
// takes precedence; otherwise falls back to the tool's codicon. Returns
// null when neither applies so callers can render unconditionally.
function ToolGlyph({
  copy,
  filePath,
  icon,
  status
}: {
  copy: ToolStatusCopy
  filePath?: string
  icon?: string
  status?: ToolStatus
}) {
  const node = status ? (
    statusGlyph(status, copy)
  ) : filePath ? (
    <FileTypeIcon className="text-(--ui-text-tertiary)" path={filePath} size="0.875rem" />
  ) : icon ? (
    <ToolIcon className="text-(--ui-text-tertiary)" name={icon} size="0.875rem" />
  ) : null

  return node ? <span className={TOOL_HEADER_GLYPH_WRAP_CLASS}>{node}</span> : null
}

// Which status (if any) should pre-empt the tool's icon in the leading
// slot. Success is silent — the row reads as "done" without a checkmark.
function leadingStatus(isPending: boolean, status: ToolStatus): ToolStatus | undefined {
  if (isPending) {
    return 'running'
  }

  return status === 'success' ? undefined : status
}

function SearchResultsList({ hits }: { hits: SearchResultRow[] }) {
  return (
    <ol className="m-0 grid list-none gap-2.5 p-0">
      {hits.map((hit, index) => {
        const key = `${hit.url || hit.title}-${index}`
        const trimmedTitle = hit.title.trim()

        return (
          <li className="grid min-w-0 gap-0.5" key={key}>
            {hit.url ? (
              <PrettyLink
                className={cn(TOOL_HEADER_TITLE_CLASS, 'block max-w-full')}
                fallbackLabel={trimmedTitle || urlSlugTitleLabel(hit.url)}
                href={hit.url}
                label={trimmedTitle || undefined}
              />
            ) : (
              <span className={TOOL_HEADER_TITLE_CLASS}>{trimmedTitle}</span>
            )}
            {hit.snippet && <p className={cn(TOOL_HEADER_SUBTITLE_CLASS, 'm-0 line-clamp-3')}>{hit.snippet}</p>}
          </li>
        )
      })}
    </ol>
  )
}

function LinkifiedText({ className, text }: { className?: string; text: string }) {
  return <SharedLinkifiedText className={className} pretty text={cleanVisibleText(text)} />
}

function ToolTitle({
  isPending,
  status,
  title,
  titleAction
}: {
  isPending: boolean
  status: ToolStatus
  title: string
  titleAction?: ToolTitleAction
}) {
  return (
    <FadeText
      className={cn(
        TOOL_HEADER_TITLE_CLASS,
        isPending && 'text-(--ui-text-tertiary)',
        status === 'error' && 'text-destructive',
        status === 'warning' && 'text-amber-700 dark:text-amber-300'
      )}
    >
      {isPending && titleAction ? (
        <>
          {titleAction.prefix}
          <span className="shimmer">{titleAction.text}</span>
          {titleAction.suffix}
        </>
      ) : (
        title
      )}
    </FadeText>
  )
}

interface ToolEntryProps {
  part: ToolPart
}

function useDisclosureOpen(disclosureId: string, fallbackOpen = false): boolean {
  const persistedOpen = useStore($toolDisclosureOpen(disclosureId))

  return persistedOpen ?? fallbackOpen
}

function ToolEntry({ part }: ToolEntryProps) {
  const { t } = useI18n()
  const copy = t.assistant.tool
  const statusCopy = t.statusStack
  const messageId = useAuiState(s => s.message.id)
  const messageRunning = useAuiState(selectMessageRunning)
  const embedded = useContext(ToolEmbedContext)
  const toolViewMode = useStore($toolViewMode)

  // `ToolFallback` rebuilds the `part` wrapper each render, defeating the memos
  // below and re-running buildToolView (full JSON.stringify of result) on every
  // stream delta — the freeze on big `/learn` runs. Re-derive a stable part from
  // the referentially-stable args/result so the memos hold across deltas.
  const { args, isError, result, toolCallId, toolName } = part

  const stablePart = useMemo<ToolPart>(
    () => ({ args, isError, result, toolCallId, toolName, type: 'tool-call' }),
    [args, isError, result, toolCallId, toolName]
  )

  const disclosureId = `tool-entry:${messageId}:${toolPartDisclosureId(stablePart)}`
  const dismissed = useStore($toolRowDismissed(disclosureId))
  const isPending = messageRunning && result === undefined
  const liveDiffs = useStore($toolInlineDiffs)
  const sideDiff = toolCallId ? liveDiffs[toolCallId] || '' : ''
  const inlineDiff = stripInlineDiffChrome(sideDiff) || inlineDiffFromResult(result)
  const isFileEdit = isFileEditTool(toolName)
  const defaultOpen = Boolean(inlineDiff)
  const open = useDisclosureOpen(disclosureId, defaultOpen)
  const canDismiss = !isPending && !embedded
  // Only animate entries that mount while their message is actively
  // streaming — historical sessions mount with `messageRunning === false`,
  // so they paint statically without a settle cascade. The wrapping group
  // handles its own enter animation, so embedded children skip it.
  const enterRef = useEnterAnimation(messageRunning && !embedded, `tool-entry:${disclosureId}`)
  const elapsed = useElapsedSeconds(isPending, `tool:${disclosureId}`)

  // Stale parts (no result, but message stopped running) get a synthetic empty
  // result so buildToolView treats them as completed-no-output. Keyed on
  // stablePart so it recomputes only when this tool's data changes.
  const view = useMemo(() => {
    const p = !isPending && result === undefined ? { ...stablePart, result: {} } : stablePart

    return buildToolView(p, inlineDiff)
  }, [inlineDiff, isPending, result, stablePart])

  // Surface a previewable artifact (HTML file / localhost URL) as a compact link
  // in the composer status stack rather than a bulky inline card. Uses the same
  // detected target the old inline card did, keyed to the active session the
  // stack reads from. Idempotent + dedup'd, so re-renders don't churn.
  const activeSessionId = useStore($activeSessionId)
  const currentCwd = useStore($currentCwd)
  const previewTarget = view.previewTarget

  useEffect(() => {
    if (isPending || !activeSessionId || !previewTarget || !isPreviewableTarget(previewTarget)) {
      return
    }

    recordPreviewArtifact(activeSessionId, previewTarget, currentCwd || '')
  }, [activeSessionId, currentCwd, isPending, previewTarget])

  const detailSections = useMemo(() => {
    if (!view.detail) {
      return { body: '', summary: '' }
    }

    if (view.status !== 'error') {
      return { body: view.detail, summary: '' }
    }

    const chunks = view.detail
      .split(/\n\s*\n+/)
      .map(chunk => chunk.trim())
      .filter(Boolean)

    const [summary = '', ...rest] = chunks
    const subtitleNorm = normalize(view.subtitle)
    const summaryDuplicatesSubtitle = summary && summary.toLowerCase() === subtitleNorm

    if (summaryDuplicatesSubtitle) {
      return { body: rest.join('\n\n').trim(), summary: '' }
    }

    return { body: rest.join('\n\n').trim(), summary }
  }, [view.detail, view.status, view.subtitle])

  const detailMatchesSubtitle = looksRedundant(view.subtitle, view.detail)

  const showDetail =
    !view.inlineDiff &&
    ((view.status === 'error' && Boolean(detailSections.summary || detailSections.body)) ||
      (view.status !== 'error' &&
        Boolean(view.detail) &&
        !looksRedundant(view.title, view.detail) &&
        !detailMatchesSubtitle))

  const renderDetailAsCode =
    view.status !== 'error' &&
    (part.toolName === 'terminal' || part.toolName === 'execute_code' || part.toolName === 'read_file')

  const hasSearchHits = Boolean(view.searchHits?.length)
  const searchResultsLabel = part.toolName === 'web_search' ? 'Search results' : view.detailLabel

  const showRawSearchDrilldown =
    part.toolName === 'web_search' &&
    part.result !== undefined &&
    toolViewMode !== 'technical' &&
    Boolean(view.rawResult.trim())

  const hasExpandableContent = Boolean(
    view.imageUrl || view.inlineDiff || showDetail || hasSearchHits || toolViewMode === 'technical'
  )

  // copyAction reads the uncapped view.detail; clampForDisplay below only bounds
  // what's painted, so the row's Copy button still yields the full output.
  const copyAction = useMemo(() => toolCopyPayload(stablePart, view), [stablePart, view])

  const diffStats = useMemo(
    () => (isFileEdit && view.inlineDiff ? countDiffLineStats(view.inlineDiff) : null),
    [isFileEdit, view.inlineDiff]
  )

  const showDiffStats = !isPending && Boolean(diffStats && (diffStats.added > 0 || diffStats.removed > 0))

  // The header trailing slot only carries the live duration timer while the
  // tool is running. The copy control used to live here too, but an
  // `opacity-0` (yet still clickable) button straddling the caret/duration made
  // the disclosure caret hard to hit. Copy now lives in the expanded body's
  // top-right, where it can't fight the caret for the right edge.
  const trailing =
    isPending && !embedded ? <ActivityTimerText className={TOOL_HEADER_DURATION_CLASS} seconds={elapsed} /> : undefined

  // Once a turn has settled, a hover/focus-revealed dismiss lets the user clear
  // a completed/failed row that would otherwise sit at the tail of the chat.
  // It goes in the in-flow `action` slot (not `trailing`) so it can't overlap
  // the disclosure caret's hit-target — see the comment above `trailing`.
  const dismissAction = canDismiss ? (
    <Tip label={statusCopy.dismiss}>
      <Button
        aria-label={statusCopy.dismiss}
        className={cn(
          'size-5 rounded-md text-(--ui-text-tertiary) transition-opacity hover:text-(--ui-text-primary) hover:opacity-100',
          open
            ? 'opacity-80'
            : 'opacity-0 group-hover/disclosure-row:opacity-80 group-focus-within/disclosure-row:opacity-80'
        )}
        onClick={event => {
          event.stopPropagation()
          dismissToolRow(disclosureId)
        }}
        size="icon-xs"
        type="button"
        variant="ghost"
      >
        <Codicon name="close" size="0.75rem" />
      </Button>
    </Tip>
  ) : undefined

  if (dismissed) {
    return null
  }

  // A completed file edit with no diff to review is a bare, unexpandable row.
  // This is almost always a `write_file` create after a reload: only `patch`
  // persists its diff in the tool result, so creates rehydrate diff-less and
  // read like dead duplicates of the real diff row. Hide them — but keep
  // in-flight writes (activity) and failures (errors) visible.
  if (isFileEdit && !isPending && view.status !== 'error' && !view.inlineDiff) {
    return null
  }

  return (
    <div
      className={cn(
        'group/tool-block min-w-0 max-w-full overflow-hidden text-[length:var(--conversation-tool-font-size)] text-(--ui-text-tertiary)',
        open && TOOL_EXPANDED_SHELL_CLASS
      )}
      data-file-edit={isFileEdit && open ? '' : undefined}
      data-slot="tool-block"
      data-tool-open={open ? '' : undefined}
      data-tool-row=""
      ref={enterRef}
    >
      <div className={cn(open && 'border-b border-(--ui-stroke-tertiary) px-2 py-1.5')}>
        <DisclosureRow
          action={dismissAction}
          onToggle={hasExpandableContent ? () => setToolDisclosureOpen(disclosureId, !open) : undefined}
          open={open}
          trailing={trailing}
        >
          <span
            className="flex min-w-0 items-center gap-1.5"
            title={isFileEdit && view.subtitle ? view.subtitle : undefined}
          >
            <ToolGlyph
              copy={copy}
              filePath={isFileEdit ? view.subtitle : undefined}
              icon={view.icon}
              status={leadingStatus(isPending, view.status)}
            />
            <ToolTitle isPending={isPending} status={view.status} title={view.title} titleAction={view.titleAction} />
            {!isPending && view.countLabel && <span className={TOOL_HEADER_DURATION_CLASS}>{view.countLabel}</span>}
            {showDiffStats && diffStats && (
              <span className="flex shrink-0 items-center gap-1 font-mono text-[0.625rem] tabular-nums">
                {diffStats.added > 0 && (
                  <span className="text-emerald-600 dark:text-emerald-400">+{diffStats.added}</span>
                )}
                {diffStats.removed > 0 && (
                  <span className="text-rose-600 dark:text-rose-400">−{diffStats.removed}</span>
                )}
              </span>
            )}
            {!isFileEdit && !isPending && view.durationLabel && (
              <span className={TOOL_HEADER_DURATION_CLASS}>{view.durationLabel}</span>
            )}
          </span>
        </DisclosureRow>
      </div>
      {isPending && <PendingToolApproval part={part} />}
      {open && (
        <div className="relative grid w-full min-w-0 max-w-full gap-1.5 overflow-hidden p-1.5">
          {copyAction.text && (
            <CopyButton
              appearance="inline"
              className="absolute right-4 top-1.5 z-10 h-5 gap-0 rounded-md px-1 opacity-5 transition-opacity group-hover/tool-block:opacity-100 hover:opacity-100 focus-visible:opacity-100"
              iconClassName="size-3"
              label={copyAction.label}
              showLabel={false}
              side="left"
              stopPropagation
              text={copyAction.text}
            />
          )}
          {view.imageUrl && (
            <div className="max-w-72 overflow-hidden rounded-[0.25rem] border border-(--ui-stroke-tertiary)">
              <ZoomableImage alt={copy.outputAlt} className="h-auto w-full object-cover" src={view.imageUrl} />
            </div>
          )}
          {hasSearchHits && view.searchHits && (
            <div className="max-w-full text-xs leading-relaxed text-(--ui-text-secondary)">
              {searchResultsLabel && <p className={TOOL_SECTION_LABEL_CLASS}>{searchResultsLabel}</p>}
              <SearchResultsList hits={view.searchHits} />
            </div>
          )}
          {view.inlineDiff && (
            <FileDiffPanel className="-mt-1.5" diff={view.inlineDiff} path={isFileEdit ? view.subtitle : undefined} />
          )}
          {showDetail &&
            toolViewMode !== 'technical' &&
            (view.status === 'error' ? (
              detailSections.summary || detailSections.body ? (
                <div className="max-w-full text-xs leading-relaxed text-destructive">
                  {detailSections.summary && (
                    <LinkifiedText className="block font-medium" text={detailSections.summary} />
                  )}
                  {detailSections.body && (
                    <pre
                      className={cn(
                        'max-h-56 overflow-auto whitespace-pre-wrap wrap-anywhere font-mono text-[0.7rem] leading-[1.55] text-destructive/90',
                        detailSections.summary && 'mt-1.5'
                      )}
                    >
                      {clampForDisplay(detailSections.body)}
                    </pre>
                  )}
                </div>
              ) : null
            ) : view.stdout || view.stderr ? (
              // Stdout + stderr split: render both as labeled blocks. stderr
              // is intentionally NOT painted destructive — many CLIs log
              // informational output there.
              <div className="max-w-full text-xs leading-relaxed text-(--ui-text-secondary)">
                {view.detailLabel && <p className={TOOL_SECTION_LABEL_CLASS}>{view.detailLabel}</p>}
                {view.stdout && (
                  <div className="space-y-0.5">
                    {view.stderr && <p className={TOOL_SECTION_LABEL_CLASS}>stdout</p>}
                    <pre className={cn(TOOL_SECTION_PRE_CLASS, 'whitespace-pre-wrap wrap-anywhere')}>
                      {view.rendersAnsi ? (
                        <AnsiText text={clampForDisplay(view.stdout)} />
                      ) : (
                        clampForDisplay(view.stdout)
                      )}
                    </pre>
                  </div>
                )}
                {view.stderr && (
                  <div className={cn('space-y-0.5', view.stdout && 'mt-1.5')}>
                    <p className={TOOL_SECTION_LABEL_CLASS}>stderr</p>
                    <pre
                      className={cn(
                        TOOL_SECTION_PRE_CLASS,
                        'whitespace-pre-wrap wrap-anywhere text-(--ui-text-tertiary)'
                      )}
                    >
                      {view.rendersAnsi ? (
                        <AnsiText text={clampForDisplay(view.stderr)} />
                      ) : (
                        clampForDisplay(view.stderr)
                      )}
                    </pre>
                  </div>
                )}
              </div>
            ) : (
              <div className="max-w-full text-xs leading-relaxed text-(--ui-text-secondary)">
                {view.detailLabel && <p className={TOOL_SECTION_LABEL_CLASS}>{view.detailLabel}</p>}
                {renderDetailAsCode ? (
                  <pre className={cn(TOOL_SECTION_PRE_CLASS, 'whitespace-pre-wrap wrap-anywhere')}>
                    {view.rendersAnsi ? <AnsiText text={clampForDisplay(view.detail)} /> : clampForDisplay(view.detail)}
                  </pre>
                ) : (
                  <CompactMarkdown
                    className={cn(TOOL_SECTION_SURFACE_CLASS, 'wrap-anywhere')}
                    text={clampForDisplay(view.detail)}
                  />
                )}
              </div>
            ))}
          {showRawSearchDrilldown && (
            <details className="max-w-full">
              <summary className={cn(TOOL_SECTION_LABEL_CLASS, 'mb-0')}>{copy.rawResponse}</summary>
              <pre className={cn(TOOL_SECTION_PRE_CLASS, 'mt-1 whitespace-pre-wrap wrap-anywhere')}>
                {view.rawResult}
              </pre>
            </details>
          )}
          {toolViewMode === 'technical' && !(isFileEdit && view.inlineDiff) && (
            <pre className={cn(TOOL_SECTION_PRE_CLASS, 'whitespace-pre-wrap wrap-anywhere')}>
              {rawTechnicalTrace(part.args, part.result)}
            </pre>
          )}
          {toolViewMode === 'technical' && isFileEdit && view.inlineDiff && (
            <details className="max-w-full">
              <summary className={cn(TOOL_SECTION_LABEL_CLASS, 'mb-0 cursor-pointer')}>Tool payload</summary>
              <pre className={cn(TOOL_SECTION_PRE_CLASS, 'mt-1 whitespace-pre-wrap wrap-anywhere')}>
                {rawTechnicalTrace(part.args, part.result)}
              </pre>
            </details>
          )}
        </div>
      )}
    </div>
  )
}

// A back-to-back run of this many tool calls collapses into the bounded,
// auto-scrolling window; fewer than this stays a plain inline stack.
const TOOL_GROUP_SCROLL_THRESHOLD = 3

// Tools whose body (an interactive form, a full-size image) must never be
// trapped behind the window's max-height + fade mask. A run holding any of
// them stays a plain, fully-visible stack no matter how long it is.
export const UNBOUNDABLE_TOOLS = new Set(['clarify', 'image_generate'])

export function shouldBoundToolGroup(childCount: number, hasUnboundable: boolean) {
  return childCount >= TOOL_GROUP_SCROLL_THRESHOLD && !hasUnboundable
}

// Pin-to-bottom + top-fade for the bounded tool window. Pins the newest row on
// growth (a call lands or a row expands) unless the user scrolled up, and fades
// the top edge once anything sits above it. Mirrors ThinkingDisclosure's live
// preview. `enabled` is false for short runs, leaving the plain flat stack.
function useToolWindow(enabled: boolean) {
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const contentRef = useRef<HTMLDivElement | null>(null)
  const stickRef = useRef(true)
  const [faded, setFaded] = useState(false)

  const syncFade = useCallback(() => setFaded((scrollRef.current?.scrollTop ?? 0) > 4), [])

  const onScroll = useCallback(() => {
    const el = scrollRef.current

    if (!el) {
      return
    }

    stickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight <= 8
    syncFade()
  }, [syncFade])

  useEffect(() => {
    const el = scrollRef.current
    const content = contentRef.current

    if (!enabled || !el || !content) {
      return
    }

    const pin = () => {
      if (stickRef.current) {
        el.scrollTop = el.scrollHeight
      }

      syncFade()
    }

    pin()
    const observer = new ResizeObserver(pin)
    observer.observe(content)

    return () => observer.disconnect()
  }, [enabled, syncFade])

  return { contentRef, faded, onScroll, scrollRef }
}

/**
 * Flat, Cursor-style tool list. assistant-ui hands us a *range* of
 * consecutive tool-call parts, but how that range is sliced is unstable: a
 * live stream interleaves narration/reasoning between calls (many tiny
 * ranges), while the settled message reconstructs every tool_call back-to-back
 * (one big range). Rendering a "Tool actions · N steps" group off that range
 * therefore reshuffled the whole turn the instant it settled.
 *
 * So we still never *label* the group: each tool is a standalone row on the
 * tight `--tool-row-gap` rhythm. Once a run reaches `TOOL_GROUP_SCROLL_THRESHOLD`
 * rows it collapses into a fixed-height, auto-scrolling window so a long run
 * doesn't shove the reply off screen; shorter runs are byte-identical to before.
 * The DOM shape is the same either way — only classes flip — so a run that
 * crosses the threshold mid-stream never remounts a row. `ToolEmbedContext` is
 * false so every row owns its own chrome (timer / preview / copy / approval).
 */
export const ToolGroupSlot: FC<PropsWithChildren<{ endIndex: number; startIndex: number }>> = ({
  children,
  endIndex,
  startIndex
}) => {
  const messageId = useAuiState(s => s.message.id)
  const messageRunning = useAuiState(selectMessageRunning)

  const hasUnboundable = useAuiState(s =>
    s.message.parts
      .slice(Math.max(0, startIndex), endIndex + 1)
      .some(part => part.type === 'tool-call' && UNBOUNDABLE_TOOLS.has(part.toolName))
  )

  const enterRef = useEnterAnimation(messageRunning, `tool-group:${messageId}:${startIndex}`)

  const bounded = shouldBoundToolGroup(Children.count(children), hasUnboundable)
  const { contentRef, faded, onScroll, scrollRef } = useToolWindow(bounded)

  return (
    <ToolEmbedContext.Provider value={false}>
      <div className="min-w-0 max-w-full overflow-hidden" data-slot="tool-block" data-tool-group="" ref={enterRef}>
        <div
          className={cn(
            bounded && 'tool-group-scroll max-h-(--tool-group-scroll-max-h) overflow-y-auto',
            bounded && faded && 'tool-group-scroll--faded'
          )}
          onScroll={bounded ? onScroll : undefined}
          ref={scrollRef}
        >
          <div className="grid min-w-0 max-w-full gap-(--tool-row-gap)" ref={contentRef}>
            {children}
          </div>
        </div>
      </div>
    </ToolEmbedContext.Provider>
  )
}

/**
 * Per-tool fallback. Now strictly returns a single ToolEntry — the
 * grouping decision lives in ToolGroupSlot above, so this never swaps
 * its return type and the underlying ToolEntry stays mounted across
 * group-shape changes.
 */
export const ToolFallback = ({ toolCallId, toolName, args, isError, result }: ToolCallMessagePartProps) => {
  const part: ToolPart = { args, isError, result, toolCallId, toolName, type: 'tool-call' }

  return <ToolEntry part={part} />
}
