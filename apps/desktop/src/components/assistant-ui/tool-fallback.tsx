'use client'

import { type ToolCallMessagePartProps, useAuiState } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { createContext, type FC, type PropsWithChildren, type ReactNode, useContext, useMemo } from 'react'
import { useShallow } from 'zustand/shallow'

import { useElapsedSeconds } from '@/components/chat/activity-timer'
import { ActivityTimerText } from '@/components/chat/activity-timer-text'
import { CompactMarkdown } from '@/components/chat/compact-markdown'
import { DiffLines } from '@/components/chat/diff-lines'
import { DisclosureRow } from '@/components/chat/disclosure-row'
import { PreviewAttachment } from '@/components/chat/preview-attachment'
import { ZoomableImage } from '@/components/chat/zoomable-image'
import { BrailleSpinner } from '@/components/ui/braille-spinner'
import { Codicon } from '@/components/ui/codicon'
import { CopyButton } from '@/components/ui/copy-button'
import { FadeText } from '@/components/ui/fade-text'
import { PrettyLink, LinkifiedText as SharedLinkifiedText, urlSlugTitleLabel } from '@/lib/external-link'
import { AlertCircle, CheckCircle2 } from '@/lib/icons'
import { useEnterAnimation } from '@/lib/use-enter-animation'
import { cn } from '@/lib/utils'
import { $toolInlineDiffs } from '@/store/tool-diffs'
import { $toolDisclosureOpen, $toolViewMode, setToolDisclosureOpen } from '@/store/tool-view'

import {
  groupCopyText as buildGroupCopyText,
  buildToolView,
  cleanVisibleText,
  groupFailedStepCount,
  groupPreviewTargets,
  groupStatus,
  groupTitle,
  groupTotalDurationLabel,
  inlineDiffFromResult,
  isPreviewableTarget,
  looksRedundant,
  type SearchResultRow,
  selectMessageRunning,
  stripInlineDiffChrome,
  toolCopyPayload,
  type ToolPart,
  toolPartDisclosureId,
  type ToolStatus
} from './tool-fallback-model'

// Tool names that ChainToolFallback intercepts and renders as something
// other than a ToolEntry — they don't count toward "is this a group of
// tool calls?" because they have no visible tool block.
const SPECIAL_TOOL_NAMES = new Set(['todo', 'image_generate', 'clarify'])

// `true` when the current ToolEntry is being rendered inside a group
// wrapper. Lets ToolEntry suppress per-row chrome (timer / preview) that
// the group already shows.
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
// than a chrome heading. Used for "COMMAND OUTPUT", "INPUT", "OUTPUT", etc.
const TOOL_SECTION_LABEL_CLASS = 'mb-1 text-[0.65rem] font-medium uppercase tracking-[0.08em] text-(--ui-text-tertiary)'

// Inset scroll surface for any detail body. The expanded tool row owns the
// border; the payload itself is just clipped raw text.
const TOOL_SECTION_SURFACE_CLASS =
  'max-h-20 max-w-full overflow-auto bg-transparent px-2 py-1.5 text-(--ui-text-secondary)'

const TOOL_SECTION_PRE_CLASS = cn(TOOL_SECTION_SURFACE_CLASS, 'font-mono text-[0.7rem] leading-relaxed')

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

  return parts.join('\n')
}

function statusGlyph(status: ToolStatus): ReactNode {
  if (status === 'running') {
    return (
      <BrailleSpinner
        ariaLabel="Running"
        className="size-3.5 shrink-0 text-[0.95rem] text-(--ui-text-tertiary)"
        spinner="breathe"
      />
    )
  }

  if (status === 'error') {
    return <AlertCircle aria-label="Error" className="size-3.5 shrink-0 text-destructive" />
  }

  if (status === 'warning') {
    return <AlertCircle aria-label="Recovered" className="size-3.5 shrink-0 text-amber-600 dark:text-amber-400" />
  }

  return <CheckCircle2 aria-label="Done" className="size-3.5 shrink-0 text-emerald-600/85 dark:text-emerald-400/85" />
}

// Leading glyph for any tool-row header. Status (running/error/warning)
// takes precedence; otherwise falls back to the tool's codicon. Returns
// null when neither applies so callers can render unconditionally.
function ToolGlyph({ icon, status }: { icon?: string; status?: ToolStatus }) {
  const node = status ? (
    statusGlyph(status)
  ) : icon ? (
    <Codicon className="text-(--ui-text-tertiary)" name={icon} size="0.875rem" />
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

interface ToolEntryProps {
  part: ToolPart
}

function useDisclosureOpen(disclosureId: string, fallbackOpen = false): boolean {
  const persistedOpen = useStore($toolDisclosureOpen(disclosureId))

  return persistedOpen ?? fallbackOpen
}

function ToolEntry({ part }: ToolEntryProps) {
  const messageId = useAuiState(s => s.message.id)
  const messageRunning = useAuiState(selectMessageRunning)
  const embedded = useContext(ToolEmbedContext)
  const toolViewMode = useStore($toolViewMode)
  const disclosureId = `tool-entry:${messageId}:${toolPartDisclosureId(part)}`
  const open = useDisclosureOpen(disclosureId)
  const isPending = messageRunning && part.result === undefined
  // Only animate entries that mount while their message is actively
  // streaming — historical sessions mount with `messageRunning === false`,
  // so they paint statically without a settle cascade. The wrapping group
  // handles its own enter animation, so embedded children skip it.
  const enterRef = useEnterAnimation(messageRunning && !embedded, `tool-entry:${disclosureId}`)
  const elapsed = useElapsedSeconds(isPending, `tool:${disclosureId}`)
  const liveDiffs = useStore($toolInlineDiffs)
  const sideDiff = part.toolCallId ? liveDiffs[part.toolCallId] || '' : ''
  const inlineDiff = stripInlineDiffChrome(sideDiff) || inlineDiffFromResult(part.result)

  // Stale parts (no result, but message stopped running) get a synthetic
  // empty result so buildToolView treats them as completed-no-output.
  const view = useMemo(() => {
    const p = !isPending && part.result === undefined ? { ...part, result: {} } : part

    return buildToolView(p, inlineDiff)
  }, [inlineDiff, isPending, part])

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
    const subtitleNorm = view.subtitle.trim().toLowerCase()
    const summaryDuplicatesSubtitle = summary && summary.toLowerCase() === subtitleNorm

    if (summaryDuplicatesSubtitle) {
      return { body: rest.join('\n\n').trim(), summary: '' }
    }

    return { body: rest.join('\n\n').trim(), summary }
  }, [view.detail, view.status, view.subtitle])

  const detailMatchesSubtitle = looksRedundant(view.subtitle, view.detail)

  const showDetail =
    (view.status === 'error' && Boolean(detailSections.summary || detailSections.body)) ||
    (view.status !== 'error' &&
      Boolean(view.detail) &&
      !looksRedundant(view.title, view.detail) &&
      !detailMatchesSubtitle)

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
    (view.previewTarget && isPreviewableTarget(view.previewTarget)) ||
    view.imageUrl ||
    showDetail ||
    hasSearchHits ||
    toolViewMode === 'technical'
  )

  const copyAction = useMemo(() => toolCopyPayload(part, view), [part, view])

  const trailing =
    isPending && !embedded ? (
      <ActivityTimerText className={TOOL_HEADER_DURATION_CLASS} seconds={elapsed} />
    ) : !isPending && copyAction.text ? (
      <CopyButton appearance="tool-row" label={copyAction.label} stopPropagation text={copyAction.text} />
    ) : undefined

  return (
    <div
      className={cn(
        'min-w-0 max-w-full overflow-hidden text-[length:var(--conversation-tool-font-size)] text-(--ui-text-tertiary)',
        open && 'rounded-[0.625rem] border border-(--ui-stroke-tertiary)'
      )}
      data-slot="tool-block"
      ref={enterRef}
    >
      <div className={cn(open && 'border-b border-(--ui-stroke-tertiary) px-2 py-1.5')}>
        <DisclosureRow
          onToggle={hasExpandableContent ? () => setToolDisclosureOpen(disclosureId, !open) : undefined}
          open={open}
          trailing={trailing}
        >
          <span className="flex min-w-0 items-center gap-1.5">
            <ToolGlyph icon={view.icon} status={leadingStatus(isPending, view.status)} />
            <FadeText
              className={cn(
                TOOL_HEADER_TITLE_CLASS,
                isPending && 'shimmer text-(--ui-text-tertiary)',
                view.status === 'error' && 'text-destructive',
                view.status === 'warning' && 'text-amber-700 dark:text-amber-300'
              )}
            >
              {view.title}
            </FadeText>
            {!isPending && view.countLabel && <span className={TOOL_HEADER_DURATION_CLASS}>{view.countLabel}</span>}
            {!isPending && view.durationLabel && (
              <span className={TOOL_HEADER_DURATION_CLASS}>{view.durationLabel}</span>
            )}
          </span>
        </DisclosureRow>
      </div>
      {open && (
        <div className="grid w-full min-w-0 max-w-full gap-1.5 overflow-hidden p-1.5">
          {!embedded && view.previewTarget && isPreviewableTarget(view.previewTarget) && (
            <PreviewAttachment source="tool-result" target={view.previewTarget} />
          )}
          {view.imageUrl && (
            <div className="max-w-72 overflow-hidden rounded-[0.25rem] border border-(--ui-stroke-tertiary)">
              <ZoomableImage alt="Tool output" className="h-auto w-full object-cover" src={view.imageUrl} />
            </div>
          )}
          {hasSearchHits && view.searchHits && (
            <div className="max-w-full text-xs leading-relaxed text-(--ui-text-secondary)">
              {searchResultsLabel && <p className={TOOL_SECTION_LABEL_CLASS}>{searchResultsLabel}</p>}
              <SearchResultsList hits={view.searchHits} />
            </div>
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
                      {detailSections.body}
                    </pre>
                  )}
                </div>
              ) : null
            ) : (
              <div className="max-w-full text-xs leading-relaxed text-(--ui-text-secondary)">
                {view.detailLabel && <p className={TOOL_SECTION_LABEL_CLASS}>{view.detailLabel}</p>}
                {renderDetailAsCode ? (
                  <pre className={cn(TOOL_SECTION_PRE_CLASS, 'whitespace-pre-wrap wrap-anywhere')}>{view.detail}</pre>
                ) : (
                  <CompactMarkdown className={cn(TOOL_SECTION_SURFACE_CLASS, 'wrap-anywhere')} text={view.detail} />
                )}
              </div>
            ))}
          {showRawSearchDrilldown && (
            <details className="max-w-full">
              <summary className={cn(TOOL_SECTION_LABEL_CLASS, 'cursor-pointer mb-0')}>Raw response</summary>
              <pre className={cn(TOOL_SECTION_PRE_CLASS, 'mt-1 whitespace-pre-wrap wrap-anywhere')}>
                {view.rawResult}
              </pre>
            </details>
          )}
          {toolViewMode === 'technical' && (
            <pre className={cn(TOOL_SECTION_PRE_CLASS, 'whitespace-pre-wrap wrap-anywhere')}>
              {rawTechnicalTrace(part.args, part.result)}
            </pre>
          )}
        </div>
      )}
      {view.inlineDiff && <DiffLines text={view.inlineDiff} />}
    </div>
  )
}

/**
 * Always-present wrapper around the consecutive tool-call range that
 * `MessagePrimitive.Parts` already grouped for us. Renders a header +
 * collapsible body when there are 2+ visible tools; otherwise it's a
 * transparent passthrough that just owns the entry animation for the
 * single ToolEntry inside.
 *
 * Crucially, the wrapper element is the SAME `<div>` regardless of
 * group size — only the optional header element appears/disappears.
 * That preserves React identity for the inner `MessagePartByIndex`
 * children when the 1→2 transition happens, so existing tool blocks
 * never remount when a new tool joins them mid-stream.
 *
 * The previous design (per-tool ToolFallback computing its own group
 * lookup and conditionally returning either `<ToolEntry>` or
 * `<ToolGroup>`) flipped the React element type at the 1→2 transition
 * and tore down the existing tool entirely, which is what showed up as
 * "the previous tool's animation resets every time a new tool arrives."
 */
export const ToolGroupSlot: FC<PropsWithChildren<{ endIndex: number; startIndex: number }>> = ({
  children,
  endIndex,
  startIndex
}) => {
  const messageId = useAuiState(s => s.message.id)
  const messageRunning = useAuiState(selectMessageRunning)

  // Pull the visible tool parts in this range. `useShallow` makes this
  // re-render only when the actual part references change (assistant-ui
  // gives stable refs for unchanged parts), not on every text/reasoning
  // delta elsewhere in the message.
  const visibleParts = useAuiState(
    useShallow((s: { message: { parts: readonly unknown[] } }) =>
      s.message.parts.slice(startIndex, endIndex + 1).filter((p): p is ToolPart => {
        if (!p || typeof p !== 'object') {
          return false
        }

        const row = p as { toolName?: unknown; type?: unknown }

        return row.type === 'tool-call' && typeof row.toolName === 'string' && !SPECIAL_TOOL_NAMES.has(row.toolName)
      })
    )
  )

  const isGroup = visibleParts.length > 1
  const isRunning = messageRunning && visibleParts.some(p => p.result === undefined)
  // Stable across the group's lifetime (start index doesn't shift when
  // tools append to the end), so user-driven open/close persists across
  // streaming.
  const disclosureId = `tool-group:${messageId}:${startIndex}`
  const open = useDisclosureOpen(disclosureId)
  const enterRef = useEnterAnimation(messageRunning, disclosureId)

  const status = groupStatus(visibleParts)
  const displayStatus = !isRunning && status === 'running' ? 'success' : status
  const failedStepCount = useMemo(() => groupFailedStepCount(visibleParts), [visibleParts])
  const totalDurationLabel = useMemo(() => groupTotalDurationLabel(visibleParts), [visibleParts])

  const statusSummary =
    displayStatus === 'running' || failedStepCount === 0
      ? ''
      : displayStatus === 'warning'
        ? failedStepCount === 1
          ? 'Recovered after 1 failed step'
          : `Recovered after ${failedStepCount} failed steps`
        : failedStepCount === 1
          ? '1 step failed'
          : `${failedStepCount} steps failed`

  const groupCopyText = useMemo(() => buildGroupCopyText(visibleParts), [visibleParts])
  const previewTargets = useMemo(() => groupPreviewTargets(visibleParts), [visibleParts])

  return (
    <ToolEmbedContext.Provider value={isGroup}>
      <div className="min-w-0 max-w-full overflow-hidden" data-slot="tool-block" ref={enterRef}>
        {isGroup && (
          <DisclosureRow
            key="header"
            onToggle={() => setToolDisclosureOpen(disclosureId, !open)}
            open={open}
            trailing={
              !isRunning && groupCopyText ? (
                <CopyButton appearance="tool-row" label="Copy activity" stopPropagation text={groupCopyText} />
              ) : undefined
            }
          >
            <span className="flex min-w-0 items-center gap-1.5">
              <ToolGlyph status={displayStatus === 'success' ? undefined : displayStatus} />
              <FadeText
                className={cn(
                  TOOL_HEADER_TITLE_CLASS,
                  displayStatus === 'error' && 'text-destructive',
                  displayStatus === 'warning' && 'text-amber-700 dark:text-amber-300'
                )}
              >
                {groupTitle(visibleParts)}
              </FadeText>
              {totalDurationLabel && <span className={TOOL_HEADER_DURATION_CLASS}>{totalDurationLabel}</span>}
            </span>
            {statusSummary && (
              <FadeText
                className={cn(
                  TOOL_HEADER_SUBTITLE_CLASS,
                  displayStatus === 'warning' ? 'text-amber-700/80 dark:text-amber-300/85' : 'text-destructive/85'
                )}
              >
                {statusSummary}
              </FadeText>
            )}
          </DisclosureRow>
        )}
        {isGroup && previewTargets.length > 0 && (
          <div className="mt-2 grid w-full min-w-0 max-w-full gap-2 overflow-hidden pr-2 pl-3">
            {previewTargets.map(target => (
              <PreviewAttachment key={target} source="tool-result" target={target} />
            ))}
          </div>
        )}
        {/* Body is always rendered so children stay mounted across collapse/
            expand and across the 1→2 group transition. `hidden` removes it
            from a11y/visual flow without unmounting React subtree. */}
        <div className={cn(isGroup && 'mt-0.5 w-full overflow-hidden pr-2 pl-3')} hidden={isGroup && !open} key="body">
          {children}
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
