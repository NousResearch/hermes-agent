import { type UIEvent, useMemo, useRef } from 'react'
import type { ThemedToken } from 'shiki'

import { useI18n } from '@/i18n'
import { shikiLanguageForFilename } from '@/lib/markdown-code'
import { cn } from '@/lib/utils'

import { parseDiff } from './diff-lines'
import { tokenStyle, useDiffTokens } from './diff-tokens'
import { chunkLines, useFixedRowWindow } from './fixed-row-window'
import { exceedsHighlightBudget } from './shiki-highlighter'
import { pairDiffLines, type SplitDiffLine } from './split-diff-rows'

const ROW_PX = 20
const CHUNK_ROWS = 100

const TINT = {
  add: 'border-(--ui-diff-add-border) bg-(--ui-diff-add-background) text-(--ui-diff-add-foreground)',
  remove: 'border-(--ui-diff-remove-border) bg-(--ui-diff-remove-background) text-(--ui-diff-remove-foreground)',
  context: 'border-transparent'
}

interface SplitDiffPanelProps {
  diff: string
  path?: string
}

interface SplitLineProps {
  line?: SplitDiffLine
  number?: number
  tokens?: ThemedToken[]
}

function SplitLine({ line, number, tokens }: SplitLineProps) {
  return (
    <div className={cn('flex h-5 min-w-max border-l-2 leading-5', line ? TINT[line.kind] : 'border-transparent')}>
      <span
        aria-hidden
        className="sticky left-0 w-10 shrink-0 select-none bg-(--ui-editor-surface-background) pr-2 text-right text-muted-foreground"
      >
        {number}
      </span>
      <span className="whitespace-pre px-2.5">
        {tokens?.length
          ? tokens.map(token => (
              <span key={token.offset} style={tokenStyle(token)}>
                {token.content}
              </span>
            ))
          : line?.text || ' '}
      </span>
    </div>
  )
}

/** Two horizontal viewports share one vertical row window, including blank partners. */
export function SplitDiffPanel({ diff, path }: SplitDiffPanelProps) {
  const { t } = useI18n()
  const c = t.statusStack.coding
  const lines = useMemo(() => parseDiff(diff), [diff])
  const rows = useMemo(() => pairDiffLines(lines), [lines])
  const chunks = useMemo(() => chunkLines(rows, CHUNK_ROWS), [rows])
  const code = useMemo(() => lines.map(line => line.text).join('\n'), [lines])
  const language = exceedsHighlightBudget(diff) ? null : (shikiLanguageForFilename(path) ?? null)
  const tokens = useDiffTokens(code, language)
  const rightRef = useRef<HTMLDivElement | null>(null)

  const { afterRows, beforeRows, endChunk, onScroll, scrollerRef, startChunk } = useFixedRowWindow({
    overscanRows: 100,
    rowPx: ROW_PX,
    rowsPerChunk: CHUNK_ROWS,
    totalRows: rows.length
  })

  const visible = chunks.slice(startChunk, endChunk + 1)

  const syncScroll = (event: UIEvent<HTMLDivElement>) => {
    const source = event.currentTarget
    const peer = source === scrollerRef.current ? rightRef.current : scrollerRef.current

    if (peer && peer.scrollTop !== source.scrollTop) {
      peer.scrollTop = source.scrollTop
    }

    onScroll(event)
  }

  return (
    <div className="grid h-full min-h-0 grid-cols-2 font-mono text-[0.7rem] text-(--ui-text-secondary)" dir="ltr">
      {(['before', 'after'] as const).map(side => (
        <div
          className={cn('flex min-h-0 min-w-0 flex-col', side === 'after' && 'border-l border-(--ui-stroke-tertiary)')}
          key={side}
        >
          <div className="shrink-0 px-2.5 py-1 font-sans text-muted-foreground">
            {side === 'before' ? c.diffBefore : c.diffAfter}
          </div>
          <div
            aria-label={side === 'before' ? c.diffBefore : c.diffAfter}
            className="min-h-0 flex-1 overflow-x-scroll overflow-y-auto overscroll-x-contain overscroll-y-auto"
            onScroll={syncScroll}
            ref={side === 'before' ? scrollerRef : rightRef}
            role="region"
            tabIndex={0}
          >
            <div className="min-w-max py-3">
              {beforeRows > 0 && <div aria-hidden style={{ height: beforeRows * ROW_PX }} />}
              {visible.flatMap(chunk =>
                chunk.lines.map((row, offset) => {
                  const line = row[side]

                  return (
                    <SplitLine
                      key={chunk.start + offset}
                      line={line}
                      number={side === 'before' ? line?.oldNo : line?.newNo}
                      tokens={line && tokens?.[line.index]}
                    />
                  )
                })
              )}
              {afterRows > 0 && <div aria-hidden style={{ height: afterRows * ROW_PX }} />}
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
