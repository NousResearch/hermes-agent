import { type ReactNode, useMemo, useState } from 'react'

import type { OfficeBlock, OfficePreview, OfficeTextRun, SpreadsheetSheet } from '@/lib/ooxml-preview'
import { cn } from '@/lib/utils'

function columnLabel(index: number) {
  let value = index
  let label = ''

  do {
    label = String.fromCharCode(65 + (value % 26)) + label
    value = Math.floor(value / 26) - 1
  } while (value >= 0)

  return label
}

export function OfficePreviewView({
  formulaBarLabel,
  preview,
  slideLabel,
  truncatedLabel
}: {
  formulaBarLabel: string
  preview: OfficePreview
  slideLabel: (index: number) => string
  truncatedLabel: string
}) {
  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden bg-transparent">
      {preview.truncated && (
        <div className="border-b border-border/60 bg-muted/35 px-3 py-1.5 text-[0.68rem] text-muted-foreground">
          {truncatedLabel}
        </div>
      )}
      {preview.kind === 'spreadsheet' ? (
        <SpreadsheetGrid formulaBarLabel={formulaBarLabel} sheets={preview.sheets} />
      ) : preview.kind === 'document' ? (
        <DocumentBlocks blocks={preview.blocks} />
      ) : (
        <SlideStack slideLabel={slideLabel} slides={preview.slides} />
      )}
    </div>
  )
}

function SpreadsheetGrid({ formulaBarLabel, sheets }: { formulaBarLabel: string; sheets: SpreadsheetSheet[] }) {
  const [active, setActive] = useState(0)
  const [selected, setSelected] = useState<{ col: number; row: number } | null>(null)
  const sheet = sheets[Math.min(active, Math.max(0, sheets.length - 1))]
  const colCount = useMemo(() => Math.max(1, ...(sheet?.rows.map(row => row.length) || [1])), [sheet])
  const selectedCell = selected ? sheet?.rows[selected.row]?.[selected.col] : undefined
  const address = selected ? `${columnLabel(selected.col)}${selected.row + 1}` : ''
  const barValue = selectedCell?.formula ? `=${selectedCell.formula}` : (selectedCell?.value ?? '')

  if (!sheet) {
    return null
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
      <div className="flex shrink-0 items-center gap-2 border-b border-border/60 px-2 py-1.5">
        <span className="w-10 shrink-0 text-center font-mono text-[0.7rem] text-muted-foreground">{address || '—'}</span>
        <input
          aria-label={formulaBarLabel}
          className="h-7 min-w-0 flex-1 rounded-md border border-border/60 bg-background px-2 font-mono text-[0.75rem] text-foreground outline-none"
          readOnly
          value={barValue}
        />
      </div>
      {sheets.length > 1 && (
        <div className="flex shrink-0 gap-1 overflow-x-auto border-b border-border/60 px-2 py-1.5" role="tablist">
          {sheets.map((item, index) => (
            <button
              aria-selected={index === active}
              className={cn(
                'shrink-0 rounded-md px-2 py-0.5 text-[0.68rem] font-medium transition-colors',
                index === active
                  ? 'bg-muted text-foreground'
                  : 'text-muted-foreground hover:bg-muted/60 hover:text-foreground'
              )}
              key={`${item.name}-${index}`}
              onClick={() => {
                setActive(index)
                setSelected(null)
              }}
              role="tab"
              type="button"
            >
              {item.name}
            </button>
          ))}
        </div>
      )}
      <div className="min-h-0 flex-1 overflow-auto">
        <table className="min-w-full border-collapse font-mono text-[0.7rem] leading-5" role="grid">
          <thead className="sticky top-0 z-10 bg-muted/80">
            <tr>
              <th className="sticky left-0 z-20 w-10 border-b border-r border-border/60 bg-muted/80 px-1 py-1 text-right font-medium text-muted-foreground" />
              {Array.from({ length: colCount }, (_, col) => (
                <th
                  className="border-b border-r border-border/60 px-2 py-1 text-center font-medium text-muted-foreground"
                  key={col}
                >
                  {columnLabel(col)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sheet.rows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                <th className="sticky left-0 z-10 border-b border-r border-border/60 bg-muted/40 px-1 py-1 text-right font-medium text-muted-foreground">
                  {rowIndex + 1}
                </th>
                {Array.from({ length: colCount }, (_, col) => {
                  const cell = row[col]
                  const isSelected = selected?.row === rowIndex && selected?.col === col

                  return (
                    <td
                      aria-selected={isSelected}
                      className={cn(
                        'max-w-64 cursor-default truncate border-b border-r border-border/50 px-2 py-1',
                        isSelected ? 'outline outline-2 outline-offset-[-2px] outline-ring' : undefined
                      )}
                      key={col}
                      onClick={() => setSelected({ col, row: rowIndex })}
                      role="gridcell"
                      style={{
                        backgroundColor: cell?.fill,
                        color: cell?.color,
                        fontStyle: cell?.italic ? 'italic' : undefined,
                        fontWeight: cell?.bold ? 700 : undefined,
                        textAlign: cell?.align
                      }}
                    >
                      {cell?.value || ''}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function DocumentBlocks({ blocks }: { blocks: OfficeBlock[] }) {
  return (
    <div className="min-h-0 flex-1 overflow-auto px-5 py-4 text-sm leading-relaxed text-foreground">
      {blocks.map((block, index) => (
        <OfficeBlockView block={block} key={index} />
      ))}
    </div>
  )
}

function OfficeBlockView({ block }: { block: OfficeBlock }) {
  if (block.type === 'table') {
    return (
      <table className="mb-4 w-full border-collapse">
        <tbody>
          {block.rows.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {row.map((cell, cellIndex) => (
                <td className="border border-border/60 px-2 py-1" key={cellIndex}>
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    )
  }

  const Tag = block.heading === 1 ? 'h1' : block.heading === 2 ? 'h2' : block.heading === 3 ? 'h3' : 'p'

  const className =
    block.heading === 1
      ? 'mb-3 text-2xl font-semibold'
      : block.heading === 2
        ? 'mb-2 mt-4 text-xl font-semibold'
        : block.heading === 3
          ? 'mb-2 mt-3 text-lg font-semibold'
          : 'mb-3 last:mb-0'

  return (
    <Tag className={className}>
      {block.runs.map((run, index) => (
        <OfficeRun key={index} run={run} />
      ))}
    </Tag>
  )
}

function OfficeRun({ run }: { run: OfficeTextRun }) {
  let node: ReactNode = run.text

  if (run.italic) {
    node = <em>{node}</em>
  }

  if (run.bold) {
    node = <strong className="font-semibold">{node}</strong>
  }

  return node
}

function SlideStack({
  slideLabel,
  slides
}: {
  slideLabel: (index: number) => string
  slides: { lines: string[] }[]
}) {
  return (
    <div className="min-h-0 flex-1 space-y-3 overflow-auto p-3">
      {slides.map((slide, index) => (
        <section className="rounded-lg border border-border/70 bg-muted/20 px-4 py-3" key={index}>
          <div className="mb-2 text-[0.65rem] font-medium uppercase tracking-wide text-muted-foreground">
            {slideLabel(index + 1)}
          </div>
          <div className="text-sm leading-relaxed text-foreground">
            {slide.lines.map((line, lineIndex) => (
              <p className="mb-2 last:mb-0" key={lineIndex}>
                {line}
              </p>
            ))}
          </div>
        </section>
      ))}
    </div>
  )
}
