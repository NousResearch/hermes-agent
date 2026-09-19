import { type ReactNode, useMemo, useState } from 'react'

import type {
  OfficeBlock,
  OfficeParagraph,
  OfficePreview,
  OfficeTextRun,
  SpreadsheetSheet
} from '@/lib/ooxml-preview'
import { cn } from '@/lib/utils'

const OFFICE_CALIBRI_STACK = 'Calibri, Carlito, "Segoe UI", "Liberation Sans", Arial, sans-serif'
const OFFICE_HEADING_STACK = 'Cambria, Calibri, Carlito, "Liberation Serif", Georgia, serif'

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
  truncatedLabel
}: {
  formulaBarLabel: string
  preview: OfficePreview
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
      ) : (
        <DocumentBlocks blocks={preview.blocks} />
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
      <div
        className="min-h-0 flex-1 overflow-auto"
        data-testid="office-sheet-scroll"
        style={{ backgroundColor: '#ffffff' }}
      >
        <table
          className="w-max min-w-full border-collapse text-[11pt] leading-5"
          role="grid"
          style={{ fontFamily: OFFICE_CALIBRI_STACK }}
        >
          <thead className="sticky top-0 z-10 bg-muted/80">
            <tr>
              <th className="sticky left-0 z-20 w-10 border-b border-r border-border/60 bg-muted/80 px-1 py-1 text-right font-medium text-muted-foreground" />
              {Array.from({ length: colCount }, (_, col) => (
                <th
                  className="min-w-24 border-b border-r border-border/60 px-2 py-1 text-center font-medium text-muted-foreground"
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
                        'min-w-24 max-w-64 cursor-default truncate border-b border-r border-border/50 px-2 py-1',
                        isSelected ? 'outline outline-2 outline-offset-[-2px] outline-ring' : undefined
                      )}
                      key={col}
                      onClick={() => setSelected({ col, row: rowIndex })}
                      role="gridcell"
                      style={{
                        backgroundColor: cell?.fill || '#ffffff',
                        color: cell?.color || '#000000',
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
    <div
      className="min-h-0 flex-1 overflow-auto"
      data-testid="office-document-scroll"
      style={{ backgroundColor: '#cfcfcf' }}
    >
      <article
        data-testid="office-document-page"
        style={{
          backgroundColor: '#ffffff',
          boxShadow: '0 1px 6px rgba(0, 0, 0, 0.18)',
          color: '#000000',
          fontFamily: OFFICE_CALIBRI_STACK,
          fontSize: '11pt',
          lineHeight: 1.15,
          margin: '24px auto',
          maxWidth: 816,
          minHeight: 1056,
          padding: 96
        }}
      >
        {renderDocumentBlocks(blocks)}
      </article>
    </div>
  )
}

function renderDocumentBlocks(blocks: OfficeBlock[]) {
  const nodes: ReactNode[] = []
  let index = 0

  while (index < blocks.length) {
    const block = blocks[index]

    if (block.type === 'paragraph' && block.list) {
      const kind = block.list
      const items: OfficeParagraph[] = []

      while (index < blocks.length) {
        const next = blocks[index]

        if (next.type !== 'paragraph' || next.list !== kind) {
          break
        }

        items.push(next)
        index += 1
      }

      const ListTag = kind === 'number' ? 'ol' : 'ul'

      nodes.push(
        <ListTag
          className="mb-3 pl-8"
          key={`list-${index}`}
          style={{ listStyleType: kind === 'number' ? 'decimal' : 'disc' }}
        >
          {items.map((item, itemIndex) => (
            <li
              dir={item.dir}
              key={itemIndex}
              style={{ direction: item.dir, marginBottom: 4, textAlign: item.align }}
            >
              {item.runs.map((run, runIndex) => (
                <OfficeRun key={runIndex} run={run} />
              ))}
            </li>
          ))}
        </ListTag>
      )

      continue
    }

    nodes.push(<OfficeBlockView block={block} key={index} />)
    index += 1
  }

  return nodes
}

function OfficeBlockView({ block }: { block: OfficeBlock }) {
  if (block.type === 'table') {
    return (
      <table className="mb-4 w-full border-collapse">
        <tbody>
          {block.rows.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {row.map((cell, cellIndex) => (
                <td
                  key={cellIndex}
                  style={{ border: '1px solid #000000', padding: '4px 8px' }}
                >
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
  const headingSize = block.heading === 1 ? '16pt' : block.heading === 2 ? '13pt' : block.heading === 3 ? '12pt' : undefined

  return (
    <Tag
      dir={block.dir}
      style={{
        direction: block.dir,
        fontFamily: block.heading ? OFFICE_HEADING_STACK : undefined,
        fontSize: headingSize,
        fontWeight: block.heading ? 700 : undefined,
        marginBottom: '10pt',
        marginTop: block.heading && block.heading > 1 ? '12pt' : 0,
        minHeight: block.runs.length ? undefined : '1em',
        textAlign: block.align
      }}
    >
      {block.runs.map((run, index) => (
        <OfficeRun key={index} run={run} />
      ))}
    </Tag>
  )
}

function cssFont(name: string | undefined): string | undefined {
  if (!name) {
    return undefined
  }

  const families = name
    .split(',')
    .map(cssFontFamily)
    .filter((family): family is string => Boolean(family))

  if (!families.length) {
    return undefined
  }

  return `${families.join(', ')}, ${OFFICE_CALIBRI_STACK}`
}

function cssFontFamily(name: string): string | undefined {
  const safe = name.replace(/["';{}\\]/g, '').replace(/\s+/g, ' ').trim()

  if (!safe || /[:()]/.test(safe)) {
    return undefined
  }

  return /[^A-Za-z0-9-]/.test(safe) ? `"${safe}"` : safe
}

function OfficeRun({ run }: { run: OfficeTextRun }) {
  return (
    <span
      style={{
        color: run.color,
        fontFamily: cssFont(run.fontFamily),
        fontSize: run.fontSize ? `${run.fontSize}pt` : undefined,
        fontStyle: run.italic ? 'italic' : undefined,
        fontWeight: run.bold ? 700 : undefined,
        textDecoration: run.underline ? 'underline' : undefined
      }}
    >
      {run.text}
    </span>
  )
}
