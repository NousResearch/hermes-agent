import { useMemo, useState } from 'react'

import type { OfficePreview, SpreadsheetSheet } from '@/lib/ooxml-preview'
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

export function OfficePreviewView({ preview, truncatedLabel }: { preview: OfficePreview; truncatedLabel: string }) {
  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden bg-transparent">
      {preview.truncated && (
        <div className="border-b border-border/60 bg-muted/35 px-3 py-1.5 text-[0.68rem] text-muted-foreground">
          {truncatedLabel}
        </div>
      )}
      {preview.kind === 'spreadsheet' ? (
        <SpreadsheetGrid sheets={preview.sheets} />
      ) : preview.kind === 'document' ? (
        <DocumentHtml html={preview.html} />
      ) : (
        <SlideStack slides={preview.slides} />
      )}
    </div>
  )
}

function SpreadsheetGrid({ sheets }: { sheets: SpreadsheetSheet[] }) {
  const [active, setActive] = useState(0)
  const sheet = sheets[Math.min(active, Math.max(0, sheets.length - 1))]
  const colCount = useMemo(() => Math.max(1, ...(sheet?.rows.map(row => row.length) || [1])), [sheet])

  if (!sheet) {
    return null
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
      {sheets.length > 1 && (
        <div className="flex shrink-0 gap-1 overflow-x-auto border-b border-border/60 px-2 py-1.5">
          {sheets.map((item, index) => (
            <button
              className={cn(
                'shrink-0 rounded-md px-2 py-0.5 text-[0.68rem] font-medium transition-colors',
                index === active
                  ? 'bg-muted text-foreground'
                  : 'text-muted-foreground hover:bg-muted/60 hover:text-foreground'
              )}
              key={`${item.name}-${index}`}
              onClick={() => setActive(index)}
              type="button"
            >
              {item.name}
            </button>
          ))}
        </div>
      )}
      <div className="min-h-0 flex-1 overflow-auto">
        <table className="min-w-full border-collapse font-mono text-[0.7rem] leading-5">
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
                {Array.from({ length: colCount }, (_, col) => (
                  <td
                    className="max-w-64 truncate border-b border-r border-border/50 px-2 py-1 text-foreground"
                    key={col}
                  >
                    {row[col] || ''}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function DocumentHtml({ html }: { html: string }) {
  return (
    <div
      className="office-doc-preview min-h-0 flex-1 overflow-auto px-5 py-4 text-sm leading-relaxed text-foreground [&_h1]:mb-3 [&_h1]:text-2xl [&_h1]:font-semibold [&_h2]:mb-2 [&_h2]:mt-4 [&_h2]:text-xl [&_h2]:font-semibold [&_h3]:mb-2 [&_h3]:mt-3 [&_h3]:text-lg [&_h3]:font-semibold [&_p]:mb-3 [&_p]:last:mb-0 [&_strong]:font-semibold [&_table]:mb-4 [&_table]:w-full [&_table]:border-collapse [&_td]:border [&_td]:border-border/60 [&_td]:px-2 [&_td]:py-1"
      // HTML is produced by parseOfficePreview from escaped OOXML text runs.
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}

function SlideStack({ slides }: { slides: { html: string }[] }) {
  return (
    <div className="min-h-0 flex-1 space-y-3 overflow-auto p-3">
      {slides.map((slide, index) => (
        <section className="rounded-lg border border-border/70 bg-muted/20 px-4 py-3" key={index}>
          <div className="mb-2 text-[0.65rem] font-medium uppercase tracking-wide text-muted-foreground">
            Slide {index + 1}
          </div>
          <div
            className="text-sm leading-relaxed text-foreground [&_p]:mb-2 [&_p]:last:mb-0"
            dangerouslySetInnerHTML={{ __html: slide.html }}
          />
        </section>
      ))}
    </div>
  )
}
