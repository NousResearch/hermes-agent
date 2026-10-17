import { useMemo } from 'react'

import { useI18n } from '@/i18n'
import { parseDelimited, type TableDelimiter, UnterminatedQuoteError } from '@/lib/delimited-table'

// Bounds the DOM, not the data: Source view and the file itself stay complete.
const MAX_ROWS = 500
const MAX_COLUMNS = 100

export function DelimitedTablePreview({
  delimiter,
  label,
  text
}: {
  delimiter: TableDelimiter
  label: string
  text: string
}) {
  const { t } = useI18n()
  const copy = t.preview.table

  const parsed = useMemo(() => {
    try {
      return { rows: parseDelimited(text, delimiter) }
    } catch (error) {
      if (error instanceof UnterminatedQuoteError) {
        return { error: true as const, rows: [] }
      }

      throw error
    }
  }, [delimiter, text])

  if ('error' in parsed) {
    return (
      <p className="px-4 py-3 text-sm text-muted-foreground" role="alert">
        {copy.unterminatedQuote}
      </p>
    )
  }

  if (parsed.rows.length === 0) {
    return <p className="px-4 py-3 text-sm text-muted-foreground">{copy.empty}</p>
  }

  const [header, ...records] = parsed.rows
  const columnCount = parsed.rows.reduce((max, row) => Math.max(max, row.length), 0)
  const shownColumns = Math.min(columnCount, MAX_COLUMNS)
  const shownRows = Math.min(records.length, MAX_ROWS)
  const columns = Array.from({ length: shownColumns }, (_, index) => index)

  return (
    <div className="flex min-h-full flex-col" data-preview-table="">
      <div
        className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-border/60 bg-muted/35 px-3 py-1.5 text-[0.68rem] text-muted-foreground"
        role="status"
      >
        <span>{copy.summary(records.length, columnCount)}</span>
        {(shownRows < records.length || shownColumns < columnCount) && (
          <span>{copy.clipped(shownRows, shownColumns)}</span>
        )}
      </div>
      <div aria-label={copy.regionLabel(label)} className="min-h-0 flex-1" role="region" tabIndex={0}>
        <table className="w-full border-collapse text-xs">
          <thead className="sticky top-0 z-10 bg-muted">
            <tr>
              <th
                className="w-10 border-b border-border px-2 py-1.5 text-right font-mono text-[0.65rem] font-normal text-muted-foreground"
                scope="col"
              >
                #
              </th>
              {columns.map(column => (
                <th
                  className="whitespace-nowrap border-b border-border px-3 py-1.5 text-left font-medium text-foreground"
                  key={column}
                  scope="col"
                >
                  {header[column] || copy.column(column + 1)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {records.slice(0, MAX_ROWS).map((row, index) => (
              <tr className="even:bg-muted/25 hover:bg-accent/40" key={index}>
                <th
                  className="border-b border-border/40 px-2 py-1 text-right align-top font-mono text-[0.65rem] font-normal text-muted-foreground"
                  scope="row"
                >
                  {index + 1}
                </th>
                {columns.map(column => (
                  <td className="whitespace-pre-wrap border-b border-border/40 px-3 py-1 align-top" key={column}>
                    {row[column] ?? ''}
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
