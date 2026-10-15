import type { OfficePreviewKind } from '@hermes/shared'
import { useEffect, useState } from 'react'

import { docxPreviewDocument } from '@/lib/office-document'
import { odtPreviewDocument } from '@/lib/office-odf'
import { parsePptx, type SlidePreview } from '@/lib/office-pptx'
import { dataUrlBytes, parseSpreadsheet, type SheetPreview } from '@/lib/office-workbook'

function columnLabel(index: number): string {
  let n = index + 1
  let label = ''

  while (n > 0) {
    const rem = (n - 1) % 26
    label = String.fromCharCode(65 + rem) + label
    n = Math.floor((n - 1) / 26)
  }

  return label
}

function isNumeric(value: string): boolean {
  return /^-?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?$/.test(value) || /^-?\d+(?:\.\d+)?%?$/.test(value) || /^\d{4}-\d{2}-\d{2}/.test(value)
}

export function OfficePreview({
  dataUrl,
  kind,
  label
}: {
  dataUrl: string
  kind: OfficePreviewKind
  label: string
}) {
  const [html, setHtml] = useState<string>()
  const [sheets, setSheets] = useState<SheetPreview[]>()
  const [slides, setSlides] = useState<SlidePreview[]>()
  const [sheetIndex, setSheetIndex] = useState(0)
  const [error, setError] = useState<string>()

  useEffect(() => {
    let active = true
    setHtml(undefined)
    setSheets(undefined)
    setSlides(undefined)
    setSheetIndex(0)
    setError(undefined)

    async function load() {
      try {
        const bytes = dataUrlBytes(dataUrl)

        if (kind === 'document') {
          const document = /\.odt$/i.test(label) ? odtPreviewDocument(bytes) : await docxPreviewDocument(bytes)

          if (active) {
            setHtml(document)
          }

          return
        }

        if (kind === 'presentation') {
          const parsed = parsePptx(bytes)

          if (active) {
            setSlides(parsed)
          }

          return
        }

        const parsed = parseSpreadsheet(bytes, label)

        if (active) {
          setSheets(parsed)
        }
      } catch (caught) {
        if (active) {
          setError(caught instanceof Error ? caught.message : String(caught))
        }
      }
    }

    void load()

    return () => {
      active = false
    }
  }, [dataUrl, kind, label])

  if (error) {
    return (
      <div className="flex h-full items-center justify-center p-6 text-center text-sm text-muted-foreground">{error}</div>
    )
  }

  if (kind === 'document' && html) {
    return <iframe className="h-full w-full border-0 bg-[#e7e6e6]" sandbox="" srcDoc={html} title={label} />
  }

  if (slides) {
    return (
      <div className="h-full overflow-auto bg-[#e7e6e6] px-4 py-6 text-[#222]">
        {slides.map((slide, index) => (
          <article
            className="mx-auto mb-6 max-w-3xl bg-white px-8 py-7 shadow-[0_1px_3px_rgba(0,0,0,0.22)]"
            key={index}
          >
            <div className="mb-3 text-[11px] font-medium tracking-wide text-[#888]">SLIDE {index + 1}</div>
            {slide.paragraphs.map((paragraph, paragraphIndex) => (
              <p
                className={paragraphIndex === 0 ? 'mb-3 text-2xl font-semibold' : 'mb-2 text-[15px] leading-6'}
                key={paragraphIndex}
              >
                {paragraph}
              </p>
            ))}
            {slide.images.map(src => (
              <img alt="" className="mt-3 max-h-80 max-w-full" key={src.slice(0, 48)} src={src} />
            ))}
          </article>
        ))}
      </div>
    )
  }

  if (sheets) {
    const sheet = sheets[Math.min(sheetIndex, sheets.length - 1)]
    const colCount = Math.max(sheet.widths?.length ?? 0, ...sheet.rows.map(row => row.length), 1)

    return (
      <div className="flex h-full min-h-0 flex-col bg-[#f3f3f3] text-[#222] [font-family:Calibri,'Segoe_UI',sans-serif]">
        <div className="min-h-0 flex-1 overflow-auto">
          <table className="border-separate border-spacing-0 text-[12px] leading-4">
            <thead className="sticky top-0 z-10">
              <tr>
                <th className="sticky left-0 z-20 h-5 w-10 border border-[#d4d4d4] bg-[#f8f9fb]" />
                {Array.from({ length: colCount }, (_, index) => (
                  <th
                    className="h-5 border border-[#d4d4d4] bg-[#f8f9fb] px-1 text-center font-normal text-[#666]"
                    key={columnLabel(index)}
                    style={{ minWidth: sheet.widths?.[index] ?? 88 }}
                  >
                    {columnLabel(index)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(sheet.rows.length ? sheet.rows : [[]]).map((row, rowIndex) => (
                <tr key={rowIndex}>
                  <th className="sticky left-0 z-10 w-10 border border-[#d4d4d4] bg-[#f8f9fb] px-1 text-center font-normal text-[#666]">
                    {rowIndex + 1}
                  </th>
                  {Array.from({ length: colCount }, (_, colIndex) => {
                    const value = row[colIndex] ?? ''

                    return (
                      <td
                        className={`h-5 border border-[#e2e2e2] bg-white px-1.5 align-middle whitespace-pre ${isNumeric(value) ? 'text-right' : 'text-left'}`}
                        key={colIndex}
                        style={{ minWidth: sheet.widths?.[colIndex] ?? 88 }}
                      >
                        {value}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="flex shrink-0 gap-0.5 overflow-x-auto border-t border-[#d4d4d4] bg-[#f3f3f3] px-1 py-0.5">
          {sheets.map((item, index) => (
            <button
              className={`shrink-0 border px-3 py-1 text-[11px] ${index === sheetIndex ? 'border-[#d4d4d4] border-b-white bg-white font-medium' : 'border-transparent text-[#666]'}`}
              key={`${item.name}-${index}`}
              onClick={() => setSheetIndex(index)}
              type="button"
            >
              {item.name}
            </button>
          ))}
        </div>
      </div>
    )
  }

  return null
}
