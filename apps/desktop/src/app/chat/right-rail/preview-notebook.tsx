import DOMPurify from 'dompurify'
import { useMemo } from 'react'

import { LazyShiki as ShikiHighlighter } from '@/components/chat/shiki-highlighter'
import { useI18n } from '@/i18n'
import type { ParsedNotebookCell, ParsedNotebookOutput } from '@/lib/notebook-preview'
import { parseNotebook } from '@/lib/notebook-preview'
import { cn } from '@/lib/utils'

import { MarkdownPreview } from './preview-file'

const SHIKI_THEME = { dark: 'github-dark-default', light: 'github-light-default' } as const

const HTML_SANITIZE = {
  FORBID_ATTR: ['action', 'formaction', 'srcdoc', 'xlink:href'],
  FORBID_TAGS: ['base', 'embed', 'form', 'frame', 'iframe', 'link', 'meta', 'object', 'script', 'template']
} as const

export function NotebookPreview({ text }: { text: string }) {
  const { t } = useI18n()
  const notebook = useMemo(() => parseNotebook(text), [text])

  if (!notebook) {
    return (
      <div className="grid place-items-center px-6 py-10 text-center">
        <div className="grid max-w-sm gap-2">
          <div className="text-sm font-medium text-foreground">{t.preview.invalidNotebookTitle}</div>
          <div className="text-xs leading-relaxed text-muted-foreground">{t.preview.invalidNotebookBody}</div>
        </div>
      </div>
    )
  }

  return (
    <div className="preview-notebook mx-auto max-w-3xl divide-y divide-border/40 px-4 text-sm text-foreground">
      {notebook.cells.map((cell, index) => (
        <NotebookCellView cell={cell} key={index} language={notebook.language} />
      ))}
    </div>
  )
}

function NotebookCellView({ cell, language }: { cell: ParsedNotebookCell; language: string }) {
  if (cell.kind === 'markdown') {
    if (!cell.source.trim()) {
      return null
    }

    return <MarkdownPreview className="mx-0 max-w-none px-0" text={cell.source} />
  }

  if (cell.kind === 'raw') {
    if (!cell.source.trim()) {
      return null
    }

    return (
      <pre className="overflow-x-auto py-3 font-mono text-xs leading-relaxed text-muted-foreground">{cell.source}</pre>
    )
  }

  const count = cell.executionCount ?? ' '
  const code = cell.source.replace(/\n$/, '')

  return (
    <div className="grid gap-2 py-3">
      <div className="font-mono text-[0.625rem] font-bold tabular-nums text-muted-foreground">{`In [${count}]:`}</div>
      {code ? (
        <div className="overflow-hidden rounded-lg border border-border bg-card [&_pre]:m-0 [&_pre]:overflow-x-auto [&_pre]:p-3 [&_pre]:font-mono [&_pre]:text-xs">
          <ShikiHighlighter code={code} language={language || 'python'} theme={SHIKI_THEME} />
        </div>
      ) : null}
      {cell.outputs.length > 0 && (
        <div className="grid gap-2">
          <div className="font-mono text-[0.625rem] font-bold tabular-nums text-muted-foreground">{`Out [${count}]:`}</div>
          {cell.outputs.map((output, index) => (
            <NotebookOutputView key={index} output={output} />
          ))}
        </div>
      )}
    </div>
  )
}

function NotebookOutputView({ output }: { output: ParsedNotebookOutput }) {
  const { t } = useI18n()

  switch (output.type) {
    case 'stream':
      return (
        <pre
          className={cn(
            'overflow-x-auto whitespace-pre-wrap font-mono text-xs leading-relaxed',
            output.name === 'stderr' ? 'text-destructive' : 'text-foreground'
          )}
        >
          {output.text}
        </pre>
      )

    case 'error':
      return (
        <pre className="overflow-x-auto whitespace-pre-wrap font-mono text-xs leading-relaxed text-destructive">
          {`${output.ename}${output.evalue ? `: ${output.evalue}` : ''}${output.traceback ? `\n${output.traceback}` : ''}`}
        </pre>
      )

    case 'image':
      return (
        <img
          alt={output.mime}
          className="max-h-96 w-auto max-w-full rounded-lg border border-border object-contain"
          src={output.dataUrl}
        />
      )

    case 'svg':
      return <NotebookSvg svg={output.svg} />

    case 'html':
      return <NotebookHtml html={output.html} />

    case 'markdown':
      return <MarkdownPreview className="mx-0 max-w-none px-0 py-0" text={output.text} />

    case 'text':
      return (
        <pre className="overflow-x-auto whitespace-pre-wrap font-mono text-xs leading-relaxed text-foreground">
          {output.text}
        </pre>
      )

    case 'widget':
      return <div className="text-xs text-muted-foreground">{t.preview.notebookWidget}</div>
  }
}

function NotebookHtml({ html }: { html: string }) {
  const clean = useMemo(() => DOMPurify.sanitize(html, HTML_SANITIZE), [html])

  if (!clean.trim()) {
    return null
  }

  return (
    <div
      className="preview-notebook-html max-w-full overflow-x-auto text-sm [&_img]:max-h-96 [&_img]:max-w-full [&_table]:w-full"
      dangerouslySetInnerHTML={{ __html: clean }}
    />
  )
}

function NotebookSvg({ svg }: { svg: string }) {
  const clean = useMemo(() => DOMPurify.sanitize(svg, { USE_PROFILES: { svg: true, svgFilters: true } }), [svg])

  if (!clean.trim()) {
    return null
  }

  return (
    <div
      className="[&_svg]:block [&_svg]:h-auto [&_svg]:w-auto [&_svg]:max-h-[33dvh] [&_svg]:max-w-full"
      dangerouslySetInnerHTML={{ __html: clean }}
    />
  )
}
