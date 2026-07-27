import DOMPurify from 'dompurify'
import { useMemo } from 'react'
import ShikiHighlighter from 'react-shiki'

import { chunkTextLines, useFixedRowWindow } from '@/components/chat/fixed-row-window'
import type { ArtifactKind } from '@/lib/artifact-detect'

const SHIKI_THEME = { dark: 'github-dark-default', light: 'github-light-default' } as const
const SOURCE_CHUNK_LINES = 200
const SOURCE_LINE_PX = 20
const SOURCE_OVERSCAN_LINES = 400

/** Windowed, Shiki-highlighted source view for artifact content. Same fixed-row
 *  windowing as the file preview's SourceView so a 5k-line artifact scrolls
 *  smoothly, minus the gutter drag/selection machinery (artifact content has no
 *  on-disk path to reference lines against). */
export function ArtifactSourceView({ language, text }: { language: string; text: string }) {
  const chunks = useMemo(() => chunkTextLines(text, SOURCE_CHUNK_LINES), [text])
  const lastChunk = chunks.at(-1)
  const totalLines = lastChunk ? lastChunk.start + lastChunk.lines.length : 0

  const { afterRows, beforeRows, endChunk, onScroll, scrollerRef, startChunk } = useFixedRowWindow({
    overscanRows: SOURCE_OVERSCAN_LINES,
    rowPx: SOURCE_LINE_PX,
    rowsPerChunk: SOURCE_CHUNK_LINES,
    totalRows: totalLines
  })

  const visibleChunks = chunks.slice(startChunk, endChunk + 1)

  return (
    <div className="h-full overflow-auto" onScroll={onScroll} ref={scrollerRef}>
      <div className="min-w-max px-3 py-2 font-mono text-[0.7rem] leading-relaxed" data-selectable-text="true">
        {beforeRows > 0 && <div aria-hidden style={{ height: beforeRows * SOURCE_LINE_PX }} />}
        {visibleChunks.map(chunk => (
          <div className="[&_pre]:m-0" key={chunk.start}>
            <ShikiHighlighter
              addDefaultStyles={false}
              as="div"
              defaultColor="light-dark()"
              delay={80}
              language={language || 'text'}
              showLanguage={false}
              theme={SHIKI_THEME}
            >
              {chunk.text}
            </ShikiHighlighter>
          </div>
        ))}
        {afterRows > 0 && <div aria-hidden style={{ height: afterRows * SOURCE_LINE_PX }} />}
      </div>
    </div>
  )
}

/** Wrap an HTML fragment in a minimal document shell; full documents pass
 *  through untouched. Keeps generated fragments (no <html>/<body>) rendering
 *  with sane defaults instead of quirks-mode soup. */
export function composeArtifactHtml(content: string): string {
  if (/<html[\s>]|<!doctype\s+html/i.test(content)) {
    return content
  }

  return [
    '<!doctype html>',
    '<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
    '<style>body{margin:0;font-family:system-ui,sans-serif}</style></head><body>',
    content,
    '</body></html>'
  ].join('\n')
}

/**
 * Sandboxed live renderer for html/svg artifact content.
 *
 * HTML runs in an `<iframe sandbox="allow-scripts">` — scripts execute in an
 * opaque origin with no same-origin access, no top navigation, no popups, no
 * form submission out of the frame. The parent app is unreachable. SVG is
 * DOMPurify-sanitized with the same profile as the inline ```svg embed.
 */
export function ArtifactLivePreview({ content, kind, title }: { content: string; kind: ArtifactKind; title: string }) {
  const svgClean = useMemo(
    () => (kind === 'svg' ? DOMPurify.sanitize(content, { USE_PROFILES: { svg: true, svgFilters: true } }) : ''),
    [content, kind]
  )

  if (kind === 'svg') {
    return (
      <div className="grid h-full place-items-center overflow-auto bg-background p-4 [&_svg]:h-auto [&_svg]:max-h-full [&_svg]:w-auto [&_svg]:max-w-full">
        <div dangerouslySetInnerHTML={{ __html: svgClean }} />
      </div>
    )
  }

  return (
    <iframe
      className="block size-full border-0 bg-white"
      sandbox="allow-scripts"
      srcDoc={composeArtifactHtml(content)}
      // Deliberately raw white + forced light scheme: the frame hosts foreign
      // generated HTML that assumes a light canvas, so it renders deterministically
      // light in both app themes instead of inheriting theme tokens it can't see.
      style={{ colorScheme: 'light' }}
      title={title}
    />
  )
}
