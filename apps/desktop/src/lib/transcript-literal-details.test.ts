import { expect, it } from 'vitest'

import { DEFAULT_TRANSCRIPT_OPTIONS, type TranscriptDocument } from './transcript-document'
import { renderTranscript } from './transcript-render'

const doc: TranscriptDocument = {
 version: 1, session_id: 'literal', title: 'Tool detail sample', exported_at: '2026-11-03T18:00:00Z',
 options: { ...DEFAULT_TRANSCRIPT_OPTIONS, include_tools: true }, warnings: [], messages: []
}


it('preserves literal tool text inside Markdown HTML blocks, including safe links and the off switch', async () => {
  const { unified } = await import('unified')
  const { default: remarkParse } = await import('remark-parse')
  const { default: remarkGfm } = await import('remark-gfm')
  const literal = 'run **bold** [label](https://example.org/tool)\n<script>no()</script>'

  for (const enabled of [true, false]) {
    const value = {
      ...doc,
      options: { ...doc.options, include_links: enabled },
      messages: [
        { role: 'assistant', speaker: 'Assistant', text: 'ok', tool_calls: [{ name: 'run', arguments: literal }] },
        { role: 'tool', speaker: 'Tool', text: literal }
      ]
    }

    const tree = unified().use(remarkParse).use(remarkGfm).parse(renderTranscript(value, 'md').content)
    const blocks: string[] = []

    const visit = (node: { type: string; value?: string; children?: unknown[] }) => {
      if (node.type === 'html') { blocks.push(node.value ?? '') }

      for (const child of node.children ?? []) { visit(child as typeof node) }
    }

    visit(tree)
    const html = new DOMParser().parseFromString(blocks.join('\n'), 'text/html')
    expect([...html.querySelectorAll('pre')].filter(block => block.textContent === literal)).toHaveLength(2)
    expect(html.querySelector('script')).toBeNull()
    const links = [...html.querySelectorAll('a')].map(a => a.getAttribute('href'))
    expect(links).toEqual(enabled ? ['https://example.org/tool', 'https://example.org/tool'] : [])
  }
})
