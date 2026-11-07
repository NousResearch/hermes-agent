import { expect, it } from 'vitest'

import { DEFAULT_TRANSCRIPT_OPTIONS, type TranscriptDocument } from './transcript-document'
import { renderMarkdown } from './transcript-markdown'

const doc: TranscriptDocument = {
  version: 1,
  session_id: 's',
  title: 'Demo',
  exported_at: 'now',
  options: DEFAULT_TRANSCRIPT_OPTIONS,
  warnings: [],
  messages: [{ role: 'user', speaker: 'Alice', text: '**Hello** world' }]
}

it.each([
  ['inline', 'Before `https://example.org` after', 'inlineCode', 'https://example.org'],
  ['fenced', '```js\nhttps://example.org\n```', 'code', 'https://example.org']
])('preserves genuine %s code as literal in a GFM reader', async (_name, text, kind, expected) => {
  const { unified } = await import('unified')
  const { default: remarkParse } = await import('remark-parse')
  const { default: remarkGfm } = await import('remark-gfm')
  const tree = unified().use(remarkParse).use(remarkGfm).parse(renderMarkdown({ ...doc, messages: [{ role: 'user', speaker: 'U', text }] }))
  const nodes: { type: string; value?: string; children?: unknown[] }[] = []

  const visit = (node: typeof nodes[number]) => {
    nodes.push(node)

    for (const child of node.children ?? []) { visit(child as typeof node) }
  }

  visit(tree)
  expect(nodes.some(node => node.type === kind && node.value === expected)).toBe(true)
  expect(nodes.filter(node => node.type === 'link')).toHaveLength(0)
})

it('keeps basic Markdown as readable Markdown', () => {
  const result = renderMarkdown(doc)
  expect(result).toContain('**Hello** world')
  expect(result).not.toContain('<pre>**Hello** world</pre>')
})

it('preserves unlinked original wording and exposes safe disclosure content', () => {
  const value: TranscriptDocument = {
    ...doc,
    options: { ...doc.options, include_links: false, collapse_details: false },
    warnings: ['Heads up'],
    messages: [
      {
        role: 'tool',
        speaker: 'Tool',
        kind: 'aside',
        text: '[link](https://example.com) <script>x</script>',
        reasoning: 'thought',
        tool_calls: [{ name: 'run', arguments: 'x < y' }],
        attachments: [{ kind: 'video', name: 'clip', reference: 'C:/clip', local: true }]
      }
    ]
  }

  const result = renderMarkdown(value)
  const page = new DOMParser().parseFromString(result, 'text/html')
  expect(result).toContain('Aside — /btw')
  expect(result).toContain('[link](https://example.com) &lt;script&gt;x&lt;/script&gt;')
  expect(page.querySelector('a')).toBeNull()
  expect(page.querySelector('script')).toBeNull()
  expect(result).toContain('may be unavailable')
  expect(result).toContain('**Reasoning**')
  expect(result).toContain('thought')
})

it('does not turn rejected credentials URLs into viewer-autolinks', async () => {
  const { unified } = await import('unified')
  const { default: remarkParse } = await import('remark-parse')
  const { default: remarkGfm } = await import('remark-gfm')

  const value: TranscriptDocument = {
    ...doc,
    title: 'https://user:pass@example.com',
    messages: [{ role: 'user', speaker: 'U', text: 'unsafe https://user:pass@example.com' }]
  }

  const tree = unified().use(remarkParse).use(remarkGfm).parse(renderMarkdown(value))
  const links: unknown[] = []

  const visit = (node: { type: string; children?: unknown[] }) => {
    if (node.type === 'link') {
      links.push(node)
    }

    for (const child of node.children ?? []) {
      visit(child as typeof node)
    }
  }

  visit(tree)
  expect(links, JSON.stringify(links)).toHaveLength(0)
})

it('keeps a later safe link active after a rejected link in the same text', async () => {
  const { unified } = await import('unified')
  const { default: remarkParse } = await import('remark-parse')
  const { default: remarkGfm } = await import('remark-gfm')

  const value: TranscriptDocument = {
    ...doc,
    messages: [{ role: 'user', speaker: 'U', text: '[bad](https://broken.example [good](https://two.example)' }]
  }

  const output = renderMarkdown(value)
  const tree = unified().use(remarkParse).use(remarkGfm).parse(output)
  const links: string[] = []

  const visit = (node: { type: string; url?: string; children?: unknown[] }) => {
    if (node.type === 'link') {
      links.push(node.url ?? '')
    }

    for (const child of node.children ?? []) {
      visit(child as typeof node)
    }
  }

  visit(tree)
  expect(links).toEqual(['https://two.example'])
  expect(output).toContain('broken.example')
  expect(output).toContain('good')
})

it('keeps code literal after rejected prefixes and links later prose', async () => {
  const { unified } = await import('unified')
  const { default: remarkParse } = await import('remark-parse')
  const { default: remarkGfm } = await import('remark-gfm')

  for (const text of [
    '[bad](javascript:alert(1)) ` [good](https://two.example) `',
    '[bad](not-a-url ` [good](https://two.example) `',
    '[bad](not-a-url\n```\n[good](https://two.example)\n```',
    '[bad](not-a-url\n~~~\n[good](https://two.example)\n~~~',
    '[bad](not-a-url\n> ~~~\n> [good](https://two.example)\n> ~~~'
  ]) {
    const value = { ...doc, messages: [{ role: 'user', speaker: 'U', text: `${text}\n\n[later](https://later.example)` }] }
    const tree = unified().use(remarkParse).use(remarkGfm).parse(renderMarkdown(value))
    const links: string[] = []

    const visit = (node: { type: string; url?: string; children?: unknown[] }) => {
      if (node.type === 'link') { links.push(node.url ?? '') }

      for (const child of node.children ?? []) { visit(child as typeof node) }
    }

    visit(tree)
    expect(links, text).toEqual(['https://later.example'])
  }
})

it('preserves a URL-bearing line with 200000 backtick runs without autolinking', async () => {
  const { unified } = await import('unified')
  const { default: remarkParse } = await import('remark-parse')
  const { default: remarkGfm } = await import('remark-gfm')
  const text = 'www.invalid.example ' + '`x'.repeat(200000)
  const value = { ...doc, messages: [{ role: 'user', speaker: 'U', text }] }
  const tree = unified().use(remarkParse).use(remarkGfm).parse(renderMarkdown(value))
  let foundLiteral = false
  let linkCount = 0

  const visit = (node: { type: string; value?: string; children?: unknown[] }) => {
    if (node.type === 'inlineCode' && node.value === text) { foundLiteral = true }

    if (node.type === 'link') { linkCount++ }

    for (const child of node.children ?? []) { visit(child as typeof node) }
  }

  visit(tree)
  expect(foundLiteral).toBe(true)
  expect(linkCount).toBe(0)
})

it('disables links in attachment names and references when links are off', async () => {
  const { unified } = await import('unified')
  const { default: remarkParse } = await import('remark-parse')
  const { default: remarkGfm } = await import('remark-gfm')

  const value: TranscriptDocument = {
    ...doc,
    options: { ...doc.options, include_links: false },
    messages: [
      {
        role: 'user',
        speaker: 'U',
        text: 'ok',
        attachments: [{ kind: 'file', name: 'https://example.com', reference: 'https://example.org' }]
      }
    ]
  }

  const tree = unified().use(remarkParse).use(remarkGfm).parse(renderMarkdown(value))
  const links: unknown[] = []

  const visit = (node: { type: string; children?: unknown[] }) => {
    if (node.type === 'link') {
      links.push(node)
    }

    for (const child of node.children ?? []) {
      visit(child as typeof node)
    }
  }

  visit(tree)
  expect(links, JSON.stringify(links)).toHaveLength(0)
})
