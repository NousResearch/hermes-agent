import remarkGfm from 'remark-gfm'
import remarkParse from 'remark-parse'
import { unified } from 'unified'
import { expect, it } from 'vitest'

import { DEFAULT_TRANSCRIPT_OPTIONS, type TranscriptDocument } from './transcript-document'
import { renderMarkdown } from './transcript-markdown'

const documentFor = (text: string): TranscriptDocument => ({
  version: 1, session_id: 's', title: 'demo', exported_at: 'now', options: DEFAULT_TRANSCRIPT_OPTIONS, warnings: [],
  messages: [{ role: 'user', speaker: 'First', text }, { role: 'assistant', speaker: 'Second', text: 'NEXT TURN' }]
})

function read(text: string) {
  const output = renderMarkdown(documentFor(text))
  const tree = unified().use(remarkParse).use(remarkGfm).parse(output)
  const nodes: { type: string; value?: string; url?: string; children?: unknown[] }[] = []

  const visit = (node: typeof nodes[number]) => {
    nodes.push(node)

    for (const child of node.children ?? []) { visit(child as typeof node) }
  }

  visit(tree)

  return { output, nodes }
}

it.each([
  ['tilde fence', '~~~js\nhttps://example.org\n~~~', 'code'],
  ['indented', '    https://example.org', 'code'],
  ['nested', '> ~~~\n> https://example.org\n> ~~~', 'code'],
  ['unclosed', '```js\nhttps://example.org', 'code']
])('preserves %s code and does not consume the next turn', (_name, text, kind) => {
  const { nodes } = read(text)
  expect(nodes.some(n => n.type === kind && n.value === 'https://example.org')).toBe(true)
  expect(nodes.some(n => n.type === 'text' && n.value === 'NEXT TURN')).toBe(true)
  expect(nodes.filter(n => n.type === 'link')).toHaveLength(0)
})

it('neutralizes untrusted HTML, image destinations and rejected links', () => {
  const { nodes } = read('<img src="https://bad.example/x">\n\n![image](https://bad.example/x) [bad](javascript:alert(1)) [good](https://safe.example)')
  expect(nodes.filter(n => n.type === 'html').every(n => !n.value?.includes('<img'))).toBe(true)
  expect(nodes.filter(n => n.type === 'image')).toHaveLength(0)
  expect(nodes.filter(n => n.type === 'link').map(n => n.url)).toEqual(['https://safe.example'])
})

it('preserves safe Markdown references while suppressing unsafe reference targets', () => {
  const { nodes } = read('[safe][ok] [bad][no]\n\n[ok]: https://safe.example\n[no]: javascript:alert(1)')
  expect(nodes.filter(n => n.type === 'link').map(n => n.url)).toEqual(['https://safe.example'])
  expect(nodes.filter(n => n.type === 'definition').map(n => n.url)).toEqual(['https://safe.example'])
})

it('keeps structured JSON literal including Markdown-like values', () => {
  const json = '{\n  "message": "**important** [label](https://example.org)"\n}'
  const output = renderMarkdown(documentFor(json))
  const page = new DOMParser().parseFromString(output, 'text/html')
  expect([...page.querySelectorAll('pre')].some(pre => pre.textContent === json)).toBe(true)
})

it.each([
  ['blockquote', '[safe][id]\n\n> [id]: https://safe.example'],
  ['list', '[safe][id]\n\n- [id]: https://safe.example']
])('resolves a safe reference defined inside a %s', (_name, input) => {
  const { nodes } = read(input)
  expect(nodes.filter(n => n.type === 'link').map(n => n.url)).toEqual(['https://safe.example'])
})

it.each([1500, 4000])('bounds %i nested quotes without losing the later prose link or next turn', depth => {
  const { nodes } = read('> '.repeat(depth) + 'hello\n\n[ok](https://safe.example)')
  expect(nodes.some(n => n.type === 'code' && n.value?.includes('hello'))).toBe(true)
  expect(nodes.filter(n => n.type === 'link').map(n => n.url)).toEqual(['https://safe.example'])
  expect(nodes.some(n => n.type === 'text' && n.value === 'NEXT TURN')).toBe(true)
})

it('preserves ordinary nested quotes instead of applying the resource fallback', () => {
  const { nodes } = read('> '.repeat(50) + 'hello')
  expect(nodes.filter(n => n.type === 'blockquote')).toHaveLength(50)
  expect(nodes.filter(n => n.type === 'code')).toHaveLength(0)
})

it('uses a bounded literal fallback for pathological multiline input', () => {
  const original = 'line\n' + '`x'.repeat(60000) + '\nhttps://example.org'
  const { output, nodes } = read(original)
  expect(output).toContain(`<pre>${original}</pre>`)
  expect(nodes.filter(n => n.type === 'link')).toHaveLength(0)
  expect(nodes.some(n => n.type === 'text' && n.value === 'NEXT TURN')).toBe(true)
})

it('bounds a deep-short 6000-quote run without losing later safe prose or the next turn', () => {
  const { nodes } = read('> '.repeat(6000) + 'hello\n\n[ok](https://safe.example)')
  expect(nodes.some(n => n.type === 'code' && n.value?.includes('> ') && n.value.includes('hello'))).toBe(true)
  expect(nodes.filter(n => n.type === 'link').map(n => n.url)).toEqual(['https://safe.example'])
  expect(nodes.some(n => n.type === 'text' && n.value === 'NEXT TURN')).toBe(true)
})

it.each([4096, 4097])('applies the delimiter budget above, not at, %i characters', count => {
  const { nodes } = read('*a'.repeat(count) + '\n\n[ok](https://safe.example)')
  expect(nodes.some(n => n.type === 'code')).toBe(count > 4096)
  expect(nodes.filter(n => n.type === 'link').map(n => n.url)).toEqual(['https://safe.example'])
})

it.each(['```', '~~~'])('keeps pathological literal content inside %s fences unchanged', fence => {
  const content = '> '.repeat(6000) + '\n' + '*a'.repeat(6000)
  const { nodes } = read(`${fence}\n${content}\n${fence}\n\n[ok](https://safe.example)`)
  expect(nodes.filter(n => n.type === 'code').map(n => n.value)).toEqual([content])
  expect(nodes.filter(n => n.type === 'link').map(n => n.url)).toEqual(['https://safe.example'])
  expect(nodes.some(n => n.type === 'text' && n.value === 'NEXT TURN')).toBe(true)
})

it.each(['```', '~~~'])('keeps a quoted %s marker from closing a top-level fence', fence => {
  const payload = `> ${fence}\n${'*'.repeat(4097)} https://example.invalid/keep`
  const { nodes } = read(`${fence}\n${payload}\n${fence}\n\n[ok](https://safe.example)`)
  expect(nodes.filter(n => n.type === 'code').map(n => n.value)).toEqual([payload])
  expect(nodes.filter(n => n.type === 'link').map(n => n.url)).toEqual(['https://safe.example'])
  expect(nodes.some(n => n.type === 'text' && n.value === 'NEXT TURN')).toBe(true)
})

it('bounds prose after leaving an unclosed quoted fence', () => {
  const { nodes } = read('> ```\n> literal\n\n' + '> '.repeat(6000) + 'tail\n\n[ok](https://safe.example)')
  expect(nodes.some(n => n.type === 'code' && String(n.value).includes('tail'))).toBe(true)
  expect(nodes.filter(n => n.type === 'link').map(n => n.url)).toEqual(['https://safe.example'])
})

it('bounds 40001 emphasis openers without losing the later prose link or next turn', () => {
  const { nodes } = read('*a'.repeat(40001) + '\n\n[ok](https://safe.example)')
  expect(nodes.some(n => n.type === 'code' && n.value?.includes('*a'))).toBe(true)
  expect(nodes.filter(n => n.type === 'link').map(n => n.url)).toEqual(['https://safe.example'])
  expect(nodes.some(n => n.type === 'text' && n.value === 'NEXT TURN')).toBe(true)
})
