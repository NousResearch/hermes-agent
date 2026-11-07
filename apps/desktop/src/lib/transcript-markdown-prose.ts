import type { Nodes, PhrasingContent, Root } from 'mdast'
import remarkGfm from 'remark-gfm'
import remarkParse from 'remark-parse'
import remarkStringify from 'remark-stringify'
import { unified } from 'unified'

import { escapeTranscriptXml as esc } from './transcript-media'
import { safeTranscriptLink, transcriptTextParts } from './transcript-text'

const parser = unified().use(remarkParse).use(remarkGfm).use(remarkStringify, {
  fences: true,
  resourceLink: true,
  handlers: {
    link(node, _parent, state, info) {
      const label = state.containerPhrasing(node, { ...info, before: '[', after: '](' })

      return `[${label}](<${esc(node.url)}>)`
    }
  }
})

const unsafeAutolink = /(?:\bwww\.|[\w.+-]+@[\w.-]+\.[a-z]{2,}|https?:\/\/)/i

// The GFM parser can be superlinear on enormous delimiter-heavy input. Keep such
// messages literal rather than letting untrusted content monopolize the export.
const MAX_MARKDOWN_INPUT = 100_000
const MAX_MARKDOWN_DEPTH = 128
// Bound parser work before its recursive tree walk; deeper quotes become literal.
const MAX_PARSER_QUOTE_DEPTH = MAX_MARKDOWN_DEPTH
// Count delimiter characters, not bytes or parsed nodes, before invoking GFM.
const MAX_LINE_DELIMITERS = 4096
const flowParents = new Set(['root', 'blockquote', 'listItem', 'footnoteDefinition'])
const inlineParents = new Set(['paragraph', 'heading', 'emphasis', 'strong', 'delete', 'link', 'linkReference', 'tableCell'])

function literalCode(value: string): PhrasingContent {
  return { type: 'inlineCode', value: value.replace(/[\r\n]+/g, ' ') }
}

function sourcePart(node: Nodes, source: string): string {
  const start = node.position?.start.offset ?? 0

  return source.slice(start, node.position?.end.offset ?? start)
}

function literalPart(node: Nodes, source: string, parent: Nodes): Nodes {
  const value = sourcePart(node, source)

  return flowParents.has(parent.type) ? { type: 'code', value } : literalCode(value)
}

function definitionsFor(tree: Root): Map<string, string> {
  const definitions = new Map<string, string>()
  const pending: Nodes[] = [tree]

  while (pending.length) {
    const node = pending.pop()!

    if (node.type === 'definition' && !definitions.has(node.identifier)) {
      definitions.set(node.identifier, node.url)
    }

    if ('children' in node) {
      for (let i = node.children.length - 1; i >= 0; i--) { pending.push(node.children[i]) }
    }
  }

  return definitions
}

// Bound the library's recursive serializer, not just the character count.
// Only the over-deep branch becomes literal; unrelated prose links still work.
function boundNesting(tree: Root, source: string): void {
  const pending: { node: Nodes; depth: number }[] = [{ node: tree, depth: 0 }]

  while (pending.length) {
    const { node, depth } = pending.pop()!

    if (!('children' in node)) { continue }
    const children = node.children as Nodes[]

    for (let i = 0; i < children.length; i++) {
      const child = children[i]

      if (depth >= MAX_MARKDOWN_DEPTH && 'children' in child && (flowParents.has(node.type) || inlineParents.has(node.type))) {
        children[i] = literalPart(child, source, node)
      } else {
        pending.push({ node: child, depth: depth + 1 })
      }
    }
  }
}

function proseParts(value: string): PhrasingContent[] {
  return transcriptTextParts(value, true).flatMap(part => {
    if (part.href) {
      return [{ type: 'link', url: part.href, children: [{ type: 'text', value: part.text }] } as PhrasingContent]
    }

    return part.text.split(/(\n)/).map(piece => unsafeAutolink.test(piece) ? literalCode(piece) : { type: 'text', value: piece })
  })
}

function sanitize(tree: Root, value: string, definitions: Map<string, string>): void {
  function link(child: Extract<Nodes, { type: 'link' }>): Nodes {
    const raw = sourcePart(child, value)
    const start = child.position?.start.offset ?? 0
    const prefix = value.slice(value.lastIndexOf('\n', start - 1) + 1, start)
    const malformed = !raw.startsWith('[') && /\[[^\]\n]*\]\([^)]*$/.test(prefix)

    if (!safeTranscriptLink(child.url) || malformed) { return literalCode(raw) }
    transform(child, true)

    return child
  }

  function reference(child: Extract<Nodes, { type: 'linkReference' }>, parent: Nodes): Nodes {
    const href = definitions.get(child.identifier) ?? ''

    if (!safeTranscriptLink(href)) { return literalPart(child, value, parent) }
    transform(child, true)

    return { type: 'link', url: href, children: child.children }
  }

  function html(child: Extract<Nodes, { type: 'html' }>, parent: Nodes): Nodes {
    return unsafeAutolink.test(child.value)
      ? literalPart(child, value, parent)
      : { type: 'html', value: esc(child.value) }
  }

  function transform(node: Nodes, inLink = false): void {
    if (!('children' in node)) { return }
    const result: Nodes[] = []

    for (const child of node.children) {
      switch (child.type) {
        case 'text':
          result.push(...(inLink ? [child] : proseParts(child.value)))

          break

        case 'html':
          result.push(html(child, node))

          break

        case 'definition':
          result.push(safeTranscriptLink(child.url) ? child : literalPart(child, value, node))

          break

        case 'linkReference':
          result.push(reference(child, node))

          break

        case 'image':

        case 'imageReference':
          result.push(literalPart(child, value, node))

          break

        case 'link':
          result.push(link(child))

          break

        default:
          transform(child)
          result.push(child)
      }
    }

    ;(node as { children: Nodes[] }).children = result
  }

  transform(tree)
}

function quotePrefix(line: string): { depth: number; content: string } {
  let i = 0
  let depth = 0

  while (i < line.length) {
    let spaces = 0

    while (spaces < 4 && line[i + spaces] === ' ') { spaces++ }
    if (spaces === 4 || line[i + spaces] !== '>') { break }
    i += spaces + 1
    depth++
    if (line[i] === ' ') { i++ }
  }

  return { depth, content: line.slice(i) }
}

function fenceAt(content: string): { marker: '`' | '~'; length: number; rest: string } | undefined {
  let spaces = 0

  while (spaces < 4 && content[spaces] === ' ') { spaces++ }
  if (spaces === 4) { return undefined }
  const marker = content[spaces]

  if (marker !== '`' && marker !== '~') { return undefined }
  let length = 0

  while (content[spaces + length] === marker) { length++ }
  if (length < 3) { return undefined }

  return { marker, length, rest: content.slice(spaces + length) }
}

// Block fences are opaque to the parser. Do not count or rewrite their contents,
// and close only on a same-depth, same-marker line long enough to end the fence.
function fenceStep(content: string, open: { marker: '`' | '~'; length: number } | undefined) {
  const found = fenceAt(content)

  if (!found) { return open }
  if (open && found.marker === open.marker && found.length >= open.length && found.rest.trim() === '') { return undefined }
  if (!open && (found.marker !== '`' || !found.rest.includes('`'))) { return { marker: found.marker, length: found.length } }

  return open
}

function fencedLiteral(value: string): string {
  let max = 2

  for (const match of value.matchAll(/`+/g)) { max = Math.max(max, match[0].length) }
  const ticks = '`'.repeat(max + 1)

  return `${ticks}\n${value}\n${ticks}`
}

function delimiterPressure(line: string): boolean {
  let count = 0

  for (const char of line) {
    if ('*_~[]'.includes(char) && ++count > MAX_LINE_DELIMITERS) { return true }
  }

  return false
}

// Only pathological lines lose formatting; later prose remains on the library path.
function boundParserInput(value: string): string {
  let fence: { marker: '`' | '~'; length: number; depth: number } | undefined
  const lines = value.split('\n')
  let changed = false

  for (let i = 0; i < lines.length; i++) {
    const cr = lines[i].endsWith('\r')
    const line = cr ? lines[i].slice(0, -1) : lines[i]
    const { depth, content } = quotePrefix(line)

    if (fence && depth < fence.depth) { fence = undefined }
    if (fence) {
      // A quoted marker is another container, not a closer for a shallower fence.
      if (depth === fence.depth) { fence = fenceStep(content, fence) && fence }
      continue
    }

    const opening = fenceStep(content, undefined)

    if (depth > MAX_PARSER_QUOTE_DEPTH || (!opening && delimiterPressure(line))) {
      lines[i] = fencedLiteral(line) + (cr ? '\r' : '')
      changed = true
      continue
    }

    fence = opening && { ...opening, depth }
  }

  return changed ? lines.join('\n') : value
}

export function markdownProse(value: string): string {
  if (value.length > MAX_MARKDOWN_INPUT) {
    if (/[\r\n]/.test(value)) { return `<pre>${esc(value)}</pre>` }
    let max = 0

    for (const match of value.matchAll(/`+/g)) { max = Math.max(max, match[0].length) }
    const ticks = '`'.repeat(max + 1)

    return `${ticks} ${value} ${ticks}`
  }

  const source = boundParserInput(value)
  const tree = parser.parse(source)
  const definitions = definitionsFor(tree)
  boundNesting(tree, source)
  sanitize(tree, source, definitions)

  return String(parser.stringify(tree))
}
