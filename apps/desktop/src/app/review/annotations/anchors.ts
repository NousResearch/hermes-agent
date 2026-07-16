import type { TextPosition } from '@/store/annotations'

export function contentFingerprint(content: string): string {
  let hash = 0x811c9dc5

  for (let index = 0; index < content.length; index += 1) {
    hash ^= content.charCodeAt(index)
    hash = Math.imul(hash, 0x01000193)
  }

  return `fnv1a-${(hash >>> 0).toString(16).padStart(8, '0')}-${content.length}`
}

function textNodes(container: Node): Text[] {
  const owner = container.ownerDocument ?? document
  const walker = owner.createTreeWalker(container, NodeFilter.SHOW_TEXT)
  const nodes: Text[] = []
  let node = walker.nextNode()

  while (node) {
    nodes.push(node as Text)
    node = walker.nextNode()
  }

  return nodes
}

function nodePath(container: Node, node: Node): number[] | undefined {
  const path: number[] = []
  let cursor: Node | null = node

  while (cursor && cursor !== container) {
    const parent: Node | null = cursor.parentNode

    if (!parent) {
      return undefined
    }

    path.unshift(Array.prototype.indexOf.call(parent.childNodes, cursor) as number)
    cursor = parent
  }

  return cursor === container ? path : undefined
}

function nodeAtPath(container: Node, path?: number[]): Node | null {
  if (!path) {
    return null
  }

  let cursor: Node = container

  for (const index of path) {
    const next = cursor.childNodes[index]

    if (!next) {
      return null
    }

    cursor = next
  }

  return cursor
}

function offsetInText(container: Node, target: Node, offset: number): number | null {
  let total = 0

  for (const node of textNodes(container)) {
    if (node === target) {
      return total + offset
    }

    total += node.data.length
  }

  return null
}

function rangeFromOffsets(container: Node, start: number, end: number): Range | null {
  const owner = container.ownerDocument ?? document
  const range = owner.createRange()
  let total = 0
  let startNode: Text | null = null
  let startOffset = 0

  for (const node of textNodes(container)) {
    const next = total + node.data.length

    if (!startNode && start < next) {
      startNode = node
      startOffset = Math.max(0, start - total)
    }

    if (startNode && end <= next) {
      range.setStart(startNode, Math.min(startOffset, startNode.data.length))
      range.setEnd(node, Math.max(0, end - total))

      return range
    }

    total = next
  }

  return null
}

export function captureTextPosition(container: Node, range: Range): TextPosition | null {
  if (!container.contains(range.startContainer) || !container.contains(range.endContainer)) {
    return null
  }

  const quote = range.toString()
  const startOffset = offsetInText(container, range.startContainer, range.startOffset)
  const endOffset = offsetInText(container, range.endContainer, range.endOffset)

  if (!quote.trim() || startOffset == null || endOffset == null) {
    return null
  }

  const fullText = container.textContent ?? ''

  return {
    endOffset,
    endNodeOffset: range.endOffset,
    endPath: nodePath(container, range.endContainer),
    prefix: fullText.slice(Math.max(0, startOffset - 48), startOffset),
    quote,
    startOffset,
    startNodeOffset: range.startOffset,
    startPath: nodePath(container, range.startContainer),
    suffix: fullText.slice(endOffset, endOffset + 48)
  }
}

function normalizeWithMap(text: string): { map: number[]; text: string } {
  const map: number[] = []
  let output = ''
  let whitespace = false

  for (let index = 0; index < text.length; index += 1) {
    if (/\s/.test(text[index])) {
      if (!whitespace) {
        output += ' '
        map.push(index)
      }

      whitespace = true
    } else {
      output += text[index]
      map.push(index)
      whitespace = false
    }
  }

  return { map, text: output }
}

export function restoreTextPosition(container: Node, position: TextPosition): Range | null {
  const owner = container.ownerDocument ?? document
  const startNode = nodeAtPath(container, position.startPath)
  const endNode = nodeAtPath(container, position.endPath)

  if (startNode?.nodeType === Node.TEXT_NODE && endNode?.nodeType === Node.TEXT_NODE) {
    try {
      const exact = owner.createRange()
      exact.setStart(startNode, Math.min(position.startNodeOffset ?? 0, startNode.textContent?.length ?? 0))
      exact.setEnd(endNode, Math.min(position.endNodeOffset ?? 0, endNode.textContent?.length ?? 0))

      if (exact.toString() === position.quote) {
        return exact
      }
    } catch {
      // The structural path changed; continue through content fallbacks.
    }
  }

  const byOffset = rangeFromOffsets(container, position.startOffset, position.endOffset)

  if (byOffset?.toString() === position.quote) {
    return byOffset
  }

  const fullText = container.textContent ?? ''
  const matches: number[] = []
  let cursor = fullText.indexOf(position.quote)

  while (cursor >= 0) {
    matches.push(cursor)
    cursor = fullText.indexOf(position.quote, cursor + 1)
  }

  if (matches.length > 0) {
    const scored = matches
      .map(start => {
        const prefix = position.prefix ?? ''
        const suffix = position.suffix ?? ''
        const prefixScore = prefix && fullText.slice(Math.max(0, start - prefix.length), start) === prefix ? 2 : 0
        const end = start + position.quote.length
        const suffixScore = suffix && fullText.slice(end, end + suffix.length) === suffix ? 2 : 0
        const distance = Math.abs(start - position.startOffset) / Math.max(1, fullText.length)

        return { score: prefixScore + suffixScore - distance, start }
      })
      .sort((left, right) => right.score - left.score)

    const best = scored[0]

    if (scored.length === 1 || best.score > scored[1].score) {
      return rangeFromOffsets(container, best.start, best.start + position.quote.length)
    }
  }

  const haystack = normalizeWithMap(fullText)
  const needle = normalizeWithMap(position.quote).text.trim()
  const normalizedStart = needle ? haystack.text.indexOf(needle) : -1

  if (normalizedStart >= 0) {
    const start = haystack.map[normalizedStart]
    const end = haystack.map[normalizedStart + needle.length - 1] + 1

    return rangeFromOffsets(container, start, end)
  }

  return null
}
