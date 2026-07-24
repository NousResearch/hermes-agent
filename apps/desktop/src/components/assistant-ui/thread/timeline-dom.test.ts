import { describe, expect, it, vi } from 'vitest'

import { activeTimelineIndexInViewport } from './timeline-dom'

function rect(top: number): DOMRect {
  return { bottom: top, height: 0, left: 0, right: 0, top, width: 0, x: 0, y: top, toJSON: () => ({}) }
}

function messageNode(id: string, top: number): HTMLElement {
  const node = document.createElement('div')
  node.dataset.messageId = id
  node.getBoundingClientRect = vi.fn(() => rect(top))

  return node
}

describe('activeTimelineIndexInViewport', () => {
  it('bounds geometry reads to the rendered-node search depth', () => {
    const viewport = document.createElement('div')
    viewport.getBoundingClientRect = vi.fn(() => rect(100))
    const unrelated = messageNode('assistant-42', 70)

    const rendered = Array.from({ length: 128 }, (_, offset) =>
      messageNode(`user-${872 + offset}`, 100 + (offset - 64) * 20)
    )

    viewport.append(unrelated, ...rendered)
    const entryIndexById = new Map(Array.from({ length: 1_000 }, (_, index) => [`user-${index}`, index]))
    const querySelectorAll = vi.spyOn(viewport, 'querySelectorAll')

    expect(activeTimelineIndexInViewport(viewport, entryIndexById)).toBe(936)
    expect(querySelectorAll).toHaveBeenCalledOnce()
    expect(unrelated.getBoundingClientRect).not.toHaveBeenCalled()

    const geometryReads = rendered.reduce(
      (total, node) => total + vi.mocked(node.getBoundingClientRect).mock.calls.length,
      0
    )

    expect(geometryReads).toBeLessThanOrEqual(8)
  })

  it('falls back to the first rendered timeline entry, then zero when none are rendered', () => {
    const viewport = document.createElement('div')
    viewport.getBoundingClientRect = vi.fn(() => rect(100))
    viewport.append(messageNode('user-8', 240), messageNode('user-9', 480))

    const entryIndexById = new Map([
      ['user-8', 8],
      ['user-9', 9]
    ])

    expect(activeTimelineIndexInViewport(viewport, entryIndexById)).toBe(8)
    expect(activeTimelineIndexInViewport(document.createElement('div'), entryIndexById)).toBe(0)
  })
})
