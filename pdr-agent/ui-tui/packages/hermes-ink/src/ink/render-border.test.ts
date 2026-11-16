import { describe, expect, it } from 'vitest'

import type { DOMNode } from './dom.js'
import Output from './output.js'
import renderBorder from './render-border.js'
import { cellAt, CellWidth, CharPool, createScreen, HyperlinkPool, type Screen, StylePool } from './screen.js'

const WIDTH = 12
const HEIGHT = 6

function createOutput() {
  const stylePool = new StylePool()
  const screen = createScreen(WIDTH, HEIGHT, stylePool, new CharPool(), new HyperlinkPool())

  return { output: new Output({ height: HEIGHT, screen, stylePool, width: WIDTH }), stylePool }
}

function borderNode(style: Record<string, unknown> = {}, width = 8, height = 4): DOMNode {
  return {
    style: { borderStyle: 'single', ...style },
    yogaNode: {
      getComputedHeight: () => height,
      getComputedWidth: () => width
    }
  } as unknown as DOMNode
}

function snapshot(screen: Screen, styles: StylePool) {
  return Array.from({ length: screen.height }, (_, y) =>
    Array.from({ length: screen.width }, (_, x) => {
      const cell = cellAt(screen, x, y)!

      return [cell.char, cell.width, styles.get(cell.styleId).map(code => code.code)] as const
    })
  )
}

function paint(
  node: DOMNode,
  visible: readonly [number, number, number, number] = [0, WIDTH, 0, HEIGHT],
  decorate?: (output: Output, phase: 'before' | 'after') => void
) {
  const { output, stylePool } = createOutput()

  decorate?.(output, 'before')
  renderBorder(1, 1, node, output, ...visible)
  decorate?.(output, 'after')

  return { cells: snapshot(output.get(), stylePool), stylePool }
}

function expectClippedParity(
  full: ReturnType<typeof paint>['cells'],
  clipped: ReturnType<typeof paint>['cells'],
  [x1, x2, y1, y2]: readonly [number, number, number, number]
) {
  for (let y = 0; y < HEIGHT; y++) {
    for (let x = 0; x < WIDTH; x++) {
      if (x >= x1 && x < x2 && y >= y1 && y < y2) {
        expect(clipped[y]![x]).toEqual(full[y]![x])
      } else {
        expect(clipped[y]![x]![0]).toBe(' ')
      }
    }
  }
}

function expectWideTextClippedParity(
  full: ReturnType<typeof paint>['cells'],
  clipped: ReturnType<typeof paint>['cells'],
  visible: readonly [number, number, number, number]
) {
  const [x1, x2, y1, y2] = visible

  for (let y = 0; y < HEIGHT; y++) {
    for (let x = 0; x < WIDTH; x++) {
      const fullCell = full[y]![x]!
      const isVisible = x >= x1 && x < x2 && y >= y1 && y < y2
      const clipsWideHead = isVisible && fullCell[1] === CellWidth.Wide && x + 1 >= x2
      const clipsWideTail = isVisible && fullCell[1] === CellWidth.SpacerTail && x - 1 < x1

      if (clipsWideHead || clipsWideTail) {
        expect(clipped[y]![x]).toEqual([' ', CellWidth.Narrow, []])
      } else if (isVisible) {
        expect(clipped[y]![x]).toEqual(fullCell)
      } else {
        expect(clipped[y]![x]![0]).toBe(' ')
      }
    }
  }
}

describe('renderBorder viewport parity', () => {
  it.each([
    ['all edges', {}],
    ['disabled edges', { borderLeft: false, borderTop: false }]
  ])('matches the unclipped border inside a clipped viewport with %s', (_name, style) => {
    const visible = [3, 8, 1, 4] as const
    const full = paint(borderNode(style)).cells
    const clipped = paint(borderNode(style), visible).cells

    expectClippedParity(full, clipped, visible)
  })

  it.each([
    ['left edge at wide title', [3, 9, 1, 2]],
    ['left edge bisects wide title', [4, 9, 1, 2]],
    ['left edge after wide grapheme', [5, 9, 1, 2]],
    ['right edge bisects wide title', [1, 4, 1, 2]],
    ['right edge after wide grapheme', [1, 5, 1, 2]],
    ['right edge after narrow suffix', [1, 6, 1, 2]]
  ] as const)('preserves ANSI wide-title cell coordinates when the %s', (_name, visible) => {
    const node = borderNode({
      borderText: {
        align: 'center',
        content: '\u001B[31m界A\u001B[39m',
        position: 'top'
      }
    })

    const full = paint(node).cells
    const clipped = paint(node, visible).cells

    expectWideTextClippedParity(full, clipped, visible)
    expect(full[1]!.some(([char]) => char === '界')).toBe(true)
    expect(full[1]!.some(([char, , codes]) => char === '界' && codes.includes('\u001B[31m'))).toBe(true)
    expect(full[1]![6]![2]).not.toContain('\u001B[31m')

    if (visible[0] === 4) {
      expect(clipped[1]![4]).toEqual([' ', CellWidth.Narrow, []])
      expect(clipped[1]![5]![0]).toBe('A')
    }
  })

  it.each([
    ['left edge inside title', [4, 9, 1, 2]],
    ['right edge inside title', [1, 5, 1, 2]]
  ] as const)('preserves narrow ANSI-title parity when the %s', (_name, visible) => {
    const node = borderNode({
      borderText: {
        align: 'center',
        content: '\u001B[32mABC\u001B[39m',
        position: 'top'
      }
    })

    const full = paint(node).cells
    const clipped = paint(node, visible).cells

    expectClippedParity(full, clipped, visible)
  })

  it('intersects nested output clips without changing surviving border cells', () => {
    const node = borderNode()
    const full = paint(node).cells
    const { output, stylePool } = createOutput()

    output.clip({ x1: 0, x2: 10, y1: 0, y2: 5 })
    output.clip({ x1: 1, x2: 7, y1: 1, y2: 4 })
    renderBorder(1, 1, node, output)
    output.unclip()
    output.unclip()

    const nested = snapshot(output.get(), stylePool)

    expectClippedParity(full, nested, [1, 7, 1, 4])
  })

  it('renders borders after opaque fills and preserves the fill interior', () => {
    const decorated = paint(borderNode(), [0, WIDTH, 0, HEIGHT], (output, phase) => {
      if (phase === 'before') {
        const line = '\u001B[44m        \u001B[49m'

        output.write(1, 1, [line, line, line, line].join('\n'))
      }
    })

    expect(decorated.cells[1]![1]![0]).toBe('┌')
    expect(decorated.cells[4]![8]![0]).toBe('┘')
    expect(decorated.cells[2]![2]![2]).toContain('\u001B[44m')
  })

  it('keeps clipped border parity when an absolute-style overlay paints afterward', () => {
    const overlay = (output: Output, phase: 'before' | 'after') => {
      if (phase === 'after') {
        output.clip({ x1: 4, x2: 6, y1: 1, y2: 2 })
        output.write(4, 1, 'OV')
        output.unclip()
      }
    }

    const visible = [3, 7, 1, 4] as const
    const full = paint(borderNode(), [0, WIDTH, 0, HEIGHT], overlay).cells
    const clipped = paint(borderNode(), visible, overlay).cells

    expectClippedParity(full, clipped, visible)
    expect(clipped[1]![4]![0]).toBe('O')
    expect(clipped[1]![5]![0]).toBe('V')
  })
})
