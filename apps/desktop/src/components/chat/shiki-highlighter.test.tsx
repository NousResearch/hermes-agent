import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { ComponentProps } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { chunkByLines, exceedsHighlightBudget, SyntaxHighlighter } from '@/components/chat/shiki-highlighter'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

const Pre = (props: ComponentProps<'pre'>) => <pre {...props} />

describe('exceedsHighlightBudget', () => {
  it('highlights normal-sized blocks', () => {
    expect(exceedsHighlightBudget('const x = 1\n'.repeat(100))).toBe(false)
  })

  it('skips highlighting past the line budget', () => {
    expect(exceedsHighlightBudget('x\n'.repeat(5_000))).toBe(true)
  })

  it('skips highlighting past the char budget on few lines', () => {
    expect(exceedsHighlightBudget('a'.repeat(200_000))).toBe(true)
  })

  it('short-circuits on char budget before line loop', () => {
    expect(exceedsHighlightBudget('y\n'.repeat(250_000))).toBe(true)
  })
})

describe('chunkByLines', () => {
  it('keeps a small block as a single chunk', () => {
    const code = 'a\nb\nc'
    expect(chunkByLines(code, 200)).toEqual([{ text: code, lines: 3 }])
  })

  it('splits a large block and reconstructs it losslessly', () => {
    const code = Array.from({ length: 1000 }, (_, i) => `line ${i}`).join('\n')
    const chunks = chunkByLines(code, 200)

    expect(chunks).toHaveLength(5)
    expect(chunks.map(chunk => chunk.text).join('\n')).toBe(code)
    expect(chunks.reduce((sum, chunk) => sum + chunk.lines, 0)).toBe(1000)
  })
})

describe('SyntaxHighlighter layout', () => {
  it('grows with a long fenced block while keeping horizontal scrolling', () => {
    const code = Array.from({ length: 1_000 }, (_, i) => `const value${i} = '${'x'.repeat(200)}'`).join('\n')

    const { container } = render(
      <SyntaxHighlighter code={code} components={{ Pre } as never} defer language="typescript" />
    )

    const pre = container.querySelector('pre')

    expect(pre).not.toBeNull()
    expect(pre?.className).toContain('max-h-none')
    expect(pre?.className).toContain('overflow-x-auto')
    expect(pre?.className).toContain('overflow-y-visible')
    expect(container.querySelector('.max-h-\\[7\\.5rem\\]')).toBeNull()
    expect(container.querySelector('.max-h-\\[40dvh\\]')).toBeNull()
    expect(screen.queryByRole('button', { name: 'Expand' })).toBeNull()
  })

  it('copies the complete untruncated source', async () => {
    const code = Array.from({ length: 1_000 }, (_, i) => `line ${i}`).join('\n')
    const writeText = vi.fn().mockResolvedValue(undefined)

    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText }
    })

    render(<SyntaxHighlighter code={code} components={{ Pre } as never} defer language="typescript" />)
    fireEvent.click(screen.getByRole('button', { name: /copy code/i }))

    await vi.waitFor(() => expect(writeText).toHaveBeenCalledWith(code))
  })
})
