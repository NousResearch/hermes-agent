import type { SyntaxHighlighterProps } from '@assistant-ui/react-streamdown'
import { render, waitFor, within } from '@testing-library/react'
import type { ComponentProps } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Mock the heavy mermaid library so tests drive render success/failure directly
const { mockRender, mockInit } = vi.hoisted(() => ({
  mockRender: vi.fn(),
  mockInit: vi.fn(),
}))

vi.mock('mermaid', () => ({
  default: {
    initialize: mockInit,
    render: mockRender,
  },
}))

import { MermaidBlock } from './mermaid-block'

function highlighterProps(code: string): SyntaxHighlighterProps {
  return {
    code,
    language: 'mermaid',
    components: { Pre: (props: ComponentProps<'pre'>) => <pre {...props} /> },
  } as unknown as SyntaxHighlighterProps
}

const SOURCE = 'graph TD; A-->B'

describe('MermaidBlock', () => {
  beforeEach(() => {
    mockRender.mockReset()
    mockInit.mockReset()
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it('renders the diagram once the source parses', async () => {
    mockRender.mockResolvedValue({ svg: '<svg data-testid="diagram">DIAGRAM</svg>' })

    const { container } = render(<MermaidBlock code={SOURCE} />)

    await waitFor(() => {
      expect(container.querySelector('[data-testid="diagram"]')).toBeTruthy()
    })
    expect(container.textContent).toContain('DIAGRAM')
  })

  it('shows the source and an error when parsing fails', async () => {
    mockRender.mockRejectedValue(new Error('Parse error on line 1'))

    const { container } = render(<MermaidBlock code="graph TD; A--" />)

    await waitFor(() => {
      expect(mockRender).toHaveBeenCalled()
    })
    // mermaid.render rejected — the fallback card should contain the source.
    // (tailwind class semantics aren't guaranteed in jsdom, so we check the
    // text content contains enough signal.)
    const text = container.textContent ?? ''
    expect(text).toMatch(/graph TD/)
    expect(text).toMatch(/render failed/)
  })

  it('renders nothing and does not call mermaid for an empty fence', () => {
    const { container } = render(<MermaidBlock code="   " />)

    expect(mockRender).not.toHaveBeenCalled()
    expect(container.querySelector('[data-testid="diagram"]')).toBeNull()
  })

  it('shows a loading placeholder while render is in flight', () => {
    // Never resolve — the promise stays pending
    mockRender.mockReturnValue(new Promise(() => {}))

    const { container } = render(<MermaidBlock code={SOURCE} />)

    expect(container.textContent).toContain('Rendering diagram')
  })
})
