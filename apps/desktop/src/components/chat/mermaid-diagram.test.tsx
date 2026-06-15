import type { SyntaxHighlighterProps } from '@assistant-ui/react-streamdown'
import { render, waitFor, within } from '@testing-library/react'
import type { ComponentProps } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Mock the heavy mermaid plugin so tests drive render success/failure directly
// and never touch the real (DOM-measuring) mermaid library.
const { mockRender } = vi.hoisted(() => ({ mockRender: vi.fn() }))

vi.mock('@streamdown/mermaid', () => ({
  mermaid: { getMermaid: () => ({ render: mockRender }) }
}))

import { MermaidDiagram } from './mermaid-diagram'

function highlighterProps(code: string): SyntaxHighlighterProps {
  return {
    code,
    language: 'mermaid',
    components: { Pre: (props: ComponentProps<'pre'>) => <pre {...props} /> }
  } as unknown as SyntaxHighlighterProps
}

const SOURCE = 'graph TD; A-->B'

describe('MermaidDiagram', () => {
  beforeEach(() => {
    mockRender.mockReset()
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it('renders the diagram once the source parses', async () => {
    mockRender.mockResolvedValue({ svg: '<svg data-testid="diagram">DIAGRAM</svg>' })

    const { container } = render(<MermaidDiagram {...highlighterProps(SOURCE)} />)

    await waitFor(() => {
      expect(container.querySelector('[role="img"]')).toBeTruthy()
    })
    expect(container.querySelector('[data-testid="diagram"]')).toBeTruthy()
    expect(container.textContent).toContain('DIAGRAM')
  })

  it('falls back to the source (no diagram) while streaming', async () => {
    const { container } = render(<MermaidDiagram {...highlighterProps(SOURCE)} defer />)

    // Defer short-circuits before any render call.
    expect(mockRender).not.toHaveBeenCalled()
    expect(container.querySelector('[role="img"]')).toBeNull()
    expect(within(container).getByText(/graph TD/)).toBeTruthy()
  })

  it('shows an error badge and the source when the source fails to parse', async () => {
    mockRender.mockRejectedValue(new Error('Parse error on line 1'))

    const { container } = render(<MermaidDiagram {...highlighterProps('graph TD; A--')} />)

    await waitFor(() => {
      expect(mockRender).toHaveBeenCalled()
    })
    await waitFor(() => {
      // No diagram, but a visible alert explaining why — not a silent fall-through.
      expect(container.querySelector('[role="img"]')).toBeNull()
      const alert = container.querySelector('[role="alert"]')
      expect(alert).toBeTruthy()
      expect(alert?.textContent).toContain('Parse error on line 1')
      expect(container.textContent).toContain('graph TD')
    })
  })

  it('falls back (no diagram, no render call) for an empty fence', () => {
    const { container } = render(<MermaidDiagram {...highlighterProps('   ')} />)

    expect(mockRender).not.toHaveBeenCalled()
    expect(container.querySelector('[role="img"]')).toBeNull()
  })
})
