import type { SyntaxHighlighterProps } from '@assistant-ui/react-streamdown'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import type { ComponentProps } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { MermaidBlock } from './mermaid-block'
import { SyntaxHighlighter } from './shiki-highlighter'

const { initialize, parse, renderDiagram } = vi.hoisted(() => ({
  initialize: vi.fn(),
  parse: vi.fn(),
  renderDiagram: vi.fn()
}))

vi.mock('mermaid', () => ({
  default: { initialize, parse, render: renderDiagram }
}))

vi.mock('react-shiki', () => ({
  default: ({ children }: { children: string }) => <pre data-testid="shiki">{children}</pre>
}))

const CHART = 'flowchart LR\n  A --> B'
const SVG = '<svg data-testid="diagram-svg"><text>diagram</text></svg>'

beforeEach(() => {
  cleanup()
  initialize.mockReset()
  parse.mockReset().mockResolvedValue(true)
  renderDiagram.mockReset().mockResolvedValue({ svg: SVG })
  delete document.documentElement.dataset.hermesMode
})

function diagramSvg(container: HTMLElement) {
  return container.querySelector('[data-slot="mermaid-block"] svg')
}

describe('MermaidBlock', () => {
  it('renders the diagram svg once mermaid settles', async () => {
    const { container } = render(<MermaidBlock chart={CHART} fallback={<div>source card</div>} />)

    expect(screen.getByText('source card')).toBeDefined()

    await waitFor(() => expect(diagramSvg(container)).not.toBeNull())
    expect(renderDiagram).toHaveBeenCalledWith(expect.stringMatching(/^aui-mermaid-/), CHART)
    expect(screen.queryByText('source card')).toBeNull()
  })

  it('keeps the fallback when the source does not parse', async () => {
    parse.mockResolvedValue(false)

    const { container } = render(<MermaidBlock chart="not a diagram" fallback={<div>source card</div>} />)

    await waitFor(() => expect(parse).toHaveBeenCalledWith('not a diagram', { suppressErrors: true }))
    expect(renderDiagram).not.toHaveBeenCalled()
    expect(diagramSvg(container)).toBeNull()
    expect(screen.getByText('source card')).toBeDefined()
  })

  it('keeps the fallback when render rejects past a successful parse', async () => {
    renderDiagram.mockRejectedValue(new Error('unsupported directive'))

    const { container } = render(<MermaidBlock chart={CHART} fallback={<div>source card</div>} />)

    await waitFor(() => expect(renderDiagram).toHaveBeenCalled())
    expect(diagramSvg(container)).toBeNull()
    expect(screen.getByText('source card')).toBeDefined()
  })

  it('follows the rendered theme from data-hermes-mode, live', async () => {
    document.documentElement.dataset.hermesMode = 'dark'

    const { container } = render(<MermaidBlock chart={CHART} />)

    await waitFor(() => expect(diagramSvg(container)).not.toBeNull())
    expect(initialize).toHaveBeenCalledWith(expect.objectContaining({ theme: 'dark' }))

    act(() => {
      document.documentElement.dataset.hermesMode = 'light'
    })

    await waitFor(() => expect(initialize).toHaveBeenCalledWith(expect.objectContaining({ theme: 'default' })))
  })
})

describe('SyntaxHighlighter mermaid fences', () => {
  const Pre = (props: ComponentProps<'pre'>) => <pre {...props} />

  function renderHighlighter(defer: boolean) {
    return render(
      <SyntaxHighlighter
        {...({ code: CHART, components: { Pre }, language: 'mermaid' } as unknown as SyntaxHighlighterProps)}
        defer={defer}
      />
    )
  }

  it('renders the diagram once streaming settles', async () => {
    const { container } = renderHighlighter(false)

    await waitFor(() => expect(diagramSvg(container)).not.toBeNull())
  })

  it('keeps the source card while streaming, no render attempt', () => {
    const { container } = renderHighlighter(true)

    expect(container.querySelector('[data-slot="code-card"]')).not.toBeNull()
    expect(container.textContent).toContain('flowchart LR')
    expect(diagramSvg(container)).toBeNull()
    expect(renderDiagram).not.toHaveBeenCalled()
  })
})
