import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { en } from '@/i18n/en'

import MermaidRenderer from './mermaid-embed'

// The renderer defers to mermaid's parse+render, which needs a real DOM layout
// pass and a browser. Stub it at the module boundary and drive the states the
// transcript actually sees: settled, streaming, and parse failure.
const renderDiagram = vi.fn()

vi.mock('mermaid', () => ({
  default: {
    initialize: vi.fn(),
    render: (...args: unknown[]) => renderDiagram(...args)
  }
}))

const SOURCE = 'flowchart LR\n  A --> B'
const SVG = '<svg viewBox="0 0 10 10"><text>A</text></svg>'

function renderFence(streaming = false) {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <MermaidRenderer code={SOURCE} streaming={streaming} />
    </I18nProvider>
  )
}

// The diagram is injected as raw SVG markup, so it is read from the rendered
// container rather than through an accessible role query.
const diagramSvg = (container: HTMLElement) => container.querySelector('svg[viewBox="0 0 10 10"]')

const sourceToggle = () => screen.queryByRole('button', { name: en.preview.source })
const diagramToggle = () => screen.queryByRole('button', { name: en.preview.renderedPreview })

afterEach(() => {
  cleanup()
  renderDiagram.mockReset()
})

describe('MermaidRenderer source view', () => {
  it('swaps the rendered diagram for the original Mermaid source and back', async () => {
    renderDiagram.mockResolvedValue({ svg: SVG })
    const { container } = renderFence()

    const toggle = await screen.findByRole('button', { name: en.preview.source })

    expect(diagramSvg(container)).not.toBeNull()
    expect(screen.queryByText(/flowchart LR/)).toBeNull()

    fireEvent.click(toggle)

    // The original text comes back verbatim — this is the whole point of the
    // view: editing it elsewhere or re-reading what the model wrote.
    expect(screen.getByText(/flowchart LR/)).toBeTruthy()
    expect(screen.getByText(/A --> B/)).toBeTruthy()
    expect(diagramSvg(container)).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: en.preview.renderedPreview }))

    expect(diagramSvg(container)).not.toBeNull()
    expect(screen.queryByText(/flowchart LR/)).toBeNull()
  })

  it('keeps the diagram viewer reachable while the source view is open, from the diagram side', async () => {
    renderDiagram.mockResolvedValue({ svg: SVG })
    renderFence()

    fireEvent.click(await screen.findByRole('button', { name: en.preview.source }))
    fireEvent.click(screen.getByRole('button', { name: en.preview.renderedPreview }))

    // Back in diagram mode the whole surface is still the zoom trigger.
    fireEvent.click(screen.getByTitle(en.desktop.openDiagram))

    await waitFor(() => expect(screen.getByRole('dialog')).toBeTruthy())
  })

  it('does not open the full-screen viewer when the toggle itself is pressed', async () => {
    renderDiagram.mockResolvedValue({ svg: SVG })
    renderFence()

    fireEvent.click(await screen.findByRole('button', { name: en.preview.source }))

    // The control sits over the zoom trigger, so it must not double as one.
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('offers no toggle while the fence is still streaming', () => {
    renderDiagram.mockResolvedValue({ svg: SVG })
    renderFence(true)

    // Mid-stream the source is what is on screen, so a toggle would be a no-op
    // on every token.
    expect(screen.getByText(/flowchart LR/)).toBeTruthy()
    expect(sourceToggle()).toBeNull()
    expect(diagramToggle()).toBeNull()
  })

  it('offers no toggle when the diagram cannot be rendered', async () => {
    renderDiagram.mockRejectedValue(new Error('parse error'))
    renderFence()

    // Failure already falls back to the source; toggling to a diagram that does
    // not exist would be a dead end.
    await waitFor(() => expect(screen.getByText(/flowchart LR/)).toBeTruthy())

    expect(sourceToggle()).toBeNull()
    expect(diagramToggle()).toBeNull()
  })

  it('offers no toggle while the diagram render is still pending', () => {
    // A promise that never settles: the fence is settled, but no SVG exists yet.
    renderDiagram.mockImplementation(() => new Promise(() => {}))

    renderFence()

    expect(screen.getByText(/flowchart LR/)).toBeTruthy()
    expect(sourceToggle()).toBeNull()
    expect(diagramToggle()).toBeNull()
  })

  it('comes back in diagram view when the fence content changes', async () => {
    renderDiagram.mockResolvedValue({ svg: SVG })
    const view = renderFence()

    fireEvent.click(await screen.findByRole('button', { name: en.preview.source }))
    expect(screen.getByText(/flowchart LR/)).toBeTruthy()

    // Same component instance, different fence: regenerated message, or a row
    // recycled by the transcript list. The open source view must not carry over.
    view.rerender(
      <I18nProvider configClient={null} initialLocale="en">
        <MermaidRenderer code={'graph TD\n  X --> Y'} />
      </I18nProvider>
    )

    await waitFor(() => expect(screen.queryByText(/flowchart LR/)).toBeNull())
    expect(screen.getByRole('button', { name: en.preview.source })).toBeTruthy()
  })

  it('labels the toggle from the localized catalog', async () => {
    renderDiagram.mockResolvedValue({ svg: SVG })
    render(
      <I18nProvider configClient={null} initialLocale="zh-hant">
        <MermaidRenderer code={SOURCE} />
      </I18nProvider>
    )

    const toggle = await screen.findByRole('button', { name: '原始碼' })

    // The zoom trigger beside it reads the same catalog, so a hardcoded English
    // label on that trigger would fail this lookup.
    expect(screen.getByTitle('開啟圖表')).toBeTruthy()

    fireEvent.click(toggle)

    expect(screen.getByRole('button', { name: '預覽' })).toBeTruthy()
  })
})
