import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import MermaidRenderer from './mermaid-embed'

vi.mock('mermaid', () => ({
  default: {
    initialize: vi.fn(),
    render: vi.fn().mockResolvedValue({
      svg: '<svg role="img" viewBox="0 0 10 10"><text>A</text></svg>'
    })
  }
}))

describe('MermaidRenderer', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('copies the original Mermaid source from the rendered diagram', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    const code = 'graph TD\n  A --> B'

    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText }
    })

    render(
      <I18nProvider configClient={null}>
        <MermaidRenderer code={code} />
      </I18nProvider>
    )

    const copyButton = await screen.findByRole('button', { name: 'Copy Mermaid source' })

    fireEvent.click(copyButton)

    await waitFor(() => expect(writeText).toHaveBeenCalledWith(code))
  })
})
