import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { requestModelOptions } from '@/lib/model-options'
import { $visibleModels } from '@/store/model-visibility'

import { ModelVisibilityDialog } from './model-visibility-dialog'

vi.mock('@/lib/model-options', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestModelOptions: vi.fn()
}))

beforeEach(() => {
  $visibleModels.set(null)
  vi.mocked(requestModelOptions).mockResolvedValue({
    providers: [
      {
        models: ['cc/claude-opus-5', 'claude/claude-opus-5'],
        name: 'OmniRoute',
        slug: 'omniroute'
      }
    ]
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('ModelVisibilityDialog', () => {
  it('associates the exact model id tooltip with the focusable visibility switch', async () => {
    vi.mocked(requestModelOptions).mockResolvedValue({
      providers: [
        {
          models: ['zai/glm-5.3'],
          name: 'OmniRoute',
          slug: 'omniroute'
        }
      ]
    })
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

    render(
      <QueryClientProvider client={client}>
        <I18nProvider>
          <ModelVisibilityDialog
            onOpenChange={() => undefined}
            onOpenProviders={() => undefined}
            open
          />
        </I18nProvider>
      </QueryClientProvider>
    )

    await screen.findByText(
      (_, element) => element?.classList.contains('truncate') === true && element.textContent === 'Glm 5.3 · zai'
    )

    const toggle = screen.getByRole('switch', { name: 'zai/glm-5.3' })

    expect(toggle.getAttribute('data-slot')).toBe('tooltip-trigger')
    expect(toggle.tabIndex).toBe(0)
    expect(toggle.getAttribute('title')).toBeNull()

    toggle.focus()
    expect(document.activeElement).toBe(toggle)

    fireEvent.pointerMove(toggle, { pointerType: 'mouse' })
    expect((await screen.findByRole('tooltip')).textContent).toBe('zai/glm-5.3')
  })
})
