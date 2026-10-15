import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $activeGatewayProfile } from '@/store/profile'

import { WorkhorseReasoningPill } from './workhorse-reasoning-pill'

const mocks = vi.hoisted(() => ({
  config: {} as Record<string, unknown>,
  save: vi.fn(),
  cacheWrite: vi.fn(),
  previous: { delegation: {} as Record<string, unknown> },
  notifyError: vi.fn(),
  onSetOptionsRef: {
    current: null as null | ((patch: { effort?: string; fast?: boolean }) => void)
  },
  onOpenChangeRef: { current: null as null | ((open: boolean) => void) },
  releaseTypingFocus: vi.fn(),
  capabilities: { reasoning: true, can_disable_reasoning: true }
}))

vi.mock('@/components/ui/keyboard-first', () => ({
  releaseTypingFocus: () => mocks.releaseTypingFocus()
}))

vi.mock('@/hermes', async importOriginal => {
  const actual = await importOriginal<Record<string, unknown>>()

  return {
    ...actual,
    saveHermesConfig: (config: Record<string, unknown>, profile?: unknown) => mocks.save(config, profile)
  }
})

vi.mock('@/app/hooks/use-config-record', () => ({
  useHermesConfigRecord: () => ({ data: mocks.config }),
  hermesConfigCacheWriter: () => (patch: unknown) => {
    const fn = patch as (prev: unknown) => unknown
    mocks.cacheWrite(fn(mocks.previous))
  }
}))

vi.mock('@/lib/model-options', () => ({
  currentModelCapabilities: () => mocks.capabilities,
  modelOptionsQueryKey: () => ['workhorse-model-options'],
  requestModelOptions: () => Promise.resolve({ providers: [] })
}))

vi.mock('@/app/shell/model-edit-submenu', () => ({
  ModelOptionsContent: ({
    onSetOptions
  }: {
    onSetOptions: (patch: { effort?: string; fast?: boolean }) => void
  }) => {
    mocks.onSetOptionsRef.current = onSetOptions

    return <div data-testid="workhorse-options-content" />
  }
}))

vi.mock('@/components/ui/dropdown-menu', () => ({
  DropdownMenu: ({ children, onOpenChange }: { children: React.ReactNode; onOpenChange?: (open: boolean) => void }) => {
    mocks.onOpenChangeRef.current = onOpenChange ?? null

    return <>{children}</>
  },
  DropdownMenuContent: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  DropdownMenuTrigger: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock('@/components/ui/tooltip', () => ({
  Tip: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock('@/store/notifications', () => ({
  notifyError: (...args: unknown[]) => mocks.notifyError(...args)
}))

afterEach(() => {
  cleanup()
  mocks.save.mockClear()
  mocks.cacheWrite.mockClear()
  mocks.notifyError.mockClear()
  mocks.releaseTypingFocus.mockClear()
  mocks.onOpenChangeRef.current = null
  mocks.capabilities = { reasoning: true, can_disable_reasoning: true }
  mocks.config = {}
  $activeGatewayProfile.set('default')
})

beforeEach(() => {
  mocks.previous = { delegation: { model: '', provider: '', reasoning_effort: '' } }
})

function renderPill(over: Partial<{ disabled: boolean }> = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return render(
    <QueryClientProvider client={client}>
      <I18nProvider configClient={null} initialLocale="en">
        <WorkhorseReasoningPill disabled={false} {...over} />
      </I18nProvider>
    </QueryClientProvider>
  )
}

describe('WorkhorseReasoningPill', () => {
  it('shows the pinned delegation effort level', () => {
    mocks.config = { delegation: { model: 'm', provider: 'p', reasoning_effort: 'high' } }
    renderPill()

    expect(screen.getByTestId('workhorse-reasoning-pill').textContent).toContain('High')
  })

  it('shows inherit-parent-effort when the delegation slot has no reasoning level', () => {
    mocks.config = { delegation: { model: 'm', provider: 'p' } }
    renderPill()

    expect(screen.getByTestId('workhorse-reasoning-pill').textContent).toContain('inherit')
  })

  it('normalizes a YAML false reasoning effort to Off', () => {
    mocks.config = { delegation: { model: 'm', provider: 'p', reasoning_effort: false } }
    renderPill()

    expect(screen.getByTestId('workhorse-reasoning-pill').textContent).toContain('Off')
  })

  it('hides entirely when no workhorse model is pinned (nothing to inherit)', () => {
    mocks.config = { delegation: { model: '', provider: '' } }
    const { container } = renderPill()

    expect(container.querySelector('[data-testid="workhorse-reasoning-pill"]')).toBeNull()
  })

  it('writes reasoning_effort to the delegation section and mirrors the cache', async () => {
    mocks.config = { delegation: { model: 'm', provider: 'p' } }
    mocks.previous = { delegation: { model: 'm', provider: 'p' } }
    mocks.save.mockResolvedValueOnce({ ok: true })
    renderPill()

    await act(async () => {
      mocks.onSetOptionsRef.current?.({ effort: 'max' })
      await new Promise(resolve => setTimeout(resolve, 0))
    })

    expect(mocks.save).toHaveBeenCalledWith({ delegation: { reasoning_effort: 'max' } }, 'default')
    expect(mocks.cacheWrite).toHaveBeenCalledWith({
      delegation: { model: 'm', provider: 'p', reasoning_effort: 'max' }
    })
  })

  it('surfaces a failed effort write without painting the cache', async () => {
    mocks.config = { delegation: { model: 'm', provider: 'p' } }
    mocks.save.mockRejectedValueOnce(new Error('boom'))
    renderPill()

    await act(async () => {
      mocks.onSetOptionsRef.current?.({ effort: 'low' })
      await new Promise(resolve => setTimeout(resolve, 0))
    })

    expect(mocks.notifyError).toHaveBeenCalled()
    expect(mocks.cacheWrite).not.toHaveBeenCalled()
  })

  it('merges effort into the latest cached delegation state', async () => {
    mocks.config = { delegation: { model: 'old', provider: 'old-provider' } }
    mocks.previous = { delegation: { model: 'latest', provider: 'latest-provider', reasoning_effort: 'low' } }
    mocks.save.mockResolvedValueOnce({ ok: true })
    renderPill()

    await act(async () => {
      mocks.onSetOptionsRef.current?.({ effort: 'max' })
      await new Promise(resolve => setTimeout(resolve, 0))
    })

    expect(mocks.cacheWrite).toHaveBeenCalledWith({
      delegation: { model: 'latest', provider: 'latest-provider', reasoning_effort: 'max' }
    })
  })

  it('releases typing focus when the reasoning menu closes', () => {
    mocks.config = { delegation: { model: 'm', provider: 'p' } }
    renderPill()

    act(() => mocks.onOpenChangeRef.current?.(false))

    expect(mocks.releaseTypingFocus).toHaveBeenCalledTimes(1)
  })

  it('hides when the resolved workhorse model has no reasoning capability', async () => {
    mocks.config = { delegation: { model: 'm', provider: 'p' } }
    mocks.capabilities = { reasoning: false, can_disable_reasoning: false }
    const { container } = renderPill()

    await waitFor(() => expect(container.querySelector('[data-testid="workhorse-reasoning-pill"]')).toBeNull())
  })
})
