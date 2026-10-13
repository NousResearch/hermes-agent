import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { WorkhorseReasoningPill } from './workhorse-reasoning-pill'

const mocks = vi.hoisted(() => ({
  delegation: {} as Record<string, unknown>,
  commit: vi.fn(),
  releaseTypingFocus: vi.fn(),
  capabilities: { reasoning: true, can_disable_reasoning: true }
}))

vi.mock('./workhorse-config', () => ({
  useWorkhorseConfig: () => ({ ...mocks, ready: true, profile: 'worker', connectionId: 'connection-a' })
}))
vi.mock('@/components/ui/keyboard-first', () => ({ releaseTypingFocus: () => mocks.releaseTypingFocus() }))
vi.mock('@/lib/model-options', () => ({
  currentModelCapabilities: () => mocks.capabilities,
  modelOptionsQueryKey: () => ['workhorse-model-options'],
  requestModelOptions: () => Promise.resolve({ providers: [] })
}))
vi.mock('@/components/ui/tooltip', () => ({
  Tip: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

beforeEach(() => {
  mocks.delegation = { model: 'm', provider: 'p' }
  mocks.capabilities = { reasoning: true, can_disable_reasoning: true }
  mocks.commit.mockReset()
  mocks.releaseTypingFocus.mockReset()
})
afterEach(cleanup)

function renderPill() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return render(
    <QueryClientProvider client={client}>
      <I18nProvider configClient={null} initialLocale="en">
        <WorkhorseReasoningPill disabled={false} />
      </I18nProvider>
    </QueryClientProvider>
  )
}

function open() {
  fireEvent.pointerDown(screen.getByTestId('workhorse-reasoning-pill'), {
    button: 0,
    ctrlKey: false,
    pointerType: 'mouse'
  })
}

describe('WorkhorseReasoningPill with real Thinking / Effort controls', () => {
  it('shows the pinned delegation effort', () => {
    mocks.delegation.reasoning_effort = 'high'
    renderPill()
    expect(screen.getByTestId('workhorse-reasoning-pill').textContent).toContain('High')
  })
  it('allows independent effort selection for an inherited model', () => {
    mocks.delegation = {}
    renderPill()
    expect(screen.getByTestId('workhorse-reasoning-pill').textContent).toContain('inherit')
    open()
    fireEvent.click(screen.getByRole('menuitemradio', { name: 'High' }))
    expect(mocks.commit).toHaveBeenCalledWith('effort', { reasoning_effort: 'high' })
  })
  it.each([false, 'false', 'disabled', 'none'])('reenables Thinking from %s', value => {
    mocks.delegation.reasoning_effort = value
    renderPill()
    expect(screen.getByTestId('workhorse-reasoning-pill').textContent).toContain('Off')
    open()
    expect(screen.getByRole('switch').getAttribute('aria-checked')).toBe('false')
    fireEvent.click(screen.getByRole('switch'))
    expect(mocks.commit).toHaveBeenCalledWith('effort', { reasoning_effort: 'medium' })
  })
  it('can disable Thinking without changing the parent model', () => {
    mocks.delegation.reasoning_effort = 'high'
    renderPill()
    open()
    fireEvent.click(screen.getByRole('switch'))
    expect(mocks.commit).toHaveBeenCalledWith('effort', { reasoning_effort: 'none' })
  })
  it('resets effort and releases typing focus on menu close', () => {
    mocks.delegation.reasoning_effort = 'high'
    renderPill()
    open()
    fireEvent.click(screen.getByRole('menuitem', { name: 'Inherit parent effort' }))
    expect(mocks.commit).toHaveBeenCalledWith('effort', { reasoning_effort: '' })
    expect(mocks.releaseTypingFocus).toHaveBeenCalledTimes(1)
  })
  it('keeps reset available for an override on a non-reasoning model', () => {
    mocks.capabilities = { reasoning: false, can_disable_reasoning: false }
    mocks.delegation.reasoning_effort = 'high'
    renderPill()
    open()
    expect(screen.queryByRole('switch')).toBeNull()
    expect(screen.getByRole('menuitem', { name: 'Inherit parent effort' })).toBeTruthy()
  })
  it('hides an unset effort on a known non-reasoning model', () => {
    mocks.capabilities = { reasoning: false, can_disable_reasoning: false }
    renderPill()
    expect(screen.queryByTestId('workhorse-reasoning-pill')).toBeNull()
  })
})
