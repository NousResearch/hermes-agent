import { QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { queryClient } from '@/lib/query-client'
import { $currentModel, $currentProvider, $currentReasoningEffort } from '@/store/session'

import { WorkhorseModelPill } from './workhorse-model-pill'

const mocks = vi.hoisted(() => ({
  delegation: {} as Record<string, unknown>,
  commit: vi.fn(),
  releaseTypingFocus: vi.fn(),
  request: vi.fn()
}))

vi.mock('./workhorse-config', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useWorkhorseConfig: () => ({ ...mocks, ready: true, profile: 'worker', connectionId: 'connection-a' })
}))
vi.mock('@/components/ui/keyboard-first', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  releaseTypingFocus: () => mocks.releaseTypingFocus()
}))

beforeEach(() => {
  queryClient.clear()
  queryClient.setDefaultOptions({ queries: { retry: false } })
  mocks.delegation = { model: 'child-model', provider: 'p' }
  mocks.request.mockReset().mockResolvedValue({
    providers: [
      {
        slug: 'p',
        name: 'Provider',
        models: ['child-model', 'other-model'],
        capabilities: { 'child-model': { reasoning: true, can_disable_reasoning: true } }
      }
    ]
  })
  mocks.commit.mockReset()
  mocks.releaseTypingFocus.mockReset()
  $currentModel.set('parent-model')
  $currentProvider.set('p')
  $currentReasoningEffort.set('')
  Element.prototype.scrollIntoView = vi.fn()
})
afterEach(() => {
  cleanup()
  queryClient.clear()
})

async function open() {
  render(
    <QueryClientProvider client={queryClient}>
      <I18nProvider configClient={null} initialLocale="en">
        <WorkhorseModelPill disabled={false} />
      </I18nProvider>
    </QueryClientProvider>
  )
  fireEvent.keyDown(screen.getByTestId('workhorse-model-pill'), { key: 'Enter' })
  await screen.findByText('Provider')
  fireEvent.keyDown(screen.getByTestId('workhorse-reasoning-menu'), { key: 'ArrowRight' })
  await screen.findByRole('menuitem', { name: 'Inherit parent effort' })
}

describe('subagent menu with real Thinking / Effort controls', () => {
  it('keeps the pinned effort inside the menu, never in the bar', async () => {
    mocks.delegation.reasoning_effort = 'high'
    await open()
    expect(screen.getByTestId('workhorse-model-pill').textContent).toBe('')
    expect(screen.getByTestId('workhorse-reasoning-menu').textContent).toContain('High')
  })
  it('allows independent effort selection for an inherited model', async () => {
    mocks.delegation = {}
    await open()
    fireEvent.click(screen.getByRole('menuitemradio', { name: 'High' }))
    expect(mocks.commit).toHaveBeenCalledWith('effort', { reasoning_effort: 'high' })
    expect(mocks.commit).toHaveBeenCalledTimes(1)
    expect($currentModel.get()).toBe('parent-model')
    expect($currentReasoningEffort.get()).toBe('')
  })
  it.each([false, 'false', 'disabled', 'none'])('reenables Thinking from %s', async value => {
    mocks.delegation.reasoning_effort = value
    await open()
    expect(screen.getByRole('switch').getAttribute('aria-checked')).toBe('false')
    fireEvent.click(screen.getByRole('switch'))
    expect(mocks.commit).toHaveBeenCalledWith('effort', { reasoning_effort: 'medium' })
  })
  it('shows inherited Thinking off without persisting a parent effort', async () => {
    mocks.delegation = {}
    $currentReasoningEffort.set('none')
    await open()
    expect(screen.getByRole('switch').getAttribute('aria-checked')).toBe('false')
    expect(mocks.commit).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('switch'))
    expect(mocks.commit).toHaveBeenCalledWith('effort', { reasoning_effort: 'medium' })
    expect($currentReasoningEffort.get()).toBe('none')
  })
  it('can disable Thinking without changing the parent model', async () => {
    mocks.delegation.reasoning_effort = 'high'
    await open()
    fireEvent.click(screen.getByRole('switch'))
    expect(mocks.commit).toHaveBeenCalledWith('effort', { reasoning_effort: 'none' })
    expect(mocks.commit).toHaveBeenCalledTimes(1)
  })
  it('resets effort and releases typing focus on menu close', async () => {
    mocks.delegation.reasoning_effort = 'high'
    await open()
    fireEvent.click(screen.getByRole('menuitem', { name: 'Inherit parent effort' }))
    expect(mocks.commit).toHaveBeenCalledWith('effort', { reasoning_effort: '' })
    expect(mocks.releaseTypingFocus).toHaveBeenCalledTimes(1)
  })
  it.each(['high', ''])('keeps reset available on a non-reasoning model (effort=%s)', async effort => {
    mocks.request.mockResolvedValue({
      providers: [
        { slug: 'p', name: 'Provider', models: ['child-model'], capabilities: { 'child-model': { reasoning: false } } }
      ]
    })
    mocks.delegation.reasoning_effort = effort
    await open()
    expect(screen.queryByRole('switch')).toBeNull()
    expect(screen.queryByRole('menuitemradio')).toBeNull()
    expect(screen.getByRole('menuitem', { name: 'Inherit parent effort' })).toBeTruthy()
  })
  it('edits a different model row through the child controller, never parent options', async () => {
    await open()
    fireEvent.keyDown(screen.getByRole('menuitemradio', { name: 'High' }), { key: 'ArrowLeft' })
    const search = screen.getByRole('textbox', { name: 'Search models' })
    fireEvent.change(search, { target: { value: 'other-model' } })
    fireEvent.keyDown(search, { key: 'ArrowRight' })
    fireEvent.click(await screen.findByRole('menuitemradio', { name: 'High' }))
    await waitFor(() => expect(mocks.commit).toHaveBeenCalledWith('effort', { reasoning_effort: 'high' }))
    expect(mocks.commit).toHaveBeenCalledWith('model', {
      model: 'other-model',
      provider: 'p',
      base_url: '',
      api_key: '',
      api_mode: ''
    })
    expect($currentModel.get()).toBe('parent-model')
    expect($currentReasoningEffort.get()).toBe('')
    expect(mocks.request.mock.calls.every(([method]) => method === 'model.options')).toBe(true)
  })
})
