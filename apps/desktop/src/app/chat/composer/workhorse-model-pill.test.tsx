import { QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { PRIMARY_SESSION_VIEW, SessionViewProvider } from '@/app/chat/session-view'
import { ModelMenuPanel } from '@/app/shell/model-menu-panel'
import { DropdownMenu, DropdownMenuContent } from '@/components/ui/dropdown-menu'
import { I18nProvider } from '@/i18n'
import { queryClient } from '@/lib/query-client'
import { $customModels } from '@/store/custom-models'
import { $favoriteModels } from '@/store/favorite-models'
import { $localModelsEnabled } from '@/store/local-models-flag'
import { $modelPresets } from '@/store/model-presets'
import { $knownModels, $visibleModels, setVisibleModels } from '@/store/model-visibility'
import { $collapsedProviders } from '@/store/provider-collapse'

import { WorkhorseModelPill } from './workhorse-model-pill'

const mocks = vi.hoisted(() => ({
  delegation: {} as Record<string, unknown>,
  ready: true,
  commit: vi.fn(),
  request: vi.fn(),
  parentSelect: vi.fn()
}))

vi.mock('./workhorse-config', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useWorkhorseConfig: () => ({ ...mocks, profile: 'worker', connectionId: 'connection-a' })
}))

const view = {
  ...PRIMARY_SESSION_VIEW,
  kind: 'tile' as const,
  $runtimeId: atom<string | null>('tile-session'),
  $model: atom('parent-model'),
  $provider: atom('p'),
  $reasoningEffort: atom('high')
}

const catalog = {
  model: 'catalog-parent',
  provider: 'p',
  providers: [
    {
      slug: 'p',
      name: 'Provider P',
      models: ['parent-model', 'child-model', 'visible-model', 'visible-model-fast', 'hidden-model', 'star-model'],
      capabilities: { 'child-model': { reasoning: true, can_disable_reasoning: true, fast: true } }
    },
    { slug: 'q', name: 'Provider Q', models: ['other-model'] },
    { slug: 'llamacpp', name: 'Local', models: ['local-model'] },
    { slug: 'moa', name: 'MoA', models: ['team'] }
  ]
}

beforeEach(() => {
  queryClient.clear()
  queryClient.setDefaultOptions({ queries: { retry: false } })
  mocks.delegation = { model: 'child-model', provider: 'p', reasoning_effort: 'low' }
  mocks.ready = true
  mocks.commit.mockReset()
  mocks.request.mockReset().mockResolvedValue(catalog)
  mocks.parentSelect.mockReset()
  view.$model.set('parent-model')
  view.$provider.set('p')
  view.$reasoningEffort.set('high')
  $favoriteModels.set([])
  $visibleModels.set(null)
  $knownModels.set(null)
  $customModels.set([])
  $collapsedProviders.set([])
  $localModelsEnabled.set(false)
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})
afterEach(() => {
  cleanup()
  queryClient.clear()
})

function mount(main = false, props: Partial<{ disabled: boolean }> = {}) {
  return render(
    <QueryClientProvider client={queryClient}>
      <I18nProvider configClient={null} initialLocale="en">
        <SessionViewProvider value={view}>
          {main ? (
            <DropdownMenu open>
              <DropdownMenuContent>
                <ModelMenuPanel
                  onSelectModel={mocks.parentSelect}
                  ownerConnectionId="connection-a"
                  profile="worker"
                  requestGateway={mocks.request}
                />
              </DropdownMenuContent>
            </DropdownMenu>
          ) : (
            <WorkhorseModelPill disabled={false} {...props} />
          )}
        </SessionViewProvider>
      </I18nProvider>
    </QueryClientProvider>
  )
}

function open() {
  fireEvent.keyDown(screen.getByTestId('workhorse-model-pill'), { key: 'Enter' })
}

async function search() {
  return screen.findByRole('textbox', { name: 'Search models' })
}

function modelRows() {
  return screen
    .getAllByRole('menuitem')
    .filter(row => row.querySelector('[aria-label="Add to favorites"], [aria-label="Remove from favorites"]'))
    .map(row => row.querySelector('.truncate')?.textContent)
}

describe('compact subagent menu', () => {
  it('is icon-only and opens and closes by keyboard', async () => {
    mount()
    const trigger = screen.getByRole('button', { name: 'Subagents' })
    expect(trigger.textContent).toBe('')
    expect(trigger.querySelector('svg')).toBeTruthy()
    expect(mocks.request).not.toHaveBeenCalled()
    open()
    await search()
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(screen.getByTestId('workhorse-model-menu').className).toContain('w-72')
    fireEvent.keyDown(await search(), { key: 'Escape' })
    await waitFor(() => expect(screen.queryByRole('menu')).toBeNull())
    await waitFor(() => expect(trigger.ownerDocument.activeElement).toBe(trigger))
  })

  it.each([
    { model: 'child-model', provider: 'p' },
    { model: 'other-model', provider: 'q' }
  ])(
    'matches the principal catalog exactly, including favorites, hidden/current rows, retired favorites and provider gates (%s)',
    async child => {
      mocks.delegation = child
      $favoriteModels.set(['p::star-model', 'p::retired-model', 'absent::absent-model', 'llamacpp::local-model'])
      setVisibleModels(new Set(['p::visible-model', 'q::other-model']), catalog.providers)
      mount(true)
      await screen.findByText('Visible Model')
      const mainRows = modelRows()
      expect(mainRows).toEqual(['Star Model', 'Parent Model', 'Visible Model', 'Other Model'])
      expect(screen.getByRole('menuitem', { name: 'MoA: team' })).toBeTruthy()
      fireEvent.change(await search(), { target: { value: 'novel-slug' } })

      const customRows = () =>
        screen
          .getAllByRole('menuitem')
          .filter(row => row.textContent?.startsWith('novel-slug'))
          .map(row => row.textContent)

      const mainCustomRows = customRows()
      expect(mainCustomRows).toEqual(['novel-slug Provider P', 'novel-slug Provider Q'])
      cleanup()
      mount()
      open()
      await screen.findByText('Visible Model')
      expect(modelRows()).toEqual(mainRows)
      expect(screen.getByRole('menuitem', { name: 'MoA: team' })).toBeTruthy()
      fireEvent.change(await search(), { target: { value: 'novel-slug' } })
      expect(customRows()).toEqual(mainCustomRows)
      expect(mocks.request).toHaveBeenCalledWith(
        'model.options',
        expect.objectContaining({ profile: 'worker', session_id: 'tile-session' })
      )
      expect(mocks.commit).not.toHaveBeenCalled()
    }
  )

  it('shares provider collapse, search and keyboard selection without touching parent or presets', async () => {
    const presets = $modelPresets.get()
    mount()
    open()
    await screen.findByText('Visible Model')
    fireEvent.click(screen.getByRole('menuitem', { name: 'Provider P' }))
    expect(modelRows()).toEqual(['Other Model'])
    fireEvent.change(await search(), { target: { value: 'hidden-model' } })
    expect(modelRows()).toEqual(['Hidden Model'])
    fireEvent.keyDown(await search(), { key: 'Enter' })
    await waitFor(() =>
      expect(mocks.commit).toHaveBeenCalledWith('model', {
        model: 'hidden-model',
        provider: 'p',
        base_url: '',
        api_key: '',
        api_mode: ''
      })
    )
    expect(mocks.commit).toHaveBeenCalledTimes(1)
    expect(mocks.parentSelect).not.toHaveBeenCalled()
    expect(view.$model.get()).toBe('parent-model')
    expect(view.$reasoningEffort.get()).toBe('high')
    expect($modelPresets.get()).toEqual(presets)
    await waitFor(() => expect(screen.queryByRole('menu')).toBeNull())
  })

  it('reaches inheritance and effort from search using only the keyboard', async () => {
    mount()
    open()
    const input = await search()
    fireEvent.keyDown(input, { key: 'Tab' })
    const inherit = screen.getByRole('menuitem', { name: 'Inherit parent model and route' })
    expect(inherit.ownerDocument.activeElement).toBe(inherit)
    fireEvent.keyDown(inherit, { key: 'ArrowDown' })
    const effort = screen.getByTestId('workhorse-reasoning-menu')
    await waitFor(() => expect(effort.ownerDocument.activeElement).toBe(effort))
    fireEvent.keyDown(effort, { key: 'ArrowRight' })
    const reset = await screen.findByRole('menuitem', { name: 'Inherit parent effort' })
    await waitFor(() => expect(reset.ownerDocument.activeElement).toBe(reset))
    fireEvent.keyDown(reset, { key: 'Enter' })
    expect(mocks.commit).toHaveBeenCalledWith('effort', { reasoning_effort: '' })
    await waitFor(() => expect(screen.queryByRole('menu')).toBeNull())
  })

  it('also reaches the principal footer without changing its model', async () => {
    mount(true)
    const input = await search()
    fireEvent.keyDown(input, { key: 'Tab' })
    const refresh = screen.getByRole('menuitem', { name: 'Refresh models' })
    expect(refresh.ownerDocument.activeElement).toBe(refresh)
    expect(mocks.parentSelect).not.toHaveBeenCalled()
  })

  it('stars through the shared favorite control without selecting a model', async () => {
    mount()
    open()
    fireEvent.change(await search(), { target: { value: 'visible-model' } })
    fireEvent.keyDown(await search(), { key: 'Enter', shiftKey: true })
    expect($favoriteModels.get()).toContain('p::visible-model')
    expect(mocks.commit).not.toHaveBeenCalled()
  })

  it('teaches the icon through the translated tooltip', async () => {
    mount()
    fireEvent.pointerMove(screen.getByRole('button', { name: 'Subagents' }), { pointerType: 'mouse' })
    expect((await screen.findByRole('tooltip')).textContent).toBe('Subagents')
  })

  it('resets only the child model and route, leaving effort independent', async () => {
    mount()
    open()
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Inherit parent model and route' }))
    expect(mocks.commit).toHaveBeenCalledWith('model', {
      model: '',
      provider: '',
      base_url: '',
      api_key: '',
      api_mode: ''
    })
    expect(mocks.commit).toHaveBeenCalledTimes(1)
  })

  it.each(['ready', 'disabled'])('disables the control for %s', reason => {
    mocks.ready = reason !== 'ready'
    mount(false, { disabled: reason === 'disabled' })
    expect(screen.getByTestId('workhorse-model-pill').hasAttribute('disabled')).toBe(true)
  })
})
