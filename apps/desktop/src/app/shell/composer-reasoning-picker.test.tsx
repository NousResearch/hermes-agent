import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ReasoningPill } from '@/app/chat/composer/reasoning-pill'
import { useModelControls } from '@/app/session/hooks/use-model-controls'
import { DropdownMenu, DropdownMenuContent } from '@/components/ui/dropdown-menu'
import { displayModelName } from '@/lib/model-status-label'
import { $composerReasoningEffort, setComposerReasoningEffort } from '@/store/composer-reasoning-effort'
import { $modelPresets, getModelPreset, setModelPreset } from '@/store/model-presets'
import { $visibleModels } from '@/store/model-visibility'
import {
  $activeSessionId,
  $selectedStoredSessionId,
  setComposerSelectionOwner,
  setCurrentModel,
  setCurrentModelSource,
  setCurrentProvider,
  setCurrentReasoningEffort,
  setDefaultReasoningEffort
} from '@/store/session'

import { ModelCatalogMenu } from './model-catalog-menu'
import { useModelMenuController } from './use-model-menu-controller'

const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })

const request = vi.fn(
  async () => ({ providers: [{ name: 'Example', slug: 'example', models: ['example-model'] }] }) as never
)

function Picker() {
  const { selectModel } = useModelControls({ queryClient, requestGateway: request })
  const { controller } = useModelMenuController({ onSelectModel: selectModel, requestGateway: request })

  return (
    <DropdownMenu open>
      <DropdownMenuContent>
        <ModelCatalogMenu controller={controller} />
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

beforeEach(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
  queryClient.clear()
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  setComposerSelectionOwner('test-connection', 'default')
  setComposerReasoningEffort('')
  $modelPresets.set({})
  $visibleModels.set(null)
  setCurrentModel('previous-model')
  setCurrentProvider('example')
  setCurrentModelSource('default')
  setCurrentReasoningEffort('low')
  setDefaultReasoningEffort('high')
})
afterEach(() => {
  cleanup()
  queryClient.clear()
})

describe('composer model picker effort provenance', () => {
  it.each([false, true])('inherits defaults unless the selected model has a deliberate preset (%s)', async preset => {
    if (preset) {
      setModelPreset('example', 'example-model', { effort: 'none' })
    }

    render(
      <QueryClientProvider client={queryClient}>
        <Picker />
      </QueryClientProvider>
    )
    fireEvent.click(await screen.findByText(displayModelName('example-model')))
    await waitFor(() => expect(getModelPreset('example', 'example-model').fast).toBe(false))
    expect($composerReasoningEffort.get()).toBe(preset ? 'none' : '')
    expect(getModelPreset('example', 'example-model').effort).toBe(preset ? 'none' : undefined)
    cleanup()
    render(
      <ReasoningPill
        disabled={false}
        model={{ canSwitch: true, model: 'example-model', provider: 'example', reasoningMenuContent: <div>menu</div> }}
      />
    )
    expect(screen.getByTestId('reasoning-pill').textContent).toBe(preset ? 'Off' : 'High')
    // Runtime mirrors cannot replace the draft's explicit pick or its default.
    act(() => setCurrentReasoningEffort('low'))
    expect(screen.getByTestId('reasoning-pill').textContent).toBe(preset ? 'Off' : 'High')
  })
})
