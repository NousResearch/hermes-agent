import { QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DelegationModelRoute } from '@/api/delegation'
import type { HermesApiRequest } from '@/global'
import { queryClient } from '@/lib/query-client'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

import { ModelRouteDialog } from './model-route-dialog'

// The owner's catalog as /api/model/options reports it: Daybreak listed for gpt-6-sol only.
const CATALOG = {
  providers: [
    {
      slug: 'openai-codex',
      name: 'OpenAI Codex',
      models: ['gpt-6-sol', 'gpt-6.1-sol'],
      capabilities: {
        'gpt-6-sol': { fast: true, reasoning: true, daybreak: true },
        'gpt-6.1-sol': { fast: true, reasoning: true }
      }
    }
  ]
}

beforeEach(() => {
  stubMenuDomApis()
  stubResizeObserver()
  queryClient.clear()
  window.hermesDesktop = {
    ...window.hermesDesktop,
    api: async (request: HermesApiRequest) => {
      if (request.path.startsWith('/api/model/options')) {
        return structuredClone(CATALOG)
      }

      throw new Error('404: Not Found')
    }
  } as typeof window.hermesDesktop
})

afterEach(cleanup)

function mount(route: DelegationModelRoute, routeKeys: readonly string[] = ['daybreak']) {
  const onSave = vi.fn(async (_route: DelegationModelRoute) => true)

  render(
    <QueryClientProvider client={queryClient}>
      <ModelRouteDialog
        onClose={() => {}}
        onSave={onSave}
        route={route}
        routeKeys={routeKeys}
        scope="work"
        taken={[]}
      />
    </QueryClientProvider>
  )

  return onSave
}

const daybreakField = () => screen.queryByRole('combobox', { name: /Daybreak/ })

describe('model route dialog: Daybreak', () => {
  it('offers Daybreak only for a model the catalog lists it for, and saves the opt-in', async () => {
    const onSave = mount({ name: 'daybreak', description: 'Sol', provider: 'openai-codex', model: 'gpt-6-sol' })

    await waitFor(() => expect(daybreakField()).not.toBeNull())
    expect(screen.getByText('OpenAI trusted access with reduced guardrails, for this route only.')).toBeTruthy()

    fireEvent.click(daybreakField()!)
    fireEvent.click(await screen.findByRole('option', { name: 'On' }))
    fireEvent.click(screen.getByRole('button', { name: 'Save route' }))

    await waitFor(() => expect(onSave).toHaveBeenCalledTimes(1))
    expect(onSave.mock.calls[0][0]).toMatchObject({ model: 'gpt-6-sol', daybreak: true })
  })

  it('shows no Daybreak control on a model the catalog does not list', async () => {
    mount({ name: 'sol', description: 'Sol', provider: 'openai-codex', model: 'gpt-6.1-sol' })

    await screen.findByText('OpenAI Codex')
    expect(daybreakField()).toBeNull()
  })

  it('shows no Daybreak control when the installed plugin does not declare it, even on a listed model', async () => {
    mount({ name: 'daybreak', description: 'Sol', provider: 'openai-codex', model: 'gpt-6-sol' }, [])

    await screen.findByText('OpenAI Codex')
    expect(daybreakField()).toBeNull()
  })

  it('keeps a stored opt-in the plugin cannot run visible with a warning, so it can be turned off', async () => {
    mount({ name: 'old', description: 'Sol', provider: 'openai-codex', model: 'gpt-6-sol', daybreak: true }, [])

    await screen.findByText(/does not run Daybreak routes yet/)
    expect(daybreakField()).not.toBeNull()
  })

  it('keeps a stored opt-in visible with a warning when its model is not listed, so it can be turned off', async () => {
    const onSave = mount({
      name: 'old',
      description: 'Sol',
      provider: 'openai-codex',
      model: 'gpt-6.1-sol',
      daybreak: true
    })

    await screen.findByText(/not listed for Daybreak on your account/)
    fireEvent.click(daybreakField()!)
    fireEvent.click(await screen.findByRole('option', { name: 'Off' }))
    fireEvent.click(screen.getByRole('button', { name: 'Save route' }))

    await waitFor(() => expect(onSave).toHaveBeenCalledTimes(1))
    expect('daybreak' in onSave.mock.calls[0][0]).toBe(false)
  })
})
