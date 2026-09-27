import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { WritableAtom } from 'nanostores'
import { afterEach, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { requestModelOptions } from '@/lib/model-options'
import { startManualOnboarding } from '@/store/onboarding'
import { $gatewayState, $modelPickerOpen, $selectedStoredSessionId } from '@/store/session'
import { $focusedTreePaneId as $focusedTreePaneIdMock } from '@/store/session-focus'
import { $sessionTiles } from '@/store/session-states'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

import { ModelPickerOverlay } from './model-picker-overlay'

// The mock below replaces the computed store with a writable atom.
const $focusedTreePaneId = $focusedTreePaneIdMock as unknown as WritableAtom<null | string>

vi.mock('@/store/session-focus', async () => {
  const { atom } = await import('nanostores')

  return { $focusedTreePaneId: atom<null | string>(null) }
})
vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getLocalModelsStatus: vi.fn().mockResolvedValue({ loading: {} })
}))
vi.mock('@/lib/model-options', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestModelOptions: vi.fn().mockResolvedValue({ model: '', provider: '', providers: [] })
}))
vi.mock('@/store/onboarding', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  startManualOnboarding: vi.fn()
}))

stubResizeObserver()
stubMenuDomApis()

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  $sessionTiles.set([])
  $focusedTreePaneId.set(null)
  $selectedStoredSessionId.set(null)
  $modelPickerOpen.set(false)
  $gatewayState.set('idle')
})

it('reads the catalog from the backend profile but hands the Desktop alias to provider setup', async () => {
  $gatewayState.set('open')
  $modelPickerOpen.set(true)
  $selectedStoredSessionId.set('primary-a')
  $sessionTiles.set([
    {
      ownerRoute: { connectionId: 'connection-b', profile: 'desktop-b', targetProfile: 'backend-b' },
      storedSessionId: 'tile-b'
    }
  ])
  $focusedTreePaneId.set('session-tile:tile-b')
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  render(
    <QueryClientProvider client={client}>
      <I18nProvider>
        <ModelPickerOverlay
          onSelect={() => undefined}
          ownerConnectionId="connection-a"
          profile="default"
          requestGateway={async () => undefined as never}
        />
      </I18nProvider>
    </QueryClientProvider>
  )

  await waitFor(() => expect(requestModelOptions).toHaveBeenCalled())
  expect(vi.mocked(requestModelOptions).mock.calls.every(([options]) => options.profile === 'backend-b')).toBe(true)

  fireEvent.click(await screen.findByRole('button', { name: 'Add provider' }))
  expect(startManualOnboarding).toHaveBeenCalledWith(undefined, { connectionId: 'connection-b', profile: 'desktop-b' })
  client.clear()
})
