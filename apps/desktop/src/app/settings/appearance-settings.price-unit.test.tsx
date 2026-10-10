// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { enModelMenu } from '@/i18n/en_model_menu'
import { $modelPriceUnit, setModelPriceUnit, setShowModelPricing } from '@/store/model-pricing'

import { AppearanceSettings } from './appearance-settings'

afterEach(() => {
  cleanup()
  setShowModelPricing(false)
  setModelPriceUnit('mtok')
})

function renderPage() {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <AppearanceSettings />
    </QueryClientProvider>
  )
}

describe('AppearanceSettings price unit', () => {
  it('offers per-1K pricing only while prices are shown, and remembers the choice', () => {
    renderPage()

    // A unit for prices the picker does not show would be noise.
    expect(screen.queryByText(enModelMenu.priceUnitPerThousand)).toBeNull()

    act(() => setShowModelPricing(true))
    fireEvent.click(screen.getByText(enModelMenu.priceUnitPerThousand))

    expect($modelPriceUnit.get()).toBe('1k')
  })
})
