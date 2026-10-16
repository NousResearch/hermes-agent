import type { ReactNode } from 'react'

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { LocalCatalogModel, LocalModelsStatus } from '@/types/hermes'

import { LocalModelsModelsSection } from './local-models-models-section'

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      settings: {
        localModels: {
          modelsTitle: 'Models',
          recommended: 'Recommended',
          noRecommendationAction: 'Find models',
          noRecommendationDetail: 'No recommendation',
          noRecommendationTitle: 'No recommendation'
        }
      }
    }
  })
}))

vi.mock('./local-models-model-rows', () => ({
  CatalogModelRow: () => null,
  SideloadedModelRow: () => null
}))

vi.mock('./primitives', () => ({
  SettingsSection: ({ title, children }: { title: string; children: ReactNode }) => (
    <section>
      <h2>{title}</h2>
      {children}
    </section>
  ),
  ListRow: ({ action }: { action?: ReactNode }) => <div>{action}</div>
}))

afterEach(() => {
  document.getElementById('local-model-browse')?.remove()
  cleanup()
  vi.unstubAllGlobals()
})

const status = { models: [] } as unknown as LocalModelsStatus

function model(recommended: boolean): LocalCatalogModel {
  return {
    id: 'qwen-test',
    model_id: 'qwen-test',
    downloaded_model_id: 'qwen-test',
    fits: true,
    spilled: false,
    recommended
  } as unknown as LocalCatalogModel
}

function addBrowseTarget(): {
  input: HTMLInputElement
  focus: ReturnType<typeof vi.fn>
  scroll: ReturnType<typeof vi.fn>
} {
  const browse = document.createElement('div')
  browse.id = 'local-model-browse'
  const input = document.createElement('input')
  const scroll = vi.fn()
  const focus = vi.fn()

  Object.defineProperty(browse, 'scrollIntoView', { configurable: true, value: scroll })
  Object.defineProperty(input, 'focus', { configurable: true, value: focus })
  browse.appendChild(input)
  document.body.appendChild(browse)

  return { input, focus, scroll }
}

describe('LocalModelsModelsSection heading', () => {
  it('identifies a Nous recommendation when the catalog contains a recommended model', () => {
    render(
      <LocalModelsModelsSection
        catalog={[model(true)]}
        jobs={[]}
        lastError={undefined}
        status={status}
      />
    )

    expect(screen.getByRole('heading', { name: 'Nous · Recommended · Models' })).toBeTruthy()
  })

  it('keeps the generic heading when there is no recommendation', () => {
    render(
      <LocalModelsModelsSection catalog={[model(false)]} jobs={[]} lastError={undefined} status={status} />
    )

    expect(screen.getByRole('heading', { name: 'Models' })).toBeTruthy()
  })
})

describe('LocalModelsModelsSection browse handoff', () => {
  it('scrolls smoothly and focuses the browse search by default', () => {
    vi.stubGlobal('matchMedia', vi.fn(() => ({ matches: false })))
    const { focus, scroll } = addBrowseTarget()

    render(
      <LocalModelsModelsSection catalog={[model(false)]} jobs={[]} lastError={undefined} status={status} />
    )

    fireEvent.click(screen.getByRole('button', { name: 'Find models' }))

    expect(scroll).toHaveBeenCalledWith({ behavior: 'smooth', block: 'start' })
    expect(focus).toHaveBeenCalledWith({ preventScroll: true })
  })

  it('uses instant scrolling when reduced motion is preferred and still focuses search', () => {
    vi.stubGlobal('matchMedia', vi.fn(() => ({ matches: true })))
    const { focus, scroll } = addBrowseTarget()

    render(
      <LocalModelsModelsSection catalog={[model(false)]} jobs={[]} lastError={undefined} status={status} />
    )

    fireEvent.click(screen.getByRole('button', { name: 'Find models' }))

    expect(scroll).toHaveBeenCalledWith({ behavior: 'auto', block: 'start' })
    expect(focus).toHaveBeenCalledWith({ preventScroll: true })
  })
})
