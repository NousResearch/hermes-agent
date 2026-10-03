import type { ReactNode } from 'react'

import { cleanup, render, screen } from '@testing-library/react'
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
  ListRow: () => null
}))

afterEach(cleanup)

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
