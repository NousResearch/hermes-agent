import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ReactNode } from 'react'

import { ConfigField } from './config-field'

// ListRow is a presentational wrapper; render it directly so the test focuses
// on the CJK dedup logic, not the row chrome.
vi.mock('./primitives', () => ({
  ListRow: ({ description, title }: { description?: ReactNode; title: ReactNode }) => (
    <div>
      <div data-testid="field-title">{title}</div>
      <div data-testid="field-description">{description}</div>
    </div>
  )
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      settings: {
        config: {
          notSet: 'Not set',
          systemDefault: 'System default',
          noResults: 'No results found',
          emptyMessage: 'No options'
        },
        fieldLabels: {},
        fieldDescriptions: {
          'desktop.repoScanRoots': '扫描本地的代码仓库目录' // CJK label that used to normalize to ''
        }
      }
    }
  })
}))

// A minimal QueryClientProvider for any QueryClient usage ConfigField pulls in.
function wrap(node: ReactNode) {
  const qc = new QueryClient()
  return <QueryClientProvider client={qc}>{node}</QueryClientProvider>
}

afterEach(() => {
  cleanup()
})

describe('ConfigField CJK dedup regression', () => {
  it('renders a CJK description even when the label is also CJK', () => {
    // Regression for the old /[^a-z0-9]+/g normalize: both the CJK label and
    // the CJK description collapsed to "", so the dedup check dropped the
    // description. With \p{L}\p{N} (and NFKC) they stay distinct.
    render(
      wrap(
        <ConfigField
          descriptionExtra={null}
          onChange={() => {}}
          schema={{ type: 'string' }}
          schemaKey="desktop.repoScanRoots"
          value=""
        />
      )
    )

    expect(screen.getByTestId('field-description').textContent).toContain('扫描本地的代码仓库目录')
  })

  it('keeps the description when label and description differ only by full-width variants', () => {
    // NFKC folds full-width digits/letters; the dedup must still treat a
    // meaningfully-different description as distinct from the label.
    render(
      wrap(
        <ConfigField
          descriptionExtra={null}
          onChange={() => {}}
          schema={{ type: 'string' }}
          schemaKey="desktop.repoScanRoots"
          value=""
        />
      )
    )

    // The CJK description is NOT identical to the schema key, so it must show.
    expect(screen.getByTestId('field-description').textContent).toBeTruthy()
  })
})
