// Webapp (browser-hosted) model selection of already-advertised Local models.
//
// The Electron-only `--local` launch flag gates local-runtime MANAGEMENT
// (status polling, downloads, Settings pane). It must not hide a `llamacpp`
// provider the backend already advertises as configured, or the web selectors
// disagree with the "Edit models" dialog. Native Desktop with the flag
// explicitly off keeps hiding it.
//
// Model rows below are test FIXTURES returned by a mocked options request; no
// real inference or llama.cpp server is involved.
import type { ModelOptionsResult } from '@hermes/shared'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ModelCatalogMenu, type ModelMenuController } from '@/app/shell/model-catalog-menu'
import { ModelPickerDialog } from '@/components/model-picker'
import { DropdownMenu, DropdownMenuContent } from '@/components/ui/dropdown-menu'
import { I18nProvider } from '@/i18n'
import { $localModelsEnabled } from '@/store/local-models-flag'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

const hermes = vi.hoisted(() => ({
  getLocalModelsJobs: vi.fn(async () => ({ jobs: [] })),
  getLocalModelsStatus: vi.fn(async () => ({ loading: {} }))
}))

vi.mock('@/hermes', () => hermes)
vi.mock('@/lib/model-options', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestModelOptions: vi.fn()
}))

import { requestModelOptions } from '@/lib/model-options'

stubResizeObserver()
stubMenuDomApis()

const LOCAL_MODEL = 'Qwen3.6-27B-UD-Q4_K_XL'
// The menu prettifies ids; the same fixture row reads like this there.
const LOCAL_MENU_ROW = /Qwen3\.6 27B/i

const OPTIONS: ModelOptionsResult = {
  providers: [
    { slug: 'llamacpp', name: 'Local', models: [LOCAL_MODEL], authenticated: true },
    { slug: 'nous', name: 'Nous', models: ['fixture-remote-model'], authenticated: true }
  ]
}

type Host = 'browser' | 'desktop-on' | 'desktop-off'

// Reproduce each host's `window.hermesDesktop` shape:
//  - browser: the web bridge (no `localModelsEnabled` key) + its host marker; the atom
//    is false because the Electron preload flag is absent.
//  - desktop-*: the Electron preload publishes an explicit boolean and no marker.
function setHost(host: Host): void {
  const bridge: Record<string, unknown> = host === 'browser' ? {} : { localModelsEnabled: host === 'desktop-on' }

  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: bridge })

  if (host === 'browser') {
    window.document.documentElement.dataset.hermesDesktopHost = 'browser'
  } else {
    delete window.document.documentElement.dataset.hermesDesktopHost
  }

  $localModelsEnabled.set(host === 'desktop-on')
}

let client: QueryClient

beforeEach(() => {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  vi.mocked(requestModelOptions).mockResolvedValue(OPTIONS)
})

afterEach(() => {
  cleanup()
  delete window.document.documentElement.dataset.hermesDesktopHost
  Reflect.deleteProperty(window, 'hermesDesktop')
  $localModelsEnabled.set(false)
  vi.clearAllMocks()
})

function renderPicker(onSelect = vi.fn()) {
  render(
    <QueryClientProvider client={client}>
      <I18nProvider>
        <ModelPickerDialog
          currentModel="fixture-remote-model"
          currentProvider="nous"
          onOpenChange={() => undefined}
          onSelect={onSelect}
          open
        />
      </I18nProvider>
    </QueryClientProvider>
  )

  return onSelect
}

function renderMenu() {
  const select = vi.fn()

  const controller: ModelMenuController = {
    applyPreset: vi.fn(),
    current: { effort: '', fast: false, model: '', provider: '' },
    presetFor: () => ({}),
    select,
    setOptions: vi.fn()
  }

  render(
    <QueryClientProvider client={client}>
      <I18nProvider>
        <DropdownMenu open>
          <DropdownMenuContent>
            <ModelCatalogMenu controller={controller} />
          </DropdownMenuContent>
        </DropdownMenu>
      </I18nProvider>
    </QueryClientProvider>
  )

  return select
}

describe('Webapp (no Electron preload flag): configured Local models are selectable', () => {
  beforeEach(() => setHost('browser'))

  it('ModelPickerDialog shows and selects the advertised Local model', async () => {
    const onSelect = renderPicker()

    const row = await screen.findByText(LOCAL_MODEL)

    fireEvent.click(row)
    expect(onSelect).toHaveBeenCalledWith({ provider: 'llamacpp', model: LOCAL_MODEL })
  })

  it('ModelCatalogMenu shows and selects the advertised Local model', async () => {
    const select = renderMenu()

    fireEvent.click(await screen.findByText(LOCAL_MENU_ROW))
    await waitFor(() => expect(select).toHaveBeenCalledWith(LOCAL_MODEL, 'llamacpp'))
  })

  it('still keeps local-runtime management off (no status/jobs polling)', async () => {
    renderPicker()
    await screen.findByText(LOCAL_MODEL)
    renderMenu()
    await screen.findAllByText(LOCAL_MENU_ROW)

    expect(hermes.getLocalModelsStatus).not.toHaveBeenCalled()
    expect(hermes.getLocalModelsJobs).not.toHaveBeenCalled()
  })
})

describe('native Desktop with the local-models flag explicitly off: Local stays hidden', () => {
  beforeEach(() => setHost('desktop-off'))

  it('ModelPickerDialog hides the Local provider', async () => {
    renderPicker()
    await screen.findByText('fixture-remote-model', { selector: '*' })
    expect(screen.queryByText(LOCAL_MODEL)).toBeNull()
  })

  it('ModelCatalogMenu hides the Local provider', async () => {
    renderMenu()
    await screen.findByText(/fixture remote model/i)
    expect(screen.queryByText(LOCAL_MENU_ROW)).toBeNull()
  })
})

describe('native Desktop with the local-models flag on: unchanged', () => {
  it('ModelPickerDialog shows the Local model', async () => {
    setHost('desktop-on')
    renderPicker()
    expect(await screen.findByText(LOCAL_MODEL)).toBeTruthy()
  })

  it('ModelCatalogMenu shows the Local model', async () => {
    setHost('desktop-on')
    renderMenu()
    expect(await screen.findByText(LOCAL_MENU_ROW)).toBeTruthy()
  })
})
