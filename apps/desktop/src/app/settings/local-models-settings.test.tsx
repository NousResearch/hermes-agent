import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $localRuntimeJobs } from '@/store/local-runtime-jobs'
import type { LocalCatalogModel, LocalHardware, LocalModelsStatus, LocalRuntimeJob } from '@/types/hermes'

import { LocalModelsSettings } from './local-models-settings'

// Mock the API layer — the pane's contract is what it RENDERS from these
// payloads, not transport.
vi.mock('@/hermes', () => ({
  activateLocalModel: vi.fn(),
  deleteLocalModel: vi.fn(),
  downloadBrowsedModel: vi.fn(),
  downloadLocalModel: vi.fn(),
  ejectLocalModel: vi.fn(),
  getLocalCatalog: vi.fn(),
  getLocalHardware: vi.fn(),
  getLocalModelsJobs: vi.fn(),
  getLocalModelsStatus: vi.fn(),
  getLocalRuntimeJob: vi.fn(),
  installLocalRuntime: vi.fn(),
  listHFRepoFiles: vi.fn(),
  quickstartLocalModels: vi.fn(),
  searchHFModels: vi.fn(),
  sideloadLocalModel: vi.fn()
}))

import * as hermes from '@/hermes'

const mocked = vi.mocked(hermes)

const BASE_STATUS: LocalModelsStatus = {
  enabled: true,
  tag: 'b10290',
  configured_tag: 'b10290',
  update_available: false,
  runtime_installed: false,
  runtime_backend: null,
  server_running: false,
  server_base_url: null,
  active_model_id: null,
  loaded_models: {},
  models: [],
  models_dir: 'C:/somewhere/models'
}

const BASE_HARDWARE: LocalHardware = {
  uma: false,
  vram_total_bytes: 32 * 2 ** 30,
  vram_usable_bytes: 26 * 2 ** 30,
  ram_total_bytes: 256 * 2 ** 30,
  ram_available_bytes: 200 * 2 ** 30,
  vram_label: '32.0 GB',
  gpu_name: 'NVIDIA GeForce RTX 5090',
  gpu_util_percent: 12,
  vram_used_bytes: 6 * 2 ** 30
}

const FITTING_MODEL: LocalCatalogModel = {
  id: 'Qwen3.6-27B-UD-Q4_K_XL',
  display_name: 'Qwen3.6 27B',
  description: 'Best all-round agent model; long context stays fast',
  size_bytes: 17.6 * 2 ** 30,
  size_label: '17.6 GB',
  native_context: 262144,
  native_context_label: '256K',
  recommended: true,
  downloaded: false,
  mtp: false,
  fits: true,
  fit_summary: 'runs at its full 256K context',
  start_window: 262144,
  start_window_label: '256K',
  spilled: false
}

const SPILLED_MODEL: LocalCatalogModel = {
  ...FITTING_MODEL,
  id: 'Spilled-Model',
  display_name: 'Spilled Model',
  recommended: false,
  fits: true,
  spilled: true,
  start_window: 65536,
  start_window_label: '64K',
  fit_summary: 'starts at 64K and grows toward 256K as you use it (larger than your GPU memory — runs slower)'
}

const REFUSED_MODEL: LocalCatalogModel = {
  ...FITTING_MODEL,
  id: 'Huge-Model',
  display_name: 'Huge Model',
  recommended: false,
  fits: false,
  fit_summary: 'Needs more memory than this machine has',
  fit_detail: 'needs ~60 GiB at the 64K floor',
  start_window: undefined,
  start_window_label: undefined
}

function renderPane() {
  return render(
    <MemoryRouter>
      <I18nProvider>
        <LocalModelsSettings />
      </I18nProvider>
    </MemoryRouter>
  )
}

// The fresh-machine states these tests exercise now lead with the
// quickstart card; the full pane (runtime rows, model list, browser)
// is one 'Configure…' click away. Render and click through.
async function renderFullPane() {
  const result = renderPane()
  const configure = await screen.findByRole('button', { name: /configure/i })

  fireEvent.click(configure)

  return result
}

beforeEach(() => {
  mocked.getLocalModelsStatus.mockResolvedValue(BASE_STATUS)
  mocked.getLocalHardware.mockResolvedValue(BASE_HARDWARE)
  mocked.getLocalCatalog.mockResolvedValue({ models: [FITTING_MODEL, SPILLED_MODEL, REFUSED_MODEL] })
  mocked.getLocalModelsJobs.mockResolvedValue({ jobs: [] })
  $localRuntimeJobs.set([])
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('LocalModelsSettings', () => {
  it('offers the runtime install with a plain-language explanation', async () => {
    await renderFullPane()

    expect(await screen.findByText('Install the local runtime')).toBeTruthy()
    expect(screen.getByText(/runs? entirely on this machine/i)).toBeTruthy()
    expect(screen.getByRole('button', { name: /install runtime/i })).toBeTruthy()
  })

  it('shows every catalog model with fit pills; unaffordable ones stay visible with the reason', async () => {
    await renderFullPane()

    expect(await screen.findByText('Qwen3.6 27B')).toBeTruthy()
    // The fitting model reads as pills, not prose: green memory pill +
    // green full-context pill (start_window == native, resident on GPU).
    expect(screen.getByText('Fits your GPU')).toBeTruthy()
    expect(screen.getByText('Full 256K context').className).toContain('emerald')

    // The refused model is NOT hidden (discoverability rule): red memory
    // pill, plus the ceiling it would have had.
    expect(screen.getByText('Huge Model')).toBeTruthy()
    expect(screen.getByText('Too big for this machine')).toBeTruthy()

    // The spilled model reads amber + ONE quiet ceiling pill — the same
    // 'Up to' shape the refused row wears; no start/grow pair.
    expect(screen.getByText('Spilled Model')).toBeTruthy()
    expect(screen.getByText('Uses system RAM')).toBeTruthy()
    expect(screen.getAllByText('Up to 256K context').length).toBe(2)
    expect(screen.queryByText(/Starts at/)).toBeNull()

    // Its download button is disabled; the fitting model's is enabled once
    // the runtime exists (here runtime_installed=false, so both disabled —
    // asserted separately below).
    const buttons = screen.getAllByRole('button', { name: /download · 17\.6 GB/i })
    expect(buttons.every(b => (b as HTMLButtonElement).disabled)).toBe(true)
  })

  it('orders the catalog by fit: resident first, then spilled, then too-big', async () => {
    // Scrambled input — the pane, not the backend, owns display order.
    mocked.getLocalCatalog.mockResolvedValue({ models: [REFUSED_MODEL, SPILLED_MODEL, FITTING_MODEL] })
    await renderFullPane()
    await screen.findByText('Qwen3.6 27B')

    // The matched element is the row-title span; the recommended row's
    // includes its nested pill copy — strip it before comparing order.
    const names = screen
      .getAllByText(/^(Qwen3\.6 27B|Spilled Model|Huge Model)$/)
      .map(el => el.textContent?.replace('Recommended', ''))

    expect(names).toEqual(['Qwen3.6 27B', 'Spilled Model', 'Huge Model'])
  })

  it('never greens the full-context pill on a system-RAM model', async () => {
    // Full native window, but earned by spilling into system RAM: the
    // pill must not wear the green that would recommend exactly the
    // wrong model.
    const spilledFull: LocalCatalogModel = {
      ...FITTING_MODEL,
      id: 'Spilled-Full',
      display_name: 'Spilled Full',
      recommended: false,
      spilled: true,
      fit_summary: 'runs its full 256K context, partly from system RAM'
    }

    mocked.getLocalCatalog.mockResolvedValue({ models: [spilledFull] })
    await renderFullPane()
    await screen.findByText('Spilled Full')

    expect(screen.getByText('Full 256K context').className).not.toContain('emerald')
  })

  it('explains the Recommended pick on hover', async () => {
    // The tooltip is the resolver's own reason, and it must actually OPEN:
    // Tip works by asChild-cloning hover handlers onto the pill, so a Pill
    // that swallows its rest props kills the tooltip silently (the pill
    // still renders, nothing appears on hover).
    mocked.getLocalCatalog.mockResolvedValue({
      models: [{ ...FITTING_MODEL, recommended_reason: 'speed-gated-quality' }]
    })
    await renderFullPane()
    await screen.findByText('Qwen3.6 27B')

    fireEvent.pointerMove(screen.getByText('Recommended'))
    fireEvent.pointerEnter(screen.getByText('Recommended'))

    await waitFor(() =>
      expect(screen.getAllByText(/would respond too slowly on its memory bandwidth/).length).toBeGreaterThan(0)
    )
  })

  it('enables downloads only once the runtime is installed', async () => {
    mocked.getLocalModelsStatus.mockResolvedValue({
      ...BASE_STATUS,
      runtime_installed: true,
      runtime_backend: 'cuda'
    })
    await renderFullPane()

    await screen.findByText('Qwen3.6 27B')
    const [fittingButton] = screen.getAllByRole('button', { name: /download · 17\.6 GB/i })
    expect((fittingButton as HTMLButtonElement).disabled).toBe(false)
  })

  it('shows hardware facts after backfill', async () => {
    await renderFullPane()

    expect(await screen.findByText('NVIDIA GeForce RTX 5090')).toBeTruthy()
    expect(screen.getByText(/32\.0 GB GPU memory/)).toBeTruthy()
    expect(screen.getByText(/256\.0 GB RAM/)).toBeTruthy()
  })

  it('tracks a download job to completion and refreshes', async () => {
    mocked.getLocalModelsStatus.mockResolvedValue({
      ...BASE_STATUS,
      runtime_installed: true,
      runtime_backend: 'cuda'
    })
    mocked.downloadLocalModel.mockResolvedValue({ job_id: 'j1' })

    const running: LocalRuntimeJob = {
      job_id: 'j1',
      kind: 'model-download',
      target: 'Qwen3.6 27B',
      model_id: FITTING_MODEL.id,
      status: 'running',
      phase: 'downloading',
      detail: 'Qwen3.6 27B — 17.6 GB',
      total_bytes: 100,
      done_bytes: 40,
      percent: 40,
      error: null
    }

    mocked.getLocalModelsJobs
      .mockResolvedValueOnce({ jobs: [running] })
      .mockResolvedValue({ jobs: [{ ...running, status: 'done', phase: 'done', done_bytes: 100, percent: 100 }] })

    await renderFullPane()
    await screen.findByText('Qwen3.6 27B')

    const [download] = screen.getAllByRole('button', { name: /download · 17\.6 GB/i })
    download.click()

    // The app-level watcher follows the job; when it settles the pane
    // refreshes (status + catalog re-fetched).
    await waitFor(() => {
      expect(mocked.getLocalModelsJobs).toHaveBeenCalled()
      expect(mocked.getLocalModelsStatus.mock.calls.length).toBeGreaterThanOrEqual(2)
    })
  })

  it('renders progress for a download discovered from the store (survives pane remount)', async () => {
    mocked.getLocalModelsStatus.mockResolvedValue({
      ...BASE_STATUS,
      runtime_installed: true,
      runtime_backend: 'cuda'
    })
    // A running job already in the app-level store — as after closing and
    // reopening the pane mid-download.
    $localRuntimeJobs.set([
      {
        job_id: 'j9',
        kind: 'model-download',
        target: 'Qwen3.6 27B',
        model_id: FITTING_MODEL.id,
        status: 'running',
        phase: 'downloading',
        detail: '',
        total_bytes: 100,
        done_bytes: 62,
        percent: 62,
        error: null
      }
    ])

    await renderFullPane()
    await screen.findByText('Qwen3.6 27B')

    // The fitting row shows byte progress; the remaining download
    // buttons belong to the other rows (spilled + refused).
    expect(screen.getAllByText(/0\.0 GB of 0\.0 GB|of/).length).toBeGreaterThan(0)
    const remaining = screen.queryAllByRole('button', { name: /download · 17\.6 GB/i })
    expect(remaining.length).toBe(2)
    expect(remaining.some(b => (b as HTMLButtonElement).disabled)).toBe(true)
  })

  it('surfaces a failed download with the backend message', async () => {
    mocked.getLocalModelsStatus.mockResolvedValue({
      ...BASE_STATUS,
      runtime_installed: true,
      runtime_backend: 'cuda'
    })
    $localRuntimeJobs.set([
      {
        job_id: 'j2',
        kind: 'model-download',
        target: 'Qwen3.6 27B',
        model_id: FITTING_MODEL.id,
        status: 'error',
        phase: 'verifying',
        detail: '',
        total_bytes: 100,
        done_bytes: 100,
        error: 'Downloaded file failed its integrity check and was removed — try again'
      }
    ])

    await renderFullPane()
    await screen.findByText('Qwen3.6 27B')

    expect(await screen.findByText(/integrity check/)).toBeTruthy()
  })
})

describe('quickstart', () => {
  it('leads with one button on a fresh machine and fires the quickstart job', async () => {
    mocked.quickstartLocalModels.mockResolvedValue({
      display_name: 'Qwen3.6 27B',
      download_bytes: FITTING_MODEL.size_bytes,
      job_id: 'q1',
      model_id: 'qwen3.6-27b',
      needs_download: true,
      needs_runtime: true
    })
    renderPane()

    // The card names the recommended model and the one-click action; the
    // runtime/model machinery is NOT on screen.
    expect(await screen.findByRole('button', { name: /set up for me/i })).toBeTruthy()
    expect(screen.queryByText('Install the local runtime')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: /set up for me/i }))
    await waitFor(() => {
      expect(mocked.quickstartLocalModels).toHaveBeenCalled()
    })
  })

  it('pins the quickstart progress view while the job runs', async () => {
    $localRuntimeJobs.set([
      {
        job_id: 'q1',
        kind: 'quickstart',
        target: 'Qwen3.6 27B',
        model_id: 'qwen3.6-27b',
        status: 'running',
        phase: 'downloading',
        detail: 'Qwen3.6 27B — 17.6 GB',
        total_bytes: 100,
        done_bytes: 30,
        percent: 30,
        error: null
      }
    ])
    renderPane()

    expect(await screen.findByText('Qwen3.6 27B — 17.6 GB')).toBeTruthy()
    // One job, one view: no Set up / Configure buttons while it runs.
    expect(screen.queryByRole('button', { name: /set up for me/i })).toBeNull()
  })

  it('skips the card entirely once a model is staged', async () => {
    mocked.getLocalModelsStatus.mockResolvedValue({
      ...BASE_STATUS,
      runtime_installed: true,
      runtime_backend: 'cuda',
      models: [{ id: 'Qwen3.6-27B-UD-Q4_K_XL', size_bytes: 17 * 2 ** 30, size_label: '17.6 GB' }]
    })
    renderPane()

    // Straight to the full pane — no quickstart hero for a working setup.
    expect(await screen.findByText('Qwen3.6 27B')).toBeTruthy()
    expect(screen.queryByRole('button', { name: /set up for me/i })).toBeNull()
  })
})

describe('BrowseSection', () => {
  it('searches HF after a pause and shows fit-priced files on demand', async () => {
    vi.useFakeTimers()

    try {
      vi.mocked(hermes.searchHFModels).mockResolvedValue({
        hits: [{ downloads: 872724, gated: false, likes: 47, repo: 'unsloth/Qwen3.8-27B-GGUF', updated: '2026-08-18' }]
      })
      vi.mocked(hermes.listHFRepoFiles).mockResolvedValue({
        files: [
          { fit: 'fits-gpu', label: 'Q4_K_M', paths: ['Qwen3.8-27B-Q4_K_M.gguf'], total_bytes: 17 * 2 ** 30 },
          { fit: 'too-big', label: 'F16', paths: ['Qwen3.8-27B-F16.gguf'], total_bytes: 56 * 2 ** 30 }
        ]
      })

      render(
        <MemoryRouter>
          <I18nProvider>
            <LocalModelsSettings />
          </I18nProvider>
        </MemoryRouter>
      )
      await act(async () => {
        await vi.runOnlyPendingTimersAsync()
      })
      // Fresh machine leads with the quickstart card — enter the full pane.
      fireEvent.click(screen.getByRole('button', { name: /configure/i }))

      const box = screen.getByPlaceholderText(/search models/i)
      fireEvent.change(box, { target: { value: 'qwen' } })
      // Debounce: no call until the pause elapses.
      expect(hermes.searchHFModels).not.toHaveBeenCalled()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(400)
      })
      expect(hermes.searchHFModels).toHaveBeenCalledWith('qwen')
      expect(screen.getByText('unsloth/Qwen3.8-27B-GGUF')).toBeTruthy()

      fireEvent.click(screen.getByRole('button', { name: /show files/i }))
      await act(async () => {
        await vi.runOnlyPendingTimersAsync()
      })
      expect(screen.getByText('Q4_K_M')).toBeTruthy()
      // Each tile has an explicit download button; the too-big quant's is
      // disabled, the fitting one is live and starts the download.
      const q4Btn = screen.getByRole('button', { name: 'Download Q4_K_M' })
      const f16Btn = screen.getByRole('button', { name: 'Download F16' })
      expect((f16Btn as HTMLButtonElement).disabled).toBe(true)
      expect((q4Btn as HTMLButtonElement).disabled).toBe(false)

      vi.mocked(hermes.downloadBrowsedModel).mockResolvedValue({ job_id: 'j1', model_id: 'Qwen3.8-27B-Q4_K_M' })
      fireEvent.click(q4Btn)
      await act(async () => {
        await vi.runOnlyPendingTimersAsync()
      })
      expect(hermes.downloadBrowsedModel).toHaveBeenCalledWith('unsloth/Qwen3.8-27B-GGUF', ['Qwen3.8-27B-Q4_K_M.gguf'])
    } finally {
      vi.useRealTimers()
    }
  })
})

describe('added-by-you rows', () => {
  it('staged models outside the catalog get the full action set', async () => {
    vi.mocked(hermes.getLocalModelsStatus).mockResolvedValue({
      ...BASE_STATUS,
      loaded_models: { 'Hermes-4.3-36B-Q5_K_M': 'loaded' },
      models: [{ id: 'Hermes-4.3-36B-Q5_K_M', size_bytes: 25 * 2 ** 30, size_label: '25.0 GB' }],
      placement: {
        'Hermes-4.3-36B-Q5_K_M': {
          granted_window_label: '96K',
          spilled: false,
          window: 98304,
          window_label: '96K'
        }
      },
      server_running: true
    })
    vi.mocked(hermes.getLocalCatalog).mockResolvedValue({ models: [] })

    renderPane()
    await screen.findByText('Hermes-4.3-36B-Q5_K_M')

    // Full management surface: Use, eject, delete, live placement pill.
    expect(screen.getByText(/added by you/i)).toBeTruthy()
    expect(screen.getByRole('button', { name: /use/i })).toBeTruthy()
    expect(screen.getByText(/96K/)).toBeTruthy()
    const buttons = screen.getAllByRole('button')
    expect(buttons.length).toBeGreaterThanOrEqual(3)
  })
})

describe('quickstart completion navigation', () => {
  it('lands on a new chat when a quickstart it watched finishes; stale done jobs on mount never navigate', async () => {
    const routeProbe = vi.fn()

    function Probe() {
      const loc = useLocation()
      routeProbe(loc.pathname)

      return null
    }

    const doneJob: LocalRuntimeJob = {
      done_bytes: 0,
      detail: '',
      error: null,
      job_id: 'stale-done',
      kind: 'quickstart',
      model_id: 'qwen3.8-27b',
      phase: 'done',
      status: 'done',
      target: 'Qwen3.8 27B',
      total_bytes: null
    }

    // A finished quickstart already in history when the pane mounts —
    // must NOT trigger navigation.
    $localRuntimeJobs.set([doneJob])

    render(
      <MemoryRouter initialEntries={['/settings']}>
        <I18nProvider>
          <LocalModelsSettings />
        </I18nProvider>
        <Probe />
      </MemoryRouter>
    )
    await act(async () => {})
    expect(routeProbe).not.toHaveBeenCalledWith('/')

    // A quickstart the pane SAW running that then completes -> navigate.
    const running: LocalRuntimeJob = { ...doneJob, job_id: 'live-run', phase: 'downloading', status: 'running' }
    await act(async () => {
      $localRuntimeJobs.set([doneJob, running])
    })
    await act(async () => {
      $localRuntimeJobs.set([doneJob, { ...running, phase: 'done', status: 'done' }])
    })
    expect(routeProbe).toHaveBeenCalledWith('/')
  })
})
