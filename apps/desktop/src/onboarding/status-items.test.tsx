import { QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import type { HermesConnection } from '@/global'
import { queryClient } from '@/lib/query-client'
import { setInterfaceMode } from '@/store/interface-mode'
import { localModelsKey, localModelsOwner, watchLocalRuntimeJobs } from '@/store/local-runtime-jobs'
import { $notifications } from '@/store/notifications'
import { setConnection } from '@/store/session'
import { $statusbarVisible } from '@/store/statusbar-prefs'
import { installRestBridge } from '@/test/rest-bridge'

import { $questionnaireDownload, LocalDownloadStatusItem, startQuestionnaireQuickstart } from './status-items'

// SAFETY: the jobs owner reads only baseUrl from the connection; the rest is window chrome.
const CONNECTION = { baseUrl: 'http://127.0.0.1:9119', isFullscreen: false, mode: 'local' } as HermesConnection

// GET /api/local-models/jobs mid-quickstart, as the Windows run answered it.
const QUICKSTART_JOB = {
  detail: '',
  done_bytes: 20,
  error: null,
  job_id: 'qs-1',
  kind: 'quickstart',
  model_id: 'qwen3.8-27b',
  percent: 20,
  phase: 'downloading',
  status: 'running',
  target: 'Qwen3.8 27B',
  total_bytes: 100
}

// A download the user started from Settings > Local Models after the questionnaire's.
const OTHER_DOWNLOAD = {
  ...QUICKSTART_JOB,
  done_bytes: 40,
  job_id: 'dl-2',
  kind: 'model-download',
  model_id: 'gemma-4-12b',
  percent: 40,
  target: 'Gemma 4 12B'
}

async function jobsRead(api: ReturnType<typeof installRestBridge>) {
  await waitFor(() => {
    expect(api).toHaveBeenCalledWith(expect.objectContaining({ path: '/api/local-models/jobs' }))
    expect(queryClient.isFetching()).toBe(0)
  })
}

function renderItem() {
  render(
    <QueryClientProvider client={queryClient}>
      <LocalDownloadStatusItem />
    </QueryClientProvider>
  )
}

afterEach(() => {
  cleanup()
  setInterfaceMode('advanced')
  $notifications.set([])
  $questionnaireDownload.set(null)
  setConnection(null)
  queryClient.clear()
})

describe('LocalDownloadStatusItem', () => {
  it('shows the quickstart download with its percent once the backend connected after app start', async () => {
    const api = installRestBridge(request =>
      request.path === '/api/local-models/jobs' ? { jobs: [QUICKSTART_JOB] } : {}
    )

    // The status bar module loads before the backend connects; the questionnaire's Start comes later.
    setConnection(CONNECTION)
    $questionnaireDownload.set('qs-1')

    render(
      <QueryClientProvider client={queryClient}>
        <LocalDownloadStatusItem />
      </QueryClientProvider>
    )

    const item = await screen.findByRole('status')

    expect(item.textContent).toContain('Qwen3.8 27B')
    await waitFor(() => expect(item.textContent).toContain('20'))
    expect(api).toHaveBeenCalledWith(expect.objectContaining({ path: '/api/local-models/jobs', profile: 'default' }))
  })

  it('names the engine, not the model, while quickstart fetches the engine first', async () => {
    const engineStage = {
      ...QUICKSTART_JOB,
      detail: 'Downloading the local engine (2/2)',
      percent: 63,
      phase: 'downloading-runtime'
    }

    installRestBridge(request => (request.path === '/api/local-models/jobs' ? { jobs: [engineStage] } : {}))
    setConnection(CONNECTION)
    $questionnaireDownload.set('qs-1')

    renderItem()

    const item = await screen.findByRole('status')

    await waitFor(() => expect(item.textContent).toContain('Downloading the local engine'))
    expect(item.textContent).not.toContain('Qwen3.8 27B')
  })

  it("shows the download Start's quickstart created, though the first jobs read came before the job", async () => {
    let answerQuickstart = () => {}
    let jobCreated = false

    // The backend creates the job while it handles the POST; until then the jobs list is empty.
    const api = installRestBridge(request => {
      if (request.path === '/api/local-models/quickstart') {
        return new Promise<object>(resolve => {
          answerQuickstart = () => {
            jobCreated = true
            resolve({ job_id: 'qs-1', model_id: 'qwen3.8-27b' })
          }
        })
      }

      return request.path === '/api/local-models/jobs' ? { jobs: jobCreated ? [QUICKSTART_JOB] : [] } : {}
    })

    setConnection(CONNECTION)
    renderItem()
    // Another surface (Settings > Local Models) read the jobs before Start's POST answered.
    watchLocalRuntimeJobs(localModelsOwner('default'))

    const started = startQuestionnaireQuickstart({ id: 'qwen3.8-27b', name: 'Qwen3.8 27B' })

    await jobsRead(api)
    answerQuickstart()
    await started

    const item = await screen.findByRole('status')

    expect(item.textContent).toContain('Qwen3.8 27B')
    await waitFor(() => expect(item.textContent).toContain('20'))
    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        body: { model_id: 'qwen3.8-27b' },
        path: '/api/local-models/quickstart',
        profile: 'default'
      })
    )
  })

  it('names only the job Start created, not a later download once that job is done', async () => {
    const api = installRestBridge(request => {
      if (request.path === '/api/local-models/quickstart') {
        return { job_id: 'qs-1', model_id: 'qwen3.8-27b' }
      }

      return request.path === '/api/local-models/jobs'
        ? { jobs: [{ ...QUICKSTART_JOB, status: 'done' }, OTHER_DOWNLOAD] }
        : {}
    })

    setConnection(CONNECTION)
    await startQuestionnaireQuickstart({ id: 'qwen3.8-27b', name: 'Qwen3.8 27B' })
    renderItem()

    await jobsRead(api)
    expect(screen.queryByRole('status')).toBeNull()
  })

  it('explains a refused quickstart and shows no download for it', async () => {
    const api = installRestBridge(request => {
      if (request.path === '/api/local-models/quickstart') {
        throw new Error('Setup is already running')
      }

      return request.path === '/api/local-models/jobs' ? { jobs: [OTHER_DOWNLOAD] } : {}
    })

    setConnection(CONNECTION)
    // Settings > Local Models is already downloading another model.
    watchLocalRuntimeJobs(localModelsOwner('default'))
    await jobsRead(api)
    await startQuestionnaireQuickstart({ id: 'qwen3.8-27b', name: 'Qwen3.8 27B' })
    renderItem()

    expect($notifications.get()).toEqual([expect.objectContaining({ kind: 'error' })])
    expect(screen.queryByRole('status')).toBeNull()
  })
})

describe('the Simple-mode status bar after Start', () => {
  it('stays up after setup closes while Start\'s download runs, and rests hidden once it is done', async () => {
    let status = 'running'

    installRestBridge(request => {
      if (request.path === '/api/local-models/quickstart') {
        return { job_id: 'qs-1', model_id: 'qwen3.8-27b' }
      }

      return request.path === '/api/local-models/jobs' ? { jobs: [{ ...QUICKSTART_JOB, status }] } : {}
    })

    setConnection(CONNECTION)
    setInterfaceMode('simple')
    // Setup is closed and there is no Sign in chip: the bar rests hidden.
    expect($statusbarVisible.get()).toBe(false)

    await startQuestionnaireQuickstart({ id: 'qwen3.8-27b', name: 'Qwen3.8 27B' })
    await waitFor(() => expect($statusbarVisible.get()).toBe(true))

    status = 'done'
    await queryClient.refetchQueries({ queryKey: localModelsKey(localModelsOwner('default'), 'jobs') })
    await waitFor(() => expect($statusbarVisible.get()).toBe(false))
  })
})
