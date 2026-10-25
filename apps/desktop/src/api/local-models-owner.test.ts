import { beforeEach, expect, it, vi } from 'vitest'

import type { LocalModelsOwner } from '@/store/local-runtime-jobs'

import { setApiRequestConnection, setApiRequestProfile } from './client'
import {
  getLocalModelsJobs,
  getLocalModelsStatus,
  pauseLocalDownload,
  resumeLocalDownload,
  setLocalGpu
} from './local-models'

beforeEach((): void => {
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { api: vi.fn().mockResolvedValue({ jobs: [] }) }
  })
  setApiRequestConnection('foreground')
  setApiRequestProfile('default')
})

it('pins delayed reads and controls to their captured connection and profile', async (): Promise<void> => {
  const owner = { connectionId: 'background', profile: 'work' }
  await getLocalModelsStatus(owner)
  await getLocalModelsJobs(owner)
  await pauseLocalDownload('download', owner)

  for (const [request] of vi.mocked(window.hermesDesktop.api).mock.calls) {
    expect(request).toMatchObject(owner)
  }
})

it.each(['auto', '1', 'GPU-b83de86f', 'Vulkan0'])(
  'posts GPU choice %s to the captured owner and preserves the canonical server result',
  async gpu => {
    const owner = { connectionId: 'compute-host', profile: 'secondary' }
    const result = { ok: true, gpu: gpu === '1' ? 'GPU-b83de86f' : gpu, server_restarted: true }
    vi.mocked(window.hermesDesktop.api).mockResolvedValue(result)
    expect(await setLocalGpu(gpu, owner)).toEqual(result)
    expect(window.hermesDesktop.api).toHaveBeenCalledExactlyOnceWith({
      ...owner,
      body: { gpu },
      method: 'POST',
      path: '/api/local-models/gpu'
    })
  }
)

it.each([422, 409, 502])('propagates a GPU HTTP %s failure without inventing success', async status => {
  const error = new Error(`HTTP ${status}: GPU selection failed`)
  vi.mocked(window.hermesDesktop.api).mockRejectedValue(error)
  await expect(setLocalGpu('GPU-missing', { connectionId: 'compute-host', profile: 'secondary' })).rejects.toBe(error)
})

it.each([
  ['pause', pauseLocalDownload, { ok: true, paused: true }],
  ['resume', resumeLocalDownload, { ok: true, resumed: false }]
] as const)(
  '%s preserves the captured owner, request body and backend verdict',
  async (
    action: string,
    control: typeof pauseLocalDownload | typeof resumeLocalDownload,
    response: { ok: boolean; paused?: boolean; resumed?: boolean }
  ): Promise<void> => {
    const owner: LocalModelsOwner = { connectionId: 'background', profile: 'work' }
    vi.mocked(window.hermesDesktop.api).mockResolvedValue(response)
    expect(await control('download', owner)).toEqual(response)
    expect(window.hermesDesktop.api).toHaveBeenCalledExactlyOnceWith({
      ...owner,
      method: 'POST',
      path: `/api/local-models/download/${action}`,
      body: { job_id: 'download' }
    })
    vi.mocked(window.hermesDesktop.api).mockRejectedValue(new Error('unknown download job'))
    await expect(control('gone', owner)).rejects.toThrow('unknown download job')
  }
)
