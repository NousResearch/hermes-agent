import { beforeEach, expect, it, vi } from 'vitest'

import type { LocalModelsOwner } from '@/store/local-runtime-jobs'

import { setApiRequestConnection, setApiRequestProfile } from './client'
import { getLocalModelsJobs, getLocalModelsStatus, pauseLocalDownload, resumeLocalDownload } from './local-models'

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

it.each([
  ['pause', pauseLocalDownload, { ok: true, paused: true }],
  ['resume', resumeLocalDownload, { ok: true, resumed: false }]
] as const)('%s preserves the captured owner, request body and backend verdict', async (action: string, control: typeof pauseLocalDownload | typeof resumeLocalDownload, response: { ok: boolean; paused?: boolean; resumed?: boolean }): Promise<void> => {
  const owner: LocalModelsOwner = { connectionId: 'background', profile: 'work' }
  vi.mocked(window.hermesDesktop.api).mockResolvedValue(response)
  expect(await control('download', owner)).toEqual(response)
  expect(window.hermesDesktop.api).toHaveBeenCalledExactlyOnceWith({
    ...owner, method: 'POST', path: `/api/local-models/download/${action}`, body: { job_id: 'download' }
  })
  vi.mocked(window.hermesDesktop.api).mockRejectedValue(new Error('unknown download job'))
  await expect(control('gone', owner)).rejects.toThrow('unknown download job')
})
