import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('./client', () => ({
  hermesApi: vi.fn(),
  profileScoped: vi.fn(() => ({}))
}))

const client = await import('./client')

const { pauseLocalDownload, resumeLocalDownload } = await import('./local-models')

const hermesApi = vi.mocked(client.hermesApi)

beforeEach(() => {
  vi.clearAllMocks()
})

describe('local download pause/resume API', () => {
  it('pauseLocalDownload posts {job_id} to the existing pause route and returns the {ok,paused} contract', async () => {
    hermesApi.mockResolvedValue({ ok: true, paused: true } as never)

    const res = await pauseLocalDownload('job-7')

    expect(hermesApi).toHaveBeenCalledTimes(1)
    expect(hermesApi.mock.calls[0][0]).toMatchObject({
      body: { job_id: 'job-7' },
      method: 'POST',
      path: '/api/local-models/download/pause'
    })
    expect(res).toEqual({ ok: true, paused: true })
  })

  it('resumeLocalDownload posts {job_id} to the existing resume route and returns the {ok,resumed} contract', async () => {
    hermesApi.mockResolvedValue({ ok: true, resumed: false } as never)

    const res = await resumeLocalDownload('job-8')

    expect(hermesApi).toHaveBeenCalledTimes(1)
    expect(hermesApi.mock.calls[0][0]).toMatchObject({
      body: { job_id: 'job-8' },
      method: 'POST',
      path: '/api/local-models/download/resume'
    })
    // A false resumed flag (no parked download) reaches the caller so the
    // UI can report "couldn't resume" rather than silently claiming success.
    expect(res).toEqual({ ok: true, resumed: false })
  })

  it('a failed pause propagates as a rejection — never a silent success', async () => {
    hermesApi.mockRejectedValue(new Error('unknown download job'))

    await expect(pauseLocalDownload('gone')).rejects.toThrow('unknown download job')
  })
})
