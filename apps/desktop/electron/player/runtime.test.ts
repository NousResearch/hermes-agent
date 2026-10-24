import { describe, expect, it, vi } from 'vitest'

import { resolveWidevineReadiness } from './runtime'

const ID = 'widevine-cdm'

describe('resolveWidevineReadiness', () => {
  it('requires the requested component in both readiness results and runtime status', async () => {
    const api = {
      WIDEVINE_CDM_ID: ID,
      whenReady: vi.fn().mockResolvedValue([{ id: ID, status: 'updated' }]),
      status: vi.fn().mockReturnValue({ [ID]: { status: 'ready', version: '4.10' } })
    }

    await expect(resolveWidevineReadiness(api)).resolves.toEqual({ state: 'ready' })
    expect(api.whenReady).toHaveBeenCalledWith([ID])
  })

  it('does not infer readiness from a different component or a missing status record', async () => {
    await expect(
      resolveWidevineReadiness({
        WIDEVINE_CDM_ID: ID,
        whenReady: async () => [{ id: 'another-component', status: 'ready' }],
        status: () => ({ [ID]: { status: 'ready' } })
      })
    ).resolves.toMatchObject({ state: 'unsupported' })
    await expect(
      resolveWidevineReadiness({
        WIDEVINE_CDM_ID: ID,
        whenReady: async () => [{ id: ID, status: 'updated' }],
        status: () => ({})
      })
    ).resolves.toMatchObject({ state: 'unsupported' })
  })

  it('requires an app restart when Linux reports a newly updated Widevine component', async () => {
    await expect(
      resolveWidevineReadiness(
        {
          WIDEVINE_CDM_ID: ID,
          whenReady: async () => [{ id: ID, status: 'updated' }],
          status: () => ({ [ID]: { status: 'updated' } })
        },
        100,
        'linux'
      )
    ).resolves.toEqual({
      state: 'restart-required',
      detail: 'Widevine was installed or updated; restart Hermes before protected playback'
    })
  })

  it('bounds component readiness instead of leaving TV startup pending forever', async () => {
    await expect(
      resolveWidevineReadiness(
        {
          WIDEVINE_CDM_ID: ID,
          whenReady: () => new Promise(() => undefined),
          status: () => ({ [ID]: { status: 'ready' } })
        },
        5
      )
    ).resolves.toEqual({ state: 'unsupported', detail: 'component-readiness-timeout' })
  })
})
