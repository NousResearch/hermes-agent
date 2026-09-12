import { afterEach, expect, it, vi } from 'vitest'

import type { HermesApiRequest } from '@/global'

import { setApiRequestConnection, setApiRequestProfile } from './client'
import * as messaging from './messaging'

afterEach(() => {
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  vi.unstubAllGlobals()
})

it.each(['local', 'fixture-remote'])(
  'keeps platform and pairing operations on the explicit %s owner',
  async connectionId => {
    const owner = { connectionId, profile: 'shared-name' }
    const requests: HermesApiRequest[] = []
    vi.stubGlobal('hermesDesktop', {
      api: vi.fn(async (request: HermesApiRequest) => {
        requests.push(request)

        return {}
      })
    })
    setApiRequestConnection('fixture-ambient')
    setApiRequestProfile('ambient-profile')

    await messaging.getMessagingPlatforms(owner)
    await messaging.updateMessagingPlatform('fixture-platform', { enabled: true }, owner)
    await messaging.testMessagingPlatform('fixture-platform', owner)
    await messaging.startTelegramOnboarding(undefined, owner)
    await messaging.getTelegramOnboardingStatus('fixture-pairing', owner)
    await messaging.applyTelegramOnboarding('fixture-pairing', ['123456'], owner)
    await messaging.cancelTelegramOnboarding('fixture-pairing', owner)
    await messaging.getPairing(owner)
    await messaging.approvePairing('fixture-platform', 'fixture-request', owner)
    await messaging.revokePairing('fixture-platform', '123456', owner)

    for (const request of requests) {
      expect.soft(request).toMatchObject(owner)
      // The renderer connection tag is routing metadata, never backend JSON.
      expect.soft(request.body ?? {}).not.toHaveProperty('connectionId')
    }

    expect(
      requests
        .filter(request => request.path.endsWith('/apply') || request.path.startsWith('/api/pairing/'))
        .map(request => request.body)
    ).toEqual([
      { allowed_user_ids: ['123456'], profile: owner.profile },
      { platform: 'fixture-platform', request_id: 'fixture-request', profile: owner.profile },
      { platform: 'fixture-platform', user_id: '123456', profile: owner.profile }
    ])
  }
)
