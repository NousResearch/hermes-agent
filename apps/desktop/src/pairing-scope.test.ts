// Pairing writes must target the profile the user is actually looking at.
// The approve/revoke endpoints read `profile` off the BODY (a POST body is
// not touched by query-param scoping), so a request that only carried
// `profileScoped()` at the top level would approve into the default
// profile's whitelist while the operator was managing another one — a grant
// the running gateway for that profile never consults.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type ProfileScope, setApiRequestConnection, setApiRequestProfile } from './api/client'
import * as messaging from './api/messaging'

const api = vi.fn().mockResolvedValue({ ok: true })

async function requestMessaging(scope?: ProfileScope) {
  await Promise.all([
    messaging.getMessagingPlatforms(scope),
    messaging.updateMessagingPlatform('telegram', { enabled: true }, scope),
    messaging.testMessagingPlatform('telegram', scope),
    messaging.getRelayLinks(scope),
    messaging.requestRelayLink('telegram', scope),
    messaging.getRelayLinkState('link-code', scope),
    messaging.unlinkRelayAccount('telegram', 'U1', scope),
    messaging.startTelegramOnboarding('Inbox', scope),
    messaging.getTelegramOnboardingStatus('pairing-id', scope),
    messaging.applyTelegramOnboarding('pairing-id', ['U1'], scope),
    messaging.cancelTelegramOnboarding('pairing-id', scope),
    messaging.getPairing(scope),
    messaging.approvePairing('telegram', 'request-id', scope),
    messaging.revokePairing('telegram', 'U1', scope),
    messaging.getWebhooks(scope),
    messaging.enableWebhooks(scope),
    messaging.createWebhook({ name: 'inbox' }, scope),
    messaging.deleteWebhook('inbox', scope),
    messaging.setWebhookEnabled('inbox', true, scope)
  ])
}

describe('messaging requests carry their owning gateway and profile', () => {
  beforeEach(() => {
    api.mockClear()
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })
  })

  afterEach(() => {
    setApiRequestConnection(null)
    setApiRequestProfile(null)
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('scopes approve and revoke by body, and the listing by query', async () => {
    const mod = await import('@/hermes')
    mod.setApiRequestProfile('work')

    await mod.approvePairing('telegram', 'a'.repeat(16))
    await mod.revokePairing('telegram', 'U1')
    await mod.getPairing()

    const [approve, revoke, list] = api.mock.calls.map(call => call[0])

    expect(approve.body.profile).toBe('work')
    expect(revoke.body.profile).toBe('work')
    expect(list.profile).toBe('work')
  })

  it('omits the profile entirely for single-profile users', async () => {
    const mod = await import('@/hermes')
    mod.setApiRequestProfile(null)

    await mod.approvePairing('telegram', 'a'.repeat(16))
    await mod.getPairing()

    const [approve, list] = api.mock.calls.map(call => call[0])

    expect(approve.body.profile).toBeUndefined()
    expect(list.profile).toBeUndefined()
  })

  it.each(['homelab', 'local'])('pins messaging flows to %s after the ambient gateway changes', async connectionId => {
    const owner = Object.freeze({ connectionId: ` ${connectionId} `, profile: ' inbox-bot ' })
    setApiRequestConnection('first-gateway')
    setApiRequestProfile('first-profile')
    await requestMessaging(owner)

    setApiRequestConnection('second-gateway')
    setApiRequestProfile('second-profile')
    await requestMessaging(owner)

    for (const [request] of api.mock.calls) {
      expect(request).toMatchObject({ connectionId, profile: 'inbox-bot', priority: 'foreground' })
    }

    const bodyProfiles = api.mock.calls
      .map(([request]) => request)
      .filter(
        request =>
          ['/api/pairing/approve', '/api/pairing/revoke'].includes(request.path) || request.path.endsWith('/apply')
      )

    for (const request of bodyProfiles) {
      expect(request.body.profile).toBe('inbox-bot')
      expect(request.body).not.toHaveProperty('connectionId')
      expect(request.body).not.toHaveProperty('priority')
    }
  })

  it.each([
    { scope: undefined, profile: 'ambient-profile', priority: undefined },
    { scope: null, profile: undefined, priority: undefined },
    { scope: '', profile: undefined, priority: 'foreground' },
    { scope: 'coder', profile: 'coder', priority: 'foreground' },
    { scope: ' coder ', profile: ' coder ', priority: 'foreground' }
  ])('preserves classic scope semantics for $scope', async ({ scope, profile, priority }) => {
    setApiRequestConnection('ambient-gateway')
    setApiRequestProfile('ambient-profile')
    await requestMessaging(scope)

    for (const [request] of api.mock.calls) {
      expect(request.connectionId).toBe('ambient-gateway')
      expect(request.profile).toBe(profile)
      expect(request.priority).toBe(priority)

      if (request.body && 'profile' in request.body) {
        expect(request.body.profile).toBe(profile)
      }
    }
  })
})
