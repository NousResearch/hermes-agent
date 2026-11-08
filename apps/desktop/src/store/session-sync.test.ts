// @vitest-environment node
import { expect, it, vi } from 'vitest'

import { captureSessionRestoreFence } from './session-sync'

it('receives restore metadata through the real window bus without invalidating another exact owner or profile', async () => {
  const affected = captureSessionRestoreFence('cloud-a', 'work', ['tip', 'root'])
  const otherSource = captureSessionRestoreFence('cloud-b', 'work', ['root'])
  const otherProfile = captureSessionRestoreFence('cloud-a', 'other', ['root'])
  const legacy = captureSessionRestoreFence(null, 'work', ['root'])
  const peer = new BroadcastChannel('hermes:sessions')

  try {
    peer.postMessage({ type: 'restored', sessionId: 'root', connectionId: 'cloud-a', profile: 'work' })
    await vi.waitFor(() => expect(affected.isCurrent()).toBe(false))
    expect(legacy.isCurrent()).toBe(false)
    expect(otherSource.isCurrent()).toBe(true)
    expect(otherProfile.isCurrent()).toBe(true)
  } finally {
    peer.close()
  }
})
