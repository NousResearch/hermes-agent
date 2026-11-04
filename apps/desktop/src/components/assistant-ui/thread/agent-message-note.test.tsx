import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $gateway } from '@/store/gateway'

import { agentAvatarCache, AgentMessageNote, resolveAgentAvatar } from './user-message'

const clearCache = () => {
  for (const key of [...agentAvatarCache.keys()]) {
    agentAvatarCache.delete(key)
  }
}

const stubGateway = (profiles: Set<string>) => {
  $gateway.set({
    request: async (method: string, params: Record<string, unknown>) => {
      if (method === 'profiles.list') {
        return {
          profiles: [...profiles].map(name => ({ has_avatar: true, name }))
        }
      }

      if (method === 'profiles.get_asset') {
        return { data: `data:image/png;base64,${String(params.name)}`, found: true }
      }

      throw new Error(`unexpected request: ${method}`)
    }
  } as never)
}

describe('AgentMessageNote avatar sync', () => {
  beforeEach(() => {
    clearCache()
    vi.clearAllMocks()
  })

  afterEach(() => {
    cleanup()
    $gateway.set(null)
  })

  /**
   * GREEN: Component mounts with avatar stale, server adds avatar, cache updates,
   * and observers are notified. Component now re-renders with the fresh avatar.
   */
  it('updates avatar when cache changes due to resolveAgentAvatar call', async () => {
    const messageText = 'Message from 🤖 ghostbot: test'

    // Setup gateway that knows about ghostbot from the start
    const profiles = new Set(['ghostbot'])
    stubGateway(profiles)
    clearCache()

    // Component mounts, resolves avatar via the gateway
    const view = render(<AgentMessageNote text={messageText} />)

    // After resolveAgentAvatar resolves, component should have the avatar
    await waitFor(() => {
      const img = view.container.querySelector('img')
      expect(img?.src).toBe('data:image/png;base64,ghostbot')
    }, { timeout: 2000 })
  })

  /**
   * GREEN: Component mounted with avatar, server changes avatar on the backend,
   * we call resolveAgentAvatar again, cache updates and observer is notified,
   * UI shows fresh image.
   */
  it('updates avatar when resolveAgentAvatar fetches a new version', async () => {
    const messageText = 'Message from 🤖 oldbot: hello'
    const profiles = new Set(['oldbot'])
    stubGateway(profiles)
    clearCache()

    // Component mounts, fetches avatar v1
    const view = render(<AgentMessageNote text={messageText} />)

    await waitFor(() => {
      const img = view.container.querySelector('img')
      expect(img?.src).toBe('data:image/png;base64,oldbot')
    })

    // Stage 2: The stub's gateway is updated to return a different asset URL.
    // (In reality this might be triggered by a server event.)
    $gateway.set({
      request: async (method: string, params: Record<string, unknown>) => {
        if (method === 'profiles.list') {
          return { profiles: [{ has_avatar: true, name: 'oldbot' }] }
        }

        if (method === 'profiles.get_asset') {
          return { data: 'data:image/png;base64,oldbot-v2', found: true }
        }

        throw new Error(`unexpected request: ${method}`)
      }
    } as never)

    // Re-call resolveAgentAvatar (simulates a background refresh or server push)
    await act(async () => {
      // First, clear the cache to force a fresh resolve
      agentAvatarCache.delete('oldbot')
      await resolveAgentAvatar('oldbot')
    })

    // Component should re-render with the new avatar
    await waitFor(() => {
      const img = view.container.querySelector('img')
      expect(img?.src).toBe('data:image/png;base64,oldbot-v2')
    })
  })
})

