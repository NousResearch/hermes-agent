import { act, cleanup, render, waitFor } from '@testing-library/react'
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

  it('updates avatar when resolveAgentAvatar fetches a new version', async () => {
    const messageText = 'Message from 🤖 oldbot: hello'
    const profiles = new Set(['oldbot'])
    stubGateway(profiles)
    clearCache()

    const view = render(<AgentMessageNote text={messageText} />)

    await waitFor(() => {
      const img = view.container.querySelector('img')
      expect(img?.src).toBe('data:image/png;base64,oldbot')
    })

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

    await act(async () => {
      agentAvatarCache.delete('oldbot')
      await resolveAgentAvatar('oldbot')
    })

    await waitFor(() => {
      const img = view.container.querySelector('img')
      expect(img?.src).toBe('data:image/png;base64,oldbot-v2')
    })
  })

  /**
   * RED TEST: Avatar removal does NOT trigger re-render.
   * Component shows old avatar when server removes it.
   * This test will FAIL until we fix the observer pattern.
   */
  it('updates avatar when it is removed (server no longer has asset)', async () => {
    const messageText = 'Message from 🤖 botremoved: hello'
    
    // Stage 1: Gateway knows about botremoved with avatar
    const profiles1 = new Set(['botremoved'])
    stubGateway(profiles1)
    clearCache()

    const view = render(<AgentMessageNote text={messageText} />)

    // Wait for initial avatar to load
    await waitFor(() => {
      const img = view.container.querySelector('img')
      expect(img?.src).toBe('data:image/png;base64,botremoved')
    })

    // Stage 2: Gateway updated - botremoved no longer has avatar
    $gateway.set({
      request: async (method: string, params: Record<string, unknown>) => {
        if (method === 'profiles.list') {
          return { profiles: [{ has_avatar: false, name: 'botremoved' }] }
        }

        if (method === 'profiles.get_asset') {
          return { data: null, found: false }
        }

        throw new Error(`unexpected request: ${method}`)
      }
    } as never)

    // Stage 3: Trigger a fresh resolve (which will update cache and notify observers)
    // Clear cache entry first to force a fresh gateway call
    await act(async () => {
      agentAvatarCache.delete('botremoved')
      await resolveAgentAvatar('botremoved')
    })

    // Stage 4: Component should show 🤖 glyph, NOT the old image
    // The observer callback should have been triggered and setAvatar(null) called
    await waitFor(() => {
      const img = view.container.querySelector('img')
      const glyph = view.container.querySelector('span[aria-hidden]')
      
      expect(img).toBeNull()
      expect(glyph?.textContent).toBe('🤖')
    }, { timeout: 2000 })
  })
})
