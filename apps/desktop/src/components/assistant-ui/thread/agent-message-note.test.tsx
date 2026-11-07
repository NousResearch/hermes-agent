import { act, cleanup, render, waitFor } from '@testing-library/react'
import type { ComponentProps } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $gateway } from '@/store/gateway'

import { AgentDeliveryNotice } from './agent-delivery'
import { agentAvatarCache, AgentMessageNote, resolveAgentAvatar } from './user-message'

let revision = 1
let image: null | string = 'data:image/png;base64,old'
let calls: string[] = []
let failAssets = false

beforeEach(() => {
  for (const key of [...agentAvatarCache.keys()]) {
    agentAvatarCache.delete(key)
  }

  revision = 1
  image = 'data:image/png;base64,old'
  calls = []
  failAssets = false
  $gateway.set({
    request: async (method: string) => {
      calls.push(method)

      if (method === 'profiles.get_asset' && failAssets) {
        throw new Error('temporarily offline')
      }

      return method === 'profiles.list'
        ? { profiles: [{ name: 'default', has_avatar: image !== null, avatar_rev: image ? revision : null }] }
        : { found: image !== null, data: image }
    }
  } as never)
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  $gateway.set(null)
})

it('refreshes a mounted notice after an external replacement without manually clearing its cache', async () => {
  vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] })
  const view = render(<AgentMessageNote text="Message from 🤖 Hermes (@hermes): test" />)
  await waitFor(() => expect(view.container.querySelector('img')?.getAttribute('src')).toBe(image))
  image = 'data:image/png;base64,new'
  revision = 2
  await act(async () => {
    await vi.advanceTimersByTimeAsync(30_000)
  })
  expect(view.container.querySelector('img')?.getAttribute('src')).toBe(image)
  const assetCalls = calls.filter(call => call === 'profiles.get_asset').length
  await act(async () => {
    await vi.advanceTimersByTimeAsync(30_000)
  })
  expect(calls.filter(call => call === 'profiles.get_asset')).toHaveLength(assetCalls)
  image = null
  await act(async () => {
    await vi.advanceTimersByTimeAsync(30_000)
  })
  expect(view.container.querySelector('img')).toBeNull()
  expect(view.container.textContent).toContain('🤖')
  image = 'data:image/png;base64,restored'
  revision = 3
  await act(async () => {
    await vi.advanceTimersByTimeAsync(30_000)
  })
  expect(view.container.querySelector('img')?.getAttribute('src')).toBe(image)
})

it('shares one probe between mounted sender and recipient notices and stops on unmount', async () => {
  vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] })

  const view = render(
    <>
      <AgentMessageNote text="Message from 🤖 Hermes (@hermes): test" />
      <AgentDeliveryNotice
        {...({
          args: { command: 'hermes -p hermes chat -q "Message from 🤖 test: hello"' },
          result: 'hello'
        } as ComponentProps<typeof AgentDeliveryNotice>)}
      />
    </>
  )

  await waitFor(() => expect(view.container.querySelectorAll('img')).toHaveLength(3))
  calls = []
  image = 'data:image/png;base64,new'
  revision = 2
  await act(async () => {
    await vi.advanceTimersByTimeAsync(30_000)
  })
  expect(calls).toEqual(['profiles.list', 'profiles.get_asset'])

  for (const img of view.container.querySelectorAll('img')) {
    expect(img.getAttribute('src')).toBe(image)
  }

  view.unmount()
  calls = []
  await act(async () => {
    await vi.advanceTimersByTimeAsync(30_000)
  })
  expect(calls).toEqual([])
})

it('retries a failed revision fetch without erasing the displayed avatar', async () => {
  vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] })
  const view = render(<AgentMessageNote text="Message from 🤖 Hermes (@hermes): test" />)
  await waitFor(() => expect(view.container.querySelector('img')?.getAttribute('src')).toBe(image))
  image = 'data:image/png;base64,new'
  revision = 2
  failAssets = true
  await act(async () => {
    await vi.advanceTimersByTimeAsync(30_000)
  })
  expect(view.container.querySelector('img')?.getAttribute('src')).toBe('data:image/png;base64,old')
  failAssets = false
  await act(async () => {
    await vi.advanceTimersByTimeAsync(30_000)
  })
  expect(view.container.querySelector('img')?.getAttribute('src')).toBe(image)
})

it('does not reuse another gateway avatar even when its revision is identical', async () => {
  expect(await resolveAgentAvatar('hermes')).toBe('data:image/png;base64,old')
  const previous = $gateway.get()!
  image = 'data:image/png;base64,other-gateway'
  $gateway.set({ request: previous.request } as never)
  expect(await resolveAgentAvatar('hermes')).toBe(image)
})

it('preserves cached art for older gateways without avatar_rev', async () => {
  const gateway = $gateway.get()!
  $gateway.set({
    request: async (method: string, params: never) => {
      const res = (await gateway.request(method as never, params)) as { profiles?: { avatar_rev?: unknown }[] }

      for (const profile of res.profiles ?? []) {
        delete profile.avatar_rev
      }

      return res
    }
  } as never)
  vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] })
  const view = render(<AgentMessageNote text="Message from 🤖 Hermes (@hermes): test" />)
  await waitFor(() => expect(view.container.querySelector('img')?.getAttribute('src')).toBe(image))
  image = 'data:image/png;base64,new'
  calls = []
  await act(async () => {
    await vi.advanceTimersByTimeAsync(30_000)
  })
  expect(calls).toEqual(['profiles.list'])
  expect(view.container.querySelector('img')?.getAttribute('src')).toBe('data:image/png;base64,old')
})
