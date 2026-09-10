import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { stubThreadEnvironment } from '../test-utils'

import { Thread } from '.'

const identity = vi.hoisted(() => ({
  value: {
    appearance: {
      avatarDataUrl: 'data:image/png;base64,c3ludGhldGlj',
      displayName: 'Kite',
      role: 'TikTok Channel Steward'
    },
    owner: { connectionId: 'source-a', profile: 'kite', targetProfile: 'kite' }
  } as {
    appearance: null | { avatarDataUrl: null | string; displayName: string; role: null | string }
    owner: null | { connectionId: string; profile: string; targetProfile: string }
  }
}))

vi.mock('@/lib/session-profile-appearance', () => ({
  useSessionProfileAppearance: () => identity.value
}))

stubThreadEnvironment()

afterEach(() => {
  cleanup()
  identity.value = {
    appearance: {
      avatarDataUrl: 'data:image/png;base64,c3ludGhldGlj',
      displayName: 'Kite',
      role: 'TikTok Channel Steward'
    },
    owner: { connectionId: 'source-a', profile: 'kite', targetProfile: 'kite' }
  }
})

const assistant = (custom: Record<string, unknown> = {}, running = false): ThreadMessage =>
  ({
    id: 'assistant-1',
    role: 'assistant',
    content: [{ type: 'text', text: 'Synthetic reply' }],
    status: running ? { type: 'running' } : { type: 'complete', reason: 'stop' },
    createdAt: new Date('2026-08-30T00:00:00Z'),
    metadata: { custom }
  }) as unknown as ThreadMessage

const user = (text = 'Synthetic question'): ThreadMessage =>
  ({
    id: 'user-1',
    role: 'user',
    content: [{ type: 'text', text }],
    attachments: [],
    createdAt: new Date('2026-08-30T00:00:00Z'),
    metadata: { custom: {} }
  }) as unknown as ThreadMessage

function Harness({
  custom = {},
  running = false,
  userText
}: {
  custom?: Record<string, unknown>
  running?: boolean
  userText?: string
}) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [user(userText), assistant(custom, running)],
    isRunning: running,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

describe('direct response identity presentation', () => {
  it('renders the exact resolved name, role, and decorative avatar without exposing route or bytes as text', async () => {
    const { container } = render(<Harness />)

    await screen.findByText('Synthetic reply')
    expect(screen.getByText('Kite')).toBeTruthy()
    expect(screen.getByText('TikTok Channel Steward')).toBeTruthy()

    const rail = container.querySelector('[data-slot="assistant-session-identity"]')
    const avatar = rail?.querySelector('img')

    expect(avatar?.getAttribute('alt')).toBe('')
    expect(avatar?.getAttribute('aria-hidden')).toBe('true')
    expect(avatar?.getAttribute('tabindex')).toBeNull()
    expect(rail?.textContent).not.toContain('source-a')
    expect(rail?.textContent).not.toContain('base64')
  })

  it('fails closed to a deterministic non-person fallback when exact identity is unavailable', async () => {
    identity.value = { appearance: null, owner: null }
    render(<Harness />)

    await screen.findByText('Synthetic reply')
    expect(screen.getByText('Assistant')).toBeTruthy()
    expect(screen.getByText('Identity unavailable')).toBeTruthy()
  })

  it('uses a quiet non-color fallback with stable geometry when the avatar asset is unavailable', async () => {
    identity.value = {
      appearance: { avatarDataUrl: null, displayName: 'Kite', role: 'TikTok Channel Steward' },
      owner: { connectionId: 'source-a', profile: 'kite', targetProfile: 'kite' }
    }
    const { container } = render(<Harness />)

    await screen.findByText('Synthetic reply')
    const rail = container.querySelector('[data-slot="assistant-session-identity"]')
    const fallback = rail?.firstElementChild

    expect(rail?.querySelector('img')).toBeNull()
    expect(fallback?.className).toContain('size-8')
    expect(fallback?.textContent).toBe('AI')
    expect(container.querySelector('[role="alert"]')).toBeNull()
    expect(rail?.className).not.toContain('animate-')
    expect(rail?.className).not.toContain('transition-')
  })

  it('retains the same identity block node when a response settles', async () => {
    const { container, rerender } = render(<Harness running />)

    await screen.findByText('Synthetic reply')
    const runningIdentity = container.querySelector('[data-slot="assistant-session-identity"]')

    rerender(<Harness running={false} />)
    await screen.findByText('Synthetic reply')

    expect(container.querySelector('[data-slot="assistant-session-identity"]')).toBe(runningIdentity)
  })

  it('keeps the responding employee identity visible when structured author metadata exists', async () => {
    identity.value = {
      appearance: { avatarDataUrl: null, displayName: 'Kite', role: 'TikTok Channel Steward' },
      owner: { connectionId: 'source-a', profile: 'kite', targetProfile: 'kite' }
    }
    const { container } = render(<Harness custom={{ author: { profile: 'structured-source' } }} />)

    await screen.findByText('Synthetic reply')
    expect(container.querySelector('[data-slot="assistant-session-identity"]')).not.toBeNull()
    expect(screen.getByText('Kite')).toBeTruthy()
    expect(screen.getByText('TikTok Channel Steward')).toBeTruthy()
  })

  it('keeps the responding employee identity visible on a collapsed inter-agent reply', async () => {
    identity.value = {
      appearance: { avatarDataUrl: null, displayName: 'Vela', role: 'Wellness and Wholeness Steward' },
      owner: { connectionId: 'source-a', profile: 'vela', targetProfile: 'vela' }
    }
    const { container } = render(<Harness userText={'Message from 🤖 sender (@sender):\nSynthetic relay'} />)

    await screen.findByText('Replied to sender')
    expect(container.querySelector('[data-slot="assistant-session-identity"]')).not.toBeNull()
    expect(screen.getByText('Vela')).toBeTruthy()
    expect(screen.getByText('Wellness and Wholeness Steward')).toBeTruthy()
  })
})
