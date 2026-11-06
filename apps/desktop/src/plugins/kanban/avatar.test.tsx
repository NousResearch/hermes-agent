/**
 * The assignee avatar's identity ladder: a bot with a look renders the Bots
 * face (image > shape+color via the shared appearance resolver), a title-only
 * bot degrades to the title initial, and free-text assignees keep the slug
 * initials. The roster rides the shared `GET /profiles` cache — mocked at the
 * api seam like every kanban test; the avatar asset probe rides
 * `host.request('profiles.get_asset')`, mocked at the sdk seam like the
 * hermes-bots tests do.
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { request } = vi.hoisted(() => ({ request: vi.fn() }))

import type * as KanbanSdk from '@hermes/plugin-sdk'

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const actual = await importOriginal<typeof KanbanSdk>()

  return { ...actual, host: { ...actual.host, request } }
})

import type * as KanbanApi from './api'
import type { KanbanProfile } from './types'

vi.mock('./api', async importOriginal => ({
  ...(await importOriginal<typeof KanbanApi>()),
  fetchProfiles: vi.fn()
}))

import { fetchProfiles } from './api'
import { resetKanbanAvatarImageCache } from './identity'
import { Avatar } from './ui'

function renderAvatar(name: string): HTMLElement {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  const { container } = render(<Avatar name={name} />, {
    wrapper: ({ children }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
  })

  return container
}

function roster(profiles: KanbanProfile[]): void {
  vi.mocked(fetchProfiles).mockResolvedValue({ profiles })
}

beforeEach(() => {
  request.mockReset()
  resetKanbanAvatarImageCache()
  roster([])
})

afterEach(cleanup)

describe('Avatar identity ladder', () => {
  it('renders the Bots face for a bot with a shape, titled by its bot name', async () => {
    roster([
      {
        name: 'engineer1',
        is_default: false,
        description: '',
        description_auto: false,
        bot: { title: 'Hepa (Engineer)', shape: 'blobatar', has_avatar: false }
      }
    ])

    const container = renderAvatar('engineer1')

    const chip = await screen.findByTitle('Hepa (Engineer)')

    expect(chip).toBeTruthy()
    // A shape avatar is the Bots roster's SVG face, not a letter chip.
    expect(container.querySelector('svg')).toBeTruthy()
    expect(container.textContent?.trim()).not.toBe('E')
  })

  it('falls back to the title initial for a bot with a title but no look', async () => {
    roster([
      {
        name: 'engineer2',
        is_default: false,
        description: '',
        description_auto: false,
        bot: { title: 'Daro (Engineer)', has_avatar: false }
      }
    ])

    const container = renderAvatar('engineer2')

    await waitFor(
      () => {
        // eslint-disable-next-line no-console
        console.log('BODY', JSON.stringify(document.body.innerHTML.slice(0, 300)))
        return document.body.innerHTML.length > 0
      },
      { timeout: 3000 }
    ).catch(() => {})
    const chip = await screen.findByTitle('Daro (Engineer)')

    // 'Daro (Engineer)' -> 'D', not the slug's 'E' and not a punctuated 'D('.
    expect(chip.textContent).toBe('D')
  })

  it('paints a legacy inline avatar image without probing the gateway', async () => {
    roster([
      {
        name: 'eng',
        is_default: false,
        description: '',
        description_auto: false,
        bot: { image: 'data:image/png;base64,AA==', shape: 'circle', has_avatar: false }
      }
    ])

    const container = renderAvatar('eng')

    await waitFor(() => expect(container.querySelector('img')).toBeTruthy())
    expect(request).not.toHaveBeenCalled()
  })

  it('probes the avatar asset when the profile carries one, and paints it', async () => {
    request.mockResolvedValue({ found: true, data: 'data:image/png;base64,AA==' })
    roster([
      {
        name: 'eng',
        is_default: false,
        description: '',
        description_auto: false,
        bot: { has_avatar: true, shape: 'circle' }
      }
    ])

    const container = renderAvatar('eng')

    await waitFor(() => expect(container.querySelector('img')).toBeTruthy())
    expect(request).toHaveBeenCalledWith('profiles.get_asset', { asset: 'avatar', name: 'eng' })
  })

  it('keeps the shape when the asset probe fails', async () => {
    request.mockRejectedValue(new Error('gateway asleep'))
    roster([
      {
        name: 'eng',
        is_default: false,
        description: '',
        description_auto: false,
        bot: { has_avatar: true, shape: 'circle' }
      }
    ])

    const container = renderAvatar('eng')

    await waitFor(() => expect(request).toHaveBeenCalled())
    // The catch path resolves null; the shape face stands in, no crash.
    expect(container.querySelector('svg')).toBeTruthy()
    expect(container.querySelector('img')).toBeNull()
  })

  it('treats a customized-but-shapeless bot as a look, not a letter', async () => {
    roster([
      {
        name: 'eng',
        is_default: false,
        description: '',
        description_auto: false,
        bot: { custom: true, has_avatar: false }
      }
    ])

    const container = renderAvatar('eng')

    // The appearance resolver owns the defaults: custom => a real face shape.
    await waitFor(() => expect(container.querySelector('svg')).toBeTruthy())
  })

  it('keeps slug initials for free-text assignees', () => {
    const container = renderAvatar('night ops')

    expect(screen.getByTitle('night ops').textContent).toBe('NO')
    expect(container.querySelector('svg')).toBeNull()
  })

  it('keeps slug initials for profiles without a bot identity', async () => {
    roster([{ name: 'an', is_default: false, description: '', description_auto: false }])

    const container = renderAvatar('an')

    await waitFor(() => expect(vi.mocked(fetchProfiles).mock.calls.length).toBeGreaterThan(0))
    expect(screen.getByTitle('an').textContent).toBe('A')
  })
})
