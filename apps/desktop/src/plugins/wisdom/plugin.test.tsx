import type * as SdkModule from '@hermes/plugin-sdk'
/**
 * Collective Wisdom desktop plugin: the install path never fires without the hash the user saw.
 * Drives the real component against a stubbed `ctx.rest`; asserts wire calls, not markup.
 */
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('@hermes/plugin-sdk', async () => {
  const real = await vi.importActual<typeof SdkModule>('@hermes/plugin-sdk')

  return { ...real, host: { ...real.host, navigate: vi.fn(), notify: vi.fn(), notifyError: vi.fn() } }
})

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'

import plugin from './plugin'

const HASH = 'sha256:' + 'a'.repeat(64)

function boot() {
  const calls: Array<[string, unknown]> = []

  const rest = vi.fn(async (path: string, opts?: { body?: unknown }) => {
    calls.push([path, opts?.body])

    if (path === '/overview') {
      return {
        entitled: true,
        skills: [{ id: 'sk1', slug: 'deploy', version: 3, installs: 4, description: 'd', security: 'pass' }],
        status: {
          installed: { sk2: { slug: 'notes', version: 1, path: '/x/notes' } },
          updates: [{ skill_id: 'sk2', slug: 'notes', installed: 1, latest: 2, required: false, mode: 'AUTO_WITH_NOTICE', modified: true, action: 'conflict' }],
          notices: []
        },
        candidates: [{ skill: 'standup', reason: 'high_usage', evidence: { consecutive_business_days: 7, used_days: 9 } }]
      }
    }

    if (path === '/share/prepare') {
      return { skill_name: 'standup', slug: 'standup', description: 'Daily standup notes', content_hash: HASH, files: [{ path: 'SKILL.md', bytes: 12 }] }
    }

    if (path === '/plan') {
      return { skill_id: 'sk1', slug: 'deploy', version: 3, content_hash: HASH, security: 'pass — clean', author: null, explanation: null, target: '/x/deploy' }
    }

    return { installed: 'sk1', slug: 'deploy', version: 3, path: '/x/deploy' }
  })

  const contributions: Array<{ id: string; render?: () => ReactNode }> = []
  plugin.register({
    rest,
    onDispose: () => undefined,
    registerMany: (items: Array<{ id: string; render?: () => ReactNode }>) => contributions.push(...items)
  } as never)
  const page = contributions.find(c => c.id === 'page')!.render!
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(<QueryClientProvider client={qc}>{page()}</QueryClientProvider>)

  return { calls }
}

afterEach(cleanup)

describe('wisdom desktop plugin', () => {
  it('resolves an update conflict without touching the package: keep pins the exact version', async () => {
    const { calls } = boot()
    fireEvent.click(await screen.findByRole('button', { name: 'Keep mine' }))
    await waitFor(() => expect(calls.find(([p]) => p === '/update/keep')?.[1]).toEqual({ skill_id: 'sk2', version: 2 }))
    expect(calls.some(([p]) => p === '/install' || p === '/plan')).toBe(false)
  })

  it('shares a candidate only after the prepared package dialog, echoing its hash', async () => {
    const { calls } = boot()
    fireEvent.click(await screen.findByRole('button', { name: 'Share…' }))
    await screen.findByText(HASH)
    expect(calls.some(([p]) => p === '/share')).toBe(false)
    fireEvent.click(await screen.findByRole('button', { name: 'Share' }))
    await waitFor(() => expect(calls.find(([p]) => p === '/share')?.[1]).toEqual({ skill_name: 'standup', description: 'Daily standup notes', content_hash: HASH }))
  })

  it('installs only after the plan dialog, echoing the exact planned hash', async () => {
    const { calls } = boot()
    fireEvent.click(await screen.findByRole('button', { name: 'Install' }))
    await waitFor(() => expect(calls.some(([p]) => p === '/plan')).toBe(true))
    expect(calls.some(([p]) => p === '/install')).toBe(false)
    await screen.findByText(HASH)
    fireEvent.click((await screen.findAllByRole('button', { name: 'Install' })).at(-1)!)
    await waitFor(() => expect(calls.find(([p]) => p === '/install')?.[1]).toEqual({ skill_id: 'sk1', version: 3, content_hash: HASH }))
  })
})
