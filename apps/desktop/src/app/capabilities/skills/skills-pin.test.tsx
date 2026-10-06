// @vitest-environment jsdom
import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { queryClient } from '@/lib/query-client'
import { notify, notifyError } from '@/store/notifications'

const getSkills = vi.fn()
const getToolsets = vi.fn()
const getUsageAnalytics = vi.fn()
const getProfiles = vi.fn()
const getSkillContent = vi.fn()
const getOfficialSkills = vi.fn()
const setSkillPinned = vi.fn()

// Partial mock, like ./index.test.tsx: the real module stays (CapabilitiesView
// pulls in @/store/profile, whose import-time subscription needs it) and only
// the calls under test are stubbed. The profile scope is forwarded, not
// swallowed — the write has to be scoped exactly like the read.
vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  getSkills: (profile?: unknown) => getSkills(profile),
  getToolsets: (profile?: unknown) => getToolsets(profile),
  getUsageAnalytics: (days: number, profile?: unknown) => getUsageAnalytics(days, profile),
  getProfiles: () => getProfiles(),
  getSkillContent: (name: string, profile?: unknown) => getSkillContent(name, profile),
  getOfficialSkills: (profile?: unknown) => getOfficialSkills(profile),
  setSkillPinned: (name: string, pinned: boolean, profile?: unknown) => setSkillPinned(name, pinned, profile)
}))

vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

// Imported at module scope so the heavy component-tree transform is paid during
// collection, not billed against the first test's timeout (see ./index.test.tsx).
const { CapabilitiesView } = await import('../index')

const SKILL_MD = { name: 'x', path: '/skills/x/SKILL.md', content: '---\nname: x\n---\n\n# x\n\nBody.' }

function skill(overrides: Record<string, unknown> = {}) {
  return {
    name: 'learned-skill',
    description: 'A learned skill',
    category: 'research',
    enabled: true,
    usage: 5,
    provenance: 'agent',
    pinned: false,
    ...overrides
  }
}

async function renderSkills() {
  let result: ReturnType<typeof render>
  await act(async () => {
    result = render(
      // CapabilitiesView reads skills via useQuery, so it needs a provider; the
      // optimistic repaint the pin button relies on is written into that cache.
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={['/capabilities?tab=skills']}>
          <CapabilitiesView />
        </MemoryRouter>
      </QueryClientProvider>
    )
  })

  return result!
}

beforeEach(() => {
  Element.prototype.scrollIntoView = vi.fn()
  getSkills.mockResolvedValue([skill()])
  getToolsets.mockResolvedValue([])
  getUsageAnalytics.mockResolvedValue({ tools: [] })
  getOfficialSkills.mockResolvedValue({ skills: [] })
  getSkillContent.mockResolvedValue(SKILL_MD)
  getProfiles.mockResolvedValue({ profiles: [{ name: 'default', is_default: true }] })
  setSkillPinned.mockResolvedValue({ ok: true, name: 'learned-skill', pinned: true, managed: true, message: 'Pinned.' })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  queryClient.clear()
})

describe('Skills pin control', { timeout: 60_000 }, () => {
  it('pins a learned skill from the detail pane, scoped like the list read', async () => {
    await renderSkills()

    const pin = await screen.findByRole('button', { name: 'Pin' })
    await act(async () => {
      fireEvent.click(pin)
    })

    await waitFor(() => expect(setSkillPinned).toHaveBeenCalled())
    expect(setSkillPinned.mock.calls[0].slice(0, 2)).toEqual(['learned-skill', true])
    // Same scope argument the list was fetched with: a pin can never land on
    // another profile than the one the pane is showing.
    expect(setSkillPinned.mock.calls[0][2]).toBe(getSkills.mock.calls.at(-1)?.[0])
    expect(vi.mocked(notify)).toHaveBeenCalled()
  })

  it('repaints the toggle immediately and offers the inverse action', async () => {
    await renderSkills()

    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: 'Pin' }))
    })

    // Optimistic: the label flips before any refetch, and unpinning is offered.
    expect(await screen.findByRole('button', { name: 'Unpin' })).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Pin' })).toBeNull()
  })

  it('rolls the toggle back and surfaces the refusal', async () => {
    // The backend refuses a name the curator would never archive anyway.
    setSkillPinned.mockResolvedValue({
      ok: false,
      name: 'learned-skill',
      pinned: false,
      reason: 'not_eligible',
      message: 'This skill is not curation-eligible (protected built-in or external mount).'
    })

    await renderSkills()
    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: 'Pin' }))
    })

    await waitFor(() => expect(vi.mocked(notifyError)).toHaveBeenCalled())
    // The refusal message is the error, not a generic failure string.
    expect((vi.mocked(notifyError).mock.calls[0][0] as Error).message).toContain('not curation-eligible')
    expect(await screen.findByRole('button', { name: 'Pin' })).toBeTruthy()
  })

  it('does not offer the pin control on a built-in skill', async () => {
    // A bundled skill is managed by its source: no edit/archive/pin at all.
    getSkills.mockResolvedValue([skill({ name: 'bundled-skill', provenance: 'bundled' })])

    await renderSkills()
    await screen.findByRole('heading', { name: 'bundled-skill' })

    expect(screen.queryByRole('button', { name: 'Pin' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Archive' })).toBeNull()
  })

  it('does not offer the pin control on an external mount, which stays editable', async () => {
    // An external mount keeps its in-place edit rights (commit 8c8fc6c1ec) but
    // the curator never archives it, so there is nothing for a pin to protect.
    getSkills.mockResolvedValue([skill({ name: 'external-skill', provenance: 'external' })])

    await renderSkills()
    await screen.findByRole('heading', { name: 'external-skill' })

    expect(screen.queryByRole('button', { name: 'Pin' })).toBeNull()
    expect(screen.getByRole('button', { name: 'Archive' })).toBeTruthy()
  })

  it('hides the control when the runtime predates the pin endpoint', async () => {
    // An older backend omits `pinned` entirely and has no /api/skills/pin to
    // write to, so the pane must not offer a button whose only outcome is an
    // error toast (desktop and runtime update on separate clocks).
    getSkills.mockResolvedValue([skill({ pinned: undefined })])

    await renderSkills()
    await screen.findByRole('heading', { name: 'learned-skill' })

    expect(screen.queryByRole('button', { name: 'Pin' })).toBeNull()
    // Still a learned skill: edit/archive are unaffected.
    expect(screen.getByRole('button', { name: 'Archive' })).toBeTruthy()
  })
})
