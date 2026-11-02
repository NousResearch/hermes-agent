// @vitest-environment jsdom
import { QueryClientProvider, useQuery } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { queryClient } from '@/lib/query-client'
import type { SkillInfo } from '@/types/hermes'

import { SkillDetail } from './skill-detail'
import { skillsQueryKey } from './skills-data'

const setSkillLocked = vi.fn()
const notifyError = vi.fn()

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  getSkillContent: async (name: string) => ({ content: `---\nname: ${name}\n---\n\n# body\n`, name, path: '' }),
  setSkillLocked: (name: string, locked: boolean, profile?: unknown) => setSkillLocked(name, locked, profile)
}))

vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: (...args: unknown[]) => notifyError(...args) }))

const PROFILE = 'researcher'

function seed(skill: SkillInfo) {
  queryClient.setQueryData<SkillInfo[]>(skillsQueryKey(PROFILE), [skill])
}

// Reads the row from the list cache like the real catalog does (a disabled observer:
// optimistic writes and rollbacks re-render it, nothing refetches).
function Harness({ name }: { name: string }) {
  const { data } = useQuery<SkillInfo[]>({ enabled: false, queryFn: () => [], queryKey: skillsQueryKey(PROFILE) })
  const skill = data?.find(row => row.name === name)

  return skill ? <SkillDetail onArchive={() => {}} onEdit={() => {}} profile={PROFILE} skill={skill} /> : null
}

function renderDetail(name: string) {
  render(
    <QueryClientProvider client={queryClient}>
      <Harness name={name} />
    </QueryClientProvider>
  )
}

const lockedOf = (name: string) =>
  queryClient.getQueryData<SkillInfo[]>(skillsQueryKey(PROFILE))?.find(row => row.name === name)?.locked

describe('SkillDetail lock control', () => {
  beforeEach(() => {
    setSkillLocked.mockReset()
    notifyError.mockReset()
    queryClient.clear()
  })

  afterEach(() => cleanup())

  it('locks any provenance (bundled here) with the scoped profile', async () => {
    seed({ category: 'x', description: '', enabled: true, locked: false, name: 'arxiv', provenance: 'bundled' })
    setSkillLocked.mockResolvedValue({ locked: true, name: 'arxiv', ok: true })
    renderDetail('arxiv')

    const toggle = screen.getByRole('button', { name: /lock/i })
    expect(toggle.getAttribute('aria-pressed')).toBe('false')
    fireEvent.click(toggle)

    await waitFor(() => expect(setSkillLocked).toHaveBeenCalledWith('arxiv', true, PROFILE))
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /locked/i }).getAttribute('aria-pressed')).toBe('true')
    )
    expect(lockedOf('arxiv')).toBe(true)
  })

  it('rolls the optimistic flip back and reports when the write fails', async () => {
    seed({ category: 'x', description: '', enabled: true, locked: true, name: 'mine', provenance: 'agent' })
    setSkillLocked.mockRejectedValue(new Error('not writable'))
    renderDetail('mine')

    fireEvent.click(screen.getByRole('button', { name: /locked/i }))

    await waitFor(() => expect(setSkillLocked).toHaveBeenCalledWith('mine', false, PROFILE))
    await waitFor(() => expect(notifyError).toHaveBeenCalled())
    expect(lockedOf('mine')).toBe(true)
    expect(screen.getByRole('button', { name: /locked/i }).getAttribute('aria-pressed')).toBe('true')
  })
})
