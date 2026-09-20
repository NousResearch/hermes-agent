import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { DROPDOWN_KIT } from '@/components/ui/actions-menu'
import { DropdownMenu, DropdownMenuContent } from '@/components/ui/dropdown-menu'
import { searchSessions } from '@/hermes'
import { requestGatewayForAgent } from '@/store/gateway'
import { refreshProjectTree } from '@/store/projects'
import { $sessions } from '@/store/session'
import { $archivedSessions } from '@/store/sidebar-archive'

import { mergeSessionSearchResults } from './search-results'
import { SessionTagsMenu } from './session-tags-menu'
import { useSessionSearch } from './use-session-search'
vi.mock('@/hermes', () => ({ searchSessions: vi.fn() }))
vi.mock('@/store/projects', () => ({ refreshProjectTree: vi.fn() }))
vi.mock('@/store/sidebar-archive', () => ({ $archivedSessions: atom([]) }))
vi.mock('@/store/session', () => ({
  $sessions: atom([{ id: 's', profile: 'p', connection_id: 'remote', tags: ['a'] }]),
  $cronSessions: atom([]),
  $messagingSessions: atom([]),
  $connection: atom({ profile: 'other', connectionId: 'wrong' }),
  sessionMatchesStoredId: (s: { id: string }, id: string) => s.id === id
}))
vi.mock('@/store/gateway', () => ({
  requestGatewayForAgent: vi.fn(async (_c, _p, method, params) =>
    method === 'session.tags.list'
      ? { tags: ['a', 'b'] }
      : {
          tags: params.assigned
            ? [...new Set([...($sessions.get()[0].tags ?? []), params.tag])]
            : ($sessions.get()[0].tags ?? []).filter(t => t !== params.tag)
        }
  )
}))
vi.mock('@/store/notifications', () => ({ notifyError: vi.fn() }))
beforeEach(() => {
  $sessions.set([{ id: 's', profile: 'p', connection_id: 'remote', tags: ['a'] }] as ReturnType<typeof $sessions.get>)
  $archivedSessions.set([])
  vi.mocked(requestGatewayForAgent).mockImplementation(async (_c, _p, method, params: any) =>
    method === 'session.tags.list'
      ? { tags: ['a', 'b'] }
      : {
          tags: params.assigned
            ? [...new Set([...($sessions.get()[0].tags ?? []), params.tag])]
            : ($sessions.get()[0].tags ?? []).filter(t => t !== params.tag)
        }
  )
})
afterEach(cleanup)

it('refetches FTS-only hits after successful menu removal without changing the query', async () => {
  $sessions.set([])
  let tags = ['a']
  vi.mocked(searchSessions).mockImplementation(async () => ({
    results: [{ session_id: 'old', snippet: 'needle', tags, model: null, role: null, source: null, session_started: null }]
  }))
  vi.mocked(requestGatewayForAgent).mockImplementation(async (_c, _p, method) => {
    if (method === 'session.tags.list') {return { tags: ['a'] }}
    tags = []

    return { tags }
  })

  function SearchList() {
    const { serverMatches } = useSessionSearch('needle')
    const rows = mergeSessionSearchResults('needle', [], serverMatches, new Map(), row => !!row.tags?.includes('a'))

    return <div data-testid="search-results">{rows.map(row => row.id).join(',')}</div>
  }

  render(<>
    <SearchList />
    <DropdownMenu open><DropdownMenuContent>
      <SessionTagsMenu connectionId="remote" kit={DROPDOWN_KIT} profile="p" sessionId="old" tags={['a']} />
    </DropdownMenuContent></DropdownMenu>
  </>)
  await waitFor(() => expect(screen.getByTestId('search-results').textContent).toBe('old'))
  fireEvent.click(await screen.findByRole('menuitemcheckbox', { name: 'a' }))
  await waitFor(() => expect(screen.getByTestId('search-results').textContent).toBe(''))
  expect(vi.mocked(searchSessions).mock.calls.every(([query]) => query === 'needle')).toBe(true)
  expect($sessions.get()).toEqual([])
})
it('updates a hydrated older row outside recents and refreshes the project tree after assignment', async () => {
  $sessions.set([])
  vi.mocked(refreshProjectTree).mockClear()
  vi.mocked(requestGatewayForAgent).mockImplementation(async (_c, _p, method) =>
    method === 'session.tags.list' ? { tags: ['a', 'b'] } : { tags: ['b'] }
  )
  render(
    <DropdownMenu open>
      <DropdownMenuContent>
        <SessionTagsMenu connectionId="remote" kit={DROPDOWN_KIT} profile="p" sessionId="old" tags={['a']} />
      </DropdownMenuContent>
    </DropdownMenu>
  )
  expect((await screen.findByRole('menuitemcheckbox', { name: 'a' })).getAttribute('aria-checked')).toBe('true')
  fireEvent.click(screen.getByRole('menuitemcheckbox', { name: 'a' }))
  await waitFor(() => expect(refreshProjectTree).toHaveBeenCalledOnce())
  expect(requestGatewayForAgent).toHaveBeenLastCalledWith('remote', 'p', 'session.tags.set', {
    profile: 'p',
    session_id: 'old',
    tag: 'a',
    assigned: false
  })
  expect(screen.getByRole('menuitemcheckbox', { name: 'b' }).getAttribute('aria-checked')).toBe('true')
})
it('creates and assigns immediately; multiple checkbox toggles leave the menu open and route to the owner', async () => {
  render(
    <DropdownMenu open>
      <DropdownMenuContent>
        <SessionTagsMenu connectionId="remote" kit={DROPDOWN_KIT} profile="p" sessionId="s" />
      </DropdownMenuContent>
    </DropdownMenu>
  )
  fireEvent.click(await screen.findByRole('menuitemcheckbox', { name: 'b' }))
  await waitFor(() => expect($sessions.get()[0].tags ?? []).toEqual(['a', 'b']))
  fireEvent.click(screen.getByRole('menuitemcheckbox', { name: 'a' }))
  await waitFor(() => expect($sessions.get()[0].tags ?? []).toEqual(['b']))
  fireEvent.change(screen.getByRole('textbox', { name: 'New tag' }), { target: { value: 'new' } })
  fireEvent.click(screen.getByRole('button', { name: 'Create and assign' }))
  await waitFor(() => expect($sessions.get()[0].tags ?? []).toEqual(['b', 'new']))
  expect(requestGatewayForAgent).toHaveBeenLastCalledWith('remote', 'p', 'session.tags.set', {
    profile: 'p',
    session_id: 's',
    tag: 'new',
    assigned: true
  })
  expect(screen.getByRole('menuitemcheckbox', { name: 'new' })).toBeTruthy()
})

it('isolates duplicate ids and profiles across connections, including archived rows', async () => {
  $sessions.set([
    { id: 's', profile: 'p', connection_id: 'wrong', tags: ['untouched'] },
    { id: 's', profile: 'p', connection_id: 'remote', tags: ['a'] }
  ] as typeof $sessions extends { get(): infer T } ? T : never)
  $archivedSessions.set([{ id: 's', profile: 'p', connection_id: 'remote', tags: ['a'] }] as ReturnType<
    typeof $archivedSessions.get
  >)
  vi.mocked(requestGatewayForAgent).mockResolvedValue({ tags: ['b'] })
  render(
    <DropdownMenu open>
      <DropdownMenuContent>
        <SessionTagsMenu connectionId="remote" kit={DROPDOWN_KIT} profile="p" sessionId="s" />
      </DropdownMenuContent>
    </DropdownMenu>
  )
  expect((await screen.findByRole('menuitemcheckbox', { name: 'a' })).getAttribute('aria-checked')).toBe('true')
  fireEvent.click(await screen.findByRole('menuitemcheckbox', { name: 'b' }))
  await waitFor(() => expect($sessions.get()[1].tags).toEqual(['b']))
  expect($sessions.get()[0].tags).toEqual(['untouched'])
  expect($archivedSessions.get()[0].tags).toEqual(['b'])
  expect(requestGatewayForAgent).toHaveBeenLastCalledWith('remote', 'p', 'session.tags.set', {
    profile: 'p',
    session_id: 's',
    tag: 'b',
    assigned: true
  })
})
