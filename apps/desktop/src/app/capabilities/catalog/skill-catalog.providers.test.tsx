import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, configure, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { queryClient } from '@/lib/query-client'
import type { SkillInfo } from '@/types/hermes'

import { parseCatalog } from './catalog-data'
import completed from './provider-fixtures/confirm-completed.json'
import official from './provider-fixtures/official-ready.json'
import stock from './provider-fixtures/official-stock.json'
import preview from './provider-fixtures/preview-ready.json'
import review from './provider-fixtures/review-acknowledge.json'
import search from './provider-fixtures/search-first-page.json'
import pageTwo from './provider-fixtures/search-page-two.json'
import { SkillCatalog } from './skill-catalog'
import { $catalogCardView } from './store'

const source = official.provider_sources[0]
const identifier = search.results[0].identifier
const api = vi.fn()
let reviewResponse: unknown
let confirmResponse: unknown
let searchResponse: unknown
const profile = { connectionId: 'local', profile: 'alice' }

beforeEach(() => {
  configure({ asyncUtilTimeout: 1000 })
  $catalogCardView.set(true)
  queryClient.clear()
  queryClient.setQueryData(['public-catalog', 'skills'], parseCatalog('skills', [
    { name: 'Example skill', description: 'Public lookalike', source: 'github', identifier: 'github:org/example', installIdentifier: 'github:org/example' }
  ]))
  reviewResponse = review
  confirmResponse = completed
  searchResponse = search
  api.mockReset().mockImplementation(async ({ path }: { path: string }) => {
    if (path === '/api/skills/hub/official') { return official }

    if (path === '/api/skills/hub/sources') { return { sources: [source], installed: {}, featured: [] } }

    if (path === '/api/skills/hub/updates') { return { updates: [] } }

    if (path.startsWith('/api/skills/hub/search')) {
      return new URL(path, 'http://fixture').searchParams.has('cursor') ? pageTwo : searchResponse
    }

    if (path.startsWith('/api/skills/hub/preview')) { return preview }

    if (path === '/api/skills/hub/review') { return reviewResponse }

    if (path === '/api/skills/hub/confirm') { return confirmResponse }

    if (path.startsWith('/api/skills/hub/operation')) { return { ...completed, settlement: 'settled' } }
    throw new Error(`Unexpected request ${path}`)
  })
  window.hermesDesktop = { api } as unknown as typeof window.hermesDesktop
})

afterEach(() => { cleanup(); queryClient.clear() })

function setup(skills: SkillInfo[] = [], scope = profile) {
  const props = { profile: scope, skills, renderInstalledDetail: (skill: SkillInfo) => <button>Local delete {skill.name}</button> }

  return render(<QueryClientProvider client={queryClient}><SkillCatalog {...props} /></QueryClientProvider>)
}

async function teamCard() {
  await screen.findByText('Team skills')

  return (await screen.findAllByRole('article')).find(row => row.textContent?.includes('Team skills'))!
}

it.each(['never-used', 'previously-used-disabled', 'flag-off', 'signed-out', 'failed-registration'])('stock fixture: %s has identical rows/actions/requests', async () => {
  api.mockImplementation(async ({ path }: { path: string }) => {
    expect(path).toBe('/api/skills/hub/official')

    return stock
  })
  const { container } = setup()
  await waitFor(() => expect(api).toHaveBeenCalledTimes(1))
  expect(container.textContent).toMatchSnapshot()
  expect(api.mock.calls.map(([request]) => request)).toEqual([{ path: '/api/skills/hub/official', connectionId: 'local', profile: 'alice', priority: 'foreground' }])
})

it('bootstraps descriptors without hub installs and keeps same-named public and provider cards distinct', async () => {
  setup()
  const card = await teamCard()
  expect(within(card).getByRole('button', { name: 'Install Example skill' })).toBeTruthy()
  expect(screen.getAllByRole('article')).toHaveLength(2)
  fireEvent.click(screen.getByRole('button', { name: 'Source' }))
  fireEvent.click(screen.getByRole('button', { name: 'Team skills' }))
  expect(screen.getAllByRole('article')).toHaveLength(1)
  expect(api.mock.calls.filter(([request]) => request.path.startsWith('/api/skills/hub/search')).every(([request]) => request.connectionId === 'local' && request.profile === 'alice')).toBe(true)
})

it('browses subsequent provider pages without hiding the later rows', async () => {
  setup()
  await teamCard()
  fireEvent.click(await screen.findByRole('button', { name: 'Load more Team skills' }))
  await screen.findByRole('button', { name: pageTwo.results[0].name })
  expect(api.mock.calls.some(([request]) => request.path.includes(`cursor=${search.next_cursor}`))).toBe(true)
})

it('previews uninstalled bytes and confirms only the exact individually acknowledged findings', async () => {
  setup()
  const card = await teamCard()
  fireEvent.click(within(card).getByRole('button', { name: 'Example skill' }))
  await screen.findByText(/Follow the approved plan/)
  fireEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Install Example skill' }))
  const dialog = await screen.findByRole('dialog', { name: 'Review install' })
  const confirm = await within(dialog).findByRole<HTMLButtonElement>('button', { name: 'Confirm install' })
  expect(confirm.disabled).toBe(true)
  expect(within(dialog).getByText('SKILL.md')).toBeTruthy()
  fireEvent.click(within(dialog).getByRole('checkbox', { name: /This skill describes network access/ }))
  expect(confirm.disabled).toBe(false)
  fireEvent.click(confirm)
  await waitFor(() => expect(api.mock.calls.some(([request]) => request.path === '/api/skills/hub/confirm')).toBe(true))
  expect(api.mock.calls.find(([request]) => request.path === '/api/skills/hub/confirm')![0]).toMatchObject({ connectionId: 'local', profile: 'alice', body: { identifier, action: 'install', review_token: review.review_token, acknowledgements: ['network-access'] } })
  await screen.findByText('installed', { selector: '[role="status"]' })
})

it.each(['block', 'unavailable'])('never confirms %s policy even if a malformed response offers a token', async policy => {
  reviewResponse = { ...review, scan: { ...review.scan, policy } }
  setup()
  fireEvent.click(within(await teamCard()).getByRole('button', { name: 'Install Example skill' }))
  const dialog = await screen.findByRole('dialog', { name: 'Review install' })
  await within(dialog).findByText('Review the finding.')
  expect(within(dialog).queryByRole('button', { name: 'Confirm install' })).toBeNull()
})

it.each(['installed', 'removed', 'unmanaged_local', 'pending', 'unavailable'] as const)('shows %s management state without local delete or name-based merging', async managedState => {
  const managed = { name: 'Example skill', description: 'Managed bytes', category: 'uncategorized', enabled: true, provenance: 'provider', provider: { ...search.results[0].provider, managed_state: managedState, actions: ['update', 'uninstall'] }, identifier, target_relative_path: 'team/example', installed_version: '1', latest_version: '2', managed_state: managedState, installation_revision: 'r1', actions: ['update', 'uninstall'], settlement: 'settled' } as SkillInfo
  setup([managed, { name: 'Example skill', description: 'Hub bytes', category: 'uncategorized', provenance: 'hub', enabled: true }])
  const rows = await screen.findAllByRole('article')
  await waitFor(() => expect(screen.getAllByRole('article').length).toBeGreaterThanOrEqual(2))
  fireEvent.click(within(await teamCard()).getByRole('button', { name: 'Example skill' }))
  const dialog = screen.getByRole('dialog')
  await within(dialog).findByText(managedState === 'installed' ? 'Installed' : { removed: 'Removed outside this source', unmanaged_local: 'Unmanaged local copy', pending: 'Operation pending', unavailable: 'Source unavailable' }[managedState])
  expect(within(dialog).queryByText('Local delete Example skill')).toBeNull()
  expect(rows.length).toBeGreaterThan(0)
})

it.each(['update', 'uninstall'] as const)('uses the same exact review flow for managed %s', async action => {
  reviewResponse = { ...review, action }
  confirmResponse = { ...completed, action, outcome: action === 'update' ? 'updated' : 'uninstalled' }
  const managed = { name: 'Example skill', description: 'Managed', category: 'uncategorized', enabled: true, provenance: 'provider', provider: { ...search.results[0].provider, managed_state: 'installed' }, identifier, target_relative_path: 'example-skill', installed_version: '1', latest_version: '2', managed_state: 'installed', installation_revision: 'r1', actions: ['update', 'uninstall'] } as SkillInfo
  setup([managed])
  fireEvent.click(within(await teamCard()).getByRole('button', { name: `Review ${action}` }))
  const confirm = await screen.findByRole('button', { name: `Confirm ${action}` })
  fireEvent.click(screen.getByRole('checkbox'))
  fireEvent.click(confirm)
  await waitFor(() => expect(api.mock.calls.find(([request]) => request.path === '/api/skills/hub/confirm')?.[0].body).toMatchObject({ identifier, action, acknowledgements: ['network-access'] }))
  expect(api.mock.calls.find(([request]) => request.path === '/api/skills/hub/review')?.[0].body).toEqual({ identifier, action, ...(action === 'update' ? { version: '2' } : {}) })
})

it.each(['sign_in_required', 'unavailable', 'hidden'] as const)('source %s never masquerades as an empty ready library', async state => {
  api.mockImplementation(async () => ({ skills: [], provider_sources: [{ ...source, state }] }))
  setup()

  if (state === 'hidden') { await waitFor(() => expect(api).toHaveBeenCalledTimes(1)); expect(screen.queryByText('Team skills')).toBeNull() }
  else { await screen.findByText(state === 'sign_in_required' ? /Sign in through/ : /This source is unavailable/) }

  expect(api.mock.calls.some(([request]) => request.path.includes('/search'))).toBe(false)
})

it('recovers a lost confirm response with the original token and no second review', async () => {
  confirmResponse = { ok: false, error: { code: 'outcome_unknown', message: 'Response lost' } }
  setup()
  fireEvent.click(within(await teamCard()).getByRole('button', { name: 'Install Example skill' }))
  fireEvent.click(await screen.findByRole('checkbox'))
  fireEvent.click(screen.getByRole('button', { name: 'Confirm install' }))
  const recover = await screen.findByRole('button', { name: 'Recover original operation' })
  confirmResponse = completed
  fireEvent.click(recover)
  await screen.findByText('installed', { selector: '[role="status"]' })
  const confirms = api.mock.calls.filter(([request]) => request.path === '/api/skills/hub/confirm')
  expect(confirms).toHaveLength(2)
  expect(confirms[0][0]).toEqual(confirms[1][0])
  expect(api.mock.calls.filter(([request]) => request.path === '/api/skills/hub/review')).toHaveLength(1)
})

it('shows a terminal outcome while finalizing and performs read-only status recovery', async () => {
  confirmResponse = { ...completed, settlement: 'reconciling' }
  setup()
  fireEvent.click(within(await teamCard()).getByRole('button', { name: 'Install Example skill' }))
  fireEvent.click(await screen.findByRole('checkbox'))
  fireEvent.click(screen.getByRole('button', { name: 'Confirm install' }))
  await waitFor(() => expect(api.mock.calls.some(([request]) => request.path.startsWith('/api/skills/hub/operation'))).toBe(true))
  expect(api.mock.calls.filter(([request]) => request.path === '/api/skills/hub/confirm')).toHaveLength(1)
})

it.each([
  { preview_complete: false },
  { scan: { ...review.scan, complete: false } },
  { acknowledgements_required: ['unshown-finding'] },
  { review_token: null },
  { expires_at_ms: 1 }
])('does not confirm an incomplete or expired plan %j', async delta => {
  reviewResponse = { ...review, ...delta }
  setup()
  fireEvent.click(within(await teamCard()).getByRole('button', { name: 'Install Example skill' }))
  await screen.findByText('Review the finding.')
  const confirm = screen.queryByRole<HTMLButtonElement>('button', { name: 'Confirm install' })

  if (confirm) { expect(confirm.disabled).toBe(true) }
  else { expect(screen.getByText('The review is incomplete or the provider policy does not allow confirmation.')).toBeTruthy() }
})

it('does not carry a foreign response into the requested review', async () => {
  reviewResponse = { ...review, identifier: 'foreign-source:item' }
  setup()
  fireEvent.click(within(await teamCard()).getByRole('button', { name: 'Install Example skill' }))
  await screen.findByRole('alert')
  expect(screen.queryByRole('button', { name: 'Confirm install' })).toBeNull()
})

it('replaces the inspector with the review instead of stacking two dialogs', async () => {
  setup()
  fireEvent.click(within(await teamCard()).getByRole('button', { name: 'Example skill' }))
  fireEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Install Example skill' }))
  await screen.findByRole('button', { name: 'Confirm install' })
  expect(window.document.querySelectorAll('[role="dialog"]')).toHaveLength(1)
})

it('keeps one working provider visible when another source fails', async () => {
  const other = { ...source, id: source.id.replace('team', 'other'), label: 'Other team' }
  api.mockImplementation(async ({ path }: { path: string }) => {
    if (path.endsWith('/official')) { return { skills: [], provider_sources: [source, other] } }

    if (path.includes('/search')) {
      if (new URL(path, 'http://fixture').searchParams.get('source') === other.id) { throw new Error('Other source offline') }

      return search
    }

    throw new Error('Unexpected request')
  })
  setup()
  await screen.findByText('Other source offline')
  expect(screen.getAllByRole('article').some(article => article.textContent?.includes('Team skills'))).toBe(true)
})

it('keeps a same-named public card visible beside a team install', async () => {
  setup([{ name: 'Example skill', description: 'Managed', category: '', enabled: true, provenance: 'provider', provider: search.results[0].provider, identifier, target_relative_path: 'team/example', installed_version: '1', latest_version: '2', managed_state: 'installed', installation_revision: 'r1', actions: ['update', 'uninstall'] } as SkillInfo])
  await teamCard()
  expect(screen.getAllByRole('article')).toHaveLength(2)
})

it.each(['review_changed', 'review_expired', 'ownership_conflict'])('%s clears all old consent and requests a new review', async code => {
  setup()
  fireEvent.click(within(await teamCard()).getByRole('button', { name: 'Install Example skill' }))
  fireEvent.click(await screen.findByRole('checkbox'))
  confirmResponse = { ok: false, error: { code, message: 'Review again' } }
  fireEvent.click(screen.getByRole('button', { name: 'Confirm install' }))
  await screen.findByText('This review is no longer current. Close it and request a new review.')
  expect(screen.queryByRole('checkbox')).toBeNull()
  expect(screen.queryByRole('button', { name: 'Recover original operation' })).toBeNull()
})

it('leaves a prior operation recoverable after closing and reopening its card', async () => {
  confirmResponse = { ok: false, error: { code: 'outcome_unknown', message: 'Lost response' } }
  setup()
  fireEvent.click(within(await teamCard()).getByRole('button', { name: 'Install Example skill' }))
  fireEvent.click(await screen.findByRole('checkbox'))
  fireEvent.click(screen.getByRole('button', { name: 'Confirm install' }))
  await screen.findByRole('button', { name: 'Recover original operation' })
  fireEvent.click(screen.getAllByRole('button', { name: 'Close' })[0])
  fireEvent.click(within(await teamCard()).getByRole('button', { name: 'Install Example skill' }))
  await screen.findByRole('button', { name: 'Recover original operation' })
  expect(api.mock.calls.filter(([request]) => request.path === '/api/skills/hub/review')).toHaveLength(1)
})

it('keeps local deletion unavailable while descriptor bootstrap is still pending', async () => {
  api.mockImplementation(() => new Promise(() => undefined))
  setup([{ name: 'Managed', description: '', category: '', enabled: true, provenance: 'provider', provider: search.results[0].provider, identifier, target_relative_path: 'team/example', installed_version: '1', latest_version: '1', managed_state: 'installed', installation_revision: 'r1', actions: ['uninstall'] } as SkillInfo])
  fireEvent.click(screen.getByRole('button', { name: 'Managed' }))
  expect(screen.queryByText('Local delete Managed')).toBeNull()
})

it('treats a disabled provider install as a normal local skill with no provider requests', async () => {
  api.mockResolvedValue(stock)
  setup([{ name: 'Local remainder', description: '', category: '', enabled: true, provenance: 'agent' }])
  fireEvent.click(screen.getByRole('button', { name: 'Local remainder' }))
  expect(screen.getByText('Local delete Local remainder')).toBeTruthy()
  expect(api.mock.calls.map(([request]) => request.path)).toEqual(['/api/skills/hub/official'])
})

it('never mistakes a provider partial failure for an empty library', async () => {
  searchResponse = { ...search, results: [], provider_errors: [{ source: source.id, code: 'provider_unavailable', message: 'Source offline' }] }
  setup()
  await screen.findByText('Source offline')
  expect(screen.queryByText('This source has no matching skills.')).toBeNull()
})

it('drops a late provider response when the connection changes', async () => {
  let resolve!: (value: unknown) => void
  api.mockImplementation(({ path, connectionId }: { path: string; connectionId: string }) => {
    if (path === '/api/skills/hub/official') { return connectionId === 'local' ? new Promise(done => { resolve = done }) : Promise.resolve(stock) }
    throw new Error('Unexpected request')
  })
  const view = setup()
  await waitFor(() => expect(resolve).toBeTruthy())
  view.rerender(<QueryClientProvider client={queryClient}><SkillCatalog profile={{ connectionId: 'remote', profile: 'alice' }} renderInstalledDetail={() => null} skills={[]} /></QueryClientProvider>)
  await act(async () => resolve(official))
  expect(screen.queryByText('Team skills')).toBeNull()
  expect(api.mock.calls.filter(([request]) => request.path.includes('/search'))).toHaveLength(0)
})
