// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as DesktopGithub from '@/lib/desktop-github'

const listGithubPullRequests = vi.fn()
const getGithubPullRequestDetail = vi.fn()
const githubProfileKey = vi.fn().mockReturnValue('default')

vi.mock('@/lib/desktop-github', async importOriginal => ({
  ...(await importOriginal<typeof DesktopGithub>()),
  githubProfileKey: () => githubProfileKey(),
  getGithubPullRequestDetail: () => getGithubPullRequestDetail(),
  listGithubPullRequests: () => listGithubPullRequests()
}))

const pullRequestsCopy: Record<string, string> = {
  title: 'Pull requests',
  created: 'Created',
  reviewRequested: 'Review requested',
  closed: 'Closed',
  search: 'Search pull requests',
  refresh: 'Refresh',
  loading: 'Loading pull requests',
  loadFailed: 'Failed to load pull requests',
  retry: 'Retry',
  noneCreated: 'No created pull requests',
  noneReview: 'Nothing awaiting your review',
  noneClosed: 'No recently closed pull requests'
}

vi.mock('@/i18n', () => ({
  useI18n: () => ({ t: { pullRequests: pullRequestsCopy } })
}))

const openExternal = vi.fn()
const writeClipboard = vi.fn()

function renderView() {
  return import('./index').then(({ PullRequestsView }) =>
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <MemoryRouter initialEntries={['/pull-requests?tab=created']}>
          <PullRequestsView />
        </MemoryRouter>
      </QueryClientProvider>
    )
  )
}

beforeEach(() => {
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) =>
    window.setTimeout(() => callback(Date.now()), 0)
  )
  vi.stubGlobal('cancelAnimationFrame', (id: number) => window.clearTimeout(id))
  vi.stubGlobal('window', {
    ...window,
    hermesDesktop: { openExternal, writeClipboard }
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  vi.unstubAllGlobals()
})

describe('PullRequestsView error states', () => {
  it('renders retryable error when the list query rejects', async () => {
    listGithubPullRequests.mockRejectedValue(new Error('Network failure'))

    await renderView()

    await screen.findByText('Failed to load pull requests')
    expect(screen.getByText('Network failure')).toBeTruthy()
    expect(screen.getByText('Retry')).toBeTruthy()
  })
})
