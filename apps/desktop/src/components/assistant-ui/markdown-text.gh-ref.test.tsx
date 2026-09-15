import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { requestGatewayForProfile } from '@/store/gateway'
import { $activeSessionId } from '@/store/session'

import { __resetGhRepoRefsCache } from './directive-text'
import { MarkdownTextContent } from './markdown-text'

// The no-repo / gateway-down paths: `git.repo_refs` unavailable → plain text,
// never a dead `#gh-ref/` anchor.
let repoRefsResult: { available: boolean; host: string; owner: string; repo: string } | null = null
let repoRefsError = false

vi.mock('@/store/gateway', async importOriginal => {
  const actual = (await importOriginal()) as Record<string, unknown>

  return {
    ...actual,
    activeGatewayProfileKey: () => 'default',
    requestGatewayForProfile: vi.fn(async () => {
      if (repoRefsError) {
        throw new Error('gateway down')
      }

      return repoRefsResult
    })
  }
})

afterEach(() => {
  cleanup()
  $activeSessionId.set(null)
  __resetGhRepoRefsCache()
  repoRefsResult = null
  repoRefsError = false
})

// End-to-end for the agent-authored path: a bare `#N` in assistant markdown has
// to survive preprocessMarkdown -> Streamdown -> MarkdownLink and come out as a
// github.com link when the session's repo resolves — and stay plain text when it
// doesn't. The chip resolves against the per-view session, so the test opens one.
describe('MarkdownTextContent github refs', () => {
  it('renders a bare ref as a link to the session repo', async () => {
    repoRefsResult = { available: true, host: 'github.com', owner: 'KaptenKatthatt', repo: 'newsAgg' }
    $activeSessionId.set('gh-test-1')

    render(<MarkdownTextContent isRunning={false} text="merga PR #46" />)

    const link = await screen.findByTitle('KaptenKatthatt/newsAgg#46')

    expect(link.tagName).toBe('A')
    expect(link.getAttribute('href')).toBe('https://github.com/KaptenKatthatt/newsAgg/issues/46')
    expect(link.textContent).toBe('#46')
  })

  it('leaves a ref as plain text when the repo cannot resolve', async () => {
    repoRefsResult = { available: false, host: '', owner: '', repo: '' }
    $activeSessionId.set('gh-test-2')

    const { container } = render(<MarkdownTextContent isRunning={false} text="merga PR #46" />)

    await waitFor(() => expect(vi.mocked(requestGatewayForProfile)).toHaveBeenCalled())

    expect(screen.queryByTitle('KaptenKatthatt/newsAgg#46')).toBeNull()
    expect(container.querySelector('a')).toBeNull()
    expect(container.textContent).toContain('#46')
  })

  it('leaves a ref as plain text when the gateway errors', async () => {
    repoRefsError = true
    $activeSessionId.set('gh-test-3')

    const { container } = render(<MarkdownTextContent isRunning={false} text="merga PR #46" />)

    await waitFor(() => expect(vi.mocked(requestGatewayForProfile)).toHaveBeenCalled())

    expect(screen.queryByTitle('KaptenKatthatt/newsAgg#46')).toBeNull()
    expect(container.querySelector('a')).toBeNull()
    expect(container.textContent).toContain('#46')
  })
})