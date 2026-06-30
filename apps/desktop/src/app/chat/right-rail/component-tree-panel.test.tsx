import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { clearBrowserTabs, createBrowserTab } from '@/store/browser'
import { highlightGuestElement, inspectGuestElement, runGuestScript } from '@/store/browser-bridge'

import { ComponentTreePanel } from './component-tree-panel'

// The guest bridge is mocked end-to-end: the real `runGuestScript` runs an IIFE
// inside an Electron <webview>, which jsdom has no equivalent for. The fixture is
// a DOM tree (react:false) whose single child carries a React component name.
vi.mock('@/store/browser-bridge', () => ({
  GUEST_HANDLE_HELPERS_SOURCE: '',
  highlightGuestElement: vi.fn(),
  inspectGuestElement: vi.fn(),
  runGuestScript: vi.fn()
}))

const FIXTURE = {
  react: false,
  root: {
    children: [{ children: [], handle: '0.0', id: '0.0', name: 'Button', tag: 'div' }],
    handle: '0',
    id: '0',
    name: 'body',
    tag: 'body'
  },
  truncated: false
}

describe('ComponentTreePanel', () => {
  beforeEach(() => {
    window.localStorage.clear()
    clearBrowserTabs()
    vi.mocked(runGuestScript).mockReset()
    vi.mocked(runGuestScript).mockResolvedValue(FIXTURE)
    vi.mocked(highlightGuestElement).mockReset()
    vi.mocked(highlightGuestElement).mockResolvedValue(undefined)
    vi.mocked(inspectGuestElement).mockReset()
    vi.mocked(inspectGuestElement).mockResolvedValue({ ref: '0.0', tag: 'div', text: 'Click me' })
  })

  afterEach(() => {
    cleanup()
    clearBrowserTabs()
    window.localStorage.clear()
  })

  it('builds the tree on mount and dims the tag when the React name differs', async () => {
    const tab = createBrowserTab({ sessionId: 'session-1', url: 'https://example.com/app' })

    render(<ComponentTreePanel tab={tab} />)

    expect(await screen.findByText('Button')).toBeDefined()
    expect(screen.getByText('div')).toBeDefined()
    expect(runGuestScript).toHaveBeenCalledWith(tab.id, expect.any(String), 'observe')
  })

  it('highlights and inspects the selected node by its handle', async () => {
    const tab = createBrowserTab({ sessionId: 'session-1', url: 'https://example.com/app' })

    render(<ComponentTreePanel tab={tab} />)
    fireEvent.click(await screen.findByText('Button'))

    await waitFor(() => expect(highlightGuestElement).toHaveBeenCalledWith(tab.id, '0.0'))
    expect(inspectGuestElement).toHaveBeenCalledWith(tab.id, '0.0')
  })

  it('notes when no React is detected', async () => {
    const tab = createBrowserTab({ sessionId: 'session-1', url: 'https://example.com/app' })

    render(<ComponentTreePanel tab={tab} />)

    expect(await screen.findByText(/No React detected/)).toBeDefined()
  })

  it('shows an error with retry when the tree script rejects, then recovers', async () => {
    vi.mocked(runGuestScript).mockReset()
    vi.mocked(runGuestScript).mockRejectedValue(new Error('Browser tab is not bound for agent access'))
    const tab = createBrowserTab({ sessionId: 'session-1', url: 'about:blank' })

    render(<ComponentTreePanel tab={tab} />)

    const retry = await screen.findByRole('button', { name: 'Retry' })

    expect(screen.getByText(/not bound for agent access/)).toBeDefined()

    vi.mocked(runGuestScript).mockReset()
    vi.mocked(runGuestScript).mockResolvedValue(FIXTURE)
    fireEvent.click(retry)

    expect(await screen.findByText('Button')).toBeDefined()
  })
})
