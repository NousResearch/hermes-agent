import { useStore } from '@nanostores/react'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { requestComposerInsert, requestComposerSubmit } from '@/app/chat/composer/focus'
import {
  $browserTabs,
  type BrowserTabId,
  clearBrowserTabs,
  createBrowserTab,
  setBrowserEnabled,
  updateBrowserTab
} from '@/store/browser'
import { runGuestScript } from '@/store/browser-bridge'
import { $browserSelection, type SelectedElement, setBrowserSelection } from '@/store/browser-guest-state'

import { buildDesignPrompt, DesignEditorPanel } from './design-editor-panel'

vi.mock('@/store/browser-bridge', () => ({
  GUEST_HANDLE_HELPERS_SOURCE: '/* guest helpers */',
  runGuestScript: vi.fn(async () => ({ ok: true }))
}))

vi.mock('@/app/chat/composer/focus', () => ({
  requestComposerInsert: vi.fn(),
  requestComposerSubmit: vi.fn()
}))

function DesignHarness({ tabId }: { tabId: BrowserTabId }) {
  const tab = useStore($browserTabs).find(candidate => candidate.id === tabId)

  if (!tab) {
    return null
  }

  return <DesignEditorPanel tab={tab} />
}

function seedSelection(tabId: BrowserTabId, overrides: Partial<SelectedElement> = {}): SelectedElement {
  const sel: SelectedElement = {
    at: Date.now(),
    componentName: 'PrimaryButton',
    cssPath: 'div#app:nth-of-type(1)>button:nth-of-type(2)',
    htmlPreview: '<button class="cta">Buy</button>',
    ref: 'p-handle-1',
    stableRef: '@sabc123',
    styles: { backgroundColor: 'rgb(0, 0, 0)', color: 'rgb(255, 255, 255)', fontSize: '16px' },
    tag: 'button',
    url: 'http://localhost:5173/',
    ...overrides
  }

  setBrowserSelection(tabId, sel)

  return sel
}

describe('DesignEditorPanel', () => {
  beforeEach(() => {
    window.localStorage.clear()
    clearBrowserTabs()
    setBrowserEnabled(true)
    $browserSelection.set({})
    vi.clearAllMocks()
  })

  afterEach(() => {
    cleanup()
    clearBrowserTabs()
    setBrowserEnabled(false)
    $browserSelection.set({})
    vi.useRealTimers()
    window.localStorage.clear()
  })

  it('applies a debounced live preview via runGuestScript with control mode and a JSON payload', () => {
    const tab = createBrowserTab({ sessionId: 'session-1', url: 'http://localhost:5173/' })

    updateBrowserTab(tab.id, { controlMode: 'control' })
    seedSelection(tab.id)
    vi.useFakeTimers()
    render(<DesignHarness tabId={tab.id} />)

    fireEvent.change(screen.getByLabelText('Font size'), { target: { value: '40' } })

    act(() => {
      vi.advanceTimersByTime(150)
    })

    expect(vi.mocked(runGuestScript)).toHaveBeenCalledWith(
      tab.id,
      expect.stringContaining('{"fontSize":"40px"}'),
      'control'
    )
  })

  it('reverts a live preview through runGuestScript on demand', () => {
    const tab = createBrowserTab({ sessionId: 'session-1', url: 'http://localhost:5173/' })

    updateBrowserTab(tab.id, { controlMode: 'control' })
    seedSelection(tab.id)
    render(<DesignHarness tabId={tab.id} />)

    vi.mocked(runGuestScript).mockClear()
    fireEvent.click(screen.getByRole('button', { name: 'Revert' }))

    expect(vi.mocked(runGuestScript)).toHaveBeenCalledWith(
      tab.id,
      expect.stringContaining('delete window.__hermesCss'),
      'control'
    )
  })

  it('drafts a source-edit prompt into the main composer', () => {
    const tab = createBrowserTab({ sessionId: 'session-1', url: 'http://localhost:5173/' })

    updateBrowserTab(tab.id, { controlMode: 'control' })
    seedSelection(tab.id)
    render(<DesignHarness tabId={tab.id} />)

    fireEvent.click(screen.getByRole('button', { name: 'Draft' }))

    expect(vi.mocked(requestComposerInsert)).toHaveBeenCalledWith(
      expect.stringContaining('div#app:nth-of-type(1)>button:nth-of-type(2)'),
      { target: 'main' }
    )
  })

  it('disables auto-send on an untrusted origin but still allows drafting', () => {
    const tab = createBrowserTab({ sessionId: 'session-1', url: 'https://example.com/app' })

    updateBrowserTab(tab.id, { controlMode: 'control' })
    seedSelection(tab.id, { url: 'https://example.com/app' })
    render(<DesignHarness tabId={tab.id} />)

    const send = screen.getByRole('button', { name: 'Apply & send' }) as HTMLButtonElement

    expect(send.disabled).toBe(true)
    fireEvent.click(send)
    expect(vi.mocked(requestComposerSubmit)).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: 'Draft' }))
    expect(vi.mocked(requestComposerInsert)).toHaveBeenCalled()
  })

  it('auto-sends the prompt to the agent on a trusted localhost origin', () => {
    const tab = createBrowserTab({ sessionId: 'session-1', url: 'http://localhost:5173/' })

    updateBrowserTab(tab.id, { controlMode: 'control' })
    seedSelection(tab.id)
    render(<DesignHarness tabId={tab.id} />)

    const send = screen.getByRole('button', { name: 'Apply & send' }) as HTMLButtonElement

    expect(send.disabled).toBe(false)
    fireEvent.click(send)

    expect(vi.mocked(requestComposerSubmit)).toHaveBeenCalledWith(
      expect.stringContaining('http://localhost:5173/'),
      { target: 'main' }
    )
  })

  it('renders the selection read-only and runs no guest script when control is not granted', () => {
    const tab = createBrowserTab({ sessionId: 'session-1', url: 'http://localhost:5173/' })

    updateBrowserTab(tab.id, { controlMode: 'observe' })
    seedSelection(tab.id)
    render(<DesignHarness tabId={tab.id} />)

    expect(screen.getByText('Bind the agent with control to edit styles.')).toBeDefined()
    expect(screen.queryByLabelText('Font size')).toBeNull()
    expect(vi.mocked(runGuestScript)).not.toHaveBeenCalled()
  })

  it('buildDesignPrompt includes the css path and the css delta', () => {
    const tab = createBrowserTab({ sessionId: 'session-1', url: 'http://localhost:5173/' })
    const sel = seedSelection(tab.id)
    const prompt = buildDesignPrompt(sel, { color: { from: 'rgb(0, 0, 0)', to: '#ffffff' } }, tab)

    expect(prompt).toContain('div#app:nth-of-type(1)>button:nth-of-type(2)')
    expect(prompt).toContain('color: rgb(0, 0, 0) → #ffffff')
  })
})
