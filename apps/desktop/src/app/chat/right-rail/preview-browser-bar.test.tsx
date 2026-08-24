import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { normalizePreviewAddress, PreviewBrowserBar } from './preview-browser-bar'

const baseProps = {
  canGoBack: false,
  canGoForward: false,
  consoleOpen: false,
  devToolsOpen: false,
  loading: false,
  onBack: vi.fn(),
  onForward: vi.fn(),
  onNavigate: vi.fn(),
  onOpenExternal: vi.fn(),
  onReload: vi.fn(),
  onToggleConsole: vi.fn(),
  onToggleDevTools: vi.fn(),
  url: 'https://example.com'
}

function address(rendered: ReturnType<typeof render>) {
  return rendered.getByRole('textbox', { name: 'Address' }) as HTMLInputElement
}

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }

function installBridge(writeClipboard: ReturnType<typeof vi.fn>) {
  desktopWindow.hermesDesktop = { writeClipboard } as unknown as Window['hermesDesktop']
}

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  delete desktopWindow.hermesDesktop
})

describe('normalizePreviewAddress', () => {
  it('passes through http and https URLs', () => {
    expect(normalizePreviewAddress('https://example.com/path')).toBe('https://example.com/path')
    expect(normalizePreviewAddress('http://localhost:3000/')).toBe('http://localhost:3000/')
  })

  it('gives a bare host https and a loopback host http', () => {
    expect(normalizePreviewAddress('example.com')).toBe('https://example.com')
    expect(normalizePreviewAddress('localhost:5173')).toBe('http://localhost:5173')
    expect(normalizePreviewAddress('127.0.0.1:8080/app')).toBe('http://127.0.0.1:8080/app')
  })

  it('rejects empty input', () => {
    expect(normalizePreviewAddress('')).toBeNull()
    expect(normalizePreviewAddress('   ')).toBeNull()
  })

  // This is a browser: the schemes a browser navigates to all work, and the
  // user already owns the machine, so `file://` is theirs to open.
  it('passes through the other schemes a browser can navigate', () => {
    expect(normalizePreviewAddress('about:blank')).toBe('about:blank')
    expect(normalizePreviewAddress('file:///etc/hosts')).toBe('file:///etc/hosts')
    expect(normalizePreviewAddress('view-source:https://example.com')).toBe('view-source:https://example.com')
  })

  // These EXECUTE in the guest partition instead of navigating it — an address
  // bar is not an eval box, and no shipping browser honors them here either.
  it('rejects script-bearing schemes', () => {
    expect(normalizePreviewAddress('javascript:alert(1)')).toBeNull()
    expect(normalizePreviewAddress('data:text/html,evil')).toBeNull()
    expect(normalizePreviewAddress('JavaScript:alert(1)')).toBeNull()
  })

  it('rejects strings that are not addresses at all', () => {
    expect(normalizePreviewAddress('://broken')).toBeNull()
    expect(normalizePreviewAddress('http://')).toBeNull()
  })

  it('trims whitespace before validating', () => {
    expect(normalizePreviewAddress('  https://example.com  ')).toBe('https://example.com')
  })
})

describe('PreviewBrowserBar', () => {
  it('renders the navigation controls and the page toggles', () => {
    const rendered = render(<PreviewBrowserBar {...baseProps} />)

    expect(rendered.getByRole('button', { name: 'Back' })).toBeTruthy()
    expect(rendered.getByRole('button', { name: 'Forward' })).toBeTruthy()
    expect(rendered.getByRole('button', { name: 'Reload page' })).toBeTruthy()
    expect(rendered.getByRole('button', { name: 'Copy URL' })).toBeTruthy()
    expect(rendered.getByRole('button', { name: 'Open in browser' })).toBeTruthy()
    expect(rendered.getByRole('button', { name: 'Show preview console' })).toBeTruthy()
    expect(rendered.getByRole('button', { name: 'Open preview DevTools' })).toBeTruthy()
    expect(address(rendered)).toBeTruthy()
  })

  it('disables back and forward when there is no history', () => {
    const rendered = render(<PreviewBrowserBar {...baseProps} />)

    expect((rendered.getByRole('button', { name: 'Back' }) as HTMLButtonElement).disabled).toBe(true)
    expect((rendered.getByRole('button', { name: 'Forward' }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('enables back and forward when there is history', () => {
    const rendered = render(<PreviewBrowserBar {...baseProps} canGoBack canGoForward />)

    expect((rendered.getByRole('button', { name: 'Back' }) as HTMLButtonElement).disabled).toBe(false)
    expect((rendered.getByRole('button', { name: 'Forward' }) as HTMLButtonElement).disabled).toBe(false)
  })

  it.each([
    ['Back', 'onBack'],
    ['Forward', 'onForward'],
    ['Reload page', 'onReload'],
    ['Open in browser', 'onOpenExternal']
  ] as const)('fires %s', (label, handler) => {
    const spy = vi.fn()
    const rendered = render(<PreviewBrowserBar {...baseProps} canGoBack canGoForward {...{ [handler]: spy }} />)

    fireEvent.click(rendered.getByRole('button', { name: label }))

    expect(spy).toHaveBeenCalledOnce()
  })

  it('toggles the console and DevTools, and labels them by current state', () => {
    const onToggleConsole = vi.fn()
    const onToggleDevTools = vi.fn()

    const rendered = render(
      <PreviewBrowserBar {...baseProps} onToggleConsole={onToggleConsole} onToggleDevTools={onToggleDevTools} />
    )

    fireEvent.click(rendered.getByRole('button', { name: 'Show preview console' }))
    fireEvent.click(rendered.getByRole('button', { name: 'Open preview DevTools' }))

    expect(onToggleConsole).toHaveBeenCalledOnce()
    expect(onToggleDevTools).toHaveBeenCalledOnce()

    rendered.rerender(<PreviewBrowserBar {...baseProps} consoleOpen devToolsOpen />)

    expect(rendered.getByRole('button', { name: 'Hide preview console' }).getAttribute('aria-pressed')).toBe('true')
    expect(rendered.getByRole('button', { name: 'Hide preview DevTools' }).getAttribute('aria-pressed')).toBe('true')
  })

  it('shows the live url while idle and follows it as the page navigates', () => {
    const rendered = render(<PreviewBrowserBar {...baseProps} />)

    expect(address(rendered).value).toBe('https://example.com')

    rendered.rerender(<PreviewBrowserBar {...baseProps} url="https://example.com/next" />)

    expect(address(rendered).value).toBe('https://example.com/next')
  })

  // The page can navigate itself (redirect, pushState) while the user is
  // mid-address; their keystrokes must survive it.
  it('does not clobber the draft while the user is typing', () => {
    const rendered = render(<PreviewBrowserBar {...baseProps} />)
    const input = address(rendered)

    fireEvent.focus(input)
    fireEvent.change(input, { target: { value: 'example.org/docs' } })
    rendered.rerender(<PreviewBrowserBar {...baseProps} url="https://example.com/redirected" />)

    expect(address(rendered).value).toBe('example.org/docs')
  })

  // Committing used to drop the field back to `url` — the page you were
  // LEAVING — so the address you typed flashed away and came back once the
  // load landed. What you asked for stays put until the page actually moves.
  it('normalizes and navigates on Enter, and holds the address it asked for', () => {
    const onNavigate = vi.fn()
    const rendered = render(<PreviewBrowserBar {...baseProps} onNavigate={onNavigate} />)
    const input = address(rendered)

    fireEvent.focus(input)
    fireEvent.change(input, { target: { value: 'localhost:5173' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    expect(onNavigate).toHaveBeenCalledWith('http://localhost:5173')
    expect(address(rendered).value).toBe('http://localhost:5173')
  })

  // A redirect means the address you asked for is no longer the truth.
  it('releases the asked-for address once the page lands somewhere', () => {
    const rendered = render(<PreviewBrowserBar {...baseProps} />)
    const input = address(rendered)

    fireEvent.focus(input)
    fireEvent.change(input, { target: { value: 'example.org' } })
    fireEvent.keyDown(input, { key: 'Enter' })
    rendered.rerender(<PreviewBrowserBar {...baseProps} url="https://www.example.org/home" />)

    expect(address(rendered).value).toBe('https://www.example.org/home')
  })

  // Re-focusing mid-flight must offer what is on screen to edit, not resurrect
  // the address of the page being left behind.
  it('hands the in-flight address to the field on re-focus', () => {
    const rendered = render(<PreviewBrowserBar {...baseProps} />)
    const input = address(rendered)

    fireEvent.focus(input)
    fireEvent.change(input, { target: { value: 'example.org' } })
    fireEvent.keyDown(input, { key: 'Enter' })
    fireEvent.blur(input)
    fireEvent.focus(input)

    expect(address(rendered).value).toBe('https://example.org')
  })

  it('refuses to navigate to a script-bearing scheme and marks the field invalid', () => {
    const onNavigate = vi.fn()
    const rendered = render(<PreviewBrowserBar {...baseProps} onNavigate={onNavigate} />)
    const input = address(rendered)

    fireEvent.focus(input)
    fireEvent.change(input, { target: { value: 'javascript:alert(1)' } })

    expect(address(rendered).getAttribute('aria-invalid')).toBe('true')

    fireEvent.keyDown(input, { key: 'Enter' })

    expect(onNavigate).not.toHaveBeenCalled()
  })

  it('navigates to about:blank like any other address', () => {
    const onNavigate = vi.fn()
    const rendered = render(<PreviewBrowserBar {...baseProps} onNavigate={onNavigate} />)
    const input = address(rendered)

    fireEvent.focus(input)
    fireEvent.change(input, { target: { value: 'about:blank' } })

    expect(address(rendered).getAttribute('aria-invalid')).toBeNull()

    fireEvent.keyDown(input, { key: 'Enter' })

    expect(onNavigate).toHaveBeenCalledWith('about:blank')
  })

  it('leaves an empty field unflagged', () => {
    const rendered = render(<PreviewBrowserBar {...baseProps} />)
    const input = address(rendered)

    fireEvent.focus(input)
    fireEvent.change(input, { target: { value: '' } })

    expect(address(rendered).getAttribute('aria-invalid')).toBeNull()
  })

  it('reverts the draft on Escape', () => {
    const onNavigate = vi.fn()
    const rendered = render(<PreviewBrowserBar {...baseProps} onNavigate={onNavigate} />)
    const input = address(rendered)

    fireEvent.focus(input)
    fireEvent.change(input, { target: { value: 'example.org' } })
    fireEvent.keyDown(input, { key: 'Escape' })

    expect(onNavigate).not.toHaveBeenCalled()
    expect(address(rendered).value).toBe('https://example.com')
  })

  // Progress belongs beside the address it describes; the reload glyph sits in
  // a row of four and reads as chrome rather than as this page's state.
  it('shows progress inside the address field only while loading', () => {
    const { container, rerender } = render(<PreviewBrowserBar {...baseProps} loading />)
    const field = screen.getByRole('textbox', { name: 'Address' }).parentElement

    expect(field?.querySelector('.codicon-loading')).toBeTruthy()

    rerender(<PreviewBrowserBar {...baseProps} />)

    expect(container.querySelector('.codicon-loading')).toBeNull()
  })

  it('spins the reload glyph only while loading', () => {
    const { container, rerender } = render(<PreviewBrowserBar {...baseProps} loading />)

    expect(container.querySelector('.codicon-refresh')?.className).toContain('codicon-modifier-spin')

    rerender(<PreviewBrowserBar {...baseProps} />)

    expect(container.querySelector('.codicon-refresh')?.className).not.toContain('codicon-modifier-spin')
  })

  it('copies the live address from the copy-URL button', async () => {
    const writeClipboard = vi.fn().mockResolvedValue(undefined)

    installBridge(writeClipboard)
    render(<PreviewBrowserBar {...baseProps} />)

    fireEvent.click(screen.getByRole('button', { name: 'Copy URL' }))

    await waitFor(() => expect(writeClipboard).toHaveBeenCalledWith('https://example.com'))
  })

  it('copies the new address after the page navigates', async () => {
    const writeClipboard = vi.fn().mockResolvedValue(undefined)

    installBridge(writeClipboard)
    const rendered = render(<PreviewBrowserBar {...baseProps} />)

    rendered.rerender(<PreviewBrowserBar {...baseProps} url="https://example.com/next" />)
    fireEvent.click(screen.getByRole('button', { name: 'Copy URL' }))

    await waitFor(() => expect(writeClipboard).toHaveBeenCalledWith('https://example.com/next'))
  })

  it('renders the copy control inside the address field wrapper, not as a bar glyph', () => {
    render(<PreviewBrowserBar {...baseProps} />)

    const copyButton = screen.getByRole('button', { name: 'Copy URL' })
    const address = screen.getByRole('textbox', { name: 'Address' })

    // Same positioned wrapper as the field = visually inside it, like the
    // code-block copy icon (inline appearance, overlay on the field's edge).
    expect(copyButton.parentElement?.contains(address)).toBe(true)
    expect(copyButton.className).toContain('absolute')
  })
})
