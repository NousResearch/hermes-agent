import { cleanup, render, screen } from '@testing-library/react'
import type { ComponentProps } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $toolDisclosureStates, setToolDisclosureOpen } from '@/store/tool-view'

import { toolEntryDisclosureId } from './fallback-model/targets'

vi.mock('@assistant-ui/react', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useAuiState: (select: (state: unknown) => unknown) =>
    select({ message: { id: 'msg-1', status: { type: 'complete' } }, thread: { isRunning: false } })
}))

const { ToolFallback } = await import('./fallback')

const ERROR_TEXT = 'MCP call failed: MCPError: create file error: repository file already exists [path: README.md]'
const WINDOWS_PATH_ERROR = 'Failed to write file: C:\\Users\\test-user\\Documents\\CODEX_PROGRESS.md is locked'
const WWW_PATH_ERROR = 'Failed to read /tmp/www.config and C:\\Users\\test-user\\Documents\\www.config'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop
let fetchLinkTitle: ReturnType<typeof vi.fn>

function renderErrorToolRow(result: unknown) {
  const props = {
    args: { repo: 'demo', path: 'README.md' },
    isError: true,
    result,
    toolCallId: 'call-err-1',
    toolName: 'mcp__forgejo__create_file'
  } as unknown as ComponentProps<typeof ToolFallback>

  // The error body only paints when the disclosure is open; part identity is
  // messageId + toolCallId, both fixed here.
  setToolDisclosureOpen(toolEntryDisclosureId('msg-1', props), true)
  render(<ToolFallback {...props} />)
}

beforeEach(() => {
  fetchLinkTitle = vi.fn().mockResolvedValue('AI Tools Directory - dealsbe.com')
  desktopWindow.hermesDesktop = {
    fetchLinkTitle,
    openExternal: vi.fn().mockResolvedValue(undefined)
  } as unknown as Window['hermesDesktop']
})

afterEach(() => {
  cleanup()
  $toolDisclosureStates.set({})
  desktopWindow.hermesDesktop = initialHermesDesktop
})

describe('tool error summary linkification', () => {
  // The bare-domain branch of the linkifier matched `README.md` as a host and
  // the pretty link then replaced the painted text with the fetched title of
  // https://readme.md, so the headline stopped matching the raw error string.
  it.each([
    ['filename token', ERROR_TEXT],
    ['Windows path', WINDOWS_PATH_ERROR],
    ['www.-prefixed local paths', WWW_PATH_ERROR]
  ])('renders a %s verbatim with no link and no title fetch', (_, error) => {
    renderErrorToolRow({ error })

    expect(screen.queryByRole('link')).toBeNull()
    expect(screen.getByText((_, element) => element?.textContent === error)).toBeTruthy()
    expect(fetchLinkTitle).not.toHaveBeenCalled()
  })

  it('keeps an explicit URL in the error clickable without rewriting the headline', () => {
    renderErrorToolRow({ error: 'MCP call failed: see https://example.com/docs/error-codes [path: README.md]' })

    const link = screen.getByRole('link')
    expect(link.getAttribute('href')).toBe('https://example.com/docs/error-codes')
    expect(link.textContent).toBe('https://example.com/docs/error-codes')
    // eslint-disable-next-line no-restricted-globals -- the headline text is split across link and text nodes
    expect(document.body.textContent).toContain('[path: README.md]')
    expect(fetchLinkTitle).not.toHaveBeenCalled()
  })
})
