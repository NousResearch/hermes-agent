import { cleanup, render, screen } from '@testing-library/react'
import type { ComponentProps } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $toolDisclosureStates, setToolDisclosureOpen } from '@/store/tool-view'

import { toolEntryDisclosureId } from './fallback-model/targets'

vi.mock('@assistant-ui/react', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useAuiState: (select: (state: unknown) => unknown) =>
    select({ message: { id: 'msg-1', status: { type: 'complete' } }, thread: { isRunning: false } })
}))

const { ToolFallback } = await import('./fallback')

const ERROR_TEXT = 'MCP call failed: MCPError: create file error: repository file already exists [path: README.md]'

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

afterEach(() => {
  cleanup()
  $toolDisclosureStates.set({})
})

describe('tool error summary linkification', () => {
  // The bare-domain branch of the linkifier matched `README.md` as a host and
  // the pretty link then replaced the painted text with the fetched title of
  // https://readme.md, so the headline stopped matching the raw error string.
  it('renders filename-shaped error tokens as plain text, not fabricated links', () => {
    renderErrorToolRow({ error: ERROR_TEXT })

    expect(screen.queryByRole('link')).toBeNull()
    expect(screen.getByText((_, element) => element?.textContent === ERROR_TEXT)).toBeTruthy()
  })

  it('keeps an explicit URL in the error clickable without rewriting the headline', () => {
    renderErrorToolRow({ error: 'MCP call failed: see https://example.com/docs/error-codes [path: README.md]' })

    const link = screen.getByRole('link')
    expect(link.getAttribute('href')).toBe('https://example.com/docs/error-codes')
    expect(link.textContent).toBe('https://example.com/docs/error-codes')
    // eslint-disable-next-line no-restricted-globals -- the headline text is split across link and text nodes
    expect(document.body.textContent).toContain('[path: README.md]')
  })
})
