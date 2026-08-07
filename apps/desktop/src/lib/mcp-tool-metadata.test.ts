import { describe, expect, it } from 'vitest'

import { formatMcpToolDetails, type McpToolHintCopy } from './mcp-tool-metadata'

const copy: McpToolHintCopy = {
  additive: 'Server reports additive updates only',
  closedWorld: 'Server reports closed-world interaction',
  destructive: 'Server reports this may be destructive',
  idempotent: 'Server reports repeat calls have no additional effect',
  mayModify: 'Server reports this may modify data',
  openWorld: 'Server reports external-system access',
  readOnly: 'Server reports read-only',
  repeatEffects: 'Server reports repeat calls may have additional effects'
}

describe('formatMcpToolDetails', () => {
  it('supports older backend payloads without optional metadata', () => {
    expect(formatMcpToolDetails({ description: 'Search pages', name: 'search_pages' }, copy)).toBe('Search pages')
  })

  it('uses top-level title before the legacy annotation title', () => {
    expect(
      formatMcpToolDetails(
        {
          annotations: { title: 'Legacy title' },
          description: 'Moves a page to trash',
          name: 'delete_page',
          title: 'Delete a page'
        },
        copy
      )
    ).toBe('Delete a page\nMoves a page to trash')
  })

  it('describes true annotations as server reports rather than verified facts', () => {
    expect(
      formatMcpToolDetails(
        {
          annotations: {
            destructiveHint: true,
            idempotentHint: true,
            openWorldHint: true,
            readOnlyHint: false
          },
          description: '',
          name: 'delete_page'
        },
        copy
      )
    ).toBe([copy.mayModify, copy.destructive, copy.idempotent, copy.openWorld].join('\n'))
  })

  it('preserves and explains explicit false annotations', () => {
    expect(
      formatMcpToolDetails(
        {
          annotations: {
            destructiveHint: false,
            idempotentHint: false,
            openWorldHint: false,
            readOnlyHint: false
          },
          description: '',
          name: 'create_page'
        },
        copy
      )
    ).toBe([copy.mayModify, copy.additive, copy.repeatEffects, copy.closedWorld].join('\n'))
  })

  it('does not show write-only hints for a read-only tool', () => {
    expect(
      formatMcpToolDetails(
        {
          annotations: { destructiveHint: true, idempotentHint: false, readOnlyHint: true },
          description: '',
          name: 'read_page'
        },
        copy
      )
    ).toBe(copy.readOnly)
  })
})
