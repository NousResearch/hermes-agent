import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { LocalAdvanced } from './local-slots'

vi.mock('../mcp/mcp-editor', () => ({
  McpJsonEditor: () => <div data-testid="mcp-json-editor" />,
  McpLogPane: () => <div data-slot="mcp-log-pane" />
}))

afterEach(cleanup)

describe('LocalAdvanced', () => {
  it('gives the fixed-height JSON editor region a flex layout', () => {
    render(
      <I18nProvider>
        <LocalAdvanced controller={{} as never} name="example" onRemove={() => undefined} />
      </I18nProvider>
    )

    const editor = screen.getByTestId('mcp-json-editor')

    expect(editor.parentElement?.className.split(' ')).toContain('flex')
  })
})
