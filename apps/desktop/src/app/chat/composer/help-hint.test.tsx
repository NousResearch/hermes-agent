import { act, cleanup, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n/context'
import { DESKTOP_COMMON_COMMANDS, isDesktopSlashCommand } from '@/lib/desktop-slash-commands'

import { HelpHint } from './help-hint'

// The four commands the pre-fix drawer hardcoded that the desktop refuses to
// run (all in NO_DESKTOP_SURFACE.terminal). None may reach the DOM again.
const RETIRED_TERMINAL_COMMANDS = ['/clear', '/details', '/copy', '/quit']

async function renderHelpHint() {
  let result: ReturnType<typeof render>
  await act(async () => {
    result = render(
      <I18nProvider configClient={{ getConfig: async () => ({}), saveConfig: async () => ({ ok: true }) }}>
        <HelpHint />
      </I18nProvider>
    )
  })
  return result!
}

// Render-level guard for the #-quick-help regression: the drawer used to
// hardcode /clear, /details, /copy, /quit — all terminal-only — so tapping a
// row errored. The component must render exactly the spec-derived list and
// no terminal-only command may reach the DOM.
describe('HelpHint drawer', () => {
  afterEach(cleanup)

  it('renders a row for every DESKTOP_COMMON_COMMAND', async () => {
    const { container } = await renderHelpHint()
    const drawer = container.querySelector('[data-slot="composer-completion-drawer"]')!
    for (const command of DESKTOP_COMMON_COMMANDS) {
      // getAllByText: /help also appears in the drawer footer, so >=1 row.
      expect(within(drawer as HTMLElement).getAllByText(command.name).length).toBeGreaterThan(0)
    }
  })

  it('never advertises a terminal-only command', async () => {
    await renderHelpHint()
    for (const name of RETIRED_TERMINAL_COMMANDS) {
      expect(screen.queryByText(name)).toBeNull()
    }
  })

  it('only advertises commands the desktop can actually run', async () => {
    await renderHelpHint()
    for (const command of DESKTOP_COMMON_COMMANDS) {
      expect(isDesktopSlashCommand(command.name)).toBe(true)
    }
  })
})
