import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'

import { JarvisDashboard } from './dashboard'
import { initialJarvisUiState } from './projector'
import type { JarvisUiState } from './types'

// The Pulpit prints the verified answer — `$jarvisUi.result` — between the
// voice controls and the conversation below it. The gateway hands that field
// the WHOLE answer (`message.complete` → `task.verified`, see
// app/session/hooks/use-message-stream/gateway-event/jarvis.ts), so while a
// Live reply lands the surface carries markdown, not a headline. Printed as a
// bare `<h1 class="text-xl …">` it dropped the markdown SOURCE at display size
// over the wallpaper: literal `**bold**`, `|---|` tables, no panel. These lock
// the pane and the rich text down so that regression goes red.

const VERIFIED_ANSWER = [
  'Sprawdziłem historię — o pamięci gadałem Ci tylko przelotnie, więc odpowiem wprost:',
  '',
  '**Kod i development** — delegowanie roboty do Claude Code / Codex',
  '**Dokumenty i biuro** — Word, Excel, PowerPoint, PDF',
  '',
  '| Warstwa | Gdzie siedzi | Stan u Ciebie |',
  '|---|---|---|',
  '| Pamięć trwała | `hermes-home/memories/` | **pusta** — zero zapisanych faktów |',
  '',
  '- Skille (procedury) siedzą w `hermes-home/skills/`',
  '- Historia sesji w `state.db`',
  '',
  '```ts',
  'const answer = 42',
  '```'
].join('\n')

function fixtureState(result?: string): JarvisUiState {
  return {
    ...initialJarvisUiState(),
    result,
    sessionId: 's1',
    task: { id: 't1', phase: 'verified' },
    activity: []
  }
}

function renderDashboard(ui: React.ReactElement) {
  return render(
    <MemoryRouter>
      <I18nProvider configClient={null} initialLocale="pl">
        {ui}
      </I18nProvider>
    </MemoryRouter>
  )
}

function renderResult(result?: string) {
  const view = renderDashboard(
    <JarvisDashboard connected state={fixtureState(result)}>
      <div data-testid="real-chat">Real transcript and composer</div>
    </JarvisDashboard>
  )

  const pane = view.container.querySelector('[data-jarvis-result-pane]')

  return { ...view, pane: pane as HTMLElement | null }
}

afterEach(() => {
  cleanup()
})

describe('desktop result pane', () => {
  it('renders the verified answer as rich text, never as the markdown source', async () => {
    const { container, pane } = renderResult(VERIFIED_ANSWER)

    expect(pane).toBeTruthy()
    expect(container.querySelector('h1')).toBeNull()

    await waitFor(() => {
      expect(pane?.querySelector('table')).toBeTruthy()
    })

    const text = pane?.textContent ?? ''

    expect(text).not.toContain('**')
    expect(text).not.toContain('|---|')
    expect(text).not.toContain('```ts')
    expect(pane?.querySelector('strong, [data-streamdown="strong"], .font-semibold')).toBeTruthy()
    expect(pane?.querySelector('th')?.textContent).toContain('Warstwa')
    expect(pane?.querySelectorAll('li').length).toBeGreaterThanOrEqual(2)
    expect(pane?.querySelector('[data-slot="code-card"]')).toBeTruthy()
    // The answer itself survives the formatting.
    expect(text).toContain('Pamięć trwała')
    expect(text).toContain('const answer = 42')
    expect(text).toContain('Kod i development')
  })

  it('keeps the verified result as the screen headline', async () => {
    const { pane } = renderResult('Notatka została utworzona.')

    expect(pane?.getAttribute('role')).toBe('heading')
    expect(pane?.getAttribute('aria-level')).toBe('1')
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: 'Notatka została utworzona.' })).toBeTruthy()
    })
    expect(pane?.textContent).toContain('Notatka została utworzona.')
  })

  it('never re-grows into display type over the wallpaper', () => {
    const { pane } = renderResult(VERIFIED_ANSWER)

    // `text-xl` was the reported wall of text. The pane owns its own scale in
    // styles.css (`[data-jarvis-result-pane]`, locked by
    // result-pane-css.test.ts); a display-size utility here regresses it.
    expect(pane?.className ?? '').not.toContain('text-xl')
    expect(pane?.className ?? '').not.toContain('text-lg')
    expect(pane?.textContent?.length ?? 0).toBeGreaterThan(100)
  })

  it('leaves the surface to the orb when no result has been verified yet', () => {
    const { container, pane } = renderResult(undefined)

    expect(pane).toBeNull()
    expect(container.textContent ?? '').not.toContain('**')
  })

  it('keeps the transcript below untouched', () => {
    const { pane } = renderResult(VERIFIED_ANSWER)

    expect(screen.getByTestId('real-chat')).toBeTruthy()
    expect(pane?.getAttribute('data-jarvis-result-pane')).toBe('')
  })
})
