import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { pl } from '@/i18n/pl'
import { applyVoiceEngineFromConfig } from '@/store/voice-prefs'

import { JarvisHomeHero } from './home-hero'
import { initialJarvisUiState } from './projector'
import { $jarvisUi, publishJarvisVoiceState } from './store'

/** The jarvis stylesheets sit next to these tests; vitest module URLs are not file URLs. */
function readJarvisCss(file: string): string {
  for (const from of ['src/app/jarvis', 'apps/desktop/src/app/jarvis']) {
    try {
      return readFileSync(resolve(process.cwd(), from, file), 'utf8')
    } catch {
      // Try the next root the runner may have been started from.
    }
  }

  throw new Error(`could not read ${file}`)
}

const insert = vi.hoisted(() => vi.fn())

const pulseApi = vi.hoisted(() => ({
  getPulse: vi.fn(async () => ({ generated_at: 0, matters: [] as unknown[] })),
  sendPulseFeedback: vi.fn(async () => ({ ok: true }))
}))

vi.mock('../chat/composer/focus', () => ({ requestComposerInsert: insert }))
vi.mock('@/api/pulse', () => pulseApi)

const FAILING_JOB = { id: 'failing_job:j1', kind: 'failing_job', params: { name: 'Poranny raport' }, score: 0.9 }

function renderHero(props: Partial<React.ComponentProps<typeof JarvisHomeHero>> = {}) {
  const onStartListening = vi.fn()

  const view = render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <I18nProvider configClient={null} initialLocale="pl">
        <JarvisHomeHero connected listening={false} onStartListening={onStartListening} {...props} />
      </I18nProvider>
    </QueryClientProvider>
  )

  return { onStartListening, ...view }
}

afterEach(() => {
  cleanup()
  insert.mockReset()
  pulseApi.getPulse.mockClear()
  pulseApi.sendPulseFeedback.mockClear()
  vi.useRealTimers()
  window.localStorage.clear()
  $jarvisUi.set(initialJarvisUiState())
})

describe('Agent CzesiekHomeHero', () => {
  it('greets the profile by name, in words that fit the time of day', () => {
    vi.useFakeTimers({ now: new Date(2026, 8, 25, 21, 30), toFake: ['Date'] })
    renderHero({ profileDisplayName: 'Chris' })

    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('Chris, dobry wieczór. Co dziś ogarniamy?')
  })

  it('wears no status or voice-engine chip under the greeting', () => {
    applyVoiceEngineFromConfig({
      voice: { engine: 'realtime', realtime: { gemini: { model: 'gemini-3.8-live' }, provider: 'gemini' } }
    })
    renderHero()

    // Both chips left the screen: the orb and the conversation's own status
    // strip already say this, and a second copy crowded the actions.
    expect(screen.queryByTestId('jarvis-home-status')).toBeNull()
    expect(screen.queryByTestId('jarvis-home-voice-engine')).toBeNull()
    expect(screen.queryByText(pl.jarvisShell.home.orbStatus.idle)).toBeNull()
    expect(screen.queryByText(new RegExp(pl.jarvisShell.home.voiceEngine.gemini))).toBeNull()
    expect(screen.queryByText(/gemini-3\.8-live/)).toBeNull()

    applyVoiceEngineFromConfig({ voice: { engine: 'classic' } })
  })

  it('keeps the orb reacting to the live states now that the status line is gone', () => {
    renderHero()

    for (const [phase, voice] of [
      ['planning', 'idle'],
      ['approval', 'idle'],
      ['running', 'speaking']
    ] as const) {
      act(() => $jarvisUi.set({ ...initialJarvisUiState(), task: { id: 't1', phase }, voice }))

      const core = screen.getByTestId('jarvis-core')

      expect(core.getAttribute('data-task')).toBe(phase)
      expect(core.getAttribute('data-voice')).toBe(voice)
    }
  })

  it('follows the Live voice state while a conversation is open, instead of freezing on listening', () => {
    renderHero({ listening: true })

    // A conversation just opened: listening until the session reports itself.
    expect(screen.getByTestId('jarvis-core').getAttribute('data-voice')).toBe('listening')

    act(() => publishJarvisVoiceState('speaking'))

    expect(screen.getByTestId('jarvis-core').getAttribute('data-voice')).toBe('speaking')

    act(() => publishJarvisVoiceState('idle'))

    expect(screen.getByTestId('jarvis-core').getAttribute('data-voice')).toBe('listening')
  })

  it('an action chip starts the request in the composer instead of sending it', () => {
    renderHero()

    const actions = screen.getByRole('group', { name: pl.jarvisShell.home.actionsLabel })

    fireEvent.click(within(actions).getByRole('button', { name: pl.jarvisShell.home.actions.plan.label }))

    expect(insert).toHaveBeenCalledWith(pl.jarvisShell.home.actions.plan.prompt, { mode: 'prefix', target: 'main' })
  })

  it('puts a pulse suggestion first; taking it fills the composer and dismissing it tells the backend', async () => {
    pulseApi.getPulse.mockResolvedValueOnce({ generated_at: 0, matters: [FAILING_JOB] })
    renderHero()

    const nav = screen.getByRole('group', { name: pl.jarvisShell.home.shortcutsLabel })
    const title = pl.jarvisShell.pulse.kinds.failing_job.title(FAILING_JOB.params)

    await waitFor(() => expect(within(nav).getByText(title)).toBeTruthy())
    // The pulse takes a slot: the list still holds at most three entries.
    expect(nav.querySelectorAll(':scope > button, :scope > [data-pulse-kind]').length).toBeLessThanOrEqual(3)

    fireEvent.click(within(nav).getByText(title))
    expect(insert.mock.calls[0][0]).toBe(pl.jarvisShell.pulse.kinds.failing_job.prompt(FAILING_JOB.params))
    expect(pulseApi.sendPulseFeedback).toHaveBeenCalledWith(FAILING_JOB, 'accept')
    await waitFor(() => expect(within(nav).queryByText(title)).toBeNull())
  })

  it('dismissing a pulse suggestion does not touch the composer', async () => {
    pulseApi.getPulse.mockResolvedValueOnce({ generated_at: 0, matters: [FAILING_JOB] })
    renderHero()

    const dismiss = await screen.findByRole('button', { name: new RegExp(`^${pl.jarvisShell.pulse.dismiss}`) })

    fireEvent.click(dismiss)
    expect(pulseApi.sendPulseFeedback).toHaveBeenCalledWith(FAILING_JOB, 'decline')
    expect(insert).not.toHaveBeenCalled()
  })

  it('offers at most three shortcuts, and a shortcut only fills the composer', () => {
    renderHero()

    const shortcuts = within(screen.getByRole('group', { name: pl.jarvisShell.home.shortcutsLabel })).getAllByRole(
      'button'
    )

    expect(shortcuts.length).toBeGreaterThan(0)
    expect(shortcuts.length).toBeLessThanOrEqual(3)

    fireEvent.click(shortcuts[0])

    expect(insert).toHaveBeenCalledTimes(1)
    expect(insert.mock.calls[0][1]).toMatchObject({ target: 'main' })
  })

  it('cannot start a conversation while the engine is disconnected', () => {
    const { onStartListening } = renderHero({ connected: false })
    const talk = screen.getByRole('button', { name: pl.jarvisShell.home.talk })

    expect((talk as HTMLButtonElement).disabled).toBe(true)
    // No offline chip any more; the disabled control and the engine's own state
    // carry it.
    expect(screen.queryByText(pl.jarvisShell.home.offline)).toBeNull()

    fireEvent.click(talk)
    expect(onStartListening).not.toHaveBeenCalled()
  })

  it('offers one primary voice action above three quiet suggestions', () => {
    renderHero()

    const actions = screen.getByRole('group', { name: pl.jarvisShell.home.actionsLabel })
    const talk = screen.getByRole('button', { name: pl.jarvisShell.home.talk })

    // The suggestions never wrap: no wrapping utility, and the stylesheet says `nowrap`.
    expect(actions.className).not.toContain('flex-wrap')
    expect(readJarvisCss('glass.css')).toMatch(/\.jarvis-home__actions\s*\{[^}]*flex-wrap:\s*nowrap/)

    // The talk button is the screen's single primary and is not one of the suggestions.
    expect(actions.querySelectorAll('.jarvis-action')).toHaveLength(3)
    expect(actions.contains(talk)).toBe(false)
    expect(talk.className).toContain('jarvis-action--talk')
    expect(screen.getByTestId('jarvis-home-hero').querySelectorAll('.jarvis-action--talk')).toHaveLength(1)
  })

  it('never puts two microphone controls on this screen', () => {
    // At rest the hero's talk button is the only voice control.
    renderHero()

    expect(screen.getAllByRole('button', { name: pl.jarvisShell.home.talk })).toHaveLength(1)
    expect(screen.queryByRole('button', { name: pl.jarvisShell.home.stopTalking })).toBeNull()

    cleanup()

    // With the conversation live the button ends it instead: the voice dock the
    // dashboard shows at the same time owns the microphone, so this screen must
    // not show a microphone of its own.
    renderHero({ listening: true })

    expect(screen.queryByRole('button', { name: pl.jarvisShell.home.talk })).toBeNull()
    expect(screen.getByRole('button', { name: pl.jarvisShell.home.stopTalking })).toBeTruthy()
    expect(screen.getByTestId('jarvis-home-hero').querySelectorAll('.jarvis-action--talk')).toHaveLength(0)
  })

  it('draws no ring or particle orbit around the orb, and keeps the orb itself', () => {
    const { container } = renderHero()

    // The loose specks and the dashed outer frame are gone; the orb stays.
    expect(container.querySelector('.jarvis-home__orbit')).toBeNull()
    expect(container.querySelector('.jarvis-core__outer-ring')).toBeNull()
    expect(container.querySelector('.jarvis-core__glass')).not.toBeNull()
    expect(screen.getByTestId('jarvis-core').getAttribute('data-variant')).toBe('hero')
  })
})
