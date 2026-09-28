import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { pl } from '@/i18n/pl'
import { applyVoiceEngineFromConfig } from '@/store/voice-prefs'

import { JarvisHomeHero } from './home-hero'
import { initialJarvisUiState } from './projector'
import { $jarvisUi, publishJarvisVoiceState } from './store'

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

  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <I18nProvider configClient={null} initialLocale="pl">
        <JarvisHomeHero connected listening={false} onStartListening={onStartListening} {...props} />
      </I18nProvider>
    </QueryClientProvider>
  )

  return { onStartListening }
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

  it('says which voice answers: Gemini Live and its model when Live runs on Gemini', () => {
    applyVoiceEngineFromConfig({
      voice: { engine: 'realtime', realtime: { gemini: { model: 'gemini-3.8-live' }, provider: 'gemini' } }
    })
    renderHero()

    const line = screen.getByTestId('jarvis-home-voice-engine').textContent ?? ''

    expect(line).toContain(pl.jarvisShell.home.voiceEngine.gemini)
    expect(line).toContain('gemini-3.8-live')

    applyVoiceEngineFromConfig({ voice: { engine: 'classic' } })
  })

  it('names the live planning, approval and speaking states under the orb', () => {
    renderHero()

    for (const [phase, voice, label] of [
      ['planning', 'idle', pl.jarvisShell.home.orbStatus.thinking],
      ['approval', 'idle', pl.jarvisShell.home.orbStatus.approval],
      ['running', 'speaking', pl.jarvisShell.home.orbStatus.speaking]
    ] as const) {
      act(() => $jarvisUi.set({ ...initialJarvisUiState(), task: { id: 't1', phase }, voice }))
      expect(screen.getByTestId('jarvis-home-status').textContent).toContain(label)
    }
  })

  it('follows the Live voice state while a conversation is open, instead of freezing on listening', () => {
    renderHero({ listening: true })

    // A conversation just opened: listening until the session reports itself.
    expect(screen.getByTestId('jarvis-core').getAttribute('data-voice')).toBe('listening')

    act(() => publishJarvisVoiceState('speaking'))

    expect(screen.getByTestId('jarvis-core').getAttribute('data-voice')).toBe('speaking')
    expect(screen.getByTestId('jarvis-home-status').textContent).toContain(pl.jarvisShell.home.orbStatus.speaking)

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
    expect(screen.getByTestId('jarvis-home-status').textContent).toContain(pl.jarvisShell.home.offline)

    fireEvent.click(talk)
    expect(onStartListening).not.toHaveBeenCalled()
  })
})
