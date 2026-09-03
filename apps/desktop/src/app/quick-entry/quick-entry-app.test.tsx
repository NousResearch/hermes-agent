import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { PromptCoachAnalysis } from '@/lib/prompt-coach'
import { DESKTOP_PET_AGENT_PROFILE_KEY } from '@/store/pet-agent'
import type { QuickEntryPromptCoachResult, QuickEntryStatePush, QuickEntrySubmitPayload } from '@/store/quick-entry'
import type { QuickEntryShownPayload } from '@/store/quick-entry'

import { hermesJokeOfTheDay, QuickEntryApp, quickEntryGroupAccent, quickEntryModeFromSearch } from './quick-entry-app'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop
const submit = vi.fn<(payload: QuickEntrySubmitPayload) => void>()
let pushState: ((payload: QuickEntryStatePush) => void) | undefined
let showQuickEntry: ((payload: QuickEntryShownPayload) => void) | undefined

beforeEach(() => {
  window.localStorage.removeItem(DESKTOP_PET_AGENT_PROFILE_KEY)
  submit.mockClear()
  pushState = undefined
  showQuickEntry = undefined
  desktopWindow.hermesDesktop = {
    quickEntry: {
      dismiss: vi.fn(),
      onLaunchResult: vi.fn(() => () => undefined),
      onShown: vi.fn(callback => {
        showQuickEntry = callback

        return () => undefined
      }),
      onState: vi.fn(callback => {
        pushState = callback

        return () => undefined
      }),
      reportLaunchResult: vi.fn(),
      submit
    }
  } as unknown as Window['hermesDesktop']
})

afterEach(() => {
  if (initialHermesDesktop) {
    desktopWindow.hermesDesktop = initialHermesDesktop
  } else {
    delete desktopWindow.hermesDesktop
  }
})

describe('QuickEntryApp pointer agent launcher', () => {
  it('guards an immediate Enter before the typing debounce can send', () => {
    render(<QuickEntryApp />)

    act(() => {
      pushState?.({ agents: [], connected: true, groups: [], sessions: [] })
      showQuickEntry?.({ mode: 'composer' })
    })

    const input = screen.getByRole('textbox', { name: 'Quick Entry' })
    fireEvent.change(input, { target: { value: 'givme that' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    expect(screen.getByLabelText('Prompt Coach preview')).toBeTruthy()
    expect(submit).not.toHaveBeenCalled()
  })

  it('offers the same local Prompt Coach preview without sending automatically', async () => {
    vi.useFakeTimers()

    try {
      render(<QuickEntryApp />)

      act(() => {
        pushState?.({ agents: [], connected: true, groups: [], sessions: [] })
        showQuickEntry?.({ mode: 'composer' })
      })

      const input = screen.getByRole('textbox', { name: 'Quick Entry' })
      fireEvent.change(input, { target: { value: 'fix it and make it better' } })

      await act(async () => vi.advanceTimersByTimeAsync(600))

      fireEvent.click(screen.getByRole('button', { name: 'Improve Quick Entry prompt' }))
      expect(screen.getByLabelText('Prompt Coach preview')).toBeTruthy()
      expect(submit).not.toHaveBeenCalled()

      fireEvent.click(screen.getByRole('button', { name: 'Replace' }))

      expect((input as HTMLTextAreaElement).value).toContain('Goal:\nfix it and make it better')
      expect(submit).not.toHaveBeenCalled()
    } finally {
      vi.useRealTimers()
    }
  })

  it('relays an ambiguous draft for AI coaching and applies only the matching unsent result', () => {
    const requestPromptCoach = vi.fn()
    let receivePromptCoach: ((result: QuickEntryPromptCoachResult) => void) | undefined

    desktopWindow.hermesDesktop!.quickEntry.requestPromptCoach = requestPromptCoach

    desktopWindow.hermesDesktop!.quickEntry.onPromptCoachResult = callback => {
      receivePromptCoach = callback

      return () => undefined
    }

    render(<QuickEntryApp />)

    act(() => {
      pushState?.({ agents: [], connected: true, groups: [], sessions: [] })
      showQuickEntry?.({ mode: 'composer' })
    })

    const input = screen.getByRole('textbox', { name: 'Quick Entry' })
    fireEvent.change(input, { target: { value: 'hwo to clean it' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    const request = requestPromptCoach.mock.calls[0]?.[0]

    const analysis: PromptCoachAnalysis = {
      generatedBy: 'ai',
      hasPotentialSecret: false,
      missing: ['target', 'constraints', 'success'],
      reason: 'Missing target, constraints and success criteria',
      score: 25,
      suggestedPrompt:
        'Request (kept exactly as written):\nhwo to clean it\n\nTarget:\n[What exactly does “it” refer to?]'
    }

    expect(request).toMatchObject({ target: 'current', text: 'hwo to clean it' })
    expect(submit).not.toHaveBeenCalled()

    act(() => receivePromptCoach?.({ analysis, requestId: request.requestId, text: request.text }))

    expect(screen.getByText('Powered by the active Hermes AI model · wording preserved')).toBeTruthy()
    expect(screen.getByText(/Request \(kept exactly as written\):/)).toBeTruthy()
    expect(submit).not.toHaveBeenCalled()
  })

  it('shows only a rounded compact agent picker when opened from the pet', () => {
    render(<QuickEntryApp />)

    act(() => {
      pushState?.({
        agents: [
          { displayName: 'Hermes', profile: 'default', reachable: true },
          { displayName: 'Jarvis', profile: 'jarvis', reachable: true }
        ],
        connected: true,
        groups: [],
        sessions: []
      })
      showQuickEntry?.({ mode: 'agents' })
    })

    const picker = screen.getByRole('region', { name: 'Choose an agent' })
    expect(picker.style.borderRadius).toBe('11px')
    // The surface is themed from Hermes tokens in the picker stylesheet rather
    // than a literal colour, so assert the contract — it opts into the shared
    // glass class and paints nothing inline. Pinning channel values is what
    // kept this panel painting one fixed blue over every background.
    const surface = screen.getByTestId('agent-picker-surface')
    expect(surface.className).toContain('hq-surface')
    expect(surface.style.background).toBe('')
    const joke = screen.getByTestId('agent-picker-joke')
    expect(joke.textContent).toContain(hermesJokeOfTheDay())
    // Decoration, never announced over the choice the user is making.
    expect(screen.queryByRole('status')).toBeNull()
    expect(joke.getAttribute('aria-hidden')).toBe('true')
    expect(joke.style.whiteSpace).toBe('nowrap')
    expect(screen.getByRole('button', { name: 'Hermes' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Jarvis' })).toBeTruthy()
    expect(screen.queryByRole('img')).toBeNull()
    expect(screen.queryByRole('textbox', { name: 'Quick Entry' })).toBeNull()
    expect(screen.queryByRole('combobox', { name: 'Target session' })).toBeNull()
    expect(screen.queryByText('Hermes is reconnecting. Agent launch is temporarily unavailable.')).toBeNull()
  })

  it('shows the shipped fallback and starts the selected HUD while the prompt gateway reconnects', () => {
    render(<QuickEntryApp />)

    act(() => {
      showQuickEntry?.({ mode: 'agents' })
    })

    expect(screen.getAllByRole('button', { name: /Hermes/ })).toHaveLength(1)
    fireEvent.click(screen.getByRole('button', { name: 'Hermes' }))

    expect(submit).toHaveBeenCalledWith({
      action: 'open-agent',
      profile: 'default',
      requestId: expect.any(String)
    })
    expect(window.localStorage.getItem(DESKTOP_PET_AGENT_PROFILE_KEY)).toBe('default')
  })

  it('opens a fresh HUD intent when a reachable agent is clicked', () => {
    render(<QuickEntryApp />)

    act(() => {
      pushState?.({
        agents: [{ displayName: 'Fizz', profile: 'fizz', reachable: true }],
        connected: true,
        groups: [],
        sessions: []
      })
    })

    fireEvent.click(screen.getByRole('button', { name: 'Fizz' }))

    expect(submit).toHaveBeenCalledWith({
      action: 'open-agent',
      profile: 'fizz',
      requestId: expect.any(String)
    })
  })

  it('keeps unreachable agents visible but disabled', () => {
    render(<QuickEntryApp />)

    act(() => {
      pushState?.({
        agents: [{ displayName: 'Remote', profile: 'remote', reachable: false }],
        connected: true,
        groups: [],
        sessions: []
      })
    })

    expect(screen.getByRole('button', { name: 'Remote' }).hasAttribute('disabled')).toBe(true)
    expect(submit).not.toHaveBeenCalled()
  })

  it('moves between reachable agents with arrows and opens the selected HUD on Enter', () => {
    render(<QuickEntryApp />)

    act(() => {
      pushState?.({
        agents: [
          { displayName: 'Hermes', profile: 'default', reachable: true },
          { displayName: 'Fizz', profile: 'fizz', reachable: true }
        ],
        connected: true,
        groups: [],
        sessions: []
      })
    })

    const input = screen.getByRole('textbox', { name: 'Quick Entry' })
    fireEvent.keyDown(input, { key: 'ArrowRight' })
    fireEvent.keyDown(input, { key: 'Enter' })

    expect(submit).toHaveBeenCalledWith({
      action: 'open-agent',
      profile: 'fizz',
      requestId: expect.any(String)
    })
  })

  it('changes the full-character pose while previewing another agent', () => {
    render(<QuickEntryApp />)

    act(() => {
      pushState?.({
        agents: [
          { displayName: 'Hermes', profile: 'default', reachable: true },
          { displayName: 'Jarvis', profile: 'jarvis', reachable: true }
        ],
        connected: true,
        groups: [],
        sessions: []
      })
    })

    expect(screen.getByRole('img', { name: 'Hermes pose' })).toBeTruthy()
    fireEvent.mouseEnter(screen.getByRole('button', { name: 'Jarvis' }))
    expect(screen.getByRole('img', { name: 'Jarvis pose' })).toBeTruthy()
    expect(window.localStorage.getItem(DESKTOP_PET_AGENT_PROFILE_KEY)).toBe('jarvis')
    expect(submit).not.toHaveBeenCalled()
  })

  it('opens a selected group from the same compact chooser', () => {
    render(<QuickEntryApp />)

    act(() => {
      pushState?.({
        agents: [{ displayName: 'Hermes', profile: 'default', reachable: true }],
        connected: true,
        groups: [{ displayName: 'Research Team', groupId: 'room-research', memberCount: 3, reachable: true }],
        sessions: []
      })
      showQuickEntry?.({ mode: 'agents' })
    })

    fireEvent.click(screen.getByRole('tab', { name: 'Groups 1' }))
    fireEvent.click(screen.getByRole('button', { name: 'Research Team' }))

    expect(submit).toHaveBeenCalledWith({
      action: 'open-group',
      groupId: 'room-research',
      requestId: expect.any(String)
    })
  })

  const roster = () => ({
    agents: [
      { displayName: 'Hermes', profile: 'default', reachable: true },
      { displayName: 'Jarvis', profile: 'jarvis', reachable: true }
    ],
    connected: true,
    groups: [
      { displayName: 'Research Team', groupId: 'room-research', memberCount: 3, reachable: true },
      { displayName: 'Ops Room', groupId: 'room-ops', memberCount: 2, reachable: true }
    ],
    sessions: []
  })

  const lit = () => screen.queryAllByRole('button').filter(button => button.dataset.lit === 'true')

  it('opens with no row lit', () => {
    render(<QuickEntryApp />)

    act(() => {
      pushState?.(roster())
      showQuickEntry?.({ mode: 'agents' })
    })

    expect(lit()).toHaveLength(0)
  })

  it('lights only the row under the pointer, and unlights it on leave', () => {
    render(<QuickEntryApp />)

    act(() => {
      pushState?.(roster())
      showQuickEntry?.({ mode: 'agents' })
    })

    const jarvis = screen.getByRole('button', { name: 'Jarvis' })
    fireEvent.mouseEnter(jarvis)

    expect(lit()).toEqual([jarvis])

    fireEvent.mouseLeave(jarvis)

    expect(lit()).toHaveLength(0)
  })

  it('gives group rows the same hover treatment as agent rows', () => {
    render(<QuickEntryApp />)

    act(() => {
      pushState?.(roster())
      showQuickEntry?.({ mode: 'agents' })
    })

    fireEvent.click(screen.getByRole('tab', { name: 'Groups 2' }))
    const ops = screen.getByRole('button', { name: 'Ops Room' })
    fireEvent.mouseEnter(ops)

    expect(lit()).toEqual([ops])
    expect(ops.className).toContain('hq-row')
  })

  it('moves the group selection with arrows in the Groups tab', () => {
    render(<QuickEntryApp />)

    act(() => {
      pushState?.(roster())
      showQuickEntry?.({ mode: 'agents' })
    })

    fireEvent.click(screen.getByRole('tab', { name: 'Groups 2' }))
    fireEvent.keyDown(screen.getByRole('region', { name: 'Choose an agent' }), { key: 'ArrowDown' })

    expect(globalThis.document.activeElement).toBe(screen.getByRole('button', { name: 'Ops Room' }))
  })

  it('does not move focus when the pointer hovers a row', () => {
    render(<QuickEntryApp />)

    act(() => {
      pushState?.(roster())
      showQuickEntry?.({ mode: 'agents' })
    })

    const before = globalThis.document.activeElement
    fireEvent.mouseEnter(screen.getByRole('button', { name: 'Jarvis' }))

    expect(globalThis.document.activeElement).toBe(before)
  })

  it('switching tabs clears any lit row', () => {
    render(<QuickEntryApp />)

    act(() => {
      pushState?.(roster())
      showQuickEntry?.({ mode: 'agents' })
    })

    fireEvent.mouseEnter(screen.getByRole('button', { name: 'Jarvis' }))
    fireEvent.click(screen.getByRole('tab', { name: 'Groups 2' }))

    expect(lit()).toHaveLength(0)
  })
})

describe('quickEntryGroupAccent', () => {
  it('is stable per room name and always a palette colour', () => {
    expect(quickEntryGroupAccent('Research Team')).toBe(quickEntryGroupAccent('Research Team'))
    expect(quickEntryGroupAccent('Research Team')).toMatch(/^#[0-9a-f]{6}$/i)
  })
})

describe('quickEntryModeFromSearch', () => {
  it('uses URL mode on the first renderer load', () => {
    expect(quickEntryModeFromSearch('?win=quick&mode=agents')).toBe('agents')
    expect(quickEntryModeFromSearch('?win=quick')).toBe('composer')
  })
})

describe('hermesJokeOfTheDay', () => {
  it('keeps the same joke throughout a UTC day', () => {
    expect(hermesJokeOfTheDay(new Date('2026-08-23T00:00:01Z'))).toBe(
      hermesJokeOfTheDay(new Date('2026-08-23T23:59:59Z'))
    )
  })
})
