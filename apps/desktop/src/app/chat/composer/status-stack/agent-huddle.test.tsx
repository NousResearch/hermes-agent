import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $subagentsBySession, type SubagentProgress, upsertSubagent } from '@/store/subagents'

import { AgentHuddle, AgentTypingIndicator, huddleState } from './agent-huddle'

// The report of a finished worker used to be painted above the composer as its
// own model answer (markdown, tables and all). It is still the truth the Live
// agent retells out loud, but on screen the user must only ever get the little
// animated huddle of agents talking to each other — never the text.

const installMatchMedia = (matches: boolean) =>
  vi.stubGlobal(
    'matchMedia',
    vi.fn(() => ({
      addEventListener: vi.fn(),
      matches,
      media: '(prefers-reduced-motion: reduce)',
      onchange: null,
      removeEventListener: vi.fn()
    }))
  )

beforeEach(() => {
  vi.stubGlobal(
    'ResizeObserver',
    class {
      disconnect() {}
      observe() {}
      unobserve() {}
    }
  )
  installMatchMedia(false)
})

afterEach(() => {
  cleanup()
  $subagentsBySession.set({})
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

const worker = (over: Partial<SubagentProgress> = {}): SubagentProgress => ({
  filesRead: [],
  filesWritten: [],
  goal: 'Zbadaj system pamięci',
  id: `worker-${Math.random()}`,
  parentId: null,
  startedAt: 0,
  status: 'running',
  stream: [],
  taskCount: 1,
  taskIndex: 0,
  updatedAt: 0,
  ...over
})

const REPORT = [
  'Sprawdziłem historię — o pamięci gadałem Ci tylko przelotnie:',
  '',
  '| Warstwa | Gdzie siedzi | Stan |',
  '|---|---|---|',
  '| Pamięć trwała | `hermes-home/memories/` | **pusta** |'
].join('\n')

describe('huddleState', () => {
  it('has no scene while nothing is delegated', () => {
    expect(huddleState([])).toBeNull()
  })

  it('waits while a worker is queued or has not said anything yet', () => {
    expect(huddleState([worker({ status: 'queued' })])).toBe('waiting')
    expect(huddleState([worker({ status: 'running', stream: [] })])).toBe('waiting')
  })

  it('talks once a live worker has real activity behind it', () => {
    const item = worker({ stream: [{ at: 1, kind: 'tool', text: 'Read File' }] })

    expect(huddleState([item])).toBe('talking')
  })

  it('thinks while the last thing a live worker said was a thought', () => {
    const item = worker({
      stream: [
        { at: 1, kind: 'tool', text: 'Read File' },
        { at: 2, kind: 'thinking', text: 'Hmm' }
      ]
    })

    expect(huddleState([item])).toBe('thinking')
  })

  it('is done when every worker on the roster has settled', () => {
    expect(huddleState([worker({ status: 'completed' })])).toBe('done')
    expect(huddleState([worker({ status: 'failed' }), worker({ status: 'interrupted' })])).toBe('done')
  })
})

describe('AgentHuddle', () => {
  it('renders no scene, and nothing of the report, when there is no work', () => {
    const { container } = render(<AgentHuddle sessionId="owner" />)

    expect(container.querySelector('[data-testid="agent-huddle"]')).toBeNull()
  })

  it('stands up the agents while work runs, in the state of that work', () => {
    upsertSubagent('owner', { goal: 'Zbadaj system pamięci', status: 'running', subagent_id: 'child-1' })

    const { container } = render(<AgentHuddle sessionId="owner" />)
    const huddle = container.querySelector('[data-testid="agent-huddle"]')

    expect(huddle?.getAttribute('data-state')).toBe('waiting')
    // Czesiek + Hermes are always on stage; the roster adds the helpers.
    expect(container.querySelectorAll('.agent-huddle__figure')).toHaveLength(3)
    expect(container.textContent).toContain('Czesiek')
    expect(container.textContent).toContain('Hermes')
  })

  it('never paints the finished report as text — not the source, not the rendering', () => {
    upsertSubagent('owner', {
      files_written: ['C:/Users/ostry/hermes-vault/CORE_MEMORY.md'],
      goal: 'Zbadaj system pamięci',
      status: 'completed',
      subagent_id: 'child-1',
      summary: REPORT
    })

    const { container } = render(<AgentHuddle sessionId="owner" />)
    const text = container.textContent ?? ''

    expect(container.querySelector('[data-testid="agent-huddle"]')?.getAttribute('data-state')).toBe('done')
    expect(container.querySelector('pre')).toBeNull()
    expect(container.querySelector('table')).toBeNull()
    expect(text).not.toContain(REPORT)
    expect(text).not.toContain('**pusta**')
    expect(text).not.toContain('hermes-home/memories/')
    expect(text).not.toContain('CORE_MEMORY.md')
    // The one thing the lane says out loud is that the huddle finished.
    expect(text).toContain('Narada zakończona')
  })

  it('keeps at most four characters on stage', () => {
    for (let i = 0; i < 5; i++) {
      upsertSubagent('owner', { goal: `Zadanie ${i}`, status: 'running', subagent_id: `child-${i}` })
    }

    const { container } = render(<AgentHuddle sessionId="owner" />)

    expect(container.querySelectorAll('.agent-huddle__figure')).toHaveLength(4)
  })

  it('holds a calm, still scene when the user asks for reduced motion', () => {
    installMatchMedia(true)
    upsertSubagent('owner', { goal: 'Zbadaj system pamięci', status: 'running', subagent_id: 'child-1' })

    const { container } = render(<AgentHuddle sessionId="owner" />)

    expect(container.querySelector('[data-testid="agent-huddle"]')?.getAttribute('data-motion')).toBe('reduced')
    expect(screen.getByTestId('agent-huddle')).toBeTruthy()
  })
})

describe('AgentTypingIndicator', () => {
  it('animates only while the child is producing output, and never carries its text', () => {
    const { container, rerender } = render(<AgentTypingIndicator active />)

    expect(container.querySelector('[data-testid="agent-typing"]')?.getAttribute('data-active')).toBe('true')
    expect(container.textContent).toBe('')

    rerender(<AgentTypingIndicator active={false} />)
    expect(container.querySelector('[data-testid="agent-typing"]')?.getAttribute('data-active')).toBe('false')
  })
})
