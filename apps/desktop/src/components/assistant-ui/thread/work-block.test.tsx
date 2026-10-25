// "Group work between replies": every stretch of thinking + tool calls between
// two visible texts is one folded line; edits fold in as steps; questions and
// other cards stay on screen. Covers the setting itself (off = stock), the
// display preferences the transcript honours, the folded label, streaming /
// reload / stop, prompts that arrive mid-stretch, and the keyboard contract.
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import { setRuntimeI18nLocale } from '@/i18n/runtime'
import { $interfaceMode, setInterfaceMode } from '@/store/interface-mode'
import { clearAllPrompts, setApprovalRequest } from '@/store/prompts'
import { $showReasoning, setReasoningCollapsedByDefault } from '@/store/reasoning-disclosure'
import { $activeSessionId } from '@/store/session'
import { setShowToolActivityFromConfig } from '@/store/tool-activity'
import { clearDismissedToolRows } from '@/store/tool-dismiss'
import { $groupWork, $toolDisclosureStates, setGroupWork, setHideCodeDiffs, setToolViewMode } from '@/store/tool-view'

import {
  at,
  edit,
  findHeader,
  Harness,
  headerButton,
  headerLabel,
  pending,
  read,
  refusedWrite,
  RUNNING,
  search,
  stubWorkBlockEnvironment,
  term,
  turn
} from './work-block-fixtures'

stubWorkBlockEnvironment()

const DONE = { type: 'text', text: 'Done.' }

const STRETCH = [
  { type: 'text', text: 'Looking.' },
  { type: 'reasoning', text: 'plan the reads' },
  read('r1', '/repo/one.ts'),
  term('t1'),
  DONE
]

const failedRead = { ...read('r9', '/repo/missing.ts'), isError: true, result: { error: 'File not found' } }

const blocks = (container: HTMLElement) => container.querySelectorAll('[data-work-block]')
const bodyRows = (container: HTMLElement) => container.querySelectorAll('[data-work-block-body] [data-tool-row]')

const thinkingRows = (scope: string, container: HTMLElement) =>
  container.querySelectorAll(`${scope} [data-slot="aui_thinking-disclosure"]`)

async function findEditRow(container: HTMLElement) {
  return waitFor(() => {
    const row = [...container.querySelectorAll<HTMLElement>('[data-tool-row]')].find(node =>
      node.textContent?.includes('a.ts')
    )

    expect(row).toBeTruthy()

    return row!
  })
}

async function openLine(container: HTMLElement, index = 0) {
  const header = await findHeader(container, index)

  fireEvent.click(headerButton(header))
  await waitFor(() => expect(headerButton(header).getAttribute('aria-expanded')).toBe('true'))

  return header
}

beforeEach(() => {
  clearAllPrompts()
  $activeSessionId.set('sess-1')
  $toolDisclosureStates.set({})
  clearDismissedToolRows()
  setGroupWork(true)
})

afterEach(() => {
  cleanup()
  setInterfaceMode('advanced')
  setGroupWork(false)
  setHideCodeDiffs(false)
  setToolViewMode('product')
  setReasoningCollapsedByDefault(false)
  setShowToolActivityFromConfig(true)
  $showReasoning.set(true)
  clearAllPrompts()
  $activeSessionId.set(null)
})

describe('the setting', () => {
  it('ships off in Advanced and leaves the transcript stock', async () => {
    setGroupWork(false)
    localStorage.removeItem('hermes.desktop.toolView.groupWork')
    const { container } = render(<Harness message={turn([...STRETCH.slice(0, 4), edit('e1'), DONE])} />)

    await screen.findByText('Done.')
    expect($groupWork.get()).toBe(false)
    expect(blocks(container)).toHaveLength(0)
    expect(container.querySelector('[data-work-block-header]')).toBeNull()
    expect(container.querySelector('[data-tool-summary], [data-tool-row]')).not.toBeNull()
  })

  it('re-renders an open chat when flipped, both ways, and persists the choice', async () => {
    setGroupWork(false)
    const { container } = render(<Harness message={turn(STRETCH)} />)

    await screen.findByText('Done.')
    expect(blocks(container)).toHaveLength(0)

    act(() => setGroupWork(true))
    await waitFor(() => expect(blocks(container)).toHaveLength(1))
    expect(localStorage.getItem('hermes.desktop.toolView.groupWork')).toBe('true')
    expect(screen.getByText('Done.')).toBeTruthy()

    act(() => setGroupWork(false))
    await waitFor(() => expect(blocks(container)).toHaveLength(0))
    expect(localStorage.getItem('hermes.desktop.toolView.groupWork')).toBe('false')
  })

  it('Simple mode groups without touching the Advanced preference', async () => {
    setGroupWork(false)
    const { container } = render(<Harness message={turn(STRETCH)} />)

    await screen.findByText('Done.')
    act(() => setInterfaceMode('simple'))
    expect($interfaceMode.get()).toBe('simple')
    await waitFor(() => expect(blocks(container)).toHaveLength(1))

    // A flip inside Simple is a session reveal, never the stored preference.
    act(() => setGroupWork(false))
    await waitFor(() => expect(blocks(container)).toHaveLength(0))
    expect(localStorage.getItem('hermes.desktop.toolView.groupWork')).not.toBe('true')

    act(() => setInterfaceMode('advanced'))
    await waitFor(() => expect(blocks(container)).toHaveLength(0))
    expect($groupWork.get()).toBe(false)
  })

  it('turning it off hands the edits the line folded back to stock, open', async () => {
    const { container } = render(<Harness message={turn([read('r1', '/repo/one.ts'), edit('e1'), DONE])} />)

    await openLine(container)
    expect((await findEditRow(container)).hasAttribute('data-tool-open')).toBe(false)

    act(() => setGroupWork(false))
    await waitFor(() => expect(blocks(container)).toHaveLength(0))
    await waitFor(async () => expect((await findEditRow(container)).hasAttribute('data-tool-open')).toBe(true))
    expect(Object.values($toolDisclosureStates.get())).not.toContain(false)
  })
})

describe('the line', () => {
  it.each([
    ['product', false],
    ['technical', true]
  ] as const)(
    'folds a stretch behind one closed line and opens to flat stock rows (%s view, thinking collapsed: %s)',
    async (mode, collapsed) => {
      setToolViewMode(mode)
      setReasoningCollapsedByDefault(collapsed)
      const { container } = render(<Harness message={turn(STRETCH)} />)

      await screen.findByText('Done.')
      expect(screen.getByText('Looking.')).toBeTruthy()
      expect(blocks(container)).toHaveLength(1)
      expect(container.querySelectorAll('[data-tool-row]')).toHaveLength(0)
      expect(thinkingRows('', container)).toHaveLength(0)

      await openLine(container)
      await waitFor(() => expect(bodyRows(container)).toHaveLength(2))
      expect(thinkingRows('[data-work-block-body]', container)).toHaveLength(1)
      // Flat: no nested run summary, and each step fades once (marks never nest).
      expect(container.querySelector('[data-work-block-body] [data-tool-summary]')).toBeNull()

      for (const row of container.querySelectorAll('[data-work-block-body] [data-conversation-scaffold]')) {
        expect(row.parentElement?.closest('[data-conversation-scaffold]')).toBeNull()
      }
    }
  )

  // Issue #76147: a single call collapses like a multi-call run, so a lone call
  // with no thinking folds behind the line too (stock ToolRun leaves it bare).
  it('folds a lone tool call with no thinking behind a line', async () => {
    const { container } = render(
      <Harness
        message={turn([
          { type: 'text', text: 'Looking.' },
          read('r1', '/repo/one.ts'),
          { type: 'text', text: 'Found it.' }
        ])}
      />
    )

    await screen.findByText('Found it.')
    expect(blocks(container)).toHaveLength(1)
    expect(headerLabel(await findHeader(container))).toBe('Explored one.ts')
    expect(container.querySelectorAll('[data-tool-row]')).toHaveLength(0)
  })

  it('folds edits in as steps that start folded and open their diff', async () => {
    const { container } = render(
      <Harness message={turn([read('r1', '/repo/one.ts'), edit('e1'), refusedWrite('w1'), term('t1'), DONE])} />
    )

    await screen.findByText('Done.')
    expect(blocks(container)).toHaveLength(1)
    expect(container.querySelectorAll('[data-tool-row]')).toHaveLength(0)

    await openLine(container)
    await waitFor(() => expect(bodyRows(container)).toHaveLength(4))
    const editRow = await findEditRow(container)

    expect(editRow.hasAttribute('data-tool-open')).toBe(false)
    fireEvent.click(editRow.querySelector('button[aria-expanded]')!)
    await waitFor(() => expect(editRow.hasAttribute('data-tool-open')).toBe(true))
    expect(editRow.textContent).toContain('afterEdit')
  })

  it('with code diffs hidden an edit inside the line shows counts, no code', async () => {
    setHideCodeDiffs(true)
    const { container } = render(<Harness message={turn([read('r1', '/repo/one.ts'), edit('e1'), DONE])} />)

    await openLine(container)
    const editRow = await findEditRow(container)

    expect(editRow.textContent).toContain('+1')
    expect(editRow.textContent).not.toContain('afterEdit')
    expect(editRow.querySelector('pre, code')).toBeNull()
  })

  it('shows one thinking row for a run of thinking (blank text between is dropped)', async () => {
    const { container } = render(
      <Harness
        message={turn([
          { type: 'reasoning', text: 'first thought' },
          { type: 'text', text: '' },
          { type: 'reasoning', text: 'second thought' },
          read('r1', '/repo/one.ts'),
          DONE
        ])}
      />
    )

    await openLine(container)
    await waitFor(() => expect(bodyRows(container)).toHaveLength(1))
    expect(thinkingRows('[data-work-block-body]', container)).toHaveLength(1)
  })

  it('with tool_progress off: no line and no tool rows, the replies stay', async () => {
    setShowToolActivityFromConfig('off')
    const { container } = render(<Harness message={turn(STRETCH)} />)

    await screen.findByText('Done.')
    expect(screen.getByText('Looking.')).toBeTruthy()
    expect(blocks(container)).toHaveLength(0)
    expect(container.querySelector('[data-tool-row], [data-tool-summary]')).toBeNull()
  })

  it.each(['new', 'all', 'verbose'])('with tool_progress %s: one line', async mode => {
    setShowToolActivityFromConfig(mode)
    const { container } = render(<Harness message={turn(STRETCH)} />)

    await screen.findByText('Done.')
    expect(blocks(container)).toHaveLength(1)
  })
})

describe('cards stay on screen and split the stretch', () => {
  const card = (toolCallId: string, toolName: string, args: object, result?: object) => ({
    type: 'tool-call',
    toolCallId,
    toolName,
    args,
    argsText: '{}',
    timestamp: at(3),
    ...(result ? { completedAt: at(4), result } : {})
  })

  it.each([
    [
      'a generated image',
      card('img-1', 'image_generate', { prompt: 'a cat' }, { success: true, image: 'https://example.com/cat.png' }),
      '[data-slot="aui_generated-image"]'
    ],
    [
      'a subagent card',
      card('d1', 'delegate_task', { goal: 'look around' }, { results: [{ status: 'completed', summary: 'ok' }] }),
      '[data-delegate-card]'
    ],
    [
      'an inter-agent delivery notice',
      card('m1', 'terminal', { command: `hermes -p turqoise chat -q "Message from Hermes: hi"` }),
      '[data-slot="aui_agent-delivery-notice"]'
    ],
    [
      'a theme picker reached through the tool_call bridge',
      card('theme-1', 'tool_call', { calls: [{ name: 'set_theme', arguments: { picker: 'accent' } }] }),
      null
    ]
  ])('%s', async (_name, part, selector) => {
    const { container } = render(<Harness message={turn([read('r1', '/repo/one.ts'), part, term('t1'), DONE])} />)

    await screen.findByText('Done.')
    expect(blocks(container)).toHaveLength(2)

    if (selector) {
      const node = container.querySelector(selector)
      expect(node).not.toBeNull()
      expect(node?.closest('[data-work-block]')).toBeNull()
    }
  })
})

describe('the folded label', () => {
  it('orders clauses edited, explored, searched, ran, then failures', async () => {
    const { container } = render(
      <Harness
        message={turn([
          term('t1'),
          search('s1', 'cats'),
          read('r1', '/repo/one.ts'),
          failedRead,
          edit('e1'),
          term('t2', 'pwd'),
          DONE
        ])}
      />
    )

    expect(headerLabel(await findHeader(container))).toBe(
      'Edited a.ts, explored 2 files, searched “cats”, ran 2 commands, 1 tool call failed'
    )
  })

  it('names a lone file, counts a settled command, singular and plural', async () => {
    const { container } = render(<Harness message={turn([read('r1', '/repo/one.ts'), term('t1'), DONE])} />)

    expect(headerLabel(await findHeader(container))).toBe('Explored one.ts, ran 1 command')
    cleanup()

    const many = render(
      <Harness message={turn([read('r1', '/repo/one.ts'), read('r2', '/repo/two.ts'), term('t1'), term('t2'), DONE])} />
    )

    expect(headerLabel(await findHeader(many.container))).toBe('Explored 2 files, ran 2 commands')
  })

  it('adds "Thought ·" only for visible thinking, and drops the thinking row when show_reasoning is off', async () => {
    const withThought = render(<Harness message={turn(STRETCH)} />)
    expect(headerLabel(await findHeader(withThought.container))).toBe('Thought · Explored one.ts, ran 1 command')
    cleanup()

    const blank = render(
      <Harness message={turn([{ type: 'reasoning', text: '   ' }, read('r1', '/repo/one.ts'), DONE])} />
    )

    expect(headerLabel(await findHeader(blank.container))).toBe('Explored one.ts')
    cleanup()

    $showReasoning.set(false)
    const hidden = render(<Harness message={turn(STRETCH)} />)
    expect(headerLabel(await findHeader(hidden.container))).toBe('Explored one.ts, ran 1 command')
    await openLine(hidden.container)
    await waitFor(() => expect(bodyRows(hidden.container)).toHaveLength(2))
    expect(thinkingRows('[data-work-block-body]', hidden.container)).toHaveLength(0)
  })

  it('speaks in the present tense and shimmers while live, past tense once settled', async () => {
    const live = [{ type: 'text', text: 'Looking.' }, read('r1', '/repo/one.ts'), pending(term('t1', 'npm test'))]
    const { container, rerender } = render(<Harness message={turn(live, RUNNING)} />)

    const header = await findHeader(container)
    await waitFor(() => expect(headerLabel(header)).toMatch(/^Explored one\.ts, running /))
    expect(header.querySelector('.shimmer')).not.toBeNull()

    await act(async () => {
      rerender(<Harness message={turn([live[0], live[1], term('t1', 'npm test'), { type: 'text', text: 'Green.' }])} />)
    })
    await screen.findByText('Green.')
    const settled = await findHeader(container)
    expect(headerLabel(settled)).toBe('Explored one.ts, ran 1 command')
    expect(settled.querySelector('.shimmer')).toBeNull()
  })

  it('localizes the Thought prefix and the failure clause (de)', async () => {
    // The app's provider sets the runtime locale at boot, before any chat renders.
    setRuntimeI18nLocale('de')

    try {
      const { container } = render(
        <I18nProvider configClient={null} initialLocale="de">
          <Harness
            message={turn([{ type: 'reasoning', text: 'plan' }, read('r1', '/repo/one.ts'), failedRead, DONE])}
          />
        </I18nProvider>
      )

      const header = await findHeader(container)
      await waitFor(() => expect(headerLabel(header)).toMatch(/^Gedanke · /))
      expect(headerLabel(header)).toContain('1 Tool-Aufruf fehlgeschlagen')
    } finally {
      setRuntimeI18nLocale('en')
    }
  })
})

describe('states', () => {
  it('keeps the same line while the thinking that opens a stretch streams', async () => {
    const t1 = term('t1')

    const streaming = (n: number) =>
      turn([{ type: 'text', text: 'Looking.' }, { type: 'reasoning', text: 'x'.repeat(n) }, t1], RUNNING)

    const { container, rerender } = render(<Harness message={streaming(1)} />)
    const header = await findHeader(container)

    for (let n = 2; n < 6; n++) {
      await act(async () => {
        rerender(<Harness message={streaming(n)} />)
      })
    }

    expect(container.querySelector('[data-work-block-header]')).toBe(header)
  })

  it('the line appears when the first tool lands, keeping the same block element', async () => {
    const thinking = [
      { type: 'text', text: 'Looking.' },
      { type: 'reasoning', text: 'plan the read' }
    ]

    const { container, rerender } = render(<Harness message={turn(thinking, RUNNING)} />)

    const wrapper = await waitFor(() => {
      const node = container.querySelector('[data-slot="tool-block"]')
      expect(node).not.toBeNull()

      return node
    })

    expect(container.querySelector('[data-work-block-header]')).toBeNull()
    expect(thinkingRows('', container)).toHaveLength(1)

    await act(async () => {
      rerender(<Harness message={turn([...thinking, pending(read('r1', '/repo/one.ts'))], RUNNING)} />)
    })

    await findHeader(container)
    expect(container.querySelector('[data-slot="tool-block"]')).toBe(wrapper)
  })

  it('reload from history renders the same line as the live turn ended with', async () => {
    const parts = [...STRETCH.slice(0, 4)]
    const live = render(<Harness message={turn(parts, RUNNING)} />)

    await act(async () => {
      live.rerender(<Harness message={turn([...parts, DONE])} />)
    })
    await screen.findByText('Done.')
    const liveLabel = headerLabel(await findHeader(live.container))
    const liveCount = blocks(live.container).length
    cleanup()

    const reloaded = render(<Harness message={turn(STRETCH)} />)
    await screen.findByText('Done.')
    expect(headerLabel(await findHeader(reloaded.container))).toBe(liveLabel)
    expect(blocks(reloaded.container)).toHaveLength(liveCount)
  })

  it('a turn stopped mid-stretch reads as finished, not running', async () => {
    const stopped = turn([read('r1', '/repo/one.ts'), pending(term('t1', 'sleep 100'))], {
      type: 'incomplete',
      reason: 'cancelled'
    })

    const { container } = render(<Harness message={stopped} />)

    const header = await findHeader(container)
    expect(headerLabel(header)).not.toMatch(/Running|Exploring/)
    expect(header.querySelector('.shimmer')).toBeNull()
  })

  it('a question arriving mid-stretch splits it and leaves the open line open', async () => {
    const before = [{ type: 'text', text: 'Looking.' }, read('r1', '/repo/one.ts'), term('t1')]
    const { container, rerender } = render(<Harness message={turn(before, RUNNING)} />)

    const header = await openLine(container)

    const clarify = {
      type: 'tool-call',
      toolCallId: 'q1',
      toolName: 'clarify',
      args: { question: 'Which file?', choices: ['a', 'b'] },
      argsText: '{}',
      timestamp: at(5)
    }

    await act(async () => {
      rerender(<Harness message={turn([...before, clarify], RUNNING)} />)
    })

    expect(container.querySelector('[data-work-block-header]')).toBe(header)
    expect(headerButton(header).getAttribute('aria-expanded')).toBe('true')
    await waitFor(() => expect(bodyRows(container)).toHaveLength(2))
    const question = container.querySelector('[data-slot="clarify-inline"]')
    expect(question?.closest('[data-work-block]') ?? null).toBeNull()
  })

  it('an approval arriving while the line is open leaves it open', async () => {
    const { container } = render(
      <Harness
        message={turn([{ type: 'text', text: 'Looking.' }, read('r1', '/repo/one.ts'), pending(term('t1'))], RUNNING)}
      />
    )

    const header = await openLine(container)

    act(() => {
      setApprovalRequest({ command: 'rm -rf /tmp/x', description: 'dangerous command', sessionId: 'sess-1' })
    })

    expect(container.querySelector('[data-work-block-header]')).toBe(header)
    expect(headerButton(header).getAttribute('aria-expanded')).toBe('true')
  })
})

describe('keyboard and a11y', () => {
  it('the line takes focus and its aria-expanded follows the fold', async () => {
    const { container } = render(<Harness message={turn(STRETCH)} />)

    const button = headerButton(await findHeader(container))

    button.focus()
    // eslint-disable-next-line no-restricted-globals -- asserting real focus requires the live document
    expect(document.activeElement).toBe(button)
    expect(button.getAttribute('aria-expanded')).toBe('false')

    fireEvent.click(button)
    await waitFor(() => expect(button.getAttribute('aria-expanded')).toBe('true'))
    expect(container.querySelector('[data-work-block-body]')).not.toBeNull()

    fireEvent.click(button)
    await waitFor(() => expect(button.getAttribute('aria-expanded')).toBe('false'))
    expect(container.querySelector('[data-work-block-body]')).toBeNull()
  })
})
