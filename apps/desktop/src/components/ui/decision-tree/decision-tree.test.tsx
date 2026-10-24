import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { createRef } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { DecisionTree } from './decision-tree'
import type { Decision, DecisionTreeHandle } from './types'

const checkout: Decision = {
  id: 'checkout',
  question: 'Checkout errors jumped to 18% after deploy #482 — what should I do?',
  context: '2,140 errors in 12 minutes · about $3.2k in orders an hour at risk.',
  agent: 'Incident agent',
  model: 'Opus 5.5',
  risk: 'high',
  options: [
    {
      label: 'Roll back to #481',
      pick: true,
      effects: [
        'Errors stop within a minute',
        {
          text: 'The 24 orders in flight are dropped',
          note: 'Refunds are manual for anything past the payment step.',
          source: 'runbook §4'
        }
      ]
    },
    {
      label: 'Ship a hotfix',
      next: {
        question: 'Wait for the test run?',
        options: [
          { label: 'Ship right away', effects: ['Skips 1,284 tests'] },
          { label: 'After the tests', effects: ['14 min · errors keep coming'] }
        ]
      }
    },
    { label: 'Turn the flag off', effects: ['New checkout off for 30%'] }
  ]
}

const refunds: Decision = {
  id: 'refunds',
  question: 'Refund the two duplicates?',
  agent: 'Billing agent',
  risk: 'low',
  options: [{ label: 'Refund both', effects: ['$294 back'] }, { label: 'Refund one' }]
}

afterEach(cleanup)

const findButton = (name: RegExp) => screen.queryAllByRole('button').find(button => name.test(button.textContent ?? ''))

const root = () => screen.getByRole('group')

describe('DecisionTree', () => {
  it('renders a real button per branch with its digit as the shortcut', () => {
    render(<DecisionTree decisions={[checkout]} />)

    expect(findButton(/Roll back to #481/)?.getAttribute('aria-keyshortcuts')).toBe('1')
    expect(findButton(/Turn the flag off/)?.getAttribute('aria-keyshortcuts')).toBe('3')
  })

  it("marks the agent's pick without hiding the other branches", () => {
    render(<DecisionTree decisions={[checkout]} />)

    expect(findButton(/Roll back to #481/)?.textContent).toContain("the agent's pick")
    expect(findButton(/Ship a hotfix/)?.textContent).not.toContain("the agent's pick")
  })

  it('picks a branch with its digit once focus is inside the tree', () => {
    const onChoose = vi.fn()

    render(<DecisionTree decisions={[checkout]} onChoose={onChoose} />)

    fireEvent.keyDown(findButton(/Turn the flag off/)!, { key: '3' })

    expect(onChoose).toHaveBeenCalledTimes(1)
    expect(onChoose.mock.calls[0][1]).toMatchObject({ index: 2, option: { label: 'Turn the flag off' } })
  })

  it('wakes a follow-up before it settles the route, then sends both picks as one choice', () => {
    const onChoose = vi.fn()

    render(<DecisionTree decisions={[checkout]} onChoose={onChoose} />)

    fireEvent.keyDown(findButton(/Ship a hotfix/)!, { key: '2' })

    // The route is not decided yet: the follow-up is a second question.
    expect(onChoose).not.toHaveBeenCalled()
    expect(screen.getAllByText('Wait for the test run?').length).toBeGreaterThan(0)

    fireEvent.keyDown(findButton(/Ship right away/)!, { key: '1' })

    expect(onChoose).toHaveBeenCalledTimes(1)
    expect(onChoose.mock.calls[0][1]).toMatchObject({
      index: 1,
      option: { label: 'Ship a hotfix' },
      then: { label: 'Ship right away' }
    })
  })

  it('keeps a dormant follow-up out of the tab order until its branch is picked', () => {
    const { container } = render(<DecisionTree decisions={[checkout]} />)

    expect(container.querySelector('[data-slot="decision-tree-follow-up"]')?.hasAttribute('inert')).toBe(true)
    expect(findButton(/Ship right away/)?.getAttribute('tabindex')).toBe('-1')
  })

  it("settles the agent's short answer into the outcome line", async () => {
    const onChoose = vi.fn(async () => 'Rolled back · #482 reverted')

    render(<DecisionTree decisions={[checkout]} onChoose={onChoose} />)

    fireEvent.keyDown(findButton(/Roll back to #481/)!, { key: '1' })

    await waitFor(() => expect(screen.getAllByText('Rolled back · #482 reverted').length).toBeGreaterThan(0))
  })

  it('sends a written instruction as its own route and keeps Enter out of the choosing keys', () => {
    const onChoose = vi.fn()

    render(<DecisionTree decisions={[checkout]} onChoose={onChoose} />)

    fireEvent.keyDown(findButton(/Ship a hotfix/)!, { key: '4' })

    const field = screen.getByRole('textbox') as HTMLTextAreaElement

    // A field owns its keys — digits must not pick anything under it.
    fireEvent.keyDown(field, { key: '1' })
    expect(onChoose).not.toHaveBeenCalled()

    fireEvent.change(field, { target: { value: 'Revert #482 and page the on-call' } })
    fireEvent.keyDown(field, { key: 'Enter', shiftKey: true })
    expect(onChoose).not.toHaveBeenCalled()

    fireEvent.keyDown(field, { key: 'Enter' })

    expect(onChoose).toHaveBeenCalledTimes(1)
    expect(onChoose.mock.calls[0][1]).toMatchObject({
      index: -1,
      option: { label: 'Revert #482 and page the on-call' },
      other: 'Revert #482 and page the on-call'
    })
  })

  it('closes "Other…" on Esc without choosing anything', () => {
    const onChoose = vi.fn()

    render(<DecisionTree decisions={[checkout]} onChoose={onChoose} />)

    fireEvent.keyDown(findButton(/Ship a hotfix/)!, { key: '4' })
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Escape' })

    expect(screen.queryByRole('textbox')).toBeNull()
    expect(onChoose).not.toHaveBeenCalled()
  })

  it('describes a flagged consequence with its note, closed until it is wanted', () => {
    render(<DecisionTree decisions={[checkout]} />)

    const flag = findButton(/The 24 orders in flight are dropped/)
    const describedBy = flag?.getAttribute('aria-describedby') ?? ''
    const note = describedBy ? screen.getByRole('tooltip') : null

    expect(note?.id).toBe(describedBy)
    expect(note?.textContent).toContain('Refunds are manual')
    expect(note?.textContent).toContain('runbook §4')
    expect(note?.className).toContain('sr-only')

    fireEvent.focus(flag!)

    expect(note?.className).toContain('dt-note-open')
  })

  it('walks the flagged consequences with F', () => {
    const { container } = render(<DecisionTree decisions={[checkout]} />)

    fireEvent.keyDown(root(), { key: 'f' })

    const open = container.querySelectorAll('[data-slot="decision-tree-note"].dt-note-open')

    expect(open).toHaveLength(1)
    expect(open[0].textContent).toContain('Refunds are manual')
  })

  it('moves between decisions with the arrows and J/K, and keeps the keys with the active one', async () => {
    const onChoose = vi.fn()

    render(<DecisionTree decisions={[checkout, refunds]} onChoose={onChoose} />)

    fireEvent.keyDown(root(), { key: 'ArrowRight' })
    expect(root().textContent).toContain('2/2')

    fireEvent.keyDown(findButton(/Refund one/)!, { key: '2' })
    expect(onChoose.mock.calls[0][0]).toMatchObject({ id: 'refunds' })

    // The settled decision still navigates: the receipt is part of the transcript.
    await waitFor(() => expect(screen.getAllByText('Chose Refund one').length).toBeGreaterThan(0))
    fireEvent.keyDown(root(), { key: 'k' })

    expect(root().textContent).toContain('1/2')
  })

  it('shows the risk tag and an empty state', () => {
    const { container, unmount } = render(<DecisionTree decisions={[checkout]} />)

    expect(container.querySelector('[data-slot="decision-tree-risk"]')?.textContent).toBe('High risk')
    unmount()

    render(<DecisionTree decisions={[]} />)
    expect(screen.queryByText('Nothing is waiting on a decision.')).not.toBeNull()
  })

  it('drives the tree from outside through its handle', async () => {
    const onChoose = vi.fn()
    const ref = createRef<DecisionTreeHandle>()

    const { container } = render(<DecisionTree decisions={[checkout]} onChoose={onChoose} ref={ref} />)

    act(() => ref.current?.hover(1))
    expect(container.querySelector('[data-slot="decision-tree-branch-node"][data-lit]')).not.toBeNull()

    act(() => ref.current?.setOther('Page the on-call'))
    expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe('Page the on-call')

    await act(async () => ref.current?.send())

    expect(onChoose).toHaveBeenCalledTimes(1)
    expect(onChoose.mock.calls[0][1]).toMatchObject({ index: -1, other: 'Page the on-call' })
  })

  it('reaches a branch and moves on through the handle', async () => {
    const onChoose = vi.fn()
    const ref = createRef<DecisionTreeHandle>()

    render(<DecisionTree decisions={[checkout, refunds]} onChoose={onChoose} ref={ref} />)

    await act(async () => ref.current?.choose(2))

    expect(onChoose.mock.calls[0][1]).toMatchObject({ index: 2 })

    act(() => ref.current?.next())
    expect(root().textContent).toContain('2/2')

    act(() => ref.current?.previous())
    expect(root().textContent).toContain('1/2')
  })
})
