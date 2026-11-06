import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it } from 'vitest'

// The drawer lives inside a composer, so the test mounts it in a real composer
// scope (the same context its SDK hook reads in the app).
// eslint-disable-next-line no-restricted-imports
import { ComposerScopeProvider, MAIN_COMPOSER_SCOPE } from '@/app/chat/composer/scope'
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import { PLUGIN_ID, PROMPT_REFINEMENT_LOCALES } from './i18n'
import { createReviewDrawer } from './review-drawer'
import { createReviewStore, type ReviewRequest, type ReviewStore } from './review-store'
import type { RefineResult, ReviewDecision } from './types'

const ORIGINAL = 'fix the flaky test in src/app/index.test.ts and do not touch other files'
const REFINED = 'Fix the flaky test in src/app/index.test.ts. Do not touch other files.'

const ANALYSIS = {
  professional: 'Find the root cause',
  rawMaterial: 'CI logs',
  methods: 'git bisect',
  verification: '50 green runs'
}

beforeAll(() => {
  registerPluginLocales(PLUGIN_ID, PROMPT_REFINEMENT_LOCALES)
})

afterEach(cleanup)

function mount(target = 'main') {
  const store = createReviewStore()
  const Drawer = createReviewDrawer(store)

  const view = render(
    <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, target }}>
      <Drawer />
    </ComposerScopeProvider>
  )

  return { store, view }
}

/** Opens a review. Returns the pending decision WRAPPED: an async function
 *  returning the bare promise would make `await open()` wait for the user. */
async function open(
  store: ReviewStore,
  result: RefineResult | 'pending',
  overrides: Partial<ReviewRequest> = {}
): Promise<{ decision: Promise<ReviewDecision> }> {
  let decision!: Promise<ReviewDecision>

  await act(async () => {
    decision = store.request({
      target: 'main',
      original: ORIGINAL,
      attachmentLabels: [],
      run: () => (result === 'pending' ? new Promise(() => {}) : Promise.resolve(result)),
      ...overrides
    })
    await Promise.resolve()
  })
  // The drawer paints folded (inert) for one frame, then slides open.
  await act(() => new Promise(resolve => setTimeout(resolve, 40)))

  return { decision }
}

const ok = (text: string, analysis: null | typeof ANALYSIS = null): RefineResult => ({
  ok: true,
  text,
  unchanged: false,
  analysis
})

const region = () => screen.queryByRole('region', { name: 'Prompt refinement' })
const button = (name: string) => screen.getByRole('button', { name }) as HTMLButtonElement

describe('refinement drawer — placement and motion', () => {
  it('opens only in the composer the review belongs to', async () => {
    const { store } = mount('main')
    await open(store, 'pending', { target: 'tile:other' })

    expect(region()).toBeNull()
  })

  it('paints folded, then slides open, and takes focus', async () => {
    const { store, view } = mount()
    await open(store, 'pending')

    const drawer = view.container.querySelector('.status-drawer')!
    expect(drawer).not.toBeNull()
    await waitFor(() => expect(drawer.hasAttribute('data-collapsed')).toBe(false))
    // eslint-disable-next-line no-restricted-globals -- asserting real focus requires the live document
    expect(document.activeElement).toBe(region())
  })

  it('cancels the review when its composer goes away, so the send is never stranded', async () => {
    const { store, view } = mount()
    const { decision } = await open(store, 'pending')

    view.unmount()

    expect(await decision).toEqual({ kind: 'cancel', reason: 'no-host' })
  })
})

describe('refinement drawer — while refining', () => {
  it('a held send shows one slim row with Send original and Cancel', async () => {
    const { store } = mount()
    const { decision } = await open(store, 'pending', { autoApprove: true })

    expect(screen.getByText('Refining your prompt before sending')).toBeTruthy()
    fireEvent.click(button('Send original'))

    expect(await decision).toEqual({ kind: 'original' })
  })

  it('Cancel keeps the draft: nothing is sent', async () => {
    const { store } = mount()
    const { decision } = await open(store, 'pending')

    fireEvent.click(button('Cancel'))

    expect(await decision).toEqual({ kind: 'cancel', reason: 'user' })
  })
})

describe('refinement drawer — review', () => {
  it('shows the refined prompt, and what was added only on request', async () => {
    const { store } = mount()
    const { decision } = await open(store, ok(REFINED, ANALYSIS))

    expect(screen.getByText(REFINED)).toBeTruthy()
    expect(screen.getByText('Everything from your prompt is kept')).toBeTruthy()
    expect(screen.queryByText('git bisect')).toBeNull()

    fireEvent.click(button('What was added and why'))
    expect(screen.getByText('Raw material')).toBeTruthy()
    expect(screen.getByText('git bisect')).toBeTruthy()

    fireEvent.click(button('Approve and send'))
    expect(await decision).toEqual({ kind: 'approve', text: REFINED })
  })

  it('Approval-First: a clean refinement waits for the user instead of sending', async () => {
    const { store } = mount()
    const { decision } = await open(store, ok(REFINED))
    let settled = false
    void decision.then(() => (settled = true))
    await act(() => new Promise(resolve => setTimeout(resolve, 20)))

    expect(settled).toBe(false)
    expect(screen.queryByText(/Not sent automatically/)).toBeNull()
    fireEvent.click(button('Approve and send'))
    expect(await decision).toEqual({ kind: 'approve', text: REFINED })
  })

  it('names the depth the refiner chose for an Auto depth request', async () => {
    const { store } = mount()
    await open(store, { ...ok(REFINED), chosenDepth: 'clarify' } as RefineResult)

    expect(screen.getByText('Depth: Clarify, chosen automatically')).toBeTruthy()
  })

  it('keeps Approve off while a file path is missing, until Edit restores it', async () => {
    const { store } = mount()
    const { decision } = await open(store, ok('Fix the flaky test. Do not touch other files.'))

    expect(screen.getByText('File path: src/app/index.test.ts')).toBeTruthy()
    expect(button('Approve and send').disabled).toBe(true)

    fireEvent.click(button('Edit'))
    fireEvent.change(screen.getByRole('textbox', { name: 'Edit the refined prompt' }), { target: { value: REFINED } })
    expect(button('Approve and send').disabled).toBe(false)
    fireEvent.click(button('Approve and send'))

    expect(await decision).toEqual({ kind: 'approve', text: REFINED })
  })

  it('lists what the rewrite asks for beyond the prompt, and needs the check before Approve', async () => {
    const { store } = mount()
    const extra = `${REFINED} Then delete the old snapshots.`
    const { decision } = await open(store, ok(extra))

    expect(screen.getByText('Asks for more than your prompt did')).toBeTruthy()
    expect(screen.getByText('Deletes or overwrites something: “delete”')).toBeTruthy()
    expect(button('Approve and send').disabled).toBe(true)

    fireEvent.click(screen.getByRole('checkbox'))
    expect(button('Approve and send').disabled).toBe(false)
    fireEvent.click(button('Approve and send'))

    expect(await decision).toEqual({ kind: 'approve', text: extra })
  })

  it('a stopped automatic send says why, shows your prompt on request, and offers to send it', async () => {
    const { store } = mount()
    const { decision } = await open(store, ok('Fix the flaky test.'), { autoApprove: true })

    expect(screen.getByText(/Not sent automatically/)).toBeTruthy()
    expect(button('Approve and send').disabled).toBe(true)
    fireEvent.click(button('Your prompt'))
    expect(screen.getByText(ORIGINAL)).toBeTruthy()

    fireEvent.click(button('Send original'))
    expect(await decision).toEqual({ kind: 'original' })
  })

  it('a failure offers Try again, never Approve', async () => {
    const { store } = mount()
    await open(store, { ok: false, failure: 'unavailable' }, { autoApprove: true })

    expect(screen.getByText(/No refinement model is available/)).toBeTruthy()
    expect(screen.getByText('Not sent: refinement didn’t finish.')).toBeTruthy()
    expect(button('Try again')).toBeTruthy()
    expect(button('Approve and send').disabled).toBe(true)
  })

  it('says how to turn on the backend plugin when the backend has no refinement slot', async () => {
    const { store } = mount()
    await open(store, { ok: false, failure: 'slot-missing' }, { autoApprove: true })

    expect(screen.getByText(/hermes plugins enable prompt-refinement/)).toBeTruthy()
    expect(button('Try again')).toBeTruthy()
    expect(button('Approve and send').disabled).toBe(true)
  })
})

describe('refinement drawer — keyboard', () => {
  it('Esc inside the drawer cancels, and is marked handled so the composer does not also stop a turn', async () => {
    const { store } = mount()
    const { decision } = await open(store, 'pending')
    let seenByWindow: boolean | null = null
    const listener = (event: KeyboardEvent) => (seenByWindow = event.defaultPrevented)
    window.addEventListener('keydown', listener)

    fireEvent.keyDown(region()!, { key: 'Escape' })
    window.removeEventListener('keydown', listener)

    expect(await decision).toEqual({ kind: 'cancel', reason: 'user' })
    expect(seenByWindow === null || seenByWindow === true).toBe(true)
  })

  it('⌘+Enter approves from where focus lands when the drawer opens (the drawer itself)', async () => {
    const { store } = mount()
    const { decision } = await open(store, ok(REFINED))

    // eslint-disable-next-line no-restricted-globals -- asserting real focus requires the live document
    expect(document.activeElement).toBe(region())
    fireEvent.keyDown(region()!, { key: 'Enter', metaKey: true })

    expect(await decision).toEqual({ kind: 'approve', text: REFINED })
  })

  it('Ctrl+Enter approves from inside the panel too', async () => {
    const { store } = mount()
    const { decision } = await open(store, ok(REFINED))

    fireEvent.keyDown(screen.getByText(REFINED), { key: 'Enter', ctrlKey: true })

    expect(await decision).toEqual({ kind: 'approve', text: REFINED })
  })

  it('the shortcut does nothing while Approve is blocked', async () => {
    const { store } = mount()
    await open(store, ok('Fix the flaky test. Do not touch other files.'))

    fireEvent.keyDown(region()!, { key: 'Enter', metaKey: true })

    expect(store.$review.get()).not.toBeNull()
  })
})
