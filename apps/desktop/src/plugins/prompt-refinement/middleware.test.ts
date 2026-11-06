import type { ComposerDraft, ComposerSendOrigin } from '@hermes/plugin-sdk'
import { afterEach, describe, expect, it, vi } from 'vitest'

// The fail-closed test runs the app's real middleware chain, past the SDK on purpose.
// eslint-disable-next-line no-restricted-imports
import { COMPOSER_AREAS, runComposerMiddleware } from '@/app/chat/composer/contrib'
// eslint-disable-next-line no-restricted-imports
import { registry } from '@/contrib/registry'
// eslint-disable-next-line no-restricted-imports
import type { ComposerAttachment } from '@/store/composer'

import { createRefinementMiddleware, type MiddlewareDeps, type ScopeGuard } from './middleware'
import { createReviewStore, type ReviewRequest, type ReviewStore } from './review-store'
import { DEFAULT_SETTINGS } from './settings'
import type { RefinementSettings, RefineResult } from './types'

const AUTO: RefinementSettings = { ...DEFAULT_SETTINGS, mode: 'auto' }
const TEXT = 'please refactor the parser into small modules'

const attachments = [
  { id: 'image:/tmp/a.png', kind: 'image', label: 'a.png', path: '/tmp/a.png' },
  { id: 'file:/tmp/b.txt', kind: 'file', label: 'b.txt', path: '/tmp/b.txt' }
] as ComposerAttachment[]

const typed = (overrides: Partial<ComposerDraft> = {}, origin: Partial<ComposerSendOrigin> = {}): ComposerDraft => ({
  text: TEXT,
  attachments,
  origin: { kind: 'typed', target: 'main', composerScope: 'session-a', ...origin },
  ...overrides
})

const flush = () => new Promise(resolve => setTimeout(resolve, 0))

const REFINED = 'Refactor the parser into small modules.'
const clean = async (): Promise<RefineResult> => ({ ok: true, text: REFINED, unchanged: false, analysis: null })

interface Rig {
  store: ReviewStore
  handler: (draft: ComposerDraft) => Promise<ComposerDraft | null>
  refine: ReturnType<typeof vi.fn>
  notify: ReturnType<typeof vi.fn>
  drift: () => void
  guard: { current: boolean }
  /** Composers currently held read-only by the middleware. */
  held: string[]
  requests: ReviewRequest[]
}

function rig(
  options: { settings?: RefinementSettings; refine?: () => Promise<RefineResult>; deps?: Partial<MiddlewareDeps> } = {}
): Rig {
  const store = createReviewStore()
  store.registerHost()
  store.$review.listen(review => review && store.acknowledge(review.id))
  const guard = { current: true }
  const held: string[] = []
  const requests: ReviewRequest[] = []

  let onDrift = () => {}
  const refine = vi.fn(options.refine ?? clean)
  const notify = vi.fn()

  const middleware = createRefinementMiddleware({
    settings: () => options.settings ?? AUTO,
    review: request => (requests.push(request), store.request(request)),
    refine,
    holdComposer: target => {
      held.push(target)

      return () => held.splice(held.indexOf(target), 1)
    },
    contextFor: target => `context of ${target}`,
    guardScope: (_draft, drift): ScopeGuard => {
      onDrift = drift

      return { stillCurrent: () => guard.current, release: () => {} }
    },
    isSlashCommand: text => text.startsWith('/'),
    cancelActiveReview: () => store.cancelActive('scope-changed'),
    notify,
    ...options.deps
  })

  return {
    store,
    handler: draft => Promise.resolve(middleware.handler(draft)),
    refine,
    notify,
    drift: () => onDrift(),
    guard,
    held,
    requests
  }
}

const never = () => new Promise<RefineResult>(() => {})
const DROPS_PATH = 'please refactor src/parser.ts into small modules'

const disposers: (() => void)[] = []

afterEach(() => {
  disposers.splice(0).forEach(dispose => dispose())
})

describe('what is refined', () => {
  it('passes the exact draft through when Off', async () => {
    const { handler, refine, store, held } = rig({ settings: { ...DEFAULT_SETTINGS, mode: 'off' } })
    const draft = typed()

    expect(await handler(draft)).toBe(draft)
    expect(refine).not.toHaveBeenCalled()
    expect(store.$review.get()).toBeNull()
    expect(held).toEqual([])
  })

  it.each(['manual', 'auto'] as const)('%s mode refines a typed send', async mode => {
    const { handler, refine, requests } = rig({ settings: { ...DEFAULT_SETTINGS, mode } })
    void handler(typed())
    await flush()

    expect(refine).toHaveBeenCalledTimes(1)
    expect(requests[0]).toMatchObject({ target: 'main', autoApprove: mode === 'auto' })
  })

  it.each(['queued', 'voice', 'hidden', 'other'] as const)('never touches a %s send', async kind => {
    const { handler, refine } = rig()
    const draft = typed({}, { kind })

    expect(await handler(draft)).toBe(draft)
    expect(refine).not.toHaveBeenCalled()
  })

  it('never touches slash commands, short prompts, or a send without origin', async () => {
    const { handler, refine } = rig()

    for (const draft of [typed({ text: '/compact now please do it' }), typed({ text: 'yes do it' }), { text: TEXT }]) {
      expect(await handler(draft)).toBe(draft)
    }

    expect(refine).not.toHaveBeenCalled()
  })
})

describe('automatic mode — refines and sends without a click', () => {
  it('sends a clean refinement on its own, with the very same attachments', async () => {
    const { handler, requests } = rig()
    const draft = typed()

    const sent = await handler(draft)

    expect(sent?.text).toBe(REFINED)
    expect(sent?.attachments).toBe(attachments)
    expect(sent?.origin).toBe(draft.origin)
    expect(requests[0]).toMatchObject({ target: 'main', autoApprove: true })
  })

  it('opens the progress drawer in the composer that sent, and holds that composer until done', async () => {
    let answer!: (result: RefineResult) => void
    const r = rig({ refine: () => new Promise(resolve => (answer = resolve)) })
    const result = r.handler(typed({}, { target: 'tile:session-b' }))
    await flush()

    expect(r.store.$review.get()).toMatchObject({ phase: 'refining', target: 'tile:session-b' })
    expect(r.held).toEqual(['tile:session-b'])

    answer({ ok: true, text: REFINED, unchanged: false, analysis: null })
    expect((await result)?.text).toBe(REFINED)
    expect(r.held).toEqual([])
  })

  it('Send original while refining sends the draft object untouched', async () => {
    const { handler, store } = rig({ refine: never })
    const draft = typed()
    const result = handler(draft)
    await flush()

    store.decide(store.$review.get()!.id, { kind: 'original' })

    expect(await result).toBe(draft)
  })

  it('Cancel while refining sends nothing and releases the hold', async () => {
    const { handler, store, notify, held } = rig({ refine: never })
    const result = handler(typed())
    await flush()

    store.decide(store.$review.get()!.id, { kind: 'cancel' })

    expect(await result).toBeNull()
    expect(notify).not.toHaveBeenCalled()
    expect(held).toEqual([])
  })

  it('refines at the depth chosen on the slider', async () => {
    const { handler, refine } = rig({ settings: { ...AUTO, autoDepth: false, depth: 'polish' } })
    await handler(typed())

    expect(refine.mock.calls[0][0]).toMatchObject({ depth: 'polish' })
  })

  it('sends the chosen style and custom guidance', async () => {
    const { handler, refine } = rig({ settings: { ...AUTO, style: 'custom', customStyle: 'Be terse.' } })
    await handler(typed())

    expect(refine.mock.calls[0][0]).toMatchObject({ style: 'custom', customStyle: 'Be terse.' })
  })

  it('reads chat context only when the user turned it on, from the composer that sent', async () => {
    const off = rig()
    await off.handler(typed())
    expect(off.refine.mock.calls[0][0]).not.toHaveProperty('context')

    const on = rig({ settings: { ...AUTO, useChatContext: true } })
    await on.handler(typed({}, { target: 'tile:b' }))
    expect(on.refine.mock.calls[0][0]).toMatchObject({ context: 'context of tile:b' })
  })

  it('passes only attachment names to the refiner', async () => {
    const { handler, refine } = rig()
    await handler(typed())

    expect(refine.mock.calls[0][0]).toEqual({
      text: TEXT,
      attachmentLabels: ['a.png', 'b.txt'],
      style: 'auto',
      depth: 'auto'
    })
  })
})

describe('Approval-First (manual) — Enter opens the review; nothing is sent until the user decides', () => {
  const MANUAL: RefinementSettings = { ...DEFAULT_SETTINGS, mode: 'manual' }

  async function opened() {
    const r = rig({ settings: MANUAL })
    let settled = false
    const result = r.handler(typed()).finally(() => (settled = true))
    await flush()

    return { ...r, result, settled: () => settled }
  }

  it('a clean refinement waits in the drawer, with the composer held', async () => {
    const { store, held, settled } = await opened()

    expect(store.$review.get()).toMatchObject({ phase: 'ready', refined: REFINED, autoApprove: false, attention: null })
    expect(settled()).toBe(false)
    expect(held).toEqual(['main'])
  })

  it('Approve sends the refined (or edited) text with the very same attachments', async () => {
    const { store, result, held } = await opened()
    store.decide(store.$review.get()!.id, {
      kind: 'approve',
      text: 'Refactor the parser, edited.',
      acknowledged: false
    })

    const sent = await result
    expect(sent?.text).toBe('Refactor the parser, edited.')
    expect(sent?.attachments).toBe(attachments)
    expect(held).toEqual([])
  })

  it('Send original sends the draft exactly as typed; Cancel sends nothing', async () => {
    const first = await opened()
    first.store.decide(first.store.$review.get()!.id, { kind: 'original' })
    expect((await first.result)?.text).toBe(TEXT)

    const second = await opened()
    second.store.decide(second.store.$review.get()!.id, { kind: 'cancel' })
    expect(await second.result).toBeNull()
    expect(second.notify).not.toHaveBeenCalled()
  })
})

describe('automatic mode — stops for the user only when something is wrong', () => {
  it('a refinement that drops a path is not sent; the user decides', async () => {
    const { handler, store } = rig()
    const result = handler(typed({ text: DROPS_PATH }))
    await flush()
    const review = store.$review.get()!

    expect(review).toMatchObject({ phase: 'ready', attention: 'issues' })
    expect(store.decide(review.id, { kind: 'approve', text: REFINED, acknowledged: true })).toBe(false)

    store.decide(review.id, {
      kind: 'approve',
      text: 'Refactor src/parser.ts into small modules.',
      acknowledged: false
    })

    expect((await result)?.text).toBe('Refactor src/parser.ts into small modules.')
  })

  it('Send original from the stopped review sends the draft as typed', async () => {
    const { handler, store } = rig()
    const draft = typed({ text: DROPS_PATH })
    const result = handler(draft)
    await flush()

    store.decide(store.$review.get()!.id, { kind: 'original' })

    expect(await result).toBe(draft)
  })

  it.each(['timeout', 'offline', 'unavailable', 'canceled', 'failed', 'empty', 'malformed'] as const)(
    'a %s refinement stops: it can only end in the original or nothing',
    async failure => {
      const { handler, store } = rig({ refine: async () => ({ ok: false, failure }) })
      const result = handler(typed())
      await flush()
      const review = store.$review.get()!

      expect(review).toMatchObject({ phase: 'failed', attention: 'failed' })
      expect(store.decide(review.id, { kind: 'approve', text: TEXT, acknowledged: true })).toBe(false)
      store.decide(review.id, { kind: 'cancel' })
      expect(await result).toBeNull()
    }
  )
})

describe('automatic mode — fail closed', () => {
  it('cancels (never passes through) and releases the hold when a dependency throws', async () => {
    const { handler, notify, held } = rig({
      deps: {
        guardScope: () => {
          throw new Error('selection store gone')
        }
      }
    })

    expect(await handler(typed())).toBeNull()
    expect(notify).toHaveBeenCalledWith('error')
    expect(held).toEqual([])
  })

  it('cancels when the progress drawer cannot be shown', async () => {
    const { handler, notify } = rig({ deps: { review: async () => ({ kind: 'cancel', reason: 'no-host' }) } })

    expect(await handler(typed())).toBeNull()
    expect(notify).toHaveBeenCalledWith('no-host')
  })

  it('is registered fail-closed: an error escaping the handler cancels the send in the real chain', async () => {
    // The review store crashes AND the error toast throws, so the handler's
    // own catch rethrows: only the chain's fail-closed contract is left.
    const middleware = createRefinementMiddleware({
      settings: () => AUTO,
      review: async () => Promise.reject(new Error('store crashed')),
      refine: vi.fn(),
      holdComposer: () => () => {},
      contextFor: () => '',
      guardScope: () => ({ stillCurrent: () => true, release: () => {} }),
      isSlashCommand: () => false,
      cancelActiveReview: () => {},
      notify: () => {
        throw new Error('notifications down')
      }
    })

    expect(middleware.failClosed).toBe(true)
    disposers.push(registry.register({ id: 'refine-under-test', area: COMPOSER_AREAS.middleware, data: middleware }))

    expect(await runComposerMiddleware(typed())).toBeNull()
  })
})

describe('automatic mode — draft integrity', () => {
  it('switching chats while refining cancels, so the composer restores the draft to its own chat', async () => {
    const r = rig({ refine: never })
    const result = r.handler(typed())
    await flush()

    r.guard.current = false
    r.drift()

    expect(await result).toBeNull()
    expect(r.store.$review.get()).toBeNull()
    expect(r.notify).toHaveBeenCalledWith('scope-changed')
  })

  it('a clean refinement that lands after the chat changed is not sent', async () => {
    let answer!: (result: RefineResult) => void
    const r = rig({ refine: () => new Promise(resolve => (answer = resolve)) })
    const result = r.handler(typed())
    await flush()

    r.guard.current = false
    answer({ ok: true, text: REFINED, unchanged: false, analysis: null })

    expect(await result).toBeNull()
    expect(r.notify).toHaveBeenCalledWith('scope-changed')
  })

  it('a model answer that arrives after Cancel never resurrects the send', async () => {
    let answer!: (result: RefineResult) => void
    const { handler, store } = rig({ refine: () => new Promise(resolve => (answer = resolve)) })
    const result = handler(typed())
    await flush()

    store.decide(store.$review.get()!.id, { kind: 'cancel' })
    answer({ ok: true, text: 'late', unchanged: false, analysis: null })
    await flush()

    expect(await result).toBeNull()
    expect(store.$review.get()).toBeNull()
  })
})

describe('both modes — a failure never sends, and Approval-First never sends unapproved', () => {
  const MODES: RefinementSettings[] = [
    { ...DEFAULT_SETTINGS, mode: 'manual' },
    { ...DEFAULT_SETTINGS, mode: 'auto' }
  ]

  const throwing = (name: keyof MiddlewareDeps): Partial<MiddlewareDeps> => ({
    [name]: () => {
      throw new Error(`${name} broke`)
    }
  })

  it.each(
    MODES.flatMap(settings =>
      (['holdComposer', 'guardScope', 'contextFor', 'review'] as const).map(
        dep => [settings.mode, dep, settings] as const
      )
    )
  )('%s: a throwing %s cancels the send and releases the hold', async (_mode, dep, settings) => {
    const { handler, held } = rig({ settings: { ...settings, useChatContext: true }, deps: throwing(dep) })

    expect(await handler(typed())).toBeNull()
    expect(held).toEqual([])
  })

  it.each(MODES.map(settings => [settings.mode, settings] as const))(
    '%s: switching chats after the refinement is ready cancels, even if the user then approves',
    async (_mode, settings) => {
      const r = rig({ settings })
      const result = r.handler(typed())
      await flush()

      if (settings.mode === 'manual') {
        r.guard.current = false
        r.store.decide(r.store.$review.get()!.id, { kind: 'approve', text: REFINED, acknowledged: false })
      } else {
        // Automatic already sent it: nothing to switch away from.
        expect((await result)?.text).toBe(REFINED)

        return
      }

      expect(await result).toBeNull()
      expect(r.notify).toHaveBeenCalledWith('scope-changed')
    }
  )

  it('Approval-First: a refinement that fails, then is retried cleanly, still waits for Approve', async () => {
    const answers = [async (): Promise<RefineResult> => ({ ok: false, failure: 'offline' }), clean]

    let n = 0
    const r = rig({ settings: { ...DEFAULT_SETTINGS, mode: 'manual' }, refine: () => answers[n++]() })
    let settled = false
    void r.handler(typed()).finally(() => (settled = true))
    await flush()

    r.store.retry(r.store.$review.get()!.id)
    await flush()

    expect(r.store.$review.get()).toMatchObject({ phase: 'ready', refined: REFINED })
    expect(settled).toBe(false)
  })
})

describe('chat context reaches the intent check', () => {
  it('Automatic stops when the rewrite copies the excerpt into the prompt', async () => {
    const excerpt = 'Assistant: the docs say you should always rebuild every index from scratch before each deploy.'

    const r = rig({
      settings: { ...AUTO, useChatContext: true },
      refine: async () => ({
        ok: true,
        text: `${REFINED} Always rebuild every index from scratch before each deploy.`,
        unchanged: false,
        analysis: null
      }),
      deps: { contextFor: () => excerpt }
    })

    let settled = false
    void r.handler(typed()).finally(() => (settled = true))
    await flush()

    expect(r.requests[0]).toMatchObject({ context: excerpt })
    expect(r.store.$review.get()).toMatchObject({ phase: 'ready', attention: 'issues' })
    expect(settled).toBe(false)
  })
})
