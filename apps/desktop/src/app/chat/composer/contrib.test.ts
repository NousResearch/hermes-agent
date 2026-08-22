import { afterEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import type { ComposerAttachment } from '@/store/composer'

import { composerAttachmentsAreAuthorized } from './attachment-relay'
import { COMPOSER_AREAS, type ComposerDraft, type ComposerMiddleware, runComposerMiddleware } from './contrib'

const disposers: Array<() => void> = []

function addMiddleware(id: string, handler: ComposerMiddleware['handler'], order?: number) {
  disposers.push(
    registry.register({ id, area: COMPOSER_AREAS.middleware, order, data: { handler } satisfies ComposerMiddleware })
  )
}

const hostAttachment = (): ComposerAttachment => ({
  id: 'host-file',
  occurrenceId: 'host-occurrence',
  kind: 'file',
  label: 'host.txt',
  path: 'C:/trusted/host.txt'
})

afterEach(() => {
  disposers.splice(0).forEach(d => d())
})

describe('runComposerMiddleware', () => {
  it('passes the draft through untouched when nothing is registered', async () => {
    const draft = { text: 'hello' }

    expect(await runComposerMiddleware(draft)).toBe(draft)
  })

  it('chains rewrites in registry order', async () => {
    addMiddleware('b', d => ({ ...d, text: `${d.text}b` }), 20)
    addMiddleware('a', d => ({ ...d, text: `${d.text}a` }), 10)

    expect(await runComposerMiddleware({ text: 'x' })).toEqual({ text: 'xab' })
  })

  it('cancels the send when a handler returns null', async () => {
    addMiddleware('gate', () => null)
    addMiddleware('later', d => ({ ...d, text: 'never' }), 99)

    expect(await runComposerMiddleware({ text: 'x' })).toBeNull()
  })

  it('treats a throwing handler as pass-through', async () => {
    addMiddleware('boom', () => {
      throw new Error('broken plugin')
    })
    addMiddleware('after', d => ({ ...d, text: `${d.text}!` }), 99)

    expect(await runComposerMiddleware({ text: 'x' })).toEqual({ text: 'x!' })
  })

  it('never exposes authoritative state and discards irreversible throwing attempts', async () => {
    const first = hostAttachment()
    const second: ComposerAttachment = {
      id: 'host-image',
      occurrenceId: 'host-image-occurrence',
      kind: 'image',
      label: 'host.png',
      detail: 'image metadata',
      path: 'C:/trusted/host.png'
    }
    const attachments = [first, second]
    const draft = { text: 'exact text', attachments }
    let firstAttempt: { attachments: ComposerAttachment[]; draft: object; first: ComposerAttachment } | undefined
    let observed: unknown
    const later = vi.fn((next: typeof draft) => {
      const nextAttachments = next.attachments
      observed = {
        authoritativeArrayExposed: nextAttachments === attachments,
        authoritativeDraftExposed: next === draft,
        authoritativeFirstExposed: nextAttachments[0] === first,
        authorized: composerAttachmentsAreAuthorized(nextAttachments),
        firstAttemptArrayReused: nextAttachments === firstAttempt?.attachments,
        firstAttemptDraftReused: next === firstAttempt?.draft,
        firstAttemptObjectReused: nextAttachments[0] === firstAttempt?.first,
        attachments: nextAttachments.map(item => ({ ...item })),
        text: next.text
      }
      return next
    })

    addMiddleware('mutate-then-throw', current => {
      firstAttempt = {
        attachments: current.attachments!,
        draft: current,
        first: current.attachments![0]!
      }
      current.text = 'corrupted'
      current.attachments!.reverse()
      current.attachments!.push({ ...first, id: 'fabricated' })
      Object.defineProperty(current.attachments![1]!, 'label', {
        configurable: false,
        enumerable: true,
        value: 'locked-mutation.txt',
        writable: false
      })
      Object.freeze(current.attachments![1]!)
      Object.setPrototypeOf(current.attachments![0]!, { poisoned: true })
      Object.preventExtensions(current.attachments![0]!)
      Object.seal(current.attachments!)
      ;(current as typeof draft & { injected?: string }).injected = 'bad'
      Object.setPrototypeOf(current, { poisoned: true })
      Object.freeze(current)
      throw new Error('restore this attempt')
    })
    addMiddleware('later', later as ComposerMiddleware['handler'])

    const result = await runComposerMiddleware(draft)

    expect(later).toHaveBeenCalledOnce()
    expect(observed).toEqual({
      authoritativeArrayExposed: false,
      authoritativeDraftExposed: false,
      authoritativeFirstExposed: false,
      authorized: true,
      firstAttemptArrayReused: false,
      firstAttemptDraftReused: false,
      firstAttemptObjectReused: false,
      attachments: [hostAttachment(), second],
      text: 'exact text'
    })
    expect(result).toEqual(draft)
    expect(draft).toEqual({ text: 'exact text', attachments: [first, second] })
    expect(Object.isExtensible(draft)).toBe(true)
    expect(Object.isExtensible(attachments)).toBe(true)
    expect(Object.isExtensible(first)).toBe(true)
    expect(Object.getPrototypeOf(draft)).toBe(Object.prototype)
    expect(Object.getPrototypeOf(first)).toBe(Object.prototype)
  })

  it('adopts valid attempt-local mutation into a fresh later attempt', async () => {
    const original = hostAttachment()
    const draft = { text: 'original', attachments: [original] }
    let firstAttempt: ComposerDraft | undefined
    let laterObserved: unknown

    addMiddleware('mutate-valid-view', current => {
      firstAttempt = current
      current.text = 'adopted'
      return current
    })
    addMiddleware('observe-adopted-copy', current => {
      laterObserved = {
        authorized: composerAttachmentsAreAuthorized(current.attachments ?? []),
        draftReused: current === firstAttempt,
        originalArrayExposed: current.attachments === draft.attachments,
        originalObjectExposed: current.attachments?.[0] === original,
        text: current.text
      }
      return current
    })

    const result = await runComposerMiddleware(draft)

    expect(laterObserved).toEqual({
      authorized: true,
      draftReused: false,
      originalArrayExposed: false,
      originalObjectExposed: false,
      text: 'adopted'
    })
    expect(result).toEqual({ text: 'adopted', attachments: [original] })
    expect(draft.text).toBe('original')
  })

  it('supports async handlers', async () => {
    addMiddleware('async', async d => ({ ...d, text: d.text.toUpperCase() }))

    expect(await runComposerMiddleware({ text: 'quiet' })).toEqual({ text: 'QUIET' })
  })

  it('supports additive pass dispositions without changing legacy draft results', async () => {
    addMiddleware('pass', d => ({ disposition: 'pass', draft: { ...d, text: `${d.text}!` } }))
    addMiddleware('legacy', d => ({ ...d, text: `${d.text}?` }))

    expect(await runComposerMiddleware({ text: 'hello' })).toEqual({ text: 'hello!?' })
  })

  it.each([
    [
      'text accessor',
      (_draft: ComposerDraft, reads: () => void) => {
        const nested = { attachments: [] as ComposerAttachment[] } as Record<string, unknown>
        Object.defineProperty(nested, 'text', {
          enumerable: true,
          get: () => {
            reads()
            return 'getter text'
          }
        })
        return nested
      }
    ],
    ['inherited text', () => Object.assign(Object.create({ text: 'inherited' }), { attachments: [] })],
    ['symbol key', () => ({ text: 'symbol', [Symbol('extra')]: true })],
    ['extra key', () => ({ text: 'extra', extra: true })],
    ['missing text', () => ({ attachments: [] })],
    ['wrong text type', () => ({ text: 42 })],
    [
      'attachments accessor',
      (_draft: ComposerDraft, reads: () => void) => {
        const nested = { text: 'getter attachments' } as Record<string, unknown>
        Object.defineProperty(nested, 'attachments', {
          enumerable: true,
          get: () => {
            reads()
            return []
          }
        })
        return nested
      }
    ],
    ['non-array attachments', () => ({ text: 'object attachments', attachments: {} })],
    [
      'sparse attachments',
      () => {
        const attachments = new Array<ComposerAttachment>(1)
        return { text: 'sparse', attachments }
      }
    ],
    [
      'exotic array prototype',
      (draft: ComposerDraft) => {
        const attachments = [...(draft.attachments ?? [])]
        Object.setPrototypeOf(attachments, { poisoned: true })
        return { text: 'exotic', attachments }
      }
    ],
    [
      'array extra key',
      (draft: ComposerDraft) => {
        const attachments = [...(draft.attachments ?? [])] as ComposerAttachment[] & { extra?: boolean }
        attachments.extra = true
        return { text: 'array extra', attachments }
      }
    ],
    [
      'hostile attachment getter',
      (_draft: ComposerDraft, reads: () => void) => {
        const item = { id: 'hostile', kind: 'file' } as Record<string, unknown>
        Object.defineProperty(item, 'label', {
          enumerable: true,
          get: () => {
            reads()
            return 'hostile.txt'
          }
        })
        return { text: 'hostile attachment', attachments: [item] }
      }
    ],
    [
      'inherited attachment semantics',
      (draft: ComposerDraft) => ({
        text: 'inherited attachment',
        attachments: [Object.create(draft.attachments?.[0] ?? null)]
      })
    ],
    [
      'attachment symbol key',
      (draft: ComposerDraft) => ({
        text: 'attachment symbol',
        attachments: [{ ...(draft.attachments?.[0] ?? hostAttachment()), [Symbol('extra')]: true }]
      })
    ],
    [
      'attachment extra key',
      (draft: ComposerDraft) => ({
        text: 'attachment extra',
        attachments: [{ ...(draft.attachments?.[0] ?? hostAttachment()), extra: true }]
      })
    ],
    ['missing attachment id', () => ({ text: 'missing id', attachments: [{ kind: 'file', label: 'x.txt' }] })],
    ['wrong attachment kind', () => ({ text: 'wrong kind', attachments: [{ id: 'x', kind: 7, label: 'x.txt' }] })],
    [
      'copied arbitrary path attachment',
      (draft: ComposerDraft) => ({ text: 'copied path', attachments: [{ ...draft.attachments?.[0] }] })
    ],
    [
      'changed authoritative view path',
      (draft: ComposerDraft) => {
        const item = draft.attachments![0]!
        item.path = 'C:/arbitrary/replaced.txt'
        return { text: 'changed path', attachments: [item] }
      }
    ]
  ] as Array<[
    string,
    (draft: ComposerDraft, reads: () => void) => unknown
  ]>)('discards invalid nested pass draft: %s', async (_name, build) => {
    const reads = vi.fn()
    const later = vi.fn((draft: ComposerDraft) => ({ ...draft, text: `${draft.text} safe` }))
    addMiddleware('invalid-pass', draft => ({
      disposition: 'pass',
      draft: build(draft, reads) as ComposerDraft
    }))
    addMiddleware('later', later)

    const result = await runComposerMiddleware({ text: 'original', attachments: [hostAttachment()] })

    expect(result).toEqual({ text: 'original safe', attachments: [hostAttachment()] })
    expect(reads).not.toHaveBeenCalled()
    expect(later).toHaveBeenCalledOnce()
  })

  it('adopts one closed valid nested pass draft and preserves exact-view relay provenance', async () => {
    let authorized = false
    addMiddleware('valid-pass', draft => ({
      disposition: 'pass',
      draft: { text: 'valid pass', attachments: [...(draft.attachments ?? [])] }
    }))
    addMiddleware('later', draft => {
      authorized = composerAttachmentsAreAuthorized(draft.attachments ?? [])
      return draft
    })

    const result = await runComposerMiddleware({ text: 'original', attachments: [hostAttachment()] })

    expect(result).toEqual({ text: 'valid pass', attachments: [hostAttachment()] })
    expect(authorized).toBe(true)
  })

  it('adopts a closed null-prototype pass draft with a safe pathless replacement attachment', async () => {
    let authorized = true
    addMiddleware('valid-null-pass', () => {
      const attachment = Object.assign(Object.create(null), {
        id: 'url-1',
        kind: 'url',
        label: 'Reference',
        refText: 'https://example.test/reference'
      }) as ComposerAttachment
      return {
        disposition: 'pass',
        draft: Object.assign(Object.create(null), { text: 'null prototype', attachments: [attachment] })
      }
    })
    addMiddleware('observe-pathless-replacement', draft => {
      authorized = composerAttachmentsAreAuthorized(draft.attachments ?? [])
      return draft
    })

    const result = await runComposerMiddleware({ text: 'original', attachments: [hostAttachment()] })

    expect(result).toEqual({
      text: 'null prototype',
      attachments: [{ id: 'url-1', kind: 'url', label: 'Reference', refText: 'https://example.test/reference' }]
    })
    expect(authorized).toBe(false)
  })

  it('authorizes only unchanged attachments originating at the initial host boundary', async () => {
    const original = hostAttachment()
    const fabricated = { ...original, path: 'C:/arbitrary/forged.txt' }
    const inherited = Object.assign(Object.create(original), { id: original.id }) as ComposerAttachment
    const seen: boolean[][] = []

    addMiddleware('replace-with-fabricated', draft => ({ ...draft, attachments: [fabricated, inherited] }))
    addMiddleware('observe-forgeries', draft => {
      seen.push([
        composerAttachmentsAreAuthorized([original]),
        composerAttachmentsAreAuthorized([fabricated]),
        composerAttachmentsAreAuthorized([inherited])
      ])
      return { ...draft, attachments: [original] }
    })
    addMiddleware('observe-restored-host', draft => {
      seen.push([composerAttachmentsAreAuthorized(draft.attachments ?? [])])
      return draft
    })

    const result = await runComposerMiddleware({ text: 'host', attachments: [original] })

    expect(seen).toEqual([[false, false, false], [true]])
    expect(result).toMatchObject({ attachments: [original] })
  })

  it('keeps host provenance private from copied, deserialized, derived, fabricated, and changed views', async () => {
    const statuses: boolean[][] = []
    addMiddleware('probe-private-provenance', draft => {
      const exact = draft.attachments![0]!
      const copied = { ...exact }
      const deserialized = JSON.parse(JSON.stringify(exact)) as ComposerAttachment
      const derived = Object.assign(Object.create(exact), { id: exact.id }) as ComposerAttachment
      const fabricated = { id: 'fabricated', kind: 'file' as const, label: 'fake.txt', path: 'C:/arbitrary/fake.txt' }
      statuses.push([
        composerAttachmentsAreAuthorized([exact]),
        composerAttachmentsAreAuthorized([copied]),
        composerAttachmentsAreAuthorized([deserialized]),
        composerAttachmentsAreAuthorized([derived]),
        composerAttachmentsAreAuthorized([fabricated])
      ])
      exact.label = 'changed.txt'
      statuses.push([composerAttachmentsAreAuthorized([exact])])
      throw new Error('discard changed attempt')
    })
    addMiddleware('fresh-attempt', draft => {
      statuses.push([composerAttachmentsAreAuthorized(draft.attachments ?? [])])
      return draft
    })

    const result = await runComposerMiddleware({ text: 'original', attachments: [hostAttachment()] })

    expect(statuses).toEqual([[true, false, false, false, false], [false], [true]])
    expect(result).toEqual({ text: 'original', attachments: [hostAttachment()] })
  })

  it('allows attachment removal without minting authority for replacement objects', async () => {
    const original = hostAttachment()
    const copied = { ...original }
    const seen: boolean[] = []

    addMiddleware('remove', draft => ({ ...draft, attachments: [] }))
    addMiddleware('observe-removal', draft => {
      seen.push(composerAttachmentsAreAuthorized(draft.attachments ?? []))
      seen.push(composerAttachmentsAreAuthorized([copied]))
      return { ...draft, attachments: [copied] }
    })
    addMiddleware('observe-copy', draft => {
      seen.push(composerAttachmentsAreAuthorized(draft.attachments ?? []))
      return draft
    })

    await runComposerMiddleware({ text: 'host', attachments: [original] })

    expect(seen).toEqual([true, false, true])
  })

  it('returns a rejected draft and prevents later middleware', async () => {
    const later = vi.fn((d: { text: string }) => d)
    addMiddleware('reject', () => ({ disposition: 'reject', reason: 'not accepted' }))
    addMiddleware('later', later, 99)

    expect(await runComposerMiddleware({ text: 'keep exactly' })).toEqual({
      disposition: 'reject',
      draft: { text: 'keep exactly' },
      reason: 'not accepted'
    })
    expect(later).not.toHaveBeenCalled()
  })

  it('returns consume and prevents later middleware after a successful side-effect', async () => {
    const later = vi.fn((d: { text: string }) => d)
    addMiddleware('consume', () => ({ disposition: 'consume', receipt: { state: 'durably_accepted' } }))
    addMiddleware('later', later, 99)

    expect(await runComposerMiddleware({ text: 'send once' })).toEqual({
      disposition: 'consume',
      receipt: { state: 'durably_accepted' }
    })
    expect(later).not.toHaveBeenCalled()
  })

  it('isolates exceptions as pass-through and rejects inherited disposition values', async () => {
    addMiddleware('boom', () => {
      throw new Error('plugin failed')
    })
    addMiddleware(
      'poisoned',
      () => Object.create({ disposition: 'consume' }) as unknown as ReturnType<ComposerMiddleware['handler']>
    )
    addMiddleware('after', d => ({ ...d, text: `${d.text} safe` }))

    expect(await runComposerMiddleware({ text: 'still' })).toEqual({ text: 'still safe' })
  })

  it('recognizes dispositions only from closed own data-property shapes without invoking getters', async () => {
    let getterCalls = 0
    const hostileGetter = Object.create(null) as Record<string, unknown>
    Object.defineProperty(hostileGetter, 'disposition', {
      enumerable: true,
      get: () => {
        getterCalls += 1
        return 'consume'
      }
    })
    const polluted = Object.assign(Object.create({ polluted: true }), { disposition: 'consume' })
    const later = vi.fn((draft: { text: string }) => ({ ...draft, text: `${draft.text} safe` }))

    addMiddleware('getter', () => hostileGetter as unknown as ReturnType<ComposerMiddleware['handler']>)
    addMiddleware('extra', () => ({ disposition: 'consume', extra: true }) as ReturnType<ComposerMiddleware['handler']>)
    addMiddleware('polluted', () => polluted as ReturnType<ComposerMiddleware['handler']>)
    addMiddleware('later', later)

    expect(await runComposerMiddleware({ text: 'still' })).toEqual({ text: 'still safe' })
    expect(getterCalls).toBe(0)
    expect(later).toHaveBeenCalledOnce()
  })

  it('preserves a legacy replacement draft with an inherited disposition property', async () => {
    const replacement = Object.assign(Object.create({ disposition: 'consume' }), { text: 'legacy replacement' })
    addMiddleware('legacy-inherited', () => replacement)

    const result = await runComposerMiddleware({ text: 'original' })

    expect(result).toEqual({ text: 'legacy replacement' })
    expect((result as { text: string }).text).toBe('legacy replacement')
  })
})
