import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'

import {
  clearTranscriptTails,
  dropTranscriptTail,
  dropTranscriptTailEverywhere,
  loadTranscriptTail,
  saveTranscriptTail,
  type TranscriptTailAuthority
} from './transcript-tail-cache'

const msg = (id: string, chars = 10): ChatMessage =>
  ({ id, parts: [{ text: 'x'.repeat(chars), type: 'text' }], role: 'assistant' }) as never

const authority = (overrides: Partial<TranscriptTailAuthority> = {}): TranscriptTailAuthority => ({
  connectionId: 'conn-1',
  displayRevision: 7,
  lineageRootId: 'root-1',
  profile: 'coder',
  resolvedTipId: 'tip-2',
  ...overrides
})

const loadedMessages = (storedSessionId: string, proof = authority()) =>
  loadTranscriptTail(storedSessionId, proof)?.messages ?? null

function storageSnapshot(): Array<[string, string | null]> {
  return Array.from({ length: window.localStorage.length }, (_, index) => window.localStorage.key(index))
    .filter((key): key is string => key !== null)
    .sort()
    .map(key => [key, window.localStorage.getItem(key)])
}

function v3NamespaceSnapshot(): Array<[string, string | null]> {
  return storageSnapshot().filter(
    ([key]) => key === 'hermes.transcript-tail.v3-index' || key.startsWith('hermes.transcript-tail.v3:')
  )
}

function onlyV3EntryKey(): string {
  const keys = Array.from({ length: window.localStorage.length }, (_, index) => window.localStorage.key(index))
  const key = keys.find(candidate => candidate?.startsWith('hermes.transcript-tail.v3:'))

  if (!key) {
    throw new Error('expected one v3 transcript-tail entry')
  }

  return key
}

function v3EntryKeys(): string[] {
  return Array.from({ length: window.localStorage.length }, (_, index) => window.localStorage.key(index)).filter(
    (key): key is string => key?.startsWith('hermes.transcript-tail.v3:') === true
  )
}

function entryKeyForStoredId(storedSessionId: string): string | undefined {
  return v3EntryKeys().find(key => {
    try {
      return JSON.parse(window.localStorage.getItem(key)!).storedSessionId === storedSessionId
    } catch {
      return false
    }
  })
}

function storagePropertyOwner(property: keyof Storage): Storage {
  let owner: object | null = window.localStorage

  while (owner && !Object.prototype.hasOwnProperty.call(owner, property)) {
    owner = Object.getPrototypeOf(owner) as object | null
  }

  if (!owner) {
    throw new Error(`localStorage property ${String(property)} is unavailable`)
  }

  return owner as Storage
}

beforeEach(() => {
  window.localStorage.clear()
})

afterEach(() => {
  window.localStorage.clear()
  vi.restoreAllMocks()
})

describe('transcript tail cache v3 authority', () => {
  it('round-trips a tail only with the same complete authority proof', () => {
    const proof = authority()

    saveTranscriptTail('root-1', [msg('1')], proof)

    expect(loadTranscriptTail('root-1', proof)?.messages[0].id).toBe('1')
    expect(loadTranscriptTail('root-other', proof)).toBeNull()
  })

  it.each([
    ['connectionId', 'conn-other'],
    ['profile', 'reviewer'],
    ['lineageRootId', 'root-other'],
    ['resolvedTipId', 'tip-other'],
    ['displayRevision', 8],
    ['storedSessionId', 'root-other']
  ] as const)('rejects an entry whose parsed %s does not match the requested proof', (field, mismatch) => {
    const proof = authority()
    saveTranscriptTail('root-1', [msg('1')], proof)

    const key = onlyV3EntryKey()
    const parsed = JSON.parse(window.localStorage.getItem(key)!)
    parsed[field] = mismatch
    window.localStorage.setItem(key, JSON.stringify(parsed))

    expect(loadTranscriptTail('root-1', proof)).toBeNull()
  })

  it('normalizes whitespace and the default profile without merging distinct scopes', () => {
    saveTranscriptTail('  root-1  ', [msg('normalized')], {
      connectionId: '  conn-1  ',
      displayRevision: 7,
      lineageRootId: '  root-1  ',
      profile: '   ',
      resolvedTipId: '  tip-2  '
    })

    expect(
      loadTranscriptTail('root-1', {
        connectionId: 'conn-1',
        displayRevision: 7,
        lineageRootId: 'root-1',
        profile: 'default',
        resolvedTipId: 'tip-2'
      })?.messages[0].id
    ).toBe('normalized')
    expect(loadedMessages('root-1', authority({ connectionId: '', profile: 'default' }))).toBeNull()
    expect(loadedMessages('root-1', authority({ connectionId: 'conn-1', profile: 'coder' }))).toBeNull()
  })

  it.each([
    ['lineageRootId', ''],
    ['resolvedTipId', ''],
    ['displayRevision', Number.NaN],
    ['displayRevision', Number.POSITIVE_INFINITY],
    ['displayRevision', -1],
    ['displayRevision', 1.5],
    ['displayRevision', '7']
  ] as const)('invalid %s=%s makes save and load storage-pure', (field, value) => {
    window.localStorage.setItem('hermes.transcript-tail.v3-index', JSON.stringify(['sentinel']))
    window.localStorage.setItem('hermes.transcript-tail.v3:sentinel', 'sentinel-v3')
    window.localStorage.setItem('hermes.transcript-tail.v1:legacy', 'sentinel-v1')
    window.localStorage.setItem('hermes.transcript-tail.v2:legacy', 'sentinel-v2')
    const before = storageSnapshot()
    const invalid = { ...authority(), [field]: value } as TranscriptTailAuthority

    saveTranscriptTail('root-1', [msg('1')], invalid)
    expect(loadTranscriptTail('root-1', invalid)).toBeNull()
    expect(storageSnapshot()).toEqual(before)
  })

  it('purges only v1/v2 cache roots on a fresh import and never returns them', async () => {
    const stale = JSON.stringify({ messages: [msg('old')], savedAt: Date.now() })

    window.localStorage.setItem('hermes.transcript-tail.v1:root-1', stale)
    window.localStorage.setItem('hermes.transcript-tail.v1-index', JSON.stringify(['root-1']))
    window.localStorage.setItem('hermes.transcript-tail.v2:root-1', stale)
    window.localStorage.setItem('hermes.transcript-tail.v2-index', JSON.stringify(['root-1']))
    window.localStorage.setItem('hermes.transcript-tail.v20:unrelated', 'keep-me')
    window.localStorage.setItem('unrelated.application.key', 'keep-me-too')
    vi.resetModules()

    try {
      const fresh = await import('./transcript-tail-cache')

      expect(fresh.loadTranscriptTail('root-1', authority())).toBeNull()
      expect(window.localStorage.getItem('hermes.transcript-tail.v1:root-1')).toBeNull()
      expect(window.localStorage.getItem('hermes.transcript-tail.v1-index')).toBeNull()
      expect(window.localStorage.getItem('hermes.transcript-tail.v2:root-1')).toBeNull()
      expect(window.localStorage.getItem('hermes.transcript-tail.v2-index')).toBeNull()
      expect(window.localStorage.getItem('hermes.transcript-tail.v20:unrelated')).toBe('keep-me')
      expect(window.localStorage.getItem('unrelated.application.key')).toBe('keep-me-too')
    } finally {
      vi.resetModules()
    }
  })

  it.each([
    [
      'missing',
      (entry: Record<string, unknown>): void => {
        delete entry.savedAt
      }
    ],
    [
      'null',
      (entry: Record<string, unknown>): void => {
        entry.savedAt = null
      }
    ],
    [
      'string',
      (entry: Record<string, unknown>): void => {
        entry.savedAt = '1234'
      }
    ],
    [
      'negative',
      (entry: Record<string, unknown>): void => {
        entry.savedAt = -1
      }
    ],
    [
      'non-finite serialized as null',
      (entry: Record<string, unknown>): void => {
        entry.savedAt = Number.POSITIVE_INFINITY
      }
    ]
  ] as const)('direct load evicts an entry with %s savedAt and preserves unrelated entries', (_name, corrupt) => {
    const targetProof = authority()
    const unrelatedProof = authority({ lineageRootId: 'root-kept', resolvedTipId: 'tip-kept' })
    saveTranscriptTail('root-1', [msg('target')], targetProof)
    saveTranscriptTail('root-kept', [msg('kept')], unrelatedProof)
    const targetKey = entryKeyForStoredId('root-1')!
    const targetSuffix = targetKey.slice('hermes.transcript-tail.v3:'.length)
    const unrelatedKey = entryKeyForStoredId('root-kept')!
    const unrelatedRaw = window.localStorage.getItem(unrelatedKey)
    const parsed = JSON.parse(window.localStorage.getItem(targetKey)!) as Record<string, unknown>

    corrupt(parsed)
    window.localStorage.setItem(targetKey, JSON.stringify(parsed))

    expect(loadTranscriptTail('root-1', targetProof)).toBeNull()
    expect(window.localStorage.getItem(targetKey)).toBeNull()
    expect(JSON.parse(window.localStorage.getItem('hermes.transcript-tail.v3-index')!)).not.toContain(targetSuffix)
    expect(window.localStorage.getItem(unrelatedKey)).toBe(unrelatedRaw)
    expect(loadTranscriptTail('root-kept', unrelatedProof)?.messages[0].id).toBe('kept')
  })
})

describe('transcript tail cache bounded behavior', () => {
  it('marks a bounded suffix as tail coverage instead of an authoritative latest page', () => {
    const long = Array.from({ length: 200 }, (_, index) => msg(`m${index}`))
    saveTranscriptTail('root-1', long, authority(), {
      pagination: { limit: 200, offset: 0, order: 'latest', returned: 200 }
    })

    const loaded = loadTranscriptTail('root-1', authority())

    expect(loaded?.coverage).toBe('latest-page-tail')
    expect(loaded?.messages).toHaveLength(40)
    expect(loaded?.messages[0].id).toBe('m160')
    expect(loaded?.messages[39].id).toBe('m199')
    expect(loaded?.pagination).toEqual({ limit: 200, offset: 0, order: 'latest', returned: 200 })
  })

  it('marks an entry authoritative only when storage retained the entire latest page', () => {
    const page = Array.from({ length: 20 }, (_, index) => msg(`m${index}`))

    saveTranscriptTail('root-1', page, authority(), {
      pagination: { limit: 120, offset: 0, order: 'latest', returned: 20 }
    })

    expect(loadTranscriptTail('root-1', authority())).toMatchObject({
      coverage: 'latest-page',
      pagination: { limit: 120, offset: 0, order: 'latest', returned: 20 }
    })
  })

  it('falls back to a shorter tail rather than caching an oversized entry', () => {
    const heavy = Array.from({ length: 40 }, (_, index) => msg(`h${index}`, 30_000))
    saveTranscriptTail('root-1', heavy, authority())

    const loaded = loadTranscriptTail('root-1', authority())

    expect(loaded?.coverage).toBe('latest-page-tail')
    expect(loaded?.messages).toHaveLength(8)
    expect(loaded?.messages[7].id).toBe('h39')
  })

  it('migrates a legacy v3 entry without coverage as a non-conditional tail', () => {
    saveTranscriptTail('root-1', [msg('legacy')], authority())
    const key = onlyV3EntryKey()
    const legacy = JSON.parse(window.localStorage.getItem(key)!)

    delete legacy.coverage
    window.localStorage.setItem(key, JSON.stringify(legacy))

    expect(loadTranscriptTail('root-1', authority())).toMatchObject({
      coverage: 'latest-page-tail',
      messages: [{ id: 'legacy' }]
    })
  })

  it('ignores empty saves and blank ids', () => {
    saveTranscriptTail('', [msg('1')], authority())
    saveTranscriptTail('root-1', [], authority())

    expect(loadTranscriptTail('', authority())).toBeNull()
    expect(loadTranscriptTail('root-1', authority())).toBeNull()
  })

  it('self-evicts a corrupt entry instead of returning garbage', () => {
    saveTranscriptTail('root-1', [msg('1')], authority())
    const key = onlyV3EntryKey()
    window.localStorage.setItem(key, '{not json')

    expect(loadTranscriptTail('root-1', authority())).toBeNull()
    expect(window.localStorage.getItem(key)).toBeNull()
  })

  it('drops one entry and wipes everything on a gateway re-home', () => {
    saveTranscriptTail('root-1', [msg('a')], authority())
    saveTranscriptTail('root-2', [msg('b')], authority({ lineageRootId: 'root-2', resolvedTipId: 'tip-3' }))

    dropTranscriptTail('root-1', authority())
    expect(loadTranscriptTail('root-1', authority())).toBeNull()
    expect(
      loadTranscriptTail('root-2', authority({ lineageRootId: 'root-2', resolvedTipId: 'tip-3' }))
    ).not.toBeNull()

    clearTranscriptTails()
    expect(loadTranscriptTail('root-2', authority({ lineageRootId: 'root-2', resolvedTipId: 'tip-3' }))).toBeNull()
  })

  it('LRU-evicts the oldest sessions past the entry cap', () => {
    for (let index = 0; index < 55; index += 1) {
      const id = `root-${index}`
      saveTranscriptTail(id, [msg(`m${index}`)], authority({ lineageRootId: id, resolvedTipId: `tip-${index}` }))
    }

    expect(loadTranscriptTail('root-0', authority({ lineageRootId: 'root-0', resolvedTipId: 'tip-0' }))).toBeNull()
    expect(loadTranscriptTail('root-4', authority({ lineageRootId: 'root-4', resolvedTipId: 'tip-4' }))).toBeNull()
    expect(loadTranscriptTail('root-5', authority({ lineageRootId: 'root-5', resolvedTipId: 'tip-5' }))).not.toBeNull()
    expect(
      loadTranscriptTail('root-54', authority({ lineageRootId: 'root-54', resolvedTipId: 'tip-54' }))
    ).not.toBeNull()
  })

  it('repairs a poisoned persisted tail carrying a duplicate toolCallId (#87857)', () => {
    const tool = (toolCallId: string) => ({
      type: 'tool-call',
      toolCallId,
      toolName: 'terminal',
      args: {},
      argsText: ''
    })
    saveTranscriptTail(
      'root-1',
      [{ id: 'assistant-p', role: 'assistant', parts: [tool('call-b'), tool('call-b')] } as never],
      authority()
    )

    const loaded = loadedMessages('root-1')
    const ids = (loaded![0].parts as { type: string; toolCallId?: string }[])
      .filter(part => part.type === 'tool-call')
      .map(part => part.toolCallId)

    expect(ids).toHaveLength(2)
    expect(new Set(ids).size).toBe(2)
    expect(ids[0]).toBe('call-b')
  })

  it('keeps same-id tails on different authorities distinct', () => {
    const local = authority({ connectionId: 'local', profile: 'default' })
    const remote = authority({ connectionId: 'conn:mimir', profile: 'default' })
    saveTranscriptTail('root-1', [msg('local-row')], local)
    saveTranscriptTail('root-1', [msg('remote-row')], remote)

    expect(loadTranscriptTail('root-1', local)?.messages[0].id).toBe('local-row')
    expect(loadTranscriptTail('root-1', remote)?.messages[0].id).toBe('remote-row')
  })

  it('delete-path everywhere-drop clears every authority for the id', () => {
    const first = authority({ connectionId: 'homelab', profile: 'ops' })
    const second = authority({ connectionId: '', profile: 'ops', resolvedTipId: 'tip-3' })
    saveTranscriptTail('root-1', [msg('routed-row')], first)
    saveTranscriptTail('root-1', [msg('local-row')], second)
    saveTranscriptTail('root-kept', [msg('other-row')], authority({ lineageRootId: 'root-kept' }))

    dropTranscriptTailEverywhere('root-1')

    expect(loadTranscriptTail('root-1', first)).toBeNull()
    expect(loadTranscriptTail('root-1', second)).toBeNull()
    expect(loadTranscriptTail('root-kept', authority({ lineageRootId: 'root-kept' }))).not.toBeNull()
  })
})

describe('transcript tail cache orphan reconciliation', () => {
  it('everywhere-drop finds every matching unindexed scope and preserves unrelated or unknowable entries', () => {
    const first = authority({ connectionId: 'conn-a' })
    const second = authority({ connectionId: 'conn-b', resolvedTipId: 'tip-3' })
    const unrelated = authority({ lineageRootId: 'root-kept', resolvedTipId: 'tip-kept' })
    saveTranscriptTail('root-1', [msg('a')], first)
    saveTranscriptTail('root-1', [msg('b')], second)
    saveTranscriptTail('root-kept', [msg('kept')], unrelated)
    window.localStorage.removeItem('hermes.transcript-tail.v3-index')
    window.localStorage.setItem('hermes.transcript-tail.v3:corrupt-unknown-owner', '{not json')

    dropTranscriptTailEverywhere('root-1')

    expect(loadTranscriptTail('root-1', first)).toBeNull()
    expect(loadTranscriptTail('root-1', second)).toBeNull()
    expect(loadTranscriptTail('root-kept', unrelated)?.messages[0].id).toBe('kept')
    expect(window.localStorage.getItem('hermes.transcript-tail.v3:corrupt-unknown-owner')).toBe('{not json')
  })

  it.each(['missing', 'corrupt'] as const)('clear removes every v3 entry with a %s index', indexState => {
    saveTranscriptTail('root-1', [msg('a')], authority())
    saveTranscriptTail(
      'root-2',
      [msg('b')],
      authority({ lineageRootId: 'root-2', resolvedTipId: 'tip-3' })
    )

    if (indexState === 'missing') {
      window.localStorage.removeItem('hermes.transcript-tail.v3-index')
    } else {
      window.localStorage.setItem('hermes.transcript-tail.v3-index', '{not json')
    }

    clearTranscriptTails()

    expect(v3EntryKeys()).toEqual([])
    expect(window.localStorage.getItem('hermes.transcript-tail.v3-index')).toBeNull()
  })

  it('reconciles actual orphaned entries and enforces the cap by savedAt when the index is corrupt', () => {
    let now = 1_000
    const nowSpy = vi.spyOn(Date, 'now').mockImplementation(() => now++)
    const seeded = new Map<string, string>()

    try {
      for (let index = 0; index < 55; index += 1) {
        const id = `root-${index}`
        saveTranscriptTail(id, [msg(`m${index}`)], authority({ lineageRootId: id, resolvedTipId: `tip-${index}` }))
        const key = entryKeyForStoredId(id)

        if (!key) {
          throw new Error(`missing seeded entry ${id}`)
        }

        seeded.set(key, window.localStorage.getItem(key)!)
      }

      for (const [key, raw] of seeded) {
        window.localStorage.setItem(key, raw)
      }

      window.localStorage.setItem('hermes.transcript-tail.v3:corrupt-entry', '{not json')
      window.localStorage.setItem('hermes.transcript-tail.v3-index', '{not json')
      window.localStorage.setItem('hermes.transcript-tail.v1:legacy', 'keep-v1')
      window.localStorage.setItem('hermes.transcript-tail.v2:legacy', 'keep-v2')
      window.localStorage.setItem('hermes.transcript-tail.v30:unrelated', 'keep-v30')

      saveTranscriptTail(
        'root-new',
        [msg('new')],
        authority({ lineageRootId: 'root-new', resolvedTipId: 'tip-new' })
      )

      const storedIds = v3EntryKeys().map(key => JSON.parse(window.localStorage.getItem(key)!).storedSessionId)

      expect(storedIds).toHaveLength(50)
      expect(storedIds).toContain('root-new')
      expect(storedIds).toContain('root-6')
      expect(storedIds).not.toContain('root-0')
      expect(storedIds).not.toContain('root-5')
      expect(window.localStorage.getItem('hermes.transcript-tail.v3:corrupt-entry')).toBeNull()
      expect(window.localStorage.getItem('hermes.transcript-tail.v1:legacy')).toBe('keep-v1')
      expect(window.localStorage.getItem('hermes.transcript-tail.v2:legacy')).toBe('keep-v2')
      expect(window.localStorage.getItem('hermes.transcript-tail.v30:unrelated')).toBe('keep-v30')
    } finally {
      nowSpy.mockRestore()
    }
  })

  it('rolls back a new entry and restores an overwritten entry when index persistence fails', () => {
    const existingProof = authority()
    saveTranscriptTail('root-1', [msg('old')], existingProof)
    const existingKey = entryKeyForStoredId('root-1')!
    const previousRaw = window.localStorage.getItem(existingKey)
    const prototype = storagePropertyOwner('setItem')
    const originalSetItem = prototype.setItem
    const setItemSpy = vi.spyOn(prototype, 'setItem').mockImplementation(function (
      this: Storage,
      key: string,
      value: string
    ) {
      if (key === 'hermes.transcript-tail.v3-index') {
        throw new Error('index write failed')
      }

      return originalSetItem.call(this, key, value)
    })

    try {
      saveTranscriptTail('root-new', [msg('new')], authority({ lineageRootId: 'root-new' }))
      saveTranscriptTail('root-1', [msg('replacement')], existingProof)
    } finally {
      setItemSpy.mockRestore()
    }

    expect(entryKeyForStoredId('root-new')).toBeUndefined()
    expect(window.localStorage.getItem(existingKey)).toBe(previousRaw)
    expect(loadTranscriptTail('root-1', existingProof)?.messages[0].id).toBe('old')
  })

  it('does not access storage at all for an invalid authority', () => {
    const getItem = vi.spyOn(storagePropertyOwner('getItem'), 'getItem')
    const setItem = vi.spyOn(storagePropertyOwner('setItem'), 'setItem')
    const removeItem = vi.spyOn(storagePropertyOwner('removeItem'), 'removeItem')
    const key = vi.spyOn(storagePropertyOwner('key'), 'key')
    const length = vi.spyOn(storagePropertyOwner('length'), 'length', 'get')
    const invalid = authority({ displayRevision: Number.NaN })

    saveTranscriptTail('root-1', [msg('never-written')], invalid)
    expect(loadTranscriptTail('root-1', invalid)).toBeNull()

    expect(getItem).not.toHaveBeenCalled()
    expect(setItem).not.toHaveBeenCalled()
    expect(removeItem).not.toHaveBeenCalled()
    expect(key).not.toHaveBeenCalled()
    expect(length).not.toHaveBeenCalled()
  })

  it('continues everywhere-delete after one target removal fails and indexes actual survivors', () => {
    const targetProofs = [
      authority({ connectionId: 'conn-a' }),
      authority({ connectionId: 'conn-b', resolvedTipId: 'tip-3' }),
      authority({ connectionId: 'conn-c', resolvedTipId: 'tip-4' })
    ]
    const unrelatedProof = authority({ lineageRootId: 'root-kept', resolvedTipId: 'tip-kept' })

    targetProofs.forEach((proof, index) => saveTranscriptTail('root-1', [msg(`target-${index}`)], proof))
    saveTranscriptTail('root-kept', [msg('kept')], unrelatedProof)

    const targetKeys = v3EntryKeys().filter(key => {
      try {
        return JSON.parse(window.localStorage.getItem(key)!).storedSessionId === 'root-1'
      } catch {
        return false
      }
    })
    const failedKey = targetKeys[0]
    const prototype = storagePropertyOwner('removeItem')
    const originalRemoveItem = prototype.removeItem
    const attemptedKeys: string[] = []
    const removeItem = vi.spyOn(prototype, 'removeItem').mockImplementation(function (this: Storage, key: string) {
      attemptedKeys.push(key)

      if (key === failedKey) {
        throw new Error('persistent remove failure')
      }

      return originalRemoveItem.call(this, key)
    })

    try {
      expect(() => dropTranscriptTailEverywhere('root-1')).not.toThrow()
    } finally {
      removeItem.mockRestore()
    }

    for (const key of targetKeys) {
      expect(attemptedKeys).toContain(key)
    }

    expect(window.localStorage.getItem(failedKey)).not.toBeNull()
    expect(targetKeys.filter(key => window.localStorage.getItem(key) !== null)).toEqual([failedKey])
    expect(loadTranscriptTail('root-kept', unrelatedProof)?.messages[0].id).toBe('kept')

    const actualSuffixes = v3EntryKeys().map(key => key.slice('hermes.transcript-tail.v3:'.length)).sort()
    const indexedSuffixes = JSON.parse(window.localStorage.getItem('hermes.transcript-tail.v3-index')!).sort()

    expect(indexedSuffixes).toEqual(actualSuffixes)
  })

  it('restores the exact v3 snapshot when quota recovery writes the entry but cannot persist its index', () => {
    saveTranscriptTail('root-1', [msg('one')], authority())
    saveTranscriptTail(
      'root-2',
      [msg('two')],
      authority({ lineageRootId: 'root-2', resolvedTipId: 'tip-3' })
    )
    window.localStorage.setItem('unrelated.application.key', 'keep-unrelated')
    window.localStorage.setItem('hermes.transcript-tail.v2:legacy-after-purge', 'keep-legacy')
    const before = v3NamespaceSnapshot()
    const newProof = authority({ lineageRootId: 'root-new', resolvedTipId: 'tip-new' })
    const newSuffix = JSON.stringify([
      newProof.connectionId,
      newProof.profile,
      'root-new',
      newProof.lineageRootId,
      newProof.resolvedTipId,
      newProof.displayRevision
    ])
    const newKey = `hermes.transcript-tail.v3:${newSuffix}`
    const prototype = storagePropertyOwner('setItem')
    const originalSetItem = prototype.setItem
    let quotaFailed = false
    let indexFailed = false
    const attemptedSetKeys: string[] = []
    const setItem = vi.spyOn(prototype, 'setItem').mockImplementation(function (
      this: Storage,
      key: string,
      value: string
    ) {
      attemptedSetKeys.push(key)

      if (key === newKey && !quotaFailed) {
        quotaFailed = true
        throw new Error('quota exceeded')
      }

      if (key === 'hermes.transcript-tail.v3-index' && quotaFailed && !indexFailed) {
        indexFailed = true
        throw new Error('recovery index write failed')
      }

      return originalSetItem.call(this, key, value)
    })

    try {
      saveTranscriptTail('root-new', [msg('new')], newProof)
    } finally {
      setItem.mockRestore()
    }

    expect(quotaFailed).toBe(true)
    expect(attemptedSetKeys).toContain('hermes.transcript-tail.v3-index')
    expect(indexFailed).toBe(true)
    expect(attemptedSetKeys).toEqual(
      expect.arrayContaining(before.filter(([key]) => key.startsWith('hermes.transcript-tail.v3:')).map(([key]) => key))
    )
    expect(v3NamespaceSnapshot()).toEqual(before)
    expect(window.localStorage.getItem(newKey)).toBeNull()
    expect(window.localStorage.getItem('unrelated.application.key')).toBe('keep-unrelated')
    expect(window.localStorage.getItem('hermes.transcript-tail.v2:legacy-after-purge')).toBe('keep-legacy')
  })
})
