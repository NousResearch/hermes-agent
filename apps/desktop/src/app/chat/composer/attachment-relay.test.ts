import { describe, expect, it, vi } from 'vitest'

import type { ComposerAttachment } from '@/store/composer'

import {
  captureHostComposerAttachmentAuthority,
  createComposerAttachmentAttempt,
  type RelayAttachmentDeps,
  type RelayAttachmentTarget,
  relayComposerAttachments as relayAuthorizedViews
} from './attachment-relay'

const attachment = (id: string, label: string): ComposerAttachment => ({
  id,
  occurrenceId: `occ-${id}`,
  kind: id.includes('image') ? 'image' : 'file',
  label,
  path: `C:/private/${label}`
})

const target: RelayAttachmentTarget = {
  connectionId: 'connection-remote',
  profile: 'worker',
  runtimeSessionId: 'runtime-1',
  storedSessionId: 'stored-1',
  lineageRootId: 'root-1'
}

const testViews = new WeakMap<ComposerAttachment, ComposerAttachment>()

const authorizeTrustedForTest = (
  attachments: readonly ComposerAttachment[],
  options: { now?: () => number; ttlMs?: number } = {}
) => {
  const authority = captureHostComposerAttachmentAuthority(attachments)
  const attempt = createComposerAttachmentAttempt(authority, attachments, attachments, options)
  attachments.forEach((attachment, index) => testViews.set(attachment, attempt.attachments[index]!))
  return () => {
    attempt.release()
    attachments.forEach(attachment => testViews.delete(attachment))
  }
}

const relayTrustedForTest = (
  relayTarget: RelayAttachmentTarget,
  attachments: readonly ComposerAttachment[],
  relayDeps: RelayAttachmentDeps
) => relayAuthorizedViews(relayTarget, attachments.map(attachment => testViews.get(attachment) ?? attachment), relayDeps)

function deps(bytesById: Record<string, Uint8Array>): RelayAttachmentDeps {
  return {
    maxBytes: 1024,
    now: () => 1234,
    read: vi.fn(async item => ({
      bytes: bytesById[item.id],
      mediaType: item.kind === 'image' ? 'image/png' : 'text/plain'
    })),
    revalidate: vi.fn(async () => undefined),
    stage: vi.fn(async (exactTarget, item) => ({
      attached: true,
      bytes: item.bytes.byteLength,
      mediaType: item.mediaType,
      name: item.name,
      order: item.order,
      runtimeSessionId: exactTarget.runtimeSessionId,
      sha256: item.sha256,
      storedName: item.name,
      refText: item.mediaType.startsWith('image/') ? undefined : `@file:${item.name}`
    }))
  }
}

describe('authorized composer attachment relay', () => {
  it('preserves complete ordered integrity and non-path provenance', async () => {
    const items = [attachment('file-1', 'notes.txt'), attachment('image-2', 'café.png')]
    const release = authorizeTrustedForTest(items, { now: () => 1234 })
    const relayDeps = deps({
      'file-1': new TextEncoder().encode('alpha'),
      'image-2': new TextEncoder().encode('β')
    })

    try {
      const relayed = await relayTrustedForTest(target, items, relayDeps)

      expect(relayed.map(item => item.order)).toEqual([0, 1])
      expect(relayed.map(item => item.name)).toEqual(['notes.txt', 'café.png'])
      expect(relayed.map(item => item.storedName)).toEqual(['notes.txt', 'café.png'])
      expect(relayed.map(item => item.mediaType)).toEqual(['text/plain', 'image/png'])
      expect(relayed.map(item => item.size)).toEqual([5, 2])
      expect(relayed.every(item => /^[0-9a-f]{64}$/.test(item.sha256))).toBe(true)
      expect(relayed.every(item => item.runtimeSessionId === target.runtimeSessionId)).toBe(true)
      expect(relayed.map(item => item.provenance)).toEqual([
        { kind: 'composer', sourceId: 'file-1', occurrenceId: 'occ-file-1', authorizedAt: 1234 },
        { kind: 'composer', sourceId: 'image-2', occurrenceId: 'occ-image-2', authorizedAt: 1234 }
      ])
      expect(JSON.stringify(relayed)).not.toContain('C:/private')
      expect(relayDeps.revalidate).toHaveBeenCalledTimes(9)
    } finally {
      release()
    }
  })

  it('rejects fabricated, inherited, revoked, and expired authorization before reads', async () => {
    const real = attachment('file-real', 'real.txt')
    const fabricated = Object.assign(Object.create(real), { id: real.id }) as ComposerAttachment
    const relayDeps = deps({ 'file-real': new Uint8Array([1]) })
    const release = authorizeTrustedForTest([real], { now: () => 1000, ttlMs: 30 })

    await expect(relayTrustedForTest(target, [fabricated], relayDeps)).rejects.toThrow(/authorized composer attachment/i)
    release()
    await expect(relayTrustedForTest(target, [real], relayDeps)).rejects.toThrow(/authorized composer attachment/i)
    const expired = authorizeTrustedForTest([real], { now: () => 1000, ttlMs: 30 })
    relayDeps.now = () => 1031
    await expect(relayTrustedForTest(target, [real], relayDeps)).rejects.toThrow(/authorized composer attachment/i)
    expired()
    expect(relayDeps.read).not.toHaveBeenCalled()
    expect(relayDeps.stage).not.toHaveBeenCalled()
  })

  it('rechecks authorization after reading and immediately before staging', async () => {
    const item = attachment('file-race', 'before.txt')
    const relayDeps = deps({ 'file-race': new Uint8Array([1]) })
    vi.mocked(relayDeps.read).mockImplementationOnce(async () => {
      item.path = 'C:/private/changed-after-read.txt'
      return { bytes: new Uint8Array([1]), mediaType: 'text/plain' }
    })
    const release = authorizeTrustedForTest([item])

    try {
      await expect(relayTrustedForTest(target, [item], relayDeps)).rejects.toThrow(/changed after authorization/i)
      expect(relayDeps.stage).not.toHaveBeenCalled()
    } finally {
      release()
    }
  })

  it('rechecks authorization after target validation immediately before reading bytes', async () => {
    const item = attachment('file-before-read', 'before-read.txt')
    const relayDeps = deps({ 'file-before-read': new Uint8Array([1]) })
    const release = authorizeTrustedForTest([item])
    let probes = 0
    vi.mocked(relayDeps.revalidate).mockImplementation(async () => {
      probes += 1
      if (probes === 2) {
        release()
      }
    })

    await expect(relayTrustedForTest(target, [item], relayDeps)).rejects.toThrow(/authorized composer attachment/i)
    expect(relayDeps.read).not.toHaveBeenCalled()
    expect(relayDeps.stage).not.toHaveBeenCalled()
  })

  it('rechecks authorization after target validation immediately before staging bytes', async () => {
    const item = attachment('file-before-stage', 'before-stage.txt')
    const relayDeps = deps({ 'file-before-stage': new Uint8Array([1]) })
    const release = authorizeTrustedForTest([item])
    let probes = 0
    vi.mocked(relayDeps.revalidate).mockImplementation(async () => {
      probes += 1
      if (probes === 4) {
        release()
      }
    })

    await expect(relayTrustedForTest(target, [item], relayDeps)).rejects.toThrow(/authorized composer attachment/i)
    expect(relayDeps.read).toHaveBeenCalledTimes(1)
    expect(relayDeps.stage).not.toHaveBeenCalled()
  })

  it('revalidates the exact target at every mutation boundary and stops before a stale second read', async () => {
    const items = [attachment('file-a', 'a.txt'), attachment('file-b', 'b.txt')]
    const relayDeps = deps({ 'file-a': new Uint8Array([1]), 'file-b': new Uint8Array([2]) })
    let probes = 0
    vi.mocked(relayDeps.revalidate).mockImplementation(async () => {
      probes += 1
      if (probes === 5) {
        throw new Error('stale exact target')
      }
    })
    const release = authorizeTrustedForTest(items)

    try {
      await expect(relayTrustedForTest(target, items, relayDeps)).rejects.toThrow(/stale exact target/i)
      expect(relayDeps.read).toHaveBeenCalledTimes(1)
      expect(relayDeps.stage).toHaveBeenCalledTimes(1)
    } finally {
      release()
    }
  })

  it.each(['../escape', '..\\escape', 'C:stream', '\\\\server\\share', '/rooted', 'NUL', 'trailing.', 'trailing '])(
    'rejects unsafe cross-platform name %s',
    async label => {
      const item = attachment('file-unsafe', label)
      const relayDeps = deps({ 'file-unsafe': new Uint8Array([1]) })
      const release = authorizeTrustedForTest([item])
      try {
        await expect(relayTrustedForTest(target, [item], relayDeps)).rejects.toThrow(/attachment name/i)
        expect(relayDeps.stage).not.toHaveBeenCalled()
      } finally {
        release()
      }
    }
  )

  it('rejects unsafe media types, oversized bytes, and invalid byte payloads before staging', async () => {
    const item = attachment('file-large', 'large.bin')
    const relayDeps = deps({ 'file-large': new Uint8Array(1025) })
    const release = authorizeTrustedForTest([item])
    try {
      await expect(relayTrustedForTest(target, [item], relayDeps)).rejects.toThrow(/size limit/i)
      vi.mocked(relayDeps.read).mockResolvedValueOnce({ bytes: new Uint8Array([1]), mediaType: 'text/plain\r\nx-bad: 1' })
      await expect(relayTrustedForTest(target, [item], relayDeps)).rejects.toThrow(/media type/i)
      expect(relayDeps.stage).not.toHaveBeenCalled()
    } finally {
      release()
    }
  })

  it.each([
    ['name', { name: 'other.txt' }],
    ['media type', { mediaType: 'application/json' }],
    ['size', { bytes: 999 }],
    ['hash', { sha256: '0'.repeat(64) }],
    ['order', { order: 9 }],
    ['stored name', { storedName: '../unsafe' }],
    ['session', { runtimeSessionId: 'runtime-other' }]
  ])('rejects target-side %s integrity mismatch', async (_field, mismatch) => {
    const item = attachment('file-integrity', 'notes.txt')
    const relayDeps = deps({ 'file-integrity': new TextEncoder().encode('hello') })
    vi.mocked(relayDeps.stage).mockImplementationOnce(async (exactTarget, staged) => ({
      attached: true,
      bytes: staged.bytes.byteLength,
      mediaType: staged.mediaType,
      name: staged.name,
      order: staged.order,
      runtimeSessionId: exactTarget.runtimeSessionId,
      sha256: staged.sha256,
      storedName: staged.name,
      ...mismatch
    }))
    const release = authorizeTrustedForTest([item])
    try {
      await expect(relayTrustedForTest(target, [item], relayDeps)).rejects.toThrow(/integrity/i)
    } finally {
      release()
    }
  })
})
