import { describe, expect, it, vi } from 'vitest'

import {
  assertAttachmentUploadBudget,
  attachmentBudgetErrorMessage,
  checkAttachmentUploadBudget
} from './attachment-upload-policy'

const MiB = 1024 * 1024
const limits = { maxFileBytes: 64 * MiB, maxTotalBytes: 128 * MiB }

describe('checkAttachmentUploadBudget', () => {
  it('accepts files within both the per-file and the per-message limit', () => {
    expect(
      checkAttachmentUploadBudget(
        [
          { label: 'a', size: 10 * MiB },
          { label: 'b', size: 64 * MiB }
        ],
        limits
      )
    ).toEqual({
      ok: true,
      totalBytes: 74 * MiB
    })
  })

  it('refuses the first file over the per-file limit, naming it', () => {
    const verdict = checkAttachmentUploadBudget(
      [
        { label: 'small.txt', size: MiB },
        { label: 'course.mp4', size: 237_853_022 }
      ],
      limits
    )

    expect(verdict).toMatchObject({ ok: false, reason: 'file', label: 'course.mp4', limitBytes: 64 * MiB })
    expect(attachmentBudgetErrorMessage(verdict as never)).toMatch(/course\.mp4 is too large .*max 64 MB/)
  })

  it('refuses individually acceptable files whose total is over the per-message limit', () => {
    const verdict = checkAttachmentUploadBudget(
      [
        { label: 'a', size: 50 * MiB },
        { label: 'b', size: 50 * MiB },
        { label: 'c', size: 50 * MiB }
      ],
      limits
    )

    expect(verdict).toMatchObject({ ok: false, reason: 'total', totalBytes: 150 * MiB, limitBytes: 128 * MiB })
    expect(attachmentBudgetErrorMessage(verdict as never)).toMatch(/total 150 MB, over the 128 MB/)
  })

  it('counts unsized entries as zero so the reader can surface their real error', () => {
    expect(checkAttachmentUploadBudget([{ label: 'gone' }, { label: 'x', size: Number.NaN }], limits)).toEqual({
      ok: true,
      totalBytes: 0
    })
  })
})

describe('assertAttachmentUploadBudget', () => {
  const stat = (sizes: Record<string, number>) =>
    vi.fn(async (paths: string[]) => ({ limits, files: paths.map(path => ({ path, size: sizes[path] })) }))

  it('does not call the bridge when nothing crosses as bytes', async () => {
    const bridge = stat({})

    await assertAttachmentUploadBudget([{ kind: 'url', path: '' }, { kind: 'terminal' }], bridge)

    expect(bridge).not.toHaveBeenCalled()
  })

  it('throws a user-facing error for an oversized file', async () => {
    await expect(
      assertAttachmentUploadBudget(
        [{ kind: 'file', label: 'v.mp4', path: '/d/v.mp4' }],
        stat({ '/d/v.mp4': 300 * MiB })
      )
    ).rejects.toThrow(/v\.mp4 is too large/)
  })

  it('throws for a batch over the total and sizes the whole batch in one call', async () => {
    const bridge = stat({ '/d/a': 60 * MiB, '/d/b': 60 * MiB, '/d/c': 60 * MiB })

    await expect(
      assertAttachmentUploadBudget(
        [
          { kind: 'file', path: '/d/a' },
          { kind: 'image', path: '/d/b' },
          { kind: 'file', path: '/d/c' }
        ],
        bridge
      )
    ).rejects.toThrow(/over the 128 MB upload limit/)
    expect(bridge).toHaveBeenCalledTimes(1)
    expect(bridge).toHaveBeenCalledWith(['/d/a', '/d/b', '/d/c'])
  })

  // The main process sizes at most 256 paths per call (ATTACHMENT_STAT_MAX_PATHS)
  // and silently drops the rest. 257 x 512 KiB is over the 128 MiB message cap,
  // yet the first 256 total exactly 128 MiB — an omitted path must not count as 0.
  const truncatingStat = (size: number) =>
    vi.fn(async (paths: string[]) => ({
      limits,
      files: paths.slice(0, 256).map(path => ({ path, size }))
    }))

  it('refuses a batch whose stat report was truncated at 256 paths', async () => {
    const bridge = truncatingStat(MiB / 2)
    const candidates = Array.from({ length: 257 }, (_, index) => ({ kind: 'file', path: `/d/sample-${index}.wav` }))

    await expect(assertAttachmentUploadBudget(candidates, bridge)).rejects.toThrow(
      /Could not check the size of 1 of 257 attachments.*separate messages/
    )
    expect(bridge).toHaveBeenCalledTimes(1)
  })

  it('refuses even when the files the report did size are tiny', async () => {
    const candidates = Array.from({ length: 300 }, (_, index) => ({ kind: 'image', path: `/d/${index}.png` }))

    await expect(assertAttachmentUploadBudget(candidates, truncatingStat(1))).rejects.toThrow(
      /Could not check the size of 44 of 300 attachments/
    )
  })

  it('accepts exactly 256 files that fit the per-message limit', async () => {
    const candidates = Array.from({ length: 256 }, (_, index) => ({ kind: 'file', path: `/d/sample-${index}.wav` }))

    await expect(assertAttachmentUploadBudget(candidates, truncatingStat(MiB / 2))).resolves.toBeUndefined()
  })

  it('counts a repeated path once per attachment, from its single stat entry', async () => {
    const bridge = vi.fn(async () => ({ limits, files: [{ path: '/d/a', size: 50 * MiB }] }))

    await expect(
      assertAttachmentUploadBudget(
        [
          { kind: 'file', path: '/d/a' },
          { kind: 'file', path: '/d/a' },
          { kind: 'file', path: '/d/a' }
        ],
        bridge
      )
    ).rejects.toThrow(/total 150 MB, over the 128 MB/)
  })

  it('is a no-op on an older shell without the stat bridge', async () => {
    await expect(assertAttachmentUploadBudget([{ kind: 'file', path: '/d/huge' }], undefined)).resolves.toBeUndefined()
  })
})
