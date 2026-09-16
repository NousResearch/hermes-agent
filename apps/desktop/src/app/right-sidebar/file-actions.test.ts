import { afterEach, describe, expect, it, vi } from 'vitest'

import { attachFileAsContext } from '@/app/right-sidebar/file-actions'
import { $composerAttachments } from '@/store/composer'

// The right-click "Add as context" path must contribute the same
// `@file:` / `@folder:` refs as the drag-drop attach path — especially
// folders, which are never uploaded at submit, so the ref staged here is
// the only one the prompt ever sees. Regression guard for the triage
// finding that folder chips contributed no refText to the submitted text.

describe('attachFileAsContext', () => {
  afterEach(() => {
    $composerAttachments.set([])
    vi.restoreAllMocks()
  })

  it('stages a folder chip with an @folder: refText', () => {
    attachFileAsContext('/repo/docs', true, '/repo')

    const chip = $composerAttachments.get()[0]!

    expect(chip.kind).toBe('folder')
    expect(chip.refText).toBe('@folder:docs')
    expect(chip.detail).toBe('docs')
  })

  it('stages a file chip with an @file: refText', () => {
    attachFileAsContext('/repo/README.md', false, '/repo')

    const chip = $composerAttachments.get()[0]!

    expect(chip.kind).toBe('file')
    expect(chip.refText).toBe('@file:README.md')
    expect(chip.detail).toBe('README.md')
  })

  it('quotes refs whose relative path contains spaces', () => {
    attachFileAsContext('/repo/my docs/notes.md', false, '/repo')

    const chip = $composerAttachments.get()[0]!

    expect(chip.refText).toBe('@file:`my docs/notes.md`')
  })

  it('falls back to the absolute path when no cwd is known', () => {
    attachFileAsContext('/repo/README.md', false, null)

    const chip = $composerAttachments.get()[0]!

    expect(chip.kind).toBe('file')
    expect(chip.refText).toBe('@file:/repo/README.md')
  })
})
