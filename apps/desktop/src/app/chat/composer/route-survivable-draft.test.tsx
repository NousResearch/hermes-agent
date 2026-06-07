import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $composerDraft, clearComposerDraft, setComposerDraft } from '@/store/composer'

// Faithful mirror of ChatBar's route-survivable draft wiring around
// `$composerDraft`, kept narrow so the regression is reproducible without
// pulling the full assistant-ui runtime into the test.
function Harness() {
  const [draft, setDraft] = useState('')
  const draftRef = useRef(draft)
  const restoredSavedDraftRef = useRef(false)

  const loadIntoComposer = useCallback((text: string) => {
    draftRef.current = text
    setDraft(text)
    setComposerDraft(text)
  }, [])

  useEffect(() => {
    if (restoredSavedDraftRef.current) {
      return
    }

    restoredSavedDraftRef.current = true

    const savedDraft = $composerDraft.get()

    if (!savedDraft || draft || draftRef.current) {
      return
    }

    loadIntoComposer(savedDraft)
  }, [draft, loadIntoComposer])

  useEffect(() => {
    if (!draft && !draftRef.current && $composerDraft.get()) {
      return
    }

    draftRef.current = draft
    setComposerDraft(draft)
  }, [draft])

  return <textarea data-testid="draft" onChange={event => setDraft(event.currentTarget.value)} value={draft} />
}

describe('composer draft persistence across route remounts', () => {
  beforeEach(() => {
    clearComposerDraft()
  })

  afterEach(() => {
    cleanup()
    clearComposerDraft()
  })

  it('restores an unsent draft after the composer unmounts and remounts (#41079)', () => {
    const first = render(<Harness />)
    const text = 'unfinished draft that should survive settings navigation'

    fireEvent.change(screen.getByTestId('draft'), { target: { value: text } })

    expect($composerDraft.get()).toBe(text)

    first.unmount()

    render(<Harness />)

    expect((screen.getByTestId('draft') as HTMLTextAreaElement).value).toBe(text)
  })

  it('does not resurrect a draft after the user cleared it before remount', () => {
    const first = render(<Harness />)

    fireEvent.change(screen.getByTestId('draft'), { target: { value: 'temporary text' } })
    fireEvent.change(screen.getByTestId('draft'), { target: { value: '' } })

    expect($composerDraft.get()).toBe('')

    first.unmount()

    render(<Harness />)

    expect((screen.getByTestId('draft') as HTMLTextAreaElement).value).toBe('')
  })
})
