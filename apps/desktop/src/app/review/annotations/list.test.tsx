import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { $annotationContext, $annotations } from '@/store/annotations'

import { ReviewAnnotationsList } from './list'

describe('ReviewAnnotationsList', () => {
  afterEach(() => {
    cleanup()
    $annotations.set([])
  })

  it('shows compact actions and selected-text context', () => {
    $annotations.set([
      {
        anchor: { excerpt: 'return result', kind: 'diff', lineEnd: 8, lineStart: 8, path: 'a.ts', side: 'new' },
        comment: 'Check this result.',
        contextId: $annotationContext.get().id,
        createdAt: 1,
        id: 'a',
        labels: [],
        status: 'active',
        type: 'concern'
      }
    ])
    const rendered = render(<ReviewAnnotationsList />)

    expect(rendered.getByRole('button', { name: 'Send feedback to agent' })).toBeTruthy()
    expect(rendered.getByRole('button', { name: 'Clear annotations' })).toBeTruthy()
    expect(rendered.getByText('return result')).toBeTruthy()
  })
})
