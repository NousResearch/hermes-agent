import { expect, it } from 'vitest'

import { sourceLinesForQuote } from './preview-file'

it('maps an ordinary source selection to its exact line range', () => {
  const source = ['const repeated = true', 'middle()', 'const repeated = true'].join('\n')
  const secondOccurrence = source.lastIndexOf('const repeated')

  expect(sourceLinesForQuote(source, 'const repeated', secondOccurrence)).toEqual({ end: 3, start: 3 })
  expect(sourceLinesForQuote(source, 'middle()\nconst repeated', source.indexOf('middle()'))).toEqual({
    end: 3,
    start: 2
  })
})
