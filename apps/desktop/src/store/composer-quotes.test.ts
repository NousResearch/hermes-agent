import { describe, expect, it } from 'vitest'

import { type ComposerQuote, createComposerQuoteScope } from './composer-quotes'

const makeQuote = (id: string): ComposerQuote => ({
  id,
  role: 'user',
  text: `text ${id}`,
  timestamp: '10:00'
})

describe('composer quote scope', () => {
  it('keeps parked quotes with their session when the main composer switches chats', () => {
    const main = createComposerQuoteScope()
    const first = main.forSession('first')
    first.add(makeQuote('first-quote'))

    expect(main.forSession('second').$quotes.get()).toEqual([])
    expect(
      main
        .forSession('first')
        .$quotes.get()
        .map(quote => quote.id)
    ).toEqual(['first-quote'])
  })

  it('adds quotes in order', () => {
    const scope = createComposerQuoteScope()

    scope.add(makeQuote('a'))
    scope.add(makeQuote('b'))

    expect(scope.$quotes.get().map(quote => quote.id)).toEqual(['a', 'b'])
  })

  it('removes by id and ignores unknown ids', () => {
    const scope = createComposerQuoteScope()

    scope.add(makeQuote('a'))
    scope.add(makeQuote('b'))
    scope.remove('a')
    scope.remove('missing')

    expect(scope.$quotes.get().map(quote => quote.id)).toEqual(['b'])
  })

  it('clears all quotes', () => {
    const scope = createComposerQuoteScope()

    scope.add(makeQuote('a'))
    scope.clear()

    expect(scope.$quotes.get()).toEqual([])
  })

  it('keeps separate scopes isolated', () => {
    const main = createComposerQuoteScope()
    const tile = createComposerQuoteScope()

    main.add(makeQuote('x'))

    expect(tile.$quotes.get()).toEqual([])
  })
})
