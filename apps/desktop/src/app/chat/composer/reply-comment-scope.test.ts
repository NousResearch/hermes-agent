import { beforeEach, describe, expect, it } from 'vitest'

import { createReplyComment } from './reply-comments'
import { createReplyCommentScope, mainReplyCommentScope } from './reply-comment-scope'

const makeComment = (seed: string) => {
  const comment = createReplyComment(`quote ${seed}`, `note ${seed}`)

  if (!comment) {
    throw new Error('expected a comment')
  }

  return comment
}

describe('createReplyCommentScope', () => {
  it('starts empty and appends comments', () => {
    const scope = createReplyCommentScope()

    expect(scope.list()).toEqual([])
    expect(scope.add(makeComment('a'))).toBe(true)
    expect(scope.list()).toHaveLength(1)
  })

  it('ignores duplicate ids', () => {
    const scope = createReplyCommentScope()
    const comment = makeComment('a')

    expect(scope.add(comment)).toBe(true)
    expect(scope.add(comment)).toBe(false)
    expect(scope.list()).toHaveLength(1)
  })

  it('refuses additions past the batch cap', () => {
    const scope = createReplyCommentScope()
    let accepted = 0

    for (let index = 0; index < 25; index += 1) {
      if (scope.add(makeComment(`c${index}`))) {
        accepted += 1
      }
    }

    expect(accepted).toBeGreaterThan(0)
    expect(scope.list()).toHaveLength(accepted)
    expect(scope.list().length).toBeLessThanOrEqual(10)
  })

  it('removes by id and clears', () => {
    const scope = createReplyCommentScope()
    const first = makeComment('a')
    const second = makeComment('b')
    scope.add(first)
    scope.add(second)

    scope.remove(first.id)

    expect(scope.list().map(item => item.id)).toEqual([second.id])

    scope.clear()

    expect(scope.list()).toEqual([])
  })

  it('updates a note', () => {
    const scope = createReplyCommentScope()
    const comment = makeComment('a')
    scope.add(comment)

    expect(scope.update(comment.id, { note: 'edited' })).toBe(true)
    expect(scope.list()[0]).toMatchObject({ note: 'edited', quote: comment.quote })
    expect(scope.update('missing', { note: 'x' })).toBe(false)
  })

  it('take() snapshots and drains exactly once', () => {
    const scope = createReplyCommentScope()
    scope.add(makeComment('a'))
    scope.add(makeComment('b'))

    const taken = scope.take()

    expect(taken).toHaveLength(2)
    expect(scope.list()).toEqual([])
    expect(scope.take()).toEqual([])
  })
})

describe('mainReplyCommentScope', () => {
  beforeEach(() => {
    mainReplyCommentScope.clear()
  })

  it('is a shared live set', () => {
    expect(mainReplyCommentScope.list()).toEqual([])

    mainReplyCommentScope.add(makeComment('shared'))

    expect(mainReplyCommentScope.list()).toHaveLength(1)
  })
})
