import { describe, expect, it } from 'vitest'

import type { HermesGithubPullRequestSummary } from '@/global'

import { matchesPullRequest } from './pull-request-utils'

const item: HermesGithubPullRequestSummary = {
  id: 'o/r#42',
  repository: 'o/r',
  number: 42,
  title: 'Improve dashboard',
  url: 'u',
  state: 'OPEN',
  isDraft: false,
  author: { login: 'octo' },
  labels: [{ name: 'desktop' }],
  commentsCount: 0,
  createdAt: '',
  updatedAt: ''
}

describe('matchesPullRequest', () => {
  it.each(['dashboard', 'o/r', '42', 'octo', 'desktop'])('searches all visible fields: %s', query =>
    expect(matchesPullRequest(item, query)).toBe(true)
  )
  it('rejects unmatched text', () => expect(matchesPullRequest(item, 'backend')).toBe(false))
})
