import { afterEach, expect, it, vi } from 'vitest'

import { reviewPrList } from './git-review-ops'

const { execFile } = vi.hoisted(() => ({ execFile: vi.fn() }))

vi.mock('node:child_process', () => ({ execFile }))

afterEach(() => vi.resetAllMocks())

it('keeps legacy branch/number results while marking failed URL hydration non-authoritative', async () => {
  const known = { headRefName: 'feature', number: 42, state: 'OPEN', url: 'https://github.com/owner/repo/pull/42' }

  const queries: string[] = []

  execFile.mockImplementation((_bin, args, _options, callback) => {
    if (args[0] === 'repo') {
      callback(null, 'owner/repo')
    } else if (args[0] === 'api') {
      queries.push(args[3])
      callback(null, JSON.stringify({ data: { repository: { b0: { nodes: [known] }, n0: known } } }))
    } else {
      callback(null, 'not JSON')
    }
  })

  const legacy = await reviewPrList(process.cwd(), 'gh', ['feature'], [42])

  expect(queries.join('\n')).toContain('headRefName: "feature"')
  expect(queries.join('\n')).toContain('pullRequest(number: 42)')
  expect(legacy.ghReady).toBe(true)
  expect(legacy.prs.map(pr => pr.url)).toContain(known.url)

  const failed = await reviewPrList(process.cwd(), 'gh', ['feature'], [42], ['https://github.com/other/repo/pull/42'])

  expect(failed.prs.map(pr => pr.url)).toContain(known.url)
  expect(failed.ghReady).toBe(false)
  execFile.mockImplementation((_bin, _args, _options, callback) => callback(new Error('network unavailable'), ''))
  expect(await reviewPrList('', 'gh', [], [], [known.url])).toEqual({ ghReady: false, prs: [] })
})

it('hydrates explicit GitHub PR identities without a checkout using bounded, validated reads', async () => {
  const urls = [
    ...Array.from({ length: 9 }, (_, i) => `https://github.com/owner/repo-${i}/pull/42`),
    'https://github.com/actions/.github/pull/221',
    'https://github.com/owner/.config/pull/42/'
  ]

  const calls: string[][] = []
  let active = 0
  let peak = 0

  execFile.mockImplementation((_bin, args, options, callback) => {
    calls.push(args)
    expect(options.cwd).toBeUndefined()
    active++
    peak = Math.max(peak, active)
    setImmediate(() => {
      active--
      callback(
        null,
        JSON.stringify({
          headRefName: 'feature',
          isDraft: true,
          number: 42,
          state: 'MERGED',
          title: args[2],
          url: args[2]
        })
      )
    })
  })

  const result = await reviewPrList(
    '',
    'gh',
    [],
    [],
    [
      ...urls,
      urls[0],
      'https://evil.example/owner/repo/pull/42',
      'https://github.com.evil.example/owner/repo/pull/42',
      'https://github.com@evil.example/owner/repo/pull/42',
      'http://github.com/owner/repo/pull/42',
      'https://github.com/owner/repo/pull/0',
      'https://github.com/owner/./pull/42',
      'https://github.com/owner/../pull/42',
      'https://github.com/./repo/pull/42',
      'https://github.com/../repo/pull/42',
      'https://github.com/owner/%2e/pull/42',
      'https://github.com/owner/%2e%2e/pull/42',
      'https://github.com/owner/.github/../repo/pull/42',
      'https://github.com/owner/repo/pull/42\n'
    ]
  )

  expect(result.ghReady).toBe(true)
  expect(result.prs.map(pr => pr.url)).toEqual(urls)
  expect(result.prs[0]).toMatchObject({ branch: 'feature', draft: true, number: 42, state: 'merged' })
  expect(calls.map(args => args.slice(0, 3))).toEqual(urls.map(url => ['pr', 'view', url]))
  expect(calls.every(args => args[3] === '--json' && args[4].includes('headRefName'))).toBe(true)
  expect(peak).toBeGreaterThan(1)
  expect(peak).toBeLessThanOrEqual(4)
})
