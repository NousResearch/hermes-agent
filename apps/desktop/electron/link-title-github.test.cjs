const assert = require('node:assert/strict')
const test = require('node:test')

const {
  fetchGithubLinkTitle,
  formatGithubLinkTitle,
  parseGithubIssueOrPullUrl
} = require('./link-title-github.cjs')

test('parseGithubIssueOrPullUrl recognizes GitHub issue and PR URLs', () => {
  assert.deepEqual(parseGithubIssueOrPullUrl('https://github.com/altworth-markets/app/issues/2174'), {
    kind: 'issue',
    number: 2174,
    owner: 'altworth-markets',
    repo: 'app'
  })
  assert.deepEqual(parseGithubIssueOrPullUrl('https://github.com/NousResearch/hermes-agent/pull/49505#discussion'), {
    kind: 'pull',
    number: 49505,
    owner: 'NousResearch',
    repo: 'hermes-agent'
  })
  assert.equal(parseGithubIssueOrPullUrl('https://gist.github.com/NousResearch/hermes-agent/issues/1'), null)
  assert.equal(parseGithubIssueOrPullUrl('https://github.com/NousResearch/hermes-agent/actions/runs/1'), null)
})

test('formatGithubLinkTitle includes repo, issue number, and fetched title', () => {
  assert.equal(
    formatGithubLinkTitle({ owner: 'altworth-markets', repo: 'app', number: 2174 }, 'Cancel strands held units'),
    'altworth-markets/app#2174 — Cancel strands held units'
  )
})

test('fetchGithubLinkTitle resolves private issue titles through gh first', async () => {
  const calls = []
  const title = await fetchGithubLinkTitle('https://github.com/altworth-markets/app/issues/2174', {
    env: {},
    execFile: (command, args, options, callback) => {
      calls.push({ args, command, options })
      callback(null, JSON.stringify({ title: 'Cancel strands held units' }), '')
    },
    ghBinary: 'gh'
  })

  assert.equal(title, 'altworth-markets/app#2174 — Cancel strands held units')
  assert.equal(calls.length, 1)
  assert.equal(calls[0].command, 'gh')
  assert.deepEqual(calls[0].args, [
    'issue',
    'view',
    '2174',
    '--repo',
    'altworth-markets/app',
    '--json',
    'title'
  ])
  assert.equal(calls[0].options.env.GH_PROMPT_DISABLED, '1')
  assert.equal(calls[0].options.env.GIT_TERMINAL_PROMPT, '0')
  assert.equal(calls[0].options.windowsHide, true)
})

test('fetchGithubLinkTitle uses gh pr view for pull request URLs', async () => {
  const calls = []
  const title = await fetchGithubLinkTitle('https://github.com/NousResearch/hermes-agent/pull/49505', {
    env: {},
    execFile: (_command, args, _options, callback) => {
      calls.push(args)
      callback(null, JSON.stringify({ title: 'Fix private GitHub titles' }), '')
    },
    ghBinary: 'gh'
  })

  assert.equal(title, 'NousResearch/hermes-agent#49505 — Fix private GitHub titles')
  assert.equal(calls[0][0], 'pr')
})

test('fetchGithubLinkTitle falls back to GITHUB_TOKEN API when gh is unavailable', async () => {
  const requests = []
  const title = await fetchGithubLinkTitle('https://github.com/altworth-markets/app/issues/2174', {
    env: { GITHUB_TOKEN: 'test-token' },
    execFile: (_command, _args, _options, callback) => callback(new Error('gh missing')),
    ghBinary: 'gh',
    requestJson: async (url, options) => {
      requests.push({ options, url })
      return { title: 'Title from REST API' }
    }
  })

  assert.equal(title, 'altworth-markets/app#2174 — Title from REST API')
  assert.equal(requests.length, 1)
  assert.equal(requests[0].url, 'https://api.github.com/repos/altworth-markets/app/issues/2174')
  assert.equal(requests[0].options.headers.Authorization, 'Bearer test-token')
})

test('fetchGithubLinkTitle returns empty for non-GitHub URLs or missing auth', async () => {
  assert.equal(await fetchGithubLinkTitle('https://example.com/altworth-markets/app/issues/2174'), '')
  assert.equal(
    await fetchGithubLinkTitle('https://github.com/altworth-markets/app/issues/2174', {
      env: {},
      execFile: (_command, _args, _options, callback) => callback(new Error('not authenticated')),
      ghBinary: 'gh'
    }),
    ''
  )
})
