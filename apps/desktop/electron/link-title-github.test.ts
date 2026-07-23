import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  cleanGitHubHtmlTitle,
  fetchGitHubTitle,
  githubApiPath,
  isUnusableLinkTitle,
  parseGitHubResource,
  titleFromGitHubApiJson
} from './link-title-github'

// ─── parseGitHubResource ─────────────────────────────────────────────────

test('parseGitHubResource recognizes an issue URL', () => {
  assert.deepEqual(parseGitHubResource('https://github.com/lassoanalytics/affiliate-brain/issues/6'), {
    kind: 'issue',
    owner: 'lassoanalytics',
    repo: 'affiliate-brain',
    number: 6
  })
})

test('parseGitHubResource recognizes /pull/N and /pulls/N as the same pull resource', () => {
  const expected = { kind: 'pull', owner: 'owner', repo: 'repo', number: 42 }

  assert.deepEqual(parseGitHubResource('https://github.com/owner/repo/pull/42'), expected)
  assert.deepEqual(parseGitHubResource('https://github.com/owner/repo/pulls/42'), expected)
})

test('parseGitHubResource recognizes a bare repo URL, including a trailing slash', () => {
  const expected = { kind: 'repo', owner: 'owner', repo: 'repo' }

  assert.deepEqual(parseGitHubResource('https://github.com/owner/repo'), expected)
  assert.deepEqual(parseGitHubResource('https://github.com/owner/repo/'), expected)
})

test('parseGitHubResource accepts the www subdomain', () => {
  assert.deepEqual(parseGitHubResource('https://www.github.com/owner/repo/issues/6'), {
    kind: 'issue',
    owner: 'owner',
    repo: 'repo',
    number: 6
  })
})

test('parseGitHubResource rejects issue/pull list pages', () => {
  assert.equal(parseGitHubResource('https://github.com/owner/repo/issues'), null)
  assert.equal(parseGitHubResource('https://github.com/owner/repo/pulls'), null)
})

test('parseGitHubResource rejects non-issue/pull subresources', () => {
  assert.equal(parseGitHubResource('https://github.com/owner/repo/actions/runs/123'), null)
  assert.equal(parseGitHubResource('https://github.com/owner/repo/blob/main/file.ts'), null)
  assert.equal(parseGitHubResource('https://github.com/owner/repo/settings'), null)
})

test('parseGitHubResource rejects non-github hosts, including github subdomains it does not special-case', () => {
  assert.equal(parseGitHubResource('https://example.com/owner/repo'), null)
  assert.equal(parseGitHubResource('https://gist.github.com/owner/abc123'), null)
  assert.equal(parseGitHubResource('https://api.github.com/repos/owner/repo'), null)
  assert.equal(parseGitHubResource('https://raw.githubusercontent.com/owner/repo/main/README.md'), null)
})

test('parseGitHubResource rejects owner/repo segments outside GitHub charset', () => {
  assert.equal(parseGitHubResource('https://github.com/owner/repo@bad'), null)
})

test('parseGitHubResource rejects an issue number of zero or non-numeric', () => {
  assert.equal(parseGitHubResource('https://github.com/owner/repo/issues/0'), null)
  assert.equal(parseGitHubResource('https://github.com/owner/repo/issues/abc'), null)
})

test('parseGitHubResource rejects an unparseable URL', () => {
  assert.equal(parseGitHubResource('not a url'), null)
})

// ─── githubApiPath / titleFromGitHubApiJson ──────────────────────────────

test('githubApiPath builds the expected gh api path per resource kind', () => {
  assert.equal(githubApiPath({ kind: 'issue', owner: 'o', repo: 'r', number: 6 }), 'repos/o/r/issues/6')
  assert.equal(githubApiPath({ kind: 'pull', owner: 'o', repo: 'r', number: 6 }), 'repos/o/r/pulls/6')
  assert.equal(githubApiPath({ kind: 'repo', owner: 'o', repo: 'r' }), 'repos/o/r')
})

test('titleFromGitHubApiJson reads .title for issues/pulls and .name for repos', () => {
  const issue = { kind: 'issue' as const, owner: 'o', repo: 'r', number: 6 }
  const pull = { kind: 'pull' as const, owner: 'o', repo: 'r', number: 6 }
  const repo = { kind: 'repo' as const, owner: 'o', repo: 'r' }

  assert.equal(titleFromGitHubApiJson(issue, { title: '  Fix bug  ' }), 'Fix bug')
  assert.equal(titleFromGitHubApiJson(pull, { title: 'Add feature' }), 'Add feature')
  assert.equal(titleFromGitHubApiJson(repo, { name: 'affiliate-brain', title: 'ignored' }), 'affiliate-brain')
})

test('titleFromGitHubApiJson returns empty for missing/non-string fields or non-object json', () => {
  const issue = { kind: 'issue' as const, owner: 'o', repo: 'r', number: 6 }

  assert.equal(titleFromGitHubApiJson(issue, {}), '')
  assert.equal(titleFromGitHubApiJson(issue, { title: 42 }), '')
  assert.equal(titleFromGitHubApiJson(issue, null), '')
  assert.equal(titleFromGitHubApiJson(issue, 'nope'), '')
})

// ─── fetchGitHubTitle ─────────────────────────────────────────────────────

test('fetchGitHubTitle resolves the issue title via an injected gh runner', async () => {
  const calls: Array<{ bin: string; args: string[] }> = []

  const runExecFile = async (bin: string, args: string[]) => {
    calls.push({ bin, args })

    return { ok: true, stdout: JSON.stringify({ title: 'feat: [Integration] add checklist template' }) }
  }

  const title = await fetchGitHubTitle('https://github.com/lassoanalytics/affiliate-brain/issues/6', {
    ghBin: '/opt/homebrew/bin/gh',
    runExecFile
  })

  assert.equal(title, 'feat: [Integration] add checklist template')
  assert.equal(calls.length, 1)
  assert.equal(calls[0].bin, '/opt/homebrew/bin/gh')
  assert.deepEqual(calls[0].args, ['api', 'repos/lassoanalytics/affiliate-brain/issues/6'])
})

test('fetchGitHubTitle returns empty when gh is missing or exits non-zero', async () => {
  const runExecFile = async () => ({ ok: false, stdout: '' })

  const title = await fetchGitHubTitle('https://github.com/owner/repo/issues/6', { runExecFile })

  assert.equal(title, '')
})

test('fetchGitHubTitle returns empty and never spawns for a non-github URL', async () => {
  let called = false

  const runExecFile = async () => {
    called = true

    return { ok: true, stdout: JSON.stringify({ title: 'should not be used' }) }
  }

  const title = await fetchGitHubTitle('https://example.com/owner/repo/issues/6', { runExecFile })

  assert.equal(title, '')
  assert.equal(called, false)
})

test('fetchGitHubTitle returns empty when gh output is not valid JSON', async () => {
  const runExecFile = async () => ({ ok: true, stdout: 'not json' })

  const title = await fetchGitHubTitle('https://github.com/owner/repo/issues/6', { runExecFile })

  assert.equal(title, '')
})

test('fetchGitHubTitle never throws when the runner itself rejects', async () => {
  const runExecFile = async () => {
    throw new Error('spawn failure')
  }

  const title = await fetchGitHubTitle('https://github.com/owner/repo/issues/6', { runExecFile })

  assert.equal(title, '')
})

// ─── cleanGitHubHtmlTitle ─────────────────────────────────────────────────

test('cleanGitHubHtmlTitle strips issue/PR/repo chrome suffixes', () => {
  assert.equal(cleanGitHubHtmlTitle('Fix bug · Issue #6 · owner/repo · GitHub'), 'Fix bug')
  assert.equal(cleanGitHubHtmlTitle('Add feature · Pull Request #12 · owner/repo · GitHub'), 'Add feature')
  assert.equal(cleanGitHubHtmlTitle('owner/repo · GitHub'), 'owner/repo')
  assert.equal(cleanGitHubHtmlTitle('Some Page · GitHub'), 'Some Page')
})

test('cleanGitHubHtmlTitle collapses the doubled 404 suffix down to the bare message', () => {
  assert.equal(cleanGitHubHtmlTitle('Page not found · GitHub · GitHub'), 'Page not found')
})

test('cleanGitHubHtmlTitle is a no-op for titles without GitHub chrome', () => {
  assert.equal(cleanGitHubHtmlTitle('Just a regular page title'), 'Just a regular page title')
})

// ─── isUnusableLinkTitle ───────────────────────────────────────────────────

test('isUnusableLinkTitle rejects GitHub 404 chrome, before and after cleaning', () => {
  assert.equal(isUnusableLinkTitle('Page not found · GitHub · GitHub'), true)
  assert.equal(isUnusableLinkTitle('Page not found'), true)
  assert.equal(isUnusableLinkTitle('PAGE NOT FOUND'), true)
  assert.equal(isUnusableLinkTitle('not found'), true)
})

test('isUnusableLinkTitle rejects known bot-wall/captcha titles and empty values', () => {
  assert.equal(isUnusableLinkTitle('Access Denied'), true)
  assert.equal(isUnusableLinkTitle('Just a moment...'), true)
  assert.equal(isUnusableLinkTitle(''), true)
})

test('isUnusableLinkTitle does not over-match legitimate titles that merely contain "found"', () => {
  assert.equal(isUnusableLinkTitle('Treasure Found in Egypt'), false)
  assert.equal(isUnusableLinkTitle('Item Not Found In Cart? Try This'), false)
  assert.equal(isUnusableLinkTitle('feat: [Integration] add checklist template'), false)
})
