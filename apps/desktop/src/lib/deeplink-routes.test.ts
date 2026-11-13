import { describe, expect, it } from 'vitest'

import { CHAT_NEW_PROMPT_MAX_CHARS, normalizeChatNewPrompt, resolveDeepLinkAction } from './deeplink-routes'

describe('resolveDeepLinkAction', () => {
  it('routes unified plugin install deeplinks', () => {
    expect(
      resolveDeepLinkAction({
        kind: 'plugin',
        name: 'install',
        params: { repo: 'owner/repo', enable: '0', force: '1' }
      })
    ).toEqual({
      type: 'plugin-install',
      repo: 'owner/repo',
      enable: false,
      force: true,
      legacyHint: null
    })
  })

  it('routes catalog= to the catalog lookup, never to a git-path install', () => {
    expect(
      resolveDeepLinkAction({ kind: 'plugin', name: 'install', params: { catalog: ' web-search-plus ' } })
    ).toEqual({ type: 'plugin-catalog-install', name: 'web-search-plus' })

    // A repo riding along must not win: the reviewed catalog verdict decides.
    expect(
      resolveDeepLinkAction({
        kind: 'plugin',
        name: 'install',
        params: { catalog: 'nope', repo: 'evil/repo' }
      })
    ).toEqual({ type: 'plugin-catalog-install', name: 'nope' })

    // An empty catalog name is still a catalog request (→ error toast), not a fall-through.
    expect(
      resolveDeepLinkAction({ kind: 'plugin', name: 'install', params: { catalog: '', repo: 'evil/repo' } })
    ).toEqual({ type: 'plugin-catalog-install', name: '' })
  })

  it('does not trust URL catalog metadata on a repository install', () => {
    const params = {
      repo: 'https://github.com/owner/repo#plugins/example',
      catalog_name: 'example',
      sha: '0123456789abcdef0123456789abcdef01234567',
      enable: '1'
    }

    const url = new URL(`hermes://plugin/install?${new URLSearchParams(params)}`)

    expect(
      resolveDeepLinkAction({
        kind: url.hostname,
        name: url.pathname.slice(1),
        params: Object.fromEntries(url.searchParams)
      })
    ).toEqual({
      type: 'plugin-install',
      repo: params.repo,
      enable: true,
      force: false,
      legacyHint: null
    })
  })

  it('routes legacy plugin-agent alias', () => {
    expect(
      resolveDeepLinkAction({
        kind: 'plugin-agent',
        name: '',
        params: { repo: 'owner/repo' }
      })
    ).toMatchObject({ type: 'plugin-install', legacyHint: 'agent' })
  })

  it('only routes skill installs with an explicit, unchanged identifier', () => {
    const identifier = 'skills-sh/owner/repo/a skill?mode=one&two#readme'
    const url = new URL(`hermes://skill/install?${new URLSearchParams({ identifier })}`)

    expect(
      resolveDeepLinkAction({
        kind: url.hostname,
        name: url.pathname.slice(1),
        params: Object.fromEntries(url.searchParams)
      })
    ).toEqual({ type: 'skill-install', identifier })

    const invalidParams: Record<string, string>[] = [
      {},
      { identifier: '' },
      { identifier: '   ' },
      { identifier: ' official/skill ' }
    ]

    for (const params of invalidParams) {
      expect(resolveDeepLinkAction({ kind: 'skill', name: 'install', params })).toEqual({ type: 'ignore' })
    }

    expect(resolveDeepLinkAction({ kind: 'skill', name: 'remove', params: { identifier } })).toEqual({ type: 'ignore' })
    expect(resolveDeepLinkAction({ kind: 'plugin', name: 'install', params: {} })).toEqual({ type: 'ignore' })
  })

  it('preserves connector completion routing without treating browser status as authority', () => {
    expect(
      resolveDeepLinkAction({
        kind: 'connections',
        name: 'done',
        params: { op: ' operation-1 ', status: ' connected ' }
      })
    ).toEqual({ type: 'connection-done', op: 'operation-1', status: 'connected' })
    expect(
      resolveDeepLinkAction({
        kind: 'connections',
        name: 'done',
        params: { status: 'connected' }
      })
    ).toEqual({ type: 'ignore' })
  })

  it('routes blueprint composer inserts', () => {
    expect(
      resolveDeepLinkAction({
        kind: 'blueprint',
        name: 'morning-brief',
        params: { time: '08:00' }
      })
    ).toEqual({
      type: 'composer-blueprint',
      name: 'morning-brief',
      params: { time: '08:00' }
    })
  })

  describe('chat/new', () => {
    it('routes a prompt to a seeded fresh chat', () => {
      expect(
        resolveDeepLinkAction({
          kind: 'chat',
          name: 'new',
          params: { prompt: '  Import my Grok Bots into Hermes  ' }
        })
      ).toEqual({ type: 'chat-new', prompt: 'Import my Grok Bots into Hermes' })
    })

    it('opens a fresh chat with no seed when the prompt is absent or empty', () => {
      expect(resolveDeepLinkAction({ kind: 'chat', name: 'new', params: {} })).toEqual({ type: 'chat-new', prompt: '' })
      expect(resolveDeepLinkAction({ kind: 'chat', name: 'new', params: { prompt: '   ' } })).toEqual({
        type: 'chat-new',
        prompt: ''
      })
    })

    it('claims only chat/new, so other chat paths stay free to design', () => {
      expect(resolveDeepLinkAction({ kind: 'chat', name: '', params: {} })).toEqual({ type: 'ignore' })
      expect(resolveDeepLinkAction({ kind: 'chat', name: 'abc123', params: {} })).toEqual({ type: 'ignore' })
      expect(resolveDeepLinkAction({ kind: 'chat', name: 'new/extra', params: {} })).toEqual({ type: 'ignore' })
    })

    it('survives the URL round trip a browser click actually produces', () => {
      const prompt = 'Audit my repo & tell me what breaks first?\nStart with CI.'
      const url = new URL(`hermes://chat/new?${new URLSearchParams({ prompt })}`)

      expect(
        resolveDeepLinkAction({
          kind: url.hostname,
          name: url.pathname.replace(/^\//, ''),
          params: Object.fromEntries(url.searchParams)
        })
      ).toEqual({ type: 'chat-new', prompt })
    })
  })

  describe('normalizeChatNewPrompt', () => {
    it('strips control characters an editor cannot show or type', () => {
      // U+001B matters most: the composer's downstream sanitizer reads escape
      // sequences, and terminal context blocks ride the same text.
      expect(normalizeChatNewPrompt('run \u001b[200~evil\u001b[201~ this')).toBe('run [200~evil[201~ this')
      expect(normalizeChatNewPrompt('a\u0000b\u0007c\u009Bd')).toBe('abcd')
    })

    it('folds every line-break form to \\n', () => {
      expect(normalizeChatNewPrompt('one\r\ntwo\rthree\u2028four\u2029five')).toBe('one\ntwo\nthree\nfour\nfive')
    })

    it('keeps legitimate prose, emoji and non-Latin text intact', () => {
      const prompt = 'Résumé l’audit 🚀 — بالعربية, 中文, and a tab\tbetween words'

      expect(normalizeChatNewPrompt(prompt)).toBe(prompt)
    })

    it('caps the seed and leaves no trailing whitespace from the cut', () => {
      const long = `${'x'.repeat(CHAT_NEW_PROMPT_MAX_CHARS - 1)} ${'y'.repeat(500)}`
      const capped = normalizeChatNewPrompt(long)

      expect(capped).toHaveLength(CHAT_NEW_PROMPT_MAX_CHARS - 1)
      expect(capped).toBe('x'.repeat(CHAT_NEW_PROMPT_MAX_CHARS - 1))
    })

    it('treats a missing value as no seed', () => {
      expect(normalizeChatNewPrompt(undefined)).toBe('')
      expect(normalizeChatNewPrompt('')).toBe('')
    })
  })
})
