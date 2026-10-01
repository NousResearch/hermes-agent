import { pluginCatalogInstallUrl, skillCatalogInstallIdentifier, skillCatalogInstallUrl } from '@hermes/shared'
import { describe, expect, it } from 'vitest'

import { resolveDeepLinkAction } from './deeplink-routes'

function route(href: string) {
  const url = new URL(href)

  return resolveDeepLinkAction({
    kind: url.hostname,
    name: url.pathname.slice(1),
    params: Object.fromEntries(url.searchParams)
  })
}

describe('public catalog install links', () => {
  it.each([
    [{ name: 'Apple Design', source: 'ClawHub', identifier: 'apple-design' }, 'clawhub/apple-design'],
    [{ name: 'Apple Design', source: 'ClawHub', identifier: 'clawhub/apple-design' }, 'clawhub/apple-design'],
    [{ name: 'pdf', source: 'Anthropic', identifier: 'anthropics/skills/skills/pdf' }, 'anthropics/skills/skills/pdf'],
    [{ name: 'pdf', source: 'optional' }, 'official/pdf'],
    [
      { name: 'pdf', source: 'built-in', installIdentifier: 'NousResearch/hermes-agent/skills/productivity/pdf' },
      'NousResearch/hermes-agent/skills/productivity/pdf'
    ],
    [
      { name: 'A name', source: 'future-source', identifier: 'provider/path?mode=one&two#readme' },
      'provider/path?mode=one&two#readme'
    ]
  ])('preserves the source-specific target for %o', (entry, identifier) => {
    expect(skillCatalogInstallIdentifier(entry)).toBe(identifier)
    expect(route(skillCatalogInstallUrl(entry)!)).toEqual({ type: 'skill-install', identifier })
  })

  it('omits bundled install links when an older snapshot has no exact target', () => {
    expect(skillCatalogInstallUrl({ name: 'pdf', source: 'built-in' })).toBeNull()
  })

  it('routes only the canonical catalog name through the reviewed lookup', () => {
    const plugin = {
      name: 'weather',
      repo: 'https://github.com/owner/plugins',
      subdir: 'packages/weather',
      sha: 'a'.repeat(40)
    }

    expect(route(pluginCatalogInstallUrl(plugin))).toEqual({
      type: 'plugin-catalog-install',
      name: plugin.name
    })
    expect([...new URL(pluginCatalogInstallUrl(plugin)).searchParams]).toEqual([['catalog', plugin.name]])
  })
})
