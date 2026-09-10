import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { AvatarChip } from '@/components/ui/avatar-chip'

import { brandFor, brandGlyphStyle, faviconSourceFor } from './mcp-brands'

afterEach(cleanup)

const CATALOG_BRANDS = [
  'airtable',
  'algolia',
  'alltrails',
  'asana',
  'atlassian',
  'betterstack',
  'buildkite',
  'calendly',
  'circleci',
  'clickup',
  'cloudflare',
  'cloudinary',
  'datadog',
  'dropbox',
  'figma',
  'gitlab',
  'grafana',
  'hugging_face',
  'indeed',
  'intercom',
  'linear',
  'miro',
  'mixpanel',
  'n8n',
  'netlify',
  'notion',
  'paypal',
  'postman',
  'prisma-postgres',
  'railway',
  'robinhood',
  'sentry',
  'square',
  'strava',
  'stripe',
  'supabase',
  'todoist',
  'trivago',
  'unreal-engine',
  'vercel',
  'webflow',
  'wolfram',
  'wordpress-com'
]

const CATALOG_FALLBACKS = [
  'amplitude',
  'attio',
  'aws-knowledge',
  'canva',
  'close',
  'comfy-cloud',
  'context7',
  'craft',
  'deepwiki',
  'fireflies',
  'gamma',
  'globalping',
  'kiwi',
  'klaviyo',
  'microsoft-learn',
  'monday',
  'motherduck',
  'plaid',
  'semgrep',
  'twelve-data',
  'twilio-docs'
]

describe('MCP catalog brand glyphs', () => {
  it('uses product origins when a catalog source is hosted on a code platform', () => {
    expect(faviconSourceFor('aws-knowledge', 'https://awslabs.github.io/mcp/servers/aws-knowledge-mcp-server/')).toBe(
      'https://aws.amazon.com'
    )
    expect(faviconSourceFor('globalping', 'https://github.com/jsdelivr/globalping-mcp-server')).toBe(
      'https://globalping.io'
    )
    expect(faviconSourceFor('gamma', 'https://developers.gamma.app/docs/gamma-mcp-server')).toBe(
      'https://developers.gamma.app'
    )
  })

  it.each(CATALOG_BRANDS)('resolves %s to a real icon instead of a letter fallback', name => {
    const brand = brandFor(name)
    const { container } = render(<AvatarChip brand={brand} name={name} />)

    expect(brand?.Icon).toBeTruthy()
    expect(container.querySelector('svg')).toBeTruthy()
  })

  it.each(CATALOG_FALLBACKS)('keeps %s on the honest monogram fallback', name => {
    const brand = brandFor(name)
    const { container } = render(<AvatarChip brand={brand} name={name} />)

    expect(brand).toBeNull()
    expect(container.querySelector('svg')).toBeNull()
    expect(container.textContent).toBe(name.charAt(0).toUpperCase())
  })

  it('keeps dense marks optically lighter without changing their SVG source', () => {
    const dense = brandFor('circleci')
    const regular = brandFor('cloudflare')

    expect(brandGlyphStyle(dense!)).toBeUndefined()
    expect(brandGlyphStyle(regular!)).toEqual({ color: '#F38020' })

    const { container } = render(<AvatarChip brand={dense} name="circleci" />)
    const transform = container.querySelector('svg > g')?.getAttribute('transform')
    expect(transform).toContain('translate(')
    expect(transform).toContain('scale(0.9)')
  })

  it('still returns no brand for an unknown MCP server', () => {
    expect(brandFor('private-company-mcp')).toBeNull()
  })
})
