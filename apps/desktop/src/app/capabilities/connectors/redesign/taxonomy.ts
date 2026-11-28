import type { ConnectorCardModel } from '../types'

/**
 * The redesigned Connections page's app taxonomy (Figma `1538:13131`).
 *
 * The taxonomy is static: it names every app the page can show, the subgroup it
 * belongs to, the order it reads in, and which apps need no sign-in. Cards are
 * matched to it by normalized slug *and* normalized display name, so a hosted
 * slug (`stripe_mcp`) and a local server (`stripe`) land on the same app.
 */
export type ConnectionCategoryId =
  | 'work-productivity'
  | 'design-content'
  | 'money'
  | 'customers-growth'
  | 'data-analytics'
  | 'engineering'
  | 'travel-fitness'
  | 'reference'

/** Every category the page renders, including the fallback for unmapped local servers. */
export type ConnectionCategoryKey = ConnectionCategoryId | 'other'

export type ConnectionInterfaceMode = 'simple' | 'advanced'

export type ConnectionSubgroupId =
  | 'docs-files'
  | 'communication'
  | 'projects-tasks'
  | 'meetings-scheduling'
  | 'design-whiteboards'
  | 'ai-media-presentations'
  | 'websites-assets'
  | 'payments-commerce'
  | 'spend-investing-crypto'
  | 'marketing-seo'
  | 'sales-crm'
  | 'customer-support'
  | 'hiring-people'
  | 'product-analytics'
  | 'bi-search-warehouses'
  | 'hosting-databases'
  | 'automation-ai-apis'
  | 'monitoring-incidents'
  | 'code-review-ci'
  | 'travel'
  | 'outdoors-fitness'
  | 'docs-knowledge'

/** Figma `1538:13131`: the Simple interface's category order. */
export const SIMPLE_CATEGORY_ORDER = [
  'work-productivity',
  'design-content',
  'money',
  'customers-growth',
  'data-analytics',
  'engineering',
  'travel-fitness',
  'reference'
] as const satisfies readonly ConnectionCategoryId[]

/** Figma `1538:13131` & `1524:26133`: the Advanced interface's category order. */
export const ADVANCED_CATEGORY_ORDER = [
  'engineering',
  'work-productivity',
  'design-content',
  'money',
  'customers-growth',
  'data-analytics',
  'reference',
  'travel-fitness'
] as const satisfies readonly ConnectionCategoryId[]

/** Figma `1538:13131`: how many apps each category previews on the main page. */
const CATEGORY_PREVIEW_LIMITS = {
  'customers-growth': 4,
  'data-analytics': 4,
  'design-content': 4,
  engineering: 6,
  money: 4,
  reference: 4,
  'travel-fitness': 4,
  'work-productivity': 6
} as const satisfies Record<ConnectionCategoryId, number>

export function categoryOrderForMode(mode: ConnectionInterfaceMode): readonly ConnectionCategoryId[] {
  return mode === 'advanced' ? ADVANCED_CATEGORY_ORDER : SIMPLE_CATEGORY_ORDER
}

export function categoryPreviewLimit(category: ConnectionCategoryId): number {
  return CATEGORY_PREVIEW_LIMITS[category]
}

export interface ConnectionTaxonomyApp {
  /** Extra slugs the app answers to (`stripe_mcp`, `discord`, …). */
  aliases: readonly string[]
  category: ConnectionCategoryId
  name: string
  /** True for apps that work without an account (`Add` rather than `Connect`). */
  noSignIn: boolean
  /** Static connected-account rank inside its category (1-based, Figma `1538:13131`). */
  rank: number
  slug: string
  subgroup: ConnectionSubgroupId
}

export interface ConnectionTaxonomySubgroup {
  apps: readonly ConnectionTaxonomyApp[]
  id: ConnectionSubgroupId
}

export interface ConnectionTaxonomyCategory {
  id: ConnectionCategoryId
  previewLimit: number
  subgroups: readonly ConnectionTaxonomySubgroup[]
}

interface AppSpec {
  aliases?: readonly string[]
  name: string
  noSignIn?: boolean
  slug: string
}

interface SubgroupSpec {
  apps: readonly AppSpec[]
  id: ConnectionSubgroupId
}

interface CategorySpec {
  id: ConnectionCategoryId
  previewLimit: number
  subgroups: readonly SubgroupSpec[]
}

const app = (
  name: string,
  slug: string,
  extra: { aliases?: readonly string[]; noSignIn?: boolean } = {}
): AppSpec => ({ name, slug, ...extra })

const CATEGORY_SPECS: readonly CategorySpec[] = [
  {
    id: 'work-productivity',
    previewLimit: 6,
    subgroups: [
      {
        id: 'docs-files',
        apps: [
          app('Notion', 'notion'),
          app('Airtable', 'airtable'),
          app('Confluence', 'confluence', { aliases: ['atlassian-confluence'] }),
          app('Box', 'box'),
          app('PandaDoc', 'pandadoc'),
          app('Craft', 'craft'),
          app('Dropbox', 'dropbox')
        ]
      },
      {
        id: 'communication',
        apps: [
          app('Discord Bot', 'discord', { aliases: ['discord-bot', 'discordbot'] }),
          app('Slack', 'slack'),
          app('Wispr Flow', 'wispr-flow', { aliases: ['wispr', 'wisprflow'] })
        ]
      },
      {
        id: 'projects-tasks',
        apps: [
          app('Linear', 'linear'),
          app('Jira', 'jira', { aliases: ['atlassian'] }),
          app('Todoist', 'todoist'),
          app('ClickUp', 'clickup'),
          app('Asana', 'asana'),
          app('monday.com', 'monday', { aliases: ['monday-com', 'mondaycom'] })
        ]
      },
      {
        id: 'meetings-scheduling',
        apps: [
          app('Granola', 'granola', { aliases: ['granola_mcp'] }),
          app('Calendly', 'calendly'),
          app('Circleback', 'circleback'),
          app('Fireflies', 'fireflies')
        ]
      }
    ]
  },
  {
    id: 'design-content',
    previewLimit: 4,
    subgroups: [
      {
        id: 'design-whiteboards',
        apps: [
          app('Canva', 'canva', { aliases: ['canva_mcp'] }),
          app('Figma', 'figma'),
          app('Miro', 'miro')
        ]
      },
      {
        id: 'ai-media-presentations',
        apps: [
          app('Higgsfield', 'higgsfield'),
          app('Memelord', 'memelord'),
          app('Comfy Cloud', 'comfy-cloud', { aliases: ['comfy', 'comfycloud'] }),
          app('Gamma', 'gamma')
        ]
      },
      {
        id: 'websites-assets',
        apps: [
          app('Cloudinary', 'cloudinary'),
          app('Webflow', 'webflow'),
          app('WordPress.com', 'wordpress-com', { aliases: ['wordpress', 'wordpresscom'] })
        ]
      }
    ]
  },
  {
    id: 'money',
    previewLimit: 4,
    subgroups: [
      {
        id: 'payments-commerce',
        apps: [
          app('Stripe', 'stripe', { aliases: ['stripe_mcp'] }),
          app('Shopify', 'shopify'),
          app('Plaid', 'plaid'),
          app('Cashfree Payments', 'cashfree-payments', { aliases: ['cashfree', 'cashfreepayments'] }),
          app('PayPal', 'paypal'),
          app('Square', 'square')
        ]
      },
      {
        id: 'spend-investing-crypto',
        apps: [
          app('Coinbase Agents', 'coinbase-agents', { aliases: ['coinbase', 'coinbaseagent', 'coinbaseagents'] }),
          app('Coinbase Wallet', 'coinbase-wallet', { aliases: ['coinbasewallet'] }),
          app('Ramp', 'ramp'),
          app('Robinhood', 'robinhood'),
          app('Twelve Data', 'twelve-data', { aliases: ['twelvedata'] })
        ]
      }
    ]
  },
  {
    id: 'customers-growth',
    previewLimit: 4,
    subgroups: [
      {
        id: 'marketing-seo',
        apps: [
          app('Adobe Marketing Agent', 'adobe-marketing-agent', {
            aliases: ['adobe', 'adobemarketingagent']
          }),
          app('Ahrefs', 'ahrefs'),
          app('SendGrid', 'sendgrid'),
          app('Semrush', 'semrush'),
          app('Customer.io', 'customer-io', { aliases: ['customerio'] }),
          app('Klaviyo', 'klaviyo')
        ]
      },
      {
        id: 'sales-crm',
        apps: [
          app('Apollo', 'apollo', { aliases: ['apollo-io', 'apolloio'] }),
          app('People Data Labs', 'people-data-labs', { aliases: ['peopledatalabs', 'pdl'] }),
          app('Attio', 'attio'),
          app('Close', 'close')
        ]
      },
      {
        id: 'customer-support',
        apps: [
          app('Intercom', 'intercom', { aliases: ['intercom_mcp'] }),
          app('Pylon', 'pylon'),
          app('Plain', 'plain')
        ]
      },
      {
        id: 'hiring-people',
        apps: [
          app('Ashby', 'ashby'),
          app('Indeed', 'indeed'),
          app('RentAHuman', 'rentahuman', { aliases: ['rent-a-human'] })
        ]
      }
    ]
  },
  {
    id: 'data-analytics',
    previewLimit: 4,
    subgroups: [
      {
        id: 'product-analytics',
        apps: [
          app('PostHog', 'posthog'),
          app('Amplitude', 'amplitude'),
          app('Statsig', 'statsig'),
          app('Mixpanel', 'mixpanel')
        ]
      },
      {
        id: 'bi-search-warehouses',
        apps: [
          app('Metabase', 'metabase'),
          app('ClickHouse', 'clickhouse'),
          app('Algolia', 'algolia'),
          app('MotherDuck', 'motherduck')
        ]
      }
    ]
  },
  {
    id: 'engineering',
    previewLimit: 6,
    subgroups: [
      {
        id: 'hosting-databases',
        apps: [
          app('Cloudflare', 'cloudflare', { aliases: ['cloudflare_mcp'] }),
          app('Vercel', 'vercel', { aliases: ['vercel_mcp'] }),
          app('Supabase', 'supabase'),
          app('Neon', 'neon', { aliases: ['neon_mcp'] }),
          app('Railway', 'railway'),
          app('Turso', 'turso'),
          app('Netlify', 'netlify'),
          app('Prisma Postgres', 'prisma-postgres', { aliases: ['prisma', 'prismapostgres'] })
        ]
      },
      {
        id: 'automation-ai-apis',
        apps: [
          app('Resend', 'resend'),
          app('Hugging Face', 'hugging-face', { aliases: ['hugging_face', 'huggingface'] }),
          app('n8n', 'n8n', { aliases: ['n8n-official', 'n8nofficial'] }),
          app('Unreal Engine', 'unreal-engine', { aliases: ['unrealengine', 'unreal'], noSignIn: true })
        ]
      },
      {
        id: 'monitoring-incidents',
        apps: [
          app('Sentry', 'sentry', { aliases: ['sentry_mcp'] }),
          app('Better Stack', 'better-stack', { aliases: ['betterstack', 'better_stack_mcp'] }),
          app('Datadog', 'datadog', { aliases: ['datadog_mcp'] }),
          app('Globalping', 'globalping'),
          app('Grafana', 'grafana')
        ]
      },
      {
        id: 'code-review-ci',
        apps: [
          app('Greptile', 'greptile'),
          app('Mintlify', 'mintlify'),
          app('Buildkite', 'buildkite'),
          app('CircleCI', 'circleci'),
          app('GitLab', 'gitlab'),
          app('Postman', 'postman'),
          app('Semgrep', 'semgrep')
        ]
      }
    ]
  },
  {
    id: 'travel-fitness',
    previewLimit: 4,
    subgroups: [
      {
        id: 'travel',
        apps: [
          app('Kiwi', 'kiwi', { aliases: ['kiwi-com', 'kiwicom'], noSignIn: true }),
          app('Trivago', 'trivago', { noSignIn: true })
        ]
      },
      {
        id: 'outdoors-fitness',
        apps: [app('AllTrails', 'alltrails', { noSignIn: true }), app('Strava', 'strava')]
      }
    ]
  },
  {
    id: 'reference',
    previewLimit: 4,
    subgroups: [
      {
        id: 'docs-knowledge',
        apps: [
          app('AWS Knowledge', 'aws-knowledge', { aliases: ['awsknowledge', 'aws'], noSignIn: true }),
          app('Context7', 'context7', { noSignIn: true }),
          app('DeepWiki', 'deepwiki', { noSignIn: true }),
          app('Microsoft Learn', 'microsoft-learn', {
            aliases: ['microsoftlearn', 'mslearn'],
            noSignIn: true
          }),
          app('Twilio Docs', 'twilio-docs', { aliases: ['twiliodocs', 'twilio'], noSignIn: true }),
          app('Wolfram', 'wolfram', { noSignIn: true })
        ]
      }
    ]
  }
]

function categoryFromSpec(spec: CategorySpec): ConnectionTaxonomyCategory {
  let rank = 0

  return {
    id: spec.id,
    previewLimit: spec.previewLimit,
    subgroups: spec.subgroups.map(subgroup => ({
      id: subgroup.id,
      apps: subgroup.apps.map(entry => {
        rank += 1

        return {
          aliases: entry.aliases ?? [],
          category: spec.id,
          name: entry.name,
          noSignIn: entry.noSignIn === true,
          rank,
          slug: entry.slug,
          subgroup: subgroup.id
        }
      })
    }))
  }
}

export const CONNECTION_TAXONOMY: readonly ConnectionTaxonomyCategory[] = CATEGORY_SPECS.map(categoryFromSpec)

export const CONNECTION_TAXONOMY_APPS: readonly ConnectionTaxonomyApp[] = CONNECTION_TAXONOMY.flatMap(category =>
  category.subgroups.flatMap(subgroup => subgroup.apps)
)

const SUBGROUP_ORDER = new Map<ConnectionCategoryId, readonly ConnectionSubgroupId[]>(
  CONNECTION_TAXONOMY.map(
    category => [category.id, category.subgroups.map(subgroup => subgroup.id)] as const
  )
)

/** Figma `1591:17731`: the subgroup display order on a category's See-All page. */
export function subgroupOrderForCategory(category: ConnectionCategoryId): readonly ConnectionSubgroupId[] {
  return SUBGROUP_ORDER.get(category) ?? []
}

function normalized(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, '')
}

const KEY_SUFFIXES = ['official', 'connector', 'mcp', 'app'] as const

/** Slugs arrive as `stripe_mcp`, `n8n-official`, or plain `notion`; names as `monday.com`. */
function lookupKeys(value: string): string[] {
  const base = normalized(value)

  if (base.length === 0) {
    return []
  }

  const keys = [base]

  for (const suffix of KEY_SUFFIXES) {
    if (base.length > suffix.length && base.endsWith(suffix)) {
      keys.push(base.slice(0, -suffix.length))
    }
  }

  return keys
}

const TAXONOMY_INDEX = new Map<string, ConnectionTaxonomyApp>()

for (const entry of CONNECTION_TAXONOMY_APPS) {
  for (const value of [entry.slug, ...entry.aliases, entry.name]) {
    for (const key of lookupKeys(value)) {
      if (!TAXONOMY_INDEX.has(key)) {
        TAXONOMY_INDEX.set(key, entry)
      }
    }
  }
}

export function taxonomyAppForCard(card: ConnectorCardModel): ConnectionTaxonomyApp | undefined {
  for (const value of [card.slug, card.name]) {
    for (const key of lookupKeys(value)) {
      const found = TAXONOMY_INDEX.get(key)

      if (found) {
        return found
      }
    }
  }

  return undefined
}

export function categoryOfCard(card: ConnectorCardModel): ConnectionCategoryKey {
  return taxonomyAppForCard(card)?.category ?? 'other'
}

const NO_RANK = Number.MAX_SAFE_INTEGER

export function connectionRank(card: ConnectorCardModel): number {
  return taxonomyAppForCard(card)?.rank ?? NO_RANK
}

export type ConnectionCtaKind = 'add' | 'connect' | 'manage'

/** Connected hosted account, or an installed local server. */
export function isConnectedCard(card: ConnectorCardModel): boolean {
  return card.ways.hosted?.connected === true || card.ways.local?.installed === true
}

/** `Manage` when connected, `Add` when no sign-in is needed, `Connect` otherwise. */
export function cardCtaKind(card: ConnectorCardModel): ConnectionCtaKind {
  if (isConnectedCard(card)) {
    return 'manage'
  }

  if (taxonomyAppForCard(card)?.noSignIn === true) {
    return 'add'
  }

  return card.ways.local?.authType === 'none' ? 'add' : 'connect'
}

const SUGGESTED_SLUGS = {
  advanced: ['cloudflare', 'vercel', 'supabase', 'notion'],
  simple: ['notion', 'canva', 'higgsfield', 'slack']
} as const satisfies Record<ConnectionInterfaceMode, readonly string[]>

/**
 * Up to four unconnected apps in the mode's suggested order. The suggestion
 * list is fixed: a connected suggestion is dropped, never replaced from below.
 */
export function suggestedCardsForMode(
  cards: readonly ConnectorCardModel[],
  mode: ConnectionInterfaceMode
): ConnectorCardModel[] {
  const suggestions = SUGGESTED_SLUGS[mode]
  const suggested: ConnectorCardModel[] = []
  const used = new Set<string>()

  for (const slug of suggestions) {
    const match = cards.find(card => {
      const entry = taxonomyAppForCard(card)

      return entry?.slug === slug && !isConnectedCard(card) && !used.has(cardIdentity(card))
    })

    if (match) {
      used.add(cardIdentity(match))
      suggested.push(match)
    }
  }

  return suggested
}

export function cardIdentity(card: ConnectorCardModel): string {
  return `${card.residency}:${normalized(card.slug) || normalized(card.name)}`
}


