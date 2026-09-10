/**
 * Curated brand glyphs for MCP server names — extracted from the mcp-tab's
 * avatar (its `MCP_BRAND_ICONS`) the moment a second surface (the composer
 * suggestion pills / inline setup card) needed the same identity ladder.
 *
 * This is the first rung only. Everything below it — the endpoint's own
 * favicon, then the initial — lives in `components/ui/connector-logo`, which
 * is what a surface should render when it wants a mark rather than a glyph.
 * Either way we never ask a third-party favicon service: an MCP URL can be a
 * private host, and that lookup would leak the hostname off-box.
 */
import {
  SiAirtable,
  SiAlgolia,
  SiAlltrails,
  SiAsana,
  SiAtlassian,
  SiBetterstack,
  SiBuildkite,
  SiCalendly,
  SiCircleci,
  SiClickup,
  SiCloudflare,
  SiCloudinary,
  SiDatadog,
  SiDropbox,
  SiFigma,
  SiGithub,
  SiGitlab,
  SiGrafana,
  SiHuggingface,
  SiIndeed,
  SiIntercom,
  SiLinear,
  SiMiro,
  SiMixpanel,
  SiN8n,
  SiNeon,
  SiNetlify,
  SiNotion,
  SiPaypal,
  SiPostgresql,
  SiPostman,
  SiPrisma,
  SiRailway,
  SiRobinhood,
  SiSentry,
  SiSquare,
  SiStrava,
  SiStripe,
  SiSupabase,
  SiTodoist,
  SiTrivago,
  SiUnrealengine,
  SiVercel,
  SiWebflow,
  SiWolfram,
  SiWordpress,
  SiZapier
} from '@icons-pack/react-simple-icons'
import type { ComponentType, SVGProps } from 'react'

export interface McpBrand {
  Icon: ComponentType<SVGProps<SVGSVGElement>>
  color: string
  /** The official mark is black/white (GitHub, Vercel, Notion): render it in
   *  `currentColor` so it follows the theme instead of vanishing on dark. The
   *  `color` stays for tint backgrounds (the avatar chip), never the glyph. */
  monochrome?: boolean
}

/**
 * Keep the source brand component intact and adapt only its rendered wrapper.
 * This follows the messaging icon treatment for marks whose normalized
 * 24×24 path is visibly heavier than neighboring marks at chip size.
 */
const withOpticalInset = (Icon: ComponentType<SVGProps<SVGSVGElement>>, scale: number) => {
  const inset = (24 * (1 - scale)) / 2

  return function OpticallyInsetBrandIcon(props: SVGProps<SVGSVGElement>) {
    return (
      <svg {...props} viewBox="0 0 24 24">
        <g transform={`translate(${inset} ${inset}) scale(${scale})`}>
          <Icon aria-hidden className="size-full" />
        </g>
      </svg>
    )
  }
}

export const MCP_BRAND_ICONS: Record<string, McpBrand> = {
  airtable: { Icon: SiAirtable, color: '#18BFFF' },
  algolia: { Icon: SiAlgolia, color: '#003DFF' },
  alltrails: { Icon: SiAlltrails, color: '#142800' },
  asana: { Icon: SiAsana, color: '#F06A6A' },
  atlassian: { Icon: SiAtlassian, color: '#0052CC' },
  betterstack: { Icon: SiBetterstack, color: '#000000', monochrome: true },
  buildkite: { Icon: SiBuildkite, color: '#14CC80' },
  calendly: { Icon: withOpticalInset(SiCalendly, 0.92), color: '#006BFF' },
  circleci: { Icon: withOpticalInset(SiCircleci, 0.9), color: '#343434', monochrome: true },
  clickup: { Icon: SiClickup, color: '#7B68EE' },
  cloudflare: { Icon: SiCloudflare, color: '#F38020' },
  cloudinary: { Icon: SiCloudinary, color: '#3448C5' },
  datadog: { Icon: SiDatadog, color: '#632CA6' },
  dropbox: { Icon: SiDropbox, color: '#0061FF' },
  figma: { Icon: SiFigma, color: '#F24E1E' },
  github: { Icon: SiGithub, color: '#181717', monochrome: true },
  gitlab: { Icon: withOpticalInset(SiGitlab, 0.9), color: '#FC6D26' },
  grafana: { Icon: SiGrafana, color: '#F46800' },
  hugging_face: { Icon: withOpticalInset(SiHuggingface, 0.9), color: '#FFD21E' },
  huggingface: { Icon: withOpticalInset(SiHuggingface, 0.9), color: '#FFD21E' },
  indeed: { Icon: SiIndeed, color: '#003A9B' },
  intercom: { Icon: withOpticalInset(SiIntercom, 0.88), color: '#6AFDEF' },
  linear: { Icon: withOpticalInset(SiLinear, 0.92), color: '#5E6AD2' },
  miro: { Icon: withOpticalInset(SiMiro, 0.88), color: '#050038', monochrome: true },
  mixpanel: { Icon: SiMixpanel, color: '#7856FF' },
  n8n: { Icon: SiN8n, color: '#EA4B71' },
  netlify: { Icon: SiNetlify, color: '#00C7B7' },
  neon: { Icon: SiNeon, color: '#34D399' },
  notion: { Icon: withOpticalInset(SiNotion, 0.9), color: '#000000', monochrome: true },
  paypal: { Icon: SiPaypal, color: '#003087' },
  postgres: { Icon: SiPostgresql, color: '#4169E1' },
  postgresql: { Icon: SiPostgresql, color: '#4169E1' },
  postman: { Icon: withOpticalInset(SiPostman, 0.88), color: '#FF6C37' },
  'prisma-postgres': { Icon: SiPrisma, color: '#2D3748', monochrome: true },
  railway: { Icon: withOpticalInset(SiRailway, 0.88), color: '#0B0D0E', monochrome: true },
  robinhood: { Icon: SiRobinhood, color: '#CCFF00' },
  sentry: { Icon: SiSentry, color: '#362D59' },
  square: { Icon: withOpticalInset(SiSquare, 0.88), color: '#3E4348', monochrome: true },
  strava: { Icon: SiStrava, color: '#FC4C02' },
  stripe: { Icon: SiStripe, color: '#635BFF' },
  supabase: { Icon: SiSupabase, color: '#3FCF8E' },
  todoist: { Icon: withOpticalInset(SiTodoist, 0.86), color: '#E44332' },
  trivago: { Icon: SiTrivago, color: '#E32851' },
  'unreal-engine': { Icon: SiUnrealengine, color: '#0E1128', monochrome: true },
  vercel: { Icon: SiVercel, color: '#000000', monochrome: true },
  webflow: { Icon: SiWebflow, color: '#146EF5' },
  'wordpress-com': { Icon: SiWordpress, color: '#21759B' },
  wolfram: { Icon: SiWolfram, color: '#DD1100' },
  zapier: { Icon: withOpticalInset(SiZapier, 0.84), color: '#FF4A00' }
}

/** Catalog sources that publish documentation on a code-hosting domain. Use
 * the product's own public origin for the favicon so the list does not paint
 * a generic GitHub mark as the MCP's identity. */
const MCP_FAVICON_SOURCES: Record<string, string> = {
  'aws-knowledge': 'https://aws.amazon.com',
  gamma: 'https://developers.gamma.app',
  globalping: 'https://globalping.io'
}

export const faviconSourceFor = (name: string, source?: null | string): null | string =>
  MCP_FAVICON_SOURCES[name] ?? source ?? null

/** Inline-glyph color for a brand: monochrome marks inherit the surrounding
 *  text color; branded marks use the brand color. */
export const brandGlyphStyle = (brand: McpBrand): { color: string } | undefined =>
  brand.monochrome ? undefined : { color: brand.color }

/** The same brand under every spelling a connector arrives with: catalog slug
 *  (`unreal-engine`), registry id (`unreal_engine`), display name (`Unreal
 *  Engine`). Compare on letters and digits alone so none of them miss. */
const squash = (value: string): string => value.toLowerCase().replace(/[^a-z0-9]/g, '')

export const brandFor = (name: string): McpBrand | null => {
  const target = squash(name)

  if (!target) {
    return null
  }

  const entries = Object.entries(MCP_BRAND_ICONS)

  return (
    entries.find(([key]) => squash(key) === target)?.[1] ??
    entries.find(([key]) => target.includes(squash(key)))?.[1] ??
    null
  )
}
