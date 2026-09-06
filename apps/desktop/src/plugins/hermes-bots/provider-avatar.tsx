import {
  SiClaude,
  SiGooglegemini,
  SiHuggingface,
  SiMeta,
  SiMinimax,
  SiMistralai,
  SiOllama,
  SiOpenrouter,
  SiPerplexity,
  SiVercel
} from '@icons-pack/react-simple-icons'
import type { ComponentType, SVGProps } from 'react'

import { avatarColor, BotFace } from './avatar'
import type { AvatarAppearance } from './types'

interface ProviderBrand {
  Icon: ComponentType<SVGProps<SVGSVGElement>>
  id: string
  label: string
  color: string
  monochrome?: boolean
}

/** OpenAI and Grok are not distributed by Simple Icons, so keep their official
 * marks local rather than fetching logos or adding another dependency. */
function OpenAiGlyph(props: SVGProps<SVGSVGElement>) {
  return (
    <svg fill="none" viewBox="0 0 24 24" {...props}>
      <path
        d="M9.205 8.658v-2.26c0-.19.072-.333.238-.428l4.543-2.616c.619-.357 1.356-.523 2.117-.523 2.854 0 4.662 2.212 4.662 4.566 0 .167 0 .357-.024.547l-4.71-2.759a.797.797 0 0 0-.856 0l-5.97 3.473zm10.609 8.8V12.06c0-.333-.143-.57-.429-.737l-5.97-3.473 1.95-1.118a.433.433 0 0 1 .476 0l4.543 2.617c1.309.76 2.189 2.378 2.189 3.948 0 1.808-1.07 3.473-2.76 4.163zM7.802 12.703l-1.95-1.142c-.167-.095-.239-.238-.239-.428V5.899c0-2.545 1.95-4.472 4.591-4.472 1 0 1.927.333 2.712.928L8.23 5.067c-.285.166-.428.404-.428.737v6.898zM12 15.128l-2.795-1.57v-3.33L12 8.658l2.795 1.57v3.33L12 15.128zm1.796 7.23c-1 0-1.927-.332-2.712-.927l4.686-2.712c.285-.166.428-.404.428-.737v-6.898l1.974 1.142c.167.095.238.238.238.428v5.233c0 2.545-1.974 4.472-4.614 4.472zm-5.637-5.303l-4.544-2.617c-1.308-.761-2.188-2.378-2.188-3.948A4.482 4.482 0 0 1 4.21 6.327v5.423c0 .333.143.571.428.738l5.947 3.449-1.95 1.118a.432.432 0 0 1-.476 0zm-.262 3.9c-2.688 0-4.662-2.021-4.662-4.519 0-.19.024-.38.047-.57l4.686 2.71c.286.167.571.167.856 0l5.97-3.448v2.26c0 .19-.07.333-.237.428l-4.543 2.616c-.619.357-1.356.523-2.117.523zm5.899 2.83a5.947 5.947 0 0 0 5.827-4.756C22.287 18.339 24 15.84 24 13.296c0-1.665-.713-3.282-1.998-4.448.119-.5.19-.999.19-1.498 0-3.401-2.759-5.947-5.946-5.947-.642 0-1.26.095-1.88.31A5.962 5.962 0 0 0 10.205 0a5.947 5.947 0 0 0-5.827 4.757C1.713 5.447 0 7.945 0 10.49c0 1.666.713 3.283 1.998 4.448-.119.5-.19 1-.19 1.499 0 3.401 2.759 5.946 5.946 5.946.642 0 1.26.095 1.88-.309a5.96 5.96 0 0 0 4.162 1.713z"
        fill="currentColor"
      />
    </svg>
  )
}

function GrokGlyph(props: SVGProps<SVGSVGElement>) {
  return (
    <svg fill="none" viewBox="0 0 24 24" {...props}>
      <path
        d="m9.27 15.29 7.978-5.897c.391-.29.95-.177 1.137.272.98 2.369.542 5.215-1.41 7.169-1.951 1.954-4.667 2.382-7.149 1.406l-2.711 1.257c3.889 2.661 8.611 2.003 11.562-.953 2.341-2.344 3.066-5.539 2.388-8.42l.006.007c-.983-4.232.242-5.924 2.75-9.383.06-.082.12-.164.179-.248l-3.301 3.305v-.01L9.267 15.292M7.623 16.723c-2.792-2.67-2.31-6.801.071-9.184 1.761-1.763 4.647-2.483 7.166-1.425l2.705-1.25a7.808 7.808 0 0 0-1.829-1A8.975 8.975 0 0 0 5.984 5.83c-2.533 2.536-3.33 6.436-1.962 9.764 1.022 2.487-.653 4.246-2.34 6.022-.599.63-1.199 1.259-1.682 1.925l7.62-6.815"
        fill="currentColor"
      />
    </svg>
  )
}

const PROVIDER_BRANDS: ProviderBrand[] = [
  {
    id: 'openai',
    label: 'OpenAI',
    Icon: OpenAiGlyph,
    color: '#10A37F'
  },
  {
    id: 'xai',
    label: 'Grok',
    Icon: GrokGlyph,
    color: '#FFFFFF',
    monochrome: true
  },
  {
    id: 'claude',
    label: 'Claude',
    Icon: SiClaude,
    color: '#D97757'
  },

  {
    id: 'gemini',
    label: 'Google Gemini',
    Icon: SiGooglegemini,
    color: '#4285F4'
  },
  {
    id: 'mistral',
    label: 'Mistral',
    Icon: SiMistralai,
    color: '#F97316'
  },
  {
    id: 'openrouter',
    label: 'OpenRouter',
    Icon: SiOpenrouter,
    color: '#6366F1'
  },
  {
    id: 'perplexity',
    label: 'Perplexity',
    Icon: SiPerplexity,
    color: '#20B8CD'
  },
  {
    id: 'meta',
    label: 'Meta',
    Icon: SiMeta,
    color: '#0866FF'
  },
  {
    id: 'minimax',
    label: 'MiniMax',
    Icon: SiMinimax,
    color: '#111111',
    monochrome: true
  },
  {
    id: 'ollama',
    label: 'Ollama',
    Icon: SiOllama,
    color: '#111111',
    monochrome: true
  },
  {
    id: 'huggingface',
    label: 'Hugging Face',
    Icon: SiHuggingface,
    color: '#FFD21E'
  },
  {
    id: 'vercel',
    label: 'Vercel',
    Icon: SiVercel,
    color: '#111111',
    monochrome: true
  }
]

const squash = (value: string): string => value.toLowerCase().replace(/[^a-z0-9]/g, '')

const PROVIDER_ALIASES: Record<string, string> = {
  anthropic: 'claude',
  claude: 'claude',
  claudecode: 'claude',
  gemini: 'gemini',
  google: 'gemini',
  googlevertex: 'gemini',
  huggingface: 'huggingface',
  meta: 'meta',
  minimax: 'minimax',
  mistral: 'mistral',
  mistralai: 'mistral',
  ollama: 'ollama',
  openai: 'openai',
  openaicodex: 'openai',
  chatgpt: 'openai',
  codex: 'openai',
  openrouter: 'openrouter',
  perplexity: 'perplexity',
  vercel: 'vercel',
  xai: 'xai',
  xaioauth: 'xai',
  grok: 'xai'
}

const brandById = new Map(PROVIDER_BRANDS.map(brand => [brand.id, brand]))

/** Resolve the provider mark from the canonical provider first, then from a
 * vendor-qualified model id for custom/aggregator providers. */
export function providerBrand(provider = '', model = ''): ProviderBrand | null {
  const providerKey = squash(provider)
  const modelKey = squash(model)
  const direct = PROVIDER_ALIASES[providerKey]

  if (direct) {
    return brandById.get(direct) ?? null
  }

  const modelMatch = Object.entries(PROVIDER_ALIASES).find(([alias]) => alias.length >= 5 && modelKey.includes(alias))

  return modelMatch ? (brandById.get(modelMatch[1]) ?? null) : null
}

interface ProviderAvatarProps {
  appearance?: AvatarAppearance
  className?: string
  model?: string
  name: string
  provider?: string
  size?: number
}

/** Provider-first identity mark for an agent. Unknown providers keep the
 * existing deterministic avatar so a new provider never turns into a blank box. */
export function ProviderAvatar({
  appearance,
  className,
  model = '',
  name,
  provider = '',
  size = 34
}: ProviderAvatarProps) {
  const brand = providerBrand(provider, model)
  const label = brand ? `${name} · ${brand.label}` : name

  if (brand) {
    return (
      <span
        aria-label={label}
        className={['inline-grid shrink-0 place-items-center rounded-md', className].filter(Boolean).join(' ')}
        data-provider={brand.id}
        role="img"
        style={{
          backgroundColor: `color-mix(in srgb, ${brand.color} 16%, transparent)`,
          color: brand.monochrome ? 'currentColor' : brand.color,
          height: size,
          width: size
        }}
      >
        <brand.Icon aria-hidden className="size-[58%]" />
      </span>
    )
  }

  if (appearance) {
    return (
      <span aria-label={label} className={className} data-provider="unknown" role="img">
        <BotFace
          color={avatarColor(appearance.color, name)}
          image={appearance.image}
          name={name}
          shape={appearance.shape}
          size={size}
        />
      </span>
    )
  }

  return (
    <span
      aria-label={label}
      className={['inline-grid shrink-0 place-items-center rounded-md bg-(--ui-bg-tertiary) text-xs', className]
        .filter(Boolean)
        .join(' ')}
      data-provider="unknown"
      role="img"
      style={{ height: size, width: size }}
    >
      {name.trim().charAt(0).toUpperCase()}
    </span>
  )
}
