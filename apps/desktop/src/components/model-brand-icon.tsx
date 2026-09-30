import {
  SiAlibabacloud,
  SiClaude,
  SiGooglegemini,
  SiHuggingface,
  SiMeta,
  SiMistralai,
  SiNvidia,
  SiOllama,
  SiOpenrouter,
  SiPerplexity,
  SiX
} from '@icons-pack/react-simple-icons'
import type { ComponentType, SVGProps } from 'react'

import deepseekMark from '@/assets/brands/deepseek.png'
import nousMark from '@/assets/brands/nous.png'
import openaiMark from '@/assets/brands/openai.png'
import { cn } from '@/lib/utils'

type Glyph = ComponentType<SVGProps<SVGSVGElement>>

/** A model maker's mark. Makers missing from the simple-icons pack get a drawn glyph of their own. */
export interface ModelBrand {
  Icon?: Glyph
  /** Supplied artwork: `mask` marks draw in the theme's text colour (black artwork), others keep their own colours. */
  image?: { mask: boolean; src: string }
  color: string
  /** Matches against the model id, the provider slug or a preset name (lowercased, punctuation squashed). */
  match: readonly string[]
  /** Drawn marks use currentColor with the brand tint behind them. */
  name: string
}

const stroke = { fill: 'none', stroke: 'currentColor', strokeLinecap: 'round', strokeLinejoin: 'round', strokeWidth: 1.7 } as const

/** An ink drop — the free models. */
const DropGlyph: Glyph = props => (
  <svg viewBox="0 0 24 24" {...props}>
    <path d="M12 3c3.2 4.6 6.2 7.7 6.2 11.2a6.2 6.2 0 0 1-12.4 0C5.8 10.7 8.8 7.6 12 3Z" fill="currentColor" />
  </svg>
)

const ChipGlyph: Glyph = props => (
  <svg viewBox="0 0 24 24" {...props}>
    <g {...stroke}>
      <rect height="10" rx="2" width="10" x="7" y="7" />
      <path d="M10 3.5v3.5M14 3.5v3.5M10 17v3.5M14 17v3.5M3.5 10H7M3.5 14H7M17 10h3.5M17 14h3.5" />
    </g>
  </svg>
)

export const MODEL_BRANDS: readonly ModelBrand[] = [
  { image: { mask: false, src: deepseekMark }, color: '#4D6BFE', match: ['deepseek'], name: 'DeepSeek' },
  { image: { mask: true, src: openaiMark }, color: '#10A37F', match: ['openai', 'gpt', 'realtime', 'o3', 'o4'], name: 'OpenAI' },
  { Icon: SiClaude, color: '#D97757', match: ['anthropic', 'claude', 'sonnet', 'opus', 'haiku'], name: 'Claude' },
  { Icon: SiGooglegemini, color: '#4285F4', match: ['google', 'gemini', 'gemma'], name: 'Gemini' },
  { Icon: SiMeta, color: '#0668E1', match: ['meta', 'llama'], name: 'Meta' },
  { Icon: SiMistralai, color: '#FA520F', match: ['mistral', 'mixtral', 'codestral'], name: 'Mistral' },
  { Icon: SiX, color: '#8B93A7', match: ['xai', 'grok'], name: 'xAI' },
  { Icon: SiAlibabacloud, color: '#FF6A00', match: ['qwen', 'alibaba'], name: 'Qwen' },
  { Icon: SiNvidia, color: '#76B900', match: ['nvidia', 'nemotron'], name: 'NVIDIA' },
  { Icon: SiPerplexity, color: '#20808D', match: ['perplexity', 'sonar'], name: 'Perplexity' },
  { Icon: SiHuggingface, color: '#FFB000', match: ['huggingface'], name: 'Hugging Face' },
  { Icon: SiOllama, color: '#7C8597', match: ['ollama', 'local'], name: 'Ollama' },
  { image: { mask: true, src: nousMark }, color: '#8B5CF6', match: ['nousresearch', 'nous', 'hermes', 'czesiek'], name: 'Hermes' },
  { Icon: SiOpenrouter, color: '#6467F2', match: ['openrouter'], name: 'OpenRouter' },
  { Icon: DropGlyph, color: '#0EA5E9', match: ['free', 'inkling'], name: 'Free' }
]

const FALLBACK: ModelBrand = { Icon: ChipGlyph, color: '#64748B', match: [], name: 'Model' }

const squash = (value: string): string => value.toLowerCase().replace(/[^a-z0-9]/g, '')

/** The maker of a model: the id's vendor first (`deepseek/…`), then the model name, then the provider. */
export function resolveModelBrand(...hints: (null | string | undefined)[]): ModelBrand {
  for (const hint of hints) {
    const text = squash(hint ?? '')

    if (!text) {
      continue
    }

    const hit = MODEL_BRANDS.find(brand => brand.match.some(key => text.includes(key)))

    if (hit) {
      return hit
    }
  }

  return FALLBACK
}

const SIZES = { lg: 'size-11 rounded-xl [&>img]:size-7 [&>span]:size-7 [&>svg]:size-6', md: 'size-9 rounded-[0.75rem] [&>img]:size-[1.35rem] [&>span]:size-[1.35rem] [&>svg]:size-[1.2rem]', sm: 'size-7 rounded-lg [&>img]:size-[1.15rem] [&>span]:size-[1.15rem] [&>svg]:size-4' } as const

export interface ModelBrandIconProps {
  className?: string
  /** Extra hints tried after `model`: a provider slug, a preset name. */
  hints?: (null | string | undefined)[]
  model?: null | string
  size?: keyof typeof SIZES
}

/** A model's own logo on a tinted glass tile — every maker looks different at a glance. */
export function ModelBrandIcon({ className, hints = [], model, size = 'md' }: ModelBrandIconProps) {
  const brand = resolveModelBrand(model, ...hints)

  return (
    <span
      aria-hidden="true"
      className={cn('model-brand-tile inline-grid shrink-0 place-items-center', SIZES[size], className)}
      data-brand={brand.name}
      style={{ '--brand': brand.color } as React.CSSProperties}
    >
      {brand.image ? (
        brand.image.mask ? (
          <span
            className="model-brand-mask"
            style={{ WebkitMaskImage: `url(${brand.image.src})`, maskImage: `url(${brand.image.src})` }}
          />
        ) : (
          <img alt="" draggable={false} src={brand.image.src} />
        )
      ) : brand.Icon ? (
        <brand.Icon className="text-(--brand)" />
      ) : null}
    </span>
  )
}
