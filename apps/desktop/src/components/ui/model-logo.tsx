import anthropic from '@/assets/model-logos/anthropic.svg'
import deepseek from '@/assets/model-logos/deepseek.svg'
import google from '@/assets/model-logos/gemini.svg'
import meta from '@/assets/model-logos/meta.svg'
import mistral from '@/assets/model-logos/mistral.svg'
import nous from '@/assets/model-logos/nousresearch.svg'
import openai from '@/assets/model-logos/openai.svg'
import qwen from '@/assets/model-logos/qwen.svg'
import xai from '@/assets/model-logos/xai.svg'
import zai from '@/assets/model-logos/zai.svg'
import { Cpu } from '@/lib/icons'
import type { ModelBrand } from '@/lib/model-brand'

const LOGOS = { anthropic, deepseek, google, meta, mistral, nous, openai, qwen, xai, zai }

/** Decorative only: no tooltip, focus target, handler, or runtime network request. */
export function ModelLogo({ brand }: { brand: ModelBrand }) {
  const mask = brand === 'unknown' ? undefined : `url("${LOGOS[brand]}")`

  return (
    <span
      aria-hidden="true"
      className="pointer-events-none inline-flex size-3.5 shrink-0 self-center"
      data-model-brand={brand}
      style={
        mask
          ? {
              backgroundColor: 'currentColor',
              maskImage: mask,
              maskPosition: 'center',
              maskRepeat: 'no-repeat',
              maskSize: 'contain'
            }
          : undefined
      }
    >
      {brand === 'unknown' && <Cpu className="size-full" />}
    </span>
  )
}
