import { useStore } from '@nanostores/react'

import { $currentReasoningEffortWire } from '@/store/session'

import { $activeComposerRevision, getActiveComposer, markActiveComposer } from '../focus'
import { useComposerVisible } from '../visibility'

import { useComposerBehavior } from './use-composer-behavior'

interface SurfaceOptions extends Omit<Parameters<typeof useComposerBehavior>[0], 'enabled' | 'revision'> {
  target: Parameters<typeof markActiveComposer>[0]
  model: { provider: unknown; model: unknown }
  blockers: readonly unknown[]
  placeholder: string
}

export function useSurfaceComposerBehavior({ target, model, blockers, placeholder, ...options }: SurfaceOptions) {
  const visible = useComposerVisible()
  useStore($activeComposerRevision)
  const reasoning = useStore($currentReasoningEffortWire)

  const behavior = useComposerBehavior({
    ...options,
    revision: `${model.provider}:${model.model}:${reasoning}`,
    enabled: visible && getActiveComposer() === target && !blockers.some(Boolean)
  })

  return {
    ...behavior,
    inputProps: {
      'data-next-prompt': behavior.text ? 'ready' : undefined,
      'data-placeholder': behavior.text || placeholder,
      title: behavior.text ? `${behavior.text} · Tab` : undefined,
      onFocus: () => {
        markActiveComposer(target)
        behavior.refresh()
      }
    }
  }
}
