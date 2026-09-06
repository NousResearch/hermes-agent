import { Codicon } from '@hermes/plugin-sdk'

import type { StepKind } from './scenario'

export type KindMarkName = StepKind | 'judge'

export const KIND_ICON: Record<KindMarkName, string> = {
  agent: 'hubot',
  gate: 'git-branch',
  human: 'person',
  wait: 'clock',
  trigger: 'zap',
  judge: 'eye'
}

export function kindMarkOf(def: { kind: StepKind; icon?: string }): KindMarkName {
  if (def.icon === 'eye') {
    return 'judge'
  }

  return def.kind
}

export function KindMark({ kind, title }: { kind: KindMarkName; title?: string }) {
  const tone = kind === 'judge' ? 'agent' : kind

  return (
    <span className={`kind-mark tile-${tone}`} title={title}>
      <Codicon name={KIND_ICON[kind]} size={16} />
    </span>
  )
}
