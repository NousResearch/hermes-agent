import type { RolloutDraft } from './rollout-config'

export type WavePlanner = (draft: RolloutDraft) => string[][]

export function canonicalWaves(draft: RolloutDraft): string[][] {
  const canary = draft.canaryInstallId ? [draft.canaryInstallId] : []
  return [canary, draft.selectedInstallIds.filter(id => id !== draft.canaryInstallId)]
}

export function WavePreview({ draft, planner }: { draft: RolloutDraft; planner: WavePlanner }) {
  return <section aria-label="Managed rollout wave preview" className="grid gap-2"><p className="text-sm">Wave preview</p>{planner(draft).map((wave, index) => <div className="text-xs" key={index}>Wave {index + 1}: {wave.length ? wave.join(', ') : 'none'}</div>)}</section>
}
