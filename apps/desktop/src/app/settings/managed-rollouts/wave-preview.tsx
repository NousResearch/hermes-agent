import type { RolloutDraft } from './rollout-config'

export function canonicalWaves(draft: RolloutDraft): string[][] {
  const canary = draft.canaryInstallId ? [draft.canaryInstallId] : []
  return [canary, draft.selectedInstallIds.filter(id => id !== draft.canaryInstallId)]
}

export function WavePreview({ draft }: { draft: RolloutDraft }) {
  return <section aria-label="Managed rollout wave preview" className="grid gap-2"><p className="text-sm">Wave preview</p>{canonicalWaves(draft).map((wave, index) => <div className="text-xs" key={index}>Wave {index + 1}: {wave.length ? wave.join(', ') : 'none'}</div>)}</section>
}
