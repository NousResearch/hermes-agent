import { analyzePromptDraft } from '@/lib/prompt-coach'
import { registerDraftProvider } from '@/store/composer-suggestions'
import { openPromptCoachWithAI, reconcilePromptCoachPreview } from '@/store/prompt-coach'

function draftRevision(text: string): string {
  let hash = 2166136261

  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index)
    hash = Math.imul(hash, 16777619)
  }

  return `${text.length}:${hash >>> 0}`
}

registerDraftProvider('prompt-coach', async ({ sessionId, text }) => {
  reconcilePromptCoachPreview(sessionId, text)

  const analysis = analyzePromptDraft(text)

  if (!analysis) {
    return []
  }

  return [
    {
      doneLabel: 'Prompt preview opened',
      doneTip: 'Review the suggestion before changing or sending anything',
      icon: analysis.hasPotentialSecret ? 'shield' : 'sparkle',
      id: 'improve',
      invoke: async () => openPromptCoachWithAI(sessionId, text, analysis),
      label: 'Improve prompt',
      provider: 'prompt-coach',
      revision: draftRevision(text),
      tip: `Prompt could be clearer — ${analysis.reason.toLowerCase()}`,
      workingLabel: 'Asking Hermes AI…',
      workingTip: 'Using the active Hermes model without rewriting your wording'
    }
  ]
})
