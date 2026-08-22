/**
 * DOM-page budget helpers for the transcript list.
 *
 * Kept out of `list.tsx` so the clamp / reveal contracts can be unit-tested
 * without mounting the React thread.
 */

export function shouldClampTranscriptBudget(hidden: boolean, renderBudget: number, paneBudget: number): boolean {
  return hidden && renderBudget > paneBudget
}

export function renderBudgetToRevealGroup(
  groups: readonly { id: string; weight: number }[],
  messageId: string
): number | null {
  const index = groups.findIndex(group => group.id === messageId)

  if (index < 0) {
    return null
  }

  let weight = 0

  for (let i = groups.length - 1; i >= index; i--) {
    weight += groups[i].weight
  }

  return weight
}
