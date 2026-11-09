/** Message lifecycle authority for an args-only Clarify preview. */
export function hasEndedClarifyPreview(
  interrupted: boolean | undefined,
  messageInterrupted: boolean,
  status: string
): boolean {
  return Boolean(interrupted || messageInterrupted || status === 'incomplete')
}
