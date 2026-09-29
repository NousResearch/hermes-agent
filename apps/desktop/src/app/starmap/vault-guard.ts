/**
 * What a request to open another note (or close the open one) should do while
 * the editor may hold unsaved text: nothing, switch at once, or ask first.
 */
export function selectDecision(dirty: boolean, current: null | string, next: null | string): 'ask' | 'noop' | 'switch' {
  if (next === current) {
    return 'noop'
  }

  return dirty ? 'ask' : 'switch'
}
