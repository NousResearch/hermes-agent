export function shouldHandleStopActiveRun(isPrimary: boolean, busy: boolean, awaitingInput: boolean): boolean {
  return isPrimary && busy && !awaitingInput
}
