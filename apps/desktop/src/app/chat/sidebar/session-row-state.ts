/** A quiet turn is still authoritatively running. Keep the unmistakable row
 * arc until the gateway reports completion; only a blocking prompt suppresses
 * it in favour of the needs-input treatment. */
export function sessionShowsRunningArc({
  isWorking,
  needsInput
}: {
  isWorking: boolean
  needsInput: boolean
}): boolean {
  return isWorking && !needsInput
}
