/** Electron flattens console arguments, so keep wake fields in the message itself.
 * The renderer console capture persists only this warning namespace to desktop.log. */
export function logBotWake(
  message: 'hydration timed out' | 'hydration timed out, retrying',
  details: Record<string, boolean | number | string>
): void {
  console.warn(`[bot-wake] ${message} ${JSON.stringify(details)}`)
}
