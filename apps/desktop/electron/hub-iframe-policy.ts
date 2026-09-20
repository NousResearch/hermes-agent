export const HERMES_HUB_ORIGIN = 'https://hermes-agent.nousresearch.com'

export function isHermesHubOrigin(value: unknown): boolean {
  if (typeof value !== 'string' || !value) {
    return false
  }

  try {
    return new URL(value).origin === HERMES_HUB_ORIGIN
  } catch {
    return false
  }
}

export function isHermesHubExternalUrl(value: unknown): boolean {
  if (typeof value !== 'string' || !value) {
    return false
  }

  try {
    return ['http:', 'https:', 'mailto:'].includes(new URL(value).protocol)
  } catch {
    return false
  }
}

export function isHermesHubClipboardWrite(
  permission: string,
  requestingOrigin?: string,
  requestingUrl?: string
): boolean {
  return (
    permission === 'clipboard-sanitized-write' &&
    (isHermesHubOrigin(requestingOrigin) || isHermesHubOrigin(requestingUrl))
  )
}
