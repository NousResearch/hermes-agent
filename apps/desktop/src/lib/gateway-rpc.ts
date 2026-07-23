/** True when a JSON-RPC call failed because the backend predates the method. */
export function isMissingRpcMethod(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error)

  return /method not found|-32601|unknown method|no such method/i.test(message)
}

/** True when a backend predates the read-only effective YOLO status key. */
export function isUnsupportedYoloStatus(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error)

  return isMissingRpcMethod(error) || /unknown config key:\s*yolo\b/i.test(message)
}

/** True when a prompt response raced a backend-side timeout / completion. */
export function isMissingPendingPromptRequest(error: unknown, key: string): boolean {
  const message = error instanceof Error ? error.message : String(error)

  return message.toLowerCase().includes(`no pending ${key.toLowerCase()} request`)
}
