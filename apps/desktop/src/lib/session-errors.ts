const errorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error))

export function isSessionNotFoundError(error: unknown): boolean {
  return /session not found/i.test(errorMessage(error))
}

// Gateway JSON-RPC calls use this message when a request stalls. Submit paths
// treat it like a dead runtime and recover the verified stored session once.
export function isGatewayTimeoutError(error: unknown): boolean {
  return /request timed out/i.test(errorMessage(error))
}

export function isSessionBusyError(error: unknown): boolean {
  return /session busy/i.test(errorMessage(error))
}
