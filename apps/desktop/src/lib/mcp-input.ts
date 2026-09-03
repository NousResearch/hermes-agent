export interface McpInputEventLike {
  currentTarget: { value: string } | null
}

/**
 * Capture an MCP credential input value synchronously so deferred state updates
 * do not depend on the input element still being available.
 */
export function readMcpInputValue(event: McpInputEventLike): string {
  const value = event.currentTarget?.value

  return typeof value === 'string' ? value : ''
}
