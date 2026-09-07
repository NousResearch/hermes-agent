/** Embedder → editor MCP requests. Contract: pen-embed-demo README. */

export function isPenSchemaAction(action: string): boolean {
  return action === 'schema' || action === 'get-mcp-schema'
}
