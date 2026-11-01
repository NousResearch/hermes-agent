/** Embedder → editor MCP requests. Contract: pen-embed-demo README. */

export function isPenSchemaAction(action: string): boolean {
  return action === 'schema' || action === 'get-mcp-schema'
}

/** Tool names from a `get-mcp-schema` answer; empty when it is not one. */
export function penToolNames(schema: unknown): string[] {
  const tools = schema && typeof schema === 'object' && 'tools' in schema ? (schema as { tools: unknown }).tools : null

  if (!Array.isArray(tools)) {
    return []
  }

  return tools
    .map(tool => (tool && typeof tool === 'object' ? (tool as { name?: unknown }).name : null))
    .filter((name): name is string => typeof name === 'string' && name !== '')
}

/**
 * The message a model gets for a tool the editor does not have. The editor has
 * no create/draw/add tools — every change is a script run through `execute` —
 * and a made-up name would otherwise come back as the editor's bare "No handler
 * found", which teaches nothing. Null when the name is real.
 */
export function unknownPenToolError(name: string, tools: readonly string[]): null | string {
  if (tools.length === 0 || tools.includes(name)) {
    return null
  }

  return (
    `'${name}' is not an editor tool. The editor's tools are: ${tools.join(', ')}. ` +
    "Every change to the canvas is a script run with execute({ input: '<pen script>' }) — " +
    'there are no create/add/draw tools. Learn the script API first: read_skill(), then ' +
    "read_skill({ path: 'pen-schema.md' }) and read_skill({ path: 'execute.md' })."
  )
}
