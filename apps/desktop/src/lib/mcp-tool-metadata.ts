import type { McpDiscoveredTool } from '@/hermes'

export interface McpToolHintCopy {
  additive: string
  closedWorld: string
  destructive: string
  idempotent: string
  mayModify: string
  openWorld: string
  readOnly: string
  repeatEffects: string
}

/** Format server-reported MCP display metadata for a tool's hover details. */
export function formatMcpToolDetails(tool: McpDiscoveredTool, copy: McpToolHintCopy): string {
  const lines: string[] = []
  const title = tool.title?.trim() || tool.annotations?.title?.trim()

  if (title && title !== tool.name) {
    lines.push(title)
  }

  if (tool.description.trim()) {
    lines.push(tool.description.trim())
  }

  const annotations = tool.annotations

  if (annotations?.readOnlyHint !== undefined) {
    lines.push(annotations.readOnlyHint ? copy.readOnly : copy.mayModify)
  }

  if (annotations?.destructiveHint !== undefined && annotations.readOnlyHint !== true) {
    lines.push(annotations.destructiveHint ? copy.destructive : copy.additive)
  }

  if (annotations?.idempotentHint !== undefined && annotations.readOnlyHint !== true) {
    lines.push(annotations.idempotentHint ? copy.idempotent : copy.repeatEffects)
  }

  if (annotations?.openWorldHint !== undefined) {
    lines.push(annotations.openWorldHint ? copy.openWorld : copy.closedWorld)
  }

  return lines.join('\n')
}
