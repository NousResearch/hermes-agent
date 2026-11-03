// The deferred-tool bridge's name, and its pre-#110442 name that transcripts recorded before the
// rename still carry: code that reads history recognises both (tools/tool_search_catalog.py).
export const BRIDGE_CALL_NAMES: ReadonlySet<string> = new Set(['invoke_tool', 'tool_call'])

export const isBridgeCallName = (name: string | null | undefined): boolean =>
  Boolean(name && BRIDGE_CALL_NAMES.has(name))
