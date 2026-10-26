import type { ToolCallMessagePart } from '@assistant-ui/react'
import type { ThemePicker } from '@hermes/shared'

import { recordOf } from '@/lib/connector-tools'

export const SET_THEME_TOOL = 'set_theme'

const PICKERS: readonly ThemePicker[] = ['accent', 'layout']

export interface SetThemeCall {
  picker: null | ThemePicker
}

// After a reload the transcript holds the model's own call, which reached the deferred tool through tool_call.
function bridgedArgs(toolName: string, args: ToolCallMessagePart['args']): null | ToolCallMessagePart['args'] {
  if (toolName !== 'tool_call') {
    return null
  }

  const calls = Array.isArray(args.calls) ? args.calls : [args]
  const call = recordOf(calls[0])

  return calls.length === 1 && call.name === SET_THEME_TOOL ? recordOf(call.arguments) : null
}

/** The set_theme call a tool row carries (live rows are named `set_theme`), or null for any other tool. */
export function setThemeCall(toolName: string, args: ToolCallMessagePart['args']): null | SetThemeCall {
  const callArgs = toolName === SET_THEME_TOOL ? args : bridgedArgs(toolName, args)

  return callArgs ? { picker: PICKERS.find(picker => picker === callArgs.picker) ?? null } : null
}
