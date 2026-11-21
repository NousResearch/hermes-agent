import { translateNow } from '@/i18n'
import { firstStringField } from '@/lib/text'

import { parseMaybeObject, toolCallFailed } from './fallback-model/format'

interface SkillCall {
  args?: unknown
  completedAt?: number
  isError?: boolean
  result?: unknown
  toolName: string
}

/** A repeat `skill_view` of a file already served in this conversation answers a
 * short stub instead of the body (`tools/skills_tool_dedup.py`: `status`
 * unchanged, `content_returned` false). Nothing new reached the context, so the
 * row must not read as another load. */
function repeatedViewStub(result: Record<string, unknown>): boolean {
  return result.status === 'unchanged' || result.content_returned === false
}

/** Loading instructions is not the same action as reading a skill resource,
 * and neither proves the task described by the skill has been completed. */
export function skillActivityTitle(part: SkillCall, live = true): string | undefined {
  if (part.toolName !== 'skill_view' && part.toolName !== 'skills_list') {
    return undefined
  }

  const args = parseMaybeObject(part.args)
  const result = parseMaybeObject(part.result)
  const failed = toolCallFailed(part)
  const pending = live && part.result === undefined && part.completedAt === undefined
  const missing = !pending && part.result === undefined
  const repeated = part.toolName === 'skill_view' && repeatedViewStub(result)
  const file = firstStringField(args, ['file_path'])
  const name = firstStringField(args, ['name'])
  const target = [name, file].filter(Boolean).join(' → ')

  const keys =
    part.toolName === 'skills_list'
      ? { pending: 'listing', done: 'listed', failed: 'listFailed' }
      : file
        ? { pending: 'readingResource', done: 'readResource', failed: 'resourceFailed' }
        : { pending: 'loading', done: 'loaded', failed: 'loadFailed' }

  const done = repeated ? 'unchanged' : keys.done
  const key = failed ? keys.failed : missing ? 'unavailable' : pending ? keys.pending : done
  const label = translateNow(`assistant.tool.skillActivity.${key}`)

  return target ? `${label}: ${target}` : label
}
