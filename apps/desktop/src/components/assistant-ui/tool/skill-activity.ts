import { translateNow } from '@/i18n'
import { firstStringField } from '@/lib/text'
import { extractToolErrorMessage } from '@/lib/tool-result-summary'

import { parseMaybeObject } from './fallback-model/format'

interface SkillCall {
  args?: unknown
  completedAt?: number
  isError?: boolean
  result?: unknown
  toolName: string
}

/** Loading instructions is not the same action as reading a skill resource,
 * and neither proves the task described by the skill has been completed. */
export function skillActivityTitle(part: SkillCall, live = true): string | undefined {
  if (part.toolName !== 'skill_view' && part.toolName !== 'skills_list') {
    return undefined
  }

  const args = parseMaybeObject(part.args)
  const result = parseMaybeObject(part.result)

  const failed = result.success !== true && result.ok !== true && Boolean(
    part.isError || extractToolErrorMessage(part.result) || result.success === false || result.ok === false
  )

  const pending = live && part.result === undefined && part.completedAt === undefined
  const missing = !pending && part.result === undefined
  const file = firstStringField(args, ['file_path'])
  const name = firstStringField(args, ['name'])
  const target = [name, file].filter(Boolean).join(' → ')

  const keys =
    part.toolName === 'skills_list'
      ? { pending: 'listing', done: 'listed', failed: 'listFailed' }
      : file
        ? { pending: 'readingResource', done: 'readResource', failed: 'resourceFailed' }
        : { pending: 'loading', done: 'loaded', failed: 'loadFailed' }

  const key = failed ? keys.failed : missing ? 'unavailable' : pending ? keys.pending : keys.done
  const label = translateNow(`assistant.tool.skillActivity.${key}`)

  return target ? `${label}: ${target}` : label
}
