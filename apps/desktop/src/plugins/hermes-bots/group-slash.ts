import { host } from '@hermes/plugin-sdk'

import { botsText } from './i18n'

/** Group rooms cannot dispatch slash commands. Keep command-shaped input out
 * of member prompts without rejecting paths such as /etc/hosts. */
export function rejectGroupSlashCommand(text: string): boolean {
  if (!/^\/[a-z][\w-]*(?:\s|$)/i.test(text)) {
    return false
  }

  host.notify({
    kind: 'warning',
    message: botsText().group.slashCommandsUnsupported
  })

  return true
}
