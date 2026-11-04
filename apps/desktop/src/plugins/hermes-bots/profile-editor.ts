import { ensureBotMetadata } from './canonical-chat'
import { requestForBot } from './routing'
import type { RosterRow } from './types'

/** Roster rows can be thin or stale. Read the selected profile before seeding
 * the editor, which also supplies the description used for avatar generation. */
export async function loadBotEditorProfile(bot: RosterRow): Promise<RosterRow> {
  await ensureBotMetadata(bot)

  const profile = await requestForBot<{ description: string }>(
    bot,
    'profiles.describe',
    { name: bot.name },
    { spawnPriority: 'foreground' }
  )

  return { ...bot, description: profile.description }
}
