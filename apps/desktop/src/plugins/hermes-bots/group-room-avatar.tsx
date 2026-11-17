import { useValue } from '@hermes/plugin-sdk'

import { avatarColor, botAppearance, BotFace } from './avatar'
import { isBackfilledFacePng } from './avatar-image'
import { $botMeta } from './data'

/** A bot's face from its Bot Mode appearance (custom image/pet, else its shape + color). */
export function RoomBotAvatar({ name, size = 24 }: { name: string; size?: number }) {
  const meta = useValue($botMeta) as Record<string, Parameters<typeof botAppearance>[1]>
  const appearance = botAppearance(name, meta?.[name])
  const photo = appearance.image && !isBackfilledFacePng(appearance.image) ? appearance.image : null

  return (
    <BotFace
      color={avatarColor(appearance.color, name)}
      image={photo}
      name={name}
      shape={appearance.shape}
      size={size}
    />
  )
}
