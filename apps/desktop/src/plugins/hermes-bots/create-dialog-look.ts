/**
 * Persisting the look a New agent dialog gave a freshly created bot: the
 * remote machine's ui_meta (and avatar asset store) for a remote target, this
 * desktop's bot meta otherwise.
 */

import { normalizeBotRole } from './bot-role'
import { saveBotMeta } from './data'

/** Gateway RPC on the create target; null when the bot lives on the active gateway. */
export type RemoteBotRequest = (method: string, params: Record<string, unknown>) => Promise<unknown>

export interface NewBotLook {
  color: null | string
  image: null | string
  role: string
  shape: string
  title: string
}

export function persistNewBotLook(slug: string, look: NewBotLook, remote: null | RemoteBotRequest): void {
  const { color, image, shape, title } = look
  const role = normalizeBotRole(look.role)
  const roleMeta = role ? { role } : {}

  if (!remote) {
    saveBotMeta(slug, {
      shape,
      color: color ?? undefined,
      image,
      imageKind: image ? 'photo' : 'shape',
      title,
      ...roleMeta,
      created: Date.now()
    })

    return
  }

  // The bot lives on ANOTHER machine — local bot-meta is scoped to the
  // active gateway, so write appearance/title into the remote
  // profile's ui_meta (and asset store) directly. Best-effort: the
  // profile exists either way.
  const { image: avatarImage, ...remoteLook } = {
    shape,
    color,
    image,
    imageKind: image ? 'photo' : 'shape',
    title,
    ...roleMeta,
    created: Date.now()
  }

  try {
    void remote('profiles.configure', {
      name: slug,
      ui_meta: {
        'hermes-bots': remoteLook
      }
    }).catch(() => undefined)

    if (avatarImage) {
      void remote('profiles.set_asset', {
        name: slug,
        asset: 'avatar',
        data: avatarImage
      }).catch(() => undefined)
    }
  } catch {
    /* older remote gateway */
  }
}
