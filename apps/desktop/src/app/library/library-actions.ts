import type { NavigateFunction } from 'react-router'

import type { ArtifactRecord } from '@/app/artifacts/artifact-utils'
import type { IndexedChat } from '@/app/artifacts/item-index-snapshot'
import { translateNow } from '@/i18n'
import {
  downloadGatewayFileWithFeedback,
  downloadGatewayMediaFile,
  isArtifactFilePath,
  isRemoteGateway
} from '@/lib/media'
import { notifyError } from '@/store/notifications'
import { setSessionOwnerHint } from '@/store/session'

import { openSessionFromPicker } from '../open-session'

/** The chat an item came from, looked up in the owner's index (for its owner route and title). */
export function chatOfItem(chats: Record<string, IndexedChat>, item: ArtifactRecord): IndexedChat | undefined {
  const profile = item.profile || 'default'

  return Object.values(chats).find(chat => chat.sessionId === item.sessionId && chat.profile === profile)
}

/** Open the item's chat on the machine and profile that own it, not on whichever is in front. */
export function openItemChat(item: ArtifactRecord, chat: IndexedChat | undefined, navigate: NavigateFunction): void {
  if (chat && chat.connectionId !== 'local') {
    setSessionOwnerHint(item.sessionId, { connectionId: chat.connectionId, profile: chat.profile })
  }

  openSessionFromPicker(item.sessionId, navigate)
}

/** The file's owner for the save bridge; this machine's own chats keep the ambient route. */
export const itemOwner = (chat: IndexedChat | undefined) =>
  chat && chat.connectionId !== 'local' ? { connectionId: chat.connectionId, profile: chat.profile } : undefined

const owner = (item: ArtifactRecord, chat: IndexedChat | undefined) => ({
  owner: itemOwner(chat) ?? { profile: chat?.profile ?? item.profile },
  profile: item.profile,
  sessionId: item.sessionId
})

/** Save the item's real bytes: a file through the owning gateway's save bridge, a web or inline
 *  image through the app's image save. */
export async function downloadItem(item: ArtifactRecord, chat: IndexedChat | undefined): Promise<void> {
  if (isArtifactFilePath(item.value)) {
    await downloadGatewayFileWithFeedback(item.value, owner(item, chat))

    return
  }

  try {
    await window.hermesDesktop?.saveImageFromUrl?.(item.href)
  } catch (error) {
    notifyError(error, translateNow('fileMenu.downloadFailed'))
  }
}

/** Open a file item the way classic's Library does: a remote gateway's file is fetched over the
 *  authenticated bridge (it is not on this disk), a local one opens with the system. */
export async function openFileItem(item: ArtifactRecord, chat: IndexedChat | undefined): Promise<void> {
  try {
    if (isRemoteGateway() && isArtifactFilePath(item.value)) {
      await downloadGatewayMediaFile(item.value, owner(item, chat))

      return
    }

    if (window.hermesDesktop?.openExternal) {
      await window.hermesDesktop.openExternal(item.href)
    } else {
      window.open(item.href, '_blank', 'noopener,noreferrer')
    }
  } catch (error) {
    notifyError(error, translateNow('artifacts.openFailed'))
  }
}
