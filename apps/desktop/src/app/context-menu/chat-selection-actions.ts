import { writeClipboardText } from '@/components/ui/copy-button'
import { resolveOwnerNow } from '@/hermes'
import { playSpeechText } from '@/lib/voice-playback'
import { $gateway } from '@/store/gateway'
import { openSelectionTranslate } from '@/store/selection-translate'
import { $activeSessionId } from '@/store/session'
import { profileScopeForSessionOwner, requestForSessionProfile } from '@/store/session-request-router'
import { knownOwnerForSession } from '@/store/session-states'

import { type ChatSelection, chatSelectionIsCurrent } from './chat-selection'

export type ChatSelectionAction = 'copy' | 'lookup' | 'read-aloud' | 'translate'

export async function runChatSelectionAction(action: ChatSelectionAction, selection: ChatSelection): Promise<boolean> {
  if (!chatSelectionIsCurrent(selection)) {
    return false
  }

  if (action === 'copy') {
    await writeClipboardText(selection.text)

    return true
  }

  if (action === 'lookup') {
    return (
      (await window.hermesDesktop?.contextMenuLookUp?.({ text: selection.text, sessionId: selection.sessionId })) ===
      true
    )
  }

  const resolvedOwner = knownOwnerForSession(selection.sessionId)
  const owner = resolvedOwner && typeof resolvedOwner === 'object' ? { ...resolvedOwner } : resolvedOwner

  // A secondary pane with no ownership record cannot borrow the active chat's credentials.
  if (owner === undefined && selection.sessionId !== $activeSessionId.get()) {
    return false
  }

  if (action === 'read-aloud') {
    const scope = profileScopeForSessionOwner(owner)

    const resolved = resolveOwnerNow(
      typeof scope === 'object' && scope ? scope : { profile: typeof scope === 'string' ? scope : undefined }
    )

    return playSpeechText(selection.text, { ...resolved, messageId: 'selection-read-aloud', source: 'read-aloud' })
  }

  const gateway = $gateway.get()

  if (!gateway) {
    return false
  }

  const ambient = gateway.request.bind(gateway)
  openSelectionTranslate(selection.text, {
    sessionId: selection.sessionId,
    request: (method, params) => requestForSessionProfile(owner, ambient, method, params)
  })

  return true
}
