/**
 * SESSION CHAT — the real chat surface for a session that isn't the workspace
 * pane, with none of the chrome that says where it lives.
 *
 * This is the whole stack the primary chat uses (`ChatView` → `Thread` +
 * `ChatBar`), mounted under this session's own `SessionView` (its slice of
 * `$sessionStates`) and its own `ComposerScope` (own attachment chips, own
 * focus-bus key), with actions from `useSessionTileActions`.
 *
 * It was `TileChat`, inside `session-tile.tsx`, and the extraction is the
 * point: rendering a conversation and owning a pane are separate jobs, and
 * only the pane part was ever tile-specific. A tile mounts this in a
 * layout-tree pane; a detached surface (the Workflows canvas) mounts it inside
 * itself. Neither gets a copy — the transcript, the tool cards, the streaming
 * indicators, attachments, voice and the model menu are the same code in both,
 * which is the only way they stay the same as the app moves.
 *
 * Presentation is the caller's job, and CSS is how it's done: HUD mode already
 * restyles this exact tree through `[data-hud-shell]` on an ancestor rather
 * than variant props. A surface that wants a different chat layout marks its
 * own container and styles from there.
 */

import { useStore } from '@nanostores/react'
import { useQueryClient } from '@tanstack/react-query'
import { useCallback, useMemo, useRef } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { useModelControls } from '@/app/session/hooks/use-model-controls'
import { blobToDataUrl } from '@/app/session/hooks/use-prompt-actions/utils'
import { ModelMenuPanel } from '@/app/shell/model-menu-panel'
import { formatRefValue } from '@/components/assistant-ui/directive-text'
import { transcribeAudio } from '@/hermes'
import { transcribeAudioClientDirect } from '@/lib/voice-client-direct'
import { createComposerAttachmentScope } from '@/store/composer'
import { $activeGatewayProfile } from '@/store/profile'
import { sessionAwaitingInput } from '@/store/prompts'
import { $gatewayState } from '@/store/session'
import { requestForSessionProfile, type SessionProfileRoute } from '@/store/session-request-router'

import { type ComposerScope, ComposerScopeProvider } from './composer/scope'
import { useComposerActions } from './hooks/use-composer-actions'
import { useSessionTileActions } from './session-tile-actions'
import { type SessionView, SessionViewProvider } from './session-view'

import { ChatView } from '.'

// Module-level so these ChatView props are referentially stable — a surface
// like this has no pin/delete affordance, and transcription needs no per-chat
// state.
const noop = () => undefined

const chatTranscribeAudio = async (audio: Blob) => {
  // Client-direct first (profile's own STT provider, no gateway audio hop);
  // relay when the provider is not client-callable. Same ladder as the main
  // composer's transcribeVoiceAudio.
  const direct = await transcribeAudioClientDirect(audio)

  if (direct !== null) {
    return direct
  }

  return (await transcribeAudio(await blobToDataUrl(audio), audio.type)).transcript
}

export interface SessionChatProps {
  /** Durable identity. Also the composer scope + attachment key. */
  storedSessionId: string
  /** The live runtime this session is bound to. The caller resumes. */
  runtimeId: string
  view: SessionView
  /** Backend that owns the session, when it isn't the ambient one. A tile
   *  keeps its persisted route; a local detached chat has none. */
  ownerRoute?: SessionProfileRoute
  /** The runtime was reaped mid-sequence and re-resumed onto a new id. The
   *  caller owns where that gets recorded (tile registry, plugin storage). */
  onRuntimeBound?: (runtimeId: string) => void
  /** Rendered in the composer's model pill. Omit for no model menu. */
  modelMenu?: boolean
  /** The thread's "retry resuming this session" affordance. */
  onRetryResume?: () => void
}

export function SessionChat({
  modelMenu = true,
  onRetryResume = noop,
  onRuntimeBound,
  ownerRoute,
  runtimeId,
  storedSessionId,
  view
}: SessionChatProps) {
  const { gateway, requestGateway } = useGatewayRequest()
  const queryClient = useQueryClient()

  const requestOwnGateway = useCallback(
    <T,>(method: string, params?: Record<string, unknown>, timeoutMs?: number, signal?: AbortSignal): Promise<T> =>
      requestForSessionProfile<T>(ownerRoute, requestGateway, method, params, timeoutMs, signal),
    [ownerRoute, requestGateway]
  )

  const { selectModel } = useModelControls({
    cacheOwnerConnectionId: ownerRoute?.connectionId || undefined,
    cacheProfile: ownerRoute?.targetProfile || ownerRoute?.profile || undefined,
    queryClient,
    requestGateway: requestOwnGateway
  })

  const activeGatewayProfile = useStore($activeGatewayProfile)
  const cwd = useStore(view.$cwd)
  const gatewayOpen = useStore($gatewayState) === 'open'

  // One attachment set + focus key per surface, stable for its lifetime.
  const attachments = useRef(createComposerAttachmentScope()).current

  const scope = useMemo<ComposerScope>(
    () => ({
      $awaitingInput: sessionAwaitingInput(runtimeId),
      $messages: view.$messages,
      attachments,
      target: `session:${storedSessionId}`
    }),
    [attachments, runtimeId, storedSessionId, view.$messages]
  )

  // Actions must keep the persisted owner route. The ambient gateway hook
  // follows foreground focus and can point at another backend during restore
  // or reconnect, which turns a recoverable stale runtime into "session not
  // found".
  const actions = useSessionTileActions({
    onRuntimeBound,
    requestGateway: requestOwnGateway,
    runtimeId,
    scope,
    storedSessionId
  })

  // The same attach/pick/paste/drop pipeline the primary composer uses,
  // pointed at this surface's chips + session.
  const composer = useComposerActions({
    activeSessionId: runtimeId,
    currentCwd: cwd,
    requestGateway: requestOwnGateway,
    scope: {
      add: attachments.add,
      remove: attachments.remove,
      target: scope.target,
      update: attachments.update,
      updateIfCurrent: attachments.updateIfCurrent
    }
  })

  // ChatView is memo()d — every callback prop must be referentially stable or
  // the memo never holds and each host render (idle ticks, unrelated store
  // updates) re-renders the whole chat shell. The individual composer
  // functions are useCallback'd inside useComposerActions, so hoisting these
  // wrappers onto them keeps identity stable across renders.
  const { addContextRefAttachment, pasteClipboardImage, pickContextPaths, pickImages, removeAttachment } = composer

  const onAddUrl = useCallback(
    (url: string) => addContextRefAttachment(`@url:${formatRefValue(url)}`, url),
    [addContextRefAttachment]
  )

  const onPasteClipboardImage = useCallback(
    (opts?: { silent?: boolean }) => pasteClipboardImage(opts),
    [pasteClipboardImage]
  )

  const onPickFiles = useCallback(() => void pickContextPaths('file'), [pickContextPaths])
  const onPickFolders = useCallback(() => void pickContextPaths('folder'), [pickContextPaths])
  const onPickImages = useCallback(() => void pickImages(), [pickImages])
  const onRemoveAttachment = useCallback((id: string) => void removeAttachment(id), [removeAttachment])

  // Rendered under THIS SessionView so the pill + switch target this runtime,
  // not the primary (which may be mid-turn).
  const modelMenuContent = useMemo(
    () =>
      modelMenu && gatewayOpen ? (
        <ModelMenuPanel
          onSelectModel={selectModel}
          ownerConnectionId={ownerRoute?.connectionId || undefined}
          profile={ownerRoute?.targetProfile || ownerRoute?.profile || activeGatewayProfile}
          requestGateway={requestOwnGateway}
        />
      ) : null,
    [
      activeGatewayProfile,
      gatewayOpen,
      modelMenu,
      ownerRoute?.connectionId,
      ownerRoute?.profile,
      ownerRoute?.targetProfile,
      requestOwnGateway,
      selectModel
    ]
  )

  return (
    <SessionViewProvider value={view}>
      <ComposerScopeProvider value={scope}>
        <ChatView
          gateway={gateway}
          modelMenuContent={modelMenuContent}
          modelOptionsOwnerConnectionId={ownerRoute?.connectionId || undefined}
          modelOptionsProfile={ownerRoute?.targetProfile || ownerRoute?.profile || activeGatewayProfile}
          onAddContextRef={addContextRefAttachment}
          onAddUrl={onAddUrl}
          onAttachDroppedItems={composer.attachDroppedItems}
          onAttachImageBlob={composer.attachImageBlob}
          onAttachPrCommentUrl={composer.attachPrCommentUrl}
          onCancel={actions.cancelRun}
          onDeleteSelectedSession={noop}
          onDismissError={actions.dismissError}
          onEdit={actions.editMessage}
          onPasteClipboardImage={onPasteClipboardImage}
          onPickFiles={onPickFiles}
          onPickFolders={onPickFolders}
          onPickImages={onPickImages}
          onReload={actions.reloadFromMessage}
          onRemoveAttachment={onRemoveAttachment}
          onRestoreToMessage={actions.restoreToMessage}
          onRetryResume={onRetryResume}
          onSteer={actions.steerPrompt}
          onSubmit={actions.submitText}
          onThreadMessagesChange={actions.handleThreadMessagesChange}
          onToggleSelectedPin={noop}
          onTranscribeAudio={chatTranscribeAudio}
          requestModelOptionsForOwner={requestOwnGateway}
        />
      </ComposerScopeProvider>
    </SessionViewProvider>
  )
}
