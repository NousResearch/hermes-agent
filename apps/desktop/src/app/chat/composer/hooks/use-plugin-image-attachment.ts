import { type RefObject, useLayoutEffect, useRef } from 'react'

import { isBrowserHostedDesktop } from '@/lib/platform'
import { useStoreSelector } from '@/lib/use-session-slice'
import { $freshDraftKey, isFreshDraftScope } from '@/store/composer'
import { $activeConnectionId } from '@/store/connections'
import { $activeGatewayRoute } from '@/store/gateway'
import { $activeSessionId, $selectedStoredSessionId } from '@/store/session'

import { registerComposerImageRecipient } from '../plugin-image-attachment'
import { useComposerScope, useComposerSurfaceId } from '../scope'
import type { ChatBarProps } from '../types'
import { useComposerVisible } from '../visibility'

interface Options {
  attachmentBlocked: () => boolean
  composerRef: RefObject<HTMLFormElement | null>
  enabled: boolean
  focusKey: ChatBarProps['focusKey']
  onAttachImageBlob: ChatBarProps['onAttachImageBlob']
  sessionId: ChatBarProps['sessionId']
  sessionKey: string | null
}

export function usePluginImageAttachment({
  attachmentBlocked,
  composerRef,
  enabled,
  focusKey,
  onAttachImageBlob,
  sessionId,
  sessionKey
}: Options) {
  const { target, connectionId, profile } = useComposerScope()
  const surfaceId = useComposerSurfaceId()
  const visible = useComposerVisible()
  const main = target === 'main'
  const freshMain = main && (sessionKey === null || isFreshDraftScope(sessionKey))
  const ownerProfile = useStoreSelector($activeGatewayRoute, store => (main ? store : profile))
  const ownerConnection = useStoreSelector($activeConnectionId, store => (main ? store : connectionId))
  const selectedKey = useStoreSelector($selectedStoredSessionId, store => (main ? store : null))
  const activeKey = useStoreSelector($activeSessionId, store => (main ? store : null))
  const fresh = useStoreSelector($freshDraftKey, store => (freshMain ? store : null))

  const attachRef = useRef(onAttachImageBlob)
  useLayoutEffect(() => {
    attachRef.current = onAttachImageBlob
  }, [onAttachImageBlob])

  const canStage = enabled && Boolean(onAttachImageBlob) && !isBrowserHostedDesktop()

  useLayoutEffect(() => {
    if (!canStage || !visible || !surfaceId || target === 'edit') {
      return undefined
    }

    let mounted = true
    let epoch = 0

    const invalidate = () => {
      epoch += 1
    }

    // Invalidate synchronously, even if React batches an A → B → A store
    // transition into one render, or staging resolves before the next commit.
    const off = main
      ? [
          $selectedStoredSessionId.listen(invalidate),
          $activeSessionId.listen(invalidate),
          $activeGatewayRoute.listen(invalidate),
          $activeConnectionId.listen(invalidate),
          ...(fresh !== null ? [$freshDraftKey.listen(invalidate)] : [])
        ]
      : []

    const matchesMain = () =>
      $selectedStoredSessionId.get() === selectedKey &&
      $activeSessionId.get() === activeKey &&
      $activeGatewayRoute.get() === ownerProfile &&
      $activeConnectionId.get() === ownerConnection &&
      (fresh === null || $freshDraftKey.get() === fresh)

    const isCurrent = () =>
      mounted && composerRef.current?.isConnected === true && !attachmentBlocked() && (!main || matchesMain())

    const unregister = registerComposerImageRecipient({
      target,
      surfaceId,
      sessionKey,
      capture: () => {
        const capturedEpoch = epoch

        return isCurrent() ? () => capturedEpoch === epoch && isCurrent() : null
      },
      attach: (blob, guard) => attachRef.current?.(blob, guard)
    })

    return () => {
      mounted = false
      unregister()
      off.forEach(dispose => dispose())
    }
  }, [
    activeKey,
    attachmentBlocked,
    canStage,
    composerRef,
    focusKey,
    fresh,
    main,
    ownerConnection,
    ownerProfile,
    selectedKey,
    sessionId,
    sessionKey,
    surfaceId,
    target,
    visible
  ])
}
