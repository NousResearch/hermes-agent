import { useStore } from '@nanostores/react'
import { useEffect, useMemo } from 'react'

import type { ArtifactRecord as IndexedItem } from '@/app/artifacts/artifact-utils'
import type { IndexedChat } from '@/app/artifacts/item-index-snapshot'
import { useAgentRoutines } from '@/app/routines/routines-by-agent'
import type { CronJob } from '@/hermes'
import { $artifactRegistry, type ArtifactRecord } from '@/store/artifacts'
import { $activeConnectionId } from '@/store/connections'
import { $itemIndex, ensureChatIndexed, ownerKeyOf, peekOwnerIndex } from '@/store/item-index'

import type { ChatOwner } from '../use-chat-owner'

export interface ChatItems {
  /** Pages, graphics and code the transcript promoted to artifacts (newest first). */
  artifacts: ArtifactRecord[]
  /** Images and files from the shared item index (newest first). */
  files: IndexedItem[]
  links: IndexedItem[]
  chat: IndexedChat | undefined
  /** The routines of the chat's agent. */
  routines: CronJob[]
  /** Everything above: the one count the simple-mode button and the pane's head both show. */
  total: number
}

type ChatOwnItems = Omit<ChatItems, 'routines' | 'total'>

const NONE: ChatOwnItems = { artifacts: [], files: [], links: [], chat: undefined }

/** One chat's items: its artifacts from the transcript's registry, its images, files and links from
 *  the item index the Library shares (store/item-index.ts), read from the index of the machine that
 *  owns the chat and indexed on first sight when the index has not read it yet; and its agent's
 *  routines, which a draft has too. */
export function useChatItems(sessionId: null | string, owner: ChatOwner): ChatItems {
  const active = useStore($activeConnectionId)
  const indexOwner = ownerKeyOf(owner.connectionId ?? active)
  const index = useStore($itemIndex)[indexOwner] ?? peekOwnerIndex(indexOwner)
  const registry = useStore($artifactRegistry)
  const routines = useAgentRoutines(owner)
  const { row } = owner

  useEffect(() => {
    if (row) {
      void ensureChatIndexed(row, indexOwner)
    }
  }, [indexOwner, row])

  const own = useMemo(() => {
    if (!sessionId) {
      return NONE
    }

    // Unknown (undefined) matches by id alone. A known owner is compared as the index stores it, where
    // a row with no profile is the default's.
    const profile = owner.profile ?? (row ? row.profile : undefined)

    const chat = Object.values(index.snapshot.chats).find(
      entry => entry.sessionId === sessionId && (profile === undefined || entry.profile === (profile || 'default'))
    )

    const items = chat?.items ?? []
    const artifacts = [...(registry[sessionId] ?? [])].sort((left, right) => right.updatedAt - left.updatedAt)
    const files = items.filter(item => item.kind !== 'link')
    const links = items.filter(item => item.kind === 'link')

    return { artifacts, files, links, chat }
  }, [index.snapshot.chats, owner.profile, registry, row, sessionId])

  return {
    ...own,
    routines,
    total: own.artifacts.length + own.files.length + own.links.length + routines.length
  }
}
