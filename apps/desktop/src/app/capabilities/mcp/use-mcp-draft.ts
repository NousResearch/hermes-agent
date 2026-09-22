
import { type RefObject, useEffect, useMemo, useRef, useState } from 'react'

import { type CodeEditorApi } from '@/components/chat/code-editor'
import { type McpImportEntry } from '@/lib/mcp-import'
import { getServers, type McpServers } from '@/lib/mcp-servers'
import type { HermesConfigRecord } from '@/types/hermes'

import { parseServersDoc, scanServerBlocks, type ServerBlock, STARTER_ENTRY, uniqueServerKey, wrapDoc } from './mcp-doc'

export interface McpDraft {
  activeBlock: null | ServerBlock
  addServer: () => void
  blocks: ServerBlock[]
  cursor: number
  dirty: boolean
  docVersion: number
  draft: string
  editorApi: RefObject<CodeEditorApi | null>
  focusServer: (name: string) => void
  importServers: (entries: McpImportEntry[]) => void
  patchDraft: (mutate: (doc: McpServers) => McpServers) => void
  reset: () => void
  resetDraft: (entries: McpServers) => void
  selected: null | string
  setCursor: (cursor: number) => void
  setDraft: (draft: string) => void
}

export interface UseMcpDraftOptions {
  config: HermesConfigRecord | null | undefined
  names: string[]
  profilePending: boolean
  servers: McpServers
  writable: boolean
}

export function useMcpDraft({ config, names, profilePending, servers, writable }: UseMcpDraftOptions): McpDraft {
  const [draft, setDraftText] = useState('')
  const [dirty, setDirty] = useState(false)
  const [docVersion, setDocVersion] = useState(0)

  const editorApi = useRef<CodeEditorApi | null>(null)
  const [cursor, setCursor] = useState(0)
  const blocks = useMemo(() => scanServerBlocks(draft), [draft])

  const activeBlock = useMemo(
    () => blocks.find(block => cursor >= block.from && cursor <= block.to) ?? null,
    [blocks, cursor]
  )

  const selected = activeBlock?.name ?? null

  const focusServer = (name: string) => {
    const block = blocks.find(b => b.name === name)

    if (block) {
      editorApi.current?.setCursor(block.from + 1)
      setCursor(block.from + 1)
    }
  }

  const resetDraft = (entries: McpServers) => {
    setDraftText(wrapDoc(entries))
    setDirty(false)
    setDocVersion(version => version + 1)
  }

  const patchDraft = (mutate: (doc: McpServers) => McpServers) => {
    try {
      setDraftText(wrapDoc(mutate(parseServersDoc(draft))))
      setDocVersion(version => version + 1)
      // eslint-disable-next-line no-empty
    } catch {
    }
  }

  const draftSeeded = useRef(false)

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    if (!config || profilePending) {
      return
    }

    if (!draftSeeded.current) {
      draftSeeded.current = true
      resetDraft(getServers(config))

      return
    }

    if (dirty || names.length === 0) {
      return
    }

    try {
      if (Object.keys(parseServersDoc(draft)).length === 0) {
        resetDraft(servers)
      }
      // eslint-disable-next-line no-empty
    } catch {
    }
  }, [config, dirty, draft, names, profilePending, servers])

  const focusKeyIn = (nextDraft: string, key: string) => {
    const from = nextDraft.indexOf(`"${key}"`)

    if (from >= 0) {
      requestAnimationFrame(() => {
        editorApi.current?.setCursor(from + 1)
        setCursor(from + 1)
      })
    }
  }

  const draftBase = (): McpServers => {
    try {
      return parseServersDoc(draft)
    } catch {
      return { ...servers }
    }
  }

  const addServer = () => {
    if (!writable) {
      return
    }

    const base = draftBase()
    const key = uniqueServerKey(base, 'my-server')
    const nextDraft = wrapDoc({ ...base, [key]: STARTER_ENTRY })

    setDraftText(nextDraft)
    setDirty(true)
    setDocVersion(version => version + 1)
    focusKeyIn(nextDraft, key)
  }

  const importServers = (entries: McpImportEntry[]) => {
    if (!writable || entries.length === 0) {
      return
    }

    let base = draftBase()
    let firstKey: null | string = null

    for (const entry of entries) {
      const key = uniqueServerKey(base, entry.name)
      base = { ...base, [key]: entry.config }
      firstKey ??= key
    }

    const nextDraft = wrapDoc(base)
    setDraftText(nextDraft)
    setDirty(true)
    setDocVersion(version => version + 1)

    if (firstKey) {
      focusKeyIn(nextDraft, firstKey)
    }
  }

  const reset = () => {
    draftSeeded.current = false
    setCursor(0)
    setDirty(false)
    setDraftText('')
    setDocVersion(version => version + 1)
  }

  const setDraft = (next: string) => {
    setDraftText(next)
    setDirty(true)
  }

  return {
    activeBlock,
    addServer,
    blocks,
    cursor,
    dirty,
    docVersion,
    draft,
    editorApi,
    focusServer,
    importServers,
    patchDraft,
    reset,
    resetDraft,
    selected,
    setCursor,
    setDraft
  }
}
