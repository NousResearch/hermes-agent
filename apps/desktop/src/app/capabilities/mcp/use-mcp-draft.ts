// The mcp.json document the editor shows: the draft text, the cursor that
// selects a server block inside it, and the two ways a write reaches it.
//
// Split out of `use-mcp-servers.ts` because it answers a different question.
// This half knows only about TEXT — what the person is editing and where their
// cursor is. The other half knows about the fleet: what is configured, what
// answers, what a save does to the backend. They meet at `resetDraft` (the
// backend won, redraw the document) and `patchDraft` (the backend won, but the
// person is mid-edit, so mirror the one field and keep their words).

import { type RefObject, useEffect, useMemo, useRef, useState } from 'react'

import { type CodeEditorApi } from '@/components/chat/code-editor'
import { type McpImportEntry } from '@/lib/mcp-import'
import { getServers, type McpServers } from '@/lib/mcp-servers'
import type { HermesConfigRecord } from '@/types/hermes'

import { parseServersDoc, scanServerBlocks, type ServerBlock, STARTER_ENTRY, uniqueServerKey, wrapDoc } from './mcp-doc'

export interface McpDraft {
  activeBlock: null | ServerBlock
  /** Seed a starter entry under a fresh key and put the cursor in it. */
  addServer: () => void
  blocks: ServerBlock[]
  cursor: number
  dirty: boolean
  docVersion: number
  draft: string
  /** The draft parsed, falling back to the saved map while it is invalid. */
  draftBase: () => McpServers
  editorApi: RefObject<CodeEditorApi | null>
  focusServer: (name: string) => void
  importServers: (entries: McpImportEntry[]) => void
  /** Mirror one backend write into a dirty draft, keeping the person's words. */
  patchDraft: (mutate: (doc: McpServers) => McpServers) => void
  /** Forget everything: a profile switch, where even a dirty draft must go. */
  reset: () => void
  /** Redraw the document from the saved map; the draft stops being dirty. */
  resetDraft: (entries: McpServers) => void
  selected: null | string
  setCursor: (cursor: number) => void
  setDraft: (draft: string) => void
}

export interface UseMcpDraftOptions {
  config: HermesConfigRecord | null | undefined
  /** Document order of the saved servers; `[]` before the first config lands. */
  names: string[]
  /** A profile switch is in flight: the config still holds the OTHER profile. */
  profilePending: boolean
  servers: McpServers
  /** Refuse a write while a switch is in flight. */
  writable: boolean
}

export function useMcpDraft({ config, names, profilePending, servers, writable }: UseMcpDraftOptions): McpDraft {
  // Master document draft. `docVersion` remounts the editor when the draft is
  // regenerated programmatically (list-side mutations); `dirty` guards user
  // edits from being clobbered by those regenerations.
  const [draft, setDraftText] = useState('')
  const [dirty, setDirty] = useState(false)
  const [docVersion, setDocVersion] = useState(0)

  // Selection IS the editor cursor: whichever server block contains it is the
  // configured server. Cursor outside every block → the list.
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
      // Land just inside the key so the block claims the cursor.
      editorApi.current?.setCursor(block.from + 1)
      setCursor(block.from + 1)
    }
  }

  const resetDraft = (entries: McpServers) => {
    setDraftText(wrapDoc(entries))
    setDirty(false)
    setDocVersion(version => version + 1)
  }

  // Mirror a list-side mutation into a dirty draft without losing the user's
  // other edits. Unparseable drafts are left alone — save resolves the race.
  const patchDraft = (mutate: (doc: McpServers) => McpServers) => {
    try {
      setDraftText(wrapDoc(mutate(parseServersDoc(draft))))
      setDocVersion(version => version + 1)
    } catch {
      // Draft is mid-edit / invalid JSON; the user's text wins until save.
    }
  }

  // Seed the editor draft from config exactly once, the first time it lands.
  // Background refetches thereafter update the list but must not clobber an
  // in-progress edit — the draft is the user's until they save or reset.
  const draftSeeded = useRef(false)

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    // profilePending: config still holds the PREVIOUS profile's record right
    // after a switch — seeding from it would latch the wrong profile's doc.
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

    // Heal the early-boot race: the first config snapshot can land before the
    // backend has mcp_servers assembled, seeding (and latching) an empty doc
    // while later refetches fill the list — saving would then wipe the real
    // servers. A PRISTINE empty draft reseeds when servers arrive; any user
    // edit (dirty) still always wins.
    try {
      if (Object.keys(parseServersDoc(draft)).length === 0) {
        resetDraft(servers)
      }
    } catch {
      // Mid-edit / invalid JSON — the user's text wins.
    }
  }, [config, dirty, draft, names, profilePending, servers])

  /** Put the cursor in a freshly written block once the editor remounts. */
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

  // "+" seeds a starter entry into the document (unique key) and marks it
  // dirty — naming happens in the editor, like every other mcp.json.
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

  // Paste-anything import: merge parsed entries into the draft exactly like
  // addServer seeds its starter — dirty draft, unique keys, focus the first
  // new block. Saving stays an explicit step, so the user can fix placeholder
  // env values (YOUR_KEY, …) in the editor first.
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
    draftBase,
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
