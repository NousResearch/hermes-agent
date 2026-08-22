import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { requestComposerInsert } from '@/app/chat/composer/focus'
import { PageLoader } from '@/components/page-loader'
import { crossInsertLearningNode, getLearningRecallDraft } from '@/hermes'
import { useI18n } from '@/i18n'
import { sessionTitle } from '@/lib/chat-runtime'
import { stashSessionDraft, takeSessionDraft } from '@/store/composer'
import { notify } from '@/store/notifications'
import { requestFreshSession } from '@/store/profile'
import { $sessions } from '@/store/session'
import { openSessionTile } from '@/store/session-states'
import {
  $starmapError,
  $starmapGraph,
  $starmapLoading,
  $starmapRecallMode,
  clearStarmapRecall,
  loadStarmapGraph
} from '@/store/starmap'
import type { StarmapGraph } from '@/types/hermes'

import { Panel, PanelEmpty } from '../overlays/panel'

import { type RecallNodeRef, resolveRecallTarget } from './recall'
import { StarMap } from './star-map'

// How many most-recently-active sessions the "Add to a session" submenu offers.
const RECALL_RECENT_SESSIONS = 5

// Star map overlay: a top-down map of what Hermes has learned for a profile,
// over a radial time axis. Data is fetched on demand into the $starmap* atoms;
// the map itself lives in ./star-map. The chrome is owned by the map itself
// (timeline scrubber + legend float over the canvas), so there's no panel
// header here.
export function StarmapView({ onClose }: { onClose: () => void }) {
  const { t } = useI18n()
  const graph = useStore($starmapGraph)
  const loading = useStore($starmapLoading)
  const error = useStore($starmapError)
  const sessions = useStore($sessions)
  // Raised by `/recall`: open the search sidebar focused and offer "insert into
  // this chat" on nodes. Cleared here on close so a later `/journey` opens plain.
  const recallMode = useStore($starmapRecallMode)

  // A pasted share code populates the map with someone else's (or an exported)
  // graph, overriding the live profile scan. Cleared by "back to my map" and
  // whenever a fresh profile graph loads in.
  const [imported, setImported] = useState<StarmapGraph | null>(null)

  useEffect(() => {
    void loadStarmapGraph()
  }, [])

  // Drop a stale import when the underlying profile graph changes out from under it.
  useEffect(() => {
    setImported(null)
  }, [graph])

  const shown = imported ?? graph

  const closeAndReset = () => {
    clearStarmapRecall()
    onClose()
  }

  // The 5 most-recently-active sessions offered in "Add to a session". The
  // $sessions atom is already recency-sorted (backend order=recent). The draft
  // is keyed by the durable lineage root so tip rotation from auto-compression
  // can't strand it on an empty key.
  const recentSessions = sessions.slice(0, RECALL_RECENT_SESSIONS).map(session => ({
    key: session._lineage_root_id ?? session.id,
    title: sessionTitle(session)
  }))

  // Fetch the injection-hardened, provenance-tagged draft for a node. The body
  // is scanned + defanged + wrapped server-side; we only place the returned
  // text. Returns null (after a toast) when the node has no recallable content.
  //
  // Multi-profile mode: node ids arrive PREFIXED (`<profile>:<id>`) and the
  // node may live in a DIFFERENT profile than the active one — resolve the
  // unprefixed id and scope the request to the node's own profile, mirroring
  // the onInsertIntoProfile contract (same bug class: the server only knows
  // original ids).
  const fetchRecallText = async (node: RecallNodeRef): Promise<null | string> => {
    const target = resolveRecallTarget(node)

    try {
      const draft = await getLearningRecallDraft(target.id, target.profile)

      if (!draft.ok || !draft.text.trim()) {
        notify({ kind: 'warning', message: t.starmap.recallFailed })

        return null
      }

      return draft.text
    } catch {
      notify({ kind: 'warning', message: t.starmap.recallFailed })

      return null
    }
  }

  return (
    <Panel closeLabel={t.starmap.close} onClose={closeAndReset}>
      {error ? (
        <PanelEmpty description={error} icon="warning" title={t.starmap.loadFailed} />
      ) : !shown && loading ? (
        <PageLoader aria-label={t.starmap.loading} className="min-h-0 flex-1" />
      ) : shown && shown.nodes.length === 0 && !imported ? (
        <PanelEmpty description={t.starmap.emptyDesc} icon="lightbulb" title={t.starmap.emptyTitle} />
      ) : shown ? (
        <StarMap
          graph={shown}
          imported={imported !== null}
          initialSearchFocus={recallMode}
          onAddToSession={async (node, sessionKey) => {
            // Feature A — stash the node's knowledge into an EXISTING session's
            // composer as a reviewed draft. Never clobber an in-progress draft:
            // read what's there and append. The user opens that session, reviews
            // the draft, and sends it (the human gate — nothing auto-sent).
            const text = await fetchRecallText(node)

            if (!text) {
              return
            }

            const existing = takeSessionDraft(sessionKey)
            const combined = existing.text.trim() ? `${existing.text.trimEnd()}\n\n${text}` : text
            stashSessionDraft(sessionKey, combined, existing.attachments)

            const target = recentSessions.find(s => s.key === sessionKey)
            notify({ kind: 'success', message: t.starmap.addToSessionDone(target?.title ?? node.label) })
          }}
          onImport={setImported}
          onInsertIntoProfile={async (node, targetProfile) => {
            // Cross-profile insert: copy this node's content into another profile's
            // MEMORY.md. The source profile is embedded in the node; we use the
            // _originalId to fetch the content without the profile prefix.
            if (!node.profile) {
              notify({ kind: 'warning', message: t.starmap.crossProfileInsertFailed })

              return
            }

            try {
              // Same addressing contract as recall: unprefixed id, node's own
              // profile (shared resolver — see recall.ts).
              const { id: nodeId } = resolveRecallTarget(node)

              const result = await crossInsertLearningNode(nodeId, node.profile, targetProfile)

              if (result.ok) {
                notify({ kind: 'success', message: t.starmap.insertIntoProfileDone(targetProfile) })
                // Refresh the graph to show the new entry
                void loadStarmapGraph(true)
              } else {
                notify({ kind: 'warning', message: result.message || t.starmap.crossProfileInsertFailed })
              }
            } catch {
              notify({ kind: 'warning', message: t.starmap.crossProfileInsertFailed })
            }
          }}
          onOpenSession={id => {
            // Drill-down from a node: open the conversation as a tile stacked
            // into the main zone, then close the map so it's readable.
            openSessionTile(id, 'center')
            closeAndReset()
          }}
          onRecallIntoChat={async node => {
            // Feature B (/recall) — insert the node's knowledge into the chat
            // the user is actually looking at, for review (never auto-sent).
            //
            // Target the ACTIVE composer, not a hardcoded 'main': the star map
            // is opened over whatever chat is on screen, and that is a session
            // TILE (data-composer-target="tile:<id>") whenever the user drilled
            // into a session — 'main' is the workspace tab sitting hidden behind
            // it. Inserting into 'main' wrote into that off-screen composer, so
            // the recall "did nothing" for anyone reading a tile. 'active'
            // resolves (and heals) to the visible chat surface — identical to
            // 'main' when the workspace tab is fronted.
            //
            // Close FIRST, then insert on the next macrotask: the overlay-close
            // navigation must settle so the insert lands in the revealed
            // composer and the composer's session-scope effect can't repaint
            // over it (the deferral the sibling onStartConversation relies on).
            const text = await fetchRecallText(node)

            if (!text) {
              return
            }

            closeAndReset()
            setTimeout(() => requestComposerInsert(text, { mode: 'block' }), 0)
          }}
          onResetMap={() => setImported(null)}
          onStartConversation={conclusion => {
            // Seed a NEW chat with the conclusion quoted as context — the user
            // reviews and sends (never auto-sent). Open the fresh session, then
            // defer the insert one tick so it lands after the chat controller
            // has reset the composer to the new session.
            requestFreshSession()
            const seed = t.starmap.conclusionSeedPrompt(conclusion.label)
            setTimeout(() => requestComposerInsert(seed, { target: 'main' }), 0)
            closeAndReset()
          }}
          recentSessions={recentSessions}
        />
      ) : null}
    </Panel>
  )
}
