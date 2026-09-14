import { useStore } from '@nanostores/react'
import { useRef, useState } from 'react'

import { ArchiveSkillConfirmDialog, fireOptimistic } from '@/app/learning/archive-skill-confirm-dialog'
import { CodeEditor } from '@/components/chat/code-editor'
import { Button } from '@/components/ui/button'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { deleteLearningNode, editLearningNode, getLearningNode } from '@/hermes'
import { useI18n } from '@/i18n'
import { notifyError } from '@/store/notifications'
import { $profiles, normalizeProfileKey, profileLabel } from '@/store/profile'
import { $starmapSelectedProfiles, evictStarmapNode, loadStarmapGraph } from '@/store/starmap'

import { useOnProfileSwitch } from '../hooks/use-on-profile-switch'

import { isProviderSource } from './sources'

export interface NodeMenuTarget {
  id: string
  /** True when this node is a Honcho conclusion (durable derived fact). Adds a
   *  "Start a conversation about this" action; conclusions are provider-backed
   *  so they stay read-only (no Edit/Delete). */
  isConclusion?: boolean
  kind: 'memory' | 'skill'
  label: string
  /** Memory nodes only: 'memory' | 'profile' | a provider name ('honcho', …).
   *  Provider-backed nodes are read-only — the menu offers no Edit/Delete. */
  memorySource?: string
  /** Multi-profile mode: which profile this node belongs to. */
  profile?: string
  /** Multi-profile mode: the original node id without the profile prefix. */
  _originalId?: string
  x: number
  y: number
}

/** One recent session offered in the "Add to a session" submenu. `key` is the
 *  durable composer scope (lineage root) the draft is stashed under; `title` is
 *  the display label. */
export interface RecallSessionOption {
  key: string
  title: string
}

interface NodeContextMenuProps {
  onClose: () => void
  onNodeRemoved: () => void
  /** Open the provenance dialog ("Where this came from…") for this node. */
  onShowProvenance: (id: string) => void
  /** Conclusion nodes only: seed a NEW chat about this conclusion (for review
   *  — never auto-sent). Absent when the host doesn't support it. */
  onStartConversation?: (target: NodeMenuTarget) => void
  /** Feature A — "Add to a session": stash this node's knowledge (as a reviewed,
   *  injection-hardened draft) into an existing session's composer. Given the
   *  durable session key. Available for ALL node kinds. */
  onAddToSession?: (target: NodeMenuTarget, sessionKey: string) => void
  /** Feature B — /recall: insert this node's knowledge into the CURRENT chat's
   *  composer for review. Available for ALL node kinds. */
  onRecallIntoChat?: (target: NodeMenuTarget) => void
  /** Cross-profile insert: copy this node's content into another profile's memory.
   *  Available when in multi-profile mode and the node kind is 'memory'. */
  onInsertIntoProfile?: (target: NodeMenuTarget, profileName: string) => void
  /** Most-recently-active sessions (already capped + ordered) for the
   *  "Add to a session" submenu. Empty/undefined hides that action. */
  recentSessions?: RecallSessionOption[]
  target: NodeMenuTarget | null
}

interface EditState {
  content: string
  id: string
  label: string
}

/** Right-click actions for a star-map node: provenance, edit (modal), delete (confirm).
 *  Provider-backed memory nodes are read-only (their storage lives in the
 *  provider's backend), so Edit/Delete are replaced by a hint. */
export function NodeContextMenu({ onAddToSession, onClose, onInsertIntoProfile, onNodeRemoved, onRecallIntoChat, onShowProvenance, onStartConversation, recentSessions, target }: NodeContextMenuProps) {
  const { t } = useI18n()
  const profiles = useStore($profiles)
  const selectedProfiles = useStore($starmapSelectedProfiles)
  const [editing, setEditing] = useState<EditState | null>(null)
  const [deleting, setDeleting] = useState<Omit<NodeMenuTarget, 'x' | 'y'> | null>(null)
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<null | string>(null)

  // In multi-profile mode, get the profiles the node can be inserted into
  // (all selected profiles except the node's own profile).
  // Guard against profiles not being an array (defensive)
  const crossProfileTargets = Array.isArray(profiles) && Array.isArray(selectedProfiles)
    ? profiles.filter(p => {
        const key = normalizeProfileKey(p.name)

        return selectedProfiles.includes(key) && key !== target?.profile
      })
    : []

  // Bumped on profile switch so an in-flight openEdit fetch from profile A can't
  // reopen the editor with A's node content after switching to B.
  const editEpoch = useRef(0)

  // A profile switch swaps the backend under an open edit/delete dialog — its
  // node id belongs to the previous profile, so a Save/Delete after the switch
  // would hit the newly active profile. Close everything on switch.
  useOnProfileSwitch(() => {
    editEpoch.current += 1
    setEditing(null)
    setDeleting(null)
    setError(null)
  })

  const noun = target?.kind === 'memory' ? 'memory' : 'skill'

  const openEdit = async () => {
    if (!target) {
      return
    }

    const epoch = editEpoch.current
    setLoading(true)
    setError(null)

    try {
      const detail = await getLearningNode(target.id)

      if (editEpoch.current !== epoch) {
        return
      }

      setEditing({ content: detail.content, id: target.id, label: target.label })
      onClose()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  const save = async () => {
    if (!editing) {
      return
    }

    setSaving(true)
    setError(null)

    try {
      const res = await editLearningNode(editing.id, editing.content)

      if (!res.ok) {
        throw new Error(res.message)
      }

      setEditing(null)
      void loadStarmapGraph(true)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  const menuOpen = target && !editing && !deleting

  return (
    <>
      {menuOpen ? (
        <DropdownMenu onOpenChange={open => !open && onClose()} open>
          <DropdownMenuTrigger asChild>
            {/* A zero-size anchor at the canvas click point, as AppContextMenu
                does: Radix positions against it like a real trigger and flips or
                shifts the menu back inside the viewport near the window edges,
                so the destructive row can never be clipped off-window. */}
            <span aria-hidden style={{ left: target.x, position: 'fixed', top: target.y }} />
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" onCloseAutoFocus={e => e.preventDefault()} side="bottom">
            <DropdownMenuLabel className="truncate text-[0.68rem] font-normal text-muted-foreground">
              {target.label}
            </DropdownMenuLabel>
            <DropdownMenuItem onSelect={() => onShowProvenance(target.id)}>{t.starmap.provenanceMenu}</DropdownMenuItem>
            {/* Feature B — /recall: insert this node's knowledge into THIS chat.
                Offered for every node kind. The host fetches an
                injection-hardened, provenance-tagged draft and inserts it into
                the live composer for the user to review + send. */}
            {onRecallIntoChat ? (
              <DropdownMenuItem onSelect={() => onRecallIntoChat(target)}>{t.starmap.recallIntoChat}</DropdownMenuItem>
            ) : null}
            {/* Feature A — "Add to a session ▸": stash this node's knowledge into
                one of the most-recently-active sessions' composers. A real Radix
                submenu now that the menu is anchored through DropdownMenu. */}
            {onAddToSession && recentSessions && recentSessions.length > 0 ? (
              <DropdownMenuSub>
                <DropdownMenuSubTrigger>{t.starmap.addToSession}</DropdownMenuSubTrigger>
                <DropdownMenuSubContent>
                  {recentSessions.map(session => (
                    <DropdownMenuItem
                      className="max-w-72 truncate"
                      key={session.key}
                      onSelect={() => onAddToSession(target, session.key)}
                      title={session.title}
                    >
                      {session.title}
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuSubContent>
              </DropdownMenuSub>
            ) : null}
            {/* Cross-profile insert — in multi-profile mode, offer to insert this
                node's content into another selected profile's memory. */}
            {onInsertIntoProfile && target.profile && crossProfileTargets.length > 0 ? (
              <DropdownMenuSub>
                <DropdownMenuSubTrigger>
                  {t.starmap.insertIntoProfile(crossProfileTargets.length === 1 ? profileLabel(crossProfileTargets[0]) : '…')}
                </DropdownMenuSubTrigger>
                <DropdownMenuSubContent>
                  {crossProfileTargets.length > 1 ? (
                    <DropdownMenuItem
                      className="font-medium"
                      onSelect={() => crossProfileTargets.forEach(p => onInsertIntoProfile(target, normalizeProfileKey(p.name)))}
                    >
                      {t.starmap.insertIntoAllSelected}
                    </DropdownMenuItem>
                  ) : null}
                  {crossProfileTargets.map(profile => (
                    <DropdownMenuItem
                      className="max-w-72 truncate"
                      key={normalizeProfileKey(profile.name)}
                      onSelect={() => onInsertIntoProfile(target, normalizeProfileKey(profile.name))}
                      title={profileLabel(profile)}
                    >
                      {profileLabel(profile)}
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuSubContent>
              </DropdownMenuSub>
            ) : null}
            {isProviderSource(target.memorySource) ? (
              <>
                {target.isConclusion && onStartConversation ? (
                  <DropdownMenuItem onSelect={() => onStartConversation(target)}>
                    {t.starmap.conclusionStartConversation}
                  </DropdownMenuItem>
                ) : null}
                <div className="max-w-56 px-2 py-1 text-[0.68rem] text-muted-foreground">
                  {t.starmap.providerReadOnly(target.memorySource)}
                </div>
              </>
            ) : (
              <>
                <DropdownMenuItem
                  disabled={loading}
                  onSelect={e => {
                    // Keep the menu up while the node content loads; openEdit closes it.
                    e.preventDefault()
                    void openEdit()
                  }}
                >
                  Edit {noun}…
                </DropdownMenuItem>
                <DropdownMenuItem
                  onSelect={() => setDeleting({ id: target.id, kind: target.kind, label: target.label })}
                  variant="destructive"
                >
                  {target.kind === 'skill' ? 'Archive skill' : 'Delete memory'}
                </DropdownMenuItem>
              </>
            )}
          </DropdownMenuContent>
        </DropdownMenu>
      ) : null}

      <Dialog onOpenChange={value => !value && !saving && setEditing(null)} open={Boolean(editing)}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Edit {editing?.label}</DialogTitle>
          </DialogHeader>
          <div className="h-80">
            {editing && (
              <CodeEditor
                filePath={noun === 'skill' ? 'SKILL.md' : 'memory.md'}
                framed
                initialValue={editing.content}
                key={editing.id}
                onCancel={() => !saving && setEditing(null)}
                onChange={content => setEditing(prev => (prev ? { ...prev, content } : prev))}
                onSave={() => void save()}
              />
            )}
          </div>
          {error ? <p className="text-xs text-destructive">{error}</p> : null}
          <DialogFooter>
            <Button disabled={saving} onClick={() => setEditing(null)} type="button" variant="ghost">
              Cancel
            </Button>
            <Button disabled={saving} onClick={() => void save()}>
              {saving ? 'Saving…' : 'Save'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {deleting?.kind === 'skill' ? (
        <ArchiveSkillConfirmDialog
          onApply={() => {
            onNodeRemoved()

            return evictStarmapNode(deleting.id)
          }}
          onClose={() => setDeleting(null)}
          onFailure={(err, name) => notifyError(err, name)}
          open
          skillId={deleting.id}
          skillName={deleting.label}
        />
      ) : (
        <ConfirmDialog
          confirmLabel="Delete"
          description="This memory is removed permanently."
          destructive
          dismissOnConfirm
          onClose={() => setDeleting(null)}
          onConfirm={() => {
            if (!deleting) {
              return
            }

            const { id, label } = deleting
            const rollback = evictStarmapNode(id)
            onNodeRemoved()

            fireOptimistic(
              deleteLearningNode(id).then(res => {
                if (!res.ok) {
                  throw new Error(res.message)
                }
              }),
              rollback,
              err => notifyError(err, label)
            )
          }}
          open={Boolean(deleting)}
          title={`Delete ${deleting?.label ?? ''}?`}
        />
      )}
    </>
  )
}
