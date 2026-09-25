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
import { evictStarmapNode, loadStarmapGraph, $starmapSelectedProfiles } from '@/store/starmap'

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
  onShowProvenance?: (id: string) => void
  /** Conclusion nodes only: seed a NEW chat about this conclusion (for review
   *  — never auto-sent). Absent when the host doesn't support it. */
  onStartConversation?: (target: NodeMenuTarget) => void
  /** "Add to a session": stash this node's knowledge (as a reviewed,
   *  injection-hardened draft) into an existing session's composer. Given the
   *  durable session key. Available for ALL node kinds. */
  onAddToSession?: (target: NodeMenuTarget, sessionKey: string) => void
  /** /recall: insert this node's knowledge into the CURRENT chat's
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

function CrossProfileSubmenu({
  onInsertIntoProfile,
  target,
  title
}: {
  onInsertIntoProfile: (target: NodeMenuTarget, profileName: string) => void
  target: NodeMenuTarget
  title: (profile: string) => string
}) {
  const { t } = useI18n()
  const profiles = useStore($profiles)
  const selectedProfiles = useStore($starmapSelectedProfiles)
  const targets =
    Array.isArray(profiles) && Array.isArray(selectedProfiles)
      ? profiles.filter(profile => {
          const key = normalizeProfileKey(profile.name)

          return selectedProfiles.includes(key) && key !== target.profile
        })
      : []

  if (targets.length === 0) {
    return null
  }

  return (
    <DropdownMenuSub>
      <DropdownMenuSubTrigger>{title(targets.length === 1 ? profileLabel(targets[0]) : '…')}</DropdownMenuSubTrigger>
      <DropdownMenuSubContent>
        {targets.length > 1 ? (
          <DropdownMenuItem
            className="font-medium"
            onSelect={() => {
              targets.forEach(profile => onInsertIntoProfile(target, normalizeProfileKey(profile.name)))
            }}
          >
            {t.starmap.insertIntoAllSelected}
          </DropdownMenuItem>
        ) : null}
        {targets.map(profile => (
          <DropdownMenuItem
            className="max-w-64"
            key={normalizeProfileKey(profile.name)}
            onSelect={() => onInsertIntoProfile(target, normalizeProfileKey(profile.name))}
            title={profileLabel(profile)}
          >
            <span className="truncate">{profileLabel(profile)}</span>
          </DropdownMenuItem>
        ))}
      </DropdownMenuSubContent>
    </DropdownMenuSub>
  )
}

/** Right-click actions for a star-map node: provenance, edit (modal), delete (confirm).
 *  Provider-backed memory nodes are read-only (their storage lives in the
 *  provider's backend), so Edit/Delete are replaced by a hint. */
export function NodeContextMenu({
  onAddToSession,
  onClose,
  onInsertIntoProfile,
  onNodeRemoved,
  onRecallIntoChat,
  onShowProvenance,
  onStartConversation,
  recentSessions,
  target
}: NodeContextMenuProps) {
  const { t } = useI18n()
  const [editing, setEditing] = useState<EditState | null>(null)
  const [deleting, setDeleting] = useState<Omit<NodeMenuTarget, 'x' | 'y'> | null>(null)
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<null | string>(null)

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
                so every feature row stays reachable. */}
            <span aria-hidden style={{ left: target.x, position: 'fixed', top: target.y }} />
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" onCloseAutoFocus={e => e.preventDefault()} side="bottom">
            <DropdownMenuLabel className="max-w-56 truncate text-[0.68rem] font-normal text-muted-foreground">
              {target.label}
            </DropdownMenuLabel>
            {onShowProvenance ? (
              <DropdownMenuItem onSelect={() => onShowProvenance(target.id)}>
                {t.starmap.provenanceMenu}
              </DropdownMenuItem>
            ) : null}
            {onRecallIntoChat ? (
              <DropdownMenuItem onSelect={() => onRecallIntoChat(target)}>{t.starmap.recallIntoChat}</DropdownMenuItem>
            ) : null}
            {onAddToSession && recentSessions && recentSessions.length > 0 ? (
              <DropdownMenuSub>
                <DropdownMenuSubTrigger>{t.starmap.addToSession}</DropdownMenuSubTrigger>
                <DropdownMenuSubContent>
                  {recentSessions.map(session => (
                    <DropdownMenuItem
                      className="max-w-64"
                      key={session.key}
                      onSelect={() => onAddToSession(target, session.key)}
                      title={session.title}
                    >
                      <span className="truncate">{session.title}</span>
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuSubContent>
              </DropdownMenuSub>
            ) : null}
            {onInsertIntoProfile && target.profile ? (
              <CrossProfileSubmenu
                onInsertIntoProfile={onInsertIntoProfile}
                target={target}
                title={t.starmap.insertIntoProfile}
              />
            ) : null}
            {isProviderSource(target.memorySource) ? (
              <>
                {target.isConclusion && onStartConversation ? (
                  <DropdownMenuItem onSelect={() => onStartConversation(target)}>
                    {t.starmap.conclusionStartConversation}
                  </DropdownMenuItem>
                ) : null}
                <DropdownMenuLabel className="max-w-56 whitespace-normal text-[0.68rem] font-normal text-muted-foreground">
                  {t.starmap.providerReadOnly(target.memorySource)}
                </DropdownMenuLabel>
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
