import {
  $codingWorkspaceDrafts, type CodingWorkspaceIntent, codingWorkspaceKey, type CodingWorkspaceOwner,
  inspectCodingWorkspace, setCodingWorkspaceIntent
} from '@/store/coding-workspaces'

/** A selection records intent and reads facts. It must never provision a checkout. */
export async function selectCodingWorkspaceIntent(owner: CodingWorkspaceOwner, intent: CodingWorkspaceIntent | null) {
  setCodingWorkspaceIntent(owner, intent)

  if (!intent) {return}
  const key = codingWorkspaceKey(owner)
  const requestId = $codingWorkspaceDrafts.get()[key]?.requestId
  const inspection = await inspectCodingWorkspace(owner)

  if ($codingWorkspaceDrafts.get()[key]?.requestId !== requestId) {return}

  if (!inspection.repoRoot && intent.mode !== 'folder') {
    setCodingWorkspaceIntent(owner, { projectId: intent.projectId, path: intent.path, mode: 'folder' })
    await inspectCodingWorkspace(owner)
  }
}
