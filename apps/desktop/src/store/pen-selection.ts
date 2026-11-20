import { $canvasTabs, $poppedCanvasTabs, type CanvasTab } from '@/app/chat/canvas-tile'
import { $composerAttachments, addComposerAttachment, type ComposerAttachment, removeComposerAttachment } from '@/store/composer'

export interface PenSelectionNode {
  id: string
  name?: string
  type?: string
}

/** The one chip the canvas selection owns in the main composer. */
export const PEN_SELECTION_ATTACHMENT_ID = 'canvas:selection'

const PEN_PROVIDER = 'pen'

function nodeLabel(node: PenSelectionNode): string {
  return node.name?.trim() || node.type || node.id
}

/** "Hero card", or "Hero card +2" for a multi-selection. */
export function penSelectionLabel(nodes: readonly PenSelectionNode[]): string {
  const [first, ...rest] = nodes

  return rest.length ? `${nodeLabel(first)} +${rest.length}` : nodeLabel(first)
}

/**
 * What the agent reads: the canvas, and each selected node's name, type and id
 * so "this" / "these" land on pen_canvas calls without a round-trip.
 */
export function penSelectionRefText(canvas: string, nodes: readonly PenSelectionNode[]): string {
  const list = nodes.map(node => `- ${nodeLabel(node)} (${[node.type, `id ${node.id}`].filter(Boolean).join(', ')})`)

  return [`[Selected on the canvas "${canvas}" — "this"/"these" means these nodes:`, ...list, ']'].join('\n')
}

export function penSelectionAttachment(canvas: string, nodes: readonly PenSelectionNode[]): ComposerAttachment {
  return {
    id: PEN_SELECTION_ATTACHMENT_ID,
    kind: 'canvas',
    label: penSelectionLabel(nodes),
    detail: canvas,
    refText: penSelectionRefText(canvas, nodes)
  }
}

const signature = (docId: string, nodes: readonly PenSelectionNode[]) =>
  `${docId}\u0000${nodes.map(node => node.id).join(',')}`

function chatCanvas(): CanvasTab | undefined {
  return $poppedCanvasTabs.get().get(PEN_PROVIDER) ?? $canvasTabs.get().find(tab => tab.provider === PEN_PROVIDER)
}

/**
 * Mirror the chat's canvas selection into the main composer as a live chip:
 * it appears on select, follows the selection, and goes on deselect or when
 * the chat's canvas is a different file (or none). ✕ — or sending it — leaves
 * the selection out until it changes; reselecting brings the chip back.
 */
export function watchPenSelection(): () => void {
  const pen = window.hermesDesktop?.pen

  if (!pen?.onEvent) {
    return () => {}
  }

  let selection: { docId: string; nodes: PenSelectionNode[] } | null = null
  let shown: null | string = null
  let dismissed: null | string = null

  const sync = () => {
    const canvas = chatCanvas()
    const nodes = selection && canvas?.docId === selection.docId ? selection.nodes : []
    const sig = selection && nodes.length ? signature(selection.docId, nodes) : null

    if (!sig || sig === dismissed) {
      if (shown) {
        shown = null
        removeComposerAttachment(PEN_SELECTION_ATTACHMENT_ID)
      }

      return
    }

    shown = sig
    addComposerAttachment(penSelectionAttachment(canvas?.title || 'Canvas', nodes))
  }

  const offEvents = pen.onEvent(({ event, payload }) => {
    if (event !== 'selection-changed') {
      return
    }

    const { docId, nodes } = (payload ?? {}) as { docId?: string; nodes?: PenSelectionNode[] }

    selection = docId ? { docId, nodes: Array.isArray(nodes) ? nodes.filter(node => node?.id) : [] } : null

    // ✕ holds only while that selection stands: any change, even a deselect
    // and reselect of the same layer, offers the chip again.
    if (!selection?.nodes.length || signature(selection.docId, selection.nodes) !== dismissed) {
      dismissed = null
    }

    sync()
  })

  // The chip went away without the selection changing: ✕ or a send.
  const offAttachments = $composerAttachments.subscribe(attachments => {
    if (shown && !attachments.some(attachment => attachment?.id === PEN_SELECTION_ATTACHMENT_ID)) {
      dismissed = shown
      shown = null
    }
  })

  const offTabs = $canvasTabs.subscribe(sync)
  const offPopped = $poppedCanvasTabs.subscribe(sync)

  return () => {
    offEvents()
    offAttachments()
    offTabs()
    offPopped()
  }
}
