import { Codicon, DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@hermes/plugin-sdk'
import { type Edge, type Node, useStore, type XYPosition } from '@xyflow/react'
import { createContext, useContext } from 'react'

import { KIND_ICON } from './kind-mark'
import { CARD_W, DEFAULT_DIR, type FlowDir, freeSpot, RANK_GAP, widthOf } from './layout'
import { type NodeData } from './nodes'
import { type EdgeState, freshRuntime } from './protocol'
import { defaultConfig, STEP_KINDS, type StepDef, type StepKind } from './scenario'

// Adding a step, in one place.
//
// Two gestures offer it — the + on a connector and ⌘-or-Shift-click on empty
// canvas — and they run the same command for the same reason a menu item and
// its shortcut do. Minting the step, wiring it, snapshotting for undo and
// leaving it selected with the inspector open all have to happen identically
// or the two routes drift into two different features that happen to share a
// verb. All either gesture gets to decide is WHERE, which is the whole of
// `AddAt`.

export type AddAt =
  /** Split a connector, so source → new step → target. */
  | { on: 'edge'; edgeId: string; at: XYPosition }
  /** Unwired, at a point in flow coordinates. */
  | { on: 'canvas'; at: XYPosition }

export interface AddedStep {
  nodes: Node[]
  edges: Edge[]
  /** null when the anchor no longer exists — the caller changes nothing. */
  id: string | null
}

const wireId = (source: string, target: string) => `${source}->${target}`

// Steps are named by the author, not by us — the id only has to be stable and
// free, since it's what `needs:` and the gate predicates will refer to.
function freeId(nodes: Node[]) {
  for (let i = 1; ; i++) {
    const id = `step-${i}`

    if (!nodes.some(n => n.id === id)) {
      return id
    }
  }
}

function mint(nodes: Node[], position: XYPosition, kind: StepKind): Node {
  const spec = STEP_KINDS.find(k => k.kind === kind) ?? STEP_KINDS[0]

  const def: StepDef = {
    id: freeId(nodes),
    kind,
    title: spec.title,
    doing: spec.doing
  }

  const config = defaultConfig(def)

  return {
    id: def.id,
    type: def.kind,
    position,
    selected: true,
    data: {
      def,
      config,
      rt: freshRuntime(),
      selected: true
    } satisfies NodeData
  }
}

/** One selected card — the new one. Everyone else drops the ring. */
function selectOnly(nodes: Node[], id: string): Node[] {
  return nodes.map(n =>
    n.id === id || n.selected || (n.data as NodeData).selected
      ? { ...n, selected: n.id === id, data: { ...n.data, selected: n.id === id } }
      : n
  )
}

/** The graph after adding a step, plus which step to select. Pure. */
export function addStep(
  nodes: Node[],
  edges: Edge[],
  where: AddAt,
  kind: StepKind = 'agent',
  dir: FlowDir = DEFAULT_DIR
): AddedStep {
  if (where.on === 'canvas') {
    // x is a left edge and y is a centre, per the canvas's nodeOrigin.
    // The listening note is scenery, not a card — don't bounce off it.
    const cards = nodes.filter(n => (n.data as NodeData).def)
    const at = freeSpot(cards, { x: where.at.x - CARD_W / 2, y: where.at.y }, dir)
    const node = mint(nodes, at, kind)

    return { nodes: selectOnly([...nodes, node], node.id), edges, id: node.id }
  }

  const split = edges.find(e => e.id === where.edgeId)

  if (!split) {
    return { nodes, edges, id: null }
  }

  // n8n (useCanvasOperations.shiftDownstreamNodesPosition): the new card sits
  // to the RIGHT of the source by one rank step, Y locked to the source so
  // the hop stays a straight line. The + is where you clicked, not where the
  // card lands — dropping on the bezier midpoint put a 212px card in a 120px
  // gap and left both new wires as stubs with the next + on the ports.
  const source = nodes.find(n => n.id === split.source)
  const target = nodes.find(n => n.id === split.target)
  const insertX = source ? source.position.x + widthOf(source) + RANK_GAP : where.at.x - CARD_W / 2
  const insertY = source?.position.y ?? where.at.y
  const node = mint(nodes, { x: insertX, y: insertY }, kind)

  // Push the tail only when the target column hasn't got a full rank of room
  // (n8n's hasSpaceForInsertion). Everything at or past the target moves as
  // one column — fan-out siblings stay aligned.
  const needed = insertX + CARD_W + RANK_GAP
  const shift = target ? Math.max(0, needed - target.position.x) : 0

  const shifted =
    shift > 0 && target
      ? nodes.map(n =>
          n.position.x < target.position.x - 1 ? n : { ...n, position: { ...n.position, x: n.position.x + shift } }
        )
      : nodes

  // The halves inherit the ends they keep: the gate's `pass` port stays the
  // source of the first hop, and a loop-back's target handle stays the target
  // of the second. Splitting a wire shouldn't quietly re-route either end.
  const wires: Edge[] = [
    {
      id: wireId(split.source, node.id),
      source: split.source,
      target: node.id,
      sourceHandle: split.sourceHandle,
      type: 'data',
      data: { state: 'idle' as EdgeState }
    },
    {
      id: wireId(node.id, split.target),
      source: node.id,
      target: split.target,
      sourceHandle: kind === 'gate' ? 'pass' : undefined,
      targetHandle: split.targetHandle,
      type: 'data',
      data: { state: 'idle' as EdgeState }
    }
  ]

  return {
    nodes: selectOnly([...shifted, node], node.id),
    edges: [...edges.filter(e => e.id !== split.id), ...wires],
    id: node.id
  }
}

// Click + / ⌘-or-Shift-click asks WHERE. The picker asks WHAT. Both land here
// so an edge add and a canvas add mint the same way.
export type RequestAdd = (where: AddAt) => void

const Ctx = createContext<RequestAdd>(() => {})
export const AddStepProvider = Ctx.Provider
export const useAddStep = () => useContext(Ctx)

/** The app's menu, parked on the + / drop point. */
export function KindPicker({
  at,
  onPick,
  onClose
}: {
  at: XYPosition
  onPick: (kind: StepKind) => void
  onClose: () => void
}) {
  const [tx, ty, zoom] = useStore(s => s.transform)
  const pane = useStore(s => s.domNode)
  const origin = pane?.getBoundingClientRect()
  const left = at.x * zoom + tx + (origin?.left ?? 0)
  const top = at.y * zoom + ty + (origin?.top ?? 0)

  return (
    <DropdownMenu onOpenChange={open => !open && onClose()} open>
      <DropdownMenuTrigger asChild>
        <span
          aria-hidden
          className="pointer-events-none fixed z-50 size-px"
          style={{ left, top }}
        />
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="start"
        className="min-w-44"
        onCloseAutoFocus={e => e.preventDefault()}
        side="bottom"
      >
        {STEP_KINDS.map(k => (
          <DropdownMenuItem key={k.kind} onSelect={() => onPick(k.kind)}>
            <Codicon name={KIND_ICON[k.kind]} size="0.8rem" />
            {k.title}
            <span className="ml-auto text-(--ui-text-tertiary)">{k.blurb}</span>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
