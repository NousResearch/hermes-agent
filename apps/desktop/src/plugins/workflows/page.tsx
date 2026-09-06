import '@xyflow/react/dist/style.css'

import {
  Button,
  cn,
  Codicon,
  composerDockCard,
  Contribute,
  EmptyState,
  PageHeader,
  PageHeaderActions,
  PageHeaderTitle,
  PageShell,
  SidePanel,
  Tip,
  TITLEBAR_AREAS,
  useTheme,
  useValue
} from '@hermes/plugin-sdk'
import {
  Background,
  BackgroundVariant,
  type EdgeChange,
  type NodeChange,
  Panel,
  ReactFlow,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  useReactFlow
} from '@xyflow/react'
import { type CSSProperties, useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { type AddAt, addStep, AddStepProvider, KindPicker } from './add-step'
import { AskDialog } from './ask'
import { CanvasChat } from './chat'
import { FlowDirProvider } from './direction'
import { $currentId, $workflows, createWorkflow, saveWorkflow, type WorkflowDoc } from './documents'
import { CutEdgeProvider, edgeTypes } from './edges'
import { fromScenario, type Graph, runPlan, toScenario, updateStep } from './graph'
import type { RunControl } from './graph-dispatch'
import { Inspector } from './inspector'
import { FIT, tidyLayout } from './layout'
import { LiveLog } from './livelog'
import { CANVAS_NOTE_ID, canvasNote, type NodeData, NodeLive, nodeTypes } from './nodes'
import { RunNowProvider, usePlayer } from './player'
import { type EdgeState, freshRuntime, type StepRuntime } from './protocol'
import { feedLine, type FeedLine } from './protocol-feed'
import { blankScenario, starterScenario, type StepConfig, type StepKind } from './scenario'
import { WorkflowSwitcher } from './switcher'
import { Timeline } from './timeline'
import { useTourPan } from './tour-pan'
import { useCanvasKeys } from './use-canvas-keys'
import { useCanvasLayout } from './use-canvas-layout'
import { useCommit } from './use-commit'
import { useUndoRedo } from './use-undo-redo'
import { useWiring } from './use-wiring'

// Steps the agent adds mid-session have no runtime in the event stream — they
// read as idle until the next run includes them.
const IDLE_RT: StepRuntime = freshRuntime()

// One width, two consumers: the panel wears it, and the canvas reads it as a
// CSS var so the run dock re-centres on what's left of the canvas.
const INSPECTOR_REM = '26rem'
const INSPECTOR_WIDTH = 'w-[26rem]'

export default function WorkflowsPage() {
  const docs = useValue($workflows)
  const currentId = useValue($currentId)
  const doc = docs.find(d => d.id === currentId)

  return (
    <>
      {/* Page-owned titlebar chrome: exists exactly while this page is mounted. */}
      <Contribute area={TITLEBAR_AREAS.center} id="workflows:switcher">
        <WorkflowSwitcher />
      </Contribute>
      {doc ? (
        // Keyed on the document: switching workflows is a fresh canvas, not a
        // re-render of this one. Undo history, selection and the run all belong
        // to the workflow you were looking at, and carrying any of them across
        // would be a bug in every case.
        <ReactFlowProvider key={doc.id}>
          <Flow doc={doc} />
        </ReactFlowProvider>
      ) : (
        <FirstWorkflow />
      )}
    </>
  )
}

/** Nothing authored yet. Two ways in: an empty canvas, or the scenario the
 *  plugin ships with — which is the faster way to learn what a gate is. */
function FirstWorkflow() {
  return (
    <PageShell className="wf-root">
      <PageHeader>
        <PageHeaderTitle>Workflows</PageHeaderTitle>
      </PageHeader>
      <EmptyState
        action={
          <div className="flex items-center gap-2">
            <Button onClick={() => createWorkflow('Untitled workflow', blankScenario())} size="sm">
              <Codicon name="add" size="0.75rem" />
              Create your first workflow
            </Button>
            <Button onClick={() => createWorkflow('Figma → PR', starterScenario())} size="sm" variant="outline">
              Start from an example
            </Button>
          </div>
        }
        className="min-h-0 flex-1"
        description="A workflow is a graph of steps an agent runs — work, checks, branches, and the places a person has to say yes. Build one by hand, or ask for it."
        icon="type-hierarchy-sub"
        title="No workflows yet"
      />
    </PageShell>
  )
}

function Flow({ doc }: { doc: WorkflowDoc }) {
  // React Flow paints its own chrome (background dots, controls, minimap) from
  // a light/dark switch of its own, so it needs the mode the host actually
  // resolved — 'system' would leave it guessing.
  const { resolvedMode } = useTheme()
  // The run is built from whatever is on the canvas when you press play, so the
  // player reads the graph through a ref rather than taking it as a prop — it's
  // mounted above the node state, and re-arming it on every keystroke would
  // rebuild the run shape while you type.
  const graphRef = useRef<Graph>({ nodes: [], edges: [] })
  const planOf = useCallback(() => runPlan(graphRef.current, doc.name, doc.id), [doc.id, doc.name])
  const player = usePlayer(planOf)

  const { world, frozenAt, live } = player
  const { steps: runtime, edges: edgeState, phase } = world

  // The feed is part of the view, so it rewinds with the playhead.
  const lines = useMemo(() => {
    const out: FeedLine[] = []

    for (const e of player.events.slice(0, player.head)) {
      const line = feedLine(e)

      if (line) {
        out.push(line)
      }
    }

    return out
  }, [player.events, player.head])

  // useNodesState takes a value, not a lazy initializer, so an inline
  // fromScenario() call would rebuild the whole graph on every render of this
  // component and throw the result away — React only keeps the first. The
  // document is fixed for this canvas's life (the page keys on its id), so
  // there's nothing for the memo to depend on.
   
  const seed = useMemo(() => {
    const g = fromScenario(doc.scenario)

    return g.nodes.length === 0 ? { ...g, nodes: [canvasNote()] } : g
  }, [])

  const [nodes, setNodes, onNodesChange] = useNodesState(seed.nodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(seed.edges)
  const [selected, setSelected] = useState<string | null>(null)
  const [draft, setDraft] = useState<AddAt | null>(null)

  // The note is scenery for an empty canvas. A real step means it's done.
  useEffect(() => {
    if (nodes.some(n => n.id !== CANVAS_NOTE_ID) && nodes.some(n => n.id === CANVAS_NOTE_ID)) {
      setNodes(ns => ns.filter(n => n.id !== CANVAS_NOTE_ID))
    }
  }, [nodes, setNodes])
  // The + lives on the edge; adding a step unmounts that edge under the
  // pointer, so the mouseup lands on the pane and would clear the selection
  // we just made. Swallow the next pane click after an add.
  const ignorePaneClick = useRef(false)
  const { fitView, screenToFlowPosition } = useReactFlow()

  // A tour step's card may be off screen; the engine asks us to bring it in.
  const canvasWrap = useRef<HTMLDivElement>(null)
  useTourPan(canvasWrap, FIT)

  // Undo/redo are keyboard-only (⌘Z / ⌘⇧Z, bound inside the hook) — the canvas
  // takes the snapshots, the rail doesn't need buttons for them.
  const { takeSnapshot } = useUndoRedo({
    nodes,
    edges,
    setNodes,
    setEdges
  })

  // Snapshot before structural mutations so cmd/ctrl+z reverts them. Drags
  // snapshot on drag-start (one entry per drag); deletions snapshot the
  // pre-remove graph the moment a "remove" change arrives.
  const handleNodesChange = useCallback(
    (changes: NodeChange[]) => {
      if (changes.some(c => c.type === 'remove')) {
        takeSnapshot()
      }

      onNodesChange(changes)
    },
    [onNodesChange, takeSnapshot]
  )

  const handleEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      if (changes.some(c => c.type === 'remove')) {
        takeSnapshot()
      }

      onEdgesChange(changes)
    },
    [onEdgesChange, takeSnapshot]
  )

  const { dir, dirRef, resetView, setDir, tidy, vertical } = useCanvasLayout({
    edges,
    nodes,
    setNodes,
    takeSnapshot
  })

  // sync run state -> node.data (preserves dragged positions + edited config).
  //
  // A card that hasn't moved keeps its exact node object. React Flow re-renders
  // a node when its data changes, so spreading a new `data` onto all of them
  // every event meant one step starting re-rendered the entire graph.
  useEffect(() => {
    setNodes(ns => {
      let dirty = false

      const next = ns.map(n => {
        const d = n.data as Partial<NodeData>
        const sel = n.id === selected
        const rt = runtime[n.id] ?? IDLE_RT

        if (d.rt === rt && d.frozenAt === frozenAt && d.selected === sel) {
          return n
        }

        dirty = true

        return { ...n, data: { ...n.data, rt, frozenAt, selected: sel } }
      })

      return dirty ? next : ns
    })
  }, [frozenAt, runtime, selected, setNodes])

  useEffect(() => {
    setEdges(es => {
      let dirty = false

      const next = es.map(e => {
        const state = edgeState[e.id] as EdgeState

        if ((e.data as { state?: EdgeState })?.state === state) {
          return e
        }

        dirty = true

        return { ...e, data: { ...e.data, state } }
      })

      return dirty ? next : es
    })
  }, [edgeState, setEdges])

  const updateConfig = (id: string, patch: Partial<StepConfig>) => {
    takeSnapshot()
    // Functional so a click can't apply the patch onto a `nodes` snapshot
    // that the overlay or a drag already replaced.
    setNodes(ns => {
      const op = updateStep({ nodes: ns, edges }, id, patch)

      return op.ok ? op.graph.nodes : ns
    })
  }

  // Click + / ⌘-or-Shift-click empty canvas only names WHERE. The picker
  // names WHAT — otherwise the seed's gate/human/wait can never be minted,
  // only edited. Space used to mint here; it is transport now.
  const requestAdd = useCallback((where: AddAt) => {
    ignorePaneClick.current = true
    setDraft(where)
    window.setTimeout(() => {
      ignorePaneClick.current = false
    }, 80)
  }, [])

  const confirmAdd = useCallback(
    (kind: StepKind) => {
      if (!draft) {
        return
      }

      const next = addStep(nodes, edges, draft, kind, dir)
      const dropped = draft.on === 'canvas'
      setDraft(null)

      if (!next.id) {
        return
      }

      takeSnapshot()
      // A canvas click already named the spot. Tidy would treat the unwired
      // card as an orphan and stack it left of the graph.
      setNodes(dropped ? next.nodes : tidyLayout(next.nodes, next.edges, dir))
      setEdges(next.edges)
      ignorePaneClick.current = true
      setSelected(next.id)
      window.setTimeout(() => {
        ignorePaneClick.current = false
      }, 80)
    },
    [dir, draft, edges, nodes, setEdges, setNodes, takeSnapshot]
  )

  const graph = useMemo<Graph>(() => ({ nodes, edges }), [nodes, edges])
  graphRef.current = graph

  // The document IS the canvas, so it's written back whenever the canvas
  // changes — no save button, and the switcher can't show you a stale step
  // count. `toScenario` drops runtime and keeps positions, so a round-trip
  // through storage returns the graph you left.
  useEffect(() => {
    saveWorkflow(doc.id, toScenario(graph))
  }, [doc.id, graph])

  // What `run_control` drives. Held in a ref for the bridge, for the same
  // reason the graph is: the bridge registers once per canvas, and a control
  // that changed identity every frame would re-register through every drag.
  const run = useMemo<RunControl>(
    () => ({
      running: player.running,
      paused: player.pauseState === 'paused',
      start: player.start,
      pause: player.requestPause,
      resume: player.resume,
      reset: player.reset
    }),
    [player]
  )

  const runRef = useRef(run)
  runRef.current = run

  const { applyOp, reflowing } = useCommit({
    dir,
    dirRef,
    docId: doc.id,
    graphRef,
    runRef,
    setEdges,
    setNodes,
    takeSnapshot
  })

  const {
    cutEdge,
    isValidConnection,
    onBeforeDelete,
    onConnect,
    onConnectEnd,
    onConnectStart,
    onReconnect,
    onReconnectEnd,
    onReconnectStart,
    removeNode
  } = useWiring({ applyOp, graph })

  useCanvasKeys({ player, tidy })

  const selNode = useMemo(
    () => (selected ? nodes.find(n => n.id === selected && (n.data as NodeData).def) : null),
    [nodes, selected]
  )

  // The live log names steps the way their cards do, renames included.
  const nodeTitles = useMemo(() => {
    const out: Record<string, string> = {}

    for (const n of nodes) {
      const cfg = (n.data as NodeData).config

      if (cfg) {
        out[n.id] = cfg.title
      }
    }

    return out
  }, [nodes])

  const askTitle = player.asking ? (nodeTitles[player.asking.nodeId] ?? player.asking.nodeId) : ''

  return (
    <RunNowProvider value={{ start: player.start, fireWebhook: player.fireWebhook, running: player.running }}>
    <PageShell className="wf-root" style={{ '--wf-inspector': selNode ? INSPECTOR_REM : '0rem' } as CSSProperties}>
      {/* Same header as the Kanban board: the page name here, the current
          workflow in the titlebar (see WorkflowSwitcher). Theme and mode are
          the host's — they live in Settings, not on this page. */}
      <PageHeader>
        <PageHeaderTitle>Workflows</PageHeaderTitle>
        <PageHeaderActions>
          {/* A divided box, not an arrow: the icon has to say "arrangement",
              and a lone chevron on a header button says "this opens something".
              It shows the layout you'd GET.
              
              The LABEL, though, names the control and doesn't change with it.
              You click this with the tip already open, so a label that swapped
              between two different-length strings resized and re-centred its
              bubble on every press — a flicker right under the header, for a
              state the icon is already showing. */}
          <Tip label="Flip layout direction">
            <Button
              aria-label="Flip layout direction"
              onClick={() => setDir(vertical ? 'LR' : 'TB')}
              size="icon-xs"
              variant="ghost"
            >
              <Codicon
                className="grid size-3.5 place-items-center"
                name={vertical ? 'split-vertical' : 'split-horizontal'}
                size="0.85rem"
              />
            </Button>
          </Tip>
        </PageHeaderActions>
      </PageHeader>

      <div
        className={cn('canvas-wrap', reflowing && 'reflowing')}
        /* Double-click empty canvas frames the graph. Caught here because
           React Flow exposes no pane double-click prop — the target check
           keeps double-clicks on cards, wires and panels meaning whatever
           those things say they mean. */
        onDoubleClick={e => {
          if ((e.target as HTMLElement).classList.contains('react-flow__pane')) {
            resetView()
          }
        }}
        ref={canvasWrap}
      >
        <NodeLive.Provider value={nodes}>
        <FlowDirProvider value={dir}>
          <AddStepProvider value={requestAdd}>
            <CutEdgeProvider value={cutEdge}>
              <ReactFlow
                colorMode={resolvedMode}
                /* n8n's `connection-radius`, triple React Flow's default 20. A 9px
           socket you have to hit dead-on is why dropping a wire felt like
           threading a needle; at 60 the socket comes to meet you, and the
           connectingto highlight tells you it has. */
                connectionRadius={60}
                deleteKeyCode={['Backspace', 'Delete']}
                edges={edges}
                edgeTypes={edgeTypes}
                elevateNodesOnSelect
                fitView={nodes.some(n => n.id !== CANVAS_NOTE_ID)}
                fitViewOptions={FIT}
                isValidConnection={isValidConnection}
                maxZoom={1.75}
                minZoom={0.35}
                multiSelectionKeyCode={['Meta', 'Control']}
                /* React Flow defaults nodeClickDistance to 0, which forwards to d3's
           .clickDistance(0): the click is swallowed if the pointer moves even
           one pixel between press and release. A trackpad almost always drifts
           a pixel or two, so selecting a node silently failed and you'd click
           again — the "dead zone". A few pixels of slack is what every native
           control allows. paneClickDistance gets the same treatment so
           deselecting doesn't have the identical problem. */
                nodeClickDistance={4}
                /* A node's y is its CENTRE, not its top edge. React Flow renders at
           `position.y - height * origin[1]`, so a card that grows takes half
           the new height off its top and half off its bottom instead of
           unrolling downward from a pinned corner.
           
           That matters because the handles sit at 50% and the cards in a rank
           are centre-aligned: growing downward walked every handle down with
           the card, so the whole graph's wiring sagged and re-settled each time
           a step produced a line. With the centre fixed, a card can change
           height without a single edge moving.
           
           Done with the library's own origin rather than a compensating
           transform on the card: origin feeds `positionAbsolute`, so bounds,
           fitView, hit-testing and edge geometry all agree. A CSS transform
           would move the paint and leave React Flow's model behind it. */
                nodeOrigin={[0, 0.5]}
                nodes={nodes}
                nodesDraggable
                nodeTypes={nodeTypes}
                onBeforeDelete={onBeforeDelete}
                onConnect={onConnect}
                onConnectEnd={onConnectEnd}
                onConnectStart={onConnectStart}
                onEdgesChange={handleEdgesChange}
                onNodeClick={(e, n) => {
                  if (n.id === CANVAS_NOTE_ID) {
                    if (e.metaKey || e.ctrlKey || e.shiftKey) {
                      requestAdd({
                        on: 'canvas',
                        at: screenToFlowPosition({ x: e.clientX, y: e.clientY })
                      })
                    }

                    return
                  }

                  setSelected(n.id)
                }}
                onNodeDragStart={() => takeSnapshot()}
                onNodesChange={handleNodesChange}
                onPaneClick={e => {
                  if (ignorePaneClick.current) {
                    ignorePaneClick.current = false

                    return
                  }

                  if (e.metaKey || e.ctrlKey || e.shiftKey) {
                    requestAdd({
                      on: 'canvas',
                      at: screenToFlowPosition({ x: e.clientX, y: e.clientY })
                    })

                    return
                  }

                  if (draft) {
                    setDraft(null)

                    return
                  }

                  setSelected(null)
                }}
                onReconnect={onReconnect}
                onReconnectEnd={onReconnectEnd}
                onReconnectStart={onReconnectStart}
                onSelectionDragStart={() => takeSnapshot()}
                panActivationKeyCode={null}
                paneClickDistance={4}
                proOptions={{ hideAttribution: true }}
                /* Default 10px puts the grab ring almost entirely under the node's own
           handle, so the gesture that unplugs a wire was reachable only in a
           couple of pixels of fringe. Matched to the edge's hit stroke. */
                reconnectRadius={22}
                selectionKeyCode="Shift"
                zoomActivationKeyCode={['Meta', 'Control']}
                /* The wrapper's onDoubleClick frames the graph. RF's default
           spend of the gesture (zoom in) is turned off to make room. */
                zoomOnDoubleClick={false}
              >
                <Background gap={20} size={1.3} variant={BackgroundVariant.Dots} />

                <LiveLog lines={lines} titles={nodeTitles} />

                {/* The app's composer dock, borrowed whole: a card fused to the top of
            the capsule (the chat's status stack is the same shape) carrying the
            transport, and the composer itself below it. Same fill, same glass,
            same seam — this reads as the app's input, because it is. */}
                <Panel className="run-panel" position="bottom-center">
                  {player.asking && player.deferred && (
                    <button className="ask-back" onClick={player.reveal}>
                      <Codicon name="bell" />
                      {askTitle} is waiting on you
                    </button>
                  )}
                  <div
                    className={cn(
                      composerDockCard('top'),
                      'canvas-dock-transport mx-2 overflow-visible rounded-b-none border-b-transparent'
                    )}
                  >
                    <Timeline p={player} />
                  </div>
                  <CanvasChat autofocus={!nodes.some(n => n.id !== CANVAS_NOTE_ID)} workflowId={doc.id} />
                </Panel>

                {player.asking && (
                  <AskDialog
                    {...player.asking}
                    onDefer={player.defer}
                    onRespond={player.respond}
                    open={!player.deferred}
                    title={askTitle}
                  />
                )}
              </ReactFlow>
            </CutEdgeProvider>
          </AddStepProvider>
        </FlowDirProvider>
        </NodeLive.Provider>
      </div>

      {/* Last child of the page root, exactly where the Kanban board hangs its
          task drawer — so it pins to the whole page and bleeds past the header,
          rather than starting below it as another inset canvas panel.

          Narrower than that drawer's 26rem: it holds prose and a run log, this
          holds a column of knobs. */}
      {selNode && (
        <SidePanel className={INSPECTOR_WIDTH} onClose={() => setSelected(null)}>
          <Inspector
            graph={graph}
            node={selNode}
            onChange={patch => updateConfig(selNode.id, patch)}
            onClose={() => setSelected(null)}
            onDelete={() => {
              setSelected(null)
              removeNode(selNode.id)
            }}
            onOp={applyOp}
            rt={runtime[selNode.id]}
          />
        </SidePanel>
      )}

      {draft && <KindPicker at={draft.at} onClose={() => setDraft(null)} onPick={confirmAdd} />}
    </PageShell>
    </RunNowProvider>
  )
}
