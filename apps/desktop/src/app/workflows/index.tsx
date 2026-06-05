import '@xyflow/react/dist/style.css'
import './workflows.css'

import { useStore } from '@nanostores/react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Background,
  Controls,
  type Edge,
  Handle,
  MarkerType,
  MiniMap,
  type Node,
  type NodeProps,
  type OnConnect,
  type OnNodeDrag,
  type OnNodesChange,
  Position,
  ReactFlow,
  ReactFlowProvider,
  useEdgesState,
  useNodesState
} from '@xyflow/react'
import type * as React from 'react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import { Button } from '@/components/ui/button'
import { CompactMarkdown } from '@/components/chat/compact-markdown'
import { Codicon } from '@/components/ui/codicon'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Textarea } from '@/components/ui/textarea'
import {
  attachWorkflowComposerFiles,
  completeWorkflowComposer,
  confirmWorkflowIntake,
  confirmWorkflowNode,
  createWorkflowSnapshot,
  executeWorkflowSlashCommand,
  generateWorkflow,
  getGlobalModelOptions,
  getSkills,
  getWorkflowFiles,
  getWorkflowProject,
  listWorkflowEvents,
  listWorkflowProjects,
  pauseWorkflowRun,
  retryWorkflowNode,
  saveWorkflow,
  sendWorkflowChat,
  sendWorkflowIntakeMessage,
  skipWorkflowNode,
  startWorkflowIntake,
  startWorkflowRun,
  submitWorkflowIntakeAnswers,
  stopWorkflowRun,
  updateWorkflowReferences,
  updateWorkflowSkills
} from '@/hermes'
import { cn } from '@/lib/utils'
import { $workflowLanguage } from '@/store/workflow-language'
import { useTheme } from '@/themes/context'
import type { ModelOptionsResponse, SkillInfo } from '@/types/hermes'
import type {
  ExecutionMode,
  ProjectBundle,
  ProjectListResponse,
  ReferenceItem,
  SkillBinding,
  StreamEvent,
  VersionSnapshot,
  Workflow,
  WorkflowComposerCompletionItem,
  WorkflowEdge,
  WorkflowFileNode,
  WorkflowIntakeAnswer,
  WorkflowIntakeBatch,
  WorkflowIntakeMessage,
  WorkflowIntakePayload,
  WorkflowIntakeResponse,
  WorkflowNode,
  WorkflowNodeStatus
} from '@/types/workflow'

import { titlebarHeaderBaseClass } from '../shell/titlebar'
import { type WorkflowCopy, workflowCopyFor } from './i18n'
import {
  applyWorkflowProjectChange,
  dispatchWorkflowProjectsChanged
} from './project-events'

type DrawerMode = 'files' | 'references' | 'skills' | 'snapshots' | 'task'

interface WorkflowNodeData extends Record<string, unknown> {
  node: WorkflowNode
}

type FlowNode = Node<WorkflowNodeData, 'workflow'>
type FlowEdge = Edge<{ kind: string }>

const STATUS_TONE: Record<WorkflowNodeStatus, string> = {
  aborted: 'danger',
  completed: 'success',
  created: 'neutral',
  failed: 'danger',
  queued: 'info',
  ready: 'ready',
  retrying: 'warning',
  reviewing: 'warning',
  revision_needed: 'warning',
  running: 'running',
  skipped: 'neutral',
  waiting_user_confirm: 'warning'
}

function useWorkflowCopy(): WorkflowCopy {
  return workflowCopyFor(useStore($workflowLanguage))
}

function statusMeta(copy: WorkflowCopy, status: WorkflowNodeStatus): { label: string; tone: string } {
  return {
    label: copy.status[status] ?? status,
    tone: STATUS_TONE[status] ?? 'neutral'
  }
}

const EVENT_ICON: Record<StreamEvent['type'], string> = {
  ai_reply: 'sparkle',
  approval: 'pass',
  error: 'error',
  node_status: 'pulse',
  process_summary: 'list-tree',
  snapshot: 'git-commit',
  stage_result: 'checklist',
  tool_call: 'tools'
}

const DEFAULT_SKILLS: SkillBinding[] = [
  { id: 'planner', name: 'planner', enabled: true, source: 'hermes' },
  { id: 'file', name: 'file', enabled: true, source: 'hermes' },
  { id: 'terminal', name: 'terminal', enabled: true, source: 'hermes' },
  { id: 'reviewer', name: 'reviewer', enabled: true, source: 'hermes' },
  { id: 'writer', name: 'writer', enabled: true, source: 'hermes' }
]
const DEFAULT_MAX_CONCURRENCY = 2
const RIGHT_DRAWER_WIDTH_KEY = 'hermes.workflow.rightDrawerWidth'
const RIGHT_DRAWER_MIN_WIDTH = 280
const RIGHT_DRAWER_MAX_WIDTH = 640
const RIGHT_DRAWER_DEFAULT_WIDTH = 352

function WorkflowNodeCard({ data, selected }: NodeProps<FlowNode>) {
  const copy = useWorkflowCopy()
  const status = statusMeta(copy, data.node.status)
  const hasReview = data.node.reviewRules.required
  const nodeType = copy.nodeType[data.node.type as keyof typeof copy.nodeType] ?? data.node.type.toUpperCase()

  return (
    <div className={cn('workflow-node-card', selected && 'is-selected', `tone-${status.tone}`)}>
      <Handle className="workflow-handle" position={Position.Left} type="target" />
      <div className="workflow-node-card__top">
        <span className="workflow-node-card__type">{nodeType}</span>
        <span className={cn('workflow-status-pill', `tone-${status.tone}`)}>{status.label}</span>
      </div>
      <div className="workflow-node-card__title">{data.node.title}</div>
      <div className="workflow-node-card__description">{data.node.description}</div>
      <div className="workflow-node-card__meta">
        <span>{data.node.skills.length ? data.node.skills.join(' / ') : 'no skill'}</span>
        {hasReview && <span>review gate</span>}
      </div>
      <Handle className="workflow-handle" position={Position.Right} type="source" />
    </div>
  )
}

const nodeTypes = { workflow: WorkflowNodeCard }

export function WorkflowsView() {
  return (
    <ReactFlowProvider>
      <WorkflowWorkbench />
    </ReactFlowProvider>
  )
}

function WorkflowWorkbench() {
  const workflowLanguage = useStore($workflowLanguage)
  const { resolvedMode } = useTheme()
  const copy = workflowCopyFor(workflowLanguage)
  const queryClient = useQueryClient()
  const [searchParams, setSearchParams] = useSearchParams()
  const [activeProjectId, setActiveProjectId] = useState<string | null>(null)
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [autoFollowRunNode, setAutoFollowRunNode] = useState(true)
  const [drawerMode, setDrawerMode] = useState<DrawerMode>('task')
  const [selectedFilePath, setSelectedFilePath] = useState<string | null>(null)
  const [streamExpanded, setStreamExpanded] = useState(false)
  const [filterSelectedNode, setFilterSelectedNode] = useState(false)
  const [executionMode, setExecutionMode] = useState<ExecutionMode>('semi_auto')
  const [streamEvents, setStreamEvents] = useState<StreamEvent[]>([])
  const [wsHealthy, setWsHealthy] = useState(false)
  const [flowNodes, setFlowNodes, onNodesChangeBase] = useNodesState<FlowNode>([])
  const [flowEdges, setFlowEdges, onEdgesChange] = useEdgesState<FlowEdge>([])
  const latestEventTimestampRef = useRef<number | undefined>(undefined)
  const requestedProjectId = searchParams.get('project')
  const requestNewProject = searchParams.get('new') === '1'

  const projectsQuery = useQuery({
    queryKey: ['workflow-projects'],
    queryFn: () => listWorkflowProjects()
  })

  useEffect(() => {
    const firstProject = projectsQuery.data?.projects[0]

    if (requestedProjectId && requestedProjectId !== activeProjectId) {
      setActiveProjectId(requestedProjectId)
      setSelectedNodeId(null)
      setAutoFollowRunNode(true)
      setStreamEvents([])
      latestEventTimestampRef.current = undefined

      return
    }

    if (requestNewProject && activeProjectId) {
      setActiveProjectId(null)
      setSelectedNodeId(null)
      setAutoFollowRunNode(true)
      setStreamEvents([])
      latestEventTimestampRef.current = undefined

      return
    }

    if (!activeProjectId && firstProject && !requestNewProject) {
      setActiveProjectId(firstProject.id)
    }
  }, [activeProjectId, projectsQuery.data?.projects, projectsQuery.isSuccess, requestNewProject, requestedProjectId])

  const bundleQuery = useQuery({
    queryKey: ['workflow-project', activeProjectId],
    queryFn: () => getWorkflowProject(activeProjectId!),
    enabled: Boolean(activeProjectId),
    refetchInterval: 1800
  })

  const filesQuery = useQuery({
    queryKey: ['workflow-files', activeProjectId],
    queryFn: () => getWorkflowFiles(activeProjectId!),
    enabled: Boolean(activeProjectId && drawerMode === 'files'),
    refetchInterval: drawerMode === 'files' ? 3000 : false
  })

  const availableSkillsQuery = useQuery({
    queryKey: ['workflow-available-skills'],
    queryFn: getSkills
  })

  const modelOptionsQuery = useQuery({
    queryKey: ['workflow-model-options'],
    queryFn: getGlobalModelOptions
  })

  const eventsQuery = useQuery({
    queryKey: ['workflow-events', activeProjectId],
    queryFn: () => listWorkflowEvents(activeProjectId!, latestEventTimestampRef.current),
    enabled: Boolean(activeProjectId) && !wsHealthy,
    refetchInterval: wsHealthy ? false : 1500
  })

  useEffect(() => {
    if (eventsQuery.data?.events.length) {
      setStreamEvents(previous => mergeEvents(previous, eventsQuery.data.events))
    }
  }, [eventsQuery.data])

  useEffect(() => {
    if (!activeProjectId) {
      return
    }

    let disposed = false
    let socket: WebSocket | null = null

    void window.hermesDesktop
      .getConnection()
      .then(connection => {
        if (disposed) {
          return
        }

        const wsBase = connection.baseUrl.replace(/^http/i, connection.baseUrl.startsWith('https') ? 'wss' : 'ws')
        const since = latestEventTimestampRef.current

        const suffix = new URLSearchParams({
          token: connection.token,
          ...(typeof since === 'number' ? { since: String(since) } : {})
        })

        socket = new WebSocket(`${wsBase}/api/workflows/projects/${encodeURIComponent(activeProjectId)}/events?${suffix}`)
        socket.onopen = () => setWsHealthy(true)
        socket.onclose = () => setWsHealthy(false)
        socket.onerror = () => setWsHealthy(false)

        socket.onmessage = event => {
          try {
            const payload = JSON.parse(String(event.data)) as StreamEvent
            setStreamEvents(previous => mergeEvents(previous, [payload]))
          } catch {
            // Ignore malformed side-channel events; polling remains active on reconnect.
          }
        }
      })
      .catch(() => setWsHealthy(false))

    return () => {
      disposed = true
      setWsHealthy(false)
      socket?.close()
    }
  }, [activeProjectId])

  useEffect(() => {
    latestEventTimestampRef.current = streamEvents.at(-1)?.timestamp
  }, [streamEvents])

  const bundle = bundleQuery.data ?? null
  const workflow = bundle?.workflow ?? null

  const selectedNode = useMemo(
    () => workflow?.nodes.find(node => node.id === selectedNodeId) ?? workflow?.nodes[0] ?? null,
    [selectedNodeId, workflow]
  )

  const activeRun = bundle?.latestRun ?? null
  const runtimeNodeId = useMemo(() => latestWorkflowRuntimeNodeId(activeRun, streamEvents), [activeRun, streamEvents])
  const runtimeNode = useMemo(
    () => workflow?.nodes.find(node => node.id === runtimeNodeId) ?? null,
    [runtimeNodeId, workflow]
  )

  useEffect(() => {
    if (selectedNode && selectedNode.id !== selectedNodeId) {
      setSelectedNodeId(selectedNode.id)
    }
  }, [selectedNode, selectedNodeId])

  useEffect(() => {
    if (!autoFollowRunNode || !runtimeNode || selectedNodeId === runtimeNode.id) {
      return
    }

    setSelectedNodeId(runtimeNode.id)

    if (activeRun?.status === 'waiting_user_confirm') {
      setDrawerMode('task')
    }
  }, [activeRun?.status, autoFollowRunNode, runtimeNode, selectedNodeId])

  useEffect(() => {
    if (!workflow) {
      setFlowNodes([])
      setFlowEdges([])

      return
    }

    setFlowNodes(toFlowNodes(workflow))
    setFlowEdges(toFlowEdges(workflow))
  }, [setFlowEdges, setFlowNodes, workflow])

  const invalidateProject = useCallback(
    async (projectId = activeProjectId) => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['workflow-projects'] }),
        queryClient.invalidateQueries({ queryKey: ['workflow-project', projectId] }),
        queryClient.invalidateQueries({ queryKey: ['workflow-events', projectId] }),
        queryClient.invalidateQueries({ queryKey: ['workflow-files', projectId] })
      ])
    },
    [activeProjectId, queryClient]
  )

  const handleIntakeComplete = useCallback(
    async (data: ProjectBundle) => {
      queryClient.setQueryData<ProjectListResponse>(['workflow-projects'], current => ({
        projects: applyWorkflowProjectChange(current?.projects ?? [], { action: 'created', project: data.project })
      }))
      dispatchWorkflowProjectsChanged({ action: 'created', project: data.project })
      setActiveProjectId(data.project.id)
      setSelectedNodeId(data.workflow.nodes[0]?.id ?? null)
      setAutoFollowRunNode(true)
      setDrawerMode('task')
      setStreamEvents([])
      latestEventTimestampRef.current = undefined
      setSearchParams({ project: data.project.id })
      await invalidateProject(data.project.id)
    },
    [invalidateProject, queryClient, setSearchParams]
  )

  const generateMutation = useMutation({
    mutationFn: () => generateWorkflow(activeProjectId!),
    onSuccess: async data => {
      queryClient.setQueryData<ProjectListResponse>(['workflow-projects'], current => ({
        projects: applyWorkflowProjectChange(current?.projects ?? [], { action: 'updated', project: data.project })
      }))
      dispatchWorkflowProjectsChanged({ action: 'updated', project: data.project })
      setSelectedNodeId(data.workflow.nodes[0]?.id ?? null)
      setAutoFollowRunNode(true)
      await invalidateProject(data.project.id)
    }
  })

  const saveWorkflowMutation = useMutation({
    mutationFn: ({ nextWorkflow, snapshotLabel }: { nextWorkflow: Workflow; snapshotLabel: string }) =>
      saveWorkflow(activeProjectId!, nextWorkflow, snapshotLabel),
    onSuccess: data => invalidateProject(data.project.id)
  })

  const runMutation = useMutation({
    mutationFn: () => startWorkflowRun(activeProjectId!, { maxConcurrency: DEFAULT_MAX_CONCURRENCY, mode: executionMode }),
    onSuccess: data => invalidateProject(data.project?.id)
  })

  const pauseMutation = useMutation({
    mutationFn: (runId: string) => pauseWorkflowRun(runId),
    onSuccess: data => invalidateProject(data.run.projectId)
  })

  const stopMutation = useMutation({
    mutationFn: (runId: string) => stopWorkflowRun(runId),
    onSuccess: data => invalidateProject(data.run.projectId)
  })

  const nodeActionMutation = useMutation({
    mutationFn: ({ action, nodeId, runId }: { action: 'confirm' | 'retry' | 'skip'; nodeId: string; runId: string }) => {
      if (action === 'confirm') {
        return confirmWorkflowNode(runId, nodeId)
      }

      if (action === 'retry') {
        return retryWorkflowNode(runId, nodeId)
      }

      return skipWorkflowNode(runId, nodeId)
    },
    onSuccess: data => invalidateProject(data.run?.projectId ?? activeProjectId)
  })

  const chatMutation = useMutation({
    mutationFn: ({ attachments, text }: { attachments: string[]; text: string }) =>
      sendWorkflowChat({
        attachments,
        projectId: activeProjectId!,
        nodeId: selectedNode?.id ?? null,
        text
      }),
    onSuccess: () => invalidateProject()
  })

  const slashMutation = useMutation({
    mutationFn: (command: string) => executeWorkflowSlashCommand(activeProjectId!, { command, nodeId: selectedNode?.id ?? null }),
    onSuccess: () => invalidateProject()
  })

  const composerAttachmentMutation = useMutation({
    mutationFn: (paths: string[]) => attachWorkflowComposerFiles(activeProjectId!, paths),
    onSuccess: data => {
      if (data.references) {
        return invalidateProject()
      }

      return undefined
    }
  })

  const referencesMutation = useMutation({
    mutationFn: (references: ReferenceItem[]) => updateWorkflowReferences(activeProjectId!, references),
    onSuccess: data => invalidateProject(data.project.id)
  })

  const skillsMutation = useMutation({
    mutationFn: (skills: SkillBinding[]) => updateWorkflowSkills(activeProjectId!, skills),
    onSuccess: data => invalidateProject(data.project.id)
  })

  const snapshotMutation = useMutation({
    mutationFn: () => createWorkflowSnapshot(activeProjectId!),
    onSuccess: () => invalidateProject()
  })

  const onNodesChange = useCallback<OnNodesChange<FlowNode>>(
    changes => {
      onNodesChangeBase(changes)
    },
    [onNodesChangeBase]
  )

  const persistNodePosition = useCallback<OnNodeDrag<FlowNode>>(
    (_event, draggedNode) => {
      if (!workflow) {
        return
      }

      const nextNodes = flowNodes.map(node => (node.id === draggedNode.id ? draggedNode : node))
      saveWorkflowMutation.mutate({
        nextWorkflow: workflowWithPositions(workflow, nextNodes),
        snapshotLabel: 'Canvas node moved'
      })
    },
    [flowNodes, saveWorkflowMutation, workflow]
  )

  const onConnect = useCallback<OnConnect>(
    connection => {
      if (!workflow || !connection.source || !connection.target) {
        return
      }

      const edge: WorkflowEdge = {
        id: `edge-${connection.source}-${connection.target}-${Date.now()}`,
        source: connection.source,
        target: connection.target,
        type: 'dependency',
        label: copy.dependency,
        optional: false
      }

      saveWorkflowMutation.mutate({
        nextWorkflow: { ...workflow, edges: [...workflow.edges, edge], updatedAt: Date.now() / 1000 },
        snapshotLabel: 'Canvas edge created'
      })
    },
    [copy.dependency, saveWorkflowMutation, workflow]
  )

  const saveNodeConfig = useCallback(
    (nextNode: WorkflowNode) => {
      if (!workflow) {
        return
      }

      saveWorkflowMutation.mutate({
        nextWorkflow: {
          ...workflow,
          nodes: workflow.nodes.map(node => (node.id === nextNode.id ? nextNode : node)),
          updatedAt: Date.now() / 1000
        },
        snapshotLabel: `Node config updated: ${nextNode.title}`
      })
    },
    [saveWorkflowMutation, workflow]
  )

  const followRuntimeNode = useCallback(() => {
    setAutoFollowRunNode(true)

    if (runtimeNode) {
      setSelectedNodeId(runtimeNode.id)
      setDrawerMode('task')
    }
  }, [runtimeNode])

  const visibleEvents = useMemo(() => {
    if (!filterSelectedNode || !selectedNode) {
      return streamEvents
    }

    return streamEvents.filter(event => !event.nodeId || event.nodeId === selectedNode.id)
  }, [filterSelectedNode, selectedNode, streamEvents])

  const busy =
    generateMutation.isPending ||
    saveWorkflowMutation.isPending ||
    runMutation.isPending ||
    pauseMutation.isPending ||
    stopMutation.isPending ||
    nodeActionMutation.isPending ||
    slashMutation.isPending ||
    composerAttachmentMutation.isPending

  const showIntake = requestNewProject || (projectsQuery.isSuccess && !activeProjectId)

  return (
    <div className={cn('workflow-workbench', workflowLanguage === 'zh' && 'workflow-workbench--zh')}>
      {showIntake ? (
        <WorkflowIntakePage onComplete={handleIntakeComplete} />
      ) : (
        <>
          <ExecutionToolbar
            activeRun={activeRun}
            busy={busy}
            executionMode={executionMode}
            onModeChange={setExecutionMode}
            onPause={() => activeRun && pauseMutation.mutate(activeRun.id)}
            onRun={() => {
              setAutoFollowRunNode(true)
              runMutation.mutate()
            }}
            onStop={() => activeRun && stopMutation.mutate(activeRun.id)}
            project={bundle?.project ?? null}
            selectedNode={selectedNode}
          />

          <div className="workflow-main">
            <section aria-label="Workflow canvas workbench" className="workflow-center">
              <WorkflowFloatingToolbar
                active={drawerMode}
                onToggle={setDrawerMode}
              />
              <WorkflowStatusOverlay
                activeRun={activeRun}
                executionMode={executionMode}
                onFollowRuntimeNode={followRuntimeNode}
                runtimeNode={runtimeNode}
                selectedNode={selectedNode}
                workflow={workflow}
              />
          {bundleQuery.isLoading || projectsQuery.isLoading ? (
            <WorkbenchLoading />
          ) : workflow && workflow.nodes.length > 0 ? (
            <ReactFlow
              className="workflow-flow"
              colorMode={resolvedMode}
              connectOnClick={false}
              edges={flowEdges}
              elementsSelectable
              fitView
              maxZoom={1.4}
              minZoom={0.35}
              nodes={flowNodes}
              nodesConnectable
              nodesDraggable
              nodeTypes={nodeTypes}
              onConnect={onConnect}
              onEdgesChange={onEdgesChange}
              onNodeClick={(_event, node) => {
                setAutoFollowRunNode(false)
                setSelectedNodeId(node.id)
                setDrawerMode('task')
              }}
              onNodeDragStop={persistNodePosition}
              onNodesChange={onNodesChange}
              proOptions={{ hideAttribution: true }}
            >
              <Background color="var(--workflow-canvas-dot)" gap={24} size={1} />
              <MiniMap
                className="workflow-minimap"
                maskColor="var(--workflow-minimap-mask)"
                nodeColor={node => statusColor((node.data as WorkflowNodeData).node.status)}
                pannable
                zoomable
              />
              <Controls className="workflow-controls" showInteractive={false} />
            </ReactFlow>
          ) : (
            <EmptyWorkbench
              busy={generateMutation.isPending}
              hasProject={Boolean(activeProjectId)}
              onAddReference={() => setDrawerMode('references')}
              onCreate={() => setSearchParams({ new: '1' })}
              onGenerate={() => activeProjectId && generateMutation.mutate()}
            />
          )}

          <StreamOutputPanel
            events={visibleEvents}
            expanded={streamExpanded}
            filterSelectedNode={filterSelectedNode}
            onFilterSelectedNode={setFilterSelectedNode}
            onToggleExpanded={() => setStreamExpanded(value => !value)}
            selectedNode={selectedNode}
            wsHealthy={wsHealthy}
          />
          <WorkflowChatBox
            disabled={!activeProjectId || chatMutation.isPending || slashMutation.isPending}
            onAttach={paths => composerAttachmentMutation.mutateAsync(paths).then(() => undefined)}
            onSlash={command => slashMutation.mutate(command)}
            onSubmit={(text, attachments) => chatMutation.mutate({ attachments, text })}
            projectId={activeProjectId}
            projectRoot={bundle?.project.root}
            selectedNode={selectedNode}
          />
            </section>

        {(drawerMode === 'task' || drawerMode === 'files' || drawerMode === 'references' || drawerMode === 'skills' || drawerMode === 'snapshots') && (
          <RightDrawer
            activeRun={activeRun}
            artifacts={bundle?.artifacts ?? []}
            availableSkills={availableSkillsQuery.data ?? []}
            files={filesQuery.data?.tree ?? []}
            filesLoading={filesQuery.isLoading}
            mode={drawerMode}
            modelOptions={modelOptionsQuery.data ?? null}
            node={selectedNode}
            onAddReferences={paths => {
              const current = bundle?.references ?? []

              const next = [
                ...current,
                ...paths.map(path => referenceFromPath(path)).filter(ref => !current.some(item => item.path === ref.path))
              ]

              referencesMutation.mutate(next)
            }}
            onClose={() => setDrawerMode('task')}
            onNodeAction={(action, nodeId, runId) => {
              setAutoFollowRunNode(true)
              setSelectedNodeId(nodeId)
              nodeActionMutation.mutate({ action, nodeId, runId })
            }}
            onOpenFile={openPath}
            onSaveNode={saveNodeConfig}
            onSelectFile={setSelectedFilePath}
            onSnapshot={() => snapshotMutation.mutate()}
            onToggleReference={(reference, enabled) => {
              const references = (bundle?.references ?? []).map(item =>
                item.id === reference.id ? { ...item, enabled } : item
              )

              referencesMutation.mutate(references)
            }}
            onToggleSkill={(skill, enabled) => {
              const current = bundle?.skills.length ? bundle.skills : DEFAULT_SKILLS
              const skills = current.map(item => (item.id === skill.id ? { ...item, enabled } : item))
              skillsMutation.mutate(skills)
            }}
            references={bundle?.references ?? []}
            root={bundle?.project.root}
            selectedFilePath={selectedFilePath}
            skills={bundle?.skills.length ? bundle.skills : DEFAULT_SKILLS}
            snapshots={bundle?.snapshots ?? []}
          />
        )}
          </div>
        </>
      )}
    </div>
  )
}

function ExecutionToolbar({
  activeRun,
  busy,
  executionMode,
  onModeChange,
  onPause,
  onRun,
  onStop,
  project,
  selectedNode
}: {
  activeRun: ProjectBundle['latestRun']
  busy: boolean
  executionMode: ExecutionMode
  onModeChange: (mode: ExecutionMode) => void
  onPause: () => void
  onRun: () => void
  onStop: () => void
  project: ProjectBundle['project'] | null
  selectedNode: WorkflowNode | null
}) {
  const copy = useWorkflowCopy()
  const running = activeRun?.status === 'running'

  return (
    <section aria-label="Workflow execution controls" className="workflow-execution-toolbar">
      <div className="workflow-execution-toolbar__identity">
        <div className="workflow-title">{project?.name ?? 'hermes-workflow'}</div>
        <div className="workflow-subtitle">
          {selectedNode ? `${copy.currentNodePrefix}${selectedNode.title}` : project?.root ?? copy.workflowStartHint}
        </div>
      </div>

      <div className="workflow-execution-toolbar__controls">
        <Select onValueChange={value => onModeChange(value as ExecutionMode)} value={executionMode}>
          <SelectTrigger className="h-8 w-28 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="single_step">{copy.mode.single_step}</SelectItem>
            <SelectItem value="semi_auto">{copy.mode.semi_auto}</SelectItem>
            <SelectItem value="auto">{copy.mode.auto}</SelectItem>
          </SelectContent>
        </Select>

        <Button disabled={!project || busy || running} onClick={onRun} size="sm" type="button">
          <Codicon name="play" size="0.875rem" />
          {copy.run}
        </Button>
        <Button disabled={!running || busy} onClick={onPause} size="sm" type="button" variant="outline">
          <Codicon name="debug-pause" size="0.875rem" />
          {copy.pause}
        </Button>
        <Button disabled={!activeRun || busy || activeRun.status === 'stopped'} onClick={onStop} size="sm" type="button" variant="outline">
          <Codicon name="debug-stop" size="0.875rem" />
          {copy.stop}
        </Button>
      </div>
    </section>
  )
}

function WorkflowPageTitlebar({
  icon,
  subtitle,
  title
}: {
  icon: string
  subtitle: string
  title: string
}) {
  return (
    <header className={cn(titlebarHeaderBaseClass, 'workflow-page-titlebar')}>
      <Codicon className="workflow-page-titlebar__icon" name={icon} size="0.9375rem" />
      <div className="workflow-page-titlebar__copy">
        <h1>{title}</h1>
        <p>{subtitle}</p>
      </div>
    </header>
  )
}

function WorkflowStatusOverlay({
  activeRun,
  executionMode,
  onFollowRuntimeNode,
  runtimeNode,
  selectedNode,
  workflow
}: {
  activeRun: ProjectBundle['latestRun']
  executionMode: ExecutionMode
  onFollowRuntimeNode: () => void
  runtimeNode: WorkflowNode | null
  selectedNode: WorkflowNode | null
  workflow: Workflow | null
}) {
  const copy = useWorkflowCopy()
  const running = activeRun?.status === 'running'
  const waiting = activeRun?.status === 'waiting_user_confirm'
  const completed = workflow ? workflow.nodes.filter(node => node.status === 'completed').length : 0
  const total = workflow?.nodes.length ?? 0
  const displayedNode = runtimeNode ?? selectedNode

  return (
    <div className="workflow-status-overlay" aria-label="Workflow execution status">
      <span className={cn('workflow-run-dot', running && 'is-running', waiting && 'is-waiting')} />
      <span>{activeRun ? runStatusLabel(copy, activeRun.status) : copy.notRun}</span>
      <span>{total ? `${completed}/${total}` : '0/0'}</span>
      <span>{activeRun ? copy.mode[activeRun.mode] : copy.mode[executionMode]}</span>
      {runtimeNode ? (
        <button className="workflow-status-overlay__node" onClick={onFollowRuntimeNode} title={runtimeNode.title} type="button">
          <strong>{runtimeNode.title}</strong>
        </button>
      ) : (
        <strong title={displayedNode?.title ?? undefined}>{displayedNode ? displayedNode.title : copy.noNodeSelected}</strong>
      )}
    </div>
  )
}

function WorkflowFloatingToolbar({
  active,
  onToggle
}: {
  active: DrawerMode
  onToggle: (mode: DrawerMode) => void
}) {
  const copy = useWorkflowCopy()
  const items: Array<{ icon: string; label: string; mode: DrawerMode }> = [
    { icon: 'graph', label: copy.nodeDetails, mode: 'task' },
    { icon: 'files', label: copy.fileTree, mode: 'files' },
    { icon: 'references', label: copy.references, mode: 'references' },
    { icon: 'symbol-misc', label: copy.skills, mode: 'skills' },
    { icon: 'git-commit', label: copy.snapshots, mode: 'snapshots' }
  ]

  return (
    <div aria-label={copy.tools} className="workflow-floating-toolbar" role="toolbar">
      {items.map(item => (
        <button
          aria-label={item.label}
          className={cn(active === item.mode && 'is-active')}
          key={item.mode}
          onClick={() => onToggle(item.mode)}
          title={item.label}
          type="button"
        >
          <Codicon name={item.icon} size="1rem" />
        </button>
      ))}
    </div>
  )
}

function FileTreeDrawer({
  files,
  loading,
  onOpenFile,
  onOpenProjectRoot,
  onSelectFile,
  root,
  selectedFilePath
}: {
  files: WorkflowFileNode[]
  loading: boolean
  onOpenFile: (path: string) => void
  onOpenProjectRoot: () => void
  onSelectFile: (path: string) => void
  root?: string
  selectedFilePath: string | null
}) {
  const copy = useWorkflowCopy()
  const [expanded, setExpanded] = useState<Set<string>>(new Set())

  useEffect(() => {
    setExpanded(current => {
      if (current.size) {
        return current
      }

      return new Set(files.filter(item => item.kind === 'folder').map(item => item.path))
    })
  }, [files])

  const toggleExpanded = useCallback((path: string) => {
    setExpanded(current => {
      const next = new Set(current)

      if (next.has(path)) {
        next.delete(path)
      } else {
        next.add(path)
      }

      return next
    })
  }, [])

  return (
    <div className="workflow-file-drawer">
      <div className="workflow-drawer-header">
        <div>
          <h2>{copy.projectFiles}</h2>
          <p>{root ?? copy.noProjectSelected}</p>
        </div>
        <Button disabled={!root} onClick={onOpenProjectRoot} size="icon-sm" title={copy.openProjectInExplorer} type="button" variant="ghost">
          <Codicon name="folder-opened" size="0.875rem" />
        </Button>
      </div>
      <div className="workflow-file-tree">
        {loading ? (
          <div className="workflow-muted">Loading...</div>
        ) : files.length ? (
          files.map(item => (
            <FileTreeItem
              expanded={expanded}
              item={item}
              key={item.path}
              onOpenFile={onOpenFile}
              onSelectFile={onSelectFile}
              onToggleExpanded={toggleExpanded}
              openLabel={copy.openFile}
              selectedFilePath={selectedFilePath}
            />
          ))
        ) : (
          <div className="workflow-muted">{copy.noFiles}</div>
        )}
      </div>
    </div>
  )
}

function FileTreeItem({
  expanded,
  item,
  level = 0,
  onOpenFile,
  onSelectFile,
  onToggleExpanded,
  openLabel,
  selectedFilePath
}: {
  expanded: Set<string>
  item: WorkflowFileNode
  level?: number
  onOpenFile: (path: string) => void
  onSelectFile: (path: string) => void
  onToggleExpanded: (path: string) => void
  openLabel: string
  selectedFilePath: string | null
}) {
  const isFolder = item.kind === 'folder'
  const isExpanded = expanded.has(item.path)
  const selected = selectedFilePath === item.path

  return (
    <div>
      <div className={cn('workflow-file-row', selected && 'is-selected')} style={{ paddingLeft: `${level * 0.75 + 0.5}rem` }}>
        <button
          className="workflow-file-main"
          onClick={() => {
            if (isFolder) {
              onToggleExpanded(item.path)
            } else {
              onSelectFile(item.path)
            }
          }}
          onDoubleClick={() => !isFolder && onOpenFile(item.path)}
          title={item.path}
          type="button"
        >
          <Codicon name={isFolder ? (isExpanded ? 'folder-opened' : 'folder') : 'file'} size="0.8125rem" />
          <span>{item.name}</span>
        </button>
        {!isFolder && (
          <Button
            aria-label={openLabel}
            className="workflow-file-open"
            onClick={() => onOpenFile(item.path)}
            size="icon-xs"
            type="button"
            variant="ghost"
          >
            <Codicon name="go-to-file" size="0.75rem" />
          </Button>
        )}
      </div>
      {isFolder && isExpanded
        ? item.children?.map(child => (
            <FileTreeItem
              expanded={expanded}
              item={child}
              key={child.path}
              level={level + 1}
              onOpenFile={onOpenFile}
              onSelectFile={onSelectFile}
              onToggleExpanded={onToggleExpanded}
              openLabel={openLabel}
              selectedFilePath={selectedFilePath}
            />
          ))
        : null}
    </div>
  )
}

function RightDrawer({
  activeRun,
  artifacts,
  availableSkills,
  files,
  filesLoading,
  mode,
  modelOptions,
  node,
  onAddReferences,
  onOpenFile,
  onSaveNode,
  onSelectFile,
  onNodeAction,
  onSnapshot,
  onToggleReference,
  onToggleSkill,
  references,
  root,
  selectedFilePath,
  skills,
  snapshots
}: {
  activeRun: ProjectBundle['latestRun']
  artifacts: ProjectBundle['artifacts']
  availableSkills: SkillInfo[]
  files: WorkflowFileNode[]
  filesLoading: boolean
  mode: DrawerMode
  modelOptions: ModelOptionsResponse | null
  node: WorkflowNode | null
  onAddReferences: (paths: string[]) => void
  onClose: () => void
  onOpenFile: (path: string) => void
  onSaveNode: (node: WorkflowNode) => void
  onSelectFile: (path: string) => void
  onNodeAction: (action: 'confirm' | 'retry' | 'skip', nodeId: string, runId: string) => void
  onSnapshot: () => void
  onToggleReference: (reference: ReferenceItem, enabled: boolean) => void
  onToggleSkill: (skill: SkillBinding, enabled: boolean) => void
  references: ReferenceItem[]
  root?: string
  selectedFilePath: string | null
  skills: SkillBinding[]
  snapshots: VersionSnapshot[]
}) {
  const [width, setWidth] = useState(readStoredRightDrawerWidth)

  const beginResize = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      event.preventDefault()
      const startX = event.clientX
      const startWidth = width

      const onMove = (moveEvent: PointerEvent) => {
        const nextWidth = clampRightDrawerWidth(startWidth + startX - moveEvent.clientX)
        setWidth(nextWidth)
      }

      const onUp = (upEvent: PointerEvent) => {
        const nextWidth = clampRightDrawerWidth(startWidth + startX - upEvent.clientX)
        setWidth(nextWidth)
        persistRightDrawerWidth(nextWidth)
        window.removeEventListener('pointermove', onMove)
        window.removeEventListener('pointerup', onUp)
      }

      window.addEventListener('pointermove', onMove)
      window.addEventListener('pointerup', onUp)
    },
    [width]
  )

  return (
    <aside className="workflow-right-drawer" style={{ width }}>
      <div aria-hidden className="workflow-right-drawer__resize" onPointerDown={beginResize} />
      {mode === 'task' && (
        <TaskDetailDrawer
          activeRun={activeRun}
          artifacts={artifacts}
          availableSkills={availableSkills}
          modelOptions={modelOptions}
          node={node}
          onNodeAction={onNodeAction}
          onOpenFile={onOpenFile}
          onSaveNode={onSaveNode}
          root={root}
          selectedFilePath={selectedFilePath}
        />
      )}
      {mode === 'files' && (
        <FileTreeDrawer
          files={files}
          loading={filesLoading}
          onOpenFile={onOpenFile}
          onOpenProjectRoot={() => root && openPath(root)}
          onSelectFile={onSelectFile}
          root={root}
          selectedFilePath={selectedFilePath}
        />
      )}
      {mode === 'references' && (
        <ReferenceDrawer onAddReferences={onAddReferences} onToggleReference={onToggleReference} references={references} />
      )}
      {mode === 'skills' && <SkillDrawer onToggleSkill={onToggleSkill} skills={skills} />}
      {mode === 'snapshots' && <SnapshotDrawer onSnapshot={onSnapshot} snapshots={snapshots} />}
    </aside>
  )
}

function TaskDetailDrawer({
  activeRun,
  artifacts,
  availableSkills,
  modelOptions,
  node,
  onNodeAction,
  onOpenFile,
  onSaveNode,
  root,
  selectedFilePath
}: {
  activeRun: ProjectBundle['latestRun']
  artifacts: ProjectBundle['artifacts']
  availableSkills: SkillInfo[]
  modelOptions: ModelOptionsResponse | null
  node: WorkflowNode | null
  onNodeAction: (action: 'confirm' | 'retry' | 'skip', nodeId: string, runId: string) => void
  onOpenFile: (path: string) => void
  onSaveNode: (node: WorkflowNode) => void
  root?: string
  selectedFilePath: string | null
}) {
  const copy = useWorkflowCopy()
  const [draft, setDraft] = useState<WorkflowNode | null>(node)
  const [skillsOpen, setSkillsOpen] = useState(false)
  const [referencesOpen, setReferencesOpen] = useState(false)
  const [changesOpen, setChangesOpen] = useState(true)
  const [openFilePreviews, setOpenFilePreviews] = useState<Set<string>>(new Set())

  useEffect(() => {
    setDraft(node)
    setSkillsOpen(false)
    setReferencesOpen(false)
    setChangesOpen(true)
    setOpenFilePreviews(new Set())
  }, [node])

  if (!node) {
    return (
      <div className="workflow-drawer-empty">
        <Codicon name="graph" size="1.25rem" />
        <span>{copy.drawerEmptyNode}</span>
      </div>
    )
  }

  const editable = draft ?? node
  const status = statusMeta(copy, node.status)
  const runId = activeRun?.id ?? null
  const waiting = Boolean(runId && node.status === 'waiting_user_confirm')
  const nodeArtifacts = artifacts.filter(artifact => artifact.nodeId === node.id || node.artifacts.includes(artifact.path))
  const modelChoices = flattenModelChoices(modelOptions)
  const references = editable.references ?? []
  const fileChanges = editable.fileChanges ?? []
  const selectedFileForReference = selectedFilePath && root ? normalizeProjectReference(root, selectedFilePath) : selectedFilePath

  const updateDraft = (updates: Partial<WorkflowNode>) => {
    setDraft(current => ({ ...(current ?? node), ...updates }))
  }

  const toggleSkill = (skillName: string, enabled: boolean) => {
    const current = editable.skills ?? []
    updateDraft({
      skills: enabled ? [...new Set([...current, skillName])] : current.filter(skill => skill !== skillName)
    })
  }

  const addReference = (path: string) => {
    if (!path) {
      return
    }

    updateDraft({ references: [...new Set([...references, path])] })
  }

  const toggleFilePreview = (changeKey: string) => {
    setOpenFilePreviews(current => {
      const next = new Set(current)

      if (next.has(changeKey)) {
        next.delete(changeKey)
      } else {
        next.add(changeKey)
      }

      return next
    })
  }

  return (
    <div className="workflow-task-detail">
      <div className="workflow-drawer-header">
        <div>
          <h2>{node.title}</h2>
          <p>{node.id}</p>
        </div>
        <span className={cn('workflow-status-pill', `tone-${status.tone}`)}>{status.label}</span>
      </div>

      <div className="workflow-task-detail__body">
      <section>
        <div className="workflow-section-header">
          <h3>{copy.editExecutionPrompt}</h3>
          <Button
            disabled={!draft}
            onClick={() => draft && onSaveNode({ ...draft, promptOverride: draft.promptOverride?.trim() || null })}
            size="xs"
            type="button"
          >
            <Codicon name="save" size="0.8125rem" />
            {copy.save}
          </Button>
        </div>
        <Textarea
          className="workflow-prompt-editor"
          onChange={event => updateDraft({ promptOverride: event.target.value })}
          placeholder={node.description || copy.taskPlaceholder}
          value={editable.promptOverride ?? ''}
        />
        <p className="workflow-muted">{node.description}</p>
      </section>

      <section>
        <h3>{copy.context}</h3>
        <div className="workflow-key-values">
          <span>{copy.type}</span>
          <strong>{copy.nodeType[node.type as keyof typeof copy.nodeType] ?? node.type}</strong>
          <span>{copy.model}</span>
          <strong>{editable.modelOverride ?? editable.model ?? copy.globalModel}</strong>
          <span>{copy.skills}</span>
          <strong>{editable.skillMode === 'manual' ? `${editable.skills.length} ${copy.manualSkillsSummary}` : 'auto'}</strong>
          <span>{copy.retry}</span>
          <strong>
            {node.retryCount}/{node.maxRetries}
          </strong>
        </div>
      </section>

      <section>
        <h3>{copy.executionModel}</h3>
        <Select
          onValueChange={value => updateDraft({ modelOverride: value === '__inherit' ? null : value })}
          value={editable.modelOverride ?? '__inherit'}
        >
          <SelectTrigger className="h-8 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="__inherit">{copy.globalModel}</SelectItem>
            {modelChoices.map(choice => (
              <SelectItem key={`${choice.provider}-${choice.model}`} value={choice.model}>
                {choice.provider} / {choice.model}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </section>

      <section>
        <button className="workflow-collapsible-heading" onClick={() => setSkillsOpen(value => !value)} type="button">
          <h3>Skills</h3>
          <span>{editable.skillMode === 'manual' ? `${editable.skills.length} selected` : 'auto'}</span>
          <Codicon name={skillsOpen ? 'chevron-down' : 'chevron-right'} size="0.8125rem" />
        </button>
        {skillsOpen && (
          <div className="workflow-config-list">
            <label className="workflow-toggle-row">
              <span>
                <strong>{copy.autoCallHermesSkills}</strong>
                <small>{copy.autoSkillDescription}</small>
              </span>
              <Switch checked={editable.skillMode !== 'manual'} onCheckedChange={checked => updateDraft({ skillMode: checked ? 'auto' : 'manual' })} />
            </label>
            {editable.skillMode === 'manual' &&
              availableSkills.map(skill => (
                <label className="workflow-check-row" key={skill.name}>
                  <input
                    checked={(editable.skills ?? []).includes(skill.name)}
                    onChange={event => toggleSkill(skill.name, event.target.checked)}
                    type="checkbox"
                  />
                  <span>
                    <strong>{skill.name}</strong>
                    <small>{skill.category} · {skill.description}</small>
                  </span>
                </label>
              ))}
            {editable.skillMode === 'manual' && !availableSkills.length && <div className="workflow-muted">{copy.manualSkillsEmpty}</div>}
          </div>
        )}
      </section>

      <section>
        <button className="workflow-collapsible-heading" onClick={() => setReferencesOpen(value => !value)} type="button">
          <h3>{copy.nodeReferences}</h3>
          <span>{references.length}</span>
          <Codicon name={referencesOpen ? 'chevron-down' : 'chevron-right'} size="0.8125rem" />
        </button>
        {referencesOpen && (
          <div className="workflow-config-list">
            <div className="workflow-inline-actions">
              <Button
                onClick={() => {
                  void window.hermesDesktop
                    .selectPaths({ multiple: true, title: copy.chooseCurrentNodeReference })
                    .then(paths => paths.forEach(addReference))
                }}
                size="xs"
                type="button"
                variant="outline"
              >
                <Codicon name="add" size="0.8125rem" />
                {copy.addFiles}
              </Button>
              <Button disabled={!selectedFileForReference} onClick={() => selectedFileForReference && addReference(selectedFileForReference)} size="xs" type="button" variant="outline">
                <Codicon name="files" size="0.8125rem" />
                {copy.addSelectedFile}
              </Button>
            </div>
            {references.length ? (
              references.map(path => (
                <div className="workflow-reference-row" key={path}>
                  <span title={path}>{path}</span>
                  <Button onClick={() => onOpenFile(resolveProjectPath(root, path))} size="icon-xs" type="button" variant="ghost">
                    <Codicon name="go-to-file" size="0.75rem" />
                  </Button>
                  <Button onClick={() => updateDraft({ references: references.filter(item => item !== path) })} size="icon-xs" type="button" variant="ghost">
                    <Codicon name="close" size="0.75rem" />
                  </Button>
                </div>
              ))
            ) : (
              <div className="workflow-muted">{copy.noNodeReferences}</div>
            )}
          </div>
        )}
      </section>

      <section>
        <h3>Review Rules</h3>
        <ul className="workflow-checklist">
          {node.reviewRules.checklist.length ? (
            node.reviewRules.checklist.map(item => (
              <li key={item}>
                <Codicon name="check" size="0.75rem" />
                {item}
              </li>
            ))
          ) : (
            <li>{copy.noExplicitReviewRules}</li>
          )}
        </ul>
      </section>

      <section>
        <h3>Artifacts</h3>
        <div className="workflow-artifacts">
          {nodeArtifacts.length ? (
            nodeArtifacts.map(artifact => (
              <div className="workflow-artifact-row" key={artifact.id}>
                <Codicon name="file-code" size="0.8125rem" />
                <span title={artifact.path}>{artifact.name}</span>
              </div>
            ))
          ) : (
            <div className="workflow-muted">{copy.noArtifacts}</div>
          )}
        </div>
      </section>

      <section>
        <button className="workflow-collapsible-heading" onClick={() => setChangesOpen(value => !value)} type="button">
          <h3>{copy.fileChangeReview}</h3>
          <span>{fileChanges.length}</span>
          <Codicon name={changesOpen ? 'chevron-down' : 'chevron-right'} size="0.8125rem" />
        </button>
        {changesOpen && (
          <div className="workflow-file-changes">
            {fileChanges.length ? (
              fileChanges.map(change => {
                const changeKey = `${change.status}-${change.path}`
                const canPreview = fileChangeCanPreview(change)
                const previewOpen = openFilePreviews.has(changeKey)
                const meta = [
                  change.status,
                  change.isArtifact ? 'artifact' : null,
                  change.truncated ? 'truncated' : null,
                  change.isBinary ? copy.binaryFile : null
                ].filter(Boolean).join(' · ')

                return (
                  <div className="workflow-file-change" key={changeKey}>
                    <div className="workflow-file-change__header">
                      <button
                        aria-expanded={canPreview ? previewOpen : undefined}
                        className="workflow-file-change__summary"
                        disabled={!canPreview}
                        onClick={() => canPreview && toggleFilePreview(changeKey)}
                        type="button"
                      >
                        <Codicon name={canPreview ? (previewOpen ? 'chevron-down' : 'chevron-right') : 'circle-slash'} size="0.8125rem" />
                        <span>
                          <strong>{change.path}</strong>
                          <small>{meta}</small>
                        </span>
                      </button>
                      <div className="workflow-file-change__actions">
                        {canPreview && (
                          <Button onClick={() => toggleFilePreview(changeKey)} size="xs" type="button" variant="ghost">
                            {previewOpen ? copy.hidePreview : copy.preview}
                          </Button>
                        )}
                        <Button onClick={() => onOpenFile(resolveProjectPath(root, change.path))} size="xs" type="button" variant="outline">
                          <Codicon name="go-to-file" size="0.8125rem" />
                          {copy.open}
                        </Button>
                      </div>
                    </div>
                    {previewOpen && canPreview ? (
                      <pre>{change.diff || copy.noDiff}</pre>
                    ) : !canPreview ? (
                      <div className="workflow-file-change__notice">
                        {change.isBinary ? copy.binaryPreviewOmitted : copy.noTextPreviewAvailable}
                      </div>
                    ) : null}
                  </div>
                )
              })
            ) : (
              <div className="workflow-muted">{copy.fileChangeReviewEmpty}</div>
            )}
          </div>
        )}
      </section>
      </div>

      <div className="workflow-node-actions">
        <Button disabled={!waiting || !runId} onClick={() => runId && onNodeAction('confirm', node.id, runId)} size="sm" type="button">
          <Codicon name="pass" size="0.875rem" />
          {copy.confirm}
        </Button>
        <Button disabled={!runId} onClick={() => runId && onNodeAction('retry', node.id, runId)} size="sm" type="button" variant="outline">
          <Codicon name="refresh" size="0.875rem" />
          {copy.retry}
        </Button>
        <Button disabled={!runId} onClick={() => runId && onNodeAction('skip', node.id, runId)} size="sm" type="button" variant="outline">
          <Codicon name="debug-step-over" size="0.875rem" />
          {copy.skip}
        </Button>
      </div>
    </div>
  )
}

function ReferenceDrawer({
  onAddReferences,
  onToggleReference,
  references
}: {
  onAddReferences: (paths: string[]) => void
  onToggleReference: (reference: ReferenceItem, enabled: boolean) => void
  references: ReferenceItem[]
}) {
  const copy = useWorkflowCopy()

  return (
    <div className="workflow-reference-drawer">
      <div className="workflow-drawer-header">
        <div>
          <h2>{copy.references}</h2>
          <p>{copy.referenceContextHint}</p>
        </div>
        <Button
          onClick={() => {
            void window.hermesDesktop
              .selectPaths({ multiple: true, title: copy.chooseReference })
              .then(paths => paths.length && onAddReferences(paths))
          }}
          size="xs"
          type="button"
          variant="outline"
        >
          <Codicon name="add" size="0.8125rem" />
          {copy.add}
        </Button>
      </div>
      <div className="workflow-list">
        {references.length ? (
          references.map(reference => (
            <label className="workflow-toggle-row" key={reference.id}>
              <span>
                <strong>{reference.name}</strong>
                <small>{reference.path}</small>
              </span>
              <Switch checked={reference.enabled} onCheckedChange={enabled => onToggleReference(reference, enabled)} />
            </label>
          ))
        ) : (
          <div className="workflow-muted">{copy.noReferenceProject}</div>
        )}
      </div>
    </div>
  )
}

function SkillDrawer({
  onToggleSkill,
  skills
}: {
  onToggleSkill: (skill: SkillBinding, enabled: boolean) => void
  skills: SkillBinding[]
}) {
  const copy = useWorkflowCopy()

  return (
    <div>
      <div className="workflow-drawer-header">
        <div>
          <h2>{copy.skills}</h2>
          <p>{copy.skillProjectHint}</p>
        </div>
      </div>
      <div className="workflow-list">
        {skills.map(skill => (
          <label className="workflow-toggle-row" key={skill.id}>
            <span>
              <strong>{skill.name}</strong>
              <small>{skill.source}</small>
            </span>
            <Switch checked={skill.enabled} onCheckedChange={enabled => onToggleSkill(skill, enabled)} />
          </label>
        ))}
      </div>
    </div>
  )
}

function SnapshotDrawer({ onSnapshot, snapshots }: { onSnapshot: () => void; snapshots: VersionSnapshot[] }) {
  const copy = useWorkflowCopy()

  return (
    <div>
      <div className="workflow-drawer-header">
        <div>
          <h2>{copy.snapshots}</h2>
          <p>{copy.snapshotsHint}</p>
        </div>
        <Button onClick={onSnapshot} size="xs" type="button" variant="outline">
          <Codicon name="git-commit" size="0.8125rem" />
          {copy.snapshot}
        </Button>
      </div>
      <div className="workflow-snapshot-list">
        {snapshots.length ? (
          snapshots.map(snapshot => (
            <div className="workflow-snapshot-row" key={snapshot.id}>
              <Codicon name="git-commit" size="0.8125rem" />
              <span>
                <strong>{snapshot.label}</strong>
                <small>{formatDate(snapshot.createdAt)}</small>
              </span>
            </div>
          ))
        ) : (
          <div className="workflow-muted">{copy.noSnapshots}</div>
        )}
      </div>
    </div>
  )
}

function StreamOutputPanel({
  events,
  expanded,
  filterSelectedNode,
  onFilterSelectedNode,
  onToggleExpanded,
  selectedNode,
  wsHealthy
}: {
  events: StreamEvent[]
  expanded: boolean
  filterSelectedNode: boolean
  onFilterSelectedNode: (value: boolean) => void
  onToggleExpanded: () => void
  selectedNode: WorkflowNode | null
  wsHealthy: boolean
}) {
  const copy = useWorkflowCopy()
  const [height, setHeight] = useState(() => {
    const stored = Number(window.localStorage.getItem('hermes.workflow.streamHeight') || 260)

    return Number.isFinite(stored) ? Math.min(Math.max(stored, 160), Math.round(window.innerHeight * 0.6)) : 260
  })

  const transcript = useMemo(() => streamTranscriptItems(events), [events])

  useEffect(() => {
    window.localStorage.setItem('hermes.workflow.streamHeight', String(height))
  }, [height])

  const beginResize = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      event.preventDefault()
      const startY = event.clientY
      const startHeight = height
      const maxHeight = Math.round(window.innerHeight * 0.6)

      const onMove = (moveEvent: PointerEvent) => {
        const next = startHeight + startY - moveEvent.clientY
        setHeight(Math.min(Math.max(next, 160), maxHeight))
      }

      const onUp = () => {
        window.removeEventListener('pointermove', onMove)
        window.removeEventListener('pointerup', onUp)
      }

      window.addEventListener('pointermove', onMove)
      window.addEventListener('pointerup', onUp)
    },
    [height]
  )

  return (
    <section
      aria-label="Stream output panel"
      className={cn('workflow-stream-panel', expanded && 'is-expanded')}
      style={expanded ? { height } : undefined}
    >
      {expanded && <div className="workflow-stream-resizer" onPointerDown={beginResize} />}
      <div className="workflow-stream-panel__header">
        <button onClick={onToggleExpanded} type="button">
          <Codicon name={expanded ? 'chevron-down' : 'chevron-up'} size="0.875rem" />
          {copy.streamOutput}
        </button>
        <div className="workflow-stream-panel__tools">
          <span className={cn('workflow-ws-dot', wsHealthy && 'is-live')} />
          <span>{wsHealthy ? 'WS live' : 'polling'}</span>
          <label>
            <Switch checked={filterSelectedNode} disabled={!selectedNode} onCheckedChange={onFilterSelectedNode} />
            {copy.streamContextCurrentNode}
          </label>
        </div>
      </div>
      {expanded && (
        <div className="workflow-stream-transcript">
          {transcript.length ? (
            transcript.map(item =>
              item.kind === 'assistant' ? (
                <div className="workflow-transcript-message" data-slot="aui_assistant-message-root" key={item.id}>
                  <div className="workflow-transcript-header">
                    <Codicon name="sparkle" size="0.875rem" />
                    <strong>{item.label}</strong>
                    <span>{formatTime(item.timestamp)}</span>
                  </div>
                  <div data-slot="aui_assistant-message-content">
                    <CompactMarkdown className="workflow-transcript-markdown" text={item.text || '...'} />
                  </div>
                </div>
              ) : (
                <div className={cn('workflow-stream-compact-event', `status-${item.status}`)} key={item.id}>
                  <Codicon name={EVENT_ICON[item.type]} size="0.8125rem" />
                  <strong>{item.label}</strong>
                  <span>{item.text}</span>
                  <time>{formatTime(item.timestamp)}</time>
                </div>
              )
            )
          ) : (
            <div className="workflow-muted">{copy.runSummaryEmpty}</div>
          )}
        </div>
      )}
    </section>
  )
}

function WorkflowChatBox({
  disabled,
  onAttach,
  onSlash,
  onSubmit,
  projectId,
  projectRoot,
  selectedNode
}: {
  disabled: boolean
  onAttach: (paths: string[]) => Promise<void>
  onSlash: (command: string) => void
  onSubmit: (text: string, attachments: string[]) => void
  projectId: null | string
  projectRoot?: string
  selectedNode: WorkflowNode | null
}) {
  const copy = useWorkflowCopy()
  const [text, setText] = useState('')
  const [attachments, setAttachments] = useState<string[]>([])
  const [completions, setCompletions] = useState<WorkflowComposerCompletionItem[]>([])

  useEffect(() => {
    if (!projectId || disabled) {
      setCompletions([])

      return
    }

    const cursor = text.length
    const prefix = text.slice(0, cursor)
    const wantsCompletion = /\/[\w-]*$/.test(prefix) || /@file:[^\s`]*$/.test(prefix)

    if (!wantsCompletion) {
      setCompletions([])

      return
    }

    const handle = window.setTimeout(() => {
      void completeWorkflowComposer(projectId, { cursor, cwd: projectRoot, text })
        .then(result => setCompletions(result.items))
        .catch(() => setCompletions([]))
    }, 160)

    return () => window.clearTimeout(handle)
  }, [disabled, projectId, projectRoot, text])

  const submit = useCallback(() => {
    const trimmed = text.trim()

    if (!trimmed && attachments.length === 0) {
      return
    }

    if (/^\/\S+/.test(trimmed) && attachments.length === 0) {
      onSlash(trimmed)
    } else {
      onSubmit(trimmed, attachments)
    }

    setText('')
    setAttachments([])
    setCompletions([])
  }, [attachments, onSlash, onSubmit, text])

  const insertCompletion = useCallback(
    (item: WorkflowComposerCompletionItem) => {
      setText(current => {
        if (item.type === 'slash') {
          return current.replace(/\/[\w-]*$/, `${item.text} `)
        }

        return current.replace(/@file:[^\s`]*$/, `${item.text} `)
      })
      setCompletions([])
    },
    []
  )

  return (
    <form
      className="workflow-chat-box"
      onSubmit={event => {
        event.preventDefault()
        submit()
      }}
    >
      <div className="workflow-chat-box__context">
        <Codicon name="target" size="0.8125rem" />
        {selectedNode ? `${copy.currentNodePrefix}${selectedNode.title}` : copy.contextGlobal}
      </div>
      {attachments.length > 0 && (
        <div className="workflow-chat-box__attachments">
          {attachments.map(path => (
            <button
              key={path}
              onClick={() => setAttachments(current => current.filter(item => item !== path))}
              title={path}
              type="button"
            >
              <Codicon name="file" size="0.75rem" />
              <span>{fileName(path)}</span>
              <Codicon name="close" size="0.7rem" />
            </button>
          ))}
        </div>
      )}
      <Textarea
        disabled={disabled}
        onChange={event => setText(event.target.value)}
        onKeyDown={event => {
          if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
            event.currentTarget.form?.requestSubmit()
          }
        }}
        placeholder={copy.workflowChatPlaceholder}
        value={text}
      />
      {completions.length > 0 && (
        <div className="workflow-completion-popover">
          {completions.slice(0, 8).map(item => (
            <button key={`${item.type}-${item.text}`} onClick={() => insertCompletion(item)} onMouseDown={event => event.preventDefault()} type="button">
              <Codicon name={item.type === 'slash' ? 'terminal' : 'file'} size="0.75rem" />
              <span>{item.label}</span>
              {item.detail && <small>{item.detail}</small>}
            </button>
          ))}
        </div>
      )}
      <Button
        disabled={disabled || !projectId}
        onClick={() => {
          void window.hermesDesktop
            .selectPaths({ multiple: true, title: copy.chooseWorkflowAttachment })
            .then(async paths => {
              if (!paths.length) {
                return
              }

              await onAttach(paths)
              setAttachments(current => [...new Set([...current, ...paths])])
            })
        }}
        size="icon-sm"
        title={copy.addAttachments}
        type="button"
        variant="outline"
      >
        <Codicon name="attach" size="0.875rem" />
      </Button>
      <Button disabled={disabled || (!text.trim() && attachments.length === 0)} size="icon-sm" type="submit">
        <Codicon name="send" size="0.875rem" />
      </Button>
    </form>
  )
}

function WorkflowIntakePage({ onComplete }: { onComplete: (bundle: ProjectBundle) => void | Promise<void> }) {
  const copy = useWorkflowCopy()
  const [name, setName] = useState<string>(copy.workflowProjectDefaultName)
  const [goal, setGoal] = useState('')
  const [root, setRoot] = useState('')
  const [references, setReferences] = useState<string[]>([])
  const [intakeId, setIntakeId] = useState<string | null>(null)
  const [projectId, setProjectId] = useState<string | null>(null)
  const [messages, setMessages] = useState<WorkflowIntakeMessage[]>([])
  const [currentBatch, setCurrentBatch] = useState<WorkflowIntakeBatch | null>(null)
  const [answeredCount, setAnsweredCount] = useState(0)
  const [intakeError, setIntakeError] = useState<string | null>(null)
  const [reply, setReply] = useState('')
  const [summary, setSummary] = useState('')
  const [ready, setReady] = useState(false)

  const payload = useMemo<WorkflowIntakePayload>(
    () => ({ goal, name, references, root: root || undefined }),
    [goal, name, references, root]
  )

  const applyIntakeResponse = useCallback((response: WorkflowIntakeResponse) => {
    setIntakeId(response.intakeId)
    setProjectId(response.projectId ?? null)
    setMessages(response.messages)
    setCurrentBatch(response.currentBatch ?? null)
    setAnsweredCount(response.answeredCount ?? 0)
    setIntakeError(response.error ?? null)
    setReady(response.ready)
    setSummary(response.summary)
  }, [])

  const startMutation = useMutation({
    mutationFn: () => startWorkflowIntake(payload),
    onSuccess: applyIntakeResponse
  })

  const messageMutation = useMutation({
    mutationFn: (message: string) => sendWorkflowIntakeMessage(intakeId!, message),
    onSuccess: data => {
      setReply('')
      applyIntakeResponse(data)
    }
  })

  const answersMutation = useMutation({
    mutationFn: (answers: WorkflowIntakeAnswer[]) => submitWorkflowIntakeAnswers(intakeId!, answers),
    onSuccess: applyIntakeResponse
  })

  const confirmMutation = useMutation({
    mutationFn: () => confirmWorkflowIntake(intakeId!, { ...payload, projectId, summary }),
    onSuccess: data => void onComplete(data)
  })

  const busy = startMutation.isPending || messageMutation.isPending || answersMutation.isPending || confirmMutation.isPending
  const error = errorText(startMutation.error || messageMutation.error || answersMutation.error || confirmMutation.error) || intakeError

  return (
    <main className="workflow-intake-page">
      <WorkflowPageTitlebar
        icon="graph"
        subtitle={copy.workflowIntakeSubtitle}
        title={copy.workflowIntakeTitle}
      />
      <section aria-label="Workflow intake" className="workflow-intake-layout">
        <div className="workflow-intake-config">
          <div className="workflow-intake-section-heading">
            <h2>{copy.projectConfig}</h2>
            <p>{copy.projectConfigHint}</p>
          </div>

          <label>
            {copy.projectName}
            <Input disabled={busy && Boolean(intakeId)} onChange={event => setName(event.target.value)} value={name} />
          </label>

          <label>
            {copy.projectDirectory}
            <div className="workflow-path-picker">
              <Input
                disabled={busy && Boolean(intakeId)}
                onChange={event => setRoot(event.target.value)}
                placeholder={copy.projectDirectoryPlaceholder}
                value={root}
              />
              <Button
                disabled={busy && Boolean(intakeId)}
                onClick={() => {
                  void window.hermesDesktop
                    .selectPaths({ directories: true, title: copy.chooseWorkflowProjectDirectory })
                    .then(paths => paths[0] && setRoot(paths[0]))
                }}
                type="button"
                variant="outline"
              >
                {copy.open}
              </Button>
            </div>
          </label>

          <label>
            {copy.projectBackground}
            <Textarea
              className="min-h-32"
              disabled={busy && Boolean(intakeId)}
              onChange={event => setGoal(event.target.value)}
              placeholder={copy.taskPlaceholder}
              value={goal}
            />
          </label>

          <div>
            <div className="workflow-dialog-row">
              <span>{copy.references}</span>
              <Button
                disabled={busy && Boolean(intakeId)}
                onClick={() => {
                  void window.hermesDesktop
                    .selectPaths({ multiple: true, title: copy.chooseReference })
                    .then(paths => setReferences(current => [...new Set([...current, ...paths])]))
                }}
                size="xs"
                type="button"
                variant="outline"
              >
                <Codicon name="add" size="0.8125rem" />
                {copy.add}
              </Button>
            </div>
            <div className="workflow-reference-preview">
              {references.length ? references.map(path => <span key={path}>{path}</span>) : <span>{copy.noReference}</span>}
            </div>
          </div>

          <Button disabled={busy || !name.trim() || Boolean(intakeId)} onClick={() => startMutation.mutate()} type="button">
            <Codicon name={startMutation.isPending ? 'loading' : 'comment-discussion'} size="0.875rem" spinning={startMutation.isPending} />
            {copy.startClarification}
          </Button>
        </div>

        <div className="workflow-intake-chat">
          <div className="workflow-intake-transcript">
            {messages.length ? (
              messages.map((message, index) => (
                <div className={cn('workflow-intake-message', message.role === 'user' && 'is-user')} key={`${message.timestamp}-${index}`}>
                  <span>{message.role === 'user' ? copy.you : 'Hermes'}</span>
                  <CompactMarkdown className="workflow-intake-message-markdown" text={message.content || '...'} />
                </div>
              ))
            ) : (
              <div className="workflow-muted">{copy.projectConfigHint}</div>
            )}
          </div>

          {currentBatch && (
            <WorkflowClarificationBatchCard
              batch={currentBatch}
              disabled={busy}
              onSubmit={answers => answersMutation.mutate(answers)}
              submitting={answersMutation.isPending}
            />
          )}

          <form
            className="workflow-intake-reply"
            onSubmit={event => {
              event.preventDefault()

              if (!intakeId || !reply.trim()) {
                return
              }

              messageMutation.mutate(reply)
            }}
          >
            <Textarea
              disabled={!intakeId || busy}
              onChange={event => setReply(event.target.value)}
              onKeyDown={event => {
                if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
                  event.currentTarget.form?.requestSubmit()
                }
              }}
              placeholder={copy.planDetailsPlaceholder}
              value={reply}
            />
            <Button disabled={!intakeId || busy || !reply.trim()} size="icon-sm" type="submit">
              <Codicon name={messageMutation.isPending ? 'loading' : 'send'} size="0.875rem" spinning={messageMutation.isPending} />
            </Button>
          </form>
        </div>

        <aside className="workflow-intake-summary">
          <div className="workflow-drawer-header">
            <div>
              <h2>{copy.summaryTitle}</h2>
              <p>{ready ? copy.intakeReady : currentBatch ? copy.intakeClarifying : copy.intakeWaiting}</p>
            </div>
          </div>
          <div className="workflow-intake-stats">
            <span>{projectId ? copy.draftProjectCreated : copy.draftProjectPending}</span>
            <span>{copy.answeredQuestions}: {answeredCount}</span>
          </div>
          <pre>{summary || copy.summaryPlaceholder}</pre>
          {error && <div className="workflow-error">{error}</div>}
          {confirmMutation.isPending && <div className="workflow-status">{copy.startingProjectStatus}</div>}
          <Button disabled={!intakeId || !ready || Boolean(currentBatch) || confirmMutation.isPending} onClick={() => confirmMutation.mutate()} type="button">
            <Codicon name={confirmMutation.isPending ? 'loading' : 'sparkle'} size="0.875rem" spinning={confirmMutation.isPending} />
            {copy.confirmAndGenerate}
          </Button>
        </aside>
      </section>
    </main>
  )
}

function WorkflowClarificationBatchCard({
  batch,
  disabled = false,
  onSubmit,
  submitting = false
}: {
  batch: WorkflowIntakeBatch
  disabled?: boolean
  onSubmit: (answers: WorkflowIntakeAnswer[]) => void
  submitting?: boolean
}) {
  const copy = useWorkflowCopy()
  const [index, setIndex] = useState(0)
  const [selectedOptions, setSelectedOptions] = useState<Record<string, string>>({})
  const [customAnswers, setCustomAnswers] = useState<Record<string, string>>({})
  const [answers, setAnswers] = useState<Record<string, WorkflowIntakeAnswer>>({})
  const questions = batch.questions
  const question = questions[index]

  useEffect(() => {
    setIndex(0)
    setSelectedOptions({})
    setCustomAnswers({})
    setAnswers({})
  }, [batch.id])

  if (!question) {
    return null
  }

  const confirmed = Boolean(answers[question.id])
  const selectedOptionId = selectedOptions[question.id] ?? null
  const customAnswer = customAnswers[question.id]?.trim() ?? ''
  const selectedOption = question.options.find(option => option.id === selectedOptionId) ?? null
  const canConfirm = Boolean(customAnswer || selectedOption)
  const confirmedCount = questions.filter(item => answers[item.id]).length

  const confirmCurrent = () => {
    if (!canConfirm || disabled || submitting) {
      return
    }

    const answer: WorkflowIntakeAnswer = {
      questionId: question.id,
      optionId: customAnswer ? selectedOptionId : selectedOption?.id ?? null,
      answer: customAnswer || selectedOption?.label || '',
      custom: Boolean(customAnswer)
    }
    const nextAnswers = { ...answers, [question.id]: answer }
    setAnswers(nextAnswers)
    const nextQuestionIndex = questions.findIndex(item => !nextAnswers[item.id])
    if (nextQuestionIndex === -1) {
      onSubmit(questions.map(item => nextAnswers[item.id]!))
      return
    }
    setIndex(nextQuestionIndex)
  }

  return (
    <section className="workflow-clarification-card" aria-label={copy.clarificationQuestions}>
      <div className="workflow-clarification-card__header">
        <div>
          <span>{copy.clarificationQuestions}</span>
          <strong>{index + 1}/{questions.length}</strong>
        </div>
        <div className="workflow-clarification-card__nav">
          <Button
            aria-label={copy.previousQuestion}
            disabled={disabled || submitting || index === 0}
            onClick={() => setIndex(value => Math.max(0, value - 1))}
            size="icon-sm"
            type="button"
            variant="ghost"
          >
            <Codicon name="chevron-left" size="0.875rem" />
          </Button>
          <Button
            aria-label={copy.nextQuestion}
            disabled={disabled || submitting || index >= questions.length - 1}
            onClick={() => setIndex(value => Math.min(questions.length - 1, value + 1))}
            size="icon-sm"
            type="button"
            variant="ghost"
          >
            <Codicon name="chevron-right" size="0.875rem" />
          </Button>
        </div>
      </div>

      <div className="workflow-clarification-question">
        <h3>{question.question}</h3>
        {question.detail && <p>{question.detail}</p>}
      </div>

      <div className="workflow-clarification-options">
        {question.options.map(option => (
          <button
            className={cn('workflow-clarification-option', selectedOptionId === option.id && 'is-selected')}
            disabled={disabled || submitting}
            key={option.id}
            onClick={() => setSelectedOptions(current => ({ ...current, [question.id]: option.id }))}
            type="button"
          >
            <span>
              {copy.priorityOption} {option.priority}
            </span>
            <strong>{option.label}</strong>
            {option.description && <small>{option.description}</small>}
          </button>
        ))}
      </div>

      <label className="workflow-clarification-custom">
        {copy.customAnswer}
        <Textarea
          disabled={disabled || submitting}
          onChange={event => setCustomAnswers(current => ({ ...current, [question.id]: event.target.value }))}
          placeholder={copy.typeCustomAnswer}
          value={customAnswers[question.id] ?? ''}
        />
      </label>

      <div className="workflow-clarification-footer">
        <span>
          {copy.answeredQuestions}: {confirmedCount}/{questions.length}
          {confirmed && ` · ${copy.answerConfirmed}`}
        </span>
        <Button disabled={!canConfirm || disabled || submitting} onClick={confirmCurrent} type="button">
          <Codicon name={submitting ? 'loading' : confirmed ? 'check' : 'send'} size="0.875rem" spinning={submitting} />
          {submitting ? copy.clarificationSubmitting : confirmed ? copy.answerConfirmed : copy.confirmAnswer}
        </Button>
      </div>
    </section>
  )
}

function WorkbenchLoading() {
  const copy = useWorkflowCopy()

  return (
    <div className="workflow-empty">
      <Codicon name="loading" size="1.5rem" spinning />
      <span>{copy.loadingWorkbench}</span>
    </div>
  )
}

function EmptyWorkbench({
  busy = false,
  hasProject = false,
  onAddReference,
  onCreate,
  onGenerate
}: {
  busy?: boolean
  hasProject?: boolean
  onAddReference?: () => void
  onCreate: () => void
  onGenerate?: () => void
}) {
  const copy = useWorkflowCopy()

  if (hasProject) {
    return (
      <div className="workflow-empty">
        <Codicon name={busy ? 'loading' : 'graph'} size="1.5rem" spinning={busy} />
        <span>{busy ? copy.agentGeneratingWorkflow : copy.canvasEmpty}</span>
        <div className="workflow-empty__actions">
          <Button disabled={busy} onClick={onGenerate} type="button">
            <Codicon name={busy ? 'loading' : 'sparkle'} size="0.875rem" spinning={busy} />
            {copy.workflowGenerationEmptyAction}
          </Button>
          <Button disabled={busy} onClick={onAddReference} type="button" variant="outline">
            <Codicon name="references" size="0.875rem" />
            {copy.workflowGenerationReferencesAction}
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="workflow-empty">
      <Codicon name="graph" size="1.5rem" />
      <span>{copy.workflowStartHint}</span>
      <Button onClick={onCreate} type="button">
        {copy.newWorkflowProject}
      </Button>
    </div>
  )
}

function toFlowNodes(workflow: Workflow): FlowNode[] {
  return workflow.nodes.map(node => ({
    id: node.id,
    type: 'workflow',
    position: node.position,
    width: 260,
    height: 112,
    data: { node }
  }))
}

function toFlowEdges(workflow: Workflow): FlowEdge[] {
  return workflow.edges.map(edge => ({
    id: edge.id,
    source: edge.source,
    target: edge.target,
    label: edge.label || undefined,
    type: edge.type === 'feedback' ? 'smoothstep' : 'default',
    animated: edge.type === 'feedback',
    markerEnd: edge.type === 'feedback' ? undefined : { type: MarkerType.ArrowClosed },
    style:
      edge.type === 'feedback'
        ? { stroke: 'var(--workflow-edge-feedback)', strokeDasharray: '6 5', strokeWidth: 1.8 }
        : { stroke: 'var(--workflow-edge)', strokeWidth: 1.6 },
    data: { kind: edge.type }
  }))
}

function workflowWithPositions(workflow: Workflow, nodes: FlowNode[]): Workflow {
  const byId = new Map(nodes.map(node => [node.id, node.position]))

  return {
    ...workflow,
    nodes: workflow.nodes.map(node => ({
      ...node,
      position: byId.get(node.id) ?? node.position
    })),
    updatedAt: Date.now() / 1000
  }
}

function mergeEvents(previous: StreamEvent[], incoming: StreamEvent[]): StreamEvent[] {
  const byId = new Map<string, StreamEvent>()

  for (const event of previous) {
    byId.set(event.id, event)
  }

  for (const event of incoming) {
    byId.set(event.id, event)
  }

  return [...byId.values()].sort((a, b) => a.timestamp - b.timestamp).slice(-500)
}

const FOLLOWABLE_RUN_STATUSES = new Set(['running', 'waiting_user_confirm', 'paused'])
const RUNTIME_EVENT_TYPES = new Set<StreamEvent['type']>(['node_status', 'approval'])

function latestWorkflowRuntimeNodeId(activeRun: ProjectBundle['latestRun'], events: StreamEvent[]): string | null {
  if (activeRun?.currentNodeId) {
    return activeRun.currentNodeId
  }

  if (!activeRun || !FOLLOWABLE_RUN_STATUSES.has(activeRun.status)) {
    return null
  }

  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index]

    if (!event.nodeId || !RUNTIME_EVENT_TYPES.has(event.type)) {
      continue
    }

    if (!event.runId || event.runId === activeRun.id) {
      return event.nodeId
    }
  }

  return null
}

function fileChangeCanPreview(change: WorkflowNode['fileChanges'][number]): boolean {
  return change.previewable !== false && !change.isBinary && Boolean(change.diff)
}

type WorkflowTranscriptItem =
  | {
      id: string
      kind: 'assistant'
      label: string
      text: string
      timestamp: number
    }
  | {
      id: string
      kind: 'event'
      label: string
      status: string
      text: string
      timestamp: number
      type: StreamEvent['type']
    }

function streamTranscriptItems(events: StreamEvent[]): WorkflowTranscriptItem[] {
  const items: WorkflowTranscriptItem[] = []
  const assistantByKey = new Map<string, Extract<WorkflowTranscriptItem, { kind: 'assistant' }>>()

  for (const event of events) {
    if (event.type === 'ai_reply') {
      const details = event.details ?? {}
      const messageId = typeof details.messageId === 'string' ? details.messageId : null
      const key = messageId ?? `legacy-${event.runId ?? 'global'}-${event.nodeId ?? 'global'}-${event.label}`

      const text =
        typeof details.text === 'string'
          ? details.text
          : typeof details.delta === 'string'
            ? details.delta
            : event.summary

      const existing = assistantByKey.get(key)

      if (existing) {
        existing.text = text || existing.text
        existing.timestamp = event.timestamp
        existing.label = event.label || existing.label
      } else {
        const item: Extract<WorkflowTranscriptItem, { kind: 'assistant' }> = {
          id: key,
          kind: 'assistant',
          label: event.label,
          text,
          timestamp: event.timestamp
        }

        assistantByKey.set(key, item)
        items.push(item)
      }

      continue
    }

    items.push({
      id: event.id,
      kind: 'event',
      label: event.label,
      status: event.status,
      text: event.summary,
      timestamp: event.timestamp,
      type: event.type
    })
  }

  return items
}

function referenceFromPath(path: string): ReferenceItem {
  const name = path.replace(/[/\\]+$/, '').split(/[/\\]/).pop() || path

  return {
    id: `ref_${crypto.randomUUID().slice(0, 12)}`,
    name,
    path,
    enabled: true,
    kind: 'file',
    addedAt: Date.now() / 1000
  }
}

function fileName(path: string): string {
  return path.replace(/[/\\]+$/, '').split(/[/\\]/).pop() || path
}

function statusColor(status: WorkflowNodeStatus): string {
  const tone = STATUS_TONE[status] ?? 'neutral'

  const colors: Record<string, string> = {
    danger: 'var(--ui-red)',
    info: 'var(--ui-blue)',
    neutral: 'var(--ui-text-tertiary)',
    ready: 'var(--ui-blue)',
    running: 'var(--ui-purple)',
    success: 'var(--ui-green)',
    warning: 'var(--ui-yellow)'
  }

  return colors[tone] ?? colors.neutral
}

function runStatusLabel(copy: WorkflowCopy, status: string): string {
  return copy.runStatus[status as keyof typeof copy.runStatus] ?? status
}

function formatTime(timestamp: number): string {
  return new Date(timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

function formatDate(timestamp: number): string {
  return new Date(timestamp * 1000).toLocaleString()
}

function flattenModelChoices(options: ModelOptionsResponse | null): Array<{ model: string; provider: string }> {
  const choices: Array<{ model: string; provider: string }> = []
  const seen = new Set<string>()

  for (const provider of options?.providers ?? []) {
    for (const model of provider.models ?? []) {
      const key = `${provider.slug}:${model}`

      if (seen.has(key)) {
        continue
      }

      seen.add(key)
      choices.push({ model, provider: provider.name || provider.slug })
    }
  }

  return choices
}

function clampRightDrawerWidth(width: number): number {
  return Math.min(RIGHT_DRAWER_MAX_WIDTH, Math.max(RIGHT_DRAWER_MIN_WIDTH, Math.round(width)))
}

function readStoredRightDrawerWidth(): number {
  if (typeof window === 'undefined') {
    return RIGHT_DRAWER_DEFAULT_WIDTH
  }
  try {
    const raw = window.localStorage.getItem(RIGHT_DRAWER_WIDTH_KEY)
    const parsed = raw ? Number(raw) : RIGHT_DRAWER_DEFAULT_WIDTH

    return Number.isFinite(parsed) ? clampRightDrawerWidth(parsed) : RIGHT_DRAWER_DEFAULT_WIDTH
  } catch {
    return RIGHT_DRAWER_DEFAULT_WIDTH
  }
}

function persistRightDrawerWidth(width: number): void {
  try {
    window.localStorage.setItem(RIGHT_DRAWER_WIDTH_KEY, String(clampRightDrawerWidth(width)))
  } catch {
    // Drawer width is a local UI preference; ignore restricted storage.
  }
}

function openPath(path: string): void {
  if (!path) {
    return
  }

  void window.hermesDesktop.openExternal(pathToFileUrl(path))
}

function pathToFileUrl(path: string): string {
  const normalized = path.replace(/\\/g, '/')

  if (normalized.startsWith('/')) {
    return encodeURI(`file://${normalized}`)
  }

  return encodeURI(`file:///${normalized}`)
}

function resolveProjectPath(root: string | undefined, path: string): string {
  if (!path || /^[a-zA-Z]:[\\/]/.test(path) || path.startsWith('/') || path.startsWith('\\\\')) {
    return path
  }

  if (!root) {
    return path
  }

  return `${root.replace(/[\\/]+$/, '')}\\${path.replace(/\//g, '\\')}`
}

function normalizeProjectReference(root: string, path: string): string {
  const normalizedRoot = root.replace(/\\/g, '/').replace(/\/+$/, '').toLowerCase()
  const normalizedPath = path.replace(/\\/g, '/')

  if (normalizedPath.toLowerCase().startsWith(`${normalizedRoot}/`)) {
    return normalizedPath.slice(normalizedRoot.length + 1)
  }

  return path
}

function errorText(error: unknown): string | undefined {
  if (!error) {
    return undefined
  }

  return error instanceof Error ? error.message : String(error)
}
