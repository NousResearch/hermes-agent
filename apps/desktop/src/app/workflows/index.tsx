import '@xyflow/react/dist/style.css'
import './workflows.css'

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
  stopWorkflowRun,
  updateWorkflowReferences,
  updateWorkflowSkills
} from '@/hermes'
import { cn } from '@/lib/utils'
import type { ModelOptionsResponse, SkillInfo } from '@/types/hermes'
import type {
  ExecutionMode,
  ProjectBundle,
  ReferenceItem,
  SkillBinding,
  StreamEvent,
  VersionSnapshot,
  Workflow,
  WorkflowComposerCompletionItem,
  WorkflowEdge,
  WorkflowFileNode,
  WorkflowIntakeMessage,
  WorkflowIntakePayload,
  WorkflowNode,
  WorkflowNodeStatus
} from '@/types/workflow'

import { titlebarHeaderBaseClass } from '../shell/titlebar'

type DrawerMode = 'files' | 'references' | 'skills' | 'snapshots' | 'task'

interface WorkflowNodeData extends Record<string, unknown> {
  node: WorkflowNode
}

type FlowNode = Node<WorkflowNodeData, 'workflow'>
type FlowEdge = Edge<{ kind: string }>

const STATUS_META: Record<WorkflowNodeStatus, { label: string; tone: string }> = {
  aborted: { label: '已中止', tone: 'danger' },
  completed: { label: '已完成', tone: 'success' },
  created: { label: '已创建', tone: 'neutral' },
  failed: { label: '失败', tone: 'danger' },
  queued: { label: '排队中', tone: 'info' },
  ready: { label: '就绪', tone: 'ready' },
  retrying: { label: '重试中', tone: 'warning' },
  reviewing: { label: '审查中', tone: 'warning' },
  revision_needed: { label: '需修订', tone: 'warning' },
  running: { label: '运行中', tone: 'running' },
  skipped: { label: '已跳过', tone: 'neutral' },
  waiting_user_confirm: { label: '待确认', tone: 'warning' }
}

const MODE_LABEL: Record<ExecutionMode, string> = {
  auto: '自动',
  semi_auto: '半自动',
  single_step: '单步'
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
  const status = STATUS_META[data.node.status]
  const hasReview = data.node.reviewRules.required

  return (
    <div className={cn('workflow-node-card', selected && 'is-selected', `tone-${status.tone}`)}>
      <Handle className="workflow-handle" position={Position.Left} type="target" />
      <div className="workflow-node-card__top">
        <span className="workflow-node-card__type">{data.node.type}</span>
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
  const queryClient = useQueryClient()
  const [searchParams, setSearchParams] = useSearchParams()
  const [activeProjectId, setActiveProjectId] = useState<string | null>(null)
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
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
      setStreamEvents([])
      latestEventTimestampRef.current = undefined

      return
    }

    if (requestNewProject && activeProjectId) {
      setActiveProjectId(null)
      setSelectedNodeId(null)
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

  useEffect(() => {
    if (selectedNode && selectedNode.id !== selectedNodeId) {
      setSelectedNodeId(selectedNode.id)
    }
  }, [selectedNode, selectedNodeId])

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
      setActiveProjectId(data.project.id)
      setSelectedNodeId(data.workflow.nodes[0]?.id ?? null)
      setDrawerMode('task')
      setStreamEvents([])
      latestEventTimestampRef.current = undefined
      setSearchParams({ project: data.project.id })
      await invalidateProject(data.project.id)
    },
    [invalidateProject, setSearchParams]
  )

  const generateMutation = useMutation({
    mutationFn: () => generateWorkflow(activeProjectId!),
    onSuccess: async data => {
      setSelectedNodeId(data.workflow.nodes[0]?.id ?? null)
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
        label: '依赖',
        optional: false
      }

      saveWorkflowMutation.mutate({
        nextWorkflow: { ...workflow, edges: [...workflow.edges, edge], updatedAt: Date.now() / 1000 },
        snapshotLabel: 'Canvas edge created'
      })
    },
    [saveWorkflowMutation, workflow]
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
    <div className="workflow-workbench">
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
            onRun={() => runMutation.mutate()}
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
              <WorkflowStatusOverlay activeRun={activeRun} executionMode={executionMode} selectedNode={selectedNode} workflow={workflow} />
          {bundleQuery.isLoading || projectsQuery.isLoading ? (
            <WorkbenchLoading />
          ) : workflow && workflow.nodes.length > 0 ? (
            <ReactFlow
              className="workflow-flow"
              colorMode="system"
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
                setSelectedNodeId(node.id)
                setDrawerMode('task')
              }}
              onNodeDragStop={persistNodePosition}
              onNodesChange={onNodesChange}
              proOptions={{ hideAttribution: true }}
            >
              <Background color="rgba(36, 54, 83, 0.16)" gap={24} size={1} />
              <MiniMap
                className="workflow-minimap"
                maskColor="rgba(245, 248, 252, 0.72)"
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
            onNodeAction={(action, nodeId, runId) => nodeActionMutation.mutate({ action, nodeId, runId })}
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
  const running = activeRun?.status === 'running'

  return (
    <section aria-label="Workflow execution controls" className="workflow-execution-toolbar">
      <div className="workflow-execution-toolbar__identity">
        <div className="workflow-title">{project?.name ?? 'hermes-workflow'}</div>
        <div className="workflow-subtitle">
          {selectedNode ? `当前节点：${selectedNode.title}` : project?.root ?? '创建项目后开始'}
        </div>
      </div>

      <div className="workflow-execution-toolbar__controls">
        <Select onValueChange={value => onModeChange(value as ExecutionMode)} value={executionMode}>
          <SelectTrigger className="h-8 w-28 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="single_step">单步</SelectItem>
            <SelectItem value="semi_auto">半自动</SelectItem>
            <SelectItem value="auto">自动</SelectItem>
          </SelectContent>
        </Select>

        <Button disabled={!project || busy || running} onClick={onRun} size="sm" type="button">
          <Codicon name="play" size="0.875rem" />
          运行
        </Button>
        <Button disabled={!running || busy} onClick={onPause} size="sm" type="button" variant="outline">
          <Codicon name="debug-pause" size="0.875rem" />
          暂停
        </Button>
        <Button disabled={!activeRun || busy || activeRun.status === 'stopped'} onClick={onStop} size="sm" type="button" variant="outline">
          <Codicon name="debug-stop" size="0.875rem" />
          停止
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
  selectedNode,
  workflow
}: {
  activeRun: ProjectBundle['latestRun']
  executionMode: ExecutionMode
  selectedNode: WorkflowNode | null
  workflow: Workflow | null
}) {
  const running = activeRun?.status === 'running'
  const waiting = activeRun?.status === 'waiting_user_confirm'
  const completed = workflow ? workflow.nodes.filter(node => node.status === 'completed').length : 0
  const total = workflow?.nodes.length ?? 0

  return (
    <div className="workflow-status-overlay" aria-label="Workflow execution status">
      <span className={cn('workflow-run-dot', running && 'is-running', waiting && 'is-waiting')} />
      <span>{activeRun ? runStatusLabel(activeRun.status) : '未运行'}</span>
      <span>{total ? `${completed}/${total}` : '0/0'}</span>
      <span>{activeRun ? MODE_LABEL[activeRun.mode] : MODE_LABEL[executionMode]}</span>
      <strong title={selectedNode?.title ?? undefined}>{selectedNode ? selectedNode.title : '未选择节点'}</strong>
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
  const items: Array<{ icon: string; label: string; mode: DrawerMode }> = [
    { icon: 'graph', label: '节点详情', mode: 'task' },
    { icon: 'files', label: '文件树', mode: 'files' },
    { icon: 'references', label: '资料', mode: 'references' },
    { icon: 'symbol-misc', label: '技能', mode: 'skills' },
    { icon: 'git-commit', label: '版本', mode: 'snapshots' }
  ]

  return (
    <div aria-label="Workflow tools" className="workflow-floating-toolbar" role="toolbar">
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
          <h2>项目文件</h2>
          <p>{root ?? '未选择项目'}</p>
        </div>
        <Button disabled={!root} onClick={onOpenProjectRoot} size="icon-sm" title="在文件资源管理器中打开" type="button" variant="ghost">
          <Codicon name="folder-opened" size="0.875rem" />
        </Button>
      </div>
      <div className="workflow-file-tree">
        {loading ? (
          <div className="workflow-muted">读取中...</div>
        ) : files.length ? (
          files.map(item => (
            <FileTreeItem
              expanded={expanded}
              item={item}
              key={item.path}
              onOpenFile={onOpenFile}
              onSelectFile={onSelectFile}
              onToggleExpanded={toggleExpanded}
              selectedFilePath={selectedFilePath}
            />
          ))
        ) : (
          <div className="workflow-muted">暂无可显示文件</div>
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
  selectedFilePath
}: {
  expanded: Set<string>
  item: WorkflowFileNode
  level?: number
  onOpenFile: (path: string) => void
  onSelectFile: (path: string) => void
  onToggleExpanded: (path: string) => void
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
            aria-label="打开文件"
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
  const [draft, setDraft] = useState<WorkflowNode | null>(node)
  const [skillsOpen, setSkillsOpen] = useState(false)
  const [referencesOpen, setReferencesOpen] = useState(false)
  const [changesOpen, setChangesOpen] = useState(true)

  useEffect(() => {
    setDraft(node)
    setSkillsOpen(false)
    setReferencesOpen(false)
    setChangesOpen(true)
  }, [node])

  if (!node) {
    return (
      <div className="workflow-drawer-empty">
        <Codicon name="graph" size="1.25rem" />
        <span>选择一个节点查看详情</span>
      </div>
    )
  }

  const editable = draft ?? node
  const status = STATUS_META[node.status]
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
          <h3>执行规划 Prompt</h3>
          <Button
            disabled={!draft}
            onClick={() => draft && onSaveNode({ ...draft, promptOverride: draft.promptOverride?.trim() || null })}
            size="xs"
            type="button"
          >
            <Codicon name="save" size="0.8125rem" />
            保存
          </Button>
        </div>
        <Textarea
          className="workflow-prompt-editor"
          onChange={event => updateDraft({ promptOverride: event.target.value })}
          placeholder={node.description || '为该节点补充执行规划 prompt'}
          value={editable.promptOverride ?? ''}
        />
        <p className="workflow-muted">{node.description}</p>
      </section>

      <section>
        <h3>上下文</h3>
        <div className="workflow-key-values">
          <span>类型</span>
          <strong>{node.type}</strong>
          <span>模型</span>
          <strong>{editable.modelOverride ?? editable.model ?? '继承全局配置'}</strong>
          <span>技能</span>
          <strong>{editable.skillMode === 'manual' ? `${editable.skills.length} 个手动技能` : '自动调用'}</strong>
          <span>重试</span>
          <strong>
            {node.retryCount}/{node.maxRetries}
          </strong>
        </div>
      </section>

      <section>
        <h3>执行模型</h3>
        <Select
          onValueChange={value => updateDraft({ modelOverride: value === '__inherit' ? null : value })}
          value={editable.modelOverride ?? '__inherit'}
        >
          <SelectTrigger className="h-8 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="__inherit">继承 workflow/global model</SelectItem>
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
                <strong>自动调用 Hermes Skills</strong>
                <small>由 Agent 根据节点目标和可用工具选择</small>
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
            {editable.skillMode === 'manual' && !availableSkills.length && <div className="workflow-muted">暂无可用 skills 列表。</div>}
          </div>
        )}
      </section>

      <section>
        <button className="workflow-collapsible-heading" onClick={() => setReferencesOpen(value => !value)} type="button">
          <h3>节点引用文件</h3>
          <span>{references.length}</span>
          <Codicon name={referencesOpen ? 'chevron-down' : 'chevron-right'} size="0.8125rem" />
        </button>
        {referencesOpen && (
          <div className="workflow-config-list">
            <div className="workflow-inline-actions">
              <Button
                onClick={() => {
                  void window.hermesDesktop
                    .selectPaths({ multiple: true, title: '选择当前节点引用文件' })
                    .then(paths => paths.forEach(addReference))
                }}
                size="xs"
                type="button"
                variant="outline"
              >
                <Codicon name="add" size="0.8125rem" />
                添加文件
              </Button>
              <Button disabled={!selectedFileForReference} onClick={() => selectedFileForReference && addReference(selectedFileForReference)} size="xs" type="button" variant="outline">
                <Codicon name="files" size="0.8125rem" />
                添加选中文件
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
              <div className="workflow-muted">暂无节点级引用文件。</div>
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
            <li>无显式审查规则</li>
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
            <div className="workflow-muted">暂无产物</div>
          )}
        </div>
      </section>

      <section>
        <button className="workflow-collapsible-heading" onClick={() => setChangesOpen(value => !value)} type="button">
          <h3>文件变更审查</h3>
          <span>{fileChanges.length}</span>
          <Codicon name={changesOpen ? 'chevron-down' : 'chevron-right'} size="0.8125rem" />
        </button>
        {changesOpen && (
          <div className="workflow-file-changes">
            {fileChanges.length ? (
              fileChanges.map(change => (
                <div className="workflow-file-change" key={`${change.status}-${change.path}`}>
                  <div className="workflow-file-change__header">
                    <span>
                      <strong>{change.path}</strong>
                      <small>{change.status}{change.isArtifact ? ' · artifact' : ''}{change.truncated ? ' · truncated' : ''}</small>
                    </span>
                    <Button onClick={() => onOpenFile(resolveProjectPath(root, change.path))} size="xs" type="button" variant="outline">
                      <Codicon name="go-to-file" size="0.8125rem" />
                      打开
                    </Button>
                  </div>
                  <pre>{change.diff || '无可显示 diff'}</pre>
                </div>
              ))
            ) : (
              <div className="workflow-muted">节点执行后会在这里显示新增、修改或删除的业务文件。</div>
            )}
          </div>
        )}
      </section>
      </div>

      <div className="workflow-node-actions">
        <Button disabled={!waiting || !runId} onClick={() => runId && onNodeAction('confirm', node.id, runId)} size="sm" type="button">
          <Codicon name="pass" size="0.875rem" />
          确认
        </Button>
        <Button disabled={!runId} onClick={() => runId && onNodeAction('retry', node.id, runId)} size="sm" type="button" variant="outline">
          <Codicon name="refresh" size="0.875rem" />
          重试
        </Button>
        <Button disabled={!runId} onClick={() => runId && onNodeAction('skip', node.id, runId)} size="sm" type="button" variant="outline">
          <Codicon name="debug-step-over" size="0.875rem" />
          跳过
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
  return (
    <div className="workflow-reference-drawer">
      <div className="workflow-drawer-header">
        <div>
          <h2>References</h2>
          <p>只有启用项会进入节点上下文</p>
        </div>
        <Button
          onClick={() => {
            void window.hermesDesktop
              .selectPaths({ multiple: true, title: '选择 reference 文件或文件夹' })
              .then(paths => paths.length && onAddReferences(paths))
          }}
          size="xs"
          type="button"
          variant="outline"
        >
          <Codicon name="add" size="0.8125rem" />
          添加
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
          <div className="workflow-muted">暂无 reference，节点会仅使用项目目标和上游输出。</div>
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
  return (
    <div>
      <div className="workflow-drawer-header">
        <div>
          <h2>Skills</h2>
          <p>项目级启用列表，节点可再绑定子集</p>
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
  return (
    <div>
      <div className="workflow-drawer-header">
        <div>
          <h2>版本快照</h2>
          <p>普通模式隐藏 Git 细节</p>
        </div>
        <Button onClick={onSnapshot} size="xs" type="button" variant="outline">
          <Codicon name="git-commit" size="0.8125rem" />
          快照
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
          <div className="workflow-muted">暂无快照</div>
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
          Stream Output
        </button>
        <div className="workflow-stream-panel__tools">
          <span className={cn('workflow-ws-dot', wsHealthy && 'is-live')} />
          <span>{wsHealthy ? 'WS live' : 'polling'}</span>
          <label>
            <Switch checked={filterSelectedNode} disabled={!selectedNode} onCheckedChange={onFilterSelectedNode} />
            当前节点
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
            <div className="workflow-muted">运行后会显示安全过程摘要、工具调用、阶段结果和 AI 回复。</div>
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
        {selectedNode ? `当前上下文：${selectedNode.title}` : '当前上下文：全局 Workflow'}
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
        placeholder="输入消息、/命令或 @file: 引用；输出会进入 Stream，而不是聊天气泡。"
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
            .selectPaths({ multiple: true, title: '选择 workflow 附件或 reference' })
            .then(async paths => {
              if (!paths.length) {
                return
              }

              await onAttach(paths)
              setAttachments(current => [...new Set([...current, ...paths])])
            })
        }}
        size="icon-sm"
        title="添加附件"
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
  const [name, setName] = useState('Hermes Workflow Project')
  const [goal, setGoal] = useState('')
  const [root, setRoot] = useState('')
  const [references, setReferences] = useState<string[]>([])
  const [intakeId, setIntakeId] = useState<string | null>(null)
  const [messages, setMessages] = useState<WorkflowIntakeMessage[]>([])
  const [reply, setReply] = useState('')
  const [summary, setSummary] = useState('')
  const [ready, setReady] = useState(false)

  const payload = useMemo<WorkflowIntakePayload>(
    () => ({ goal, name, references, root: root || undefined }),
    [goal, name, references, root]
  )

  const applyIntakeResponse = useCallback((response: { intakeId: string; messages: WorkflowIntakeMessage[]; ready: boolean; summary: string }) => {
    setIntakeId(response.intakeId)
    setMessages(response.messages)
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

  const confirmMutation = useMutation({
    mutationFn: () => confirmWorkflowIntake(intakeId!, { ...payload, summary }),
    onSuccess: data => void onComplete(data)
  })

  const busy = startMutation.isPending || messageMutation.isPending || confirmMutation.isPending
  const error = errorText(startMutation.error || messageMutation.error || confirmMutation.error)

  return (
    <main className="workflow-intake-page">
      <WorkflowPageTitlebar
        icon="graph"
        subtitle="Hermes 会先澄清规划细节，再生成可执行节点图。"
        title="新建 Workflow"
      />
      <section aria-label="Workflow intake" className="workflow-intake-layout">
        <div className="workflow-intake-config">
          <div className="workflow-intake-section-heading">
            <h2>项目配置</h2>
            <p>填写项目背景后开始澄清。</p>
          </div>

          <label>
            项目名称
            <Input disabled={busy && Boolean(intakeId)} onChange={event => setName(event.target.value)} value={name} />
          </label>

          <label>
            项目目录
            <div className="workflow-path-picker">
              <Input
                disabled={busy && Boolean(intakeId)}
                onChange={event => setRoot(event.target.value)}
                placeholder="留空则使用 Hermes 默认 workflows 目录"
                value={root}
              />
              <Button
                disabled={busy && Boolean(intakeId)}
                onClick={() => {
                  void window.hermesDesktop
                    .selectPaths({ directories: true, title: '选择 Workflow 项目目录' })
                    .then(paths => paths[0] && setRoot(paths[0]))
                }}
                type="button"
                variant="outline"
              >
                选择
              </Button>
            </div>
          </label>

          <label>
            任务背景
            <Textarea
              className="min-h-32"
              disabled={busy && Boolean(intakeId)}
              onChange={event => setGoal(event.target.value)}
              placeholder="描述目标、输入资料、验收标准和约束。"
              value={goal}
            />
          </label>

          <div>
            <div className="workflow-dialog-row">
              <span>References</span>
              <Button
                disabled={busy && Boolean(intakeId)}
                onClick={() => {
                  void window.hermesDesktop
                    .selectPaths({ multiple: true, title: '选择 reference 文件或文件夹' })
                    .then(paths => setReferences(current => [...new Set([...current, ...paths])]))
                }}
                size="xs"
                type="button"
                variant="outline"
              >
                <Codicon name="add" size="0.8125rem" />
                添加
              </Button>
            </div>
            <div className="workflow-reference-preview">
              {references.length ? references.map(path => <span key={path}>{path}</span>) : <span>暂无 reference</span>}
            </div>
          </div>

          <Button disabled={busy || !name.trim() || Boolean(intakeId)} onClick={() => startMutation.mutate()} type="button">
            <Codicon name={startMutation.isPending ? 'loading' : 'comment-discussion'} size="0.875rem" spinning={startMutation.isPending} />
            开始澄清
          </Button>
        </div>

        <div className="workflow-intake-chat">
          <div className="workflow-intake-transcript">
            {messages.length ? (
              messages.map((message, index) => (
                <div className={cn('workflow-intake-message', message.role === 'user' && 'is-user')} key={`${message.timestamp}-${index}`}>
                  <span>{message.role === 'user' ? '你' : 'Hermes'}</span>
                  <p>{message.content}</p>
                </div>
              ))
            ) : (
              <div className="workflow-muted">填写项目背景后开始澄清。</div>
            )}
          </div>

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
              placeholder="补充规划细节..."
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
              <h2>规划摘要</h2>
              <p>{ready ? '可以生成 Workflow' : '等待澄清完成'}</p>
            </div>
          </div>
          <pre>{summary || '摘要会随澄清对话更新。'}</pre>
          {error && <div className="workflow-error">{error}</div>}
          {confirmMutation.isPending && <div className="workflow-status">正在创建项目、初始化 Git、生成 Workflow...</div>}
          <Button disabled={!intakeId || !ready || confirmMutation.isPending} onClick={() => confirmMutation.mutate()} type="button">
            <Codicon name={confirmMutation.isPending ? 'loading' : 'sparkle'} size="0.875rem" spinning={confirmMutation.isPending} />
            确认并生成
          </Button>
        </aside>
      </section>
    </main>
  )
}

function WorkbenchLoading() {
  return (
    <div className="workflow-empty">
      <Codicon name="loading" size="1.5rem" spinning />
      <span>正在加载 Workflow 工作台...</span>
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
  if (hasProject) {
    return (
      <div className="workflow-empty">
        <Codicon name={busy ? 'loading' : 'graph'} size="1.5rem" spinning={busy} />
        <span>{busy ? '正在通过 Hermes Agent 生成 Workflow...' : '当前项目还没有可显示的 Workflow 节点'}</span>
        <div className="workflow-empty__actions">
          <Button disabled={busy} onClick={onGenerate} type="button">
            <Codicon name={busy ? 'loading' : 'sparkle'} size="0.875rem" spinning={busy} />
            生成 Workflow
          </Button>
          <Button disabled={busy} onClick={onAddReference} type="button" variant="outline">
            <Codicon name="references" size="0.875rem" />
            添加资料
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="workflow-empty">
      <Codicon name="graph" size="1.5rem" />
      <span>创建项目后开始编排 AI Agent Workflow</span>
      <Button onClick={onCreate} type="button">
        新建 Workflow 项目
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
        ? { stroke: '#cf806d', strokeDasharray: '6 5', strokeWidth: 1.8 }
        : { stroke: 'rgba(25, 39, 64, 0.42)', strokeWidth: 1.6 },
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
  const tone = STATUS_META[status].tone

  const colors: Record<string, string> = {
    danger: '#cf2d56',
    info: '#4c7f8c',
    neutral: '#9aa3af',
    ready: '#0053fd',
    running: '#7c5cff',
    success: '#1f8a65',
    warning: '#c08532'
  }

  return colors[tone] ?? colors.neutral
}

function runStatusLabel(status: string): string {
  const labels: Record<string, string> = {
    completed: '已完成',
    failed: '失败',
    idle: '空闲',
    paused: '已暂停',
    running: '运行中',
    stopped: '已停止',
    waiting_user_confirm: '等待确认'
  }

  return labels[status] ?? status
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
