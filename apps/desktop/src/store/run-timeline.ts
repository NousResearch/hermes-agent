// Pure, UI-free model for the Observatory delegation timeline (Variant D, v1).
//
// buildRunTimeline() turns the live subagent tree + the parent session's
// per-tool blocks into ordered swimlanes on a single BACKEND-epoch axis:
//   - parent lane  -> per-tool blocks (kind: 'blocks')
//   - child lanes  -> one spawn->complete bar each (kind: 'bar')
//
// No React, no nanostores reads, no Date.now() on the axis — every startMs /
// endMs is a backend epoch (ms). This is the P2 deliverable; the canvas (P3)
// only renders what this returns.

import { buildSubagentTree, type SubagentNode, type SubagentProgress } from './subagents'

export type ToolFamily = 'terminal' | 'file' | 'browser' | 'think' | 'delegation' | 'web' | 'other'

export interface ToolBlock {
  toolCallId: string
  name: string
  family: ToolFamily
  /** Backend epoch ms. */
  startMs: number
  /** null while still running. */
  durationMs: number | null
  isOutlier: boolean
}

export type LaneKind = 'blocks' | 'bar'

export interface LaneBar {
  toolCount: number
  isOutlier: boolean
}

export interface TimelineLane {
  id: string
  parentLaneId: string | null
  depth: number
  label: string
  kind: LaneKind
  /** Backend epoch ms. */
  startMs: number
  /** null while still running. */
  endMs: number | null
  running: boolean
  blocks: ToolBlock[]
  bar: LaneBar | null
}

export interface RunTimeline {
  lanes: TimelineLane[]
  /** Earliest lane/block start across the whole run (backend epoch ms). */
  startMs: number
  /** Latest known end, or now-ish for a running run (backend epoch ms). */
  endMs: number
}

export interface BuildRunTimelineOpts {
  /** Tool calls over this many ms are flagged as outliers. Default 20s. */
  outlierMs?: number
  /**
   * Upper bound for the run's live edge when work is still running (backend
   * epoch ms). The caller passes a backend-epoch "now" so the pure function
   * never touches the renderer clock. Falls back to the latest known end.
   */
  nowMs?: number
}

const DEFAULT_OUTLIER_MS = 20_000

// Static tool-name -> family map. No backend dependency; extend as tools land.
const FAMILY_BY_TOOL: Record<string, ToolFamily> = {
  terminal: 'terminal',
  process: 'terminal',
  read_file: 'file',
  write_file: 'file',
  patch: 'file',
  search_files: 'file',
  edit_file: 'file',
  browser_navigate: 'browser',
  browser_click: 'browser',
  browser_snapshot: 'browser',
  browser_vision: 'browser',
  browser_type: 'browser',
  browser_scroll: 'browser',
  delegate_task: 'delegation',
  workflow: 'delegation',
  web_search: 'web',
  web_extract: 'web'
}

export function toolFamily(name: string): ToolFamily {
  const key = (name || '').toLowerCase()

  if (FAMILY_BY_TOOL[key]) {
    return FAMILY_BY_TOOL[key]
  }

  if (key.startsWith('browser_')) {
    return 'browser'
  }

  if (key.startsWith('web_')) {
    return 'web'
  }

  return 'other'
}

/**
 * A minimal projection of a message tool-call part — the fields the timeline
 * needs. The caller extracts these from ChatMessagePart before calling
 * buildRunTimeline, keeping this module free of the assistant-ui types.
 */
export interface ParentToolInput {
  toolCallId: string
  name: string
  /** Backend epoch SECONDS (as emitted on tool.start started_at). */
  startedAt?: number
  /** Seconds (as emitted on tool.complete duration_s); undefined = running. */
  durationS?: number
}

function makeParentBlocks(
  inputs: readonly ParentToolInput[],
  fallbackStartMs: number,
  outlierMs: number
): ToolBlock[] {
  // Sequential fallback: when a tool has no backend started_at (pre-P1
  // sessions), lay blocks end-to-end by arrival order from fallbackStartMs so
  // the lane still renders, just without true timing.
  let cursorMs = fallbackStartMs

  return inputs.map(input => {
    const durationMs = input.durationS !== undefined ? Math.max(0, input.durationS * 1000) : null
    const startMs = input.startedAt !== undefined ? input.startedAt * 1000 : cursorMs

    cursorMs = startMs + (durationMs ?? 0)

    return {
      toolCallId: input.toolCallId,
      name: input.name,
      family: toolFamily(input.name),
      startMs,
      durationMs,
      isOutlier: durationMs !== null && durationMs > outlierMs
    }
  })
}

function laneFromNode(
  node: SubagentNode,
  depth: number,
  parentLaneId: string,
  parentStartMs: number,
  outlierMs: number
): TimelineLane[] {
  const running = node.status === 'queued' || node.status === 'running'
  const durationMs = node.durationSeconds !== undefined ? Math.max(0, node.durationSeconds * 1000) : null
  // subagent.start carries no backend epoch today (v1): anchor the child lane
  // to the parent's backend-epoch start until a real spawn epoch exists (P6).
  const startMs = parentStartMs
  const endMs = durationMs !== null ? startMs + durationMs : null
  const toolCount = node.toolCount ?? 0

  const lane: TimelineLane = {
    id: node.id,
    parentLaneId,
    depth,
    label: node.goal || node.id,
    kind: 'bar',
    startMs,
    endMs,
    running,
    blocks: [],
    bar: {
      toolCount,
      isOutlier: durationMs !== null && durationMs > outlierMs
    }
  }

  const nested = node.children.flatMap(child => laneFromNode(child, depth + 1, node.id, startMs, outlierMs))

  return [lane, ...nested]
}

export function buildRunTimeline(
  subagents: readonly SubagentProgress[],
  parentTools: readonly ParentToolInput[],
  sessionCreatedAtMs: number,
  parentLaneId: string,
  parentLabel: string,
  opts: BuildRunTimelineOpts = {}
): RunTimeline {
  const outlierMs = opts.outlierMs ?? DEFAULT_OUTLIER_MS

  const blocks = makeParentBlocks(parentTools, sessionCreatedAtMs, outlierMs)

  const blockEnd = blocks.reduce((max, b) => Math.max(max, b.startMs + (b.durationMs ?? 0)), sessionCreatedAtMs)

  const parentLane: TimelineLane = {
    id: parentLaneId,
    parentLaneId: null,
    depth: 0,
    label: parentLabel,
    kind: 'blocks',
    startMs: sessionCreatedAtMs,
    endMs: null, // parent session is live
    running: true,
    blocks,
    bar: null
  }

  const tree = buildSubagentTree(subagents)
  const childLanes = tree.flatMap(node => laneFromNode(node, 1, parentLaneId, sessionCreatedAtMs, outlierMs))

  const lanes = [parentLane, ...childLanes]

  const startMs = lanes.reduce((min, l) => Math.min(min, l.startMs), sessionCreatedAtMs)

  const knownEnd = lanes.reduce(
    (max, l) => Math.max(max, l.endMs ?? l.startMs),
    blockEnd
  )

  const endMs = Math.max(knownEnd, opts.nowMs ?? knownEnd)

  return { lanes, startMs, endMs }
}

/**
 * A minimal projection of a persisted delegation child (the SessionChild the
 * /api/sessions/{id}/children endpoint returns), kept free of the hermes.ts
 * type so this module stays pure. Timestamps are backend epoch SECONDS;
 * endedAtS null = still running.
 */
export interface HistoricalChildInput {
  id: string
  label: string
  /** Backend epoch SECONDS. */
  startedAtS: number
  /** Backend epoch SECONDS; null = running. */
  endedAtS: number | null
  toolCount: number
}

/**
 * Pure: build child swimlane bars from PERSISTED delegation children (the
 * historical path — used when the live $subagentsBySession store is empty
 * because the run finished or the app reloaded). Mirrors laneFromNode's bar
 * shape so the canvas renders them identically to live child lanes.
 *
 * `parentEndMs` clamps a still-running child (endedAtS null) whose parent has
 * already finished, so a crashed/interrupted child bar doesn't stretch to
 * wall-clock now. Pass null for a live parent (bar falls back to nowMs at
 * render time via endMs=null).
 */
export function buildHistoricalChildLanes(
  children: readonly HistoricalChildInput[],
  parentLaneId: string,
  parentEndMs: number | null,
  opts: BuildRunTimelineOpts = {}
): TimelineLane[] {
  const outlierMs = opts.outlierMs ?? DEFAULT_OUTLIER_MS

  return children.map(child => {
    const startMs = child.startedAtS * 1000
    const rawEndMs = child.endedAtS !== null ? child.endedAtS * 1000 : null
    // A running child under a finished parent clamps to the parent's end.
    const endMs = rawEndMs !== null ? rawEndMs : parentEndMs
    const running = child.endedAtS === null
    const durationMs = endMs !== null ? Math.max(0, endMs - startMs) : null

    return {
      id: child.id,
      parentLaneId,
      depth: 1,
      label: child.label,
      kind: 'bar' as const,
      startMs,
      endMs,
      running,
      blocks: [],
      bar: {
        toolCount: child.toolCount,
        isOutlier: durationMs !== null && durationMs > outlierMs
      }
    }
  })
}
