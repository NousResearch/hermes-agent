// Sanitized local export of a capture bundle (U4, R2/R3, KTD3).
//
// The controller (diagnostics-capture.ts) hands over three streams stamped on
// three different monotonic clocks. This module is the only place that turns
// them into files, which makes it the only place the sanitization contract has
// to hold:
//
//   * Nothing outbound. The bundle is a directory under Electron's userData;
//     the user attaches it by hand or does not.
//   * The MANIFEST IS INSIDE THE CONTRACT. The process tree records pid, ppid
//     and the executable BASENAME — never argv, never a full path. Command
//     lines on this machine routinely carry API tokens, worktree paths and
//     session ids, so `sanitizeProcessTree` drops every field it is not
//     explicitly told to keep rather than filtering known-bad ones.
//   * Stream events are already sanitized at record time in each process
//     (counts/sizes/durations/ids only); the exporter never enriches them.
//
// Layout, one directory per capture:
//
//   diagnostics/<captureId>/
//     manifest.json        capture id, anchors, app version, process tree, stream index
//     classification.json  the R3 summary (see diagnostics-classify.ts)
//     renderer-<windowId>.jsonl
//     main.jsonl
//     gateway.jsonl        absent when the gateway stream is absent
//
// JSONL rather than one JSON blob because the streams are the part a reviewer
// greps, sorts and pipes; the manifest is the part they read.
//
// Alignment (KTD3): every stream file carries an added `wall_clock_ms` beside
// the process's own monotonic `t`, computed as
// `wallClockAnchorMs + (t - <that stream's> monotonicAnchorMs)`. The original
// monotonic value is preserved untouched so a reviewer can always fall back to
// the raw stream if an anchor turns out to be wrong.
//
// fs/path/process are injected so this unit-tests under the `electron` vitest
// project without Electron or a real temp directory.

import nodeFs from 'node:fs/promises'
import nodePath from 'node:path'

import type { DiagnosticsCaptureBundle } from './diagnostics-capture'
import { classifyCapture, type HitchClassification } from './diagnostics-classify'

/** Bumped when the on-disk shape changes in a way a reader must notice. */
const BUNDLE_SCHEMA_VERSION = 1

/** Directory under userData that holds one subdirectory per capture. */
const BUNDLE_DIR_NAME = 'diagnostics'

/** Raw process description, as a platform probe hands it over. Only the three
 * named fields are ever read; anything else on the object is dropped. */
export interface RawProcessEntry {
  pid?: unknown
  ppid?: unknown
  /** Executable path OR name. Reduced to its basename either way. */
  name?: unknown
}

export interface SanitizedProcessEntry {
  pid: number
  ppid: number
  /** Executable basename ONLY, e.g. `hermes.exe`. Never a path, never argv. */
  name: string
}

interface FsLike {
  mkdir(dir: string, options: { recursive: boolean }): Promise<unknown>
  writeFile(file: string, data: string, encoding: string): Promise<void>
}

export interface ExportDiagnosticsOptions {
  bundle: DiagnosticsCaptureBundle
  /** Electron's `app.getPath('userData')`. */
  userDataPath: string
  appVersion?: string
  /** `app.getAppMetrics()`-shaped or probe-shaped process list. Optional: an
   * export with no process tree is still a valid bundle. */
  processes?: RawProcessEntry[]
  /** Wall clock for the manifest's `exported_at_ms`. */
  wallClock?: () => number
  platform?: string
  fs?: FsLike
  join?: (...parts: string[]) => string
}

export interface StreamIndexEntry {
  name: string
  kind: 'gateway' | 'main' | 'renderer'
  /** Relative to the bundle directory; null when the stream is absent. */
  file: null | string
  events: number
  dropped: number
  /** That stream's own monotonic clock at arm time (KTD3). */
  monotonic_anchor_ms: number
  window_id?: number
  /** Set only when `file` is null. */
  absent?: string
}

export interface DiagnosticsManifest {
  schema_version: number
  capture_id: string
  exported_at_ms: number
  app_version: string
  platform: string
  anchors: {
    wall_clock_anchor_ms: number
    main_monotonic_anchor_ms: number
  }
  process_tree: SanitizedProcessEntry[]
  streams: StreamIndexEntry[]
}

export interface ExportResult {
  /** Absolute path of the bundle directory — what the UI shows the user. */
  directory: string
  manifestPath: string
  manifest: DiagnosticsManifest
  classification: HitchClassification
  /** Absolute paths of every file written, manifest first. */
  files: string[]
}

/**
 * Reduce an executable path or name to its basename. Handles both separators
 * regardless of host platform, because a bundle exported on Windows can be read
 * on a Mac and a probe can report either form.
 */
export function executableBasename(value: unknown): string {
  const text = typeof value === 'string' ? value.trim() : ''

  if (!text) {
    return 'unknown'
  }

  const parts = text.split(/[\\/]/).filter(Boolean)
  const basename = parts.length ? parts[parts.length - 1] : ''

  // A basename is one token. Anything with whitespace in it is a command line
  // that was handed over as if it were a path, and its tail is argv — drop it.
  return (basename.split(/\s/)[0] || 'unknown').slice(0, 128)
}

/**
 * Whitelist a raw process list down to {pid, ppid, name-basename}. This is a
 * REBUILD, not a filter: the returned objects share no fields with the input,
 * so a probe that starts reporting `cmdline` cannot leak it through here.
 */
export function sanitizeProcessTree(entries: RawProcessEntry[] | undefined): SanitizedProcessEntry[] {
  if (!Array.isArray(entries)) {
    return []
  }

  return entries.map(entry => ({
    pid: Math.trunc(Number(entry?.pid)) || 0,
    ppid: Math.trunc(Number(entry?.ppid)) || 0,
    name: executableBasename(entry?.name)
  }))
}

/**
 * Shift one stream's monotonic timestamps onto the shared wall clock (KTD3).
 * `t` survives untouched; `wall_clock_ms` is added beside it.
 */
export function alignEvent(event: unknown, wallClockAnchorMs: number, monotonicAnchorMs: number): unknown {
  if (!event || typeof event !== 'object' || Array.isArray(event)) {
    return event
  }

  const record = event as Record<string, unknown>

  // Renderer/main stamp `t` in MILLISECONDS (`performance.now()`); the gateway
  // ring stamps `t_monotonic` in SECONDS (`time.monotonic()`) while reporting
  // its anchor already scaled to ms. Normalize the event, never the anchor.
  const monotonicMs =
    typeof record.t === 'number'
      ? record.t
      : typeof record.t_monotonic === 'number'
        ? record.t_monotonic * 1000
        : null

  if (monotonicMs === null || !Number.isFinite(wallClockAnchorMs)) {
    return record
  }

  return { ...record, wall_clock_ms: Math.round(wallClockAnchorMs + (monotonicMs - monotonicAnchorMs)) }
}

function toJsonl(events: unknown[], wallClockAnchorMs: number, monotonicAnchorMs: number): string {
  return events.map(event => JSON.stringify(alignEvent(event, wallClockAnchorMs, monotonicAnchorMs))).join('\n')
}

/** The gateway already scales its anchor to ms on the wire
 * (`monotonic_anchor_ms = time.monotonic() * 1000`), so no conversion here —
 * only the per-event `t_monotonic` seconds need scaling, in `alignEvent`. */
function gatewayAnchorMs(stream: { monotonicAnchorMs?: unknown }): number {
  const raw = Number(stream?.monotonicAnchorMs)

  return Number.isFinite(raw) ? raw : 0
}

/**
 * Write the bundle. Creates `<userData>/diagnostics/<captureId>/`, one JSONL
 * per stream, `classification.json` and `manifest.json`, and returns the paths.
 *
 * An absent gateway stream is NOT an error (R2 error path): no `gateway.jsonl`
 * is written and the manifest's stream entry carries `absent: <reason>`.
 */
export async function exportDiagnosticsBundle(options: ExportDiagnosticsOptions): Promise<ExportResult> {
  const fs = options.fs ?? (nodeFs as unknown as FsLike)
  const join = options.join ?? nodePath.join
  const wallClock = options.wallClock ?? (() => Date.now())
  const bundle = options.bundle
  const wallClockAnchorMs = Number(bundle.anchors?.wallClockAnchorMs) || 0
  const mainMonotonicAnchorMs = Number(bundle.anchors?.mainMonotonicAnchorMs) || 0

  const directory = join(options.userDataPath, BUNDLE_DIR_NAME, bundle.captureId)

  await fs.mkdir(directory, { recursive: true })

  const files: string[] = []
  const streams: StreamIndexEntry[] = []

  async function writeStream(fileName: string, body: string) {
    const target = join(directory, fileName)

    await fs.writeFile(target, body ? `${body}\n` : '', 'utf8')
    files.push(target)
  }

  for (const stream of bundle.renderer ?? []) {
    const fileName = `renderer-${stream.windowId}.jsonl`
    const events = Array.isArray(stream.events) ? stream.events : []

    await writeStream(fileName, toJsonl(events, wallClockAnchorMs, Number(stream.monotonicAnchorMs) || 0))
    streams.push({
      name: `renderer-${stream.windowId}`,
      kind: 'renderer',
      file: fileName,
      events: events.length,
      dropped: Number(stream.droppedEvents) || 0,
      monotonic_anchor_ms: Number(stream.monotonicAnchorMs) || 0,
      window_id: stream.windowId
    })
  }

  const mainEvents = Array.isArray(bundle.main) ? bundle.main : []

  await writeStream('main.jsonl', toJsonl(mainEvents, wallClockAnchorMs, mainMonotonicAnchorMs))
  streams.push({
    name: 'main',
    kind: 'main',
    file: 'main.jsonl',
    events: mainEvents.length,
    dropped: Number(bundle.mainDropped) || 0,
    monotonic_anchor_ms: mainMonotonicAnchorMs
  })

  const gateway = bundle.gateway as { absent?: string; events?: unknown[]; dropped?: unknown; monotonicAnchorMs?: unknown }

  if (gateway && !gateway.absent && Array.isArray(gateway.events)) {
    const anchorMs = gatewayAnchorMs(gateway)

    await writeStream('gateway.jsonl', toJsonl(gateway.events, wallClockAnchorMs, anchorMs))
    streams.push({
      name: 'gateway',
      kind: 'gateway',
      file: 'gateway.jsonl',
      events: gateway.events.length,
      dropped: Number(gateway.dropped) || 0,
      monotonic_anchor_ms: anchorMs
    })
  } else {
    streams.push({
      name: 'gateway',
      kind: 'gateway',
      file: null,
      events: 0,
      dropped: 0,
      monotonic_anchor_ms: 0,
      absent: typeof gateway?.absent === 'string' ? gateway.absent : 'unavailable'
    })
  }

  const classification = classifyCapture(bundle)
  const classificationPath = join(directory, 'classification.json')

  await fs.writeFile(classificationPath, `${JSON.stringify(classification, null, 2)}\n`, 'utf8')
  files.push(classificationPath)

  const manifest: DiagnosticsManifest = {
    schema_version: BUNDLE_SCHEMA_VERSION,
    capture_id: bundle.captureId,
    exported_at_ms: wallClock(),
    app_version: options.appVersion ?? 'unknown',
    platform: options.platform ?? process.platform,
    anchors: {
      wall_clock_anchor_ms: wallClockAnchorMs,
      main_monotonic_anchor_ms: mainMonotonicAnchorMs
    },
    process_tree: sanitizeProcessTree(options.processes),
    streams
  }

  const manifestPath = join(directory, 'manifest.json')

  await fs.writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, 'utf8')

  return { directory, manifestPath, manifest, classification, files: [manifestPath, ...files] }
}

/**
 * Build the manifest's process tree from what Electron can see of itself:
 * this process, plus every process `app.getAppMetrics()` reports. Electron
 * spawns all of them (renderers, GPU, utility) as children of main, so main's
 * pid is their ppid.
 *
 * `getAppMetrics()` reports a `name` for utility processes only ("Audio
 * Service", "Network Service"), which is a role rather than an executable; the
 * process `type` is the honest label for the rest. Neither is a path and
 * neither is argv, which is the property that matters here.
 */
export function electronProcessTree(input: {
  mainPid: number
  mainPpid: number
  execPath: string
  metrics?: { pid?: unknown; type?: unknown; name?: unknown }[]
}): RawProcessEntry[] {
  const tree: RawProcessEntry[] = [
    { pid: input.mainPid, ppid: input.mainPpid, name: input.execPath }
  ]

  for (const metric of input.metrics ?? []) {
    const pid = Math.trunc(Number(metric?.pid)) || 0

    if (!pid || pid === input.mainPid) {
      continue
    }

    tree.push({ pid, ppid: input.mainPid, name: metric?.name || metric?.type })
  }

  return tree
}

export { BUNDLE_DIR_NAME, BUNDLE_SCHEMA_VERSION }
