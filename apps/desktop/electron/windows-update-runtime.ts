export type WindowsProcessInfo = {
  pid: number | null
  parentPid: number | null
  creationTimeMs: number | null
  executablePath: string
  commandLine: string
}

function parsedCreationTimeMs(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value) && value > 0) {
    return value
  }

  const text = String(value ?? '').trim()

  if (!text) {
    return null
  }

  const numeric = Number(text)

  if (Number.isFinite(numeric) && numeric > 0) {
    return numeric
  }

  const parsed = Date.parse(text)

  return Number.isFinite(parsed) && parsed > 0 ? parsed : null
}

function normalizedProcessInfo(value: any): WindowsProcessInfo {
  const rawPid = value?.pid ?? value?.ProcessId
  const numericPid = Number(rawPid)
  const rawParent = value?.parentPid ?? value?.ParentProcessId
  const numericParent = Number(rawParent)

  return {
    pid: Number.isInteger(numericPid) && numericPid > 0 ? numericPid : null,
    parentPid: Number.isInteger(numericParent) && numericParent > 0 ? numericParent : null,
    creationTimeMs: parsedCreationTimeMs(value?.creationTimeMs ?? value?.CreationTimeMs ?? value?.CreationDate),
    executablePath: String(value?.executablePath ?? value?.ExecutablePath ?? ''),
    commandLine: String(value?.commandLine ?? value?.CommandLine ?? '')
  }
}

export function parseWindowsProcessList(raw: unknown): WindowsProcessInfo[] {
  const text = String(raw ?? '')
    .replace(/^\uFEFF/, '')
    .trim()

  if (!text) {
    return []
  }

  try {
    const parsed = JSON.parse(text)
    const rows = Array.isArray(parsed) ? parsed : [parsed]
    return rows.map(normalizedProcessInfo)
  } catch {
    return []
  }
}

/**
 * Enumerate a process tree from a live inventory instead of `taskkill /T`.
 *
 * taskkill /T builds its tree from bare ParentProcessId links with no
 * creation-time validation. The desktop's own parent (the task-launcher
 * PowerShell) dies immediately after boot, so its PID gets recycled — and
 * when a process in the tree being killed recycles it, taskkill considers
 * the DESKTOP a descendant and kills it mid-handoff (observed: the whole
 * Electron tree died within 1s of the backend tree-kill). This walk has the
 * parent/child edge is accepted only when both creation times exist and the
 * parent predates the child. That rejects stale ParentProcessId links after
 * PID reuse. Callers MUST additionally pass their own pid (and anything else
 * that must survive) in excludePids: excluded pids are neither killed nor
 * expanded.
 */
export function collectProcessTreePids(
  processes: Array<WindowsProcessInfo | Record<string, unknown>>,
  rootPid: number,
  {
    excludePids = [],
    expectedRootCreationTimeMs
  }: { excludePids?: number[]; expectedRootCreationTimeMs: number }
): number[] {
  if (
    !Number.isInteger(rootPid) ||
    rootPid <= 0 ||
    !Number.isFinite(expectedRootCreationTimeMs) ||
    expectedRootCreationTimeMs <= 0
  ) {
    return []
  }

  const excluded = new Set(excludePids.filter(pid => Number.isInteger(pid) && pid > 0))

  if (excluded.has(rootPid)) {
    return []
  }

  const infoByPid = new Map<number, WindowsProcessInfo>()

  for (const value of processes || []) {
    const info = normalizedProcessInfo(value)

    if (info.pid === null) {
      continue
    }

    infoByPid.set(info.pid, info)
  }

  // The root must still exist in this exact inventory. If the tracked child
  // already exited, do not kill a newly recycled PID with the same number.
  if (infoByPid.get(rootPid)?.creationTimeMs !== expectedRootCreationTimeMs) {
    return []
  }

  const childrenByParent = new Map<number, number[]>()

  for (const info of infoByPid.values()) {
    if (info.parentPid === null || info.creationTimeMs === null) {
      continue
    }

    const parent = infoByPid.get(info.parentPid)

    if (parent?.creationTimeMs === null || parent?.creationTimeMs === undefined) {
      continue
    }

    // A real parent was already alive when its child started. If the current
    // process carrying parentPid was created later, the PPID is a stale link
    // through a recycled PID and must not be traversed.
    if (parent.creationTimeMs > info.creationTimeMs) {
      continue
    }

    const siblings = childrenByParent.get(info.parentPid) || []

    siblings.push(info.pid as number)
    childrenByParent.set(info.parentPid, siblings)
  }

  const tree: number[] = []
  const seen = new Set<number>([rootPid])
  const queue = [rootPid]

  while (queue.length) {
    const pid = queue.shift() as number

    if (excluded.has(pid)) {
      continue
    }

    tree.push(pid)

    for (const child of childrenByParent.get(pid) || []) {
      if (!seen.has(child)) {
        seen.add(child)
        queue.push(child)
      }
    }
  }

  return tree
}
