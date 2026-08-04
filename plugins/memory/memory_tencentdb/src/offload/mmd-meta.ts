/**
 * MMD metadata parsing utility.
 * Extracted from prompts/l15.ts — pure data parsing, not a prompt.
 */

export interface MmdMeta {
  filename: string;
  path: string;
  taskGoal: string;
  createdTime: string | null;
  updatedTime: string | null;
  doneCount: number;
  doingCount: number;
  todoCount: number;
  nodeSummaries: Array<{ nodeId: string; status: string; summary: string }>;
}

export function parseMmdMeta(
  filename: string,
  mmdPath: string,
  content: string,
): MmdMeta {
  const meta: MmdMeta = {
    filename,
    path: mmdPath,
    taskGoal: "",
    createdTime: null,
    updatedTime: null,
    doneCount: 0,
    doingCount: 0,
    todoCount: 0,
    nodeSummaries: [],
  };
  const metaMatch = content.match(/^%%\{\s*(.*?)\s*\}%%/);
  if (metaMatch) {
    try {
      const p = JSON.parse(`{${metaMatch[1]}}`) as Record<string, unknown>;
      meta.taskGoal = (p.taskGoal as string) || "";
      meta.createdTime = (p.createdTime as string) || null;
      meta.updatedTime = (p.updatedTime as string) || null;
    } catch {
      /* ignore */
    }
  }
  meta.doneCount = (content.match(/status:\s*done/gi) || []).length;
  meta.doingCount = (content.match(/status:\s*doing/gi) || []).length;
  meta.todoCount = (content.match(/status:\s*todo/gi) || []).length;
  const nodeRe = /(\d{3}-N\d+)\["([^"]*?)"\]/g;
  let m: RegExpExecArray | null;
  while ((m = nodeRe.exec(content)) !== null) {
    const nodeText = m[2];
    const summaryMatch = nodeText.match(/summary:\s*(.+?)(?:<br\/>|$)/i);
    const statusMatch = nodeText.match(/status:\s*(\w+)/i);
    if (summaryMatch) {
      meta.nodeSummaries.push({
        nodeId: m[1],
        status: statusMatch ? statusMatch[1] : "unknown",
        summary: summaryMatch[1].trim().slice(0, 100),
      });
    }
  }
  if (meta.nodeSummaries.length > 2) {
    meta.nodeSummaries = meta.nodeSummaries.slice(-2);
  }
  return meta;
}
