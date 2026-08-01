import type { TaskPayload, TimerEntry } from "../core/state/types.js";

export interface LocalTimerTaskOptions {
  entry: TimerEntry;
  defaultInstanceId: string;
  now?: number;
}

export function createLocalTimerTask(options: LocalTimerTaskOptions): TaskPayload {
  const member = options.entry.member;
  const now = options.now ?? Date.now();
  let taskType: TaskPayload["type"];
  let instanceId: string;
  let sessionId: string;

  const firstColon = member.indexOf(":");
  const prefix = firstColon > 0 ? member.slice(0, firstColon) : member;

  if (prefix === "offload-l1" || prefix === "offload-l15" || prefix === "offload-l2") {
    taskType = prefix;
    const rest = member.slice(firstColon + 1);
    const instanceEnd = rest.indexOf(":");
    if (instanceEnd > 0) {
      instanceId = rest.slice(0, instanceEnd);
      sessionId = rest.slice(instanceEnd + 1);
    } else {
      instanceId = options.defaultInstanceId;
      sessionId = rest;
    }
    if (prefix === "offload-l2" && sessionId.endsWith(".mmd")) {
      const lastColon = sessionId.lastIndexOf(":");
      if (lastColon > 0) {
        sessionId = sessionId.slice(0, lastColon);
      }
    }
  } else {
    const lastColon = member.lastIndexOf(":");
    const suffix = lastColon >= 0 ? member.slice(lastColon + 1) : "";
    sessionId = lastColon >= 0 ? member.slice(0, lastColon) : member;
    taskType = suffix === "L2_schedule" ? "L2" : suffix === "L1_idle" ? "L1" : "L3";
    instanceId = options.defaultInstanceId;
  }

  let targetMmdFile: string | undefined;
  if (taskType === "offload-l2") {
    const mmdMatch = member.match(/(\d+-[^:]+\.mmd)$/);
    if (mmdMatch) targetMmdFile = mmdMatch[1];
  }

  return {
    id: `${taskType}-${sessionId}-${now}`,
    type: taskType,
    instanceId,
    sessionId,
    priority: 0,
    createdAt: now,
    data: { triggeredBy: "timer_scanner", timerMember: member, instanceId, targetMmdFile },
  };
}
