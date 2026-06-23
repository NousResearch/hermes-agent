import { useEffect, useMemo, useState, type CSSProperties } from "react";
import {
  createTask,
  listAdapters,
  listHandoffs,
  listLogs,
  listTasks,
  patchTask,
  requeueTask,
  retryTask,
  runTask,
  type AdapterRecord,
  type HandoffRecord,
  type LogRecord,
  type TaskAgent,
  type TaskPriority,
  type TaskRecord,
} from "../lib/controlLayer";

const rooms = ["main-office", "trade-room", "agents-panel", "console-logs", "handoff-logs"];
const agents: TaskAgent[] = ["codex", "hermes", "chez", "system"];
const priorities: TaskPriority[] = ["low", "medium", "high", "urgent"];

const shellCard: CSSProperties = {
  borderRadius: 18,
  background: "rgba(15, 23, 42, 0.78)",
  border: "1px solid rgba(148, 163, 184, 0.16)",
  padding: 18,
};

const inputStyle: CSSProperties = {
  width: "100%",
  borderRadius: 12,
  border: "1px solid rgba(148, 163, 184, 0.2)",
  padding: "10px 12px",
  background: "rgba(15, 23, 42, 0.9)",
  color: "#e5eef7",
};

const textAreaStyle: CSSProperties = {
  ...inputStyle,
  resize: "vertical",
  minHeight: 92,
};

const buttonBase: CSSProperties = {
  border: 0,
  borderRadius: 12,
  padding: "10px 14px",
  fontWeight: 700,
  cursor: "pointer",
};

function badgeColor(status: string): string {
  switch (status) {
    case "completed":
      return "#22c55e";
    case "failed":
      return "#ef4444";
    case "in_progress":
    case "running":
      return "#f59e0b";
    default:
      return "#64748b";
  }
}

function sortByNewest<T extends { updated_at?: string; created_at?: string }>(items: T[]): T[] {
  return [...items].sort((left, right) =>
    String(right.updated_at || right.created_at || "").localeCompare(String(left.updated_at || left.created_at || "")),
  );
}

export default function TaskBoard() {
  const [tasks, setTasks] = useState<TaskRecord[]>([]);
  const [handoffs, setHandoffs] = useState<HandoffRecord[]>([]);
  const [logs, setLogs] = useState<LogRecord[]>([]);
  const [adapters, setAdapters] = useState<AdapterRecord[]>([]);
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [selectedLog, setSelectedLog] = useState<LogRecord | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [busyTaskId, setBusyTaskId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [form, setForm] = useState({
    title: "",
    goal: "",
    context: "",
    room: "main-office",
    agent: "codex" as TaskAgent,
    priority: "medium" as TaskPriority,
    autoRun: true,
  });

  const loadControlLayer = async () => {
    setLoading(true);
    setError(null);
    try {
      const [taskData, handoffData, logData, adapterData] = await Promise.all([
        listTasks(),
        listHandoffs(),
        listLogs(),
        listAdapters(),
      ]);
      const orderedTasks = sortByNewest(taskData);
      setTasks(orderedTasks);
      setHandoffs(sortByNewest(handoffData));
      setLogs([...logData].sort((left, right) => right.timestamp.localeCompare(left.timestamp)));
      setAdapters(adapterData);
      setSelectedTaskId((current) => current ?? orderedTasks[0]?.id ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load control layer");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadControlLayer();
  }, []);

  const selectedTask = useMemo(
    () => tasks.find((task) => task.id === selectedTaskId) ?? null,
    [selectedTaskId, tasks],
  );

  const relatedHandoffs = useMemo(() => {
    if (!selectedTask) {
      return [];
    }
    return handoffs.filter((handoff) => {
      const resultTaskId = typeof handoff.result?.task_id === "string" ? handoff.result.task_id : null;
      const payloadTaskId = typeof handoff.payload?.task_id === "string" ? String(handoff.payload.task_id) : null;
      return (
        handoff.id === selectedTask.handoff_id ||
        resultTaskId === selectedTask.id ||
        payloadTaskId === selectedTask.id
      );
    });
  }, [handoffs, selectedTask]);

  const relatedLogs = useMemo(() => {
    if (!selectedTask) {
      return [];
    }
    const handoffIds = new Set(relatedHandoffs.map((handoff) => handoff.id));
    return logs.filter((log) => {
      const taskMatch = log.metadata?.task_id === selectedTask.id;
      const handoffMatch = log.metadata?.handoff_id ? handoffIds.has(log.metadata.handoff_id) : false;
      return taskMatch || handoffMatch;
    });
  }, [logs, relatedHandoffs, selectedTask]);

  const timeline = useMemo(() => {
    if (!selectedTask) {
      return [] as Array<{ id: string; when: string; label: string; kind: "task" | "handoff" | "log" }>;
    }
    const taskItem = {
      id: `task-${selectedTask.id}`,
      when: selectedTask.updated_at || selectedTask.created_at,
      label: `Task ${selectedTask.status} in ${selectedTask.room || "main-office"}`,
      kind: "task" as const,
    };
    const handoffItems = relatedHandoffs.map((handoff) => ({
      id: `handoff-${handoff.id}`,
      when: handoff.completed_at || handoff.created_at,
      label: `${handoff.from_agent} -> ${handoff.to_agent} ${handoff.status}`,
      kind: "handoff" as const,
    }));
    const logItems = relatedLogs.map((log) => ({
      id: `log-${log.id}`,
      when: log.timestamp,
      label: `${log.level}: ${log.message}`,
      kind: "log" as const,
    }));
    return [taskItem, ...handoffItems, ...logItems].sort((left, right) => left.when.localeCompare(right.when));
  }, [relatedHandoffs, relatedLogs, selectedTask]);

  const onlineAgents = useMemo(() => new Set(adapters.filter((item) => item.status === "online").map((item) => item.name)), [adapters]);

  const submitTask = async () => {
    if (!form.title.trim() || !form.goal.trim()) {
      setError("Title and goal are required.");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const created = await createTask({
        title: form.title.trim(),
        goal: form.goal.trim(),
        context: form.context.trim(),
        room: form.room,
        agent: form.agent,
        priority: form.priority,
      });
      if (form.autoRun) {
        await runTask(created.id);
      }
      setForm({
        title: "",
        goal: "",
        context: "",
        room: form.room,
        agent: form.agent,
        priority: form.priority,
        autoRun: form.autoRun,
      });
      await loadControlLayer();
      setSelectedTaskId(created.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to create task");
    } finally {
      setSaving(false);
    }
  };

  const updateAssignment = async (taskId: string, payload: Partial<TaskRecord>) => {
    setBusyTaskId(taskId);
    setError(null);
    try {
      await patchTask(taskId, payload);
      await loadControlLayer();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to update task");
    } finally {
      setBusyTaskId(null);
    }
  };

  const actOnTask = async (taskId: string, action: "run" | "retry" | "requeue") => {
    setBusyTaskId(taskId);
    setError(null);
    try {
      if (action === "run") {
        await runTask(taskId);
      } else if (action === "retry") {
        await retryTask(taskId);
      } else {
        await requeueTask(taskId);
      }
      await loadControlLayer();
      setSelectedTaskId(taskId);
    } catch (err) {
      setError(err instanceof Error ? err.message : `Unable to ${action} task`);
    } finally {
      setBusyTaskId(null);
    }
  };

  return (
    <section style={{ display: "grid", gap: 20 }}>
      <div>
        <h2 style={{ margin: 0 }}>Task Board</h2>
        <p style={{ color: "#94a3b8", marginTop: 8 }}>
          Operator control layer for task creation, assignment, Hermes handoff orchestration, and result review.
        </p>
      </div>

      {error ? (
        <div style={{ ...shellCard, borderColor: "rgba(239, 68, 68, 0.4)", color: "#fecaca" }}>{error}</div>
      ) : null}

      <section style={{ ...shellCard, display: "grid", gap: 12 }}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <h3 style={{ margin: 0 }}>Create Operator Task</h3>
          <button type="button" onClick={() => void loadControlLayer()} style={{ ...buttonBase, background: "#1d4ed8", color: "#eff6ff" }}>
            Refresh Board
          </button>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 12 }}>
          <label style={{ display: "grid", gap: 6 }}>
            <span>Title</span>
            <input style={inputStyle} value={form.title} onChange={(event) => setForm((current) => ({ ...current, title: event.target.value }))} />
          </label>
          <label style={{ display: "grid", gap: 6 }}>
            <span>Room</span>
            <select style={inputStyle} value={form.room} onChange={(event) => setForm((current) => ({ ...current, room: event.target.value }))}>
              {rooms.map((room) => (
                <option key={room} value={room}>
                  {room}
                </option>
              ))}
            </select>
          </label>
          <label style={{ display: "grid", gap: 6 }}>
            <span>Assigned Agent</span>
            <select
              style={inputStyle}
              value={form.agent}
              onChange={(event) => setForm((current) => ({ ...current, agent: event.target.value as TaskAgent, autoRun: event.target.value === "codex" }))}
            >
              {agents.map((agent) => (
                <option key={agent} value={agent}>
                  {agent}
                </option>
              ))}
            </select>
          </label>
          <label style={{ display: "grid", gap: 6 }}>
            <span>Priority</span>
            <select style={inputStyle} value={form.priority} onChange={(event) => setForm((current) => ({ ...current, priority: event.target.value as TaskPriority }))}>
              {priorities.map((priority) => (
                <option key={priority} value={priority}>
                  {priority}
                </option>
              ))}
            </select>
          </label>
        </div>
        <label style={{ display: "grid", gap: 6 }}>
          <span>Goal</span>
          <textarea
            style={textAreaStyle}
            value={form.goal}
            onChange={(event) => setForm((current) => ({ ...current, goal: event.target.value }))}
            placeholder="What should Hermes hand off to Codex?"
          />
        </label>
        <label style={{ display: "grid", gap: 6 }}>
          <span>Context</span>
          <textarea
            style={textAreaStyle}
            value={form.context}
            onChange={(event) => setForm((current) => ({ ...current, context: event.target.value }))}
            placeholder="Optional extra notes or operator context"
          />
        </label>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <label style={{ display: "flex", gap: 8, color: "#cbd5e1" }}>
            <input
              type="checkbox"
              checked={form.autoRun}
              onChange={(event) => setForm((current) => ({ ...current, autoRun: event.target.checked }))}
              disabled={form.agent !== "codex"}
            />
            <span>Trigger Hermes → task → Codex immediately</span>
          </label>
          <button
            type="button"
            onClick={() => void submitTask()}
            disabled={saving}
            style={{ ...buttonBase, background: saving ? "#475569" : "#34d399", color: "#052e16" }}
          >
            {saving ? "Saving..." : form.autoRun ? "Create + Run" : "Create Task"}
          </button>
        </div>
        <div style={{ color: "#94a3b8", fontSize: 13 }}>
          Codex is the only agent auto-routed in this stage. Other assignments stay queued without breaking the current flow.
        </div>
      </section>

      <section style={{ display: "grid", gridTemplateColumns: "minmax(320px, 0.95fr) minmax(380px, 1.2fr)", gap: 20 }}>
        <article style={{ ...shellCard, display: "grid", gap: 12, alignContent: "start" }}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "center" }}>
            <h3 style={{ margin: 0 }}>Board</h3>
            <span style={{ color: "#94a3b8", fontSize: 13 }}>{tasks.length} tasks</span>
          </div>
          {loading ? (
            <div style={{ color: "#94a3b8" }}>Loading tasks...</div>
          ) : tasks.length === 0 ? (
            <div style={{ color: "#94a3b8" }}>No tasks yet. Create the first operator task above.</div>
          ) : (
            sortByNewest(tasks).map((task) => {
              const active = task.id === selectedTaskId;
              const runningDisabled = !onlineAgents.has("codex") && task.agent === "codex";
              return (
                <div
                  key={task.id}
                  style={{
                    borderRadius: 14,
                    padding: 14,
                    background: active ? "rgba(30, 41, 59, 0.95)" : "rgba(30, 41, 59, 0.72)",
                    border: `1px solid ${active ? "rgba(96, 165, 250, 0.45)" : "rgba(148, 163, 184, 0.12)"}`,
                  }}
                >
                  <button
                    type="button"
                    onClick={() => setSelectedTaskId(task.id)}
                    style={{ background: "transparent", border: 0, color: "inherit", width: "100%", textAlign: "left", padding: 0, cursor: "pointer" }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                      <strong>{task.title}</strong>
                      <span
                        style={{
                          display: "inline-flex",
                          alignItems: "center",
                          borderRadius: 999,
                          padding: "4px 10px",
                          background: `${badgeColor(task.status)}22`,
                          color: badgeColor(task.status),
                          fontSize: 12,
                          fontWeight: 700,
                        }}
                      >
                        {task.status}
                      </span>
                    </div>
                    <div style={{ marginTop: 8, color: "#cbd5e1" }}>{task.goal}</div>
                    <div style={{ marginTop: 8, color: "#94a3b8", fontSize: 13 }}>
                      {task.room || "main-office"} | {task.agent} | {task.priority}
                    </div>
                  </button>
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 12 }}>
                    <select
                      style={{ ...inputStyle, width: 140, padding: "8px 10px" }}
                      value={task.room || "main-office"}
                      onChange={(event) => void updateAssignment(task.id, { room: event.target.value })}
                      disabled={busyTaskId === task.id}
                    >
                      {rooms.map((room) => (
                        <option key={room} value={room}>
                          {room}
                        </option>
                      ))}
                    </select>
                    <select
                      style={{ ...inputStyle, width: 120, padding: "8px 10px" }}
                      value={task.agent}
                      onChange={(event) => void updateAssignment(task.id, { agent: event.target.value as TaskAgent })}
                      disabled={busyTaskId === task.id}
                    >
                      {agents.map((agent) => (
                        <option key={agent} value={agent}>
                          {agent}
                        </option>
                      ))}
                    </select>
                    <button
                      type="button"
                      onClick={() => void actOnTask(task.id, task.status === "failed" ? "retry" : "run")}
                      disabled={busyTaskId === task.id || runningDisabled}
                      style={{ ...buttonBase, background: "#22c55e", color: "#052e16" }}
                    >
                      {busyTaskId === task.id ? "Working..." : task.status === "failed" ? "Retry" : task.status === "completed" ? "Run Again" : "Run"}
                    </button>
                    <button
                      type="button"
                      onClick={() => void actOnTask(task.id, "requeue")}
                      disabled={busyTaskId === task.id}
                      style={{ ...buttonBase, background: "#334155", color: "#e2e8f0" }}
                    >
                      Requeue
                    </button>
                  </div>
                  {runningDisabled ? <div style={{ marginTop: 8, color: "#fca5a5", fontSize: 13 }}>Codex is offline.</div> : null}
                </div>
              );
            })
          )}
        </article>

        <article style={{ ...shellCard, display: "grid", gap: 14, alignContent: "start" }}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "center" }}>
            <h3 style={{ margin: 0 }}>Task Detail</h3>
            {selectedTask ? <span style={{ color: "#94a3b8", fontSize: 13 }}>{selectedTask.id}</span> : null}
          </div>
          {!selectedTask ? (
            <div style={{ color: "#94a3b8" }}>Select a task to inspect workflow details.</div>
          ) : (
            <>
              <div style={{ display: "grid", gap: 8 }}>
                <strong style={{ fontSize: 18 }}>{selectedTask.title}</strong>
                <div style={{ color: "#cbd5e1" }}>{selectedTask.goal}</div>
                <div style={{ color: "#94a3b8", fontSize: 13 }}>
                  Room {selectedTask.room || "main-office"} | Agent {selectedTask.agent} | Priority {selectedTask.priority}
                </div>
                <div style={{ color: "#94a3b8", fontSize: 13 }}>
                  Updated {selectedTask.updated_at}
                </div>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 12 }}>
                <div style={{ ...shellCard, padding: 14 }}>
                  <div style={{ color: "#94a3b8", fontSize: 13, marginBottom: 6 }}>Result State</div>
                  <div style={{ color: badgeColor(selectedTask.status), fontWeight: 700 }}>{selectedTask.status}</div>
                  <div style={{ marginTop: 8, color: selectedTask.error ? "#fecaca" : "#cbd5e1", whiteSpace: "pre-wrap" }}>
                    {selectedTask.error || selectedTask.result || "No result yet."}
                  </div>
                </div>
                <div style={{ ...shellCard, padding: 14 }}>
                  <div style={{ color: "#94a3b8", fontSize: 13, marginBottom: 6 }}>Context</div>
                  <div style={{ color: "#cbd5e1", whiteSpace: "pre-wrap" }}>{selectedTask.context || "No extra context."}</div>
                </div>
              </div>

              <div style={{ ...shellCard, padding: 14 }}>
                <div style={{ color: "#94a3b8", fontSize: 13, marginBottom: 8 }}>Handoff Timeline</div>
                {timeline.length === 0 ? (
                  <div style={{ color: "#94a3b8" }}>No timeline yet.</div>
                ) : (
                  <div style={{ display: "grid", gap: 10 }}>
                    {timeline.map((item) => (
                      <div key={item.id} style={{ display: "grid", gap: 4, paddingLeft: 12, borderLeft: `3px solid ${badgeColor(item.kind === "log" ? "running" : selectedTask.status)}` }}>
                        <div style={{ color: "#cbd5e1" }}>{item.label}</div>
                        <div style={{ color: "#94a3b8", fontSize: 12 }}>{item.when}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 12 }}>
                <div style={{ ...shellCard, padding: 14 }}>
                  <div style={{ color: "#94a3b8", fontSize: 13, marginBottom: 8 }}>Linked Handoffs</div>
                  {relatedHandoffs.length === 0 ? (
                    <div style={{ color: "#94a3b8" }}>No handoffs yet.</div>
                  ) : (
                    <div style={{ display: "grid", gap: 10 }}>
                      {relatedHandoffs.map((handoff) => (
                        <div key={handoff.id} style={{ padding: 12, borderRadius: 12, background: "rgba(30, 41, 59, 0.75)" }}>
                          <div style={{ display: "flex", justifyContent: "space-between", gap: 8, flexWrap: "wrap" }}>
                            <strong>
                              {handoff.from_agent} -&gt; {handoff.to_agent}
                            </strong>
                            <span style={{ color: badgeColor(handoff.status), fontWeight: 700 }}>{handoff.status}</span>
                          </div>
                          <div style={{ marginTop: 6, color: "#94a3b8", fontSize: 13 }}>{handoff.completed_at || handoff.created_at}</div>
                          <div style={{ marginTop: 6, color: "#cbd5e1", whiteSpace: "pre-wrap" }}>
                            {typeof handoff.result?.output_preview === "string"
                              ? handoff.result.output_preview
                              : typeof handoff.result?.error === "string"
                                ? handoff.result.error
                                : JSON.stringify(handoff.result || {}, null, 2)}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
                <div style={{ ...shellCard, padding: 14 }}>
                  <div style={{ color: "#94a3b8", fontSize: 13, marginBottom: 8 }}>Related Logs</div>
                  {relatedLogs.length === 0 ? (
                    <div style={{ color: "#94a3b8" }}>No logs yet.</div>
                  ) : (
                    <div style={{ display: "grid", gap: 10 }}>
                      {relatedLogs.map((log) => (
                        <button
                          key={log.id}
                          type="button"
                          onClick={() => setSelectedLog(log)}
                          style={{
                            textAlign: "left",
                            padding: 12,
                            borderRadius: 12,
                            border: "1px solid rgba(148, 163, 184, 0.12)",
                            background: "rgba(30, 41, 59, 0.75)",
                            color: "inherit",
                            cursor: "pointer",
                          }}
                        >
                          <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
                            <span style={{ color: badgeColor(log.level), fontWeight: 700 }}>{log.level}</span>
                            <span style={{ color: "#94a3b8", fontSize: 12 }}>{log.timestamp}</span>
                          </div>
                          <div style={{ marginTop: 6 }}>{log.message}</div>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </>
          )}
        </article>
      </section>

      {selectedLog ? (
        <section style={{ ...shellCard, display: "grid", gap: 10 }}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "center" }}>
            <h3 style={{ margin: 0 }}>Log Detail</h3>
            <button type="button" onClick={() => setSelectedLog(null)} style={{ ...buttonBase, background: "#334155", color: "#e2e8f0" }}>
              Close
            </button>
          </div>
          <div style={{ color: badgeColor(selectedLog.level), fontWeight: 700 }}>{selectedLog.level}</div>
          <div style={{ color: "#94a3b8", fontSize: 13 }}>{selectedLog.timestamp}</div>
          <div style={{ color: "#cbd5e1", whiteSpace: "pre-wrap" }}>{selectedLog.message}</div>
          <pre style={{ margin: 0, padding: 12, borderRadius: 12, background: "rgba(15, 23, 42, 0.9)", overflowX: "auto" }}>
            {JSON.stringify(selectedLog.metadata || {}, null, 2)}
          </pre>
        </section>
      ) : null}
    </section>
  );
}
