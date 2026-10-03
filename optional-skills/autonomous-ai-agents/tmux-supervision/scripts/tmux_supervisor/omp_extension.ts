import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";
import { randomUUID } from "node:crypto";
import {
  chmodSync, closeSync, constants, existsSync, fstatSync, fsyncSync, lstatSync,
  openSync, readFileSync, realpathSync, renameSync, unlinkSync, writeFileSync,
} from "node:fs";
import { createServer } from "node:net";
import type { Server, Socket } from "node:net";
import { basename, dirname, isAbsolute, join } from "node:path";

type State = "idle" | "busy" | "revoked" | "closed";
type Kind = "started" | "turn_settled" | "needs_input" | "error" | "session_revoked" | "shutdown" | "observation_lost";
type Event = {
  version: 2; type: "event"; run_id: string; epoch: string; seq: number;
  app_session_id: string; kind: Kind; at_ms: number;
};
type Binding = {
  mode: "tmux"; adapter: "omp";
  version: 2; run_id: string; workspace: string; tmux_session: string;
  owner: { hermes_home: string; platform: string; session_key: string;
    session_id: string; chat_id: string; thread_id: string | null };
  created_at: number;
};
const uuid = () => randomUUID().replaceAll("-", "");
const text = (value: unknown): value is string =>
  typeof value === "string" && value.length > 0 && value.length <= 1024 && !/[\x00-\x1f\x7f]/.test(value);

function realDirectory(path: string): boolean {
  return isAbsolute(path) && realpathSync(path) === path && lstatSync(path).isDirectory();
}

function readBinding(path: string): Binding {
  if (!isAbsolute(path) || basename(path) !== "binding.json" || realpathSync(path) !== path) throw new Error("binding");
  const dir = lstatSync(dirname(path));
  if (!dir.isDirectory() || dir.uid !== process.getuid!() || (dir.mode & 0o777) !== 0o700) throw new Error("directory");
  const fd = openSync(path, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
  try {
    const stat = fstatSync(fd);
    if (!stat.isFile() || stat.uid !== process.getuid!() || (stat.mode & 0o777) !== 0o600 || stat.nlink !== 1 || stat.size > 16384) throw new Error("binding");
    const value = JSON.parse(readFileSync(fd, "utf8"));
    if (value?.version !== 2 || value.mode !== "tmux" || value.adapter !== "omp" || !/^[a-f0-9]{32}$/.test(value.run_id) || basename(dirname(path)) !== value.run_id ||
        !text(value.workspace) || !realDirectory(value.workspace) ||
        !text(value.tmux_session) || !/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(value.tmux_session) ||
        !text(value.owner?.platform) || !/^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/.test(value.owner.platform) ||
        !text(value.owner.hermes_home) || !realDirectory(value.owner.hermes_home) ||
        !text(value.owner.session_key) || !text(value.owner.session_id) || !text(value.owner.chat_id) ||
        !(value.owner.thread_id === null || value.owner.thread_id === "" || text(value.owner.thread_id)) ||
        !Number.isFinite(value.created_at) || value.created_at < 0) throw new Error("binding");
    return value;
  } finally { closeSync(fd); }
}

/** Observe one interactive OMP session; never steer it or retain its content. */
export default function tuiBridge(pi: ExtensionAPI): void {
  const bindingPath = process.env.OMP_HERMES_BINDING_FILE;
  if (!bindingPath) return;
  let binding: Binding;
  try { binding = readBinding(bindingPath); } catch { return; }
  const directory = dirname(bindingPath);
  const directoryIdentity = lstatSync(directory);
  const socketPath = join(directory, "bridge.sock");
  const journalPath = join(directory, "journal.json");
  if (Buffer.byteLength(socketPath) > 107) return;
  const epoch = uuid();
  let server: Server | undefined;
  let initialized = false;
  let stopped = false;
  let sessionId = "";
  let seq = 0;
  let state: State = "idle";
  let events: Event[] = [];
  let cycleOpen = false;
  const clients = new Set<Socket>();
  const subscribers = new Set<Socket>();

  function closeObservation(graceful = false): void {
    stopped = true;
    for (const socket of clients) {
      if (graceful) {
        socket.end();
        socket.setTimeout(1000, () => socket.destroy());
      } else socket.destroy();
    }
    server?.close();
  }

  function persist(nextEvents: Event[], nextSeq: number, nextState: State): void {
    const stat = lstatSync(directory);
    if (realpathSync(directory) !== directory || stat.dev !== directoryIdentity.dev || stat.ino !== directoryIdentity.ino ||
        stat.uid !== process.getuid!() || (stat.mode & 0o777) !== 0o700) throw new Error("directory");
    const temporary = join(directory, `.journal-${uuid()}`);
    let fd: number | undefined;
    try {
      fd = openSync(temporary, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW, 0o600);
      writeFileSync(fd, JSON.stringify({ version: 2, run_id: binding.run_id, epoch,
        app_session_id: sessionId, pid: process.pid, seq: nextSeq, state: nextState, events: nextEvents }) + "\n");
      fsyncSync(fd);
      closeSync(fd);
      fd = undefined;
      renameSync(temporary, journalPath);
      const directoryFd = openSync(directory, constants.O_RDONLY | constants.O_DIRECTORY | constants.O_NOFOLLOW);
      try { fsyncSync(directoryFd); } finally { closeSync(directoryFd); }
    } finally {
      if (fd !== undefined) closeSync(fd);
      if (existsSync(temporary)) unlinkSync(temporary);
    }
  }

  function send(socket: Socket, value: object): boolean {
    if (socket.destroyed) return false;
    const line = JSON.stringify(value) + "\n";
    if (socket.writableLength + Buffer.byteLength(line) > 65536) { socket.destroy(); return false; }
    try { socket.write(line); return true; } catch { socket.destroy(); return false; }
  }

  function publish(kind: Kind, nextState: State = state): Event | undefined {
    if (stopped) return;
    const event: Event = { version: 2, type: "event", run_id: binding.run_id, epoch,
      seq: seq + 1, app_session_id: sessionId, kind, at_ms: Date.now() };
    const nextEvents = [...events, event].slice(-128);
    try { persist(nextEvents, event.seq, nextState); }
    catch { closeObservation(); return; }
    events = nextEvents;
    seq = event.seq;
    state = nextState;
    for (const socket of subscribers) send(socket, event);
    return event;
  }

  function accept(socket: Socket): void {
    if (stopped || clients.size >= 32) { socket.destroy(); return; }
    clients.add(socket);
    socket.unref();
    socket.on("error", () => socket.destroy());
    socket.on("close", () => { clients.delete(socket); subscribers.delete(socket); });
    socket.setTimeout(5000, () => socket.destroy());
    let input = Buffer.alloc(0);
    let subscribed = false;
    socket.on("data", (chunk: Buffer) => {
      if (subscribed || input.length + chunk.length > 4096) { socket.destroy(); return; }
      input = Buffer.concat([input, chunk]);
      const newline = input.indexOf(10);
      if (newline < 0) return;
      if (newline !== input.length - 1) { socket.destroy(); return; }
      let request;
      try { request = JSON.parse(input.subarray(0, newline).toString("utf8")); }
      catch { socket.destroy(); return; }
      if (!request || typeof request !== "object" || Array.isArray(request) ||
          Object.keys(request).some(key => !["version", "type", "run_id", "after_seq", "epoch"].includes(key)) ||
          request.version !== 2 || request.type !== "observe" || request.run_id !== binding.run_id ||
          !Number.isSafeInteger(request.after_seq) || request.after_seq < 0 ||
          (request.epoch !== undefined && (typeof request.epoch !== "string" || !/^[a-f0-9]{32}$/.test(request.epoch)))) {
        socket.destroy(); return;
      }
      subscribed = true;
      input = Buffer.alloc(0);
      socket.setTimeout(0);
      send(socket, { version: 2, type: "hello", run_id: binding.run_id, epoch,
        app_session_id: sessionId, pid: process.pid, seq, state });
      if ((request.epoch !== undefined && request.epoch !== epoch) || request.after_seq > seq ||
          (request.after_seq > 0 && request.epoch === undefined) ||
          request.after_seq < (events[0]?.seq ?? 1) - 1) {
        // A bad cursor belongs to this client, not the shared OMP event stream.
        send(socket, { version: 2, type: "rejected", run_id: binding.run_id,
          reason: request.epoch !== undefined && request.epoch !== epoch ? "epoch_mismatch" : "invalid_cursor" });
        socket.end();
        socket.setTimeout(1000, () => socket.destroy());
        return;
      }
      for (const event of events) if (event.seq > request.after_seq) send(socket, event);
      if (state === "revoked" || state === "closed") {
        socket.end();
        socket.setTimeout(1000, () => socket.destroy());
      } else if (!socket.destroyed) subscribers.add(socket);
    });
  }

  function active(ctx: { mode: string; sessionManager: { getSessionId(): string } }): boolean {
    if (!initialized || stopped || state === "revoked" || state === "closed") return false;
    if (ctx.mode !== "tui") { closeObservation(); return false; }
    if (ctx.sessionManager.getSessionId() !== sessionId) {
      publish("session_revoked", "revoked");
      cycleOpen = false;
      for (const socket of clients) { socket.end(); socket.setTimeout(1000, () => socket.destroy()); }
      return false;
    }
    return true;
  }

  pi.on("session_start", async (_event, ctx) => {
    if (initialized || stopped) return;
    initialized = true;
    try {
      if (ctx.mode !== "tui" || realpathSync(ctx.cwd) !== binding.workspace ||
          lstatSync(socketPath, { throwIfNoEntry: false }) || lstatSync(journalPath, { throwIfNoEntry: false })) { stopped = true; return; }
      sessionId = ctx.sessionManager.getSessionId();
      if (!text(sessionId)) { stopped = true; return; }
      server = createServer(accept);
      server.on("error", () => closeObservation());
      await new Promise<void>((resolve, reject) => {
        server!.once("error", reject);
        server!.listen(socketPath, () => { server!.removeListener("error", reject); resolve(); });
      });
      chmodSync(socketPath, 0o600);
      persist([], 0, "idle");
      server.unref();
    } catch { closeObservation(); }
  });
  pi.on("agent_start", (_event, ctx) => {
    try { if (active(ctx)) { cycleOpen = true; publish("started", "busy"); } }
    catch { closeObservation(); }
  });
  pi.on("agent_end", (event, ctx) => {
    try {
      if (!active(ctx) || !cycleOpen) return;
      cycleOpen = false;
      // A scheduled continuation or queued prompt is not a user-visible settle.
      if (event.willContinue === true || ctx.hasPendingMessages()) return;
      const outcome = event.messages.findLast(message => message?.role === "assistant");
      publish(outcome?.stopReason === "error" ? "error" : "turn_settled", "idle");
    } catch { closeObservation(); }
  });
  pi.on("tool_approval_requested", (event, ctx) => {
    try { if (active(ctx) && event.sessionId === sessionId) publish("needs_input"); }
    catch { closeObservation(); }
  });
  pi.on("session_switch", () => {
    if (!initialized || stopped || state === "revoked" || state === "closed") return;
    publish("session_revoked", "revoked");
    cycleOpen = false;
    for (const socket of clients) { socket.end(); socket.setTimeout(1000, () => socket.destroy()); }
  });
  pi.on("session_branch", (_event, ctx) => {
    try { active(ctx); } catch { closeObservation(); }
  });
  pi.on("session_shutdown", () => {
    if (!initialized || stopped) return;
    if (state !== "revoked") publish("shutdown", "closed");
    closeObservation(true);
  });
}
