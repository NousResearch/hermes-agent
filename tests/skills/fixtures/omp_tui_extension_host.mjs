// Synthetic OMP hook driver; the extension, socket and journal are real.
import { createInterface } from "node:readline";
import { lstatSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";

const extensionUrl = process.argv[2] ? pathToFileURL(process.argv[2]) :
  new URL("../../../optional-skills/autonomous-ai-agents/tmux-supervision/scripts/tmux_supervisor/omp_extension.ts", import.meta.url);
const { default: extension } = await import(extensionUrl.href);

const binding = JSON.parse(readFileSync(process.env.OMP_HERMES_BINDING_FILE, "utf8"));
const hooks = new Map();
let sessionId = "synthetic-cross-language-session";
const ctx = {
  mode: "tui", cwd: binding.workspace, hasPendingMessages: () => false,
  sessionManager: { getSessionId: () => sessionId },
};
extension({ on(name, handler) { hooks.set(name, handler); } });
await hooks.get("session_start")({ type: "session_start" }, ctx);
const run = dirname(process.env.OMP_HERMES_BINDING_FILE);
if (!lstatSync(join(run, "bridge.sock")).isSocket() ||
    JSON.parse(readFileSync(join(run, "journal.json"), "utf8")).pid !== process.pid) {
  throw new Error("Synthetic bridge did not start");
}
console.log("READY");
for await (const line of createInterface({ input: process.stdin })) {
  const command = JSON.parse(line);
  if (command.sessionId) sessionId = command.sessionId;
  await hooks.get(command.type)(command, ctx);
  console.log("ACK");
  if (command.type === "session_shutdown") break;
}
