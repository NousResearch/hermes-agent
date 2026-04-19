// Darwinian Evolver — VS Code extension (C4, v1.0 skeleton).
//
// The extension wraps the existing CLI (`evolver.py`) + the FastAPI
// dashboard. Every command here is a thin invocation of the CLI
// through `child_process.spawn` so the extension never duplicates
// evolver logic — it just sits on top of the already-tested Python
// surface.
//
// v1.0 ships five commands:
//   * Run Experiment         → `evolver run`
//   * Open Dashboard         → `evolver dashboard` + browser open
//   * Synthesise fitness.py  → `evolver synthesise-fitness`
//   * Show Candidate Lineage → opens /lineage/<cid> in dashboard
//   * Submit Human Edit      → POST /api/candidate/<cid>/edit
//
// A richer tree view + live monitoring lands in v1.1 when the
// WebSocket stream can hydrate into a VS Code WebView.

import * as cp from "child_process";
import * as fs from "fs";
import * as path from "path";
import * as vscode from "vscode";

type Cfg = {
  cliPath:       string;
  pythonPath:    string;
  dashboardPort: number;
  dashboardHost: string;
};

function readConfig(): Cfg {
  const c = vscode.workspace.getConfiguration("darwinianEvolver");
  return {
    cliPath:       c.get<string>("cliPath",    ""),
    pythonPath:    c.get<string>("pythonPath", "python3"),
    dashboardPort: c.get<number>("dashboardPort", 8787),
    dashboardHost: c.get<string>("dashboardHost", "127.0.0.1"),
  };
}

function resolveCli(cfg: Cfg): string {
  if (cfg.cliPath && fs.existsSync(cfg.cliPath)) {
    return cfg.cliPath;
  }
  const home = process.env.HERMES_HOME ?? path.join(process.env.HOME ?? "~", ".hermes");
  // Installed location for the skill's scripts.
  const installed = path.join(
    home, "skills", "research", "darwinian-evolver", "scripts", "evolver.py",
  );
  if (fs.existsSync(installed)) { return installed; }
  // Fallback: live development tree.
  if (vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders.length > 0) {
    const root = vscode.workspace.workspaceFolders[0].uri.fsPath;
    const dev = path.join(
      root, "optional-skills", "research", "darwinian-evolver", "scripts", "evolver.py",
    );
    if (fs.existsSync(dev)) { return dev; }
  }
  throw new Error(
    "Could not locate evolver.py. Set darwinianEvolver.cliPath in your settings, " +
    "or ensure the skill is installed under $HERMES_HOME.",
  );
}

function runCli(
  args: readonly string[],
  ctx: { cwd?: string; env?: NodeJS.ProcessEnv } = {},
): Promise<{ code: number; stdout: string; stderr: string }> {
  const cfg = readConfig();
  const cli = resolveCli(cfg);
  return new Promise((resolve) => {
    const proc = cp.spawn(cfg.pythonPath, [cli, ...args], {
      cwd: ctx.cwd,
      env: { ...process.env, ...(ctx.env ?? {}) },
    });
    const out: string[] = []; const err: string[] = [];
    proc.stdout.on("data", (chunk) => out.push(chunk.toString()));
    proc.stderr.on("data", (chunk) => err.push(chunk.toString()));
    proc.on("close", (code) => resolve({
      code:   code ?? 0,
      stdout: out.join(""),
      stderr: err.join(""),
    }));
  });
}

async function pickExperimentDir(): Promise<string | undefined> {
  const folders = await vscode.window.showOpenDialog({
    canSelectFiles:   false,
    canSelectFolders: true,
    canSelectMany:    false,
    openLabel:        "Select experiment directory",
  });
  return folders?.[0]?.fsPath;
}

// ---------------------------------------------------------------------------
// Command: Run Experiment
// ---------------------------------------------------------------------------

async function cmdRun(): Promise<void> {
  const dir = await pickExperimentDir();
  if (!dir) { return; }

  const gens = await vscode.window.showInputBox({
    prompt: "Generations", value: "10", validateInput: (v) => {
      const n = Number(v); return Number.isInteger(n) && n > 0 ? null : "positive integer please";
    },
  });
  if (!gens) { return; }

  const pop = await vscode.window.showInputBox({
    prompt: "Population size", value: "8",
  }) ?? "8";
  const budget = await vscode.window.showInputBox({
    prompt: "Budget USD cap (0 disables)", value: "0.5",
  }) ?? "0.5";

  const output = vscode.window.createOutputChannel("Darwinian Evolver");
  output.show(true);
  output.appendLine(`Running evolver on ${dir} ...`);

  const { code, stdout, stderr } = await runCli([
    "run", dir, "--generations", gens, "--pop", pop, "--budget", budget,
  ], { cwd: dir });
  output.append(stdout);
  if (stderr) { output.append("\n[stderr]\n" + stderr); }
  output.appendLine(`\nexit code: ${code}`);
}

// ---------------------------------------------------------------------------
// Command: Open Dashboard
// ---------------------------------------------------------------------------

async function cmdDashboard(): Promise<void> {
  const dir = await pickExperimentDir();
  if (!dir) { return; }
  const cfg = readConfig();
  const output = vscode.window.createOutputChannel("Darwinian Evolver Dashboard");
  output.show(true);

  // Launch detached so VS Code doesn't block waiting on the server.
  const cli = resolveCli(cfg);
  const proc = cp.spawn(cfg.pythonPath, [
    cli, "dashboard", dir, "--host", cfg.dashboardHost, "--port", String(cfg.dashboardPort),
  ], { cwd: dir, detached: true, stdio: "ignore" });
  proc.unref();

  const url = `http://${cfg.dashboardHost}:${cfg.dashboardPort}/`;
  output.appendLine(`Dashboard listening at ${url}`);
  await vscode.env.openExternal(vscode.Uri.parse(url));
}

// ---------------------------------------------------------------------------
// Command: Synthesise fitness.py
// ---------------------------------------------------------------------------

async function cmdSynthesiseFitness(): Promise<void> {
  const dir = await pickExperimentDir();
  if (!dir) { return; }
  const file = await vscode.window.showOpenDialog({
    canSelectFiles: true, canSelectFolders: false, canSelectMany: false,
    openLabel: "Select examples JSONL",
    filters: { JSONL: ["jsonl", "json"] },
  });
  if (!file?.[0]) { return; }
  const criterion = await vscode.window.showInputBox({
    prompt: "Criterion (e.g. brevity, correctness, clarity)", value: "correctness",
  }) ?? "correctness";
  const output = vscode.window.createOutputChannel("Darwinian Evolver");
  output.show(true);
  const { code, stdout, stderr } = await runCli([
    "synthesise-fitness", dir, "--examples", file[0].fsPath, "--criterion", criterion,
  ]);
  output.append(stdout);
  if (stderr) { output.append("\n[stderr]\n" + stderr); }
  output.appendLine(`\nexit code: ${code}`);
}

// ---------------------------------------------------------------------------
// Command: Show Candidate Lineage
// ---------------------------------------------------------------------------

async function cmdShowLineage(): Promise<void> {
  const cid = await vscode.window.showInputBox({
    prompt: "Candidate id (blake2b-16)", validateInput: (v) =>
      /^[0-9a-f]{16}$/.test(v) ? null : "expected 16 hex chars",
  });
  if (!cid) { return; }
  const cfg = readConfig();
  const url = `http://${cfg.dashboardHost}:${cfg.dashboardPort}/api/lineage/${cid}`;
  await vscode.env.openExternal(vscode.Uri.parse(url));
}

// ---------------------------------------------------------------------------
// Command: Submit Human Edit to Dashboard
// ---------------------------------------------------------------------------

async function cmdAcceptHumanEdit(): Promise<void> {
  const cid = await vscode.window.showInputBox({ prompt: "Parent candidate id" });
  if (!cid) { return; }
  const genome = await vscode.window.showInputBox({ prompt: "Edited genome text" });
  if (!genome) { return; }
  const cfg = readConfig();
  const url = `http://${cfg.dashboardHost}:${cfg.dashboardPort}/api/candidate/${cid}/edit`;
  try {
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ genome }),
    });
    if (!resp.ok) {
      vscode.window.showErrorMessage(`Dashboard rejected edit: ${resp.status}`);
      return;
    }
    const payload = await resp.json() as { id: string; parent: string; generation: number };
    vscode.window.showInformationMessage(
      `Edit recorded → child ${payload.id} (gen ${payload.generation})`,
    );
  } catch (exc) {
    vscode.window.showErrorMessage(
      `Could not reach dashboard at ${url}. Is it running?`,
    );
  }
}

// ---------------------------------------------------------------------------
// Activation
// ---------------------------------------------------------------------------

export function activate(context: vscode.ExtensionContext): void {
  const register = (id: string, fn: (...args: unknown[]) => unknown) =>
    context.subscriptions.push(vscode.commands.registerCommand(id, fn));

  register("darwinianEvolver.run",                cmdRun);
  register("darwinianEvolver.dashboard",          cmdDashboard);
  register("darwinianEvolver.synthesiseFitness",  cmdSynthesiseFitness);
  register("darwinianEvolver.showLineage",        cmdShowLineage);
  register("darwinianEvolver.acceptHumanEdit",    cmdAcceptHumanEdit);
}

export function deactivate(): void { /* nothing to tear down */ }
