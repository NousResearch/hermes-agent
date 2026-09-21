#!/usr/bin/env node

import { spawnSync as runProcessSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HOOK_NAME = "clawsec-advisory-guardian";
const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const SUITE_DIR = path.resolve(SCRIPT_DIR, "..");
const SOURCE_HOOK_DIR = path.join(SUITE_DIR, "hooks", HOOK_NAME);
const HOOKS_ROOT = path.join(os.homedir(), ".openclaw", "hooks");
const TARGET_HOOK_DIR = path.join(HOOKS_ROOT, HOOK_NAME);

function sh(cmd, args) {
  const result = runProcessSync(cmd, args, {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });

  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    const details = (result.stderr || result.stdout || "").trim();
    throw new Error(`${cmd} ${args.join(" ")} failed${details ? `: ${details}` : ""}`);
  }

  return result.stdout;
}

function requireOpenClawCli() {
  try {
    sh("openclaw", ["--version"]);
  } catch (error) {
    throw new Error(
      "openclaw CLI is required. Install OpenClaw and ensure `openclaw` is available in PATH. " +
        `Original error: ${String(error)}`,
      { cause: error },
    );
  }
}

function assertSourceHookExists() {
  const requiredFiles = [
    "HOOK.md",
    "handler.ts",
    "lib/utils.mjs",
    "lib/version.mjs",
    "lib/feed.mjs",
  ];
  for (const file of requiredFiles) {
    const fullPath = path.join(SOURCE_HOOK_DIR, file);
    if (!fs.existsSync(fullPath)) {
      throw new Error(`Missing required hook file: ${fullPath}`);
    }
  }
}

function installHookFiles() {
  fs.mkdirSync(HOOKS_ROOT, { recursive: true });
  fs.rmSync(TARGET_HOOK_DIR, { recursive: true, force: true });
  fs.cpSync(SOURCE_HOOK_DIR, TARGET_HOOK_DIR, { recursive: true });
}

function printPreflightSummary() {
  const lines = [
    "Preflight review:",
    `- This setup installs a persistent OpenClaw hook under ${TARGET_HOOK_DIR} and enables it globally.`,
    "- Required runtime: openclaw CLI, node.",
    "- The installed hook fetches signed advisory feed data and may recommend removal of risky skills, but destructive actions remain approval-gated.",
    `- Source hook files: ${SOURCE_HOOK_DIR}`,
    "- Restart your OpenClaw gateway process after setup so the hook loads intentionally.",
  ];

  process.stdout.write(lines.join("\n") + "\n\n");
}

function enableHook() {
  sh("openclaw", ["hooks", "enable", HOOK_NAME]);
}

function main() {
  assertSourceHookExists();
  printPreflightSummary();
  requireOpenClawCli();
  installHookFiles();
  enableHook();

  process.stdout.write(`Installed hook files to: ${TARGET_HOOK_DIR}\n`);
  process.stdout.write(`Enabled hook: ${HOOK_NAME}\n`);
  process.stdout.write("Restart your OpenClaw gateway process so the hook is loaded.\n");
  process.stdout.write("After restart, run /new once to trigger an immediate advisory scan.\n");
}

try {
  main();
} catch (error) {
  process.stderr.write(`${String(error)}\n`);
  process.exit(1);
}
