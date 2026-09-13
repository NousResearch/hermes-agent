#!/usr/bin/env node

import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { spawn } from "node:child_process";
import { execFileSync } from "node:child_process";

const args = process.argv.slice(2);
const isWindows = process.platform === "win32";
const hermesHome = process.env.HERMES_HOME || join(homedir(), ".hermes");

function localLaunchers() {
  if (isWindows) {
    return [
      join(hermesHome, "hermes-agent", "venv", "Scripts", "hermes-acp.exe"),
      join(hermesHome, "hermes-agent", "venv", "Scripts", "hermes-acp.bat"),
    ];
  }
  return [
    join(hermesHome, "hermes-agent", "venv", "bin", "hermes-acp"),
    join(homedir(), ".local", "bin", "hermes-acp"),
  ];
}

function commandExists(command) {
  try {
    execFileSync(isWindows ? "where.exe" : "which", [command], {
      stdio: "ignore",
    });
    return true;
  } catch {
    return false;
  }
}

function run(command, commandArgs) {
  const child = spawn(command, commandArgs, {
    stdio: "inherit",
    env: process.env,
    windowsHide: true,
  });
  child.on("error", (error) => {
    console.error(`Hermes ACP launcher failed to start: ${error.message}`);
    process.exitCode = 1;
  });
  child.on("exit", (code, signal) => {
    if (signal) {
      process.kill(process.pid, signal);
    } else {
      process.exitCode = code ?? 1;
    }
  });
}

const launcher = localLaunchers().find((candidate) => existsSync(candidate));
if (launcher) {
  run(launcher, args);
} else if (commandExists("hermes-acp")) {
  run("hermes-acp", args);
} else if (!isWindows && commandExists("uvx")) {
  // This keeps the registry entry useful for machines that have uv but use a
  // managed Hermes install elsewhere. The normal install path still prefers
  // the user's current hermes-acp launcher above.
  run("uvx", ["--from", "hermes-agent[acp]", "hermes-acp", ...args]);
} else {
  const pathHint = isWindows
    ? "%USERPROFILE%\\.hermes\\hermes-agent\\venv\\Scripts\\hermes-acp.exe"
    : "~/.local/bin/hermes-acp";
  console.error(
    "Hermes Agent is not installed. Install it from " +
      "https://hermes-agent.nousresearch.com/install.sh, then retry.\n" +
      `Expected ACP launcher: ${pathHint}`,
  );
  process.exitCode = 1;
}
