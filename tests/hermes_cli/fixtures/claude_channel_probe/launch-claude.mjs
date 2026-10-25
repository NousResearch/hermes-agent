#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";

import { openNewPrivateOutputPath } from "./contract.mjs";

const debugPath = process.argv[2];
const separator = process.argv[3];
const command = process.argv[4];
const args = process.argv.slice(5);
if (!debugPath || separator !== "--" || !command) {
  throw new Error("usage: node launch-claude.mjs /absolute/debug-file -- command [args...]");
}

const debugDescriptor = openNewPrivateOutputPath(path.dirname(debugPath), debugPath);
let child;
try {
  child = spawn(command, args, {
    stdio: ["inherit", "inherit", "inherit", debugDescriptor],
  });
} finally {
  fs.closeSync(debugDescriptor);
}

const signalExit = Object.freeze({ SIGHUP: 129, SIGINT: 130, SIGTERM: 143 });
let forwardedSignal = null;
for (const signal of Object.keys(signalExit)) {
  process.on(signal, () => {
    if (forwardedSignal) return;
    forwardedSignal = signal;
    child.kill(signal);
  });
}
child.once("error", (error) => {
  process.stderr.write(`Claude launch failed: ${String(error.code ?? "SPAWN_ERROR")}\n`);
  process.exitCode = 127;
});
child.once("exit", (code, signal) => {
  if (Number.isInteger(code)) process.exitCode = code;
  else process.exitCode = signalExit[forwardedSignal ?? signal] ?? 1;
});

