#!/usr/bin/env node

import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { spawnSync } from "node:child_process";

const DEFAULT_SDK_SPEC = "@cursor/sdk@1.0.13";

function emit(event) {
  process.stdout.write(`${JSON.stringify(event)}\n`);
}

function readStdinJson() {
  return new Promise((resolve, reject) => {
    let data = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", (chunk) => {
      data += chunk;
    });
    process.stdin.on("end", () => {
      try {
        resolve(JSON.parse(data || "{}"));
      } catch (error) {
        reject(error);
      }
    });
    process.stdin.on("error", reject);
  });
}

function managedPackageDir() {
  if (process.env.CURSOR_SDK_NODE_DIR) {
    return process.env.CURSOR_SDK_NODE_DIR;
  }
  const hermesHome = process.env.HERMES_HOME || path.join(os.homedir(), ".hermes");
  return path.join(hermesHome, "cursor-sdk-node");
}

function ensureManagedSdk() {
  const dir = managedPackageDir();
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
  const pkgJson = path.join(dir, "package.json");
  if (!fs.existsSync(pkgJson)) {
    fs.writeFileSync(
      pkgJson,
      JSON.stringify({ private: true, dependencies: {} }, null, 2),
      { mode: 0o600 },
    );
  }
  const spec = process.env.CURSOR_SDK_PACKAGE_SPEC || DEFAULT_SDK_SPEC;
  const result = spawnSync(
    "npm",
    ["install", "--silent", "--no-audit", "--no-fund", spec],
    { cwd: dir, encoding: "utf8" },
  );
  if (result.status !== 0) {
    const stderr = (result.stderr || result.stdout || "").trim();
    throw new Error(`Failed to install ${spec}: ${stderr || `exit ${result.status}`}`);
  }
  return dir;
}

async function loadCursorSdk() {
  const localRequire = createRequire(import.meta.url);
  try {
    return localRequire("@cursor/sdk");
  } catch (_) {
    let dir = managedPackageDir();
    let managedRequire = createRequire(path.join(dir, "package.json"));
    try {
      return managedRequire("@cursor/sdk");
    } catch (_) {
      dir = ensureManagedSdk();
      managedRequire = createRequire(path.join(dir, "package.json"));
    }
    try {
      return managedRequire("@cursor/sdk");
    } catch (error) {
      const resolved = managedRequire.resolve("@cursor/sdk");
      return import(pathToFileURL(resolved).href);
    }
  }
}

function modelSelection(model) {
  const id = String(model || "").trim();
  if (!id) {
    throw new Error("Cursor SDK model id is required");
  }
  return { id };
}

function textFromAssistantMessage(message) {
  const content = message?.message?.content;
  if (!Array.isArray(content)) {
    return "";
  }
  return content
    .filter((part) => part && part.type === "text" && typeof part.text === "string")
    .map((part) => part.text)
    .join("");
}

function deltaText(update) {
  if (!update || typeof update !== "object") {
    return "";
  }
  const type = String(update.type || update.kind || "");
  if (!/(text|token).*delta|delta.*(text|token)/i.test(type)) {
    return "";
  }
  for (const key of ["text", "delta", "token", "content"]) {
    if (typeof update[key] === "string") {
      return update[key];
    }
  }
  return "";
}

function thinkingText(update) {
  if (!update || typeof update !== "object") {
    return "";
  }
  const type = String(update.type || update.kind || "");
  if (!/thinking/i.test(type)) {
    return "";
  }
  for (const key of ["text", "delta", "token", "content"]) {
    if (typeof update[key] === "string") {
      return update[key];
    }
  }
  return "";
}

async function runPrompt(sdk, request) {
  const { Agent } = sdk;
  const apiKey = request.apiKey || process.env.CURSOR_API_KEY;
  if (!apiKey) {
    throw new Error("CURSOR_API_KEY is required for the Cursor SDK provider");
  }

  const options = {
    apiKey,
    model: modelSelection(request.model),
    name: "Hermes Cursor SDK Provider",
  };
  const runtime = String(request.runtime || "local").toLowerCase();
  if (runtime === "cloud") {
    options.cloud = request.cloud || {};
  } else {
    options.local = { cwd: request.cwd || process.cwd() };
  }

  const agent = await Agent.create(options);
  let streamedText = "";
  try {
    const run = await agent.send(String(request.prompt || ""), {
      onDelta: ({ update }) => {
        const text = deltaText(update);
        if (text) {
          streamedText += text;
          emit({ type: "delta", text });
        }
        const thinking = thinkingText(update);
        if (thinking) {
          emit({ type: "reasoning_delta", text: thinking });
        }
      },
    });

    let assistantText = "";
    try {
      for await (const message of run.stream()) {
        if (message?.type === "assistant") {
          assistantText = textFromAssistantMessage(message) || assistantText;
        } else if (message?.type === "thinking" && typeof message.text === "string") {
          emit({ type: "reasoning_delta", text: message.text });
        } else if (message?.type === "status" && typeof message.message === "string") {
          emit({ type: "status", message: message.message, status: message.status });
        }
      }
    } catch (error) {
      emit({ type: "status", message: `Cursor SDK stream ended: ${error.message}` });
    }

    const result = await run.wait();
    emit({
      type: "final",
      status: result.status,
      result: result.result || assistantText || streamedText,
      model: result.model || options.model,
      runId: result.id,
      agentId: run.agentId,
      durationMs: result.durationMs,
    });
  } finally {
    agent.close();
  }
}

async function listModels(sdk, request) {
  const { Cursor } = sdk;
  const apiKey = request.apiKey || process.env.CURSOR_API_KEY;
  if (!apiKey) {
    throw new Error("CURSOR_API_KEY is required for Cursor model listing");
  }
  const models = await Cursor.models.list({ apiKey });
  emit({ type: "final", models });
}

async function main() {
  const request = await readStdinJson();
  const sdk = await loadCursorSdk();
  if (request.operation === "models") {
    await listModels(sdk, request);
  } else {
    await runPrompt(sdk, request);
  }
}

main().catch((error) => {
  emit({
    type: "error",
    code: error?.code || error?.name || "CursorSdkBridgeError",
    message: error?.message || String(error),
  });
  process.exitCode = 1;
});
