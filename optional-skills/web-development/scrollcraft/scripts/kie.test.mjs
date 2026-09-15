import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

const HERE = path.dirname(fileURLToPath(import.meta.url));

test("key resolution continues past a nearer env file without the key", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "scrollcraft-kie-"));
  const project = path.join(root, "project");
  const nested = path.join(project, "build", "nested");
  fs.mkdirSync(nested, { recursive: true });
  fs.writeFileSync(path.join(project, ".env"), "KIE_AI_API_KEY=synthetic-parent\n");
  fs.writeFileSync(path.join(project, "build", ".env"), "UNRELATED=value\n");

  const result = spawnSync(process.execPath, [path.join(HERE, "kie.mjs"), "invalid-command"], {
    cwd: nested,
    encoding: "utf8",
    env: { ...process.env, KIE_AI_API_KEY: "" },
    timeout: 5_000,
  });

  assert.equal(result.status, 1);
  assert.match(result.stderr, /scrollcraft asset generator/);
  assert.doesNotMatch(result.stderr, /KIE_AI_API_KEY (?:not found|not set)/);
});

test("key resolution continues past an empty nearer assignment", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "scrollcraft-kie-empty-"));
  const project = path.join(root, "project");
  const nested = path.join(project, "build", "nested");
  fs.mkdirSync(nested, { recursive: true });
  fs.writeFileSync(path.join(project, ".env"), "KIE_AI_API_KEY=synthetic-parent\n");
  fs.writeFileSync(path.join(project, "build", ".env"), 'KIE_AI_API_KEY=""\n');

  const result = spawnSync(process.execPath, [path.join(HERE, "kie.mjs"), "invalid-command"], {
    cwd: nested,
    encoding: "utf8",
    env: { ...process.env, KIE_AI_API_KEY: "" },
    timeout: 5_000,
  });

  assert.equal(result.status, 1);
  assert.match(result.stderr, /scrollcraft asset generator/);
  assert.doesNotMatch(result.stderr, /KIE_AI_API_KEY (?:not found|not set)/);
});
