import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const packageRoot = join(here, "..");

test("package exposes the ACP launcher bin", async () => {
  const packageJson = JSON.parse(
    await readFile(join(packageRoot, "package.json"), "utf8"),
  );
  assert.equal(packageJson.bin["hermes-agent"], "bin/hermes-agent.js");
  assert.equal(packageJson.type, "module");
});

test("launcher contains the managed and uvx fallback paths", async () => {
  const source = await readFile(join(packageRoot, "bin/hermes-agent.js"), "utf8");
  assert.match(source, /hermes-acp/);
  assert.match(source, /--from/);
  assert.match(source, /hermes-agent\[acp\]/);
});
