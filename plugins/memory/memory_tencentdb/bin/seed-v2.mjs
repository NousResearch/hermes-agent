#!/usr/bin/env node

// 薄启动器：seed-v2 通过 v2 API 把历史对话灌入 memory-tencentdb gateway。
//
// 优先用预编译产物（生产场景），找不到时 fallback 到 tsx 跑源码（开发场景）。
//
// 构建：npm run build:seed-v2
// 使用：
//   npm run seed-v2 -- --input ./scripts/seed-v2/fixtures/minimal.json
//   node ./bin/seed-v2.mjs --input fixture.json --endpoint http://127.0.0.1:18420

import path from "node:path";
import fs from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { spawnSync } from "node:child_process";

const thisDir = path.dirname(fileURLToPath(import.meta.url));
const distEntry = path.resolve(thisDir, "../scripts/seed-v2/dist/seed-v2.js");
const srcEntry  = path.resolve(thisDir, "../scripts/seed-v2/seed-v2.ts");

if (fs.existsSync(distEntry)) {
  // 预编译产物存在：直接 dynamic import
  await import(pathToFileURL(distEntry).href);
} else if (fs.existsSync(srcEntry)) {
  // 没编译过：fallback 到 tsx（开发期常见）
  const result = spawnSync(
    process.execPath,
    ["--import", "tsx", srcEntry, ...process.argv.slice(2)],
    { stdio: "inherit" },
  );
  process.exit(result.status ?? 1);
} else {
  console.error("❌  neither dist nor source found:");
  console.error("    " + distEntry);
  console.error("    " + srcEntry);
  console.error("    run: npm run build:seed-v2");
  process.exit(1);
}
