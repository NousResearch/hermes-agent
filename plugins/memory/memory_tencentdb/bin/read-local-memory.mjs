#!/usr/bin/env node

// 薄启动器：加载预编译好的本地 Memory 数据查询脚本。
// 构建：npm run build:read-local-memory
// 使用：npm run read-local-memory -- [参数]  或  node ./bin/read-local-memory.mjs [参数]

import path from "node:path";
import { fileURLToPath } from "node:url";

import fs from "node:fs";

const thisDir = path.dirname(fileURLToPath(import.meta.url));
const entryScript = path.resolve(thisDir, "../scripts/read-local-memory/dist/read-local-memory.js");

if (!fs.existsSync(entryScript)) {
  console.error("❌  预编译产物不存在: " + entryScript);
  console.error("   请先执行: npm run build:read-local-memory");
  process.exit(1);
}

import(entryScript);
