#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";

const DEFAULT_FILES = ["feed-signing-public.pem", "feed.json", "feed.json.sig"];

function usage() {
  process.stderr.write(
    [
      "Usage:",
      "  node scripts/generate_checksums_json.mjs --out advisories/checksums.json [--base advisories] [--file feed.json --file feed.json.sig ...]",
      "",
      "Defaults:",
      "  --base <dirname(--out)>",
      `  --file ${DEFAULT_FILES.join(" --file ")}`,
      "",
    ].join("\n"),
  );
}

function parseArgs(argv) {
  const parsed = { files: [] };

  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === "--out") {
      parsed.outPath = argv[++i];
    } else if (token === "--base") {
      parsed.baseDir = argv[++i];
    } else if (token === "--file") {
      parsed.files.push(argv[++i]);
    } else if (token === "-h" || token === "--help") {
      parsed.help = true;
    } else {
      throw new Error(`Unknown argument: ${token}`);
    }
  }

  return parsed;
}

function sha256Hex(buffer) {
  return crypto.createHash("sha256").update(buffer).digest("hex");
}

async function main() {
  const { outPath, baseDir, files, help } = parseArgs(process.argv.slice(2));

  if (help) {
    usage();
    process.exit(0);
  }

  if (!outPath) {
    usage();
    throw new Error("Missing required argument: --out");
  }

  const resolvedBase = path.resolve(baseDir ?? path.dirname(outPath));
  const fileList = files.length > 0 ? files : DEFAULT_FILES;

  const checksums = {};

  for (const relativePath of [...fileList].sort((a, b) => a.localeCompare(b))) {
    const absolutePath = path.resolve(resolvedBase, relativePath);
    const content = await fs.readFile(absolutePath);
    checksums[relativePath] = sha256Hex(content);
  }

  const payload = {
    schema_version: "1.0",
    algorithm: "sha256",
    files: checksums,
  };

  await fs.writeFile(`${outPath}`, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
  process.stdout.write(`Wrote ${outPath}\n`);
}

main().catch((error) => {
  process.stderr.write(`${String(error)}\n`);
  process.exit(1);
});
