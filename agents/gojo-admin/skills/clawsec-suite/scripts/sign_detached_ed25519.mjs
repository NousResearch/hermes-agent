#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs/promises";

function usage() {
  process.stderr.write(
    [
      "Usage:",
      "  node scripts/sign_detached_ed25519.mjs --key <private-key.pem> --in <file> --out <file.sig>",
      "",
      "Signs <file> with Ed25519 private key and writes base64 detached signature to <file.sig>.",
      "",
    ].join("\n"),
  );
}

function parseArgs(argv) {
  const parsed = {};

  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === "--key") {
      parsed.keyPath = argv[++i];
    } else if (token === "--in") {
      parsed.inPath = argv[++i];
    } else if (token === "--out") {
      parsed.outPath = argv[++i];
    } else if (token === "-h" || token === "--help") {
      parsed.help = true;
    } else {
      throw new Error(`Unknown argument: ${token}`);
    }
  }

  return parsed;
}

async function main() {
  const { keyPath, inPath, outPath, help } = parseArgs(process.argv.slice(2));

  if (help) {
    usage();
    process.exit(0);
  }

  if (!keyPath || !inPath || !outPath) {
    usage();
    throw new Error("Missing required arguments: --key, --in, --out");
  }

  const privateKeyPem = await fs.readFile(keyPath, "utf8");
  const privateKey = crypto.createPrivateKey(privateKeyPem);
  const data = await fs.readFile(inPath);
  const signature = crypto.sign(null, data, privateKey);
  const signatureBase64 = signature.toString("base64");

  await fs.writeFile(outPath, `${signatureBase64}\n`, "utf8");
  process.stdout.write(`Signed ${inPath} -> ${outPath}\n`);
}

main().catch((error) => {
  process.stderr.write(`${String(error)}\n`);
  process.exit(1);
});
