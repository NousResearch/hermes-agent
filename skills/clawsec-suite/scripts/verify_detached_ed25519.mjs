#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs/promises";

function usage() {
  process.stderr.write(
    [
      "Usage:",
      "  node scripts/verify_detached_ed25519.mjs --key <public-key.pem> --in <file> --sig <file.sig>",
      "",
      "Verifies Ed25519 detached signature against <file>.",
      "Exits 0 on success, 1 on verification failure.",
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
    } else if (token === "--sig") {
      parsed.sigPath = argv[++i];
    } else if (token === "-h" || token === "--help") {
      parsed.help = true;
    } else {
      throw new Error(`Unknown argument: ${token}`);
    }
  }

  return parsed;
}

async function main() {
  const { keyPath, inPath, sigPath, help } = parseArgs(process.argv.slice(2));

  if (help) {
    usage();
    process.exit(0);
  }

  if (!keyPath || !inPath || !sigPath) {
    usage();
    throw new Error("Missing required arguments: --key, --in, --sig");
  }

  const publicKeyPem = await fs.readFile(keyPath, "utf8");
  const publicKey = crypto.createPublicKey(publicKeyPem);
  const data = await fs.readFile(inPath);
  const signatureRaw = await fs.readFile(sigPath, "utf8");
  const signature = Buffer.from(signatureRaw.trim(), "base64");

  const valid = crypto.verify(null, data, publicKey, signature);

  if (valid) {
    process.stdout.write(`Signature valid: ${inPath}\n`);
    process.exit(0);
  } else {
    process.stderr.write(`Signature INVALID: ${inPath}\n`);
    process.exit(1);
  }
}

main().catch((error) => {
  process.stderr.write(`${String(error)}\n`);
  process.exit(1);
});
