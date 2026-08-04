#!/usr/bin/env node

// Thin wrapper: runs the pre-compiled migration CLI entry.
// Build first: npm run build:migrate-sqlite-to-vdb (or pnpm build:migrate-sqlite-to-vdb)

import path from "node:path";
import { fileURLToPath } from "node:url";

const thisDir = path.dirname(fileURLToPath(import.meta.url));
const entryScript = path.resolve(thisDir, "../scripts/migrate-sqlite-to-tcvdb/dist/scripts/migrate-sqlite-to-tcvdb/cli-entry.js");

import(entryScript);
