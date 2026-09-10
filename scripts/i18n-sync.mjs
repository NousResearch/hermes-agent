#!/usr/bin/env node
/**
 * i18n sync tool — keeps every locale structurally in lockstep with en.ts.
 *
 * en.ts is the single source of truth. Every other locale is migrated to the
 * partial-locale pattern (`defineLocale({...})`), listing ONLY the keys whose
 * value differs from English; everything else inherits English automatically
 * at runtime. This means:
 *
 *   - When en.ts GAINS a key, every locale inherits it for free — no locale
 *     can silently lag behind (the failure mode that left 79 keys untranslated
 *     across all locales).
 *   - When en.ts DROPS a key, `--check` flags the now-orphaned override so it
 *     cannot linger as dead weight.
 *
 * Usage:
 *   node scripts/i18n-sync.mjs --check   # CI gate: fail on any drift
 *   node scripts/i18n-sync.mjs --fix     # rewrite locales to the synced form
 *
 * Run `--fix` after editing en.ts, then commit. `--check` runs in CI to ensure
 * nobody merges a drifted tree. No third-party dependencies — plain Node, so it
 * runs in CI, locally, and on deployment hosts alike.
 */

import { readFileSync, writeFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const I18N_DIR = join(__dirname, "..", "web", "src", "i18n");

const SKIP = new Set(["en.ts", "types.ts", "define-locale.ts", "index.ts", "context.tsx"]);

// ── Loading locale objects without a TS toolchain ───────────────────────────
// Locale files are simple data modules. We strip the TS-only bits and evaluate
// the object literal so we get the real JS values (no fragile regex parsing of
// nested structures). For `defineLocale({...})` files we evaluate just the
// override argument and merge it onto en ourselves.

function stripTsImports(src) {
  return src.replace(/^\s*import[^\n]*\n/gm, "");
}

/** Extract the balanced `{...}` starting at the first `{` at/after `from`. */
function extractObjectLiteral(src, from) {
  const start = src.indexOf("{", from);
  if (start === -1) return null;
  let depth = 0;
  let quote = null;
  let escape = false;
  for (let i = start; i < src.length; i++) {
    const ch = src[i];
    if (escape) {
      escape = false;
      continue;
    }
    if (quote) {
      if (ch === "\\") escape = true;
      else if (ch === quote) quote = null;
      continue;
    }
    if (ch === '"' || ch === "'" || ch === "`") {
      quote = ch;
      continue;
    }
    if (ch === "{") depth++;
    else if (ch === "}") {
      depth--;
      if (depth === 0) return src.slice(start, i + 1);
    }
  }
  return null;
}

function evalObject(literal) {
  // eslint-disable-next-line no-new-func
  return new Function(`return (${literal});`)();
}

/** Deep-merge `override` onto `base` (override wins), mirroring defineLocale. */
function deepMerge(base, override) {
  if (!isRecord(base) || !isRecord(override)) return override ?? base;
  const out = { ...base };
  for (const [k, v] of Object.entries(override)) {
    if (v === undefined) continue;
    out[k] = isRecord(out[k]) && isRecord(v) ? deepMerge(out[k], v) : v;
  }
  return out;
}

function isRecord(v) {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function loadEn() {
  const src = stripTsImports(readFileSync(join(I18N_DIR, "en.ts"), "utf8"));
  const lit = extractObjectLiteral(src, src.indexOf("="));
  if (!lit) throw new Error("could not parse en.ts object literal");
  return evalObject(lit);
}

function loadLocale(file) {
  const raw = readFileSync(join(I18N_DIR, file), "utf8");
  const exportName = (raw.match(/export\s+const\s+([A-Za-z0-9_]+)\s*[=:]/) || [])[1];
  if (!exportName) throw new Error(`${file}: no 'export const NAME' found`);
  const isPartial = /defineLocale\s*\(/.test(raw);
  const stripped = stripTsImports(raw);
  let resolved;
  let enRef;
  if (isPartial) {
    const argStart = stripped.indexOf("defineLocale(");
    const argLit = extractObjectLiteral(stripped, argStart + "defineLocale(".length - 1);
    if (!argLit) throw new Error(`${file}: could not parse defineLocale(...) argument`);
    const overrides = evalObject(argLit);
    enRef = loadEn();
    resolved = deepMerge(enRef, overrides);
  } else {
    const lit = extractObjectLiteral(stripped, stripped.indexOf("="));
    if (!lit) throw new Error(`${file}: could not parse object literal`);
    resolved = evalObject(lit);
    enRef = loadEn();
  }
  // Preserve any leading comment block above the first import/export.
  const headerLines = [];
  for (const line of raw.split("\n")) {
    const t = line.trim();
    if (t === "" || t.startsWith("//") || t.startsWith("/*") || t.startsWith("*") || t.startsWith("*/")) {
      headerLines.push(line);
    } else break;
  }
  const header = headerLines.join("\n").replace(/\n+$/, "");
  return { exportName, resolved, enRef, header, isPartial };
}

// ── Diffing: build the minimal override tree (keys that differ from en) ─────

function leafEqual(a, b) {
  if (typeof a === "function" && typeof b === "function") return a.toString() === b.toString();
  if (Array.isArray(a) && Array.isArray(b)) return JSON.stringify(a) === JSON.stringify(b);
  return a === b;
}

/**
 * Walk en's structure against the locale's resolved structure. Returns the
 * override subtree containing only leaves that differ from en (orphans in the
 * locale that en no longer has are dropped and reported separately).
 */
function buildOverride(enNode, locNode, path, orphans) {
  if (!isRecord(enNode)) {
    // en leaf — locale should match or inherit; nothing to override here.
    return undefined;
  }
  const out = {};
  let hasKey = false;
  for (const k of Object.keys(enNode)) {
    const childPath = path ? `${path}.${k}` : k;
    const enChild = enNode[k];
    const locChild = isRecord(locNode) ? locNode[k] : undefined;
    if (isRecord(enChild)) {
      const sub = buildOverride(enChild, locChild, childPath, orphans);
      if (sub !== undefined) {
        out[k] = sub;
        hasKey = true;
      }
    } else {
      // en leaf. Include only if the locale's resolved value differs.
      if (locChild !== undefined && !leafEqual(locChild, enChild)) {
        out[k] = locChild;
        hasKey = true;
      }
    }
  }
  // Detect orphan keys present in the locale but absent from en.
  if (isRecord(locNode)) {
    for (const k of Object.keys(locNode)) {
      if (!(k in enNode)) orphans.push(path ? `${path}.${k}` : k);
    }
  }
  return hasKey ? out : undefined;
}

function leafCount(node) {
  if (!isRecord(node)) return 1;
  return Object.values(node).reduce((n, v) => n + leafCount(v), 0);
}

function collectLeafPaths(node, path = "", out = []) {
  if (!isRecord(node)) {
    out.push(path);
    return out;
  }
  for (const k of Object.keys(node)) collectLeafPaths(node[k], path ? `${path}.${k}` : k, out);
  return out;
}

// ── Deterministic serializer for the override tree ──────────────────────────

function serializeValue(v, indent) {
  if (typeof v === "function") return v.toString();
  if (Array.isArray(v)) return JSON.stringify(v);
  if (typeof v === "string") return serializeString(v);
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  if (v === null || v === undefined) return "null";
  return serializeObject(v, indent);
}

function serializeString(s) {
  // Use double quotes; escape backslashes, double quotes, and control chars.
  const escaped = s
    .replace(/\\/g, "\\\\")
    .replace(/"/g, '\\"')
    .replace(/\n/g, "\\n")
    .replace(/\r/g, "\\r")
    .replace(/\t/g, "\\t");
  return `"${escaped}"`;
}

function keyNeedsQuotes(k) {
  return !/^[A-Za-z_$][A-Za-z0-9_$]*$/.test(k);
}

function serializeObject(obj, indent) {
  const entries = Object.entries(obj);
  if (entries.length === 0) return "{}";
  const pad = " ".repeat(indent);
  const inner = " ".repeat(indent + 2);
  const lines = entries.map(([k, v]) => {
    const key = keyNeedsQuotes(k) ? serializeString(k) : k;
    return `${inner}${key}: ${serializeValue(v, indent + 2)},`;
  });
  return `{\n${lines.join("\n")}\n${pad}}`;
}

function renderFile(exportName, override, header) {
  const body =
    override === undefined
      ? "defineLocale({})"
      : `defineLocale(${serializeObject(override, 0)})`;
  const preamble = header ? `${header}\n` : "";
  return (
    `${preamble}import { defineLocale } from "./define-locale";

// Auto-synced by scripts/i18n-sync.mjs — only keys that differ from en.ts are
// listed below; everything else inherits English automatically. After editing
// en.ts, run \`npm run i18n:sync\` to propagate, and \`npm run i18n:check\`
// (wired into CI) guards against drift.
export const ${exportName} = ${body};
`
  );
}

// ── Commands ────────────────────────────────────────────────────────────────

function localeFiles() {
  return readdirSync(I18N_DIR)
    .filter((f) => /^[a-z][a-z0-9-]*\.ts$/.test(f) && !SKIP.has(f))
    .sort();
}

function cmdCheck() {
  const en = loadEn();
  const enPaths = new Set(collectLeafPaths(en));
  let failures = 0;
  for (const file of localeFiles()) {
    const { resolved } = loadLocale(file);
    const locPaths = collectLeafPaths(resolved);
    const missing = [...enPaths].filter((p) => !new Set(locPaths).has(p));
    const extra = locPaths.filter((p) => !enPaths.has(p));
    if (missing.length || extra.length) {
      failures++;
      console.error(`✗ ${file} drifted from en.ts`);
      if (missing.length) console.error(`    missing ${missing.length} key(s): ${missing.slice(0, 5).join(", ")}${missing.length > 5 ? ", …" : ""}`);
      if (extra.length) console.error(`    orphan  ${extra.length} key(s): ${extra.slice(0, 5).join(", ")}${extra.length > 5 ? ", …" : ""}`);
    }
  }
  if (failures) {
    console.error(`\n${failures} locale(s) out of sync. Run \`npm run i18n:sync\` to fix.`);
    process.exit(1);
  }
  console.log(`✓ all ${localeFiles().length} locales in sync with en.ts (${enPaths.size} keys)`);
}

function cmdFix() {
  const en = loadEn();
  let changed = 0;
  for (const file of localeFiles()) {
    const { exportName, resolved, header } = loadLocale(file);
    const orphans = [];
    const override = buildOverride(en, resolved, "", orphans);
    const out = renderFile(exportName, override, header);
    const path = join(I18N_DIR, file);
    const before = readFileSync(path, "utf8");
    if (before !== out) {
      writeFileSync(path, out);
      changed++;
      const kept = override ? leafCount(override) : 0;
      console.log(`↻ ${file} → defineLocale (${kept} translated key(s)${orphans.length ? `, ${orphans.length} orphan(s) dropped` : ""})`);
    } else {
      console.log(`· ${file} already synced`);
    }
  }
  console.log(`\n${changed} file(s) updated. Run \`npm run i18n:check\` to confirm parity.`);
}

const arg = process.argv[2];
if (arg === "--check") cmdCheck();
else if (arg === "--fix") cmdFix();
else {
  console.error("Usage: node scripts/i18n-sync.mjs --check | --fix");
  process.exit(2);
}
