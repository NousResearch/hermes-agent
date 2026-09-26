#!/usr/bin/env node
/**
 * Hardcoded user-facing English strings in web/src → build failure.
 *
 * The fa-IR localization sweep (PR #112035) wired every user-facing string
 * on the dashboard pages through the i18n catalogs (web/src/i18n/en.ts +
 * fa.ts, deep-merged per locale). This scanner keeps it that way: it greps
 * web/src for the string idioms the sweep had to fix — raw toasts, confirm
 * dialogs, label/placeholder/aria attribute literals, JSX text nodes,
 * label-map entries, and templated aria labels — and fails when it finds
 * new ones that are not in the allowlist.
 *
 * Scope: web/src only. The desktop app (apps/desktop/src) has its own i18n
 * catalog and a much larger surface; extending there needs its own audit
 * before it can gate.
 *
 * Usage:
 *   node scripts/check_i18n_hardcoded.mjs                   # gate
 *   node scripts/check_i18n_hardcoded.mjs --update-allowlist
 *                                                           # re-baseline
 *
 * The allowlist (scripts/i18n-hardcoded-allowlist.json) records
 * file:line-insensitive findings as `file:::pattern-tag` so unrelated
 * edits above a line do not silently re-arm old findings. Regenerate it
 * ONLY when a newly added string is genuinely not user-facing (a proper
 * noun, a technical token, a brand name); prefer fixing by wiring the
 * string through useI18n().
 */
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const targetDir = join(root, "web", "src");
const allowlistPath = join(root, "scripts", "i18n-hardcoded-allowlist.json");
const update = process.argv.includes("--update-allowlist");

/** Walk web/src recursively. */
function walk(dir, out = []) {
  for (const entry of readdirSorted(dir)) {
    if (entry.isDirectory()) {
      if (entry.name === "i18n" || entry.name === "__tests__") continue; // catalogs & tests
      walk(join(dir, entry.name), out);
    } else if (/\.(tsx?|mjs)$/.test(entry.name) && !/\.test\./.test(entry.name)) {
      out.push(join(dir, entry.name));
    }
  }
  return out;
}

import { readdirSync, statSync } from "node:fs";
function readdirSorted(dir) {
  return readdirSync(dir, { withFileTypes: true }).sort((a, b) =>
    a.name.localeCompare(b.name),
  );
}

/** Lines that can never carry user-facing copy. */
const LINE_NOISE =
  /^\s*(\/\/|\/\*|\*|import\b|export\b.*\bfrom\b|console\.|className=|data-|key=|ref=|href=|src=|type\s+\w|interface\b|declare\b)/;

/** Literals that are identifiers/paths/URLs/commands rather than prose. */
const VALUE_NOISE = new RegExp(
  [
    "^https?://",
    "^/",
    "^\\.{0,2}/",
    "^-", // CLI flags: "-y", "--config"
    "^[a-z]+(-[a-z]+)*$",
    "^[A-Za-z][\\w.-]*\\.(tsx?|png|svg|css|md|json|woff2?)$",
    "^[a-z][\\w.-]*\\.(ts|js|py|sh)$",
    "^#[0-9a-fA-F]{3,8}$",
    "^[a-z-]+:",
    "^@", // npm scopes / decorators
    "^\\\$\\{", // pure interpolation
  ].join("|"),
);

/**
 * Static residue of a template literal: the text outside ${...} jumps.
 * A template whose residue is only punctuation/whitespace around t.*
 * interpolations is translated; one with English residue is not.
 */
function templateResidue(literal) {
  return literal.replace(/^`|`$/g, "").replace(/\$\{[^}]*\}/g, "");
}

/**
 * Pattern classes, each with a tag. `userFacing` heuristics: the literal
 * must look like English prose (contains a space-separated word with a
 * lowercase letter, at least 2 words OR a known UI word) — this keeps
 * brand names ("Telegram"), technical tokens ("owner/repo"), and enum
 * values ("pre_tool_call") out of the report.
 */
const PATTERNS = [
  {
    tag: "toast",
    re: /showToast\((?!"\s*t\.)(`[^`]*`|"[^"]+")/,
    lit: (m) => m[1],
  },
  {
    tag: "confirm",
    re: /(?:window\.)?confirm\((`[^`]*`|"[^"]+")\)/,
    lit: (m) => m[1],
  },
  {
    tag: "attr",
    re:
      /\b(label|placeholder|title|tooltip|aria-label|emptyLabel|confirmLabel|cancelLabel|description|hint)="([^"]{3,})"/,
    lit: (m) => `"${m[2]}"`,
  },
  {
    tag: "attr-tpl",
    re: /\b(aria-label|title|label)=\{(`[^`]*\$\{[^`]*`)\}/,
    lit: (m) => m[2],
  },
  {
    tag: "jsx-text",
    re: /(>|^)\s*([A-Z][a-z]+\s+[a-z][A-Za-z'’,.!?;:()\- ]{8,})\s*(<|$)/,
    lit: (m) => m[2],
  },
  {
    tag: "label-map",
    re: /\b(\w+):\s*"([A-Z][A-Za-z]*(?:\s+[a-z][A-Za-z]*)+)"/,
    lit: (m) => `"${m[2]}"`,
  },
];

function looksUserFacing(literal) {
  // Templates are judged on their static residue: `${t.a}: ${err(e)}` has
  // none (translated), `${n} pastes${redacted}` has English residue.
  const s = (
    literal.startsWith("`") ? templateResidue(literal) : literal
  ).replace(/^[`"']|[`"']$/g, "");
  if (VALUE_NOISE.test(s)) return false;
  const words = s.split(/\s+/).filter(Boolean);
  if (words.length < 2) return false;
  return words.some((w) => /[a-z]/.test(w)) && words.some((w) => /[A-Za-z]/.test(w));
}

const findings = [];
for (const file of walk(targetDir)) {
  const rel = file.slice(root.length + 1).replace(/\\/g, "/");
  const lines = readFileSync(file, "utf8").split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (LINE_NOISE.test(line)) continue;
    for (const p of PATTERNS) {
      const m = line.match(p.re);
      if (!m) continue;
      const literal = p.lit(m);
      if (!literal || !looksUserFacing(literal)) continue;
      findings.push({ file: rel, tag: p.tag, line: i + 1, text: literal });
    }
  }
}

/* ── Allowlist ────────────────────────────────────────────────────── */
let allowlist = new Set();
if (existsSync(allowlistPath)) {
  allowlist = new Set(JSON.parse(readFileSync(allowlistPath, "utf8")));
}

const keyOf = (f) => `${f.file}:::${f.tag}:::${f.text}`;
const novel = findings.filter((f) => !allowlist.has(keyOf(f)));

if (update) {
  const keep = [...new Set(findings.map(keyOf))].sort();
  writeFileSync(allowlistPath, JSON.stringify(keep, null, 2) + "\n");
  console.log(
    `allowlist updated: ${keep.length} entr(y|ies) -> ${allowlistPath}`,
  );
  process.exit(0);
}

if (novel.length === 0) {
  console.log(
    `ok: no hardcoded user-facing strings in web/src (${findings.length} allowlisted, ${allowlist.size} allowlist entries)`,
  );
  process.exit(0);
}

console.error(
  `FAIL: ${novel.length} hardcoded user-facing string(s) in web/src.\n` +
    `Wire them through useI18n() (catalogs: web/src/i18n/en.ts + fa.ts + types.ts).\n` +
    `If a finding is genuinely not user-facing, re-baseline with:\n` +
    `  node scripts/check_i18n_hardcoded.mjs --update-allowlist\n`,
);
for (const f of novel) {
  console.error(`  ${f.file}:${f.line} [${f.tag}] ${f.text}`);
}
process.exit(1);
