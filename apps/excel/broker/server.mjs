import http from "node:http";
import { readFile, writeFile, mkdir, readdir, stat, unlink } from "node:fs/promises";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import zlib from "node:zlib";
import { translateMatrixFormulas, anchorFromAddress } from "./formula-rebase.mjs";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const uploadsDir = path.join(root, "uploads");
const port = Number(process.env.PORT || 8787);
const llmBaseUrl = (process.env.HERMES_EXCEL_LLM_BASE_URL || "http://127.0.0.1:8642/v1").replace(/\/$/, "");
const llmModel = process.env.HERMES_EXCEL_LLM_MODEL || "hermes-agent";
const llmApiKey = process.env.HERMES_EXCEL_LLM_API_KEY || readHermesApiServerKey() || "local";
const llmTimeoutMs = Number(process.env.HERMES_EXCEL_LLM_TIMEOUT_MS || 180000);
const llmMaxTokens = Number(process.env.HERMES_EXCEL_LLM_MAX_TOKENS || 8000);
const doclingBaseUrl = (process.env.HERMES_EXCEL_DOCLING_URL || "http://127.0.0.1:8200").replace(/\/$/, "");
const doclingTimeoutMs = Number(process.env.HERMES_EXCEL_DOCLING_TIMEOUT_MS || 300000);
const maxExtractedCharsPerFile = Number(process.env.HERMES_EXCEL_MAX_EXTRACTED_CHARS_PER_FILE || 32000);
const maxExtractedCharsTotal = Number(process.env.HERMES_EXCEL_MAX_EXTRACTED_CHARS_TOTAL || 96000);
const execFileAsync = promisify(execFile);
const iconPng = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAIklEQVR42mP8z8Dwn4ECwESJ5lEDRg0YNWDUgFEDBgAxygQh6tKGxAAAAABJRU5ErkJggg==",
  "base64",
);

const mime = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".xml": "application/xml; charset=utf-8",
  ".png": "image/png",
  ".json": "application/json; charset=utf-8",
};

function readHermesApiServerKey() {
  const configPath = path.join(process.env.LOCALAPPDATA || "", "hermes", "config.yaml");
  if (!configPath || !existsSync(configPath)) return "";

  try {
    const text = readFileSync(configPath, "utf8");
    const lines = text.split(/\r?\n/);
    const matches = [];
    for (let index = 0; index < lines.length; index += 1) {
      const section = lines[index].match(/^(\s*)api_server:\s*$/);
      if (!section) continue;
      const sectionIndent = section[1].length;
      for (let next = index + 1; next < lines.length; next += 1) {
        const line = lines[next];
        if (!line.trim() || line.trim().startsWith("#")) continue;
        const indent = line.match(/^(\s*)/)?.[1].length || 0;
        if (indent <= sectionIndent) break;
        const key = line.match(/^\s*(?:key|api_key):\s*['"]?([^'"\r\n#]+)['"]?/);
        if (key?.[1]) matches.push(key[1].trim());
      }
    }
    if (matches.length) return matches.at(-1);
  } catch {}

  return "";
}

function send(res, status, body, contentType = "application/json; charset=utf-8") {
  res.writeHead(status, {
    "content-type": contentType,
    "access-control-allow-origin": "*",
    "access-control-allow-methods": "GET,POST,OPTIONS",
    "access-control-allow-headers": "content-type",
  });
  res.end(typeof body === "string" ? body : JSON.stringify(body, null, 2));
}

async function healthStatus() {
  const status = {
    ok: true,
    service: "hermes-excel-bridge",
    port,
    llmBaseUrl,
    llmModel,
    doclingBaseUrl,
    time: new Date().toISOString(),
    hermes: { ok: false },
    docling: { ok: false },
  };

  try {
    const response = await fetch(`${llmBaseUrl}/models`, {
      headers: { authorization: `Bearer ${llmApiKey}` },
      signal: AbortSignal.timeout(3000),
    });
    status.hermes = { ok: response.ok, status: response.status };
  } catch (error) {
    status.hermes = { ok: false, error: error.message };
  }

  try {
    const response = await fetch(`${doclingBaseUrl}/healthz`, { signal: AbortSignal.timeout(3000) });
    status.docling = { ok: response.ok, status: response.status };
    if (response.ok) status.docling.detail = await response.json().catch(() => null);
  } catch (error) {
    status.docling = { ok: false, error: error.message };
  }

  status.ok = status.hermes.ok && status.docling.ok;
  return status;
}

async function readJson(req) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > 50 * 1024 * 1024) throw new Error("request too large");
    chunks.push(chunk);
  }
  return JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
}

function compactSelection(selection) {
  const values = Array.isArray(selection?.values) ? selection.values : [];
  return {
    address: selection?.address || "",
    rowCount: selection?.rowCount || values.length,
    columnCount: selection?.columnCount || values[0]?.length || 0,
    values: values.slice(0, 100).map((row) => row.slice(0, 16)),
    formulas: Array.isArray(selection?.formulas)
      ? selection.formulas.slice(0, 100).map((row) => row.slice(0, 16))
      : [],
  };
}

function truncateText(text, maxChars = maxExtractedCharsPerFile) {
  const value = String(text || "").replace(/\r\n/g, "\n").trim();
  if (value.length <= maxChars) return value;
  return `${value.slice(0, maxChars)}\n\n[truncated ${value.length - maxChars} characters]`;
}

function windowsPathToWsl(filePath) {
  const resolved = path.resolve(filePath);
  const driveMatch = resolved.match(/^([a-zA-Z]):\\(.*)$/);
  if (!driveMatch) return resolved.replace(/\\/g, "/");
  const drive = driveMatch[1].toLowerCase();
  const rest = driveMatch[2].replace(/\\/g, "/");
  return `/mnt/${drive}/${rest}`;
}

function extensionOf(fileName) {
  return path.extname(String(fileName || "")).toLowerCase();
}

function parseDelimitedText(text, delimiter) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;
  const input = String(text || "");

  for (let index = 0; index < input.length; index += 1) {
    const char = input[index];
    const next = input[index + 1];
    if (quoted) {
      if (char === '"' && next === '"') {
        cell += '"';
        index += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        cell += char;
      }
      continue;
    }
    if (char === '"') {
      quoted = true;
      continue;
    }
    if (char === delimiter) {
      row.push(cell);
      cell = "";
      continue;
    }
    if (char === "\n") {
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
      continue;
    }
    if (char !== "\r") cell += char;
  }
  row.push(cell);
  rows.push(row);
  return rows.filter((items) => items.some((item) => String(item || "").trim()));
}

function markdownTableFromRows(rows, limit = 80) {
  const trimmed = rows.slice(0, limit).map((row) => row.slice(0, 20));
  if (!trimmed.length) return "";
  const width = Math.max(...trimmed.map((row) => row.length));
  const padded = trimmed.map((row) => {
    const copy = [...row];
    while (copy.length < width) copy.push("");
    return copy.map((cell) => String(cell ?? "").replace(/\|/g, "\\|").replace(/\s+/g, " ").trim());
  });
  const header = padded[0];
  const body = padded.slice(1);
  return [
    `| ${header.join(" | ")} |`,
    `| ${header.map(() => "---").join(" | ")} |`,
    ...body.map((row) => `| ${row.join(" | ")} |`),
    rows.length > limit ? `\n[${rows.length - limit} more row(s) omitted]` : "",
  ]
    .filter(Boolean)
    .join("\n");
}

function extractSimpleText(file, bytes) {
  const ext = extensionOf(file.name);
  if (![".txt", ".md", ".csv", ".tsv"].includes(ext)) return null;
  const text = bytes.toString("utf8");
  if (ext === ".csv" || ext === ".tsv") {
    const rows = parseDelimitedText(text, ext === ".tsv" ? "\t" : ",");
    return truncateText(markdownTableFromRows(rows) || text);
  }
  return truncateText(text);
}

function decodePdfLiteral(value) {
  return String(value || "")
    .replace(/\\n/g, "\n")
    .replace(/\\r/g, "\n")
    .replace(/\\t/g, "\t")
    .replace(/\\\(/g, "(")
    .replace(/\\\)/g, ")")
    .replace(/\\\\/g, "\\");
}

function decodeAscii85(input) {
  let text = String(input || "").trim();
  if (text.startsWith("<~")) text = text.slice(2);
  const terminator = text.indexOf("~>");
  if (terminator >= 0) text = text.slice(0, terminator);

  const out = [];
  let group = [];
  for (const char of text) {
    if (/\s/.test(char)) continue;
    if (char === "z" && group.length === 0) {
      out.push(0, 0, 0, 0);
      continue;
    }
    const code = char.charCodeAt(0);
    if (code < 33 || code > 117) continue;
    group.push(code - 33);
    if (group.length === 5) {
      let value = 0;
      for (const item of group) value = value * 85 + item;
      out.push((value >>> 24) & 255, (value >>> 16) & 255, (value >>> 8) & 255, value & 255);
      group = [];
    }
  }

  if (group.length) {
    const length = group.length;
    while (group.length < 5) group.push(84);
    let value = 0;
    for (const item of group) value = value * 85 + item;
    const bytes = [(value >>> 24) & 255, (value >>> 16) & 255, (value >>> 8) & 255, value & 255];
    out.push(...bytes.slice(0, length - 1));
  }

  return Buffer.from(out);
}

function decodePdfStream(filters, streamText) {
  let data = Buffer.from(streamText, "latin1");
  for (const filter of filters) {
    if (filter === "ASCII85Decode" || filter === "A85") data = decodeAscii85(data.toString("latin1"));
    if (filter === "FlateDecode" || filter === "Fl") data = zlib.inflateSync(data);
  }
  return data.toString("latin1");
}

function extractPdfTextOperators(text) {
  const chunks = [];
  for (const match of text.matchAll(/\((?:\\.|[^\\)]){1,2000}\)\s*Tj/g)) {
    chunks.push(decodePdfLiteral(match[0].replace(/\)\s*Tj$/, "").slice(1)));
  }
  for (const match of text.matchAll(/\[((?:\s*\((?:\\.|[^\\)]){1,2000}\)\s*)+)\]\s*TJ/g)) {
    const inner = match[1];
    const parts = [...inner.matchAll(/\((?:\\.|[^\\)]){1,2000}\)/g)].map((part) =>
      decodePdfLiteral(part[0].slice(1, -1)),
    );
    if (parts.length) chunks.push(parts.join(""));
  }
  return chunks;
}

function extractBasicPdfText(bytes) {
  const text = bytes.toString("latin1");
  const chunks = extractPdfTextOperators(text);

  for (const match of text.matchAll(/<<(?:[\s\S]{0,2000}?\/Filter\s+(?:\[\s*)?([^\]\r\n<>]+)(?:\s*\])?[\s\S]{0,2000}?)>>\s*stream\r?\n([\s\S]*?)\r?\n?endstream/g)) {
    const filters = [...match[1].matchAll(/\/([A-Za-z0-9]+)/g)].map((filter) => filter[1]);
    if (!filters.length) continue;
    try {
      chunks.push(...extractPdfTextOperators(decodePdfStream(filters, match[2])));
    } catch {}
  }

  const extracted = chunks
    .join("\n")
    .replace(/[^\S\r\n]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
  return extracted.length > 40 ? truncateText(extracted) : null;
}

async function fetchJsonWithTimeout(url, options = {}, timeoutMs = 30000) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text().catch(() => "")}`);
    return await response.json();
  } finally {
    clearTimeout(timeout);
  }
}

async function readWslText(wslPath) {
  const { stdout } = await execFileAsync("wsl.exe", ["-d", "Ubuntu-24.04", "--", "cat", wslPath], {
    maxBuffer: maxExtractedCharsPerFile * 4,
    windowsHide: true,
  });
  return stdout;
}

async function extractWithDocling(filePath) {
  const sourcePath = windowsPathToWsl(filePath);
  const created = await fetchJsonWithTimeout(
    `${doclingBaseUrl}/jobs`,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ source_path: sourcePath, priority: 4 }),
    },
    30000,
  );
  const jobId = created.id;
  if (!jobId) throw new Error("Docling did not return a job id");

  const started = Date.now();
  let latest = created;
  while (Date.now() - started < doclingTimeoutMs) {
    await new Promise((resolve) => setTimeout(resolve, 1500));
    latest = await fetchJsonWithTimeout(`${doclingBaseUrl}/jobs/${jobId}`, {}, 30000);
    if (latest.status === "done") {
      const result = await fetchJsonWithTimeout(`${doclingBaseUrl}/jobs/${jobId}/result`, {}, 30000);
      if (result.markdown || result.text) return truncateText(result.markdown || result.text);
      const mdPath = result.md_path || result.result_md_path || result.markdown_path;
      if (!mdPath) throw new Error(`Docling job ${jobId} finished without markdown path`);
      return truncateText(await readWslText(mdPath));
    }
    if (["failed", "review"].includes(latest.status)) {
      throw new Error(`Docling job ${jobId} ${latest.status}: ${latest.error || "no details"}`);
    }
  }
  throw new Error(`Docling timed out after ${Math.round(doclingTimeoutMs / 1000)}s`);
}

async function prepareAttachedFiles(files = []) {
  await mkdir(uploadsDir, { recursive: true });
  const contexts = [];
  let totalChars = 0;

  for (const file of files) {
    const safeName = String(file.name || "upload.bin").replace(/[^a-zA-Z0-9._-]/g, "_").slice(-100);
    const out = path.join(uploadsDir, `${Date.now()}-${safeName}`);
    const bytes = Buffer.from(file.base64 || "", "base64");
    await writeFile(out, bytes);

    const context = {
      name: file.name || safeName,
      type: file.type || "application/octet-stream",
      size: file.size || bytes.length,
      saved_path: out,
      extraction_status: "unread",
      extracted_text: "",
    };

    const simple = extractSimpleText(file, bytes);
    if (simple !== null) {
      context.extraction_status = "parsed";
      context.extraction_method = "local-text";
      context.extracted_text = simple;
    } else {
      try {
        context.extracted_text = await extractWithDocling(out);
        context.extraction_status = "parsed";
        context.extraction_method = "docling";
      } catch (error) {
        const basicPdfText = extensionOf(file.name) === ".pdf" ? extractBasicPdfText(bytes) : null;
        if (basicPdfText) {
          context.extracted_text = basicPdfText;
          context.extraction_status = "parsed";
          context.extraction_method = "basic-pdf";
          context.extraction_error = `Docling unavailable, used basic PDF text extraction: ${error.message}`;
        } else {
          context.extraction_status = "failed";
          context.extraction_error = error.message;
        }
      }
    }

    if (context.extracted_text) {
      const remaining = Math.max(0, maxExtractedCharsTotal - totalChars);
      context.extracted_text = truncateText(context.extracted_text, remaining);
      totalChars += context.extracted_text.length;
    }
    contexts.push(context);
  }

  return contexts;
}

function repairJsonText(text, dropMismatched) {
  let out = "";
  const stack = [];
  let inString = false;
  let escaped = false;
  for (const char of text) {
    if (out && !stack.length) break; // root value closed; ignore trailing text
    if (inString) {
      out += char;
      if (escaped) escaped = false;
      else if (char === "\\") escaped = true;
      else if (char === '"') inString = false;
      continue;
    }
    if (char === '"') {
      inString = true;
      out += char;
      continue;
    }
    if (char === "{" || char === "[") {
      stack.push(char);
      out += char;
      continue;
    }
    if (char === "}" || char === "]") {
      const opener = char === "}" ? "{" : "[";
      if (stack.length && stack[stack.length - 1] === opener) {
        stack.pop();
        out += char;
      } else if (!dropMismatched && stack.includes(opener)) {
        while (stack.length && stack[stack.length - 1] !== opener) {
          out += stack.pop() === "{" ? "}" : "]";
        }
        stack.pop();
        out += char;
      }
      // Otherwise the closer is stray; drop it.
      continue;
    }
    out += char;
  }
  if (inString) out += '"';
  while (stack.length) out += stack.pop() === "{" ? "}" : "]";
  out = out.replace(/,\s*([}\]])/g, "$1");
  try {
    return JSON.parse(out);
  } catch {
    return null;
  }
}

function extractJsonObject(text) {
  const trimmed = String(text || "").trim();
  try {
    return JSON.parse(trimmed);
  } catch {}

  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenced) {
    try {
      return JSON.parse(fenced[1].trim());
    } catch {}
  }

  const start = trimmed.indexOf("{");
  const end = trimmed.lastIndexOf("}");
  if (start >= 0 && end > start) {
    try {
      return JSON.parse(trimmed.slice(start, end + 1));
    } catch {}
  }

  // Local models sometimes emit one stray or missing bracket; repair before giving up.
  if (start >= 0) {
    const candidate = trimmed.slice(start);
    return repairJsonText(candidate, true) || repairJsonText(candidate, false);
  }
  return null;
}

function normalizeMatrix(matrix) {
  if (!Array.isArray(matrix)) return null;
  const rows = matrix
    .filter((row) => Array.isArray(row))
    .slice(0, 200)
    .map((row) =>
      row.slice(0, 30).map((cell) =>
        cell === null || ["string", "number", "boolean"].includes(typeof cell) ? cell : String(cell),
      ),
    );
  const width = Math.max(0, ...rows.map((row) => row.length));
  for (const row of rows) {
    while (row.length < width) row.push("");
  }
  return rows.length ? rows : null;
}

// Re-read of a just-written range, scanned for the two failure signatures we
// have actually seen: Excel error values, and formula columns that came out
// all-zero (the symptom of formulas anchored at the wrong cell).
function scanWrittenCells(values) {
  if (!Array.isArray(values)) return { errors: [], zeroFormulaColumns: [], ok: true };
  const errorRegex = /^#(REF|DIV\/0|VALUE|NAME\?|N\/A|NULL|NUM)!?/i;
  const errors = [];
  const nonEmpty = {};
  const zeros = {};
  for (let r = 0; r < values.length; r += 1) {
    const row = values[r];
    if (!Array.isArray(row)) continue;
    for (let c = 0; c < row.length; c += 1) {
      const cell = row[c];
      const str = cell === null || cell === undefined ? "" : String(cell);
      if (str && errorRegex.test(str)) errors.push({ r, c, value: str });
      if (r === 0 || str === "") continue; // header excluded from the all-zero heuristic
      nonEmpty[c] = (nonEmpty[c] || 0) + 1;
      if (cell === 0 || str === "0") zeros[c] = (zeros[c] || 0) + 1;
    }
  }
  const zeroFormulaColumns = [];
  for (const key of Object.keys(nonEmpty)) {
    const c = Number(key);
    if (nonEmpty[c] >= 2 && nonEmpty[c] === (zeros[c] || 0)) zeroFormulaColumns.push(c);
  }
  zeroFormulaColumns.sort((a, b) => a - b);
  return { errors, zeroFormulaColumns, ok: errors.length === 0 && zeroFormulaColumns.length === 0 };
}

const exportsDir = path.join(root, "exports");

function safeExportName(name) {
  const sanitized = String(name || "hermes-export")
    .replace(/[^A-Za-z0-9._-]/g, "")
    .replace(/^\.+/, "");
  const base = sanitized || "hermes-export";
  return base.toLowerCase().endsWith(".csv") ? base : `${base}.csv`;
}

function matrixToCsv(values) {
  if (!Array.isArray(values)) return "";
  return values
    .map((row) =>
      (Array.isArray(row) ? row : [])
        .map((cell) => {
          const str = cell === null || cell === undefined ? "" : String(cell);
          return /[",\r\n]/.test(str) ? `"${str.replace(/"/g, '""')}"` : str;
        })
        .join(","),
    )
    .join("\r\n");
}

async function handleExport(req, res) {
  const body = await readJson(req);
  if (!body.values) return send(res, 400, { error: "Missing values" });
  await mkdir(exportsDir, { recursive: true });
  const filePath = path.join(exportsDir, safeExportName(body.name));
  await writeFile(filePath, matrixToCsv(normalizeMatrix(body.values) || []), "utf8");
  return send(res, 200, { ok: true, path: filePath });
}

async function pruneUploads(maxAgeMs = Number(process.env.HERMES_EXCEL_UPLOADS_TTL_MS || 7 * 24 * 60 * 60 * 1000)) {
  try {
    if (!existsSync(uploadsDir)) return 0;
    const entries = await readdir(uploadsDir);
    const now = Date.now();
    let count = 0;
    for (const entry of entries) {
      try {
        const info = await stat(path.join(uploadsDir, entry));
        if (info.isFile() && now - info.mtimeMs > maxAgeMs) {
          await unlink(path.join(uploadsDir, entry));
          count += 1;
        }
      } catch {}
    }
    return count;
  } catch {
    return 0;
  }
}

function normalizeAction(action) {
  if (!action || typeof action !== "object") return null;
  const type = String(action.type || "");
  if (type === "write_cells") {
    const values = normalizeMatrix(action.values || action.table);
    if (!values) return null;
    return {
      type,
      // Empty when omitted; normalizeActions resolves it to the selection or A1.
      start_cell: String(action.start_cell || action.startCell || ""),
      values,
      allow_overwrite: action.allow_overwrite !== false,
      auto_format: typeof action.auto_format === "boolean" ? action.auto_format : undefined,
    };
  }
  if (type === "create_sheet") {
    const values = normalizeMatrix(action.values || action.table);
    if (!values) return null;
    return {
      type,
      name: action.name ? String(action.name).slice(0, 31) : "Hermes Output",
      values,
    };
  }
  if (type === "format_cells") {
    return {
      type,
      range: String(action.range || "A1"),
      style: Array.isArray(action.style)
        ? action.style.map(String)
        : action.style
          ? String(action.style)
          : undefined,
      bold: typeof action.bold === "boolean" ? action.bold : undefined,
      italic: typeof action.italic === "boolean" ? action.italic : undefined,
      underline: typeof action.underline === "boolean" ? action.underline : undefined,
      font_color: action.font_color ? String(action.font_color) : undefined,
      font_size: Number.isFinite(Number(action.font_size)) ? Number(action.font_size) : undefined,
      font_name: action.font_name ? String(action.font_name) : undefined,
      fill_color: action.fill_color ? String(action.fill_color) : undefined,
      number_format: action.number_format ? String(action.number_format) : undefined,
      number_format_dp: Number.isFinite(Number(action.number_format_dp)) ? Number(action.number_format_dp) : undefined,
      currency_symbol: action.currency_symbol ? String(action.currency_symbol) : undefined,
      horizontal_alignment: action.horizontal_alignment ? String(action.horizontal_alignment) : undefined,
      vertical_alignment: action.vertical_alignment ? String(action.vertical_alignment) : undefined,
      wrap_text: typeof action.wrap_text === "boolean" ? action.wrap_text : undefined,
      column_width: Number.isFinite(Number(action.column_width)) ? Number(action.column_width) : undefined,
      row_height: Number.isFinite(Number(action.row_height)) ? Number(action.row_height) : undefined,
      auto_fit: typeof action.auto_fit === "boolean" ? action.auto_fit : undefined,
      borders: action.borders ? String(action.borders) : undefined,
      border_top: action.border_top ? String(action.border_top) : undefined,
      border_bottom: action.border_bottom ? String(action.border_bottom) : undefined,
      border_left: action.border_left ? String(action.border_left) : undefined,
      border_right: action.border_right ? String(action.border_right) : undefined,
      border_color: action.border_color ? String(action.border_color) : undefined,
    };
  }
  if (type === "execute_office_js") {
    const code = String(action.code || "").trim();
    if (!code) return null;
    return {
      type,
      explanation: String(action.explanation || "workbook edit").slice(0, 200),
      code,
    };
  }
  if (type === "read_range") {
    if (!action.range) return null;
    return {
      type: "read_range",
      range: String(action.range).slice(0, 80),
      reason: String(action.reason || "").slice(0, 200),
    };
  }
  if (type === "conditional_format") {
    const range = String(action.range || "").trim();
    if (!range) return null;
    // Map the many ways a model writes an operator onto Office.js's enum names.
    const operatorMap = {
      lessthan: "lessThan", "<": "lessThan", below: "lessThan", under: "lessThan",
      lessthanorequal: "lessThanOrEqual", "<=": "lessThanOrEqual",
      greaterthan: "greaterThan", ">": "greaterThan", above: "greaterThan", over: "greaterThan",
      greaterthanorequal: "greaterThanOrEqual", ">=": "greaterThanOrEqual",
      equalto: "equalTo", equal: "equalTo", "=": "equalTo", "==": "equalTo",
      notequalto: "notEqualTo", "!=": "notEqualTo", "<>": "notEqualTo",
      between: "between", notbetween: "notBetween",
    };
    const opKey = String(action.operator || "lessThan").toLowerCase().replace(/[\s_-]/g, "");
    const operator = operatorMap[opKey] || "lessThan";
    const rawValue = action.value !== undefined ? action.value : action.threshold;
    const value = typeof rawValue === "number" || typeof rawValue === "string" ? rawValue : 0;
    const out = {
      type,
      range: range.slice(0, 80),
      operator,
      value,
      fill_color: action.fill_color ? String(action.fill_color).slice(0, 9) : "#FFC7CE",
      font_color: action.font_color ? String(action.font_color).slice(0, 9) : "#9C0006",
    };
    if ((operator === "between" || operator === "notBetween") && action.value2 !== undefined && action.value2 !== null) {
      out.value2 = typeof action.value2 === "number" || typeof action.value2 === "string" ? action.value2 : 0;
    }
    return out;
  }
  return null;
}

function legacyWriteToActions(write, body) {
  if (!write || typeof write !== "object") return [];
  const mode = String(write.mode || "none");
  const values = normalizeMatrix(write.values || write.table);
  if (!values || mode === "none") return [];
  if (mode === "new_sheet") {
    return [{ type: "create_sheet", name: write.name || "Hermes Output", values }];
  }
  if (mode === "selection") {
    return [{ type: "write_cells", start_cell: body.selection?.address || "A1", values, allow_overwrite: true }];
  }
  return [];
}

function normalizeActions(parsed, body) {
  const actions = Array.isArray(parsed?.actions)
    ? parsed.actions.map(normalizeAction).filter(Boolean)
    : legacyWriteToActions(parsed?.write, body);
  // Resolve the write anchor (selection when the model omitted start_cell) and
  // rebase A1-authored formulas to it. create_sheet always lands at A1.
  for (const action of actions) {
    if (action.type === "write_cells") {
      action.start_cell = action.start_cell || body?.selection?.address || "A1";
      const { rowOffset, colOffset } = anchorFromAddress(action.start_cell);
      action.values = translateMatrixFormulas(action.values, rowOffset, colOffset);
    }
  }
  return actions;
}

function extractedField(text, label) {
  const escaped = label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = String(text || "").match(new RegExp(`${escaped}:\\s*([^\\n]+)`, "i"));
  return match?.[1]?.trim() || "";
}

function followingValue(text, label) {
  const lines = String(text || "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  const index = lines.findIndex((line) => line.toLowerCase() === label.toLowerCase());
  return index >= 0 ? lines[index + 1] || "" : "";
}

function parseMoney(value) {
  const raw = String(value || "");
  if (!raw.trim()) return "";

  let text = raw.replace(/[$,]/g, "").trim();
  let negate = false;
  if (/^\(.*\)$/.test(text)) {
    negate = true;
    text = text.replace(/^\(|\)$/g, "");
  }

  text = text.replace(/[^0-9.-]/g, "");
  if (!text || text === "-" || text === "." || text === "-.") return "";

  const number = Number(text);
  if (!Number.isFinite(number)) return value;

  return negate ? -number : number;
}

function moneyValueForLabel(text, label) {
  const direct = followingValue(text, label);
  if (direct) return direct;
  const escaped = label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "\\s*");
  const match = String(text || "").match(new RegExp(`${escaped}[^\\n$()\\d-]*([($-]?[\\d,]+\\.\\d{2}\\)?)`, "i"));
  return match?.[1]?.trim() || "";
}

function statementSummaryRows(file) {
  const text = file.extracted_text || "";
  const summary = [
    ["Field", "Value"],
    ["Source file", file.name],
    ["Parser", file.extraction_method || "parsed"],
    ["Bank", text.match(/^(.*Bank.*)$/im)?.[1]?.trim() || ""],
    ["Account holder", extractedField(text, "Account Holder")],
    ["Statement period", extractedField(text, "Statement Period")],
    ["Statement date", extractedField(text, "Statement Date")],
    ["Account number", extractedField(text, "Account Number")],
    ["Beginning balance", parseMoney(moneyValueForLabel(text, "Beginning Balance"))],
    ["Deposits / credits", parseMoney(moneyValueForLabel(text, "Total Deposits / Credits"))],
    ["Withdrawals / debits", parseMoney(moneyValueForLabel(text, "Total Withdrawals / Debits"))],
    ["Ending balance", parseMoney(moneyValueForLabel(text, "Ending Balance"))],
  ].filter((row, index) => index === 0 || row[1] !== "");
  return summary.length > 1 ? summary : null;
}

function statementTransactionRows(file) {
  const lines = String(file.extracted_text || "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  const rows = [["Date", "Description", "Debit", "Credit", "Balance"]];
  for (let index = 0; index < lines.length; index += 1) {
    if (!/^\d{2}\/\d{2}\/\d{4}$/.test(lines[index])) continue;
    const description = lines[index + 1] || "";
    const money = [];
    for (let next = index + 2; next < Math.min(lines.length, index + 8); next += 1) {
      if (/^\$?\(?[\d,]+\.\d{2}\)?$/.test(lines[next])) money.push(lines[next]);
      if (money.length >= 2) break;
    }
    if (!description || !money.length) continue;
    const amount = parseMoney(money[0]);
    const balance = money.length > 1 ? parseMoney(money[1]) : "";
    const isCredit = /\b(credit|deposit|interest|wire from|inv pmt)\b/i.test(description);
    rows.push([lines[index], description, isCredit ? "" : amount, isCredit ? amount : "", balance]);
  }
  return rows.length > 1 ? rows.slice(0, 201) : null;
}

function buildAccountingDataSheet(file) {
  const summary = statementSummaryRows(file);
  const transactions = statementTransactionRows(file);
  const rows = [];
  if (summary) rows.push(["Bank Statement Summary", "", "", "", ""], ...summary, ["", "", "", "", ""]);
  if (transactions) rows.push(["Transactions", "", "", "", ""], ...transactions);
  if (rows.length) return rows;
  return [
    ["Source file", "Extraction", "Preview"],
    [file.name, file.extraction_method || "parsed", String(file.extracted_text || "").replace(/\s+/g, " ").slice(0, 2000)],
  ];
}

function readableFileAnswer(body, reason = null) {
  const readable = (body.files || []).filter((file) => file.extracted_text);
  if (!readable.length) return null;

  const primary = readable[0];
  const text = primary.extracted_text || "";
  const accountHolder = extractedField(text, "Account Holder");
  const statementPeriod = extractedField(text, "Statement Period");
  const accountNumber = extractedField(text, "Account Number");
  const endingBalance = followingValue(text, "Ending Balance");
  const beginningBalance = followingValue(text, "Beginning Balance");
  const deposits = followingValue(text, "Total Deposits / Credits");
  const withdrawals = followingValue(text, "Total Withdrawals / Debits");

  const facts = [
    accountHolder ? `Account holder: ${accountHolder}` : "",
    statementPeriod ? `Statement period: ${statementPeriod}` : "",
    accountNumber ? `Account number: ${accountNumber}` : "",
    beginningBalance ? `Beginning balance: ${beginningBalance}` : "",
    deposits ? `Deposits/credits: ${deposits}` : "",
    withdrawals ? `Withdrawals/debits: ${withdrawals}` : "",
    endingBalance ? `Ending balance: ${endingBalance}` : "",
  ].filter(Boolean);

  const preview = facts.length
    ? facts.join("\n")
    : text.replace(/\s+/g, " ").slice(0, 700);

  return {
    message: [
      `Yes. I read ${primary.name} using ${primary.extraction_method || "the local parser"}.`,
      reason ? `Note: ${reason}.` : "",
      preview,
    ]
      .filter(Boolean)
      .join("\n"),
    actions: [],
    files: readable.map((file) => ({
      name: file.name,
      type: file.type,
      size: file.size,
      extraction_status: file.extraction_status,
      extraction_method: file.extraction_method,
      extraction_error: file.extraction_error,
    })),
    source: "file-parser",
  };
}

function messageRefusesReadableFile(message, body) {
  if (!(body.files || []).some((file) => file.extracted_text)) return false;
  return /cannot\s+(directly\s+)?(read|parse|view)|can't\s+(directly\s+)?(read|parse|view)|please\s+(copy|paste|provide)\s+(the\s+)?(text|data)/i.test(
    String(message || ""),
  );
}

// Write-intent verb in the prompt itself.
function promptWantsWorkbookOutput(prompt) {
  return /\b(put|organize|populate|create|write|insert|build|add|extract|fill|make|generate|import|reconcile|parse)\b/i.test(
    String(prompt || ""),
  );
}

// The user asked for workbook output (write-intent verb, or a parsed file is
// attached and presumably destined for the sheet).
function expectsWorkbookActions(body) {
  return promptWantsWorkbookOutput(body?.prompt) || (body?.files || []).some((file) => file.extracted_text);
}

// The reply asserts that the workbook was changed.
function claimsWorkbookChange(message) {
  return /\b(done|populated|created|added|wrote|written|inserted|built|filled|updated|placed|imported|organized)\b/i.test(
    String(message || ""),
  );
}

function buildAttachmentPromptContext(fileContexts) {
  const readable = (fileContexts || []).filter((file) => file.extracted_text);
  const failed = (fileContexts || []).filter((file) => file.extraction_status === "failed");
  if (!readable.length && !failed.length) return "";

  const sections = [];
  if (readable.length) {
    sections.push(
      "ATTACHED FILE CONTENTS ALREADY EXTRACTED FOR YOU. Use this text as if you read the files directly. Do not say you cannot read the file.",
    );
    for (const file of readable) {
      sections.push(
        [
          `--- FILE: ${file.name}`,
          `TYPE: ${file.type || "unknown"}`,
          `EXTRACTION: ${file.extraction_method || "parsed"}`,
          file.extraction_error ? `NOTE: ${file.extraction_error}` : "",
          "CONTENT:",
          file.extracted_text,
        ]
          .filter(Boolean)
          .join("\n"),
      );
    }
  }
  if (failed.length) {
    sections.push("ATTACHED FILES THAT COULD NOT BE READ:");
    for (const file of failed) {
      sections.push(`- ${file.name}: ${file.extraction_error || "unknown extraction error"}`);
    }
  }
  return sections.join("\n\n");
}

function normalizeWorkbook(workbook) {
  if (!workbook || typeof workbook !== "object") return { activeSheet: "", sheets: [] };
  const activeSheet = String(workbook.activeSheet || "").slice(0, 50);
  const rawSheets = Array.isArray(workbook.sheets) ? workbook.sheets : [];
  const sheets = [];
  for (let index = 0; index < Math.min(rawSheets.length, 30); index += 1) {
    const entry = rawSheets[index];
    if (typeof entry === "string") {
      sheets.push({ name: entry.slice(0, 50), usedRange: "", rowCount: 0, columnCount: 0 });
    } else if (entry && typeof entry === "object") {
      sheets.push({
        name: String(entry.name || "").slice(0, 50),
        usedRange: String(entry.usedRange || "").slice(0, 30),
        rowCount: Number(entry.rowCount || 0),
        columnCount: Number(entry.columnCount || 0),
      });
    }
  }
  return { activeSheet, sheets };
}

function buildSystemPrompt(loopBudgetExhausted) {
  const prompt = [
    "You are Hermes inside an Excel task pane.",
    "The user's workbook is already open in Excel in front of them. You cannot touch the filesystem: do NOT create, save, or write any files, and do NOT use any of your own tools. The ONLY way to change the workbook is to return actions in the JSON reply; the Excel add-in executes them in the open workbook.",
    "Return JSON only. No markdown.",
    "The JSON schema is:",
    '{"message":"short user-facing reply","actions":[{"type":"write_cells","start_cell":"Sheet1!A1","values":[["matrix"]],"allow_overwrite":true,"auto_format":true},{"type":"create_sheet","name":"Sheet Name","values":[["matrix"]]},{"type":"format_cells","range":"Sheet Name!A1:D1","style":["header"],"auto_fit":true},{"type":"conditional_format","range":"Sheet Name!G2:G31","operator":"lessThan","value":0.25,"fill_color":"#FFC7CE","font_color":"#9C0006"},{"type":"execute_office_js","explanation":"what changes","code":"Office.js code body that uses context"}]}',
    'A read action is also available: {"type":"read_range","range":"Sheet Name!A1:D200","reason":"short why"}.',
    "COMPLETE THE ENTIRE TASK IN THIS ONE REPLY. You get no follow-up turn except to receive read_range results you explicitly request. Never say you will continue 'in a couple of actions', 'next', or 'then' — emit every action the task needs right now, in this single actions array.",
    "Each cell value must be short — a label, a number, or a formula. Formulas are encouraged (see the A1-relative formula rule below); the only limit is on prose, not on formulas or numbers. NEVER put a sentence, explanation, or multi-clause note (more than ~40 characters of prose) inside a cell, and never build a 'QA Notes' block out of long prose rows — that corrupts the output and makes the model stop mid-reply. If the user wants a QA note, keep it to a few short cells or put the explanation in the 'message' field instead of in the sheet.",
    "Use conditional_format (NOT execute_office_js) to highlight cells by value, e.g. Margin % below a threshold. operator is one of lessThan, lessThanOrEqual, greaterThan, greaterThanOrEqual, equalTo, notEqualTo, between, notBetween. For percentage columns the underlying cell value is a decimal, so 'below 25%' means value 0.25 (not 25). fill_color/font_color are hex; default is light-red fill #FFC7CE with dark-red font #9C0006.",
    "Prefer the native actions (write_cells, create_sheet, format_cells, conditional_format) over execute_office_js — they are reliable. Only use execute_office_js for structural changes that have no native action. If you do, the code is a JSON string: prefer single quotes for any JS string literal so you do not have to escape double quotes, and keep the code short.",
    "When you need cell data that was not provided (another worksheet, a wider range), return ONLY read_range actions (max 5) plus a one-line message; the add-in will run the reads and call you again with the data under TOOL RESULTS. Then give the final answer with write actions.",
    "The workbook context lists every sheet with its used range; you may read from any of them.",
    "Never invent, estimate, or placeholder financial numbers. If data is missing and cannot be read, state exactly what is missing.",
    "Never claim you created, populated, or wrote anything unless THIS reply includes the actions that do it. A success message with an empty actions array is a failure: the add-in writes nothing without actions.",
    "Formulas you put inside a values matrix must be written as if the table's top-left cell is A1 (header in row 1, first data row in row 2): e.g. a Total in the first data row is =B2*C2 and a column total is =SUM(D2:D4). The add-in automatically relocates these to wherever the table is placed. Do not try to guess the absolute anchor yourself.",
    "Choose create_sheet for a brand-new report/schedule/export that should stand alone. Choose write_cells to add to or edit the sheet the user is on; omit start_cell to write at the user's current selection, or set start_cell explicitly (e.g. Sheet1!A1).",
    "Use actions whenever the user asks you to create, edit, format, populate, reconcile, match, or write spreadsheet output.",
    "Use create_sheet for new reports, schedules, balance sheets, exports, and generated tables.",
    "Use write_cells for edits to the current sheet or a requested range. start_cell may include a sheet name like Sheet1!A1.",
    "Use format_cells after writing when you know the final ranges. Named styles: number, integer, currency, percent, ratio, text, header, total-row, subtotal, input, blank-section.",
    "Use execute_office_js only when cell values are not enough, such as formatting, formulas across dynamic ranges, or workbook structure changes.",
    "Do not include nested Excel.run in execute_office_js code; the add-in already supplies context.",
    "In execute_office_js code, context is an Excel RequestContext: worksheets are context.workbook.worksheets (never context.worksheets), and you must not call context.sync() at the end — the add-in does.",
    "If no workbook action is needed, return actions: [].",
    "Keep message short; the workbook actions are the real output.",
    "Keep numeric/accounting output compact and presentation-ready.",
    "Formatting conventions: default font Arial 10; first rows of tables should be header style; totals should use total-row; hardcoded/input assumptions should be blue font or input style; formulas should stay black; linked values may be green.",
    "Prefer formulas over hardcoded totals. Include headers in row 1. Use currency or number formats on numeric columns and autofit columns.",
    "Keep generated grids compact: maximum 30 columns and 200 rows; ordinary accounting schedules should usually be 4 to 8 columns.",
    "Never represent a report as one long row. Use normal row records, one worksheet row per logical line.",
    "For balance sheets, use rows like ['Cash',100000,'Accounts Payable',50000], not hundreds of blank columns.",
    "Attached files may include extracted_text from Docling. Use it as source evidence for summaries, schedules, matching, tie-outs, and workbook edits.",
    "If a file extraction_status is failed, tell the user which file could not be read and continue with any readable files and workbook context.",
    "If attached file content is provided, you have read the file. Never tell the user you cannot directly read or parse that PDF/document.",
    "If the user asks whether you can see or read an attached file, answer from the extracted content and briefly summarize the file.",
  ].join("\n");
  if (loopBudgetExhausted) {
    return `${prompt}\n\nREAD BUDGET EXHAUSTED: do not return read_range actions; answer using only the data already provided.`;
  }
  return prompt;
}

function buildChatMessages(body) {
  const loopBudgetExhausted = Number(body.loop_count || 0) >= 5;
  const messages = [{ role: "system", content: buildSystemPrompt(loopBudgetExhausted) }];

  const history = Array.isArray(body.history) ? body.history : [];
  messages.push(
    ...history
      .filter((turn) => turn && (turn.role === "user" || turn.role === "assistant"))
      .slice(-12)
      .map((turn) => ({ role: turn.role, content: String(turn.content || "").slice(0, 4000) })),
  );

  const fileSummary = (body.files || []).map((file) => ({
    name: file.name,
    type: file.type,
    size: file.size,
    extraction_status: file.extraction_status,
    extraction_method: file.extraction_method,
    extraction_error: file.extraction_error,
  }));
  const fileContexts = (body.files || []).map((file) => ({
    name: file.name,
    type: file.type,
    size: file.size,
    extraction_status: file.extraction_status,
    extraction_method: file.extraction_method,
    extraction_error: file.extraction_error,
    extracted_text: file.extracted_text || "",
  }));

  messages.push({
    role: "user",
    content: JSON.stringify({
      prompt: body.prompt,
      workbook: normalizeWorkbook(body.workbook),
      selection: compactSelection(body.selection),
      files: fileSummary,
    }),
  });

  const attachmentPromptContext = buildAttachmentPromptContext(fileContexts);
  if (attachmentPromptContext) {
    messages.push({ role: "user", content: attachmentPromptContext });
  }

  if (Array.isArray(body.tool_results) && body.tool_results.length) {
    const capped = body.tool_results.map((result) => {
      const entry = { range: result.range };
      if (result.error) entry.error = String(result.error).slice(0, 400);
      if (result.values) entry.values = normalizeMatrix(result.values);
      if (result.formulas) entry.formulas = normalizeMatrix(result.formulas);
      if (result.truncated) entry.truncated = true;
      return entry;
    });
    messages.push({
      role: "user",
      content: `TOOL RESULTS (workbook reads you requested). Use these values as ground truth:\n${JSON.stringify(capped)}`,
    });
  }

  return messages;
}

async function callHermesModel(body) {
  const fileSummary = (body.files || []).map((file) => ({
    name: file.name,
    type: file.type,
    size: file.size,
    extraction_status: file.extraction_status,
    extraction_method: file.extraction_method,
    extraction_error: file.extraction_error,
  }));
  const payload = {
    model: llmModel,
    temperature: 0.1,
    max_tokens: llmMaxTokens,
    messages: buildChatMessages(body),
  };

  // Each call gets its own full timeout: a busy gateway can eat most of the
  // budget on the first generation, and corrective retries must not start
  // life already half-expired.
  const postOnce = async (messages) => {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), llmTimeoutMs);
    try {
      const response = await fetch(`${llmBaseUrl}/chat/completions`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${llmApiKey}`,
        },
        body: JSON.stringify({ ...payload, messages }),
        signal: controller.signal,
      });
      if (!response.ok) {
        throw new Error(`LLM HTTP ${response.status}: ${await response.text().catch(() => "")}`);
      }
      const json = await response.json();
      return json.choices?.[0]?.message?.content || "";
    } finally {
      clearTimeout(timeout);
    }
  };

  try {
    const content = await postOnce(payload.messages);
    let parsed = extractJsonObject(content);
    if (!parsed || typeof parsed !== "object") {
      // One corrective retry: show the model its own reply and demand strict JSON.
      const retryContent = await postOnce([
        ...payload.messages,
        { role: "assistant", content },
        {
          role: "user",
          content:
            "Your previous reply was not valid JSON. Resend the exact same answer as strictly valid JSON matching the schema. Output JSON only, no other text.",
        },
      ]);
      parsed = extractJsonObject(retryContent);
    }
    if (!parsed || typeof parsed !== "object") {
      return fallbackResponse(body, "LLM returned invalid JSON twice");
    }
    if (messageRefusesReadableFile(parsed.message, body)) {
      return readableFileAnswer(body, "the model tried to refuse a file that the local parser already read");
    }
    let actions = normalizeActions(parsed, body);
    if (Number(body.loop_count || 0) >= 5) {
      actions = actions.filter((action) => action.type !== "read_range");
    }
    // The model sometimes claims success ("Done.", "has been populated")
    // while returning zero actions — nothing reaches the workbook. Demand
    // the actions once; failing that, make the message honest.
    if (!actions.length && expectsWorkbookActions(body) && claimsWorkbookChange(parsed.message)) {
      const retryContent = await postOnce([
        ...payload.messages,
        { role: "assistant", content },
        {
          role: "user",
          content:
            "Your reply claimed the workbook was changed, but it contained no actions — nothing was written. Return the same answer WITH the actions array that performs the work (create_sheet or write_cells with the full values matrix). If you cannot do it, say plainly that no changes were made.",
        },
      ]);
      const retryParsed = extractJsonObject(retryContent);
      if (retryParsed && typeof retryParsed === "object") {
        let retryActions = normalizeActions(retryParsed, body);
        if (Number(body.loop_count || 0) >= 5) {
          retryActions = retryActions.filter((action) => action.type !== "read_range");
        }
        if (retryActions.length) {
          parsed = retryParsed;
          actions = retryActions;
        }
      }
      if (!actions.length) {
        parsed.message = `${String(parsed.message || "")}\n(No workbook changes were made — the model returned no actions.)`;
      }
    }
    const result = {
      message: String(parsed.message || "Done."),
      actions,
      files: fileSummary,
      source: "llm",
    };
    if (actions.some((action) => action.type === "read_range")) {
      // Echoed back by the pane on loop rounds so attachments are not re-parsed.
      result.parsed_files = (body.files || []).map((file) => ({
        name: file.name,
        type: file.type,
        size: file.size,
        extraction_status: file.extraction_status,
        extraction_method: file.extraction_method,
        extraction_error: file.extraction_error,
        extracted_text: file.extracted_text || "",
      }));
    }
    return result;
  } finally {
    // Per-call timeouts live inside postOnce now.
  }
}

function fallbackResponse(body, reason = null) {
  const prompt = String(body.prompt || "");
  const lower = prompt.toLowerCase();
  const parsedFiles = (body.files || []).filter((file) => file.extracted_text);
  const failedFiles = (body.files || []).filter((file) => file.extraction_status === "failed");
  const modelFallbackLabel =
    reason && /invalid json|refus|not usable|could not safely/i.test(reason)
      ? "Hermes model returned output the bridge could not safely apply"
      : "Hermes model is not reachable";
  const modelFallbackSentence = reason ? `${modelFallbackLabel} (${reason}).` : `${modelFallbackLabel}.`;
  const fileSummary = (body.files || [])
    .map((file) =>
      file.extraction_status === "parsed"
        ? `${file.name} (${file.extraction_method})`
        : file.extraction_status === "failed"
          ? `${file.name} (read failed)`
          : file.name,
    )
    .join(", ") || "no files";

  // Any write-intent prompt with parsed file data gets a real sheet built from
  // that data — never a chatty preview dump (found live: "Parse this PDF and
  // make a workbook" timed out, then routed to the see/read branch below).
  if (parsedFiles.length && (promptWantsWorkbookOutput(prompt) || /\b(transactions?|statement|summar)/i.test(prompt))) {
    const primary = parsedFiles[0];
    return {
      message: [
        modelFallbackSentence,
        `I created a worksheet strictly from data parsed out of ${primary.name}; nothing was estimated or invented.`,
      ].join("\n"),
      actions: [
        {
          type: "create_sheet",
          name: "Bank Statement Data",
          values: buildAccountingDataSheet(primary),
        },
      ],
      files: parsedFiles.map((file) => ({
        name: file.name,
        type: file.type,
        size: file.size,
        extraction_status: file.extraction_status,
        extraction_method: file.extraction_method,
        extraction_error: file.extraction_error,
      })),
      source: "file-parser",
    };
  }

  if (
    parsedFiles.length &&
    (/\b(can you|do you|did you)?\s*(see|read|parse|view)\b/i.test(lower) ||
      lower.includes("pdf") ||
      lower.includes("ending balance") ||
      lower.includes("account holder"))
  ) {
    return readableFileAnswer(body, reason);
  }

  return {
    message: [
      modelFallbackSentence,
      "No workbook changes were made; Hermes does not write numbers it could not source.",
      `Workbook active sheet: ${body.workbook?.activeSheet || "unknown"}`,
      `Selection: ${body.selection?.address || "unknown"}`,
      `Files attached: ${fileSummary}`,
      parsedFiles.length ? `Readable file text: ${parsedFiles.map((file) => file.name).join(", ")}` : "",
      failedFiles.length ? `Files not read: ${failedFiles.map((file) => `${file.name}: ${file.extraction_error}`).join("; ")}` : "",
      `Configured LLM: ${llmBaseUrl} / ${llmModel}`,
      "Check the Hermes window started by start-windows-native-hermes-api.cmd, then ask again.",
    ].filter(Boolean).join("\n"),
    actions: [],
    files: (body.files || []).map((file) => ({
      name: file.name,
      type: file.type,
      size: file.size,
      extraction_status: file.extraction_status,
      extraction_method: file.extraction_method,
      extraction_error: file.extraction_error,
    })),
    source: "fallback",
  };
}

async function handleChat(req, res) {
  pruneUploads().catch(() => {});
  const body = await readJson(req);
  // Loop rounds echo parsed_files back so attachments are not re-uploaded or re-parsed.
  body.files =
    Array.isArray(body.parsed_files) && body.parsed_files.length
      ? body.parsed_files
      : await prepareAttachedFiles(body.files || []);

  try {
    return send(res, 200, await callHermesModel(body));
  } catch (error) {
    return send(res, 200, fallbackResponse(body, error.message));
  }
}

async function serveStatic(req, res) {
  const url = new URL(req.url, `http://${req.headers.host}`);
  let pathname = decodeURIComponent(url.pathname);
  if (pathname === "/") pathname = "/taskpane.html";
  // Real icons ship in assets/; the embedded pixel is only a fallback so a
  // bare checkout still renders a ribbon button.
  if (pathname.startsWith("/assets/icon-") && !existsSync(path.normalize(path.join(root, pathname)))) {
    res.writeHead(200, {
      "content-type": "image/png",
      "access-control-allow-origin": "*",
    });
    return res.end(iconPng);
  }
  const file = path.normalize(path.join(root, pathname));
  if (!file.startsWith(root) || !existsSync(file)) {
    return send(res, 404, "not found", "text/plain; charset=utf-8");
  }
  const ext = path.extname(file);
  const bytes = await readFile(file);
  res.writeHead(200, {
    "content-type": mime[ext] || "application/octet-stream",
    "access-control-allow-origin": "*",
  });
  res.end(bytes);
}

async function checkHermesEndpoint() {
  const response = await fetch(`${llmBaseUrl}/models`, {
    headers: {
      authorization: `Bearer ${llmApiKey}`,
    },
  });
  if (!response.ok) {
    throw new Error(`Hermes API Server check failed: HTTP ${response.status} ${await response.text().catch(() => "")}`);
  }
  const json = await response.json();
  const models = Array.isArray(json.data) ? json.data.map((item) => item.id).filter(Boolean).join(", ") : "unknown";
  console.log(`Windows-native Hermes API Server OK: ${llmBaseUrl}`);
  console.log(`Models: ${models || "unknown"}`);
}

async function checkDoclingEndpoint() {
  const json = await fetchJsonWithTimeout(`${doclingBaseUrl}/healthz`, {}, 10000);
  console.log(`Docling parser OK: ${doclingBaseUrl}`);
  console.log(`Docling health: ${JSON.stringify(json)}`);
}

async function extractFileCli() {
  const filePath = process.argv[process.argv.indexOf("--extract-file") + 1];
  if (!filePath) throw new Error("usage: node broker/server.mjs --extract-file <path>");
  const bytes = await readFile(filePath);
  const context = await prepareAttachedFiles([
    {
      name: path.basename(filePath),
      type: extensionOf(filePath) === ".pdf" ? "application/pdf" : "application/octet-stream",
      size: bytes.length,
      base64: bytes.toString("base64"),
    },
  ]);
  console.log(JSON.stringify(context[0], null, 2));
}

const isMainModule = process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href;

if (isMainModule) {
  if (process.argv.includes("--check-hermes")) {
    checkHermesEndpoint().then(
      () => process.exit(0),
      (error) => {
        console.error(error.message);
        process.exit(1);
      },
    );
  } else if (process.argv.includes("--check-docling")) {
    checkDoclingEndpoint().then(
      () => process.exit(0),
      (error) => {
        console.error(error.message);
        process.exit(1);
      },
    );
  } else if (process.argv.includes("--extract-file")) {
    extractFileCli().then(
      () => process.exit(0),
      (error) => {
        console.error(error.message);
        process.exit(1);
      },
    );
  } else {
    pruneUploads().catch(() => {});
    // Never let one bad request kill the bridge.
    process.on("unhandledRejection", (reason) => {
      console.error("unhandledRejection:", reason);
    });
    process.on("uncaughtException", (error) => {
      console.error("uncaughtException:", error);
    });
    const server = http.createServer(async (req, res) => {
      try {
        if (req.method === "OPTIONS") return send(res, 204, "");
        if (req.method === "GET" && req.url === "/api/health") return send(res, 200, await healthStatus());
        // `await` is load-bearing: without it a rejection inside a handler
        // (e.g. an aborted upload stream) escapes this try/catch and crashes
        // the process as an unhandled rejection.
        if (req.method === "POST" && req.url === "/api/chat") return await handleChat(req, res);
        if (req.method === "POST" && req.url === "/api/export") return await handleExport(req, res);
        return await serveStatic(req, res);
      } catch (error) {
        return send(res, 500, { error: error.message });
      }
    });

    server.listen(port, "127.0.0.1", () => {
      console.log(`Hermes Excel add-in running at http://localhost:${port}`);
      console.log(`Manifest: ${path.join(root, "manifest.xml")}`);
      console.log(`Hermes backend endpoint: ${llmBaseUrl}`);
    });
  }
}

export {
  parseDelimitedText,
  markdownTableFromRows,
  extractJsonObject,
  normalizeMatrix,
  normalizeAction,
  normalizeActions,
  legacyWriteToActions,
  parseMoney,
  extractedField,
  followingValue,
  moneyValueForLabel,
  statementSummaryRows,
  statementTransactionRows,
  buildAccountingDataSheet,
  truncateText,
  windowsPathToWsl,
  extensionOf,
  compactSelection,
  normalizeWorkbook,
  buildChatMessages,
  buildSystemPrompt,
  fallbackResponse,
  readHermesApiServerKey,
  scanWrittenCells,
  matrixToCsv,
  safeExportName,
  expectsWorkbookActions,
  claimsWorkbookChange,
  promptWantsWorkbookOutput,
};
