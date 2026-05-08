import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { createUIResource, type UIResource } from "@mcp-ui/server";

export function escapeHtml(value: unknown): string {
  if (value === null || value === undefined) return "";
  const s = String(value);
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

type Ctx = Record<string, unknown>;

function lookup(ctx: Ctx, key: string): unknown {
  if (key === ".") return (ctx as { _self?: unknown })._self;
  return ctx[key];
}

export function renderTemplate(template: string, data: Ctx): string {
  // Sections (#) and conditionals (?) first.
  let out = template;
  const sectionRe = /\{\{([#?])(\w+)\}\}([\s\S]*?)\{\{\/\2\}\}/;
  while (true) {
    const m = out.match(sectionRe);
    if (!m) break;
    const [full, kind, name, body] = m;
    const value = data[name];
    let replacement = "";
    if (kind === "?") {
      if (value) {
        replacement =
          value && typeof value === "object"
            ? renderTemplate(body, value as Ctx)
            : renderTemplate(body, data);
      }
    } else {
      if (Array.isArray(value)) {
        replacement = value
          .map((item) => {
            if (item && typeof item === "object") {
              return renderTemplate(body, item as Ctx);
            }
            return renderTemplate(body, { _self: item } as Ctx);
          })
          .join("");
      }
    }
    out = out.slice(0, m.index!) + replacement + out.slice(m.index! + full.length);
  }
  // Triple-brace raw.
  out = out.replace(/\{\{\{(\.|\w+)\}\}\}/g, (_, k) => {
    const v = lookup(data, k);
    return v === undefined || v === null ? "" : String(v);
  });
  // Double-brace escaped.
  out = out.replace(/\{\{(\.|\w+)\}\}/g, (_, k) => escapeHtml(lookup(data, k)));
  return out;
}

export type { UIResource };

export function buildResource(view: string, id: string, html: string): UIResource {
  return createUIResource({
    uri: `ui://aviation/${view}/${id}`,
    content: { type: "rawHtml", htmlString: html },
    encoding: "text",
  });
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const TEMPLATE_DIR = join(__dirname, "..", "templates");

const cache = new Map<string, string>();

export function loadTemplate(name: string): string {
  if (!cache.has(name)) {
    cache.set(name, readFileSync(join(TEMPLATE_DIR, name), "utf8"));
  }
  return cache.get(name)!;
}
