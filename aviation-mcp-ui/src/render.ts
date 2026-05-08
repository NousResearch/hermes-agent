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
      if (value) replacement = renderTemplate(body, data);
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
