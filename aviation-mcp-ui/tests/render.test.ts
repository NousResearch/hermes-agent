import { describe, it, expect } from "vitest";
import { escapeHtml, renderTemplate, buildResource } from "../src/render.js";

describe("escapeHtml", () => {
  it("escapes ampersand, lt, gt, quotes", () => {
    expect(escapeHtml(`<script>alert("xss")</script>&'`)).toBe(
      "&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;&amp;&#39;"
    );
  });

  it("returns empty string for null/undefined", () => {
    expect(escapeHtml(null as unknown as string)).toBe("");
    expect(escapeHtml(undefined as unknown as string)).toBe("");
  });

  it("stringifies non-strings", () => {
    expect(escapeHtml(42 as unknown as string)).toBe("42");
  });
});

describe("renderTemplate", () => {
  it("substitutes a single field with escaping", () => {
    const out = renderTemplate("hello {{name}}", { name: "<b>" });
    expect(out).toBe("hello &lt;b&gt;");
  });

  it("renders raw with triple braces", () => {
    const out = renderTemplate("style: {{{css}}}", { css: ".x{color:red}" });
    expect(out).toBe("style: .x{color:red}");
  });

  it("repeats a section with item objects", () => {
    const out = renderTemplate(
      "<ul>{{#items}}<li>{{label}}</li>{{/items}}</ul>",
      { items: [{ label: "A" }, { label: "B" }] }
    );
    expect(out).toBe("<ul><li>A</li><li>B</li></ul>");
  });

  it("repeats a section with primitive items via {{.}}", () => {
    const out = renderTemplate(
      "{{#tags}}[{{.}}]{{/tags}}",
      { tags: ["one", "two"] }
    );
    expect(out).toBe("[one][two]");
  });

  it("renders conditional iff truthy", () => {
    expect(renderTemplate("{{?on}}YES{{/on}}", { on: true })).toBe("YES");
    expect(renderTemplate("{{?on}}YES{{/on}}", { on: false })).toBe("");
    expect(renderTemplate("{{?on}}YES{{/on}}", {})).toBe("");
  });

  it("renders conditional object bodies against the object context", () => {
    const out = renderTemplate("{{?alert}}{{headline}} {{distance}}{{/alert}}", {
      alert: { headline: "Hantavirus", distance: 10 },
    });
    expect(out).toBe("Hantavirus 10");
  });

  it("treats missing fields as empty", () => {
    expect(renderTemplate("a={{a}} b={{b}}", { a: "x" })).toBe("a=x b=");
  });
});

describe("buildResource", () => {
  it("returns an MCP-UI EmbeddedResource shape", () => {
    const r = buildResource("airport", "LSZH", "<!DOCTYPE html><html></html>");
    expect(r.type).toBe("resource");
    expect(r.resource.uri).toBe("ui://aviation/airport/LSZH");
    expect(r.resource.mimeType).toMatch(/^text\/html/);
    expect(r.resource.text).toContain("<!DOCTYPE html>");
  });
});
