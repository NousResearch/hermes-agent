// Security/behavior tests for the build-time plugin README renderer
// (apps/docs/scripts/lib/plugin-readme.mjs), ported from the Docusaurus plugin
// along with it. These cover ONLY the pure helpers: the serializer's tag/attribute
// allowlist, URL safety, pinned-commit link rewriting and the fetch host guard.
// renderReadme itself needs the unified/remark-* dependencies and is exercised by
// the build (and by CI browser checks), not here.
//
// Run: node --test apps/docs/tests/   (from apps/docs; no test runner install needed)

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  absolutize,
  fetchReadme,
  forgeBases,
  readmeCandidates,
  rewriteUrls,
  safeUrl,
  serialize,
  serializeAttrs,
} from "../scripts/lib/plugin-readme.mjs";

// ── hast fixtures (the shape remark-rehype produces) ─────────────────────────

const el = (tagName, properties = {}, children = []) => ({
  type: "element",
  tagName,
  properties,
  children,
});
const text = (value) => ({ type: "text", value });
const root = (children) => ({ type: "root", children });

const SHA = "0123456789abcdef0123456789abcdef01234567";
const GH_README = `https://raw.githubusercontent.com/acme/plug/${SHA}/sub/README.md`;
const GL_README = `https://gitlab.com/grp/proj/-/raw/${SHA}/dir/README.md`;

// ── safeUrl: only http(s), mailto and fragments survive ──────────────────────

test("safeUrl passes https, http, mailto and bare fragments", () => {
  assert.equal(safeUrl("https://example.com/x"), "https://example.com/x");
  assert.equal(safeUrl("http://example.com/x"), "http://example.com/x");
  assert.equal(safeUrl("mailto:a@b.example"), "mailto:a@b.example");
  assert.equal(safeUrl("#anchor"), "#anchor");
  assert.equal(safeUrl(""), "");
});

test("safeUrl drops javascript:, data:, vbscript: and protocol-relative URLs", () => {
  assert.equal(safeUrl("javascript:alert(1)"), null);
  assert.equal(safeUrl("JaVaScRiPt:alert(1)"), null);
  assert.equal(safeUrl("data:text/html,<script>"), null);
  assert.equal(safeUrl("vbscript:msg"), null);
  assert.equal(safeUrl("//evil.example/x"), null);
  assert.equal(safeUrl(" javascript:alert(1) "), null);
});

// ── serialize: tag/attribute allowlist, escaping, unwrapping ─────────────────

test("serialize escapes text nodes", () => {
  assert.equal(serialize(root([text("<b> & \"'")])), "&lt;b&gt; &amp; &quot;&#39;");
});

test("serialize drops comments and non-element nodes", () => {
  assert.equal(serialize(root([{ type: "comment", value: "hi" }])), "");
});

test("serialize unwraps unknown tags but keeps their text", () => {
  assert.equal(serialize(root([el("script", {}, [text("alert(1)")])])), "alert(1)");
  assert.equal(serialize(root([el("marquee", {}, [text("x")])])), "x");
});

test("serialize keeps allowlisted structural tags", () => {
  assert.equal(
    serialize(root([el("ul", {}, [el("li", {}, [text("a")])])])),
    "<ul><li>a</li></ul>",
  );
});

test("serialize emits void tags without a closing tag", () => {
  assert.equal(serialize(root([el("img", { src: "https://e/x.png", alt: "x" })])), '<img src="https://e/x.png" alt="x">');
  assert.equal(serialize(root([el("br")])), "<br>");
});

test("serialize drops event-handler attributes", () => {
  assert.equal(
    serialize(root([el("a", { href: "https://ok.example", onclick: "evil()" }, [text("x")])])),
    '<a href="https://ok.example">x</a>',
  );
});

test("serialize drops javascript: hrefs entirely", () => {
  assert.equal(serialize(root([el("a", { href: "javascript:alert(1)" }, [text("x")])])), "<a>x</a>");
});

test("serialize keeps only checkbox inputs", () => {
  assert.equal(serialize(root([el("input", { type: "text" })])), "<input>");
  assert.equal(serialize(root([el("input", { type: "checkbox", checked: true })])), '<input type="checkbox" checked>');
});

test("serialize maps React-style attribute names to HTML", () => {
  const out = serializeAttrs("td", { className: ["a", "b"], colSpan: 2, rowSpan: 3, align: "left" });
  assert.equal(out, ' class="a b" colspan="2" rowspan="3" align="left"');
});

test("serializeAttrs escapes attribute values", () => {
  const out = serializeAttrs("a", { title: '"><script>' });
  assert.equal(out, ' title="&quot;&gt;&lt;script&gt;"');
});

// ── pinned-commit URL rewriting ──────────────────────────────────────────────

test("forgeBases derives raw and blob bases for a GitHub README at a sha", () => {
  assert.deepEqual(forgeBases(GH_README), {
    raw: `https://raw.githubusercontent.com/acme/plug/${SHA}/sub/`,
    blob: `https://github.com/acme/plug/blob/${SHA}/sub/`,
  });
});

test("forgeBases maps gitlab raw paths to blob paths", () => {
  const bases = forgeBases(GL_README);
  assert.equal(bases.raw, `https://gitlab.com/grp/proj/-/raw/${SHA}/dir/`);
  assert.equal(bases.blob, `https://gitlab.com/grp/proj/-/blob/${SHA}/dir/`);
});

test("rewriteUrls resolves relative img/a URLs at the pinned commit", () => {
  const tree = root([
    el("p", {}, [
      el("img", { src: "shot.png" }),
      el("a", { href: "docs/page.md" }, [text("docs")]),
      el("a", { href: "https://absolute.example/x" }, [text("abs")]),
      el("a", { href: "#frag" }, [text("frag")]),
    ]),
  ]);
  rewriteUrls(tree, forgeBases(GH_README));
  const [img, rel, abs, frag] = tree.children[0].children;
  assert.equal(img.properties.src, `https://raw.githubusercontent.com/acme/plug/${SHA}/sub/shot.png`);
  assert.equal(rel.properties.href, `https://github.com/acme/plug/blob/${SHA}/sub/docs/page.md`);
  assert.equal(abs.properties.href, "https://absolute.example/x");
  assert.equal(frag.properties.href, "#frag");
});

test("rewriteUrls prefixes README heading ids so they never clash with page chrome", () => {
  const tree = root([el("h2", { id: "install" }, [text("Install")])]);
  rewriteUrls(tree, forgeBases(GH_README));
  assert.equal(tree.children[0].properties.id, "readme-install");
});

test("absolutize passes through absolute, scheme'd, fragment and protocol-relative values", () => {
  const base = "https://raw.example/base/";
  assert.equal(absolutize("https://x.example/y", base), "https://x.example/y");
  assert.equal(absolutize("mailto:a@b.example", base), "mailto:a@b.example");
  assert.equal(absolutize("#x", base), "#x");
  assert.equal(absolutize("//cdn.example/y", base), "//cdn.example/y");
  assert.equal(absolutize("", base), "");
});

test("absolutize resolves relative paths (including ./) against the base", () => {
  assert.equal(absolutize("a/b.png", "https://r.example/d/"), "https://r.example/d/a/b.png");
  assert.equal(absolutize("./c.png", "https://r.example/d/"), "https://r.example/d/c.png");
});

// ── README candidate discovery + fetch guard ─────────────────────────────────

test("readmeCandidates tries the subdir first, then the repo root, in common casings", () => {
  const candidates = readmeCandidates(GH_README);
  assert.equal(candidates[0], `https://raw.githubusercontent.com/acme/plug/${SHA}/sub/README.md`);
  // the root fallback starts after the six subdir casings
  assert.equal(candidates[6], `https://raw.githubusercontent.com/acme/plug/${SHA}/README.md`);
  assert.ok(candidates.some((c) => c.endsWith("/sub/readme.md")));
  assert.ok(candidates.some((c) => c.endsWith("/README.MD")));
  assert.ok(candidates.some((c) => c.endsWith("/docs/README.md")));
  assert.equal(candidates.length, 12); // 2 paths × 6 names
});

test("readmeCandidates collapses to one path when the README already sits at the root", () => {
  const rootUrl = `https://raw.githubusercontent.com/acme/plug/${SHA}/README.md`;
  const candidates = readmeCandidates(rootUrl);
  assert.equal(candidates.length, 6);
  assert.ok(candidates.every((c) => !c.includes("/sub/")));
});

test("fetchReadme refuses non-allowlisted hosts without any network fetch", async () => {
  const logs = [];
  const got = await fetchReadme("https://evil.example/x/README.md", "/tmp/nonexistent-cache", (m) => logs.push(m));
  assert.equal(got, null);
  assert.ok(logs.some((m) => m.includes("refusing README fetch from evil.example")));
});

test("fetchReadme refuses plain-http URLs even on an allowlisted host", async () => {
  const logs = [];
  const got = await fetchReadme("http://raw.githubusercontent.com/x/README.md", "/tmp/nonexistent-cache", (m) => logs.push(m));
  assert.equal(got, null);
  assert.ok(logs.some((m) => m.includes("refusing README fetch from raw.githubusercontent.com")));
});
