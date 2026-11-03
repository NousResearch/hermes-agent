// Security/behavior tests for the build-time plugin README renderer
// (apps/docs/scripts/lib/plugin-readme.mjs), ported from the Docusaurus plugin
// along with it. The pure-helper tests cover the serializer's tag/attribute
// allowlist, URL safety, pinned-commit link rewriting and the fetch host guard.
// The anchor tests run the REAL renderReadme pipeline (unified/remark-* and
// github-slugger are direct dependencies), so heading ids and fragment links
// are proven against actual rendered output, not hand-built HAST.
//
// Run from apps/docs: node --test tests/catalog-readme.test.mjs

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  absolutize,
  fetchReadme,
  forgeBases,
  readmeCandidates,
  renderReadme,
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

// ── renderReadme: GitHub-compatible anchors, proven through the real pipeline ──
//
// The unified pipeline by itself never assigns heading ids, but GitHub-flavoured
// READMEs link to their own headings with bare `#slug` hrefs. The contract the
// output validator's fragment gate relies on: every heading carries a
// readme-<github-slug> id (duplicate-aware), every id on the page is namespaced
// under readme- (footnotes included), and every non-empty #fragment href either
// resolves to one of those ids or was rewritten to the pinned source README on
// the forge — never a guessed local target.

const render = (markdown, readmeUrl = GH_README) => renderReadme(markdown, readmeUrl);

/** All id="…" values on the page. */
function pageIds(html) {
  return new Set([...html.matchAll(/\bid="([^"]*)"/g)].map((m) => m[1]));
}

/** Every non-empty #fragment href, without the leading #. */
function fragmentHrefs(html) {
  return [...html.matchAll(/\shref="#([^"]+)"/g)].map((m) => m[1]);
}

test("renderReadme assigns GitHub-style duplicate-aware readme- ids to headings", async () => {
  const html = await render("# Usage\n\n## Usage\n\n## Usage\n");
  assert.match(html, /<h1 id="readme-usage">Usage<\/h1>/);
  assert.match(html, /<h2 id="readme-usage-1">Usage<\/h2>/);
  assert.match(html, /<h2 id="readme-usage-2">Usage<\/h2>/);
  assert.equal(pageIds(html).size, 3); // unique — no duplicate ids on the page
});

test("rich inline headings slug from their text content", async () => {
  const html = await render("## The `install` **command** — 実行\n");
  assert.match(
    html,
    /<h2 id="readme-the-install-command--実行">The <code>install<\/code> <strong>command<\/strong> — 実行<\/h2>/,
  );
});

test("bare fragment links resolve to the namespaced heading ids", async () => {
  const html = await render("## Setup\n\nSee [setup](#setup) and [top](#).\n");
  assert.ok(html.includes('<a href="#readme-setup">setup</a>'));
  assert.ok(html.includes('<a href="#">top</a>')); // empty # keeps current behavior
});

test("fragment links resolve to duplicate-suffixed heading ids", async () => {
  const html = await render("## Setup\n\n## Setup\n\n[first](#setup) [second](#setup-1)\n");
  assert.ok(html.includes('<a href="#readme-setup">first</a>'));
  assert.ok(html.includes('<a href="#readme-setup-1">second</a>'));
});

test("percent-encoded and CJK fragments decode to their heading slugs", async () => {
  const html = await render(
    "## 中文标题\n\n[a](#中文标题) [b](#%E4%B8%AD%E6%96%87%E6%A0%87%E9%A2%98) [c](#%e4%b8%ad%e6%96%87%e6%a0%87%e9%a2%98)\n",
  );
  assert.match(html, /<h2 id="readme-中文标题">/);
  for (const label of ["a", "b", "c"]) {
    assert.ok(html.includes(`<a href="#readme-中文标题">${label}</a>`), `link ${label}`);
  }
});

test("fragment matching normalizes case and %20 like GitHub's anchor slugs", async () => {
  const html = await render("## Foo Bar\n\n## Usage\n\n[x](#foo%20bar) [y](#FOO%20BAR) [z](#foo-bar) [w](#USAGE)\n");
  assert.match(html, /<h2 id="readme-foo-bar">/);
  for (const label of ["x", "y", "z"]) {
    assert.ok(html.includes(`<a href="#readme-foo-bar">${label}</a>`), `link ${label}`);
  }
  assert.ok(html.includes('<a href="#readme-usage">w</a>'));
});

test("unmatched fragments fall back to the pinned source README, not a local guess", async () => {
  const html = await render('## Usage\n\n<a name="custom"></a> [x](#custom)\n');
  const source = `https://github.com/acme/plug/blob/${SHA}/sub/README.md`;
  assert.ok(!pageIds(html).has("custom"));
  assert.ok(!html.includes("name=")); // the raw anchor was dropped, not rendered
  assert.ok(html.includes(`<a href="${source}#custom">x</a>`));
});

test("unmatched fragments on gitlab READMEs fall back to the gitlab blob form", async () => {
  const html = await render("[x](#nope)\n", GL_README);
  const source = `https://gitlab.com/grp/proj/-/blob/${SHA}/dir/README.md`;
  assert.ok(html.includes(`<a href="${source}#nope">x</a>`));
});

test("raw HTML never contributes ids; its fragments leave the page for the pinned source", async () => {
  const html = await render('<h2 id="evil">Hi</h2>\n\n<a name="custom"></a>\n\n## Hi\n\n[x](#evil) [y](#custom) [z](#hi)\n');
  const ids = pageIds(html);
  assert.ok(!ids.has("evil"));
  assert.ok(!ids.has("custom"));
  assert.ok(ids.has("readme-hi")); // only the markdown heading got an id
  const source = `https://github.com/acme/plug/blob/${SHA}/sub/README.md`;
  assert.ok(html.includes(`<a href="${source}#evil">x</a>`));
  assert.ok(html.includes(`<a href="${source}#custom">y</a>`));
  assert.ok(html.includes('<a href="#readme-hi">z</a>'));
});

test("footnote ids and refs stay consistent and are namespaced under readme-", async () => {
  const html = await render("Note[^1].\n\n[^1]: Body.\n");
  assert.match(html, /<sup><a href="#readme-user-content-fn-1" id="readme-user-content-fnref-1">1<\/a><\/sup>/);
  assert.match(html, /<li id="readme-user-content-fn-1">/);
  assert.match(html, /<a href="#readme-user-content-fnref-1" class="data-footnote-backref">/);
  assert.match(html, /<h2 class="sr-only" id="readme-footnote-label">/); // never slug-collides with a `## Footnotes` heading
  const ids = pageIds(html);
  for (const frag of fragmentHrefs(html)) {
    assert.ok(ids.has(frag), `footnote fragment #${frag} has no matching id`);
  }
});

test("every non-empty fragment href resolves to an id on the page (fragment-gate contract)", async () => {
  const html = await render([
    "# Title",
    "",
    "## Setup",
    "",
    "## Setup",
    "",
    "## 中文 标题",
    "",
    "See [setup](#setup), [second](#setup-1), [cjk](#%E4%B8%AD%E6%96%87%20%E6%A0%87%E9%A2%98), [missing](#nope), [top](#).",
    "",
    "Note[^1]",
    "",
    "[^1]: Footnote body.",
  ].join("\n"));
  const ids = pageIds(html);
  for (const id of ids) {
    assert.ok(id.startsWith("readme-"), `id ${id} is not namespaced under readme-`);
  }
  const frags = fragmentHrefs(html);
  assert.ok(frags.length >= 4, `expected several fragment hrefs, got ${frags.length}`);
  for (const frag of frags) {
    assert.ok(ids.has(frag), `fragment #${frag} has no id on the page`);
  }
  // Unmatched fragments retain their source destination, not a guessed local id.
  assert.ok(!frags.includes("nope"));
  assert.ok(html.includes(`href="https://github.com/acme/plug/blob/${SHA}/sub/README.md#nope"`));
});
