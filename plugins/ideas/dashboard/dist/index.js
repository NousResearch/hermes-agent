(function () {
  "use strict";

  const SDK = window.__HERMES_PLUGIN_SDK__;
  const REG = window.__HERMES_PLUGINS__;
  if (!SDK || !REG) return;

  const { React, fetchJSON } = SDK;
  const h = React.createElement;
  const { Card, CardContent, CardHeader, CardTitle, Badge, Button, Input, Label, Select, SelectOption } = SDK.components;
  const { useState, useEffect, useCallback, useMemo, useRef } = SDK.hooks;
  const { timeAgo } = SDK.utils;

  const API = "/api/plugins/ideas";
  const LS_BOARD_KEY = "hermes.kanban.selectedBoard";
  const EMPTY_IDEA = { title: "", summary: "", body: "# New idea\n\n", status: "draft", tags: [] };
  const STATUSES = ["draft", "active", "parked", "converted", "archived"];

  function readSelectedBoard() {
    try { return (window.localStorage.getItem(LS_BOARD_KEY) || "").trim() || "default"; }
    catch (_e) { return "default"; }
  }
  function writeSelectedBoard(slug) {
    try { if (slug) window.localStorage.setItem(LS_BOARD_KEY, slug); } catch (_e) {}
  }
  function withBoard(url, board) {
    const sep = url.indexOf("?") >= 0 ? "&" : "?";
    return `${url}${sep}board=${encodeURIComponent(board || "default")}`;
  }
  function json(method, url, body) {
    return fetchJSON(url, {
      method,
      headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  }

  // Hermes dashboard Select is shadcn-style: it emits onValueChange(value),
  // not a native DOM change event. Wire both shapes so this plugin also keeps
  // working if the SDK swaps back to a native select later.
  function selectChangeHandler(setter) {
    return {
      onValueChange: function (v) { setter(v == null ? "" : v); },
      onChange: function (e) {
        const v = e && e.target ? e.target.value : e;
        setter(v == null ? "" : v);
      },
    };
  }

  function fmtTags(tags) { return (tags || []).join(", "); }
  function parseTags(value) {
    return (value || "").split(",").map((x) => x.trim().replace(/^#/, "")).filter(Boolean);
  }
  function escapeHtml(value) {
    return String(value || "").replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[ch]));
  }
  function inlineMd(value) {
    return escapeHtml(value)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*]+)\*(?!\*)/g, "$1<em>$2</em>")
      .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
  }
  function renderMarkdown(md) {
    const lines = String(md || "").split(/\r?\n/);
    let html = "";
    let inCode = false;
    let listOpen = false;
    function closeList() { if (listOpen) { html += "</ul>"; listOpen = false; } }
    for (const raw of lines) {
      const line = raw.replace(/\s+$/, "");
      if (line.startsWith("```")) {
        closeList();
        html += inCode ? "</code></pre>" : "<pre><code>";
        inCode = !inCode;
        continue;
      }
      if (inCode) { html += escapeHtml(raw) + "\n"; continue; }
      if (!line.trim()) { closeList(); html += "<br/>"; continue; }
      const heading = line.match(/^(#{1,4})\s+(.*)$/);
      if (heading) {
        closeList();
        const level = heading[1].length;
        html += `<h${level}>${inlineMd(heading[2])}</h${level}>`;
        continue;
      }
      const task = line.match(/^[-*]\s+\[( |x|X)\]\s*(.*)$/);
      if (task) {
        if (listOpen) { html += "</ul>"; listOpen = false; }
        const checked = task[1].toLowerCase() === "x" ? " checked" : "";
        html += `<label class="ideas-md-task"><input type="checkbox" disabled${checked}/> <span>${inlineMd(task[2])}</span></label>`;
        continue;
      }
      const bareTask = line.match(/^\[( |x|X)?\]\s*(.*)$/);
      if (bareTask) {
        if (listOpen) { html += "</ul>"; listOpen = false; }
        const checked = (bareTask[1] || "").toLowerCase() === "x" ? " checked" : "";
        html += `<label class="ideas-md-task"><input type="checkbox" disabled${checked}/> <span>${inlineMd(bareTask[2])}</span></label>`;
        continue;
      }
      const bullet = line.match(/^[-*]\s+(.*)$/);
      if (bullet) {
        if (!listOpen) { html += "<ul>"; listOpen = true; }
        html += `<li>${inlineMd(bullet[1])}</li>`;
        continue;
      }
      closeList();
      html += `<p>${inlineMd(line)}</p>`;
    }
    closeList();
    if (inCode) html += "</code></pre>";
    return html;
  }

  function editorLineHtml(kind, content, extra) {
    const text = inlineMd(content || "");
    if (kind === "h1") return `<h1 data-md-kind="h1">${text || "<br/>"}</h1>`;
    if (kind === "h2") return `<h2 data-md-kind="h2">${text || "<br/>"}</h2>`;
    if (kind === "h3") return `<h3 data-md-kind="h3">${text || "<br/>"}</h3>`;
    if (kind === "quote") return `<blockquote data-md-kind="quote">${text || "<br/>"}</blockquote>`;
    if (kind === "bullet") return `<div data-md-kind="bullet" class="ideas-md-bullet"><span contenteditable="false" class="ideas-md-marker">•</span><span class="ideas-md-content">${text || "<br/>"}</span></div>`;
    if (kind === "task") {
      const checked = extra && extra.checked ? " checked" : "";
      return `<div data-md-kind="task" class="ideas-md-taskline"><input contenteditable="false" type="checkbox"${checked}/><span class="ideas-md-content">${text || "<br/>"}</span></div>`;
    }
    return `<div data-md-kind="p" class="ideas-md-paragraph">${text || "<br/>"}</div>`;
  }

  function markdownToEditorHtml(md) {
    const lines = String(md || "").split(/\r?\n/);
    if (!lines.length) return editorLineHtml("p", "");
    return lines.map((raw) => {
      const line = raw.replace(/\s+$/, "");
      if (!line.trim()) return editorLineHtml("p", "");
      const h3m = line.match(/^###\s+(.*)$/);
      if (h3m) return editorLineHtml("h3", h3m[1]);
      const h2m = line.match(/^##\s+(.*)$/);
      if (h2m) return editorLineHtml("h2", h2m[1]);
      const h1m = line.match(/^#\s+(.*)$/);
      if (h1m) return editorLineHtml("h1", h1m[1]);
      const task = line.match(/^[-*]\s+\[( |x|X)\]\s*(.*)$/) || line.match(/^\[( |x|X)?\]\s*(.*)$/);
      if (task) return editorLineHtml("task", task[2] || "", { checked: (task[1] || "").toLowerCase() === "x" });
      const bullet = line.match(/^[-*]\s+(.*)$/) || line.match(/^[-*]\s+$/);
      if (bullet) return editorLineHtml("bullet", bullet[1] || "");
      const quote = line.match(/^>\s*(.*)$/);
      if (quote) return editorLineHtml("quote", quote[1] || "");
      return editorLineHtml("p", line);
    }).join("");
  }

  function inlineHtmlToMarkdown(node) {
    let out = "";
    for (const child of Array.from(node.childNodes || [])) {
      if (child.nodeType === Node.TEXT_NODE) { out += child.textContent || ""; continue; }
      if (child.nodeType !== Node.ELEMENT_NODE) continue;
      const tag = child.tagName.toLowerCase();
      if (tag === "br") { continue; }
      const inner = inlineHtmlToMarkdown(child);
      if (!inner) continue;
      if (tag === "strong" || tag === "b") out += `**${inner}**`;
      else if (tag === "em" || tag === "i") out += `*${inner}*`;
      else if (tag === "code") out += `\`${inner}\``;
      else if (tag === "a" && child.getAttribute("href")) out += `[${inner}](${child.getAttribute("href")})`;
      else out += inner;
    }
    return out.replace(/\u00a0/g, " ").replace(/\u200b/g, "");
  }

  function inlineFragmentToMarkdown(fragment) {
    const div = document.createElement("div");
    div.appendChild(fragment.cloneNode(true));
    return inlineHtmlToMarkdown(div);
  }

  function removeEmptyInlineFormatting(node) {
    if (!node) return;
    for (const child of Array.from(node.querySelectorAll("strong,b,em,i,code,a"))) {
      if (!String(child.textContent || "").replace(/\u200b/g, "").length) child.replaceWith(document.createTextNode(""));
    }
    if (!String(node.textContent || "").replace(/\u200b/g, "").length) node.innerHTML = "";
  }

  function editorToMarkdown(root) {
    if (!root) return "";
    const lines = [];
    const children = Array.from(root.children || []);
    for (const child of children) {
      const kind = child.getAttribute("data-md-kind") || "p";
      const contentNode = child.querySelector(".ideas-md-content") || child;
      const text = inlineHtmlToMarkdown(contentNode).trimEnd();
      if (kind === "h1") lines.push(text ? `# ${text}` : "# ");
      else if (kind === "h2") lines.push(text ? `## ${text}` : "## ");
      else if (kind === "h3") lines.push(text ? `### ${text}` : "### ");
      else if (kind === "quote") lines.push(text ? `> ${text}` : "> ");
      else if (kind === "bullet") lines.push(`- ${text}`);
      else if (kind === "task") {
        const checked = child.querySelector('input[type="checkbox"]')?.checked;
        lines.push(`- [${checked ? "x" : " "}] ${text}`);
      } else lines.push(text);
    }
    return lines.join("\n");
  }

  function normalizeMarkdownShortcuts(md) {
    return String(md || "").split(/\r?\n/).map((line) => {
      if (/^\[\]\s+/.test(line)) return line.replace(/^\[\]\s+/, "- [ ] ");
      if (/^\[ \]\s+/.test(line)) return line.replace(/^\[ \]\s+/, "- [ ] ");
      if (/^\[x\]\s+/i.test(line)) return line.replace(/^\[x\]\s+/i, "- [x] ");
      return line;
    }).join("\n");
  }

  function caretOffset(root) {
    const sel = window.getSelection && window.getSelection();
    if (!sel || sel.rangeCount === 0 || !root.contains(sel.anchorNode)) return null;
    const range = sel.getRangeAt(0).cloneRange();
    range.selectNodeContents(root);
    range.setEnd(sel.anchorNode, sel.anchorOffset);
    return range.toString().length;
  }

  function restoreCaret(root, offset) {
    restoreCaretInside(root, offset);
  }

  function restoreCaretInside(node, offset) {
    if (offset == null || !node) return;
    const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT);
    let remaining = offset;
    let textNode;
    while ((textNode = walker.nextNode())) {
      const len = textNode.textContent.length;
      if (remaining <= len) {
        const range = document.createRange();
        range.setStart(textNode, remaining);
        range.collapse(true);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        return;
      }
      remaining -= len;
    }
    if (node.focus) node.focus();
  }

  function blockContentNode(block) {
    return block?.querySelector?.(".ideas-md-content") || block;
  }

  function currentEditorPosition(root) {
    const sel = window.getSelection && window.getSelection();
    if (!sel || sel.rangeCount === 0 || !root.contains(sel.anchorNode)) return null;
    let node = sel.anchorNode;
    if (node.nodeType === Node.TEXT_NODE) node = node.parentElement;
    const block = node?.closest?.("[data-md-kind]");
    if (!block || !root.contains(block)) return null;
    const blocks = Array.from(root.children || []);
    const index = Math.max(0, blocks.indexOf(block));
    const content = blockContentNode(block);
    const range = sel.getRangeAt(0).cloneRange();
    range.selectNodeContents(content);
    try { range.setEnd(sel.anchorNode, sel.anchorOffset); }
    catch (_e) { return { index, offset: 0, kind: block.getAttribute("data-md-kind") || "p" }; }
    return { index, offset: range.toString().length, kind: block.getAttribute("data-md-kind") || "p" };
  }

  function markdownPrefixLength(line) {
    if (/^###\s+/.test(line)) return 4;
    if (/^##\s+/.test(line)) return 3;
    if (/^#\s+/.test(line)) return 2;
    if (/^>\s+/.test(line)) return 2;
    const task = line.match(/^[-*]\s+\[[ xX]\]\s*/);
    if (task) return task[0].length;
    const bullet = line.match(/^[-*]\s+/);
    if (bullet) return bullet[0].length;
    return 0;
  }

  function renderAndPlace(root, markdown, lineIndex, contentOffset) {
    root.innerHTML = markdownToEditorHtml(markdown || "");
    const block = (root.children || [])[Math.max(0, Math.min(lineIndex || 0, root.children.length - 1))];
    const content = blockContentNode(block || root);
    restoreCaretInside(content, contentOffset || 0);
  }

  let nextBlockSeq = 1;
  function makeBlockId() {
    nextBlockSeq += 1;
    return `ideas-block-${Date.now().toString(36)}-${nextBlockSeq.toString(36)}`;
  }

  function plainText(value) {
    return String(value || "").replace(/\u00a0/g, " ").replace(/\u200b/g, "").replace(/[\r\n]+/g, "");
  }

  function blockTextValue(type, value) {
    const text = String(value || "").replace(/\u00a0/g, " ").replace(/\u200b/g, "");
    return type === "code" ? text.replace(/\r\n/g, "\n") : text.replace(/[\r\n]+/g, "");
  }

  function inlineEditorHtml(value) {
    return inlineMd(value || "");
  }

  function clampIndent(value) {
    const n = Number.isFinite(value) ? value : parseInt(value || 0, 10);
    return Math.max(0, Math.min(12, n || 0));
  }

  function makeBlock(type, text, extra) {
    return {
      id: makeBlockId(),
      type: type || "paragraph",
      text: blockTextValue(type || "paragraph", text),
      checked: !!(extra && extra.checked),
      indent: clampIndent(extra && extra.indent),
    };
  }

  function markdownLineToBlock(raw) {
    const original = String(raw || "").replace(/\s+$/, "");
    const indentMatch = original.match(/^(\s*)/);
    const indent = Math.floor(((indentMatch && indentMatch[1]) || "").replace(/\t/g, "  ").length / 2);
    const line = original.trimStart();
    if (line === "###") return makeBlock("h3", "");
    if (line === "##") return makeBlock("h2", "");
    if (line === "#") return makeBlock("h1", "");
    if (line === "-" || line === "*") return makeBlock("bullet", "", { indent });
    let m = line.match(/^###\s+(.*)$/);
    if (m) return makeBlock("h3", m[1]);
    m = line.match(/^##\s+(.*)$/);
    if (m) return makeBlock("h2", m[1]);
    m = line.match(/^#\s+(.*)$/);
    if (m) return makeBlock("h1", m[1]);
    m = line.match(/^[-*]\s+\[( |x|X)\]\s*(.*)$/) || line.match(/^\[( |x|X)?\]\s*(.*)$/);
    if (m) return makeBlock("task", m[2] || "", { checked: (m[1] || "").toLowerCase() === "x", indent });
    m = line.match(/^[-*]\s+(.*)$/);
    if (m) return makeBlock("bullet", m[1] || "", { indent });
    m = line.match(/^>\s*(.*)$/);
    if (m) return makeBlock("quote", m[1] || "");
    return makeBlock("paragraph", original);
  }

  function markdownToBlocks(md) {
    const lines = String(md || "").split(/\r?\n/);
    const blocks = [];
    let inCode = false;
    let codeLines = [];
    for (const line of lines) {
      if (line.trim() === "```") {
        if (inCode) {
          blocks.push(makeBlock("code", codeLines.join("\n")));
          codeLines = [];
          inCode = false;
        } else {
          inCode = true;
          codeLines = [];
        }
        continue;
      }
      if (inCode) codeLines.push(line);
      else blocks.push(markdownLineToBlock(line));
    }
    if (inCode) blocks.push(makeBlock("code", codeLines.join("\n")));
    return blocks.length ? blocks : [makeBlock("paragraph", "")];
  }

  function blockToMarkdown(block) {
    if (!block) return "";
    const text = block.type === "code" ? String(block.text || "").replace(/\r\n/g, "\n") : plainText(block && block.text);
    const indent = (block.type === "bullet" || block.type === "task") ? "  ".repeat(clampIndent(block.indent)) : "";
    if (block.type === "h1") return text ? `# ${text}` : "# ";
    if (block.type === "h2") return text ? `## ${text}` : "## ";
    if (block.type === "h3") return text ? `### ${text}` : "### ";
    if (block.type === "bullet") return `${indent}- ${text}`;
    if (block.type === "task") return `${indent}- [${block.checked ? "x" : " "}] ${text}`;
    if (block.type === "quote") return text ? `> ${text}` : "> ";
    if (block.type === "code") return "```\n" + text + "\n```";
    return text;
  }

  function blocksToMarkdown(blocks) {
    const list = Array.isArray(blocks) && blocks.length ? blocks : [makeBlock("paragraph", "")];
    return list.map(blockToMarkdown).join("\n");
  }

  function blockDomId(id) {
    return `ideas-block-edit-${String(id || "").replace(/[^a-zA-Z0-9_-]/g, "")}`;
  }

  function textOffsetIn(node) {
    const sel = window.getSelection && window.getSelection();
    if (!sel || sel.rangeCount === 0 || !node || !node.contains(sel.anchorNode)) return 0;
    const range = sel.getRangeAt(0).cloneRange();
    range.selectNodeContents(node);
    try { range.setEnd(sel.anchorNode, sel.anchorOffset); }
    catch (_e) { return 0; }
    return range.toString().replace(/\u200b/g, "").length;
  }

  function markdownAroundSelection(node) {
    const sel = window.getSelection && window.getSelection();
    if (!sel || sel.rangeCount === 0 || !node || !node.contains(sel.anchorNode)) return { before: inlineHtmlToMarkdown(node), selected: "", after: "" };
    const range = sel.getRangeAt(0);
    const before = document.createRange();
    before.selectNodeContents(node);
    const after = document.createRange();
    after.selectNodeContents(node);
    try {
      before.setEnd(range.startContainer, range.startOffset);
      after.setStart(range.endContainer, range.endOffset);
    } catch (_e) {
      return { before: inlineHtmlToMarkdown(node), selected: "", after: "" };
    }
    return {
      before: inlineFragmentToMarkdown(before.cloneContents()),
      selected: inlineFragmentToMarkdown(range.cloneContents()),
      after: inlineFragmentToMarkdown(after.cloneContents()),
    };
  }

  function closestBlockFromNode(node, root) {
    let current = node;
    if (current && current.nodeType === Node.TEXT_NODE) current = current.parentElement;
    const block = current?.closest?.(".ideas-block[data-block-id]");
    return block && root && root.contains(block) ? block : null;
  }

  function placeCaret(node, offset) {
    if (!node) return;
    const editorRoot = node.closest?.(".ideas-block-editor");
    (editorRoot || node).focus();
    const wanted = Math.max(0, offset || 0);
    const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT);
    let remaining = wanted;
    let textNode;
    while ((textNode = walker.nextNode())) {
      const len = textNode.textContent.length;
      if (remaining <= len) {
        const range = document.createRange();
        range.setStart(textNode, remaining);
        range.collapse(true);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        return;
      }
      remaining -= len;
    }
    const range = document.createRange();
    range.selectNodeContents(node);
    range.collapse(false);
    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
  }

  function placeCaretAtEnd(node) {
    if (!node) return;
    node.focus();
    const range = document.createRange();
    range.selectNodeContents(node);
    range.collapse(false);
    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
  }

  function Toolbar({ onWrap, onBlockType }) {
    const items = [
      ["Text", () => onBlockType("paragraph")], ["H1", () => onBlockType("h1")], ["H2", () => onBlockType("h2")], ["H3", () => onBlockType("h3")],
      ["Bold", () => onWrap("**", "**")], ["Italic", () => onWrap("*", "*")], ["Code", () => onWrap("`", "`")], ["Code block", () => onBlockType("code")],
      ["List", () => onBlockType("bullet")], ["Task", () => onBlockType("task")], ["Quote", () => onBlockType("quote")],
    ];
    return h("div", { className: "ideas-toolbar" }, items.map(([label, fn]) => h(Button, { key: label, type: "button", variant: "outline", onClick: fn }, label)));
  }


  function LexicalToolbar() {
    const L = SDK.lexical;
    const [editor] = L.useLexicalComposerContext();
    const S = L.selection;
    const setBlock = useCallback((factory) => {
      editor.update(() => {
        const selection = S.$getSelection();
        if (S.$isRangeSelection(selection)) S.$setBlocksType(selection, factory);
      });
    }, [editor]);
    const button = (label, fn) => h(Button, { key: label, type: "button", variant: "outline", onMouseDown: (e) => e.preventDefault(), onClick: () => editor.focus(fn) }, label);
    return h("div", { className: "ideas-toolbar ideas-lexical-toolbar" }, [
      button("Text", () => { editor.dispatchCommand(L.commands.REMOVE_LIST_COMMAND, undefined); setBlock(() => S.$createParagraphNode()); }),
      button("H1", () => setBlock(() => S.$createHeadingNode("h1"))),
      button("H2", () => setBlock(() => S.$createHeadingNode("h2"))),
      button("H3", () => setBlock(() => S.$createHeadingNode("h3"))),
      button("Bold", () => editor.dispatchCommand(L.commands.FORMAT_TEXT_COMMAND, "bold")),
      button("Italic", () => editor.dispatchCommand(L.commands.FORMAT_TEXT_COMMAND, "italic")),
      button("Code", () => editor.dispatchCommand(L.commands.FORMAT_TEXT_COMMAND, "code")),
      button("Code block", () => setBlock(() => S.$createCodeNode())),
      button("List", () => editor.dispatchCommand(L.commands.INSERT_UNORDERED_LIST_COMMAND, undefined)),
      button("Task", () => editor.dispatchCommand(L.commands.INSERT_CHECK_LIST_COMMAND, undefined)),
      button("Quote", () => setBlock(() => S.$createQuoteNode())),
    ]);
  }

  function LexicalMarkdownEditor({ markdown, onMarkdownChange, editorKey }) {
    const L = SDK.lexical;
    if (!L) {
      return h("textarea", {
        className: "ideas-lexical-missing",
        value: markdown || "",
        onChange: (e) => onMarkdownChange(e.target.value),
      });
    }
    const markdownTransformers = useMemo(() => {
      const base = Array.isArray(L.TRANSFORMERS) ? L.TRANSFORMERS : [];
      // @lexical/markdown's catch-all TRANSFORMERS intentionally omits the
      // CHECK_LIST transformer in newer releases. The Ideas editor treats
      // Notion-style [] / [ ] / [x] + Space as first-class task shortcuts, so
      // include CHECK_LIST explicitly when the host SDK exposes it. Keep it
      // before unordered-list so "- [ ] " is parsed as a checklist, not a ul.
      const preferred = [L.CHECK_LIST, L.HEADING, L.UNORDERED_LIST, L.ORDERED_LIST].filter(Boolean);
      return preferred.concat(base.filter((t) => preferred.indexOf(t) === -1));
    }, [L]);
    const initialConfig = useMemo(() => ({
      namespace: "ideas-markdown-editor-" + String(editorKey || "draft"),
      theme: {
        root: "ideas-lexical-root",
        paragraph: "ideas-lexical-paragraph",
        heading: { h1: "ideas-lexical-h1", h2: "ideas-lexical-h2", h3: "ideas-lexical-h3" },
        quote: "ideas-lexical-quote",
        list: { ul: "ideas-lexical-ul", ol: "ideas-lexical-ol", listitem: "ideas-lexical-li", nested: { listitem: "ideas-lexical-nested-li" }, checklist: "ideas-lexical-checklist", listitemChecked: "ideas-lexical-checked", listitemUnchecked: "ideas-lexical-unchecked" },
        text: { bold: "ideas-lexical-bold", italic: "ideas-lexical-italic", code: "ideas-lexical-inline-code" },
        code: "ideas-lexical-codeblock",
        link: "ideas-lexical-link",
      },
      nodes: [
        L.nodes.HeadingNode,
        L.nodes.QuoteNode,
        L.nodes.ListNode,
        L.nodes.ListItemNode,
        L.nodes.CodeNode,
        L.nodes.CodeHighlightNode,
        L.nodes.LinkNode,
        L.nodes.AutoLinkNode,
      ],
      onError(error) { console.error("Ideas Lexical editor error", error); },
      editorState: () => L.$convertFromMarkdownString(markdown || "", markdownTransformers),
    }), [editorKey, markdownTransformers]);
    const handleChange = useCallback((editorState) => {
      editorState.read(() => {
        onMarkdownChange(L.$convertToMarkdownString(markdownTransformers));
      });
    }, [L, markdownTransformers, onMarkdownChange]);
    return h(L.LexicalComposer, { initialConfig },
      h("div", { className: "ideas-lexical-editor" },
        h(L.RichTextPlugin, {
          contentEditable: h(L.ContentEditable, { className: "ideas-lexical-content", spellCheck: true }),
          placeholder: h("div", { className: "ideas-lexical-placeholder" }, "Type markdown shortcuts naturally: #, ##, ###, -, [], ``` …"),
          ErrorBoundary: L.LexicalErrorBoundary,
        }),
        h(L.HistoryPlugin, null),
        L.ListPlugin ? h(L.ListPlugin, null) : null,
        L.CheckListPlugin ? h(L.CheckListPlugin, null) : null,
        L.TabIndentationPlugin ? h(L.TabIndentationPlugin, { maxIndent: 7 }) : null,
        h(L.MarkdownShortcutPlugin, { transformers: markdownTransformers }),
        h(L.OnChangePlugin, { onChange: handleChange, ignoreSelectionChange: true })
      )
    );
  }

  function BlockEditor({ blocks, onBlocksChange, onMarkdownChange, activeBlockId, setActiveBlockId }) {
    const rootRef = useRef(null);
    const dragIdRef = useRef(null);

    const focusBlock = useCallback((id, offset) => {
      window.setTimeout(() => placeCaret(document.getElementById(blockDomId(id)), offset), 0);
    }, []);

    const blockText = useCallback((block) => {
      const node = document.getElementById(blockDomId(block.id));
      if (!node) return blockTextValue(block.type, block.text);
      if (block.type === "code") return String(node.textContent || "").replace(/\r\n/g, "\n");
      removeEmptyInlineFormatting(node);
      return inlineHtmlToMarkdown(node);
    }, []);

    const snapshotBlocks = useCallback(() => blocks.map((b) => ({ ...b, text: blockText(b) })), [blocks, blockText]);

    const commit = useCallback((next, focus) => {
      const safe = next.length ? next : [makeBlock("paragraph", "")];
      onBlocksChange(safe);
      if (focus) focusBlock(focus.id, focus.offset);
    }, [focusBlock, onBlocksChange]);

    const syncMarkdownOnly = useCallback(() => {
      onMarkdownChange(blocksToMarkdown(snapshotBlocks()));
    }, [onMarkdownChange, snapshotBlocks]);

    const transformInlineNode = useCallback((node) => {
      if (!node) return false;
      removeEmptyInlineFormatting(node);
      const raw = inlineHtmlToMarkdown(node);
      if (!/(\*\*[^*]+\*\*|(^|[^*])\*[^*]+\*(?!\*)|`[^`]+`)/.test(raw)) return false;
      node.innerHTML = inlineEditorHtml(raw);
      placeCaretAtEnd(node);
      return true;
    }, []);

    const updateText = useCallback((block, _value) => {
      const node = document.getElementById(blockDomId(block.id));
      if (block.type !== "code") {
        removeEmptyInlineFormatting(node);
        transformInlineNode(node);
      }
      syncMarkdownOnly();
    }, [syncMarkdownOnly, transformInlineNode]);

    const splitBlock = useCallback((block) => {
      const currentBlocks = snapshotBlocks();
      const index = currentBlocks.findIndex((b) => b.id === block.id);
      if (index < 0) return;
      const currentBlock = currentBlocks[index];
      const node = document.getElementById(blockDomId(block.id));
      const parts = currentBlock.type === "code" ? null : markdownAroundSelection(node);
      const text = currentBlock.text;
      const offset = Math.max(0, Math.min(text.length, textOffsetIn(node)));
      const before = parts ? parts.before : text.slice(0, offset);
      const after = parts ? parts.after : text.slice(offset);
      const current = { ...currentBlock, text: before };
      if ((currentBlock.type === "bullet" || currentBlock.type === "task" || currentBlock.type === "quote") && !before && !after) {
        const paragraph = { ...currentBlock, type: "paragraph", text: "", checked: false, indent: 0 };
        const next = currentBlocks.slice();
        next[index] = paragraph;
        commit(next, { id: paragraph.id, offset: 0 });
        return;
      }
      const nextType = currentBlock.type === "h1" || currentBlock.type === "h2" || currentBlock.type === "h3" ? "paragraph" : currentBlock.type;
      const inserted = makeBlock(nextType, after, { checked: currentBlock.type === "task" ? false : currentBlock.checked, indent: currentBlock.indent });
      const next = currentBlocks.slice();
      next.splice(index, 1, current, inserted);
      commit(next, { id: inserted.id, offset: 0 });
    }, [commit, snapshotBlocks]);

    const mergeBackward = useCallback((block) => {
      const currentBlocks = snapshotBlocks();
      const index = currentBlocks.findIndex((b) => b.id === block.id);
      if (index < 0) return;
      const currentBlock = currentBlocks[index];
      if (currentBlock.type !== "paragraph" && !currentBlock.text) {
        const next = currentBlocks.slice();
        next[index] = { ...currentBlock, type: "paragraph", checked: false, indent: 0 };
        commit(next, { id: currentBlock.id, offset: 0 });
        return;
      }
      if (index === 0) return;
      const prev = currentBlocks[index - 1];
      const merged = { ...prev, text: prev.text + currentBlock.text };
      const next = currentBlocks.slice();
      next.splice(index - 1, 2, merged);
      commit(next, { id: prev.id, offset: prev.text.length });
    }, [commit, snapshotBlocks]);

    const deleteSelectionIfNeeded = useCallback((block) => {
      const sel = window.getSelection && window.getSelection();
      const root = rootRef.current;
      const node = document.getElementById(blockDomId(block.id));
      if (!sel || !root || sel.rangeCount === 0 || sel.isCollapsed || !root.contains(sel.anchorNode) || !root.contains(sel.focusNode)) return false;
      const startBlockEl = closestBlockFromNode(sel.anchorNode, root);
      const endBlockEl = closestBlockFromNode(sel.focusNode, root);
      if (!startBlockEl || !endBlockEl) return false;
      const currentBlocks = snapshotBlocks();
      const ids = currentBlocks.map((b) => b.id);
      let startIndex = ids.indexOf(startBlockEl.getAttribute("data-block-id"));
      let endIndex = ids.indexOf(endBlockEl.getAttribute("data-block-id"));
      if (startIndex < 0 || endIndex < 0) return false;
      if (startIndex > endIndex) { const tmp = startIndex; startIndex = endIndex; endIndex = tmp; }
      if (startIndex === endIndex && node && node.contains(sel.anchorNode) && node.contains(sel.focusNode)) {
        sel.getRangeAt(0).deleteContents();
        removeEmptyInlineFormatting(node);
        syncMarkdownOnly();
        return true;
      }
      const first = currentBlocks[startIndex];
      const last = currentBlocks[endIndex];
      const firstNode = document.getElementById(blockDomId(first.id));
      const lastNode = document.getElementById(blockDomId(last.id));
      const firstParts = first.type === "code" ? { before: String(firstNode?.textContent || "").slice(0, textOffsetIn(firstNode)) } : markdownAroundSelection(firstNode);
      const lastParts = last.type === "code" ? { after: String(lastNode?.textContent || "").slice(textOffsetIn(lastNode)) } : markdownAroundSelection(lastNode);
      const mergedText = blockTextValue(first.type, (firstParts.before || "") + (lastParts.after || ""));
      const replacement = { ...first, text: mergedText };
      const next = currentBlocks.slice(0, startIndex).concat([replacement], currentBlocks.slice(endIndex + 1));
      commit(next, { id: replacement.id, offset: String(firstParts.before || "").replace(/\*|`/g, "").length });
      return true;
    }, [commit, snapshotBlocks, syncMarkdownOnly]);

    const changeIndent = useCallback((block, delta) => {
      const currentBlocks = snapshotBlocks();
      const current = currentBlocks.find((b) => b.id === block.id);
      if (!current || (current.type !== "bullet" && current.type !== "task")) return false;
      const node = document.getElementById(blockDomId(block.id));
      const offset = textOffsetIn(node);
      const nextIndent = clampIndent((current.indent || 0) + delta);
      if (nextIndent === (current.indent || 0)) return true;
      const next = currentBlocks.map((b) => b.id === block.id ? { ...b, indent: nextIndent } : b);
      commit(next, { id: block.id, offset });
      return true;
    }, [commit, snapshotBlocks]);

    const handleKeyDown = useCallback((e, block) => {
      const node = document.getElementById(blockDomId(block.id));
      if (!node) return;
      if (e.key === "ArrowUp" || e.key === "ArrowDown") {
        const currentBlocks = snapshotBlocks();
        const index = currentBlocks.findIndex((b) => b.id === block.id);
        const target = currentBlocks[index + (e.key === "ArrowUp" ? -1 : 1)];
        if (target) {
          e.preventDefault();
          focusBlock(target.id, Math.min(textOffsetIn(node), blockTextValue(target.type, target.text).length));
        }
        return;
      }
      if (e.key === "Tab") {
        if (block.type === "bullet" || block.type === "task") {
          e.preventDefault();
          changeIndent(block, e.shiftKey ? -1 : 1);
        }
        return;
      }
      if (e.key === "Enter") {
        if (block.type === "code") {
          e.preventDefault();
          document.execCommand("insertText", false, "\n");
          syncMarkdownOnly();
          return;
        }
        e.preventDefault();
        if (deleteSelectionIfNeeded(block)) return;
        splitBlock(block);
        return;
      }
      if (e.key === "Backspace" || e.key === "Delete") {
        if (deleteSelectionIfNeeded(block)) { e.preventDefault(); return; }
      }
      if (e.key === "Backspace" && textOffsetIn(node) === 0) {
        e.preventDefault();
        mergeBackward(block);
        return;
      }
      if (e.key !== " " || !(window.getSelection && window.getSelection()?.isCollapsed)) return;
      const text = blockText(block);
      const offset = textOffsetIn(node);
      const before = text.slice(0, offset);
      const after = text.slice(offset);
      let nextType = null;
      let checked = false;
      if (before === "-" || before === "*") nextType = "bullet";
      else if (before === "###") nextType = "h3";
      else if (before === "##") nextType = "h2";
      else if (before === "#") nextType = "h1";
      else if (before === "```") nextType = "code";
      else if (before === ">") nextType = "quote";
      else if (before === "[]" || before === "[ ]") nextType = "task";
      else if (before.toLowerCase() === "[x]") { nextType = "task"; checked = true; }
      if (!nextType) return;
      e.preventDefault();
      // The active contenteditable node is intentionally left unmanaged during
      // normal typing so the caret doesn't jump on every keystroke. When a
      // markdown shortcut transforms the block, update that live DOM node
      // immediately; otherwise React will keep focus in the old text node and
      // the typed marker ("-", "[]", etc.) can remain visible with the caret
      // before it.
      node.textContent = after;
      const currentBlocks = snapshotBlocks();
      const next = currentBlocks.map((b) => b.id === block.id ? { ...b, type: nextType, text: blockTextValue(nextType, after), checked: nextType === "task" ? checked : false, indent: (nextType === "bullet" || nextType === "task") ? clampIndent(b.indent) : 0 } : b);
      commit(next, { id: block.id, offset: 0 });
    }, [blockText, changeIndent, commit, deleteSelectionIfNeeded, focusBlock, mergeBackward, snapshotBlocks, splitBlock]);

    const moveBlock = useCallback((fromId, toId) => {
      if (!fromId || !toId || fromId === toId) return;
      const currentBlocks = snapshotBlocks();
      const from = currentBlocks.findIndex((b) => b.id === fromId);
      const to = currentBlocks.findIndex((b) => b.id === toId);
      if (from < 0 || to < 0) return;
      const next = currentBlocks.slice();
      const [moved] = next.splice(from, 1);
      next.splice(to, 0, moved);
      commit(next, { id: moved.id, offset: moved.text.length });
    }, [commit, snapshotBlocks]);

    const renderBlock = (block) => {
      const isTask = block.type === "task";
      const isCode = block.type === "code";
      const indent = clampIndent(block.indent);
      const markerClass = `ideas-bullet-marker ideas-bullet-marker-${indent % 4}`;
      const className = `ideas-block ideas-block-${block.type || "paragraph"} ideas-indent-${Math.min(indent, 6)}` + (activeBlockId === block.id ? " active" : "");
      return h("div", {
        key: block.id,
        className,
        "data-block-id": block.id,
        onDragOver: (e) => { e.preventDefault(); e.currentTarget.classList.add("drag-over"); },
        onDragLeave: (e) => e.currentTarget.classList.remove("drag-over"),
        onDrop: (e) => { e.preventDefault(); e.currentTarget.classList.remove("drag-over"); moveBlock(e.dataTransfer.getData("text/plain") || dragIdRef.current, block.id); },
      },
        h("button", {
          type: "button",
          className: "ideas-block-handle",
          title: "Drag to move block",
          draggable: true,
          contentEditable: false,
          onMouseDown: () => setActiveBlockId(block.id),
          onDragStart: (e) => { dragIdRef.current = block.id; e.dataTransfer.setData("text/plain", block.id); e.dataTransfer.effectAllowed = "move"; },
          onDragEnd: () => { dragIdRef.current = null; },
        }, "⋮⋮"),
        block.type === "bullet" ? h("span", { className: markerClass, "aria-hidden": "true", contentEditable: false }) : null,
        isTask ? h("input", {
          className: "ideas-block-checkbox",
          type: "checkbox",
          checked: !!block.checked,
          onChange: (e) => {
            const currentBlocks = snapshotBlocks();
            commit(currentBlocks.map((b) => b.id === block.id ? { ...b, checked: e.target.checked } : b), { id: block.id, offset: blockText(block).length });
          },
        }) : null,
        h("div", {
          id: blockDomId(block.id),
          className: "ideas-block-content",
          suppressContentEditableWarning: true,
          spellCheck: !isCode,
          ref: (node) => {
            if (!node || document.activeElement === node) return;
            const wanted = block.text || "";
            if (isCode) {
              if (node.textContent !== wanted) node.textContent = wanted;
            } else {
              const html = inlineEditorHtml(wanted);
              if (node.innerHTML !== html) node.innerHTML = html;
            }
          },
          "data-placeholder": block.type === "h1" ? "Heading 1" : block.type === "h2" ? "Heading 2" : block.type === "h3" ? "Heading 3" : block.type === "bullet" ? "List item" : block.type === "task" ? "To-do" : block.type === "code" ? "Code" : block.type === "quote" ? "Quote" : "Type '/' for blocks, '-' for list, '[]' for task…",
          onFocus: () => setActiveBlockId(block.id),
          onClick: () => setActiveBlockId(block.id),
          onInput: (e) => updateText(block, e.currentTarget.textContent || ""),
          onKeyDown: (e) => handleKeyDown(e, block),
        })
      );
    };

    return h("div", {
      ref: rootRef,
      className: "ideas-block-editor",
      role: "textbox",
      "aria-multiline": "true",
      contentEditable: true,
      suppressContentEditableWarning: true,
      onInput: (e) => {
        const sel = window.getSelection && window.getSelection();
        const blockEl = sel && rootRef.current ? closestBlockFromNode(sel.anchorNode, rootRef.current) : closestBlockFromNode(e.target, rootRef.current);
        const block = blockEl ? blocks.find((b) => b.id === blockEl.getAttribute("data-block-id")) : null;
        if (block) updateText(block, blockEl.querySelector(".ideas-block-content")?.textContent || "");
        else syncMarkdownOnly();
      },
      onKeyDownCapture: (e) => {
        const sel = window.getSelection && window.getSelection();
        const blockEl = sel && rootRef.current ? closestBlockFromNode(sel.anchorNode, rootRef.current) : null;
        const block = blockEl ? blocks.find((b) => b.id === blockEl.getAttribute("data-block-id")) : null;
        if (block) handleKeyDown(e, block);
      },
      onKeyDown: (e) => {
        if (e.defaultPrevented) return;
        const sel = window.getSelection && window.getSelection();
        const blockEl = sel && rootRef.current ? closestBlockFromNode(sel.anchorNode, rootRef.current) : null;
        const block = blockEl ? blocks.find((b) => b.id === blockEl.getAttribute("data-block-id")) : null;
        if (block) handleKeyDown(e, block);
      },
      onClick: () => {
        const sel = window.getSelection && window.getSelection();
        const blockEl = sel && rootRef.current ? closestBlockFromNode(sel.anchorNode, rootRef.current) : null;
        if (blockEl) setActiveBlockId(blockEl.getAttribute("data-block-id"));
      },
    }, blocks.map(renderBlock));
  }

  function Editor({ idea, draft, setDraft, dirty, saving, onSave, onDelete, onDuplicate, onConvert }) {
    const [blocks, setBlocks] = useState([makeBlock("paragraph", "")]);
    const [activeBlockId, setActiveBlockId] = useState(null);
    const lastBodyRef = useRef(null);

    useEffect(() => {
      if (!draft) return;
      const body = draft.body || "";
      if (lastBodyRef.current === body) return;
      const parsed = markdownToBlocks(body);
      lastBodyRef.current = body;
      setBlocks(parsed);
      setActiveBlockId(parsed[0]?.id || null);
    }, [idea && idea.id, draft && draft.body]);

    const updateBlocks = useCallback((nextBlocks) => {
      const safe = nextBlocks.length ? nextBlocks : [makeBlock("paragraph", "")];
      const body = blocksToMarkdown(safe);
      lastBodyRef.current = body;
      setBlocks(safe);
      if (draft) setDraft({ ...draft, body });
    }, [draft, setDraft]);

    const updateMarkdownOnly = useCallback((body) => {
      lastBodyRef.current = body;
      if (draft) setDraft({ ...draft, body });
    }, [draft, setDraft]);

    const activeBlock = useMemo(() => blocks.find((b) => b.id === activeBlockId) || blocks[0], [blocks, activeBlockId]);

    const setActiveBlockType = useCallback((type) => {
      if (!activeBlock) return;
      updateBlocks(blocks.map((b) => b.id === activeBlock.id ? { ...b, type, text: blockTextValue(type, b.text), checked: type === "task" ? b.checked : false, indent: (type === "bullet" || type === "task") ? clampIndent(b.indent) : 0 } : b));
      window.setTimeout(() => placeCaret(document.getElementById(blockDomId(activeBlock.id)), activeBlock.text.length), 0);
    }, [activeBlock, blocks, updateBlocks]);

    const insertAround = useCallback((before, after) => {
      const sel = window.getSelection && window.getSelection();
      if (!sel || sel.rangeCount === 0 || !activeBlock) return;
      const node = document.getElementById(blockDomId(activeBlock.id));
      if (!node || !node.contains(sel.anchorNode)) return;
      const range = sel.getRangeAt(0);
      const selected = range.toString();
      const start = textOffsetIn(node);
      range.deleteContents();
      range.insertNode(document.createTextNode(before + selected + after));
      const text = activeBlock.type === "code" ? String(node.textContent || "") : inlineHtmlToMarkdown(node);
      if (activeBlock.type !== "code") node.innerHTML = inlineEditorHtml(text);
      updateBlocks(blocks.map((b) => b.id === activeBlock.id ? { ...b, text: blockTextValue(b.type, text) } : b));
      window.setTimeout(() => placeCaret(node, start + before.length + selected.length + after.length), 0);
    }, [activeBlock, blocks, updateBlocks]);

    if (!draft) {
      return h(Card, { className: "ideas-editor-card" }, h(CardContent, { className: "ideas-empty big" }, "Select an idea or create a new draft."));
    }

    return h(Card, { className: "ideas-editor-card" },
      h(CardHeader, null,
        h("div", { className: "ideas-row ideas-between" },
          h(CardTitle, null, idea && idea.id ? "Edit idea" : "New idea"),
          h("div", { className: "ideas-actions" },
            dirty ? h(Badge, { variant: "secondary" }, "Unsaved") : null,
            h(Button, { onClick: onSave, disabled: saving || !draft.title.trim() }, saving ? "Saving…" : "Save")
          )
        )
      ),
      h(CardContent, null,
        h("div", { className: "ideas-fields" },
          h("div", null, h(Label, null, "Title"), h(Input, { value: draft.title, onChange: (e) => setDraft({ ...draft, title: e.target.value }) })),
          h("div", null, h(Label, null, "Status"), h(Select, { value: draft.status, ...selectChangeHandler((value) => setDraft({ ...draft, status: value })) }, STATUSES.map((s) => h(SelectOption, { key: s, value: s }, s)))),
          h("div", null, h(Label, null, "Tags"), h(Input, { placeholder: "agent, product, ux", value: fmtTags(draft.tags), onChange: (e) => setDraft({ ...draft, tags: parseTags(e.target.value) }) }))
        ),
        h("div", { className: "idea-summary-field" }, h(Label, null, "One-line summary"), h(Input, { value: draft.summary || "", onChange: (e) => setDraft({ ...draft, summary: e.target.value }) })),
        h("div", { className: "ideas-editor-meta" },
          idea && idea.file_path ? h("span", null, `Markdown file: ${idea.file_path}`) : h("span", null, "Saved ideas become markdown files under $HERMES_HOME/ideas."),
          idea && idea.task_id ? h("span", null, `Kanban task: ${idea.task_id}`) : null
        ),
        h(LexicalMarkdownEditor, {
          key: (idea && idea.id) || "new-idea-editor",
          editorKey: (idea && idea.id) || "new-idea-editor",
          markdown: draft.body || "",
          onMarkdownChange: updateMarkdownOnly,
        }),
        h("div", { className: "ideas-editor-hint" }, "Lexical markdown editor: type shortcuts like '-', '[]', '#', '##', '###', '>' or '```' followed by Space. Inline **bold**, *italic*, and `code` edit as visible rich text and save back to Markdown."),
        h("div", { className: "ideas-bottom-actions" },
          idea && idea.id ? h(Button, { variant: "outline", onClick: onDuplicate }, "Duplicate") : null,
          idea && idea.id ? h(Button, { variant: "outline", onClick: onConvert }, "Create Kanban task") : null,
          idea && idea.id ? h(Button, { variant: "outline", onClick: onDelete }, "Delete") : null
        )
      )
    );
  }

  function IdeaList({ ideas, selectedId, onSelect, onNew, query, setQuery, statusFilter, setStatusFilter }) {
    return h(Card, { className: "ideas-sidebar" },
      h(CardHeader, null,
        h("div", { className: "ideas-row ideas-between" },
          h(CardTitle, null, "Ideas"),
          h(Button, { onClick: onNew }, "+ New")
        ),
        h(Input, { placeholder: "Search title or summary…", value: query, onChange: (e) => setQuery(e.target.value) }),
        h(Select, { value: statusFilter, ...selectChangeHandler(setStatusFilter) },
          h(SelectOption, { value: "" }, "All active statuses"),
          STATUSES.map((s) => h(SelectOption, { key: s, value: s }, s))
        )
      ),
      h(CardContent, { className: "ideas-list" },
        ideas.length === 0 ? h("div", { className: "ideas-empty" }, "No ideas yet for this project.") : null,
        ideas.map((idea) => h("button", {
          key: idea.id,
          className: "idea-list-item" + (idea.id === selectedId ? " selected" : ""),
          onClick: () => onSelect(idea.id),
        },
          h("div", { className: "ideas-row ideas-between" },
            h("strong", null, idea.title),
            h(Badge, { variant: idea.status === "converted" ? "default" : "secondary" }, idea.status)
          ),
          idea.summary ? h("p", null, idea.summary) : null,
          h("div", { className: "idea-meta" },
            (idea.tags || []).slice(0, 4).map((t) => h("span", { key: t }, `#${t}`)),
            h("span", null, idea.updated_at ? `Updated ${timeAgo(idea.updated_at)}` : "")
          )
        ))
      )
    );
  }

  function IdeasPage() {
    const [boards, setBoards] = useState([]);
    const [board, setBoard] = useState(readSelectedBoard());
    const [ideas, setIdeas] = useState([]);
    const [selectedId, setSelectedId] = useState(null);
    const [selected, setSelected] = useState(null);
    const [draft, setDraft] = useState(null);
    const [query, setQuery] = useState("");
    const [statusFilter, setStatusFilter] = useState("");
    const [loading, setLoading] = useState(false);
    const [saving, setSaving] = useState(false);
    const [confirmDelete, setConfirmDelete] = useState(false);
    const [deleting, setDeleting] = useState(false);
    const [error, setError] = useState("");

    const dirty = useMemo(() => {
      if (!draft) return false;
      if (!selected) return true;
      return ["title", "summary", "body", "status"].some((k) => (draft[k] || "") !== (selected[k] || "")) || fmtTags(draft.tags) !== fmtTags(selected.tags);
    }, [draft, selected]);

    const loadBoards = useCallback(() => {
      fetchJSON(`${API}/boards`).then((r) => {
        setBoards(r.boards || []);
        if (!board && r.current) setBoard(r.current);
      }).catch((e) => setError(String(e.message || e)));
    }, [board]);

    const loadIdeas = useCallback(() => {
      setLoading(true);
      const qs = new URLSearchParams();
      if (query.trim()) qs.set("q", query.trim());
      if (statusFilter) qs.set("status", statusFilter);
      fetchJSON(withBoard(`${API}/ideas?${qs.toString()}`, board))
        .then((r) => setIdeas(r.ideas || []))
        .catch((e) => setError(String(e.message || e)))
        .finally(() => setLoading(false));
    }, [board, query, statusFilter]);

    const loadIdea = useCallback((id) => {
      if (!id) return;
      fetchJSON(`${API}/ideas/${encodeURIComponent(id)}`).then((r) => {
        setSelected(r.idea);
        setDraft({ title: r.idea.title || "", summary: r.idea.summary || "", body: r.idea.body || "", status: r.idea.status || "draft", tags: r.idea.tags || [] });
      }).catch((e) => setError(String(e.message || e)));
    }, []);

    useEffect(() => { loadBoards(); }, [loadBoards]);
    useEffect(() => { writeSelectedBoard(board); setSelectedId(null); setSelected(null); setDraft(null); }, [board]);
    useEffect(() => { const t = setTimeout(loadIdeas, 150); return () => clearTimeout(t); }, [loadIdeas]);
    useEffect(() => { if (selectedId) loadIdea(selectedId); }, [selectedId, loadIdea]);

    function newIdea() {
      setSelected(null); setSelectedId(null); setDraft({ ...EMPTY_IDEA, title: "" });
    }
    function save() {
      if (!draft || !draft.title.trim()) return;
      setSaving(true); setError("");
      const payload = { title: draft.title, summary: draft.summary, body: draft.body, status: draft.status, tags: draft.tags };
      const req = selected && selected.id
        ? json("PUT", `${API}/ideas/${encodeURIComponent(selected.id)}`, payload)
        : json("POST", withBoard(`${API}/ideas`, board), payload);
      req.then((r) => {
        setSelected(r.idea); setSelectedId(r.idea.id);
        setDraft({ title: r.idea.title || "", summary: r.idea.summary || "", body: r.idea.body || "", status: r.idea.status || "draft", tags: r.idea.tags || [] });
        loadIdeas(); loadBoards();
      }).catch((e) => setError(String(e.message || e))).finally(() => setSaving(false));
    }
    function del() {
      if (!selected || !selected.id) return;
      setDeleting(true); setError("");
      json("DELETE", `${API}/ideas/${encodeURIComponent(selected.id)}?delete_file=true`).then(() => {
        setConfirmDelete(false);
        setSelected(null); setSelectedId(null); setDraft(null); loadIdeas(); loadBoards();
      }).catch((e) => setError(String(e.message || e))).finally(() => setDeleting(false));
    }
    function duplicate() {
      if (!selected) return;
      json("POST", `${API}/ideas/${encodeURIComponent(selected.id)}/duplicate`, {}).then((r) => {
        setSelectedId(r.idea.id); loadIdeas(); loadBoards();
      }).catch((e) => setError(String(e.message || e)));
    }
    function convert() {
      if (!selected) return;
      json("POST", `${API}/ideas/${encodeURIComponent(selected.id)}/task`, { triage: true, priority: 0 }).then((r) => {
        setSelected(r.idea); setDraft({ title: r.idea.title, summary: r.idea.summary || "", body: r.idea.body || "", status: r.idea.status, tags: r.idea.tags || [] });
        loadIdeas();
        window.alert(`Created Kanban task ${r.task_id} on board ${r.board}.`);
      }).catch((e) => setError(String(e.message || e)));
    }

    return h("div", { className: "ideas-page" },
      h("div", { className: "ideas-header" },
        h("div", null,
          h("h1", null, "Ideas"),
          h("p", null, "Draft markdown-rich project ideas tied to Kanban boards. Agents can later read the saved markdown files or convert ideas into board tasks.")
        ),
        h("div", { className: "ideas-board-picker" },
          h(Label, null, "Project board"),
          h(Select, { value: board || "default", ...selectChangeHandler(setBoard) },
            boards.map((b) => h(SelectOption, { key: b.slug, value: b.slug }, `${b.icon ? b.icon + " " : ""}${b.name || b.slug} (${b.idea_count || 0})`))
          )
        )
      ),
      error ? h("div", { className: "ideas-error" }, error) : null,
      loading ? h("div", { className: "ideas-loading" }, "Loading ideas…") : null,
      h("div", { className: "ideas-layout" },
        h(IdeaList, { ideas, selectedId, onSelect: setSelectedId, onNew: newIdea, query, setQuery, statusFilter, setStatusFilter }),
        h(Editor, { idea: selected, draft, setDraft, dirty, saving, onSave: save, onDelete: () => setConfirmDelete(true), onDuplicate: duplicate, onConvert: convert })
      ),
      confirmDelete && selected ? h("div", { className: "ideas-modal-backdrop", role: "presentation", onMouseDown: (e) => { if (e.target === e.currentTarget && !deleting) setConfirmDelete(false); } },
        h("div", { className: "ideas-confirm", role: "dialog", "aria-modal": "true", "aria-labelledby": "ideas-delete-title" },
          h("h3", { id: "ideas-delete-title" }, "Delete idea?"),
          h("p", null, `This will remove “${selected.title}” from the Ideas tracker and delete its markdown file from disk.`),
          selected.file_path ? h("p", { className: "ideas-confirm-path" }, selected.file_path) : null,
          h("div", { className: "ideas-confirm-actions" },
            h(Button, { variant: "outline", disabled: deleting, onClick: () => setConfirmDelete(false) }, "Cancel"),
            h(Button, { disabled: deleting, onClick: del }, deleting ? "Deleting…" : "Delete idea and file")
          )
        )
      ) : null
    );
  }

  REG.register("ideas", IdeasPage);
})();
