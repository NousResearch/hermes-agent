// In-app viewer: everything clicked in the dashboard opens here, not in a
// new tab. Two modes — READER (server-side text extraction, always works for
// articles) and EMBED (live iframe; some sites refuse framing, so an
// external-open escape hatch is always visible).

import { h, clear, hostOf } from "./utils.js";
import { api } from "./api.js";
import { showSummary } from "./summarize.js";

let overlay = null;

export function closeViewer() {
  overlay?.remove();
  overlay = null;
  document.body.classList.remove("viewer-open");
}

export function openViewer({ url, title = "", summary = "", source = "", mode = "reader" }) {
  closeViewer();
  const host = hostOf(url);
  let activeMode = mode;

  const body = h("div.viewer-body");
  const tabReader = h("button.viewer-tab", { type: "button" }, "READER");
  const tabEmbed = h("button.viewer-tab", { type: "button" }, "EMBED");

  const setMode = (next) => {
    activeMode = next;
    tabReader.setAttribute("aria-selected", String(next === "reader"));
    tabEmbed.setAttribute("aria-selected", String(next === "embed"));
    next === "reader" ? renderReader() : renderEmbed();
  };
  tabReader.addEventListener("click", () => setMode("reader"));
  tabEmbed.addEventListener("click", () => setMode("embed"));

  async function renderReader() {
    clear(body).append(h("div.widget-loading", {}, "EXTRACTING TEXT…"));
    let doc;
    try {
      doc = await api.reader(url);
    } catch (err) {
      doc = { source: "sample", blocks: [], note: err.message };
    }
    const article = h("article.viewer-article");
    article.append(h("h1.viewer-title", {}, doc.title || title || host));
    if (doc.blocks?.length) {
      for (const block of doc.blocks) {
        if (block.tag === "li") article.append(h("p.viewer-li", {}, "• " + block.text));
        else if (block.tag.startsWith("h")) article.append(h("h3", {}, block.text));
        else if (block.tag === "blockquote") article.append(h("blockquote", {}, block.text));
        else article.append(h("p", {}, block.text));
      }
    } else {
      if (summary) article.append(h("p.viewer-summary", {}, summary));
      article.append(
        h("p.muted", {}, doc.note || "No readable text could be extracted."),
        h("p.muted.small", {}, "Try EMBED mode, or open externally ↗."),
      );
    }
    if (source) article.append(h("p.viewer-source.muted.small", {}, `SOURCE: ${source} · ${host}`));
    clear(body).append(article);
  }

  function renderEmbed() {
    clear(body).append(
      h("div.viewer-embed-note.muted.small", {},
        "Rendering live site. Some sites refuse to be embedded — if this stays blank, use ↗."),
      h("iframe.viewer-frame", {
        src: url,
        title: title || host,
        referrerpolicy: "no-referrer",
        loading: "eager",
      }),
    );
  }

  overlay = h("div.viewer-backdrop", {
    onclick: (ev) => { if (ev.target === overlay) closeViewer(); },
  },
    h("div.viewer", { role: "dialog", "aria-modal": "true", "aria-label": title || host },
      h("header.viewer-head", {},
        h("span.viewer-mark", { "aria-hidden": "true" }, "◆"),
        h("div.viewer-heading", {},
          h("div.viewer-name", {}, title || host),
          h("div.viewer-url.muted.small", {}, url),
        ),
        h("div.viewer-tabs", { role: "tablist" }, tabReader, tabEmbed),
        h("button.btn.viewer-sum", {
          type: "button",
          title: "Summarize this page with AI",
          "aria-label": "Summarize this page",
          onclick: async () => {
            let content = summary || "";
            try {
              const doc = await api.reader(url);
              if (doc.blocks?.length) {
                content = doc.blocks.map((b) => b.text).join("\n");
              }
            } catch { /* fall back to feed summary */ }
            showSummary({
              kind: "article",
              title: title || host,
              content: content || `${title}\n${url}`,
            });
          },
        }, "∑"),
        h("a.btn.viewer-ext", {
          href: url, target: "_blank", rel: "noopener noreferrer",
          title: "Open in a new tab",
        }, "↗"),
        h("button.btn", {
          type: "button", "aria-label": "Close viewer", onclick: closeViewer,
        }, "✕"),
      ),
      body,
    ),
  );

  document.body.append(overlay);
  document.body.classList.add("viewer-open");
  setMode(activeMode);
}

document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape" && overlay) closeViewer();
});

/**
 * Intercept plain left-clicks on a link so it opens in the viewer instead;
 * modified clicks (ctrl/cmd/middle) keep native new-tab behavior.
 */
export function viewerLink(anchor, options) {
  anchor.addEventListener("click", (ev) => {
    if (ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.button !== 0) return;
    if (document.body.classList.contains("edit-mode")) return;
    ev.preventDefault();
    openViewer(options);
  });
  return anchor;
}
