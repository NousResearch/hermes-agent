// "News sources" settings panel: manage topics and their RSS/Atom feeds.
// Works with any feed URL — blogs, YouTube channels (/feeds/videos.xml?…),
// subreddits (….rss), podcasts. Changes apply server-side for all devices.

import { h, clear, toast } from "./utils.js";
import { api } from "./api.js";

let panelEl = null;

function closeSources() {
  panelEl?.remove();
  panelEl = null;
}

document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape" && panelEl) closeSources();
});

function feedsChanged() {
  window.dispatchEvent(new CustomEvent("hub:feeds-changed"));
}

export async function openSources() {
  closeSources();
  const body = h("div.sum-body.sources-body", {}, h("div.widget-loading", {}, "LOADING SOURCES…"));
  panelEl = h("div.sum-backdrop", {
    onclick: (ev) => { if (ev.target === panelEl) closeSources(); },
  },
    h("div.sum-pop.sources-pop", { role: "dialog", "aria-label": "News sources" },
      h("header.sum-head", {},
        h("span.sum-title", {}, "NEWS SOURCES"),
        h("button.icon-btn", { type: "button", "aria-label": "Close", onclick: closeSources }, "✕"),
      ),
      body,
    ),
  );
  document.body.append(panelEl);

  const run = async (op, payload, okMsg) => {
    try {
      await api.feedsOp({ op, ...payload });
      if (okMsg) toast(okMsg);
      feedsChanged();
      await draw();
    } catch (err) {
      toast(err.message, "error");
    }
  };

  async function draw() {
    let config;
    try {
      config = await api.feeds();
    } catch (err) {
      clear(body).append(h("div.widget-error", {}, `Cannot load sources: ${err.message}`));
      return;
    }
    clear(body);

    for (const [topic, sources] of Object.entries(config.sources)) {
      const section = h("section.sources-topic", {},
        h("div.sources-topic-head", {},
          h("span.sources-topic-name", {}, topic),
          h("button.link-btn", {
            type: "button",
            onclick: () => {
              if (confirm(`Remove the “${topic}” topic and its ${sources.length} feed(s)?`)) {
                run("remove_topic", { name: topic }, `Topic “${topic}” removed`);
              }
            },
          }, "remove topic"),
        ),
      );
      for (const source of sources) {
        section.append(h("div.sources-row", {},
          h("span.sources-name", {}, source.name),
          h("span.sources-url.muted.small", {}, source.url),
          h("button.icon-btn", {
            type: "button", title: "Remove feed", "aria-label": `Remove ${source.name}`,
            onclick: () => run("remove_source", { topic, url: source.url }, "Feed removed"),
          }, "✕"),
        ));
      }
      if (!sources.length) {
        section.append(h("div.muted.small.sources-empty", {}, "No feeds yet — add one below."));
      }
      const nameInput = h("input.input", { type: "text", placeholder: "Feed name", "aria-label": `Feed name for ${topic}` });
      const urlInput = h("input.input", { type: "url", placeholder: "https://…/rss.xml", "aria-label": `Feed URL for ${topic}` });
      section.append(h("form.sources-add", {
        onsubmit: (ev) => {
          ev.preventDefault();
          run("add_source", { topic, name: nameInput.value, url: urlInput.value }, "Feed added");
        },
      }, nameInput, urlInput, h("button.btn.btn-primary", { type: "submit" }, "Add")));
      body.append(section);
    }

    const topicInput = h("input.input", { type: "text", placeholder: "New topic name (becomes a tab)", "aria-label": "New topic name" });
    body.append(
      h("form.sources-add.sources-newtopic", {
        onsubmit: (ev) => {
          ev.preventDefault();
          run("add_topic", { name: topicInput.value }, "Topic added");
        },
      }, topicInput, h("button.btn.btn-primary", { type: "submit" }, "Add topic")),
      h("div.sources-footer", {},
        h("span.muted.small", {}, "Any RSS/Atom URL works: sites, YouTube channels, subreddits (.rss), podcasts."),
        h("button.link-btn", {
          type: "button",
          onclick: () => {
            if (confirm("Restore the default topics and feeds?")) run("reset", {}, "Sources reset to defaults");
          },
        }, "Reset defaults"),
      ),
    );
  }

  await draw();
}
