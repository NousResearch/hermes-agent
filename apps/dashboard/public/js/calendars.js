// "Calendar feeds" settings panel: subscribe to read-only ICS calendars
// (Google/Apple/Outlook all provide a private .ics / webcal address).

import { h, clear, toast } from "./utils.js";
import { api } from "./api.js";

let panelEl = null;

function closePanel() {
  panelEl?.remove();
  panelEl = null;
}

document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape" && panelEl) closePanel();
});

function changed() {
  window.dispatchEvent(new CustomEvent("hub:calendars-changed"));
}

export async function openCalendars() {
  closePanel();
  const body = h("div.sum-body.sources-body", {}, h("div.widget-loading", {}, "LOADING CALENDARS…"));
  panelEl = h("div.sum-backdrop", {
    onclick: (ev) => { if (ev.target === panelEl) closePanel(); },
  },
    h("div.sum-pop.sources-pop", { role: "dialog", "aria-label": "Calendar feeds" },
      h("header.sum-head", {},
        h("span.sum-title", {}, "CALENDAR FEEDS"),
        h("button.icon-btn", { type: "button", "aria-label": "Close", onclick: closePanel }, "✕"),
      ),
      body,
    ),
  );
  document.body.append(panelEl);

  async function draw() {
    let data;
    try {
      data = await api.calendars();
    } catch (err) {
      clear(body).append(h("div.widget-error", {}, `Cannot load calendars: ${err.message}`));
      return;
    }
    clear(body);

    for (const cal of data.calendars) {
      body.append(h("div.sources-row", {},
        h("span.sources-name", {}, cal.name),
        h("span.sources-url.muted.small", {}, cal.url),
        h("button.icon-btn", {
          type: "button", title: "Unsubscribe", "aria-label": `Unsubscribe ${cal.name}`,
          onclick: async () => {
            try {
              await api.calendarsOp({ op: "remove", url: cal.url });
              toast("Calendar removed");
              changed();
              draw();
            } catch (err) {
              toast(err.message, "error");
            }
          },
        }, "✕"),
      ));
    }
    if (!data.calendars.length) {
      body.append(h("div.muted.small.sources-empty", {},
        "No calendar subscriptions yet."));
    }

    const nameInput = h("input.input", { type: "text", placeholder: "Name (e.g. Personal)", "aria-label": "Calendar name" });
    const urlInput = h("input.input", { type: "url", placeholder: "https://…/basic.ics or webcal://…", "aria-label": "Calendar URL" });
    body.append(
      h("form.sources-add", {
        onsubmit: async (ev) => {
          ev.preventDefault();
          try {
            await api.calendarsOp({ op: "add", name: nameInput.value, url: urlInput.value });
            toast("Calendar subscribed");
            changed();
            draw();
          } catch (err) {
            toast(err.message, "error");
          }
        },
      }, nameInput, urlInput, h("button.btn.btn-primary", { type: "submit" }, "Subscribe")),
      h("p.muted.small.sources-empty", {},
        "Read-only: events appear in the Calendar widget and briefings. ",
        "Google Calendar: Settings → your calendar → “Secret address in iCal format”. ",
        "Apple: Calendar → Share → Public Calendar. Outlook: Settings → Shared calendars → ICS."),
    );
  }

  await draw();
}
