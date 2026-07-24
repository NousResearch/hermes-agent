import { h, clear, uid, timeAgo, debounce } from "../utils.js";

function titleOf(note) {
  const first = (note.text || "").split("\n")[0].trim();
  return first.slice(0, 40) || "Untitled";
}

export default {
  type: "notes",
  title: "Notes",
  icon: "📝",
  defaultSize: "m",

  render(body, ctx) {
    const { store } = ctx;

    const active = () => {
      const { items, activeNote } = store.state.notes;
      return items.find((n) => n.id === activeNote) || items[0] || null;
    };

    const draw = () => {
      const { items } = store.state.notes;
      const note = active();

      const tabs = h("div.note-tabs", { role: "tablist", "aria-label": "Notes" },
        items.map((n) =>
          h("button.note-tab", {
            type: "button",
            role: "tab",
            "aria-selected": String(note && n.id === note.id),
            onclick: () => {
              store.update((state) => { state.notes.activeNote = n.id; }, "notes");
              draw();
            },
          }, titleOf(n)),
        ),
        h("button.note-tab.note-new", {
          type: "button",
          onclick: () => {
            const id = uid();
            store.update((state) => {
              state.notes.items.push({ id, text: "", updated: new Date().toISOString() });
              state.notes.activeNote = id;
            }, "notes");
            draw();
          },
        }, "+ New"),
      );

      clear(body).append(tabs);
      if (!note) {
        body.append(h("div.muted", {}, "No notes — create one."));
        return;
      }

      const status = h("span.muted.small", {}, `Saved ${timeAgo(note.updated)}`);
      const save = debounce((text) => {
        store.update((state) => {
          const target = state.notes.items.find((n) => n.id === note.id);
          target.text = text;
          target.updated = new Date().toISOString();
        }, "notes");
        status.textContent = "Saved just now";
      }, 400);

      const pad = h("textarea.note-pad", {
        value: note.text,
        placeholder: "Type a note… first line becomes the title.",
        "aria-label": "Note text",
        oninput: (ev) => {
          status.textContent = "Saving…";
          save(ev.target.value);
        },
      });

      const footer = h("div.note-footer", {},
        status,
        h("button.link-btn", {
          type: "button",
          onclick: () => {
            if (!confirm(`Delete note “${titleOf(note)}”?`)) return;
            store.update((state) => {
              state.notes.items = state.notes.items.filter((n) => n.id !== note.id);
              state.notes.activeNote = state.notes.items[0]?.id ?? null;
            }, "notes");
            draw();
          },
        }, "Delete note"),
      );

      body.append(pad, footer);
    };

    ctx.onSummarize(() => {
      const note = active();
      return note && {
        kind: "note",
        title: titleOf(note),
        content: note.text,
      };
    });
    ctx.onStore((topic) => {
      if (topic === "notes-external") draw();
    });
    draw();
  },
};
