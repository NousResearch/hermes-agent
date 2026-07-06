import { h, clear, uid, hueFor, hostOf, isDark, SERIES_LIGHT, SERIES_DARK, toast } from "../utils.js";
import { viewerLink } from "../viewer.js";

function monogram(name) {
  const palette = isDark() ? SERIES_DARK : SERIES_LIGHT;
  const color = palette[hueFor(name.toLowerCase())];
  return h("span.app-monogram", { style: { background: color } },
    name.trim().charAt(0).toUpperCase() || "?");
}

function linkForm(existing, onSave, onCancel) {
  const name = h("input.input", {
    type: "text", placeholder: "Name", value: existing?.name || "", required: true,
    "aria-label": "App name",
  });
  const url = h("input.input", {
    type: "url", placeholder: "https://…", value: existing?.url || "", required: true,
    "aria-label": "App URL",
  });
  const form = h("form.launcher-form", {
    onsubmit: (ev) => {
      ev.preventDefault();
      let value = url.value.trim();
      if (!/^https?:\/\//i.test(value)) value = "https://" + value;
      try {
        new URL(value);
      } catch {
        toast("That URL doesn't look valid", "error");
        return;
      }
      onSave({ name: name.value.trim() || hostOf(value), url: value });
    },
  },
    name, url,
    h("div.form-row", {},
      h("button.btn.btn-primary", { type: "submit" }, existing ? "Save" : "Add"),
      h("button.btn", { type: "button", onclick: onCancel }, "Cancel"),
    ),
  );
  return form;
}

export default {
  type: "launcher",
  title: "Apps",
  icon: "🚀",
  defaultSize: "m",

  render(body, ctx) {
    const { store } = ctx;
    let editing = null; // null | "new" | link id

    const draw = () => {
      const { links } = store.state.launcher;
      const grid = h("div.launcher-grid", { role: "list" });

      for (const link of links) {
        const tile = h("a.app-tile", {
          href: link.url, target: "_blank", rel: "noopener noreferrer",
          role: "listitem", title: `${link.name} — ${hostOf(link.url)}`,
        },
          monogram(link.name),
          h("span.app-name", {}, link.name),
        );
        viewerLink(tile, { url: link.url, title: link.name, mode: "embed" });
        if (store.state.editMode) {
          tile.addEventListener("click", (ev) => ev.preventDefault());
          tile.append(
            h("span.app-actions", {},
              h("button.icon-btn", {
                type: "button", title: "Edit", "aria-label": `Edit ${link.name}`,
                onclick: (ev) => { ev.preventDefault(); editing = link.id; draw(); },
              }, "✎"),
              h("button.icon-btn", {
                type: "button", title: "Remove", "aria-label": `Remove ${link.name}`,
                onclick: (ev) => {
                  ev.preventDefault();
                  store.update((state) => {
                    state.launcher.links = state.launcher.links.filter((l) => l.id !== link.id);
                  }, "launcher");
                  draw();
                },
              }, "✕"),
            ),
          );
        }
        grid.append(tile);

        if (editing === link.id) {
          grid.append(linkForm(link, (values) => {
            store.update((state) => {
              const target = state.launcher.links.find((l) => l.id === link.id);
              Object.assign(target, values);
            }, "launcher");
            editing = null;
            draw();
          }, () => { editing = null; draw(); }));
        }
      }

      const addBtn = h("button.app-tile.app-add", {
        type: "button", "aria-label": "Add app",
        onclick: () => { editing = "new"; draw(); },
      }, h("span.app-monogram.app-add-mono", {}, "+"), h("span.app-name", {}, "Add app"));
      grid.append(addBtn);

      clear(body).append(grid);
      if (editing === "new") {
        body.append(linkForm(null, (values) => {
          store.update((state) => {
            state.launcher.links.push({ id: uid(), ...values });
          }, "launcher");
          editing = null;
          draw();
        }, () => { editing = null; draw(); }));
      }
    };

    ctx.onStore((topic) => {
      if (topic === "editMode" || topic === "launcher-external") draw();
    });
    draw();
  },
};
