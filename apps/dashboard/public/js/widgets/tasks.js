import { h, clear, uid, toast } from "../utils.js";

export default {
  type: "tasks",
  title: "Lists",
  icon: "✅",
  defaultSize: "m",

  render(body, ctx) {
    const { store } = ctx;

    const activeList = () => {
      const { lists, activeList: id } = store.state.tasks;
      return lists.find((l) => l.id === id) || lists[0];
    };

    const draw = () => {
      const { lists } = store.state.tasks;
      const list = activeList();

      const picker = h("select.select", {
        "aria-label": "Choose list",
        onchange: (ev) => {
          if (ev.target.value === "__new__") {
            const name = prompt("Name for the new list:");
            if (name?.trim()) {
              const id = uid();
              store.update((state) => {
                state.tasks.lists.push({ id, name: name.trim(), items: [] });
                state.tasks.activeList = id;
              }, "tasks");
            }
          } else {
            store.update((state) => { state.tasks.activeList = ev.target.value; }, "tasks");
          }
          draw();
        },
      },
        lists.map((l) => h("option", { value: l.id, selected: l.id === list.id }, l.name)),
        h("option", { value: "__new__" }, "+ New list…"),
      );

      const total = list.items.length;
      const done = list.items.filter((i) => i.done).length;
      const progress = h("div.task-progress", { "aria-label": `${done} of ${total} done` },
        h("div.task-progress-bar", { style: { width: total ? `${(done / total) * 100}%` : "0%" } }),
      );

      const input = h("input.input", {
        type: "text",
        placeholder: `Add to ${list.name}…`,
        "aria-label": `Add task to ${list.name}`,
      });
      const form = h("form.task-form", {
        onsubmit: (ev) => {
          ev.preventDefault();
          const text = input.value.trim();
          if (!text) return;
          store.update((state) => {
            state.tasks.lists.find((l) => l.id === list.id).items.push({ id: uid(), text, done: false });
          }, "tasks");
          draw();
        },
      }, input, h("button.btn.btn-primary", { type: "submit" }, "Add"));

      const items = h("ul.task-items", { role: "list" });
      for (const task of list.items) {
        const checkbox = h("input", {
          type: "checkbox", checked: task.done, id: `task-${task.id}`,
          onchange: () => {
            store.update((state) => {
              const t = state.tasks.lists.find((l) => l.id === list.id).items.find((i) => i.id === task.id);
              t.done = !t.done;
            }, "tasks");
            draw();
          },
        });
        items.append(
          h("li.task-item", { class: task.done ? "task-item task-done" : "task-item" },
            checkbox,
            h("label.task-text", { for: `task-${task.id}` }, task.text),
            h("button.icon-btn", {
              type: "button", "aria-label": `Delete “${task.text}”`, title: "Delete",
              onclick: () => {
                store.update((state) => {
                  const target = state.tasks.lists.find((l) => l.id === list.id);
                  target.items = target.items.filter((i) => i.id !== task.id);
                }, "tasks");
                draw();
              },
            }, "✕"),
          ),
        );
      }
      if (!list.items.length) {
        items.append(h("li.muted.small", {}, "Nothing here yet — add something above."));
      }

      const footer = h("div.task-footer", {},
        h("span.muted.small", {}, total ? `${done}/${total} done` : ""),
        done > 0
          ? h("button.link-btn", {
            type: "button",
            onclick: () => {
              store.update((state) => {
                const target = state.tasks.lists.find((l) => l.id === list.id);
                target.items = target.items.filter((i) => !i.done);
              }, "tasks");
              draw();
            },
          }, "Clear completed")
          : null,
        lists.length > 1
          ? h("button.link-btn", {
            type: "button",
            onclick: () => {
              if (!confirm(`Delete the list “${list.name}” and its ${total} item(s)?`)) return;
              store.update((state) => {
                state.tasks.lists = state.tasks.lists.filter((l) => l.id !== list.id);
                state.tasks.activeList = state.tasks.lists[0].id;
              }, "tasks");
              toast("List deleted");
              draw();
            },
          }, "Delete list")
          : null,
      );

      clear(body).append(picker, progress, form, items, footer);
    };

    ctx.onSummarize(() => {
      const list = activeList();
      return {
        kind: "to-do list",
        title: list.name,
        content: list.items.map((i) => `${i.done ? "[done]" : "[open]"} ${i.text}`).join("\n"),
      };
    });
    ctx.onStore((topic) => {
      if (topic === "tasks-external") draw();
    });
    draw();
  },
};
