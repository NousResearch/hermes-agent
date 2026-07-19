import { h, clear, uid, toast } from "../utils.js";

const PRIORITY_RANK = { high: 0, normal: 1, low: 2 };

function ymd(date) {
  const m = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  return `${date.getFullYear()}-${m}-${d}`;
}

// Parse inline tokens from a task's text: "!high"/"!low"/"!med" set priority;
// "@2026-07-20", "@today", "@tomorrow" (or "by <date>") set a due date. Tokens
// are stripped from the stored text. Returns {text, due, priority}.
export function parseTaskInput(raw) {
  let text = raw;
  let priority;
  let due;
  text = text.replace(/(?:^|\s)!(high|hi|urgent|low|med|medium|normal)\b/i, (_, p) => {
    const k = p.toLowerCase();
    priority = (k === "hi" || k === "urgent") ? "high"
      : (k === "med" || k === "medium" || k === "normal") ? "normal" : k;
    return "";
  });
  text = text.replace(/(?:^|\s)(?:@|by\s+)(\d{4}-\d{2}-\d{2}|today|tomorrow)\b/i, (_, d) => {
    const k = d.toLowerCase();
    if (k === "today") due = ymd(new Date());
    else if (k === "tomorrow") due = ymd(new Date(Date.now() + 86400000));
    else due = d;
    return "";
  });
  return { text: text.replace(/\s+/g, " ").trim(), due, priority };
}

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
        placeholder: `Add to ${list.name}…  (!high  @2026-07-20)`,
        title: "Tip: add !high/!low for priority and @YYYY-MM-DD (or @tomorrow) for a due date",
        "aria-label": `Add task to ${list.name}`,
      });
      const form = h("form.task-form", {
        onsubmit: (ev) => {
          ev.preventDefault();
          const raw = input.value.trim();
          if (!raw) return;
          const { text, due, priority } = parseTaskInput(raw);
          if (!text) return;
          store.update((state) => {
            const item = { id: uid(), text, done: false };
            if (due) item.due = due;
            if (priority) item.priority = priority;
            state.tasks.lists.find((l) => l.id === list.id).items.push(item);
          }, "tasks");
          draw();
        },
      }, input, h("button.btn.btn-primary", { type: "submit" }, "Add"));

      const todayStr = ymd(new Date());
      const dueChip = (task) => {
        if (!task.due || task.done) return null;
        const overdue = task.due < todayStr;
        const label = task.due === todayStr ? "today"
          : task.due === ymd(new Date(Date.now() + 86400000)) ? "tomorrow"
          : new Date(task.due + "T12:00:00").toLocaleDateString(undefined, { month: "short", day: "numeric" });
        return h("span.task-due", {
          class: overdue ? "task-due task-due-overdue" : "task-due",
          title: `Due ${task.due}${overdue ? " (overdue)" : ""}`,
        }, overdue ? `⚠ ${label}` : label);
      };

      // Open tasks sort by priority then due date; done tasks sink to the
      // bottom in their original order. Sorting is stable on a copy so the
      // stored list order is untouched.
      const ordered = list.items.map((t, i) => [t, i]).sort((a, b) => {
        if (a[0].done !== b[0].done) return a[0].done ? 1 : -1;
        const pa = PRIORITY_RANK[a[0].priority] ?? 1;
        const pb = PRIORITY_RANK[b[0].priority] ?? 1;
        if (pa !== pb) return pa - pb;
        const da = a[0].due || "9999-99-99";
        const db = b[0].due || "9999-99-99";
        if (da !== db) return da < db ? -1 : 1;
        return a[1] - b[1];
      }).map(([t]) => t);

      const items = h("ul.task-items", { role: "list" });
      for (const task of ordered) {
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
        const cls = ["task-item"];
        if (task.done) cls.push("task-done");
        if (task.priority && !task.done) cls.push(`task-prio-${task.priority}`);
        items.append(
          h("li.task-item", { class: cls.join(" ") },
            checkbox,
            h("label.task-text", { for: `task-${task.id}` }, task.text),
            dueChip(task),
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
