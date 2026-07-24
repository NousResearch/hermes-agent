import { h, clear, uid } from "../utils.js";

function ymd(date) {
  const m = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  return `${date.getFullYear()}-${m}-${d}`;
}

export default {
  type: "calendar",
  title: "Calendar",
  icon: "📅",
  defaultSize: "m",

  render(body, ctx) {
    const { store } = ctx;
    const today = new Date();
    let viewYear = today.getFullYear();
    let viewMonth = today.getMonth();
    let selected = ymd(today);
    let external = []; // read-only events from subscribed ICS calendars

    const eventsOn = (dateStr) =>
      store.state.calendar.events.filter((ev) => ev.date === dateStr);
    const externalOn = (dateStr) => external.filter((ev) => ev.date === dateStr);
    // Open tasks with a due date surface on the calendar as read-only markers.
    const dueTasksOn = (dateStr) =>
      (store.state.tasks?.lists || []).flatMap((l) => l.items)
        .filter((t) => t.due === dateStr && !t.done);

    const loadExternal = async () => {
      try {
        const res = await ctx.api.icsEvents(90);
        external = res.events;
        draw();
      } catch { /* offline or none subscribed */ }
    };

    const draw = () => {
      const first = new Date(viewYear, viewMonth, 1);
      const monthName = first.toLocaleDateString(undefined, { month: "long", year: "numeric" });

      const nav = h("div.cal-nav", {},
        h("button.icon-btn", {
          type: "button", "aria-label": "Previous month",
          onclick: () => { viewMonth--; if (viewMonth < 0) { viewMonth = 11; viewYear--; } draw(); },
        }, "‹"),
        h("div.cal-title", {}, monthName),
        h("button.icon-btn", {
          type: "button", "aria-label": "Next month",
          onclick: () => { viewMonth++; if (viewMonth > 11) { viewMonth = 0; viewYear++; } draw(); },
        }, "›"),
      );

      const grid = h("div.cal-grid", { role: "grid" });
      const weekdayFmt = new Intl.DateTimeFormat(undefined, { weekday: "narrow" });
      // Week starts Monday.
      for (let i = 0; i < 7; i++) {
        grid.append(h("div.cal-head", { role: "columnheader" },
          weekdayFmt.format(new Date(2024, 0, i + 1)))); // 2024-01-01 was a Monday
      }
      const lead = (first.getDay() + 6) % 7; // days before the 1st
      for (let i = 0; i < lead; i++) grid.append(h("div.cal-cell.cal-empty"));
      const daysInMonth = new Date(viewYear, viewMonth + 1, 0).getDate();
      for (let day = 1; day <= daysInMonth; day++) {
        const dateStr = ymd(new Date(viewYear, viewMonth, day));
        const isToday = dateStr === ymd(today);
        const hasEvents = eventsOn(dateStr).length > 0;
        const hasExternal = externalOn(dateStr).length > 0;
        const hasDue = dueTasksOn(dateStr).length > 0;
        grid.append(
          h("button.cal-cell", {
            type: "button",
            role: "gridcell",
            class: [
              "cal-cell",
              isToday ? "cal-today" : "",
              dateStr === selected ? "cal-selected" : "",
            ].filter(Boolean).join(" "),
            "aria-label": dateStr + (hasEvents || hasExternal ? ", has events" : ""),
            onclick: () => { selected = dateStr; draw(); },
          },
            String(day),
            hasEvents ? h("span.cal-dot", { "aria-hidden": "true" }) : null,
            hasExternal && !hasEvents ? h("span.cal-dot.cal-dot-ext", { "aria-hidden": "true" }) : null,
            hasDue ? h("span.cal-dot.cal-dot-task", { "aria-hidden": "true" }) : null,
          ),
        );
      }

      const selDate = new Date(selected + "T12:00:00");
      const selLabel = selDate.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" });
      const dayEvents = eventsOn(selected);
      const dayExternal = externalOn(selected);

      const input = h("input.input", {
        type: "text",
        placeholder: `Add event on ${selLabel}…`,
        "aria-label": `Add event on ${selLabel}`,
      });
      const form = h("form.cal-form", {
        onsubmit: (ev) => {
          ev.preventDefault();
          const title = input.value.trim();
          if (!title) return;
          store.update((state) => {
            state.calendar.events.push({ id: uid(), date: selected, title });
          }, "calendar");
          draw();
        },
      }, input, h("button.btn.btn-primary", { type: "submit" }, "Add"));

      const list = h("ul.cal-events", { role: "list" },
        dayEvents.map((event) =>
          h("li.cal-event", {},
            h("span.cal-dot", { "aria-hidden": "true" }),
            h("span.cal-event-title", {}, event.title),
            h("button.icon-btn", {
              type: "button", "aria-label": `Delete event ${event.title}`, title: "Delete",
              onclick: () => {
                store.update((state) => {
                  state.calendar.events = state.calendar.events.filter((e) => e.id !== event.id);
                }, "calendar");
                draw();
              },
            }, "✕"),
          ),
        ),
        dayExternal.map((event) =>
          h("li.cal-event.cal-event-ext", {},
            h("span.cal-dot.cal-dot-ext", { "aria-hidden": "true" }),
            h("span.cal-event-title", {},
              event.time ? `${event.time} ` : "", event.title),
            h("span.muted.small.cal-event-cal", {}, event.calendar),
          ),
        ),
        dueTasksOn(selected).map((task) =>
          h("li.cal-event.cal-event-task", {},
            h("span.cal-dot.cal-dot-task", { "aria-hidden": "true" }),
            h("span.cal-event-title", {}, task.text),
            h("span.muted.small.cal-event-cal", {}, "task due"),
          ),
        ),
        !dayEvents.length && !dayExternal.length && !dueTasksOn(selected).length
          ? h("li.muted.small", {}, `Nothing on ${selLabel}.`) : null,
      );

      // Next few upcoming events across local + subscribed calendars.
      const upcoming = [
        ...store.state.calendar.events,
        ...external.map((e) => ({ ...e, ext: true })),
      ]
        .filter((event) => event.date >= ymd(today))
        .sort((a, b) => a.date.localeCompare(b.date) || (a.time || "").localeCompare(b.time || ""))
        .slice(0, 4);

      clear(body).append(nav, grid, form, list);
      if (upcoming.length) {
        body.append(
          h("div.cal-upcoming", {},
            h("div.muted.small", {}, "Upcoming"),
            upcoming.map((event) =>
              h("div.small", {},
                new Date(event.date + "T12:00:00").toLocaleDateString(undefined, { month: "short", day: "numeric" }),
                event.time ? ` ${event.time}` : "",
                " — ", event.title,
                event.ext ? h("span.muted.small", {}, ` (${event.calendar})`) : null),
            ),
          ),
        );
      }
    };

    ctx.onSummarize(() => ({
      kind: "calendar",
      title: "Events",
      content: [...external.map((e) => ({ ...e, title: `${e.title} [${e.calendar}]` })),
        ...store.state.calendar.events]
        .slice()
        .sort((a, b) => a.date.localeCompare(b.date))
        .map((e) => `${e.date}: ${e.title}`)
        .join("\n") || "No events scheduled.",
    }));
    ctx.onStore((topic) => {
      // redraw on calendar edits and on task edits (due-date markers) and
      // on a full state replace (sync adoption / restore).
      if (topic === "calendar-external" || topic === "tasks" ||
          topic === "tasks-external" || topic === "replace") draw();
    });
    window.addEventListener("hub:calendars-changed", loadExternal);
    draw();
    loadExternal();
    ctx.every(15 * 60_000, loadExternal);
  },
};
