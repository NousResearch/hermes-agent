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

    const eventsOn = (dateStr) =>
      store.state.calendar.events.filter((ev) => ev.date === dateStr);

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
        grid.append(
          h("button.cal-cell", {
            type: "button",
            role: "gridcell",
            class: [
              "cal-cell",
              isToday ? "cal-today" : "",
              dateStr === selected ? "cal-selected" : "",
            ].filter(Boolean).join(" "),
            "aria-label": dateStr + (hasEvents ? ", has events" : ""),
            onclick: () => { selected = dateStr; draw(); },
          },
            String(day),
            hasEvents ? h("span.cal-dot", { "aria-hidden": "true" }) : null,
          ),
        );
      }

      const selDate = new Date(selected + "T12:00:00");
      const selLabel = selDate.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" });
      const dayEvents = eventsOn(selected);

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
        !dayEvents.length ? h("li.muted.small", {}, `Nothing on ${selLabel}.`) : null,
      );

      // Next few upcoming events across the whole calendar.
      const upcoming = store.state.calendar.events
        .filter((event) => event.date >= ymd(today))
        .sort((a, b) => a.date.localeCompare(b.date))
        .slice(0, 3);

      clear(body).append(nav, grid, form, list);
      if (upcoming.length) {
        body.append(
          h("div.cal-upcoming", {},
            h("div.muted.small", {}, "Upcoming"),
            upcoming.map((event) =>
              h("div.small", {},
                new Date(event.date + "T12:00:00").toLocaleDateString(undefined, { month: "short", day: "numeric" }),
                " — ", event.title),
            ),
          ),
        );
      }
    };

    ctx.onSummarize(() => ({
      kind: "calendar",
      title: "Events",
      content: store.state.calendar.events
        .slice()
        .sort((a, b) => a.date.localeCompare(b.date))
        .map((e) => `${e.date}: ${e.title}`)
        .join("\n") || "No events scheduled.",
    }));
    ctx.onStore((topic) => {
      if (topic === "calendar-external") draw();
    });
    draw();
  },
};
