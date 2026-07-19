import { h, clear } from "../utils.js";

function greeting(hour) {
  if (hour < 5) return "Burning the midnight oil";
  if (hour < 12) return "Good morning";
  if (hour < 17) return "Good afternoon";
  if (hour < 21) return "Good evening";
  return "Good night";
}

export default {
  type: "clock",
  title: "Clock",
  icon: "🕒",
  defaultSize: "s",

  render(body, ctx) {
    const timeEl = h("div.clock-time", { "aria-label": "current time" });
    const dateEl = h("div.clock-date");
    const greetEl = h("div.clock-greeting");
    clear(body).append(h("div.clock", {}, greetEl, timeEl, dateEl));

    const tick = () => {
      const now = new Date();
      timeEl.textContent = now.toLocaleTimeString(undefined, {
        hour: "2-digit",
        minute: "2-digit",
      });
      dateEl.textContent = now.toLocaleDateString(undefined, {
        weekday: "long",
        month: "long",
        day: "numeric",
      });
      greetEl.textContent = greeting(now.getHours());
    };
    tick();
    ctx.every(30_000, tick);
  },
};
