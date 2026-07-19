// Focus timer — a minimalist Pomodoro. The countdown survives reloads and
// cross-device sync because only the target end-time (epoch ms) is stored, not
// a live tick; every device recomputes the remaining time from the clock.

import { h, clear, toast } from "../utils.js";

const PRESETS = [
  { label: "Focus 25", minutes: 25, mode: "focus" },
  { label: "Focus 50", minutes: 50, mode: "focus" },
  { label: "Break 5", minutes: 5, mode: "break" },
];

function today() {
  return new Date().toISOString().slice(0, 10);
}

function fmt(ms) {
  const total = Math.max(0, Math.round(ms / 1000));
  const m = String(Math.floor(total / 60)).padStart(2, "0");
  const s = String(total % 60).padStart(2, "0");
  return `${m}:${s}`;
}

export default {
  type: "focus",
  title: "Focus",
  icon: "◷",
  defaultSize: "s",

  render(body, ctx) {
    const { store } = ctx;
    if (!store.state.focus) {
      store.state.focus = { running: false, endsAt: null, remainingMs: 25 * 60000, minutes: 25, mode: "focus", sessions: [] };
    }
    const f = () => store.state.focus;

    const remaining = () => {
      const s = f();
      return s.running && s.endsAt ? s.endsAt - Date.now() : s.remainingMs;
    };

    const complete = () => {
      const s = f();
      store.update((state) => {
        if (state.focus.mode === "focus") {
          state.focus.sessions.push({ date: today(), minutes: state.focus.minutes });
          if (state.focus.sessions.length > 500) state.focus.sessions = state.focus.sessions.slice(-500);
        }
        state.focus.running = false;
        state.focus.endsAt = null;
        state.focus.remainingMs = state.focus.minutes * 60000;
      }, "focus");
      toast(s.mode === "focus" ? "Focus session complete ✓" : "Break over — back to it");
      draw();
    };

    const setPreset = (preset) => {
      store.update((state) => {
        state.focus.minutes = preset.minutes;
        state.focus.mode = preset.mode;
        state.focus.running = false;
        state.focus.endsAt = null;
        state.focus.remainingMs = preset.minutes * 60000;
      }, "focus");
      draw();
    };

    const start = () => {
      store.update((state) => {
        state.focus.running = true;
        state.focus.endsAt = Date.now() + state.focus.remainingMs;
      }, "focus");
      draw();
    };

    const pause = () => {
      store.update((state) => {
        state.focus.remainingMs = Math.max(0, state.focus.endsAt - Date.now());
        state.focus.running = false;
        state.focus.endsAt = null;
      }, "focus");
      draw();
    };

    const reset = () => {
      store.update((state) => {
        state.focus.running = false;
        state.focus.endsAt = null;
        state.focus.remainingMs = state.focus.minutes * 60000;
      }, "focus");
      draw();
    };

    const draw = () => {
      const s = f();
      if (s.running && remaining() <= 0) { complete(); return; }

      const sessionsToday = s.sessions.filter((x) => x.date === today());
      const minsToday = sessionsToday.reduce((a, x) => a + x.minutes, 0);

      const presetRow = h("div.focus-presets", {},
        PRESETS.map((p) => h("button.note-tab", {
          type: "button",
          "aria-pressed": String(!s.running && s.minutes === p.minutes && s.mode === p.mode),
          class: (!s.running && s.minutes === p.minutes && s.mode === p.mode)
            ? "note-tab focus-preset-on" : "note-tab",
          onclick: () => setPreset(p),
        }, p.label)),
      );

      const controls = h("div.focus-controls", {},
        h("button.btn.btn-primary", { type: "button", onclick: () => (s.running ? pause() : start()) },
          s.running ? "Pause" : (remaining() < s.minutes * 60000 ? "Resume" : "Start")),
        h("button.btn", { type: "button", onclick: reset }, "Reset"),
      );

      clear(body).append(
        presetRow,
        h("div.focus-clock", { class: s.mode === "break" ? "focus-clock focus-break" : "focus-clock" },
          fmt(remaining())),
        h("div.focus-mode.muted.small", {}, (s.mode === "break" ? "BREAK" : "FOCUS") + (s.running ? " · RUNNING" : "")),
        controls,
        h("div.focus-stats.muted.small", {},
          `${sessionsToday.length} session${sessionsToday.length === 1 ? "" : "s"} today · ${minsToday}m focused`),
      );
    };

    ctx.onSummarize(() => {
      const s = f();
      const byDay = {};
      for (const x of s.sessions) byDay[x.date] = (byDay[x.date] || 0) + x.minutes;
      const recent = Object.entries(byDay).sort().slice(-7)
        .map(([d, m]) => `${d}: ${m}m`).join("\n");
      return {
        kind: "focus timer stats",
        title: "Focus history",
        content: `Today: ${s.sessions.filter((x) => x.date === today()).length} sessions.\n${recent || "No sessions logged yet."}`,
      };
    });

    draw();
    ctx.every(1000, () => { if (f().running) draw(); });
    ctx.onStore((topic) => { if (topic === "replace" || topic === "import" || topic === "reset") draw(); });
  },
};
