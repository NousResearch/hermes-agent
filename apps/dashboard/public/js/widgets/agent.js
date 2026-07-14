// HERMES//AGENT — the dashboard's built-in assistant.
//
// Runs a client-driven agent loop: the server relays the conversation to
// Claude (or the local command parser); tool calls come back as content
// blocks and are executed HERE, against the browser's own data, then results
// are fed back until the turn ends. Nothing personal is stored server-side.

import { h, clear, toast } from "../utils.js";
import { executeAction, buildContext } from "../actions.js";
import { api } from "../api.js";
import { enableSystemAlerts } from "../notifications.js";

const MAX_LOOPS = 6;
const MAX_HISTORY = 60;

// Conversation state for the current page session (display history persists
// in the store; the raw API message list intentionally does not).
let apiMessages = [];

async function fullContext() {
  const context = buildContext();
  try {
    const [news, world] = await Promise.all([api.news("top", 8), api.worldstate()]);
    context.headlines = news.items.map((i) => i.title);
    context.worldstate = {
      score: world.overall.score,
      level: world.overall.level,
      watch: world.domains
        .filter((d) => d.level === "elevated" || d.level === "critical")
        .map((d) => d.name)
        .join(", ") || null,
    };
  } catch { /* context stays partial if feeds are down */ }
  return context;
}

export default {
  type: "agent",
  title: "Agent",
  icon: "◆",
  defaultSize: "m",

  render(body, ctx) {
    const { store } = ctx;
    if (!store.state.agent) store.state.agent = { history: [] };

    const log = h("div.agent-log", { role: "log", "aria-live": "polite" });
    const modeChip = h("span.agent-mode", {}, "…");
    const input = h("input.input.agent-input", {
      type: "text",
      placeholder: "Ask or command… e.g. “add task water plants”",
      "aria-label": "Message the agent",
      autocomplete: "off",
    });
    const sendBtn = h("button.btn.btn-primary", { type: "submit" }, "Send");

    const pushEntry = (entry) => {
      store.update((state) => {
        state.agent.history.push(entry);
        if (state.agent.history.length > MAX_HISTORY) {
          state.agent.history = state.agent.history.slice(-MAX_HISTORY);
        }
      }, "agent");
      renderLog();
    };

    const renderLog = () => {
      clear(log);
      const { history } = store.state.agent;
      if (!history.length) {
        log.append(h("div.agent-empty.muted", {},
          "Standing by. Try “brief me”, “add task …”, “open GitHub”, or ask anything."));
      }
      for (const entry of history) {
        const bubble = h(`div.agent-msg.agent-${entry.role}`, {});
        for (const line of (entry.text || "").split("\n")) {
          bubble.append(h("p.agent-line", {}, line));
        }
        if (entry.actions?.length) {
          bubble.append(h("div.agent-actions", {},
            entry.actions.map((a) => {
              const long = a.result.length > 140 || a.result.includes("\n");
              if (long && a.ok) {
                return h("pre.agent-tool-block", {}, a.result);
              }
              return h("span.agent-chip", { class: a.ok ? "agent-chip" : "agent-chip agent-chip-err" },
                `${a.ok ? "✓" : "✗"} ${a.result}`);
            }),
          ));
        }
        log.append(bubble);
      }
      log.scrollTop = log.scrollHeight;
    };

    let busy = false;
    const setBusy = (value) => {
      busy = value;
      sendBtn.disabled = value;
      input.disabled = value;
      form.classList.toggle("agent-busy", value);
      if (!value) input.focus();
    };

    async function runTurn(userText) {
      pushEntry({ role: "user", text: userText });
      apiMessages.push({ role: "user", content: userText });
      setBusy(true);
      const context = await fullContext();
      const texts = [];
      const actions = [];
      let mode = "local";
      try {
        for (let hop = 0; hop < MAX_LOOPS; hop++) {
          const res = await api.chat(apiMessages, hop === 0 ? context : {});
          mode = res.mode;
          apiMessages.push({ role: "assistant", content: res.content });
          const toolResults = [];
          for (const block of res.content) {
            if (block.type === "text" && block.text.trim()) texts.push(block.text.trim());
            if (block.type === "tool_use") {
              const outcome = await executeAction(block.name, block.input);
              actions.push(outcome);
              toolResults.push({
                type: "tool_result",
                tool_use_id: block.id,
                content: outcome.result,
                is_error: !outcome.ok || undefined,
              });
            }
          }
          if (res.stop_reason !== "tool_use") break;
          apiMessages.push({ role: "user", content: toolResults });
        }
        const replyText = texts.join("\n\n") || "Done.";
        pushEntry({ role: "assistant", text: replyText, actions, mode });
        speak(replyText);
      } catch (err) {
        pushEntry({ role: "assistant", text: `Something went wrong: ${err.message}`, actions });
      } finally {
        setBusy(false);
      }
    }

    async function runBriefing() {
      pushEntry({ role: "user", text: "Brief me." });
      setBusy(true);
      try {
        const res = await api.briefing(await fullContext());
        pushEntry({ role: "assistant", text: res.briefing, mode: res.mode });
      } catch (err) {
        pushEntry({ role: "assistant", text: `Briefing failed: ${err.message}` });
      } finally {
        setBusy(false);
      }
    }

    const submitText = (text) => {
      if (!text || busy) return;
      input.value = "";
      if (/^brief(\s+me)?\.?$/i.test(text)) runBriefing();
      else runTurn(text);
    };

    const form = h("form.agent-form", {
      onsubmit: (ev) => {
        ev.preventDefault();
        submitText(input.value.trim());
      },
    }, input, sendBtn);

    // -- voice: push-to-talk (Chrome and friends) + spoken replies ---------
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (Recognition) {
      let rec = null;
      const micBtn = h("button.btn.agent-mic", {
        type: "button", title: "Push to talk", "aria-label": "Voice input",
        onclick: () => {
          if (rec) { rec.stop(); return; }
          rec = new Recognition();
          rec.lang = navigator.language || "en-US";
          rec.interimResults = true;
          micBtn.classList.add("agent-mic-live");
          const oldPlaceholder = input.placeholder;
          input.placeholder = "Listening…";
          rec.onresult = (ev) => {
            input.value = [...ev.results].map((r) => r[0].transcript).join("");
          };
          const finish = () => {
            micBtn.classList.remove("agent-mic-live");
            input.placeholder = oldPlaceholder;
            rec = null;
            submitText(input.value.trim());
          };
          rec.onend = finish;
          rec.onerror = () => { input.value = ""; finish(); };
          rec.start();
        },
      }, "🎙");
      form.insertBefore(micBtn, sendBtn);
    }

    const speakOn = () => !!store.state.agent.speak;
    const speak = (text) => {
      if (!speakOn() || !("speechSynthesis" in window) || !text) return;
      speechSynthesis.cancel();
      speechSynthesis.speak(new SpeechSynthesisUtterance(text.slice(0, 500)));
    };

    const quick = h("div.agent-quick", {},
      h("button.link-btn", { type: "button", onclick: () => !busy && runBriefing() }, "▸ Morning briefing"),
      h("button.link-btn", {
        type: "button",
        onclick: () => !busy && runTurn("What should I focus on right now?"),
      }, "▸ Focus check"),
      h("button.link-btn", {
        type: "button",
        title: "Show standing automations",
        onclick: () => !busy && runTurn("list automations"),
      }, "▸ Automations"),
      ("speechSynthesis" in window) ? h("button.link-btn", {
        type: "button",
        title: "Read replies aloud",
        onclick: () => {
          store.update((state) => { state.agent.speak = !state.agent.speak; }, "agent");
          if (!store.state.agent.speak) speechSynthesis.cancel();
          toast(store.state.agent.speak ? "Voice replies on" : "Voice replies off");
        },
      }, "🔊 Voice") : null,
      h("button.link-btn", {
        type: "button",
        title: "Allow system notifications for automation alerts",
        onclick: async () => {
          const result = await enableSystemAlerts();
          toast(result === "granted"
            ? "System alerts enabled"
            : result === "unsupported"
              ? "This browser doesn't support notifications"
              : "Notifications not allowed");
        },
      }, "🔔 Alerts"),
      h("button.link-btn", {
        type: "button",
        onclick: () => {
          store.update((state) => { state.agent.history = []; }, "agent");
          apiMessages = [];
          renderLog();
        },
      }, "Clear"),
    );

    clear(body).append(
      h("div.agent-head-row", {},
        h("span.muted.small", {}, "SECURE CHANNEL"),
        modeChip,
      ),
      log, quick, form,
    );
    renderLog();

    api.assistantStatus().then((status) => {
      modeChip.textContent = status.mode === "claude"
        ? `CLAUDE · ${status.model}` : "LOCAL MODE";
      modeChip.title = status.hint || "Full Claude agent active";
      modeChip.className = status.mode === "claude" ? "agent-mode agent-mode-ai" : "agent-mode";
    }).catch(() => { modeChip.textContent = "OFFLINE"; });

    ctx.onSummarize(() => ({
      kind: "conversation with the dashboard agent",
      title: "Agent log",
      content: store.state.agent.history.map((e) => `${e.role}: ${e.text}`).join("\n"),
    }));
  },
};
