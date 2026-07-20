// Claude & AI Hub — a curated directory of learning material and tools for AI,
// Claude, Claude Code, skills, plugins, connectors and vibe coding. Offline
// (static, hand-picked links) with a rotating "resource of the day". Opens links
// in the in-app viewer.

import { h, clear } from "../utils.js";
import { viewerLink } from "../viewer.js";

// Categories → curated resources. url opens in the viewer.
const CATEGORIES = [
  ["Courses", [
    ["Anthropic Academy (Skilljar)", "Free official Anthropic courses — the ones you enjoyed.", "https://anthropic.skilljar.com/"],
    ["Anthropic Academy hub", "Guided learning paths for building with Claude.", "https://www.anthropic.com/learn"],
    ["Prompt engineering guide", "Official interactive prompting tutorial.", "https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview"],
    ["Prompt engineering interactive tutorial", "Hands-on notebook course (GitHub).", "https://github.com/anthropics/prompt-eng-interactive-tutorial"],
    ["Building effective agents", "Anthropic's guide to agent design patterns.", "https://www.anthropic.com/engineering/building-effective-agents"],
    ["DeepLearning.AI short courses", "Free short courses, several with Anthropic.", "https://www.deeplearning.ai/short-courses/"],
    ["Hugging Face learn", "Open courses on LLMs, agents & more.", "https://huggingface.co/learn"],
    ["fast.ai — Practical Deep Learning", "Free top-to-bottom DL course.", "https://course.fast.ai/"],
    ["Google ML Crash Course", "Fundamentals with exercises.", "https://developers.google.com/machine-learning/crash-course"],
  ]],
  ["Claude Code", [
    ["Claude Code docs", "Install, workflows, config, IDE integrations.", "https://docs.claude.com/en/docs/claude-code/overview"],
    ["Claude Code on the web", "Cloud sessions, environments, triggers.", "https://code.claude.com/docs/en/claude-code-on-the-web"],
    ["Claude Code best practices", "Anthropic's engineering guidance.", "https://www.anthropic.com/engineering/claude-code-best-practices"],
    ["Common workflows", "Recipes for real tasks.", "https://docs.claude.com/en/docs/claude-code/common-workflows"],
  ]],
  ["Skills · Plugins · Connectors", [
    ["Agent Skills", "What skills are and how to build them.", "https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview"],
    ["anthropics/skills (GitHub)", "Official example skills to learn from.", "https://github.com/anthropics/skills"],
    ["Claude Code plugins", "Package commands, agents & hooks.", "https://docs.claude.com/en/docs/claude-code/plugins"],
    ["Model Context Protocol (MCP)", "The open standard for connectors/tools.", "https://modelcontextprotocol.io/"],
    ["Connectors & remote MCP", "Connect Claude to your tools & data.", "https://docs.claude.com/en/docs/agents-and-tools/mcp"],
  ]],
  ["Build & API", [
    ["Claude API — get started", "First call, models, parameters.", "https://docs.claude.com/en/docs/get-started"],
    ["Anthropic Cookbook (GitHub)", "Runnable notebooks & patterns.", "https://github.com/anthropics/anthropic-cookbook"],
    ["Agent SDK", "Build custom agents on Claude.", "https://docs.claude.com/en/api/agent-sdk/overview"],
    ["Tool use / function calling", "Give Claude tools.", "https://docs.claude.com/en/docs/build-with-claude/tool-use"],
    ["Prompt caching", "Cut cost & latency on long prompts.", "https://docs.claude.com/en/docs/build-with-claude/prompt-caching"],
  ]],
  ["Models & updates", [
    ["Release notes", "What changed across Claude & tooling.", "https://docs.claude.com/en/release-notes/overview"],
    ["Models overview", "Capabilities, context, pricing.", "https://docs.claude.com/en/docs/about-claude/models/overview"],
    ["Anthropic news", "Announcements & research.", "https://www.anthropic.com/news"],
    ["Anthropic research", "Papers & interpretability.", "https://www.anthropic.com/research"],
  ]],
  ["Community & vibe coding", [
    ["r/ClaudeAI", "Community tips, builds & news.", "https://www.reddit.com/r/ClaudeAI/"],
    ["Hacker News", "Front page of tech & AI.", "https://news.ycombinator.com/"],
    ["Awesome Claude Code (GitHub)", "Curated list of tools & resources.", "https://github.com/hesreallyhim/awesome-claude-code"],
    ["Product Hunt", "New AI products daily.", "https://www.producthunt.com/topics/artificial-intelligence"],
  ]],
];

const ALL = CATEGORIES.flatMap(([cat, items]) => items.map((i) => [cat, ...i]));

export default {
  type: "ailearn",
  title: "Claude & AI Hub",
  icon: "✦",
  defaultSize: "l",

  render(body, ctx) {
    const { store } = ctx;
    let cat = store.state.ailearn?.cat || CATEGORIES[0][0];

    const dayIndex = () => {
      const now = new Date();
      const start = new Date(now.getFullYear(), 0, 0);
      return Math.floor((now - start) / 86400000);
    };

    const draw = () => {
      const pick = ALL[dayIndex() % ALL.length];
      const chips = h("div.ail-cats", {}, CATEGORIES.map(([c]) =>
        h("button.ail-chip", {
          type: "button", class: c === cat ? "ail-chip ail-chip-on" : "ail-chip",
          onclick: () => { cat = c; store.update((s) => { s.ailearn = { cat }; }, "ailearn"); draw(); },
        }, c)));

      const items = (CATEGORIES.find(([c]) => c === cat) || CATEGORIES[0])[1];
      const list = h("div.ail-list", {}, items.map(([title, desc, url]) =>
        viewerLink(
          h("a.ail-item", { href: url, target: "_blank", rel: "noopener noreferrer" },
            h("div.ail-item-title", {}, title),
            h("div.muted.small.ail-item-desc", {}, desc)),
          { url, title, source: cat, mode: "embed" })));

      const daily = viewerLink(
        h("a.ail-daily", { href: pick[3], target: "_blank", rel: "noopener noreferrer" },
          h("span.ail-daily-label", {}, "★ TODAY'S PICK"),
          h("div.ail-daily-title", {}, pick[1]),
          h("div.muted.small", {}, `${pick[0]} · ${pick[2]}`)),
        { url: pick[3], title: pick[1], source: pick[0], mode: "embed" });

      clear(body).append(daily, chips, list);
    };

    ctx.onRefresh(draw);
    draw();
  },
};
