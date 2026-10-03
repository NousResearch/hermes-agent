"use client";

import { useState } from "react";
import { cn } from "../utils/cn";
import { MCP_SERVERS, RULES, AGENTS } from "../data/catalog";
import { REPOS } from "../data/extra";
import { THEMES, useApp } from "../lib/app";
import { AgentMark, IconBolt, IconBrain, IconCheck, IconCoin, IconFile, IconGit, IconLayers, IconMcp, IconMic, IconMonitor, IconPalette, IconPlug, IconPlus, IconSearch, IconSettings, IconShield, IconSpark, IconTerminal, IconX, IconAt, IconStar } from "./Icons";
import { Badge, Bar, Button, Input, Kbd, Segmented, Select, Toggle } from "./ui";
import { SHORTCUTS, SHORTCUT_GROUPS } from "../lib/shortcuts";
import { getBackend, setBackend, testBackend, type BackendConfig, type BackendMode } from "../lib/live";
import { AgentsView, BridgesView } from "./views";
import { DesignSystemView } from "./DesignSystemView";
import { modeDefs } from "./Workbench";
import { ProviderSettings } from "./ProviderSettings";

type Section = { id: string; label: string; icon: React.ComponentType<{ size?: number; className?: string }>; keywords: string };
const GROUPS: { label: string; items: Section[] }[] = [
  { label: "Workspace", items: [
    { id: "general", label: "General", icon: IconSettings, keywords: "startup language notifications sound telemetry updates" },
    { id: "appearance", label: "Appearance", icon: IconPalette, keywords: "theme dark light font density" },
    { id: "keyboard", label: "Keyboard", icon: IconAt, keywords: "shortcuts keybindings vim" },
    { id: "voice", label: "Voice & audio", icon: IconMic, keywords: "audio voice speech dictation handsfree microphone speak replies" },
  ] },
  { label: "Agents", items: [
    { id: "agents", label: "Agents & models", icon: IconLayers, keywords: "claude codex cursor aro glm provider api key routing" },
    { id: "providers", label: "Providers & models", icon: IconBrain, keywords: "ollama lm studio local offline cloud inference openai compatible model endpoint" },
    { id: "backend", label: "Backend", icon: IconBolt, keywords: "demo live sandbox real aro agent api server openai compatible endpoint 8642 replies chat completions" },
    { id: "permissions", label: "Permissions", icon: IconShield, keywords: "mode plan read only full access approvals allowlist sandbox spend cap" },
    { id: "rules", label: "Rules & memory", icon: IconFile, keywords: "agents.md claude.md soul memory instructions" },
    { id: "connectors", label: "Connectors (MCP)", icon: IconMcp, keywords: "mcp tools servers github linear sentry" },
    { id: "skills", label: "Skills", icon: IconStar, keywords: "skills hub commands" },
  ] },
  { label: "Integrations", items: [
    { id: "editors", label: "Editors", icon: IconMonitor, keywords: "ide cursor vscode zed windsurf jetbrains bridge" },
    { id: "git", label: "Git", icon: IconGit, keywords: "commit pr merge branch github gitlab" },
    { id: "terminal", label: "Terminal & sandbox", icon: IconTerminal, keywords: "shell backend docker ssh modal timeout" },
    { id: "gateway", label: "Messaging gateway", icon: IconPlug, keywords: "slack telegram discord email notifications" },
  ] },
  { label: "Account", items: [
    { id: "usage", label: "Usage & billing", icon: IconCoin, keywords: "cost spend tokens plan limit" },
    { id: "design", label: "Design system", icon: IconSpark, keywords: "tokens components colors" },
  ] },
];

/* ---------------- primitives ---------------- */
function Card({ title, sub, children, right }: { title: string; sub?: string; children: React.ReactNode; right?: React.ReactNode }) {
  return (
    <section className="overflow-hidden rounded-[12px] border border-line-soft bg-raise">
      <div className="flex items-start justify-between gap-3 border-b border-line-soft px-4 py-3">
        <div><h3 className="text-[13px] font-semibold text-ink">{title}</h3>{sub && <p className="mt-0.5 text-[11.5px] text-ink-3">{sub}</p>}</div>{right}
      </div>
      <div className="divide-y divide-line-soft">{children}</div>
    </section>
  );
}
function Row({ label, desc, children }: { label: string; desc?: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-6 px-4 py-3">
      <div className="min-w-0"><div className="text-[12.5px] font-medium text-ink">{label}</div>{desc && <div className="mt-0.5 text-[11.5px] leading-[1.5] text-ink-3">{desc}</div>}</div>
      <div className="shrink-0">{children}</div>
    </div>
  );
}
function T({ on = true }: { on?: boolean }) { const [v, setV] = useState(on); return <Toggle checked={v} onChange={setV} />; }
function S({ value, options, w = 170 }: { value: string; options: string[]; w?: number }) { const [v, setV] = useState(value); return <div style={{ width: w }}><Select value={v} onChange={setV} options={options.map((o) => ({ value: o, label: o }))} align="right" /></div>; }
function Page({ title, sub, children }: { title: string; sub: string; children: React.ReactNode }) {
  return (
    <div className="scroll-thin flex-1 overflow-y-auto">
      <div className="mx-auto max-w-[780px] space-y-4 px-6 py-6">
        <div><h1 className="text-[19px] font-semibold tracking-[-.02em] text-ink">{title}</h1><p className="mt-1 text-[12.5px] text-ink-3">{sub}</p></div>
        {children}
      </div>
    </div>
  );
}

/* ---------------- sections ---------------- */
function General() {
  return (
    <Page title="General" sub="How Aro starts, notifies and updates.">
      <Card title="Startup">
        <Row label="Default product mode" desc="Which side opens on launch."><S value="Code" options={["Code", "Agent", "Last used"]} /></Row>
        <Row label="Restore last session" desc="Reopen the session and panels you had open."><T /></Row>
        <Row label="Default agent for new sessions"><S value="Claude Code" options={AGENTS.filter((a) => a.status === "connected").map((a) => a.name)} /></Row>
      </Card>
      <Card title="Notifications">
        <Row label="Notify when an agent needs approval" desc="Desktop notification + dock badge."><T /></Row>
        <Row label="Notify when a background run finishes"><T /></Row>
        <Row label="Sound on completion"><T on={false} /></Row>
      </Card>
      <Card title="Privacy & updates">
        <Row label="Privacy mode" desc="Never store code on remote servers. Disables cloud runs."><T on={false} /></Row>
        <Row label="Anonymous usage data"><T on={false} /></Row>
        <Row label="Update channel"><S value="Stable" options={["Stable", "Beta", "Nightly"]} /></Row>
      </Card>
    </Page>
  );
}
function Appearance() {
  const { theme, setTheme } = useApp();
  const [density, setDensity] = useState<"compact" | "comfortable">("comfortable");
  return (
    <Page title="Appearance" sub="Themes share one token contract — every surface adapts.">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
        {THEMES.map((t) => (
          <button key={t.id} onClick={() => setTheme(t.id)} className={cn("cursor-pointer overflow-hidden rounded-[12px] border text-left transition-all hover:shadow-e3", theme === t.id ? "border-iris/60 shadow-glow-iris" : "border-line-soft hover:border-line-strong")}>
            <div className="relative h-[84px] p-2.5" style={{ background: t.swatch[0] }}>
              <div className="flex h-full gap-1.5"><div className="w-[22%] rounded-[5px]" style={{ background: t.swatch[1] }} /><div className="flex-1 rounded-[5px] p-1.5" style={{ background: t.swatch[1] }}><div className="h-[5px] w-1/2 rounded-full" style={{ background: t.swatch[2] }} /><div className="mt-1.5 h-[4px] w-3/4 rounded-full opacity-40" style={{ background: t.swatch[3] }} /><div className="mt-1 h-[4px] w-2/3 rounded-full opacity-25" style={{ background: t.swatch[3] }} /></div></div>
              {theme === t.id && <span className="absolute top-2 right-2 flex size-[18px] items-center justify-center rounded-full bg-iris text-on-iris"><IconCheck size={10} /></span>}
            </div>
            <div className="bg-raise px-3 py-2"><div className="text-[12.5px] font-semibold text-ink">{t.label}</div><div className="text-[10.5px] text-ink-3">{t.desc}</div></div>
          </button>
        ))}
      </div>
      <Card title="Layout">
        <Row label="Density"><Segmented value={density} onChange={setDensity} items={[{ value: "compact", label: "Compact" }, { value: "comfortable", label: "Comfortable" }]} /></Row>
        <Row label="Show reasoning blocks" desc="Collapsed by default when on."><T /></Row>
        <Row label="Auto-expand diffs in transcript"><T /></Row>
        <Row label="Code font"><S value="JetBrains Mono" options={["JetBrains Mono", "SF Mono", "Fira Code", "Berkeley Mono"]} /></Row>
        <Row label="Transcript font size"><S value="13.5px" options={["12.5px", "13.5px", "14.5px"]} w={110} /></Row>
      </Card>
    </Page>
  );
}
function Keyboard() {
  const [preset, setPreset] = useState("VS Code");
  const [rebind, setRebind] = useState<string | null>(null);
  return (
    <Page title="Keyboard" sub="Defaults follow VS Code and Cursor conventions so muscle memory transfers. Press a shortcut to rebind it.">
      <Card title="Preset" sub="Applies the whole map at once; individual keys stay overridable.">
        <Row label="Keymap"><div style={{ width: 150 }}><Select value={preset} onChange={setPreset} options={["VS Code", "JetBrains", "Vim", "Emacs"].map((o) => ({ value: o, label: o }))} align="right" /></div></Row>
      </Card>
      {SHORTCUT_GROUPS.map((g) => (
        <Card key={g} title={g}>
          {SHORTCUTS.filter((s) => s.group === g).map((s) => (
            <Row key={s.id} label={s.label} desc={s.note}>
              <button onClick={() => setRebind(rebind === s.id ? null : s.id)} className={cn("flex cursor-pointer items-center gap-1 rounded-[7px] border px-1.5 py-1 transition-colors", rebind === s.id ? "border-iris bg-iris-tint" : "border-line hover:border-line-strong")}>
                {rebind === s.id ? <span className="font-mono text-[11px] text-iris-soft">press keys…</span> : s.keys.map((k, i) => <Kbd key={i} className="h-[20px] min-w-[20px] px-1.5 text-[10.5px]">{k}</Kbd>)}
              </button>
            </Row>
          ))}
        </Card>
      ))}
    </Page>
  );
}
function VoiceSettings() {
  const [speak, setSpeak] = useState(() => localStorage.getItem("aro.voice.speakBack") === "true");
  const [handsFree, setHandsFree] = useState(() => localStorage.getItem("aro.voice.handsFree") === "true");
  const [orb, setOrb] = useState(() => localStorage.getItem("aro.voice.orb") !== "false");
  const setOrbMode = (v: boolean) => { setOrb(v); localStorage.setItem("aro.voice.orb", String(v)); window.dispatchEvent(new CustomEvent("aro:voice:orb", { detail: v })); };
  const [language, setLanguage] = useState(() => localStorage.getItem("aro.voice.language") || navigator.language || "en-US");
  const [engine, setEngine] = useState<"browser" | "whisper-local" | "whisper-cloud">(() => (localStorage.getItem("aro.voice.engine") as "browser" | "whisper-local" | "whisper-cloud") || "browser");
  const [daemon, setDaemon] = useState(() => localStorage.getItem("aro.voice.daemon") || "http://localhost:4735/v1/audio");
  const setSpeakMode = (v: boolean) => { setSpeak(v); localStorage.setItem("aro.voice.speakBack", String(v)); window.dispatchEvent(new CustomEvent("aro:voice:speakBack", { detail: v })); };
  const setHandsFreeMode = (v: boolean) => { setHandsFree(v); localStorage.setItem("aro.voice.handsFree", String(v)); window.dispatchEvent(new CustomEvent("aro:voice:handsFree", { detail: v })); };
  const setVoiceLanguage = (v: string) => { setLanguage(v); localStorage.setItem("aro.voice.language", v); };
  return (
    <Page title="Voice & audio" sub="Dictation is built into the composer — press the mic beside Send, or ⌘⇧V anywhere. The hands-free assistant is optional and off by default.">
      <Card title="Dictation" sub="The microphone beside Send turns speech into text in the composer. ⌘⇧V toggles it from anywhere.">
        <Row label="Engine" desc="Whisper routes through the Aro daemon; if it's offline you fall back to the browser engine automatically.">
          <Segmented value={engine} onChange={(v) => { setEngine(v); localStorage.setItem("aro.voice.engine", v); window.dispatchEvent(new CustomEvent("aro:voice:engine", { detail: v })); }}
            items={[{ value: "browser", label: "Browser" }, { value: "whisper-local", label: "Whisper local" }, { value: "whisper-cloud", label: "Whisper cloud" }]} />
        </Row>
        <Row label="Daemon endpoint" desc="Only used by Whisper engines."><div className="w-[210px]"><Input value={daemon} onChange={(e) => { setDaemon(e.target.value); localStorage.setItem("aro.voice.daemon", e.target.value); }} className="h-[28px] font-mono text-[11px]" /></div></Row>
        <Row label="Recognition language" desc="Applied next time you start dictating."><div className="w-[160px]"><Select value={language} onChange={setVoiceLanguage} align="right" options={["en-US", "en-GB", "fr-FR", "de-DE", "es-ES", "ja-JP"].map((l) => ({ value: l, label: l }))} /></div></Row>
        <Row label="Auto-punctuate" desc="Whisper only — adds commas and full stops as you speak."><T /></Row>
      </Card>
      <Card title="Voice orb" sub="A live, draggable orb that reflects what the agent is doing — idle, listening, thinking, speaking, done or needs attention — even when minimised.">
        <Row label="Show the orb" desc="Click to open, double-click to talk, drag anywhere."><Toggle checked={orb} onChange={setOrbMode} /></Row>
        <Row label="Hands-free default" desc="Send each recognised phrase directly when the panel is open."><Toggle checked={handsFree} onChange={setHandsFreeMode} /></Row>
        <Row label="Speak assistant replies" desc="Uses your device's built-in speech synthesis voice."><Toggle checked={speak} onChange={setSpeakMode} /></Row>
      </Card>
      <div className="flex items-start gap-2 rounded-[10px] border border-amber/20 bg-amber-tint/40 px-3 py-2.5 text-[11px] leading-[1.55] text-amber"><IconMic size={12} className="mt-0.5 shrink-0" /><span>Microphone access is requested only after you press Listen. Browser speech recognition may use the browser vendor's speech service; it is not guaranteed to be on-device. Text fallback remains available.</span></div>
      <Card title="Short voice commands" sub="Only while the floating assistant is on. Navigation is local; any other phrase goes to your active thread.">
        {["Open tasks", "Open changes", "Open browser", "Open Git", "Open fleet", "Open terminal", "Open providers", "Code mode", "Agent mode", "Stop listening"].map((v) => <Row key={v} label={v}><Badge tone={v === "Stop listening" ? "amber" : "neutral"} mono>voice</Badge></Row>)}
      </Card>
    </Page>
  );
}
function Permissions() {
  const { product } = useApp();
  const modes = modeDefs(product);
  const [def, setDef] = useState(modes[2].id as string);
  const [allow, setAllow] = useState(["pnpm test", "pnpm lint", "git status", "git diff"]);
  const [deny] = useState(["rm -rf", "git push --force", "curl | sh", "DROP TABLE"]);
  const [cmd, setCmd] = useState("");
  return (
    <Page title="Permissions" sub="What agents may do without asking. Per-session mode lives in the composer dropdown.">
      <Card title="Default mode for new sessions">
        <div className="grid gap-2 p-3 sm:grid-cols-2">
          {modes.map((m) => (
            <button key={m.id} onClick={() => setDef(m.id)} className={cn("flex cursor-pointer items-start gap-2.5 rounded-[9px] border p-3 text-left", def === m.id ? "border-iris/50 bg-iris-tint" : "border-line-soft hover:border-line-strong")}>
              <m.icon size={14} className="mt-0.5 text-iris-soft" /><span><span className="block text-[12.5px] font-medium text-ink">{m.label}</span><span className="block text-[11px] text-ink-3">{m.desc}</span></span>
            </button>
          ))}
        </div>
      </Card>
      <Card title="Command allowlist" sub="Run without asking, in any mode except Plan / Read only.">
        <div className="flex flex-wrap gap-1.5 p-3">
          {allow.map((a) => <span key={a} className="inline-flex items-center gap-1.5 rounded-[6px] border border-mint/25 bg-mint-tint px-2 py-1 font-mono text-[11px] text-mint">{a}<button onClick={() => setAllow((x) => x.filter((y) => y !== a))} className="cursor-pointer opacity-60 hover:opacity-100"><IconX size={10} /></button></span>)}
          <div className="flex gap-1.5"><Input value={cmd} onChange={(e) => setCmd(e.target.value)} placeholder="add command…" className="h-[28px] w-[160px] font-mono" onKeyDown={(e) => { if (e.key === "Enter" && cmd) { setAllow((x) => [...x, cmd]); setCmd(""); } }} /></div>
        </div>
      </Card>
      <Card title="Always deny" sub="Blocked in every mode, including Full access.">
        <div className="flex flex-wrap gap-1.5 p-3">{deny.map((a) => <span key={a} className="rounded-[6px] border border-rose/25 bg-rose-tint px-2 py-1 font-mono text-[11px] text-rose">{a}</span>)}</div>
      </Card>
      <Card title="Guardrails">
        <Row label="Checkpoint before every write" desc="Rewind any edit with esc esc."><T /></Row>
        <Row label="Network access in Agent mode"><S value="Ask" options={["Block", "Ask", "Allow"]} w={110} /></Row>
        <Row label="Spend cap per session"><S value="$25" options={["$5", "$10", "$25", "$100", "No cap"]} w={110} /></Row>
        <Row label="Auto-deny approvals after" desc="Unanswered approvals time out safely."><S value="30s" options={["Never", "18s", "30s", "2m"]} w={110} /></Row>
      </Card>
    </Page>
  );
}
function Rules() {
  return (
    <Page title="Rules & memory" sub="Instructions every agent reads. Compatible with AGENTS.md, CLAUDE.md, .cursor/rules and Aro SOUL.md / MEMORY.md.">
      <Card title="Project rules" sub="aro/platform · synced to AGENTS.md" right={<Button size="xs" variant="secondary" icon={IconPlus}>Add rule</Button>}>
        {RULES.map((r) => <Row key={r.id} label={r.text} desc={`scope ${r.scope}`}><T on={r.on} /></Row>)}
      </Card>
      <Card title="Memory" sub="Learned facts the agent may reuse across sessions.">
        {["Prefers tables over prose for comparisons", "Uses pnpm, never npm", "Platform leads: Mira, Ola, Dev"].map((m) => <Row key={m} label={m}><Button size="xs" variant="ghost">Forget</Button></Row>)}
        <Row label="Auto-capture memory" desc="Agent proposes memories; you approve."><T /></Row>
      </Card>
      <Card title="Import">
        <Row label="Import from other agents" desc="Pull rules from .cursor/rules, CLAUDE.md, ~/.aro/SOUL.md, ~/.codex/AGENTS.md"><Button size="xs" variant="secondary">Scan</Button></Row>
      </Card>
    </Page>
  );
}
function Connectors() {
  return (
    <Page title="Connectors (MCP)" sub="Tools available to every agent. Each server can be scoped per product mode.">
      <Card title="Servers" right={<Button size="xs" variant="primary" icon={IconPlus}>Add server</Button>}>
        {MCP_SERVERS.map((m) => (
          <Row key={m.name} label={m.name} desc={`${m.tools} tools · ${m.status === "live" ? "connected" : "degraded — reconnecting"}`}>
            <div className="flex items-center gap-2"><Badge tone={m.status === "live" ? "mint" : "amber"} mono dot className="text-[9.5px]">{m.status}</Badge><T on /></div>
          </Row>
        ))}
      </Card>
      <Card title="Indexing">
        <Row label="Index repository for semantic search" desc="1,284 files · last indexed 4m ago"><Button size="xs" variant="secondary">Re-index</Button></Row>
        <Row label="Ignore patterns" desc="node_modules, dist, generated/**"><Button size="xs" variant="ghost">Edit</Button></Row>
      </Card>
    </Page>
  );
}
function Skills() {
  const skills = [["release-notes", "Draft release notes from merged PRs", true], ["migrate-vitest", "Convert jest specs to vitest", true], ["sentry-triage", "Open tasks for new P0 issues", true], ["design", "Generate UI artboards before building", false], ["db-migration", "Write and dry-run a SQL migration", false]] as const;
  return (
    <Page title="Skills" sub="Reusable, versioned procedures the agent can invoke with / or pick automatically.">
      <Card title="Installed" right={<Button size="xs" variant="secondary">Browse hub</Button>}>
        {skills.map(([n, d, on]) => <Row key={n} label={`/${n}`} desc={d}><T on={on} /></Row>)}
      </Card>
    </Page>
  );
}
function GitSettings() {
  return (
    <Page title="Git" sub="How agents commit, branch and open pull requests.">
      <Card title="Repositories" right={<Button size="xs" variant="primary" icon={IconPlus}>Connect</Button>}>
        {REPOS.map((r) => <Row key={r.id} label={r.name} desc={`${r.path} · ${r.remote}`}>{r.default ? <Badge tone="iris" mono className="text-[9.5px]">default</Badge> : <Button size="xs" variant="ghost">Make default</Button>}</Row>)}
      </Card>
      <Card title="Behaviour">
        <Row label="Agents work on a new branch" desc="Never commit directly to the default branch."><T /></Row>
        <Row label="Branch prefix"><S value="agent/" options={["agent/", "feat/", "aro/"]} w={120} /></Row>
        <Row label="Commit message author"><S value="Agent drafts, you edit" options={["Agent drafts, you edit", "Agent", "Me"]} w={190} /></Row>
        <Row label="Auto-merge agent PRs" desc="When all checks pass and one human approved."><T /></Row>
        <Row label="Merge strategy"><S value="Squash" options={["Squash", "Rebase", "Merge commit"]} w={130} /></Row>
      </Card>
    </Page>
  );
}
function TerminalSettings() {
  return (
    <Page title="Terminal & sandbox" sub="Where agent commands execute. Isolated backends keep the agent away from your machine.">
      <Card title="Execution backend">
        <Row label="Backend" desc="local · docker · ssh · modal · daytona"><S value="docker" options={["local", "docker", "ssh", "modal", "daytona"]} w={130} /></Row>
        <Row label="Image"><S value="node:22-bookworm" options={["node:22-bookworm", "python:3.12-slim", "custom…"]} w={180} /></Row>
        <Row label="Persist container between sessions"><T /></Row>
        <Row label="Command timeout"><S value="180s" options={["60s", "180s", "600s"]} w={110} /></Row>
        <Row label="Forward env vars" desc="Secrets are proxied, never written to the sandbox."><Button size="xs" variant="ghost">Edit</Button></Row>
      </Card>
      <Card title="Integrated terminal">
        <Row label="Shell"><S value="zsh" options={["zsh", "bash", "fish", "pwsh"]} w={110} /></Row>
        <Row label="Mirror agent commands to terminal" desc="Shows every command the agent runs in the agent tab."><T /></Row>
        <Row label="Prefix ? sends to agent"><T /></Row>
      </Card>
    </Page>
  );
}
function Gateway() {
  const p = [["Slack", "#platform · DMs", true], ["Telegram", "@aro_bot", false], ["Discord", "not connected", false], ["Email", "dev@acme.dev", true]] as const;
  return (
    <Page title="Messaging gateway" sub="Talk to the agent and get approvals from chat apps when you're away from the desk.">
      <Card title="Platforms">{p.map(([n, d, on]) => <Row key={n} label={n} desc={d}>{on ? <T /> : <Button size="xs" variant="secondary">Connect</Button>}</Row>)}</Card>
      <Card title="Routing"><Row label="Send approvals to chat" desc="Approve risky commands from Slack."><T /></Row><Row label="Daily digest"><S value="09:00" options={["Off", "09:00", "17:00"]} w={110} /></Row></Card>
    </Page>
  );
}
function BackendSettings() {
  const { toast } = useApp();
  const [cfg, setCfg] = useState<BackendConfig>(() => getBackend());
  const [testing, setTesting] = useState(false);
  const [result, setResult] = useState<{ ok: boolean; detail: string; ms: number } | null>(null);
  const setMode = (mode: BackendMode) => { setBackend({ ...cfg, mode }); setCfg({ ...cfg, mode }); setResult(null); };
  const runTest = async () => { setTesting(true); setResult(null); const r = await testBackend(cfg); setResult(r); setTesting(false); };
  const save = () => { setBackend(cfg); toast("Live backend saved — replies now come from your Aro agent", "mint"); };
  return (
    <Page title="Backend" sub="Where replies come from. Demo uses this sandbox's z-ai model; live points the workbench at any OpenAI-compatible endpoint — like your Aro agent's api_server on :8642.">
      <Card title="Mode" sub="Takes effect immediately for new messages.">
        <Row label="Reply backend" desc="Demo: this sandbox's z-ai model. Live: any OpenAI-compatible endpoint.">
          <Segmented value={cfg.mode} onChange={setMode} items={[{ value: "demo", label: "Demo" }, { value: "live", label: "Live" }]} />
        </Row>
      </Card>
      {cfg.mode === "demo" ? (
        <Card title="Demo" sub="Scripted tool steps close with a real reply from the sandbox model.">
          <div className="flex items-center justify-between gap-6 px-4 py-3">
            <div className="min-w-0">
              <div className="font-mono text-[11px] text-ink-2">POST /api/chat · sandbox model · GLM via z-ai</div>
              <div className="mt-0.5 text-[11.5px] text-ink-3">the reply you see in chat</div>
            </div>
            <Badge tone="mint" mono dot>connected</Badge>
          </div>
        </Card>
      ) : (
        <Card title="Live endpoint" sub="Any server that speaks POST /chat/completions.">
          <Row label="Base URL" desc="Aro agent's api_server speaks OpenAI-compatible on :8642.">
            <div className="w-[250px]"><Input value={cfg.baseUrl} onChange={(e) => setCfg((c) => ({ ...c, baseUrl: e.target.value }))} placeholder="http://localhost:8642/v1" className="h-[28px] font-mono text-[11px]" /></div>
          </Row>
          <Row label="Model" desc="Model id sent with every completion.">
            <div className="w-[200px]"><Input value={cfg.model} onChange={(e) => setCfg((c) => ({ ...c, model: e.target.value }))} placeholder="aro-4-70b" className="h-[28px] font-mono text-[11px]" /></div>
          </Row>
          <Row label="API key" desc="Optional for local servers.">
            <div className="w-[200px]"><Input type="password" value={cfg.key} onChange={(e) => setCfg((c) => ({ ...c, key: e.target.value }))} placeholder="sk-…" className="h-[28px] font-mono text-[11px]" /></div>
          </Row>
          <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3">
            <div className="flex min-w-0 items-center gap-2">
              {result ? (
                result.ok
                  ? <Badge tone="mint" mono dot>reachable · {result.ms}ms</Badge>
                  : <Badge tone="rose" mono dot>unreachable · check host</Badge>
              ) : <Badge tone="neutral" mono>{testing ? "testing…" : "not tested"}</Badge>}
              {result && <span className="truncate font-mono text-[10.5px] text-ink-4">{result.detail}{result.ok ? "" : ` · ${result.ms}ms`}</span>}
            </div>
            <div className="flex shrink-0 items-center gap-2">
              <Button variant="secondary" disabled={testing} onClick={runTest}>{testing ? "Testing…" : "Test connection"}</Button>
              <Button variant="primary" onClick={save}>Save</Button>
            </div>
          </div>
        </Card>
      )}
      <div className="flex items-start gap-2 rounded-[10px] border border-amber/20 bg-amber-tint/40 px-3 py-2.5 text-[11px] leading-[1.55] text-amber"><IconBolt size={12} className="mt-0.5 shrink-0" /><span>Live mode points the workbench at a real agent. Demo data elsewhere (sessions, tasks) stays seeded.</span></div>
    </Page>
  );
}
function Usage() {
  const rows = [["claude-code", 84.2], ["codex", 51.6], ["cursor", 22.1], ["aro", 14.8], ["glm-code", 6.4], ["darwin", 5.1]] as const;
  return (
    <Page title="Usage & billing" sub="Spend across all agents, this month.">
      <div className="grid grid-cols-3 gap-3">
        {[["Spent", "$184.20", "of $400"], ["Tokens", "18.4M", "62% cached"], ["Sessions", "212", "41 parallel"]].map(([l, v, s]) => <div key={l} className="rounded-[11px] border border-line-soft bg-raise p-3"><div className="font-mono text-[9.5px] tracking-[.12em] text-ink-4 uppercase">{l}</div><div className="mt-1 text-[20px] font-semibold text-ink">{v}</div><div className="text-[11px] text-ink-3">{s}</div></div>)}
      </div>
      <Card title="By agent">
        {rows.map(([id, v]) => { const a = AGENTS.find((x) => x.id === id)!; return (
          <div key={id} className="flex items-center gap-3 px-4 py-2.5"><AgentMark glyph={a.glyph} from={a.from} to={a.to} size={18} /><span className="w-[110px] text-[12.5px] text-ink">{a.name}</span><Bar value={(v / 84.2) * 100} /><span className="w-[60px] text-right font-mono text-[11px] text-ink-2">${v}</span></div>); })}
      </Card>
      <Card title="Plan"><Row label="Max" desc="$400 / month pooled across agents"><Button size="xs" variant="secondary">Manage</Button></Row><Row label="Hard limit" desc="Stop all runs when reached"><T /></Row></Card>
    </Page>
  );
}

/* ---------------- settings shell ---------------- */
export function SettingsView({ onStart }: { onStart: (a: string) => void }) {
  const { settingsSection: sec, setSettingsSection: setSec } = useApp();
  const [q, setQ] = useState("");
  const match = (s: Section) => (s.label + " " + s.keywords).toLowerCase().includes(q.toLowerCase());
  return (
    <div className="flex min-h-0 flex-1">
      <aside className="flex w-[230px] shrink-0 flex-col border-r border-line-soft bg-sunken">
        <div className="flex h-[46px] items-center gap-2 border-b border-line-soft px-3"><IconSettings size={14} className="text-ink-3" /><span className="text-[13px] font-semibold text-ink">Settings</span><Kbd className="ml-auto">⌘,</Kbd></div>
        <div className="relative px-2 py-2"><IconSearch size={12} className="pointer-events-none absolute top-1/2 left-4 -translate-y-1/2 text-ink-4" /><Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search settings…" className="h-[28px] pl-7" /></div>
        <nav className="scroll-thin flex-1 overflow-y-auto px-1.5 pb-3">
          {GROUPS.map((g) => { const items = g.items.filter(match); if (!items.length) return null; return (
            <div key={g.label} className="mb-2">
              <div className="px-2 pt-2 pb-1 font-mono text-[9px] font-semibold tracking-[.16em] text-ink-4 uppercase">{g.label}</div>
              {items.map((s) => <button key={s.id} onClick={() => setSec(s.id)} className={cn("flex w-full cursor-pointer items-center gap-2.5 rounded-[7px] px-2 py-[7px] text-left text-[12.5px] transition-colors", sec === s.id ? "bg-raise font-medium text-ink shadow-e1" : "text-ink-2 hover:bg-raise/60 hover:text-ink")}><s.icon size={13} className={sec === s.id ? "text-iris-soft" : "text-ink-4"} />{s.label}</button>)}
            </div>); })}
        </nav>
        <div className="border-t border-line-soft p-3 font-mono text-[10px] text-ink-4">Aro 1.5.0 · <span className="text-mint">up to date</span></div>
      </aside>
      <div key={sec} className="flex min-w-0 flex-1 animate-fade flex-col bg-base">
        {sec === "general" && <General />}
        {sec === "appearance" && <Appearance />}
        {sec === "keyboard" && <Keyboard />}
        {sec === "voice" && <VoiceSettings />}
        {sec === "agents" && <AgentsView onStart={onStart} />}
        {sec === "providers" && <ProviderSettings />}
        {sec === "backend" && <BackendSettings />}
        {sec === "permissions" && <Permissions />}
        {sec === "rules" && <Rules />}
        {sec === "connectors" && <Connectors />}
        {sec === "skills" && <Skills />}
        {sec === "editors" && <BridgesView />}
        {sec === "git" && <GitSettings />}
        {sec === "terminal" && <TerminalSettings />}
        {sec === "gateway" && <Gateway />}
        {sec === "usage" && <Usage />}
        {sec === "design" && <DesignSystemView />}
      </div>
    </div>
  );
}
export { IconBolt, IconBrain };
