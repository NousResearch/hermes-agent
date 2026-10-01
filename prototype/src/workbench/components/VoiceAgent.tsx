"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { cn } from "../utils/cn";
import { IconArrowUp, IconEye, IconGit, IconList, IconTerminal, IconX } from "./Icons";
import { Kbd, Toggle } from "./ui";
import { ORB_LABEL, Orb, useMicLevel, type OrbState } from "./Orb";
import type { ViewId } from "./Sidebar";

type SpeechResultEvent = { resultIndex: number; results: ArrayLike<ArrayLike<{ transcript: string }> & { isFinal: boolean }> };
type Recognition = {
  lang: string; continuous: boolean; interimResults: boolean;
  onresult: ((e: SpeechResultEvent) => void) | null;
  onerror: ((e: { error: string }) => void) | null;
  onend: (() => void) | null;
  start: () => void; stop: () => void; abort: () => void;
};
type RecognitionCtor = new () => Recognition;
const voiceWindow = () => window as Window & { SpeechRecognition?: RecognitionCtor; webkitSpeechRecognition?: RecognitionCtor };

const NAV: { label: string; icon: typeof IconList; words: string[]; view: ViewId }[] = [
  { label: "Tasks", icon: IconList, words: ["tasks", "task list"], view: "tasks" },
  { label: "Changes", icon: IconGit, words: ["changes", "review", "diff"], view: "review" },
  { label: "Browser", icon: IconEye, words: ["browser", "preview"], view: "browser" },
  { label: "Fleet", icon: IconTerminal, words: ["fleet", "runs", "automations"], view: "runs" },
];
const STATE_HINT: Record<OrbState, string> = {
  idle: "Tap the orb to talk",
  listening: "Speak naturally — I'm listening",
  thinking: "The agent is working on it",
  speaking: "Reading the reply aloud",
  completed: "Finished — reply is in the thread",
  error: "Something needs your attention",
};

export function VoiceAgent({ onSend, onNavigate, reply, streaming }: { onSend: (text: string) => void; onNavigate: (view: ViewId) => void; reply: string; streaming: boolean }) {
  const [open, setOpen] = useState(false);
  const [enabled, setEnabled] = useState(() => localStorage.getItem("aro.voice.orb") !== "false");
  const [handsFree, setHandsFree] = useState(() => localStorage.getItem("aro.voice.handsFree") === "true");
  const [speakBack, setSpeakBack] = useState(() => localStorage.getItem("aro.voice.speakBack") === "true");
  const [listening, setListening] = useState(false);
  const [dictating, setDictating] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [completed, setCompleted] = useState(false);
  const [errorFlash, setErrorFlash] = useState(false);
  const [draft, setDraft] = useState("");
  const [interim, setInterim] = useState("");
  const [error, setError] = useState("");
  const [lastHeard, setLastHeard] = useState("");
  const [position, setPosition] = useState<{ x: number; y: number }>(() => {
    try { const p = JSON.parse(localStorage.getItem("aro.voice.position") || "null"); return p && typeof p.x === "number" ? p : { x: 0, y: 0 }; } catch { return { x: 0, y: 0 }; }
  });
  const dragRef = useRef<{ x: number; y: number; px: number; py: number; moved: boolean } | null>(null);
  const suppressClick = useRef(false);
  const recRef = useRef<Recognition | null>(null);
  const restartOnEnd = useRef(false);
  const spokenRef = useRef("");
  const supported = Boolean(voiceWindow().SpeechRecognition ?? voiceWindow().webkitSpeechRecognition);

  /* the orb listens to the mic whenever *anything* is capturing speech */
  const micLevel = useMicLevel(listening || dictating);

  const orbState: OrbState = errorFlash ? "error" : listening || dictating ? "listening" : speaking ? "speaking" : streaming ? "thinking" : completed ? "completed" : "idle";

  /* persistence + cross-component events */
  useEffect(() => localStorage.setItem("aro.voice.position", JSON.stringify(position)), [position]);
  useEffect(() => localStorage.setItem("aro.voice.speakBack", String(speakBack)), [speakBack]);
  useEffect(() => localStorage.setItem("aro.voice.handsFree", String(handsFree)), [handsFree]);
  useEffect(() => {
    const on = (name: string, fn: (e: Event) => void) => { window.addEventListener(name, fn); return () => window.removeEventListener(name, fn); };
    const offs = [
      on("aro:voice", () => { setEnabled(true); setOpen(true); }),
      on("aro:voice:orb", (e) => setEnabled(Boolean((e as CustomEvent<boolean>).detail))),
      on("aro:voice:speakBack", (e) => setSpeakBack(Boolean((e as CustomEvent<boolean>).detail))),
      on("aro:voice:handsFree", (e) => setHandsFree(Boolean((e as CustomEvent<boolean>).detail))),
      on("aro:dictation", (e) => setDictating(Boolean((e as CustomEvent<boolean>).detail))),
    ];
    return () => offs.forEach((off) => off());
  }, []);

  /* thinking → completed pulse (adjust during render; the fade-out is a timer) */
  const [prevStreaming, setPrevStreaming] = useState(streaming);
  if (prevStreaming !== streaming) {
    setPrevStreaming(streaming);
    if (prevStreaming && !streaming) setCompleted(true);
  }
  useEffect(() => {
    if (!completed) return;
    const t = window.setTimeout(() => setCompleted(false), 2800);
    return () => window.clearTimeout(t);
  }, [completed]);

  const flashError = (msg: string) => {
    setError(msg);
    setErrorFlash(true);
    window.setTimeout(() => setErrorFlash(false), 3200);
  };

  const say = useCallback((text: string) => {
    if (!text || !("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text.slice(0, 900));
    u.rate = 1.02;
    u.onstart = () => setSpeaking(true);
    u.onend = () => setSpeaking(false);
    u.onerror = () => setSpeaking(false);
    spokenRef.current = text;
    window.speechSynthesis.speak(u);
  }, []);
  useEffect(() => {
    if (speakBack && reply && reply !== spokenRef.current && !listening && !streaming) say(reply);
  }, [reply, speakBack, listening, streaming, say]);

  function stopRecognition() {
    restartOnEnd.current = false;
    try { recRef.current?.stop(); } catch { /* already stopped */ }
    setListening(false);
    setInterim("");
  }

  const routeCommand = useCallback((raw: string) => {
    const text = raw.trim();
    if (!text) return "";
    const lower = text.toLowerCase().replace(/[.!?]+$/, "");
    setLastHeard(text);
    const hit = NAV.find((n) => n.words.some((w) => [w, `open ${w}`, `go to ${w}`, `show ${w}`].includes(lower)));
    if (hit) { onNavigate(hit.view); return `Opening ${hit.label}.`; }
    const events: [string[], string, string, unknown?][] = [
      [["open terminal", "show terminal"], "aro:terminal", "Terminal toggled."],
      [["open settings", "settings"], "aro:settings", "Opening settings."],
      [["open providers", "open local models"], "aro:providers", "Opening model providers."],
      [["show todos", "open todo"], "aro:todo", "Opening your todo list."],
      [["new thread", "new chat", "new session"], "aro:new-session", "Starting a new thread."],
      [["code mode", "switch to code mode"], "aro:product", "Switched to Code.", "code"],
      [["agent mode", "switch to agent mode"], "aro:product", "Switched to Agent.", "agent"],
    ];
    for (const [phrases, evt, ack, detail] of events) {
      if (phrases.includes(lower)) { window.dispatchEvent(new CustomEvent(evt, { detail })); return ack; }
    }
    if (["stop listening", "stop voice", "stop"].includes(lower)) { setHandsFree(false); stopRecognition(); return "Paused."; }
    onSend(text);
    return "Sent to your thread.";
  }, [onNavigate, onSend]);

  const startRecognition = useCallback((continuous: boolean) => {
    const C = voiceWindow().SpeechRecognition ?? voiceWindow().webkitSpeechRecognition;
    if (!C) { flashError("Voice isn't supported in this browser — Chrome or Edge work best."); return; }
    try {
      recRef.current?.abort();
      const r = new C();
      r.lang = localStorage.getItem("aro.voice.language") || navigator.language || "en-US";
      r.continuous = continuous;
      r.interimResults = true;
      restartOnEnd.current = continuous;
      r.onresult = (e) => {
        let fin = "", tmp = "";
        for (let i = e.resultIndex; i < e.results.length; i += 1) { const it = e.results[i]; const ph = it[0]?.transcript ?? ""; if (it.isFinal) fin += ph; else tmp += ph; }
        setInterim(tmp);
        if (fin.trim()) {
          setInterim("");
          if (continuous) { const ack = routeCommand(fin); if (speakBack && ack) say(ack); }
          else setDraft((v) => `${v}${v ? " " : ""}${fin.trim()}`);
        }
      };
      r.onerror = (e) => {
        if (e.error === "not-allowed" || e.error === "service-not-allowed") { restartOnEnd.current = false; flashError("Microphone blocked — allow it in the browser's site settings."); }
        else if (e.error !== "no-speech" && e.error !== "aborted") { restartOnEnd.current = false; flashError(`Voice stopped: ${e.error}`); }
        setListening(false);
      };
      r.onend = () => {
        if (restartOnEnd.current) window.setTimeout(() => { try { recRef.current?.start(); setListening(true); } catch { setListening(false); } }, 240);
        else setListening(false);
      };
      recRef.current = r;
      setError(""); setListening(true); r.start();
    } catch { setListening(false); flashError("Couldn't start the microphone."); }
  }, [routeCommand, speakBack, say]);

  useEffect(() => () => { try { recRef.current?.abort(); window.speechSynthesis?.cancel(); } catch { /* teardown */ } }, []);

  const toggleListen = () => (listening ? stopRecognition() : startRecognition(handsFree));
  const sendDraft = () => { if (!draft.trim()) return; routeCommand(draft); setDraft(""); };

  /* dragging — shared by orb and panel header */
  const beginDrag = (e: React.PointerEvent, allowOnControls = false) => {
    if (!allowOnControls && (e.target as HTMLElement).closest("button,input,textarea,[role=switch]")) return;
    dragRef.current = { x: e.clientX, y: e.clientY, px: position.x, py: position.y, moved: false };
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  };
  const moveDrag = (e: React.PointerEvent) => {
    if (!dragRef.current) return;
    const dx = e.clientX - dragRef.current.x, dy = e.clientY - dragRef.current.y;
    if (Math.abs(dx) + Math.abs(dy) > 4) dragRef.current.moved = true;
    const maxX = window.innerWidth - 80, maxY = window.innerHeight - 90;
    setPosition({ x: Math.max(-16, Math.min(maxX, dragRef.current.px - dx)), y: Math.max(-30, Math.min(maxY, dragRef.current.py - dy)) });
  };
  const endDrag = () => { suppressClick.current = Boolean(dragRef.current?.moved); dragRef.current = null; };

  if (!enabled) return null;

  const tone = orbState === "listening" ? "text-rose" : orbState === "thinking" ? "text-amber" : orbState === "speaking" ? "text-cyan" : orbState === "completed" ? "text-mint" : orbState === "error" ? "text-rose" : "text-ink-3";

  return (
    <div className="fixed right-5 bottom-[40px] z-[80] flex flex-col items-end" style={{ transform: `translate(${-position.x}px, ${-position.y}px)` }}>
      {open && (
        <section role="dialog" aria-label="Voice assistant" className="mb-3 w-[min(340px,calc(100vw-28px))] animate-pop overflow-hidden rounded-[20px] border border-line-strong bg-overlay/95 shadow-e4 backdrop-blur-xl">
          <div onPointerDown={beginDrag} onPointerMove={moveDrag} onPointerUp={endDrag} className="flex cursor-grab items-center gap-2 px-3.5 pt-3 active:cursor-grabbing">
            <span className="font-display text-[12.5px] font-semibold text-ink">Aro</span>
            <span className={cn("flex items-center gap-1 font-mono text-[10px]", tone)}><i className="size-[5px] rounded-full bg-current" />{ORB_LABEL[orbState].toLowerCase()}</span>
            <button onClick={() => { stopRecognition(); setOpen(false); }} className="ml-auto flex size-[26px] cursor-pointer items-center justify-center rounded-full text-ink-4 transition-colors hover:bg-hover hover:text-ink" aria-label="Minimise to orb"><IconX size={13} /></button>
          </div>

          {/* hero orb — tap to talk */}
          <div className="flex flex-col items-center px-4 pt-2 pb-3">
            <button onClick={toggleListen} className="relative cursor-pointer rounded-full transition-transform active:scale-95 focus-visible:ring-focus" aria-label={listening ? "Stop listening" : "Start listening"} aria-pressed={listening}>
              <Orb state={orbState} size={118} levelRef={micLevel} />
            </button>
            <p className={cn("mt-1 text-[12.5px] font-medium", tone)}>{STATE_HINT[orbState]}</p>
            <p className="mt-1 min-h-[34px] max-w-[280px] text-center text-[11.5px] leading-[1.45] text-ink-3">
              {interim ? <span className="italic">“{interim}”</span> : lastHeard ? <span className="text-ink-4">Last: “{lastHeard.slice(0, 90)}”</span> : supported ? <span className="text-ink-4">Try “open tasks”, “new thread”, or just ask.</span> : <span className="text-amber">Voice recognition isn't available here.</span>}
            </p>
            {error && <p role="alert" className="mt-1 text-center text-[10.5px] text-amber">{error}</p>}
          </div>

          {!handsFree && (
            <div className="flex gap-1.5 px-3.5 pb-3">
              <input value={draft} onChange={(e) => setDraft(e.target.value)} onKeyDown={(e) => e.key === "Enter" && sendDraft()} placeholder="Dictated text or a command…" className="h-[34px] min-w-0 flex-1 rounded-[10px] border border-line bg-base px-3 text-[12px] text-ink placeholder:text-ink-4 focus:border-iris/50 focus:outline-none" />
              <button onClick={sendDraft} disabled={!draft.trim()} className={cn("flex size-[34px] shrink-0 cursor-pointer items-center justify-center rounded-[10px] transition-colors", draft.trim() ? "bg-iris text-on-iris" : "bg-track text-ink-4")} aria-label="Send"><IconArrowUp size={14} /></button>
            </div>
          )}

          <div className="space-y-2.5 border-t border-line-soft px-3.5 py-3">
            <label className="flex items-center justify-between gap-3">
              <span><span className="block text-[12px] font-medium text-ink-2">Hands-free</span><span className="block text-[10.5px] text-ink-4">Every phrase is sent or run</span></span>
              <Toggle checked={handsFree} onChange={(v) => { setHandsFree(v); if (listening) { stopRecognition(); if (v) startRecognition(true); } }} size="sm" />
            </label>
            <label className="flex items-center justify-between gap-3">
              <span><span className="block text-[12px] font-medium text-ink-2">Speak replies</span><span className="block text-[10.5px] text-ink-4">Read new answers aloud</span></span>
              <Toggle checked={speakBack} onChange={setSpeakBack} size="sm" />
            </label>
            <div className="flex flex-wrap gap-1.5 pt-0.5">
              {NAV.map((n) => <button key={n.view} onClick={() => { onNavigate(n.view); setOpen(false); }} className="inline-flex h-[26px] cursor-pointer items-center gap-1.5 rounded-full border border-line-soft bg-well px-2.5 text-[11px] text-ink-3 transition-colors hover:border-line-strong hover:text-ink"><n.icon size={11} />{n.label}</button>)}
              {reply && <button onClick={() => say(reply)} className="inline-flex h-[26px] cursor-pointer items-center gap-1.5 rounded-full border border-line-soft bg-well px-2.5 text-[11px] text-ink-3 transition-colors hover:border-line-strong hover:text-ink">Read last reply</button>}
            </div>
          </div>
        </section>
      )}

      {/* minimised: just the orb — still alive, still draggable */}
      {!open && (
        <div className="group relative">
          <button
            onPointerDown={(e) => beginDrag(e, true)} onPointerMove={moveDrag} onPointerUp={endDrag}
            onClick={() => { if (suppressClick.current) { suppressClick.current = false; return; } setOpen(true); }}
            onDoubleClick={toggleListen}
            className="relative flex size-[54px] cursor-grab items-center justify-center rounded-full transition-transform hover:scale-105 active:cursor-grabbing focus-visible:ring-focus"
            aria-label={`Aro voice · ${ORB_LABEL[orbState]} · click to open, double-click to talk`}
          >
            <Orb state={orbState} size={46} levelRef={micLevel} />
          </button>
          <span className={cn("pointer-events-none absolute top-1/2 right-[62px] -translate-y-1/2 rounded-full border border-line-strong bg-overlay px-2.5 py-1 font-mono text-[10px] whitespace-nowrap shadow-e3 transition-all duration-200",
            orbState === "idle" ? "translate-x-1 opacity-0 group-hover:translate-x-0 group-hover:opacity-100" : "opacity-100", tone)}>
            {ORB_LABEL[orbState]}{orbState === "idle" && <Kbd className="ml-1.5">dbl-click</Kbd>}
          </span>
        </div>
      )}
    </div>
  );
}
