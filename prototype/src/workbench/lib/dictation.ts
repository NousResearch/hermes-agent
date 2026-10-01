"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type ResultEvent = { resultIndex: number; results: ArrayLike<ArrayLike<{ transcript: string }> & { isFinal: boolean }> };
type Rec = { lang: string; continuous: boolean; interimResults: boolean; onresult: ((e: ResultEvent) => void) | null; onerror: ((e: { error: string }) => void) | null; onend: (() => void) | null; start: () => void; stop: () => void; abort: () => void };
type Ctor = new () => Rec;
const ctor = () => (window as Window & { SpeechRecognition?: Ctor; webkitSpeechRecognition?: Ctor }).SpeechRecognition ?? (window as Window & { webkitSpeechRecognition?: Ctor }).webkitSpeechRecognition;

export type Engine = "browser" | "whisper-local" | "whisper-cloud";
export const ENGINES: { id: Engine; label: string; short: string; note: string }[] = [
  { id: "browser", label: "Browser engine", short: "browser", note: "Built-in Web Speech recognition. No setup." },
  { id: "whisper-local", label: "Whisper · local daemon", short: "whisper", note: "Faster, punctuation-aware. Needs the Aro daemon on :4735." },
  { id: "whisper-cloud", label: "Whisper · cloud", short: "turbo", note: "Lowest latency streaming. Audio leaves the device." },
];
export const engineMeta = (id: Engine) => ENGINES.find((e) => e.id === id) ?? ENGINES[0];
export const readEngine = (): Engine => {
  const v = localStorage.getItem("aro.voice.engine") as Engine | null;
  return v && ENGINES.some((e) => e.id === v) ? v : "browser";
};

/** Plain dictation: speech → text appended to a field. No commands, no hands-free loop. */
export function useDictation(onFinal: (text: string) => void, onError?: (msg: string) => void) {
  const [listening, setListening] = useState(false);
  const [interim, setInterim] = useState("");
  const [engine, setEngine] = useState<Engine>(() => (typeof window === "undefined" ? "browser" : readEngine()));
  const ref = useRef<Rec | null>(null);
  const supported = typeof window !== "undefined" && Boolean(ctor());
  /* let the voice orb mirror composer dictation */
  useEffect(() => { window.dispatchEvent(new CustomEvent("aro:dictation", { detail: listening })); }, [listening]);
  useEffect(() => {
    const h = (e: Event) => setEngine((e as CustomEvent<Engine>).detail);
    window.addEventListener("aro:voice:engine", h);
    return () => window.removeEventListener("aro:voice:engine", h);
  }, []);

  const stop = useCallback(() => { try { ref.current?.stop(); } catch { /* already stopped */ } setListening(false); setInterim(""); }, []);
  const start = useCallback(() => {
    const C = ctor();
    if (!C) { onError?.("Dictation isn't supported in this browser. Chrome or Edge work best."); return; }
    try {
      ref.current?.abort();
      const r = new C();
      r.lang = localStorage.getItem("aro.voice.language") || navigator.language || "en-US";
      r.continuous = true;
      r.interimResults = true;
      r.onresult = (e) => {
        let fin = "", tmp = "";
        for (let i = e.resultIndex; i < e.results.length; i += 1) { const it = e.results[i]; const t = it[0]?.transcript ?? ""; if (it.isFinal) fin += t; else tmp += t; }
        setInterim(tmp);
        if (fin.trim()) { onFinal(fin.trim()); setInterim(""); }
      };
      r.onerror = (e) => { if (e.error === "not-allowed" || e.error === "service-not-allowed") onError?.("Microphone permission denied — allow it in the browser's site settings."); else if (e.error !== "no-speech" && e.error !== "aborted") onError?.(`Dictation stopped: ${e.error}`); setListening(false); };
      r.onend = () => setListening(false);
      ref.current = r; r.start(); setListening(true);
    } catch { onError?.("Couldn't start the microphone."); setListening(false); }
  }, [onFinal, onError]);
  const toggle = useCallback(() => {
    if (listening) { stop(); return; }
    /* Whisper engines stream through the daemon; without it we fall back rather than fail. */
    if (engine !== "browser") onError?.(`${engineMeta(engine).label} unavailable — using the browser engine for now.`);
    start();
  }, [listening, start, stop, engine, onError]);
  useEffect(() => () => { try { ref.current?.abort(); } catch { /* teardown */ } }, []);
  return { listening, interim, supported, engine, setEngine, start, stop, toggle };
}
