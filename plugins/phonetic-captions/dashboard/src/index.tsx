/**
 * Phonetic Captions — Dashboard Plugin UI
 *
 * IIFE plugin that registers with the Hermes dashboard.
 * Uses SDK globals (React, fetchJSON, components) — no bundled React.
 * State-based navigation (no react-router-dom).
 */

import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  Download,
  RefreshCw,
  Play,
  ChevronLeft,
  VideoOff,
  AlertCircle,
  Scissors,
} from "lucide-react";

// ---------------------------------------------------------------------------
// SDK access
// ---------------------------------------------------------------------------

declare global {
  interface Window {
    __HERMES_PLUGIN_SDK__: any;
    __HERMES_PLUGINS__: any;
  }
}

const SDK = window.__HERMES_PLUGIN_SDK__;
const PLUGINS = window.__HERMES_PLUGINS__;

// Grab what we need from SDK
const fetchJSON = SDK?.fetchJSON ?? (async (url: string, opts?: any) => {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
});

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Word {
  word: string;
  start: number;
  end: number;
}

interface CaptionSegment {
  id: number;
  start: number;
  end: number;
  text: string;
  lang: "en" | "vi" | "";  // may be "" on pre-phonetics jobs
  phonetic: string;
  words?: Word[];
}

interface CaptionStyle {
  font: string;
  font_size: number;
  primary_color: string;
  outline_color: string;
  outline_width: number;
  alignment: number;
  margin_bottom: number;
  max_line_length: number;
}

interface CaptionJob {
  id: string;
  created_at: string;
  video_path: string;
  output_path: string;
  style: CaptionStyle;
  segments: CaptionSegment[];
}

interface JobSummary {
  id: string;
  created_at: string;
  video_filename: string;
  segment_count: number;
}

// ---------------------------------------------------------------------------
// API base
// ---------------------------------------------------------------------------

const API = "/api/plugins/phonetic-captions";

// ---------------------------------------------------------------------------
// Spinner (local component — not in SDK)
// ---------------------------------------------------------------------------

function Spinner({ className = "" }: { className?: string }) {
  return (
    <div
      className={`animate-spin rounded-full border-2 border-current border-t-transparent w-4 h-4 ${className}`}
      role="status"
      aria-label="Loading"
    />
  );
}

// ---------------------------------------------------------------------------
// Job List
// ---------------------------------------------------------------------------

function JobListView({ onSelect }: { onSelect: (id: string) => void }) {
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchJSON(`${API}/jobs`)
      .then((data: JobSummary[]) => setJobs(data))
      .catch((e: any) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center p-16 text-muted-foreground">
        <Spinner className="w-5 h-5 mr-3" />
        <span className="text-sm">Loading jobs…</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 flex items-start gap-3 text-destructive">
        <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
        <div>
          <p className="font-medium text-sm">Failed to load caption jobs</p>
          <p className="text-xs mt-1 text-destructive/70">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-2xl">
      <h1 className="text-xl font-semibold mb-1 text-foreground">Caption Jobs</h1>
      <p className="text-sm text-muted-foreground mb-5">
        Jobs are created when Hermes captions a video. Click a job to edit segments and re-burn.
      </p>
      {jobs.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-16 text-muted-foreground border border-dashed border-border rounded-xl">
          <VideoOff className="w-10 h-10 mb-3 opacity-40" />
          <p className="text-sm font-medium">No caption jobs yet</p>
          <p className="text-xs mt-1 text-muted-foreground/70">Send a video via Telegram and Hermes will create one.</p>
        </div>
      ) : (
        <div className="space-y-2">
          {jobs.map((job) => (
            <button
              key={job.id}
              onClick={() => onSelect(job.id)}
              className="w-full flex items-center justify-between rounded-lg border border-border p-4 bg-card hover:bg-accent transition-colors text-left group"
            >
              <div className="min-w-0">
                <div className="font-medium text-sm truncate text-foreground">{job.video_filename || job.id}</div>
                <div className="text-xs text-muted-foreground mt-0.5">
                  {job.segment_count} segment{job.segment_count !== 1 ? "s" : ""}
                  {job.created_at ? ` · ${new Date(job.created_at).toLocaleString()}` : ""}
                </div>
              </div>
              <Play className="w-4 h-4 text-muted-foreground group-hover:text-foreground shrink-0 ml-4 transition-colors" />
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Video Player with error / loading states
// ---------------------------------------------------------------------------

function VideoPlayer({
  src,
}: {
  src: string;
}) {
  const [state, setState] = useState<"loading" | "ready" | "error">("loading");

  useEffect(() => {
    setState("loading");
  }, [src]);

  return (
    <div
      className="relative w-full rounded-lg overflow-hidden bg-zinc-900"
      style={{ maxHeight: "55vh", minHeight: "160px" }}
    >
      {state === "error" ? (
        <div className="flex flex-col items-center justify-center gap-2 py-10 px-4 text-muted-foreground">
          <VideoOff className="w-8 h-8 opacity-40" />
          <p className="text-sm font-medium">Video not available</p>
          <p className="text-xs text-muted-foreground/70 text-center">
            The output file may not exist yet.{" "}
            <span className="font-medium">Re-burn</span> to generate it.
          </p>
        </div>
      ) : (
        <video
          key={src}
          src={src}
          controls
          className="w-full h-full object-contain"
          style={{ maxHeight: "55vh", display: state === "loading" ? "none" : "block" }}
          onCanPlay={() => setState("ready")}
          onError={() => setState("error")}
        />
      )}
      {state === "loading" && (
        <div className="absolute inset-0 flex items-center justify-center">
          <Spinner className="w-5 h-5 text-zinc-400" />
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Word Split Editor
// ---------------------------------------------------------------------------

function SplitEditor({
  seg,
  splitBefore,
  onToggle,
  onApply,
  onCancel,
}: {
  seg: CaptionSegment;
  splitBefore: Set<number>;
  onToggle: (wordIdx: number) => void;
  onApply: () => void;
  onCancel: () => void;
}) {
  const words = seg.words ?? [];
  return (
    <div className="mt-2 rounded-lg bg-muted border border-border p-3">
      <p className="text-xs text-muted-foreground mb-2.5">
        Click ✂ between words to mark split points:
      </p>
      <div className="flex flex-wrap items-end gap-y-2 gap-x-0.5">
        {words.map((w, i) => (
          <React.Fragment key={i}>
            {i > 0 && (
              <button
                onClick={() => onToggle(i)}
                title={`Split before "${w.word.trim()}"`}
                className={`self-center w-6 h-5 flex items-center justify-center rounded text-xs font-bold transition-colors ${
                  splitBefore.has(i)
                    ? "bg-orange-400 dark:bg-orange-500 text-white"
                    : "bg-secondary text-muted-foreground hover:bg-orange-200/40 hover:text-orange-500"
                }`}
              >
                ✂
              </button>
            )}
            <div className="flex flex-col items-center">
              <span className="px-1.5 py-0.5 rounded bg-card border border-border text-xs font-medium text-foreground whitespace-nowrap">
                {w.word.trim()}
              </span>
              <span className="text-[10px] text-muted-foreground tabular-nums mt-0.5">
                {formatTime(w.start)}
              </span>
            </div>
          </React.Fragment>
        ))}
      </div>
      <div className="flex gap-2 mt-3">
        <button
          onClick={onApply}
          disabled={splitBefore.size === 0}
          className="px-3 py-1 rounded bg-orange-500 text-white text-xs font-medium hover:bg-orange-600 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {splitBefore.size > 0
            ? `Apply — ${splitBefore.size + 1} segments`
            : "Select a split point"}
        </button>
        <button
          onClick={onCancel}
          className="px-3 py-1 rounded border border-border text-xs font-medium text-foreground hover:bg-accent transition-colors"
        >
          Cancel
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Editor
// ---------------------------------------------------------------------------

function EditorView({ jobId, onBack }: { jobId: string; onBack: () => void }) {
  const [job, setJob] = useState<CaptionJob | null>(null);
  const [segments, setSegments] = useState<CaptionSegment[]>([]);
  const [style, setStyle] = useState<CaptionStyle | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [burning, setBurning] = useState(false);
  const [burnError, setBurnError] = useState<string | null>(null);
  const [burnSuccess, setBurnSuccess] = useState(false);
  const [burnTimestamp, setBurnTimestamp] = useState(Date.now());
  const videoRef = useRef<HTMLVideoElement>(null);
  const [splitState, setSplitState] = useState<{ segIdx: number; splitBefore: Set<number> } | null>(null);

  useEffect(() => {
    setLoading(true);
    fetchJSON(`${API}/jobs/${jobId}`)
      .then((data: CaptionJob) => {
        console.log("[captions] job loaded", data.id, "segments:", data.segments?.length, data.segments?.[0]);
        setJob(data);
        setSegments(data.segments);
        setStyle(data.style);
      })
      .catch((e: any) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [jobId]);

  const updateSegment = useCallback((idx: number, field: keyof CaptionSegment, value: string) => {
    setSegments((prev) =>
      prev.map((s, i) => (i === idx ? { ...s, [field]: value } : s))
    );
  }, []);

  const toggleLang = useCallback((idx: number) => {
    setSegments((prev) =>
      prev.map((s, i) => {
        if (i !== idx) return s;
        const wasVi = (s.lang || "en") === "vi";
        return { ...s, lang: wasVi ? "en" : "vi", phonetic: wasVi ? "" : s.phonetic };
      })
    );
  }, []);

  const openSplit = useCallback((idx: number) => {
    setSplitState({ segIdx: idx, splitBefore: new Set() });
  }, []);

  const toggleSplitPoint = useCallback((wordIdx: number) => {
    setSplitState((prev) => {
      if (!prev) return null;
      const next = new Set(prev.splitBefore);
      if (next.has(wordIdx)) next.delete(wordIdx);
      else next.add(wordIdx);
      return { ...prev, splitBefore: next };
    });
  }, []);

  const applyWordSplit = useCallback(() => {
    if (!splitState) return;
    const { segIdx, splitBefore } = splitState;
    const seg = segments[segIdx];
    const words = seg.words ?? [];
    if (words.length < 2 || splitBefore.size === 0) {
      setSplitState(null);
      return;
    }
    const sortedSplits = Array.from(splitBefore).sort((a, b) => a - b);
    const boundaries = [0, ...sortedSplits, words.length];
    const newSegs: CaptionSegment[] = [];
    for (let j = 0; j < boundaries.length - 1; j++) {
      const group = words.slice(boundaries[j], boundaries[j + 1]);
      const text = group.map((w) => w.word.trim()).join(" ").trim();
      if (!text) continue;
      newSegs.push({
        ...seg,
        id: 0,
        start: group[0].start,
        end: group[group.length - 1].end,
        text,
        phonetic: "",
        words: group,
      });
    }
    if (newSegs.length < 2) {
      setSplitState(null);
      return;
    }
    setSegments((prev) =>
      [...prev.slice(0, segIdx), ...newSegs, ...prev.slice(segIdx + 1)].map((s, i) => ({
        ...s,
        id: i,
      }))
    );
    setSplitState(null);
  }, [splitState, segments]);

  const handleReburn = async () => {
    if (!style) return;
    setBurning(true);
    setBurnError(null);
    setBurnSuccess(false);

    try {
      await fetchJSON(`${API}/jobs/${jobId}/segments`, {
        method: "PUT",
        body: JSON.stringify({ segments }),
        headers: { "Content-Type": "application/json" },
      });
      await fetchJSON(`${API}/jobs/${jobId}/style`, {
        method: "PUT",
        body: JSON.stringify({ style }),
        headers: { "Content-Type": "application/json" },
      });
      await fetchJSON(`${API}/jobs/${jobId}/burn`, { method: "POST" });
      setBurnTimestamp(Date.now());
      setBurnSuccess(true);
    } catch (e: any) {
      setBurnError(String(e));
    } finally {
      setBurning(false);
    }
  };

  const handleDownload = async () => {
    setBurnError(null);
    try {
      const res = await fetch(`${API}/jobs/${jobId}/download`);
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `captioned_${jobId}.mp4`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e: any) {
      setBurnError(`Download failed: ${e}`);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground">
        <Spinner className="w-5 h-5 mr-3" />
        <span className="text-sm">Loading job…</span>
      </div>
    );
  }
  if (error) {
    return (
      <div className="p-8 flex items-start gap-3 text-destructive">
        <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
        <div>
          <p className="font-medium">Failed to load job</p>
          <p className="text-sm mt-1 text-destructive/70">{error}</p>
        </div>
      </div>
    );
  }
  if (!job || !style) return null;

  const filename = job.video_path.split("/").pop() ?? job.id;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* ── Header ── */}
      <div className="flex items-center gap-2 px-5 py-3 border-b border-border shrink-0">
        <button
          onClick={onBack}
          className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          <ChevronLeft className="w-4 h-4" />
          All jobs
        </button>
        <span className="text-muted-foreground opacity-40">/</span>
        <span className="text-sm font-medium truncate text-foreground">{filename}</span>
        <span className="ml-auto text-xs text-muted-foreground shrink-0">
          {segments.length} segment{segments.length !== 1 ? "s" : ""}
        </span>
      </div>

      {/* ── Body ── */}
      <div className="flex flex-1 overflow-hidden">

        {/* ── Left: video + actions ── */}
        <div className="w-[420px] shrink-0 flex flex-col border-r border-border overflow-y-auto">
          <div className="p-4">
            <VideoPlayer src={`${API}/jobs/${jobId}/video?t=${burnTimestamp}`} />
          </div>

          {/* Actions */}
          <div className="px-4 pb-3 flex gap-2">
            <button
              onClick={handleReburn}
              disabled={burning}
              className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-orange-500 hover:bg-orange-600 active:bg-orange-700 text-white font-semibold text-sm disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-150"
            >
              {burning ? <Spinner className="w-4 h-4" /> : <RefreshCw className="w-4 h-4" />}
              {burning ? "Burning…" : "Re-burn"}
            </button>
            <button
              onClick={handleDownload}
              className="flex items-center gap-2 px-4 py-2 rounded-lg border border-border hover:bg-accent text-foreground transition-colors text-sm font-medium"
            >
              <Download className="w-4 h-4" />
              Download
            </button>
          </div>

          {/* Feedback */}
          {burnError && (
            <div className="mx-4 mb-3 flex items-start gap-2 rounded-lg bg-destructive/10 border border-destructive/30 px-3 py-2">
              <AlertCircle className="w-4 h-4 text-destructive shrink-0 mt-0.5" />
              <p className="text-xs text-destructive break-words">{burnError}</p>
            </div>
          )}
          {burnSuccess && !burnError && (
            <div className="mx-4 mb-3 rounded-lg bg-success/10 border border-success/30 px-3 py-2">
              <p className="text-xs text-success">✓ Video re-burned successfully.</p>
            </div>
          )}

        </div>

        {/* ── Right: segment editor + style settings ── */}
        <div className="flex-1 overflow-y-auto p-5">
          <div className="max-w-2xl mx-auto">
          <div className="flex items-baseline gap-2 mb-4">
            <h2 className="text-base font-semibold text-foreground">Segments</h2>
            <span className="text-xs text-muted-foreground">{segments.length} total · edits are saved on Re-burn</span>
          </div>

          {segments.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-muted-foreground border border-dashed border-border rounded-xl">
              <p className="text-sm">No segments in this job.</p>
              <p className="text-xs mt-1 text-muted-foreground/70">The transcription may have returned nothing.</p>
            </div>
          ) : (
            <div className="space-y-2 pb-2">
              {segments.map((seg, idx) => {
                const lang = (seg.lang === "vi" ? "vi" : "en") as "en" | "vi";
                return (
                  <div
                    key={`${seg.id}-${idx}`}
                    className="rounded-lg border border-border bg-card p-3 space-y-2 text-sm transition-colors"
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono text-muted-foreground w-7 shrink-0 text-right">#{idx + 1}</span>
                      <span className="text-xs tabular-nums text-muted-foreground shrink-0 w-[5.5rem]">
                        {formatTime(seg.start)}–{formatTime(seg.end)}
                      </span>
                      <button
                        onClick={() => toggleLang(idx)}
                        title="Toggle EN / VI"
                        className={`px-2 py-0.5 rounded text-xs font-semibold shrink-0 transition-colors ${
                          lang === "vi"
                            ? "bg-blue-500/20 text-blue-300 hover:bg-blue-500/30"
                            : "bg-secondary text-secondary-foreground hover:bg-accent"
                        }`}
                      >
                        {lang.toUpperCase()}
                      </button>
                      <input
                        className="flex-1 min-w-0 bg-transparent border-b border-transparent hover:border-border focus:border-ring outline-none px-1 py-0.5 transition-colors text-foreground"
                        value={seg.text}
                        onChange={(e) => updateSegment(idx, "text", e.target.value)}
                        placeholder="(no text)"
                      />
                      {(seg.words?.length ?? 0) >= 2 && (
                        <button
                          onClick={() =>
                            splitState?.segIdx === idx
                              ? setSplitState(null)
                              : openSplit(idx)
                          }
                          title="Split segment at word boundary"
                          className={`shrink-0 p-1 rounded transition-colors ${
                            splitState?.segIdx === idx
                              ? "text-orange-500"
                              : "text-muted-foreground/40 hover:text-muted-foreground"
                          }`}
                        >
                          <Scissors className="w-3.5 h-3.5" />
                        </button>
                      )}
                    </div>
                    {lang === "vi" && (
                      <div className="flex items-center gap-2 pl-[calc(1.75rem+5.5rem+3rem+0.75rem)]">
                        <span className="text-xs text-muted-foreground shrink-0">phonetic</span>
                        <input
                          className="flex-1 min-w-0 bg-transparent border-b border-transparent hover:border-border focus:border-ring outline-none px-1 py-0.5 italic text-muted-foreground transition-colors"
                          value={seg.phonetic}
                          onChange={(e) => updateSegment(idx, "phonetic", e.target.value)}
                          placeholder="[pronunciation guide]"
                        />
                      </div>
                    )}
                    {splitState?.segIdx === idx && (
                      <SplitEditor
                        seg={seg}
                        splitBefore={splitState.splitBefore}
                        onToggle={toggleSplitPoint}
                        onApply={applyWordSplit}
                        onCancel={() => setSplitState(null)}
                      />
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {/* ── Style settings (below segments) ── */}
          <div className="border-t border-border mt-6 pb-8">
            <h2 className="text-base font-semibold mt-6 mb-4 text-foreground">Caption Style</h2>
            <div className="space-y-2.5">
              <StyleField label="Font" value={style.font} onChange={(v) => setStyle((s) => s && ({ ...s, font: v }))} placeholder="e.g. Arial, Impact, Trebuchet MS" />
              <StyleNumberField label="Font size" value={style.font_size} onChange={(v) => setStyle((s) => s && ({ ...s, font_size: v }))} />
              <StyleColorField label="Text color" value={style.primary_color} onChange={(v) => setStyle((s) => s && ({ ...s, primary_color: v }))} />
              <StyleColorField label="Outline" value={style.outline_color} onChange={(v) => setStyle((s) => s && ({ ...s, outline_color: v }))} />
              <StyleNumberField label="Outline width" value={style.outline_width} onChange={(v) => setStyle((s) => s && ({ ...s, outline_width: v }))} />
              <StyleNumberField label="Bottom margin" value={style.margin_bottom} onChange={(v) => setStyle((s) => s && ({ ...s, margin_bottom: v }))} />
            </div>
            <p className="text-xs text-muted-foreground mt-3">Style changes apply on the next Re-burn.</p>
          </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Style field helpers
// ---------------------------------------------------------------------------

function StyleField({ label, value, onChange, placeholder }: { label: string; value: string; onChange: (v: string) => void; placeholder?: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-24 shrink-0 text-xs text-muted-foreground">{label}</span>
      <input
        className="flex-1 min-w-0 bg-card border border-border rounded px-2 py-1 text-xs text-foreground outline-none focus:border-ring focus:ring-1 focus:ring-ring/30 placeholder:text-muted-foreground transition-colors"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
      />
    </div>
  );
}

function StyleNumberField({ label, value, onChange }: { label: string; value: number; onChange: (v: number) => void }) {
  const [raw, setRaw] = useState(String(value));

  // Sync if parent value changes externally (e.g. job load)
  useEffect(() => {
    setRaw(String(value));
  }, [value]);

  return (
    <div className="flex items-center gap-2">
      <span className="w-24 shrink-0 text-xs text-muted-foreground">{label}</span>
      <input
        type="number"
        className="w-20 bg-card border border-border rounded px-2 py-1 text-xs text-foreground outline-none focus:border-ring focus:ring-1 focus:ring-ring/30 transition-colors"
        value={raw}
        onChange={(e) => {
          setRaw(e.target.value);
          const n = Number(e.target.value);
          if (e.target.value !== "" && !Number.isNaN(n)) onChange(n);
        }}
        onBlur={() => {
          // On blur, snap back to the committed numeric value if field is empty/invalid
          const n = Number(raw);
          if (raw === "" || Number.isNaN(n)) setRaw(String(value));
        }}
      />
    </div>
  );
}

function StyleColorField({ label, value, onChange }: { label: string; value: string; onChange: (v: string) => void }) {
  const toHex = (ass: string): string => {
    const m = ass.match(/^&H[0-9A-Fa-f]{2}([0-9A-Fa-f]{2})([0-9A-Fa-f]{2})([0-9A-Fa-f]{2})$/);
    if (!m) return "#ffffff";
    const [, bb, gg, rr] = m;
    return `#${rr}${gg}${bb}`.toLowerCase();
  };
  const toAss = (hex: string): string => {
    const m = hex.match(/^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i);
    if (!m) return value;
    const [, r, g, b] = m;
    return `&H00${b}${g}${r}`.toUpperCase();
  };

  return (
    <div className="flex items-center gap-2">
      <span className="w-24 shrink-0 text-xs text-muted-foreground">{label}</span>
      <div className="flex items-center gap-2">
        <input
          type="color"
          className="w-7 h-6 rounded cursor-pointer border border-border p-0"
          value={toHex(value)}
          onChange={(e) => onChange(toAss(e.target.value))}
        />
        <span className="text-xs font-mono text-muted-foreground">{toHex(value)}</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

function formatTime(seconds: number): string {
  if (seconds == null || isNaN(Number(seconds))) return "?:??";
  const n = Number(seconds);
  const m = Math.floor(n / 60);
  const s = (n % 60).toFixed(1).padStart(4, "0");
  return `${m}:${s}`;
}

// ---------------------------------------------------------------------------
// Main Plugin Component
// ---------------------------------------------------------------------------

function CaptionApp() {
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);

  // Check URL on mount — if /captions/<id>, auto-select
  useEffect(() => {
    const path = window.location.pathname;
    const match = path.match(/\/captions\/([^/]+)/);
    if (match) {
      setSelectedJobId(match[1]);
    }
  }, []);

  const handleSelect = (id: string) => {
    setSelectedJobId(id);
    window.history.pushState(null, "", `/captions/${id}`);
  };

  const handleBack = () => {
    setSelectedJobId(null);
    window.history.pushState(null, "", "/captions");
  };

  if (selectedJobId) {
    return <EditorView jobId={selectedJobId} onBack={handleBack} />;
  }
  return <JobListView onSelect={handleSelect} />;
}

// ---------------------------------------------------------------------------
// Register with Hermes
// ---------------------------------------------------------------------------

if (PLUGINS && SDK) {
  PLUGINS.register("phonetic-captions", CaptionApp);
}
