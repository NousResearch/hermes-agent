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

interface CaptionSegment {
  id: number;
  start: number;
  end: number;
  text: string;
  lang: "en" | "vi" | "";  // may be "" on pre-phonetics jobs
  phonetic: string;
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
      <div className="flex items-center justify-center p-16 text-zinc-400">
        <Spinner className="w-5 h-5 mr-3" />
        <span className="text-sm">Loading jobs…</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 flex items-start gap-3 text-red-500">
        <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
        <div>
          <p className="font-medium text-sm">Failed to load caption jobs</p>
          <p className="text-xs mt-1 text-red-400">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-2xl">
      <h1 className="text-xl font-semibold mb-1">Caption Jobs</h1>
      <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-5">
        Jobs are created when Hermes captions a video. Click a job to edit segments and re-burn.
      </p>
      {jobs.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-16 text-zinc-400 border border-dashed border-zinc-300 dark:border-zinc-700 rounded-xl">
          <VideoOff className="w-10 h-10 mb-3 opacity-40" />
          <p className="text-sm font-medium">No caption jobs yet</p>
          <p className="text-xs mt-1 text-zinc-500">Send a video via Telegram and Hermes will create one.</p>
        </div>
      ) : (
        <div className="space-y-2">
          {jobs.map((job) => (
            <button
              key={job.id}
              onClick={() => onSelect(job.id)}
              className="w-full flex items-center justify-between rounded-lg border border-zinc-200 dark:border-zinc-700 p-4 hover:bg-zinc-50 dark:hover:bg-zinc-800/50 transition-colors text-left group"
            >
              <div className="min-w-0">
                <div className="font-medium text-sm truncate">{job.video_filename || job.id}</div>
                <div className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                  {job.segment_count} segment{job.segment_count !== 1 ? "s" : ""}
                  {job.created_at ? ` · ${new Date(job.created_at).toLocaleString()}` : ""}
                </div>
              </div>
              <Play className="w-4 h-4 text-zinc-400 group-hover:text-zinc-600 dark:group-hover:text-zinc-200 shrink-0 ml-4 transition-colors" />
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
        <div className="flex flex-col items-center justify-center gap-2 py-10 px-4 text-zinc-400">
          <VideoOff className="w-8 h-8 opacity-40" />
          <p className="text-sm font-medium">Video not available</p>
          <p className="text-xs text-zinc-500 text-center">
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
      <div className="flex items-center justify-center h-full text-zinc-400">
        <Spinner className="w-5 h-5 mr-3" />
        <span className="text-sm">Loading job…</span>
      </div>
    );
  }
  if (error) {
    return (
      <div className="p-8 flex items-start gap-3 text-red-500">
        <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
        <div>
          <p className="font-medium">Failed to load job</p>
          <p className="text-sm mt-1 text-red-400">{error}</p>
        </div>
      </div>
    );
  }
  if (!job || !style) return null;

  const filename = job.video_path.split("/").pop() ?? job.id;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* ── Header ── */}
      <div className="flex items-center gap-2 px-5 py-3 border-b border-zinc-200 dark:border-zinc-800 shrink-0">
        <button
          onClick={onBack}
          className="flex items-center gap-1 text-sm text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100 transition-colors"
        >
          <ChevronLeft className="w-4 h-4" />
          All jobs
        </button>
        <span className="text-zinc-300 dark:text-zinc-600">/</span>
        <span className="text-sm font-medium truncate">{filename}</span>
        <span className="ml-auto text-xs text-zinc-400 shrink-0">
          {segments.length} segment{segments.length !== 1 ? "s" : ""}
        </span>
      </div>

      {/* ── Body ── */}
      <div className="flex flex-1 overflow-hidden">

        {/* ── Left: video + actions ── */}
        <div className="w-[420px] shrink-0 flex flex-col border-r border-zinc-200 dark:border-zinc-800 overflow-y-auto">
          <div className="p-4">
            <VideoPlayer src={`${API}/jobs/${jobId}/video?t=${burnTimestamp}`} />
          </div>

          {/* Actions */}
          <div className="px-4 pb-3 flex gap-2">
            <button
              onClick={handleReburn}
              disabled={burning}
              className="flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-zinc-900 text-white hover:bg-zinc-700 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-300 disabled:opacity-50 transition-colors text-sm font-medium"
            >
              {burning ? <Spinner className="w-4 h-4" /> : <RefreshCw className="w-4 h-4" />}
              {burning ? "Burning…" : "Re-burn"}
            </button>
            <button
              onClick={handleDownload}
              className="flex items-center gap-2 px-4 py-2 rounded-lg border border-zinc-300 dark:border-zinc-600 hover:bg-zinc-50 dark:hover:bg-zinc-800 transition-colors text-sm font-medium"
            >
              <Download className="w-4 h-4" />
              Download
            </button>
          </div>

          {/* Feedback */}
          {burnError && (
            <div className="mx-4 mb-3 flex items-start gap-2 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 px-3 py-2">
              <AlertCircle className="w-4 h-4 text-red-500 shrink-0 mt-0.5" />
              <p className="text-xs text-red-600 dark:text-red-400 break-words">{burnError}</p>
            </div>
          )}
          {burnSuccess && !burnError && (
            <div className="mx-4 mb-3 rounded-lg bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 px-3 py-2">
              <p className="text-xs text-green-700 dark:text-green-400">✓ Video re-burned successfully.</p>
            </div>
          )}

        </div>

        {/* ── Right: segment editor + style settings ── */}
        <div className="flex-1 overflow-y-auto p-5">
          <div className="max-w-2xl mx-auto">
          <div className="flex items-baseline gap-2 mb-4">
            <h2 className="text-base font-semibold">Segments</h2>
            <span className="text-xs text-zinc-400">{segments.length} total · edits are saved on Re-burn</span>
          </div>

          {segments.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-zinc-400 border border-dashed border-zinc-300 dark:border-zinc-700 rounded-xl">
              <p className="text-sm">No segments in this job.</p>
              <p className="text-xs mt-1 text-zinc-500">The transcription may have returned nothing.</p>
            </div>
          ) : (
            <div className="space-y-2 pb-2">
              {segments.map((seg, idx) => {
                const lang = (seg.lang === "vi" ? "vi" : "en") as "en" | "vi";
                return (
                  <div
                    key={`${seg.id}-${idx}`}
                    className="rounded-lg border border-zinc-200 dark:border-zinc-700 p-3 space-y-2 text-sm hover:border-zinc-300 dark:hover:border-zinc-600 transition-colors"
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono text-zinc-400 w-7 shrink-0 text-right">#{idx + 1}</span>
                      <span className="text-xs tabular-nums text-zinc-400 shrink-0 w-[5.5rem]">
                        {formatTime(seg.start)}–{formatTime(seg.end)}
                      </span>
                      <button
                        onClick={() => toggleLang(idx)}
                        title="Toggle EN / VI"
                        className={`px-2 py-0.5 rounded text-xs font-semibold shrink-0 transition-colors ${
                          lang === "vi"
                            ? "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300 hover:bg-blue-200 dark:hover:bg-blue-800/60"
                            : "bg-zinc-100 text-zinc-600 dark:bg-zinc-700 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-600"
                        }`}
                      >
                        {lang.toUpperCase()}
                      </button>
                      <input
                        className="flex-1 min-w-0 bg-transparent border-b border-transparent hover:border-zinc-300 focus:border-blue-500 dark:hover:border-zinc-600 dark:focus:border-blue-400 outline-none px-1 py-0.5 transition-colors text-zinc-900 dark:text-zinc-100"
                        value={seg.text}
                        onChange={(e) => updateSegment(idx, "text", e.target.value)}
                        placeholder="(no text)"
                      />
                    </div>
                    {lang === "vi" && (
                      <div className="flex items-center gap-2 pl-[calc(1.75rem+5.5rem+3rem+0.75rem)]">
                        <span className="text-xs text-zinc-400 shrink-0">phonetic</span>
                        <input
                          className="flex-1 min-w-0 bg-transparent border-b border-transparent hover:border-zinc-300 focus:border-blue-500 dark:hover:border-zinc-600 dark:focus:border-blue-400 outline-none px-1 py-0.5 italic text-zinc-600 dark:text-zinc-300 transition-colors"
                          value={seg.phonetic}
                          onChange={(e) => updateSegment(idx, "phonetic", e.target.value)}
                          placeholder="[pronunciation guide]"
                        />
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {/* ── Style settings (below segments) ── */}
          <div className="border-t border-zinc-200 dark:border-zinc-800 mt-6 pt-5 pb-8">
            <p className="text-xs font-semibold uppercase tracking-wide text-zinc-400 dark:text-zinc-500 mb-3">Caption Style</p>
            <div className="space-y-2.5">
              <StyleField label="Font" value={style.font} onChange={(v) => setStyle((s) => s && ({ ...s, font: v }))} />
              <StyleNumberField label="Font size" value={style.font_size} onChange={(v) => setStyle((s) => s && ({ ...s, font_size: v }))} />
              <StyleColorField label="Text color" value={style.primary_color} onChange={(v) => setStyle((s) => s && ({ ...s, primary_color: v }))} />
              <StyleColorField label="Outline" value={style.outline_color} onChange={(v) => setStyle((s) => s && ({ ...s, outline_color: v }))} />
              <StyleNumberField label="Outline width" value={style.outline_width} onChange={(v) => setStyle((s) => s && ({ ...s, outline_width: v }))} />
              <StyleNumberField label="Bottom margin" value={style.margin_bottom} onChange={(v) => setStyle((s) => s && ({ ...s, margin_bottom: v }))} />
            </div>
            <p className="text-xs text-zinc-400 mt-3">Style changes apply on the next Re-burn.</p>
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

function StyleField({ label, value, onChange }: { label: string; value: string; onChange: (v: string) => void }) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-24 shrink-0 text-xs text-zinc-500">{label}</span>
      <input
        className="flex-1 min-w-0 bg-transparent border border-zinc-300 dark:border-zinc-600 rounded px-2 py-1 text-xs outline-none focus:border-blue-500 transition-colors text-zinc-900 dark:text-zinc-100"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    </div>
  );
}

function StyleNumberField({ label, value, onChange }: { label: string; value: number; onChange: (v: number) => void }) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-24 shrink-0 text-xs text-zinc-500">{label}</span>
      <input
        type="number"
        className="w-20 bg-transparent border border-zinc-300 dark:border-zinc-600 rounded px-2 py-1 text-xs outline-none focus:border-blue-500 transition-colors text-zinc-900 dark:text-zinc-100"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
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
      <span className="w-24 shrink-0 text-xs text-zinc-500">{label}</span>
      <div className="flex items-center gap-2">
        <input
          type="color"
          className="w-7 h-6 rounded cursor-pointer border border-zinc-300 dark:border-zinc-600 p-0"
          value={toHex(value)}
          onChange={(e) => onChange(toAss(e.target.value))}
        />
        <span className="text-xs font-mono text-zinc-400">{toHex(value)}</span>
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
