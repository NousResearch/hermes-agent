/**
 * Phonetic Captions — Dashboard Plugin UI
 *
 * IIFE plugin that registers with the Hermes dashboard.
 * Uses SDK globals (React, fetchJSON, components) — no bundled React.
 * State-based navigation (no react-router-dom).
 */

import { useState, useEffect, useRef, useCallback } from "react";
import { Download, RefreshCw, Play, ChevronLeft, FileText } from "lucide-react";

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
  lang: "en" | "vi";
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

  if (loading) return <div className="flex justify-center p-12"><Spinner /></div>;
  if (error) return <div className="p-6 text-red-500">{error}</div>;

  return (
    <div className="p-6 max-w-3xl">
      <h1 className="text-xl font-semibold mb-4">Caption Jobs</h1>
      {jobs.length === 0 ? (
        <p className="text-zinc-500 dark:text-zinc-400">
          No caption jobs yet. Send a video via Telegram and Hermes will create one.
        </p>
      ) : (
        <div className="space-y-2">
          {jobs.map((job) => (
            <button
              key={job.id}
              onClick={() => onSelect(job.id)}
              className="w-full flex items-center justify-between rounded-lg border border-zinc-200 dark:border-zinc-700 p-4 hover:bg-zinc-50 dark:hover:bg-zinc-800/50 transition-colors text-left"
            >
              <div>
                <div className="font-medium">{job.video_filename || job.id}</div>
                <div className="text-sm text-zinc-500 dark:text-zinc-400">
                  {job.segment_count} segments · {job.created_at ? new Date(job.created_at).toLocaleString() : ""}
                </div>
              </div>
              <Play className="w-4 h-4 text-zinc-400" />
            </button>
          ))}
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
  const [burnTimestamp, setBurnTimestamp] = useState(Date.now());
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    setLoading(true);
    fetchJSON(`${API}/jobs/${jobId}`)
      .then((data: CaptionJob) => {
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
      prev.map((s, i) =>
        i === idx ? { ...s, lang: s.lang === "en" ? "vi" : "en", phonetic: s.lang === "en" ? s.phonetic : "" } : s
      )
    );
  }, []);

  const handleReburn = async () => {
    if (!style) return;
    setBurning(true);
    setBurnError(null);

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
      if (videoRef.current) videoRef.current.load();
    } catch (e: any) {
      setBurnError(String(e));
    } finally {
      setBurning(false);
    }
  };

  const handleDownload = async () => {
    const res = await fetch(`${API}/jobs/${jobId}/download`);
    if (!res.ok) return;
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `captioned_${jobId}.mp4`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (loading) return <div className="flex justify-center p-12"><Spinner /></div>;
  if (error) return <div className="p-6 text-red-500">Error: {error}</div>;
  if (!job || !style) return null;

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-3 px-6 py-3 border-b border-zinc-200 dark:border-zinc-700">
        <button onClick={onBack} className="flex items-center gap-1 text-sm text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100">
          <ChevronLeft className="w-4 h-4" />
          All jobs
        </button>
        <span className="text-zinc-400">/</span>
        <span className="text-sm font-medium truncate">{job.video_path.split("/").pop()}</span>
      </div>

      {/* Main two-column layout */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left: video player */}
        <div className="w-2/5 min-w-0 flex flex-col gap-3 p-4 border-r border-zinc-200 dark:border-zinc-700">
          <video
            ref={videoRef}
            key={burnTimestamp}
            controls
            className="w-full rounded-lg bg-black"
            src={`${API}/jobs/${jobId}/video?t=${burnTimestamp}`}
          />
          <div className="flex gap-2">
            <button
              onClick={handleReburn}
              disabled={burning}
              className="flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-md bg-zinc-900 text-white hover:bg-zinc-700 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-300 disabled:opacity-50 transition-colors text-sm font-medium"
            >
              {burning ? <Spinner className="w-4 h-4" /> : <RefreshCw className="w-4 h-4" />}
              {burning ? "Re-burning…" : "Re-burn"}
            </button>
            <button
              onClick={handleDownload}
              className="flex items-center gap-2 px-4 py-2 rounded-md border border-zinc-300 dark:border-zinc-600 hover:bg-zinc-50 dark:hover:bg-zinc-800 transition-colors text-sm font-medium"
            >
              <Download className="w-4 h-4" />
              Download
            </button>
          </div>
          {burnError && <p className="text-sm text-red-500">{burnError}</p>}

          {/* Style panel */}
          <details className="group">
            <summary className="cursor-pointer text-sm font-medium text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100 py-1">
              Style settings
            </summary>
            <div className="mt-2 space-y-2 text-sm">
              <StyleField label="Font" value={style.font} onChange={(v) => setStyle((s) => s && ({ ...s, font: v }))} />
              <StyleNumberField label="Font size" value={style.font_size} onChange={(v) => setStyle((s) => s && ({ ...s, font_size: v }))} />
              <StyleColorField label="Text color" value={style.primary_color} onChange={(v) => setStyle((s) => s && ({ ...s, primary_color: v }))} />
              <StyleColorField label="Outline color" value={style.outline_color} onChange={(v) => setStyle((s) => s && ({ ...s, outline_color: v }))} />
              <StyleNumberField label="Outline width" value={style.outline_width} onChange={(v) => setStyle((s) => s && ({ ...s, outline_width: v }))} />
              <StyleNumberField label="Margin bottom" value={style.margin_bottom} onChange={(v) => setStyle((s) => s && ({ ...s, margin_bottom: v }))} />
            </div>
          </details>
        </div>

        {/* Right: segment editor */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          <h2 className="text-lg font-medium mb-3">
            Segments ({segments.length})
          </h2>
          {segments.map((seg, idx) => (
            <div key={seg.id} className="rounded-lg border border-zinc-200 dark:border-zinc-700 p-3 space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <span className="text-xs text-zinc-500 w-16 shrink-0">
                  {formatTime(seg.start)}–{formatTime(seg.end)}
                </span>
                <button
                  onClick={() => toggleLang(idx)}
                  className={`px-2 py-0.5 rounded text-xs font-medium shrink-0 transition-colors ${
                    seg.lang === "vi"
                      ? "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300"
                      : "bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300"
                  }`}
                >
                  {seg.lang.toUpperCase()}
                </button>
                <input
                  className="flex-1 min-w-0 bg-transparent border-b border-transparent hover:border-zinc-300 focus:border-blue-500 dark:hover:border-zinc-600 dark:focus:border-blue-400 outline-none px-1"
                  value={seg.text}
                  onChange={(e) => updateSegment(idx, "text", e.target.value)}
                  placeholder="Caption text"
                />
              </div>
              {seg.lang === "vi" && (
                <div className="flex items-center gap-2 pl-[4.5rem]">
                  <span className="text-xs text-zinc-500 shrink-0">phonetic</span>
                  <input
                    className="flex-1 min-w-0 bg-transparent border-b border-transparent hover:border-zinc-300 focus:border-blue-500 dark:hover:border-zinc-600 dark:focus:border-blue-400 outline-none px-1 italic text-zinc-500"
                    value={seg.phonetic}
                    onChange={(e) => updateSegment(idx, "phonetic", e.target.value)}
                    placeholder="[pronunciation guide]"
                  />
                </div>
              )}
            </div>
          ))}
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
      <span className="w-28 shrink-0 text-zinc-500">{label}</span>
      <input
        className="flex-1 bg-transparent border border-zinc-300 dark:border-zinc-600 rounded px-2 py-0.5 text-sm outline-none focus:border-blue-500"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    </div>
  );
}

function StyleNumberField({ label, value, onChange }: { label: string; value: number; onChange: (v: number) => void }) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-28 shrink-0 text-zinc-500">{label}</span>
      <input
        type="number"
        className="w-24 bg-transparent border border-zinc-300 dark:border-zinc-600 rounded px-2 py-0.5 text-sm outline-none focus:border-blue-500"
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
      <span className="w-28 shrink-0 text-zinc-500">{label}</span>
      <input
        type="color"
        className="w-8 h-6 rounded cursor-pointer border border-zinc-300 dark:border-zinc-600"
        value={toHex(value)}
        onChange={(e) => onChange(toAss(e.target.value))}
      />
      <span className="text-xs text-zinc-400">{toHex(value)}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = (seconds % 60).toFixed(1).padStart(4, "0");
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
