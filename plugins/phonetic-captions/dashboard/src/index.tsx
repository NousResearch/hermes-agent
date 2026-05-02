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
  Upload,
  Wand2,
  ShieldCheck,
  X,
  Check,
  FileUp,
  Loader2,
  Sparkles,
  Trash2,
  Plus,
  ChevronDown,
  MessageSquare,
  Clapperboard,
  Pen,
  FlaskConical,
  CheckCircle2,
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
  margin_edge: number;
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
  status?: string;
  status_message?: string;
}

// Patch operations returned by the NL-edit endpoint
type NLPatch =
  | { op: "edit"; segment_id: number; field: string; old: any; new: any }
  | { op: "merge"; segment_ids: number[] }
  | { op: "split"; segment_id: number; at_word_index: number };

interface QAFlag {
  segment_id: number;
  type?: "issue" | "praise";
  issue: string;
  suggestion: string;
}

interface StyleSuggestion {
  available: boolean;
  style?: CaptionStyle;
  explanation?: string;
}

interface CaptionPreset {
  name: string;
  style: CaptionStyle;
}

// ---------------------------------------------------------------------------
// API base
// ---------------------------------------------------------------------------

const API = "/api/plugins/phonetic-captions";

// ---------------------------------------------------------------------------
// Style defaults + merge helper
// Ensures every preset/AI style applied always contains every field,
// preventing uncontrolled→controlled jumps in the Caption Style inputs.
// ---------------------------------------------------------------------------

const STYLE_DEFAULTS: CaptionStyle = {
  font: "Arial",
  font_size: 48,
  primary_color: "&H00FFFFFF",
  outline_color: "&H00000000",
  outline_width: 3,
  alignment: 2,
  margin_edge: 80,
  max_line_length: 42,
};

function mergeStyle(incoming: Partial<CaptionStyle>): CaptionStyle {
  return { ...STYLE_DEFAULTS, ...incoming };
}

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

// ---------------------------------------------------------------------------
// Info Sidebar — shown on the job list page
// ---------------------------------------------------------------------------

function InfoSidebar() {
  const steps = [
    {
      icon: <MessageSquare className="w-4 h-4" />,
      title: "Send a video",
      body: "Send any video file via Telegram (or upload directly here). Hermes transcribes it with Whisper and generates phonetic captions automatically.",
    },
    {
      icon: <Clapperboard className="w-4 h-4" />,
      title: "Review & edit",
      body: "Open the job to see the segment list. Correct text, phonetics, or timing — one segment at a time, or describe changes in plain English and let AI apply them.",
    },
    {
      icon: <FlaskConical className="w-4 h-4" />,
      title: "Style it",
      body: "Adjust font, size, colour, and outline in the Caption Style section. Save named presets to reuse across videos, or describe a look and generate one with AI.",
    },
    {
      icon: <Play className="w-4 h-4" />,
      title: "Re-burn & download",
      body: "Hit Re-burn to bake the final captions into the video, then download the finished file. Hermes also replies with the download link in Telegram.",
    },
  ];

  const features = [
    { icon: <ShieldCheck className="w-3.5 h-3.5" />, label: "AI QA review — flags awkward phrasing or timing issues" },
    { icon: <Pen className="w-3.5 h-3.5" />, label: 'Natural-language edits — "merge segments 3 and 4"' },
    { icon: <Sparkles className="w-3.5 h-3.5" />, label: "Style presets — save and reuse your favourite looks" },
    { icon: <MessageSquare className="w-3.5 h-3.5" />, label: "Telegram-native — full pipeline from a single message" },
  ];

  return (
    <div className="space-y-6">
      {/* How it works */}
      <div>
        <h2 className="text-sm font-semibold text-foreground mb-4">How it works</h2>
        <ol className="space-y-4">
          {steps.map((s, i) => (
            <li key={i} className="flex gap-3">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary">
                {s.icon}
              </div>
              <div>
                <p className="text-sm font-semibold text-foreground mb-0.5">{s.title}</p>
                <p className="text-xs text-muted-foreground leading-relaxed">{s.body}</p>
              </div>
            </li>
          ))}
        </ol>
      </div>

      {/* Divider */}
      <div className="border-t border-border" />

      {/* Key features */}
      <div>
        <h2 className="text-sm font-semibold text-foreground mb-3">Key features</h2>
        <ul className="space-y-2">
          {features.map((f, i) => (
            <li key={i} className="flex items-start gap-2 text-xs text-foreground">
              <span className="text-primary mt-0.5 shrink-0">{f.icon}</span>
              {f.label}
            </li>
          ))}
        </ul>
      </div>

      {/* Divider */}
      <div className="border-t border-border" />

      {/* Tip */}
      <div className="rounded-lg bg-muted/50 border border-border p-3 text-xs text-muted-foreground leading-relaxed">
        <span className="font-semibold text-foreground">Tip:</span> You can skip the auto-pipeline and paste your own segments JSON directly in the upload modal — useful when you already have a transcript.
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Prerequisites Banner
// ---------------------------------------------------------------------------

interface HealthStatus {
  ffmpeg: boolean;
  faster_whisper: boolean;
  phonetics_source: "nvidia" | "hermes" | "unavailable";
  hermes_model: string;
}

function PrerequisitesBanner() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    fetchJSON(`${API}/health`)
      .then((data: HealthStatus) => setHealth(data))
      .catch(() => { /* non-fatal — banner simply doesn't render */ });
  }, []);

  if (!health || dismissed) return null;

  const errors: string[] = [];
  if (!health.ffmpeg) errors.push("ffmpeg not found (brew install ffmpeg / apt install ffmpeg)");
  if (!health.faster_whisper) errors.push("faster-whisper not installed (pip install faster-whisper)");

  const phoneticsLabel =
    health.phonetics_source === "nvidia"
      ? "NVIDIA Kimi K2.6"
      : health.phonetics_source === "hermes"
      ? `${health.hermes_model || "configured Hermes model"} (no NVIDIA key — fallback)`
      : "unavailable (no faster-whisper)";

  return (
    <div className="mb-5 space-y-2">
      {errors.length > 0 && (
        <div className="relative flex items-start gap-3 rounded-lg border border-destructive/40 bg-destructive/5 px-4 py-3 text-xs text-destructive">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
          <div className="flex-1 space-y-0.5">
            <p className="font-semibold text-sm">Missing prerequisites</p>
            {errors.map((e, i) => <p key={i} className="font-mono">{e}</p>)}
          </div>
          <button onClick={() => setDismissed(true)} className="p-1 rounded hover:bg-destructive/10">
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}
      <div className="relative flex items-center gap-2 rounded-lg border border-border bg-muted/40 px-4 py-2.5 text-xs text-muted-foreground">
        <Sparkles className="w-3.5 h-3.5 shrink-0 text-primary" />
        <span>
          <span className="font-medium text-foreground">Phonetics engine: </span>
          {phoneticsLabel}
        </span>
        <button onClick={() => setDismissed(true)} className="ml-auto p-1 rounded hover:bg-muted">
          <X className="w-3 h-3" />
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Job List View
// ---------------------------------------------------------------------------

function JobListView({ onSelect }: { onSelect: (id: string) => void }) {
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showUpload, setShowUpload] = useState(false);

  const loadJobs = useCallback(() => {
    setLoading(true);
    fetchJSON(`${API}/jobs`)
      .then((data: JobSummary[]) => setJobs(data))
      .catch((e: any) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { loadJobs(); }, [loadJobs]);

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
    <div className="p-6 grid grid-cols-1 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)] gap-8 max-w-6xl">
      {/* ── Left: job list ── */}
      <div>
        <PrerequisitesBanner />
        <div className="flex items-center justify-between mb-1">
          <h1 className="text-xl font-semibold text-foreground">Caption Jobs</h1>
          <button
            onClick={() => setShowUpload(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-primary text-primary-foreground text-xs font-semibold hover:opacity-90 transition-opacity"
          >
            <Upload className="w-3.5 h-3.5" />
            New Job
          </button>
        </div>
        <p className="text-sm text-muted-foreground mb-5">
          Click a job to edit segments and re-burn, or upload a new video.
        </p>

        {showUpload && (
          <UploadModal
            onDone={(id) => { setShowUpload(false); onSelect(id); }}
            onClose={() => { setShowUpload(false); loadJobs(); }}
          />
        )}

        {jobs.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-16 text-muted-foreground border border-dashed border-border rounded-xl">
            <VideoOff className="w-10 h-10 mb-3 opacity-40" />
            <p className="text-sm font-medium">No caption jobs yet</p>
            <p className="text-xs mt-1 text-muted-foreground/70">Upload a video above or send one via Telegram.</p>
          </div>
        ) : (
          <div className="space-y-2">
            {jobs.map((job) => {
              const isPending = job.status && job.status !== "ready";
              const isError = job.status === "error";
              return (
                <button
                  key={job.id}
                  onClick={() => !isPending && onSelect(job.id)}
                  disabled={!!isPending}
                  className={`w-full flex items-center justify-between rounded-lg border border-border p-4 bg-card text-left group transition-colors ${
                    isPending ? "opacity-60 cursor-not-allowed" : "hover:bg-accent"
                  }`}
                >
                  <div className="min-w-0">
                    <div className="font-medium text-sm truncate text-foreground">{job.video_filename || job.id}</div>
                    <div className="text-xs text-muted-foreground mt-0.5 flex items-center gap-2">
                      <span>
                        {job.segment_count} segment{job.segment_count !== 1 ? "s" : ""}
                        {job.created_at ? ` · ${new Date(job.created_at).toLocaleString()}` : ""}
                      </span>
                      {isPending && !isError && (
                        <span className="flex items-center gap-1 text-amber-500">
                          <Loader2 className="w-3 h-3 animate-spin" />
                          {job.status_message || job.status}
                        </span>
                      )}
                      {isError && (
                        <span className="text-destructive">
                          {job.status_message || "Pipeline error"}
                        </span>
                      )}
                    </div>
                  </div>
                  {!isPending && <Play className="w-4 h-4 text-muted-foreground group-hover:text-foreground shrink-0 ml-4 transition-colors" />}
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* ── Right: info sidebar ── */}
      <div className="hidden xl:block border-l border-border pl-8 pt-1">
        <InfoSidebar />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Upload Modal
// ---------------------------------------------------------------------------

function UploadModal({
  onDone,
  onClose,
}: {
  onDone: (jobId: string) => void;
  onClose: () => void;
}) {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [segmentsFile, setSegmentsFile] = useState<File | null>(null);
  const [runPipeline, setRunPipeline] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [statusMsg, setStatusMsg] = useState("");
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPoll = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  // Cleanup on unmount
  useEffect(() => () => stopPoll(), []);

  const handleSubmit = async () => {
    if (!videoFile) return;
    setUploading(true);
    setError(null);
    setStatusMsg("Uploading…");

    try {
      const fd = new FormData();
      fd.append("video", videoFile);
      if (segmentsFile) fd.append("segments", segmentsFile);
      fd.append("run_pipeline", String(runPipeline && !segmentsFile));

      const res = await fetch(`${API}/upload`, { method: "POST", body: fd });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `${res.status} ${res.statusText}`);
      }
      const { job_id, status } = await res.json();

      if (status === "ready") {
        onDone(job_id);
        return;
      }

      // Pipeline running — poll for completion
      pollRef.current = setInterval(async () => {
        try {
          const s = await fetchJSON(`${API}/jobs/${job_id}/status`);
          setStatusMsg(s.status_message || s.status);
          if (s.status === "ready") {
            stopPoll();
            onDone(job_id);
          } else if (s.status === "error") {
            stopPoll();
            setError(s.status_message || "Pipeline failed");
            setUploading(false);
          }
        } catch {
          // transient network error — keep polling
        }
      }, 2000);
    } catch (e: any) {
      setError(String(e));
      setUploading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-card border border-border rounded-xl shadow-xl w-full max-w-md mx-4 p-6">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-base font-semibold text-foreground">New Caption Job</h2>
          <button onClick={onClose} disabled={uploading} className="text-muted-foreground hover:text-foreground transition-colors">
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="space-y-4">
          {/* Video picker */}
          <div>
            <label className="block text-xs font-medium text-muted-foreground mb-1.5">Video file *</label>
            <label className={`flex items-center gap-2 px-3 py-2.5 rounded-lg border-2 border-dashed cursor-pointer transition-colors ${
              videoFile ? "border-primary/50 bg-primary/5" : "border-border hover:border-primary/30"
            }`}>
              <FileUp className="w-4 h-4 text-muted-foreground shrink-0" />
              <span className="text-sm truncate text-foreground">
                {videoFile ? videoFile.name : "Choose .mp4, .mov, .mkv…"}
              </span>
              <input
                type="file"
                accept=".mp4,.mov,.avi,.mkv,.webm,.m4v,.ts,.mts"
                className="hidden"
                onChange={(e) => setVideoFile(e.target.files?.[0] ?? null)}
              />
            </label>
          </div>

          {/* Auto-pipeline toggle */}
          <label className="flex items-center gap-3 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={runPipeline}
              onChange={(e) => {
                setRunPipeline(e.target.checked);
                if (e.target.checked) setSegmentsFile(null);
              }}
              className="w-4 h-4 accent-primary"
            />
            <span className="text-sm text-foreground">Auto-transcribe &amp; generate phonetics</span>
          </label>

          {/* Manual segments upload (when pipeline is off) */}
          {!runPipeline && (
            <div>
              <label className="block text-xs font-medium text-muted-foreground mb-1.5">Segments JSON (optional)</label>
              <label className={`flex items-center gap-2 px-3 py-2.5 rounded-lg border-2 border-dashed cursor-pointer transition-colors ${
                segmentsFile ? "border-primary/50 bg-primary/5" : "border-border hover:border-primary/30"
              }`}>
                <FileUp className="w-4 h-4 text-muted-foreground shrink-0" />
                <span className="text-sm truncate text-foreground">
                  {segmentsFile ? segmentsFile.name : "Choose segments .json…"}
                </span>
                <input
                  type="file"
                  accept=".json"
                  className="hidden"
                  onChange={(e) => setSegmentsFile(e.target.files?.[0] ?? null)}
                />
              </label>
            </div>
          )}

          {/* Status / error */}
          {uploading && !error && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="w-4 h-4 animate-spin shrink-0" />
              <span>{statusMsg || "Working…"}</span>
            </div>
          )}
          {error && (
            <div className="flex items-start gap-2 text-sm text-destructive">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}
        </div>

        <div className="flex justify-end gap-2 mt-6">
          <button
            onClick={onClose}
            disabled={uploading}
            className="px-4 py-2 rounded-lg border border-border text-sm font-medium text-foreground hover:bg-accent transition-colors disabled:opacity-40"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={!videoFile || uploading}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-semibold hover:opacity-90 transition-opacity disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {uploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
            {uploading ? "Processing…" : "Create Job"}
          </button>
        </div>
      </div>
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
// NL Edit Panel
// ---------------------------------------------------------------------------

function NLEditPanel({
  jobId,
  segments,
  onApplyPatches,
  prefillInstruction,
}: {
  jobId: string;
  segments: CaptionSegment[];
  onApplyPatches: (patches: NLPatch[]) => void;
  prefillInstruction?: string;
}) {
  const [instruction, setInstruction] = useState(prefillInstruction ?? "");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [patches, setPatches] = useState<NLPatch[] | null>(null);
  const [checked, setChecked] = useState<Set<number>>(new Set());

  // Sync prefill from QA "Fix with AI" — also auto-submit so the fix is triggered immediately
  useEffect(() => {
    if (prefillInstruction !== undefined) {
      setInstruction(prefillInstruction);
      if (prefillInstruction.trim()) handleSubmit(prefillInstruction);
    }
  }, [prefillInstruction]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleSubmit = async (instructionOverride?: string) => {
    const inst = instructionOverride ?? instruction;
    if (!inst.trim()) return;
    setLoading(true);
    setError(null);
    setPatches(null);
    try {
      const res = await fetchJSON(`${API}/jobs/${jobId}/nl-edit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ instruction: inst }),
      });
      const ps: NLPatch[] = res.patches ?? [];
      setPatches(ps);
      setChecked(new Set(ps.map((_, i) => i)));
    } catch (e: any) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const applySelected = () => {
    if (!patches) return;
    onApplyPatches(patches.filter((_, i) => checked.has(i)));
    setPatches(null);
    setInstruction("");
    setChecked(new Set());
  };

  const patchDescription = (p: NLPatch): string => {
    if (p.op === "edit") return `#${p.segment_id + 1} ${p.field}: "${p.old}" → "${p.new}"`;
    if (p.op === "merge") return `Merge segments ${p.segment_ids.map((id) => `#${id + 1}`).join(" + ")}`;
    if (p.op === "split") return `Split segment #${p.segment_id + 1} at word ${p.at_word_index}`;
    return "Unknown operation";
  };

  return (
    <div className="space-y-2">
      <div className="flex gap-2">
        <input
          className="flex-1 min-w-0 bg-card border border-border rounded-lg px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground outline-none focus:border-ring focus:ring-1 focus:ring-ring/30 transition-colors"
          placeholder="e.g. fix diacritics in segment 3, merge segments 4 and 5…"
          value={instruction}
          onChange={(e) => setInstruction(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !loading && handleSubmit()}
          disabled={loading}
        />
        <button
          onClick={() => handleSubmit()}
          disabled={!instruction.trim() || loading}
          className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-primary text-primary-foreground text-xs font-semibold hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed transition-opacity"
        >
          {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Wand2 className="w-3.5 h-3.5" />}
          {loading ? "Thinking…" : "Apply"}
        </button>
      </div>

      {error && (
        <div className="mt-2 flex items-start gap-2 text-xs text-destructive">
          <AlertCircle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
          {error}
        </div>
      )}

      {patches !== null && (
        <div className="mt-3 rounded-lg border border-border bg-muted/40 p-3">
          <p className="text-xs font-semibold text-foreground mb-2">
            {patches.length === 0 ? "No changes suggested." : `${patches.length} proposed change${patches.length !== 1 ? "s" : ""}:`}
          </p>
          {patches.length > 0 && (
            <>
              <div className="space-y-1 max-h-40 overflow-y-auto">
                {patches.map((p, i) => (
                  <label key={i} className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={checked.has(i)}
                      onChange={() =>
                        setChecked((prev) => {
                          const next = new Set(prev);
                          next.has(i) ? next.delete(i) : next.add(i);
                          return next;
                        })
                      }
                      className="accent-primary"
                    />
                    <span className="text-xs text-foreground">{patchDescription(p)}</span>
                  </label>
                ))}
              </div>
              <div className="flex gap-2 mt-3">
                <button
                  onClick={applySelected}
                  disabled={checked.size === 0}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-primary text-primary-foreground text-xs font-semibold hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed transition-opacity"
                >
                  <Check className="w-3 h-3" />
                  Apply {checked.size} change{checked.size !== 1 ? "s" : ""}
                </button>
                <button
                  onClick={() => { setPatches(null); setChecked(new Set()); }}
                  className="px-3 py-1.5 rounded border border-border text-xs font-medium text-foreground hover:bg-accent transition-colors"
                >
                  Dismiss
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Preset Gallery (inside Hermes Panel)
// ---------------------------------------------------------------------------

function PresetGallery({
  style,
  onApply,
}: {
  style: CaptionStyle;
  onApply: (s: CaptionStyle) => void;
}) {
  const [presets, setPresets] = useState<CaptionPreset[]>([]);
  const [suggestion, setSuggestion] = useState<StyleSuggestion | null>(null);
  const [suggestionDismissed, setSuggestionDismissed] = useState(false);

  // Save current
  const [savingName, setSavingName] = useState("");
  const [showSaveInput, setShowSaveInput] = useState(false);
  const [saving, setSaving] = useState(false);

  // AI generation
  const [showGenerate, setShowGenerate] = useState(false);
  const [genDesc, setGenDesc] = useState("");
  const [genLoading, setGenLoading] = useState(false);
  const [genPreview, setGenPreview] = useState<CaptionStyle | null>(null);
  const [genName, setGenName] = useState("");
  const [genError, setGenError] = useState<string | null>(null);

  const loadPresets = useCallback(() => {
    fetchJSON(`${API}/presets`)
      .then((data: CaptionPreset[]) => setPresets(data))
      .catch(() => {});
  }, []);

  useEffect(() => {
    loadPresets();
    fetchJSON(`${API}/style/suggestion`)
      .then((s: StyleSuggestion) => setSuggestion(s))
      .catch(() => {});
  }, [loadPresets]);

  const handleDelete = async (name: string) => {
    await fetchJSON(`${API}/presets/${encodeURIComponent(name)}`, { method: "DELETE" });
    loadPresets();
  };

  const handleSaveCurrent = async () => {
    if (!savingName.trim()) return;
    setSaving(true);
    try {
      await fetchJSON(`${API}/presets/${encodeURIComponent(savingName.trim())}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ style }),
      });
      setSavingName("");
      setShowSaveInput(false);
      loadPresets();
    } catch { /* ignore */ }
    setSaving(false);
  };

  const handleGenerate = async () => {
    if (!genDesc.trim()) return;
    setGenLoading(true);
    setGenError(null);
    setGenPreview(null);
    try {
      const res = await fetchJSON(`${API}/presets/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ description: genDesc }),
      });
      setGenPreview(res.style);
    } catch (e: any) {
      setGenError(String(e));
    }
    setGenLoading(false);
  };

  const handleSaveGenerated = async () => {
    if (!genPreview || !genName.trim()) return;
    setSaving(true);
    try {
      await fetchJSON(`${API}/presets/${encodeURIComponent(genName.trim())}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ style: genPreview }),
      });
      setGenPreview(null);
      setGenDesc("");
      setGenName("");
      setShowGenerate(false);
      loadPresets();
    } catch { /* ignore */ }
    setSaving(false);
  };

  return (
    <div className="space-y-3">
      {/* Learned preset card */}
      {suggestion?.available && !suggestionDismissed && suggestion.style && (
        <div className="rounded-lg border border-amber-400/50 bg-amber-400/5 p-3">
          <div className="flex items-center gap-1.5 mb-1">
            <Sparkles className="w-3.5 h-3.5 text-amber-400 shrink-0" />
            <span className="text-xs font-semibold text-amber-400">Learned</span>
            <button
              onClick={() => setSuggestionDismissed(true)}
              className="ml-auto text-muted-foreground hover:text-foreground"
            >
              <X className="w-3 h-3" />
            </button>
          </div>
          {suggestion.explanation && (
            <p className="text-xs text-amber-400/80 mb-2">{suggestion.explanation}</p>
          )}
          <div className="flex gap-2">
            <button
              onClick={() => onApply(mergeStyle(suggestion.style!))}
              className="flex-1 px-2 py-1 rounded bg-amber-400/20 text-amber-400 text-xs font-medium hover:bg-amber-400/30 transition-colors"
            >
              Apply
            </button>
            <button
              onClick={async () => {
                const name = prompt("Preset name:");
                if (!name?.trim()) return;
                await fetchJSON(`${API}/presets/${encodeURIComponent(name.trim())}`, {
                  method: "PUT",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ style: suggestion.style }),
                });
                loadPresets();
              }}
              className="px-2 py-1 rounded border border-amber-400/30 text-amber-400 text-xs font-medium hover:bg-amber-400/10 transition-colors"
            >
              Save as…
            </button>
          </div>
        </div>
      )}

      {/* Named preset cards */}
      {presets.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {presets.map((p) => (
            <div
              key={p.name}
              className="flex items-center gap-1 pl-2.5 pr-1 py-1 rounded-lg border border-border bg-card text-xs text-foreground group hover:border-ring/50 transition-colors"
            >
              <button
                onClick={() => onApply(mergeStyle(p.style))}
                className="font-medium hover:text-primary transition-colors"
                title={`Apply preset: ${p.name}`}
              >
                {p.name}
              </button>
              <button
                onClick={() => handleDelete(p.name)}
                className="text-muted-foreground/40 hover:text-destructive transition-colors p-0.5 rounded"
                title={`Delete preset: ${p.name}`}
              >
                <X className="w-3 h-3" />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Save current style */}
      {!showSaveInput ? (
        <button
          onClick={() => setShowSaveInput(true)}
          className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          <Plus className="w-3.5 h-3.5" />
          Save current style
        </button>
      ) : (
        <div className="flex gap-2">
          <input
            className="flex-1 min-w-0 bg-card border border-border rounded px-2 py-1 text-xs text-foreground outline-none focus:border-ring focus:ring-1 focus:ring-ring/30 placeholder:text-muted-foreground"
            placeholder="Preset name…"
            value={savingName}
            onChange={(e) => setSavingName(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !saving && handleSaveCurrent()}
            autoFocus
          />
          <button
            onClick={handleSaveCurrent}
            disabled={!savingName.trim() || saving}
            className="px-2 py-1 rounded bg-primary text-primary-foreground text-xs font-semibold hover:opacity-90 disabled:opacity-40 transition-opacity"
          >
            {saving ? <Loader2 className="w-3 h-3 animate-spin" /> : <Check className="w-3 h-3" />}
          </button>
          <button
            onClick={() => { setShowSaveInput(false); setSavingName(""); }}
            className="px-2 py-1 rounded border border-border text-xs text-foreground hover:bg-accent transition-colors"
          >
            <X className="w-3 h-3" />
          </button>
        </div>
      )}

      {/* AI style generation */}
      <button
        onClick={() => setShowGenerate(!showGenerate)}
        className="flex items-center gap-1.5 w-full px-3 py-2 rounded-lg border border-dashed border-border text-xs font-medium text-foreground hover:bg-accent hover:border-primary transition-colors"
      >
        <Sparkles className="w-3.5 h-3.5 text-primary" />
        Style with Hermes
        <ChevronDown className={`w-3 h-3 ml-auto transition-transform ${showGenerate ? "rotate-180" : ""}`} />
      </button>

      {showGenerate && (
        <div className="rounded-lg border border-border bg-muted/30 p-3 space-y-2">
          <div className="flex gap-2">
            <input
              className="flex-1 min-w-0 bg-card border border-border rounded px-2 py-1.5 text-xs text-foreground outline-none focus:border-ring focus:ring-1 focus:ring-ring/30 placeholder:text-muted-foreground"
              placeholder="e.g. bold yellow Impact, thick outline, TikTok style…"
              value={genDesc}
              onChange={(e) => setGenDesc(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !genLoading && handleGenerate()}
              disabled={genLoading}
            />
            <button
              onClick={handleGenerate}
              disabled={!genDesc.trim() || genLoading}
              className="flex items-center gap-1 px-2.5 py-1.5 rounded bg-primary text-primary-foreground text-xs font-semibold hover:opacity-90 disabled:opacity-40 transition-opacity"
            >
              {genLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : <Sparkles className="w-3 h-3" />}
              {genLoading ? "…" : "Go"}
            </button>
          </div>
          {genError && (
            <p className="text-xs text-destructive">{genError}</p>
          )}
          {genPreview && (
            <div className="rounded border border-border bg-card p-2 space-y-1">
              <p className="text-xs font-semibold text-foreground mb-1.5">Preview:</p>
              {(Object.entries(genPreview) as [string, string | number | null][]).map(([k, v]) => {
                const LABELS: Record<string, string> = { font: "Font", font_size: "Font size", primary_color: "Text color", outline_color: "Outline color", outline_width: "Outline width", alignment: "Alignment", margin_edge: "Margin from edge", max_line_length: "Line length" };
                const displayValue = k === "alignment" && typeof v === "number"
                  ? `${ALIGNMENT_ICONS[v] ?? ""} ${ALIGNMENT_NAMES[v] ?? v}`
                  : String(v);
                return (
                  <div key={k} className="flex gap-2 text-xs">
                    <span className="text-muted-foreground w-28 shrink-0">{LABELS[k] ?? k}</span>
                    <span className="text-foreground font-mono">{displayValue}</span>
                  </div>
                );
              })}
              <div className="flex gap-2 mt-2 pt-2 border-t border-border">
                <input
                  className="flex-1 min-w-0 bg-card border border-border rounded px-2 py-1 text-xs text-foreground outline-none focus:border-ring placeholder:text-muted-foreground"
                  placeholder="Name this preset…"
                  value={genName}
                  onChange={(e) => setGenName(e.target.value)}
                />
                <button
                  onClick={handleSaveGenerated}
                  disabled={!genName.trim() || saving}
                  className="px-2 py-1 rounded bg-primary text-primary-foreground text-xs font-semibold hover:opacity-90 disabled:opacity-40 transition-opacity"
                >
                  Save
                </button>
                <button
                  onClick={() => { onApply(mergeStyle(genPreview)); }}
                  className="px-2 py-1 rounded border border-border text-xs text-foreground hover:bg-accent transition-colors"
                >
                  Apply
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Hermes Panel (Col 3)
// ---------------------------------------------------------------------------

function HermesPanel({
  jobId,
  segments,
  qaFlags,
  qaLoading,
  qaError,
  nlPrefill,
  style,
  onApplyPatches,
  onQAReview,
  onClearQA,
  onFlagClick,
  onFixWithAI,
  onApplyStyle,
}: {
  jobId: string;
  segments: CaptionSegment[];
  qaFlags: QAFlag[];
  qaLoading: boolean;
  qaError: string | null;
  nlPrefill?: string;
  style: CaptionStyle;
  onApplyPatches: (patches: NLPatch[]) => void;
  onQAReview: () => void;
  onClearQA: () => void;
  onFlagClick: (segmentId: number) => void;
  onFixWithAI: (suggestion: string) => void;
  onApplyStyle: (s: CaptionStyle) => void;
}) {
  return (
    <div className="flex flex-col gap-0 h-full">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-border shrink-0">
        <Sparkles className="w-4 h-4 text-primary" />
        <span className="text-sm font-semibold text-foreground">Hermes</span>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-6">

        {/* ── Edit segments ── */}
        <section>
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">Edit segments</p>
          <NLEditPanel
            jobId={jobId}
            segments={segments}
            onApplyPatches={onApplyPatches}
            prefillInstruction={nlPrefill}
          />
        </section>

        {/* ── QA ── */}
        <section>
          <div className="flex items-center gap-2 mb-3">
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide flex-1">QA</p>
            <button
              onClick={onQAReview}
              disabled={qaLoading || segments.length === 0}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded border border-border text-xs font-medium text-foreground hover:bg-accent disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {qaLoading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <ShieldCheck className="w-3.5 h-3.5" />}
              {qaLoading ? "Reviewing…" : "Review"}
            </button>
          </div>
          {qaError && (
            <p className="text-xs text-destructive mb-2">{qaError}</p>
          )}
          {qaFlags.length === 0 && !qaLoading && (
            <p className="text-xs text-muted-foreground">No issues found. Run Review to check.</p>
          )}
          {qaFlags.length > 0 && (() => {
            const issues = qaFlags.filter(f => (f.type ?? "issue") === "issue");
            const praises = qaFlags.filter(f => f.type === "praise");
            return (
              <div className="space-y-2">
                <div className="flex items-center gap-1 text-xs mb-1">
                  {issues.length > 0 && (
                    <span className="flex items-center gap-1 text-amber-500">
                      <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                      {issues.length} issue{issues.length !== 1 ? "s" : ""}
                    </span>
                  )}
                  {issues.length > 0 && praises.length > 0 && <span className="text-muted-foreground mx-1">·</span>}
                  {praises.length > 0 && (
                    <span className="flex items-center gap-1 text-emerald-500">
                      <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
                      {praises.length} good
                    </span>
                  )}
                  {issues.length > 0 && <span className="text-muted-foreground text-xs ml-1">— amber in list</span>}
                  <button onClick={onClearQA} className="ml-auto text-muted-foreground hover:text-foreground">
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
                {qaFlags.map((f, i) => {
                  const isIssue = (f.type ?? "issue") === "issue";
                  return (
                    <div key={i} className={`rounded-lg border p-2.5 space-y-1 ${
                      isIssue
                        ? "border-amber-400/30 bg-amber-400/5"
                        : "border-emerald-400/30 bg-emerald-400/5"
                    }`}>
                      <button
                        onClick={() => onFlagClick(f.segment_id)}
                        className={`text-xs font-semibold hover:underline text-left w-full ${
                          isIssue ? "text-amber-400" : "text-emerald-400"
                        }`}
                      >
                        Segment #{f.segment_id + 1}
                      </button>
                      <p className="text-xs text-foreground">{f.issue}</p>
                      {f.suggestion && <p className="text-xs text-muted-foreground">{f.suggestion}</p>}
                      {isIssue && f.suggestion && (
                        <button
                          onClick={() => onFixWithAI(f.suggestion)}
                          className="text-xs text-primary hover:underline"
                        >
                          Fix →
                        </button>
                      )}
                    </div>
                  );
                })}
              </div>
            );
          })()}
        </section>

        {/* ── Style presets ── */}
        <section>
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">Style presets</p>
          <PresetGallery style={style} onApply={onApplyStyle} />
        </section>

      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Editor View (3-column layout)
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
  const [burnStyleUsed, setBurnStyleUsed] = useState<{ alignment?: number; margin_edge?: number } | null>(null);
  const [burnTimestamp, setBurnTimestamp] = useState(Date.now());
  const [splitState, setSplitState] = useState<{ segIdx: number; splitBefore: Set<number> } | null>(null);

  // QA state
  const [qaFlags, setQaFlags] = useState<QAFlag[]>([]);
  const [qaLoading, setQaLoading] = useState(false);
  const [qaError, setQaError] = useState<string | null>(null);

  // NL panel pre-fill (from QA "Fix with AI")
  const [nlPrefill, setNlPrefill] = useState<string | undefined>(undefined);

  // Ref for scrolling to a flagged segment
  const segmentRefs = useRef<Map<number, HTMLDivElement>>(new Map());

  useEffect(() => {
    setLoading(true);
    fetchJSON(`${API}/jobs/${jobId}`)
      .then((data: CaptionJob) => {
        console.log("[captions] job loaded", data.id, "segments:", data.segments?.length, data.segments?.[0]);
        setJob(data);
        setSegments(data.segments);
        setStyle(mergeStyle(data.style));
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

  const handleQAReview = async () => {
    setQaLoading(true);
    setQaError(null);
    setQaFlags([]);
    try {
      const res = await fetchJSON(`${API}/jobs/${jobId}/qa`, { method: "POST" });
      setQaFlags(res.flags ?? []);
    } catch (e: any) {
      setQaError(String(e));
    } finally {
      setQaLoading(false);
    }
  };

  const handleApplyNLPatches = useCallback((patches: NLPatch[]) => {
    setSegments((prev) => {
      let segs = [...prev];
      // Process in reverse order to keep indices stable for splits/merges
      const sorted = [...patches].reverse();
      for (const p of sorted) {
        if (p.op === "edit") {
          const idx = segs.findIndex((s) => s.id === p.segment_id);
          if (idx !== -1) segs[idx] = { ...segs[idx], [p.field]: p.new };
        } else if (p.op === "merge") {
          const indices = p.segment_ids
            .map((id) => segs.findIndex((s) => s.id === id))
            .filter((i) => i !== -1)
            .sort((a, b) => a - b);
          if (indices.length >= 2) {
            const merged: CaptionSegment = {
              ...segs[indices[0]],
              end: segs[indices[indices.length - 1]].end,
              text: indices.map((i) => segs[i].text).join(" ").trim(),
              phonetic: "",
            };
            segs = [
              ...segs.slice(0, indices[0]),
              merged,
              ...segs.filter((_, i) => !indices.includes(i) || i === indices[0]).slice(indices[0] + 1),
            ].filter((_, i) => !indices.slice(1).map((x) => x).includes(i));
            // Rebuild properly using filter
            const keep = new Set(indices.slice(1));
            segs = segs.filter((s, i) => !keep.has(i));
          }
        }
        // split op is handled via existing applyWordSplit UI; skip here
      }
      return segs.map((s, i) => ({ ...s, id: i }));
    });
    // Auto-save after applying patches
    setTimeout(async () => {
      try {
        await fetchJSON(`${API}/jobs/${jobId}/segments`, {
          method: "PUT",
          body: JSON.stringify({ segments }),
          headers: { "Content-Type": "application/json" },
        });
      } catch { /* non-fatal */ }
    }, 0);
  }, [jobId, segments]);

  const handleReburn = async () => {
    if (!style) return;
    setBurning(true);
    setBurnError(null);
    setBurnSuccess(false);
    setBurnStyleUsed(null);

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
      const burnResult = await fetchJSON(`${API}/jobs/${jobId}/burn`, { method: "POST" });
      setBurnTimestamp(Date.now());
      setBurnSuccess(true);
      setBurnStyleUsed(burnResult?.style_used ?? null);
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

      {/* ── 3-column body ── */}
      {/* On xl+ screens: 3 side-by-side columns. Below xl: col 3 wraps below col 2. */}
      <div className="flex-1 overflow-hidden flex flex-wrap xl:flex-nowrap">

        {/* ── Col 1: Video + actions (360px) ── */}
        <div className="w-full xl:w-[360px] xl:shrink-0 flex flex-col border-r border-border overflow-y-auto">
          <div className="p-4">
            <VideoPlayer src={`${API}/jobs/${jobId}/video?t=${burnTimestamp}`} />
          </div>
          <div className="px-4 pb-3 flex gap-2">
            <button
              onClick={handleReburn}
              disabled={burning}
              className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-primary text-primary-foreground font-semibold text-sm hover:opacity-90 active:opacity-80 disabled:opacity-40 disabled:cursor-not-allowed transition-opacity duration-150"
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
          {burnError && (
            <div className="mx-4 mb-3 flex items-start gap-2 rounded-lg bg-destructive/10 border border-destructive/30 px-3 py-2">
              <AlertCircle className="w-4 h-4 text-destructive shrink-0 mt-0.5" />
              <p className="text-xs text-destructive break-words">{burnError}</p>
            </div>
          )}
          {burnSuccess && !burnError && (
            <div className="mx-4 mb-3 rounded-lg bg-success/10 border border-success/30 px-3 py-2">
              <p className="text-xs text-success">✓ Video re-burned successfully.</p>
              {burnStyleUsed && (
                <p className="text-xs text-muted-foreground mt-0.5">
                  {ALIGNMENT_NAMES[burnStyleUsed.alignment!] ?? `align ${burnStyleUsed.alignment}`} · margin {burnStyleUsed.margin_edge}px
                </p>
              )}
            </div>
          )}
        </div>

        {/* ── Col 2: Segments + style fields (flex) ── */}
        <div className="flex-1 min-w-0 overflow-y-auto p-5 border-r border-border">
          <div className="max-w-2xl mx-auto">
            <div className="flex items-center gap-2 mb-4">
              <h2 className="text-base font-semibold text-foreground">Segments</h2>
              <span className="text-xs text-muted-foreground flex-1">{segments.length} total · edits are saved on Re-burn</span>
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
                  const flag = qaFlags.find((f) => f.segment_id === seg.id);
                  return (
                    <div
                      key={`${seg.id}-${idx}`}
                      ref={(el) => { if (el) segmentRefs.current.set(seg.id, el); }}
                      className={`rounded-lg border bg-card p-3 space-y-2 text-sm transition-colors ${
                        flag ? "border-amber-400/60 dark:border-amber-500/50" : "border-border"
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-mono text-muted-foreground w-7 shrink-0 text-right">#{idx + 1}</span>
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
                        <span className="text-xs tabular-nums text-muted-foreground shrink-0">
                          {formatTime(seg.start)}–{formatTime(seg.end)}
                        </span>
                      </div>
                      {lang === "vi" && (
                        <div className="flex items-center gap-2 pl-9">
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

            {/* ── Caption Style fields ── */}
            <div className="border-t border-border mt-6 pb-8">
              <h2 className="text-base font-semibold text-foreground mt-6 mb-4">Caption Style</h2>
              <div className="space-y-2.5">
                <StyleField label="Font" value={style.font} onChange={(v) => setStyle((s) => s && ({ ...s, font: v }))} placeholder="e.g. Arial, Impact, Trebuchet MS" />
                <StyleNumberField label="Font size" value={style.font_size} onChange={(v) => setStyle((s) => s && ({ ...s, font_size: v }))} />
                <StyleColorField label="Text color" value={style.primary_color} onChange={(v) => setStyle((s) => s && ({ ...s, primary_color: v }))} />
                <StyleColorField label="Outline" value={style.outline_color} onChange={(v) => setStyle((s) => s && ({ ...s, outline_color: v }))} />
                <StyleNumberField label="Outline width" value={style.outline_width} onChange={(v) => setStyle((s) => s && ({ ...s, outline_width: v }))} />
                <AlignmentPicker value={style.alignment} onChange={(v) => setStyle((s) => s && ({ ...s, alignment: v }))} />
                <StyleNumberField label="Margin from edge" value={style.margin_edge} onChange={(v) => setStyle((s) => s && ({ ...s, margin_edge: v }))} />
                <StyleNumberField label="Line length" value={style.max_line_length} onChange={(v) => setStyle((s) => s && ({ ...s, max_line_length: v }))} />
              </div>
              <p className="text-xs text-muted-foreground mt-3">Style changes apply on the next Re-burn.</p>
            </div>
          </div>
        </div>

        {/* ── Col 3: Hermes panel (380px) ── */}
        <div className="w-full xl:w-[380px] xl:shrink-0 border-t xl:border-t-0 border-border overflow-y-auto">
          <HermesPanel
            jobId={jobId}
            segments={segments}
            qaFlags={qaFlags}
            qaLoading={qaLoading}
            qaError={qaError}
            nlPrefill={nlPrefill}
            style={style}
            onApplyPatches={handleApplyNLPatches}
            onQAReview={handleQAReview}
            onClearQA={() => setQaFlags([])}
            onFlagClick={(segId) => {
              const el = segmentRefs.current.get(segId);
              if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
            }}
            onFixWithAI={(suggestion) => setNlPrefill(suggestion)}
            onApplyStyle={(s) => setStyle(mergeStyle(s))}
          />
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
// Caption style — alignment picker
// ---------------------------------------------------------------------------

const ALIGNMENT_GRID = [7, 8, 9, 4, 5, 6, 1, 2, 3] as const;
const ALIGNMENT_ICONS: Record<number, string> = {
  7: "↖", 8: "↑", 9: "↗",
  4: "←", 5: "·", 6: "→",
  1: "↙", 2: "↓", 3: "↘",
};
const ALIGNMENT_NAMES: Record<number, string> = {
  7: "top-left", 8: "top-center", 9: "top-right",
  4: "mid-left", 5: "mid-center", 6: "mid-right",
  1: "btm-left", 2: "btm-center", 3: "btm-right",
};

function AlignmentPicker({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-24 shrink-0 text-xs text-muted-foreground">Alignment</span>
      <div className="flex items-center gap-2">
        <div className="grid grid-cols-3 gap-0.5">
          {ALIGNMENT_GRID.map((n) => (
            <button
              key={n}
              type="button"
              onClick={() => onChange(n)}
              title={ALIGNMENT_NAMES[n]}
              className={`h-7 w-7 rounded text-xs border transition-colors ${
                value === n
                  ? "bg-amber-500 text-white border-amber-500"
                  : "bg-muted text-muted-foreground border-border hover:bg-accent"
              }`}
            >
              {ALIGNMENT_ICONS[n]}
            </button>
          ))}
        </div>
        <span className="text-xs text-muted-foreground">{ALIGNMENT_NAMES[value] ?? value}</span>
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
