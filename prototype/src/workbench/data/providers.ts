export type ProviderKind = "local" | "cloud";
export type ProviderStatus = "ready" | "offline" | "needs-key" | "checking" | "error";
export type ProviderModel = { id: string; label: string; family?: string; context?: string; size?: string };
export type ModelProvider = {
  id: string;
  name: string;
  kind: ProviderKind;
  description: string;
  baseUrl?: string;
  defaultUrl?: string;
  status: ProviderStatus;
  models: ProviderModel[];
  glyph: string;
  accent: string;
  docs?: string;
  hint?: string;
};

export type Project = {
  id: string;
  name: string;
  path?: string;
  color: string;
  kind: "repo" | "workspace";
  /** Inherited by every thread in the project unless the thread overrides it. */
  defaults?: { agentId?: string; modelId?: string; mode?: string; rules?: string[]; budget?: string };
};

export const PROJECTS: Project[] = [
  { id: "platform", name: "Platform", path: "aro/platform", color: "#8f82ff", kind: "repo",
    defaults: { agentId: "claude-code", modelId: "opus-4.6", mode: "agent", budget: "$25 / session", rules: ["Never touch generated/", "pnpm only", "Run biome before done"] } },
  { id: "payments", name: "Payments", path: "internal/payments", color: "#36c5b6", kind: "repo",
    defaults: { agentId: "codex", modelId: "gpt-5.4-codex", mode: "plan", budget: "$15 / session", rules: ["No migrations without a dry run", "Idempotency keys required"] } },
  { id: "editors", name: "Editor extensions", path: "aro/editors", color: "#ec9acb", kind: "repo",
    defaults: { agentId: "cursor", modelId: "composer-2.5", mode: "agent", budget: "$10 / session", rules: ["Match the host editor's UX conventions"] } },
  { id: "personal", name: "Personal", color: "#eab45a", kind: "workspace",
    defaults: { agentId: "glm-code", modelId: "glm-5", mode: "readonly", budget: "$5 / session", rules: [] } },
];

export const PROVIDER_SEEDS: ModelProvider[] = [
  {
    id: "ollama", name: "Ollama", kind: "local", description: "Local models · private · OpenAI-compatible", baseUrl: "http://localhost:11434", defaultUrl: "http://localhost:11434",
    status: "offline", glyph: "OL", accent: "#93c5a7", models: [], docs: "https://docs.ollama.com/api/tags", hint: "GET /api/tags · default port 11434",
  },
  {
    id: "lm-studio", name: "LM Studio", kind: "local", description: "Desktop model library · local inference", baseUrl: "http://localhost:1234/v1", defaultUrl: "http://localhost:1234/v1",
    status: "offline", glyph: "LM", accent: "#a89cff", models: [], docs: "https://lmstudio.ai/docs/developer/openai-compat/models", hint: "GET /v1/models · default port 1234",
  },
  {
    id: "openai-compatible", name: "OpenAI-compatible", kind: "local", description: "Local server or custom gateway", baseUrl: "http://localhost:8000/v1", defaultUrl: "http://localhost:8000/v1",
    status: "offline", glyph: "API", accent: "#6da8ff", models: [], hint: "LM Studio · vLLM · llama.cpp · LocalAI",
  },
  {
    id: "anthropic", name: "Anthropic", kind: "cloud", description: "Claude models · hosted API", status: "needs-key", glyph: "A", accent: "#d5a98c",
    models: [{ id: "claude-opus-4-6", label: "Claude Opus 4.6", context: "200k" }, { id: "claude-sonnet-4-6", label: "Claude Sonnet 4.6", context: "200k" }, { id: "claude-haiku-4-5", label: "Claude Haiku 4.5", context: "200k" }], hint: "API key stored locally in this prototype",
  },
  {
    id: "openai", name: "OpenAI", kind: "cloud", description: "GPT and Codex models · hosted API", status: "needs-key", glyph: "◎", accent: "#79ceb6",
    models: [{ id: "gpt-5.4", label: "GPT-5.4", context: "400k" }, { id: "gpt-5.4-codex", label: "GPT-5.4-Codex", context: "400k" }, { id: "gpt-5-mini", label: "GPT-5 mini", context: "400k" }], hint: "API key stored locally in this prototype",
  },
  {
    id: "google", name: "Google AI Studio", kind: "cloud", description: "Gemini models · hosted API", status: "needs-key", glyph: "G", accent: "#80a9ff",
    models: [{ id: "gemini-2.5-pro", label: "Gemini 2.5 Pro", context: "1M" }, { id: "gemini-2.5-flash", label: "Gemini 2.5 Flash", context: "1M" }], hint: "API key stored locally in this prototype",
  },
  {
    id: "openrouter", name: "OpenRouter", kind: "cloud", description: "Multi-provider model gateway", status: "needs-key", glyph: "OR", accent: "#f2b45c",
    models: [{ id: "openrouter/auto", label: "Auto Router", context: "varies" }, { id: "deepseek/deepseek-r1", label: "DeepSeek R1", context: "128k" }, { id: "qwen/qwen3-coder", label: "Qwen3 Coder", context: "256k" }], hint: "One key · multiple model families",
  },
];

export const DEFAULT_PROJECT_FOR_SESSION: Record<string, string> = {
  s1: "platform", s2: "payments", s3: "platform", s4: "platform", s5: "platform", s6: "platform", s7: "editors", s8: "platform",
  a1: "platform", a2: "personal", a3: "personal", a4: "platform", a5: "personal",
};

export function normalizeProjectId(id?: string) {
  return id && PROJECTS.some((p) => p.id === id) ? id : "platform";
}

/** Local servers expose model discovery endpoints; failed CORS is reported separately from offline. */
export async function probeProvider(provider: ModelProvider): Promise<{ status: ProviderStatus; models: ProviderModel[]; message: string }> {
  if (provider.kind !== "local") return { status: "needs-key", models: provider.models, message: "Cloud API key required" };
  const root = (provider.baseUrl || provider.defaultUrl || "").replace(/\/$/, "");
  const url = provider.id === "ollama" ? `${root}/api/tags` : `${root}/models`;
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), 2400);
  try {
    const res = await fetch(url, { signal: controller.signal, headers: { Accept: "application/json" } });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json() as { models?: { name?: string; model?: string; id?: string; details?: { family?: string; parameter_size?: string }; context_length?: number }[]; data?: { id?: string }[] };
    const raw = provider.id === "ollama" ? (data.models ?? []).map((m) => ({ id: m.name ?? m.model ?? "", label: m.name ?? m.model ?? "", family: m.details?.family, size: m.details?.parameter_size })) : (data.data ?? []).map((m) => ({ id: m.id ?? "", label: m.id ?? "", context: "server managed" }));
    const models = raw.filter((m) => m.id).map((m) => ({ ...m }));
    return { status: "ready", models, message: `${models.length} model${models.length === 1 ? "" : "s"} found` };
  } catch (error) {
    const message = error instanceof Error && error.name === "AbortError" ? "Timed out. Is the local server running?" : error instanceof TypeError ? "Could not reach the server. Check that it is running and allows this app's origin." : error instanceof Error ? error.message : "Connection failed";
    return { status: "offline", models: [], message };
  } finally {
    window.clearTimeout(timer);
  }
}