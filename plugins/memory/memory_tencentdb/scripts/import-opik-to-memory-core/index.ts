#!/usr/bin/env npx tsx

import { createHash } from "node:crypto";
import { existsSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { resolve as resolvePath } from "node:path";
import { parseArgs } from "node:util";
import { pathToFileURL } from "node:url";

type JsonObject = Record<string, unknown>;

type Role = "user" | "assistant";

interface MemoryMessage {
  role: Role;
  content: string;
  timestamp: string;
}

interface OpikProject {
  id: string;
  name: string;
}

interface OpikTrace {
  id: string;
  project_id?: string;
  project_name?: string;
  thread_id?: string;
  name?: string;
  start_time?: string;
  end_time?: string;
  created_at?: string;
  input?: unknown;
  output?: unknown;
  metadata?: unknown;
}

interface Page<T> {
  page: number;
  size: number;
  total: number;
  content: T[];
}

interface ImportState {
  version: 1;
  completed: Record<string, { imported_at: string; accepted: number }>;
}

interface CliOptions {
  opikApiBase: string;
  memoryUrl: string;
  workspace: string;
  projects: string[];
  teamId: string;
  agentId: string;
  userId: string;
  taskId?: string;
  serviceId: string;
  pageSize: number;
  maxTraces: number;
  stateFile: string;
  resume: boolean;
  dryRun: boolean;
  includeSystem: boolean;
  waitEvery: number;
  finalWait: boolean;
  pollMs: number;
  maxWaitMs: number;
  timeoutMs: number;
  retries: number;
}

interface ApiEnvelope<T> {
  code: number;
  message: string;
  request_id?: string;
  data?: T;
}

const MAX_MESSAGE_CHARS = 8192;
const MAX_MESSAGES_PER_REQUEST = 100;

function usage(): string {
  return `
从 Opik 导入对话 Trace 到 Memory Core L0。

用法：
  npm run import:opik -- [选项]

数据源：
  --opik-url <url>       Opik UI 或 API 地址；也可用 OPIK_URL
                         例如 http://opik.example.com:5173/default/projects?size=25
  --workspace <name>     Opik workspace，默认 OPIK_WORKSPACE 或 URL 首段/default
  --project <name>       只导入指定项目；可重复或逗号分隔，默认全部项目
  --page-size <n>        Opik 每页数量，默认 100
  --max-traces <n>       本次最多处理 Trace 数，0 表示不限，默认 0

Memory Core：
  --memory-url <url>     Gateway 地址，默认 MEMORY_CORE_URL 或 http://127.0.0.1:8420
  --service-id <id>      x-tdai-service-id；也可用 MEMORY_CORE_SERVICE_ID
  --team-id <id>         目标 team；也可用 MEMORY_CORE_TEAM_ID
  --agent-id <id>        目标 agent；也可用 MEMORY_CORE_AGENT_ID
  --user-id <id>         目标 user；也可用 MEMORY_CORE_USER_ID
  --task-id <id>         可选目标 task；也可用 MEMORY_CORE_TASK_ID

执行控制：
  --state-file <path>    断点文件，默认 .opik-memory-import-state.json
  --no-resume            忽略已有断点
  --dry-run              拉取并转换，但不写 Memory Core/断点
  --include-system       将 system/developer 消息以 user 消息导入
  --wait-every <n>       每 N 个写请求等待 L1 空闲；0 禁用，默认 20
  --no-final-wait        完成后不等待 L1/L2/L3 全部空闲
  --poll-ms <n>          Pipeline 轮询间隔，默认 1000
  --max-wait-ms <n>      单次等待上限，默认 600000
  --timeout-ms <n>       HTTP 请求超时，默认 30000
  --retries <n>          429/5xx/网络错误重试次数，默认 4
  -h, --help             显示帮助

密钥仅从环境变量读取：
  OPIK_API_KEY           可选；原样写入 Opik Authorization header
  OPIK_AUTH_SCHEME       可选，如 Bearer；默认不加前缀
  MEMORY_CORE_API_KEY    必填（dry-run 除外）

示例：
  OPIK_URL='http://opik.example.com:5173/default/projects?size=25' \\
  MEMORY_CORE_URL='http://127.0.0.1:8420' \\
  MEMORY_CORE_API_KEY='***' MEMORY_CORE_SERVICE_ID='default' \\
  MEMORY_CORE_TEAM_ID='team-1' MEMORY_CORE_AGENT_ID='agent-1' \\
  MEMORY_CORE_USER_ID='user-1' \\
  npm run import:opik -- --dry-run
`.trim();
}

function parsePositiveInt(value: string | undefined, fallback: number, name: string, allowZero = false): number {
  if (value == null || value === "") return fallback;
  const parsed = Number(value);
  const valid = Number.isSafeInteger(parsed) && (allowZero ? parsed >= 0 : parsed > 0);
  if (!valid) throw new Error(`${name} 必须是${allowZero ? "非负" : "正"}整数`);
  return parsed;
}

function required(value: string | undefined, name: string): string {
  const normalized = value?.trim();
  if (!normalized) throw new Error(`缺少 ${name}`);
  return normalized;
}

function parseOpikBase(raw: string, explicitWorkspace?: string): { apiBase: string; workspace: string } {
  const url = new URL(raw);
  if (url.username || url.password) throw new Error("OPIK_URL 不允许包含凭据，请使用环境变量传递密钥");

  const parts = url.pathname.split("/").filter(Boolean);
  const workspaceFromUi = parts[0] && !["api", "v1"].includes(parts[0]) ? parts[0] : undefined;
  let apiPath: string;
  const apiIndex = parts.indexOf("api");
  const v1Index = parts.indexOf("v1");
  if (apiIndex >= 0 && parts[apiIndex + 1] === "v1") {
    apiPath = `/${parts.slice(0, apiIndex + 2).join("/")}/private`;
  } else if (v1Index >= 0) {
    apiPath = `/${parts.slice(0, v1Index + 1).join("/")}/private`;
  } else {
    apiPath = "/api/v1/private";
  }

  url.pathname = apiPath.replace(/\/private\/private$/, "/private");
  url.search = "";
  url.hash = "";
  return {
    apiBase: url.toString().replace(/\/$/, ""),
    workspace: explicitWorkspace?.trim() || workspaceFromUi || "default",
  };
}

function parseCli(argv: string[]): CliOptions {
  const { values } = parseArgs({
    args: argv,
    options: {
      "opik-url": { type: "string" },
      "memory-url": { type: "string" },
      workspace: { type: "string" },
      project: { type: "string", multiple: true },
      "team-id": { type: "string" },
      "agent-id": { type: "string" },
      "user-id": { type: "string" },
      "task-id": { type: "string" },
      "service-id": { type: "string" },
      "page-size": { type: "string" },
      "max-traces": { type: "string" },
      "state-file": { type: "string" },
      "no-resume": { type: "boolean" },
      "dry-run": { type: "boolean" },
      "include-system": { type: "boolean" },
      "wait-every": { type: "string" },
      "no-final-wait": { type: "boolean" },
      "poll-ms": { type: "string" },
      "max-wait-ms": { type: "string" },
      "timeout-ms": { type: "string" },
      retries: { type: "string" },
      help: { type: "boolean", short: "h" },
    },
    strict: true,
    allowPositionals: false,
  });

  if (values.help) {
    console.log(usage());
    process.exit(0);
  }

  const rawOpikUrl = required((values["opik-url"] as string | undefined) ?? process.env.OPIK_URL, "--opik-url 或 OPIK_URL");
  const opik = parseOpikBase(rawOpikUrl, (values.workspace as string | undefined) ?? process.env.OPIK_WORKSPACE);
  const projectValues = (values.project as string[] | undefined) ?? [];
  const projects = projectValues.flatMap((item) => item.split(",")).map((item) => item.trim()).filter(Boolean);
  const dryRun = Boolean(values["dry-run"]);

  if (!dryRun) required(process.env.MEMORY_CORE_API_KEY, "MEMORY_CORE_API_KEY");

  return {
    opikApiBase: opik.apiBase,
    memoryUrl: ((values["memory-url"] as string | undefined) ?? process.env.MEMORY_CORE_URL ?? "http://127.0.0.1:8420").replace(/\/$/, ""),
    workspace: opik.workspace,
    projects,
    teamId: required((values["team-id"] as string | undefined) ?? process.env.MEMORY_CORE_TEAM_ID, "--team-id 或 MEMORY_CORE_TEAM_ID"),
    agentId: required((values["agent-id"] as string | undefined) ?? process.env.MEMORY_CORE_AGENT_ID, "--agent-id 或 MEMORY_CORE_AGENT_ID"),
    userId: required((values["user-id"] as string | undefined) ?? process.env.MEMORY_CORE_USER_ID, "--user-id 或 MEMORY_CORE_USER_ID"),
    taskId: ((values["task-id"] as string | undefined) ?? process.env.MEMORY_CORE_TASK_ID)?.trim() || undefined,
    serviceId: required((values["service-id"] as string | undefined) ?? process.env.MEMORY_CORE_SERVICE_ID, "--service-id 或 MEMORY_CORE_SERVICE_ID"),
    pageSize: parsePositiveInt(values["page-size"] as string | undefined, 100, "--page-size"),
    maxTraces: parsePositiveInt(values["max-traces"] as string | undefined, 0, "--max-traces", true),
    stateFile: resolvePath((values["state-file"] as string | undefined) ?? ".opik-memory-import-state.json"),
    resume: !values["no-resume"],
    dryRun,
    includeSystem: Boolean(values["include-system"]),
    waitEvery: parsePositiveInt(values["wait-every"] as string | undefined, 20, "--wait-every", true),
    finalWait: !values["no-final-wait"],
    pollMs: parsePositiveInt(values["poll-ms"] as string | undefined, 1000, "--poll-ms"),
    maxWaitMs: parsePositiveInt(values["max-wait-ms"] as string | undefined, 600_000, "--max-wait-ms"),
    timeoutMs: parsePositiveInt(values["timeout-ms"] as string | undefined, 30_000, "--timeout-ms"),
    retries: parsePositiveInt(values.retries as string | undefined, 4, "--retries", true),
  };
}

function isRecord(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringifyPrimitive(value: unknown): string | undefined {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return undefined;
}

function contentToText(value: unknown): string | undefined {
  const primitive = stringifyPrimitive(value);
  if (primitive != null) return primitive;
  if (Array.isArray(value)) {
    const parts = value.map((item) => {
      if (typeof item === "string") return item;
      if (!isRecord(item)) return undefined;
      if (item.type === "text" && typeof item.text === "string") return item.text;
      if (typeof item.content === "string") return item.content;
      if (typeof item.text === "string") return item.text;
      return undefined;
    }).filter((item): item is string => Boolean(item?.trim()));
    return parts.length > 0 ? parts.join("\n") : undefined;
  }
  if (!isRecord(value)) return undefined;
  for (const key of ["content", "text", "responseContent", "answer", "response", "completion"]) {
    const text = contentToText(value[key]);
    if (text?.trim()) return text;
  }
  if (isRecord(value.message)) return contentToText(value.message);
  if (Array.isArray(value.choices)) {
    for (const choice of value.choices) {
      if (!isRecord(choice)) continue;
      const text = contentToText(choice.message ?? choice.text);
      if (text?.trim()) return text;
    }
  }
  return undefined;
}

function normalizeRole(value: unknown, includeSystem: boolean): Role | undefined {
  if (typeof value !== "string") return undefined;
  const role = value.toLowerCase();
  if (["user", "human"].includes(role)) return "user";
  if (["assistant", "ai", "model", "bot"].includes(role)) return "assistant";
  if (includeSystem && ["system", "developer"].includes(role)) return "user";
  return undefined;
}

function normalizeTimestamp(value: unknown, fallback: string): string {
  const candidate = typeof value === "number" ? new Date(value).toISOString() : typeof value === "string" ? value : fallback;
  const date = new Date(candidate);
  return Number.isNaN(date.getTime()) ? fallback : date.toISOString();
}

function splitContent(text: string): string[] {
  const normalized = text.trim();
  if (!normalized) return [];
  const chunks: string[] = [];
  let offset = 0;
  while (offset < normalized.length) {
    let end = Math.min(offset + MAX_MESSAGE_CHARS, normalized.length);
    if (end < normalized.length) {
      const lastCode = normalized.charCodeAt(end - 1);
      if (lastCode >= 0xd800 && lastCode <= 0xdbff) end--;
    }
    chunks.push(normalized.slice(offset, end));
    offset = end;
  }
  return chunks;
}

function normalizeMessage(value: unknown, fallbackTimestamp: string, includeSystem: boolean): MemoryMessage[] {
  if (!isRecord(value)) return [];
  const role = normalizeRole(value.role ?? value.type ?? (isRecord(value.message) ? value.message.role : undefined), includeSystem);
  if (!role) return [];
  const rawContent = value.content ?? (isRecord(value.message) ? value.message.content : undefined) ?? value.text;
  const text = contentToText(rawContent);
  if (!text) return [];
  const timestamp = normalizeTimestamp(value.timestamp ?? value.created_at ?? value.createdAt, fallbackTimestamp);
  const rolePrefix = includeSystem && ["system", "developer"].includes(String(value.role ?? value.type).toLowerCase())
    ? `[${String(value.role ?? value.type).toLowerCase()}] `
    : "";
  return splitContent(`${rolePrefix}${text}`).map((content) => ({ role, content, timestamp }));
}

function findMessageArrays(value: unknown, depth = 0): unknown[][] {
  if (depth > 5) return [];
  if (Array.isArray(value)) {
    const looksLikeMessages = value.some((item) => isRecord(item) && ("role" in item || (isRecord(item.message) && "role" in item.message)));
    return looksLikeMessages ? [value] : value.flatMap((item) => findMessageArrays(item, depth + 1));
  }
  if (!isRecord(value)) return [];
  const preferred = ["messages", "conversation", "history"]
    .flatMap((key) => key in value ? findMessageArrays(value[key], depth + 1) : []);
  if (preferred.length > 0) return preferred;
  return Object.values(value).flatMap((item) => findMessageArrays(item, depth + 1));
}

function extractPrompt(value: unknown, keys: string[]): string | undefined {
  const primitive = stringifyPrimitive(value);
  if (primitive?.trim()) return primitive;
  if (!isRecord(value)) return undefined;
  for (const key of keys) {
    if (!(key in value)) continue;
    const text = contentToText(value[key]);
    if (text?.trim()) return text;
  }
  return contentToText(value);
}

function mergeMessages(input: MemoryMessage[], output: MemoryMessage[]): MemoryMessage[] {
  const same = (a: MemoryMessage, b: MemoryMessage): boolean => a.role === b.role && a.content === b.content;
  const maxOverlap = Math.min(input.length, output.length);
  for (let overlap = maxOverlap; overlap > 0; overlap--) {
    const inputStart = input.length - overlap;
    if (output.slice(0, overlap).every((message, index) => same(message, input[inputStart + index]!))) {
      return [...input, ...output.slice(overlap)];
    }
  }
  return [...input, ...output];
}

function bestMessageArray(value: unknown, fallbackTimestamp: string, includeSystem: boolean): MemoryMessage[] {
  return findMessageArrays(value)
    .map((items) => items.flatMap((item) => normalizeMessage(item, fallbackTimestamp, includeSystem)))
    .sort((a, b) => b.length - a.length)[0] ?? [];
}

export function extractMessages(trace: OpikTrace, includeSystem = false): MemoryMessage[] {
  const inputTime = normalizeTimestamp(trace.start_time ?? trace.created_at, new Date(0).toISOString());
  const outputTime = normalizeTimestamp(trace.end_time ?? trace.start_time ?? trace.created_at, inputTime);
  const inputMessages = bestMessageArray(trace.input, inputTime, includeSystem);
  const outputMessages = bestMessageArray(trace.output, outputTime, includeSystem);
  const merged = mergeMessages(inputMessages, outputMessages);
  if (merged.length > 0) return merged;

  const result: MemoryMessage[] = [];
  const prompt = extractPrompt(trace.input, ["userPrompt", "user_prompt", "prompt", "query", "input", "content", "text"]);
  const answer = extractPrompt(trace.output, ["responseContent", "response_content", "answer", "response", "completion", "output", "content", "text"]);
  if (prompt) result.push(...splitContent(prompt).map((content) => ({ role: "user" as const, content, timestamp: inputTime })));
  if (answer) result.push(...splitContent(answer).map((content) => ({ role: "assistant" as const, content, timestamp: outputTime })));
  return result;
}

function chunk<T>(items: T[], size: number): T[][] {
  const chunks: T[][] = [];
  for (let i = 0; i < items.length; i += size) chunks.push(items.slice(i, i + size));
  return chunks;
}

function stableHash(value: unknown, length = 20): string {
  return createHash("sha256").update(JSON.stringify(value)).digest("hex").slice(0, length);
}

function buildSessionId(project: OpikProject, trace: OpikTrace): string {
  const source = trace.thread_id?.trim() || trace.id;
  const readable = source.replace(/[^a-zA-Z0-9._-]/g, "-").slice(0, 48) || "trace";
  return `opik:${project.id}:${readable}:${stableHash(source, 12)}`;
}

function loadState(path: string, resume: boolean): ImportState {
  if (!resume || !existsSync(path)) return { version: 1, completed: {} };
  const parsed = JSON.parse(readFileSync(path, "utf8")) as Partial<ImportState>;
  if (parsed.version !== 1 || !isRecord(parsed.completed)) throw new Error(`断点文件格式不支持: ${path}`);
  return parsed as ImportState;
}

function saveState(path: string, state: ImportState): void {
  const tmp = `${path}.tmp-${process.pid}`;
  writeFileSync(tmp, `${JSON.stringify(state, null, 2)}\n`, { encoding: "utf8", mode: 0o600 });
  renameSync(tmp, path);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

class NonRetryableHttpError extends Error {}

class HttpClient {
  constructor(private readonly opts: Pick<CliOptions, "timeoutMs" | "retries">) {}

  async json<T>(url: string, init: RequestInit): Promise<T> {
    let lastError: unknown;
    for (let attempt = 0; attempt <= this.opts.retries; attempt++) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.opts.timeoutMs);
      try {
        const response = await fetch(url, { ...init, signal: controller.signal });
        const text = await response.text();
        if (!response.ok) {
          const message = `HTTP ${response.status} ${response.statusText}: ${text.slice(0, 500)}`;
          if (response.status !== 429 && response.status < 500) throw new NonRetryableHttpError(message);
          lastError = new Error(message);
        } else {
          try {
            return JSON.parse(text) as T;
          } catch {
            throw new Error(`HTTP ${response.status} 返回非 JSON: ${text.slice(0, 500)}`);
          }
        }
      } catch (error) {
        if (error instanceof NonRetryableHttpError) throw error;
        lastError = error;
        if (attempt >= this.opts.retries) break;
      } finally {
        clearTimeout(timer);
      }
      const delay = Math.min(1000 * 2 ** attempt, 10_000) + Math.floor(Math.random() * 250);
      console.warn(`[retry] ${attempt + 1}/${this.opts.retries}，${delay}ms 后重试: ${String(lastError)}`);
      await sleep(delay);
    }
    throw lastError instanceof Error ? lastError : new Error(String(lastError));
  }
}

class OpikClient {
  constructor(private readonly opts: CliOptions, private readonly http: HttpClient) {}

  private headers(): Record<string, string> {
    const headers: Record<string, string> = {
      Accept: "application/json",
      "Comet-Workspace-Name": this.opts.workspace,
    };
    const apiKey = process.env.OPIK_API_KEY?.trim();
    if (apiKey) {
      const scheme = process.env.OPIK_AUTH_SCHEME?.trim();
      headers.Authorization = scheme ? `${scheme} ${apiKey}` : apiKey;
    }
    return headers;
  }

  private async getPage<T>(path: string, params: URLSearchParams): Promise<Page<T>> {
    const url = `${this.opts.opikApiBase}${path}?${params.toString()}`;
    const page = await this.http.json<Page<T>>(url, { headers: this.headers() });
    if (!Array.isArray(page.content) || typeof page.total !== "number") {
      throw new Error(`Opik 分页响应格式异常: ${url}`);
    }
    return page;
  }

  async projects(): Promise<OpikProject[]> {
    const result: OpikProject[] = [];
    for (let pageNo = 1; ; pageNo++) {
      const page = await this.getPage<OpikProject>("/projects", new URLSearchParams({ page: String(pageNo), size: String(this.opts.pageSize) }));
      result.push(...page.content.filter((item) => typeof item.id === "string" && typeof item.name === "string"));
      if (result.length >= page.total || page.content.length === 0) break;
    }
    return result;
  }

  async traces(project: OpikProject): Promise<OpikTrace[]> {
    const result: OpikTrace[] = [];
    for (let pageNo = 1; ; pageNo++) {
      const page = await this.getPage<OpikTrace>("/traces", new URLSearchParams({
        page: String(pageNo),
        size: String(this.opts.pageSize),
        project_id: project.id,
        truncate: "false",
        strip_attachments: "true",
      }));
      result.push(...page.content.filter((item) => typeof item.id === "string"));
      if (result.length >= page.total || page.content.length === 0) break;
    }
    return result.sort((a, b) => {
      const ta = Date.parse(a.start_time ?? a.created_at ?? "") || 0;
      const tb = Date.parse(b.start_time ?? b.created_at ?? "") || 0;
      return ta - tb || a.id.localeCompare(b.id);
    });
  }
}

class MemoryCoreClient {
  constructor(private readonly opts: CliOptions, private readonly http: HttpClient) {}

  private headers(): Record<string, string> {
    return {
      Accept: "application/json",
      "Content-Type": "application/json",
      Authorization: `Bearer ${required(process.env.MEMORY_CORE_API_KEY, "MEMORY_CORE_API_KEY")}`,
      "x-tdai-service-id": this.opts.serviceId,
    };
  }

  private async post<T>(path: string, body: unknown): Promise<ApiEnvelope<T>> {
    const response = await this.http.json<ApiEnvelope<T>>(`${this.opts.memoryUrl}${path}`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(body),
    });
    if (response.code !== 0) throw new Error(`${path} 失败: code=${response.code} message=${response.message} request_id=${response.request_id ?? "-"}`);
    return response;
  }

  async add(sessionId: string, messages: MemoryMessage[]): Promise<number> {
    const response = await this.post<{ accepted_ids: string[]; total_count: number }>("/v3/conversation/add", {
      team_id: this.opts.teamId,
      agent_id: this.opts.agentId,
      user_id: this.opts.userId,
      ...(this.opts.taskId ? { task_id: this.opts.taskId } : {}),
      session_id: sessionId,
      messages,
    });
    return response.data?.accepted_ids?.length ?? response.data?.total_count ?? 0;
  }

  async waitForIdle(mode: "l1" | "all", label: string): Promise<void> {
    const startedAt = Date.now();
    while (Date.now() - startedAt <= this.opts.maxWaitMs) {
      const response = await this.post<{
        l1: { idle: boolean; queued: number; running: number };
        l2: { idle: boolean; queued: number; running: number };
        l3: { idle: boolean; queued: number; running: number };
      }>("/v2/pipeline/status", {});
      const status = response.data;
      if (!status) throw new Error("/v2/pipeline/status 缺少 data");
      const idle = mode === "l1" ? status.l1.idle : status.l1.idle && status.l2.idle && status.l3.idle;
      if (idle) {
        console.log(`[pipeline] ${label} 已空闲`);
        return;
      }
      await sleep(this.opts.pollMs);
    }
    console.warn(`[pipeline] ${label} 等待超过 ${this.opts.maxWaitMs}ms，导入数据已落 L0，后台将继续处理`);
  }
}

async function main(): Promise<void> {
  const opts = parseCli(process.argv.slice(2));
  const http = new HttpClient(opts);
  const opik = new OpikClient(opts, http);
  const memory = new MemoryCoreClient(opts, http);
  const state = loadState(opts.stateFile, opts.resume);

  console.log(`[config] opik=${opts.opikApiBase} workspace=${opts.workspace} memory=${opts.memoryUrl}`);
  console.log(`[config] target team=${opts.teamId} agent=${opts.agentId} user=${opts.userId} service=${opts.serviceId} dryRun=${opts.dryRun}`);

  const allProjects = await opik.projects();
  const selected = opts.projects.length === 0
    ? allProjects
    : allProjects.filter((project) => opts.projects.includes(project.name) || opts.projects.includes(project.id));
  const missingProjects = opts.projects.filter((name) => !selected.some((project) => project.name === name || project.id === name));
  if (missingProjects.length > 0) throw new Error(`Opik 项目不存在: ${missingProjects.join(", ")}`);
  console.log(`[opik] 找到 ${allProjects.length} 个项目，本次处理 ${selected.length} 个`);

  let seenTraces = 0;
  let importedTraces = 0;
  let skippedTraces = 0;
  let importedMessages = 0;
  let writeCount = 0;

  outer: for (const project of selected) {
    const traces = await opik.traces(project);
    console.log(`[project] ${project.name} (${project.id}) traces=${traces.length}`);
    for (const trace of traces) {
      if (opts.maxTraces > 0 && seenTraces >= opts.maxTraces) break outer;
      seenTraces++;
      const messages = extractMessages(trace, opts.includeSystem);
      if (messages.length === 0) {
        skippedTraces++;
        console.warn(`[skip] project=${project.name} trace=${trace.id} 原始 input/output 中没有可导入对话`);
        continue;
      }

      const batches = chunk(messages, MAX_MESSAGES_PER_REQUEST);
      const sessionId = buildSessionId(project, trace);
      let wroteTrace = false;
      for (let index = 0; index < batches.length; index++) {
        const batch = batches[index]!;
        const checkpointKey = `${project.id}:${trace.id}:${stableHash(batch)}:${index}`;
        if (opts.resume && state.completed[checkpointKey]) continue;
        if (opts.dryRun) {
          console.log(`[dry-run] project=${project.name} trace=${trace.id} session=${sessionId} batch=${index + 1}/${batches.length} messages=${batch.length}`);
          continue;
        }

        const accepted = await memory.add(sessionId, batch);
        if (accepted !== batch.length) throw new Error(`Memory Core 接收数量不一致: trace=${trace.id} expected=${batch.length} accepted=${accepted}`);
        state.completed[checkpointKey] = { imported_at: new Date().toISOString(), accepted };
        saveState(opts.stateFile, state);
        importedMessages += accepted;
        writeCount++;
        wroteTrace = true;
        console.log(`[import] project=${project.name} trace=${trace.id} batch=${index + 1}/${batches.length} accepted=${accepted}`);

        if (opts.waitEvery > 0 && writeCount % opts.waitEvery === 0) {
          await memory.waitForIdle("l1", `write-${writeCount}`);
        }
      }
      if (wroteTrace) importedTraces++;
    }
  }

  if (!opts.dryRun && opts.finalWait && writeCount > 0) await memory.waitForIdle("all", "final");
  console.log(`[done] seen_traces=${seenTraces} imported_traces=${importedTraces} skipped_traces=${skippedTraces} writes=${writeCount} imported_messages=${importedMessages}`);
  if (!opts.dryRun) console.log(`[done] 断点文件: ${opts.stateFile}`);
}

const entry = process.argv[1] ? pathToFileURL(resolvePath(process.argv[1])).href : "";
if (import.meta.url === entry) {
  main().catch((error) => {
    console.error(`[fatal] ${error instanceof Error ? error.stack ?? error.message : String(error)}`);
    process.exitCode = 1;
  });
}
