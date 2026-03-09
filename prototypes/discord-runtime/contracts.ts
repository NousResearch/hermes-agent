export type SurfaceKind = "channel" | "thread" | "forum_post";

export type IdentityMode = "distinct_actor" | "shared_persona";

export type DeliveryMode = "send" | "edit" | "interaction_reply" | "interaction_followup";

export type SummaryScope = "local" | "parent" | "server";

export interface DiscordSurfaceRef {
  serverId: string;
  surfaceId: string;
  surfaceKind: SurfaceKind;
  channelId: string;
  threadId?: string;
  parentSurfaceId?: string;
  parentChannelId?: string;
  forumChannelId?: string;
  name?: string;
}

export interface AgentRef {
  agentId: string;
  displayName: string;
  identityMode: IdentityMode;
}

export interface ActorRef {
  userId: string;
  username?: string;
  displayName?: string;
  roleIds?: string[];
}

export interface MessageReference {
  messageId: string;
  authorLabel?: string;
  excerpt: string;
  surfaceId?: string;
  timestamp?: string;
}

export interface AttachmentRef {
  attachmentId: string;
  kind: "image" | "audio" | "video" | "file";
  mimeType?: string;
  filename?: string;
  localPath?: string;
  sourceUrl?: string;
  sizeBytes?: number;
}

export interface SummaryRef {
  summaryId: string;
  scope: SummaryScope;
  sourceSessionKey?: string;
  text: string;
  createdAt: string;
}

export interface LineageRef {
  parentSessionKey?: string;
  parentSurfaceId?: string;
  rootSessionKey?: string;
  edgeReason?: "surface_branch" | "subagent_spawn" | "manual_handoff";
  handoffSummary?: SummaryRef;
}

export interface InteractionInput {
  type: "slash_command" | "button" | "select" | "modal";
  interactionId: string;
  commandName?: string;
  customId?: string;
  values?: string[];
  fields?: Record<string, string>;
}

export interface HumanInput {
  text: string;
  messageId?: string;
  replyTo?: MessageReference;
  attachments?: AttachmentRef[];
  interaction?: InteractionInput;
}

export interface TunableContextPolicy {
  localMessageBudget: number;
  localTokenBudget: number;
  parentSummaryBudget: number;
  serverSummaryBudget: number;
  maxReplyExcerptChars: number;
  allowServerSummaryAwareness: boolean;
  discrawlMode: "disabled" | "ops_only" | "on_demand" | "background";
}

export interface TraceEnvelope {
  traceId: string;
  turnId: string;
  parentSpanId?: string;
  startedAt: string;
  tags?: Record<string, string>;
}

export interface TurnRequest {
  trace: TraceEnvelope;
  agent: AgentRef;
  surface: DiscordSurfaceRef;
  sessionKey: string;
  actor: ActorRef;
  input: HumanInput;
  lineage?: LineageRef;
  localSummaries?: SummaryRef[];
  parentSummaries?: SummaryRef[];
  serverSummaries?: SummaryRef[];
  channelTopic?: string;
  policy: TunableContextPolicy;
  metadata?: Record<string, string>;
}

export interface UsageStats {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
  model?: string;
  provider?: string;
}

export interface DebugArtifactRef {
  artifactId: string;
  kind:
    | "prompt_snapshot"
    | "tool_timeline"
    | "message_trace"
    | "summary_trace"
    | "raw_backend_response";
  uri?: string;
}

export interface TextReplyOp {
  op: "text";
  opId: string;
  delivery: DeliveryMode;
  text: string;
  replyToMessageId?: string;
  targetSurfaceId?: string;
  ephemeral?: boolean;
}

export interface AttachmentReplyOp {
  op: "attachment";
  opId: string;
  delivery: DeliveryMode;
  attachments: AttachmentRef[];
  caption?: string;
  replyToMessageId?: string;
  targetSurfaceId?: string;
}

export interface ButtonSpec {
  actionId: string;
  label: string;
  style?: "primary" | "secondary" | "success" | "danger";
  value?: string;
}

export interface SelectOptionSpec {
  value: string;
  label: string;
  description?: string;
}

export interface ButtonsReplyOp {
  op: "buttons";
  opId: string;
  delivery: DeliveryMode;
  text?: string;
  buttons: ButtonSpec[];
  replyToMessageId?: string;
  targetSurfaceId?: string;
}

export interface SelectReplyOp {
  op: "select";
  opId: string;
  delivery: DeliveryMode;
  text?: string;
  customId: string;
  placeholder?: string;
  options: SelectOptionSpec[];
  minValues?: number;
  maxValues?: number;
  replyToMessageId?: string;
  targetSurfaceId?: string;
}

export interface ModalFieldSpec {
  fieldId: string;
  label: string;
  style?: "short" | "paragraph";
  required?: boolean;
  placeholder?: string;
  value?: string;
}

export interface ModalReplyOp {
  op: "modal";
  opId: string;
  delivery: DeliveryMode;
  customId: string;
  title: string;
  submitLabel?: string;
  fields: ModalFieldSpec[];
  triggerText?: string;
}

export interface ThreadActionReplyOp {
  op: "thread_action";
  opId: string;
  action: "spawn_child_surface" | "handoff_to_child" | "rebind_surface";
  targetAgent?: AgentRef;
  childSurfaceName?: string;
  handoffSummary?: string;
  parentSessionKey?: string;
}

export interface SummaryMutationReplyOp {
  op: "summary_mutation";
  opId: string;
  scope: SummaryScope;
  action: "append" | "replace";
  text: string;
  sourceSessionKey?: string;
}

export interface SpawnAgentReplyOp {
  op: "spawn_agent";
  opId: string;
  agent: AgentRef;
  reason?: string;
  preferredSurface?: "thread" | "forum_post";
  handoffSummary?: string;
}

export type ReplyOp =
  | TextReplyOp
  | AttachmentReplyOp
  | ButtonsReplyOp
  | SelectReplyOp
  | ModalReplyOp
  | ThreadActionReplyOp
  | SummaryMutationReplyOp
  | SpawnAgentReplyOp;

export interface TurnResponse {
  trace: TraceEnvelope;
  sessionKey: string;
  agent: AgentRef;
  replyOps: ReplyOp[];
  usage?: UsageStats;
  debugArtifacts?: DebugArtifactRef[];
  warnings?: string[];
}

export interface HealthResponse {
  status: "ok" | "degraded" | "error";
  version: string;
  uptimeSeconds: number;
}

export interface TurnStatusResponse {
  turnId: string;
  sessionKey?: string;
  state: "running" | "completed" | "failed" | "canceled";
  usage?: UsageStats;
  warnings?: string[];
}

export interface TraceQueryResponse {
  trace: TraceEnvelope;
  usage?: UsageStats;
  artifacts?: DebugArtifactRef[];
  warnings?: string[];
}
