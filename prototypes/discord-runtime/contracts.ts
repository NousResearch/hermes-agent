import { Schema } from "effect"

const StringMapSchema = Schema.Record({
  key: Schema.String,
  value: Schema.String,
})

export const SurfaceKindSchema = Schema.Literal(
  "channel",
  "thread",
  "forum_post",
)
export type SurfaceKind = Schema.Schema.Type<typeof SurfaceKindSchema>

export const IdentityModeSchema = Schema.Literal(
  "distinct_actor",
  "shared_persona",
)
export type IdentityMode = Schema.Schema.Type<typeof IdentityModeSchema>

export const DeliveryModeSchema = Schema.Literal(
  "send",
  "edit",
  "interaction_reply",
  "interaction_followup",
)
export type DeliveryMode = Schema.Schema.Type<typeof DeliveryModeSchema>

export const SummaryScopeSchema = Schema.Literal("local", "parent", "server")
export type SummaryScope = Schema.Schema.Type<typeof SummaryScopeSchema>

export const EdgeReasonSchema = Schema.Literal(
  "surface_branch",
  "subagent_spawn",
  "manual_handoff",
)
export type EdgeReason = Schema.Schema.Type<typeof EdgeReasonSchema>

export const InteractionTypeSchema = Schema.Literal(
  "slash_command",
  "button",
  "select",
  "modal",
)
export type InteractionType = Schema.Schema.Type<typeof InteractionTypeSchema>

export const DiscrawlModeSchema = Schema.Literal(
  "disabled",
  "ops_only",
  "on_demand",
  "background",
)
export type DiscrawlMode = Schema.Schema.Type<typeof DiscrawlModeSchema>

export const DebugArtifactKindSchema = Schema.Literal(
  "prompt_snapshot",
  "tool_timeline",
  "message_trace",
  "summary_trace",
  "raw_backend_response",
)
export type DebugArtifactKind = Schema.Schema.Type<
  typeof DebugArtifactKindSchema
>

export const DiscordSurfaceRefSchema = Schema.Struct({
  serverId: Schema.String,
  surfaceId: Schema.String,
  surfaceKind: SurfaceKindSchema,
  channelId: Schema.String,
  threadId: Schema.optionalWith(Schema.String, { exact: true }),
  parentSurfaceId: Schema.optionalWith(Schema.String, { exact: true }),
  parentChannelId: Schema.optionalWith(Schema.String, { exact: true }),
  forumChannelId: Schema.optionalWith(Schema.String, { exact: true }),
  name: Schema.optionalWith(Schema.String, { exact: true }),
})
export type DiscordSurfaceRef = Schema.Schema.Type<
  typeof DiscordSurfaceRefSchema
>

export const AgentRefSchema = Schema.Struct({
  agentId: Schema.String,
  displayName: Schema.String,
  identityMode: IdentityModeSchema,
})
export type AgentRef = Schema.Schema.Type<typeof AgentRefSchema>

export const ActorRefSchema = Schema.Struct({
  userId: Schema.String,
  username: Schema.optionalWith(Schema.String, { exact: true }),
  displayName: Schema.optionalWith(Schema.String, { exact: true }),
  roleIds: Schema.optionalWith(Schema.Array(Schema.String), { exact: true }),
})
export type ActorRef = Schema.Schema.Type<typeof ActorRefSchema>

export const MessageReferenceSchema = Schema.Struct({
  messageId: Schema.String,
  authorLabel: Schema.optionalWith(Schema.String, { exact: true }),
  excerpt: Schema.String,
  surfaceId: Schema.optionalWith(Schema.String, { exact: true }),
  timestamp: Schema.optionalWith(Schema.String, { exact: true }),
})
export type MessageReference = Schema.Schema.Type<typeof MessageReferenceSchema>

export const AttachmentKindSchema = Schema.Literal(
  "image",
  "audio",
  "video",
  "file",
)
export type AttachmentKind = Schema.Schema.Type<typeof AttachmentKindSchema>

export const AttachmentRefSchema = Schema.Struct({
  attachmentId: Schema.String,
  kind: AttachmentKindSchema,
  mimeType: Schema.optionalWith(Schema.String, { exact: true }),
  filename: Schema.optionalWith(Schema.String, { exact: true }),
  localPath: Schema.optionalWith(Schema.String, { exact: true }),
  sourceUrl: Schema.optionalWith(Schema.String, { exact: true }),
  sizeBytes: Schema.optionalWith(Schema.Number, { exact: true }),
})
export type AttachmentRef = Schema.Schema.Type<typeof AttachmentRefSchema>

export const SummaryRefSchema = Schema.Struct({
  summaryId: Schema.String,
  scope: SummaryScopeSchema,
  sourceSessionKey: Schema.optionalWith(Schema.String, { exact: true }),
  text: Schema.String,
  createdAt: Schema.String,
})
export type SummaryRef = Schema.Schema.Type<typeof SummaryRefSchema>

export const LineageRefSchema = Schema.Struct({
  parentSessionKey: Schema.optionalWith(Schema.String, { exact: true }),
  parentSurfaceId: Schema.optionalWith(Schema.String, { exact: true }),
  rootSessionKey: Schema.optionalWith(Schema.String, { exact: true }),
  edgeReason: Schema.optionalWith(EdgeReasonSchema, { exact: true }),
  handoffSummary: Schema.optionalWith(SummaryRefSchema, { exact: true }),
})
export type LineageRef = Schema.Schema.Type<typeof LineageRefSchema>

export const InteractionInputSchema = Schema.Struct({
  type: InteractionTypeSchema,
  interactionId: Schema.String,
  commandName: Schema.optionalWith(Schema.String, { exact: true }),
  customId: Schema.optionalWith(Schema.String, { exact: true }),
  values: Schema.optionalWith(Schema.Array(Schema.String), { exact: true }),
  fields: Schema.optionalWith(StringMapSchema, { exact: true }),
})
export type InteractionInput = Schema.Schema.Type<typeof InteractionInputSchema>

export const HumanInputSchema = Schema.Struct({
  text: Schema.String,
  messageId: Schema.optionalWith(Schema.String, { exact: true }),
  replyTo: Schema.optionalWith(MessageReferenceSchema, { exact: true }),
  attachments: Schema.optionalWith(Schema.Array(AttachmentRefSchema), {
    exact: true,
  }),
  interaction: Schema.optionalWith(InteractionInputSchema, { exact: true }),
})
export type HumanInput = Schema.Schema.Type<typeof HumanInputSchema>

export const TunableContextPolicySchema = Schema.Struct({
  localMessageBudget: Schema.Number,
  localTokenBudget: Schema.Number,
  parentSummaryBudget: Schema.Number,
  serverSummaryBudget: Schema.Number,
  maxReplyExcerptChars: Schema.Number,
  allowServerSummaryAwareness: Schema.Boolean,
  discrawlMode: DiscrawlModeSchema,
})
export type TunableContextPolicy = Schema.Schema.Type<
  typeof TunableContextPolicySchema
>

export const TraceEnvelopeSchema = Schema.Struct({
  traceId: Schema.String,
  turnId: Schema.String,
  parentSpanId: Schema.optionalWith(Schema.String, { exact: true }),
  startedAt: Schema.String,
  tags: Schema.optionalWith(StringMapSchema, { exact: true }),
})
export type TraceEnvelope = Schema.Schema.Type<typeof TraceEnvelopeSchema>

export const TurnRequestSchema = Schema.Struct({
  trace: TraceEnvelopeSchema,
  agent: AgentRefSchema,
  surface: DiscordSurfaceRefSchema,
  sessionKey: Schema.String,
  actor: ActorRefSchema,
  input: HumanInputSchema,
  lineage: Schema.optionalWith(LineageRefSchema, { exact: true }),
  localSummaries: Schema.optionalWith(Schema.Array(SummaryRefSchema), {
    exact: true,
  }),
  parentSummaries: Schema.optionalWith(Schema.Array(SummaryRefSchema), {
    exact: true,
  }),
  serverSummaries: Schema.optionalWith(Schema.Array(SummaryRefSchema), {
    exact: true,
  }),
  channelTopic: Schema.optionalWith(Schema.String, { exact: true }),
  policy: TunableContextPolicySchema,
  metadata: Schema.optionalWith(StringMapSchema, { exact: true }),
})
export type TurnRequest = Schema.Schema.Type<typeof TurnRequestSchema>

export const UsageStatsSchema = Schema.Struct({
  inputTokens: Schema.optionalWith(Schema.Number, { exact: true }),
  outputTokens: Schema.optionalWith(Schema.Number, { exact: true }),
  totalTokens: Schema.optionalWith(Schema.Number, { exact: true }),
  model: Schema.optionalWith(Schema.String, { exact: true }),
  provider: Schema.optionalWith(Schema.String, { exact: true }),
})
export type UsageStats = Schema.Schema.Type<typeof UsageStatsSchema>

export const DebugArtifactRefSchema = Schema.Struct({
  artifactId: Schema.String,
  kind: DebugArtifactKindSchema,
  uri: Schema.optionalWith(Schema.String, { exact: true }),
})
export type DebugArtifactRef = Schema.Schema.Type<typeof DebugArtifactRefSchema>

export const TextReplyOpSchema = Schema.Struct({
  op: Schema.Literal("text"),
  opId: Schema.String,
  delivery: DeliveryModeSchema,
  text: Schema.String,
  replyToMessageId: Schema.optionalWith(Schema.String, { exact: true }),
  targetSurfaceId: Schema.optionalWith(Schema.String, { exact: true }),
  ephemeral: Schema.optionalWith(Schema.Boolean, { exact: true }),
})
export type TextReplyOp = Schema.Schema.Type<typeof TextReplyOpSchema>

export const AttachmentReplyOpSchema = Schema.Struct({
  op: Schema.Literal("attachment"),
  opId: Schema.String,
  delivery: DeliveryModeSchema,
  attachments: Schema.Array(AttachmentRefSchema),
  caption: Schema.optionalWith(Schema.String, { exact: true }),
  replyToMessageId: Schema.optionalWith(Schema.String, { exact: true }),
  targetSurfaceId: Schema.optionalWith(Schema.String, { exact: true }),
})
export type AttachmentReplyOp = Schema.Schema.Type<
  typeof AttachmentReplyOpSchema
>

export const ButtonStyleSchema = Schema.Literal(
  "primary",
  "secondary",
  "success",
  "danger",
)
export type ButtonStyle = Schema.Schema.Type<typeof ButtonStyleSchema>

export const ButtonSpecSchema = Schema.Struct({
  actionId: Schema.String,
  label: Schema.String,
  style: Schema.optionalWith(ButtonStyleSchema, { exact: true }),
  value: Schema.optionalWith(Schema.String, { exact: true }),
})
export type ButtonSpec = Schema.Schema.Type<typeof ButtonSpecSchema>

export const SelectOptionSpecSchema = Schema.Struct({
  value: Schema.String,
  label: Schema.String,
  description: Schema.optionalWith(Schema.String, { exact: true }),
})
export type SelectOptionSpec = Schema.Schema.Type<typeof SelectOptionSpecSchema>

export const ButtonsReplyOpSchema = Schema.Struct({
  op: Schema.Literal("buttons"),
  opId: Schema.String,
  delivery: DeliveryModeSchema,
  text: Schema.optionalWith(Schema.String, { exact: true }),
  buttons: Schema.Array(ButtonSpecSchema),
  replyToMessageId: Schema.optionalWith(Schema.String, { exact: true }),
  targetSurfaceId: Schema.optionalWith(Schema.String, { exact: true }),
})
export type ButtonsReplyOp = Schema.Schema.Type<typeof ButtonsReplyOpSchema>

export const SelectReplyOpSchema = Schema.Struct({
  op: Schema.Literal("select"),
  opId: Schema.String,
  delivery: DeliveryModeSchema,
  text: Schema.optionalWith(Schema.String, { exact: true }),
  customId: Schema.String,
  placeholder: Schema.optionalWith(Schema.String, { exact: true }),
  options: Schema.Array(SelectOptionSpecSchema),
  minValues: Schema.optionalWith(Schema.Number, { exact: true }),
  maxValues: Schema.optionalWith(Schema.Number, { exact: true }),
  replyToMessageId: Schema.optionalWith(Schema.String, { exact: true }),
  targetSurfaceId: Schema.optionalWith(Schema.String, { exact: true }),
})
export type SelectReplyOp = Schema.Schema.Type<typeof SelectReplyOpSchema>

export const ModalFieldStyleSchema = Schema.Literal("short", "paragraph")
export type ModalFieldStyle = Schema.Schema.Type<typeof ModalFieldStyleSchema>

export const ModalFieldSpecSchema = Schema.Struct({
  fieldId: Schema.String,
  label: Schema.String,
  style: Schema.optionalWith(ModalFieldStyleSchema, { exact: true }),
  required: Schema.optionalWith(Schema.Boolean, { exact: true }),
  placeholder: Schema.optionalWith(Schema.String, { exact: true }),
  value: Schema.optionalWith(Schema.String, { exact: true }),
})
export type ModalFieldSpec = Schema.Schema.Type<typeof ModalFieldSpecSchema>

export const ModalReplyOpSchema = Schema.Struct({
  op: Schema.Literal("modal"),
  opId: Schema.String,
  delivery: DeliveryModeSchema,
  customId: Schema.String,
  title: Schema.String,
  submitLabel: Schema.optionalWith(Schema.String, { exact: true }),
  fields: Schema.Array(ModalFieldSpecSchema),
  triggerText: Schema.optionalWith(Schema.String, { exact: true }),
})
export type ModalReplyOp = Schema.Schema.Type<typeof ModalReplyOpSchema>

export const ThreadActionSchema = Schema.Literal(
  "spawn_child_surface",
  "handoff_to_child",
  "rebind_surface",
)
export type ThreadAction = Schema.Schema.Type<typeof ThreadActionSchema>

export const ThreadActionReplyOpSchema = Schema.Struct({
  op: Schema.Literal("thread_action"),
  opId: Schema.String,
  action: ThreadActionSchema,
  targetAgent: Schema.optionalWith(AgentRefSchema, { exact: true }),
  childSurfaceName: Schema.optionalWith(Schema.String, { exact: true }),
  handoffSummary: Schema.optionalWith(Schema.String, { exact: true }),
  parentSessionKey: Schema.optionalWith(Schema.String, { exact: true }),
})
export type ThreadActionReplyOp = Schema.Schema.Type<
  typeof ThreadActionReplyOpSchema
>

export const SummaryMutationActionSchema = Schema.Literal("append", "replace")
export type SummaryMutationAction = Schema.Schema.Type<
  typeof SummaryMutationActionSchema
>

export const SummaryMutationReplyOpSchema = Schema.Struct({
  op: Schema.Literal("summary_mutation"),
  opId: Schema.String,
  scope: SummaryScopeSchema,
  action: SummaryMutationActionSchema,
  text: Schema.String,
  sourceSessionKey: Schema.optionalWith(Schema.String, { exact: true }),
})
export type SummaryMutationReplyOp = Schema.Schema.Type<
  typeof SummaryMutationReplyOpSchema
>

export const PreferredSurfaceSchema = Schema.Literal("thread", "forum_post")
export type PreferredSurface = Schema.Schema.Type<typeof PreferredSurfaceSchema>

export const SpawnAgentReplyOpSchema = Schema.Struct({
  op: Schema.Literal("spawn_agent"),
  opId: Schema.String,
  agent: AgentRefSchema,
  reason: Schema.optionalWith(Schema.String, { exact: true }),
  preferredSurface: Schema.optionalWith(PreferredSurfaceSchema, {
    exact: true,
  }),
  handoffSummary: Schema.optionalWith(Schema.String, { exact: true }),
})
export type SpawnAgentReplyOp = Schema.Schema.Type<
  typeof SpawnAgentReplyOpSchema
>

export const ReplyOpSchema = Schema.Union(
  TextReplyOpSchema,
  AttachmentReplyOpSchema,
  ButtonsReplyOpSchema,
  SelectReplyOpSchema,
  ModalReplyOpSchema,
  ThreadActionReplyOpSchema,
  SummaryMutationReplyOpSchema,
  SpawnAgentReplyOpSchema,
)
export type ReplyOp = Schema.Schema.Type<typeof ReplyOpSchema>

export const TurnResponseSchema = Schema.Struct({
  trace: TraceEnvelopeSchema,
  sessionKey: Schema.String,
  agent: AgentRefSchema,
  replyOps: Schema.Array(ReplyOpSchema),
  usage: Schema.optionalWith(UsageStatsSchema, { exact: true }),
  debugArtifacts: Schema.optionalWith(Schema.Array(DebugArtifactRefSchema), {
    exact: true,
  }),
  warnings: Schema.optionalWith(Schema.Array(Schema.String), { exact: true }),
})
export type TurnResponse = Schema.Schema.Type<typeof TurnResponseSchema>

export const HealthStatusSchema = Schema.Literal("ok", "degraded", "error")
export type HealthStatus = Schema.Schema.Type<typeof HealthStatusSchema>

export const HealthResponseSchema = Schema.Struct({
  status: HealthStatusSchema,
  version: Schema.String,
  uptimeSeconds: Schema.Number,
})
export type HealthResponse = Schema.Schema.Type<typeof HealthResponseSchema>

export const TurnStateSchema = Schema.Literal(
  "running",
  "completed",
  "failed",
  "canceled",
)
export type TurnState = Schema.Schema.Type<typeof TurnStateSchema>

export const TurnStatusResponseSchema = Schema.Struct({
  turnId: Schema.String,
  sessionKey: Schema.optionalWith(Schema.String, { exact: true }),
  state: TurnStateSchema,
  usage: Schema.optionalWith(UsageStatsSchema, { exact: true }),
  warnings: Schema.optionalWith(Schema.Array(Schema.String), { exact: true }),
})
export type TurnStatusResponse = Schema.Schema.Type<
  typeof TurnStatusResponseSchema
>

export const TraceQueryResponseSchema = Schema.Struct({
  trace: TraceEnvelopeSchema,
  usage: Schema.optionalWith(UsageStatsSchema, { exact: true }),
  artifacts: Schema.optionalWith(Schema.Array(DebugArtifactRefSchema), {
    exact: true,
  }),
  warnings: Schema.optionalWith(Schema.Array(Schema.String), { exact: true }),
})
export type TraceQueryResponse = Schema.Schema.Type<
  typeof TraceQueryResponseSchema
>
