# Hosted settings

The settings mock is implemented at `/settings` and the dashboard root. It uses
the existing dashboard server, session authentication and profile stores. Native
administration pages remain available at their existing URLs. There is no new
configuration database or chat surface.

| Page | Stored settings and behavior |
| --- | --- |
| Profile | Instructions edit native `SOUL.md`; timezone uses native `timezone`, retaining the server default when unset. The native conversation freezes SOUL contents. |
| Models | Native Codex device login, cancellation, disconnection and model assignment. One server-side login serves chat, images and Hindsight. Chat effort reads native effective reasoning and writes `agent.reasoning_overrides` for the selected model. Changes follow native next-turn resolution; explicit chat model overrides win. |
| Memory models | `hindsight.llm_model`, `llm_reasoning_effort`, `reflect_llm_reasoning_effort`. Defaults remain the reference Luna/low/medium policy. The private Hindsight supervisor applies edits automatically for its owning profile; secondary profiles show these controls read-only. |
| Access | Native pairing approval/revocation and Telegram group grants, mention/free-response rules, channel prompts and topic reply rules. Topics appear from native conversation discovery; people and groups can be added by numeric ID. |
| Service keys | Telegram, Browser Use, Parallel and OpenRouter. Verify before replacing a native profile secret; never return stored key values. One OpenRouter key supplies video, embeddings and reranking. |
| Admin login | Shared native Basic-auth account. Verify the old password, persist a salted hash and new session-signing secret, and invalidate existing sessions. Railway login variables bootstrap the account; saved rotations take precedence afterward. |

## Exact behavior differences

- Previously the native dashboard exposed these settings through several pages
  and a configuration editor. The new page edits the same owners through focused
  authenticated `/api/settings` routes, reusing native OAuth and model endpoints.
- Instructions edit native `SOUL.md` verbatim, without a separate name field or
  generated prefix. Native ephemeral overlays/personality remain separate. No
  core prompt assembly changes. Edits affect future conversations; warm prompts
  and history are untouched. Timezone edits clear the local timezone cache and
  request the native gateway restart so scheduling uses the new zone.
- Topics come from the native channel directory and existing topic rules. Group
  titles are discovered; unnamed topics use IDs. Available native topic names
  remain visible, including configured names. Reply-policy edits never create or rewrite native topic bindings.
- Telegram adds **chat-scoped silent topics** in `telegram.silent_topics`, as
  `chat_id:topic_id` values. They skip dispatch (including commands) and observation;
  a same-numbered topic in another group is unaffected. Other reply policies use
  native gates. The deployment seeds mention-by-default. When adopting that
  policy on an older profile, existing explicitly allowed groups retain their
  free-response behavior. Additional native access rules remain in force and
  are identified on the Access page.
- Service-key changes use native credential writers and request a gateway restart
  so standalone processes use the new credentials. Managed keys are rejected
  before verification or writing. Configuration and access edits also respect
  native installation and administrator locks, returning an error before mutation.
- Group instructions follow native `channel_prompts` behavior. Access edits
  request the native gateway restart; pairing-only grants apply immediately.
  Advanced native allowlists are updated on person edits and require restart.
- Hindsight formerly read every inference setting from its pinned deployment
  snapshot. Only the three memory-model settings and the OpenRouter credential
  are now editable. All remaining policy and managed-bank settings stay pinned.
  The supervisor polls the existing private inference service, restarts its
  children on changes, and acknowledges the applied revision and service health.
  A control-service outage preserves the running child. The public UI receives
  status, never the private control payload or credentials.

## Verification and limits

Router tests cover native stores, rejected writes, profile isolation, session
invalidation and Hindsight acknowledgements. Gateway tests exercise silent-topic
intake through the native config loader. Supervisor tests cover revision changes,
unchanged polls, outages and cleanup. UI tests cover saves, failed key checks,
topic edits and native model/effort writes.

A fresh Railway deployment still needs the infrastructure variables in the
[deployment instructions](../../deploy/railway/README.md). Account entitlement,
real Telegram delivery and deployed service restarts require live acceptance.
Normal setup afterward uses this UI; infrastructure provisioning stays in Railway.

Key checks use [Telegram getMe](https://core.telegram.org/bots/api#getme),
[Browser Use account billing](https://github.com/browser-use/browser-use/blob/main/CLOUD.md),
[OpenRouter current key](https://openrouter.ai/docs/api/api-reference/api-keys/get-current-key),
and one [Parallel search](https://docs.parallel.ai/api-reference/search/search).
