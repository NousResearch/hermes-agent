# Railway deployment and service selection

Status: service selections agreed; topology proposed; integration validation
pending. No deployment made. See [the master specification](employee.md).

## Selected direction

Deploy this Hermes fork on Railway, run Hindsight privately in the same project,
and use Browser Use Cloud through direct credentials. “Local Hindsight” means
self-hosted alongside Hermes, not Hindsight Cloud and not embedded in the agent
process. This supersedes the earlier assumption that the primary deployment is
on the user's laptop; native Hermes remains the runtime.

Use the native Hermes administration dashboard rather than porting the hosted
employee dashboard. Fixed employee behavior is enforced in code; operational
defaults and deployment credentials use native configuration. The earlier
proposal for a custom dashboard and strictly two-folder filesystem sandbox was
withdrawn. Folder conventions remain; they are not an OS security boundary.

## Proposed topology

- Hermes service: native gateway/scheduler, dashboard and terminal execution;
  persistent volume for profile state and working files. One active instance
  initially. Use native container lifecycle support, not systemd assumptions.
- Hindsight service: private API and worker; stable worker identity and pinned
  image. Hermes connects via Railway private DNS. Preserve the agreed employee
  memory integration and model-facing recall tool.
- PostgreSQL service: persistent storage with a supported vector extension for
  Hindsight. Keep native Hermes session storage; this is not a migration of all
  Hermes state to PostgreSQL.
- Browser Use Cloud: native `browser_exec` plus direct cloud provider, no Nous
  gateway. A durable cloud profile per employee is proposed to preserve logins;
  verify current native provisioning/configuration support before promising it.

Persist CLI login files, knowledge, artifacts and sessions on the Hermes volume;
container image files alone do not survive deployment. Plan database and file
backups separately. Public routing for the dashboard and webhook endpoint,
including appropriate dashboard authentication, requires an explicit deployment
configuration. Hindsight and PostgreSQL need no public endpoints.

## Provider selection

| Capability | Selected direction / remaining verification |
| --- | --- |
| Main model | Native Codex login; selected main models use that authentication. No paid API fallback without a separate decision. |
| Hindsight extraction/reflection | Preserve the reference Luna model and settings; authenticate through Codex login. Verify model availability and protocol compatibility before declaring this route ready. |
| Hindsight embeddings/reranking | Preserve the reference OpenRouter Qwen embeddings and Cohere reranking configuration exactly. Codex login is not assumed to supply these endpoints. |
| Web search/extraction | Parallel selected for both, through the existing native plugin with direct credentials. No Nous gateway. |
| Transcription | Local faster-whisper selected. Use native local transcription; no external STT API. Deployment packaging must include its model/runtime. |
| Image generation | OpenAI image generation using Codex authentication selected. Implement and verify the supported execution path; no silent API-key fallback. Exact model/route compatibility remains to be verified. |
| Image understanding | Prefer main-model native vision when supported; auxiliary fallback still needs selection if required. |
| Video understanding | Gemini through OpenRouter selected; include `video_analyze`. Use the reference configuration, verifying model availability before pinning. |
| Messaging | Telegram via native adapter and bot token; other channels remain optional. |

The main model and Hindsight's LLM are separate settings. Self-hosting Hindsight
does not imply self-hosting its inference: extraction/reflection still needs an
LLM, and the selected embeddings/reranking run through OpenRouter.

## Configuration ownership

Choose one authoritative source per credential. Railway variables are suitable
for deployment/service credentials; native dashboard-managed profile secrets
are suitable for later user-managed integrations. Do not advertise dashboard
key rotation while a higher-precedence Railway variable silently overrides it.
The exact initial provisioning flow remains to be specified.

## Evidence checked

- Railway services, volumes and private networking:
  https://docs.railway.com/services and
  https://docs.railway.com/networking/private-networking
- Hindsight production database requirements and full/slim images:
  https://hindsight.vectorize.io/developer/installation
- Browser Use direct API authentication:
  https://docs.browser-use.com/cloud/api-reference
- This checkout's `plugin-catalog/hindsight.yaml` notes external endpoint support
  and an embedded-install limitation; use the external-service integration.
- Native browser provider: `plugins/browser/browser_use/provider.py`.

Reference provider choices were inspected in the local employee repository's
runtime manifest and Hindsight infrastructure definition. They are evidence of
that checkout's configuration, not verification of current account entitlement
or model availability. Those checks precede final model pins.

## Hindsight configuration fidelity

Decision: copy the existing employee Hindsight configuration 1:1, including its
pinned image, bank missions/template, extraction and observation settings,
retention/recall configuration, embedding dimensions, reranker, reasoning effort,
and other explicit operational settings. Do not replace it with Hindsight
upstream defaults or an independently tuned slim deployment. The [reference snapshot](reference/hindsight-config.json) preserves the image,
explicit API environment and bank template, including pool sizes resolved from
the reference production configuration. Port bank reconciliation too; check
remaining launcher/version defaults against the pinned image during implementation.

The inspected reference selects `gpt-5.6-luna`, low LLM reasoning effort and
medium reflect effort; Qwen `qwen/qwen3-embedding-8b` at 1536 dimensions; Cohere
`cohere/rerank-4-fast`; pgvector; native text search; and exponential recency
decay with a 60-day half-life. These examples are not the complete config.

Necessary deployment substitutions are Railway addresses, storage/secrets and
stable worker identity. The explicitly requested LLM authentication substitution
is Codex login. Keep these differences visible; do not describe byte-identical
configuration where endpoints/authentication have changed.

The source Hindsight service currently sends OpenAI Responses through a hosted
usage gateway backed by API credentials, not Codex login. Native Hermes has Codex
login/token refresh, but that does not automatically give Hindsight a compatible
inference endpoint. Trace and implement the necessary protocol/auth integration,
including token refresh ownership and shared usage limits, without importing the
hosted metering stack. No model downgrade or paid-API fallback is implicit.
Qwen embeddings and Cohere reranking through OpenRouter are explicitly confirmed;
they retain their source configuration and separate credentials. Implementing
Codex subscription authentication for Hindsight is explicitly in scope for this
fork, subject to end-to-end compatibility verification.
Image generation is also explicitly selected to use Codex authentication. Its
execution route must be verified independently of text-model authentication;
a working Codex text session is not proof that the image-generation tool works.
Video understanding uses Gemini through OpenRouter, as explicitly agreed.
