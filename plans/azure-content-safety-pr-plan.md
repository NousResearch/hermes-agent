# Planned PR: feat(azure-foundry): Azure AI Content Safety guardrails + parity polish

Status: **scaffold only** — no provider code in this commit. This file
documents the planned scope so the branch is reservable and reviewable
before implementation begins.

## Why this scope

PR #15845 (teknium1, MERGED) shipped the Azure Foundry provider but stopped
short of the parity items the AWS Bedrock PR #10549 set as the bar:

| Bedrock PR #10549                          | Azure Foundry (#15845)             | Gap |
|--------------------------------------------|------------------------------------|-----|
| AWS Guardrails wired into config           | none                               | ✅ this PR |
| `hermes doctor` provider check             | none                               | ✅ this PR |
| `hermes auth` provider status              | none                               | ✅ this PR |
| `pip install hermes-agent[azure]` extra    | none                               | ✅ this PR |
| `/usage` pricing entries (7 models)        | none                               | ✅ this PR |
| Error-classifier patterns                  | none                               | ✅ this PR |

## Touchpoints (mirrors the 6-touchpoint pattern Bedrock used)

1. **`agent/azure_content_safety.py`** (new, ~250 LOC) — thin client over
   Azure AI Content Safety REST API (`/contentsafety/text:analyze`,
   `/contentsafety/text:shieldPrompt`, jailbreak / prompt-shield). Async,
   stateless, lazy-imported.
2. **`agent/error_classifier.py`** — add patterns for
   `ResponsibleAIPolicyViolation`, `content_filter`, Azure throttling
   (`429 Retry-After`), and Azure Foundry `model_not_found` /
   `DeploymentNotFound`.
3. **`agent/usage_pricing.py`** — pricing entries for the Azure Foundry
   catalog the wizard prefills today (gpt-5, gpt-5-mini, gpt-5-codex,
   gpt-4.1, gpt-4.1-mini, o3, o4-mini). Use Azure's published per-1M
   token list price; fall back to OpenAI list price tagged with
   `source: azure-foundry-list`.
4. **`hermes_cli/runtime_provider.py::_resolve_azure_foundry_runtime`** —
   read optional `guardrails:` block (mirrors Bedrock guardrails block);
   surface `content_safety_endpoint`, `content_safety_key_env`,
   `prompt_shield: true|false`, `block_categories: [...]`.
5. **`hermes_cli/doctor.py`** — Azure Foundry section: probe
   `AZURE_FOUNDRY_BASE_URL` reachability, `GET /models` count,
   credential source, optional Content Safety endpoint reachability.
   Reuses `hermes_cli/azure_detect.py::detect()`.
6. **`hermes_cli/auth_commands.py`** — Azure Foundry status row: shows
   resource hostname, key source (env / dotenv / config), Content Safety
   resource name if configured.
7. **`pyproject.toml`** — `azure = ["azure-ai-contentsafety>=1.0.0,<2",
   "azure-core>=1.30,<2"]` optional extra; documented as **optional** —
   provider works without it, only guardrails require it.
8. **Docs** — extend `website/docs/guides/azure-foundry.md` with a
   "Content Safety guardrails" section + env-vars page entry
   (`AZURE_CONTENT_SAFETY_ENDPOINT`, `AZURE_CONTENT_SAFETY_KEY`).

## Tests (targeting Bedrock-PR parity: ~80–130)

- `tests/agent/test_azure_content_safety.py` — analyze/shield request
  shape, retry on 429, lazy-import guard, redacts key from error logs
  (~25 tests).
- `tests/agent/test_error_classifier_azure.py` — content-filter and
  Foundry-specific error mapping (~10 tests).
- `tests/agent/test_usage_pricing_azure.py` — list-price entries
  resolved for both deployment-name and canonical-model paths (~8 tests).
- `tests/hermes_cli/test_doctor_azure.py` — reachability probe, missing
  key, missing Content Safety endpoint (~10 tests).
- `tests/hermes_cli/test_auth_commands_azure.py` — env/dotenv/config
  source detection (~6 tests).
- `tests/hermes_cli/test_runtime_provider_resolution.py` — extend with
  guardrails-block resolution cases (~12 tests).

## Defensibility / non-conflict

- Existing open PRs touching Azure (#22358, #16548, #20450, #19370,
  #17829, #20929) are all **bug fixes inside files #15845 already
  shipped**. None of them add new touchpoints; this PR adds the
  Bedrock-parity surface area they don't touch.
- `azure_content_safety.py` is a brand-new file (no merge conflicts
  with any in-flight PR).
- `pyproject.toml` extras are append-only.
- Pricing additions are append-only in `usage_pricing.py`.
- `doctor.py` / `auth_commands.py` Azure sections are new blocks
  appended after the Bedrock blocks — same pattern Bedrock followed.

## Estimated diff

- New files: ~600 LOC (adapter + 5 test files + doc section)
- Edits: ~150 LOC across 6 existing files
- Total: **~750 / -0**, ~71 tests

## Out of scope (intentional)

- ARM-based deployment enumeration (covered in #15845 commentary).
- Entra ID / managed-identity auth — separate follow-up; this PR keeps
  the API-key path #15845 already validated.
- Anthropic-on-Foundry routing — already shipped in #15845 via the
  salvaged commits from #4599.
