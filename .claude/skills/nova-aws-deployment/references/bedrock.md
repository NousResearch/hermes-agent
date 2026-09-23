# Bedrock

## Contents
1. Configuration model  2. Model vs profile vs application profile  3. IAM
4. Failure taxonomy  5. Diagnosis procedure  6. Incident: "Operation not allowed"

## 1. Configuration model
Model choice is deployment config, never application code. Minimum schema:

```yaml
model:
  provider: bedrock
  region: eu-west-2                  # region the client calls
  model_id: eu.anthropic.claude-sonnet-4-6   # model ID OR inference-profile ID/ARN
  kind: system_inference_profile     # foundation_model | system_inference_profile | application_inference_profile
  streaming: true
  max_tokens: 4096
  temperature: 0.2                   # only if the model supports it
  fallback:                          # optional, explicit, tested
    model_id: <other id>
```
The runtime reads this at boot; `nova model verify` validates it before workers start.

## 2. Three kinds of target
- **Foundation model** — `anthropic.claude-sonnet-4-6`; ARN `arn:aws:bedrock:<region>::foundation-model/<id>` (no account).
- **System-defined (cross-region) inference profile** — `eu.anthropic.claude-sonnet-4-6`; ARN
  `arn:aws:bedrock:<region>:<account>:inference-profile/<id>`. It routes each request to one of
  several destination regions defined by AWS.
- **Application inference profile** — customer-created, ARN
  `arn:aws:bedrock:<region>:<account>:application-inference-profile/<random-id>`; wraps a model or
  system profile for cost tracking.

Destination regions are **discovered**, never guessed:
```bash
aws bedrock get-inference-profile --inference-profile-identifier eu.anthropic.claude-sonnet-4-6 \
  --region eu-west-2 --query 'models[].modelArn'
```

## 3. IAM for a cross-region profile
Invocation through a profile is authorized against the profile **and** the foundation model in
whichever region it routes to. Grant both, scoped:

```json
{
  "Effect": "Allow",
  "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
  "Resource": [
    "arn:aws:bedrock:eu-west-2:<account>:inference-profile/eu.anthropic.claude-sonnet-4-6",
    "arn:aws:bedrock:*::foundation-model/anthropic.claude-sonnet-4-6"
  ]
}
```
Better than `*` region for the model: generate one ARN per region returned by
`get-inference-profile`. Converse/ConverseStream are authorized by the InvokeModel actions.
Never use `Resource: "*"` to clear an error; if documentation genuinely requires it for a
specific action, record the doc link and the security review in the journal.

## 4. Failure taxonomy (classify before changing anything)
`scripts/classify_bedrock_error.py` implements this.

| Cat | Name | Typical signal | Owner |
|---|---|---|---|
| A | IAM AccessDenied | `AccessDeniedException ... not authorized to perform: bedrock:Invoke...` | NOVA IAM |
| B | Model unavailable | `ResourceNotFoundException`, model not in region, legacy/EOL | config |
| C | Profile unavailable | invalid profile id / "on-demand throughput isn't supported" → must use profile | config |
| D | Account/model authorization | `Operation not allowed`, console "account is not authorized", model access not granted, use-case form not submitted | AWS account admin / support |
| E | Region mismatch | profile prefix (eu/us/apac) vs client region | config |
| F | Request schema | `ValidationException` naming a field (messages, max_tokens, roles) | NOVA code |
| G | Quota/throttling | `ThrottlingException`, `ServiceQuotaExceededException` | quotas / backoff |
| H | Provider restriction | provider-specific terms / geography | account admin |
| I | Service-side | `ModelErrorException`, `ServiceUnavailable`, 5xx | retry w/ backoff, AWS health |

Never retry a failed call blindly. Retries are only for G and I, with backoff and a cap.

## 5. Diagnosis procedure — three vantage points
Run the **same minimal Converse request** from:
1. the runtime role on the instance (via SSM) — `scripts/bedrock_probe.sh`
2. independent local credentials — same script
3. the Bedrock console playground

| Runtime | Local | Console | Conclusion |
|---|---|---|---|
| fail | ok | ok | NOVA: role policy, instance profile, VPC endpoint policy, config |
| fail | fail | ok | request/config shared by both (region, model id, schema) |
| fail | fail | fail | **account authorization** — stop changing NOVA, escalate |
| ok | ok | ok | Bedrock fine; problem is in worker/profile/gateway path |

Only after the direct probe passes should you test through NOVA (`nova model verify`, then a real task).

## 6. Incident: "Operation not allowed" (test environment)
Facts established:
- `eu.anthropic.claude-sonnet-4-6` profile ACTIVE; `anthropic.claude-sonnet-4-6` ACTIVE.
- Runtime policy grants InvokeModel/InvokeModelWithResponseStream on the profile ARN and
  `arn:aws:bedrock:*::foundation-model/anthropic.claude-sonnet-4-6`.
- Converse → `ValidationException: Operation not allowed` from runtime role **and** local creds.
- Console: "Your account is not authorized to perform this action."

Classification: **D**. The Terraform IAM change is correct and stays.
Next steps belong to the account owner: confirm model access for Anthropic models is granted in
the account/region (including any first-use / use-case details Anthropic models require), check
for Organization SCPs denying Bedrock or the destination regions, and open an AWS Support case
with request IDs from the failing calls. Treat specific root causes as hypotheses until AWS confirms.

**Lesson → automation:** `nova preflight` must run a real minimal Converse call against the
configured target from the deployment credentials *before* Terraform apply, and classify failures
with the taxonomy above. Category D fails preflight with an account-owner action, not a retry.
