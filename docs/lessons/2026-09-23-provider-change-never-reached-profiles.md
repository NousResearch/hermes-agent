# A tenant model change never reached the agent profiles

**Date found:** 2026-09-23 · **Environment:** test (eu-west-2) · **Phase:** PROFILE_APPLY

## Symptom

`bundles/test/deployment.yaml` was changed from the foundation-model id
`anthropic.claude-sonnet-4-6` to the EU inference profile `eu.anthropic.claude-sonnet-4-6`,
and the bundle was unpacked on the instance. Every later apply — including the
`NOVA_APPLY_ON_START` one — reported:

```
tenant=test runtime=hermes created=0 changed=0 unchanged=2
```

and `profiles/*/config.yaml` still said `model: anthropic.claude-sonnet-4-6`. Workers then
called Bedrock with the bare model id (`errors.log`: `provider=bedrock
model=anthropic.claude-sonnet-4-6`), not the inference profile the bundle and IAM policy
declare. It was masked because Bedrock was refusing every call anyway ("Operation not
allowed", account not authorized), so it would only have surfaced after AWS granted access.

## Cause

Apply skips an agent whose recorded digest matches the expected one. The digest
(`nova.policy.agent_digest`) covered the agent spec, compiled policy and knowledge grant —
but not the tenant-level provider or `runtime_config` from `deployment.yaml`, which are
written into the same `config.yaml`. A deployment-only change therefore could not move the
digest, apply took the "unchanged" fast path, and the Control Center reported agents
`in_sync` while they ran the old model.

## Fix

`agent_digest` takes a `deployment` payload (resolved provider + `runtime_config`), passed by
the materializer, `HermesRuntime.expected_digest`, `nova/reconcile.py` and the Control API
drift check. Tenants that declare no deployment keep their existing digests. Regression
tests: `tests/platform/test_apply.py::test_a_tenant_model_change_reaches_every_agent` and
`::test_drift_check_agrees_with_apply_when_a_deployment_is_declared` (both red before the fix).

Rollout: new control-plane image → restart `nova.service`; apply-on-start will report
`changed=2` once and rewrite both profiles with the inference profile id. No `--prune` needed.

## Preflight check to build

`nova verify` level 7 ("profiles materialized == bundle profiles") must compare the
*effective* model/provider in each profile's `config.yaml` with `TenantBundle.provider_for()`,
not only the digest — a digest that omits an input can never catch that input's drift.
