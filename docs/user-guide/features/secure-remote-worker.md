# Secure Remote Worker

`hermes secure-worker` preserves useful remote-model coding workflows without placing the real
repository, host credentials, or arbitrary network access inside the model-controlled task
environment.

## What The Boundary Does

- Copies only explicitly named, tracked UTF-8 files from a clean Git worktree.
- Rejects secrets, binaries, symlinks, path escapes, untracked work, and size-limit violations.
- Binds the disposable Git workspace to a canonical manifest stored outside the workspace.
- Requires Docker, networking off, nonpersistent state, no forwarded environment variables, no
  extra volumes, no extra Docker arguments, and a digest-pinned admitted image.
- Suppresses Hermes' automatic credential, skill, cache, and egress-proxy mounts.
- Rejects cloud fallbacks from repo-capable local profiles.
- Gives Ox GitHub issues, branches, commits, pull requests, and reviews only through one
  configured staging repository.

Remote inference still receives the task prompt and sanitized files. Privacy Mode and
`data_collection: deny` reduce provider-side retention; they do not make a remote model local.

## Prerequisites

1. A clean isolated source worktree.
2. A reviewed worker image built without the source repository mounted. Record the immutable image
   digest and the SHA-256 digests of its Dockerfile and locked requirements in an image-lock JSON
   matching `examples/secure-worker/worker-image.lock.example.json`. A private local build may use
   its immutable `sha256:...` image ID directly; no registry push is required.
3. A dedicated private GitHub staging repository containing no production-repository history or
   secrets.
4. A fine-grained token limited to that repository and only the contents, issues, pull requests,
   and metadata permissions the broker needs. Store it only as
   `HERMES_STAGING_GITHUB_TOKEN`; the generated MCP configuration passes only that explicitly
   named credential, and the broker ignores `GH_TOKEN` and `GITHUB_TOKEN`. The token value is
   never written into the profile.
   Classic `ghp_` tokens are rejected. Before MCP starts, the broker verifies that GitHub reports
   the exact configured repository, private visibility, write access, and no admin/maintain access.
5. Docker/Colima running. If Docker is unavailable, preflight denies the run and never uses the
   host terminal.

## Operator Flow

Build a pack from an explicit file list:

```bash
hermes secure-worker pack \
  --source /path/to/clean-isolated-worktree \
  --file src/example.py \
  --file tests/test_example.py \
  --pack /private/tmp/secure-worker-pack \
  --manifest /private/tmp/secure-worker-pack.manifest.json \
  --policy examples/secure-worker/policy.json
```

After checking Privacy Mode in the Nous account, record a short-lived attestation:

```bash
hermes secure-worker attest \
  --confirm-privacy-mode \
  --ttl-minutes 60 \
  --output /private/tmp/nous-privacy-attestation.json
```

Render a new profile to a new file. This refuses to overwrite an existing profile:

```bash
hermes secure-worker profile-render \
  --kind ox-sanitized \
  --cwd /private/tmp/secure-worker-pack \
  --worker-image 'registry/worker@sha256:ACTUAL_DIGEST' \
  --broker-python /absolute/path/to/pinned/python \
  --staging-owner STAGING_OWNER \
  --staging-repo STAGING_REPOSITORY \
  --output /private/tmp/ox-sanitized.config.yaml
```

Preflight before inference:

```bash
hermes secure-worker audit \
  --config /private/tmp/ox-sanitized.config.yaml \
  --pack /private/tmp/secure-worker-pack \
  --manifest /private/tmp/secure-worker-pack.manifest.json \
  --policy examples/secure-worker/policy.json \
  --attestation /private/tmp/nous-privacy-attestation.json \
  --image-lock /path/to/worker-image.lock.json \
  --receipt /private/tmp/secure-worker-admission.json
```

Launch only through the receipt-consuming gate. It re-verifies the exact effective configuration,
full policy, manifest, image, endpoint, broker executable, and broker module immediately before
constructing the Hermes process:

```bash
hermes secure-worker run \
  --config /private/tmp/ox-sanitized.config.yaml \
  --pack /private/tmp/secure-worker-pack \
  --manifest /private/tmp/secure-worker-pack.manifest.json \
  --policy examples/secure-worker/policy.json \
  --attestation /private/tmp/nous-privacy-attestation.json \
  --image-lock /path/to/worker-image.lock.json \
  --receipt /private/tmp/secure-worker-admission.json \
  -- "review the admitted proposal"
```

Verify the working-tree proposal before any trusted promotion:

```bash
hermes secure-worker verify \
  --pack /private/tmp/secure-worker-pack \
  --manifest /private/tmp/secure-worker-pack.manifest.json \
  --policy examples/secure-worker/policy.json
```

The verifier does not apply, commit, push, merge, or publish anything to the production repository.
A trusted local operator reviews and promotes an approved patch separately.

Destroy the pack using its external manifest binding and retain only a content-free receipt:

```bash
hermes secure-worker destroy \
  --pack /private/tmp/secure-worker-pack \
  --manifest /private/tmp/secure-worker-pack.manifest.json \
  --receipt /private/tmp/secure-worker-destroy-receipt.json \
  --quarantine /private/tmp/secure-worker-quarantine
```

## Local Profiles

Use `profile-render --kind local-safe` for repository-capable Ollama work. The rendered profile has
no web toolset and no cloud fallback. Moving to OpenAI, Anthropic, Nous, or another remote provider
is a separate sanitized-profile transition, not an automatic fallback.

## Rollback

Do not overwrite profiles in place. Keep the prior profile directory and gateway process until the
sanitized canary passes. Rollback stops only the new profile/gateway, restores the prior named
profile, and leaves the production repository untouched. Never reuse a quarantined pack.
