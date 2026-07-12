# Hermes Lanyard release artifact

LanyardBrain-specific path to publish a **runtime tarball** for company-brain
hosts. This is separate from upstream Nous workflows (`docker.yml` → GHCR,
`upload_to_pypi.yml` → PyPI).

## Contract

| Field | Value |
|-------|--------|
| S3 object shape | `s3://$ARTIFACT_BUCKET/hermes/<version>/hermes-<version>.tar.gz` |
| Companion checksum | `…/hermes-<version>.sha256` (sha256sum format) |
| Install path (bootstrap) | `/opt/hermes-lanyard` |
| Layout | Single top-level dir `hermes-<version>/` flattened on extract; `bin/hermes` executable |
| Versions | Immutable labels only (`v0.1.0`, CalVer tags, etc.) — **never** `main` / `master` / `HEAD` / `latest` |
| OIDC role key | `hermes` → IAM role name `${name_prefix}-gha-hermes` |
| Default trust | `repo:LanyardBrain/hermes-agent:ref:refs/tags/*` |

Deploy-owned infra and pin validation live in **company-brain-deploy**
(`modules/product-artifact-infra`, `artifacts/README.md`,
`references/product-artifact-release-contract.md`). This repo only **builds and
publishes** the object.

## Local pack (no AWS)

```bash
# Full pack (needs uv, network for deps; npm optional unless you skip frontend)
./scripts/pack-lanyard-runtime.sh --version v0.0.0-test

# Faster smoke (skip web/TUI rebuild)
./scripts/pack-lanyard-runtime.sh --version v0.0.0-test --skip-frontend

# Outputs
ls -la dist/lanyard/hermes-v0.0.0-test.tar.gz \
       dist/lanyard/hermes-v0.0.0-test.sha256
```

Verify extract layout:

```bash
tmpdir=$(mktemp -d)
tar -xzf dist/lanyard/hermes-v0.0.0-test.tar.gz -C "$tmpdir"
# Bootstrap flattens a single top-level directory:
# → expects $tmpdir/hermes-v0.0.0-test/bin/hermes (pre-flatten)
test -x "$tmpdir/hermes-v0.0.0-test/bin/hermes"
sha256sum -c dist/lanyard/hermes-v0.0.0-test.sha256
```

## GitHub Actions workflow

File: [`.github/workflows/release-lanyard-artifact.yml`](../.github/workflows/release-lanyard-artifact.yml)

| Trigger | Version source |
|---------|----------------|
| `push` tags `v*` | `github.ref_name` |
| `release` published | release tag name |
| `workflow_dispatch` | required `version` input (still rejects branch names) |

Dispatch inputs:

- `version` — immutable label
- `skip_upload` — pack only (no S3)
- `skip_frontend` — faster pack smoke

### Required GitHub configuration

Set on **LanyardBrain/hermes-agent** (repository **Variables** preferred for
non-secrets; secrets also accepted by the workflow):

| Name | Kind | Purpose |
|------|------|---------|
| `ARTIFACT_BUCKET` | variable | Platform artifact bucket name (no `s3://` prefix) |
| `AWS_ROLE_ARN` | variable or secret | OIDC role ARN for publisher (`…-gha-hermes`) |
| `AWS_REGION` | variable | AWS region (default `us-east-1` if unset) |

Do **not** commit account IDs, role ARNs, or bucket names into this repository.

OIDC role is created by company-brain-deploy `product-artifact-infra` when
`enable_oidc_publishers = true` and an existing GitHub OIDC provider ARN is
supplied. Until that apply is done, tag pushes will pack successfully but S3
upload will fail closed if vars are missing.

### Operator release process

1. Land code on the branch you release from (e.g. `main` or
   `lanyard/portal-integration`).
2. Create an **immutable** git tag and push it (or publish a GitHub Release):
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
3. Workflow packs the runtime, writes SHA-256, assumes the Hermes OIDC role,
   and uploads under `hermes/v0.1.0/`.
4. Copy pin fields into company-brain-deploy compatibility / tenant pins
   (**G7.H2** / deploy track — not this PR):

   | Pin | Example shape |
   |-----|----------------|
   | `hermes.version` / `hermes_artifact_version` | `v0.1.0` |
   | `hermes.url` / `hermes_artifact_url` | `s3://$ARTIFACT_BUCKET/hermes/v0.1.0/hermes-v0.1.0.tar.gz` |
   | `hermes.sha256` / `hermes_artifact_sha256` | 64-char hex from the `.sha256` object |

5. Bootstrap on the brain host downloads with the instance role, verifies
   SHA-256, extracts into `/opt/hermes-lanyard`, and symlinks
   `/usr/local/bin/hermes` when `bin/hermes` is present.

Optional Ed25519 attestation (`HERMES_ARTIFACT_SIGNATURE_B64`) is a deploy-side
concern; this workflow records SHA-256 only.

## Tarball layout (after extract + flatten)

```text
/opt/hermes-lanyard/
  bin/hermes              # relocatable wrapper
  .venv/                  # uv relocatable virtualenv + hermes-agent
  share/skills/           # bundled skills (HERMES_BUNDLED_SKILLS)
  share/optional-skills/
  VERSION
  .install_method         # "lanyard-artifact"
  .seed/build.env         # build metadata
```

## Explicit non-goals

- Updating company-brain-deploy pin files (later track)
- Replacing Docker/GHCR or PyPI publish paths
- Embedding live AWS account IDs, bucket names, or employee-vault content
