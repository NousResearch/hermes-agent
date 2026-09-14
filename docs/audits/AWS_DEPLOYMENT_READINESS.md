# NOVA on AWS — deployment readiness

**Status:** the image exists, is built, and passes 49 local checks against the running
container. It is **not yet deployed**, and nothing has been pushed to any registry.

**Scope of this document.** It reports what was built and what was proven locally, and it
is explicit about the difference between the two kinds of statement it makes:

* **LOCAL VALIDATION** — observed, in a container, on this machine. Reproducible with
  `deploy/docker/validate-local.sh`.
* **FIELD VALIDATION** — not performed. No AWS account was touched. Anything that depends
  on ECR, EC2, IAM, KMS, CloudWatch or SSM behaving as configured is unproven here and is
  listed under *Remaining deployment requirements*.

No customer AWS account, account id, role ARN or credential appears anywhere in this work.
The example bundle ships AWS's documented placeholders (`111122223333`, `vpc-0123…`) and
nothing else.

---

## 1. Audit of what already existed

Done first, before anything was written.

| Artefact | What it actually was |
|---|---|
| `Dockerfile` (repo root) | Builds the **upstream Hermes runtime** — s6 supervision, SQLite 3.53.0400 compiled from source, the full toolchain. Not a NOVA image, and not what `deploy/aws` runs. |
| `deploy/aws/*.tf` | Complete: EC2, EBS, KMS, CloudWatch Logs, IAM runtime role, per-integration roles with `sts:ExternalId`, a security group with **no ingress rules**, IMDSv2 required. |
| `deploy/aws/user_data.sh.tftpl` | `docker pull "${image_uri}"` from ECR, then a systemd unit that runs it. |
| `deploy/aws/README.md` | *"**Bring your own image.** This module grants the instance permission to pull the image and does not build it."* |
| **A NOVA image** | **Did not exist.** The deployment pulled something that was never built. |

That gap is what this work closes. The Terraform was not redesigned; the image was written
to fit the contract the Terraform already states.

Two further audit findings, both verified rather than assumed:

* `pyproject.toml`'s `[tool.setuptools.packages.find]` does not include `nova`, so the image
  runs the application from source on `PYTHONPATH` rather than `pip install .`. Widening the
  packaging config would be an upstream change this deployment does not need.
* NOVA imports with only PyYAML present, and `hermes_cli.kanban_db` and `cron.jobs` do too —
  checked in an environment containing nothing else before the Dockerfile was written. That
  is what makes a small image possible: the runtime's model SDKs are never called by a
  control plane.

---

## 2. Image name

```
nova-control-plane
```

Built by `deploy/docker/build.sh` from `deploy/docker/Dockerfile.nova`. It is a **control
plane, not an agent worker**: it reads and governs runtime state. Executing an agent needs
the full Hermes runtime image and that tenant's provider credentials.

## 3. Image tag

Format `<version>-g<commit>`: the NOVA version from `nova/__init__.py`, then the commit the
tree was built from. A dirty working tree appends `-dirty.<hash>` covering both modified
tracked files and the contents of untracked ones.

The guarantee is one-directional and worth stating exactly: **a tag never names two
different trees.** The reverse does not hold — a commit that changes only documentation
produces a new tag for byte-identical image content, which is harmless. `build.sh
--print-tag` is authoritative for the tree you are on.

The build these results came from:

```
nova-control-plane:0.1.0-g2457731df55f
sha256:fd5756c2dc2d945ddfb2a6cbcfb1a32f9b68d90c4dc2a6701656aaf306033db1
```

A registry assigns its own digest on push; pin **that** one in `image_uri`.

Size: **304 MB**. `build.sh` also writes `nova-control-plane:local` for local work. That tag
is mutable and must not be deployed — `image_uri` should carry the immutable tag above, or
better, a digest.

## 4. Exposed ports

| Port | Protocol | Notes |
|---|---|---|
| `8787/tcp` | HTTP | The Control API and the Control Center. Declared with `EXPOSE`; **not published** by the `deploy/aws` systemd unit. |

The image binds `127.0.0.1` by default. Binding anything else is allowed but not made easy,
and deliberately so: `nova/control/server.py` refuses a non-loopback bind without a
principals file, and refuses it again without TLS (a certificate, or an explicit statement
that a proxy terminates TLS in front). Both refusals were **observed firing inside the
container**, not read in the source.

## 5. Required environment variables

**None are required for the image to start.** Every one has a default, and the defaults are
what the `deploy/aws` unit gets.

| Variable | Default | Purpose |
|---|---|---|
| `NOVA_BUNDLE` | `/var/lib/nova/bundle` | The tenant's declarations. **Must exist on the volume** — see remaining requirements. |
| `NOVA_HOME` | `/var/lib/nova/home` | Runtime and control-plane state. Read by `nova/runtime/hermes/paths.py`. |
| `HERMES_HOME` | mirrors `NOVA_HOME` | The documented alias chain, so a `hermes_cli` command run inside the container for debugging reads the state the control plane is serving. |
| `NOVA_BIND_HOST` | `127.0.0.1` | Anything else triggers the principals + TLS requirements above. |
| `NOVA_BIND_PORT` | `8787` | |
| `NOVA_PRINCIPALS` | `$NOVA_HOME/control-principals.yaml` | Who may call the control plane, as SHA-256 digests. NOVA never stores a token. |
| `NOVA_TLS_CERT` / `NOVA_TLS_KEY` | unset | TLS terminated by this process. |
| `NOVA_BEHIND_TLS_PROXY` | unset | Any non-empty value states a proxy terminates TLS in front. It also **disables loopback trust**, which is the correct behaviour and has a consequence for the health check (§9). |
| `NOVA_APPLY_ON_START` | unset | Any non-empty value runs `nova apply` before serving. |
| `NOVA_LOG_LEVEL` | unset | Read by `nova/observability.py`. |
| `NOVA_LOG_FORMAT` | `json` | Read by `nova/observability.py`. |

**A correction to what the Terraform implies.** `user_data.sh.tftpl` writes four variables
into `/etc/nova.env` and passes them with `--env-file`:

```
NOVA_TENANT_ID  NOVA_AWS_REGION  NOVA_INTEGRATION_EXTERNAL_ID  NOVA_INTEGRATIONS
```

**No code reads any of them** (`grep` across the repository returns nothing). They are
declarative. The tenant id the control plane actually uses comes from
`organization.yaml` in the bundle. They are harmless, and they are not configuration.

## 6. Required volumes

| Mount | Purpose |
|---|---|
| `/var/lib/nova` | Everything durable: the tenant bundle, the materialised agent profiles, the Kanban and cron databases, the NOVA audit log (`home/nova/audit.jsonl`), and the automation provenance registry (`home/nova/automations.json`). |

`deploy/aws` already mounts an encrypted XFS EBS volume here. Declared as `VOLUME` in the
image so an operator who forgets the flag gets an anonymous volume rather than a container
whose audit log dies with it.

State and provenance were confirmed to survive a container restart on the mounted volume.

## 7. Required AWS services

From `deploy/aws/*.tf` as it stands — not a proposal, an inventory:

| Service | Resource | Why |
|---|---|---|
| **ECR** | not created by the module | Holds the image. The instance role is granted pull. |
| **EC2** | `aws_instance`, `aws_security_group`, `aws_vpc_security_group_egress_rule` | One host. Single-node by design: state is SQLite on an attached volume, and those guarantees hold across processes on one host, not across hosts. **No ingress rules**; egress 443 only. IMDSv2 required. |
| **EBS** | `aws_ebs_volume`, `aws_volume_attachment` | The state volume, encrypted with the KMS key. |
| **KMS** | `aws_kms_key`, `aws_kms_alias` | Volume and log encryption. An existing key ARN can be supplied instead. |
| **CloudWatch Logs** | `aws_cloudwatch_log_group` | The unit runs with `--log-driver awslogs`. |
| **IAM** | `aws_iam_role` ×2, `aws_iam_role_policy` ×2, `aws_iam_policy`, `aws_iam_role_policy_attachment`, `aws_iam_instance_profile` | A runtime role that carries **no customer-data permissions**, plus one role per declared integration, each assumed with `sts:ExternalId`. |
| **SSM Session Manager** | via the managed policy on the instance role | The only way in. No inbound port, no SSH key, no bastion. |
| **Bedrock** | referenced by `bedrock_model_ids` / `bedrock_inference_profile_arns` | Enumerated model ids, never wildcarded. Only if the tenant's models live there. |
| **Secrets Manager** | referenced by `secret_prefix` | The runtime may read secrets under one prefix and nowhere else. |

VPC and subnet are inputs, not created. The subnet needs egress — a NAT gateway or the
relevant VPC endpoints.

## 8. Health check

```
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3
    CMD ["/usr/local/bin/nova-healthcheck"]
```

`deploy/docker/healthcheck.py`, standard library only, probing
`GET /platform/v1/health` on the configured bind address.

**Liveness, not readiness.** `/platform/v1/health` answers before a tenant bundle is
healthy, so a failure means the process is down — not that someone's configuration is
wrong. Restarting a container because a tenant's policy file has a typo is a loop, not a
recovery.

**200, 401 and 403 all count as healthy, and that is the point.** With
`NOVA_BEHIND_TLS_PROXY` set, the control plane stops trusting loopback callers — a proxy
terminating TLS on the same host makes every forwarded request look local, so trusting
local would mean trusting the internet. The probe is such a caller. A 401 proves both that
the server is accepting connections and that its authentication layer is refusing an
unauthenticated request. Handing the container a bearer token to get a 200 instead would put
a live credential in the container's environment to learn strictly less.

This was found by running it: the first version demanded 200 and reported the container
unhealthy forever.

## 9. Local validation results

`deploy/docker/validate-local.sh nova-control-plane:0.1.0-g2457731df55f` — **49 checks, 49
passed, 0 failed.** Two tenants, two containers, one image, side by side.

| Group | Checks | Result |
|---|---|---|
| NOVA starts in the container | `python -m nova --version`; runs as uid 10001 `nova`, not root | 2/2 |
| Hermes starts in the container | Real Kanban store opened and written; `cron.jobs` write API present; tenant A's list and `get` scoped, tenant B's likewise, each blind to the other | 2/2 |
| `apply` materialises agents | Both tenants: 3 profiles created onto the mounted volume, audit log written | 2/2 |
| Exposure guard | `0.0.0.0` with no principals file → refused; with principals but no TLS statement → refused | 2/2 |
| Two tenants concurrently | Both containers reach Docker-reported `healthy` | 2/2 |
| Authentication | No token → 401; garbage token → 401; tenant B's token against tenant A → 401; tenant A's against tenant B → 401 | 4/4 |
| RBAC | viewer: `/agents` 200, `/policy` 403, `/decisions` 403, `/budget` 403; admin: `/policy` 200 | 5/5 |
| Write surface | viewer create/pause/delete → 403; no token → 401; form-encoded → 415; cross-`Origin` → 403; admin create → **201** | 7/7 |
| Tenant isolation of a created automation | A sees 1, B sees 0; the objective text is withheld from a viewer; B has no provenance registry at all | 4/4 |
| Audit | `intent` → `committed` as the last two records, actor `admin-a` (the authenticated principal, not a service name), tenant `tenant-a` | 3/3 |
| Control Center | `/` 200; **CSP byte-identical** to the documented policy; `X-Frame-Options: DENY`; every referenced asset resolves | 4/4 |
| Lifecycle | Graceful stop: exit code 0, under 5 s; the automation and its provenance survive a restart unchanged | 4/4 |
| No secrets in the image | No `.env`, key, `.netrc`, `.git-credentials`, `.p12`/`.pfx`; no `/run/secrets`; no `.git`; no bundle baked in; no `.tfvars`/`.tfstate`; no AWS access keys; the build-time CA never entered a layer; no secret-shaped env in the image config | 8/8 |

The CSP served by the container, unchanged from before this work:

```
default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:;
connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'
```

### Defects this found, which reading the code did not

Three in the image, three in the validation script itself. Recorded because each is the kind
that only appears once something actually runs.

1. **`nova serve` ignored SIGTERM.** It handled `KeyboardInterrupt` and nothing else. A
   container's main process is PID 1, and the kernel applies no default signal dispositions
   to PID 1 — so the signal was not defaulted, it was *ignored*. Every `docker stop` sat out
   its full 30-second grace period and exited 137, dropping in-flight requests. `serve` now
   installs a SIGTERM handler (main thread only, restoring what it replaced, shutdown driven
   from a separate thread because `shutdown()` blocks on the serving loop). Stop is now
   immediate with exit code 0. Two regression tests pin it.
2. **`croniter` was missing.** It is a core runtime dependency in `pyproject.toml`, imported
   lazily by `cron.jobs`. Without it the Automations screen rendered, the create route
   authorised, and the schedule was rejected at the last step. Now pinned in the image.
3. **A partially copied module root.** Enumerating the handful of top-level Hermes modules
   the control plane appeared to need produced an image that imported cleanly and failed at
   runtime on a transitive sibling (`hermes_state` → `hermes_state_errors`). The whole
   module root is copied now; it is ~1.5 MB.

And in the script — each would have passed while testing nothing:

4. The exposure-guard check wrote a principals file before asserting the *no-principals*
   refusal, so it was asserting the wrong refusal.
5. The AWS-key scan flagged `AKIAIOSFODNN7EXAMPLE` — AWS's own documented placeholder, in a
   comment explaining why a shape check alone is insufficient.
6. The secret-env scan flagged `GPG_KEY` from the upstream Python base, which is a public key
   fingerprint used to verify the CPython tarball.

The repository's own suite also passes: **754 tests**, `tests/platform`.

## 10. Remaining deployment requirements

Nothing below is a defect in the image. Each is a decision or a step that has to happen
outside it, and none of it has been done.

1. **Push the image to the tenant's ECR registry.** Not done, and deliberately not
   attempted: `deploy/aws/README.md` already treats pushing as a separate, deliberate step,
   and no AWS account has been touched. Tag the pushed image with the immutable tag above,
   then set `image_uri` to the **registry digest**, not to `:local` and not to a moving tag.

2. **Put the tenant bundle on the state volume.** The image ships no tenant configuration
   and refuses to serve without one (`no tenant bundle at /var/lib/nova/bundle`). The
   Terraform does not place it. Decide how it gets there — baked into a tenant-specific
   image layer, synced from S3 by `user_data`, or written during provisioning — and whether
   `NOVA_APPLY_ON_START` should run `nova apply` on each boot.

3. **Decide how the Control Center is reached.** This is the one that blocks first use. The
   systemd unit publishes no ports, the security group has no ingress, and access is over
   SSM Session Manager — which gives a shell on the **host**, not inside the container. As
   configured, nothing on the host can reach port 8787. The two supported answers:

   * **Loopback publish.** Add `--publish 127.0.0.1:8787:8787` to the unit and set
     `NOVA_BIND_HOST=0.0.0.0` in `/etc/nova.env`. That bind then requires a principals file
     **and** `NOVA_BEHIND_TLS_PROXY=1` (or a certificate) — both refusals were confirmed
     firing. Reach it with an SSM port-forward session. Still no security-group ingress.
   * **No publish.** Leave the unit as it is and use `docker exec` over an SSM shell for CLI
     work, accepting that the dashboard is not reachable.

   This is an exposure decision, so it is stated rather than made here.

4. **Create the principals file.** `nova token new <name> --role admin|viewer` prints the
   entry; it **does not write the file** — the operator does, at
   `/var/lib/nova/home/control-principals.yaml`. Without it, a loopback caller is a local
   admin by design, and a non-loopback bind is refused outright.

5. **Provide the model credentials.** `nova apply` reports honestly that each agent
   "cannot run yet" until the tenant's credential variables exist in
   `<profile>/.env`. NOVA never writes credentials; that file is the customer's and survives
   `apply`.

6. **Consider the SQLite version.** The `python:3.12-slim-bookworm` base links SQLite
   3.40.1, which carries the upstream WAL-reset corruption bug. Hermes detects this and
   **degrades safely** — it uses `journal_mode=DELETE` instead of WAL and logs a warning
   per database. Nothing is at risk of corruption; what is lost is WAL's reader/writer
   concurrency, which matters for a 24/7 multi-tenant board. The fix already exists in this
   repository: the root `Dockerfile` compiles SQLite 3.53.0400 in a builder stage and
   shadows Debian's `libsqlite3.so.0`. Porting that stage into `Dockerfile.nova` is the
   right answer and was **not done here**, because this environment's egress policy blocks
   `deb.debian.org` and `sqlite.org`, and an unbuilt, untested build stage is worse than a
   documented gap.

7. **Field-validate on AWS.** Everything in §9 is LOCAL VALIDATION. The ECR pull, the
   instance profile, KMS-encrypted volume mounting, `awslogs` delivery, SSM access and the
   `sts:ExternalId` integration roles are configured but unexercised. First deployment is
   where they are proven.

### A note on building in a restricted network

This environment's egress policy denies Docker Hub's blob CDN
(`production.cloudfront.docker.com`), `deb.debian.org` and `sqlite.org`. Two consequences
are visible in the build files, both of which make the image *more* portable rather than
less:

* `Dockerfile.nova` takes the base as `ARG PYTHON_BASE`, so a build behind a registry
  allowlist can name a mirror without the Dockerfile drifting from the canonical image. The
  content digest is recorded in a comment and is identical whichever registry served it.
* There is **no apt layer at all** — the health check is Python, not curl. That removes the
  package index from the build entirely and drops a runtime dependency.
* A TLS-intercepting proxy's CA can be supplied to the one networked step as a BuildKit
  secret (`NOVA_BUILD_CA=…`). It is mounted, never copied, and was confirmed absent from both
  the image filesystem and the build history.

## 11. Files

| File | What it is |
|---|---|
| `deploy/docker/Dockerfile.nova` | The image. |
| `deploy/docker/entrypoint.sh` | Resolves bundle, state and bind address, then `exec`s Python so it is PID 1. |
| `deploy/docker/healthcheck.py` | The health check. |
| `deploy/docker/build.sh` | Immutable tagging and the build. |
| `deploy/docker/validate-local.sh` | The 49 checks in §9. |
| `nova/control/server.py` | Gained `_install_shutdown_handlers`. |
| `tests/platform/test_control_server.py` | Two regression tests for the SIGTERM behaviour. |

The repository's root `Dockerfile` is untouched.
