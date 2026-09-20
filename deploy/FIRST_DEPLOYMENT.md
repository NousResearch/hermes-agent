# First AWS deployment — runbook

Steps 3–5 of the playbook were executed in the build container on 15 September 2026 against
commit `e7b4eaae`. Results are recorded in §A. Everything from step 6 onward has to happen
on **your** machine and in **your** AWS account, and is written out below as commands you
can paste.

> **Why you build it again on your own machine.** Pushing to your ECR needs your AWS
> credentials. Do not put those into an agent session — build locally and push from there.
> The build is deterministic from the commit, so you will get the same tag.

---

## A. What has already been proven (build container, not AWS)

| | |
|---|---|
| Commit | `e7b4eaae`, clean tree — no `-dirty` suffix |
| Image tag | `nova-control-plane:0.1.0-ge7b4eaaed212` |
| Local image ID | `sha256:742eba1891b8044060a568dbfc73afc73616a5ed3952a7efad9c3645b52dc745` |
| Size | 307 MB |
| Validation | **49 checks, 49 passed, 0 failed** |

The local image ID above is **not** the ECR digest. A registry assigns its own on push, and
that is the one to pin in `image_uri` (step 9).

What the 49 covered: NOVA and Hermes both start in-container and run as uid 10001; `apply`
materialises 3 profiles for each of two tenants onto the mounted volume; the exposure guard
refuses `0.0.0.0` without a principals file and refuses it again without TLS; two tenants
run side by side and each token 401s against the other; viewer is refused on policy,
decisions, budget and every write; a form-encoded write is 415 and a cross-origin write is
403; the audit log records `intent` → `committed` with the human's name; the CSP is
byte-identical to the documented policy; a graceful stop exits 0 in under 5 s; state and
provenance survive a restart; and eight separate checks find no secret in the image.

---

## Step 3 — Install the local tools

**macOS**

```bash
brew install --cask docker                 # then launch Docker Desktop once
brew install awscli terraform
brew install --cask session-manager-plugin
```

**Ubuntu / Debian**

```bash
# Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"            # log out and back in

# AWS CLI v2
curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip -q awscliv2.zip && sudo ./aws/install && rm -rf aws awscliv2.zip

# Terraform
wget -qO- https://apt.releases.hashicorp.com/gpg \
  | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp.gpg] \
https://apt.releases.hashicorp.com $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update && sudo apt install -y terraform

# Session Manager plugin — install it NOW, not at step 18
curl -fsSL "https://s3.amazonaws.com/session-manager-downloads/plugin/latest/ubuntu_64bit/session-manager-plugin.deb" \
  -o /tmp/smp.deb && sudo dpkg -i /tmp/smp.deb
```

**Windows** — use WSL2 and follow the Ubuntu instructions inside it. Native Windows works
but the shell quoting in this runbook assumes bash.

**Verify all four:**

```bash
docker --version && aws --version && terraform --version && session-manager-plugin --version
```

**Then configure the CLI — with an IAM user, never root:**

```bash
# In the AWS console: IAM → Users → Create user → attach AdministratorAccess
# → Security credentials → Create access key → "Command Line Interface"
aws configure            # paste the key, secret, region eu-west-2, output json
aws sts get-caller-identity
```

That last command must print **your** account id. Check it every time before a `terraform
apply`; it is the cheapest way to avoid building in the wrong account.

> **Security.** Turn on MFA for the root user and then never use root again. The access key
> lives in `~/.aws/credentials` — never commit it, never paste it into a chat.

---

## Step 4 — Build the image

```bash
cd /path/to/hermes-agen-
git status --short                 # expect empty
deploy/docker/build.sh --print-tag # expect nova-control-plane:0.1.0-ge7b4eaaed212
deploy/docker/build.sh
```

A `-dirty.<hash>` suffix means you have uncommitted changes that would land in the image.
That is the script working correctly. Commit, then rebuild.

---

## Step 5 — Validate locally

```bash
deploy/docker/validate-local.sh nova-control-plane:0.1.0-ge7b4eaaed212
```

Expect `=== 49 passed, 0 failed ===`. Anything else: stop, do not push. A failure here is a
failure in production too, only harder to see.

---

## Step 6 — Create the ECR repository

```bash
export AWS_REGION=eu-west-2
export ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
export ECR="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

aws ecr create-repository \
  --repository-name nova-control-plane \
  --region "$AWS_REGION" \
  --image-tag-mutability IMMUTABLE \
  --image-scanning-configuration scanOnPush=true
```

`IMMUTABLE` means a tag can never be repointed at different content. `scanOnPush` gets you
a free CVE report. Both matter and neither costs anything.

---

## Step 7 — Authenticate Docker to ECR

```bash
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$ECR"
```

Expect `Login Succeeded`. The token lasts 12 hours. The pipe keeps the password out of your
shell history — do not paste it as an argument.

---

## Step 8 — Push

```bash
export TAG=0.1.0-ge7b4eaaed212
docker tag  "nova-control-plane:${TAG}" "${ECR}/nova-control-plane:${TAG}"
docker push "${ECR}/nova-control-plane:${TAG}"
```

Never push `:local`. Never push a `-dirty` tag.

---

## Step 9 — Get the immutable digest

```bash
aws ecr describe-images \
  --repository-name nova-control-plane \
  --region "$AWS_REGION" \
  --image-ids imageTag="${TAG}" \
  --query 'imageDetails[0].imageDigest' --output text
```

Prints `sha256:…`. Build the URI you will actually deploy:

```bash
export IMAGE_URI="${ECR}/nova-control-plane@sha256:<paste the digest>"
echo "$IMAGE_URI"
```

**Deploy the digest, not the tag.** A digest is the content; a tag is a label. This is a
supply-chain control, and it is the difference between a reproducible deployment and a
hopeful one.

---

## Step 10 — Your tenant bundle

```bash
cp -r nova/examples/acme  bundles/yourfirm
```

Edit, in this order:

| File | What to change |
|---|---|
| `organization.yaml` | `tenant_id` (lowercase, no spaces — it is stamped on every audit event), legal name, region, contact |
| `identity.yaml` | Product name, company name, accent colour, support email. This is the white-labelling and it is real |
| `agents/*.yaml` | Your agents. Start with one |
| `prompts/*.md` | Each agent's persona |
| `policy.yaml` | Permissions and which actions need approval |
| `knowledge.yaml` | Corpora. Point `root` at a real directory |
| `channels.yaml` | Leave empty for the first deployment |
| `deployment.yaml` | Provider and model. This feeds `bedrock_model_ids` |

Validate before going further:

```bash
python3 -m nova validate bundles/yourfirm
```

Errors name the file and the field. A permission not defined in the policy, a knowledge id
typo, or an agent delegating to one that does not exist all fail here — by design.

> **Never put a credential in the bundle.** `token`, `api_key`, `secret`, `password`,
> `credential` and `app_secret` are refused at parse time at any nesting depth.

---

## Step 11 — Principals

```bash
python3 -m nova token new alice --role admin
python3 -m nova token new bob   --role viewer
```

Each prints the token **once** and a YAML entry containing only its SHA-256. NOVA never
stores the token — lose it and you mint a new one.

**The command does not write the file; you do.** Collect the entries into
`control-principals.yaml`, which goes on the state volume at
`/var/lib/nova/home/control-principals.yaml` in step 20.

Partners get `admin`. Everyone else gets `viewer`.

---

## Step 12 — Terraform variables

Two files. One you write, one NOVA generates.

```bash
cd deploy/aws
cp terraform.tfvars.example terraform.tfvars
```

Fill in:

```hcl
tenant_id = "yourfirm"
region    = "eu-west-2"
vpc_id    = "vpc-…"            # yours
subnet_id = "subnet-…"         # PRIVATE, with egress via NAT or VPC endpoints
image_uri = "…@sha256:…"       # from step 9
ami_id    = ""                 # pin this before production
instance_type = "t3.large"
volume_gb     = 100
bedrock_model_ids = ["eu.anthropic.claude-sonnet-4-20250514-v1:0"]
secret_prefix     = "nova/yourfirm/"
```

Then let NOVA write the integrations from the bundle, so what the agents may reach is
derived from the declaration rather than maintained beside it:

```bash
cd ../..
python3 -m nova deploy show   bundles/yourfirm    # read this — it is the blast radius
python3 -m nova deploy render bundles/yourfirm    # writes deploy/aws/nova.auto.tfvars.json
```

> **The subnet needs outbound internet.** A NAT gateway, or VPC endpoints for ECR, S3, SSM,
> CloudWatch Logs and Bedrock. Do **not** solve it by putting the host in a public subnet
> with a public IP.

**Replace every placeholder.** `111122223333` and `vpc-0123456789abcdef0` are AWS's own
documented examples and Terraform will not catch them — they are syntactically valid.

---

## Steps 13–15 — init, validate, plan

```bash
cd deploy/aws
terraform init
terraform validate
terraform plan -out=tfplan
```

Before applying, confirm you are pointed at the right account:

```bash
aws sts get-caller-identity
```

---

## Step 16 — Read the plan

Expect `Plan: 13 to add, 0 to change, 0 to destroy`:

1 EC2 instance · 1 EBS volume + attachment · 1 KMS key + alias · 1 CloudWatch log group ·
1 security group · 2 IAM roles · 2 role policies · 1 permissions-boundary policy ·
1 attachment · 1 instance profile.

**Confirm two things by eye:**

```bash
terraform show -json tfplan | python3 -c "
import json,sys
p=json.load(sys.stdin)
for r in p['resource_changes']:
    if 'security_group' in r['type'] and 'ingress' in str(r['change']['after']):
        print('INGRESS RULE FOUND — investigate:', r['address'])
    if r['type']=='aws_instance':
        md=r['change']['after'].get('metadata_options') or [{}]
        print('IMDSv2 required:', md[0].get('http_tokens'))
"
```

You want **no ingress rule** and `http_tokens: required`. Those two lines are most of the
host's security posture. Save the plan output — it is your first piece of deployment
evidence and the thing to show a customer later.

---

## Step 17 — Apply

```bash
terraform apply tfplan
terraform output
```

Two to four minutes. Common failures: a service quota, an instance type not offered in the
region, or your own IAM user lacking a permission. AWS names which in the error. Quota
increases are a support ticket and can take a day — hit that now rather than with a
customer watching.

**You are now spending money.** Do step 18b today.

---

## Step 18 — Get a shell, over SSM

```bash
export INSTANCE_ID="$(terraform output -raw instance_id)"
aws ssm start-session --target "$INSTANCE_ID" --region eu-west-2
```

`TargetNotConnected` usually means the instance is still booting (wait two minutes) or the
subnet cannot reach the SSM service (check the route or the VPC endpoints).

This is the only way in. No SSH key exists. **Do not "temporarily" add an ingress rule.**

### 18b — Cost guardrails, before you forget

```bash
aws logs put-retention-policy \
  --log-group-name "/nova/yourfirm" --retention-in-days 30 --region eu-west-2

aws ecr put-lifecycle-policy --repository-name nova-control-plane --region eu-west-2 \
  --lifecycle-policy-text '{"rules":[{"rulePriority":1,"selection":{"tagStatus":"untagged","countType":"sinceImagePushed","countUnit":"days","countNumber":7},"action":{"type":"expire"}}]}'
```

Then in the console: **Billing → Budgets → Create budget**, monthly, with an 80% forecast
alert to an address someone reads. CloudWatch logs default to *never expiring* and that is
the classic surprise on the first bill.

---

## Steps 19–21 — Verify the host

In the SSM session:

```bash
df -h /var/lib/nova                    # 19: XFS mounted — state is on the encrypted EBS
lsblk -f                               #     confirm it is the EBS volume, not the root disk

sudo docker ps                         # 20: one container, from your digest
sudo docker inspect --format '{{.State.Health.Status}}' nova

sudo docker exec nova /usr/local/bin/nova-healthcheck; echo "exit=$?"   # 21
```

Expect exit `0`. **HTTP 200, 401 and 403 all count as healthy** — with a TLS proxy declared
the control plane stops trusting loopback, so an unauthenticated probe correctly gets 401,
and that proves both the server and its auth layer are alive.

If the container restarts in a loop: `sudo docker logs nova`. Two causes, in the order
you will meet them:

1. **`not writable by uid 10001`** — the state volume is not owned by the container
   user. `user_data` now chowns it (see §B); on a host built before that fix:
   `sudo chown -R 10001:10001 /var/lib/nova && sudo systemctl restart nova`.
2. **`no tenant bundle`** — the image ships none and refuses to serve without one.
   That is step 23.

---

## Step 22 — Reaching the Control Center

**This was an open decision in the playbook. It has now been settled and tested, and the
deployment template has been fixed — see §B at the end for what changed and why.**

The unit now runs the container with `--network host`. Nothing extra to configure:

```bash
aws ssm start-session --target "$INSTANCE_ID" --region eu-west-2 \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["8787"],"localPortNumber":["8787"]}'
```

Then open **http://localhost:8787** in your browser. You get the dashboard, as a local
admin, with no token.

**Why it has to be host networking.** The dashboard is a browser and cannot send an
`Authorization` header, so the control plane trusts callers arriving on loopback — the
reasoning being that anyone with a shell on the host can already read the principals file,
the bundle and the audit log off disk anyway. Under Docker's *default* bridge network a
published port is NAT'd, so the container sees the bridge gateway (`172.17.0.1`) as the
client rather than `127.0.0.1`, loopback trust never applies, and **every request including
`GET /` answers 401.** Both the `--publish` and the TLS-certificate variants were tried and
both hit the same 401 wall. Host networking makes a connection from the host's own loopback
genuinely arrive as `127.0.0.1`.

**Nothing is exposed by this.** The process still binds `127.0.0.1` by default, the security
group still has no ingress rule, and the only route in is still an SSM port-forward, which
is gated by IAM and logged by AWS.

**Who can reach it, and as what:**

| Caller | Sees |
|---|---|
| You, through the SSM tunnel | Local admin. Full dashboard. |
| A remote caller with an admin token | Admin, if you ever bind non-loopback |
| A remote caller with a viewer token | Read-only: 403 on policy, decisions, budget, and every write |
| Anyone else | Nothing — there is no inbound route |

**Never** open this with a security-group ingress rule.

> **If you do need genuinely remote access later** — a customer's staff using it from their
> own machines rather than through your tunnel — that is a different build: an internal
> load balancer terminating TLS, `NOVA_BEHIND_TLS_PROXY=1`, a principals file, and every
> user holding their own token. The bearer-token path works fine for API clients; it is
> only the browser that depends on loopback. Do not attempt that for the first deployment.

---

## Steps 23–24 — Tenant and agents

Get the bundle onto the state volume. Simplest for a first deployment:

```bash
# from your laptop
tar czf /tmp/bundle.tgz -C bundles yourfirm
aws s3 cp /tmp/bundle.tgz "s3://your-private-bucket/bundle.tgz"

# in the SSM session
sudo aws s3 cp s3://your-private-bucket/bundle.tgz /tmp/
sudo mkdir -p /var/lib/nova/bundle
sudo tar xzf /tmp/bundle.tgz -C /var/lib/nova/bundle --strip-components=1

# and the principals file from step 11
sudo vi /var/lib/nova/home/control-principals.yaml

# Anything you copied in over SSM arrived owned by root. The container runs as uid 10001.
sudo chown -R 10001:10001 /var/lib/nova

sudo systemctl restart nova
sudo docker exec nova python -m nova apply /var/lib/nova/bundle
```

Warnings that each agent "cannot run yet" are correct, not errors. Step 25 fixes them.

> `apply` **overwrites** derived files, including each agent's `SOUL.md`. Edit the bundle,
> never the generated output.

---

## Step 25 — The model credential

Bedrock on the instance role needs no key — the role already carries
`bedrock:InvokeModel` for the models you enumerated. For a non-Bedrock provider:

```bash
sudo tee /var/lib/nova/home/profiles/<agent>/.env >/dev/null <<'EOF'
YOURFIRM_LLM_KEY=...
EOF
sudo chmod 600 /var/lib/nova/home/profiles/<agent>/.env
sudo docker exec nova python -m nova doctor /var/lib/nova/bundle
```

> **Put it in the profile's `.env`, never the host environment.** A credential exported into
> the host environment is readable by *every* agent on that host — that is verified
> behaviour, not a theory. Per-agent isolation holds only when credentials live in the
> per-agent store.

---

## Steps 26–28 — Knowledge and channels

```bash
sudo docker exec nova python -m nova knowledge status /var/lib/nova/bundle
sudo docker exec nova python -m nova knowledge ingest /var/lib/nova/bundle
sudo docker exec nova python -m nova knowledge search /var/lib/nova/bundle "refund policy"
```

`0 indexed` almost always means the include globs do not match your files. The Control
Center shows what each corpus accepts.

For an S3-backed corpus, add an `origin:` block and grant `s3:ListBucket` + `s3:GetObject`
on **that one prefix** through the bundle's integrations — nothing broader. A mirrored
corpus is read-only from NOVA: uploads to it are refused, because the next sync would
delete them.

**Channels: skip for the first deployment.** Zero of the 22 platforms are field-validated,
and a webhook platform needs a publicly reachable HTTPS endpoint this deployment
deliberately does not have.

---

## Steps 29–30 — The two that have never been done

### 29 — Scheduled execution

```bash
sudo docker exec nova python -m nova status          # note: no bundle argument
```

Declare an automation, then watch whether it actually fires. **The cron ticker lives inside
the runtime's gateway; there is no standalone daemon.** A deployment can hold a perfectly
correct schedule that nothing executes, which is why the Automations screen leads with
scheduler liveness. You will need a gateway process running alongside the control plane.
This has never been proven in a NOVA deployment — budget real time for it.

### 30 — A real model call

```bash
sudo docker exec nova python -m nova objective list   /var/lib/nova/bundle
sudo docker exec nova python -m nova objective submit /var/lib/nova/bundle <objective-id>
sudo docker exec nova python -m nova objective status /var/lib/nova/bundle <objective-id>
sudo docker exec nova python -m nova status           # the board
#   `nova work` is for acting on items (release / resume / reject / note), not listing them.
```

**This is the most important step in the whole runbook.** Nothing in NOVA has ever called a
real model provider. Expect to find things. Check the Usage screen's token accounting
against the actual AWS bill afterwards — that comparison is the point.

---

## Steps 31–36 — Prove it and write it down

```bash
# 31 logs
aws logs tail /nova/yourfirm --follow --region eu-west-2

# 32 audit — show this file to a customer; it is your most persuasive artefact
sudo tail -5 /var/lib/nova/home/nova/audit.jsonl | python3 -m json.tool
sudo docker exec nova python -m nova audit status

# Tamper-evidence is two steps: seal now, verify later against that seal.
# Write the seal somewhere the runtime user CANNOT write, or it proves nothing.
sudo docker exec nova python -m nova audit seal /var/lib/nova/audit.seal
sudo aws s3 cp /var/lib/nova/audit.seal s3://your-evidence-bucket/   # off the host
sudo docker exec nova python -m nova audit verify /var/lib/nova/audit.seal

# 33 restart and recovery
sudo systemctl restart nova
sudo docker exec nova python -m nova status
sudo docker exec nova python -m nova backup /var/lib/nova/bundle /var/lib/nova/backup.tar.gz
#    restore is: nova restore <archive> [--dry-run]
#    then actually restore it onto a fresh host. A backup you have never restored is not a backup.

# 34 isolation — one tenant per deployment, full stop
sudo docker exec nova python -m nova status | head -5

# 35 permissions — do this in front of a customer
curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $VIEWER" localhost:8787/platform/v1/agents  # 200
curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $VIEWER" localhost:8787/platform/v1/policy  # 403
```

**36 — the evidence pack.** A dated folder per deployment containing: the plan output, the
image digest, `terraform output`, the 49-check result, screenshots of steps 30–35, and the
account and region. This becomes the runbook for customer two and the evidence for the
first security questionnaire you are asked to fill in. Store it somewhere the credentials
are not — evidence packs get emailed.

---

## The order that matters

Steps 1–17 build infrastructure. 18–28 configure NOVA. **29 and 30 are the ones that have
never been done anywhere**, and they are what turn this from a control plane into a
product. Do not promise a customer scheduled work until you have watched a job fire, and do
not quote a model cost until you have seen a real bill.

---

## B. Two defects found while validating this runbook, and fixed

Both were found by running the image against a mounted volume rather than by reading the
template. Both would have failed the first AWS deployment outright, and neither is visible
by inspection — which is the whole argument for doing step 5 and this dry run before a
customer is watching.

### B1. The state volume was never chowned — the first boot always failed

`user_data.sh.tftpl` formatted and mounted the EBS volume but never changed its ownership.
A freshly formatted XFS volume is `root:root`; the image runs as uid 10001. So the
entrypoint's writability check failed and the container exited immediately:

```
nova-entrypoint: /var/lib/nova/home is not writable by uid 10001.
                 The state volume must be owned by the container user.
```

`Restart=always` then restarts it forever. **Every first deployment would have hit this.**

*Fixed:* one line in `deploy/aws/user_data.sh.tftpl`, after `mount -a`:

```bash
chown -R 10001:10001 "$STATE_MOUNT"
```

It has to come after the mount — applied to the mountpoint beforehand it changes the
directory underneath, which the mount then hides.

### B2. The dashboard answered 401 to everything, under every configuration tried

The playbook left "how is the Control Center reached" as an open decision with two options.
Both turn out not to work:

| Tried | Result |
|---|---|
| `--publish 127.0.0.1:8787:8787` + `NOVA_BIND_HOST=0.0.0.0` + `NOVA_BEHIND_TLS_PROXY=1` | `GET /` → **401** |
| `--publish` + `NOVA_BIND_HOST=0.0.0.0` + a TLS certificate | `GET /` → **401** |
| `--network host` | `GET /` → **200** ✅ |

The dashboard is a browser and cannot send an `Authorization` header, so it depends on the
control plane trusting loopback callers. Under Docker's default bridge network a published
port is NAT'd and the container sees the bridge gateway as the client, so loopback trust
never applies. `NOVA_BEHIND_TLS_PROXY=1` disables loopback trust deliberately on top of
that, which is correct behaviour for a real proxy and exactly wrong here.

*Fixed:* `--network host` in the systemd unit. The process still binds `127.0.0.1`, so
nothing is exposed; the security group still has no ingress; SSM remains the only route in.

### Both are pinned by tests

`tests/platform/test_deploy_aws.py`:

- `test_the_state_volume_is_chowned_to_the_container_user` — also asserts the chown follows
  the mount, which is the part that is easy to get wrong when someone tidies the script.
- `test_the_container_runs_with_host_networking` — and that `--publish` has not been
  reintroduced alongside it, since the two together mean somebody has brought the bridge
  assumption back.

### What this means for the playbook

`NOVA_Commercial_and_AWS_Deployment_Playbook.pdf` §24 action 4 is "settle and document
Control Center access". **That action is now done** — the decision is host networking, it is
in the template, and it is tested. §23's `CRITICAL` row for Control Center exposure can be
closed when the playbook is next regenerated.

The other three blockers are untouched and still stand: **no AWS field validation**, **no
real model provider has ever been called**, and **scheduled execution has never been
observed firing**. Those are steps 17, 30 and 29 of this runbook, in that order.
