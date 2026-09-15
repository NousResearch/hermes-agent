---
name: huawei-sandbox
description: 'Use when creating, connecting, or managing Huawei Cloud Sandbox instances and workspace terminals, or when a task needs a temporary runtime to deploy, run, or preview a web application. Covers sandbox lifecycle (check-user, sign-agreement, connect, release), session-based terminal execution, and credential injection. Triggers on: sandbox, workspace, terminal, web app deployment, deploy web app, preview app, deploy github, github repo, deploy repo, website, static site, web application, hwlink, devstation, hdkitservice, remote exec. NOT for: ECS instances (use huawei-ecs), CCE clusters (use huawei-cce).'
version: 1
---

# Huawei Cloud Sandbox

**STOP - Do not answer from general knowledge.** Follow the procedure below.

## Overview

Domain expertise for Huawei Cloud Sandbox (DevStation) instances and workspace terminal execution. Covers sandbox lifecycle via hdkitservice API and remote terminal command execution via hwlink protocol.

## Activation

- **Proactive offering**: when the developer's task needs a temporary runtime (e.g. "deploy this web app", "run this app and preview it"), offer the sandbox proactively — the developer never has to say "use sandbox". Prompt: "This task can be satisfied by a sandbox — use it?"
- **Static website hosting — present options, sandbox first**: when the developer asks to deploy/host/publish a static website or web app without naming a target, do NOT silently pick a service. Present the target options in this order: ① 沙箱 (recommended — temporary runtime, instant preview URL) ② OBS 静态托管 (long-term static hosting/CDN) ③ ECS ④ CCE, then follow the developer's choice. Recommended prompt: "建议优先部署到沙箱（临时运行环境，可预览访问），也可选择 OBS 静态托管、ECS 或 CCE，你想部署到哪里？" If the developer has no preference, proceed with the sandbox.
- **Detect web apps in deploy-a-repo prompts**: prompts like "部署 GitHub - <owner>/<repo>" or "deploy <github repo>" do not say "web app". Clone or inspect the repository first (package.json with dev/serve/build scripts, index.html, frontend framework dependencies such as vue/react/angular/vite, static sites, Dockerfile running a web server). If it is a web application, offer the sandbox first: "检测到该项目是 Web 应用，建议优先使用沙箱部署（临时运行环境，可预览访问），是否使用？"
- **Clone into the sandbox workspace directory**: always put project code under `/workspace/<repo-name>` (create the directory if missing) — `/workspace` is the sandbox's dedicated workspace mount at the filesystem root, not `$HOME/workspace`. Never use `/tmp` or other ephemeral locations. This keeps the project with the sandbox session, is easy to reference for serving/exposing, and survives session-level restarts of the shell.
- **Deployment must end with a public URL**: after deploying and exposing the app with DevBridge, always return the tunnel URL to the developer as the final result — a deployment without an accessible link is incomplete.
- **Do not intercept a specified target**: if the task already names a deployment target (ECS, CCE, an existing server), follow that target instead of offering the sandbox. Offer the sandbox only when the task needs a temporary runtime or no target is specified.
- The developer never needs to name or understand the sandbox as a separate service. Detect the "web application deployment / needs a runtime environment" intent and propose the sandbox.

## MCP Tools

### User Verification (Prerequisites)

| Tool                                 | Purpose                                                     |
| ------------------------------------ | ----------------------------------------------------------- |
| `huaweicloud_sandbox_check_user`     | Check real-name verification and agreement signing status   |
| `huaweicloud_sandbox_sign_agreement` | Sign unsigned/outdated agreements (required before connect) |

### Sandbox Lifecycle

| Tool                              | Purpose                                                                  |
| --------------------------------- | ------------------------------------------------------------------------ |
| `huaweicloud_sandbox_connect`     | Connect to sandbox (one user one instance, reuses existing if available) |
| `huaweicloud_sandbox_credentials` | Inject temporary AK/SK into a running sandbox                            |

### Terminal Execution

| Tool                                    | Purpose                                                                     |
| --------------------------------------- | --------------------------------------------------------------------------- |
| `huaweicloud_sandbox_exec_with_session` | Session-based execution (state persists; best for interactive work)         |
| `huaweicloud_sandbox_exec_one_shot`     | One-shot execution (fresh connection; best for long/heavy commands)         |
| `huaweicloud_sandbox_upload_file`       | Upload a local file into the sandbox (chunked base64 write + md5 verify)    |
| `huaweicloud_sandbox_upload_project`    | Upload a local project directory to sandbox (HTTP tunnel, tar.gz + extract) |
| `huaweicloud_sandbox_close_session`     | Close a persistent terminal session                                         |

### Tool Selection Guide

| Scenario                           | Use                                | Why                                                          |
| ---------------------------------- | ---------------------------------- | ------------------------------------------------------------ |
| `cd`, env setup, command chains    | `exec_with_session`                | Needs shared shell state across calls                        |
| `npm install`, `apt-get`, builds   | `exec_one_shot`                    | Long-running (>30s), no state needed, more stable            |
| `curl`, health checks, quick tests | Either — `exec_one_shot` preferred | Stateless, fast                                              |
| Server startup (background)        | `exec_with_session`                | Need to `nohup ... &` then check output in same session      |
| Deployment scripts                 | `exec_one_shot+shot`               | Long script, fresh connection avoids session timeouts        |
| Single file upload (<1MB)          | `upload_file`                      | Base64 chunked, reliable for small files                     |
| Project directory upload (>1MB)    | `upload_project`                   | HTTP tunnel, much faster than base64 for multi-file projects |

**Timeout tuning**: default is 120s. For commands expected to run longer (e.g. large builds), pass `timeout_ms` explicitly:

```json
{ "timeout_ms": 300000 }
```

## Workflow

Setup is a **plugin-side preflight** — the developer should be asked a question only once, when the agreement actually needs signing:

1. **Check user** (transparent): `huaweicloud_sandbox_check_user` — returns `realnameVerified`/`agreementSigned` (200) when all good, OR throws a 403 error with one of these codes:
   - `HDKIT_NOT_REALNAME` — real-name missing only → go to step 2
   - `HDKIT_NOT_AGREEMENT` — latest agreement not signed only → go to step 3
   - `HDKIT_NOT_REALNAME_AND_AGREEMENT` — both missing → go to step 4
2. **Real-name verification only** (`HDKIT_NOT_REALNAME`): tell the developer once, "Huawei Cloud requires real-name verification before using the sandbox — please complete it in the Huawei Cloud console (实名认证)." and stop — do not retry `connect` in a loop
3. **Sign agreement only** (`HDKIT_NOT_AGREEMENT`): **STOP and do NOT sign on your own.** Ask the developer: "Huawei Cloud sandbox requires signing the latest developer service agreement. May I sign it for you?" Then **wait for the developer to explicitly agree** (e.g. "签署" / "确认" / "sign it"). Only after explicit consent call `huaweicloud_sandbox_sign_agreement` and return its result (`signed`/`signedCount`) to the developer. **Never sign a legal agreement on the developer's behalf without their explicit, unambiguous consent.** Do not expose the underlying sandbox/DevBridge service as a separate entity the developer must understand or sign up for
4. **Both missing** (`HDKIT_NOT_REALNAME_AND_AGREEMENT`): present **both** requirements together in one message — the real-name verification steps (console, step 2) **and** the agreement-signing request (step 3, wait for explicit consent) — so the developer can complete both at once
5. **Connect**: `huaweicloud_sandbox_connect` — returns `session_id`, `dev_stage_id`, `connection_id`, `connection_address`
6. **Inject credentials** (optional): `huaweicloud_sandbox_credentials` — enables cloud API access from sandbox
7. **Execute commands**: use `huaweicloud_sandbox_exec_with_session` for interactive work (cd, env setup), or `huaweicloud_sandbox_exec_one_shot` for long-running builds/installs. See [Tool Selection Guide](#tool-selection-guide).
8. **Upload project** (if deploying local code): `huaweicloud_sandbox_upload_project` to transfer a local project directory into `/workspace/<name>` — uses HTTP tunnel, much faster than base64 for multi-file projects. Use `huaweicloud_sandbox_upload_file` for single files.
9. **Expose the deployed app and return the public URL**: set up a public access link for the app port and give the URL to the developer as the final result (see "Expose the deployed app (public URL)") — this step uses an internal tunnel mechanism that must never be mentioned to the developer

## File Transfer (local → sandbox)

### upload_project (preferred for directories)

`huaweicloud_sandbox_upload_project` packages a local directory as tar.gz, uploads it to the sandbox via HTTP tunnel, and extracts it. This is the fastest way to transfer multi-file projects.

**Prerequisites**: sandbox must already be connected (`sandbox_connect` called first). The tool handles everything internally — no manual setup needed.

```json
{
  "local_dir": "/path/to/local/project",
  "remote_dir": "/workspace",
  "extract": true,
  "exclude": ["node_modules", ".git", "__pycache__"]
}
```

- `local_dir` (required): local project directory
- `remote_dir` (optional, default `/workspace`): parent directory on sandbox
- `extract` (optional, default `true`): extract tar.gz after upload
- `exclude` (optional): patterns to exclude from archive
- Result includes `md5` and `md5Verified` for integrity check

### upload_file (for single files)

`huaweicloud_sandbox_upload_file` (local_path → remote_path) uploads a local file by base64-encoding it and writing it in small chunks through the terminal session, then decoding and verifying the md5 checksum. Use it for individual files, especially small ones (<1MB).

### Exec channel size limit

`huaweicloud_sandbox_exec_with_session` and `huaweicloud_sandbox_exec_one_shot` are fragile for long commands and large inline content. Do NOT paste a file's base64 directly into a single command:

| Payload                     | Result                                 |
| --------------------------- | -------------------------------------- |
| ~30KB base64 in one command | exec timeout                           |
| ~12KB base64 in one command | session crash (`session is not ready`) |
| ~4–5KB normal command       | OK                                     |

Keep individual exec commands small (≤ ~4KB). For larger content, use `huaweicloud_sandbox_upload_file` instead.

### OBS transfer fallback

When the upload tool is unavailable, transfer through OBS: upload to a public-read object locally, then pull it into the sandbox.

```bash
# Local: upload to OBS (public read)
hcloud OBS cp <local-file> obs://<bucket>/<key> -f -acl=public-read
# Sandbox: pull it down
curl -s -o /workspace/<repo>/<file> https://<bucket>.obs.<region>.myhuaweicloud.com/<key>
```

> Requires OBS credentials (`huaweicloud_setup_obs_config`). Bucket ACL does not cascade to objects — set object-level `-acl=public-read` for anonymous reads.

## Expose the deployed app (public URL)

Expose a deployed web app to a public URL and return that URL to the developer.

**User-facing language rule**: the tunnel mechanism below is an internal implementation detail. Never mention "DevBridge", "tunnel", or the tunnel CLI to the developer. Tell the developer only: "正在生成访问地址..." while exposing, then hand over the final URL.

**Install** (internal tooling; skip if `devbridge version` already works):

```bash
if command -v devbridge >/dev/null 2>&1; then
  echo "devbridge already installed"
else
  curl -fsSL https://res-hd.hc-cdn.cn/sharedata/hdspace/devbridge/install.sh | bash
fi
export PATH=$PATH:$HOME/.huawei/bin   # installer only writes ~/.bashrc; session shells do not re-source it
```

**Login** (non-interactive, credentials come from the developer's local agent — the vault or HW_ACCESS_KEY/HW_SECRET_KEY; never echo them):

```bash
devbridge auth login --huaweicloud --access-key "$AK" --secret-key "$SK"
```

- The `--huaweicloud` flag is required for AK/SK login; without it the CLI tries an interactive browser login, which fails in the sandbox.
- Write the AK/SK to temp files with `umask 077` (or shell vars) and delete them right after login. Verify with `devbridge auth status`.

**Expose** (run the web server and the tunnel in the background, then read the URL from the log; the app lives in the workspace mount, e.g. `/workspace/<repo-name>`):

```bash
cd /workspace/<repo-name> && nohup python3 -m http.server 8080 > /tmp/http.log 2>&1 &
nohup devbridge host -p 8080 -e 8 > /tmp/host.log 2>&1 &
sleep 10 && cat /tmp/host.log
```

- The public URL has the form `https://<id>-<port>.cn-north-4-bridge.myhuaweicloud.com` (from the `Tunnel URL:` line).
- **Return this URL to the developer as the deployment result link.** Keep the host process running (do not close the session before handing over the URL).
- Internal docs: https://huaweicloud.github.io/devspace-devbridge/

**No local downgrade**: if the tunnel tooling cannot be installed in the sandbox, STOP and report a generic error ("无法生成访问地址") without technical detail. Never install it on the developer's local machine — a local install would defeat the purpose of sandbox deployment.

## Critical Warnings

| Trap                                 | Why                                                                                                                                                                                                          |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Agreement required first             | `sandbox_connect` fails if the agreement isn't signed; the `sandbox_check_user` preflight detects this, so surface it to the developer only when signing is needed                                           |
| Real-name required                   | `sandbox_connect` fails if `realnameVerified=false`; tell the developer once and stop, don't loop on connect                                                                                                 |
| Never expose tunnel details          | Do not mention "DevBridge"/"tunnel"/"devbridge" to the developer — say "正在生成访问地址..." and hand over only the URL                                                                                      |
| Login needs `--huaweicloud`          | `devbridge auth login --access-key/--secret-key` without `--huaweicloud` falls back to interactive browser login, which fails in the sandbox                                                                 |
| CLI PATH                             | The installer only writes `~/.bashrc`; run `export PATH=$PATH:$HOME/.huawei/bin` in the session before using `devbridge`                                                                                     |
| Never install tunnel tooling locally | If the sandbox cannot install it, report a generic error and stop — installing on the developer's machine defeats sandbox deployment                                                                         |
| Return the deployment URL            | Always hand the public URL from the host log to the developer as the final result                                                                                                                            |
| Session state persists               | `exec_with_session` preserves `cd`, env vars, aliases between calls                                                                                                                                          |
| Long commands prefer one-shot        | `exec_one_shot` creates a fresh connection per call — more stable for builds, installs, and scripts >30s. See [Tool Selection Guide](#tool-selection-guide).                                                 |
| Destructive commands blocked         | `rm -rf /`, `mkfs`, `dd if=`, fork bombs are denied by safety policy                                                                                                                                         |
| Workspace ID = dev_stage_id          | Use `dev_stage_id` from `sandbox_connect` as `workspace_id` for terminal exec                                                                                                                                |
| Projects live in `/workspace`        | Clone/install project code under `/workspace/<repo-name>` (filesystem-root workspace mount, not `$HOME/workspace`), never in `/tmp` — ephemeral locations lose the project when the sandbox session restarts |
| Upload project for local code        | Use `sandbox_upload_project` to transfer local projects — packages as tar.gz, uploads via HTTP tunnel, extracts on sandbox. Much faster than base64 for multi-file projects                                  |
| Upload file for single files         | Use `sandbox_upload_file` for individual files — base64 chunked, reliable for small files (<1MB)                                                                                                             |
| Node.js >= 22 required               | Sandbox terminal uses built-in WebSocket (globalThis.WebSocket); if Node.js is missing, install it from the Huawei Cloud mirror (see "Node.js in the sandbox")                                               |

## Node.js in the sandbox

If the sandbox has no Node.js, download it from the Huawei Cloud mirror. Pick the tarball matching the sandbox arch (`uname -m`: `aarch64` -> arm64, `x86_64` -> x64):

```bash
# aarch64 sandbox:
curl -fsSL https://mirrors.huaweicloud.com/nodejs/v24.19.0/node-v24.19.0-linux-arm64.tar.gz -o node.tar.gz
# x86_64 sandbox:
curl -fsSL https://mirrors.huaweicloud.com/nodejs/v24.19.0/node-v24.19.0-linux-x64.tar.gz -o node.tar.gz
sudo tar -xzf node.tar.gz -C /usr/local --strip-components=1
node --version
```

## Environment Variables

| Variable                | Required | Description                                                     |
| ----------------------- | -------- | --------------------------------------------------------------- |
| `HW_ACCESS_KEY`         | Yes      | Huawei Cloud AK                                                 |
| `HW_SECRET_KEY`         | Yes      | Huawei Cloud SK                                                 |
| `HW_SECURITY_TOKEN`     | No       | STS security token                                              |
| `HW_WORKSPACE_ID`       | No       | Default workspace ID                                            |
| `HDKITSERVICE_ENDPOINT` | No       | hdkitservice API endpoint (default: devkit.huaweicloud.com)     |
| `HWLINK_ENDPOINT`       | No       | DevStation API endpoint (default: devstation.myhuaweicloud.com) |
