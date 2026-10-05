---
name: runpod
description: Rent RunPod GPU pods with a reusable model volume.
version: 0.1.0
author: rstephens11-ux, Hermes Agent
license: MIT
platforms: [linux, macos]
required_environment_variables:
  - name: RUNPOD_API_KEY
    prompt: RunPod API key (starts with rpa_)
    help: "console.runpod.io → Settings → API Keys → Create API Key (Read & Write)."
    required_for: "every RunPod API call: availability scans, deploy, poll, terminate"
metadata:
  hermes:
    tags: [Infrastructure, GPU Cloud, RunPod, Inference, Rendering, ComfyUI]
    related_skills: [lambda-labs, modal]
---

# RunPod GPU Pods

Rent short-lived RunPod GPU pods for heavy diffusion, video, or model jobs the local machine can't run, driven entirely through RunPod's GraphQL/REST APIs. A persistent **network volume** caches model weights so large stacks download once, not once per pod. The skill covers the whole lifecycle — scan, deploy, bootstrap, run, download, terminate — and is built around one rule: **a pod that is not working is costing money**.

## When to Use

- A job needs more VRAM than the local GPU has (e.g. a ~50 GB int8 video-model stack)
- Repeated sessions against the same large model set (the volume pays for itself)
- One-off jobs where an hourly GPU beats buying hardware

**Don't use for:** multi-day training clusters (see `lambda-labs`), or serverless function-style inference (see `modal`).

## Prerequisites

- RunPod account with funds. Keep the balance low (~$10–25) as a circuit-breaker against a forgotten pod.
- `RUNPOD_API_KEY` set through Hermes' secret setup — never pasted into chat.
- `curl`, `ssh`, `scp`, and an SSH key pair. On a fresh machine:
  `terminal(command="mkdir -p ~/.ssh && chmod 700 ~/.ssh && test -f ~/.ssh/id_ed25519 || ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519 -q")`

## How to Run

All GraphQL calls go to `https://api.runpod.io/graphql` with `Authorization: Bearer $RUNPOD_API_KEY`. Use **curl**, and pass the header via stdin so the key never appears in the process list or logs:

```
terminal(command="printf 'header = \"Authorization: Bearer %s\"\\n' \"$RUNPOD_API_KEY\" | curl -s --config - -H 'Content-Type: application/json' https://api.runpod.io/graphql -d '{\"query\":\"query { myself { clientBalance } }\"}'", timeout=30)
```

Parse responses in Python (`json.loads`) — never eyeball availability or IDs out of raw JSON.

## Quick Reference

| Action | Call |
|---|---|
| Balance | `query { myself { clientBalance } }` |
| List volumes | `query { myself { networkVolumes { id name size dataCenterId } } }` |
| Availability, every datacenter | `query { dataCenters { id gpuAvailability(input: {minDisk: 40, minMemoryInGb: 8}) { available stockStatus gpuTypeDisplayName } } }` |
| Register SSH key (account-wide) | `mutation { updateUserSettings(input: {pubKey: "ssh-ed25519 AAAA..."}) { id } }` |
| Pod status / ports | `query { pod(input: {podId: "<id>"}) { desiredStatus runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort } } } }` |
| Terminate | `mutation { podTerminate(input: {podId: "<id>"}) }` |
| Create volume (REST) | `POST https://rest.runpod.io/v1/networkvolumes` body `{"name":"...","size":200,"dataCenterId":"<dc>"}` |
| Delete volume (REST) | `DELETE https://rest.runpod.io/v1/networkvolumes/<id>` |

Deploy:

```
mutation { podFindAndDeployOnDemand(input: { cloudType: SECURE, gpuCount: 1,
  gpuTypeId: "NVIDIA L40S", name: "<job-name>",
  imageName: "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
  containerDiskInGb: 40, networkVolumeId: "<vol-id>", volumeMountPath: "/workspace",
  ports: "8188/http,22/tcp", startSsh: true, env: [] }) { id costPerHr } }
```

## Procedure

1. **Stage locally first.** Inputs, workflow files, and a bootstrap script are ready before anything is deployed — compute bills from the moment a pod exists, including setup and idle time. *Done when:* every input file the job needs exists locally.
2. **Scan availability across ALL datacenters** and pick a card plus a datacenter that stocks several interchangeable cards of the same VRAM tier. *Done when:* the target card shows `available: true` in a datacenter that also supports network volumes.
3. **Volume in the same datacenter.** Reuse an existing volume there, or create one now. *Done when:* the volume's `dataCenterId` matches the chosen datacenter.
4. **Register the SSH public key** before deploying (pods copy the account key at creation). *Done when:* `updateUserSettings` returns an id.
5. **Deploy, then poll** every ~10 s until `runtime.ports` is populated (usually 25–60 s). The SSH endpoint is the entry with `privatePort: 22` and `isIpPublic: true` — use its `ip:publicPort`. *Done when:* that entry exists. Report `costPerHr` to the user.
6. **Bootstrap.** `ssh-keyscan -p <port> <ip> >> ~/.ssh/known_hosts`, `scp` the bootstrap script, start it detached: `nohup bash bootstrap.sh > /workspace/bootstrap.log 2>&1 &`. The script should skip files already on the volume (`[ -s "$dest" ] && return`), resume partial downloads (`curl -C -`), and end by printing a unique sentinel line. *Done when:* the sentinel appears in the log.
7. **Run the job** through the service's own API (e.g. ComfyUI on :8188), then download every output. *Done when:* outputs exist locally with non-zero size.
8. **Terminate and verify in the same turn.** Call `podTerminate`, then query the pod. *Done when:* the pod query returns `null`. Report total spend, false starts included.

## Pitfalls

1. **Volume and pod must share a datacenter.** Create a volume only after confirming stock there. An empty volume is cheap to recreate elsewhere; a full one means re-downloading the whole stack.
2. **Not every datacenter supports network volumes.** A rejected create returns an error body listing the supported ones — read it rather than guessing.
3. **`SUPPLY_CONSTRAINT` on deploy** means that card is sold out in that datacenter. Don't retry-loop. Re-scan and switch card or datacenter; keep a fallback tier ready (e.g. L40S ⇄ RTX 6000 Ada ⇄ RTX A6000 at 48 GB).
4. **Phantom stock.** The availability API can report `available: true` while every deploy still fails — the inventory snapshot runs ahead of the scheduler, and you're racing the whole market. Scanning all datacenters, not just the volume's, is the fix; a volume that pins you to a saturated datacenter is usually the root cause.
5. **SSH "Permission denied (publickey)"** after deploy means the key was registered too late. Register it, terminate, redeploy — cheaper than fighting the running pod.
6. **Python `urllib` can get a Cloudflare 403 (`error code: 1010`)** on a request curl completes fine with the same key. Read the response body before assuming the key is bad or rotating it.
7. **The `runpod/pytorch` image's system Python already has a driver-matched torch.** A fresh venv with pip-installed torch can fail with `The NVIDIA driver on your system is too old`. Run the service with `/usr/bin/python3`; don't reinstall torch to fight it.
8. **`pkill -f 'main.py'` over SSH kills your own SSH command** because the command line contains the pattern. Use a self-excluding pattern: `pkill -f 'ComfyUI/[m]ain.py'`.
9. **Forgotten pods are the real cost.** An idle 48 GB pod left running overnight after the last job finished burned ~17 hours of compute for nothing. Never end a turn with a live pod unless more work is explicitly queued. Cheap insurance: a `cronjob` that terminates any pod with uptime above your longest legitimate job.
10. **Hermes approval prompts can expire while the meter runs.** SSH/scp to a raw IP may raise a security-approval prompt that times out if the user isn't watching. Before flagged commands on a paid pod, ask the user to be at the screen; if one is blocked, re-run it with them present — do not work around it.
11. **Failed deploys still bill.** A false-start pod that runs a few minutes costs a few cents. Include it in the spend report.
12. **Prices change.** As of September 2026: L40S ~$0.99/hr, RTX 6000 Ada ~$0.84/hr, network volume $0.07/GB/month (under 1 TB), container disk $0.10/GB/month. 24 GB cards are false economy for ~50 GB stacks (offloading eats the savings). Check https://docs.runpod.io/pods/pricing before quoting numbers to the user.
13. **Stopping is not terminating.** A stopped pod still bills for its disk, and a network volume bills until deleted. Delete volumes the user no longer needs.

## Verification

- The pod query returns `null` after termination.
- `myself { clientBalance }` dropped by roughly `costPerHr × hours used` — anything more means something is still running; list pods and volumes to find it.
- Every expected output file exists locally with non-zero size.
