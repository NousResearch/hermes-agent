# Remote Runtime Bridge Pattern — VPS to Local Machine

## When to use

When the VPS (OpenClaw/CLOE stack) needs to offload a **bounded generation or execution task** to a **local machine** (Mac, workstation) that has specialized hardware/software unavailable on the VPS. Examples:

- Image generation (Draw Things, ComfyUI, Automatic1111) on a GPU-equipped Mac
- Video encoding on local hardware
- Local model inference on Apple Silicon
- USB/peripheral-bound operations
- Licensed software that runs only on the local machine

## Architecture

```
┌─────────────────────────────────┐    ┌─────────────────────────────────┐
│ VPS (orchestrator / server)     │    │ Local Machine (worker)          │
│                                 │    │                                 │
│  ┌───────────────────────┐      │    │  ┌───────────────────────┐      │
│  │  1. Job contract       │──────┼────┼─→│  4. Poll for jobs     │      │
│  │  (structured params,   │      │    │  │  5. Validate locally   │      │
│  │   no executable code)  │      │    │  │  6. Execute generation  │      │
│  └───────────────────────┘      │    │  │  7. Validate output      │      │
│                                 │    │  │  8. Upload result        │      │
│  ┌───────────────────────┐      │    │  └───────────────────────┘      │
│  │  2. Atomic job store   │←─────┼────┼─── (claim, status, result)    │
│  │  (queue + state +      │      │    │                                 │
│  │   lease management)    │      │    │  ┌───────────────────────┐      │
│  └───────────────────────┘      │    │  │  9. Fixed output dir    │      │
│                                 │    │  │  10. Cleanup            │      │
│  ┌───────────────────────┐      │    │  └───────────────────────┘      │
│  │  3. Asset store        │←─────┼────┼─── (image, checksum, meta)     │
│  │  (verify → persist →   │      │    │                                 │
│  │   serve via auth route)│      │    │                                 │
│  └───────────────────────┘      │    │                                 │
│                                 │    │                                 │
│  11. WebUI preview              │    │                                 │
│  (authenticated route only)     │    │                                 │
└─────────────────────────────────┘    └─────────────────────────────────┘
```

## Principle: Pull-based worker

**The VPS does not push commands to the local machine.** The local machine polls for available jobs. This avoids:

- Opening inbound ports on the local machine
- Maintaining stateful connections
- NAT/firewall traversal
- Exposing the local machine to network attacks

## Key components

### 1. Job contract (strict schema)

The VPS produces a **structured, validated JSON object** — never executable code.

```json
{
  "schema": "cloe_generation_job_v1",
  "job_id": "dtj_<id>",
  "prompt": "<bounded text>",
  "negative_prompt": "<optional bounded text>",
  "seed": 123456,
  "width": 1024,
  "height": 1024,
  "model": "Z Image Turbo 1.0 (6-bit)",
  "tool": "Draw Things",
  "status": "QUEUED"
}
```

**Enforce:**
- Exact supported model/tool only
- Bounded prompt length
- Allowed image dimensions
- Valid seed range
- **No shell metacharacters** in prompts (regex: `[\w\s,.:\-!?@%+=/()]+`)
- No file paths supplied by the VPS
- No arbitrary script body supplied by the VPS
- No remote URL supplied as execution target
- Freeze the resulting object to prevent mutation

### 2. Atomic job store (persistent queue)

Use tmp+rename atomic writes to persist job state. Each job has a lifecycle:

```
QUEUED → CLAIMED → GENERATING → GENERATED → UPLOADING → COMPLETED
                                                          → FAILED
                                                          → EXPIRED
                                                          → CANCELLED
                                                          → UNKNOWN_OUTCOME
```

**Required features:**
- Atomic writes (write to `.tmp`, `renameSync` to final path)
- Claim lease with expiry (stale claim recovery)
- Bounded store size (e.g. 500 jobs max)
- No duplicate generation hash (same parameters = reject unless terminal)
- No replay on terminal jobs
- Corruption-safe fail-closed (invalid JSON → error, don't overwrite)

### 3. Asset store (verify before persist)

When the worker returns a generated file:

1. **Reject symlinks** — `statSync().isSymbolicLink()`
2. **Validate MIME type from magic bytes**, not extension:
   - PNG: `89 50 4E 47 0D 0A 1A 0A`
   - JPEG: `FF D8 FF`
   - WebP: `52 49 46 46 xx xx xx xx 57 45 42 50`
3. **Compute SHA-256 checksum** after reading
4. **Verify declared format matches detected MIME**
5. **Check file size bounds** (e.g. 10MB max)
6. **Copy to canonical path** under a dedicated data directory per asset
7. **Write separate metadata JSON** next to the file

### 4. Pull-based worker (local machine)

```
Worker starts → Poll /job/next → {Job available?} → No = poll again
                                    ↓ Yes
                                 Claim job → Validate locally → Execute local tool
                                                                    ↓
                                    ┌─────────────────────────────────┐
                                    │  Output valid?                  │
                                    │  ├─ Yes → Upload result         │
                                    │  └─ No → Report failure         │
                                    └────────┬────────────────────────┘
                                             ↓
                                        Cleanup temp files → poll again
```

**Worker restrictions (enforce locally):**
- One job at a time (V1)
- Fixed output directory (no arbitrary paths)
- No `eval()` on any received data
- No network uploads of unrelated files
- Cleanup temp files after confirmed upload

### 5. Authentication and transport

**Authentication contract:**
- Dedicated scoped bridge credential (e.g. `CLOE_BRIDGE_API_KEY`)
- Must NOT reuse: ExoClick token, admin key, GitHub token, LAHB key, user password
- Transmitted only over protected channel
- Supported by `requireOperatorBridgeApiKey` middleware (falls back to admin key)

**Transport (pick one, in preference order):**
1. Existing Tailscale/private VPN (no new surface)
2. Existing authenticated reverse tunnel
3. Outbound HTTPS polling from local machine
4. SSH-based bounded invocation **only if safe command wrapper exists**

**Replay protection:**
Include in each request: `job_id`, `claim_id`, timestamp, nonce/idempotency key, payload hash.

## Known pitfalls

| Trap | Symptom | Fix |
|------|---------|------|
| **VPS sends prompt with shell metacharacters** | `; rm -rf /` in prompt reaches worker's command line | Validate with shell-safe regex in the job contract on the VPS side |
| **Worker accepts arbitrary output path** | Image written outside dedicated directory | Hardcode output directory on the worker, reject env overrides outside allowed list |
| **Worker uploads unrelated files** | Wrong image or metadata submitted | Scanned files must match job ID pattern, in the dedicated directory only |
| **Claim lease expires during long generation** | Two workers claim the same job | Lease expiry must be > generation timeout + safety margin |
| **Worker loses contact mid-generation** | Image may exist but job stuck in GENERATING | Inspect dedicated directory at job ID path; only upload if valid, report UNKNOWN_OUTCOME otherwise |
| **Duplicate generation from same parameters** | VPS creates multiple jobs for same creative | Check generation hash (computed from params) before accepting new job |
| **File extension doesn't match actual content** | .png file is actually a JPEG (or worse, executable) | Always validate MIME from magic bytes, not extension |

## Verification checklist

- [ ] Job contract validates all parameters before creating
- [ ] No shell metacharacters accepted in text fields
- [ ] Model/tool enforcement (only approved values)
- [ ] Atomic store writes (tmp+rename)
- [ ] Claim leases expire after timeout
- [ ] Stale claims automatically recoverable
- [ ] No duplicate generation accepted
- [ ] No replay on terminal job states
- [ ] Asset MIME validated from magic bytes
- [ ] Symlinks rejected
- [ ] Checksum computed and verified on upload
- [ ] All VPS endpoints require authentication
- [ ] Bridge credential is scoped and not reused from other services
- [ ] Worker runs one job at a time
- [ ] Worker output directory is fixed and validated
