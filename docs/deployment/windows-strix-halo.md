# Running Hermes Agent on Windows + AMD Strix Halo — field notes

> **Field notes, not official docs.** Every item below was hit in practice
> running Hermes Agent **v0.15.1** natively on Windows 11 with a Ryzen AI MAX+ 395
> (Strix Halo, 128 GB unified memory), Ollama, and a llama.cpp server. Numbers are
> from one setup — treat them as directional, not guarantees. Hermes-behavior
> claims are version-specific (v0.15.1); verify against your version. Where a known
> upstream issue or PR exists, it's cited inline.

This is operational knowledge for two audiences the existing docs don't cover
well: people running the **gateway as an always-on Windows service**, and people
running Hermes on an **AMD unified-memory iGPU (Strix Halo)**.

---

## Part 1 — Running the gateway as a Windows service (LocalSystem)

To get an always-on agent that survives reboot-without-login, the gateway is
run as a Windows service (NSSM, or a Task Scheduler task). That puts the gateway
process in **session 0 under a service account** (e.g. LocalSystem), which has
several non-obvious consequences:

### 1.1 LocalSystem Python sees only the system site-packages

A service running as LocalSystem uses the machine-wide Python on the system
`PATH`, and it sees **only the system `site-packages`** — *not* a user's
`--user` site. So skill/plugin Python dependencies installed with a normal
`pip install --user` are invisible to the gateway.

Worse: if you then run an elevated `pip install <dep>` while a `--user` copy
exists, pip reports **"Requirement already satisfied"** and installs nothing
system-wide. The fix is to ignore the user site explicitly:

```powershell
# Install where the LocalSystem gateway can actually see it:
python -s -m pip install <dep>     # -s ignores the per-user site-packages
```

### 1.2 `~/.hermes` resolves to the wrong profile under LocalSystem

When the gateway runs as LocalSystem, `USERPROFILE` points at
`C:\Windows\System32\config\systemprofile`, so anything that resolves
`~/.hermes` (config, skills, memory, `state.db`) silently points at the wrong
directory — recall/skills/memory appear "empty" on the messaging platform while
working fine from the interactive CLI.

Fix: set **`HERMES_HOME`** in the service environment to the intended user's
`.hermes` dir. Hermes' `get_hermes_home()` honors it. Any tool a skill shells
out to should resolve the home dir the same way (env-first), not by re-expanding
`~`.

### 1.3 Session-0 isolation: no GUI, no browser, no interactive desktop

A session-0 service can't pop a GUI, can't open a browser for an **interactive
OAuth consent flow**, and can't screenshot the interactive desktop. For Hermes
that bites the Google-Workspace OAuth setup and any screenshot/GUI skill.

Pattern that works: pre-register a **Task Scheduler task that runs in the
interactive user's context** and have the service trigger it (`schtasks /Run`
wrapped from the service). The task runs in the real desktop session where the
browser/GUI is available, and writes its result somewhere the service can read.

### 1.4 Scheduled helpers that read user creds must run as the *user* (S4U), not SYSTEM

If a scheduled task needs the user's `APPDATA`/`USERPROFILE` (e.g. an `rclone`
remote config, a stored token), run it as the **user with S4U** logon
(`Register-ScheduledTask -LogonType S4U`), never as SYSTEM — under SYSTEM those
paths resolve to the systemprofile and the task fails *silently*, logging into a
hidden systemprofile AppData you'll never look at.

Gotcha: creating an S4U task via `schtasks.exe /np` **hangs** in a
non-interactive shell. Use the PowerShell `Register-ScheduledTask -LogonType
S4U` cmdlet instead.

### 1.5 PowerShell 5.1 is what Task Scheduler runs

Task Scheduler invokes `powershell.exe` = **Windows PowerShell 5.1**, not
PowerShell 7. .NET 5+ APIs silently fail there (exit 1, no output) — e.g.
`[System.Globalization.ISOWeek]`. Use the 5.1-available equivalent
(`Calendar.GetWeekOfYear(..., FirstFourDayWeek, Monday)`).

### 1.6 ffmpeg must be on the *service's* PATH for voice

Local STT/TTS (faster-whisper, piper) need `ffmpeg`. The service inherits the
**machine** `PATH`, not your user `PATH`, so an ffmpeg you added to the user
PATH won't be found. Put it on the machine PATH (or point Hermes at an absolute
binary).

---

## Part 2 — Hermes specifics that bite on Windows

### 2.1 Cron `--script` dispatches by file extension

The cron scheduler (as of v0.15.1) routes `.sh`/`.bash` to bash and **everything
else to Python** (`sys.executable`). On a WSL-less Windows host, a `.cmd`/`.bat`/`.ps1`
cron script is fed to the Python interpreter and crashes. Either write the cron
payload as a `.py` that `subprocess`-shells to your command, or — better for
periodic + always-on work on Windows — use **Task Scheduler / a service** and
reserve Hermes crons for agent-internal cadence.

### 2.2 The agent's file tools don't expand environment variables

`write_file` / `read_file` / `patch` use `pathlib` directly — they do **not**
expand `$USERPROFILE`, `%USERPROFILE%`, or `~`. Passing a bash-style path
creates a phantom literal folder. Pass absolute Windows paths.

### 2.3 `.bat`/`.cmd` written by the agent need CRLF

Files written via the agent's file tools default to LF line endings; `cmd.exe`
misparses an LF-only batch file (we saw it drop into an interactive `time`
prompt). Normalize batch/cmd files to CRLF after writing.

### 2.4 `MEDIA:` file delivery and unquoted Windows paths

In **v0.15.1**, the `MEDIA:<path>` delivery convention (used by the `send_message`
tool and the gateway send path) only matched a **quoted** path or a `/`-rooted
path; an **unquoted Windows drive-letter path** (`MEDIA:C:/Users/.../x.png`)
failed to match and leaked into the chat as literal text — for *all* file types.
Workaround: emit the path quoted (`MEDIA:"C:/..."`). Upstream has since refactored
the extraction path on `main` (post-v0.15.1) and has open PRs adding Windows path
support (e.g. #34021, #32735, #26368, #26098) — so verify your version before
relying on the workaround.

---

## Part 3 — Strix Halo / Ryzen AI MAX+ 395 (unified-memory iGPU)

The Strix Halo APU exposes a large slice of the 128 GB unified memory to the
iGPU. The dominant performance fact is that this is a **memory-bandwidth-bound**
inference target, not a compute-bound one. That reframes most model choices.

### 3.1 Prefer MoE over dense

A ~30B **MoE** (e.g. a 30–35B-A3B model) runs several times faster than a dense
≥27B model here, because MoE activates only a few billion params per token and
the bottleneck is bandwidth. In our testing a 35B-A3B MoE was ~3x the throughput
of a dense 27B at equal quant. Default to MoE for interactive use.

### 3.2 Use a Q4_K-class quant, not BF16

BF16 erases the MoE advantage: in our testing BF16 ran ~10 t/s vs ~48 t/s for
Q4_K_XL on the same MoE model (~4.8x slower), which matches the bandwidth math.
A Q4_K_XL-class quant is the right operating point on this hardware.

### 3.3 MTP draft can crash on some MoE builds

Multi-token-prediction (speculative draft) crashed for us on a MoE build (a
35B-A3B) — not on all builds, but if you enable MTP and hit crashes on Strix
Halo, disable it; the throughput hit is usually acceptable on bandwidth-bound
hardware.

### 3.4 The scarce resource is *system* RAM, not VRAM

A large BIOS GPU carveout (we ran 96 GB of 128 GB to the iGPU) leaves a **small
Windows-visible system-RAM pool** (~31.6 GB in our case). That small pool — and
the Windows **commit limit** — is the constraint to watch, not VRAM.

- **Keep the pagefile system-managed.** With a small RAM pool the commit limit
  is load-bearing; a fixed/undersized pagefile thrashed model loads for us.
- **`--no-mmap` frees system RAM.** With full GPU offload *and* default mmap,
  llama.cpp pins a redundant **host copy** of the GGUF in that scarce system-RAM
  pool. Adding `--no-mmap` freed ~21 GB → ~2 GB of system RAM for us with no
  throughput cost (the weights are on the GPU anyway).

### 3.5 Throughput is depth-bound, not window-bound

Generation t/s falls with conversation **depth** (KV occupancy), not with the
configured context size `-c`. A bigger context *window* is throughput-neutral at
equal depth — so size the window for your longest conversation without fear, but
expect slowdown as a single conversation grows long.

### 3.6 Watch concurrent-model over-commit

Running several large models concurrently (a coder model + a chat model + an aux
model) can exceed the addressable VRAM and send the stack into a crash/restart
loop. Stagger loads or stop idle models (`ollama stop`). A latency-drift
**watchdog** that restarts the inference server handles the slow-drift case
(handle/latency creep under sustained load); interactive workloads rarely hit it.

### 3.7 Aux-model constraints (Hermes-specific)

Hermes' internal auxiliary functions (compression, triage, etc.) need a
**non-thinking** model with **≥64K context** (observed on v0.15.1). Two traps:
- A *thinking* model served via Ollama's OpenAI-compatible `/v1` endpoint can
  return **empty content** (the reasoning goes to a field the aux path doesn't
  read). Use a non-thinking aux model, or its native endpoint.
- Ollama loads a model at its **full training context** if you don't cap it —
  which over-commits VRAM on a multi-model box. Cap aux context with a
  `Modelfile` `num_ctx`, and re-apply after every `ollama pull`.

### 3.8 NPU (XDNA2): not for LLM inference (yet)

The on-die NPU is not a usable LLM-inference target today. The realistic NPU use
on this platform is **Whisper STT via the Ryzen AI stack** — worth an experiment
if you want to offload speech-to-text off the iGPU, but treat it as experimental,
not a supported path.

---

## What this guide deliberately omits

Host-hardening choices (UAC behavior, elevated task runners, service ACLs) are
environment-specific security decisions, not portable advice — configure those
per your own threat model, not by copying someone else's lab.
