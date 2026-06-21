<p align="center">
  <img src="assets/banner.png" alt="Hermes AgentCyber" width="100%">
</p>

# Hermes AgentCyber ☤
<p align="center">
  <a href="https://hermes-agent.nousresearch.com/">Hermes Agent</a> | <a href="https://hermes-agent.nousresearch.com/">Hermes Desktop</a>
</p>
<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="Documentation"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/NousResearch/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://nousresearch.com"><img src="https://img.shields.io/badge/Built%20by-Nous%20Research-blueviolet?style=for-the-badge" alt="Built by Nous Research"></a>
  <a href="README.zh-CN.md"><img src="https://img.shields.io/badge/Lang-中文-red?style=for-the-badge" alt="中文"></a>
  <a href="README.ur-pk.md"><img src="https://img.shields.io/badge/Lang-اردو-green?style=for-the-badge" alt="اردو"></a>
  <a href="README.es.md"><img src="https://img.shields.io/badge/Lang-Español-orange?style=for-the-badge" alt="Español"></a>
</p>

**Hermes AgentCyber is a cybersecurity-focused fork of [Hermes Agent](https://github.com/NousResearch/hermes-agent) — the self-improving AI agent by [Nous Research](https://nousresearch.com).** It adds a native cybersecurity toolkit (threat intelligence, IOC extraction, vulnerability triage, incident response) and a live USB system that boots any x86-64 PC into an autonomous cyber operations environment. The base agent's learning loop, multi-platform gateway, and subagent delegation are all intact.

Use any model — [Nous Portal](https://portal.nousresearch.com), [OpenRouter](https://openrouter.ai) (200+ models), OpenAI, Anthropic, Bedrock, or your own endpoint. Switch with `hermes model` — no lock-in.

## Standalone AgentCyber quickstart

AgentCyber should be operated as its own runtime from this checkout, not by
turning the default `~/.hermes` assistant into a cyber agent. The repo-local
wrapper uses this fork's code and a dedicated `.agentcyber-home` by default:

```bash
cd /home/kbun/Desktop/hermes-agentcyber
scripts/agentcyber setup --apply
scripts/agentcyber status --json
scripts/agentcyber                  # interactive AgentCyber chat
scripts/agentcyber chat -q "Summarize the AgentCyber runtime boundary."
scripts/agentcyber hermes config path
```

Acceptance checks:

- `scripts/agentcyber hermes config path` resolves under `.agentcyber-home`,
  not default `~/.hermes`.
- `scripts/agentcyber status --json` reports Cyber routing enabled, local/open
  weight routing configured, built-in BC assets loaded, `cyber` enabled, and
  `live_usb` disabled.
- No external shell alias, gateway service, cron job, push, deploy, or default
  Hermes profile/config change is required for the standalone CLI runtime.

Full operator runbook: [`docs/AGENTCYBER_STANDALONE_RUNBOOK.md`](docs/AGENTCYBER_STANDALONE_RUNBOOK.md).

<table>
<tr><td><b>Cybersecurity toolkit</b></td><td>Native tools for CVE lookup (NVD), IOC reputation (VirusTotal, AbuseIPDB), MITRE ATT&CK TTP lookup, IOC extraction from logs/reports, CVSS+EPSS vulnerability triage, and an in-session incident response playbook. All in the <code>cyber</code> toolset — no wiring required.</td></tr>
<tr><td><b>Live USB — plug in and act</b></td><td>Builds on <b>Kali Linux Rolling</b> — the full offensive/defensive toolkit (nmap, Metasploit, Burp, sqlmap, Hydra, hashcat, Wireshark, aircrack-ng, and 300+ more) on x86-64 or ARM64. Boot any PC, complete a 60-second wizard, and the gateway starts. Add <code>--persistence 8G</code> to survive reboots; <code>HERMES_AUTOUPDATE=true</code> to update hermes on each boot.</td></tr>
<tr><td><b>SOC audit trail</b></td><td>Set <code>HERMES_CYBER_AUDIT=true</code> to get a tamper-evident NDJSON log of every tool call under the active AgentCyber home (for the repo-local wrapper, <code>.agentcyber-home/logs/cyber_audit.jsonl</code>). Credentials redacted. File mode 0600.</td></tr>
<tr><td><b>Protected subagent operations</b></td><td>In-flight subagents (parallel CVE triage, recon sweeps) are shielded from accidental interrupts — a follow-up message queues rather than killing running work. <code>/stop</code> still force-cancels everything.</td></tr>
<tr><td><b>A real terminal interface</b></td><td>Full TUI with multiline editing, slash-command autocomplete, conversation history, interrupt-and-redirect, and streaming tool output.</td></tr>
<tr><td><b>Lives where you do</b></td><td>Telegram, Discord, Slack, WhatsApp, Signal, and CLI — all from a single gateway process. Voice memo transcription, cross-platform conversation continuity.</td></tr>
<tr><td><b>A closed learning loop</b></td><td>Agent-curated memory with periodic nudges. Autonomous skill creation after complex tasks. Skills self-improve during use. FTS5 session search with LLM summarization for cross-session recall. <a href="https://github.com/plastic-labs/honcho">Honcho</a> dialectic user modeling. Compatible with the <a href="https://agentskills.io">agentskills.io</a> open standard.</td></tr>
<tr><td><b>Scheduled automations</b></td><td>Built-in cron scheduler with delivery to any platform. Daily reports, nightly backups, weekly audits — all in natural language, running unattended.</td></tr>
<tr><td><b>Delegates and parallelizes</b></td><td>Spawn isolated subagents for parallel workstreams — run simultaneous CVE triage, network scans, and log analysis. Write Python scripts that call tools via RPC, collapsing multi-step pipelines into zero-context-cost turns.</td></tr>
<tr><td><b>Runs anywhere, not just your laptop</b></td><td>Terminal backends include local, Docker, SSH, Singularity, Modal, Daytona, and Vercel Sandbox. Daytona and Modal offer serverless persistence; AgentCyber can also boot directly from the live USB on any x86-64 or ARM64 machine.</td></tr>
<tr><td><b>Research-ready</b></td><td>Batch trajectory generation, trajectory compression for training the next generation of tool-calling models.</td></tr>
</table>

---

## Cybersecurity Features

### Cyber Toolset

In the standalone runtime, enable the `cyber` toolset in the dedicated
AgentCyber config under `.agentcyber-home/config.yaml`, not in default `~/.hermes/config.yaml`,
to get four purpose-built security tools:

```yaml
toolsets:
  enabled: [cyber, web, terminal, file, delegation]
```

| Tool | What it does | APIs |
|---|---|---|
| `threat_intel` | CVE lookup, IOC reputation, MITRE ATT&CK TTP | NVD v2 (free), VirusTotal (`VT_API_KEY`), AbuseIPDB (`ABUSEIPDB_API_KEY`), ATT&CK TAXII (free) |
| `extract_iocs` | Extract IPs, domains, hashes, URLs, CVEs, emails from any text. Handles analyst defanging (`hxxp`, `[.]`, `[at]`). | None — local |
| `vuln_triage` | CVSS + EPSS exploitation probability + asset matching → CRITICAL/HIGH/MEDIUM/LOW priority | NVD v2 (free), EPSS first.org (free) |
| `ir_incident` | IR state machine: create incident, timeline, evidence log, markdown report | None — in-session |

**Example — triage a CVE from Telegram:**
```
"Is CVE-2024-3400 critical for my Palo Alto firewalls?"
→ vuln_triage correlates CVSS 10.0 + EPSS 94th percentile + asset match
→ Returns: CRITICAL — patch immediately (within 24 hours)
```

**Example — parse a phishing email:**
```
"Here's the raw email [paste]. What's in it?"
→ extract_iocs pulls IPs, domains, URLs, hashes
→ threat_intel checks each IOC's reputation in parallel via delegate_task
→ ir_incident opens a P2 incident and logs all IOCs as evidence
```

### Cybersecurity Skills

Three agent playbooks live in `skills/cybersecurity/` and are available as slash commands:

| Skill | What it does |
|---|---|
| `threat-intel` | Full TI synthesis workflow: extract → enrich in parallel → map to ATT&CK → brief |
| `ir-copilot` | End-to-end IR lifecycle: triage → investigation → containment → report |
| `vuln-triage` | CVE backlog prioritisation with bulk parallel triage pattern |

### SOC Audit Log

```bash
export HERMES_CYBER_AUDIT=true
scripts/agentcyber hermes gateway run
```

Every `agent:step` (tool name + redacted inputs + result preview) and `agent:end` (stop reason, token counts) is appended to the active AgentCyber home (for the repo-local wrapper, `.agentcyber-home/logs/cyber_audit.jsonl`) in NDJSON format. The file is created with mode `0600`. Rotate externally with `logrotate`.

### Optional API Keys

Add to the dedicated AgentCyber `.env` (for the repo-local wrapper,
`.agentcyber-home/.env`) — all cyber tools degrade gracefully when keys are
absent:

```bash
VT_API_KEY=your-virustotal-key        # IOC reputation via VirusTotal
ABUSEIPDB_API_KEY=your-abuseipdb-key  # IP reputation via AbuseIPDB
```

NVD, EPSS (first.org), and MITRE ATT&CK TAXII are all free with no API key required.

---

## Live USB

Build a bootable USB that turns any PC into a Hermes cyber operations node. The ISO is based on **Kali Linux Rolling**, which ships the full offensive and defensive toolkit pre-built: nmap, Metasploit, Burp Suite, sqlmap, Hydra, hashcat, Wireshark, aircrack-ng, John the Ripper, Gobuster, enum4linux, Impacket, and hundreds more — all maintained by Offensive Security.

Plug it in, pick a boot mode from the GRUB menu, and the agent starts — no install, no persistent traces on the host.

The examples below are manual operator procedures for an approved maintenance window. Inside AgentCyber, the `live_usb` tool is disabled by default: `status` and `list_usb` are read-only, while `build`, `write`, and `provision` require root plus exact operator approval via `HERMES_AGENTCYBER_LIVE_USB_APPROVAL`/`operator_approval`. `write` and `provision` additionally require verified removable Linux block-device metadata and pass a canonical `/dev/...` target to the scripts. Unattended cron lanes must not set the approval token, run `sudo`, build an ISO, write a USB, or provision media.

**Supported targets:** Any x86-64 PC (UEFI or legacy BIOS) and ARM64 boards (Raspberry Pi 4/5, ARM servers).

---

### Step 1 — Install build dependencies

**amd64 ISO (builds on any Debian/Ubuntu/Kali host):**

```bash
sudo apt-get install -y \
  debootstrap squashfs-tools xorriso \
  grub-efi-amd64-bin grub-pc-bin mtools dosfstools
```

**arm64 ISO (cross-compiles from an x86-64 host):**

```bash
sudo apt-get install -y \
  debootstrap squashfs-tools xorriso \
  grub-efi-arm64-bin mtools dosfstools \
  qemu-user-static binfmt-support
```

---

### Step 2 — Build the ISO

```bash
# Default: Kali Rolling + kali-linux-headless (~5 GB ISO)
sudo live-usb/build_iso.sh

# Smaller ISO — top 10 tools only (~3 GB)
sudo live-usb/build_iso.sh --kali-meta kali-tools-top10

# Full Kali including GUI tools (~8 GB)
sudo live-usb/build_iso.sh --kali-meta kali-linux-default

# ARM64 — EFI-only (Raspberry Pi 4/5, ARM servers)
sudo live-usb/build_iso.sh --arch arm64

# Debian base instead of Kali (no metapackage, lighter)
sudo live-usb/build_iso.sh --suite bookworm --mirror http://deb.debian.org/debian

# All options
sudo live-usb/build_iso.sh \
  --arch amd64 \                         # amd64 (default) or arm64
  --suite kali-rolling \                 # default; use bookworm for Debian
  --kali-meta kali-linux-headless \      # default Kali metapackage
  --output /tmp/hermes-kali.iso \
  --headless-scan \                      # enable auto-scan mode
  --verbose                              # show full debootstrap output
```

The build bundles the current repo into the ISO so the target PC needs no internet on boot. Build time is typically 20–40 minutes for `kali-linux-headless` (mirror-dependent).

**Kali metapackage sizes:**

| `--kali-meta` | What's included | ISO size |
|---|---|---|
| `kali-tools-top10` | nmap, Metasploit, Burp, sqlmap, Aircrack, Hydra, John, Wireshark, Responder, Maltego | ~3 GB |
| `kali-linux-headless` | All of the above + 100+ additional tools, no desktop | ~5 GB |
| `kali-linux-default` | Full default Kali install with XFCE desktop | ~8 GB |

---

### Step 3 — Write to USB

```bash
# Find your USB drive
sudo live-usb/write_usb.sh --list

# Basic write (will prompt for confirmation)
sudo live-usb/write_usb.sh --device /dev/sdb

# Write + SHA-256 verify
sudo live-usb/write_usb.sh --device /dev/sdb --verify

# Write with a persistence partition (changes survive reboots)
sudo live-usb/write_usb.sh --device /dev/sdb --persistence      # 4 GB (default)
sudo live-usb/write_usb.sh --device /dev/sdb --persistence 8G   # custom size

# Write + inject pre-configured credentials (skip first-boot wizard)
sudo live-usb/write_usb.sh --device /dev/sdb --provision ~/.hermes

# All at once
sudo live-usb/write_usb.sh \
  --iso live-usb/hermes-cyber-live.iso \
  --device /dev/sdb \
  --persistence 8G \
  --provision ~/.hermes \
  --verify \
  --yes                     # skip confirmation prompt (non-interactive)
```

> **Warning:** `--device` is written with `dd`. Double-check the path — all data on the target drive will be erased.

---

### Step 4 — Boot and configure

Plug the USB into the target PC. Select a boot mode from the GRUB menu (5-second timeout):

| Boot entry | What happens |
|---|---|
| **Gateway Mode** | First-boot wizard (60 seconds) → enter Telegram token + API key → `hermes-gateway.service` starts → send commands from your phone |
| **Terminal Mode** | First-boot wizard → interactive `hermes` shell on the console |
| **Headless Scan** | Skips wizard, reads provisioned config → autonomous network scan → CVE triage → saves report to `~/hermes-scan-<date>.md` |
| **Forensic — No Automount** | `noautomount noswap nopersistent` — safe for evidence collection; host drives are never touched |
| **With Persistence** | Same as Gateway Mode but all changes (config, logs, memory) survive reboots. Requires the `--persistence` partition created in Step 3. |

On **first boot**, the wizard runs once and creates `/home/hermes/.hermes/config.yaml`. All subsequent boots skip the wizard and start the gateway directly.

---

### Persistence partition

When you write with `--persistence`, the script appends an ext4 partition (labeled `HERMESPST`) after the ISO data. Selecting **"Hermes AgentCyber (with persistence)"** from GRUB activates live-boot's union overlay — every write goes to `HERMESPST` instead of the read-only squashfs. Your credentials, memories, logs, and incidents survive across reboots.

```
Drive layout after write --persistence 8G:
  p1+p2   ISO hybrid MBR/EFI data (from dd)
  p3      FAT32 HERMESCFG  256 MB  (config, created with --provision)
  p4      ext4  HERMESPST  8 GB    (persistence layer)
```

The persistence partition pre-creates `/home/hermes/.hermes/logs` so logs are immediately writable on first boot without needing root.

---

### Auto-update on boot

To keep Hermes current without rebuilding the ISO:

1. Enable auto-update in your config:

```bash
# On the running USB, edit the config file
hermes-live config

# Add or set this line:
HERMES_AUTOUPDATE=true
```

Or add it directly:

```bash
echo "HERMES_AUTOUPDATE=true" >> ~/.hermes/config.env
```

2. On the **next boot**, `hermes-autoupdate.service` runs before the gateway:
   - Checks for network connectivity
   - If source is bundled: runs `git pull` then reinstalls
   - If installed from PyPI: runs `uv pip install --upgrade hermes-agentcyber`
   - Logs to `~/.hermes/logs/autoupdate.log`
   - **Always exits 0** — a failed update never blocks the gateway from starting

---

### Pre-provision for fleet / ops deployment

Skip the first-boot wizard entirely by injecting credentials before shipping the USB:

```bash
# From a ~/.hermes config directory
sudo live-usb/provision.sh --usb /dev/sdb --config ~/.hermes

# From individual flags
sudo live-usb/provision.sh --usb /dev/sdb \
  --telegram-token "7xxx:AAA..." \
  --allowed-users "123456789,987654321" \
  --model-key "sk-ant-..." \
  --model-provider anthropic \
  --audit
```

The provisioned config is written to the `HERMESCFG` FAT32 partition. On first boot, the wizard detects it and starts the gateway automatically — no keyboard input needed.

---

### Orchestrate from inside the agent

The `live_usb` tool lets you build and write USBs from a chat message. Keep it
disabled in the standalone AgentCyber runtime unless the operator explicitly
approves the hardware lane, then enable it only inside the dedicated
AgentCyber home:

```bash
# In .agentcyber-home/config.yaml
platform_toolsets:
  cli:
    - cyber
    - live_usb

# Or from the standalone wrapper
scripts/agentcyber hermes tools enable live_usb
```

Then ask the agent:

```
"List my USB drives"              → live_usb(action="list_usb")
"Check if I can build"            → live_usb(action="status")
"Build an arm64 ISO"              → live_usb(action="build", arch="arm64",
                                             operator_approval="<matching one-time token>")
"Build a headless-scan ISO"       → live_usb(action="build", headless_scan=true,
                                             operator_approval="<matching one-time token>")
"Write it to /dev/sdb"            → live_usb(action="write", device="/dev/sdb",
                                             operator_approval="<matching one-time token>")
"Provision with my config"        → live_usb(action="provision", device="/dev/sdb",
                                             config=".agentcyber-home",
                                             operator_approval="<matching one-time token>")
```

The `list_usb` and `status` actions are safe read-only checks and need no root or approval token. `build`, `write`, and `provision` require the agent session to run as root plus an exact operator approval token; set `HERMES_AGENTCYBER_LIVE_USB_APPROVAL` only for the approved maintenance session and pass the exact same value as `operator_approval`. `write` and `provision` also reject targets unless Linux reports verified removable block-device metadata, then use the canonical `/dev/...` target rather than an operator-supplied alias. Root/sudo alone is not sufficient for these agent tool calls, and unattended cron lanes must not perform them.

---

### Commands on the running USB

```bash
hermes-live start          # start the gateway service
hermes-live stop           # stop the gateway service
hermes-live restart        # restart
hermes-live status         # show service status + last 20 log lines
hermes-live logs           # tail live logs (Ctrl+C to exit)
hermes-live scan 192.168.1.0/24  # ad-hoc cyber sweep → markdown report
hermes-live shell          # drop into interactive agent
hermes-live config         # edit ~/.hermes/config.yaml in $EDITOR
hermes-live version        # show installed version
```

---

## Quick Install

### Linux, macOS, WSL2, Termux

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

### Windows (native, PowerShell)

> **Heads up:** Native Windows runs Hermes without WSL — CLI, gateway, TUI, and tools all work natively. If you'd rather use WSL2, the Linux/macOS one-liner above works there too. Found a bug? Please [file issues](https://github.com/NousResearch/hermes-agent/issues).

Run this in PowerShell:

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

The installer handles everything: uv, Python 3.11, Node.js, ripgrep, ffmpeg, **and a portable Git Bash** (MinGit, unpacked to `%LOCALAPPDATA%\hermes\git` — no admin required, completely isolated from any system Git install). Hermes uses this bundled Git Bash to run shell commands.

If you already have Git installed, the installer detects it and uses that instead. Otherwise a ~45MB MinGit download is all you need — it won't touch or interfere with any system Git.

> **Android / Termux:** The tested manual path is documented in the [Termux guide](https://hermes-agent.nousresearch.com/docs/getting-started/termux). On Termux, Hermes installs a curated `.[termux]` extra because the full `.[all]` extra currently pulls Android-incompatible voice dependencies.
>
> **Windows:** Native Windows is fully supported — the PowerShell one-liner above installs everything. If you'd rather use WSL2, the Linux command works there too. Native Windows install lives under `%LOCALAPPDATA%\hermes`; WSL2 installs under `~/.hermes` as on Linux.

After installation:

```bash
source ~/.bashrc    # reload shell (or: source ~/.zshrc)
hermes              # start chatting!
```

### Troubleshooting

#### Windows Defender or antivirus flags `uv.exe` as malware

If your antivirus (Bitdefender, Windows Defender, etc.) quarantines `uv.exe` from the Hermes `bin` folder (`%LOCALAPPDATA%\hermes\bin\uv.exe`), this is a **false positive**. The file is Astral's `uv` — the Rust Python package manager Hermes bundles to manage its Python environment. ML-based antivirus engines commonly flag unsigned Rust binaries that download and install packages.

**To verify your copy is authentic:**

```powershell
# Install GitHub CLI if needed
winget install --id GitHub.cli

# Login to GitHub
gh auth login

# Run verification
$uv = "$env:LOCALAPPDATA\hermes\bin\uv.exe"
$ver = (& $uv --version).Split(' ')[1]
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$zip = "$env:TEMP\uv.zip"
Invoke-WebRequest "https://github.com/astral-sh/uv/releases/download/$ver/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip -UseBasicParsing
gh attestation verify $zip --repo astral-sh/uv
Expand-Archive $zip "$env:TEMP\uv_x" -Force
(Get-FileHash "$env:TEMP\uv_x\uv.exe").Hash -eq (Get-FileHash $uv).Hash
```

If attestation says "Verification succeeded" and the last line prints `True`, you're good.

**To whitelist Hermes:**
- **Windows Defender:** Run PowerShell as Admin → `Add-MpPreference -ExclusionPath "$env:LOCALAPPDATA\hermes\bin"`
- **Bitdefender:** Add an exception in the Bitdefender console (Protection > Antivirus > Settings > Manage Exceptions)
- Whitelist the **folder**, not the file hash — Hermes updates `uv` and the hash changes every version

For more context, see the upstream Astral reports: [astral-sh/uv#13553](https://github.com/astral-sh/uv/issues/13553), [astral-sh/uv#15011](https://github.com/astral-sh/uv/issues/15011), [astral-sh/uv#10079](https://github.com/astral-sh/uv/issues/10079).

---

## Getting Started

**1. Install**

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
source ~/.bashrc
```

**2. Configure your model and (optionally) the cyber toolset**

```bash
hermes setup        # interactive wizard: model provider, API key, gateway
```

Or set each piece individually:

```bash
hermes model        # pick LLM provider + model
hermes config set toolsets.enabled "[cyber, web, terminal, file, delegation]"
```

**3. Start chatting**

```bash
hermes              # interactive terminal UI
hermes gateway      # start Telegram/Discord/Slack gateway (take commands from your phone)
```

**Other useful commands:**

```bash
hermes tools        # list and toggle tools
hermes config set   # set any config value
hermes update       # update to the latest version
hermes doctor       # diagnose configuration issues
hermes claw migrate # import settings from OpenClaw
```

📖 **[Full documentation →](https://hermes-agent.nousresearch.com/docs/)**

---

## Skip the API-key collection — Nous Portal

Hermes works with whatever provider you want — that's not changing. But if you'd rather not collect five separate API keys for the model, web search, image generation, TTS, and a cloud browser, **[Nous Portal](https://portal.nousresearch.com)** covers all of them under one subscription:

- **300+ models** — pick any of them with `/model <name>`
- **Tool Gateway** — web search (Firecrawl), image generation (FAL), text-to-speech (OpenAI), cloud browser (Browser Use), all routed through your sub. No extra accounts.

One command from a fresh install:

```bash
hermes setup --portal
```

That logs you in via OAuth, sets Nous as your provider, and turns on the Tool Gateway. Check what's wired up any time with `hermes portal info`. Full details on the [Tool Gateway docs page](https://hermes-agent.nousresearch.com/docs/user-guide/features/tool-gateway).

You can still bring your own keys per-tool whenever you want — the gateway is per-backend, not all-or-nothing.

---

## CLI vs Messaging Quick Reference

Hermes has two entry points: start the terminal UI with `hermes`, or run the gateway and talk to it from Telegram, Discord, Slack, WhatsApp, Signal, or Email. Once you're in a conversation, many slash commands are shared across both interfaces.

| Action                         | CLI                                           | Messaging platforms                                                              |
| ------------------------------ | --------------------------------------------- | -------------------------------------------------------------------------------- |
| Start chatting                 | `hermes`                                      | Run `hermes gateway setup` + `hermes gateway start`, then send the bot a message |
| Start fresh conversation       | `/new` or `/reset`                            | `/new` or `/reset`                                                               |
| Change model                   | `/model [provider:model]`                     | `/model [provider:model]`                                                        |
| Set a personality              | `/personality [name]`                         | `/personality [name]`                                                            |
| Retry or undo the last turn    | `/retry`, `/undo`                             | `/retry`, `/undo`                                                                |
| Compress context / check usage | `/compress`, `/usage`, `/insights [--days N]` | `/compress`, `/usage`, `/insights [days]`                                        |
| Browse skills                  | `/skills` or `/<skill-name>`                  | `/<skill-name>`                                                                  |
| Interrupt current work         | `Ctrl+C` or send a new message                | `/stop` or send a new message                                                    |
| Platform-specific status       | `/platforms`                                  | `/status`, `/sethome`                                                            |

For the full command lists, see the [CLI guide](https://hermes-agent.nousresearch.com/docs/user-guide/cli) and the [Messaging Gateway guide](https://hermes-agent.nousresearch.com/docs/user-guide/messaging).

---

## Documentation

Base agent docs live at **[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)**. AgentCyber-specific features are documented in this repo under `skills/cybersecurity/` and `live-usb/`.

| Section | What's Covered |
|---------|---------------|
| [Quickstart](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart) | Install → setup → first conversation in 2 minutes |
| [CLI Usage](https://hermes-agent.nousresearch.com/docs/user-guide/cli) | Commands, keybindings, personalities, sessions |
| [Configuration](https://hermes-agent.nousresearch.com/docs/user-guide/configuration) | Config file, providers, models, all options |
| [Messaging Gateway](https://hermes-agent.nousresearch.com/docs/user-guide/messaging) | Telegram, Discord, Slack, WhatsApp, Signal, Home Assistant |
| [Security](https://hermes-agent.nousresearch.com/docs/user-guide/security) | Command approval, DM pairing, container isolation |
| [Tools & Toolsets](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools) | 40+ tools, toolset system, terminal backends |
| [Skills System](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills) | Procedural memory, Skills Hub, creating skills |
| [Memory](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory) | Persistent memory, user profiles, best practices |
| [MCP Integration](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp) | Connect any MCP server for extended capabilities |
| [Cron Scheduling](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron) | Scheduled tasks with platform delivery |
| [Context Files](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files) | Project context that shapes every conversation |
| [Architecture](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture) | Project structure, agent loop, key classes |
| [Contributing](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) | Development setup, PR process, code style |
| [CLI Reference](https://hermes-agent.nousresearch.com/docs/reference/cli-commands) | All commands and flags |
| [Environment Variables](https://hermes-agent.nousresearch.com/docs/reference/environment-variables) | Complete env var reference |
| **[Cyber Toolset](skills/cybersecurity/)** | **TI synthesis, IR co-pilot, vuln triage playbooks** |
| **[Live USB](live-usb/)** | **Build scripts, provision guide, boot modes** |

---

## Migrating from OpenClaw

If you're coming from OpenClaw, Hermes can automatically import your settings, memories, skills, and API keys.

**During first-time setup:** The setup wizard (`hermes setup`) automatically detects `~/.openclaw` and offers to migrate before configuration begins.

**Anytime after install:**

```bash
hermes claw migrate              # Interactive migration (full preset)
hermes claw migrate --dry-run    # Preview what would be migrated
hermes claw migrate --preset user-data   # Migrate without secrets
hermes claw migrate --overwrite  # Overwrite existing conflicts
```

What gets imported:

- **SOUL.md** — persona file
- **Memories** — MEMORY.md and USER.md entries
- **Skills** — user-created skills → `~/.hermes/skills/openclaw-imports/`
- **Command allowlist** — approval patterns
- **Messaging settings** — platform configs, allowed users, working directory
- **API keys** — allowlisted secrets (Telegram, OpenRouter, OpenAI, Anthropic, ElevenLabs)
- **TTS assets** — workspace audio files
- **Workspace instructions** — AGENTS.md (with `--workspace-target`)

See `hermes claw migrate --help` for all options, or use the `openclaw-migration` skill for an interactive agent-guided migration with dry-run previews.

---

## Contributing

We welcome contributions! See the [Contributing Guide](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) for development setup, code style, and PR process.

Quick start for AgentCyber contributors:

```bash
git clone https://github.com/breakingcircuits1337/hermes-agentcyber.git
cd hermes-agentcyber
./setup-hermes.sh     # installs uv, creates venv, installs .[all], symlinks ~/.local/bin/hermes
./hermes              # auto-detects the venv, no need to `source` first
```

Managed upstream-style installer path:

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
cd "${HERMES_HOME:-$HOME/.hermes}/hermes-agent"
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

Manual clone fallback (for throwaway clones/CI where you intentionally do not
want the managed install layout):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

**Cyber-specific tests** (no network, no root needed):

```bash
python -m pytest tests/cyber/ -v
```

---

## Community

- 💬 [Discord](https://discord.gg/NousResearch)
- 📚 [Skills Hub](https://agentskills.io)
- 🐛 [Issues](https://github.com/NousResearch/hermes-agent/issues)
- 🔌 [computer-use-linux](https://github.com/avifenesh/computer-use-linux) — Linux desktop-control MCP server for Hermes and other MCP hosts, with AT-SPI accessibility trees, Wayland/X11 input, screenshots, and compositor window targeting.
- 🔌 [HermesClaw](https://github.com/AaronWong1999/hermesclaw) — Community WeChat bridge: Run Hermes Agent and OpenClaw on the same WeChat account.

---

## License

MIT — see [LICENSE](LICENSE).

Base agent built by [Nous Research](https://nousresearch.com). AgentCyber fork by [breakingcircuits1337](https://github.com/breakingcircuits1337).
