---
name: security-recon-assistant
description: Modular reconnaissance tool with strict scope control, multiple output formats (JSON/HTML/Markdown), and safety guardian for pentests and bug bounty workflows.
version: 0.1.0
author: Mathis A, enhanced with Claude Code
license: MIT
metadata:
  hermes:
    tags: [security, recon, pentest, bug-bounty, nmap, subfinder, nuclei]
    category: security
setup:
  help: "Configure a scope.yaml file defining allowed_domains, excluded_domains, and max_depth. Never scan targets outside your authorized scope. Install dependencies with scripts/install_deps.sh."
---

# Security Recon Assistant

A professional, scope‑guarded reconnaissance tool for security audits. Integrates multiple scanners (nmap, subfinder, nuclei, ffuf, sslscan, gowitness, whatweb) with a **Guardian** that prevents out‑of‑scope scanning.

## When to use

Use this skill when you need to:
- Run targeted reconnaissance on an authorized perimeter
- Generate structured, triage‑ready findings
- Standardize a reproducible security scan workflow
- Enforce strict scope compliance (whitelist/exclusion, wildcard support)

## Requirements

**System dependencies** (scan binaries):
- `nmap` (network scanning)
- `subfinder` (subdomain enumeration)
- `nuclei` (vulnerability scanning)
- `ffuf` (web fuzzing)
- `sslscan` (TLS/SSL analysis)
- `gowitness` (web screenshotting)
- `whatweb` (technology fingerprinting)

**Python dependencies** (installed automatically if skill is installed via Hermes):
- `pydantic>=2.0`
- `pyyaml`
- `jinja2`
- `click>=8.0`

## Installation

### Automatic (recommended)

The skill includes an installation script that sets up system dependencies:

```bash
# Clone or place the skill in optional-skills/security/security-recon-assistant
cd optional-skills/security/security-recon-assistant
bash scripts/install_deps.sh
```

The script installs Go tools (`subfinder`, `nuclei`, `gowitness`, `ffuf`) and system packages (nmap, whatweb, sslscan) where possible.

### Manual Python install

```bash
# Install the package in editable mode
pip install -e . --break-system-packages

# Install dev dependencies for testing
pip install pytest pytest-mock --break-system-packages
```

### Verify installation

```bash
# Check all binaries are available
security-recon-assistant --check-deps

# Run the test suite (should show 120 passed)
pytest tests/ -v
```

## Configuration

1. Create a scope file from the template:
   ```bash
   cp templates/scope.example.yaml scope.yaml
   ```

2. Edit `scope.yaml` to define your authorized targets:
   ```yaml
   allowed_domains:
     - example.com
     - *.test.com
   excluded_domains:
     - admin.example.com
   max_depth: 2
   rate_limit: 50
   ```

**Important**: The Guardian blocks any target not matching the scope. Scope enforcement happens *before* any scan command is executed.

## Usage

### From the command line

```bash
# Basic single‑target scan
python -m security_recon_assistant \
  --target scanme.nmap.org \
  --scope scope.yaml \
  --output-format json \
  --output report.json

# Multiple targets, HTML output
python -m security_recon_assistant \
  --target scanme.nmap.org \
  --target example.com \
  --scope scope.yaml \
  --output-format html \
  --output report.html

# Verbose debugging
python -m security_recon_assistant \
  --target scanme.nmap.org \
  --scope scope.yaml \
  --verbose \
  --log-level DEBUG
```

### From Hermes (delegate_task)

```python
delegate_task(
  goal="Perform reconnaissance on scanme.nmap.org with strict scope control",
  context="""
    target: scanme.nmap.org
    scope_file: optional-skills/security/security-recon-assistant/templates/scope.example.yaml
    output_format: json
  """,
  toolsets=["terminal"]
)
```

## Output formats

- `json` – machine‑readable, suitable for further processing
- `html` – human‑readable report with summary statistics
- `markdown` – plain‑text report with tables

Reports include:
- Metadata (targets, scanners used, timestamp)
- Per‑scanner results with findings
- Scope information (for audit trail)

## How it works

1. **Guardian** validates every target against the scope file. Command‑line arguments, flags, and positional targets are extracted and checked.
2. **Pipeline** runs scanners sequentially or in parallel (configurable).
3. **Cache** (SQLite) stores previous results to avoid duplicate scans.
4. **Orchestrator** aggregates findings into a final `ScanResult` per scanner.
5. **Reporter** renders JSON/HTML/Markdown output.

## Verification

- The output file exists (`report.json` or `report.html`).
- Out‑of‑scope targets are blocked with `ViolationError`.
- Each finding is traceable to the scanner and command that produced it.

## Pitfalls

- **Scope too permissive** → higher operational risk and noise.
- **Missing binaries** → scanners marked as failed in the report.
- **Running unauthorized scans** → serious legal/ethical violation. Only scan what you own or have explicit permission to test.

## Development

The test suite covers unit tests, integration tests, and edge cases:
```bash
pytest tests/ -v
# 120 tests across:
# - CLI parsing and options
# - Scope Guardian (wildcards, IP ranges, case‑insensitivity)
# - Scanner implementations (Subfinder, Nmap, etc.)
# - Pipeline orchestration (sequential/parallel, retries, error handling)
# - Reporting generators (JSON, HTML, Markdown)
```

## Architecture Highlights

- **Modular scanners**: auto‑discovered in `security_recon_assistant/scanners/`. Add a new `*_scanner.py` implementing `BaseScanner` to extend.
- **Pydantic v2** models for validation and serialization.
- **SQLite cache** with TTL for result reuse.
- **Executor** with timeout handling and duration measurement.
- **Type‑hints** and comprehensive docstrings.
