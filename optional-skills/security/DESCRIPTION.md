# Security

Official security‑focused skills for secrets management, reconnaissance, and forensics.

## Skills in this category

- **`1password`** – Integrate 1Password CLI for secret retrieval and injection into commands. Supports service accounts, desktop app integration, and Connect server.

- **`security-recon-assistant`** – Modular reconnaissance tool with scope guardian. Integrates nmap, subfinder, nuclei, ffuf, sslscan, gowitness, whatweb. Generates JSON/HTML/Markdown reports. Enforces strict whitelist/exclusion before any scan.

- **`oss-forensics`** – Open‑source digital forensics toolkit (disk analysis, file carving, timeline creation).

- **`sherlock`** – Hunt down usernames across hundreds of social networks and platforms.

## Installation

Skills in `optional-skills/` are *not* bundled by default. Users can discover and install them via:

```bash
hermes skills browse security  # see available security skills
hermes skills install security-recon-assistant  # install without leaving the CLI
```

All optional skills ship with Hermes Agent but require explicit activation.

## Security notice

These skills interact with powerful system tools and sensitive data. Always:
- Verify the scope and authorization before running scans.
- Store secrets using Hermes' secret management (env vars, 1Password, etc.).
- Review generated reports before sharing.
