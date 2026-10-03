# Security Recon Assistant

A professional, scope‑guarded reconnaissance tool for security audits.

## Usage

```bash
python -m security_recon_assistant --target scanme.nmap.org --scope scope.yaml --output-format json --output report.json
```

See `SKILL.md` for Hermes integration.

## Key operational examples

### 1) Basic scoped recon

```bash
python -m security_recon_assistant \
  --target scanme.nmap.org \
  --scope scope.yaml \
  --output-format json \
  --output recon-report.json
```

### 2) Multi-target run

```bash
python -m security_recon_assistant \
  --target scanme.nmap.org \
  --target example.com \
  --scope scope.yaml \
  --output-format html \
  --output recon-report.html
```

### 3) Verbose troubleshooting

```bash
python -m security_recon_assistant \
  --target scanme.nmap.org \
  --scope scope.yaml \
  --verbose --log-level DEBUG
```
