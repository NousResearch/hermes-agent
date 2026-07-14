# Authentication & Setup

## Install the CLI

```bash
curl -fsSL cli.inference.sh | sh
```

## Login

```bash
belt login
```

This opens a browser for authentication. After login, credentials are stored locally.

## Check Authentication

```bash
belt me
```

Shows your user info if authenticated.

## Environment Variable

For CI/CD or scripts, set your API key:

```bash
export INFSH_API_KEY=your-api-key
```

Get your key from https://inference.sh/settings/api-keys. The environment variable overrides the config file.

## Update CLI

```bash
belt update
```

Or reinstall:

```bash
curl -fsSL cli.inference.sh | sh
```

## Troubleshooting

| Error | Solution |
|-------|----------|
| "not authenticated" | Run `belt login` |
| "command not found" | Reinstall CLI or add to PATH |
| "API key invalid" | Check `INFSH_API_KEY` or re-login |
