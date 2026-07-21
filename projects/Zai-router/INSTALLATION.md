# Installation and application

These instructions assume a Windows source installation at the standard location:

```text
%LOCALAPPDATA%\hermes\hermes-agent
```

Run `git rev-parse --show-toplevel` from the checkout if you need to confirm the exact path on a particular machine.

## 1. Configure the credential

Add the key to the active Hermes profile's `.env` file:

```dotenv
ZAI_INDIRECT_API_KEY=your-zai-api-key
```

Do not commit the `.env` file or place the key in source code, tests, documentation, or shell history.

## 2. Refresh the provider catalogue

From Git Bash in the repository root:

```bash
.venv/Scripts/python.exe -m hermes_cli.main model --refresh
```

Select **Leave unchanged** if you only want to rebuild the catalogue.

## 3. Restart the gateway

```bash
.venv/Scripts/python.exe -m hermes_cli.main gateway restart
.venv/Scripts/python.exe -m hermes_cli.main gateway status
```

## 4. Refresh Hermes Desktop

1. Reopen the composer model picker.
2. Select **Refresh Models**.
3. If necessary, close and reopen Hermes Desktop.
4. Select **Z.ai Indirect → GLM-5.2**.

## 5. Direct CLI use

```bash
.venv/Scripts/python.exe -m hermes_cli.main chat \
  --provider zai-indirect \
  --model glm-5.2
```

Do not manually set the base URL or API mode for normal use. The provider profile supplies both.
