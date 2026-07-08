## Activating the Hermes Development Environment

Before running Hermes locally, activate the Hermes-managed Python virtual environment and load the project's environment variables.

### 1. Activate the Hermes Virtual Environment

```bash
source ~/.hermes/venvs/hermes-dev/bin/activate
```

This activates the shared Python virtual environment managed by Hermes.

It ensures that:

- The correct Python interpreter is used.
- All Hermes dependencies are available.
- Commands such as `./hermes` use the correct packages instead of your system Python.

You should see your terminal prompt change to something similar to:

```text
(hermes-dev)
```

---

### 2. Load the Project Environment Variables

```bash
set -a
source .env
set +a
```

These commands load every variable defined in the project's `.env` file into your current shell session.

---

### Why this is necessary

Several Hermes components—including MCP servers—read credentials from environment variables.

For example, the ClickUp MCP server uses:

```text
CLICKUP_API_TOKEN
CLICKUP_TEAM_ID
```

Loading the `.env` file ensures these credentials are available whenever Hermes launches or starts external tools.

---

### Starting Hermes

Once the environment is ready:

```bash
./hermes
```

Hermes will launch using the correct Python environment and will have access to all configured environment variables.