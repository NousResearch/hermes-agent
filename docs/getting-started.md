# Get started with this Hermes fork

This project runs a personal AI employee on Railway. You talk to it through
Telegram, configure it in a web dashboard, and connect the accounts it needs for
your work. It can use a terminal, research the web, work with files and run
scheduled responsibilities. Hindsight keeps shared work memory across sessions.

**Joining an existing deployment?** Ask its administrator for the Telegram bot
link, send the bot a direct message, and have the administrator approve you in
**Settings → Access → People**. Then start with [your first tasks](#your-first-tasks).
Only administrators need the dashboard login or the Railway account.

**Setting up your own?** Follow the steps below. Use this repository's code and
Dockerfiles; the upstream installers in the main README install upstream Hermes.

## 1. Have these accounts ready

| Account or credential | Used for | Where you connect it |
| --- | --- | --- |
| GitHub | Your copy of this fork and automatic deployments | Railway |
| Railway | Hosting Hermes, Hindsight and PostgreSQL | Railway project |
| OpenAI account with Codex access | Chat, memory learning/reflection and images, subject to model access | Settings → Models → Connect Codex |
| Telegram bot token | Receiving and replying to messages | Settings → Service keys |
| OpenRouter key with credit | Memory embeddings/reranking and video analysis | Settings → Service keys |
| Parallel key | Web search and page extraction | Settings → Service keys |
| Browser Use Cloud key | Browser tasks | Settings → Service keys |

Parallel and Browser Use can be added when you need those capabilities. The
configured search and browser routes need their own keys. Codex login does not
pay for OpenRouter, Parallel, Browser Use or Railway. There is no automatic paid
API fallback for Codex inference.

## 2. Create the Railway project

Fork this repository into your GitHub account, including its current `main`
branch. Create an empty Railway project and use one environment and region for
three services named exactly **`postgres`**, **`hermes`** and **`hindsight`**.
The configuration uses these names for private networking.

Create the services and configure their variables and volumes before deploying.
If connecting GitHub starts a build immediately, cancel it until setup is complete.
Use one replica per service, disable sleeping, and use the **On Failure** restart
policy with **10 retries**. Keep the repository root as the build root for both
GitHub services.

| Service | Source | Persistent volume | Start command | Health check |
| --- | --- | --- | --- | --- |
| `postgres` | pgvector Docker image from the [deployment reference](../deploy/railway/README.md#services) | `/var/lib/postgresql/data` | Image default | No HTTP check |
| `hermes` | Your GitHub fork; Dockerfile `Dockerfile` | `/opt/data` | `/opt/hermes/docker/entrypoint-dispatch.sh sleep infinity` | `/api/health`, port `9119`, timeout `300` seconds |
| `hindsight` | Your GitHub fork; Dockerfile `deploy/railway/hindsight/Dockerfile` | Uses PostgreSQL | Image default; leave override blank | `/health`, port `8888`, timeout `300` seconds |

Hermes' start command must include the dispatcher: `sleep infinity` alone does
not start the dashboard or bot. The Hindsight build needs the whole repository
because it also copies the reference memory configuration.

### PostgreSQL variables

Create `postgres` from the pinned pgvector image, then add:

| Variable | Value |
| --- | --- |
| `POSTGRES_USER` | `hindsight` |
| `POSTGRES_DB` | `hindsight` |
| `POSTGRES_PASSWORD` | A new random password |
| `PGDATA` | `/var/lib/postgresql/data/pgdata` |

Use a hex password to avoid connection-string escaping. On macOS or Linux,
`openssl rand -hex 32` generates a suitable value. Generate a fresh value for
each password or secret below. Keep them in Railway or your password manager.

### Hermes variables

| Variable | Value |
| --- | --- |
| `PORT` | `9119` |
| `HERMES_DASHBOARD` | `1` |
| `HERMES_GATEWAY_BOOTSTRAP_STATE` | `running` |
| `HERMES_DASHBOARD_BASIC_AUTH_USERNAME` | Your chosen administrator username |
| `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD` | A new strong administrator password |
| `HINDSIGHT_API_KEY` | A new random secret for the memory API |
| `HINDSIGHT_INFERENCE_KEY` | A different random secret for private inference |

### Hindsight variables

Railway supports references to another service's variables. Enter these values
literally in the `hindsight` service; Railway resolves the `${{…}}` expressions:

| Variable | Value |
| --- | --- |
| `PORT` | `8888` |
| `HINDSIGHT_API_KEY` | `${{hermes.HINDSIGHT_API_KEY}}` |
| `HINDSIGHT_INFERENCE_KEY` | `${{hermes.HINDSIGHT_INFERENCE_KEY}}` |
| `DATABASE_URL` | `postgresql://${{postgres.POSTGRES_USER}}:${{postgres.POSTGRES_PASSWORD}}@postgres.railway.internal:5432/${{postgres.POSTGRES_DB}}` |

The two Hindsight keys are internal service secrets you generate yourself.
You do not need a Hindsight Cloud account or API key. Hindsight calls Hermes'
private inference service using your server's Codex login. It receives the
OpenRouter key from Hermes after you save it in the dashboard.

This setup uses PostgreSQL with pgvector for Hindsight. Hermes' own sessions,
configuration and files remain on its separate volume. Running all three services
in the same Railway project does not remove the database requirement.

### Deploy and open the dashboard

1. Deploy `postgres`, then `hermes`, then `hindsight`. The first Hermes build can
   take several minutes; it includes browser tools and transcription weights.
2. In Hermes **Settings → Networking**, generate a public HTTPS domain targeting
   port **9119**. Open `https://<your-domain>/settings` and sign in with the
   administrator credentials above.
3. Keep Hindsight, PostgreSQL and Hermes port **8879** private. Telegram uses
   polling and needs no incoming webhook domain. Port **8648** is only needed
   later for responsibility webhooks; see the [deployment reference](../deploy/railway/README.md).
4. Enable daily and weekly backups for both persistent volumes.

A passing health check means the service has started. Complete the model and
messaging checks below before treating it as ready for work.

## 3. Connect models and service keys

In **Settings → Models → Connect Codex**, open the sign-in link and complete the
device-code flow. Wait for **Signed in on this server**. This stores the login on
the Hermes volume; do not copy auth tokens from your laptop.

Refresh the model list and select a chat model your account can use. Chat and
memory have separate selectors. The checked-in defaults are:

| Capability | Default route |
| --- | --- |
| Chat | `gpt-6-astra` through Codex |
| Memory learning/reflection | `gpt-5.6-luna` through Codex; low learning effort, medium recall effort |
| Memory embeddings/reranking | OpenRouter: `qwen/qwen3-embedding-8b` and `cohere/rerank-4-fast` |
| Web search/extraction | Parallel; search defaults to its `agentic` mode, with no user-selected LLM |
| Browser | Browser Use Cloud |
| Images | `gpt-image-2-high` through Codex |
| Video analysis | OpenRouter: `google/gemini-3.1-flash-lite` |
| Voice transcription | Local multilingual Whisper `base`; no transcription API key |

Model names in the configuration are defaults, not a guarantee of access on
every account. If a model is unavailable, choose an available compatible model
in **Models** and test it. A working chat login alone does not prove image or
memory-model access.

In **Service keys**, paste your OpenRouter key and click **Check and save**.
Add Parallel and Browser Use keys the same way. Save these keys here rather than
also defining them as Railway variables. Each save automatically restarts the
gateway; allow it to reconnect before testing. The Parallel key check performs
one search.

Under **Models → Memory**, wait for status **ready**. Changes to its model,
efforts or OpenRouter key apply to the separate Hindsight service automatically.
No second Codex login or provider-key copy is needed there. Memory processing is
asynchronous; validate actual learning and recall as well as service health.

## 4. Connect Telegram

1. Open [@BotFather](https://t.me/BotFather) in Telegram. Send `/newbot`, choose a
   name and username, and copy the bot token.
2. Save it in **Settings → Service keys → Telegram bot**. The key check identifies
   the bot, and the gateway restarts automatically.
3. Wait for the dashboard sidebar to say **Online**. Open your bot in Telegram
   and send a **fresh direct message**.
4. Refresh **Settings → Access → People** and click **Approve** on your request.
5. Send another message, such as “Hello, tell me what you can help me with.”
   Confirm you receive a model-generated reply.

Each additional person follows the same request-and-approval flow. You can also
add someone by numeric Telegram user ID. Bot access and dashboard administrator
access are separate; ordinary bot users do not need the administrator password.

### Groups and topics

Add the bot to a Telegram group, then use **Access → Add group** with its numeric
group ID (usually `-100…`). For a supergroup message link such as
`https://t.me/c/1234567890/42`, the group ID is `-1001234567890`.
Allowing a group lets everyone in that group talk to Hermes. Approved people can
also talk to it in any group the bot is in.

Start with **When mentioned**. Choose **Every message** if the bot should respond
without being addressed. Topics appear after the bot receives messages in them;
expand the group to select **Same as group**, **Every message** or **Silent** and
to add group instructions. These edits restart Telegram automatically.

For ordinary group messages to reach the bot, disable privacy mode with
BotFather's `/setprivacy`, then remove and re-add the bot to the group. Telegram
also delivers these messages to group-admin bots. Visibility and reply policy
are separate: receiving a message does not necessarily trigger a reply.

## 5. Set up how it works for you

In **Profile**, select your timezone before creating schedules. Add instructions
about your work, preferred output and when to ask before acting. Timezone changes
restart the gateway; instruction changes apply from the next conversation.
Use `/new` in Telegram when you want to start a fresh conversation.

The dashboard uses a shared administrator account. You can change its username
and password in **Admin login**; saved changes take precedence over the initial
Railway login variables and sign out existing sessions.

### Your first tasks

Try a small task for each capability you connected:

- “Search the web for current information about X and include source links.”
- Send a voice note and ask for a summary.
- “Create a simple image of a blue bicycle.”
- “Open example.com in the browser and tell me its heading.”
- Explain a real project decision, then ask about it in a later conversation to
  check memory. Hindsight intentionally filters out throwaway memory-test facts.

To connect work accounts, ask “Help me connect Google Workspace”, “Connect my
GitHub account”, or describe the service and account you need. Hermes has
[connection guides](../guides/connections/guide.md) for Google Workspace, email,
GitHub and MCP. Complete the requested authorization, then have it verify access
with a harmless read. These accounts are not connected just because deployment
finished. Enter reusable secrets through credential setup, not chat messages.

For ongoing work, describe the responsibility, schedule, destination and authority:
“Every weekday at 9am, review our project issues and send a summary here. Ask me
before changing any issue.” Have Hermes confirm the schedule and delivery with
a test run. It maintains the responsibility and connection instructions on its
persistent volume.

## Updates, storage and troubleshooting

Connect both GitHub services to **your fork's `main` branch**, with automatic
deployments enabled. Leave Hermes watch paths empty so all repository changes
qualify. Set Hindsight watch paths to:

```text
/deploy/railway/hindsight/**
/docs/specs/reference/hindsight-config.json
/.dockerignore
```

Pushes or merges to your `main` deploy Hermes; pushes to other branches do not.
Hindsight skips changes outside its watch paths. Changes in the original fork
only reach your deployment after you bring them into your own `main`. Keep
PostgreSQL on its pinned image and plan database upgrades separately. Update
the deployed application through GitHub and Railway.

The `/opt/data` volume retains logins, keys, configuration, sessions, documents,
repositories and responsibilities. PostgreSQL retains Hindsight memory.
Redeploys reuse these volumes; deleting a volume deletes its state. The repository
`config.yaml` is a first-boot seed, so changing it in Git does not overwrite saved
settings. Test restoring both backups into an isolated project before relying on
them for recovery. Browser login persistence follows your Browser Use account's
behavior and should be tested separately.

| Symptom | What to check |
| --- | --- |
| No pending Telegram request | Wait for Online, send a new DM, refresh Access. An already approved person appears under People. Check that only this gateway is polling that bot token. |
| Bot is Online but cannot answer | Check Codex login and selected model access. Approval and Telegram connectivity do not verify inference. |
| Bot ignores group messages | Check group access, mention/topic policy and Telegram privacy settings. |
| Memory stays pending or errors | Check the OpenRouter key/credit, memory model access, matching internal secrets and private database URL. Read Hindsight logs in Railway. |
| Search or browser fails | Check its own key and account balance in Service keys. |
| Dashboard does not start | Check the full Hermes start command, port 9119 and Railway startup logs. |
| Push did not deploy | Check the connected repository/branch, GitHub access and watch paths in Railway. |

For operator details and deeper acceptance checks, see the
[deployment reference](../deploy/railway/README.md). For this fork's behavior and
boundaries, see the [employee specification](specs/employee.md).
