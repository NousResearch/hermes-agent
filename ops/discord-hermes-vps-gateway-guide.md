# Running Hermes Agent from Discord on a DigitalOcean VPS

This guide walks through setting up Hermes Agent so you can talk to it from a Discord channel. The final result is:

- A Discord bot in your server
- Hermes Agent running on a DigitalOcean VPS
- The Hermes gateway connected to Discord
- A specific Discord channel where Hermes can respond without needing to be mentioned
- Auto-threading in normal channels, while the free-response/home channel stays inline
- SSH access from your local machine so Hermes can help debug the server directly

---

## 1. Architecture

```text
Discord Server
   ↓
Discord Bot
   ↓
Hermes Gateway running on VPS
   ↓
Hermes Agent with tools, memory, terminal access, etc.
```

The Discord bot receives messages. The Hermes gateway listens for those messages, turns them into Hermes sessions, and sends Hermes' responses back into Discord.

---

## 2. Create a Discord Bot

Open the Discord Developer Portal:

```text
https://discord.com/developers/applications
```

Create the app:

1. Click **New Application**.
2. Give it a name, for example `Hermes`.
3. Open the application.
4. Go to **Bot**.
5. Click **Add Bot**.

### Enable Message Content Intent

This is critical. Without it, the bot can appear online but silently ignore messages.

In the bot settings, enable:

```text
Privileged Gateway Intents
  ✅ Message Content Intent
```

Optional but sometimes useful:

```text
  ✅ Server Members Intent
```

---

## 3. Copy the Discord Bot Token

In the Discord bot page:

1. Go to **Bot**.
2. Under **Token**, click **Reset Token** or **Copy**.
3. Save it somewhere safe temporarily.

You will use it later as:

```env
DISCORD_BOT_TOKEN=your_token_here
```

Do not share this token publicly.

---

## 4. Invite the Bot to Your Discord Server

In the Developer Portal:

1. Go to **OAuth2**.
2. Go to **URL Generator**.
3. Select scopes:

```text
✅ bot
```

4. Under bot permissions, select at least:

```text
✅ View Channels
✅ Send Messages
✅ Read Message History
✅ Create Public Threads
✅ Send Messages in Threads
```

For initial testing, you can temporarily give it Administrator, then reduce permissions later.

5. Copy the generated URL.
6. Open it in your browser.
7. Invite the bot to your server.

---

## 5. Get Your Discord User ID and Channel ID

You need:

- Your Discord user ID
- The Discord channel ID where Hermes should respond inline

### Enable Developer Mode in Discord

```text
User Settings → Advanced → Developer Mode
```

Turn it on.

### Copy your user ID

Right-click your own Discord profile and click:

```text
Copy User ID
```

Example:

```text
366361021725802506
```

### Copy the channel ID

Right-click the Discord channel where you want Hermes to live and click:

```text
Copy Channel ID
```

Example:

```text
1499931792378167467
```

---

## 6. Create a DigitalOcean VPS

Create a DigitalOcean Droplet.

Recommended starting point:

```text
Ubuntu 22.04 or 24.04
Basic Droplet
1–2 GB RAM minimum
SSH key login enabled
```

Once it is created, note the public IP address.

Example:

```text
64.227.27.57
```

---

## 7. Set Up a Local SSH Alias

On your local machine, edit:

```text
~/.ssh/config
```

On macOS, that might be:

```text
/Users/YOUR_USERNAME/.ssh/config
```

Add an entry like:

```sshconfig
Host hermes-01
  HostName YOUR_VPS_IP
  User YOUR_VPS_USER
```

Example:

```sshconfig
Host hermes-01
  HostName 64.227.27.57
  User mike
```

Now you can SSH in with:

```bash
ssh hermes-01
```

This is very useful because you can ask local Hermes to SSH into the VPS and debug logs/config directly.

---

## 8. Install Hermes on the VPS

SSH into the server:

```bash
ssh hermes-01
```

Install Hermes:

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

After installation, restart your shell if needed, then run:

```bash
hermes doctor
```

---

## 9. Configure Your Model Provider

Run:

```bash
hermes setup
```

or:

```bash
hermes model
```

Set up your model provider, such as:

- OpenRouter
- Anthropic
- OpenAI
- Nous
- Gemini
- etc.

API keys usually go in:

```text
~/.hermes/.env
```

---

## 10. Configure Discord Environment Variables

On the VPS, edit Hermes' env file:

```bash
nano ~/.hermes/.env
```

Add your Discord token and IDs:

```env
DISCORD_BOT_TOKEN=your_discord_bot_token_here

DISCORD_ALLOWED_USERS=your_discord_user_id
DISCORD_HOME_CHANNEL=your_discord_channel_id
DISCORD_FREE_RESPONSE_CHANNELS=your_discord_channel_id
```

Example:

```env
DISCORD_BOT_TOKEN=xxxxx.yyyyy.zzzzz

DISCORD_ALLOWED_USERS=366361021725802506
DISCORD_HOME_CHANNEL=1499931792378167467
DISCORD_FREE_RESPONSE_CHANNELS=1499931792378167467
```

Meaning:

- `DISCORD_ALLOWED_USERS` — only these Discord users can talk to Hermes.
- `DISCORD_HOME_CHANNEL` — default Discord channel for Hermes.
- `DISCORD_FREE_RESPONSE_CHANNELS` — channels where Hermes can reply without being mentioned.

---

## 11. Configure Hermes Discord Settings

Edit Hermes config:

```bash
hermes config edit
```

Or edit the file directly:

```bash
nano ~/.hermes/config.yaml
```

Working baseline:

```yaml
discord:
  require_mention: false
  free_response_channels: "YOUR_CHANNEL_ID"
  allowed_channels: ""
  auto_thread: true
```

Example:

```yaml
discord:
  require_mention: false
  free_response_channels: "1499931792378167467"
  allowed_channels: ""
  auto_thread: true
```

Important details:

### `require_mention: false`

Allows Hermes to respond in configured free-response channels without needing to be tagged.

### `free_response_channels`

This should include the channel ID where you want Hermes to freely respond inline.

### `allowed_channels: ""`

Leaving this empty avoids accidentally restricting Hermes to the wrong channel. Only set it if you intentionally want a whitelist.

### `auto_thread: true`

Allows Hermes to create/use threads in normal channels.

---

## 12. Install and Start the Hermes Gateway

Run:

```bash
hermes gateway install
```

Then start it:

```bash
hermes gateway start
```

Check status:

```bash
hermes gateway status
```

If you change config or `.env`, restart the gateway:

```bash
hermes gateway restart
```

---

## 13. Test in Discord

Go to the Discord channel you configured.

Try sending:

```text
hello
```

or:

```text
what tools do you have?
```

If everything is working, Hermes should reply.

---

## 14. Auto-Thread Behavior

Hermes auto-threading is intentionally skipped in free-response channels.

The effective logic is:

```python
skip_thread = bool(channel_ids & no_thread_channels) or is_free_channel
```

So behavior is:

- Normal server channel + mention → auto-thread
- DM → no thread
- Already inside a thread → no new thread
- Free-response channel → no auto-thread; Hermes replies inline
- `DISCORD_NO_THREAD_CHANNELS` channel → no auto-thread

This means if your home/inbox channel is configured as:

```env
DISCORD_FREE_RESPONSE_CHANNELS=1499931792378167467
```

or:

```yaml
discord:
  free_response_channels: "1499931792378167467"
```

then Hermes will respond inline there instead of creating a thread.

That is usually desirable for a dedicated `#inbox` or `#hermes` channel. Other channels can still auto-thread when you mention Hermes.

If you want a channel to auto-thread, do **not** include it in `free_response_channels`. You can still mention Hermes there to trigger a threaded conversation.

---

## 15. Debugging

### Check gateway logs

On the VPS:

```bash
grep -i "discord\|error\|failed" ~/.hermes/logs/gateway.log | tail -80
```

Or watch logs live:

```bash
tail -f ~/.hermes/logs/gateway.log
```

### Check gateway status

```bash
hermes gateway status
```

### Restart the gateway

```bash
hermes gateway restart
```

### Check config

```bash
hermes config
```

### Check env path

```bash
hermes config env-path
```

Then inspect:

```bash
nano ~/.hermes/.env
```

---

## 16. Common Problems

### Bot is online but never responds

Likely causes:

1. **Message Content Intent is not enabled**

Fix:

```text
Discord Developer Portal → Application → Bot → Privileged Gateway Intents → Message Content Intent
```

2. **Wrong channel ID**

Make sure the channel ID appears in:

```env
DISCORD_HOME_CHANNEL=
DISCORD_FREE_RESPONSE_CHANNELS=
```

and:

```yaml
discord:
  free_response_channels: "..."
```

3. **User ID not allowed**

Make sure your Discord user ID is in:

```env
DISCORD_ALLOWED_USERS=
```

4. **Gateway needs restart**

Run:

```bash
hermes gateway restart
```

---

### Hermes only responds when mentioned

Set:

```yaml
discord:
  require_mention: false
```

Also make sure your channel is listed in:

```env
DISCORD_FREE_RESPONSE_CHANNELS=your_channel_id
```

and:

```yaml
discord:
  free_response_channels: "your_channel_id"
```

---

### Hermes does not respond in the desired channel

A known-good pattern is:

```env
DISCORD_ALLOWED_USERS=your_user_id
DISCORD_HOME_CHANNEL=your_channel_id
DISCORD_FREE_RESPONSE_CHANNELS=your_channel_id
```

and:

```yaml
discord:
  require_mention: false
  free_response_channels: "your_channel_id"
  allowed_channels: ""
  auto_thread: true
```

Then restart:

```bash
hermes gateway restart
```

---

### Auto-thread works everywhere except the home/inbox channel

That is expected if the home/inbox channel is also a free-response channel.

Free-response channels skip auto-threading so Hermes can behave like a live inline chat there.

To make a channel auto-thread, remove it from:

```env
DISCORD_FREE_RESPONSE_CHANNELS=
```

and:

```yaml
discord:
  free_response_channels: "..."
```

Then restart the gateway.

---

### Gateway stops after SSH logout

Enable user service lingering:

```bash
sudo loginctl enable-linger $USER
```

Then restart the gateway:

```bash
hermes gateway restart
```

---

## 17. Useful Commands

```bash
# SSH into the VPS
ssh hermes-01

# Check Hermes health
hermes doctor

# Edit config
hermes config edit

# Show config
hermes config

# Show .env path
hermes config env-path

# Start gateway
hermes gateway start

# Restart gateway
hermes gateway restart

# Gateway status
hermes gateway status

# Watch logs
tail -f ~/.hermes/logs/gateway.log

# Search errors
grep -i "failed\|error\|discord" ~/.hermes/logs/gateway.log | tail -80
```

---

## 18. Minimal Working Config Example

`.env`:

```env
DISCORD_BOT_TOKEN=your_discord_bot_token

DISCORD_ALLOWED_USERS=your_discord_user_id
DISCORD_HOME_CHANNEL=your_discord_channel_id
DISCORD_FREE_RESPONSE_CHANNELS=your_discord_channel_id
```

`~/.hermes/config.yaml`:

```yaml
discord:
  require_mention: false
  free_response_channels: "your_discord_channel_id"
  allowed_channels: ""
  auto_thread: true
```

Then:

```bash
hermes gateway restart
```

---

## 19. Recommended Debugging Workflow

1. Add an SSH alias locally:

```sshconfig
Host hermes-01
  HostName YOUR_VPS_IP
  User YOUR_VPS_USER
```

2. Confirm you can connect:

```bash
ssh hermes-01
```

3. If Hermes is not responding in Discord, ask your local Hermes:

```text
ssh into hermes-01 and debug why the Discord gateway is not responding
```

Then Hermes can inspect:

```text
~/.hermes/.env
~/.hermes/config.yaml
~/.hermes/logs/gateway.log
systemd gateway status
```

This is much faster than manually copying logs and config back and forth.

---

## 20. Final Checklist

- [ ] Discord bot created
- [ ] Bot invited to server
- [ ] Message Content Intent enabled
- [ ] Bot has permission to read/send messages in the channel
- [ ] Bot has permission to create/send in threads if using auto-threading
- [ ] VPS created
- [ ] Hermes installed on VPS
- [ ] Model provider configured
- [ ] `DISCORD_BOT_TOKEN` set in `~/.hermes/.env`
- [ ] `DISCORD_ALLOWED_USERS` set
- [ ] `DISCORD_HOME_CHANNEL` set
- [ ] `DISCORD_FREE_RESPONSE_CHANNELS` set for the inline/home channel
- [ ] `discord.require_mention: false`
- [ ] `discord.free_response_channels` contains the inline/home channel ID
- [ ] `discord.allowed_channels: ""`
- [ ] `discord.auto_thread: true`
- [ ] Gateway installed and running
- [ ] Hermes responds in Discord
- [ ] Normal non-free-response channels auto-thread when Hermes is mentioned
- [ ] Free-response/home channel replies inline

---

## Known-Good Config Shape

```env
DISCORD_ALLOWED_USERS=YOUR_USER_ID
DISCORD_HOME_CHANNEL=YOUR_CHANNEL_ID
DISCORD_FREE_RESPONSE_CHANNELS=YOUR_CHANNEL_ID
```

```yaml
discord:
  require_mention: false
  free_response_channels: "YOUR_CHANNEL_ID"
  allowed_channels: ""
  auto_thread: true
```

The key detail is that `allowed_channels` is empty, while `free_response_channels` and `home_channel` point at the target Discord inline/home channel.
