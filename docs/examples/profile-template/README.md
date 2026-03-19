# Profile Template

Copy this folder to `~/.hermes/profiles/<profile_name>/` to create a profile-scoped Hermes home.

Example:

```bash
mkdir -p ~/.hermes/profiles
cp -R docs/examples/profile-template ~/.hermes/profiles/alice
```

Then add a matching profile entry to `~/.hermes/config.yaml`:

```yaml
profiles:
  alice:
    users:
      telegram: ["123456789"]
      signal: ["+15551234567"]
    config:
      model: "anthropic/claude-sonnet-4.6"
      agent:
        reasoning_effort: "medium"
      isolation:
        enabled: true
      workspace: "/srv/workspaces/alice"
      messaging:
        disabled_slash_commands: ["model", "update"]
      platform_toolsets:
        telegram: ["web", "file", "memory", "skills"]
      tools:
        disabled_toolsets: ["browser"]
      skills:
        disabled: ["telephony"]
        platform_disabled:
          telegram: ["voice-mode"]
```

Minimal isolated setup:

```yaml
profiles:
  alice:
    users:
      telegram: ["123456789"]
    config:
      isolation:
        enabled: true
      messaging:
        disabled_slash_commands: ["model", "update"]
```

`workspace` is optional. If you omit it, the profile defaults to:

```text
~/.hermes/profiles/<profile_name>/workspace
```

When profile isolation is enabled, that workspace becomes the default base
directory for terminal/file access unless you explicitly set a different
isolation root.

To block messaging slash commands for a profile, set:

```yaml
messaging:
  disabled_slash_commands: ["model", "update", "my-skill"]
```

Disabled commands are blocked in messaging and hidden from `/help` for that profile.

What lives here:

- `SOUL.md` is this profile's personality and standing instructions.
- `memories/MEMORY.md` is long-term notes for this profile's Hermes.
- `memories/USER.md` is the user profile memory for this profile's owner.
- `workspace/` is the profile's default working directory.
- `cron/` stores this profile's scheduled jobs.
- `sessions/` stores this profile's conversation transcripts.
- `skills/` contains profile-private skills that are only visible to this profile.
- `state.db` will be created automatically when the profile starts using session search/history.

Shared global config still lives in `~/.hermes/config.yaml` and `~/.hermes/.env`. Global shared skills remain in `~/.hermes/skills/`.
