# Advertising Agent Orchestration Profile Map

## 0. Why the new Agents do not appear as conversations immediately

- Profiles and conversations are different objects in Hermes Desktop.
- Creating a profile creates an isolated runtime identity/config directory, not a chat transcript.
- The left sidebar conversation list only shows sessions that already have messages.
- A newly created profile has no conversation until a chat is started under that profile.
- Profiles created from the CLI while Desktop is already open may require a profile-list refresh, route change, `/reset`, or app reload before the bottom-left profile rail shows them.
- The right file explorer is the current project folder (`apps/desktop`), not the Hermes runtime home. Runtime profile files live under `HERMES_HOME`, so their `SOUL.md` files do not automatically appear in the project file tree.

## 1. CAP — Capabilities

### CAP-1 Default advertising orchestrator

The default profile is configured as the Creative Director / Orchestrator. It owns brief analysis, task decomposition, expert routing, conflict resolution, and final packaging.

Runtime identity file:

```text
/tmp/hermes-agent-desktop-dev-home/SOUL.md
```

### CAP-2 爆款专家 profile

Profile name:

```text
baokuan-expert
```

Responsibility: hooks, user psychology, viral angles, selling-point hierarchy, short-video opening structure.

Runtime identity file:

```text
/tmp/hermes-agent-desktop-dev-home/profiles/baokuan-expert/SOUL.md
```

### CAP-3 广告文案专家 profile

Profile name:

```text
ad-copywriter
```

Responsibility: titles, ad copy, short-video voiceover, CTA, landing-page copy, A/B copy variants.

Runtime identity file:

```text
/tmp/hermes-agent-desktop-dev-home/profiles/ad-copywriter/SOUL.md
```

### CAP-4 生图助手 profile

Profile name:

```text
image-assistant
```

Responsibility: visual direction, image prompts, reference planning, product/character/scene consistency.

Runtime identity file:

```text
/tmp/hermes-agent-desktop-dev-home/profiles/image-assistant/SOUL.md
```

### CAP-5 视频专家 profile

Profile name:

```text
video-expert
```

Responsibility: storyboard, shot language, editing rhythm, Seedance/Dreamina prompts, audio-video synchronization.

Runtime identity file:

```text
/tmp/hermes-agent-desktop-dev-home/profiles/video-expert/SOUL.md
```

## 2. PAGE — Desktop surfaces

### PAGE-1 Chat sidebar

Shows existing conversations, not all profiles. A profile appears here only after it owns at least one conversation with messages.

### PAGE-2 Bottom-left profile rail

Shows selectable profiles. If profiles were created outside the UI while Desktop was open, this rail may need refresh/reload before it displays the new profile squares.

### PAGE-3 Manage Profiles page

The ellipsis button in the bottom-left profile rail opens profile management. This is the UI surface for editing each profile's `SOUL.md` from Desktop.

### PAGE-4 File explorer

The right-side file explorer is scoped to the current project folder. It will not show `/tmp/hermes-agent-desktop-dev-home/...` unless that path is opened as the workspace or copied into the repo.

## 3. BTN — Buttons / entries

### BTN-1 `+` in profile rail

Creates a new profile through the Desktop UI. It does not create a conversation by itself.

### BTN-2 Profile square

Switches the next/new chat to that profile and lazily connects that profile's backend.

### BTN-3 Ellipsis in profile rail

Opens profile management, including viewing/editing profile identity text.

### BTN-4 New chat

Creates a conversation for the currently selected profile. This is what makes a profile acquire visible chat history.

## 4. ACT — Actions

### ACT-1 Verify profiles from CLI

```bash
hermes profile list
```

### ACT-2 Start a manual expert chat

```bash
hermes -p baokuan-expert chat -q "为这个产品设计 10 个爆款前三秒钩子：..."
hermes -p ad-copywriter chat -q "基于这些卖点写 30 秒广告口播：..."
hermes -p image-assistant chat -q "为这个广告设计主视觉和生图 prompt：..."
hermes -p video-expert chat -q "把这个脚本转成 Seedance/Dreamina 分镜 prompt：..."
```

### ACT-3 Refresh Desktop visibility

Use one of:

```text
/reset
```

or reload/restart Hermes Desktop. Then open the bottom-left profile rail / manage profiles page.

### ACT-4 Create starter conversations if needed

If visible chat entries are desired for each expert, start one chat under each profile. Do not treat profile creation itself as conversation creation.

## 5. FLOW — Advertising orchestration flow

### FLOW-1 Brief intake

Default profile collects product, audience, platform, conversion target, style, available assets, and constraints.

### FLOW-2 Expert fan-out

Default profile calls or simulates the expert profiles depending on task complexity:

1. `baokuan-expert` for viral angle and hooks.
2. `ad-copywriter` for copy and voiceover.
3. `image-assistant` for visual assets and image prompts.
4. `video-expert` for storyboard and Seedance/Dreamina prompts.

### FLOW-3 Integration

Default profile resolves contradictions, removes weak ideas, unifies tone and continuity, and creates the final ad package.

### FLOW-4 Delivery

Final output is a structured Markdown production package: strategy, hooks, copy, visual prompts, video prompts, assets, A/B tests, and execution checklist.

## 6. DATA — Runtime files

### DATA-1 Default runtime home

```text
/tmp/hermes-agent-desktop-dev-home
```

### DATA-2 Default config

```text
/tmp/hermes-agent-desktop-dev-home/config.yaml
```

Relevant settings:

```yaml
delegation:
  max_concurrent_children: 4
  max_spawn_depth: 1
  child_timeout_seconds: 0
```

### DATA-3 Profile runtime directories

```text
/tmp/hermes-agent-desktop-dev-home/profiles/baokuan-expert
/tmp/hermes-agent-desktop-dev-home/profiles/ad-copywriter
/tmp/hermes-agent-desktop-dev-home/profiles/image-assistant
/tmp/hermes-agent-desktop-dev-home/profiles/video-expert
```

### DATA-4 Project-visible documentation

This file is a repo-visible map for the profile group:

```text
apps/desktop/docs/advertising-agent-orchestration.md
```

## 7. API — Desktop/profile backend concepts

### API-1 Profile list

Desktop reads profiles from the backend profile API. If profiles are created externally after the renderer has already mounted, the cached profile list may not update until refresh.

### API-2 Session list

Desktop's session list is separate from the profile list. It aggregates conversations that exist in session storage.

### API-3 Profile SOUL editor

Desktop can read/write profile `SOUL.md` through the profile management flow, but those runtime files are not part of the project file explorer by default.

## 8. RULE — Operating rules

### RULE-1 Do not confuse profile creation with session creation

A profile is an Agent identity/config island. A conversation is created only when a chat is started under that profile.

### RULE-2 Do not expect runtime files in project explorer

The file explorer follows the project root, not Hermes runtime home.

### RULE-3 Use default as orchestrator

Default should coordinate and integrate; specialists should produce focused expert outputs.

### RULE-4 Verify before claiming expert consultation

Default should not say it consulted a specialist unless it actually invoked a profile, subagent, or received a specialist result.

## 9. TEST — Verification checklist

### TEST-1 CLI profiles visible

```bash
hermes profile list
```

Expected profiles:

```text
default
baokuan-expert
ad-copywriter
image-assistant
video-expert
```

### TEST-2 SOUL files exist

```bash
python3 - <<'PY'
from pathlib import Path
root = Path('/tmp/hermes-agent-desktop-dev-home')
for rel in [
    'SOUL.md',
    'profiles/baokuan-expert/SOUL.md',
    'profiles/ad-copywriter/SOUL.md',
    'profiles/image-assistant/SOUL.md',
    'profiles/video-expert/SOUL.md',
]:
    p = root / rel
    print(rel, p.exists(), p.stat().st_size if p.exists() else 0)
PY
```

### TEST-3 Desktop rail refresh

After `/reset` or app reload, the bottom-left rail should show the named profile squares or show them in Manage Profiles.

### TEST-4 Conversation appears only after first chat

Start a new chat under one specialist profile, send a message, then verify that the conversation appears in the sidebar under that profile or in all-profile mode.
