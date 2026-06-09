---
name: coagency-hermes-update-acceptance-gate
description: Use when planning, testing, or rolling out a Hermes update across multiple agents; stage the update on one sacrificial agent first, run human-visible critical capability checks, capture regressions as issues, and only then upgrade the wider agent fleet.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [coagency, hermes, update, acceptance-gate, regression, gateway, media]
    related_skills: [hermes-agent, coagency-hermes-update-reconciliation, coagency-hermes-evolution-sync]
---

# Coagency Hermes Update Acceptance Gate

## Overview

Use this skill before or immediately after a `hermes update` when the operator depends on Hermes for live work across Telegram, Chronicle audio, GitHub coordination, cron delivery, skills, and other agent-to-human workflows.

The posture is release engineering, not optimism. Update one separated staging agent or profile first, prove the important capabilities still work in front of the human, then roll the update to the rest of the agent fleet.

A motivating disruptive event: Telegram `MEDIA:` audio delivery for a Chronicle teaser appeared successful from the agent side while the human received no attachment. Direct Telegram `sendVoice` worked, which proved the file and Bot API were fine while Hermes' higher-level media path silently degraded. That kind of regression must become an acceptance-gate test, not a repeated surprise.

## When to Use

- The user says Hermes was updated, is about to be updated, or should be upgraded across multiple agents.
- A gateway/platform capability changed after an update.
- A security tightening or allowlist change may affect media, files, tools, or delivery.
- The user is planning a separated staging agent before upgrading the wider army.
- You need to convert operator frustration into a concrete regression issue and a future-proof checklist.

Do not use this as a substitute for semantic reconciliation after an update. Pair it with `coagency-hermes-update-reconciliation` when local customizations, bundled skills, or patches need to be merged with upstream changes.

## Core Rule

Do not declare an update healthy because the agent received a successful tool result. For human-facing workflows, the human-visible outcome is the acceptance signal.

Examples:
- Telegram attachment test passes only when the human receives a playable native attachment.
- Chronicle publishing test passes only when the audio exists in canonical storage and is delivered from that path.
- GitHub issue test passes only when the created/commented issue URL resolves in the intended repository.

## Staged Rollout Pattern

1. **Freeze the current working agent**
   - Preserve local skill/code changes with scoped commits or a pre-update baseline tag.
   - Record which profile/agent remains the fallback.

2. **Update only a staging agent/profile first**
   - Use a separated Hermes checkout/profile or sacrificial agent.
   - Keep production/living agents on the known-good version until the gate passes.

3. **Run the acceptance gate with the human available**
   - Prefer short, observable probes.
   - Do not batch so much that the failing capability becomes ambiguous.

4. **Record failures as issues immediately**
   - Include operator-visible symptom, tool/log evidence, exact paths or commands when safe, and expected behavior.
   - If the repo is private, avoid leaking secrets or private content; summarize the shape and keep sensitive logs local.

5. **Only then roll out to other agents**
   - If any critical check fails, stop rollout and either fix, pin, or document a workaround.

## Critical Capability Checklist

Use the checklist as an operator-observed smoke suite. Mark each with direct evidence.

### Messaging and Gateway

- [ ] Telegram text reply reaches the human.
- [ ] Telegram native voice/audio attachment reaches the human.
- [ ] Telegram attachment comes from canonical archive storage when expected, not only `audio_cache/`.
- [ ] Slack/Discord home-channel send works if configured for that operator.
- [ ] Gateway restart reconnects enabled platforms.
- [ ] Home-channel routing uses numeric IDs where required, especially Telegram DM chat IDs.

### Voice and Chronicle

- [ ] STT transcribes an inbound voice message through the configured provider.
- [ ] TTS produces a non-empty audio file.
- [ ] Generated audio is verified with `file` / `ffprobe` when format matters.
- [ ] Voice episode artifacts land under `~/.hermes/voice-episodes/` with script, narration, audio, and index entries.
- [ ] A `MEDIA:` response or tool-send path delivers the audio as a playable attachment.

### Tools and Local Execution

- [ ] `read_file`, `write_file`, `patch`, and `search_files` work in the active profile.
- [ ] `terminal` can run a short command and return output.
- [ ] Git status/diff commands work in the intended checkout.
- [ ] Browser or web tools work if those are part of the operator's normal workflow.

### GitHub and Coordination

- [ ] `gh repo view` resolves the intended repository.
- [ ] A test issue/comment can be created or a dry-run artifact can be prepared when issue creation is not desired.
- [ ] Created issues are linked to the skill/commit or regression they describe.
- [ ] Private-repo context is not pasted into public issues.

### Skills and Memory

- [ ] Bundled and user-local skills still load.
- [ ] New or patched in-repo skills validate frontmatter and are committed.
- [ ] User-local runtime skills are not accidentally overwritten by bundled updates.
- [ ] Durable lessons are added to skills, not only to memory.

### Cron and Autonomous Delivery

- [ ] A small cron/no-agent or manual equivalent can deliver to the expected origin.
- [ ] Cron delivery target preserves thread/topic when applicable.
- [ ] Failures are visible; empty output is only silent when intentionally configured.

## Telegram Audio Probe

Use this when media attachment delivery is a critical gate.

```bash
source ~/.hermes/hermes-agent/venv/bin/activate
python - <<'PY'
import os, requests, json
from pathlib import Path
from dotenv import load_dotenv
from hermes_constants import get_env_path

load_dotenv(get_env_path(), override=True)
token = os.environ['TELEGRAM_BOT_TOKEN']
chat_id = os.environ.get('TELEGRAM_HOME_CHANNEL') or '<numeric-chat-id>'
path = Path('<canonical-audio.ogg>')

with path.open('rb') as f:
    r = requests.post(
        f'https://api.telegram.org/bot{token}/sendVoice',
        data={'chat_id': chat_id, 'caption': 'Hermes update acceptance-gate audio probe'},
        files={'voice': (path.name, f, 'audio/ogg')},
        timeout=60,
    )
print(r.status_code)
data = r.json()
print(json.dumps({
    'ok': data.get('ok'),
    'message_id': data.get('result', {}).get('message_id'),
    'has_voice': 'voice' in data.get('result', {}),
    'voice': data.get('result', {}).get('voice'),
}, indent=2))
PY
```

Then also test the Hermes path, not just the direct Bot API path:

```text
[[audio_as_voice]]
MEDIA:/absolute/path/under/voice-episodes/or/allowed/root/file.ogg
```

Passing the direct Bot API probe but failing the Hermes `MEDIA:` path means the regression is inside Hermes media extraction, allowlisting, platform dispatch, or result reporting.

## Regression Issue Shape

When the gate fails, create an issue with:

1. **Human-visible symptom** — what the operator saw or did not receive.
2. **Agent/tool claim** — what Hermes reported.
3. **Log evidence** — relevant warnings or errors, redacted.
4. **Direct probe result** — if an underlying provider/API succeeds or fails independently.
5. **Expected behavior** — user-visible acceptance criteria.
6. **Rollout consequence** — whether the wider update is blocked.
7. **Chronicle/evolution note** — why this matters to Iris/Hermes operating shape.

## Common Pitfalls

1. **Trusting success JSON without checking the human-visible result.**
   A text message can succeed while an attachment is dropped.

2. **Testing only provider APIs.**
   A direct Telegram `sendVoice` success does not prove Hermes `MEDIA:` delivery works.

3. **Rolling updates to every agent before the staging agent passes.**
   This destroys the fallback path.

4. **Letting operator anger evaporate without artifacts.**
   Convert disruptive frustration into a regression issue, a checklist item, or a skill patch.

5. **Publishing private context into a public issue.**
   Keep private repo details local or use a private issue tracker.

## Verification Checklist

- [ ] A staging agent/profile was tested before fleet rollout.
- [ ] Human-visible checks were used for messaging/media workflows.
- [ ] Direct provider probes and Hermes-level probes were distinguished.
- [ ] Any regression has an issue URL or local issue artifact.
- [ ] The rollout decision is explicit: proceed, block, pin, or fix first.
- [ ] Lessons learned were added to the relevant skill or checklist.
