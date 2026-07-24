# Manual operations

Use these commands in two modes:

1. upstream repo operations
2. Hermes-native workflow operations

## Hermes-native workflow commands

### Start a run

```bash
python3 ~/.hermes/skills/media/youtube-automation-agent/scripts/youtube_automation_helper.py init-run \
  --channel "Ladera Labs" \
  --niche "AI productivity" \
  --audience "founders and operators" \
  --style "educational" \
  --frequency daily \
  --topic "AI workflow automations"
```

### Check run status

```bash
python3 ~/.hermes/skills/media/youtube-automation-agent/scripts/youtube_automation_helper.py status \
  --workspace /path/to/run.json
```

### Get the current stage brief

```bash
python3 ~/.hermes/skills/media/youtube-automation-agent/scripts/youtube_automation_helper.py brief \
  --workspace /path/to/run.json
```

### Complete a stage

```bash
python3 ~/.hermes/skills/media/youtube-automation-agent/scripts/youtube_automation_helper.py complete-stage \
  --workspace /path/to/run.json \
  --stage strategy \
  --notes "selected founder ops angle" \
  --artifacts-json '{"selected_topic":"AI ops automations","content_type":"Explainer"}'
```

### Export final deliverables

```bash
python3 ~/.hermes/skills/media/youtube-automation-agent/scripts/youtube_automation_helper.py export \
  --workspace /path/to/run.json
```

## Upstream repo operations

### Health

```bash
curl http://localhost:3456/health
```

### View schedule

```bash
curl http://localhost:3456/schedule
```

### View analytics

```bash
curl http://localhost:3456/analytics
```

### Generate content manually

```bash
curl -X POST http://localhost:3456/generate \
  -H 'Content-Type: application/json' \
  -d '{"topic":"Top 10 Life Hacks","style":"listicle","length":"medium"}'
```

### Publish a queued item manually

```bash
curl -X POST http://localhost:3456/publish/CONTENT_ID
```
