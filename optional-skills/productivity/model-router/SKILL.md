# model-router Skill

Multi-model auto-routing skill for Hermes Agent. Automatically routes complex writing and analysis tasks to a dedicated model via Profile Spawn, while keeping simple conversations on the main model.

## When to Use

- User submits a complex writing task (audit reports, compliance docs, risk analysis, structured output >500 chars)
- User requests multi-step analysis, comparative evaluation, or deep-dive reasoning
- Task requires large context window (50K+ tokens)
- User explicitly says "Use [model] to write this" or "Use [model] to handle this"

## Routing Decision Flow

```
Receive request
    │
    ├─ User explicitly specifies a model?
    │   └─ Yes → Route directly (skip evaluation)
    │
    ├─ Task matches trigger conditions?
    │   └─ Yes → Route to dedicated model via Profile Spawn
    │       ├─ Notify user
    │       ├─ terminal("{alias} chat -q 'prompt'", timeout=120)
    │       └─ Present result
    │
    └─ No → Stay on main model (handle directly)
```

### Trigger Conditions

| Category | Examples |
|----------|---------|
| Complex writing | Audit reports, workpapers, remediation opinions, regulatory compliance docs |
| Complex analysis | Multi-step reasoning, risk evaluation, comparative analysis |
| Large context | 50K+ token tasks, multi-page document interpretation |

### Excluded (stays on main model)

- Short Q&A (<100 chars)
- Simple lookups, small talk
- File operations, system commands

## Route Execution

### Auto-routing (user did not specify a model)

```
1. Receive user request
2. Evaluate → does not match → handle directly → done
3. Evaluate → matches →
   a. Notify user
   b. Build sub-process prompt with full context
   c. terminal("{alias} chat -q 'prompt'", timeout=120)
   d. Wait for result → present to user
```

### User explicitly specifies

```
1. User says "Use [model] to write this"
2. Skip evaluation → execute auto-routing step a-e
```

### Timeout & Degradation

| Scenario | Action |
|----------|--------|
| No response for 30s | Notify user, fall back to main model |
| Empty/error response | Notify user, fall back to main model |
| terminal timeout (120s) | Suggest splitting into smaller chunks |

## Requirements

| Type | Details |
|------|---------|
| **Hermes** | Version >= 0.12.0 |
| **API keys** | Three LLM providers configured in `~/.hermes/.env` or `config.yaml` |
| **Profiles** | Two Hermes profiles: one for writing model, one optional for backup |

## Deployment

### 1. Configure API Keys

```bash
# ~/.hermes/.env (example)
API_KEY_MAIN="your-api-key-for-main-model"
API_KEY_WRITER="your-api-key-for-writing-model"
API_KEY_BACKUP="your-api-key-for-backup-model"
```

### 2. Create Profiles

```bash
# Writing/analysis model profile
hermes profile create {writing-profile-name} --clone
# Edit ~/.hermes/profiles/{writing-profile-name}/config.yaml:
#   model.default: {writing-model-name}
#   model.provider: {writing-provider-name}

# Backup model profile (use alias if command name conflicts)
hermes profile create {backup-profile-name} --clone
hermes profile alias {backup-profile-name} --name {backup-alias}
# Edit ~/.hermes/profiles/{backup-profile-name}/config.yaml:
#   model.default: {backup-model-name}
#   model.provider: {backup-provider-name}
```

### 3. Provider Config Template

```yaml
# ~/.hermes/config.yaml — providers section
providers:
  {provider-a}:
    api_mode: chat_completions
    model: {model-name-a}
    model_display_name: {display-name-a}
  {provider-b}:
    api_mode: chat_completions
    model: {model-name-b}
    model_display_name: {display-name-b}
```

### 4. Load the Skill

```
/skill model-router
```

## Known Technical Limitation

**`delegation.provider` runtime changes do NOT work mid-session.** This config is read once at session initialization. Writing to config.yaml has no effect on the current session. Always use Profile Spawn for model routing.

## Pitfalls

1. **Do not modify current session config**: Always use Profile Spawn. `hermes config set delegation.provider` does not take effect in the current session.
2. **Context isolation**: Sub-processes start a fresh session. Pass full context via the `-q` parameter.
3. **Timeout**: Set terminal timeout to 120s+ for long tasks.
4. **Sub-process limitations**: Writing/analysis only — no file or system operations in sub-processes.
5. **No redundant notification**: Inform the user once about routing, do not repeat.

## Additional Resources

- **GitHub:** https://github.com/xd258/model-router
- **Release:** https://github.com/xd258/model-router/releases/tag/v1.0.0
