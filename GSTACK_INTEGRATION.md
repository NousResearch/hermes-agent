# gstack Personas Integration Guide

Complete technical documentation for the gstack personas feature: architecture, implementation, deployment, and customization.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Persona Definitions](#persona-definitions)
3. [Command Dispatch](#command-dispatch)
4. [Subagent Spawning](#subagent-spawning)
5. [Toolset Curation](#toolset-curation)
6. [Model Selection](#model-selection)
7. [Token Budgets](#token-budgets)
8. [Adding New Personas](#adding-new-personas)
9. [Deployment Checklist](#deployment-checklist)
10. [Troubleshooting](#troubleshooting)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Input (CLI or Gateway)                  │
│  e.g., /reviewer ~/myproject/src/auth.py                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Command Dispatch    │
                  │  (commands.py)       │
                  └──────────┬───────────┘
                             │
                             ▼
        ┌────────────────────────────────────┐
        │  gstack_commands.py Handler        │
        │  e.g., handle_reviewer_command()   │
        └────────────────┬───────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │  _delegate_persona_review()            │
        │  Builds task + context                 │
        └────────────────┬───────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────────────┐
        │  delegate_task() [from tools/delegate_tool.py] │
        │  • Spawns subagent process                     │
        │  • Injects system prompt                       │
        │  • Curates toolset                             │
        │  • Sets iteration limit                        │
        └────────────────┬───────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
    ┌─────────────────┐         ┌──────────────────┐
    │ Child Agent     │         │ Parent Agent     │
    │ (subprocess)    │         │ (continues)      │
    │                 │         │                  │
    │ • System Prompt │         │ Only sees result │
    │ • Limited Tools │         │ & summary        │
    │ • Context Only  │         │                  │
    │ • Max Iter=20   │         │                  │
    └────────┬────────┘         │                  │
             │                  │                  │
             ▼                  │                  │
         ┌─────────────┐        │                  │
         │ Review Runs │        │                  │
         │ (isolated)  │        │                  │
         └────────┬────┘        │                  │
                  │             │                  │
                  ▼             │                  │
         ┌──────────────────┐   │                  │
         │ Review Report    │   │                  │
         │ (markdown)       ├───┤                  │
         │ Saved to disk    │   │                  │
         └──────────────────┘   │                  │
                                │                  │
                                └──────────────────┘
                                   User receives
                                   report location
```

### Key Design Principles

1. **Isolation:** Child agents have no access to parent history or dangerous tools
2. **Safety:** Blocked tools are always stripped (delegate_task, clarify, memory, send_message, execute_code)
3. **Focus:** Each persona gets only their task + context, not the full conversation
4. **Clarity:** Output format is strictly specified in system prompt (markdown sections)
5. **Logging:** Reports are saved to disk for audit trail and offline review

## Persona Definitions

All personas are defined in `tools/gstack_personas.py`:

```python
@dataclass
class PersonaRole(Enum):
    CEO = "ceo"
    ENG_MANAGER = "eng_manager"
    DESIGNER = "designer"
    REVIEWER = "reviewer"
    QA_LEAD = "qa_lead"
    CSO = "cso"
    RELEASE_ENGINEER = "release_engineer"

PERSONA_DEFINITIONS: Dict[PersonaRole, Dict[str, Any]] = {
    PersonaRole.REVIEWER: {
        "name": "Reviewer",
        "title": "Code Quality & Production Safety Lead",
        "emoji": "🔍",
        "toolsets": ["terminal", "file"],
        "system_prompt": """You are the Code Reviewer...""",
        "max_iterations": 30,
    },
    # ... 6 more personas
}
```

### Persona Structure

Each persona definition must include:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `name` | str | Persona name | "Reviewer" |
| `title` | str | Long title | "Code Quality & Production Safety Lead" |
| `emoji` | str | Unicode emoji | "🔍" |
| `toolsets` | list[str] | Allowed tool categories | ["terminal", "file"] |
| `system_prompt` | str | Complete system prompt (>200 chars) | "You are the Code Reviewer..." |
| `max_iterations` | int | Max agent iterations (10-50) | 30 |

### System Prompt Best Practices

```python
"""You are the [ROLE] reviewing [DOMAIN].

Your role:
1. [Concern 1] — [Specific question]
2. [Concern 2] — [Specific question]
3. [Concern 3] — [Specific question]
4. [Concern 4] — [Specific question]
5. [Concern 5] — [Specific question]

[General guidance about mindset and approach]

Format your response:
- **Section 1**: Details
- **Section 2**: Details
- **Section 3**: Details
- **Decision/Recommendation**: Clear next steps"""
```

**Key elements:**
- Clear role definition
- 3-5 specific numbered concerns (numbered list helps agent focus)
- Behavioral guidance (opinionated, thorough, pragmatic, etc.)
- Markdown format specification with `**bold**` sections
- Sections using bullet points or numbered lists

## Command Dispatch

Commands are registered in `hermes_cli/commands.py`:

```python
COMMAND_REGISTRY: list[CommandDef] = [
    # ... other commands
    
    # gstack personas
    CommandDef("reviewer", "Code review: quality, testing, safety", "gstack",
               aliases=("review",), args_hint="<target> [context]"),
    CommandDef("ceo-review", "CEO review: strategic fit, user value", "gstack",
               aliases=("ceo",), args_hint="<target> [context]"),
    # ... other personas
]
```

### Adding a New Command

To add a new persona command:

1. **Add to `PersonaRole` enum** (`tools/gstack_personas.py`):
   ```python
   class PersonaRole(Enum):
       CONSULTANT = "consultant"
   ```

2. **Add to `PERSONA_DEFINITIONS`** (`tools/gstack_personas.py`):
   ```python
   PERSONA_DEFINITIONS: Dict[PersonaRole, Dict[str, Any]] = {
       PersonaRole.CONSULTANT: {
           "name": "Consultant",
           "title": "Strategic Advisor",
           "emoji": "💼",
           "toolsets": ["terminal", "file", "web"],
           "system_prompt": "You are the Consultant...",
           "max_iterations": 22,
       }
   }
   ```

3. **Register command** (`hermes_cli/commands.py`):
   ```python
   CommandDef("consultant", "Consultant review: strategic advising", "gstack",
              aliases=("consult",), args_hint="<target> [context]"),
   ```

4. **Add handler** (`hermes_cli/gstack_commands.py`):
   ```python
   def handle_consultant_command(target: str, context: Optional[str] = None, cli_obj=None) -> str:
       """Handle /consultant <target>"""
       return _delegate_persona_review(PersonaRole.CONSULTANT, target, context, cli_obj, "consultant")
   ```

5. **Connect in gateway dispatch** (e.g., `gateway/command_handler.py`):
   ```python
   if cmd == "consultant":
       result = handle_consultant_command(target, context, cli_obj)
   ```

## Subagent Spawning

Subagents are spawned via `tools/delegate_tool.py`:

```python
def delegate_task(
    goal: str,
    context: Optional[str] = None,
    toolsets: list[str] = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    parent_agent=None,
) -> str:
    """Delegate a task to a subagent.
    
    Returns JSON string with subagent results.
    """
```

### How Delegation Works

1. **Validation:**
   - Check recursion depth (max 2 levels)
   - Verify concurrent children < MAX_CONCURRENT_CHILDREN (3)
   - Strip blocked tools from requested toolsets

2. **Child Preparation:**
   - Generate unique task_id (own terminal session)
   - Build child system prompt from goal + context
   - Create restricted context (only goal + explicit context, no parent history)
   - Filter toolsets (remove delegate_task, clarify, memory, send_message, execute_code)

3. **Spawning:**
   - Fork new subprocess with child task_id
   - Pass system prompt as context
   - Set iteration limit
   - Pass parent_agent reference (so child knows it's a child)

4. **Execution:**
   - Child agent runs isolated with focused context
   - Parent blocks until child completes (or times out)
   - Parent never sees child's intermediate tool calls, only final result

5. **Result Processing:**
   - Parse child's JSON result
   - Extract summary from child's results
   - Save report to disk
   - Return to parent

### Blocking Tools for Safety

These tools are **always** blocked for child agents, even if requested:

```python
DELEGATE_BLOCKED_TOOLS = frozenset([
    "delegate_task",   # No recursive delegation (max depth = 2)
    "clarify",         # No user interaction (child is focused)
    "memory",          # No writes to shared MEMORY.md (isolation)
    "send_message",    # No cross-platform side effects
    "execute_code",    # Children reason step-by-step, not script generation
])
```

## Toolset Curation

Each persona has access to a **safe subset** of tools:

| Persona | Toolsets | Justification |
|---------|----------|---------------|
| CEO | terminal, file, web | Strategic thinking requires broader context |
| Eng Manager | terminal, file | Code/architecture analysis needs file access |
| Designer | browser, web | Visual inspection of UI requires rendering |
| Reviewer | terminal, file | Code analysis and testing |
| QA Lead | browser | Functional testing in browser |
| CSO | terminal, file, web | Security audit needs full access (but controlled) |
| Release Engineer | terminal | Deployment scripts and system analysis |

### Safe Toolsets

Allowed tool categories:

- **terminal:** Command execution, system calls (controlled)
- **file:** Read/write filesystem (restricted paths)
- **browser:** Browser automation (for visual inspection)
- **web:** Web requests (for API testing)

### Dangerous Tools (Blocked)

These are **always** stripped:

- **delegate_task:** Prevents recursive delegation/infinite loops
- **clarify:** Prevents user interaction during focused task
- **memory:** Prevents cross-task state pollution
- **send_message:** Prevents unintended side effects (Slack, Discord, etc.)
- **execute_code:** Prevents unrestricted code execution

## Model Selection

Personas use appropriate models based on task complexity:

```python
# In gstack_commands.py, when calling delegate_task:
result_json = delegate_task(
    goal=task,
    context=full_context,
    toolsets=toolsets,
    max_iterations=max_iter,
    parent_agent=parent_agent,
    # model is chosen by enforce_token_discipline.py
)
```

### Model Recommendation Matrix

| Persona | Complexity | Recommended Model | Rationale |
|---------|-----------|-------------------|-----------|
| CEO | High | claude-opus | Strategic thinking, complex tradeoffs |
| Eng Manager | High | claude-opus | Architecture decisions, tech depth |
| Designer | Medium | claude-3.5-sonnet | Visual perception, UX reasoning |
| Reviewer | High | claude-opus | Code analysis, best practices |
| QA Lead | Medium | claude-3.5-sonnet | Functional testing, pattern matching |
| CSO | Very High | claude-opus | Security requires extreme care |
| Release Engineer | Medium | claude-3.5-sonnet | Procedural, structured thinking |

## Token Budgets

Each persona has a token budget based on task complexity:

```python
# In delegate_task() internals
TOKEN_BUDGETS = {
    "ceo": 2000,           # Strategic thinking is complex
    "eng_manager": 2500,   # Architecture requires depth
    "designer": 1500,      # Visual assessment is faster
    "reviewer": 2000,      # Code review is thorough
    "qa_lead": 1500,       # Testing is iterative but focused
    "cso": 2500,           # Security is very thorough
    "release_engineer": 1500,  # Deployment is procedural
}
```

### Token Enforcement

The system uses `enforce_token_discipline.py` to:

1. Check persona's token budget
2. Check system remaining tokens
3. Choose appropriate model (fast/cheap vs. thorough/expensive)
4. Log model selection for auditing

## Iteration Limits

Max iterations per persona (10-50 range):

```python
PERSONA_DEFINITIONS[PersonaRole.REVIEWER]["max_iterations"] = 30  # Most thorough
PERSONA_DEFINITIONS[PersonaRole.CEO]["max_iterations"] = 20       # Strategic, fewer tools
PERSONA_DEFINITIONS[PersonaRole.DESIGNER]["max_iterations"] = 15  # Focused, visual
```

**Design philosophy:**
- Reviewers need more iterations (thorough code analysis)
- Designers need fewer iterations (visual assessment is faster)
- Security/Architecture need many iterations (complexity)
- Designers/QA need fewer iterations (more focused)

## Adding New Personas

### Template

```python
# Step 1: Add to PersonaRole enum
class PersonaRole(Enum):
    NEW_ROLE = "new_role"

# Step 2: Add definition
PERSONA_DEFINITIONS[PersonaRole.NEW_ROLE] = {
    "name": "New Role",
    "title": "Long title describing the role",
    "emoji": "🎯",
    "toolsets": ["terminal", "file"],
    "system_prompt": """You are the New Role reviewing...
    
Your role:
1. Concern 1 — Question?
2. Concern 2 — Question?
3. Concern 3 — Question?

[Guidance paragraph]

Format your response:
- **Section 1**: [description]
- **Section 2**: [description]
- **Decision**: [Clear recommendation]""",
    "max_iterations": 20,  # Tune based on complexity
}

# Step 3: Register command
CommandDef("new-role", "New role review: description", "gstack",
           aliases=("newrole",), args_hint="<target> [context]"),

# Step 4: Add handler
def handle_new_role_command(target: str, context: Optional[str] = None, cli_obj=None) -> str:
    return _delegate_persona_review(PersonaRole.NEW_ROLE, target, context, cli_obj, "new_role")

# Step 5: Connect in gateway dispatch
if cmd == "new-role":
    result = handle_new_role_command(args[0], args[1] if len(args) > 1 else None, cli_obj)
```

### Validation Checklist

Before shipping a new persona:

- [ ] Persona has clear, specific focus (not overlapping existing personas)
- [ ] System prompt is >200 characters
- [ ] System prompt includes 3-5 numbered concerns
- [ ] Output format is specified with markdown sections
- [ ] max_iterations is 15-30 (reasonable for the task)
- [ ] Toolsets are only from safe list (terminal, file, browser, web)
- [ ] No dangerous keywords in system prompt (subprocess.run, eval, etc.)
- [ ] Command is registered in COMMAND_REGISTRY
- [ ] Handler function is implemented
- [ ] Gateway dispatch includes new command
- [ ] Tests pass for new persona

## Deployment Checklist

### Pre-Deployment

- [ ] All tests pass: `pytest tests/tools/test_gstack_personas.py -v`
- [ ] Coverage ≥85%: `pytest --cov=...`
- [ ] No dangerous commands in prompts
- [ ] All 7 personas defined and tested
- [ ] Commands registered in command registry
- [ ] Handlers implemented and callable
- [ ] Documentation updated
- [ ] Example workflows tested manually

### Deployment

- [ ] Create PR with:
  - [ ] Test suite (tests/tools/test_gstack_personas.py)
  - [ ] User docs (website/docs/features/gstack-personas.md)
  - [ ] Integration guide (GSTACK_INTEGRATION.md)
  - [ ] Coverage report
- [ ] Request review from:
  - [ ] Security team (toolset safety, system prompts)
  - [ ] Architecture team (subagent design)
  - [ ] Product team (workflow UX)
- [ ] Merge with passing CI/CD
- [ ] Deploy to staging first
- [ ] Monitor error rates for 24h
- [ ] Full rollout to production

### Post-Deployment

- [ ] Monitor usage metrics:
  - [ ] How many persona reviews per day?
  - [ ] Which personas are most used?
  - [ ] Average review time per persona
  - [ ] Error rate
  - [ ] User satisfaction
- [ ] Collect feedback from users
- [ ] Iterate on prompts based on feedback
- [ ] Add new personas based on user requests
- [ ] Optimize token budgets based on actual usage

## Troubleshooting

### Persona Doesn't Work

**Problem:** Command registered but handler doesn't work

**Steps:**
1. Verify command is in COMMAND_REGISTRY
2. Verify handler exists in gstack_commands.py
3. Verify handler is connected in gateway dispatch
4. Run tests: `pytest tests/tools/test_gstack_personas.py::TestCommandRegistration -v`
5. Check logs: `tail -f ~/.hermes/logs/hermes.log`

### Subagent Crashes

**Problem:** Child agent fails or times out

**Debugging:**
1. Check child agent logs (separate task_id)
2. Verify toolsets are valid: `["terminal", "file"]` etc.
3. Check system prompt for dangerous keywords
4. Verify max_iterations is reasonable (10-50)
5. Try reducing context size (might be too much for child)

### Review Output Incorrect

**Problem:** Persona output doesn't match expected format

**Fix:**
1. Review system prompt (check format specification)
2. Add more explicit format instructions
3. Use markdown headers: `- **Section**:`
4. Include examples in system prompt
5. Increase max_iterations (agent might be rushed)

### Blocked Tools Error

**Problem:** "Tool X is blocked for delegated subagents"

**Expected:** These tools are always blocked:
- delegate_task (recursion prevention)
- clarify (user interaction)
- memory (isolation)
- send_message (side effects)
- execute_code (safety)

**Solution:** Use allowed tools instead:
- terminal for command execution
- file for reading/writing files
- browser for UI inspection
- web for API calls

### Token Budget Exceeded

**Problem:** "Subagent exceeded token budget"

**Fix:**
1. Reduce max_iterations for persona (fewer steps = fewer tokens)
2. Provide smaller/more focused context
3. Use a faster/cheaper model
4. Break task into smaller sub-tasks

## Example: Extending gstack

### Add "Compliance Officer" Persona

```python
# tools/gstack_personas.py
class PersonaRole(Enum):
    COMPLIANCE_OFFICER = "compliance_officer"

PERSONA_DEFINITIONS[PersonaRole.COMPLIANCE_OFFICER] = {
    "name": "Compliance Officer",
    "title": "Chief Compliance Officer",
    "emoji": "📋",
    "toolsets": ["terminal", "file", "web"],
    "system_prompt": """You are the Compliance Officer auditing for compliance requirements.

Your role:
1. **Regulatory Requirements** — Does this meet industry regulations?
2. **Data Governance** — Is data handled per compliance policy?
3. **Audit Trail** — Are actions properly logged for auditing?
4. **Consent Management** — Does this respect user consent choices?
5. **Documentation** — Are compliance measures documented?

Be thorough. Assume regulators will review this. Note all compliance gaps.

Format your response:
- **Regulatory Analysis**: Applicable regulations and requirements
- **Data Governance**: Compliance with data policies
- **Audit Requirements**: Logging and traceability
- **Consent Management**: User choices and preferences
- **Documentation**: Compliance documentation
- **Critical Gaps**: Must fix for compliance
- **Recommendations**: Suggested improvements
- **Compliance Status**: Approved / Approved with caveats / Not approved""",
    "max_iterations": 25,
}

# hermes_cli/commands.py
CommandDef("compliance", "Compliance Officer review: regulatory, audit, governance", "gstack",
           aliases=("compliance-check",), args_hint="<target> [context]"),

# hermes_cli/gstack_commands.py
def handle_compliance_command(target: str, context: Optional[str] = None, cli_obj=None) -> str:
    """Handle /compliance <target>"""
    return _delegate_persona_review(PersonaRole.COMPLIANCE_OFFICER, target, context, cli_obj, "compliance")

# gateway/command_handler.py (or wherever commands are dispatched)
elif cmd == "compliance":
    result = handle_compliance_command(args[0], args[1] if len(args) > 1 else None, cli_obj)
```

Then add tests:
```python
# tests/tools/test_gstack_personas.py
def test_handle_compliance_command(self, mock_cli_obj, mock_delegate_result):
    """Test compliance officer command handler."""
    with patch("hermes_cli.gstack_commands.delegate_task") as mock_delegate:
        mock_delegate.return_value = mock_delegate_result
        with patch("hermes_cli.gstack_commands._save_review_report"):
            result = handle_compliance_command("policy.md", cli_obj=mock_cli_obj)
            assert "complete" in result.lower()
```

## References

- [gstack by Garry Tan](https://github.com/garrytan/gstack) — Inspiration for this feature
- [Hermes Agent Delegation Architecture](docs/architecture/delegation.md)
- [System Prompt Best Practices](docs/guides/system-prompts.md)
- [Tool Safety and Isolation](docs/architecture/tool-safety.md)
