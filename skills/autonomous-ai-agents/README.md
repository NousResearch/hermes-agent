# Autonomous AI Agents Skills

AI-powered coding assistants and autonomous development agents for Hermes.

## Overview

This category contains 4 skills for delegating coding tasks to autonomous AI agents. These skills enable you to leverage AI coding assistants like Claude Code, OpenAI Codex, and OpenCode for implementing features, reviewing pull requests, and extending Hermes itself. Whether you're building new features, conducting code reviews, or contributing to open source, these agents provide powerful autonomous coding capabilities.

## Available Skills

### AI Coding Assistants

#### **claude-code**
Delegate coding tasks to Claude Code CLI for feature implementation and pull request creation.

**Use when:** You want Claude to autonomously implement features, fix bugs, or create pull requests.

**Key features:**
- Autonomous feature implementation
- Pull request creation and management
- Multi-file code changes
- Context-aware coding decisions
- CLI-based workflow
- Integration with version control

**Capabilities:**
- Implement complete features from descriptions
- Refactor existing codebases
- Fix bugs with root cause analysis
- Create well-documented PRs
- Follow project conventions and style

---

#### **codex** (OpenAI Codex)
Delegate coding tasks to OpenAI Codex CLI for feature development and pull requests.

**Use when:** Using OpenAI's Codex model for autonomous code generation and implementation.

**Key features:**
- OpenAI Codex-powered coding
- Feature implementation from natural language
- Pull request workflows
- Multi-language support
- CLI-based agent control

**Capabilities:**
- Generate code from descriptions
- Complete partial implementations
- Create test suites
- Document code automatically
- Refactor and optimize

---

#### **opencode**
Delegate coding tasks to OpenCode CLI for features and pull request reviews.

**Use when:** You need autonomous code implementation and PR review capabilities.

**Key features:**
- Feature implementation
- Automated code review
- Pull request analysis
- Code quality suggestions
- Security and best practice checks

**Capabilities:**
- Implement features autonomously
- Review PRs for bugs and issues
- Suggest improvements
- Identify security vulnerabilities
- Enforce coding standards

---

### Meta-Agent Development

#### **hermes-agent**
Configure, extend, or contribute to Hermes Agent itself.

**Use when:** Contributing to Hermes, creating custom skills, or extending agent capabilities.

**Key features:**
- Hermes configuration management
- Custom skill development
- Agent extension and modification
- Contribution workflows
- Internal architecture access

**Capabilities:**
- Create new Hermes skills
- Modify agent behavior
- Contribute to Hermes project
- Build custom integrations
- Extend tool capabilities

---

## Quick Start

### Example: Implement a Feature

```bash
# 1. Delegate to Claude Code
/claude-code "Implement user authentication with JWT tokens, create login/logout endpoints, add middleware"

# 2. Or use Codex
/codex "Add OAuth2 integration to the authentication system"

# 3. Or use OpenCode
/opencode "Implement rate limiting middleware with Redis backend"
```

### Example: Code Review Workflow

```bash
# 1. Create feature with agent
/claude-code "Implement search functionality with full-text indexing"

# 2. Review with OpenCode
/opencode "Review the search feature PR for performance and security issues"

# 3. Address feedback
/claude-code "Fix the security concerns identified in PR review"
```

### Example: Extend Hermes

```bash
# 1. Create new skill
/hermes-agent "Create a new skill for Jira integration"

# 2. Test the skill
/hermes-agent "Run tests for the new Jira skill"

# 3. Submit contribution
/hermes-agent "Create PR for Jira skill contribution"
```

## Skill Combinations

**Complete Feature Development:**
1. Use `claude-code` to implement feature
2. Use `opencode` to review implementation
3. Use `claude-code` to address review feedback
4. Merge and deploy

**Open Source Contribution:**
1. Use `hermes-agent` to understand project structure
2. Use `claude-code` to implement contribution
3. Use `opencode` for self-review
4. Submit PR to project

**Multi-Agent Workflow:**
1. Use `claude-code` for backend implementation
2. Use `codex` for frontend components
3. Use `opencode` to review full stack changes
4. Use `hermes-agent` to integrate into Hermes

**Quality Assurance:**
1. Use any agent to implement feature
2. Use `opencode` for automated code review
3. Use `claude-code` to write comprehensive tests
4. Use `hermes-agent` to validate Hermes-specific concerns

## Choosing the Right Agent

**For feature implementation:**
- Claude-based projects → `claude-code`
- OpenAI ecosystem → `codex`
- General purpose → `opencode`

**For code review:**
- Automated review → `opencode`
- Human-in-loop → Combine agent + manual review

**For Hermes development:**
- Extending Hermes → `hermes-agent`
- Contributing skills → `hermes-agent`

**General guidelines:**
- Complex features → `claude-code` (strong reasoning)
- Quick prototypes → `codex` (fast generation)
- Security-critical → `opencode` (thorough review)

## Common Workflows

### Full Feature Implementation

```bash
# 1. Plan the feature
"I need to implement user profile management with photo uploads"

# 2. Delegate to agent
/claude-code "Implement user profile management:
- Profile CRUD endpoints
- Photo upload with S3 storage
- Image resizing and optimization
- Validation and error handling
- Unit and integration tests"

# 3. Review implementation
/opencode "Review profile management PR for:
- Security vulnerabilities
- Performance issues
- Missing edge cases
- Code quality"

# 4. Address feedback
/claude-code "Fix the issues identified in code review"

# 5. Deploy
"Merge PR and deploy to staging"
```

### Bug Fix Workflow

```bash
# 1. Describe the bug
/claude-code "Users report 500 error when uploading files > 10MB. Investigate and fix."

# 2. Agent investigates and fixes
# Reviews logs, identifies issue, implements solution

# 3. Verify fix
/opencode "Review bug fix PR for completeness and test coverage"

# 4. Merge
"Confirm tests pass and deploy"
```

### Hermes Skill Development

```bash
# 1. Create skill structure
/hermes-agent "Create new skill for Notion integration with CRUD operations"

# 2. Implement core functionality
/claude-code "Implement Notion API integration:
- Authentication
- Database queries
- Page creation and updates
- Block manipulation"

# 3. Add documentation
/hermes-agent "Add comprehensive SKILL.md with examples and use cases"

# 4. Test and validate
/hermes-agent "Run skill validation and integration tests"

# 5. Submit contribution
/hermes-agent "Create PR with skill contribution"
```

### Multi-Language Project

```bash
# 1. Backend (Python)
/claude-code "Implement FastAPI backend with async endpoints"

# 2. Frontend (React)
/codex "Create React components for the new feature"

# 3. Infrastructure (Terraform)
/opencode "Write Terraform configs for deployment"

# 4. Review everything
/opencode "Review entire stack for consistency and integration issues"
```

## Best Practices

**When Delegating Tasks:**
- Be specific about requirements and constraints
- Include context about existing code and conventions
- Specify test requirements upfront
- Mention any security or performance concerns
- Provide examples when helpful

**Code Review:**
- Always review agent-generated code
- Run tests before merging
- Check for security vulnerabilities
- Verify edge cases are handled
- Ensure documentation is complete

**Agent Selection:**
- Start with one agent for consistency
- Use multiple agents for different perspectives
- Cross-check critical implementations
- Leverage each agent's strengths

**Iteration:**
- Review and refine iteratively
- Provide feedback to improve results
- Don't accept first version blindly
- Iterate until production-ready

## Agent-Specific Tips

### Claude Code
- Excellent for complex reasoning and architecture
- Handles multi-file refactors well
- Strong at following project patterns
- Good documentation generation

### Codex
- Fast code generation
- Wide language support
- Good for boilerplate
- Strong completion capabilities

### OpenCode
- Thorough code review
- Security-focused analysis
- Best practice enforcement
- Detailed feedback

### Hermes Agent
- Deep Hermes integration
- Skill development expertise
- Contribution workflows
- Agent customization

## Integration with Development Workflow

**Git Integration:**
```bash
# Agents create branches automatically
# Review PRs in GitHub/GitLab
# Merge when approved
```

**CI/CD:**
```bash
# Agent-generated code triggers CI
# Automated tests run
# Code review checks pass
# Deploy on merge
```

**Team Collaboration:**
```bash
# Agents create PRs for review
# Team provides feedback
# Agents iterate based on comments
# Human approval before merge
```

## Limitations and Considerations

**What Agents Excel At:**
- Implementing well-defined features
- Refactoring and optimization
- Writing tests and documentation
- Following established patterns
- Routine code reviews

**What Requires Human Judgment:**
- Architecture decisions
- Business logic validation
- UX/design choices
- Security trade-offs
- Production deployment decisions

**Best Approach:**
- Use agents for implementation
- Human review for validation
- Iterate collaboratively
- Maintain human oversight

## Security Considerations

**When Using AI Agents:**
- Review all generated code
- Never deploy without testing
- Check for hardcoded secrets
- Validate input sanitization
- Verify authentication/authorization
- Test for injection vulnerabilities

**Agent Access:**
- Configure appropriate permissions
- Limit repository access
- Use read-only tokens when possible
- Audit agent actions
- Review changes before commit

## Contributing

Found a bug or have an enhancement idea?

1. Open an issue describing the improvement
2. Fork the repository
3. Make changes to the relevant `SKILL.md`
4. Submit a pull request

## Related Categories

- **software-development/** - Manual development workflows
- **github/** - GitHub integration and workflows
- **productivity/** - Task automation
- **mlops/** - ML model development and deployment

---

**Questions?** Check the [Hermes Agent documentation](https://hermes-agent.nousresearch.com/docs/) or ask in the [Discord community](https://discord.gg/nousresearch).
