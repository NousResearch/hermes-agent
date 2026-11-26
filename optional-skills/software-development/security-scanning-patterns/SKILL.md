---
name: security-scanning-patterns
description: >
  Regex patterns for detecting secrets, injection attacks, and unsafe code.
  Use during code review or pre-commit checks. Complements requesting-code-review.
tags: [security, scanning, secrets, injection, regex]
---

# Security Scanning Patterns

Regex patterns for code review and pre-commit security checks.

## Secret Detection

Scan source files (NOT .env.example, test fixtures, node_modules/):

```
password\s*[=:]\s*['"][^'"]{8,}['"]
api_key\s*[=:]\s*['"][^'"]{16,}['"]
secret\s*[=:]\s*['"][^'"]{16,}['"]
token\s*[=:]\s*['"][^'"]{16,}['"]
AKIA[0-9A-Z]{16}                          # AWS access key
sk-[a-zA-Z0-9]{48}                        # OpenAI API key
ghp_[a-zA-Z0-9]{36}                       # GitHub personal token
glpat-[a-zA-Z0-9\-]{20}                   # GitLab token
xox[bprs]-[a-zA-Z0-9\-]{10,}             # Slack token
```

## Injection Patterns

### SQL Injection
```
SELECT.*WHERE.*\+                          # String concatenation in queries
query\(.*\+.*\)                            # Concatenation in query call
```

### Command Injection
```
exec\(.*\+.*\)                             # String concat in exec
spawn\(.*\+.*\)                            # String concat in spawn
child_process.*\+                          # Concatenation with child_process
```

### XSS
```
innerHTML\s*=                              # Direct innerHTML
dangerouslySetInnerHTML                    # React dangerous HTML
document\.write\(                          # document.write
```

### Unsafe Code Execution
```
eval\(                                     # eval()
new Function\(                             # Function constructor
pickle\.loads?\(                           # Python unsafe deserialization
```

## Quick Scan Command

```bash
# Scan staged changes for security issues
git diff --cached | grep "^+" | grep -iE \
  "(api_key|secret|password|token|passwd)\s*=\s*['\"][^'\"]{6,}['\"]|eval\(|exec\(|os\.system\(|subprocess.*shell=True|innerHTML\s*=|pickle\.loads?\("
```

## Severity

| Finding           | Severity | Action                     |
|-------------------|----------|----------------------------|
| Hardcoded secret  | CRITICAL | Remove, rotate immediately |
| SQL injection     | CRITICAL | Parameterize queries       |
| Command injection | CRITICAL | Use execFile with arrays   |
| XSS               | HIGH     | Escape output              |
| Unsafe eval       | HIGH     | Remove or sandbox          |
