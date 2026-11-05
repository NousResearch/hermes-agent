---
name: quality-test-implementation
license: MIT
version: 1.1.1
author: afonsoft
platforms: [linux, macos, windows]
description: "Raise code quality and test coverage in repos."
metadata:
  hermes:
    tags: [Testing, Coverage, Quality, TDD]
  version: "1.1.1"
  visibility: public
  author: afonsoft
  url: https://github.com/afonsoft/skills
  subagent: "false"
  user-invokable: "true"
  argument-hint: "<repo-name> [--language=dotnet|java|python] [--base=main] [--output-branch=feature/YYYYMMDD-name]"
---

# Multi-Language Code Quality & Test Coverage

## Role

Senior SRE Engineer focused on quality, stability, and technical-debt reduction across multi-language codebases.

## Goal

Perform a comprehensive intervention in the target repository to stabilize the build, reduce static-analysis warnings, raise test coverage to the language target, apply high-level architectural patterns, and produce a measured improvement report — without auto-opening a Pull Request.

## When to Use

- A repository (`.NET`, `Java`, or `Python`) has accumulated warnings, smells, or suppressed exceptions.
- Test coverage is below the language target and needs a structured push.
- Technical debt must be reduced before a release or handoff.
- Security CVEs (NU1903, OWASP, Bandit, Safety, Snyk) must be cleared.

**When NOT to use:** one small file, a single change review, or a quick lint pass. For reviewing a single change, use `code-review-and-quality`.

- User asks or mentions this skill in English (e.g., "use /quality-test-implementation", "run quality-test-implementation").
- O usuário pede ou menciona esta skill em português (ex.: "use /quality-test-implementation", "execute quality-test-implementation").

## Inputs

- `REPO_NAME`: full name (`owner/repo`).
- `BASE_BRANCH`: branch to start from (default `main`/`develop`).
- `OUTPUT_BRANCH`: `feature/{YYYYMMDD}-{function-name}`.
- `PRIMARY_LANGUAGE`: `dotnet`, `java`, or `python`.

---

## Phase 1 — Preparation and Environment

1. Clone `{REPO_NAME}` if not already present.
2. Create working branch `feature/{YYYYMMDD}-{function-name}`.
3. Identify the build/test tooling:
   - **.NET**: `*.sln`, `*.csproj`, `Directory.Build.props`, `global.json`.
   - **Java**: `pom.xml` (Maven) or `build.gradle*` (Gradle).
   - **Python**: `pyproject.toml`, `setup.py`, `requirements*.txt`, `tox.ini`.
4. Keep any `Environment.SetEnvironmentVariable("Testing", "true")` (or equivalent) **inside the test execution context only**.

---

## Phase 2 — Static Analysis and Warning Correction

Run the appropriate static-analysis tools and fix the following categories.

### .NET

| Category | Codes | Fix |
|---|---|---|
| Logging | CA2017, S2629, CA2254 | Use static templates and consistent placeholders |
| Asynchronism | CS4014, CS1998 | Add `await` or remove unnecessary `async` |
| Cleanup | CS0105, CS0219 | Remove duplicate usings / unused variables |
| Exceptions | S3445, S2139 | Replace `throw ex;` with `throw;`; add context on rethrow |
| Web/API | ASP0019 | Use `.Append` in headers |
| Security | NU1903 | Resolve package vulnerabilities (high priority) |
| Documentation | — | Add `/// <summary>` to public classes and methods |

Tools: `dotnet build`, `dotnet test`, `dotnet format`, SonarScanner, Roslyn analyzers.

### Java

| Category | Codes / Tools | Fix |
|---|---|---|
| Logging | SLF4J placeholders, Checkstyle | Parameterized logging; avoid string concatenation in logs |
| Asynchronism | SpotBugs NP_NULL, Sonar S2190 | Proper `CompletableFuture` chaining; avoid fire-and-forget async |
| Cleanup | PMD, Checkstyle | Remove unused imports and variables |
| Exceptions | Sonar S1166, S2221 | Preserve stack trace; do not swallow exceptions |
| Web/API | Sonar S3751, S2658 | Use correct header APIs; avoid mutable static state |
| Security | OWASP dependency-check, Snyk | Update vulnerable dependencies |
| Documentation | Javadoc | Add Javadoc to public classes and methods |

Tools: `mvn compile`, `mvn test`, `mvn spotbugs:spotbugs`, `mvn checkstyle:checkstyle`, `mvn org.owasp:dependency-check-maven:check`.

### Python

| Category | Codes / Tools | Fix |
|---|---|---|
| Logging | Pylint W1203, Ruff G001 | Use `%`/f-string formatting with `logging` correctly |
| Asynchronism | Pylint W0707, Ruff ASYNC | Use `await` properly; avoid `asyncio` fire-and-forget |
| Cleanup | F401, F841 (Ruff/Flake8) | Remove unused imports and variables |
| Exceptions | Pylint W0706, W0719 | Re-raise with `raise` or `raise Custom()` with `from` |
| Web/API | Bandit B104 | Avoid hard-coded `*` in CORS; validate headers |
| Security | Bandit, Safety, Snyk | Fix high/critical CVEs in `requirements.txt` / `pyproject.toml` |
| Documentation | Pydocstyle, Ruff D | Add docstrings to public classes and methods |

Tools: `ruff check .`, `ruff format .`, `mypy`, `pylint`, `bandit -r .`, `pytest --cov=src --cov-report=xml`.

---

## Phase 3 — Architecture and Style

Refactor only when it reduces warnings or improves testability.

### SOLID
- **Single Responsibility**: split classes/modules that mix persistence, business logic, and presentation.
- **Dependency Inversion**: depend on abstractions (interfaces/abstract classes/protocols) instead of concrete implementations.

### DDD
- Identify **Aggregates**, **Entities**, **Value Objects**, and **Repositories**.
- Keep domain logic independent of frameworks and UI.

### Clean Architecture
- Validate separation between **Domain**, **Application**, **Infrastructure**, and **Presentation**.
- Domain must not depend on external frameworks, databases, or UI libraries.

---

## Phase 4 — Tests and Coverage

For framework-specific commands (xUnit/NUnit/MSTest, Maven/Gradle, pytest/unittest), thresholds, and HTML reports, see the `references/` files:

- `references/coverage-dotnet.md`
- `references/coverage-java.md`
- `references/coverage-python.md`

Use the auxiliary dispatcher to auto-detect the stack and run coverage:

```bash
bash references/run-coverage.sh
```

### .NET
```bash
dotnet test --collect:"XPlat Code Coverage" --results-directory ./TestResults
```

### Java
```bash
mvn test
# or
./gradlew test jacocoTestReport
```

### Python
```bash
pytest --cov=src --cov-report=term-missing --cov-report=xml
```

### Stabilization Rules
- Fix existing failures before creating new tests.
- Remove or skip problematic infrastructure tests (e.g. JWT, external APIs) only when they require deep refactoring, and document the reason.
- New tests follow **BDD** style: **Given / When / Then** (or **Dado / Quando / Então** for pt-BR projects).

### Coverage Goals
| Language | Minimum target |
|---|---|
| .NET | 90% line and branch |
| Java | 85% line and branch |
| Python | 90% line and branch |

Generate reports with `reportgenerator` (.NET), `JaCoCo` (Java), or `pytest-coverage` (Python).

---

## Phase 5 — Documentation and Delivery

### README.md
Update with:
- Repository structure (hierarchical tree with descriptions).
- Test coverage table: Total Tests, % Lines, % Branches.
- Technical stack list.
- Business Vision and Technical Vision sections.

### CHANGELOG.md
- Follow [Keep a Changelog](https://keepachangelog.com/)
- Use [Semantic Versioning](https://semver.org/)
- Link the changelog in the README.

### Finalization
Use **Conventional Commits**:
- `feat:` — new features
- `fix:` — bug fixes
- `test:` — tests
- `docs:` — documentation
- `refactor:` — refactorings
- `chore:` — maintenance tasks

**Restriction:** Do not open the Pull Request automatically. Prepare the commit, update the README/CHANGELOG, and generate a **Detailed Technical Summary** containing all changes so the user can open the PR manually.

---

## Quality Checklist

- [ ] No high/critical security vulnerabilities remain.
- [ ] Static-analysis warnings reduced to acceptable baseline.
- [ ] All existing tests pass.
- [ ] Coverage report generated and meets the language target.
- [ ] README updated with coverage and architecture sections.
- [ ] CHANGELOG updated with `Unreleased` changes.
- [ ] Commit message follows Conventional Commits.

## Common Mistakes

| Mistake | Consequence | How to avoid |
|------|-------------|-------------|
| `throw ex;` instead of `raise`/`throw` | Stack trace lost, root cause hidden | Re-raise with original trace; add context, do not reset |
| String concatenation in logs | Allocation/SQLi-style risk, no structured params | Use parameterized logging placeholders |
| Auto-opening the PR | User loses control of merge timing | Generate the summary; let the user open the PR |
| New tests before fixing red suite | Unstable baseline, false confidence | Stabilize existing failures first |
| Skip coverage report | No evidence target was met | Always emit `cobertura`/`jacoco`/`xml` coverage |

## References

- `references/coverage-dotnet.md` — .NET coverage (xUnit/NUnit/MSTest, Coverlet, reportgenerator, thresholds).
- `references/coverage-java.md` — Java coverage (Maven JaCoCo, Gradle JaCoCo, thresholds).
- `references/coverage-python.md` — Python coverage (pytest-cov, unittest + coverage.py, mypy, thresholds).
- `references/run-coverage.sh` — Stack-detecting dispatcher that runs the right coverage command.

## See Also

- For reviewing a single change before merge, see `code-review-and-quality`.
- For automated SonarQube issue remediation across stacks, see `sonarqube-autofix`.

## Origin

Adapted from the `devin/playbooks/multi-language-quality/PLAYBOOK.md` playbook into an agentskills.io-format skill, following the catalog standards (`license: MIT`, `metadata.version`, tripartite `description` with explicit `Do NOT use for` clause).
