# Investigation Templates

Pre-built hypothesis and investigation templates for common supply chain attack scenarios.
Each template includes: attack pattern, key evidence to collect, and hypothesis starters.

---

## Template 1: Maintainer Account Compromise

**Pattern**: Attacker gains access to a legitimate maintainer account (phishing, credential stuffing)
and uses it to push malicious code, create backdoored releases, or exfiltrate CI secrets.

**Real-world examples**: XZ Utils (2024), Codecov (2021), event-stream (2018)

**Key Evidence to Collect**:
- [ ] Push events from maintainer account outside normal working hours/timezone
- [ ] Commits adding new dependencies, obfuscated code, or modified build scripts
- [ ] Release creation immediately after suspicious push (to maximize package distribution)
- [ ] MemberEvent adding unknown collaborators (attacker adding backup access)
- [ ] WorkflowRunEvent with unexpected secret access or exfiltration-like behavior
- [ ] Account login location changes (check social media, conference talks for corroboration)

**Hypothesis Starters**:
```
[HYPOTHESIS] Actor <HANDLE>'s account was compromised on or around <DATE>, 
based on anomalous commit timing [EV-XXXX] and geographic access patterns [EV-YYYY].
```
```
[HYPOTHESIS] Release <VERSION> was published by the compromised account to push 
malicious code to downstream users, evidenced by the malicious commit [EV-XXXX] 
being added <N> hours before the release [EV-YYYY].
```

---

## Template 2: Malicious Dependency Injection

**Pattern**: A trusted package is modified to include malicious code in a dependency,
or a new malicious dependency is injected into an existing package.

**Key Evidence to Collect**:
- [ ] Diff of `package.json`/`requirements.txt`/`go.mod` before and after suspicious commit
- [ ] The new dependency's publication timestamp vs. the injection commit timestamp
- [ ] Whether the new dependency exists on npm/PyPI and who owns it
- [ ] Any obfuscation patterns in the injected dependency code
- [ ] Install-time scripts (`postinstall`, `setup.py`, etc.) that execute code on install

**Hypothesis Starters**:
```
[HYPOTHESIS] Commit <SHA> [EV-XXXX] introduced dependency <PACKAGE@VERSION> 
which appears to be a malicious package published by actor <HANDLE> [EV-YYYY], 
designed to execute <BEHAVIOR> during installation.
```

---

## Template 3: CI/CD Pipeline Injection

**Pattern**: Attacker modifies GitHub Actions workflows to steal secrets, exfiltrate code,
or inject malicious artifacts into the build output.

**Key Evidence to Collect**:
- [ ] Diff of all `.github/workflows/*.yml` files before/after suspicious period
- [ ] WorkflowRunEvents triggered by the modified workflows
- [ ] Any `curl`, `wget`, or network calls added to workflow steps
- [ ] New or modified `env:` sections referencing `secrets.*`
- [ ] Artifacts produced by modified workflow runs

**Hypothesis Starters**:
```
[HYPOTHESIS] Workflow file <FILE> was modified in commit <SHA> [EV-XXXX] to 
exfiltrate repository secrets via <METHOD>, as evidenced by the added network 
call pattern [EV-YYYY].
```

---

## Template 4: Typosquatting / Dependency Confusion

**Pattern**: Attacker registers a package with a name similar to a popular package
(or an internal package name) to intercept installs from users who mistype.

**Key Evidence to Collect**:
- [ ] Registration timestamp of the suspicious package on the registry
- [ ] Package content: does it contain malicious code or is it a stub?
- [ ] Download statistics for the suspicious package
- [ ] Names of internal packages that could be targeted (if private repo scope)
- [ ] Any references to the legitimate package in the malicious one's metadata

**Hypothesis Starters**:
```
[HYPOTHESIS] Package <MALICIOUS_NAME> was registered on <DATE> [EV-XXXX] to 
typosquat on <LEGITIMATE_NAME>, targeting users who misspell the package name. 
The package contains <BEHAVIOR> [EV-YYYY].
```

---

## Template 5: Force-Push History Rewrite (Evidence Erasure)

**Pattern**: After a malicious commit is detected (or before wider notice), the attacker
force-pushes to remove the malicious commit from branch history.

**Detection is key** — this template focuses on proving the erasure happened.

**Key Evidence to Collect**:
- [ ] GH Archive PushEvent with `distinct_size=0` (force push indicator) [EV-XXXX]
- [ ] The SHA of the commit BEFORE the force push (from GH Archive `payload.before`)
- [ ] Recovery of the erased commit via direct URL or `git fetch origin SHA`
- [ ] Wayback Machine snapshot of the commit page before erasure
- [ ] Timeline gap in git log (N commits visible in archive but M < N in current repo)

**Hypothesis Starters**:
```
[HYPOTHESIS] Actor <HANDLE> force-pushed branch <BRANCH> on <DATE> [EV-XXXX] 
to erase commit <SHA> [EV-YYYY], which contained <MALICIOUS_CONTENT>. 
The erased commit was recovered via <METHOD> [EV-ZZZZ].
```

---

## Cross-Cutting Investigation Checklist

Apply to every investigation regardless of template:

- [ ] Check all contributors for newly created accounts (< 30 days old at time of malicious activity)
- [ ] Check if any maintainer account changed email in the period (sign of account takeover)
- [ ] Verify GPG signatures on suspicious commits match known maintainer keys
- [ ] Check if the repository changed ownership or transferred orgs near the incident
- [ ] Look for "cleanup" commits immediately after the malicious commit (cover-up pattern)
- [ ] Check related packages/repos by the same author for similar patterns
