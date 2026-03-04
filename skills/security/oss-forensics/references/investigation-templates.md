# Investigation Scenarios and Templates

Common supply chain attack patterns and how to investigate them.

## 1. The Maintainer Hijack (XZ Utils Style)
**Trigger**: Long-term contributor suddenly pushes radical changes or becomes uncharacteristically active after a long lull.
- **Steps**:
  1. Analyze `CreateEvent` vs `PushEvent` for the actor in GitHub Archive.
  2. Search for linked accounts or social engineering attempts in Issues/PRs.
  3. Compare commit timestamps with the actor's historical patterns.

## 2. Dependency Confusion / Typosquatting
**Trigger**: A new package version or a similar-sounding package is released.
- **Steps**:
  1. Check `package.json` or `requirements.txt` changes.
  2. Investigate the downstream impact of the suspicious dependency using `execute_code`.

## 3. Workflow Poisoning (CI/CD Compromise)
**Trigger**: Malicious action added to `.github/workflows/`.
- **Steps**:
  1. Inspect `WorkflowRunEvents` in GitHub Archive.
  2. Check for unauthorized secrets usage or exfiltration attempts in logs.

## 4. Forced History Overwrite
**Trigger**: Repository history is rewritten to hide a previous compromise.
- **Steps**:
  1. Use GitHub Archive to find the original SHAs from before the force-push.
  2. Compare the pre-overwrite and post-overwrite states.
