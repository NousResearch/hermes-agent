# Git and GitHub Data Recovery Techniques

This reference guide outlines techniques for recovering deleted or tampered evidence during an investigation.

## 1. Recovering Force-Pushed (Orphaned) Commits

If a developer force-pushes, the previous commits become orphaned (not reachable by any branch) but still exist in the Git database until garbage collection.

### Local Repo Recovery
If you have the repo cloned locally:
```bash
git reflog
# Look for 'forced-update' or 'reset' entries
git show [SHA]
```

### GitHub Remote Recovery (Direct Access)
GitHub often keeps orphaned objects reachable via direct SHA URL, even if not in the UI:
- `https://github.com/[owner]/[repo]/commit/[SHA]`
- `https://github.com/[owner]/[repo]/archive/[SHA].zip`

## 2. GitHub Archive (BigQuery)
Used for recovering event data (who did what and when) even if the event was followed by a deletion.
- **PushEvent**: Contains SHAs of pushed commits.
- **MemberEvent**: Tracks when collaborators are added/removed.

## 3. Wayback Machine (Web Archive)
Recovers snapshots of GitHub's HTML views.
- **URL Pattern**: `https://github.com/[owner]/[repo]/issues/[ID]`
- **Search API**: Use `web_extract(url="https://web.archive.org/cdx/search/cdx?url=github.com/[owner]/[repo]&output=json")`

## 4. Recovering Infrastructure Logs
- **WorkflowRunEvents**: Check for malicious runner logs or modified CI YAML files.
- **ReleaseEvent**: Check for differences between the release tag and the source code at that tag.
