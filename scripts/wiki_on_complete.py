#!/usr/bin/env python3
"""
Wiki On Complete — post-processing step that reads recently completed kanban tasks
and updates the corresponding LLM-WIKI repo entry with 'implemented' adoption_status.

Runs after the quality gate as a no_agent=true cron at X:15 past each hour.

Scans kanban tasks completed in the last 2 hours. If the task result or body
references a wiki repo page (via "wiki:" or "repo:" prefix), updates that page's
frontmatter to adoption_status: implemented and tags: [implemented].
"""
import sqlite3, json, re, os, sys, time
from pathlib import Path
from datetime import datetime, timezone, timedelta

# P13 isolation: paths derive from HERMES_HOME / WIKI_DIR (env-overridable)
# so local disposable runs never touch /home/kensei/.hermes or the
# production wiki repos checkout.
_HERMES_HOME = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
KANBAN_DB = _HERMES_HOME / "kanban.db"
WIKI_REPOS = Path(os.environ.get("WIKI_DIR", os.path.expanduser("~/docs/wiki"))) / "repos"

# P13 isolation: when --dry-run is passed, the wiki page rewrite is
# suppressed. Read paths (kanban DB query, find_wiki_reference) run
# unchanged so the task scan is still exercised.
_DRY_RUN = False

TZ = timezone(timedelta(hours=1))
now = datetime.now(TZ)
cutoff = int((datetime.now(timezone.utc) - timedelta(hours=2)).timestamp())

def find_wiki_reference(text):
    """Extract repo short-name from text matching wiki:short-name or repo:owner/repo."""
    if not text or not isinstance(text, str):
        return None
    try:
        m = re.search(r'wiki:(\S+)', text)
        if m:
            return m.group(1)
        m = re.search(r'repo:(\S+)', text)
        if m:
            full = m.group(1)
            short = full.split('/')[-1] if '/' in full else full
            return short.lower()
        return None
    except (re.error, TypeError) as e:
        print(f"⚠️ Wiki sync · reference parse error: {e}")
        return None

def update_wiki_page(repo_name):
    """Update a wiki repo page's frontmatter to mark it as implemented."""
    if not repo_name or not isinstance(repo_name, str):
        return "invalid"

    page_path = WIKI_REPOS / f"{repo_name}.md"
    if not page_path.exists():
        return "missing"

    try:
        content = page_path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as e:
        print(f"⚠️ Wiki sync · read error for {repo_name}: {e}")
        return "read_error"

    # Check if already implemented
    if 'adoption_status: implemented' in content:
        return "already"

    # Update frontmatter adoption_status
    updated = re.sub(
        r'(adoption_status:\s*)\S+',
        r'\1implemented',
        content
    )

    # Add implemented tag if not present
    if 'implemented' not in updated:
        updated = re.sub(
            r'(tags:\s*\[.*?)\]',
            r'\1, implemented]',
            updated
        )

    updated = re.sub(
        r'(updated:\s*)\S+',
        f'\\1{now.strftime("%Y-%m-%d")}',
        updated
    )

    # Add what_we_did note if empty
    if 'what_we_did: ""' in updated:
        updated = updated.replace(
            'what_we_did: ""',
            f'what_we_did: "Built and deployed via kanban pipeline. Completed {now.strftime("%Y-%m-%d")}."'
        )

    if _DRY_RUN:
        return "would-update"
    try:
        page_path.write_text(updated, encoding='utf-8')
    except OSError as e:
        print(f"⚠️ Wiki sync · write error for {repo_name}: {e}")
        return "write_error"

    return "updated"

def main():
    global _DRY_RUN
    if "--dry-run" in sys.argv:
        _DRY_RUN = True
    if not KANBAN_DB.exists():
        print("SKIP: kanban DB not found")
        return

    conn = None
    try:
        conn = sqlite3.connect(str(KANBAN_DB))
        conn.execute("PRAGMA query_only = 1")  # safety valve — read-only
        cur = conn.cursor()

        # Find tasks completed in the last 2 hours with wiki references
        cur.execute("""
            SELECT id, title, result, body
            FROM tasks
            WHERE status = 'done'
              AND completed_at >= ?
              AND (result LIKE '%wiki:%' OR result LIKE '%repo:%' 
                   OR body LIKE '%wiki:%' OR body LIKE '%repo:%')
            ORDER BY completed_at DESC
        """, (cutoff,))

        tasks = cur.fetchall()
    except sqlite3.Error as e:
        print(f"❌ Wiki sync · query error: {e}")
        return
    except Exception as e:
        print(f"❌ Wiki sync · unexpected query error: {e}")
        return
    finally:
        if conn:
            conn.close()

    if not tasks:
        return

    updated = []
    missing = []
    for task_id, title, result, body in tasks:
        ref = find_wiki_reference(result) or find_wiki_reference(body) or find_wiki_reference(title)
        if not ref:
            continue
        outcome = update_wiki_page(ref)
        if outcome == "updated":
            updated.append(ref)
        elif outcome == "missing":
            missing.append(ref)
    # no_agent silence contract: no stdout when there was no action and no real error.
    if not updated and not missing:
        return

    if updated:
        print(f"📚 Wiki sync · {now.strftime('%d/%m/%Y %H:%M:%S')}")
        print(f"updated · {len(updated)} repo page(s): {', '.join(updated)}")
    if missing:
        print(f"⚠️ Wiki sync missing pages · {now.strftime('%d/%m/%Y %H:%M:%S')}")
        print(f"missing · {len(missing)} repo page(s): {', '.join(missing)}")

if __name__ == "__main__":
    try:
        main()
    except sqlite3.Error as e:
        print(f"❌ Wiki sync · database error: {e}")
        sys.exit(1)
    except OSError as e:
        print(f"❌ Wiki sync · file error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Wiki sync · unexpected error: {e}")
        sys.exit(1)