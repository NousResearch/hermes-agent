#!/usr/bin/env python3
"""
Upstream monitor for GitHub releases and arXiv updates.
Creates kanban tasks for remii-deep to do deep-dive research.
Outputs a summary message for cron job consumption.
"""

import json
import os
import subprocess
from datetime import datetime, timezone, timedelta
import sys

# Configuration
GITHUB_REPOS = [
    "mem0ai/mem0",
    "memgpt/memgpt",
    "topoteretes/cognee",
    "getzep/zep",
    "openbrain-ai/openbrain"
]
ARXIV_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.RO", "stat.ML"]
STATE_FILE = "/home/kensei/.hermes/cron_states/upstream_monitor.json"
HERMES_CMD = "/home/kensei/repos/KenseiAgent/.venv/bin/hermes"  # Use the venv hermes

def run_hermes_cmd(args):
    """Run a hermes CLI command and return (success, output)."""
    try:
        env = os.environ.copy()
        env['HOME'] = '/home/kensei'
        # Set HERMES_HOME to the user's hermes directory
        env['HERMES_HOME'] = '/home/kensei/.hermes'
        # Optionally set the profile to remii to ensure correct config is loaded
        env['HERMES_PROFILE'] = 'remii'
        result = subprocess.run(
            [HERMES_CMD] + args,
            capture_output=True,
            text=True,
            timeout=10,
            env=env
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            return False, result.stderr.strip()
    except Exception as e:
        return False, str(e)

def get_available_profiles():
    """Get list of available Hermes profiles by checking the profiles directory."""
    profiles_dir = "/home/kensei/.hermes/profiles"
    if not os.path.isdir(profiles_dir):
        return []
    profiles = []
    for entry in os.listdir(profiles_dir):
        if os.path.isdir(os.path.join(profiles_dir, entry)):
            profiles.append(entry)
    return profiles

def get_assignee():
    """Determine the appropriate assignee for remii-deep tasks."""
    available = get_available_profiles()
    # Preferred order: remii-deep, remii, kensei, default
    for candidate in ["remii-deep", "remii", "kensei", "default"]:
        if candidate in available:
            return candidate
    # Fallback to first available profile if none of the above
    if available:
        return available[0]
    # Ultimate fallback
    return "default"

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {
        "github": {repo: "" for repo in GITHUB_REPOS},
        "arxiv": {"last_checked": (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()}
    }

def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def fetch_json(url):
    """Fetch JSON from URL using curl."""
    result = subprocess.run(
        ["curl", "-s", url],
        capture_output=True,
        text=True,
        timeout=10
    )
    if result.returncode == 0 and result.stdout.strip():
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return None
    return None

def check_github_release(repo, stored_tag):
    """Check for new release/tag in a GitHub repo."""
    # Try releases/latest first
    release_url = f"https://api.github.com/repos/{repo}/releases/latest"
    data = fetch_json(release_url)
    tag_name = None
    url = None

    if data and 'tag_name' in data:
        tag_name = data['tag_name']
        url = data.get('html_url', f"https://github.com/{repo}/releases/tag/{tag_name}")
    else:
        # Fallback to tags API
        tags_url = f"https://api.github.com/repos/{repo}/tags"
        tags_data = fetch_json(tags_url)
        if tags_data and isinstance(tags_data, list) and len(tags_data) > 0:
            tag_name = tags_data[0]['name']
            url = f"https://github.com/{repo}/tree/{tag_name}"

    if tag_name and tag_name != stored_tag:
        return {'tag': tag_name, 'url': url}
    return None

def check_arxiv_updates(last_checked_str):
    """
    Check for new arXiv papers in specified categories since last_checked.
    Returns list of paper dicts if found, else empty list.
    """
    # Convert ISO timestamp to arXiv date format (YYYYMMDDHHMMSS)
    try:
        dt = datetime.fromisoformat(last_checked_str.replace('Z', '+00:00'))
    except ValueError:
        # Fallback if format is unexpected
        dt = datetime.now(timezone.utc) - timedelta(days=7)
    last_checked_formatted = dt.strftime("%Y%m%d%H%M%S")

    # Build query
    categories_query = "+OR+".join([f"cat:{cat}" for cat in ARXIV_CATEGORIES])
    query = f"search_query={categories_query}+AND+submittedDate:[{last_checked_formatted} TO *]&max_results=100&sortBy=submittedDate&sortOrder=descending"
    url = f"http://export.arxiv.org/api/query?{query}"

    result = subprocess.run(
        ["curl", "-s", url],
        capture_output=True,
        text=True,
        timeout=10
    )
    if result.returncode != 0:
        return []

    xml_content = result.stdout
    try:
        import xml.etree.ElementTree as ET
        namespace = {'a': 'http://www.w3.org/2005/Atom'}
        root = ET.fromstring(xml_content)
        entries = root.findall('a:entry', namespace)

        papers = []
        for entry in entries:
            # Extract arXiv ID
            id_tag = entry.find('a:id', namespace)
            if id_tag is None:
                continue
            id_url = id_tag.text.strip()
            arxiv_id = id_url.split('/')[-1]  # e.g., 2401.01234v1
            
            # Extract title
            title_tag = entry.find('a:title', namespace)
            title = title_tag.text.strip() if title_tag is not None else ""
            
            # Extract categories
            categories = []
            for category in entry.findall('a:category', namespace):
                term = category.get('term')
                if term:
                    categories.append(term)
            
            # Construct arXiv URL
            paper_url = f"http://arxiv.org/abs/{arxiv_id}"
            
            papers.append({
                'id': arxiv_id,
                'title': title,
                'categories': categories,
                'url': paper_url
            })
        return papers
    except Exception as e:
        # Log error but don't fail
        print(f"Error parsing arXiv XML: {e}", file=sys.stderr)
        return []

def create_kanban_task(title, body, assignee):
    """Create a kanban task using hermes CLI."""
    # Note: title is a positional argument after "create"
    success, output = run_hermes_cmd([
        "kanban", "create",
        title,  # positional argument for title
        "--assignee", assignee,
        "--body", body
    ])
    if success:
        return True
    else:
        print(f"Failed to create kanban task: {output}", file=sys.stderr)
        return False

def main():
    """Main monitoring logic."""
    try:
        state = load_state()
        updates_found = 0
        tasks_created = 0
        assignee = get_assignee()

        # Check GitHub repos
        for repo in GITHUB_REPOS:
            stored_tag = state['github'].get(repo, "")
            update = check_github_release(repo, stored_tag)
            if update:
                updates_found += 1
                title = f"[remii-deep] New release: {repo} {update['tag']}"
                body = (
                    f"New release/tag detected for {repo}: {update['tag']}\n\n"
                    f"URL: {update['url']}\n\n"
                    f"Please do a deep-dive research on this update and determine if it's relevant for feature adoption."
                )
                if create_kanban_task(title, body, assignee):
                    tasks_created += 1
                    state['github'][repo] = update['tag']

        # Check arXiv
        last_checked = state['arxiv']['last_checked']
        new_papers = check_arxiv_updates(last_checked)
        if new_papers:
            updates_found += len(new_papers)
            title = f"[remii-deep] New arXiv papers in {', '.join(ARXIV_CATEGORIES)}"
            body_lines = [
                f"Found {len(new_papers)} new paper(s) in the last week:\n"
            ]
            for paper in new_papers:
                body_lines.append(
                    f"- {paper['title']} ({paper['id']})\n"
                    f"  URL: {paper['url']}\n"
                    f"  Categories: {', '.join(paper['categories'])}\n"
                )
            body_lines.append(
                "\nPlease do a deep-dive research on these papers and determine if any are relevant for feature adoption."
            )
            body = "\n".join(body_lines)
            if create_kanban_task(title, body, assignee):
                tasks_created += 1
                state['arxiv']['last_checked'] = datetime.now(timezone.utc).isoformat()

        # Save state
        save_state(state)

        # Determine output
        if updates_found > 0 and tasks_created > 0:
            print(f"Found {updates_found} updates and created {tasks_created} research tasks.")
        elif updates_found > 0:
            print(f"Found {updates_found} updates but failed to create any research tasks.")
        else:
            print("[SILENT]")
            
    except Exception as e:
        print(f"Error in upstream monitor: {e}", file=sys.stderr)
        print(f"Error in upstream monitor: {e}")

if __name__ == "__main__":
    main()