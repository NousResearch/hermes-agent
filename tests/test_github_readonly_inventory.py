from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills/github/github-auth/scripts/github-readonly-inventory.py"


def test_inventory_prefers_authenticated_gh_without_token_or_topics(
    tmp_path: Path,
) -> None:
    gh = tmp_path / "gh"
    gh.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail

            if [ "$1" = "auth" ] && [ "$2" = "status" ]; then
              echo "Logged in to github.com"
              exit 0
            fi

            if [ "$1" = "api" ] && [ "$2" = "user" ]; then
              echo "ryouchida715"
              exit 0
            fi

            if [ "$1" = "repo" ] && [ "$2" = "view" ]; then
              cat <<'JSON'
            {
              "nameWithOwner": "ryouchida715/GH_Hermes",
              "description": "",
              "defaultBranchRef": {"name": "main"},
              "isPrivate": true,
              "pushedAt": "2026-06-25T12:50:12Z",
              "updatedAt": "2026-06-25T12:50:17Z",
              "url": "https://github.com/ryouchida715/GH_Hermes"
            }
            JSON
              exit 0
            fi

            if [ "$1" = "issue" ] && [ "$2" = "list" ]; then
              echo '[]'
              exit 0
            fi

            if [ "$1" = "pr" ] && [ "$2" = "list" ]; then
              echo '[]'
              exit 0
            fi

            echo "unexpected gh args: $*" >&2
            exit 2
            """
        )
    )
    gh.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}",
        "HOME": str(tmp_path),
    }
    env.pop("GITHUB_TOKEN", None)
    env.pop("GH_TOKEN", None)

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "ryouchida715/GH_Hermes"],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    data = json.loads(completed.stdout)
    assert data["auth"]["method"] == "gh"
    assert data["auth"]["gh_authenticated"] is True
    assert data["auth"]["user"] == "ryouchida715"
    assert data["repos"][0]["status"] == "ok"
    assert data["repos"][0]["summary"]["nameWithOwner"] == "ryouchida715/GH_Hermes"
    assert data["repos"][0]["summary"]["defaultBranch"] == "main"
    assert data["repos"][0]["openIssues"] == []
    assert data["repos"][0]["openPullRequests"] == []
