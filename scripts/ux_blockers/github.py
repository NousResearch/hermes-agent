"""GitHub API transport with complete pagination and explicit mutation methods."""
from __future__ import annotations

import json
import re
import subprocess
from urllib.parse import quote


def output(argv: list[str]) -> str:
    return subprocess.run(argv, check=True, capture_output=True, text=True,
                          encoding="utf-8", timeout=120).stdout.strip()


class GitHub:
    def __init__(self, repository: str, run=output):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("Expected an owner/repository")
        self.repository = repository
        self.prefix = f"repos/{repository}"
        self.run = run
        self.permissions: dict[str, bool] = {}

    def get(self, endpoint: str):
        return json.loads(self.run(["gh", "api", endpoint]))

    def pages(self, endpoint: str) -> list[dict]:
        pages = json.loads(self.run(["gh", "api", "--paginate", "--slurp", endpoint]))
        if not isinstance(pages, list) or any(not isinstance(page, list) for page in pages):
            raise ValueError("Incomplete or invalid GitHub inventory")
        return [row for page in pages for row in page]

    def write(self, endpoint: str, method: str, **fields):
        # JSON field values are passed as individual argv entries, never shell text.
        args = ["gh", "api", "--method", method, endpoint]
        for key, value in fields.items():
            if isinstance(value, list):
                for item in value:
                    args.extend(["--raw-field", f"{key}[]={item}"])
            else:
                args.extend(["--raw-field", f"{key}={value}"])
        return self.run(args)

    def issue(self, number: int) -> dict:
        row = self.get(f"{self.prefix}/issues/{number}")
        if "pull_request" in row:
            raise ValueError("UX dispositions belong on issues, not pull requests")
        return row

    def comments(self, number: int) -> list[dict]:
        return self.pages(f"{self.prefix}/issues/{number}/comments?per_page=100")

    def indexed(self) -> list[dict]:
        return self.pages(f"{self.prefix}/issues?state=all&labels=ux%3Aconfirmed&per_page=100")

    def contains(self, candidate: str, commit: str) -> bool:
        if not re.fullmatch(r"[a-f0-9]{40}", commit):
            raise ValueError("Fix relationship has no immutable merge commit")
        comparison = self.get(f"{self.prefix}/compare/{commit}...{candidate}")
        return comparison["status"] in ("identical", "ahead")

    def trusted(self, login: str) -> bool:
        if login not in self.permissions:
            try:
                value = self.get(f"{self.prefix}/collaborators/{quote(login, safe='')}/permission")
            except subprocess.CalledProcessError as error:
                if "HTTP 404" not in str(error.stderr):
                    raise
                value = {"permission": "none"}
            self.permissions[login] = value.get("permission") in ("admin", "maintain", "write")
        return self.permissions[login]

    def links(self, number: int) -> list[dict]:
        owner, name = self.repository.split("/")
        query = """query($owner:String!,$name:String!,$number:Int!,$cursor:String) {
          repository(owner:$owner,name:$name) { issue(number:$number) {
            closedByPullRequestsReferences(first:100,after:$cursor) {
              pageInfo { hasNextPage endCursor }
              nodes { number url state mergedAt mergeCommit { oid } }
            }
          } }
        }"""
        cursor = ""
        result = []
        while True:
            args = ["gh", "api", "graphql", "-f", f"query={query}", "-f", f"owner={owner}",
                    "-f", f"name={name}", "-F", f"number={number}"]
            if cursor:
                args.extend(["-f", f"cursor={cursor}"])
            response = json.loads(self.run(args))
            if response.get("errors"):
                raise ValueError("GitHub could not return complete fixing-PR relationships")
            connection = response["data"]["repository"]["issue"]["closedByPullRequestsReferences"]
            result.extend(connection["nodes"])
            page = connection["pageInfo"]
            if not page["hasNextPage"]:
                return result
            if not page["endCursor"] or page["endCursor"] == cursor:
                raise ValueError("Fix-link pagination did not advance")
            cursor = page["endCursor"]
