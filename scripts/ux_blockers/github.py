"""GitHub API transport with complete pagination and explicit mutation methods."""
from __future__ import annotations

import json
import os
import re
import subprocess
from urllib.parse import quote

from scripts.ux_blockers.policy import BOT, LEDGER_ISSUE, RECEIPT


def output(argv: list[str]) -> str:
    return subprocess.run(argv, check=True, capture_output=True, text=True,
                          encoding="utf-8", timeout=120).stdout.strip()


class GitHub:
    def __init__(self, repository: str, run=output, ledger_issue: int | None = None):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("Expected an owner/repository")
        self.repository = repository
        self.prefix = f"repos/{repository}"
        self.run = run
        self.permissions: dict[str, bool] = {}
        self.ledger_issue = ledger_issue or int(os.environ.get("UX_LEDGER_ISSUE") or LEDGER_ISSUE)
        if self.ledger_issue <= 0:
            raise ValueError("Select a positive UX confirmation ledger issue number")
        self._receipts = None

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
        rows = {row["number"]: row for row in self.pages(
            f"{self.prefix}/issues?state=all&labels=ux%3Aconfirmed&per_page=100")}
        # Labels are a projection. The fixed issue ledger keeps acknowledged
        # confirmations discoverable after source deletion or label removal.
        for receipt in self.receipts():
            number = receipt["issue"]
            if number not in rows:
                rows[number] = self.issue(number)
            rows[number].setdefault("_ux_receipts", []).append(receipt["comment"])
        return list(rows.values())

    def receipts(self) -> list[dict]:
        if self._receipts is None:
            receipts = []
            for row in self.comments(self.ledger_issue):
                if row["user"].get("login") != BOT or row["user"].get("type") != "Bot":
                    continue
                body = row.get("body") or ""
                if not body.startswith(RECEIPT):
                    continue
                if row["updated_at"] != row["created_at"]:
                    raise ValueError("UX confirmation ledger record was edited")
                value = json.loads(body[len(RECEIPT):].strip())
                if (not isinstance(value.get("issue"), int) or value["issue"] <= 0
                        or not isinstance(value.get("id"), int) or value["id"] <= 0
                        or re.fullmatch(r"[a-f0-9]{64}", value.get("sha256", "")) is None):
                    raise ValueError("Invalid UX confirmation ledger record")
                receipts.append({**value, "comment": row})
            self._receipts = receipts
        return self._receipts

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
            closedByPullRequestsReferences(first:100,after:$cursor,includeClosedPrs:true) {
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
