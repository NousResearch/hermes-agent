"""``hermes prreview`` — operator/agent CLI for the PR-review buttons flow.

Two subcommands, both used by the cron review sweep:

* ``stage`` — record ONE PR's verbatim review (read from ``--body-file`` or
  stdin) into the pending store. The agent calls this after reviewing each PR.
* ``publish`` — post the buttoned Block Kit digest of all not-yet-posted staged
  reviews to the Slack home channel, then mark them posted. Called once at the
  end of the sweep.

``stage`` never touches Slack (works offline / in CI). ``publish`` needs
``SLACK_BOT_TOKEN`` + a channel; ``hermes`` loads ``~/.hermes/.env`` at import
so the token is present.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from . import blocks, slackio, store


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="prreview_action")

    stage_p = subs.add_parser("stage", help="Stage one PR's verbatim review for button-posting")
    stage_p.add_argument("--repo", required=True, help="owner/repo, e.g. Strike48/matrix")
    stage_p.add_argument("--number", required=True, type=int, help="PR number")
    stage_p.add_argument("--head-sha", required=True, help="PR head SHA at review time")
    stage_p.add_argument("--title", default="", help="PR title")
    stage_p.add_argument("--url", default="", help="PR URL")
    stage_p.add_argument("--verdict", default="", help="One-line verdict (LGTM / Needs work / Blocker)")
    stage_p.add_argument("--body-file", default="", help="File with the review body; '-' or omit reads stdin")

    pub_p = subs.add_parser("publish", help="Post the buttoned digest of staged reviews to Slack")
    pub_p.add_argument("--channel", default="", help="Slack channel id (default: SLACK_HOME_CHANNEL)")

    subs.add_parser("list", help="List staged (pending) reviews")


def prreview_command(args: argparse.Namespace) -> int:
    action = getattr(args, "prreview_action", None)
    if action == "stage":
        return _cmd_stage(args)
    if action == "publish":
        return asyncio.run(_cmd_publish(args))
    if action == "list":
        return _cmd_list(args)
    print("usage: hermes prreview {stage|publish|list}", file=sys.stderr)
    return 2


def _read_body(body_file: str) -> str:
    if not body_file or body_file == "-":
        return sys.stdin.read()
    with open(body_file, "r", encoding="utf-8") as fh:
        return fh.read()


def _cmd_stage(args: argparse.Namespace) -> int:
    body = _read_body(args.body_file).strip()
    if not body:
        print("prreview stage: empty review body", file=sys.stderr)
        return 1
    key = store.stage(
        {
            "repo": args.repo,
            "number": args.number,
            "head_sha": args.head_sha,
            "title": args.title,
            "url": args.url,
            "verdict": args.verdict,
            "body": body,
        }
    )
    print(f"staged {key}")
    return 0


async def _cmd_publish(args: argparse.Namespace) -> int:
    pending = store.unpublished()
    if not pending:
        print("nothing to publish")
        return 0

    channel = args.channel or slackio.home_channel()
    if not channel:
        print("prreview publish: no channel (set --channel or SLACK_HOME_CHANNEL)", file=sys.stderr)
        return 1

    try:
        client = slackio.make_client()
    except Exception as exc:
        print(f"prreview publish: {exc}", file=sys.stderr)
        return 1

    bk = blocks.build_digest_blocks(pending)
    text = f"{len(pending)} PR review(s) awaiting your decision"
    try:
        ts = await slackio.post_message(client, channel, text, bk)
    except Exception as exc:
        print(f"prreview publish: Slack post failed: {exc}", file=sys.stderr)
        return 1

    for entry in pending:
        store.mark_published(entry["_key"], message_ts=ts)
    print(f"published {len(pending)} review(s) to {channel} (ts={ts})")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    data = store.load(store.default_path())
    if not data:
        print("no staged reviews")
        return 0
    for key, entry in data.items():
        state = "posted" if entry.get("message_ts") else "PENDING"
        print(f"{state:8} {key}  {entry.get('verdict','')}")
    return 0
