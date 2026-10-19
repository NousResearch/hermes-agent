"""Idempotent, bot-authenticated publication; no worker prose crosses this boundary."""
import re


def publish_verified_comment(auth, cfg, pins, marker, body):
    from hermes_cli.issue_ingress import AdmissionRefused
    installation = auth.verify_installation()
    slug = installation.get("app_slug") if isinstance(installation, dict) else None
    if (not isinstance(installation, dict) or installation.get("app_id") != pins.app_id
            or not isinstance(slug, str) or not re.fullmatch(r"[A-Za-z0-9-]{1,100}", slug)):
        raise AdmissionRefused("RESULT_APP_IDENTITY_UNVERIFIED")
    bot = f"{slug}[bot]".casefold()

    def read_match():
        rows = auth.list_issue_comments(cfg.owner, cfg.repo, pins.issue_number, per_page=100)
        if not isinstance(rows, list) or len(rows) > 100:
            raise AdmissionRefused("RESULT_COMMENT_PAGE_INVALID")
        matches = [row for row in rows if isinstance(row, dict) and marker in str(row.get("body") or "")]
        if len(matches) > 1:
            raise AdmissionRefused("RESULT_MARKER_AMBIGUOUS")
        if not matches:
            return None, len(rows)
        row = matches[0]; user = row.get("user")
        if (type(row.get("id")) is not int or row["id"] <= 0 or row.get("body") != body
                or not isinstance(user, dict) or user.get("type") != "Bot"
                or str(user.get("login") or "").casefold() != bot):
            raise AdmissionRefused("RESULT_MARKER_UNTRUSTED")
        return row["id"], len(rows)

    result_id, count = read_match()
    if result_id is not None:
        return result_id
    if count >= 100:
        raise AdmissionRefused("RESULT_COMMENT_HISTORY_FULL")
    posted = auth.post_issue_comment(cfg.owner, cfg.repo, pins.issue_number, body)
    result_id, _count = read_match()
    if not isinstance(posted, dict) or result_id is None or result_id != posted.get("id"):
        raise AdmissionRefused("RESULT_POST_UNVERIFIED")
    return result_id
