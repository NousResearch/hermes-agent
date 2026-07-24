#!/usr/bin/env python3
"""
Facebook Graph API Helper — Hermes Agent Skill

Usage:
  python3 facebook_graph.py <action> [args...]

Actions:
  list-pages             List your Facebook Pages
  get-page <page-id>     Get page details
  create-post <page-id>  Post something on a page
  get-posts <page-id>    See recent posts
  get-engagement <page-id>  See likes, comments, shares
  reply-comment <cid>    Reply to a comment
  list-conversations <id>  Read page inbox
  send-message <convo-id>  Send a message
  list-groups            List your groups
  group-post <group-id>  Post to a group
  group-members <id>     See group members
  token-check            Verify your access token

Requires:
  FACEBOOK_ACCESS_TOKEN environment variable to be set.
"""

import os
import sys
import json
import urllib.request
import urllib.parse
import urllib.error

API_VERSION = os.environ.get("FACEBOOK_API_VERSION", "v19.0")
TOKEN = os.environ.get("FACEBOOK_ACCESS_TOKEN", "")


# ---------------------------------------------------------------------------
#  API helper
# ---------------------------------------------------------------------------

def _api(endpoint, method="GET", params=None, data=None):
    """Make a Facebook Graph API call. Returns dict with 'success' and 'data'."""
    url = f"https://graph.facebook.com/{API_VERSION}/{endpoint}"
    params = dict(params or {})
    params["access_token"] = TOKEN

    if method == "GET":
        qs = urllib.parse.urlencode(params)
        full_url, body = f"{url}?{qs}", None
    else:
        full_url, body = url, urllib.parse.urlencode(params).encode()
        data = None  # params already encoded above

    req = urllib.request.Request(full_url, method=method)
    if method != "GET":
        req.add_header("Content-Type", "application/x-www-form-urlencoded")

    try:
        with urllib.request.urlopen(req, data=body or data, timeout=30) as resp:
            return {"success": True, "data": json.loads(resp.read().decode())}
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return {"success": False, "error": json.loads(raw)}
        except json.JSONDecodeError:
            return {"success": False, "error": {"raw": raw, "code": e.code}}


def _require_token():
    if not TOKEN:
        print("❌ FACEBOOK_ACCESS_TOKEN is not set.", file=sys.stderr)
        print("   Add it to ~/.hermes/profiles/mybot/.env:", file=sys.stderr)
        print("   export FACEBOOK_ACCESS_TOKEN='your_token_here'", file=sys.stderr)
        sys.exit(1)


def _read_message():
    """If no message passed on command line, read from stdin."""
    if len(sys.argv) > 3:
        return " ".join(sys.argv[3:])
    print("📝 Type your message (Ctrl+D to finish):", file=sys.stderr)
    return sys.stdin.read().strip()


# ---------------------------------------------------------------------------
#  Actions
# ---------------------------------------------------------------------------

def action_list_pages():
    """List pages you manage."""
    r = _api("me/accounts", params={"fields": "id,name,category,access_token,followers_count,likes"})
    if not r["success"]:
        print("Error:", json.dumps(r["error"], indent=2), file=sys.stderr); sys.exit(1)
    pages = r["data"].get("data", [])
    if not pages:
        print("No pages found. Make sure you're logged into the correct account.")
        return
    for p in pages:
        print(f"\n📄 [{p['id']}] {p['name']}")
        print(f"   Category: {p.get('category', '?')}")
        print(f"   Followers: {p.get('followers_count', '?')}")


def action_get_page(page_id):
    """Get page details."""
    r = _api(page_id, params={"fields": "id,name,about,description,category,followers_count,likes,link,website"})
    if not r["success"]:
        print("Error:", json.dumps(r["error"], indent=2), file=sys.stderr); sys.exit(1)
    d = r["data"]
    print(f"\n📄 {d.get('name', '?')}")
    print(f"   ID: {d.get('id', '?')}")
    print(f"   About: {d.get('about', d.get('description', 'N/A'))}")
    print(f"   Category: {d.get('category', '?')}")
    print(f"   Followers: {d.get('followers_count', '?')}")
    print(f"   Likes: {d.get('likes', '?')}")
    print(f"   Link: {d.get('link', '?')}")


def action_create_post(page_id):
    """Create a post on a page."""
    message = _read_message()
    if not message:
        print("❌ No message provided.", file=sys.stderr); sys.exit(1)
    r = _api(f"{page_id}/feed", method="POST", data={"message": message})
    if not r["success"]:
        print("Error:", json.dumps(r["error"], indent=2), file=sys.stderr); sys.exit(1)
    post_id = r["data"].get("id", "unknown")
    print(f"✅ Post created!")
    print(f"   ID: {post_id}")
    print(f"   URL: https://facebook.com/{post_id}")


def action_get_posts(page_id):
    """Get recent posts from a page."""
    r = _api(f"{page_id}/posts", params={
        "fields": "id,message,created_time,permalink_url,story,status_type",
        "limit": 20
    })
    if not r["success"]:
        print("Error:", json.dumps(r["error"], indent=2), file=sys.stderr); sys.exit(1)
    posts = r["data"].get("data", [])
    print(f"\n📝 Recent posts ({len(posts)}):\n")
    for p in posts:
        msg = (p.get("message") or p.get("story") or "(no text)")[:100]
        print(f"  [{p['id']}]")
        print(f"     {msg}")
        print(f"     🕐 {p.get('created_time', '?')}")
        print(f"     🔗 {p.get('permalink_url', '?')}\n")


def action_get_engagement(page_id):
    """Get post engagement stats."""
    r = _api(f"{page_id}/posts", params={
        "fields": "id,message,created_time,permalink_url,"
                  "likes.limit(1).summary(true),"
                  "comments.limit(1).summary(true),"
                  "shares",
        "limit": 10
    })
    if not r["success"]:
        print("Error:", json.dumps(r["error"], indent=2), file=sys.stderr); sys.exit(1)
    posts = r["data"].get("data", [])
    print("\n📊 Post Engagement:\n")
    for p in posts:
        msg = (p.get("message") or "(no text)")[:80]
        likes = p.get("likes", {}).get("summary", {}).get("total_count", 0)
        comments = p.get("comments", {}).get("summary", {}).get("total_count", 0)
        shares = p.get("shares", {})
        shares_c = shares.get("count", 0) if isinstance(shares, dict) else shares
        print(f"  [{p['id']}] {msg}")
        print(f"     ❤️ {likes}  💬 {comments}  🔄 {shares_c}\n")


def action_reply_comment(comment_id):
    """Reply to a comment."""
    message = _read_message()
    if not message:
        print("❌ No reply provided.", file=sys.stderr); sys.exit(1)
    r = _api(f"{comment_id}/comments", method="POST", data={"message": message})
    if not r["success"]:
        print("Error:", json.dumps(r["error"], indent=2), file=sys.stderr); sys.exit(1)
    print(f"✅ Reply posted! ID: {r['data'].get('id')}")


def action_list_conversations(page_id):
    """List inbox conversations for a page."""
    r = _api(f"{page_id}/conversations", params={
        "fields": "id,message_count,unread_count,updated_time,"
                  "senders,messages.limit(1){message,from,created_time}",
        "limit": 20
    })
    if not r["success"]:
        print("Error:", json.dumps(r["error"], indent=2), file=sys.stderr); sys.exit(1)
    convos = r["data"].get("data", [])
    print(f"\n📨 Conversations ({len(convos)}):\n")
    for c in convos:
        msgs = c.get("messages", {}).get("data", [])
        last = msgs[0]["message"][:80] if msgs else "(no messages)"
        sender = msgs[0].get("from", {}).get("name", "?") if msgs else "?"
        print(f"  [{c['id']}]")
        print(f"     From: {sender}")
        print(f"     Last: {last}")
        print(f"     📬 {c.get('message_count', '?')} msgs ({c.get('unread_count', '?')} unread)")


def action_send_message(conversation_id):
    """Send a message in a conversation."""
    message = _read_message()
    if not message:
        print("❌ No message provided.", file=sys.stderr); sys.exit(1)
    r = _api(f"{conversation_id}/messages", method="POST", data={"message": message})
    if not r["success"]:
        print("Error:", json.dumps(r["error"], indent=2), file=sys.stderr); sys.exit(1)
    print(f"✅ Message sent! ID: {r['data'].get('id')}")


def action_list_groups():
    """List groups you belong to."""
    r = _api("me/groups", params={
        "fields": "id,name,description,member_count,privacy,administrator",
        "limit": 50
    })
    if not r["success"]:
        print("Error:", json.dumps(r["error"], indent=2), file=sys.stderr); sys.exit(1)
    groups = r["data"].get("data", [])
    print(f"\n👥 Groups ({len(groups)}):\n")
    for g in groups:
        admin = " [Admin]" if g.get("administrator", False) else ""
        print(f"  [{g['id']}] {g['name']}{admin}")
        print(f"     Members: {g.get('member_count', '?')}  Privacy: {g.get('privacy', '?')}\n")


def action_group_post(group_id):
    """Post to a group."""
    message = _read_message()
    if not message:
        print("❌ No message provided.", file=sys.stderr); sys.exit(1)
    r = _api(f"{group_id}/feed", method="POST", data={"message": message})
    if not r["success"]:
        print("Error:", json.dumps(r["error"], indent=2), file=sys.stderr); sys.exit(1)
    post_id = r["data"].get("id", "unknown")
    print(f"✅ Posted to group!")
    print(f"   ID: {post_id}")
    print(f"   URL: https://facebook.com/{post_id}")


def action_group_members(group_id):
    """List group members."""
    r = _api(f"{group_id}/members", params={
        "fields": "id,name,administrator,owner",
        "limit": 50
    })
    if not r["success"]:
        print("Error:", json.dumps(r["error"], indent=2), file=sys.stderr); sys.exit(1)
    members = r["data"].get("data", [])
    print(f"\n👤 Members ({len(members)}):\n")
    for m in members:
        roles = []
        if m.get("administrator"): roles.append("Admin")
        if m.get("owner"): roles.append("Owner")
        tag = f" [{', '.join(roles)}]" if roles else ""
        print(f"  [{m['id']}] {m['name']}{tag}")


def action_token_check():
    """Check if your token is valid and what permissions it has."""
    r = _api("debug_token", params={"input_token": TOKEN})
    if not r["success"]:
        print("Error:", json.dumps(r["error"], indent=2), file=sys.stderr); sys.exit(1)
    d = r["data"]
    print("\n=== 🔑 Token Info ===")
    print(f"  Valid:       {'✅' if d.get('is_valid') else '❌'} {d.get('is_valid')}")
    print(f"  App ID:      {d.get('app_id', '?')}")
    print(f"  Type:        {d.get('type', '?')}")
    print(f"  User ID:     {d.get('user_id', '?')}")
    scopes = d.get("granular_scopes", [])
    print(f"\n  Permissions ({len(scopes)}):")
    for s in scopes:
        print(f"    - {s.get('scope', '?')}")


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

def main():
    _require_token()

    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable actions:")
        for name in sorted(globals()):
            if name.startswith("action_"):
                print(f"  {name.replace('action_', '')}")
        sys.exit(0)

    action = sys.argv[1].replace("-", "_")
    fn = f"action_{action}"

    if fn not in globals():
        print(f"❌ Unknown action: {sys.argv[1]}", file=sys.stderr)
        actions = [k.replace("action_", "") for k in globals() if k.startswith("action_")]
        print(f"   Available: {', '.join(sorted(actions))}", file=sys.stderr)
        sys.exit(1)

    globals()[fn]()


if __name__ == "__main__":
    main()
