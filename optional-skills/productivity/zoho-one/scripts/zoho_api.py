#!/usr/bin/env python3
"""Small stdlib-only client for the Zoho REST APIs (CRM, Mail, WorkDrive).

Reads its OAuth client from environment variables:
  ZOHO_CLIENT_ID, ZOHO_CLIENT_SECRET, ZOHO_REFRESH_TOKEN  (required)
  ZOHO_DC            data center: com (default), eu, in, com.au, jp, ca, sa, com.cn
  ZOHO_ACCOUNTS_URL  optional override, e.g. https://accounts.zoho.eu
  ZOHO_MAIL_BASE     optional override, e.g. https://mail.zoho.eu
  ZOHO_WORKDRIVE_BASE optional override, e.g. https://workdrive.zoho.eu

The access token is cached in $HERMES_HOME/cache/zoho-one/access.json (mode 600)
and reused until five minutes before it expires. Zoho throttles the token
endpoint after roughly 20 refreshes in a row, so never refresh per request.
The script never prints the access token or the client secret.

Commands (run with --help for options):
  auth-url                    one-time setup: print the consent URL
  exchange-code CODE          one-time setup: print the refresh token (run it yourself)
  check                       verify credentials, print data center and API hosts
  request METHOD PRODUCT PATH generic call; PRODUCT is crm, mail or workdrive
  mail-accounts               list mail accounts with accountId and send-as aliases
  mail-folders                list folders of a mail account
  mail-send                   send a new mail or save it as a draft (dry run by default)
  mail-reply                  reply in the thread of an existing mail (dry run by default)
  wd-list FOLDER_ID           list a WorkDrive folder
  wd-mkdir PARENT_ID NAME     create a WorkDrive folder
  wd-upload PARENT_ID FILE..  upload files, then download each one and compare SHA-256
"""
import argparse
import hashlib
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

# Zoho data centers. CRM uses the api_domain returned with the access token,
# so only the accounts, Mail and WorkDrive hosts are derived here.
DC_HOSTS = {
    "com": ("https://accounts.zoho.com", "https://mail.zoho.com", "https://workdrive.zoho.com"),
    "eu": ("https://accounts.zoho.eu", "https://mail.zoho.eu", "https://workdrive.zoho.eu"),
    "in": ("https://accounts.zoho.in", "https://mail.zoho.in", "https://workdrive.zoho.in"),
    "com.au": ("https://accounts.zoho.com.au", "https://mail.zoho.com.au", "https://workdrive.zoho.com.au"),
    "jp": ("https://accounts.zoho.jp", "https://mail.zoho.jp", "https://workdrive.zoho.jp"),
    "ca": ("https://accounts.zohocloud.ca", "https://mail.zohocloud.ca", "https://workdrive.zohocloud.ca"),
    "sa": ("https://accounts.zoho.sa", "https://mail.zoho.sa", "https://workdrive.zoho.sa"),
    "com.cn": ("https://accounts.zoho.com.cn", "https://mail.zoho.com.cn", "https://workdrive.zoho.com.cn"),
}
CRM_VERSION = "v8"
JSONAPI = "application/vnd.api+json"
TIMEOUT = 60


class ZohoError(Exception):
    pass


def _env(name, default=""):
    return (os.environ.get(name) or default).strip()


def hosts():
    dc = _env("ZOHO_DC", "com").lower().lstrip(".")
    if dc not in DC_HOSTS:
        raise ZohoError(f"Unknown ZOHO_DC '{dc}'. Use one of: {', '.join(DC_HOSTS)}")
    acc, mail, wd = DC_HOSTS[dc]
    return {
        "dc": dc,
        "accounts": _env("ZOHO_ACCOUNTS_URL", acc).rstrip("/"),
        "mail": _env("ZOHO_MAIL_BASE", mail).rstrip("/"),
        "workdrive": _env("ZOHO_WORKDRIVE_BASE", wd).rstrip("/"),
    }


def _cache_file():
    home = _env("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(home) / "cache" / "zoho-one" / "access.json"


def _client_key():
    raw = _env("ZOHO_CLIENT_ID") + "|" + hosts()["accounts"]
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _read_cache():
    try:
        data = json.loads(_cache_file().read_text())
    except (OSError, ValueError):
        return None
    if data.get("client") != _client_key() or data.get("expires_at", 0) < time.time() + 300:
        return None
    return data


def _write_cache(data):
    path = _cache_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}")
    with open(tmp, "w") as fh:
        json.dump(data, fh)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def drop_cache():
    try:
        _cache_file().unlink()
    except OSError:
        pass


def access(force=False):
    """Return {'value': access token, 'api_domain': ...}, cached between runs."""
    if not force:
        cached = _read_cache()
        if cached:
            return cached
    missing = [n for n in ("ZOHO_CLIENT_ID", "ZOHO_CLIENT_SECRET", "ZOHO_REFRESH_TOKEN") if not _env(n)]
    if missing:
        raise ZohoError("Missing environment variables: " + ", ".join(missing))
    form = urllib.parse.urlencode({
        "grant_type": "refresh_token",
        "client_id": _env("ZOHO_CLIENT_ID"),
        "client_secret": _env("ZOHO_CLIENT_SECRET"),
        "refresh_token": _env("ZOHO_REFRESH_TOKEN"),
    }).encode()
    req = urllib.request.Request(hosts()["accounts"] + "/oauth/v2/token", data=form, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            body = json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", "replace")
        if "too many requests" in text.lower():
            raise ZohoError("Zoho throttles token refreshes right now. Wait 60 seconds and retry once.")
        raise ZohoError(f"Token refresh failed: HTTP {exc.code} {text[:300]}")
    if "access_token" not in body:
        hint = ""
        if body.get("error") == "invalid_client":
            hint = " Check ZOHO_DC: the client must belong to this data center."
        raise ZohoError(f"Token refresh failed: {body.get('error', body)}.{hint}")
    data = {
        "value": body["access_token"],
        "api_domain": body.get("api_domain", "https://www.zohoapis.com"),
        "expires_at": time.time() + int(body.get("expires_in", 3600)),
        "client": _client_key(),
    }
    _write_cache(data)
    return data


def base_url(product, tok):
    h = hosts()
    if product == "crm":
        return f"{tok['api_domain'].rstrip('/')}/crm/{CRM_VERSION}"
    if product == "mail":
        return f"{h['mail']}/api"
    if product == "workdrive":
        return f"{h['workdrive']}/api/v1"
    raise ZohoError("PRODUCT must be crm, mail or workdrive")


def call(method, product, path, body=None, query=None, raw=False, content_type=None, retry=True):
    """One API call. Returns (status, parsed JSON or bytes). Retries once on an invalid token."""
    tok = access()
    url = base_url(product, tok) + "/" + path.lstrip("/")
    if query:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(query)
    headers = {"Authorization": "Zoho-oauthtoken " + tok["value"]}
    if product == "workdrive":
        headers["Accept"] = JSONAPI
    data = None
    if body is not None:
        if isinstance(body, (bytes, bytearray)):
            data = bytes(body)
        else:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = JSONAPI if product == "workdrive" else "application/json"
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=data, method=method.upper(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT if not raw else 600) as resp:
            status, payload = resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        status, payload = exc.code, exc.read()
    text = payload.decode("utf-8", "replace") if not raw or status >= 400 else ""
    if retry and status == 401 and ("INVALID_OAUTHTOKEN" in text or "Invalid OAuth" in text or not text):
        drop_cache()
        access(force=True)
        return call(method, product, path, body, query, raw, content_type, retry=False)
    if raw and status < 400:
        return status, payload
    try:
        return status, json.loads(text) if text.strip() else {}
    except ValueError:
        return status, {"raw": text[:2000]}


def out(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def _content(args):
    if args.html_file:
        return Path(args.html_file).read_text(encoding="utf-8"), "html"
    if args.text is not None:
        return args.text, "plaintext"
    raise ZohoError("Give the body with --html-file or --text")


# ---------- commands ----------

def cmd_check(args):
    h = hosts()
    tok = access(force=args.refresh)
    out({
        "ok": True,
        "data_center": h["dc"],
        "accounts": h["accounts"],
        "crm": base_url("crm", tok),
        "mail": base_url("mail", tok),
        "workdrive": base_url("workdrive", tok),
        "token_valid_for_min": int((tok["expires_at"] - time.time()) / 60),
    })


def cmd_request(args):
    body = None
    if args.data_file:
        body = json.loads(Path(args.data_file).read_text(encoding="utf-8"))
    elif args.data:
        body = json.loads(args.data)
    query = dict(q.split("=", 1) for q in args.query or [])
    status, payload = call(args.method, args.product, args.path, body, query)
    out({"status": status, "body": payload})
    if status >= 400:
        sys.exit(1)


def cmd_mail_accounts(args):
    status, payload = call("GET", "mail", "accounts")
    rows = []
    for a in payload.get("data", []) if isinstance(payload, dict) else []:
        rows.append({
            "accountId": a.get("accountId"),
            "primary": a.get("primaryEmailAddress") or a.get("mailboxAddress") or a.get("incomingUserName"),
            "displayName": a.get("displayName") or a.get("accountDisplayName"),
            "type": a.get("type"),
            "default": a.get("isDefaultAccount"),
            "send_as": [
                {"address": s.get("fromAddress"), "displayName": s.get("displayName")}
                for s in a.get("sendMailDetails", [])
            ],
        })
    out({"status": status, "accounts": rows} if status < 400 else {"status": status, "body": payload})


def cmd_mail_folders(args):
    status, payload = call("GET", "mail", f"accounts/{args.account}/folders")
    rows = [{"folderId": f.get("folderId"), "folderName": f.get("folderName"), "path": f.get("path")}
            for f in payload.get("data", [])] if status < 400 else payload
    out({"status": status, "folders": rows})


def _mail_payload(args):
    content, fmt = _content(args)
    body = {"fromAddress": args.sender, "toAddress": args.to, "subject": args.subject,
            "content": content, "mailFormat": fmt}
    if args.cc:
        body["ccAddress"] = args.cc
    if args.bcc:
        body["bccAddress"] = args.bcc
    return body


def cmd_mail_send(args):
    body = _mail_payload(args)
    if args.draft:
        body["mode"] = "draft"
    if not args.send and not args.draft:
        out({"dry_run": True, "would_post": f"accounts/{args.account}/messages",
             "payload": {**body, "content": body["content"][:300]}})
        return
    status, payload = call("POST", "mail", f"accounts/{args.account}/messages", body)
    out({"status": status, "body": payload})
    if status >= 400:
        sys.exit(1)


def cmd_mail_reply(args):
    body = _mail_payload(args)
    body["action"] = "reply"
    path = f"accounts/{args.account}/messages/{args.message}"
    if not args.send:
        out({"dry_run": True, "would_post": path, "payload": {**body, "content": body["content"][:300]}})
        return
    status, payload = call("POST", "mail", path, body)
    out({"status": status, "body": payload})
    if status >= 400:
        sys.exit(1)


def cmd_wd_list(args):
    status, payload = call("GET", "workdrive", f"files/{args.folder}/files", query={"page[limit]": args.limit})
    if status >= 400:
        out({"status": status, "body": payload})
        sys.exit(1)
    rows = [{"id": f.get("id"), "name": f.get("attributes", {}).get("name"),
             "type": f.get("attributes", {}).get("type"),
             "size": f.get("attributes", {}).get("storage_info", {}).get("size_in_bytes")}
            for f in payload.get("data", [])]
    out({"status": status, "files": rows})


def cmd_wd_mkdir(args):
    body = {"data": {"attributes": {"name": args.name, "parent_id": args.parent}, "type": "files"}}
    status, payload = call("POST", "workdrive", "files", body)
    if status >= 400:
        out({"status": status, "body": payload})
        sys.exit(1)
    out({"status": status, "id": payload.get("data", {}).get("id"), "name": args.name})


def _upload(parent, path):
    name = path.name
    mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
    boundary = "----zohoone" + uuid.uuid4().hex
    parts = []
    # override-name-exist=false: "true" does not replace, it renames the new file
    # to "<stem>.Final.<ext>", which breaks it for the recipient.
    for key, val in (("parent_id", parent), ("override-name-exist", "false")):
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{val}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="content"; filename="{name}"\r\n'
                 f'Content-Type: {mime}\r\n\r\n'.encode())
    parts.append(path.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return call("POST", "workdrive", "upload", b"".join(parts),
                content_type=f"multipart/form-data; boundary={boundary}")


def cmd_wd_upload(args):
    results, failed = [], False
    for f in args.files:
        path = Path(f)
        status, payload = _upload(args.parent, path)
        if status >= 400 or not payload.get("data"):
            results.append({"file": path.name, "ok": False, "status": status, "body": payload})
            failed = True
            continue
        attrs = payload["data"][0].get("attributes", {})
        fid = attrs.get("resource_id")
        stored = attrs.get("FileName") or attrs.get("name") or path.name
        dstatus, blob = call("GET", "workdrive", f"download/{fid}", raw=True)
        local = hashlib.sha256(path.read_bytes()).hexdigest()
        remote = hashlib.sha256(blob).hexdigest() if dstatus < 400 and isinstance(blob, bytes) else None
        ok = remote == local and stored == path.name
        failed |= not ok
        results.append({"file": path.name, "stored_as": stored, "id": fid, "sha256": local,
                        "sha256_match": remote == local, "ok": ok})
        time.sleep(0.3)
    out({"results": results})
    if failed:
        sys.exit(1)


DEFAULT_SCOPES = ",".join([
    "ZohoCRM.modules.ALL", "ZohoCRM.settings.READ", "ZohoCRM.users.READ", "ZohoCRM.coql.READ",
    "ZohoMail.accounts.READ", "ZohoMail.folders.READ", "ZohoMail.messages.ALL",
    "ZohoMail.messages.CREATE", "ZohoMail.messages.READ", "ZohoMail.messages.UPDATE",
    "WorkDrive.files.ALL", "WorkDrive.teamfolders.ALL", "WorkDrive.workspace.ALL", "WorkDrive.users.READ",
])


def cmd_auth_url(args):
    if not _env("ZOHO_CLIENT_ID"):
        raise ZohoError("Set ZOHO_CLIENT_ID and ZOHO_DC first")
    query = urllib.parse.urlencode({
        "scope": args.scopes, "client_id": _env("ZOHO_CLIENT_ID"), "response_type": "code",
        "access_type": "offline", "prompt": "consent", "redirect_uri": args.redirect_uri,
    })
    out({"open_in_browser": f"{hosts()['accounts']}/oauth/v2/auth?{query}",
         "next": "Approve, then copy the code parameter from the address bar and run exchange-code "
                 "within two minutes."})


def cmd_exchange_code(args):
    missing = [n for n in ("ZOHO_CLIENT_ID", "ZOHO_CLIENT_SECRET") if not _env(n)]
    if missing:
        raise ZohoError("Missing environment variables: " + ", ".join(missing))
    form = urllib.parse.urlencode({
        "grant_type": "authorization_code", "client_id": _env("ZOHO_CLIENT_ID"),
        "client_secret": _env("ZOHO_CLIENT_SECRET"), "redirect_uri": args.redirect_uri, "code": args.code,
    }).encode()
    req = urllib.request.Request(hosts()["accounts"] + "/oauth/v2/token", data=form, method="POST")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        body = json.loads(resp.read() or b"{}")
    if "refresh_token" not in body:
        raise ZohoError(f"No refresh token returned: {body.get('error', body)}. Codes expire after "
                        "about two minutes and work once; start again with auth-url.")
    # Printed once on purpose: the user stores it as ZOHO_REFRESH_TOKEN in the .env file of your Hermes home.
    out({"ZOHO_REFRESH_TOKEN": body["refresh_token"], "api_domain": body.get("api_domain"),
         "scope": body.get("scope"),
         "next": "Store the value as ZOHO_REFRESH_TOKEN in the .env file of your Hermes home, then run check."})


def main():
    p = argparse.ArgumentParser(description="Zoho REST helper (CRM, Mail, WorkDrive)")
    sub = p.add_subparsers(dest="cmd", required=True)

    au = sub.add_parser("auth-url", help="print the consent URL (one-time setup)")
    au.add_argument("--scopes", default=DEFAULT_SCOPES)
    au.add_argument("--redirect-uri", default="http://localhost:8080/callback")
    au.set_defaults(fn=cmd_auth_url)
    ex = sub.add_parser("exchange-code", help="turn the consent code into a refresh token (one-time setup)")
    ex.add_argument("code")
    ex.add_argument("--redirect-uri", default="http://localhost:8080/callback")
    ex.set_defaults(fn=cmd_exchange_code)

    c = sub.add_parser("check", help="verify credentials and print API hosts")
    c.add_argument("--refresh", action="store_true", help="ignore the cache and refresh now")
    c.set_defaults(fn=cmd_check)

    r = sub.add_parser("request", help="generic API call")
    r.add_argument("method", choices=["GET", "POST", "PUT", "PATCH", "DELETE"])
    r.add_argument("product", choices=["crm", "mail", "workdrive"])
    r.add_argument("path", help="path after the product base, e.g. Leads/search")
    r.add_argument("--query", action="append", help="query parameter key=value (repeatable)")
    r.add_argument("--data", help="JSON body as a string")
    r.add_argument("--data-file", help="JSON body from a file")
    r.set_defaults(fn=cmd_request)

    sub.add_parser("mail-accounts", help="list mail accounts").set_defaults(fn=cmd_mail_accounts)
    mf = sub.add_parser("mail-folders", help="list folders")
    mf.add_argument("--account", required=True)
    mf.set_defaults(fn=cmd_mail_folders)

    for name, fn, helptext in (("mail-send", cmd_mail_send, "send a new mail or save a draft"),
                               ("mail-reply", cmd_mail_reply, "reply in the thread of a mail")):
        m = sub.add_parser(name, help=helptext)
        m.add_argument("--account", required=True, help="accountId from mail-accounts")
        if name == "mail-reply":
            m.add_argument("--message", required=True, help="messageId of the mail you answer")
        m.add_argument("--from", dest="sender", required=True, help="a send-as address of the account")
        m.add_argument("--to", required=True)
        m.add_argument("--cc")
        m.add_argument("--bcc")
        m.add_argument("--subject", required=True)
        m.add_argument("--html-file")
        m.add_argument("--text")
        m.add_argument("--send", action="store_true", help="really send (default is a dry run)")
        if name == "mail-send":
            m.add_argument("--draft", action="store_true", help="save in Drafts instead of sending")
        m.set_defaults(fn=fn)

    wl = sub.add_parser("wd-list", help="list a WorkDrive folder")
    wl.add_argument("folder")
    wl.add_argument("--limit", default="50")
    wl.set_defaults(fn=cmd_wd_list)
    wm = sub.add_parser("wd-mkdir", help="create a WorkDrive folder")
    wm.add_argument("parent")
    wm.add_argument("name")
    wm.set_defaults(fn=cmd_wd_mkdir)
    wu = sub.add_parser("wd-upload", help="upload and verify by SHA-256")
    wu.add_argument("parent")
    wu.add_argument("files", nargs="+")
    wu.set_defaults(fn=cmd_wd_upload)

    args = p.parse_args()
    try:
        args.fn(args)
    except ZohoError as exc:
        out({"ok": False, "error": str(exc)})
        sys.exit(2)
    except urllib.error.URLError as exc:
        out({"ok": False, "error": f"Network error: {exc.reason}"})
        sys.exit(2)


if __name__ == "__main__":
    main()
