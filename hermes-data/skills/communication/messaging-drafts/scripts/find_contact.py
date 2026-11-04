#!/usr/bin/env python3
"""
find_contact.py — one-shot contact lookup for WhatsApp/message drafting (Step 1).

Looks up a person across:
  1. Google People API (searchContacts + connections list fallback)
  2. NDR DRAAS Contacts Sheet (enriched: Work/Mobile labels, org, title, notes)
  3. NDR CONTACTS sheet (traditional business contacts)

Prints labelled phone numbers so the agent can pick the PRIMARY number
(prefer Work / Phone 1 label) before building a wa.me link.

Usage:
    /opt/hermes/.venv/bin/python find_contact.py "Bharat Hawaldar"
    /opt/hermes/.venv/bin/python find_contact.py "9845890316"

Automatic vault-socket recovery: if GWS_VAULT_SOCKET points at a stale path
(observed: /opt/data/gws-vault/run/vault.sock dead, /run/gws-vault/vault.sock live),
this script re-points GWS_VAULT_SOCKET / GWS_VAULT_TOKEN_DIR at the live socket
before building any service. Do NOT skip this — People API + Sheets are the
backbone of contact lookup.
"""
import os, sys, json, subprocess

sys.path.insert(0, "/opt/hermes")

QUERY = " ".join(sys.argv[1:]).strip()
if not QUERY:
    QUERY = input("Contact query: ").strip()
if not QUERY:
    print("No query given.")
    sys.exit(1)


def live_socket():
    candidates = [
        os.environ.get("GWS_VAULT_SOCKET"),
        "/run/gws-vault/vault.sock",
        "/opt/data/gws-vault/run/vault.sock",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    try:
        out = subprocess.run(
            ["find", "/", "-name", "vault.sock", "-maxdepth", "6"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        if out:
            return out.splitlines()[0]
    except Exception:
        pass
    return None


sock = live_socket()
if sock:
    os.environ["GWS_VAULT_SOCKET"] = sock
    token_dir = os.path.join(os.path.dirname(os.path.dirname(sock)), "tokens")
    os.environ["GWS_VAULT_TOKEN_DIR"] = token_dir if os.path.isdir(token_dir) else "/run/gws-vault/tokens"

from tools import gws_auth  # noqa: E402

# --- 1. People API -----------------------------------------------------------
try:
    people = gws_auth.build_service("people", "v1", service_name="google-draas")
    try:
        res = people.people().searchContacts(
            query=QUERY,
            readMask="names,phoneNumbers,emailAddresses,organizations",
            pageSize=10,
        ).execute()
        print("=== People API matches ===")
        for p in res.get("results", []):
            person = p.get("person", {})
            print(json.dumps({
                "names": [n.get("displayName") for n in person.get("names", [])],
                "phones": [ph.get("value") for ph in person.get("phoneNumbers", [])],
                "emails": [e.get("value") for e in person.get("emailAddresses", [])],
                "orgs": [o.get("name") for o in person.get("organizations", [])],
            }, ensure_ascii=False))
    except Exception as e:
        print(f"People searchContacts FAIL: {type(e).__name__}: {e}")
    try:
        conn = people.people().connections().list(
            resourceName="people/me",
            personFields="names,phoneNumbers,emailAddresses",
            pageSize=200,
        ).execute()
        for person in conn.get("connections", []):
            names = [n.get("displayName") or "" for n in person.get("names", [])]
            if any(QUERY.lower() in n.lower() for n in names):
                print(json.dumps({
                    "names": names,
                    "phones": [ph.get("value") for ph in person.get("phoneNumbers", [])],
                    "emails": [e.get("value") for e in person.get("emailAddresses", [])],
                }, ensure_ascii=False))
    except Exception as e:
        print(f"People connections FAIL: {type(e).__name__}: {e}")
except Exception as e:
    print(f"People API FAIL: {type(e).__name__}: {e}")

# --- 2/3. Contacts sheets ------------------------------------------------------
try:
    sheets = gws_auth.build_service("sheets", "v4", service_name="google-draas")
    targets = [
        ("1XbSRAXxPLY4cXMTm2rmvKh11Nx3x0aKUxxuWualoV9g", "NDR DRAAS Google contacts.csv"),
        ("1fYa-t2RY1siy2qBgAH8uu_Jd2chjJ716BbcpxilpOK0", "Sheet1"),
    ]
    for sid, rng in targets:
        try:
            resp = sheets.spreadsheets().values().get(spreadsheetId=sid, range=rng).execute()
            rows = resp.get("values", [])
            if not rows:
                continue
            header = rows[0]
            print(f"\n=== Sheet {sid[:6]} ({len(rows) - 1} data rows) ===")
            found = False
            for row in rows[1:]:
                joined = " | ".join(str(c) for c in row)
                if QUERY.lower() not in joined.lower():
                    continue
                found = True
                d = dict(zip(header, row + [""] * (len(header) - len(row))))

                def g(*keys):
                    for k in keys:
                        if d.get(k):
                            return d[k]
                    return ""

                out = {}
                name = f"{g('First Name')} {g('Last Name')}".strip()
                out["name"] = name or g("NAME", "Name") or QUERY
                out["org"] = g("Organization Name", "COMPANY")
                out["title"] = g("Organization Title", "DESIGNATION")
                out["email"] = g("E-mail 1 - Value", "E-MAIL")
                phones = []
                for i in range(1, 7):
                    label = d.get(f"Phone {i} - Label") or ""
                    val = d.get(f"Phone {i} - Value") or ""
                    if val:
                        phones.append(f"{label}: {val}".strip(": "))
                if not phones and d.get("MOBILE"):
                    phones.append(f"MOBILE: {d['MOBILE']}")
                if not phones and d.get("TELEPHONE"):
                    phones.append(f"TELEPHONE: {d['TELEPHONE']}")
                out["phones"] = phones
                notes = (d.get("Notes") or d.get("NOTES") or "")[:200]
                if notes:
                    out["notes"] = notes
                print(json.dumps(out, ensure_ascii=False))
            if not found:
                print("(no matching rows)")
        except Exception as e:
            print(f"Sheet {sid} FAIL: {type(e).__name__}: {e}")
except Exception as e:
    print(f"Sheets API FAIL: {type(e).__name__}: {e}")
