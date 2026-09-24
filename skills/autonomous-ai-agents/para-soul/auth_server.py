"""Para-Soul Auth — Google/GitHub/Microsoft OAuth2 + encrypted key management."""
import json, base64, urllib.request, urllib.parse
from datetime import datetime
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption, PublicFormat

router = APIRouter()
USERS = {}
OAUTH_CLIENTS = {}

def _user_key(p, oid): return f"{p}:{oid}"

class RegisterRequest(BaseModel):
    provider: str; oauth_token: str; encrypted_private_key: str; salt: str

class RecoverRequest(BaseModel):
    provider: str; oauth_token: str

def _verify_google(token):
    req = urllib.request.Request("https://www.googleapis.com/oauth2/v3/userinfo", headers={"Authorization": f"Bearer {token}"})
    d = json.loads(urllib.request.urlopen(req, timeout=10).read().decode())
    return d["sub"], d.get("email", "")

def _verify_github(token):
    req = urllib.request.Request("https://api.github.com/user", headers={"Authorization": f"Bearer {token}", "User-Agent": "ParaSoul"})
    d = json.loads(urllib.request.urlopen(req, timeout=10).read().decode())
    return str(d["id"]), d.get("email", "")

def _verify_microsoft(token):
    req = urllib.request.Request("https://graph.microsoft.com/v1.0/me", headers={"Authorization": f"Bearer {token}"})
    d = json.loads(urllib.request.urlopen(req, timeout=10).read().decode())
    return d["id"], d.get("mail") or d.get("userPrincipalName", "")

OAUTH_PROVIDERS = {"google": _verify_google, "github": _verify_github, "microsoft": _verify_microsoft}

def _verify(p, t):
    if p not in OAUTH_PROVIDERS: raise HTTPException(400, "Unknown provider")
    return OAUTH_PROVIDERS[p](t)

def _gen_keypair():
    sk = Ed25519PrivateKey.generate()
    pk = sk.public_key()
    sp = sk.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()).decode()
    pb = pk.public_bytes(Encoding.Raw, PublicFormat.Raw)
    return "did:key:z" + base64.b64encode(pb).decode().rstrip("="), sp

@router.post("/register")
def register(req: RegisterRequest):
    oid, email = _verify(req.provider, req.oauth_token)
    k = _user_key(req.provider, oid)
    if k in USERS: raise HTTPException(409, "Not available")
    did, sk = _gen_keypair()
    USERS[k] = {"p": req.provider, "e": email, "did": did, "enc": req.encrypted_private_key, "salt": req.salt, "ts": datetime.utcnow().isoformat()}
    return {"success": True, "did": did, "private_key": sk}

@router.post("/recover")
def recover(req: RecoverRequest):
    oid, email = _verify(req.provider, req.oauth_token)
    k = _user_key(req.provider, oid)
    if k not in USERS: raise HTTPException(404, "Not available")
    u = USERS[k]
    return {"success": True, "did": u["did"], "encrypted_private_key": u["enc"], "pbkdf2_salt": u.get("salt",""), "email": u["e"]}

@router.get("/me")
def me(provider: str, oauth_token: str):
    oid, email = _verify(provider, oauth_token)
    k = _user_key(provider, oid)
    if k not in USERS: raise HTTPException(404, "Not found")
    u = USERS[k]
    return {"success": True, "did": u["did"], "email": u["e"], "provider": u["p"], "has_encrypted_key": bool(u.get("enc"))}

@router.get("/{provider}/login")
async def oauth_login(provider: str):
    if provider not in OAUTH_CLIENTS: raise HTTPException(400, f"Unknown: {provider}")
    cfg = OAUTH_CLIENTS[provider]
    url = cfg["auth_url"].format(client_id=cfg["client_id"], redirect_uri=cfg["redirect_uri"])
    return RedirectResponse(url=url)

@router.get("/callback/{provider}")
async def oauth_callback(provider: str, code: str):
    if provider not in OAUTH_CLIENTS: raise HTTPException(400, f"Unknown: {provider}")
    cfg = OAUTH_CLIENTS[provider]
    data = urllib.parse.urlencode({"client_id":cfg["client_id"],"client_secret":cfg["client_secret"],"code":code,"redirect_uri":cfg["redirect_uri"],"grant_type":"authorization_code"}).encode()
    req = urllib.request.Request(cfg["token_url"], data=data, headers={"Accept":"application/json"})
    token = json.loads(urllib.request.urlopen(req,timeout=10).read().decode())["access_token"]
    oid, email = _verify(provider, token)
    return HTMLResponse(content=f'''<!DOCTYPE html><html><head><script>window.opener.postMessage({{"type":"oauth-callback","token":"{token}","email":"{email}"}},"*");window.close();</script></head><body>Done</body></html>''')
