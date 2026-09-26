"""Install-wide configuration with easiest-first defaults.

Everything works zero-config. `setup --auto` (the autonomous path) detects what
this environment can do and picks the MOST AUTONOMOUS valid configuration without
asking anyone; plain `setup` keeps the easiest-first defaults and only upgrades a
setting when a flag opts in.

`autonomy` is policy, orthogonal to capability:
  full     - intake consent is standing authorization; the agent submits T0-T2
             opt-outs without pausing per submission (default).
  assisted - the agent pauses for operator confirmation before each submission.
"""
from __future__ import annotations

import os
from pathlib import Path
from shutil import which

import emailer
import paths
import storage

DEFAULT_CONFIG = {
    "autonomy": "full",                # hands-off after intake+consent
    "email_mode": "draft_only",        # zero credentials
    "browser_backend": "auto",         # auto = Browserbase when BROWSERBASE_API_KEY is set
                                       # (recommended default; clears soft CAPTCHAs), else plain browser
    "tracker_backend": "local-json",   # no external dependency
    "encryption": "none",              # files still written 0600
    "default_rescan_interval_days": 120,
    "email_min_interval_seconds": 20,  # pace SMTP sends so a run can't torch the account
}

VALID = {
    "autonomy": {"full", "assisted"},
    # email_mode:
    #   draft_only   - render drafts; the operator sends + clicks verify links (zero setup)
    #   browser      - the agent sends + opens verify links through the operator's logged-in
    #                  webmail via browser_* tools (NO password stored; needs a browser the
    #                  operator's inbox is signed into)
    #   programmatic - CLI sends via SMTP + reads verify links via IMAP (needs EMAIL_* creds)
    #   alias        - AgentMail agent-owned inboxes / per-broker aliases
    "email_mode": {"draft_only", "browser", "programmatic", "alias"},
    "browser_backend": {"auto", "browserbase", "agent-browser", "camofox"},
    "tracker_backend": {"local-json", "google-sheets"},
    "encryption": {"none", "age"},
}


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(storage.read_json(paths.config_path(), {}) or {})
    return cfg


def save_config(cfg: dict) -> Path:
    merged = dict(DEFAULT_CONFIG)
    merged.update(cfg)
    for key, allowed in VALID.items():
        if merged.get(key) not in allowed:
            raise ValueError(f"invalid {key!r}: {merged.get(key)!r} (allowed: {sorted(allowed)})")
    return storage.write_json(paths.config_path(), merged)


def dotenv_env() -> dict:
    """Shell env overlaid on `$HERMES_HOME/.env`, so capability detection sees the creds Hermes
    loads for its own tools (BROWSERBASE_API_KEY, EMAIL_*, AGENTMAIL_API_KEY, ...) even though the
    terminal-tool shell doesn't export them. Shell env wins; the .env only fills gaps."""
    merged: dict = {}
    p = paths.hermes_home() / ".env"
    if p.exists():
        try:
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                merged[k.strip()] = v.strip().strip('"').strip("'")
        except OSError:
            pass
    merged.update(os.environ)
    return merged


# Env vars that indicate a configured stealth/cloud browser backend (Hermes browser feature). Any of
# these clears soft/managed CAPTCHAs, so any one shifts a captcha-gated broker T2 -> T1.
_CLOUD_BROWSER_ENV = ("BROWSERBASE_API_KEY", "BROWSER_USE_API_KEY", "FIRECRAWL_API_KEY", "CAMOFOX_URL")


def _has_cloud_browser(env: dict) -> bool:
    return any(env.get(k) for k in _CLOUD_BROWSER_ENV)


def hermes_browser_configured() -> bool:
    """Best-effort: does Hermes ALREADY have a stealth browser wired up (no skill env keys needed)?

    Two no-key paths the env alone can't see (see the Hermes browser docs):
      - a provider selected in `$HERMES_HOME/config.yaml` (`browser.cloud_provider`, e.g. browserbase,
        browser-use, firecrawl, or the Nous Subscription gateway);
      - a Nous Portal subscriber token in `$HERMES_HOME/auth.json` (the Tool Gateway then provides the
        browser -- Browserbase is a built-in gateway tool, so Portal auth works instead of an API key).
    Defensive: any read error just returns the env-only picture.
    """
    home = paths.hermes_home()
    try:
        cfg_text = (home / "config.yaml").read_text(encoding="utf-8", errors="replace")
        for line in cfg_text.splitlines():
            s = line.strip()
            if s.startswith("cloud_provider:"):
                val = s.split(":", 1)[1].strip().strip('"').strip("'")
                if val and not val.startswith("#"):
                    return True
    except OSError:
        pass
    try:
        auth_text = (home / "auth.json").read_text(encoding="utf-8", errors="replace").lower()
        if "nous" in auth_text and "access_token" in auth_text:
            return True
    except OSError:
        pass
    return False


def detect_capabilities(env: dict | None = None) -> dict:
    """Report which opt-in upgrades are available without extra setup."""
    env = os.environ if env is None else env
    home = paths.hermes_home()
    google = (
        (home / "google_token.json").exists()
        or (home / "skills" / "productivity" / "google-workspace").exists()
        or (home / "skills" / "google-workspace").exists()
    )
    mail = emailer.available(env)
    return {
        # Browserbase needs BOTH the key and a project id to actually run (per the Hermes browser docs).
        "browserbase": bool(env.get("BROWSERBASE_API_KEY")),
        "browserbase_ready": bool(env.get("BROWSERBASE_API_KEY") and env.get("BROWSERBASE_PROJECT_ID")),
        # Other Hermes browser backends that also provide managed stealth (clear soft CAPTCHAs):
        "browser_use": bool(env.get("BROWSER_USE_API_KEY")),
        "firecrawl": bool(env.get("FIRECRAWL_API_KEY")),
        "camofox": bool(env.get("CAMOFOX_URL")),
        # Any configured stealth/cloud browser (Browserbase | Browser Use | Firecrawl | Camofox). Nous
        # Portal subscribers also get the browser via the Tool Gateway with NO keys (not env-detectable).
        "cloud_browser": _has_cloud_browser(env),
        "agentmail": bool(env.get("AGENTMAIL_API_KEY")),
        "email_imap_smtp": bool(env.get("EMAIL_ADDRESS") and env.get("EMAIL_PASSWORD")),
        "smtp_send": mail["smtp"],      # CLI can SEND opt-out emails itself
        "imap_read": mail["imap"],      # CLI can POLL verification links itself
        "google_workspace": google,
        "age": which("age") is not None,
    }


def auto_configure(env: dict | None = None) -> dict:
    """Pick the most autonomous configuration this environment supports (no questions).

    - email: programmatic when SMTP creds exist (CLI sends + IMAP-verifies itself);
      alias mode when only AgentMail exists; draft_only as the capability floor.
    - browser: browserbase when the key exists (clears soft CAPTCHAs -> more T1).
    - encryption: age when the binary is installed (free privacy, zero human cost).
    - tracker: stays local-json (google-sheets needs a sheet id -> a human choice).
    """
    caps = detect_capabilities(env)
    cfg = load_config()
    cfg["autonomy"] = "full"
    if caps["smtp_send"]:
        cfg["email_mode"] = "programmatic"
    elif caps["agentmail"]:
        cfg["email_mode"] = "alias"
    else:
        cfg["email_mode"] = "draft_only"
    # Prefer an explicit stealth backend when its creds are present; else 'auto' (which still picks up
    # a Browserbase/Browser Use/Firecrawl key, a Camofox URL, or the Nous Portal gateway at runtime).
    cfg["browser_backend"] = ("browserbase" if caps["browserbase"]
                              else "camofox" if caps["camofox"] else "auto")
    if caps["age"]:
        cfg["encryption"] = "age"
    return cfg


def browser_clears_captcha(cfg: dict, env: dict | None = None) -> bool:
    """True if the chosen browser backend can clear soft CAPTCHAs (shifts T2 -> T1).

    Any of Hermes' stealth backends qualifies: Browserbase / Browser Use / Firecrawl (managed cloud
    browsers on residential IPs) or Camofox (local anti-detect Firefox) pass soft/managed challenges
    (Turnstile, hCaptcha/reCAPTCHA checkbox) as normal operation. This is NOT solving/spoofing - hard
    interactive challenges still escalate to a human. `auto` inherits this whenever any such backend is
    configured; Nous Portal subscribers get it via the Tool Gateway (assume true if set up).
    """
    backend = cfg.get("browser_backend", "auto")
    if backend in ("browserbase", "camofox"):
        return True
    if backend == "auto":
        env = os.environ if env is None else env
        # env keys (direct) OR a browser Hermes already has via config/Portal (no skill keys needed).
        return _has_cloud_browser(env) or hermes_browser_configured()
    return False
