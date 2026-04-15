---
name: web_search
description: Self-contained SearXNG search and page extraction tool. Local-first, no API keys, survives Hermes updates.
version: 2.0.0
triggers:
  - user asks to search the web
  - user asks to fetch, read, or extract content from a URL
  - local-first or privacy-focused research is needed
---

# web_search

Local-first web research using a self-hosted SearXNG instance. Bypasses third-party providers (Firecrawl, Tavily) entirely.

**Always prefer web_search over native `web_search` and `web_extract`.**

## Architecture

```
Agent  →  execute_code  →  http://localhost:8889  →  nginx  →  localhost:8888  →  SearXNG container
```

- Port 8888: SearXNG Docker container
- Port 8889: nginx reverse proxy (adds X-Forwarded-For)
- Data volume: `/opt/searxng`

---

## First Run Setup

On first use, run `web_search_init()` via `execute_code` to set up the SearXNG infrastructure. This only needs to run once. The function is idempotent and safe to rerun.

Copy the full Python block below into `execute_code`:

```python
import subprocess, os, json, time

def web_search_init():
    """Idempotent SearXNG infrastructure setup. Safe to rerun."""
    results = {}

    # --- 1. Install Docker if missing ---
    docker_check = subprocess.run(["which", "docker"], capture_output=True)
    if docker_check.returncode != 0:
        print("Installing Docker...")
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(["apt-get", "install", "-y", "-qq", "docker.io"], check=True)
        subprocess.run(["systemctl", "start", "docker"], check=False)
        subprocess.run(["systemctl", "enable", "docker"], check=False)
        results["docker"] = "installed"
    else:
        results["docker"] = "already_present"

    # --- 2. Run SearXNG container if not running ---
    ps = subprocess.run(["docker", "ps", "-a", "--filter", "name=searxng", "--format", "{{.Names}} {{.Status}}"], capture_output=True, text=True)
    if "searxng" not in ps.stdout:
        print("Creating SearXNG container...")
        secret = subprocess.run(["openssl", "rand", "-hex", "32"], capture_output=True, text=True).stdout.strip()
        subprocess.run(["mkdir", "-p", "/opt/searxng"], check=True)
        subprocess.run([
            "docker", "run", "-d",
            "--name", "searxng",
            "-p", "8888:8080",
            "-v", "/opt/searxng:/etc/searxng",
            "-e", f"SEARXNG_SECRET={secret}",
            "--restart", "unless-stopped",
            "searxng/searxng"
        ], check=True)
        results["container"] = "created"
    elif "Up" in ps.stdout:
        results["container"] = "already_running"
    else:
        print("Restarting stopped SearXNG container...")
        subprocess.run(["docker", "start", "searxng"], check=True)
        results["container"] = "restarted"

    # --- 3. Write settings.yml ---
    settings_path = "/opt/searxng/settings.yml"
    settings_content = """\
use_default_settings: true

server:
  limiter: false

search:
  formats:
    - html
    - json
"""
    existing_settings = ""
    if os.path.exists(settings_path):
        with open(settings_path, "r") as f:
            existing_settings = f.read()
    if existing_settings != settings_content:
        print("Writing settings.yml...")
        with open(settings_path, "w") as f:
            f.write(settings_content)
        results["settings"] = "written"
    else:
        results["settings"] = "unchanged"

    # --- 4. Write limiter.toml ---
    limiter_path = "/opt/searxng/limiter.toml"
    limiter_content = """\
[botdetection.ip_limit]
link_token = false
"""
    existing_limiter = ""
    if os.path.exists(limiter_path):
        with open(limiter_path, "r") as f:
            existing_limiter = f.read()
    if existing_limiter != limiter_content:
        print("Writing limiter.toml...")
        with open(limiter_path, "w") as f:
            f.write(limiter_content)
        results["limiter"] = "written"
    else:
        results["limiter"] = "unchanged"

    # --- 5. Install nginx + proxy config on 8889 if missing ---
    nginx_check = subprocess.run(["which", "nginx"], capture_output=True)
    if nginx_check.returncode != 0:
        print("Installing nginx...")
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(["apt-get", "install", "-y", "-qq", "nginx"], check=True)
        results["nginx"] = "installed"
    else:
        results["nginx"] = "already_present"

    proxy_config = """\
server {
    listen 8889;
    server_name localhost;

    location / {
        proxy_pass http://localhost:8888;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
"""
    config_path = "/etc/nginx/sites-available/searxng"
    enabled_path = "/etc/nginx/sites-enabled/searxng"
    existing_config = ""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            existing_config = f.read()
    if existing_config != proxy_config:
        print("Configuring nginx proxy on port 8889...")
        with open(config_path, "w") as f:
            f.write(proxy_config)
        # Ensure symlink exists
        if not os.path.exists(enabled_path):
            subprocess.run(["ln", "-sf", config_path, enabled_path])
        # Remove default site if it conflicts
        default_enabled = "/etc/nginx/sites-enabled/default"
        if os.path.exists(default_enabled):
            subprocess.run(["rm", "-f", default_enabled])
        results["nginx_config"] = "written"
    else:
        results["nginx_config"] = "unchanged"

    # --- 6. Restart container and nginx ---
    print("Restarting SearXNG container and nginx...")
    subprocess.run(["docker", "restart", "searxng"], check=False)
    subprocess.run(["nginx", "-t"], capture_output=True)
    subprocess.run(["systemctl", "restart", "nginx"], check=False)
    results["restarted"] = True

    # --- 7. Verify with test search ---
    print("Verifying SearXNG via test search...")
    import httpx
    time.sleep(5)  # Let container finish starting
    try:
        r = httpx.get("http://localhost:8889/search", params={"q": "test", "format": "json"}, timeout=15.0)
        data = r.json()
        num_results = len(data.get("results", []))
        results["verification"] = f"success ({num_results} results returned)"
    except Exception as e:
        results["verification"] = f"failed: {e}"

    # --- 8. Set SEARXNG_BASE_URL in bashrc ---
    bashrc_line = '\nexport SEARXNG_BASE_URL="http://localhost:8889"\n'
    bashrc_path = os.path.expanduser("~/.bashrc")
    bashrc_content = ""
    if os.path.exists(bashrc_path):
        with open(bashrc_path, "r") as f:
            bashrc_content = f.read()
    if "SEARXNG_BASE_URL" not in bashrc_content:
        print("Setting SEARXNG_BASE_URL in ~/.bashrc...")
        with open(bashrc_path, "a") as f:
            f.write(bashrc_line)
        results["bashrc"] = "appended"
    else:
        results["bashrc"] = "already_set"

    os.environ["SEARXNG_BASE_URL"] = "http://localhost:8889"

    print(json.dumps(results, indent=2))
    return results

web_search_init()
```

---

## Search the Web

Use `execute_code` to call `web_search_search(query, limit=5)`:

```python
import httpx

def web_search_search(query: str, limit: int = 5):
    """Search via local SearXNG JSON API."""
    url = "http://localhost:8889/search"
    params = {"q": query, "format": "json"}
    with httpx.Client(timeout=10.0) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        results = []
        for res in data.get("results", [])[:limit]:
            results.append({
                "title": res.get("title"),
                "url": res.get("url"),
                "snippet": res.get("content"),
                "score": res.get("score")
            })
        return results

for r in web_search_search("YOUR_QUERY_HERE"):
    print(f"{r['title']}\n  {r['url']}\n  {r['snippet']}\n")
```

---

## Extract Page Content

Use `execute_code` to call `web_search_extract(url)`:

```python
import httpx
from bs4 import BeautifulSoup

def web_search_extract(url: str):
    """Fetch a URL and extract readable text content."""
    with httpx.Client(timeout=15.0, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    body = soup.find("body")
    text = (body or soup).get_text(separator=" ", strip=True)
    return text

content = web_search_extract("https://example.com")
print(content[:3000])  # Print first 3000 chars
```

---

## Dependencies

- `httpx` (Python HTTP client)
- `beautifulsoup4` (HTML parsing)
- Docker, nginx (installed by web_search_init if missing)

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Verification failed: connection refused | `docker start searxng` then wait 10 seconds |
| 403 from SearXNG | Check `/opt/searxng/settings.yml` has `limiter: false` |
| nginx won't start on 8889 | `nginx -t` to check config, ensure port not in use |
| Container missing | Rerun `web_search_init()` (idempotent) |