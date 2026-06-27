# Mobile HTML Report Sharing via Cloudflare Tunnel

Use this reference when a user asks to view a generated menu-analysis HTML report on a phone or from outside the local machine.

## When to use

Trigger phrases:

- "모바일에서 볼 수 있게 링크도 줘"
- "외부 접속 가능하게 만들어줘"
- "HTML을 폰에서 열어보고 싶어"
- "html 링크줘" / "HTML 링크줘"
- "serve/share this report"

Only create a public URL when the user explicitly asks for a link/browser/mobile/external view. Otherwise attach the HTML via `MEDIA:` only.

## Safe serving pattern

Serve **only the single report directory**, never the whole `~/.hermes`, `media_cache`, or image cache.

```bash
cd /Users/aa/.hermes/media_cache/<report_dir>
python3 -m http.server 8877 --bind 127.0.0.1
```

Run the local server as a tracked background process when using Hermes tools. Verify locally:

```bash
python3 - <<'PY'
import urllib.request
url='http://127.0.0.1:8877/menu_analysis.html'
with urllib.request.urlopen(url, timeout=5) as r:
    print(r.status, r.headers.get('content-type'))
PY
```

## Cloudflare quick tunnel

Install if needed:

```bash
brew install cloudflared
cloudflared --version
```

Open the tunnel:

```bash
cloudflared tunnel --url http://127.0.0.1:8877
```

Extract the generated URL from logs:

```text
https://<random>.trycloudflare.com
```

The final report URL is usually:

```text
https://<random>.trycloudflare.com/menu_analysis.html
```

Verify it returns HTTP 200 before sharing:

```bash
python3 - <<'PY'
import urllib.request
url='https://<random>.trycloudflare.com/menu_analysis.html'
with urllib.request.urlopen(url, timeout=20) as r:
    print(r.status, r.headers.get('content-type'))
PY
```

## Final response pattern

Keep the chat concise:

- Provide the public URL first.
- Say it was verified.
- Mention it works only while both processes are running.
- Warn that anyone with the link can access it.
- Offer to stop the server/tunnel when done.

Example:

```markdown
모바일 URL 열어뒀어요:
https://<random>.trycloudflare.com/menu_analysis.html

확인 결과 HTTP 200입니다. 이 링크는 로컬 서버와 cloudflared 프로세스가 살아 있는 동안만 동작하고, 링크를 아는 사람은 접근할 수 있어요. 끄고 싶으면 말해 주세요.
```

## Cleanup

When the user says they are done, stop both background processes:

- local `python3 -m http.server ...`
- `cloudflared tunnel ...`

Do not leave tunnels running indefinitely.

## Security notes

- Bind the local server to `127.0.0.1`, not `0.0.0.0`.
- Serve a report-specific directory, not parent caches.
- Do not expose reports containing private images or personal data unless the user explicitly requested the public link.
- Quick tunnels are not permanent hosting. For durable hosting, suggest a private/static hosting workflow instead.
