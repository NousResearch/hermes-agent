# Browser `aweme_detail` fallback for Douyin extraction

Use this when mcporter/agent-reach Douyin parser or `yt-dlp` cannot directly parse a public Douyin link, but the web page can load and play.

## Proven session pattern

1. Resolve short link to canonical page and capture `aweme_id`:
   - `https://v.douyin.com/.../` usually redirects through `www.iesdouyin.com/share/video/<aweme_id>/...` to `https://www.douyin.com/video/<aweme_id>`.
2. Open canonical page in the browser and let it load/play.
3. Inspect browser resource entries for the detail API:
   ```javascript
   performance.getEntriesByType('resource')
     .map(e => e.name)
     .filter(n => n.includes('/aweme/v1/web/aweme/detail/'))
   ```
4. Fetch that exact URL from page context so the page's cookies, generated params, and signatures are reused:
   ```javascript
   (async()=>{
     const u = performance.getEntriesByType('resource')
       .map(e=>e.name)
       .find(n=>n.includes('/aweme/v1/web/aweme/detail/'));
     const j = await (await fetch(u,{credentials:'include'})).json();
     const a = j.aweme_detail || j.aweme_list?.[0] || j;
     return {
       aweme_id: a.aweme_id,
       desc: a.desc,
       author: a.author && {nickname:a.author.nickname, uid:a.author.uid, sec_uid:a.author.sec_uid},
       stats: a.statistics,
       duration_ms: a.video?.duration,
       video_urls: [
         ...(a.video?.play_addr?.url_list || []),
         ...((a.video?.bit_rate || []).flatMap(b => b.play_addr?.url_list || []))
       ],
       image_urls: (a.images || []).flatMap(img => img.url_list || img.download_url_list || [])
     };
   })()
   ```
5. Download a returned play URL immediately; these URLs can expire:
   ```bash
   curl -L --fail --retry 2 --connect-timeout 20 \
     -A 'Mozilla/5.0' \
     -e 'https://www.douyin.com/' \
     '<play_url>' \
     -o "/tmp/douyin_${AWEME_ID}/video.mp4" \
     -w '\nhttp=%{http_code} size=%{size_download} type=%{content_type}\n'
   file "/tmp/douyin_${AWEME_ID}/video.mp4"
   ```

## Why this works

The browser page has already solved Douyin's web parameters/signatures for the active session. Reusing the exact API URL from `performance` avoids inventing `a_bogus`, `verifyFp`, or related parameters.

## Failure diagnosis

- Tiny output or HTML instead of MP4: URL expired, wrong candidate, or HTML challenge. Re-open/re-fetch `aweme_detail`, choose another `url_list` candidate, and retry with user-agent + referer.
- API list empty: wait, click play, scroll, or inspect `document.body.innerText` first; the page may not have triggered video detail loading yet.
- Image post: `video` may be absent; use `images[].url_list` and vision/OCR instead of STT.
