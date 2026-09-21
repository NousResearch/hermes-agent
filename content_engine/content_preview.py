"""CeeCee visual preview generator -- renders short posts and articles as
styled HTML cards for Discord approval, matching the SahilBlog approval format.

Two public functions:
  render_short_post(draft)  -> standalone HTML (Twitter/LinkedIn mockup)
  render_article(draft, bundle_dir) -> standalone HTML (article preview)

Output files are written to content_engine/previews/ and returned as the HTML
string so the caller can attach them via Discord's file upload mechanism.
"""

from __future__ import annotations
import base64
import mimetypes
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────

_ENGINE_ROOT = Path(__file__).resolve().parent
_PREVIEWS_DIR = _ENGINE_ROOT / "previews"
_ARTICLES_DIR = _ENGINE_ROOT / "output" / "articles"

# ── Professional Dark Theme CSS ────────────────────────────────────────────

_DARK_CSS = r"""
* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  background: #1a1a2e;
  color: #ffffff;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen,
               Ubuntu, Cantarell, sans-serif;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}

.preview-banner {
  background: linear-gradient(90deg, #e94560, #0f3460);
  padding: 10px 20px;
  text-align: center;
  font-size: 13px;
  font-weight: 600;
  color: #fff;
  position: sticky;
  top: 0;
  z-index: 10000;
  letter-spacing: 0.5px;
}
.preview-banner span {
  background: rgba(0,0,0,0.4);
  padding: 2px 10px;
  margin-left: 10px;
  border-radius: 4px;
  font-size: 11px;
  font-family: 'SF Mono', 'Consolas', monospace;
}

/* ── Short post card (Twitter / LinkedIn) ── */

.post-card {
  max-width: 600px;
  margin: 40px auto;
  background: #16213e;
  border-radius: 12px;
  padding: 24px;
  border: 1px solid #1a1a3e;
}

/* Twitter mockup */
.twitter-card .avatar-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
}
.twitter-card .avatar {
  width: 48px;
  height: 48px;
  border-radius: 50%;
  background: #e94560;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  font-size: 20px;
  color: #fff;
  flex-shrink: 0;
}
.twitter-card .name-row {
  display: flex;
  flex-direction: column;
}
.twitter-card .display-name {
  font-weight: 700;
  font-size: 15px;
  color: #fff;
}
.twitter-card .handle {
  font-size: 13px;
  color: #8899a6;
}
.twitter-card .post-body {
  font-size: 15px;
  line-height: 1.5;
  color: #e1e8ed;
  margin-bottom: 12px;
  white-space: pre-wrap;
  word-wrap: break-word;
}
.twitter-card .post-meta {
  font-size: 13px;
  color: #8899a6;
  border-top: 1px solid #1a1a3e;
  padding-top: 12px;
  margin-top: 4px;
}

/* LinkedIn mockup */
.linkedin-card .header-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}
.linkedin-card .avatar {
  width: 48px;
  height: 48px;
  border-radius: 50%;
  background: #e94560;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  font-size: 20px;
  color: #fff;
  flex-shrink: 0;
}
.linkedin-card .name-details {
  display: flex;
  flex-direction: column;
}
.linkedin-card .display-name {
  font-weight: 700;
  font-size: 14px;
  color: #fff;
}
.linkedin-card .headline {
  font-size: 12px;
  color: #8899a6;
}
.linkedin-card .post-body {
  font-size: 14px;
  line-height: 1.5;
  color: #e1e8ed;
  margin-bottom: 12px;
  white-space: pre-wrap;
  word-wrap: break-word;
}
.linkedin-card .post-meta {
  font-size: 12px;
  color: #8899a6;
  border-top: 1px solid #1a1a3e;
  padding-top: 10px;
  margin-top: 4px;
}

/* ── Article preview ── */

.article-container {
  max-width: 720px;
  margin: 0 auto;
  padding: 40px 24px 80px;
}

.article-header {
  margin-bottom: 32px;
  padding-bottom: 24px;
  border-bottom: 1px solid #1a1a3e;
}

.article-title {
  font-size: 32px;
  font-weight: 700;
  line-height: 1.25;
  letter-spacing: -0.3px;
  margin-bottom: 12px;
  color: #ffffff;
}

.article-lede {
  font-size: 16px;
  color: #a0a0b8;
  line-height: 1.6;
  margin-bottom: 8px;
}

.article-meta {
  font-size: 13px;
  color: #8899a6;
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  align-items: center;
}
.article-meta .pillar-tag {
  background: #0f3460;
  padding: 2px 10px;
  border-radius: 4px;
  font-size: 11px;
  color: #e94560;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

/* Hero image */
.hero-image {
  width: 100%;
  margin-bottom: 32px;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid #1a1a3e;
  position: relative;
}
.hero-image img {
  width: 100%;
  height: auto;
  display: block;
}
.hero-image .hero-label {
  position: absolute;
  top: 8px;
  left: 8px;
  background: rgba(0,0,0,0.75);
  color: #e94560;
  font-size: 11px;
  padding: 3px 10px;
  border-radius: 4px;
  font-weight: 600;
  letter-spacing: 0.5px;
}

/* Article body sections */
.article-body {
  font-size: 15px;
  color: #d0d0e0;
  line-height: 1.7;
}

.article-body h2 {
  font-size: 22px;
  font-weight: 700;
  margin: 32px 0 12px;
  color: #ffffff;
  padding-left: 14px;
  border-left: 3px solid #e94560;
}

.article-body h3 {
  font-size: 18px;
  font-weight: 600;
  margin: 24px 0 10px;
  color: #ffffff;
}

.article-body p {
  margin-bottom: 16px;
}

.article-body a {
  color: #e94560;
  text-decoration: none;
  border-bottom: 1px solid transparent;
}
.article-body a:hover {
  border-bottom-color: #e94560;
}

.article-body strong {
  color: #ffffff;
  font-weight: 600;
}

.article-body em {
  color: #a0a0b8;
}

.article-body code {
  font-family: 'SF Mono', 'Consolas', monospace;
  font-size: 13px;
  background: #0f3460;
  padding: 2px 6px;
  border-radius: 4px;
  color: #e94560;
}

.article-body pre {
  background: #0f3460;
  border: 1px solid #1a1a3e;
  border-radius: 8px;
  padding: 16px;
  margin: 20px 0;
  overflow-x: auto;
}
.article-body pre code {
  background: none;
  border: none;
  padding: 0;
  color: #d0d0e0;
  font-size: 13px;
  line-height: 1.5;
}

.article-body blockquote {
  border-left: 3px solid #e94560;
  padding: 12px 20px;
  margin: 20px 0;
  background: #0f3460;
  border-radius: 0 8px 8px 0;
  color: #a0a0b8;
  font-style: italic;
}

.article-body ul, .article-body ol {
  margin: 12px 0;
  padding-left: 24px;
}
.article-body li {
  margin-bottom: 6px;
}
.article-body ul li::marker {
  color: #e94560;
}

.article-body hr {
  border: none;
  border-top: 1px solid #1a1a3e;
  margin: 32px 0;
}

.article-body img {
  max-width: 100%;
  height: auto;
  border-radius: 8px;
  border: 1px solid #1a1a3e;
  margin: 20px 0;
}

/* ── Approval action bar ── */

.approval-bar {
  position: sticky;
  bottom: 0;
  background: #16213e;
  border-top: 1px solid #1a1a3e;
  padding: 14px 24px;
  display: flex;
  gap: 12px;
  justify-content: center;
  align-items: center;
  z-index: 9998;
  flex-wrap: wrap;
}
.approval-bar .draft-id {
  font-family: 'SF Mono', 'Consolas', monospace;
  font-size: 12px;
  color: #8899a6;
  margin-right: auto;
}
.approval-bar .approval-commands {
  font-family: 'SF Mono', 'Consolas', monospace;
  font-size: 12px;
  color: #a0a0b8;
  display: flex;
  gap: 16px;
}
.approval-bar .cmd {
  padding: 4px 10px;
  border-radius: 4px;
  font-weight: 600;
}
.approval-bar .cmd.approve {
  color: #2ecc71;
  background: rgba(46, 204, 113, 0.1);
}
.approval-bar .cmd.reject {
  color: #e74c3c;
  background: rgba(231, 76, 60, 0.1);
}
.approval-bar .cmd.amend {
  color: #f39c12;
  background: rgba(243, 156, 18, 0.1);
}

/* Responsive */
@media (max-width: 640px) {
  .article-title { font-size: 24px; }
  .article-container { padding: 24px 16px 60px; }
  .post-card { margin: 20px 12px; }
  .approval-bar { flex-direction: column; }
  .approval-bar .draft-id { margin-right: 0; }
}
"""


# ── Helpers ──────────────────────────────────────────────────────────────

def _escape(text: str) -> str:
    """HTML-escape a string."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )


def _inline(text: str) -> str:
    """Process inline markdown formatting in a text string."""
    text = _escape(text)
    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Italic
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', text)
    # Inline code
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    # Links
    text = re.sub(
        r'\[([^\]]+)\]\(([^)]+)\)',
        lambda m: f'<a href="{_escape(m.group(2))}" target="_blank">{m.group(1)}</a>',
        text,
    )
    return text


def _md_to_html_simple(md_text: str) -> str:
    """Convert markdown to HTML for article body sections.

    Handles: headings, paragraphs, bold, italic, code, links, images,
    blockquotes, lists, horizontal rules, inline code, pre blocks.
    """
    lines = md_text.splitlines()
    html_parts: list[str] = []
    i = 0
    in_pre = False
    in_ul = False
    in_ol = False

    while i < len(lines):
        line = lines[i]

        # Code block
        if line.strip().startswith("```"):
            if in_pre:
                html_parts.append("</code></pre>\n")
                in_pre = False
            else:
                lang = line.strip().strip("`").strip()
                html_parts.append(f'<pre><code class="lang-{lang}">')
                in_pre = True
            i += 1
            continue

        if in_pre:
            html_parts.append(_escape(line) + "\n")
            i += 1
            continue

        stripped = line.strip()

        # Headings
        h_match = re.match(r'^(#{1,3})\s+(.+)$', stripped)
        if h_match:
            _close_list(html_parts, in_ul, in_ol)
            in_ul = in_ol = False
            level = len(h_match.group(1))
            text = _inline(h_match.group(2))
            if level == 1:
                html_parts.append(f'<h2>{text}</h2>\n')
            elif level == 2:
                html_parts.append(f'<h2>{text}</h2>\n')
            else:
                html_parts.append(f'<h3>{text}</h3>\n')
            i += 1
            continue

        # Horizontal rule
        if re.match(r'^-{3,}$', stripped) or re.match(r'^\*{3,}$', stripped):
            _close_list(html_parts, in_ul, in_ol)
            in_ul = in_ol = False
            html_parts.append("<hr>\n")
            i += 1
            continue

        # Blockquote
        if stripped.startswith(">"):
            _close_list(html_parts, in_ul, in_ol)
            in_ul = in_ol = False
            qtext = stripped.lstrip("> ").strip()
            html_parts.append(f"<blockquote><p>{_inline(qtext)}</p></blockquote>\n")
            i += 1
            continue

        # Unordered list
        if re.match(r'^[-*+]\s+', stripped):
            if not in_ul:
                _close_list(html_parts, in_ul, in_ol)
                html_parts.append("<ul>\n")
                in_ul = True
                in_ol = False
            text = re.sub(r'^[-*+]\s+', '', stripped)
            html_parts.append(f"<li>{_inline(text)}</li>\n")
            i += 1
            continue

        # Ordered list
        if re.match(r'^\d+[.)]\s+', stripped):
            if not in_ol:
                _close_list(html_parts, in_ul, in_ol)
                html_parts.append("<ol>\n")
                in_ol = True
                in_ul = False
            text = re.sub(r'^\d+[.)]\s+', '', stripped)
            html_parts.append(f"<li>{_inline(text)}</li>\n")
            i += 1
            continue

        # Image
        img_match = re.match(r'^!\[([^\]]*)\]\(([^)]+)\)', stripped)
        if img_match:
            _close_list(html_parts, in_ul, in_ol)
            in_ul = in_ol = False
            alt = img_match.group(1)
            src = img_match.group(2)
            html_parts.append(
                f'<div class="section-image">'
                f'<img src="{src}" alt="{_escape(alt)}" loading="lazy">'
                f'</div>\n'
            )
            i += 1
            continue

        # Empty line
        if not stripped:
            _close_list(html_parts, in_ul, in_ol)
            in_ul = in_ol = False
            i += 1
            continue

        # Paragraph
        _close_list(html_parts, in_ul, in_ol)
        in_ul = in_ol = False
        html_parts.append(f"<p>{_inline(stripped)}</p>\n")
        i += 1

    if in_pre:
        html_parts.append("</code></pre>\n")
    _close_list(html_parts, in_ul, in_ol)

    return "".join(html_parts)


def _close_list(parts: list[str], in_ul: bool, in_ol: bool) -> None:
    if in_ul:
        parts.append("</ul>\n")
    if in_ol:
        parts.append("</ol>\n")


def _image_to_data_uri(image_path: str) -> tuple[str, str]:
    """Embed a local image as a base64 data URI.

    Returns (data_uri, meta_string). Falls back to the original path if
    the file cannot be read.
    """
    p = Path(image_path)
    if not p.exists() or not p.is_file():
        return image_path, "image not found in preview bundle"
    try:
        data = p.read_bytes()
        mime = mimetypes.guess_type(str(p))[0] or "image/png"
        encoded = base64.b64encode(data).decode("ascii")
        size_kb = len(data) / 1024
        return f"data:{mime};base64,{encoded}", f"embedded {p.name} ({size_kb:.0f} KB)"
    except OSError:
        return image_path, "image could not be embedded"


def _section_headers(body_md: str) -> list[str]:
    """Return the list of H2 headings from a markdown article body."""
    out: list[str] = []
    for line in body_md.splitlines():
        if line.startswith("## "):
            out.append(line[3:].strip())
    return out


def _first_sections(body_md: str, count: int = 3) -> str:
    """Extract the first N sections (## heading + content until next ##)."""
    lines = body_md.splitlines()
    sections: list[str] = []
    current: list[str] = []
    in_section = False
    section_count = 0

    for line in lines:
        if line.startswith("## "):
            if in_section and current:
                sections.append("\n".join(current))
                current = []
                section_count += 1
                if section_count >= count:
                    break
            in_section = True
            current = [line]
        elif in_section:
            current.append(line)

    if in_section and current and section_count < count:
        sections.append("\n".join(current))

    return "\n\n".join(sections)


def _lede(body_md: str, n: int = 240) -> str:
    """First 1-3 non-heading paragraphs, truncated to n chars."""
    parts: list[str] = []
    for line in body_md.splitlines():
        s = line.strip()
        if not s:
            if parts:
                break
            continue
        if s.startswith("#"):
            continue
        parts.append(s)
        if len(" ".join(parts)) > n:
            break
    text = " ".join(parts)
    return text[:n]


# ── Public API ────────────────────────────────────────────────────────────

def render_short_post(draft: dict) -> str:
    """Render a short post (Twitter or LinkedIn) as standalone HTML.

    Args:
        draft: A draft dict with keys: id, brand, platform, body_text,
               title, content_type, ai_image_path, visual_path, pillar.

    Returns:
        Standalone HTML string with embedded CSS.
    """
    draft_id = draft.get("id", "unknown")
    platform = (draft.get("platform") or "").lower()
    body_text = (draft.get("body_text") or "").strip()
    title = (draft.get("title") or "").strip()
    pillar = (draft.get("pillar") or "").strip()
    brand = (draft.get("brand") or "sahil").strip()
    content_type = draft.get("content_type", "text")

    # Determine platform for mockup
    is_twitter = "twitter" in platform or "x.com" in platform
    is_linkedin = "linkedin" in platform

    # Build the body content
    body_html = _inline(body_text)
    if title:
        body_html = f"<strong>{_escape(title)}</strong>\n\n{body_html}"

    # Timestamp
    stamp = datetime.now().strftime("%d %b %Y · %H:%M UTC")

    # Platform-specific mockup
    if is_twitter:
        avatar_initial = "S"
        display_name = "Sahil Saghir"
        handle = "@Sahil_Saghir"
        card_class = "twitter-card"

        card_html = f"""
<div class="post-card {card_class}">
  <div class="avatar-row">
    <div class="avatar">{avatar_initial}</div>
    <div class="name-row">
      <span class="display-name">{display_name}</span>
      <span class="handle">{handle}</span>
    </div>
  </div>
  <div class="post-body">{body_html}</div>
  <div class="post-meta">
    {stamp} · {pillar} · {content_type}
  </div>
</div>"""
    elif is_linkedin:
        avatar_initial = "S"
        display_name = "Sahil Saghir"
        headline = "Building Kensei · AI Infrastructure · Indie Dev"

        card_html = f"""
<div class="post-card linkedin-card">
  <div class="header-row">
    <div class="avatar">{avatar_initial}</div>
    <div class="name-details">
      <span class="display-name">{display_name}</span>
      <span class="headline">{headline}</span>
    </div>
  </div>
  <div class="post-body">{body_html}</div>
  <div class="post-meta">
    {stamp} · {pillar} · {content_type}
  </div>
</div>"""
    else:
        # Generic card for other platforms
        card_html = f"""
<div class="post-card">
  <div style="margin-bottom:12px;">
    <span style="font-size:13px;color:#8899a6;text-transform:uppercase;letter-spacing:0.5px;">
      {_escape(platform or brand)} · {pillar}
    </span>
  </div>
  <div class="post-body">{body_html}</div>
  <div class="post-meta">
    {stamp} · {content_type}
  </div>
</div>"""

    # Build the full HTML page
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Preview: {_escape(draft_id)}</title>
<style>{_DARK_CSS}</style>
</head>
<body>

<div class="preview-banner">
  📝 CEE CEE PREVIEW — not live · {_escape(brand)}
  <span>{_escape(draft_id)}</span>
</div>

{card_html}

<div class="approval-bar">
  <span class="draft-id">{_escape(draft_id)}</span>
  <div class="approval-commands">
    <span class="cmd approve">!approve {_escape(draft_id)}</span>
    <span class="cmd reject">!reject {_escape(draft_id)} [reason]</span>
    <span class="cmd amend">!amend {_escape(draft_id)} [notes]</span>
  </div>
</div>

</body>
</html>"""

    # Write to previews dir
    _PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _PREVIEWS_DIR / f"{draft_id}.html"
    out_path.write_text(html, encoding="utf-8")

    return html


def render_article(draft: dict, bundle_dir: Optional[str] = None) -> str:
    """Render an article as standalone HTML with hero image and first sections.

    Args:
        draft: A draft dict with keys: id, brand, platform, body_text,
               title, content_type, ai_image_path, visual_path, pillar.
        bundle_dir: Path to the article bundle directory (containing
                    article.md and imgs/). If None, derived from the
                    draft's title or id.

    Returns:
        Standalone HTML string with embedded CSS.
    """
    draft_id = draft.get("id", "unknown")
    title = (draft.get("title") or "Untitled Article").strip()
    body_text = (draft.get("body_text") or "").strip()
    pillar = (draft.get("pillar") or "").strip()
    brand = (draft.get("brand") or "sahil").strip()
    content_type = draft.get("content_type", "text")

    # Resolve bundle directory
    bundle_path: Optional[Path] = None
    if bundle_dir:
        bundle_path = Path(bundle_dir)
    else:
        # Try to find the bundle by scanning output/articles/
        if _ARTICLES_DIR.exists():
            candidates = sorted(_ARTICLES_DIR.iterdir(), reverse=True)
            for c in candidates:
                if c.is_dir() and (c / "article.md").exists():
                    bundle_path = c
                    break

    # Read article.md if available
    article_md = body_text
    if bundle_path and (bundle_path / "article.md").exists():
        article_md = (bundle_path / "article.md").read_text(encoding="utf-8", errors="replace")

    # Extract lede
    lede_text = _lede(article_md, 300)

    # Extract first 2-3 sections
    sections_html = _md_to_html_simple(_first_sections(article_md, 3))

    # Hero image: try bundle imgs/ first, then draft's ai_image_path
    hero_html = ""
    hero_img_src = ""
    if bundle_path:
        imgs_dir = bundle_path / "imgs"
        if imgs_dir.exists():
            img_files = sorted(imgs_dir.iterdir())
            if img_files:
                hero_path = str(img_files[0])
                data_uri, meta = _image_to_data_uri(hero_path)
                hero_img_src = data_uri
                hero_html = f"""
<div class="hero-image">
  <img src="{_escape(data_uri)}" alt="Hero: {_escape(title)}"
       onerror="this.outerHTML='<div style=\\\'padding:40px;text-align:center;color:#8899a6;border:1px dashed #1a1a3e;border-radius:8px;\\'>[HERO IMAGE — {_escape(title[:60])}]</div>'">
  <div class="hero-label">▼ HERO · {_escape(meta)}</div>
</div>"""
    elif draft.get("ai_image_path"):
        img_path = draft["ai_image_path"]
        if os.path.exists(img_path):
            data_uri, meta = _image_to_data_uri(img_path)
            hero_img_src = data_uri
            hero_html = f"""
<div class="hero-image">
  <img src="{_escape(data_uri)}" alt="Hero: {_escape(title)}"
       onerror="this.outerHTML='<div style=\\\'padding:40px;text-align:center;color:#8899a6;border:1px dashed #1a1a3e;border-radius:8px;\\'>[HERO IMAGE — {_escape(title[:60])}]</div>'">
  <div class="hero-label">▼ HERO · {_escape(meta)}</div>
</div>"""

    # Section headers list
    headers = _section_headers(article_md)
    headers_html = ""
    if headers:
        header_items = "\n".join(
            f"    <li>{_escape(h)}</li>" for h in headers[:6]
        )
        headers_html = f"""
<div style="background:#0f3460;border-radius:8px;padding:16px 20px;margin-bottom:24px;">
  <div style="font-size:12px;color:#8899a6;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">Sections</div>
  <ol style="margin:0;padding-left:20px;color:#d0d0e0;font-size:14px;line-height:1.8;">
{header_items}
  </ol>
</div>"""

    # Build the full HTML page
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Article Preview: {_escape(title)}</title>
<style>{_DARK_CSS}</style>
</head>
<body>

<div class="preview-banner">
  📰 ARTICLE PREVIEW — not live · {_escape(brand)}
  <span>{_escape(draft_id)}</span>
</div>

<div class="article-container">

  <div class="article-header">
    <h1 class="article-title">{_escape(title)}</h1>
    {f'<p class="article-lede">{_escape(lede_text)}</p>' if lede_text else ''}
    <div class="article-meta">
      <span class="pillar-tag">{_escape(pillar or brand)}</span>
      <span>{_escape(content_type)}</span>
      <span>{datetime.now().strftime("%d %b %Y")}</span>
    </div>
  </div>

  {hero_html}

  {headers_html}

  <div class="article-body">
    {sections_html}
  </div>

</div>

<div class="approval-bar">
  <span class="draft-id">{_escape(draft_id)}</span>
  <div class="approval-commands">
    <span class="cmd approve">!approve {_escape(draft_id)}</span>
    <span class="cmd reject">!reject {_escape(draft_id)} [reason]</span>
    <span class="cmd amend">!amend {_escape(draft_id)} [notes]</span>
  </div>
</div>

</body>
</html>"""

    # Write to previews dir
    _PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _PREVIEWS_DIR / f"{draft_id}.html"
    out_path.write_text(html, encoding="utf-8")

    return html


# ── CLI (quick test) ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Test with a sample short post
    sample_tweet = {
        "id": "test-tweet-001",
        "brand": "sahil_twitter",
        "platform": "twitter",
        "body_text": "Just shipped a new feature in Kensei that auto-generates visual previews for every draft. No more squinting at raw markdown in Discord. The approval flow now shows exactly what the post will look like before it goes live.\n\nThis is the kind of quality-of-life improvement that compounds over time.",
        "title": "",
        "content_type": "text",
        "ai_image_path": "",
        "visual_path": "",
        "pillar": "building-in-public",
    }

    sample_linkedin = {
        "id": "test-linkedin-001",
        "brand": "sahil_linkedin",
        "platform": "linkedin",
        "body_text": "I've been thinking about why most AI features fail in production.\n\nIt's not the model. It's everything around the model.\n\nThe context engineering. The error handling. The observability. The fallback chains.\n\nHermes taught me that orchestration is the real bottleneck. Not capability.",
        "title": "",
        "content_type": "text",
        "ai_image_path": "",
        "visual_path": "",
        "pillar": "ai-infrastructure",
    }

    sample_article = {
        "id": "test-article-001",
        "brand": "sahil",
        "platform": "blog",
        "body_text": "",
        "title": "How I Use Hermes to Ship AI Features Faster",
        "content_type": "article",
        "ai_image_path": "",
        "visual_path": "",
        "pillar": "building-in-public",
    }

    if "--article" in sys.argv:
        html = render_article(sample_article)
        print(f"Article preview written to {_PREVIEWS_DIR / 'test-article-001.html'}")
    elif "--linkedin" in sys.argv:
        html = render_short_post(sample_linkedin)
        print(f"LinkedIn preview written to {_PREVIEWS_DIR / 'test-linkedin-001.html'}")
    else:
        html = render_short_post(sample_tweet)
        print(f"Twitter preview written to {_PREVIEWS_DIR / 'test-tweet-001.html'}")
