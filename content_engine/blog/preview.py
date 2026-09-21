#!/usr/bin/env python3
"""Blog visual preview generator — renders MDX drafts as standalone styled HTML.

Produces a pixel-cyberpunk HTML file matching algorithmiccompass.com's v4 design
system, so Sahil can evaluate exact appearance before approving or amending.

Usage:
  python3 blog/preview.py <slug> [--out-dir ./previews]
  python3 blog/preview.py --all-pending [--out-dir ./previews]

Output: standalone HTML file (<slug>.html) with embedded CSS that mirrors the
production theme — no network dependencies (fonts inlined as fallbacks).
"""

from __future__ import annotations
import argparse
import base64
import json
import mimetypes
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────

_ENGINE_ROOT = Path(__file__).resolve().parent.parent
_SAHILBLOG = Path.home() / "repos" / "SahilBlog"
_DEFAULT_OUT = _ENGINE_ROOT / "previews"

PENDING_PATH = _ENGINE_ROOT / "blog_topics" / "pending_approvals.jsonl"

# ── Pixel Cyberpunk CSS (embedded, zero dependencies) ────────────────────

_PIXEL_CSS = r"""
* { margin: 0; padding: 0; box-sizing: border-box; }

@font-face {
  font-family: 'Inter';
  src: local('Inter'), local('Segoe UI'), local('system-ui');
  font-display: swap;
}
@font-face {
  font-family: 'VT323';
  src: local('VT323'), local('Courier New');
  font-display: swap;
}
@font-face {
  font-family: 'JetBrains Mono';
  src: local('JetBrains Mono'), local('Consolas'), local('monospace');
  font-display: swap;
}

:root {
  --bg: #1a1426;
  --bg-elev: #251a3a;
  --bg-card: #2d1f47;
  --bg-card-2: #3a2855;
  --rule: #3a1d4a;
  --rule-soft: #2a1538;
  --fg: #ffffff;
  --fg-soft: #d4cee8;
  --fg-mute: #968aa8;
  --neon-magenta: #ff2bd6;
  --neon-cyan: #00f0ff;
  --neon-yellow: #fff02e;
  --neon-purple: #a855f7;
  --neon-green: #5ffb8c;
  --neon-red: #ff3a5c;
  --neon-orange: #ff8a3d;
  --max-content: 720px;
  --max-read: 680px;
}

body {
  background: var(--bg);
  color: var(--fg);
  font-family: 'Inter', system-ui, sans-serif;
  line-height: 1.75;
  -webkit-font-smoothing: antialiased;
  position: relative;
  overflow-x: hidden;
}

/* Scanline overlay */
body::after {
  content: '';
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 9999;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,0,0,0.06) 2px,
    rgba(0,0,0,0.06) 4px
  );
}

.preview-banner {
  background: linear-gradient(90deg, var(--neon-magenta), var(--neon-cyan));
  padding: 8px 16px;
  text-align: center;
  font-family: 'VT323', monospace;
  font-size: 14px;
  color: #000;
  position: sticky;
  top: 0;
  z-index: 10000;
  letter-spacing: 1px;
  text-transform: uppercase;
}
.preview-banner span {
  background: #000;
  color: var(--neon-yellow);
  padding: 1px 8px;
  margin-left: 8px;
  font-size: 11px;
}

.article-container {
  max-width: var(--max-content);
  margin: 0 auto;
  padding: 48px 24px 80px;
}

/* Header / Hero */
.article-header {
  margin-bottom: 48px;
  padding-bottom: 32px;
  border-bottom: 1px solid var(--rule);
  position: relative;
}
.article-header::after {
  content: '';
  position: absolute;
  bottom: -2px;
  left: 0;
  width: 60px;
  height: 3px;
  background: var(--neon-magenta);
  box-shadow: 0 0 8px var(--neon-magenta);
}

.article-stream {
  display: inline-block;
  font-family: 'VT323', monospace;
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 3px;
  color: var(--neon-cyan);
  margin-bottom: 12px;
  padding: 2px 10px;
  border: 1px solid var(--neon-cyan);
  animation: pixel-blink 2.4s steps(1) infinite;
}
.article-stream.ai { color: var(--neon-magenta); border-color: var(--neon-magenta); }
.article-stream.pm { color: var(--neon-yellow); border-color: var(--neon-yellow); }
.article-stream.builder { color: var(--neon-green); border-color: var(--neon-green); }

.article-title {
  font-size: 36px;
  font-weight: 700;
  line-height: 1.2;
  letter-spacing: -0.5px;
  margin-bottom: 16px;
  background: linear-gradient(135deg, var(--fg), var(--neon-magenta));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.article-meta {
  font-size: 14px;
  color: var(--fg-mute);
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  align-items: center;
}

.article-meta .tag {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  padding: 2px 8px;
  border: 1px solid var(--rule);
  color: var(--fg-soft);
  border-radius: 0;
  text-transform: uppercase;
  letter-spacing: 1px;
}

.article-meta .date {
  font-family: 'VT323', monospace;
  font-size: 15px;
  color: var(--fg-mute);
}

/* Hero image */
.hero-image {
  width: 100%;
  margin-bottom: 40px;
  border: 1px solid var(--rule);
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
  background: rgba(0,0,0,0.8);
  color: var(--neon-cyan);
  font-family: 'VT323', monospace;
  font-size: 11px;
  padding: 2px 8px;
  letter-spacing: 1px;
  border: 1px solid var(--neon-cyan);
}

/* Body content */
.article-body {
  max-width: var(--max-read);
  font-size: 16px;
  color: var(--fg-soft);
}

.article-body h2 {
  font-size: 24px;
  font-weight: 700;
  margin: 40px 0 16px;
  color: var(--fg);
  padding-left: 16px;
  border-left: 3px solid var(--neon-magenta);
  position: relative;
}
.article-body h2::before {
  content: attr(data-number);
  position: absolute;
  left: -40px;
  top: 2px;
  font-family: 'VT323', monospace;
  font-size: 18px;
  color: var(--fg-mute);
  opacity: 0.6;
}

.article-body h3 {
  font-size: 19px;
  font-weight: 600;
  margin: 32px 0 12px;
  color: var(--fg);
}

.article-body p {
  margin-bottom: 20px;
}

.article-body a {
  color: var(--neon-cyan);
  text-decoration: none;
  border-bottom: 1px solid transparent;
  transition: border-color 0.2s;
}
.article-body a:hover {
  border-bottom-color: var(--neon-cyan);
}

.article-body strong {
  color: var(--fg);
  font-weight: 600;
}

.article-body em {
  color: var(--neon-purple);
}

.article-body code {
  font-family: 'JetBrains Mono', monospace;
  font-size: 14px;
  background: var(--bg-elev);
  padding: 2px 6px;
  border: 1px solid var(--rule);
  color: var(--neon-green);
}

.article-body pre {
  background: var(--bg-elev);
  border: 1px solid var(--rule);
  padding: 20px;
  margin: 24px 0;
  overflow-x: auto;
  position: relative;
}
.article-body pre::before {
  content: '▌ CODE';
  position: absolute;
  top: 6px;
  right: 10px;
  font-family: 'VT323', monospace;
  font-size: 11px;
  color: var(--fg-mute);
  letter-spacing: 2px;
}
.article-body pre code {
  background: none;
  border: none;
  padding: 0;
  color: var(--fg-soft);
  font-size: 14px;
  line-height: 1.6;
}

.article-body blockquote {
  border-left: 3px solid var(--neon-purple);
  padding: 16px 20px;
  margin: 24px 0;
  background: var(--bg-elev);
  color: var(--fg-soft);
  font-style: italic;
  position: relative;
}
.article-body blockquote::before {
  content: '"';
  position: absolute;
  top: 4px;
  left: 8px;
  font-size: 32px;
  color: var(--neon-purple);
  font-family: 'VT323', monospace;
  opacity: 0.5;
}

.article-body ul, .article-body ol {
  margin: 16px 0;
  padding-left: 24px;
}
.article-body li {
  margin-bottom: 8px;
}
.article-body ul li::marker {
  color: var(--neon-magenta);
}
.article-body ol li::marker {
  color: var(--neon-cyan);
  font-family: 'VT323', monospace;
}

.article-body hr {
  border: none;
  border-top: 1px solid var(--rule);
  margin: 40px 0;
  position: relative;
}
.article-body hr::after {
  content: '◆ ◆ ◆';
  position: absolute;
  top: -10px;
  left: 50%;
  transform: translateX(-50%);
  background: var(--bg);
  padding: 0 12px;
  color: var(--fg-mute);
  font-size: 12px;
  letter-spacing: 8px;
}

.article-body img {
  max-width: 100%;
  height: auto;
  border: 1px solid var(--rule);
  margin: 24px 0;
}

/* Section image container */
.section-image {
  margin: 32px 0;
  border: 1px solid var(--rule-soft);
  position: relative;
}
.section-image img {
  width: 100%;
  height: auto;
  display: block;
}
.section-image .section-label {
  position: absolute;
  bottom: 8px;
  right: 8px;
  background: rgba(0,0,0,0.8);
  color: var(--neon-yellow);
  font-family: 'VT323', monospace;
  font-size: 10px;
  padding: 1px 6px;
  letter-spacing: 1px;
}

/* Approval action bar */
.approval-bar {
  position: sticky;
  bottom: 0;
  background: var(--bg-elev);
  border-top: 1px solid var(--rule);
  padding: 12px 24px;
  display: flex;
  gap: 12px;
  justify-content: center;
  align-items: center;
  z-index: 9998;
  flex-wrap: wrap;
}
.approval-bar .slug {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  color: var(--fg-mute);
  margin-right: auto;
}
.approval-bar .approved-badge {
  font-family: 'VT323', monospace;
  font-size: 14px;
  color: var(--neon-green);
  text-transform: uppercase;
  letter-spacing: 2px;
  padding: 4px 16px;
  border: 1px solid var(--neon-green);
}
.approval-bar .rejected-badge {
  font-family: 'VT323', monospace;
  font-size: 14px;
  color: var(--neon-red);
  text-transform: uppercase;
  letter-spacing: 2px;
  padding: 4px 16px;
  border: 1px solid var(--neon-red);
}
.approval-bar .action-btn {
  font-family: 'VT323', monospace;
  font-size: 14px;
  text-transform: uppercase;
  letter-spacing: 2px;
  padding: 8px 24px;
  border: 1px solid var(--neon-cyan);
  background: transparent;
  color: var(--neon-cyan);
  cursor: pointer;
  transition: all 0.2s;
}
.approval-bar .action-btn.approve {
  border-color: var(--neon-green);
  color: var(--neon-green);
}
.approval-bar .action-btn.approve:hover {
  background: var(--neon-green);
  color: #000;
}
.approval-bar .action-btn.reject {
  border-color: var(--neon-red);
  color: var(--neon-red);
}
.approval-bar .action-btn.reject:hover {
  background: var(--neon-red);
  color: #000;
}
.approval-bar .action-btn.amend {
  border-color: var(--neon-yellow);
  color: var(--neon-yellow);
}
.approval-bar .action-btn.amend:hover {
  background: var(--neon-yellow);
  color: #000;
}

/* Responsive */
@media (max-width: 768px) {
  .article-title { font-size: 26px; }
  .article-container { padding: 24px 16px 60px; }
  .approval-bar { flex-direction: column; }
  .approval-bar .slug { margin-right: 0; }
}

/* Animations */
@keyframes pixel-blink {
  0%, 49% { opacity: 1; }
  50%, 100% { opacity: 0.4; }
}
@keyframes glitch {
  0%, 90%, 100% { transform: translate(0, 0); }
  92% { transform: translate(-2px, 0); }
  94% { transform: translate(2px, 0); }
  96% { transform: translate(-1px, 1px); }
}

/* Framework / Mermaid fallback */
.mermaid-fallback {
  background: var(--bg-card-2);
  border: 1px solid var(--neon-purple);
  padding: 24px;
  margin: 24px 0;
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  white-space: pre;
  overflow-x: auto;
  color: var(--neon-cyan);
}
.mermaid-fallback::before {
  content: '【FRAMEWORK】';
  display: block;
  font-family: 'VT323', monospace;
  font-size: 12px;
  color: var(--neon-purple);
  letter-spacing: 3px;
  margin-bottom: 12px;
}
"""

# ── Markdown → HTML converter (lightweight, no dependencies) ──────────────

def md_to_html(md_text: str) -> tuple[str, str]:
    """Convert markdown body to HTML. Returns (body_html, extracted_heading).
    
    Handles: headings, paragraphs, bold, italic, code, links, images,
    blockquotes, lists, horizontal rules, inline code, pre blocks.
    No external deps — pure regex + string processing.
    """
    lines = md_text.splitlines()
    html_parts: list[str] = []
    i = 0
    in_pre = False
    in_ul = False
    in_ol = False
    
    heading_text = ""
    
    while i < len(lines):
        line = lines[i]
        
        # ── Code block ──
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
        
        # ── Headings ──
        h_match = re.match(r'^(#{1,3})\s+(.+)$', stripped)
        if h_match:
            _close_list(html_parts, in_ul, in_ol)
            in_ul = in_ol = False
            level = len(h_match.group(1))
            text = _inline(h_match.group(2))
            if level == 1 and not heading_text:
                # Extract first H1 as heading for preview
                heading_text = h_match.group(2).strip()
            section_num = ""
            if level == 2:
                # Extract H2 number from text like "## The wrong mental model"
                heading_text_h2 = h_match.group(2).strip()
                # Find the H2 that has content after it — leave a data attr
                section_num = re.sub(r'[^a-z0-9]', '-', heading_text_h2.lower())[:30]
                html_parts.append(
                    f'<h2 id="{section_num}" data-number="◆">{text}</h2>\n'
                )
            elif level == 3:
                html_parts.append(f'<h3>{text}</h3>\n')
            else:
                html_parts.append(f'<h{level}>{text}</h{level}>\n')
            i += 1
            continue
        
        # ── Horizontal rule ──
        if re.match(r'^-{3,}$', stripped) or re.match(r'^\*{3,}$', stripped):
            _close_list(html_parts, in_ul, in_ol)
            in_ul = in_ol = False
            html_parts.append("<hr>\n")
            i += 1
            continue
        
        # ── Blockquote ──
        if stripped.startswith(">"):
            _close_list(html_parts, in_ul, in_ol)
            in_ul = in_ol = False
            qtext = stripped.lstrip("> ").strip()
            html_parts.append(f"<blockquote><p>{_inline(qtext)}</p></blockquote>\n")
            i += 1
            continue
        
        # ── Unordered list ──
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
        
        # ── Ordered list ──
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
        
        # ── Image ──
        img_match = re.match(r'^!\[([^\]]*)\]\(([^)]+)\)', stripped)
        if img_match:
            _close_list(html_parts, in_ul, in_ol)
            in_ul = in_ol = False
            alt = img_match.group(1)
            src = img_match.group(2)
            html_parts.append(
                f'<div class="section-image">'
                f'<img src="{src}" alt="{_escape(alt)}" loading="lazy">'
                f'<div class="section-label">{_escape(alt or "section")}</div>'
                f'</div>\n'
            )
            i += 1
            continue
        
        # ── Empty line ──
        if not stripped:
            _close_list(html_parts, in_ul, in_ol)
            in_ul = in_ol = False
            i += 1
            continue
        
        # ── Paragraph ──
        _close_list(html_parts, in_ul, in_ol)
        in_ul = in_ol = False
        html_parts.append(f"<p>{_inline(stripped)}</p>\n")
        i += 1
    
    if in_pre:
        html_parts.append("</code></pre>\n")
    _close_list(html_parts, in_ul, in_ol)
    
    return "".join(html_parts), heading_text


def _inline(text: str) -> str:
    """Process inline formatting in a text string."""
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
    # Image with inline
    text = re.sub(
        r'!\[([^\]]*)\]\(([^)]+)\)',
        lambda m: f'<img src="{_escape(m.group(2))}" alt="{_escape(m.group(1))}" style="max-width:100%">',
        text,
    )
    return text


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )


def _close_list(parts: list[str], in_ul: bool, in_ol: bool) -> None:
    if in_ul:
        parts.append("</ul>\n")
    if in_ol:
        parts.append("</ol>\n")


# ── Frontmatter parser ────────────────────────────────────────────────────

def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Extract frontmatter dict and body from MDX text."""
    fm: dict[str, str] = {}
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return fm, text
    
    end = 1
    while end < len(lines) and lines[end].strip() != "---":
        m = re.match(r'^(\w+):\s*(.+)$', lines[end])
        if m:
            key = m.group(1)
            val = m.group(2).strip().strip('"').strip("'")
            fm[key] = val
        end += 1
    
    body = "\n".join(lines[end + 1:]) if end < len(lines) else ""
    return fm, body


# ── Preview builder ───────────────────────────────────────────────────────

def _resolve_blog_asset_path(src: str) -> Optional[Path]:
    """Resolve a blog image reference to a local file path when possible."""
    if not src:
        return None
    if src.startswith("/blog/"):
        return Path.home() / "repos/SahilBlog/public" / src.lstrip("/")
    p = Path(src)
    return p if p.is_absolute() else None


def _image_to_data_uri(src: str, max_width: int = 800, jpeg_quality: int = 80) -> tuple[str, str]:
    """Return (browser_src, meta) for preview images.

    Approval HTML is delivered as a Discord attachment. Absolute local paths like
    /home/kensei/repos/SahilBlog/public/blog/<slug>/hero.png are not readable on
    Sahil's machine after download, so embed existing local images as data URIs.

    Images are compressed (resize to max_width, JPEG quality jpeg_quality) before
    embedding to keep the preview file under 1MB. Falls back to raw embedding when
    Pillow is unavailable or the image format can't be processed.
    """
    p = _resolve_blog_asset_path(src)
    if not p or not p.exists() or not p.is_file():
        return src, "image missing from preview bundle"
    try:
        # Try compressed path via Pillow
        from PIL import Image as PILImage
        pil_img = PILImage.open(p)
        ow, oh = pil_img.size
        if ow > max_width:
            nh = int(oh * max_width / ow)
            pil_img = pil_img.resize((max_width, nh), PILImage.LANCZOS)
        import io
        buf = io.BytesIO()
        # Convert to RGB if RGBA (JPEG doesn't support alpha)
        if pil_img.mode in ("RGBA", "P"):
            pil_img = pil_img.convert("RGB")
        pil_img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        compressed = buf.getvalue()
        encoded = base64.b64encode(compressed).decode("ascii")
        orig_kb = p.stat().st_size / 1024
        new_kb = len(compressed) / 1024
        return (
            f"data:image/jpeg;base64,{encoded}",
            f"compressed · {p.name} · {new_kb:.0f} KB (was {orig_kb:.0f} KB)",
        )
    except ImportError:
        # Pillow not available — fall back to raw embed
        pass
    except Exception:
        # Any compression failure — fall back to raw embed
        pass
    try:
        data = p.read_bytes()
        mime = mimetypes.guess_type(str(p))[0] or "image/png"
        encoded = base64.b64encode(data).decode("ascii")
        size_kb = len(data) / 1024
        return f"data:{mime};base64,{encoded}", f"embedded · {p.name} · {size_kb:.1f} KB"
    except OSError:
        return src, "image could not be embedded"


def build_preview_html(
    slug: str,
    title: str,
    stream: str,
    tier: str,
    tags: list[str],
    body_md: str,
    description: str = "",
    pub_date: str = "",
    hero_src: str = "",
    section_images: Optional[list[str]] = None,
    status: str = "pending",
    embed_images: bool = True,
) -> str:
    """Build a standalone HTML file from post data.

    When embed_images is False, images are referenced by their blog-relative
    paths instead of being embedded as base64 data URIs. This keeps the preview
    file small (< 50 KB) for Discord attachment, at the cost of showing image
    placeholders when the file is opened offline.
    """
    body_html, extracted_h = md_to_html(body_md)
    
    # Build tag spans
    tag_spans = "".join(
        f'<span class="tag">{_escape(t)}</span>'
        for t in tags[:6]
    )
    
    # Date
    date_display = pub_date or datetime.now().strftime("%Y-%m-%d")
    
    # Stream badge class
    stream_cls = stream.lower().replace(" ", "")
    if stream_cls not in ("ai", "pm"):
        stream_cls = "builder"
    
    # Hero image HTML
    hero_html = ""
    if hero_src:
        if embed_images:
            img_path, hero_meta = _image_to_data_uri(hero_src)
            hero_label = "▼ ARTICLE HERO IMAGE — REVIEW THIS"
        else:
            img_path = hero_src
            hero_meta = "placeholders — lightweight preview (no embedded images)"
            hero_label = "▼ HERO (placeholder in lightweight preview)"
        hero_html = f"""\
        <div class="hero-image">
          <img src="{_escape(img_path)}" alt="Hero: {_escape(title)}" onerror="this.outerHTML='<div class=\\\\'hero-image\\\\' style=\\\\'padding:40px;text-align:center;color:var(--fg-mute);border:1px dashed var(--rule);font-family:VT323,monospace\\\\'>[HERO IMAGE — {_escape(title[:60])}]</div>'">
          <div class="hero-label">{hero_label}</div>
          <div class="hero-label" style="top:auto;bottom:8px;left:8px;color:var(--neon-yellow);">{_escape(hero_meta)}</div>
        </div>"""

    # Section images
    section_html = ""
    if section_images:
        for i, img in enumerate(section_images):
            if isinstance(img, str) and img.strip():
                if embed_images:
                    section_src, section_meta = _image_to_data_uri(img.strip())
                else:
                    section_src = img.strip()
                    section_meta = "placeholders — lightweight preview"
                section_html += f"""\
                <div class="section-image">
                  <img src="{_escape(section_src)}" alt="Section {i+1}" loading="lazy" onerror="this.outerHTML='<div style=\\\\'padding:20px;text-align:center;color:var(--fg-mute);border:1px dashed var(--rule-soft);margin:16px 0;font-family:VT323,monospace\\\\'>[SECTION IMAGE {i+1}]</div>'">
                  <div class="section-label">§ {i+1:02d} · {_escape(section_meta)}</div>
                </div>"""
    
    # Status badge
    if status == "approved":
        badge = '<span class="approved-badge">✅ APPROVED</span>'
    elif status == "rejected":
        badge = '<span class="rejected-badge">❌ REJECTED</span>'
    else:
        badge = '<span class="approved-badge" style="border-color:var(--neon-orange);color:var(--neon-orange);">⏳ PENDING REVIEW</span>'
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Preview: {_escape(title)}</title>
<style>{_PIXEL_CSS}</style>
</head>
<body>

<div class="preview-banner">
  ⚡ PREVIEW — not live · {_escape(title[:60])}
  <span>{_escape(slug)}</span>
</div>

<div class="article-container">

  <div class="article-header">
    <div class="article-stream {stream_cls}">{_escape(stream)}</div>
    <h1 class="article-title">{_escape(title)}</h1>
    {f'<p style="color:var(--fg-mute);font-size:15px;margin-bottom:16px;">{_escape(description)}</p>' if description else ''}
    <div class="article-meta">
      <span class="date">◆ {date_display}</span>
      <span class="tag">tier: {_escape(tier or stream)}</span>
      {tag_spans}
    </div>
  </div>

  {hero_html}

  <div class="article-body">
    {body_html}
  </div>

  {section_html}

</div>

<div class="approval-bar">
  <span class="slug">{_escape(slug)}</span>
  {badge}
</div>

</body>
</html>'''
    return html


# ── CLI ───────────────────────────────────────────────────────────────────

def _read_mdx(slug: str) -> Optional[dict]:
    """Read an MDX file from SahilBlog by slug. Returns post dict or None."""
    mdx_dir = _SAHILBLOG / "src/content/blog"
    mdx_path = mdx_dir / f"{slug}.mdx"
    if not mdx_path.exists():
        # Try .md
        mdx_path = mdx_dir / f"{slug}.md"
    if not mdx_path.exists():
        return None
    
    text = mdx_path.read_text(encoding="utf-8", errors="replace")
    fm, body = parse_frontmatter(text)
    
    # Parse tags
    tags_raw = fm.get("tags", "[]")
    tags = json.loads(tags_raw) if tags_raw.startswith("[") else [t.strip() for t in tags_raw.strip("[]").split(",") if t.strip()]
    
    # Find images
    hero = fm.get("heroImage", "")
    sections = re.findall(
        r'!\[([^\]]*)\]\(([^)]+)\)',
        body,
    )
    section_images = [s[1] for s in sections[1:6]]  # everything after hero
    
    return {
        "slug": slug,
        "title": fm.get("title", slug),
        "description": fm.get("description", ""),
        "stream": "builder" if fm.get("tier") == "builder" else fm.get("tier", "ai"),
        "tier": fm.get("tier", "ai"),
        "tags": tags,
        "body_md": body,
        "pub_date": str(fm.get("pubDate", datetime.now().strftime("%Y-%m-%d"))),
        "hero_src": hero,
        "section_images": section_images,
        "status": fm.get("approved", "false"),
    }


def _read_from_pending(slug: str) -> Optional[dict]:
    """Read approval entry from pending_approvals.jsonl."""
    if not PENDING_PATH.exists():
        return None
    for line in PENDING_PATH.read_text().splitlines():
        try:
            entry = json.loads(line)
            if entry.get("slug") == slug:
                return entry
        except json.JSONDecodeError:
            continue
    return None


def _pending_slugs() -> list[str]:
    """Get all pending approval slugs."""
    if not PENDING_PATH.exists():
        return []
    slugs = []
    for line in PENDING_PATH.read_text().splitlines():
        try:
            entry = json.loads(line)
            if entry.get("slug"):
                slugs.append(entry["slug"])
        except json.JSONDecodeError:
            continue
    return slugs


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate visual HTML preview of blog posts")
    parser.add_argument("slug", nargs="?", help="Post slug to preview")
    parser.add_argument("--all-pending", action="store_true", help="Preview all pending approval posts")
    parser.add_argument("--out-dir", default=str(_DEFAULT_OUT), help="Output directory")
    parser.add_argument("--open", action="store_true", help="Open in browser after generation")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    slugs = []
    if args.all_pending:
        slugs = _pending_slugs()
        if not slugs:
            print("No pending approvals found.")
            return 1
    elif args.slug:
        slugs = [args.slug]
    else:
        parser.print_help()
        return 1
    
    generated = []
    for slug in slugs:
        post = _read_mdx(slug)
        if not post:
            print(f"⚠  Could not read MDX for slug: {slug}")
            continue
        
        html = build_preview_html(**post)
        out_path = out_dir / f"{slug}.html"
        out_path.write_text(html, encoding="utf-8")
        generated.append((slug, str(out_path)))
        print(f"  ✓ {slug} → {out_path}")
    
    if generated:
        print(f"\n{len(generated)} preview(s) generated in {out_dir}")
        if args.open:
            import subprocess
            for slug, path in generated:
                subprocess.run(["xdg-open", path], check=False)
        return 0
    else:
        print("No previews generated.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
